//! Isolated Python vocal separation followed by hardware large-v3-family inference.
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use filmcraft_project::{Transcript, Word};
use serde::Deserialize;

use crate::{Options, ProgressFn, SpeechError, Transcriber};

const LARGE_V3: &str = "whisper-large-v3";
const LARGE_V3_TURBO: &str = "whisper-large-v3-turbo";
const LARGE_V3_Q5_0: &str = "whisper-large-v3-q5_0";
const MAX_WORKER_LOG_BYTES: u64 = 4 * 1024 * 1024;
const MAX_PROGRESS_LINE_BYTES: usize = 512;
const PROGRESS_PREFIX: &[u8] = b"FILMCRAFT_PROGRESS ";

pub struct VoiceWorker {
    python: PathBuf,
    script: PathBuf,
    model: String,
}

#[derive(Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct WorkerProgress {
    fraction: f32,
    stage: String,
}

#[derive(Default)]
struct ProgressState {
    fraction: f32,
    stage: String,
}

impl ProgressState {
    fn accept(&mut self, event: WorkerProgress) -> bool {
        if event.fraction < self.fraction {
            return false;
        }
        self.fraction = event.fraction;
        self.stage = event.stage;
        true
    }
}

struct ProgressReader {
    file: std::fs::File,
    read_bytes: u64,
    partial: Vec<u8>,
    discard_line: bool,
}

impl ProgressReader {
    fn open(path: &Path) -> Result<Self, SpeechError> {
        Ok(Self { file: std::fs::File::open(path)?, read_bytes: 0, partial: Vec::new(), discard_line: false })
    }

    fn poll(&mut self, max_bytes: u64) -> Result<Vec<WorkerProgress>, SpeechError> {
        let mut bytes = Vec::new();
        let remaining = max_bytes.saturating_sub(self.read_bytes);
        let read = (&mut self.file).take(remaining).read_to_end(&mut bytes)?;
        self.read_bytes = self.read_bytes.saturating_add(read as u64);
        let mut events = Vec::new();
        for byte in bytes {
            if byte == b'\n' {
                if !self.discard_line
                    && let Some(event) = parse_progress_line(&self.partial)
                {
                    events.push(event);
                }
                self.partial.clear();
                self.discard_line = false;
            } else if !self.discard_line {
                if self.partial.len() < MAX_PROGRESS_LINE_BYTES {
                    self.partial.push(byte);
                } else {
                    self.partial.clear();
                    self.discard_line = true;
                }
            }
        }
        Ok(events)
    }
}

fn parse_progress_line(line: &[u8]) -> Option<WorkerProgress> {
    let payload = line.strip_prefix(PROGRESS_PREFIX)?;
    let event: WorkerProgress = serde_json::from_slice(payload).ok()?;
    if !event.fraction.is_finite()
        || !(0.0..=1.0).contains(&event.fraction)
        || event.stage.trim().is_empty()
        || event.stage.len() > 160
        || event.stage.chars().any(char::is_control)
    {
        return None;
    }
    Some(event)
}

fn valid_model(model: &str) -> bool {
    matches!(model, LARGE_V3 | LARGE_V3_TURBO | LARGE_V3_Q5_0)
}

struct ChildGuard(std::process::Child);

impl Drop for ChildGuard {
    fn drop(&mut self) {
        if matches!(self.0.try_wait(), Ok(None)) {
            let _ = self.0.kill();
        }
        let _ = self.0.wait();
    }
}

impl VoiceWorker {
    pub fn load(python: PathBuf) -> Result<Self, SpeechError> {
        Self::load_model(python, LARGE_V3)
    }

    pub fn load_model(python: PathBuf, model: &str) -> Result<Self, SpeechError> {
        if !valid_model(model) {
            return Err(SpeechError::UnknownModel(model.into()));
        }
        let script = std::env::var_os("FILMCRAFT_VOICE_SCRIPT")
            .map(PathBuf::from)
            .ok_or_else(|| SpeechError::Unavailable("voice worker script is not configured".into()))?;
        for path in [&python, &script] {
            if !path.is_absolute() || !path.is_file() {
                return Err(SpeechError::Unavailable("voice worker paths must be existing absolute files".into()));
            }
        }
        Ok(Self { python, script, model: model.into() })
    }
}

impl Transcriber for VoiceWorker {
    fn id(&self) -> String {
        self.model.clone()
    }
    fn input_sample_rate(&self) -> u32 {
        48_000
    }
    fn transcribe(&self, audio: &[f32], opts: &Options, progress: ProgressFn) -> Result<Transcript, SpeechError> {
        self.run(&[audio], crate::SAMPLE_RATE, opts, progress)
    }
    fn transcribe_channels(&self, channels: &[Vec<f32>], opts: &Options, progress: ProgressFn) -> Result<Transcript, SpeechError> {
        if channels.len() > 2 {
            let mono = crate::downmix(channels);
            return self.run(&[&mono], self.input_sample_rate(), opts, progress);
        }
        let slices: Vec<&[f32]> = channels.iter().map(Vec::as_slice).collect();
        self.run(&slices, self.input_sample_rate(), opts, progress)
    }
}

impl VoiceWorker {
    fn run(&self, channels: &[&[f32]], rate: u32, _opts: &Options, progress: ProgressFn) -> Result<Transcript, SpeechError> {
        if !progress(0.0, "Loading model") {
            return Err(SpeechError::Cancelled);
        }
        let dir = tempfile::tempdir()?;
        let input = dir.path().join("original.wav");
        let output = dir.path().join("transcript.json");
        let log = dir.path().join("worker.log");
        let frames = write_wav(&input, channels, rate)?;
        let stderr = std::fs::File::create(&log)?;
        let mut progress_log = ProgressReader::open(&log)?;
        let mut child = ChildGuard(
            Command::new(&self.python)
                .env("TMPDIR", dir.path())
                .env("FILMCRAFT_VOICE_MODEL", &self.model)
                .arg(&self.script)
                .arg(&input)
                .arg(&output)
                .stdout(Stdio::null())
                .stderr(Stdio::from(stderr))
                .spawn()?,
        );
        let started = Instant::now();
        let result = (|| {
            let mut latest = ProgressState { fraction: 0.0, stage: "Loading model".into() };
            loop {
                if started.elapsed() > Duration::from_secs(7200) || std::fs::metadata(&log)?.len() > MAX_WORKER_LOG_BYTES {
                    return Err(SpeechError::Model("voice worker exceeded time or log limit".into()));
                }
                let mut reported = false;
                for event in progress_log.poll(MAX_WORKER_LOG_BYTES)? {
                    if latest.accept(event) {
                        reported = true;
                        if !progress(latest.fraction, &latest.stage) {
                            return Err(SpeechError::Cancelled);
                        }
                    }
                }
                if !reported && !progress(latest.fraction, &latest.stage) {
                    return Err(SpeechError::Cancelled);
                }
                if let Some(status) = child.0.try_wait()? {
                    if !status.success() {
                        let log = bounded(&log, MAX_WORKER_LOG_BYTES)?;
                        let message = String::from_utf8_lossy(&log);
                        return Err(SpeechError::Model(format!(
                            "voice worker failed ({status}): {}",
                            message.chars().rev().take(3000).collect::<String>().chars().rev().collect::<String>()
                        )));
                    }
                    break;
                }
                std::thread::sleep(Duration::from_millis(100));
            }
            let transcript = parse(&bounded(&output, 64 * 1024 * 1024)?, frames as f64 / f64::from(rate), &self.model)?;
            eprintln!("FilmCraft voice transcript source: {}", transcript.source);
            Ok(transcript)
        })();
        if result.is_err() {
            if matches!(child.0.try_wait(), Ok(None)) {
                // Never signal a reaped PID: it could already belong to another process.
                let _ = Command::new("kill").args(["-TERM", &child.0.id().to_string()]).status();
                let deadline = Instant::now() + Duration::from_secs(3);
                while Instant::now() < deadline && child.0.try_wait().ok().flatten().is_none() {
                    std::thread::sleep(Duration::from_millis(50));
                }
            }
            let _ = child.0.kill();
        }
        let _ = child.0.wait();
        if result.is_ok() {
            progress(1.0, "Done");
        }
        result
    }
}

fn bounded(path: &Path, limit: u64) -> Result<Vec<u8>, SpeechError> {
    let mut bytes = Vec::new();
    std::fs::File::open(path)?.take(limit + 1).read_to_end(&mut bytes)?;
    if bytes.len() as u64 > limit {
        return Err(SpeechError::Model("voice output exceeds limit".into()));
    }
    Ok(bytes)
}

#[derive(Deserialize)]
struct Output {
    language: String,
    backend: String,
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    isolated: bool,
    words: Vec<Segment>,
}
#[derive(Deserialize)]
struct Segment {
    start: f64,
    end: f64,
    text: String,
    #[serde(default)]
    p: Option<f32>,
    #[serde(default)]
    timestamp_type: Option<String>,
}

fn parse(bytes: &[u8], duration: f64, requested_model: &str) -> Result<Transcript, SpeechError> {
    if !valid_model(requested_model) {
        return Err(SpeechError::UnknownModel(requested_model.into()));
    }
    let out: Output = serde_json::from_slice(bytes).map_err(|e| SpeechError::Model(format!("invalid voice JSON: {e}")))?;
    if out.language.is_empty() || !matches!(out.backend.as_str(), "amd" | "nvidia" | "cpu") {
        return Err(SpeechError::Model("voice worker returned invalid hardware or language".into()));
    }
    let model = match out.model {
        Some(model) if valid_model(&model) && model == requested_model => model,
        Some(_) => return Err(SpeechError::Model("voice worker returned an unsupported or unexpected model".into())),
        None if requested_model == LARGE_V3 => LARGE_V3.into(),
        None => return Err(SpeechError::Model("voice worker did not report the selected model".into())),
    };
    let estimated_timestamps = out.words.iter().any(|word| word.timestamp_type.as_deref() == Some("estimated"));
    let mut words = Vec::new();
    for row in out.words {
        if !row.start.is_finite()
            || !row.end.is_finite()
            || row.start < 0.0
            || row.end < row.start
            || row.text.trim().is_empty()
            || row.p.is_some_and(|p| !p.is_finite() || !(0.0..=1.0).contains(&p))
            || row.timestamp_type.as_deref().is_some_and(|kind| kind != "estimated")
        {
            return Err(SpeechError::Model("voice worker returned invalid word timing, confidence, or text".into()));
        }
        let mut word = Word::new(row.text.trim(), crate::seconds_tick(row.start.min(duration)), crate::seconds_tick(row.end.min(duration)));
        word.confidence = row.p.unwrap_or_default();
        words.push(word);
    }
    let isolation = if out.isolated { "+demucs" } else { "" };
    let timestamp_source = if estimated_timestamps { "+estimated-timestamps" } else { "" };
    let mut transcript =
        Transcript { language: out.language, source: format!("{model}/{}{isolation}{timestamp_source}", out.backend), words, ..Default::default() };
    transcript.normalize();
    Ok(transcript)
}

fn write_wav(path: &Path, channels: &[&[f32]], rate: u32) -> Result<usize, SpeechError> {
    let frames = channels.first().map_or(0, |c| c.len());
    if !(1..=2).contains(&channels.len()) || frames == 0 || rate == 0 || channels.iter().any(|c| c.len() != frames || c.iter().any(|x| !x.is_finite())) {
        return Err(SpeechError::Model("voice isolation requires finite mono or stereo audio of equal length".into()));
    }
    let size = frames
        .checked_mul(channels.len())
        .and_then(|n| n.checked_mul(2))
        .and_then(|n| u32::try_from(n).ok())
        .filter(|n| n.checked_add(36).is_some())
        .ok_or_else(|| SpeechError::Model("voice WAV size overflow".into()))?;
    let count = channels.len() as u16;
    let mut out = std::io::BufWriter::new(std::fs::File::create(path)?);
    out.write_all(b"RIFF")?;
    out.write_all(&(size + 36).to_le_bytes())?;
    out.write_all(b"WAVEfmt ")?;
    out.write_all(&16u32.to_le_bytes())?;
    out.write_all(&1u16.to_le_bytes())?;
    out.write_all(&count.to_le_bytes())?;
    out.write_all(&rate.to_le_bytes())?;
    out.write_all(&(rate * u32::from(count) * 2).to_le_bytes())?;
    out.write_all(&(count * 2).to_le_bytes())?;
    out.write_all(&16u16.to_le_bytes())?;
    out.write_all(b"data")?;
    out.write_all(&size.to_le_bytes())?;
    for i in 0..frames {
        for channel in channels {
            out.write_all(&((channel[i].clamp(-1.0, 1.0) * 32768.0).clamp(-32768.0, 32767.0) as i16).to_le_bytes())?;
        }
    }
    out.flush()?;
    Ok(frames)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn exited_worker_error_cleans_audio_without_signalling_reaped_pid() {
        let dir = tempfile::tempdir().unwrap();
        let script = dir.path().join("worker.sh");
        let observed = dir.path().join("input-path");
        let python_temp = dir.path().join("python-temp-path");
        std::fs::write(
            &script,
            format!(
                "printf '%s' \"$1\" > '{}'\nmkdir -p \"$TMPDIR/python-temp\"\nprintf '%s/python-temp' \"$TMPDIR\" > '{}'\nprintf '{{}}' > \"$2\"\n",
                observed.display(),
                python_temp.display()
            ),
        )
        .unwrap();
        let worker = VoiceWorker { python: "/bin/sh".into(), script, model: LARGE_V3.into() };
        assert!(worker.transcribe(&[0.0; 160], &Options::default(), &mut |_, _| true).is_err());
        let input = PathBuf::from(std::fs::read_to_string(observed).unwrap());
        assert!(!input.exists());
        assert!(!input.parent().unwrap().exists());
        let worker_temp = PathBuf::from(std::fs::read_to_string(python_temp).unwrap());
        assert!(worker_temp.starts_with(input.parent().unwrap()));
        assert!(!worker_temp.exists());
    }
    #[test]
    fn stereo_wav_preserves_rate_channels_and_interleaving() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("input.wav");
        assert_eq!(write_wav(&path, &[&[1.0, 0.0], &[-1.0, 0.5]], 48_000).unwrap(), 2);
        let b = std::fs::read(path).unwrap();
        assert_eq!(u16::from_le_bytes([b[22], b[23]]), 2);
        assert_eq!(u32::from_le_bytes(b[24..28].try_into().unwrap()), 48_000);
        assert_eq!(&b[44..], &[255, 127, 0, 128, 0, 0, 0, 64]);
    }
    #[test]
    fn voice_parser_preserves_language_and_rejects_unknown_backend() {
        let t = parse(br#"{"language":"de","backend":"amd","words":[{"start":0.1,"end":1.2,"text":"Hallo"}]}"#, 1.0, LARGE_V3).unwrap();
        assert_eq!(t.language, "de");
        assert_eq!(t.source, "whisper-large-v3/amd");
        assert_eq!(t.words[0].end, crate::seconds_tick(1.0));
        assert!(parse(br#"{"language":"de","backend":"other","words":[]}"#, 1.0, LARGE_V3).is_err());
        assert!(parse(br#"{"language":"de","backend":"amd","words":[{"start":2,"end":1,"text":"Hallo"}]}"#, 3.0, LARGE_V3).is_err());
    }
    #[test]
    fn voice_parser_requires_turbo_worker_to_report_its_model() {
        let t = parse(br#"{"language":"en","backend":"amd","model":"whisper-large-v3-turbo","words":[]}"#, 1.0, LARGE_V3_TURBO).unwrap();
        assert_eq!(t.source, "whisper-large-v3-turbo/amd");
        assert!(parse(br#"{"language":"en","backend":"amd","words":[]}"#, 1.0, LARGE_V3_TURBO).is_err());
        assert!(parse(br#"{"language":"en","backend":"amd","model":"whisper-large-v3","words":[]}"#, 1.0, LARGE_V3_TURBO,).is_err());
        assert!(parse(br#"{"language":"en","backend":"amd","model":"whisper-large-v3-small","words":[]}"#, 1.0, LARGE_V3,).is_err());
    }
    #[test]
    fn fallback_transcript_does_not_claim_demucs_isolation() {
        let t = parse(br#"{"language":"en","backend":"amd","isolated":false,"words":[]}"#, 1.0, LARGE_V3).unwrap();
        assert_eq!(t.source, "whisper-large-v3/amd");
    }
    #[test]
    fn estimated_word_timestamps_are_identified_in_source() {
        let t = parse(br#"{"language":"en","backend":"amd","words":[{"start":0.1,"end":0.4,"text":"hello","timestamp_type":"estimated"}]}"#, 1.0, LARGE_V3)
            .unwrap();
        assert_eq!(t.source, "whisper-large-v3/amd+estimated-timestamps");
    }
    #[test]
    fn cpu_output_preserves_backend_and_decoder_confidence() {
        let t = parse(
            br#"{"language":"en","backend":"cpu","isolated":true,"words":[{"start":0.1,"end":0.4,"text":"hello","p":0.37},{"start":0.5,"end":0.8,"text":"world"}]}"#,
            1.0,
            LARGE_V3,
        )
        .unwrap();
        assert_eq!(t.source, "whisper-large-v3/cpu+demucs");
        assert_eq!(t.words[0].confidence, 0.37);
        assert_eq!(t.words[1].confidence, 0.0, "missing decoder score must not claim perfect confidence");
    }
    #[test]
    fn voice_model_ids_are_an_exact_whitelist() {
        assert!(valid_model(LARGE_V3));
        assert!(valid_model(LARGE_V3_TURBO));
        assert!(!valid_model("whisper-large-v3-turbo;touch /tmp/x"));
        assert!(!valid_model("whisper-large-v3-turbo\nFILMCRAFT_GPU_REQUIRED"));
        assert!(!valid_model("large-v3-turbo"));
    }
    #[test]
    fn progress_parser_accepts_only_small_well_formed_events() {
        assert_eq!(
            parse_progress_line(br#"FILMCRAFT_PROGRESS {"fraction":0.4,"stage":"Transcribing"}"#),
            Some(WorkerProgress { fraction: 0.4, stage: "Transcribing".into() })
        );
        let lines: &[&[u8]] = &[
            b"ordinary log",
            br#"FILMCRAFT_PROGRESS {"fraction":-0.1,"stage":"Loading"}"#,
            br#"FILMCRAFT_PROGRESS {"fraction":1.1,"stage":"Done"}"#,
            br#"FILMCRAFT_PROGRESS {"fraction":0.2,"stage":"  "}"#,
            br#"FILMCRAFT_PROGRESS {"fraction":0.2,"stage":"Stage","audio":"private"}"#,
            b"FILMCRAFT_PROGRESS not-json",
        ];
        for line in lines {
            assert!(parse_progress_line(line).is_none(), "accepted hostile progress line: {:?}", String::from_utf8_lossy(line));
        }
    }
    #[test]
    fn progress_state_ignores_backwards_events() {
        let mut state = ProgressState::default();
        assert!(state.accept(WorkerProgress { fraction: 0.2, stage: "isolation".into() }));
        assert!(state.accept(WorkerProgress { fraction: 0.7, stage: "transcription".into() }));
        assert!(!state.accept(WorkerProgress { fraction: 0.6, stage: "stale".into() }));
        assert_eq!(state.fraction, 0.7);
        assert_eq!(state.stage, "transcription");
    }
    #[test]
    fn worker_passes_model_id_and_reports_real_progress_without_forwarding_options() {
        let dir = tempfile::tempdir().unwrap();
        let script = dir.path().join("worker.sh");
        let observed_model = dir.path().join("model-id");
        let observed_args = dir.path().join("args");
        std::fs::write(
            &script,
            format!(
                "printf '%s' \"$FILMCRAFT_VOICE_MODEL\" > '{}'\nprintf '%s\\n' \"$# $1 $2\" > '{}'\nprintf '%s\\n' 'FILMCRAFT_PROGRESS {{\"fraction\":0.4,\"stage\":\"Isolating vocals\"}}' >&2\nprintf '%s\\n' 'FILMCRAFT_PROGRESS {{\"fraction\":0.8,\"stage\":\"Transcribing\"}}' >&2\nprintf '%s' '{{\"language\":\"en\",\"backend\":\"amd\",\"model\":\"whisper-large-v3-turbo\",\"isolated\":true,\"words\":[]}}' > \"$2\"\n",
                observed_model.display(),
                observed_args.display()
            ),
        )
        .unwrap();
        let worker = VoiceWorker { python: "/bin/sh".into(), script, model: LARGE_V3_TURBO.into() };
        assert_eq!(worker.id(), LARGE_V3_TURBO);
        let opts = Options { language: Some("must-not-be-logged".into()), ..Default::default() };
        let mut updates = Vec::new();
        let transcript = worker
            .transcribe(&[0.0; 160], &opts, &mut |fraction, stage| {
                updates.push((fraction, stage.to_string()));
                true
            })
            .unwrap();
        assert_eq!(std::fs::read_to_string(observed_model).unwrap(), LARGE_V3_TURBO);
        let args = std::fs::read_to_string(observed_args).unwrap();
        assert_eq!(args.split_whitespace().next(), Some("2"));
        assert!(!args.contains("must-not-be-logged"));
        assert_eq!(transcript.source, "whisper-large-v3-turbo/amd+demucs");
        assert!(updates.windows(2).all(|pair| pair[0].0 <= pair[1].0));
        assert!(updates.iter().any(|(fraction, stage)| *fraction == 0.4 && stage == "Isolating vocals"));
        assert!(updates.iter().any(|(fraction, stage)| *fraction == 0.8 && stage == "Transcribing"));
        assert_eq!(updates.first().map(|u| u.0), Some(0.0));
        assert_eq!(updates.last().map(|u| u.0), Some(1.0));
        assert!(!updates.iter().any(|(_, stage)| stage.contains("must-not-be-logged")));
    }
    #[test]
    fn dropping_voice_worker_reaps_running_child() {
        let child = Command::new("sleep").arg("60").spawn().unwrap();
        let pid = child.id();
        drop(ChildGuard(child));
        assert!(!PathBuf::from(format!("/proc/{pid}")).exists());
    }
}
