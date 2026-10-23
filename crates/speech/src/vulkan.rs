//! Isolated Vulkan Whisper worker; requesting this backend never falls back to Candle.
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use filmcraft_project::{Transcript, Word};
use filmcraft_time::{TICKS_PER_SECOND, Tick};
use serde::Deserialize;

use crate::{Options, ProgressFn, SAMPLE_RATE, SpeechError, Transcriber};

const MAX_OUTPUT: u64 = 16 * 1024 * 1024;
const MAX_LOG: u64 = 1024 * 1024;

pub struct VulkanWhisper {
    executable: PathBuf,
    model: PathBuf,
    id: String,
}

impl VulkanWhisper {
    pub fn load(executable: PathBuf, models_dir: &Path, id: &str) -> Result<Self, SpeechError> {
        let metadata = crate::models::vulkan_catalogue().iter().find(|m| m.id == id).ok_or_else(|| SpeechError::UnknownModel(id.into()))?;
        let file = metadata.files.first().ok_or_else(|| SpeechError::Model("Vulkan model has no weight file".into()))?;
        let model = models_dir.join(id).join(file.name);
        if !executable.is_absolute() || !executable.is_file() {
            return Err(SpeechError::Unavailable("Vulkan worker must be an existing absolute executable path".into()));
        }
        if !crate::models::installed(models_dir, metadata) {
            return Err(SpeechError::NotInstalled(format!("{id} (Vulkan GGML weights)")));
        }
        Ok(Self { executable, model, id: id.into() })
    }
}

struct Worker(Child);
impl Drop for Worker {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

impl Transcriber for VulkanWhisper {
    fn id(&self) -> String {
        self.id.clone()
    }

    fn transcribe(&self, audio: &[f32], opts: &Options, progress: ProgressFn) -> Result<Transcript, SpeechError> {
        if !progress(0.0, "Whisper Vulkan") {
            return Err(SpeechError::Cancelled);
        }
        if audio.is_empty() {
            return Ok(Transcript { language: opts.language.clone().unwrap_or_default(), source: format!("{}/vulkan", self.id), ..Default::default() });
        }
        if audio.len() > SAMPLE_RATE as usize * 3600 {
            return Err(SpeechError::Model("Vulkan transcription is limited to one hour per media item".into()));
        }
        let dir = tempfile::tempdir()?;
        let input = dir.path().join("audio.wav");
        let mut wav = std::io::BufWriter::new(std::fs::File::create(&input)?);
        write_wav(&mut wav, audio)?;
        wav.flush()?;
        let log = dir.path().join("worker.log");
        let prefix = dir.path().join("transcript");
        let stderr = std::fs::File::create(&log)?;
        let mut worker = Worker(
            Command::new(&self.executable)
                .arg("--model")
                .arg(&self.model)
                .arg("--file")
                .arg(&input)
                .arg("--language")
                .arg(opts.language.as_deref().filter(|l| !l.is_empty()).unwrap_or("auto"))
                .args(["--output-json", "--max-len", "1", "--split-on-word", "--threads", "2"])
                .arg("--output-file")
                .arg(&prefix)
                .stdout(Stdio::null())
                .stderr(Stdio::from(stderr))
                .spawn()?,
        );
        let started = Instant::now();
        let status = loop {
            if !progress(0.5, "Transcribing on Vulkan GPU") {
                return Err(SpeechError::Cancelled);
            }
            if started.elapsed() > Duration::from_secs(3600) || std::fs::metadata(&log)?.len() > MAX_LOG {
                return Err(SpeechError::Model("Vulkan worker exceeded its time or log limit".into()));
            }
            if let Some(status) = worker.0.try_wait()? {
                break status;
            }
            std::thread::sleep(Duration::from_millis(100));
        };
        let stderr = read_bounded(&log, MAX_LOG)?;
        let stderr = String::from_utf8_lossy(&stderr);
        if !status.success() {
            return Err(SpeechError::Model(format!("Vulkan worker failed ({status}): {}", stderr.chars().take(3000).collect::<String>())));
        }
        let gpu = gpu_confirmation(&stderr)?;
        eprintln!("Whisper inference GPU: {gpu}");
        if std::env::var_os("FILMCRAFT_SPEECH_TRACE").is_some() {
            eprintln!("{stderr}");
        }
        let data = read_bounded(&prefix.with_extension("json"), MAX_OUTPUT)?;
        let mut transcript = parse_transcript(&data, audio.len(), &self.id)?;
        crate::vad::tighten_words(audio, &mut transcript.words);
        if opts.diarize && !transcript.words.is_empty() {
            if !progress(0.96, "Labelling speakers") {
                return Err(SpeechError::Cancelled);
            }
            crate::diarize::diarize(audio, &mut transcript, &crate::diarize::Params { max_speakers: opts.max_speakers, ..Default::default() });
        }
        progress(1.0, "Done");
        Ok(transcript)
    }
}

fn read_bounded(path: &Path, limit: u64) -> Result<Vec<u8>, SpeechError> {
    let mut bytes = Vec::new();
    std::fs::File::open(path)?.take(limit + 1).read_to_end(&mut bytes)?;
    if bytes.len() as u64 > limit {
        return Err(SpeechError::Model("Vulkan worker output exceeded its size limit".into()));
    }
    Ok(bytes)
}

fn gpu_confirmation(log: &str) -> Result<&str, SpeechError> {
    let device = log.lines().find_map(|l| l.strip_prefix("FILMCRAFT_GPU_REQUIRED: "));
    match device {
        Some(device) if device.starts_with("Vulkan") && !device.to_lowercase().contains("llvmpipe") && !device.to_lowercase().contains("lavapipe") => {
            Ok(device)
        }
        _ => Err(SpeechError::Unavailable("worker did not confirm a physical Vulkan GPU; CPU fallback refused".into())),
    }
}

#[derive(Deserialize)]
struct Output {
    result: Language,
    transcription: Vec<Segment>,
}
#[derive(Deserialize)]
struct Language {
    language: String,
}
#[derive(Deserialize)]
struct Segment {
    text: String,
    offsets: Offsets,
}
#[derive(Deserialize)]
struct Offsets {
    from: i64,
    to: i64,
}

fn parse_transcript(data: &[u8], samples: usize, id: &str) -> Result<Transcript, SpeechError> {
    let output: Output = serde_json::from_slice(data).map_err(|e| SpeechError::Model(format!("invalid Vulkan worker JSON: {e}")))?;
    let limit = i64::try_from(samples)
        .ok()
        .and_then(|n| n.checked_mul(1000))
        .map(|n| n / i64::from(SAMPLE_RATE))
        .ok_or_else(|| SpeechError::Model("audio duration overflow".into()))?;
    let mut words = Vec::new();
    for segment in output.transcription {
        let text = segment.text.trim();
        if text.is_empty() {
            continue;
        }
        let Offsets { from, to } = segment.offsets;
        if from < 0 || to < from || text.split_whitespace().count() != 1 {
            return Err(SpeechError::Model("invalid Vulkan word text or timestamps".into()));
        }
        // Whisper may align the last word beyond the input's end; media time stays bounded.
        let ticks =
            |ms: i64| ms.min(limit).checked_mul(TICKS_PER_SECOND / 1000).map(Tick).ok_or_else(|| SpeechError::Model("Vulkan timestamp overflow".into()));
        words.push(Word::new(text, ticks(from)?, ticks(to)?));
    }
    let mut transcript = Transcript { language: output.result.language, source: format!("{id}/vulkan"), speakers: Vec::new(), words };
    transcript.normalize();
    Ok(transcript)
}

fn write_wav(out: &mut impl Write, audio: &[f32]) -> Result<(), SpeechError> {
    if audio.iter().any(|x| !x.is_finite()) {
        return Err(SpeechError::Model("audio contains non-finite samples".into()));
    }
    let size = u32::try_from(audio.len())
        .ok()
        .and_then(|n| n.checked_mul(2))
        .filter(|n| n.checked_add(36).is_some())
        .ok_or_else(|| SpeechError::Model("audio exceeds WAV size limit".into()))?;
    out.write_all(b"RIFF")?;
    out.write_all(&(size + 36).to_le_bytes())?;
    out.write_all(b"WAVEfmt ")?;
    out.write_all(&16u32.to_le_bytes())?;
    out.write_all(&1u16.to_le_bytes())?;
    out.write_all(&1u16.to_le_bytes())?;
    out.write_all(&SAMPLE_RATE.to_le_bytes())?;
    out.write_all(&(SAMPLE_RATE * 2).to_le_bytes())?;
    out.write_all(&2u16.to_le_bytes())?;
    out.write_all(&16u16.to_le_bytes())?;
    out.write_all(b"data")?;
    out.write_all(&size.to_le_bytes())?;
    for sample in audio {
        out.write_all(&((sample.clamp(-1.0, 1.0) * 32768.0).clamp(-32768.0, 32767.0) as i16).to_le_bytes())?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vulkan_model_installation_uses_ggml_weights() {
        let base = crate::models::vulkan_catalogue().iter().find(|m| m.id == "whisper-base").unwrap();
        let dir = tempfile::tempdir().unwrap();
        assert!(!crate::models::installed(dir.path(), base));
        let model_dir = dir.path().join("whisper-base");
        std::fs::create_dir(&model_dir).unwrap();
        std::fs::File::create(model_dir.join("ggml-base.bin")).unwrap().set_len(147_951_465).unwrap();
        assert!(crate::models::installed(dir.path(), base));
        assert_eq!(crate::models::missing_bytes(dir.path(), base), 0);
    }

    #[test]
    fn gpu_confirmation_rejects_missing_or_software_device() {
        assert!(gpu_confirmation("no GPU found").is_err());
        assert!(gpu_confirmation("FILMCRAFT_GPU_REQUIRED: Vulkan0|llvmpipe").is_err());
        assert_eq!(gpu_confirmation("FILMCRAFT_GPU_REQUIRED: Vulkan0|AMD Radeon Graphics").unwrap(), "Vulkan0|AMD Radeon Graphics");
        assert!(gpu_confirmation("FILMCRAFT_GPU_REQUIRED: Vulkan0|NVIDIA GeForce RTX").is_ok());
    }

    #[test]
    fn worker_words_preserve_portuguese_and_exact_milliseconds() {
        let data = br#"{"result":{"language":"pt"},"transcription":[{"text":" Ol\u00e1!","offsets":{"from":120,"to":450}},{"text":" mundo.","offsets":{"from":500,"to":900}}]}"#;
        let transcript = parse_transcript(data, 16_000, "whisper-base").unwrap();
        assert_eq!(transcript.language, "pt");
        assert_eq!(transcript.words.len(), 2);
        assert_eq!(transcript.words[0].text, "Olá!");
        assert_eq!(transcript.words[0].start.0, 30_481_920_000);
        assert_eq!(transcript.words[1].end.0, 228_614_400_000);
    }

    #[test]
    fn malformed_worker_timing_is_an_error() {
        for data in [
            r#"{"result":{"language":"pt"},"transcription":[{"text":"oi","offsets":{"from":-1,"to":20}}]}"#,
            r#"{"result":{"language":"pt"},"transcription":[{"text":"oi","offsets":{"from":50,"to":20}}]}"#,
            r#"{"result":{"language":"pt"},"transcription":[{"text":"duas palavras","offsets":{"from":0,"to":20}}]}"#,
            "{}",
        ] {
            assert!(parse_transcript(data.as_bytes(), 16_000, "whisper-base").is_err(), "{data}");
        }
    }

    #[test]
    fn worker_word_end_is_clamped_to_the_real_audio_duration() {
        let data = br#"{"result":{"language":"en"},"transcription":[{"text":"well.","offsets":{"from":5500,"to":5960}}]}"#;
        let transcript = parse_transcript(data, 94_400, "whisper-base").unwrap();
        assert_eq!(transcript.words[0].end.0, 1_498_694_400_000);
        assert_eq!(transcript.words[0].text, "well.");
    }

    #[test]
    fn wav_rejects_nonfinite_samples() {
        let mut bytes = Vec::new();
        assert!(write_wav(&mut bytes, &[f32::NAN]).is_err());
        write_wav(&mut bytes, &[0.0, -1.0, 1.0]).unwrap();
        assert_eq!(&bytes[..4], b"RIFF");
        assert_eq!(bytes.len(), 50);
        assert_eq!(&bytes[44..], &[0, 0, 0, 128, 255, 127]);
    }

    #[cfg(unix)]
    #[test]
    fn cancellation_stops_the_worker_before_completion() {
        use crate::{Options, SpeechError, Transcriber};
        use std::os::unix::fs::PermissionsExt;
        let dir = tempfile::tempdir().unwrap();
        let exe = dir.path().join("worker");
        std::fs::write(&exe, "#!/bin/sh\nexec sleep 30\n").unwrap();
        std::fs::set_permissions(&exe, std::fs::Permissions::from_mode(0o700)).unwrap();
        let worker = VulkanWhisper { executable: exe, model: dir.path().join("model"), id: "whisper-base".into() };
        let started = std::time::Instant::now();
        let mut calls = 0;
        let result = worker.transcribe(&[0.0; 160], &Options::default(), &mut |_, _| {
            calls += 1;
            calls == 1
        });
        assert_eq!(result, Err(SpeechError::Cancelled));
        assert!(started.elapsed().as_secs() < 2);
    }
}
