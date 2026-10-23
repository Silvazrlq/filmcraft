//! The speech model catalogue and the model store.
//!
//! Models are downloaded on first use, after the user confirms (the dialog shows the size, the
//! source URL and the licence), into `<data dir>/models/<id>/`. They are never bundled with
//! FilmCraft or committed to the repository. Every file is pinned to a Hugging Face revision of
//! OpenAI's own repositories and checked against its SHA-256 before it is used.
//!
//! Licence: OpenAI released the Whisper code and model weights under the MIT licence
//! (<https://github.com/openai/whisper/blob/main/LICENSE>); the safetensors conversions OpenAI
//! publishes at `huggingface.co/openai/whisper-*` are labelled Apache-2.0. Both are permissive.

use std::path::{Path, PathBuf};

use serde::Serialize;

/// One file of a model.
#[derive(Clone, Copy, Debug, Serialize)]
pub struct ModelFile {
    pub name: &'static str,
    pub url: &'static str,
    pub sha256: &'static str,
    pub size: u64,
}

/// A downloadable speech model.
#[derive(Clone, Copy, Debug, Serialize)]
pub struct ModelInfo {
    pub id: &'static str,
    pub name: &'static str,
    /// Recognises languages other than English (and detects the language).
    pub multilingual: bool,
    pub description: &'static str,
    pub license: &'static str,
    pub license_url: &'static str,
    pub author: &'static str,
    /// Human-readable source page.
    pub source: &'static str,
    pub files: &'static [ModelFile],
}

impl ModelInfo {
    pub fn size(&self) -> u64 {
        self.files.iter().map(|f| f.size).sum()
    }
}

const LICENSE: &str = "MIT (OpenAI Whisper weights); Hugging Face conversion: Apache-2.0";
const LICENSE_URL: &str = "https://github.com/openai/whisper/blob/main/LICENSE";
const AUTHOR: &str = "OpenAI";

macro_rules! hf {
    ($repo:literal, $rev:literal, $name:literal, $sha:literal, $size:expr) => {
        ModelFile { name: $name, url: concat!("https://huggingface.co/openai/", $repo, "/resolve/", $rev, "/", $name), sha256: $sha, size: $size }
    };
}

const TOKENIZER_MULTI: &str = "27fc476bfe7f17299480be2273fc0608e4d5a99aba2ab5dec5374b4482d1a566";

const CATALOGUE: &[ModelInfo] = &[
    ModelInfo {
        id: "whisper-tiny",
        name: "Whisper tiny (multilingual)",
        multilingual: true,
        description: "39 M parameters. Fastest; fine for clear speech, rough on noisy audio.",
        license: LICENSE,
        license_url: LICENSE_URL,
        author: AUTHOR,
        source: "https://huggingface.co/openai/whisper-tiny",
        files: &[
            hf!(
                "whisper-tiny",
                "169d4a4341b33bc18d8881c4b69c2e104e1cc0af",
                "config.json",
                "ffdccec4f3211f4c63310f2b7098f309fe70f3952cedc5e4d11e43f5b2379b98",
                1983
            ),
            hf!(
                "whisper-tiny",
                "169d4a4341b33bc18d8881c4b69c2e104e1cc0af",
                "generation_config.json",
                "a5d5325911f16e74001a72fa13d6e208eee51548f994646de1f4b4cc8b35b512",
                3747
            ),
            hf!(
                "whisper-tiny",
                "169d4a4341b33bc18d8881c4b69c2e104e1cc0af",
                "tokenizer.json",
                "27fc476bfe7f17299480be2273fc0608e4d5a99aba2ab5dec5374b4482d1a566",
                2480466
            ),
            hf!(
                "whisper-tiny",
                "169d4a4341b33bc18d8881c4b69c2e104e1cc0af",
                "model.safetensors",
                "7ebd0e69e78190ffe1438491fa05cc1f5c1aa3a4c4db3bc1723adbb551ea2395",
                151061672
            ),
        ],
    },
    ModelInfo {
        id: "whisper-base",
        name: "Whisper base (multilingual)",
        multilingual: true,
        description: "74 M parameters. The default: a good balance of speed and accuracy.",
        license: LICENSE,
        license_url: LICENSE_URL,
        author: AUTHOR,
        source: "https://huggingface.co/openai/whisper-base",
        files: &[
            hf!(
                "whisper-base",
                "e37978b90ca9030d5170a5c07aadb050351a65bb",
                "config.json",
                "a153c53883a6799b6f056b4a8d1a515c9926d03994682ba88a7616618d7da0c1",
                1983
            ),
            hf!(
                "whisper-base",
                "e37978b90ca9030d5170a5c07aadb050351a65bb",
                "generation_config.json",
                "444b3f636d2fff89dd9ecf549e2a085b61f7ff0fa0246d4628bac6a3b8cc9ba4",
                3807
            ),
            hf!(
                "whisper-base",
                "e37978b90ca9030d5170a5c07aadb050351a65bb",
                "tokenizer.json",
                "27fc476bfe7f17299480be2273fc0608e4d5a99aba2ab5dec5374b4482d1a566",
                2480466
            ),
            hf!(
                "whisper-base",
                "e37978b90ca9030d5170a5c07aadb050351a65bb",
                "model.safetensors",
                "07cadb9f25677c8d50df603e66a98fbd842cce45047139baeb16e6219a1e807b",
                290403936
            ),
        ],
    },
    ModelInfo {
        id: "whisper-small",
        name: "Whisper small (multilingual)",
        multilingual: true,
        description: "244 M parameters. Most accurate here; about 4× slower than base.",
        license: LICENSE,
        license_url: LICENSE_URL,
        author: AUTHOR,
        source: "https://huggingface.co/openai/whisper-small",
        files: &[
            hf!(
                "whisper-small",
                "973afd24965f72e36ca33b3055d56a652f456b4d",
                "config.json",
                "e6a2b489da1b5aed65a8eb8d1e7466fa867ad5643a8bc138ba708bd56b2875c4",
                1967
            ),
            hf!(
                "whisper-small",
                "973afd24965f72e36ca33b3055d56a652f456b4d",
                "generation_config.json",
                "71565b8ef50d0bf7a1193ed4bbed195b94e70c18894d81bba2f1233dcec3ab53",
                3868
            ),
            hf!(
                "whisper-small",
                "973afd24965f72e36ca33b3055d56a652f456b4d",
                "tokenizer.json",
                "27fc476bfe7f17299480be2273fc0608e4d5a99aba2ab5dec5374b4482d1a566",
                2480466
            ),
            hf!(
                "whisper-small",
                "973afd24965f72e36ca33b3055d56a652f456b4d",
                "model.safetensors",
                "1d7734884874f1a1513ed9aa760a4f8e97aaa02fd6d93a3a85d27b2ae9ca596b",
                966995080
            ),
        ],
    },
];

/// The default model.
pub const DEFAULT_MODEL: &str = "whisper-base";

#[cfg(target_os = "linux")]
macro_rules! ggml {
    ($name:literal, $sha:literal, $size:expr) => {
        ModelFile {
            name: $name,
            url: concat!("https://huggingface.co/ggerganov/whisper.cpp/resolve/5359861c739e955e79d9a303bcbc70fb988958b1/", $name),
            sha256: $sha,
            size: $size,
        }
    };
}

#[cfg(target_os = "linux")]
pub(crate) fn vulkan_catalogue() -> &'static [ModelInfo] {
    const MODELS: &[ModelInfo] = &[
        ModelInfo { files: &[ggml!("ggml-tiny.bin", "be07e048e1e599ad46341c8d2a135645097a538221678b7acdd1b1919c6e1b21", 77_691_713)], ..CATALOGUE[0] },
        ModelInfo { files: &[ggml!("ggml-base.bin", "60ed5bc3dd14eea856493d334349b405782ddcaf0028d4b5df4088345fba2efe", 147_951_465)], ..CATALOGUE[1] },
        ModelInfo { files: &[ggml!("ggml-small.bin", "1be3a9b2063867b937e64e2ec7483364a79917e157fa98c5d94b5c1fffea987b", 487_601_967)], ..CATALOGUE[2] },
        ModelInfo {
            id: "whisper-large-v3",
            name: "Whisper large-v3 (multilingual, Vulkan F16)",
            description: "1.55 B parameters. Unquantized F16 GGML weights for large-v3.",
            source: "https://huggingface.co/openai/whisper-large-v3",
            files: &[ggml!("ggml-large-v3.bin", "64d182b440b98d5203c4f9bd541544d84c605196c4f7b845dfa11fb23594d1e2", 3_095_033_483)],
            ..CATALOGUE[2]
        },
        ModelInfo {
            id: "whisper-large-v3-q5_0",
            name: "Whisper large-v3 (multilingual, Vulkan Q5_0)",
            description: "1.55 B parameters. Quantized Q5_0 GGML weights for full large-v3.",
            source: "https://huggingface.co/ggerganov/whisper.cpp/tree/5359861c739e955e79d9a303bcbc70fb988958b1",
            files: &[ggml!("ggml-large-v3-q5_0.bin", "d75795ecff3f83b5faa89d1900604ad8c780abd5739fae406de19f23ecd98ad1", 1_081_140_203)],
            ..CATALOGUE[2]
        },
        ModelInfo {
            id: "whisper-large-v3-turbo",
            name: "Whisper large-v3 Turbo (multilingual, Vulkan F16)",
            description: "809 M parameters. Faster than large-v3; accuracy varies by language and audio.",
            source: "https://huggingface.co/openai/whisper-large-v3-turbo",
            files: &[ggml!("ggml-large-v3-turbo.bin", "1fc70f774d38eb169993ac391eea357ef47c88757ef72ee5943879b7e8e2bc69", 1_624_555_275)],
            ..CATALOGUE[2]
        },
    ];
    MODELS
}

pub fn catalogue() -> &'static [ModelInfo] {
    #[cfg(target_os = "linux")]
    if std::env::var_os("FILMCRAFT_WHISPER_VULKAN").is_some() {
        return vulkan_catalogue();
    }
    CATALOGUE
}

/// Model choices available through the currently configured backend.
pub fn settings_choices() -> Vec<(&'static str, &'static str)> {
    catalogue().iter().map(|m| (m.id, m.name)).collect()
}

pub fn find(id: &str) -> Option<&'static ModelInfo> {
    catalogue().iter().find(|m| m.id == id)
}

/// Where model `m` lives under `models_dir`.
pub fn model_dir(models_dir: &Path, m: &ModelInfo) -> PathBuf {
    models_dir.join(m.id)
}

/// All files present with their catalogue sizes (contents were verified when downloaded).
pub fn installed(models_dir: &Path, m: &ModelInfo) -> bool {
    let d = model_dir(models_dir, m);
    m.files.iter().all(|f| std::fs::metadata(d.join(f.name)).is_ok_and(|md| md.len() == f.size))
}

/// Bytes still to download for `m` (files already present are skipped).
pub fn missing_bytes(models_dir: &Path, m: &ModelInfo) -> u64 {
    let d = model_dir(models_dir, m);
    m.files.iter().filter(|f| !std::fs::metadata(d.join(f.name)).is_ok_and(|md| md.len() == f.size)).map(|f| f.size).sum()
}

/// The tokenizer shared by the multilingual models has one checksum.
pub fn shared_tokenizer_sha() -> &'static str {
    TOKENIZER_MULTI
}

/// Download progress: `(bytes done, bytes total, file)`; return false to cancel.
#[cfg(feature = "download")]
pub type DownloadProgress<'a> = &'a mut dyn FnMut(u64, u64, &str) -> bool;

/// Download the missing files of `m` into `models_dir/<id>/`: each file streams into `<name>.part`,
/// its SHA-256 is checked, then it is renamed into place. A wrong checksum deletes the part file
/// and fails. Pure Rust TLS (rustls + RustCrypto) with the operating system's certificate
/// verifier.
#[cfg(feature = "download")]
pub fn download(models_dir: &Path, m: &ModelInfo, progress: DownloadProgress) -> Result<(), crate::SpeechError> {
    download_files(&model_dir(models_dir, m), m.files, progress)
}

/// [`download`] for any pinned file list (the neural voices use it too): missing files of
/// `files` go into `dir`, each verified against its SHA-256 and size before it is renamed into
/// place.
#[cfg(feature = "download")]
pub fn download_files(dir: &Path, files: &[ModelFile], progress: DownloadProgress) -> Result<(), crate::SpeechError> {
    use crate::SpeechError;
    use sha2::Digest;
    use std::io::{Read, Write};
    std::fs::create_dir_all(dir)?;
    let total: u64 = files.iter().filter(|f| !std::fs::metadata(dir.join(f.name)).is_ok_and(|md| md.len() == f.size)).map(|f| f.size).sum();
    let agent = agent();
    let mut done = 0u64;
    for f in files {
        let dest = dir.join(f.name);
        if std::fs::metadata(&dest).is_ok_and(|md| md.len() == f.size) {
            continue;
        }
        let part = dir.join(format!("{}.part", f.name));
        let resp = agent.get(f.url).call().map_err(|e| SpeechError::Download(format!("{}: {e}", f.url)))?;
        let mut body = resp.into_body();
        let mut reader = body.as_reader();
        let mut out = std::io::BufWriter::new(std::fs::File::create(&part)?);
        let mut hasher = sha2::Sha256::new();
        let mut buf = vec![0u8; 1 << 20];
        let mut got = 0u64;
        loop {
            let n = reader.read(&mut buf).map_err(|e| SpeechError::Download(format!("{}: {e}", f.name)))?;
            if n == 0 {
                break;
            }
            hasher.update(&buf[..n]);
            out.write_all(&buf[..n])?;
            got += n as u64;
            if !progress(done + got, total, f.name) {
                drop(out);
                let _ = std::fs::remove_file(&part);
                return Err(SpeechError::Cancelled);
            }
        }
        out.flush()?;
        drop(out);
        let sha: String = hasher.finalize().iter().map(|b| format!("{b:02x}")).collect();
        if sha != f.sha256 || got != f.size {
            let _ = std::fs::remove_file(&part);
            return Err(SpeechError::Download(format!("{}: checksum mismatch (got {sha}, {got} bytes; expected {}, {} bytes)", f.name, f.sha256, f.size)));
        }
        std::fs::rename(&part, &dest)?;
        done += f.size;
    }
    Ok(())
}

#[cfg(feature = "download")]
fn agent() -> ureq::Agent {
    let provider = std::sync::Arc::new(rustls_rustcrypto::provider());
    ureq::Agent::config_builder()
        .tls_config(
            ureq::tls::TlsConfig::builder()
                .provider(ureq::tls::TlsProvider::Rustls)
                .root_certs(ureq::tls::RootCerts::PlatformVerifier)
                .unversioned_rustls_crypto_provider(provider)
                .build(),
        )
        .timeout_connect(Some(std::time::Duration::from_secs(30)))
        .build()
        .into()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn catalogue_is_pinned_and_checksummed() {
        assert!(find(DEFAULT_MODEL).is_some());
        for m in CATALOGUE {
            assert!(m.size() > 100_000_000, "{}", m.id);
            for f in m.files {
                assert!(f.url.starts_with("https://huggingface.co/openai/whisper-"), "{}", f.url);
                assert!(f.url.contains("/resolve/") && f.url.ends_with(f.name));
                assert_eq!(f.sha256.len(), 64);
                assert!(f.sha256.bytes().all(|b| b.is_ascii_hexdigit()));
                if f.name == "tokenizer.json" {
                    assert_eq!(f.sha256, shared_tokenizer_sha());
                }
            }
        }
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn vulkan_catalogue_pins_large_v3_without_reordering_existing_models() {
        let models = vulkan_catalogue();
        assert_eq!(
            models.iter().map(|m| m.id).collect::<Vec<_>>(),
            vec!["whisper-tiny", "whisper-base", "whisper-small", "whisper-large-v3", "whisper-large-v3-q5_0", "whisper-large-v3-turbo"]
        );
        let model = models.iter().find(|m| m.id == "whisper-large-v3").expect("large-v3 metadata");
        let file = model.files.first().expect("large-v3 GGML file");
        assert_eq!(file.name, "ggml-large-v3.bin");
        assert_eq!(file.size, 3_095_033_483);
        assert_eq!(file.sha256, "64d182b440b98d5203c4f9bd541544d84c605196c4f7b845dfa11fb23594d1e2");
        assert_eq!(file.url, "https://huggingface.co/ggerganov/whisper.cpp/resolve/5359861c739e955e79d9a303bcbc70fb988958b1/ggml-large-v3.bin");
        assert!(model.multilingual);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn vulkan_catalogue_pins_multilingual_large_v3_turbo() {
        let model = vulkan_catalogue().last().expect("turbo metadata");
        let file = model.files.first().expect("turbo GGML file");
        assert_eq!(model.id, "whisper-large-v3-turbo");
        assert!(model.multilingual);
        assert_eq!(file.name, "ggml-large-v3-turbo.bin");
        assert_eq!(file.size, 1_624_555_275);
        assert_eq!(file.sha256, "1fc70f774d38eb169993ac391eea357ef47c88757ef72ee5943879b7e8e2bc69");
        assert_eq!(file.url, "https://huggingface.co/ggerganov/whisper.cpp/resolve/5359861c739e955e79d9a303bcbc70fb988958b1/ggml-large-v3-turbo.bin");
    }

    #[test]
    fn installed_checks_sizes() {
        let dir = std::env::temp_dir().join(format!("filmcraft-speech-models-{}", std::process::id()));
        let m = find("whisper-tiny").unwrap();
        assert!(!installed(&dir, m));
        assert_eq!(missing_bytes(&dir, m), m.size());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
