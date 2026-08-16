use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
};

use omnivoice_infer::contracts::{GenerationConfig, GenerationRequest, ReferenceAudioInput};
use serde::Deserialize;

use crate::{audio::parse_data_uri_audio, error::ServerError};

const DEFAULT_LANGUAGE: &str = "en";

#[derive(Clone, Debug)]
pub struct VoiceCatalog {
    voices: BTreeMap<String, VoicePreset>,
    default_voice: String,
}

#[derive(Clone, Debug)]
pub struct VoicePreset {
    pub name: String,
    pub description: String,
    pub languages: Vec<String>,
    pub language: Option<String>,
    pub duration: Option<f32>,
    pub speed: Option<f32>,
    pub reference_audio: Option<ReferenceAudioInput>,
    pub reference_text: Option<String>,
    pub instruct: Option<String>,
    pub asr_model: Option<String>,
    pub seed: Option<u64>,
    pub generation_config: GenerationConfig,
    pub response_format: Option<String>,
    pub stream_format: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct VoiceCatalogFile {
    #[serde(default)]
    default_voice: Option<String>,
    #[serde(default = "default_languages")]
    languages: Vec<String>,
    voice: BTreeMap<String, Vec<VoicePresetFile>>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct VoicePresetFile {
    #[serde(default)]
    description: Option<String>,
    #[serde(default)]
    languages: Option<Vec<String>>,
    #[serde(default)]
    language: Option<String>,
    #[serde(default)]
    duration: Option<f32>,
    #[serde(default)]
    speed: Option<f32>,
    #[serde(default)]
    reference: Option<PathBuf>,
    #[serde(default)]
    ref_audio: Option<String>,
    #[serde(default, alias = "ref_text")]
    reference_text: Option<String>,
    #[serde(default)]
    instruct: Option<String>,
    #[serde(default)]
    asr_model: Option<String>,
    #[serde(default)]
    seed: Option<u64>,
    #[serde(default)]
    num_step: Option<usize>,
    #[serde(default)]
    guidance_scale: Option<f32>,
    #[serde(default)]
    t_shift: Option<f32>,
    #[serde(default)]
    layer_penalty_factor: Option<f32>,
    #[serde(default)]
    position_temperature: Option<f32>,
    #[serde(default)]
    class_temperature: Option<f32>,
    #[serde(default)]
    preprocess_prompt: Option<bool>,
    #[serde(default)]
    postprocess_output: Option<bool>,
    #[serde(default)]
    denoise: Option<bool>,
    #[serde(default)]
    audio_chunk_duration: Option<f32>,
    #[serde(default)]
    audio_chunk_threshold: Option<f32>,
    #[serde(default)]
    response_format: Option<String>,
    #[serde(default)]
    stream_format: Option<String>,
}

impl VoiceCatalog {
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self, ServerError> {
        let path = path.as_ref();
        let source = fs::read_to_string(path).map_err(|error| {
            ServerError::validation(format!(
                "failed to read voice config {}: {error}",
                path.display()
            ))
        })?;
        let base_dir = path.parent().unwrap_or_else(|| Path::new("."));
        Self::from_toml(&source, base_dir)
    }

    pub fn from_toml(source: &str, base_dir: &Path) -> Result<Self, ServerError> {
        let parsed: VoiceCatalogFile = toml::from_str(source)
            .map_err(|error| ServerError::validation(format!("invalid voice config: {error}")))?;
        validate_languages(&parsed.languages, "languages")?;
        if parsed.voice.is_empty() {
            return Err(ServerError::validation(
                "voice config must define at least one voice",
            ));
        }

        let mut voices = BTreeMap::new();
        for (raw_name, entries) in parsed.voice {
            let name = non_empty(raw_name, "voice name")?;
            if entries.len() != 1 {
                return Err(ServerError::validation(format!(
                    "voice `{name}` must have exactly one [[voice.{name}]] table"
                )));
            }
            let entry = entries.into_iter().next().expect("entry count checked");
            let preset = VoicePreset::from_file(name.clone(), entry, &parsed.languages, base_dir)?;
            voices.insert(name, preset);
        }

        let default_voice = match parsed.default_voice {
            Some(name) => {
                let name = non_empty(name, "default_voice")?;
                if !voices.contains_key(&name) {
                    return Err(ServerError::validation(format!(
                        "default_voice `{name}` is not defined"
                    )));
                }
                name
            }
            None => voices.keys().next().expect("voices are not empty").clone(),
        };

        Ok(Self {
            voices,
            default_voice,
        })
    }

    pub fn default_voice(&self) -> &str {
        &self.default_voice
    }

    pub fn voices(&self) -> impl Iterator<Item = &VoicePreset> {
        self.voices.values()
    }

    pub fn get(&self, name: &str) -> Option<&VoicePreset> {
        self.voices.get(name)
    }

    pub fn resolve(
        &self,
        name: Option<&str>,
        language: Option<&str>,
    ) -> Result<&VoicePreset, ServerError> {
        if let Some(name) = name.filter(|name| !name.trim().is_empty()) {
            return self.get(name).ok_or_else(|| {
                ServerError::validation(format!("voice `{name}` is not configured"))
            });
        }
        if let Some(language) = language.filter(|language| !language.trim().is_empty()) {
            if let Some(voice) = self
                .voices()
                .find(|voice| voice.languages.iter().any(|item| item == language))
            {
                return Ok(voice);
            }
        }
        self.get(&self.default_voice)
            .ok_or_else(|| ServerError::internal("default voice is missing"))
    }
}

impl VoicePreset {
    fn from_file(
        name: String,
        file: VoicePresetFile,
        catalog_languages: &[String],
        base_dir: &Path,
    ) -> Result<Self, ServerError> {
        let description = file.description.unwrap_or_else(|| name.clone());
        let description = non_empty(description, &format!("voice `{name}` description"))?;
        let languages = file.languages.unwrap_or_else(|| catalog_languages.to_vec());
        validate_languages(&languages, &format!("voice `{name}` languages"))?;
        let language = optional_non_empty(file.language, &format!("voice `{name}` language"))?;
        validate_positive(file.duration, &format!("voice `{name}` duration"))?;
        validate_positive(file.speed, &format!("voice `{name}` speed"))?;

        let reference_audio = match (file.reference, file.ref_audio) {
            (Some(_), Some(_)) => {
                return Err(ServerError::validation(format!(
                    "voice `{name}` cannot set both reference and ref_audio"
                )));
            }
            (Some(path), None) => {
                let path = if path.is_absolute() {
                    path
                } else {
                    base_dir.join(path)
                };
                if !path.is_file() {
                    return Err(ServerError::validation(format!(
                        "voice `{name}` reference audio does not exist or is not a file: {}",
                        path.display()
                    )));
                }
                Some(ReferenceAudioInput::from_path(path))
            }
            (None, Some(data_uri)) => {
                Some(parse_data_uri_audio(&serde_json::Value::String(data_uri))?)
            }
            (None, None) => None,
        };
        if file.reference_text.is_some() && reference_audio.is_none() {
            return Err(ServerError::validation(format!(
                "voice `{name}` sets reference_text without reference audio"
            )));
        }

        let mut generation_config = GenerationConfig::default();
        apply(file.num_step, &mut generation_config.num_step);
        apply(file.guidance_scale, &mut generation_config.guidance_scale);
        apply(file.t_shift, &mut generation_config.t_shift);
        apply(
            file.layer_penalty_factor,
            &mut generation_config.layer_penalty_factor,
        );
        apply(
            file.position_temperature,
            &mut generation_config.position_temperature,
        );
        apply(
            file.class_temperature,
            &mut generation_config.class_temperature,
        );
        apply(file.denoise, &mut generation_config.denoise);
        apply(
            file.preprocess_prompt,
            &mut generation_config.preprocess_prompt,
        );
        apply(
            file.postprocess_output,
            &mut generation_config.postprocess_output,
        );
        apply(
            file.audio_chunk_duration,
            &mut generation_config.audio_chunk_duration,
        );
        apply(
            file.audio_chunk_threshold,
            &mut generation_config.audio_chunk_threshold,
        );
        crate::audio::validate_generation_config(&generation_config)?;

        let response_format = normalize_choice(
            file.response_format,
            &format!("voice `{name}` response_format"),
            &["wav", "pcm", "mp3"],
        )?;
        let stream_format = normalize_choice(
            file.stream_format,
            &format!("voice `{name}` stream_format"),
            &["audio", "sse"],
        )?;

        Ok(Self {
            name,
            description,
            languages,
            language,
            duration: file.duration,
            speed: file.speed,
            reference_audio,
            reference_text: file.reference_text,
            instruct: optional_non_empty(file.instruct, "instruct")?,
            asr_model: optional_non_empty(file.asr_model, "asr_model")?,
            seed: file.seed,
            generation_config,
            response_format,
            stream_format,
        })
    }

    pub fn generation_request(
        &self,
        text: impl Into<String>,
        requested_language: Option<&str>,
    ) -> GenerationRequest {
        let mut request = GenerationRequest::new_text_only(text);
        request.languages = vec![
            self.language
                .clone()
                .or_else(|| requested_language.map(ToOwned::to_owned)),
        ];
        request.durations = vec![self.duration];
        request.speeds = vec![self.speed];
        request.ref_audios = vec![self.reference_audio.clone()];
        request.ref_texts = vec![self.reference_text.clone()];
        request.instructs = vec![self.instruct.clone()];
        request.asr_model = self.asr_model.clone();
        request.generation_config = self.generation_config.clone();
        request
    }
}

fn default_languages() -> Vec<String> {
    vec![DEFAULT_LANGUAGE.to_string()]
}

fn apply<T>(value: Option<T>, target: &mut T) {
    if let Some(value) = value {
        *target = value;
    }
}

fn validate_languages(values: &[String], field: &str) -> Result<(), ServerError> {
    if values.is_empty() {
        return Err(ServerError::validation(format!(
            "{field} must contain at least one language"
        )));
    }
    if values.iter().any(|value| value.trim().is_empty()) {
        return Err(ServerError::validation(format!(
            "{field} cannot contain an empty language"
        )));
    }
    Ok(())
}

fn validate_positive(value: Option<f32>, field: &str) -> Result<(), ServerError> {
    if value.is_some_and(|value| !value.is_finite() || value <= 0.0) {
        return Err(ServerError::validation(format!(
            "{field} must be finite and greater than zero"
        )));
    }
    Ok(())
}

fn non_empty(value: String, field: &str) -> Result<String, ServerError> {
    if value.trim().is_empty() {
        return Err(ServerError::validation(format!("{field} cannot be empty")));
    }
    Ok(value)
}

fn optional_non_empty(value: Option<String>, field: &str) -> Result<Option<String>, ServerError> {
    value.map(|value| non_empty(value, field)).transpose()
}

fn normalize_choice(
    value: Option<String>,
    field: &str,
    choices: &[&str],
) -> Result<Option<String>, ServerError> {
    value
        .map(|value| {
            let value = value.trim().to_ascii_lowercase();
            if choices.contains(&value.as_str()) {
                Ok(value)
            } else {
                Err(ServerError::validation(format!(
                    "unsupported {field} `{value}`; expected one of {}",
                    choices.join(", ")
                )))
            }
        })
        .transpose()
}

#[cfg(test)]
mod tests {
    use std::{
        fs,
        path::Path,
        time::{SystemTime, UNIX_EPOCH},
    };

    use omnivoice_infer::contracts::ReferenceAudioInput;

    use super::VoiceCatalog;

    fn temp_dir() -> std::path::PathBuf {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("omnivoice-voices-{unique}"));
        fs::create_dir_all(&path).unwrap();
        path
    }

    #[test]
    fn parses_requested_array_of_tables_and_all_generation_fields() {
        let dir = temp_dir();
        fs::write(dir.join("alice.wav"), b"placeholder").unwrap();
        let source = r#"
            default_voice = "alice"
            languages = ["en-US", "de-DE"]

            [[voice.alice]]
            description = "Alice"
            languages = ["en-US"]
            reference = "./alice.wav"
            reference_text = "Reference words."
            language = "en"
            duration = 2.5
            speed = 1.25
            instruct = "friendly"
            asr_model = "whisper"
            seed = 42
            num_step = 16
            guidance_scale = 1.5
            t_shift = 0.2
            layer_penalty_factor = 4.0
            position_temperature = 3.0
            class_temperature = 0.5
            preprocess_prompt = false
            postprocess_output = false
            denoise = false
            audio_chunk_duration = 10.0
            audio_chunk_threshold = 20.0
            response_format = "PCM"
            stream_format = "audio"
        "#;

        let catalog = VoiceCatalog::from_toml(source, &dir).unwrap();
        let voice = catalog.get("alice").unwrap();
        assert_eq!(catalog.default_voice(), "alice");
        assert_eq!(voice.description, "Alice");
        assert_eq!(voice.languages, ["en-US"]);
        assert_eq!(voice.seed, Some(42));
        assert_eq!(voice.response_format.as_deref(), Some("pcm"));
        assert_eq!(voice.stream_format.as_deref(), Some("audio"));
        assert!(matches!(
            voice.reference_audio,
            Some(ReferenceAudioInput::FilePath(_))
        ));

        let request = voice.generation_request("hello", Some("de"));
        assert_eq!(request.texts, ["hello"]);
        assert_eq!(request.languages[0].as_deref(), Some("en"));
        assert_eq!(request.durations, [Some(2.5)]);
        assert_eq!(request.speeds, [Some(1.25)]);
        assert_eq!(request.ref_texts[0].as_deref(), Some("Reference words."));
        assert_eq!(request.instructs[0].as_deref(), Some("friendly"));
        assert_eq!(request.asr_model.as_deref(), Some("whisper"));
        assert_eq!(request.generation_config.num_step, 16);
        assert_eq!(request.generation_config.guidance_scale, 1.5);
        assert!(!request.generation_config.preprocess_prompt);
        assert!(!request.generation_config.postprocess_output);
        assert!(!request.generation_config.denoise);

        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn defaults_to_first_voice_and_default_language() {
        let source = r#"
            [[voice.alice]]
            speed = 1.1

            [[voice.bob]]
            language = "de"
        "#;
        let catalog = VoiceCatalog::from_toml(source, Path::new(".")).unwrap();

        assert_eq!(catalog.default_voice(), "alice");
        assert_eq!(catalog.get("alice").unwrap().languages, ["en"]);
        assert_eq!(catalog.resolve(None, Some("de")).unwrap().name, "alice");
        assert_eq!(catalog.resolve(Some("bob"), None).unwrap().name, "bob");
    }

    #[test]
    fn resolves_by_advertised_language_before_default() {
        let source = r#"
            default_voice = "alice"

            [[voice.alice]]
            languages = ["en-US"]

            [[voice.bob]]
            languages = ["de-DE"]
            language = "de"
        "#;
        let catalog = VoiceCatalog::from_toml(source, Path::new(".")).unwrap();

        assert_eq!(catalog.resolve(None, Some("de-DE")).unwrap().name, "bob");
    }

    #[test]
    fn rejects_duplicate_tables_for_a_voice() {
        let source = r#"
            [[voice.alice]]
            [[voice.alice]]
        "#;
        let error = VoiceCatalog::from_toml(source, Path::new(".")).unwrap_err();
        assert!(format!("{error:?}").contains("exactly one"));
    }

    #[test]
    fn rejects_unknown_fields_and_missing_references() {
        let unknown = r#"
            [[voice.alice]]
            speeed = 2.0
        "#;
        assert!(VoiceCatalog::from_toml(unknown, Path::new(".")).is_err());

        let missing = r#"
            [[voice.alice]]
            reference = "missing.wav"
        "#;
        assert!(VoiceCatalog::from_toml(missing, Path::new(".")).is_err());
    }
}
