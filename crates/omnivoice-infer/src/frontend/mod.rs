use std::{fs, path::Path};

use serde::Deserialize;
use tokenizers::Tokenizer;

use crate::{
    artifacts::RuntimeArtifacts,
    contracts::{
        GenerationRequest, GenerationTask, I64Tensor3, PreparedPrompt, PromptTensorBundle,
        VoiceClonePrompt,
    },
    error::{OmniVoiceError, Result},
    BoolTensor2,
};

mod duration;
mod language;
mod text;
mod voice_design;

pub use duration::RuleDurationEstimator;
pub use language::resolve_language;
pub use text::{add_punctuation, chunk_text_punctuation, combine_text};
pub use voice_design::{contains_cjk, resolve_instruct};

#[derive(Debug, Clone)]
pub struct Frontend {
    tokenizer: Tokenizer,
    num_audio_codebook: usize,
    audio_mask_id: i64,
    frame_rate: usize,
    duration_estimator: RuleDurationEstimator,
}

#[derive(Debug, Deserialize)]
struct ModelConfigFile {
    num_audio_codebook: usize,
    audio_mask_id: i64,
}

impl Frontend {
    pub fn from_model_root(model_root: impl AsRef<Path>) -> Result<Self> {
        let runtime = RuntimeArtifacts::from_model_root(model_root)?;
        Self::from_runtime_artifacts(&runtime)
    }

    pub fn from_runtime_artifacts(runtime: &RuntimeArtifacts) -> Result<Self> {
        let tokenizer = Tokenizer::from_file(runtime.text_tokenizer().tokenizer_path())?;
        let model_config: ModelConfigFile =
            serde_json::from_str(&fs::read_to_string(runtime.generator().config_path())?)?;
        Ok(Self {
            tokenizer,
            num_audio_codebook: model_config.num_audio_codebook,
            audio_mask_id: model_config.audio_mask_id,
            frame_rate: runtime.contracts().frame_rate,
            duration_estimator: RuleDurationEstimator::default(),
        })
    }

    pub fn build_task(&self, request: &GenerationRequest) -> Result<GenerationTask> {
        let batch_size = request.texts.len();
        let languages = normalize_option_list(&request.languages, batch_size, "languages")?
            .into_iter()
            .map(|item| resolve_language(item.as_deref()))
            .collect();
        let request_ref_texts = normalize_option_list(&request.ref_texts, batch_size, "ref_texts")?;
        let instructs_raw = normalize_option_list(&request.instructs, batch_size, "instructs")?;
        let voice_clone_prompts =
            normalize_option_voice_prompts(&request.voice_clone_prompts, batch_size)?;
        let requested_speeds = normalize_option_f32_list(&request.speeds, batch_size)?;
        let durations = normalize_option_f32_list(&request.durations, batch_size)?;

        let mut instructs = Vec::with_capacity(batch_size);
        for (index, item) in instructs_raw.into_iter().enumerate() {
            let use_zh = contains_cjk(&request.texts[index]);
            instructs.push(resolve_instruct(item.as_deref(), use_zh)?);
        }

        let mut ref_texts = Vec::with_capacity(batch_size);
        let mut ref_audio_tokens = Vec::with_capacity(batch_size);
        let mut ref_rms = Vec::with_capacity(batch_size);
        let mut target_lens = Vec::with_capacity(batch_size);
        let mut effective_speeds = Vec::with_capacity(batch_size);

        for (index, text) in request.texts.iter().enumerate() {
            let (reference_text, reference_audio_tokens, reference_rms) =
                if let Some(prompt) = voice_clone_prompts[index].clone() {
                    (
                        Some(prompt.ref_text),
                        Some(prompt.ref_audio_tokens),
                        prompt.ref_rms,
                    )
                } else {
                    (request_ref_texts[index].clone(), None, None)
                };

            let requested_speed = requested_speeds[index].unwrap_or(1.0);
            let estimated_target_length = self.estimate_target_tokens(
                text,
                reference_text.as_deref(),
                reference_audio_tokens
                    .as_ref()
                    .map(|tokens| tokens.dims().1),
                if durations[index].is_some() {
                    1.0
                } else {
                    requested_speed
                },
            );

            let (target_length, effective_speed) = if let Some(duration_seconds) = durations[index]
            {
                let target_tokens = (duration_seconds * self.frame_rate as f32).max(1.0) as usize;
                let speed = if target_tokens > 0 {
                    estimated_target_length as f32 / target_tokens as f32
                } else {
                    1.0
                };
                (target_tokens, speed)
            } else {
                (estimated_target_length, requested_speed)
            };

            ref_texts.push(reference_text);
            ref_audio_tokens.push(reference_audio_tokens);
            ref_rms.push(reference_rms);
            target_lens.push(target_length);
            effective_speeds.push(effective_speed);
        }

        Ok(GenerationTask {
            texts: request.texts.clone(),
            target_lens,
            langs: languages,
            instructs,
            ref_texts,
            ref_audio_tokens,
            ref_rms,
            speed: effective_speeds,
            generation_config: request.generation_config.clone(),
        })
    }

    pub fn prepare_prompt(&self, task: &GenerationTask, index: usize) -> Result<PreparedPrompt> {
        let text = task.texts.get(index).ok_or_else(|| {
            OmniVoiceError::InvalidRequest(format!("missing task item at {index}"))
        })?;
        let lang = task.langs.get(index).cloned().flatten();
        let instruct = task.instructs.get(index).cloned().flatten();
        let ref_text = task.ref_texts.get(index).cloned().flatten();
        let ref_audio_tokens = task.ref_audio_tokens.get(index).cloned().flatten();
        let target_length = *task.target_lens.get(index).ok_or_else(|| {
            OmniVoiceError::InvalidRequest(format!("missing target length at {index}"))
        })?;
        let mode = task.mode_for(index);

        let mut style_text = String::new();
        if task.generation_config.denoise && ref_audio_tokens.is_some() {
            style_text.push_str("<|denoise|>");
        }
        style_text.push_str(&format!(
            "<|lang_start|>{}<|lang_end|><|instruct_start|>{}<|instruct_end|>",
            lang.clone().unwrap_or_else(|| "None".to_string()),
            instruct.clone().unwrap_or_else(|| "None".to_string())
        ));
        let style_encoding = self.tokenizer.encode(style_text.clone(), false)?;
        let style_token_ids = style_encoding.get_ids().to_vec();

        let full_text = combine_text(text, ref_text.as_deref());
        let text_prompt = format!("<|text_start|>{full_text}<|text_end|>");
        let text_encoding = self.tokenizer.encode(text_prompt, false)?;
        let text_token_ids = text_encoding.get_ids().to_vec();

        let style_len = style_token_ids.len();
        let text_len = text_token_ids.len();
        let ref_audio_length = ref_audio_tokens
            .as_ref()
            .map(|tokens| tokens.dims().1)
            .unwrap_or(0);
        let total_length = style_len + text_len + ref_audio_length + target_length;
        let target_start_idx = style_len + text_len + ref_audio_length;

        let mut input_ids = I64Tensor3::full(
            (1, self.num_audio_codebook, total_length),
            self.audio_mask_id,
        );
        for codebook in 0..self.num_audio_codebook {
            for (position, token_id) in style_token_ids.iter().enumerate() {
                input_ids.set(0, codebook, position, i64::from(*token_id));
            }
            for (position, token_id) in text_token_ids.iter().enumerate() {
                input_ids.set(0, codebook, style_len + position, i64::from(*token_id));
            }
            if let Some(reference_tokens) = &ref_audio_tokens {
                for position in 0..ref_audio_length {
                    input_ids.set(
                        0,
                        codebook,
                        style_len + text_len + position,
                        reference_tokens.get(codebook, position),
                    );
                }
            }
        }

        let mut audio_mask = BoolTensor2::zeros((1, total_length));
        let audio_start_idx = total_length - target_length - ref_audio_length;
        for position in audio_start_idx..total_length {
            audio_mask.set(0, position, true);
        }

        Ok(PreparedPrompt {
            mode,
            style_text,
            full_text,
            style_token_ids,
            text_token_ids,
            prompt: PromptTensorBundle {
                input_ids,
                audio_mask,
            },
            target_start_idx,
            total_length,
            target_length,
            audio_mask_id: self.audio_mask_id,
        })
    }

    pub fn frame_rate(&self) -> usize {
        self.frame_rate
    }

    pub fn chunk_text(
        &self,
        text: &str,
        target_len: usize,
        audio_chunk_duration: f32,
    ) -> Vec<String> {
        let avg_tokens_per_char = target_len as f32 / text.chars().count().max(1) as f32;
        let text_chunk_len = ((audio_chunk_duration * self.frame_rate as f32) / avg_tokens_per_char)
            .max(1.0) as usize;
        chunk_text_punctuation(text, text_chunk_len, Some(3))
    }

    pub fn estimate_target_tokens(
        &self,
        text: &str,
        ref_text: Option<&str>,
        num_ref_audio_tokens: Option<usize>,
        speed: f32,
    ) -> usize {
        let reference_duration = num_ref_audio_tokens
            .map(|tokens| tokens as f32)
            .unwrap_or(25.0);
        let estimated = self.duration_estimator.estimate_duration(
            text,
            ref_text.unwrap_or("Nice to meet you."),
            reference_duration,
            Some(50.0),
            3.0,
        );
        let adjusted = if (speed - 1.0).abs() > f32::EPSILON {
            estimated / speed
        } else {
            estimated
        };
        adjusted.max(1.0) as usize
    }
}

fn normalize_option_list(
    values: &[Option<String>],
    batch_size: usize,
    name: &str,
) -> Result<Vec<Option<String>>> {
    if values.len() == batch_size {
        Ok(values.to_vec())
    } else if values.len() == 1 {
        Ok(vec![values[0].clone(); batch_size])
    } else {
        Err(OmniVoiceError::InvalidRequest(format!(
            "{name} should contain either 1 or {batch_size} items, got {}",
            values.len()
        )))
    }
}

fn normalize_option_f32_list(
    values: &[Option<f32>],
    batch_size: usize,
) -> Result<Vec<Option<f32>>> {
    if values.len() == batch_size {
        Ok(values.to_vec())
    } else if values.len() == 1 {
        Ok(vec![values[0]; batch_size])
    } else {
        Err(OmniVoiceError::InvalidRequest(format!(
            "duration list should contain either 1 or {batch_size} items, got {}",
            values.len()
        )))
    }
}

fn normalize_option_voice_prompts(
    values: &[Option<VoiceClonePrompt>],
    batch_size: usize,
) -> Result<Vec<Option<VoiceClonePrompt>>> {
    if values.len() == batch_size {
        Ok(values.to_vec())
    } else if values.len() == 1 {
        Ok(vec![values[0].clone(); batch_size])
    } else {
        Err(OmniVoiceError::InvalidRequest(format!(
            "voice_clone_prompts should contain either 1 or {batch_size} items, got {}",
            values.len()
        )))
    }
}
