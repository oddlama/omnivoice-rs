use std::path::PathBuf;

use clap::Parser;
use omnivoice_infer::{
    model_source::resolve_tts_model_root_from_path, DTypeSpec, DeviceSpec, RuntimeOptions,
};
use shine_rs::SUPPORTED_BITRATES;

use crate::error::ServerError;

#[derive(Debug, Clone, Parser)]
#[command(name = "omnivoice-server", author = "FerrisMind")]
pub struct ServerArgs {
    #[arg(long = "model", alias = "model-dir")]
    pub model_dir: Option<PathBuf>,

    #[arg(long, default_value = "127.0.0.1")]
    pub host: String,

    #[arg(long, default_value_t = 8000)]
    pub port: u16,

    #[arg(long, default_value = "auto")]
    pub device: String,

    #[arg(long, default_value = "auto")]
    pub dtype: String,

    #[arg(long, default_value = "omnivoice-tts")]
    pub served_model_id: String,

    #[arg(long, env = "OMNIVOICE_API_KEY")]
    pub api_key: Option<String>,

    #[arg(long, default_value_t = 50)]
    pub max_body_mb: usize,

    #[arg(long, default_value_t = 1)]
    pub max_concurrent_requests: usize,

    #[arg(long, default_value_t = 128)]
    pub mp3_bitrate_kbps: u32,
}

impl ServerArgs {
    pub fn runtime_options(&self) -> Result<RuntimeOptions, ServerError> {
        let device = DeviceSpec::parse(&self.device).map_err(ServerError::from)?;
        let dtype = DTypeSpec::parse(&self.dtype).map_err(ServerError::from)?;
        let model_root = resolve_tts_model_root_from_path(self.model_dir.as_deref())
            .map_err(ServerError::from)?;

        Ok(RuntimeOptions::new(model_root)
            .with_device(device)
            .with_dtype(dtype))
    }

    pub fn validate(&self) -> Result<(), ServerError> {
        if self.served_model_id.trim().is_empty() {
            return Err(ServerError::validation("served model id cannot be empty"));
        }
        if self.max_body_mb == 0 {
            return Err(ServerError::validation(
                "max_body_mb must be greater than zero",
            ));
        }
        if self.max_concurrent_requests == 0 {
            return Err(ServerError::validation(
                "max_concurrent_requests must be greater than zero",
            ));
        }
        if !SUPPORTED_BITRATES.contains(&self.mp3_bitrate_kbps) {
            return Err(ServerError::validation(format!(
                "unsupported mp3 bitrate {}; supported values: {:?}",
                self.mp3_bitrate_kbps, SUPPORTED_BITRATES
            )));
        }
        if self
            .api_key
            .as_deref()
            .is_none_or(|value| value.trim().is_empty())
        {
            return Err(ServerError::validation(
                "api key must be provided via --api-key or OMNIVOICE_API_KEY",
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use clap::Parser;
    use std::path::PathBuf;

    use super::ServerArgs;

    #[test]
    fn model_dir_is_optional() {
        let args =
            ServerArgs::try_parse_from(["omnivoice-server", "--api-key", "test-key"]).unwrap();

        assert!(args.model_dir.is_none());
    }

    #[test]
    fn model_alias_is_accepted() {
        let args = ServerArgs::try_parse_from([
            "omnivoice-server",
            "--api-key",
            "test-key",
            "--model",
            "k2-fsa/OmniVoice",
        ])
        .unwrap();

        assert_eq!(args.model_dir, Some(PathBuf::from("k2-fsa/OmniVoice")));
    }
}
