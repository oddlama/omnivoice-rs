use std::{
    fs,
    path::PathBuf,
    time::{SystemTime, UNIX_EPOCH},
};

use omnivoice_infer::{
    model_source::{
        classify_tts_model_source, load_official_omnivoice_manifest, manifest_download_targets,
        ModelSource, DEFAULT_OMNIVOICE_REPO,
    },
    Result,
};

fn unique_temp_dir(name: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    std::env::temp_dir().join(format!(
        "omnivoice-model-source-tests-{name}-{}-{nanos}",
        std::process::id()
    ))
}

#[test]
fn classify_tts_model_source_prefers_existing_local_directory() -> Result<()> {
    let root = unique_temp_dir("local-dir");
    fs::create_dir_all(&root)?;

    let resolved = classify_tts_model_source(Some(root.to_str().unwrap()))?;

    assert_eq!(resolved, ModelSource::LocalPath(root.clone()));

    fs::remove_dir_all(root)?;
    Ok(())
}

#[test]
fn classify_tts_model_source_uses_default_repo_when_omitted() -> Result<()> {
    let resolved = classify_tts_model_source(None)?;

    assert_eq!(
        resolved,
        ModelSource::HuggingFaceRepo(DEFAULT_OMNIVOICE_REPO.to_string())
    );
    Ok(())
}

#[test]
fn official_omnivoice_manifest_download_targets_match_expected_runtime_files() -> Result<()> {
    let manifest = load_official_omnivoice_manifest()?;

    let targets = manifest_download_targets(&manifest);

    assert_eq!(
        targets,
        vec![
            "config.json",
            "model.safetensors",
            "tokenizer.json",
            "tokenizer_config.json",
            "chat_template.jinja",
            "audio_tokenizer/config.json",
            "audio_tokenizer/model.safetensors",
            "audio_tokenizer/preprocessor_config.json",
            "audio_tokenizer/LICENSE",
        ]
    );
    Ok(())
}
