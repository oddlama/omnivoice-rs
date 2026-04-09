#![cfg(feature = "metal")]

mod support;

use candle_core::DType;
use omnivoice_infer::{
    artifacts::ReferenceArtifactBundle,
    contracts::{GenerationRequest, ReferenceAudioInput, VoiceClonePrompt},
    gpu_lock::acquire_gpu_test_lock,
    pipeline::Phase3Pipeline,
    runtime::{DTypeSpec, DeviceSpec, RuntimeOptions},
    workspace_phase_marker,
};
use support::{
    assert_audio_matches_reference_with_frame_tolerance, deterministic_reference_root,
    live_oracle_clone_prompt, model_root, ref_audio_path, reference_root,
};

fn metal_f32_pipeline() -> Phase3Pipeline {
    Phase3Pipeline::from_options(
        RuntimeOptions::new(model_root())
            .with_device(DeviceSpec::Metal)
            .with_dtype(DTypeSpec::F32)
            .with_seed(1234),
    )
    .unwrap()
}

fn auto_pipeline() -> Phase3Pipeline {
    Phase3Pipeline::from_options(
        RuntimeOptions::new(model_root())
            .with_device(DeviceSpec::Auto)
            .with_dtype(DTypeSpec::Auto)
            .with_seed(1234),
    )
    .unwrap()
}

#[test]
fn phase10_marker_is_current_metal() {
    assert_eq!(workspace_phase_marker(), "omnivoice-phase10");
}

#[test]
fn phase10_metal_auto_matches_reference_audio() {
    let _guard = acquire_gpu_test_lock().unwrap();
    let bundle = ReferenceArtifactBundle::from_root(deterministic_reference_root()).unwrap();
    let case = bundle.case_by_id("det_auto_en_short").unwrap();
    let request = case.build_generation_request().unwrap();
    let actual = metal_f32_pipeline().generate(&request).unwrap();
    let expected = case.load_final_audio().unwrap();
    assert_eq!(actual.len(), 1);
    assert_audio_matches_reference_with_frame_tolerance(
        &actual[0], &expected, 480, 2.0e-4, 3.0e-4, 5.0e-3,
    );
}

#[test]
fn phase10_metal_design_matches_reference_audio() {
    let _guard = acquire_gpu_test_lock().unwrap();
    let bundle = ReferenceArtifactBundle::from_root(deterministic_reference_root()).unwrap();
    let case = bundle.case_by_id("det_design_en_british").unwrap();
    let request = case.build_generation_request().unwrap();
    let actual = metal_f32_pipeline().generate(&request).unwrap();
    let expected = case.load_final_audio().unwrap();
    assert_audio_matches_reference_with_frame_tolerance(
        &actual[0], &expected, 480, 2.0e-3, 6.0e-3, 0.35,
    );
}

#[test]
fn phase10_metal_clone_with_asr_succeeds() {
    let _guard = acquire_gpu_test_lock().unwrap();
    let bundle = ReferenceArtifactBundle::from_root(reference_root()).unwrap();
    let case = bundle.case_by_id("clone_user_ref").unwrap();
    let definition = case.load_case_definition().unwrap();
    let mut request = GenerationRequest::new_text_only(definition.request.texts[0].clone());
    request.languages = definition.request.languages.clone();
    request.ref_audios = vec![Some(ReferenceAudioInput::from_path(
        ref_audio_path().display().to_string(),
    ))];
    request.ref_texts = vec![None];
    request.generation_config = definition.request.generation_config.clone();
    let actual = metal_f32_pipeline().generate(&request).unwrap();
    assert_eq!(actual.len(), 1);
    assert_eq!(actual[0].sample_rate, 24_000);
    assert!(!actual[0].samples.is_empty());
}

#[test]
fn phase10_metal_clone_with_prebuilt_prompt_succeeds() {
    let _guard = acquire_gpu_test_lock().unwrap();
    let bundle = ReferenceArtifactBundle::from_root(reference_root()).unwrap();
    let case = bundle.case_by_id("clone_user_ref").unwrap();
    let mut request = case.build_generation_request().unwrap();
    let (reference_tokens, reference_text) = live_oracle_clone_prompt();
    request.ref_audios = vec![None];
    request.ref_texts = vec![None];
    request.voice_clone_prompts = vec![Some(VoiceClonePrompt {
        ref_audio_tokens: reference_tokens,
        ref_text: reference_text,
        ref_rms: Some(0.1),
    })];
    let actual = metal_f32_pipeline().generate(&request).unwrap();
    assert_eq!(actual.len(), 1);
    assert_eq!(actual[0].sample_rate, 24_000);
    assert!(!actual[0].samples.is_empty());
}

#[test]
fn phase10_metal_long_form_auto_succeeds() {
    let _guard = acquire_gpu_test_lock().unwrap();
    let bundle = ReferenceArtifactBundle::from_root(deterministic_reference_root()).unwrap();
    let case = bundle.case_by_id("det_auto_long_chunked").unwrap();
    let request = case.build_generation_request().unwrap();
    let actual = metal_f32_pipeline().generate(&request).unwrap();
    assert_eq!(actual.len(), 1);
    assert_eq!(actual[0].sample_rate, 24_000);
    assert!(actual[0].frame_count() > 240_000);
    assert!(!actual[0].samples.is_empty());
}

#[test]
fn phase10_metal_batch_request_preserves_order() {
    let _guard = acquire_gpu_test_lock().unwrap();
    let bundle = ReferenceArtifactBundle::from_root(deterministic_reference_root()).unwrap();
    let auto_case = bundle.case_by_id("det_auto_en_short").unwrap();
    let auto_request = auto_case.build_generation_request().unwrap();
    let mut request = GenerationRequest::new_text_only(auto_request.texts[0].clone());
    request.texts = vec![auto_request.texts[0].clone(), auto_request.texts[0].clone()];
    request.languages = vec![
        auto_request.languages[0].clone(),
        auto_request.languages[0].clone(),
    ];
    request.instructs = vec![
        auto_request.instructs[0].clone(),
        auto_request.instructs[0].clone(),
    ];
    request.ref_texts = vec![None, None];
    request.ref_audios = vec![None, None];
    request.voice_clone_prompts = vec![None, None];
    request.speeds = vec![auto_request.speeds[0], auto_request.speeds[0]];
    request.durations = vec![auto_request.durations[0], auto_request.durations[0]];
    request.generation_config = auto_request.generation_config.clone();
    let actual = metal_f32_pipeline().generate(&request).unwrap();
    assert_eq!(actual.len(), 2);
    assert_eq!(actual[0].sample_rate, 24_000);
    assert_eq!(actual[1].sample_rate, 24_000);
    assert!(!actual[0].samples.is_empty());
    assert!(!actual[1].samples.is_empty());
}

#[test]
fn phase10_metal_auto_device_dtype_prioritize_gpu() {
    let _guard = acquire_gpu_test_lock().unwrap();
    let pipeline = auto_pipeline();
    assert!(pipeline.stage0().device().is_metal());
    assert_eq!(pipeline.stage0().runtime_dtype(), DType::F16);

    let bundle = ReferenceArtifactBundle::from_root(deterministic_reference_root()).unwrap();
    let case = bundle.case_by_id("det_auto_en_short").unwrap();
    let request = case.build_generation_request().unwrap();
    let actual = pipeline.generate(&request).unwrap();

    assert_eq!(actual.len(), 1);
    assert_eq!(actual[0].sample_rate, 24_000);
    assert!(!actual[0].samples.is_empty());
}
