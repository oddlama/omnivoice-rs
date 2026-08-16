use std::{
    path::Path,
    sync::{Arc, Mutex},
    time::Duration,
};

use omnivoice_infer::{
    GeneratedAudioResult,
    contracts::{DecodedAudio, GenerationRequest, GenerationUsage},
};
use omnivoice_server::{
    VoiceCatalog,
    runtime::{AppState, ServerConfig, SpeechRuntime},
    wyoming::{WyomingEvent, read_event, serve, write_event},
};
use serde_json::{Value, json};
use tokio::{
    io::BufReader,
    net::{TcpListener, TcpStream},
    sync::watch,
};

#[derive(Clone)]
struct RecordingRuntime {
    requests: Arc<Mutex<Vec<GenerationRequest>>>,
    seeds: Arc<Mutex<Vec<u64>>>,
}

impl RecordingRuntime {
    fn new() -> Self {
        Self {
            requests: Arc::new(Mutex::new(Vec::new())),
            seeds: Arc::new(Mutex::new(Vec::new())),
        }
    }

    fn requests(&self) -> Vec<GenerationRequest> {
        self.requests.lock().unwrap().clone()
    }

    fn seeds(&self) -> Vec<u64> {
        self.seeds.lock().unwrap().clone()
    }
}

impl SpeechRuntime for RecordingRuntime {
    fn synthesize(
        &self,
        request: GenerationRequest,
    ) -> omnivoice_infer::Result<GeneratedAudioResult> {
        self.requests.lock().unwrap().push(request);
        let samples = (0..5_000)
            .map(|index| ((index as f32 / 10.0).sin() * 0.25).clamp(-1.0, 1.0))
            .collect();
        Ok(GeneratedAudioResult {
            audio: DecodedAudio::new(samples, 24_000),
            usage: GenerationUsage::new(4, 8),
        })
    }

    fn set_seed(&self, seed: u64) -> omnivoice_infer::Result<()> {
        self.seeds.lock().unwrap().push(seed);
        Ok(())
    }
}

fn config() -> ServerConfig {
    ServerConfig {
        served_model_id: "omnivoice-tts".to_string(),
        api_key: "test-key".to_string(),
        base_path: String::new(),
        max_body_bytes: 1024 * 1024,
        max_concurrent_requests: 1,
        mp3_bitrate_kbps: 128,
        request_timeout: Duration::from_secs(10),
    }
}

fn catalog() -> Arc<VoiceCatalog> {
    Arc::new(
        VoiceCatalog::from_toml(
            r#"
                default_voice = "alice"

                [[voice.alice]]
                description = "Alice voice"
                languages = ["en-US"]
                language = "en"
                speed = 1.25
                seed = 42
                num_step = 16
                guidance_scale = 1.5

                [[voice.bob]]
                description = "Bob voice"
                languages = ["de-DE"]
                language = "de"
            "#,
            Path::new("."),
        )
        .unwrap(),
    )
}

async fn start_server(
    state: AppState,
) -> (
    TcpStream,
    watch::Sender<bool>,
    tokio::task::JoinHandle<std::io::Result<()>>,
) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let (shutdown_tx, shutdown_rx) = watch::channel(false);
    let handle = tokio::spawn(serve(listener, state, catalog(), shutdown_rx));
    let client = TcpStream::connect(address).await.unwrap();
    (client, shutdown_tx, handle)
}

#[tokio::test]
async fn advertises_catalog_and_synthesizes_selected_voice_as_pcm() {
    let runtime = RecordingRuntime::new();
    let (client, shutdown, server) = start_server(AppState::new(runtime.clone(), config())).await;
    let (read_half, mut write_half) = client.into_split();
    let mut reader = BufReader::new(read_half);

    write_event(&mut write_half, &WyomingEvent::new("describe"))
        .await
        .unwrap();
    let info = read_event(&mut reader, 1024 * 1024).await.unwrap().unwrap();
    assert_eq!(info.event_type, "info");
    assert_eq!(info.data["tts"][0]["name"], "omnivoice");
    assert_eq!(info.data["tts"][0]["supports_synthesize_streaming"], false);
    let voices = info.data["tts"][0]["voices"].as_array().unwrap();
    assert_eq!(voices.len(), 2);
    assert!(
        voices
            .iter()
            .any(|voice| voice["name"] == "alice" && voice["languages"] == json!(["en-US"]))
    );

    let synthesize = WyomingEvent::new("synthesize")
        .with_data(json!({
            "text": "Hello from Home Assistant",
            "voice": {"name": "alice"},
        }))
        .unwrap();
    write_event(&mut write_half, &synthesize).await.unwrap();

    let start = read_event(&mut reader, 1024 * 1024).await.unwrap().unwrap();
    assert_eq!(start.event_type, "audio-start");
    assert_eq!(start.data["rate"], 24_000);
    assert_eq!(start.data["width"], 2);
    assert_eq!(start.data["channels"], 1);

    let mut pcm = Vec::new();
    loop {
        let event = read_event(&mut reader, 1024 * 1024).await.unwrap().unwrap();
        match event.event_type.as_str() {
            "audio-chunk" => {
                assert_eq!(event.data["rate"], 24_000);
                pcm.extend(event.payload);
            }
            "audio-stop" => {
                assert_eq!(event.data["timestamp"], 208);
                break;
            }
            other => panic!("unexpected Wyoming event: {other}"),
        }
    }
    assert_eq!(pcm.len(), 5_000 * 2);

    let requests = runtime.requests();
    assert_eq!(requests.len(), 1);
    assert_eq!(requests[0].texts, ["Hello from Home Assistant"]);
    assert_eq!(requests[0].languages[0].as_deref(), Some("en"));
    assert_eq!(requests[0].speeds, [Some(1.25)]);
    assert_eq!(requests[0].generation_config.num_step, 16);
    assert_eq!(requests[0].generation_config.guidance_scale, 1.5);
    assert_eq!(runtime.seeds(), [42]);

    shutdown.send(true).unwrap();
    server.await.unwrap().unwrap();
}

#[tokio::test]
async fn returns_protocol_errors_for_unknown_voices_and_starting_runtime() {
    let runtime = RecordingRuntime::new();
    let (client, shutdown, server) = start_server(AppState::new(runtime.clone(), config())).await;
    let (read_half, mut write_half) = client.into_split();
    let mut reader = BufReader::new(read_half);
    let request = WyomingEvent::new("synthesize")
        .with_data(json!({"text": "hello", "voice": {"name": "missing"}}))
        .unwrap();
    write_event(&mut write_half, &request).await.unwrap();

    let error = read_event(&mut reader, 1024 * 1024).await.unwrap().unwrap();
    assert_eq!(error.event_type, "error");
    assert_eq!(error.data["code"], "unknown-voice");
    assert!(
        error.data["text"]
            .as_str()
            .unwrap()
            .contains("not configured")
    );
    assert!(runtime.requests().is_empty());
    shutdown.send(true).unwrap();
    server.await.unwrap().unwrap();

    let (client, shutdown, server) = start_server(AppState::starting(config())).await;
    let (read_half, mut write_half) = client.into_split();
    let mut reader = BufReader::new(read_half);
    let request = WyomingEvent::new("synthesize")
        .with_data(json!({"text": "hello"}))
        .unwrap();
    write_event(&mut write_half, &request).await.unwrap();
    let error = read_event(&mut reader, 1024 * 1024).await.unwrap().unwrap();
    assert_eq!(error.event_type, "error");
    assert_eq!(error.data["code"], "synthesis-error");
    assert_eq!(
        error.data["text"],
        Value::String("runtime is not ready".into())
    );
    shutdown.send(true).unwrap();
    server.await.unwrap().unwrap();
}
