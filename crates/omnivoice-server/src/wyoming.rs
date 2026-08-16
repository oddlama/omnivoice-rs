use std::{io, sync::Arc};

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};
use tokio::{
    io::{AsyncBufRead, AsyncBufReadExt, AsyncReadExt, AsyncWrite, AsyncWriteExt, BufReader},
    net::{TcpListener, TcpStream},
    sync::watch,
    task::JoinSet,
};
use tracing::{debug, warn};

use crate::{
    runtime::{AppState, run_synthesis},
    voice::VoiceCatalog,
};

const MAX_HEADER_BYTES: usize = 64 * 1024;
const AUDIO_CHUNK_BYTES: usize = 8 * 1024;
const PROGRAM_NAME: &str = "omnivoice";

#[derive(Clone, Debug, PartialEq)]
pub struct WyomingEvent {
    pub event_type: String,
    pub data: Map<String, Value>,
    pub payload: Vec<u8>,
}

impl WyomingEvent {
    pub fn new(event_type: impl Into<String>) -> Self {
        Self {
            event_type: event_type.into(),
            data: Map::new(),
            payload: Vec::new(),
        }
    }

    pub fn with_data(mut self, data: Value) -> io::Result<Self> {
        self.data = data.as_object().cloned().ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "Wyoming event data must be an object",
            )
        })?;
        Ok(self)
    }

    pub fn with_payload(mut self, payload: Vec<u8>) -> Self {
        self.payload = payload;
        self
    }
}

#[derive(Debug, Deserialize)]
struct EventHeader {
    #[serde(rename = "type")]
    event_type: String,
    #[serde(default)]
    data: Map<String, Value>,
    #[serde(default)]
    data_length: Option<usize>,
    #[serde(default)]
    payload_length: Option<usize>,
}

#[derive(Debug, Serialize)]
struct OutgoingHeader<'a> {
    #[serde(rename = "type")]
    event_type: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    data_length: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    payload_length: Option<usize>,
}

pub async fn read_event<R>(
    reader: &mut R,
    max_data_bytes: usize,
) -> io::Result<Option<WyomingEvent>>
where
    R: AsyncBufRead + Unpin,
{
    let Some(header_bytes) = read_header_line(reader).await? else {
        return Ok(None);
    };
    let header: EventHeader = serde_json::from_slice(&header_bytes).map_err(invalid_data)?;
    if header.event_type.trim().is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Wyoming event type cannot be empty",
        ));
    }

    let data_length = header.data_length.unwrap_or(0);
    let payload_length = header.payload_length.unwrap_or(0);
    if data_length > max_data_bytes {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("Wyoming event data exceeds {max_data_bytes} bytes"),
        ));
    }
    if payload_length > max_data_bytes {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("Wyoming event payload exceeds {max_data_bytes} bytes"),
        ));
    }

    let mut data = header.data;
    if data_length > 0 {
        let mut bytes = vec![0_u8; data_length];
        reader.read_exact(&mut bytes).await?;
        let additional: Map<String, Value> =
            serde_json::from_slice(&bytes).map_err(invalid_data)?;
        data.extend(additional);
    }

    let mut payload = vec![0_u8; payload_length];
    if payload_length > 0 {
        reader.read_exact(&mut payload).await?;
    }

    Ok(Some(WyomingEvent {
        event_type: header.event_type,
        data,
        payload,
    }))
}

pub async fn write_event<W>(writer: &mut W, event: &WyomingEvent) -> io::Result<()>
where
    W: AsyncWrite + Unpin,
{
    let data = if event.data.is_empty() {
        None
    } else {
        Some(serde_json::to_vec(&event.data).map_err(invalid_data)?)
    };
    let header = OutgoingHeader {
        event_type: &event.event_type,
        data_length: data.as_ref().map(Vec::len),
        payload_length: (!event.payload.is_empty()).then_some(event.payload.len()),
    };
    let header = serde_json::to_vec(&header).map_err(invalid_data)?;
    writer.write_all(&header).await?;
    writer.write_all(b"\n").await?;
    if let Some(data) = data {
        writer.write_all(&data).await?;
    }
    if !event.payload.is_empty() {
        writer.write_all(&event.payload).await?;
    }
    writer.flush().await
}

pub async fn serve(
    listener: TcpListener,
    state: AppState,
    catalog: Arc<VoiceCatalog>,
    mut shutdown: watch::Receiver<bool>,
) -> io::Result<()> {
    let mut clients = JoinSet::new();
    loop {
        tokio::select! {
            changed = shutdown.changed() => {
                if changed.is_err() || *shutdown.borrow() {
                    break;
                }
            }
            accepted = listener.accept() => {
                let (stream, peer) = accepted?;
                let state = state.clone();
                let catalog = Arc::clone(&catalog);
                let client_shutdown = shutdown.clone();
                clients.spawn(async move {
                    debug!(%peer, "Wyoming client connected");
                    if let Err(error) = handle_connection(stream, state, catalog, client_shutdown).await {
                        debug!(%peer, %error, "Wyoming client disconnected with an error");
                    }
                });
            }
            completed = clients.join_next(), if !clients.is_empty() => {
                if let Some(Err(error)) = completed {
                    warn!(%error, "Wyoming client task failed");
                }
            }
        }
    }
    clients.abort_all();
    Ok(())
}

pub async fn handle_connection(
    stream: TcpStream,
    state: AppState,
    catalog: Arc<VoiceCatalog>,
    mut shutdown: watch::Receiver<bool>,
) -> io::Result<()> {
    let (read_half, mut write_half) = stream.into_split();
    let mut reader = BufReader::new(read_half);

    loop {
        let event = tokio::select! {
            changed = shutdown.changed() => {
                if changed.is_err() || *shutdown.borrow() {
                    return Ok(());
                }
                continue;
            }
            event = read_event(&mut reader, state.config.max_body_bytes) => event?,
        };
        let Some(event) = event else {
            return Ok(());
        };

        match event.event_type.as_str() {
            "describe" => write_event(&mut write_half, &info_event(&catalog)?).await?,
            "select-program" => {
                if event.data.get("name").and_then(Value::as_str) != Some(PROGRAM_NAME) {
                    write_error(
                        &mut write_half,
                        "unknown-program",
                        "unknown Wyoming program",
                    )
                    .await?;
                }
            }
            "synthesize" => {
                handle_synthesize(&mut write_half, &state, &catalog, &event).await?;
            }
            _ => debug!(event_type = %event.event_type, "ignoring unsupported Wyoming event"),
        }
    }
}

async fn handle_synthesize<W>(
    writer: &mut W,
    state: &AppState,
    catalog: &VoiceCatalog,
    event: &WyomingEvent,
) -> io::Result<()>
where
    W: AsyncWrite + Unpin,
{
    let Some(text) = event.data.get("text").and_then(Value::as_str) else {
        return write_error(writer, "invalid-request", "synthesize text is required").await;
    };
    if text.trim().is_empty() {
        return write_error(writer, "invalid-request", "synthesize text cannot be empty").await;
    }
    if event
        .data
        .get("text_format")
        .and_then(Value::as_str)
        .is_some_and(|format| !format.eq_ignore_ascii_case("text"))
    {
        return write_error(
            writer,
            "unsupported-text-format",
            "only plain text is supported",
        )
        .await;
    }

    let voice = event.data.get("voice").and_then(Value::as_object);
    let voice_name = voice
        .and_then(|voice| voice.get("name"))
        .and_then(Value::as_str);
    let voice_language = voice
        .and_then(|voice| voice.get("language"))
        .and_then(Value::as_str);
    let preset = match catalog.resolve(voice_name, voice_language) {
        Ok(preset) => preset,
        Err(error) => return write_error(writer, "unknown-voice", &error.to_string()).await,
    };
    let request = preset.generation_request(text, voice_language);
    let result = match run_synthesis(state, request, preset.seed).await {
        Ok(result) => result,
        Err(error) => return write_error(writer, "synthesis-error", &error.to_string()).await,
    };
    if let Err(error) = result.audio.validate() {
        return write_error(writer, "invalid-audio", &error.to_string()).await;
    }

    let sample_rate = result.audio.sample_rate;
    write_event(
        writer,
        &WyomingEvent::new("audio-start").with_data(json!({
            "rate": sample_rate,
            "width": 2,
            "channels": 1,
        }))?,
    )
    .await?;

    let pcm = encode_pcm16(&result.audio.samples);
    let bytes_per_chunk = AUDIO_CHUNK_BYTES - (AUDIO_CHUNK_BYTES % 2);
    for (index, chunk) in pcm.chunks(bytes_per_chunk).enumerate() {
        let sample_offset = index * (bytes_per_chunk / 2);
        let timestamp = sample_offset.saturating_mul(1000) / sample_rate as usize;
        write_event(
            writer,
            &WyomingEvent::new("audio-chunk")
                .with_data(json!({
                    "rate": sample_rate,
                    "width": 2,
                    "channels": 1,
                    "timestamp": timestamp,
                }))?
                .with_payload(chunk.to_vec()),
        )
        .await?;
    }

    let timestamp = result.audio.samples.len().saturating_mul(1000) / sample_rate as usize;
    write_event(
        writer,
        &WyomingEvent::new("audio-stop").with_data(json!({ "timestamp": timestamp }))?,
    )
    .await
}

fn info_event(catalog: &VoiceCatalog) -> io::Result<WyomingEvent> {
    let voices = catalog
        .voices()
        .map(|voice| {
            json!({
                "name": voice.name,
                "description": voice.description,
                "attribution": attribution(),
                "installed": true,
                "languages": voice.languages,
                "speakers": null,
                "version": null,
            })
        })
        .collect::<Vec<_>>();
    WyomingEvent::new("info").with_data(json!({
        "asr": [],
        "tts": [{
            "name": PROGRAM_NAME,
            "description": "OmniVoice text-to-speech",
            "attribution": attribution(),
            "installed": true,
            "version": env!("CARGO_PKG_VERSION"),
            "voices": voices,
            "supports_synthesize_streaming": false,
        }],
        "handle": [],
        "intent": [],
        "wake": [],
        "mic": [],
        "snd": [],
    }))
}

fn attribution() -> Value {
    json!({
        "name": "FerrisMind",
        "url": "https://github.com/FerrisMind/omnivoice-rs",
    })
}

async fn write_error<W>(writer: &mut W, code: &'static str, text: &str) -> io::Result<()>
where
    W: AsyncWrite + Unpin,
{
    write_event(
        writer,
        &WyomingEvent::new("error").with_data(json!({
            "text": text,
            "code": code,
        }))?,
    )
    .await
}

fn encode_pcm16(samples: &[f32]) -> Vec<u8> {
    samples
        .iter()
        .flat_map(|sample| {
            let sample = (sample.clamp(-1.0, 1.0) * i16::MAX as f32).round() as i16;
            sample.to_le_bytes()
        })
        .collect()
}

async fn read_header_line<R>(reader: &mut R) -> io::Result<Option<Vec<u8>>>
where
    R: AsyncBufRead + Unpin,
{
    let mut line = Vec::new();
    loop {
        let available = reader.fill_buf().await?;
        if available.is_empty() {
            return if line.is_empty() {
                Ok(None)
            } else {
                Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "Wyoming header ended before newline",
                ))
            };
        }
        let end = available
            .iter()
            .position(|byte| *byte == b'\n')
            .map_or(available.len(), |index| index + 1);
        if line.len().saturating_add(end) > MAX_HEADER_BYTES {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Wyoming header exceeds {MAX_HEADER_BYTES} bytes"),
            ));
        }
        line.extend_from_slice(&available[..end]);
        reader.consume(end);
        if line.last() == Some(&b'\n') {
            line.pop();
            if line.last() == Some(&b'\r') {
                line.pop();
            }
            return Ok(Some(line));
        }
    }
}

fn invalid_data(error: impl std::fmt::Display) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, error.to_string())
}

#[cfg(test)]
mod tests {
    use serde_json::json;
    use tokio::io::{AsyncWriteExt, BufReader, duplex};

    use super::{MAX_HEADER_BYTES, WyomingEvent, read_event, write_event};

    #[tokio::test]
    async fn codec_merges_inline_and_additional_data_and_reads_payload() {
        let (client, mut server) = duplex(1024);
        let writer = tokio::spawn(async move {
            server
                .write_all(
                    b"{\"type\":\"audio-chunk\",\"data\":{\"rate\":24000},\"data_length\":11,\"payload_length\":4}\n{\"width\":2}pcm!",
                )
                .await
                .unwrap();
        });
        let mut reader = BufReader::new(client);

        let event = read_event(&mut reader, 1024).await.unwrap().unwrap();
        assert_eq!(event.event_type, "audio-chunk");
        assert_eq!(event.data["rate"], 24_000);
        assert_eq!(event.data["width"], 2);
        assert_eq!(event.payload, b"pcm!");
        writer.await.unwrap();
    }

    #[tokio::test]
    async fn codec_writes_official_length_delimited_shape() {
        let (mut client, server) = duplex(1024);
        let event = WyomingEvent::new("audio-chunk")
            .with_data(json!({"rate": 24000, "width": 2, "channels": 1}))
            .unwrap()
            .with_payload(vec![1, 2, 3, 4]);
        let writer = tokio::spawn(async move { write_event(&mut client, &event).await.unwrap() });
        let mut reader = BufReader::new(server);

        let decoded = read_event(&mut reader, 1024).await.unwrap().unwrap();
        assert_eq!(decoded.event_type, "audio-chunk");
        assert_eq!(decoded.data["channels"], 1);
        assert_eq!(decoded.payload, [1, 2, 3, 4]);
        writer.await.unwrap();
    }

    #[tokio::test]
    async fn codec_rejects_oversized_header_and_payload() {
        let (client, mut server) = duplex(MAX_HEADER_BYTES * 2);
        let writer = tokio::spawn(async move {
            server
                .write_all(&vec![b'x'; MAX_HEADER_BYTES + 1])
                .await
                .unwrap();
        });
        let mut reader = BufReader::new(client);
        assert!(read_event(&mut reader, 1024).await.is_err());
        writer.abort();

        let (client, mut server) = duplex(1024);
        server
            .write_all(b"{\"type\":\"audio-chunk\",\"payload_length\":1025}\n")
            .await
            .unwrap();
        let mut reader = BufReader::new(client);
        assert!(read_event(&mut reader, 1024).await.is_err());
    }
}
