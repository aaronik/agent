use base64::Engine;
use base64::engine::general_purpose::STANDARD;
use futures_util::{SinkExt, StreamExt};
use serde_json::{Map, Value, json};
use tokio::net::TcpStream;
use tokio_tungstenite::MaybeTlsStream;
use tokio_tungstenite::WebSocketStream;
use tokio_tungstenite::connect_async;
use tokio_tungstenite::tungstenite::client::IntoClientRequest;
use tokio_tungstenite::tungstenite::http::HeaderValue;
use tokio_tungstenite::tungstenite::protocol::Message;

use crate::agent::{AgentMessage, ToolCall, Usage};
use crate::tools::ToolDefinition;
use crate::voice::audio::REALTIME_SAMPLE_RATE;

const OPENAI_REALTIME_URL: &str = "wss://api.openai.com/v1/realtime";
const DEFAULT_VOICE: &str = "cedar";
const DEFAULT_VOICE_SPEED: f64 = 1.15;
const DEFAULT_TRANSCRIPTION_MODEL: &str = "gpt-4o-mini-transcribe";

pub struct RealtimeClient {
    socket: WebSocketStream<MaybeTlsStream<TcpStream>>,
}

impl RealtimeClient {
    pub async fn connect(config: &RealtimeConfig) -> Result<Self, RealtimeError> {
        let request = build_realtime_request(config)?;
        let (socket, _) = connect_async(request)
            .await
            .map_err(|err| RealtimeError::Connection(err.to_string()))?;
        let mut client = Self { socket };
        client.send_json(session_update_event(config)).await?;
        for item in conversation_item_create_events(&config.history) {
            client.send_json(item).await?;
        }
        Ok(client)
    }

    pub async fn append_input_audio(&mut self, samples: &[i16]) -> Result<(), RealtimeError> {
        if samples.is_empty() {
            return Ok(());
        }
        self.send_json(json!({
            "type": "input_audio_buffer.append",
            "audio": encode_pcm16_base64(samples)
        }))
        .await
    }

    pub async fn cancel_response(&mut self) -> Result<(), RealtimeError> {
        self.send_json(json!({ "type": "response.cancel" })).await
    }

    pub async fn send_function_call_output(
        &mut self,
        call_id: &str,
        output: &str,
    ) -> Result<(), RealtimeError> {
        self.send_json(function_call_output_event(call_id, output))
            .await
    }

    pub async fn create_response(&mut self) -> Result<(), RealtimeError> {
        self.send_json(response_create_event()).await
    }

    pub async fn next_event(&mut self) -> Result<Option<RealtimeEvent>, RealtimeError> {
        while let Some(message) = self.socket.next().await {
            let message = message.map_err(|err| RealtimeError::Connection(err.to_string()))?;
            match message {
                Message::Text(text) => return parse_realtime_event(&text).map(Some),
                Message::Close(_) => return Ok(None),
                Message::Ping(payload) => {
                    self.socket
                        .send(Message::Pong(payload))
                        .await
                        .map_err(|err| RealtimeError::Connection(err.to_string()))?;
                }
                _ => {}
            }
        }
        Ok(None)
    }

    async fn send_json(&mut self, value: Value) -> Result<(), RealtimeError> {
        self.socket
            .send(Message::Text(value.to_string().into()))
            .await
            .map_err(|err| RealtimeError::Connection(err.to_string()))
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct RealtimeConfig {
    pub model: String,
    pub api_key: String,
    pub instructions: String,
    pub voice: String,
    pub voice_speed: f64,
    pub base_url: String,
    pub transcription_model: String,
    pub tools: Vec<ToolDefinition>,
    pub history: Vec<AgentMessage>,
}

impl RealtimeConfig {
    pub fn new(model: String, api_key: String, instructions: String) -> Self {
        Self {
            model,
            api_key,
            instructions,
            voice: std::env::var("AGENT_VOICE")
                .ok()
                .filter(|voice| !voice.trim().is_empty())
                .unwrap_or_else(|| DEFAULT_VOICE.to_string()),
            voice_speed: std::env::var("AGENT_VOICE_SPEED")
                .ok()
                .and_then(|speed| speed.parse::<f64>().ok())
                .filter(|speed| (0.25..=4.0).contains(speed))
                .unwrap_or(DEFAULT_VOICE_SPEED),
            base_url: std::env::var("AGENT_REALTIME_URL")
                .ok()
                .filter(|url| !url.trim().is_empty())
                .unwrap_or_else(|| OPENAI_REALTIME_URL.to_string()),
            transcription_model: std::env::var("AGENT_TRANSCRIPTION_MODEL")
                .ok()
                .filter(|model| !model.trim().is_empty())
                .unwrap_or_else(|| DEFAULT_TRANSCRIPTION_MODEL.to_string()),
            tools: Vec::new(),
            history: Vec::new(),
        }
    }

    pub fn with_tools(mut self, tools: Vec<ToolDefinition>) -> Self {
        self.tools = tools;
        self
    }

    pub fn with_history(mut self, history: Vec<AgentMessage>) -> Self {
        self.history = history;
        self
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum RealtimeEvent {
    AudioDelta(Vec<i16>),
    UserTranscript(String),
    AssistantTranscript(String),
    SpeechStarted,
    ResponseDone {
        tool_calls: Vec<ToolCall>,
        usage: Option<Usage>,
    },
    Error(String),
    Other(String),
}

#[derive(Debug, thiserror::Error)]
pub enum RealtimeError {
    #[error("realtime connection failed: {0}")]
    Connection(String),
    #[error("realtime event was invalid: {0}")]
    InvalidEvent(String),
    #[error("realtime request was invalid: {0}")]
    InvalidRequest(String),
}

pub fn build_realtime_request(
    config: &RealtimeConfig,
) -> Result<tokio_tungstenite::tungstenite::handshake::client::Request, RealtimeError> {
    let mut url = url::Url::parse(config.base_url.trim_end_matches('/'))
        .map_err(|err| RealtimeError::InvalidRequest(err.to_string()))?;
    url.query_pairs_mut().append_pair("model", &config.model);
    let mut request = url
        .as_str()
        .into_client_request()
        .map_err(|err| RealtimeError::InvalidRequest(err.to_string()))?;
    let headers = request.headers_mut();
    headers.insert(
        "Authorization",
        HeaderValue::from_str(&format!("Bearer {}", config.api_key))
            .map_err(|err| RealtimeError::InvalidRequest(err.to_string()))?,
    );
    Ok(request)
}

pub fn session_update_event(config: &RealtimeConfig) -> Value {
    let mut session = json!({
        "type": "realtime",
        "model": config.model,
        "instructions": config.instructions,
        "output_modalities": ["audio"],
        "audio": {
            "input": {
                "format": {
                    "type": "audio/pcm",
                    "rate": REALTIME_SAMPLE_RATE
                },
                "transcription": {
                    "model": config.transcription_model
                },
                "turn_detection": {
                    "type": "server_vad",
                    "threshold": 0.5,
                    "prefix_padding_ms": 300,
                    "silence_duration_ms": 500,
                    "create_response": true,
                    "interrupt_response": true
                }
            },
            "output": {
                "format": {
                    "type": "audio/pcm",
                    "rate": REALTIME_SAMPLE_RATE
                },
                "voice": config.voice,
                "speed": config.voice_speed
            }
        }
    });

    if !config.tools.is_empty()
        && let Value::Object(session) = &mut session
    {
        session.insert(
            "tools".to_string(),
            Value::Array(realtime_tools(&config.tools)),
        );
        session.insert("tool_choice".to_string(), Value::String("auto".to_string()));
    }

    json!({
        "type": "session.update",
        "session": session
    })
}

pub fn function_call_output_event(call_id: &str, output: &str) -> Value {
    json!({
        "type": "conversation.item.create",
        "item": {
            "type": "function_call_output",
            "call_id": call_id,
            "output": output
        }
    })
}

pub fn conversation_item_create_events(messages: &[AgentMessage]) -> Vec<Value> {
    messages
        .iter()
        .flat_map(conversation_items_for_message)
        .map(|item| json!({ "type": "conversation.item.create", "item": item }))
        .collect()
}

fn conversation_items_for_message(message: &AgentMessage) -> Vec<Value> {
    match message {
        AgentMessage::User { content } if !content.trim().is_empty() => vec![json!({
            "type": "message",
            "role": "user",
            "content": [{ "type": "input_text", "text": content }]
        })],
        AgentMessage::Assistant(assistant) => {
            let mut items = Vec::new();
            if !assistant.content.trim().is_empty() {
                items.push(json!({
                    "type": "message",
                    "role": "assistant",
                    "content": [{ "type": "output_text", "text": assistant.content }]
                }));
            }
            items.extend(assistant.tool_calls.iter().map(|call| {
                json!({
                    "type": "function_call",
                    "call_id": call.id,
                    "name": call.name,
                    "arguments": call.arguments.to_string()
                })
            }));
            items
        }
        AgentMessage::Tool(result) => vec![json!({
            "type": "function_call_output",
            "call_id": result.tool_call_id,
            "output": result.content
        })],
        _ => Vec::new(),
    }
}

pub fn response_create_event() -> Value {
    json!({ "type": "response.create" })
}

pub fn parse_realtime_event(text: &str) -> Result<RealtimeEvent, RealtimeError> {
    let value: Value = serde_json::from_str(text)
        .map_err(|err| RealtimeError::InvalidEvent(format!("{err}: {text}")))?;
    let event_type = value
        .get("type")
        .and_then(Value::as_str)
        .unwrap_or_default();

    match event_type {
        "response.audio.delta" | "response.output_audio.delta" => {
            let delta = value
                .get("delta")
                .and_then(Value::as_str)
                .ok_or_else(|| RealtimeError::InvalidEvent("audio delta missing delta".into()))?;
            Ok(RealtimeEvent::AudioDelta(decode_pcm16_base64(delta)?))
        }
        "conversation.item.input_audio_transcription.completed" => {
            Ok(RealtimeEvent::UserTranscript(
                value
                    .get("transcript")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string(),
            ))
        }
        "response.audio_transcript.done" | "response.output_audio_transcript.done" => {
            Ok(RealtimeEvent::AssistantTranscript(
                value
                    .get("transcript")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string(),
            ))
        }
        "input_audio_buffer.speech_started" => Ok(RealtimeEvent::SpeechStarted),
        "response.done" => Ok(RealtimeEvent::ResponseDone {
            tool_calls: parse_response_done_tool_calls(&value),
            usage: parse_response_done_usage(&value),
        }),
        "error" => Ok(RealtimeEvent::Error(
            value
                .pointer("/error/message")
                .and_then(Value::as_str)
                .unwrap_or(text)
                .to_string(),
        )),
        other => Ok(RealtimeEvent::Other(other.to_string())),
    }
}

fn realtime_tools(tools: &[ToolDefinition]) -> Vec<Value> {
    tools
        .iter()
        .map(|tool| {
            json!({
                "type": "function",
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters
            })
        })
        .collect()
}

fn parse_response_done_tool_calls(value: &Value) -> Vec<ToolCall> {
    value
        .pointer("/response/output")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(parse_realtime_tool_call)
        .collect()
}

fn parse_response_done_usage(value: &Value) -> Option<Usage> {
    let usage = value.pointer("/response/usage")?;
    let input_tokens = usage
        .get("input_tokens")
        .or_else(|| usage.get("prompt_tokens"))
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let output_tokens = usage
        .get("output_tokens")
        .or_else(|| usage.get("completion_tokens"))
        .and_then(Value::as_u64)
        .unwrap_or(0);

    if input_tokens == 0 && output_tokens == 0 {
        return None;
    }

    Some(Usage {
        input_tokens,
        output_tokens,
        raw: Some(usage.clone()),
    })
}

fn parse_realtime_tool_call(value: &Value) -> Option<ToolCall> {
    if value.get("type").and_then(Value::as_str) != Some("function_call") {
        return None;
    }
    let id = value.get("call_id").and_then(Value::as_str)?.to_string();
    let name = value.get("name").and_then(Value::as_str)?.to_string();
    let arguments = value
        .get("arguments")
        .and_then(Value::as_str)
        .and_then(|raw| serde_json::from_str(raw).ok())
        .unwrap_or_else(|| Value::Object(Map::new()));
    Some(ToolCall {
        id,
        name,
        arguments,
    })
}

pub fn encode_pcm16_base64(samples: &[i16]) -> String {
    let mut bytes = Vec::with_capacity(samples.len() * 2);
    for sample in samples {
        bytes.extend(sample.to_le_bytes());
    }
    STANDARD.encode(bytes)
}

pub fn decode_pcm16_base64(encoded: &str) -> Result<Vec<i16>, RealtimeError> {
    let bytes = STANDARD
        .decode(encoded)
        .map_err(|err| RealtimeError::InvalidEvent(err.to_string()))?;
    if bytes.len() % 2 != 0 {
        return Err(RealtimeError::InvalidEvent(
            "pcm16 payload has odd byte length".to_string(),
        ));
    }
    Ok(bytes
        .chunks_exact(2)
        .map(|chunk| i16::from_le_bytes([chunk[0], chunk[1]]))
        .collect())
}

pub fn voice_instructions(base_system_prompt: &str) -> String {
    format!(
        "{base_system_prompt}\n\nYou are in a realtime voice session. Keep responses conversational and concise. You have access to the same tools as the text agent and should use them when helpful. For any substantial code changes, use the sub-agent via the agent command. The user may interrupt you at any time; stop cleanly and answer the latest utterance. Audio sample rate: {REALTIME_SAMPLE_RATE} Hz."
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> RealtimeConfig {
        RealtimeConfig {
            model: "gpt-realtime".to_string(),
            api_key: "sk-test".to_string(),
            instructions: "be helpful".to_string(),
            voice: "cedar".to_string(),
            voice_speed: DEFAULT_VOICE_SPEED,
            base_url: OPENAI_REALTIME_URL.to_string(),
            transcription_model: DEFAULT_TRANSCRIPTION_MODEL.to_string(),
            tools: Vec::new(),
            history: Vec::new(),
        }
    }

    #[test]
    fn pcm16_base64_round_trips_little_endian_samples() {
        let samples = vec![-32_768, -1, 0, 1, 32_767];
        let encoded = encode_pcm16_base64(&samples);
        assert_eq!(decode_pcm16_base64(&encoded).unwrap(), samples);
    }

    #[test]
    fn session_update_enables_server_vad_and_barge_in() {
        let event = session_update_event(&test_config());
        assert_eq!(event["session"]["type"], "realtime");
        assert_eq!(
            event["session"]["audio"]["input"]["turn_detection"]["type"],
            "server_vad"
        );
        assert_eq!(
            event["session"]["audio"]["input"]["turn_detection"]["interrupt_response"],
            true
        );
        assert_eq!(
            event["session"]["audio"]["input"]["format"]["type"],
            "audio/pcm"
        );
        assert_eq!(
            event["session"]["audio"]["input"]["format"]["rate"],
            REALTIME_SAMPLE_RATE
        );
        assert_eq!(
            event["session"]["audio"]["output"]["format"]["type"],
            "audio/pcm"
        );
        assert_eq!(
            event["session"]["audio"]["output"]["format"]["rate"],
            REALTIME_SAMPLE_RATE
        );
        assert_eq!(event["session"]["audio"]["output"]["voice"], "cedar");
        assert_eq!(
            event["session"]["audio"]["output"]["speed"],
            DEFAULT_VOICE_SPEED
        );
    }

    #[test]
    fn session_update_includes_realtime_function_tools() {
        let config = test_config().with_tools(vec![ToolDefinition {
            name: "read_file".to_string(),
            description: "Read a file".to_string(),
            parameters: json!({"type": "object"}),
        }]);

        let event = session_update_event(&config);

        assert_eq!(event["session"]["tool_choice"], "auto");
        assert_eq!(event["session"]["tools"][0]["type"], "function");
        assert_eq!(event["session"]["tools"][0]["name"], "read_file");
        assert_eq!(event["session"]["tools"][0]["parameters"]["type"], "object");
    }

    #[test]
    fn parses_speech_started_event_for_interruption() {
        assert_eq!(
            parse_realtime_event(r#"{"type":"input_audio_buffer.speech_started"}"#).unwrap(),
            RealtimeEvent::SpeechStarted
        );
    }

    #[test]
    fn parses_audio_delta_event() {
        let encoded = encode_pcm16_base64(&[7, 8, 9]);
        let event = parse_realtime_event(
            &json!({
                "type": "response.audio.delta",
                "delta": encoded
            })
            .to_string(),
        )
        .unwrap();
        assert_eq!(event, RealtimeEvent::AudioDelta(vec![7, 8, 9]));
    }

    #[test]
    fn parses_response_done_function_calls() {
        let event = parse_realtime_event(
            &json!({
                "type": "response.done",
                "response": {
                    "output": [{
                        "type": "function_call",
                        "name": "read_file",
                        "call_id": "call_1",
                        "arguments": "{\"path\":\"README.md\",\"intent\":\"inspect docs\"}"
                    }]
                }
            })
            .to_string(),
        )
        .unwrap();

        assert_eq!(
            event,
            RealtimeEvent::ResponseDone {
                tool_calls: vec![ToolCall {
                    id: "call_1".to_string(),
                    name: "read_file".to_string(),
                    arguments: json!({"path": "README.md", "intent": "inspect docs"}),
                }],
                usage: None,
            }
        );
    }

    #[test]
    fn function_call_output_and_response_create_events_match_ga_shape() {
        assert_eq!(
            function_call_output_event("call_1", "ok"),
            json!({
                "type": "conversation.item.create",
                "item": {
                    "type": "function_call_output",
                    "call_id": "call_1",
                    "output": "ok"
                }
            })
        );
        assert_eq!(response_create_event(), json!({"type": "response.create"}));
    }

    #[test]
    fn realtime_request_has_auth_header_and_model_query() {
        let request = build_realtime_request(&test_config()).unwrap();
        assert_eq!(
            request.headers().get("Authorization").unwrap(),
            "Bearer sk-test"
        );
        assert!(request.uri().to_string().contains("gpt-realtime"));
    }
}
