use agent_rs::agent::ToolCall;
use agent_rs::tools::ToolDefinition;
use agent_rs::voice::audio::{LinearResampler, PlaybackQueue};
use agent_rs::voice::realtime::{
    RealtimeEvent, build_realtime_request, decode_pcm16_base64, encode_pcm16_base64,
    function_call_output_event, parse_realtime_event, response_create_event, session_update_event,
};
use agent_rs::voice::session::talk_model_name;
use serde_json::json;

#[test]
fn realtime_audio_payloads_are_pcm16_little_endian_base64() {
    let samples = vec![-32_768, -1, 0, 1, 32_767];
    let encoded = encode_pcm16_base64(&samples);
    assert_eq!(decode_pcm16_base64(&encoded).unwrap(), samples);
}

#[test]
fn realtime_session_update_configures_voice_vad_and_interruption() {
    let config = test_config();
    let event = session_update_event(&config);

    assert_eq!(event["session"]["type"], "realtime");
    assert_eq!(event["session"]["model"], "gpt-realtime-2");
    assert_eq!(event["session"]["output_modalities"], json!(["audio"]));
    assert_eq!(
        event["session"]["audio"]["input"]["format"]["type"],
        "audio/pcm"
    );
    assert_eq!(event["session"]["audio"]["input"]["format"]["rate"], 24_000);
    assert_eq!(
        event["session"]["audio"]["output"]["format"]["type"],
        "audio/pcm"
    );
    assert_eq!(
        event["session"]["audio"]["output"]["format"]["rate"],
        24_000
    );
    assert_eq!(event["session"]["audio"]["output"]["voice"], "cedar");
    assert_eq!(event["session"]["audio"]["output"]["speed"], 1.15);
    assert_eq!(
        event["session"]["audio"]["input"]["turn_detection"]["type"],
        "server_vad"
    );
    assert_eq!(
        event["session"]["audio"]["input"]["turn_detection"]["interrupt_response"],
        true
    );
}

#[test]
fn realtime_events_drive_audio_transcripts_and_barge_in() {
    let encoded = encode_pcm16_base64(&[11, 12]);
    assert_eq!(
        parse_realtime_event(&json!({"type":"response.audio.delta", "delta": encoded}).to_string())
            .unwrap(),
        RealtimeEvent::AudioDelta(vec![11, 12])
    );
    assert_eq!(
        parse_realtime_event(r#"{"type":"input_audio_buffer.speech_started"}"#).unwrap(),
        RealtimeEvent::SpeechStarted
    );
    assert_eq!(
        parse_realtime_event(r#"{"type":"conversation.item.input_audio_transcription.completed","transcript":"hello"}"#).unwrap(),
        RealtimeEvent::UserTranscript("hello".to_string())
    );
}

#[test]
fn realtime_session_update_exposes_agent_tools() {
    let mut config = test_config();
    config.tools = vec![ToolDefinition {
        name: "read_file".to_string(),
        description: "Read a file".to_string(),
        parameters: json!({"type": "object"}),
    }];

    let event = session_update_event(&config);

    assert_eq!(event["session"]["tool_choice"], "auto");
    assert_eq!(event["session"]["tools"][0]["type"], "function");
    assert_eq!(event["session"]["tools"][0]["name"], "read_file");
}

#[test]
fn realtime_response_done_parses_function_calls() {
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
fn realtime_response_done_usage_contributes_to_cost_line() {
    let event = parse_realtime_event(
        &json!({
            "type": "response.done",
            "response": {
                "usage": {
                    "input_tokens": 100,
                    "output_tokens": 12,
                    "total_tokens": 112,
                    "total_cost": 1.2345
                }
            }
        })
        .to_string(),
    )
    .unwrap();

    let RealtimeEvent::ResponseDone { usage, .. } = event else {
        panic!("expected response done event");
    };

    let messages = vec![agent_rs::agent::AgentMessage::Assistant(
        agent_rs::agent::AssistantMessage {
            content: "hello".to_string(),
            tool_calls: Vec::new(),
            usage,
            metadata: Default::default(),
        },
    )];
    let line =
        agent_rs::providers::format_cost_and_context_line(&messages, "openai:gpt-realtime-2");

    assert!(line.contains("Cost: $1.2345"));
}

#[test]
fn realtime_tool_output_events_match_ga_shape() {
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
fn realtime_request_contains_openai_auth_and_model_query() {
    let request = build_realtime_request(&test_config()).unwrap();
    assert_eq!(
        request.headers().get("Authorization").unwrap(),
        "Bearer sk-test"
    );
    assert!(request.uri().to_string().contains("model=gpt-realtime-2"));
}

#[test]
fn audio_playback_queue_can_be_cleared_for_interruption() {
    let queue = PlaybackQueue::default();
    queue.push_pcm16(&[1, 2, 3]);
    assert_eq!(queue.len(), 3);
    queue.clear();
    assert_eq!(queue.len(), 0);
}

#[test]
fn resampler_handles_common_mac_sample_rate_to_realtime_rate() {
    let mut resampler = LinearResampler::new(48_000, 24_000);
    assert_eq!(resampler.process([1, 2, 3, 4, 5]), vec![0, 2, 4]);
}

#[test]
fn talk_mode_defaults_text_models_to_realtime_model() {
    assert_eq!(talk_model_name("gpt-5.5"), "gpt-realtime-2");
    assert_eq!(talk_model_name("openai:gpt-realtime-2"), "gpt-realtime-2");
}

fn test_config() -> agent_rs::voice::realtime::RealtimeConfig {
    agent_rs::voice::realtime::RealtimeConfig {
        model: "gpt-realtime-2".to_string(),
        api_key: "sk-test".to_string(),
        instructions: "be helpful".to_string(),
        voice: "cedar".to_string(),
        voice_speed: 1.15,
        base_url: "wss://api.openai.com/v1/realtime".to_string(),
        transcription_model: "gpt-4o-mini-transcribe".to_string(),
        tools: Vec::new(),
    }
}
