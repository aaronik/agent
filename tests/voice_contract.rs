use agent_rs::agent::{AgentMessage, AssistantMessage, ToolCall, ToolResult, ToolStatus};
use agent_rs::tools::ToolDefinition;
use agent_rs::voice::audio::{LinearResampler, PlaybackQueue};
use agent_rs::voice::realtime::{
    RealtimeEvent, build_realtime_request, conversation_item_create_events, decode_pcm16_base64,
    encode_pcm16_base64, function_call_output_event, parse_realtime_event, response_create_event,
    session_update_event,
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
    assert_eq!(event["session"]["model"], "gpt-realtime");
    assert_eq!(event["session"]["output_modalities"], json!(["audio"]));
    assert!(
        event["session"]["instructions"]
            .as_str()
            .expect("instructions")
            .contains("Use low verbosity")
    );
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
fn wake_word_session_waits_for_explicit_response() {
    let config = test_config().with_wake_word("Computer".to_string());
    let event = session_update_event(&config);
    let detection = &event["session"]["audio"]["input"]["turn_detection"];
    assert_eq!(detection["create_response"], false);
    assert_eq!(detection["interrupt_response"], false);
    assert_eq!(
        session_update_event(&test_config())["session"]["audio"]["input"]["turn_detection"]["create_response"],
        true
    );
}

#[test]
fn transcription_contains_item_id_for_ignored_turn_deletion() {
    assert_eq!(
        parse_realtime_event(r#"{"type":"conversation.item.input_audio_transcription.completed","item_id":"item_42","transcript":"Computer, status?"}"#).unwrap(),
        RealtimeEvent::UserTranscript { transcript: "Computer, status?".to_string(), item_id: "item_42".to_string() }
    );
}

#[test]
fn realtime_session_instructions_include_agents_memory_and_working_directory() {
    let project = tempfile::tempdir().unwrap();
    std::fs::write(project.path().join("AGENTS.md"), "Project rule: run tests.").unwrap();
    let memory = agent_rs::memory::load_all_agents_memory(Some(project.path()));
    let config = test_config().with_history(vec![
        AgentMessage::System {
            content: "Base system prompt".to_string(),
        },
        AgentMessage::System { content: memory },
        AgentMessage::System {
            content: "[SYSTEM INFO] pwd: /example/project".to_string(),
        },
        AgentMessage::User {
            content: "User content must stay in history".to_string(),
        },
    ]);

    let event = session_update_event(&config);
    let instructions = event["session"]["instructions"].as_str().unwrap();
    assert!(instructions.contains("Base system prompt"));
    assert!(instructions.contains("Project rule: run tests."));
    assert!(instructions.contains("[SYSTEM INFO] pwd: /example/project"));
    assert!(instructions.contains("be helpful"));
    assert!(instructions.contains("Use low verbosity"));
    assert!(!instructions.contains("User content must stay in history"));
    assert_eq!(conversation_item_create_events(&config.history).len(), 1);
}

#[test]
fn realtime_reconnect_uses_current_system_messages_without_accumulating_instructions() {
    let original = test_config().with_history(vec![AgentMessage::System {
        content: "Old project instructions".to_string(),
    }]);
    let reconnected = original.clone().with_history(vec![AgentMessage::System {
        content: "Updated project instructions".to_string(),
    }]);

    let event = session_update_event(&reconnected);
    let instructions = event["session"]["instructions"].as_str().unwrap();
    assert!(!instructions.contains("Old project instructions"));
    assert_eq!(
        instructions.matches("Updated project instructions").count(),
        1
    );
    assert_eq!(session_update_event(&reconnected), event);
    assert_eq!(reconnected.instructions, original.instructions);
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
        RealtimeEvent::UserTranscript { transcript: "hello".to_string(), item_id: String::new() }
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
            model: None,
            metadata: Default::default(),
        },
    )];
    let line =
        agent_rs::providers::format_cost_and_context_line(&messages, "openai:gpt-realtime", false);

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

#[tokio::test]
async fn wake_word_protocol_deletes_audio_item_then_sends_command_and_response() {
    use futures_util::{SinkExt, StreamExt};
    use tokio::net::TcpListener;
    use tokio_tungstenite::tungstenite::Message;

    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        let (stream, _) = listener.accept().await.unwrap();
        let mut socket = tokio_tungstenite::accept_async(stream).await.unwrap();
        let update: serde_json::Value =
            serde_json::from_str(&socket.next().await.unwrap().unwrap().into_text().unwrap())
                .unwrap();
        assert_eq!(
            update["session"]["audio"]["input"]["turn_detection"]["create_response"],
            false
        );
        socket.send(Message::Text(json!({"type":"conversation.item.input_audio_transcription.completed","item_id":"audio_1","transcript":"Computer, status?"}).to_string().into())).await.unwrap();
        let mut events = Vec::new();
        for _ in 0..3 {
            events.push(
                serde_json::from_str::<serde_json::Value>(
                    &socket.next().await.unwrap().unwrap().into_text().unwrap(),
                )
                .unwrap(),
            );
        }
        assert_eq!(
            events[0],
            json!({"type":"conversation.item.delete","item_id":"audio_1"})
        );
        assert_eq!(events[1]["item"]["content"][0]["text"], "status?");
        assert_eq!(events[2], response_create_event());
    });
    let mut config = test_config().with_wake_word("Computer".to_string());
    config.base_url = format!("ws://{address}/v1/realtime");
    let mut client = agent_rs::voice::realtime::RealtimeClient::connect(&config)
        .await
        .unwrap();
    let event = client.next_event().await.unwrap().unwrap();
    assert!(
        matches!(event, RealtimeEvent::UserTranscript { ref item_id, .. } if item_id == "audio_1")
    );
    client.delete_item("audio_1").await.unwrap();
    client.create_user_text("status?").await.unwrap();
    client.create_response().await.unwrap();
    server.await.unwrap();
}

#[test]
fn realtime_request_contains_openai_auth_and_model_query() {
    let request = build_realtime_request(&test_config()).unwrap();
    assert_eq!(
        request.headers().get("Authorization").unwrap(),
        "Bearer sk-test"
    );
    assert!(request.uri().to_string().contains("model=gpt-realtime"));
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
    assert_eq!(talk_model_name("gpt-5.5"), "gpt-realtime");
    assert_eq!(talk_model_name("openai:gpt-realtime"), "gpt-realtime");
}

#[test]
fn realtime_history_sends_image_input() {
    let events = conversation_item_create_events(&[AgentMessage::UserWithImages {
        content: "What is in this image?".to_string(),
        images: vec![agent_rs::agent::ImageAttachment {
            media_type: "image/png".to_string(),
            data: "aGVsbG8=".to_string(),
        }],
    }]);

    assert_eq!(events.len(), 1);
    assert_eq!(events[0]["item"]["content"][0]["type"], "input_text");
    assert_eq!(
        events[0]["item"]["content"][1],
        json!({
            "type": "input_image",
            "image_url": "data:image/png;base64,aGVsbG8="
        })
    );
}

#[test]
fn realtime_reconnect_restores_prior_chat_messages_and_tool_context() {
    let events = conversation_item_create_events(&[
        AgentMessage::System {
            content: "system prompt".to_string(),
        },
        AgentMessage::User {
            content: "remember blue".to_string(),
        },
        AgentMessage::Assistant(AssistantMessage {
            content: "I will remember blue.".to_string(),
            tool_calls: vec![ToolCall {
                id: "call_1".to_string(),
                name: "run_shell_command".to_string(),
                arguments: json!({ "cmd": "echo blue", "intent": "recall color", "timeout": 30 }),
            }],
            usage: None,
            model: None,
            metadata: Default::default(),
        }),
        AgentMessage::Tool(ToolResult {
            tool_call_id: "call_1".to_string(),
            name: "run_shell_command".to_string(),
            status: ToolStatus::Success,
            content: "blue\n".to_string(),
            elapsed_ms: None,
            subagent_usages: Vec::new(),
        }),
    ]);

    assert_eq!(events.len(), 4);
    assert_eq!(events[0]["item"]["role"], "user");
    assert_eq!(events[0]["item"]["content"][0]["text"], "remember blue");
    assert_eq!(events[1]["item"]["role"], "assistant");
    assert_eq!(
        events[1]["item"]["content"][0]["text"],
        "I will remember blue."
    );
    assert_eq!(events[2]["item"]["type"], "function_call");
    assert_eq!(events[2]["item"]["call_id"], "call_1");
    assert_eq!(events[2]["item"]["name"], "run_shell_command");
    assert_eq!(
        events[2]["item"]["arguments"],
        r#"{"cmd":"echo blue","intent":"recall color","timeout":30}"#
    );
    assert_eq!(events[3]["item"]["type"], "function_call_output");
    assert_eq!(events[3]["item"]["call_id"], "call_1");
    assert_eq!(events[3]["item"]["output"], "blue\n");
}

fn test_config() -> agent_rs::voice::realtime::RealtimeConfig {
    agent_rs::voice::realtime::RealtimeConfig {
        model: "gpt-realtime".to_string(),
        api_key: "sk-test".to_string(),
        instructions: "be helpful".to_string(),
        voice: "cedar".to_string(),
        voice_speed: 1.15,
        base_url: "wss://api.openai.com/v1/realtime".to_string(),
        transcription_model: "gpt-4o-mini-transcribe".to_string(),
        tools: Vec::new(),
        history: Vec::new(),
        initial_response: false,
        wake_word: None,
    }
}
