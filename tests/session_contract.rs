use agent_rs::agent::{AgentMessage, AssistantMessage, ToolCall, ToolResult, ToolStatus, Usage};
use agent_rs::providers::format_cost_and_context_line;
use agent_rs::session::{Session, SessionStore};
use serde_json::json;

#[test]
fn session_schema_v1_round_trip() {
    let temp = tempfile::tempdir().expect("temp dir");
    let store = SessionStore::with_root(temp.path().join(".agent"));
    let session = Session::new(
        "s1".to_string(),
        vec![
            AgentMessage::System {
                content: "system".to_string(),
            },
            AgentMessage::User {
                content: "hello".to_string(),
            },
            AgentMessage::Assistant(AssistantMessage {
                content: String::new(),
                tool_calls: vec![ToolCall {
                    id: "call_1".to_string(),
                    name: "run_shell_command".to_string(),
                    arguments: json!({"cmd": "echo hi", "timeout": 30}),
                }],
                usage: Some(Usage {
                    input_tokens: 100,
                    output_tokens: 12,
                    raw: None,
                }),
                metadata: Default::default(),
            }),
            AgentMessage::Tool(ToolResult {
                tool_call_id: "call_1".to_string(),
                name: "run_shell_command".to_string(),
                status: ToolStatus::Success,
                content: "hi\n".to_string(),
                elapsed_ms: Some(7),
            }),
        ],
    );

    store.save(&session).expect("save");
    let loaded = store.load(Some("s1")).expect("load");
    assert_eq!(loaded.schema_version, 1);
    assert_eq!(loaded.session_id, "s1");
    assert_eq!(loaded.messages, session.messages);
    assert_eq!(
        std::fs::read_to_string(store.latest_session_path()).expect("latest"),
        "s1\n"
    );
}

#[test]
fn new_session_ids_are_guids() {
    let temp = tempfile::tempdir().expect("temp dir");
    let store = SessionStore::with_root(temp.path().join(".agent"));

    let first = store.new_session_id();
    let second = store.new_session_id();

    uuid::Uuid::parse_str(&first).expect("first session id is a guid");
    uuid::Uuid::parse_str(&second).expect("second session id is a guid");
    assert_ne!(first, second);
}

#[test]
fn prompt_metadata_recomputes_cost_and_context_from_session_messages() {
    let messages = vec![AgentMessage::Assistant(AssistantMessage {
        content: String::new(),
        tool_calls: Vec::new(),
        usage: Some(Usage {
            input_tokens: 100,
            output_tokens: 12,
            raw: Some(json!({"total_cost": 1.2345})),
        }),
        metadata: Default::default(),
    })];

    let line = format_cost_and_context_line(&messages, "openai:gpt-5.2");

    assert!(line.contains("Cost: $1.2345"));
    assert!(line.contains("/400,000 tokens)"));
    assert!(line.contains("Model: openai:gpt-5.2"));
}

#[test]
fn prompt_metadata_does_not_render_negative_zero_cost() {
    let line = format_cost_and_context_line(&[], "mock");

    assert!(line.contains("Cost: $0.0000"));
    assert!(!line.contains("Cost: $-0.0000"));
}
