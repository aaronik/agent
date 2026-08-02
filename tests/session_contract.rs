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
fn session_labels_are_ordered_by_recently_saved_first() {
    let temp = tempfile::tempdir().expect("temp dir");
    let store = SessionStore::with_root(temp.path().join(".agent"));

    store
        .save(&Session::new(
            "zz-old".to_string(),
            vec![AgentMessage::User {
                content: "old conversation".to_string(),
            }],
        ))
        .expect("save old");
    std::thread::sleep(std::time::Duration::from_millis(20));
    store
        .save(&Session::new(
            "aa-new".to_string(),
            vec![AgentMessage::User {
                content: "new conversation".to_string(),
            }],
        ))
        .expect("save new");

    let labels = store.list_session_labels(80).expect("labels");

    assert_eq!(labels[0], "aa-new\tnew conversation");
    assert_eq!(labels[1], "zz-old\told conversation");
}

#[test]
fn session_search_finds_subject_across_conversation_history() {
    let temp = tempfile::tempdir().expect("temp dir");
    let store = SessionStore::with_root(temp.path().join(".agent"));

    store
        .save(&Session::new(
            "unrelated".to_string(),
            vec![AgentMessage::User {
                content: "discuss database migrations".to_string(),
            }],
        ))
        .expect("save unrelated");
    store
        .save(&Session::new(
            "shasta-session".to_string(),
            vec![
                AgentMessage::User {
                    content: "Please update the private land data importer".to_string(),
                },
                AgentMessage::Assistant(AssistantMessage {
                    content: "Updated shasta_private_land.py and added coverage".to_string(),
                    tool_calls: Vec::new(),
                    usage: None,
                    metadata: Default::default(),
                }),
            ],
        ))
        .expect("save matching");

    let matches = store
        .find_sessions("when we worked on shasta_private_land.py", 80)
        .expect("search sessions");

    assert_eq!(matches.len(), 1);
    assert_eq!(matches[0].session_id, "shasta-session");
    assert!(matches[0].excerpt.contains("shasta_private_land.py"));
}

#[test]
fn session_search_matches_filename_components_in_conversation_text() {
    let temp = tempfile::tempdir().expect("temp dir");
    let store = SessionStore::with_root(temp.path().join(".agent"));
    store
        .save(&Session::new(
            "shasta-map".to_string(),
            vec![AgentMessage::User {
                content: "map private land parcels around Shasta".to_string(),
            }],
        ))
        .expect("save relevant");
    store
        .save(&Session::new(
            "shasta-only".to_string(),
            vec![AgentMessage::User {
                content: "visit Mount Shasta".to_string(),
            }],
        ))
        .expect("save weak match");

    let matches = store
        .find_sessions("working on shasta_private_land.py", 80)
        .expect("search sessions");

    assert_eq!(matches.len(), 1);
    assert_eq!(matches[0].session_id, "shasta-map");
}

#[test]
fn session_search_ignores_tool_output_and_weak_generic_matches() {
    let temp = tempfile::tempdir().expect("temp dir");
    let store = SessionStore::with_root(temp.path().join(".agent"));
    store
        .save(&Session::new(
            "tool-noise".to_string(),
            vec![AgentMessage::Tool(ToolResult {
                tool_call_id: "call-1".to_string(),
                name: "read_file".to_string(),
                status: ToolStatus::Success,
                content: "source mentions shasta_private_land.py".to_string(),
                elapsed_ms: None,
            })],
        ))
        .expect("save tool noise");
    store
        .save(&Session::new(
            "generic".to_string(),
            vec![AgentMessage::User {
                content: "working on an unrelated project".to_string(),
            }],
        ))
        .expect("save generic");
    store
        .save(&Session::new(
            "relevant".to_string(),
            vec![AgentMessage::User {
                content: "fix shasta_private_land.py parcel boundaries".to_string(),
            }],
        ))
        .expect("save relevant");

    let matches = store
        .find_sessions_excluding(
            "working on shasta_private_land.py",
            80,
            Some("current-session"),
            10,
        )
        .expect("search sessions");

    assert_eq!(
        matches
            .iter()
            .map(|found| found.session_id.as_str())
            .collect::<Vec<_>>(),
        vec!["relevant"]
    );
}

#[test]
fn session_search_excludes_current_session_and_limits_results() {
    let temp = tempfile::tempdir().expect("temp dir");
    let store = SessionStore::with_root(temp.path().join(".agent"));
    for id in ["current", "one", "two", "three"] {
        store
            .save(&Session::new(
                id.to_string(),
                vec![AgentMessage::User {
                    content: "discuss needle".to_string(),
                }],
            ))
            .expect("save session");
    }

    let matches = store
        .find_sessions_excluding("needle", 80, Some("current"), 2)
        .expect("search sessions");

    assert_eq!(matches.len(), 2);
    assert!(matches.iter().all(|found| found.session_id != "current"));
}

#[test]
fn session_search_truncates_unicode_excerpts_safely() {
    let temp = tempfile::tempdir().expect("temp dir");
    let store = SessionStore::with_root(temp.path().join(".agent"));
    let content = format!("needle {}é tail", "a".repeat(71));
    store
        .save(&Session::new(
            "unicode-session".to_string(),
            vec![AgentMessage::User { content }],
        ))
        .expect("save session");

    let matches = store.find_sessions("needle", 80).expect("search sessions");

    assert_eq!(matches.len(), 1);
    assert!(matches[0].excerpt.ends_with("..."));
    assert!(
        matches[0]
            .excerpt
            .is_char_boundary(matches[0].excerpt.len())
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
