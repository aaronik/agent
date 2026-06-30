use agent_rs::cli::{completion_values_for_line, completion_values_for_line_with_models};
use agent_rs::session::SessionStore;
use assert_cmd::Command;
use serde_json::json;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

#[test]
fn mock_single_turn_executes_tool_and_saves_session() {
    let temp_home = tempfile::tempdir().expect("temp home");

    let mut cmd = Command::cargo_bin("agent").expect("agent binary");
    let output = cmd
        .env("HOME", temp_home.path())
        .args(["--model", "mock", "--single", "run echo hi"])
        .output()
        .expect("agent output");
    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).expect("utf8 stdout");
    let printed_session_id = assert_single_guid_session_id_line(&stdout);
    let running_index = stdout.find("[> Running]").expect("running tool panel");
    let ok_index = stdout.find("[OK Done]").expect("completed tool panel");
    let final_index = stdout
        .find("Tool completed: hi")
        .expect("final assistant response");
    assert!(running_index < ok_index);
    assert!(ok_index < final_index);

    let sessions_dir = temp_home.path().join(".agent-rs").join("sessions");
    let entries = std::fs::read_dir(&sessions_dir)
        .expect("sessions dir")
        .collect::<Result<Vec<_>, _>>()
        .expect("session entries");
    assert_eq!(entries.len(), 1);

    let payload = std::fs::read_to_string(entries[0].path()).expect("session payload");
    let value: serde_json::Value = serde_json::from_str(&payload).expect("session json");
    assert_eq!(value["schema_version"], 1);
    assert_eq!(value["session_id"], printed_session_id.as_str());
    assert_eq!(value["messages"][0]["role"], "system");
    assert!(
        value["messages"]
            .as_array()
            .expect("messages")
            .iter()
            .any(|message| message["role"] == "tool"
                && message["name"] == "run_shell_command"
                && message["content"].as_str().unwrap_or("").contains("hi"))
    );
}

#[test]
fn submitted_user_message_shows_immediate_working_feedback() {
    let temp_home = tempfile::tempdir().expect("temp home");

    let mut cmd = Command::cargo_bin("agent").expect("agent binary");
    let output = cmd
        .env("HOME", temp_home.path())
        .args(["--model", "mock", "--single", "run echo hi"])
        .output()
        .expect("agent output");
    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).expect("utf8 stdout");
    assert_single_guid_session_id_line(&stdout);

    let working_index = stdout.find("Working...").expect("working feedback");
    let running_index = stdout.find("[> Running]").expect("running tool panel");
    let final_index = stdout
        .find("Tool completed: hi")
        .expect("final assistant response");

    assert!(working_index < running_index);
    assert!(working_index < final_index);
}

#[test]
fn resume_replay_shows_tool_commands() {
    let temp_home = tempfile::tempdir().expect("temp home");

    let mut initial = Command::cargo_bin("agent").expect("agent binary");
    initial
        .env("HOME", temp_home.path())
        .args(["--model", "mock", "--single", "run echo hi"])
        .assert()
        .success();

    let mut replay = Command::cargo_bin("agent").expect("agent binary");
    replay
        .env("HOME", temp_home.path())
        .args(["--model", "mock", "--single", "--resume"])
        .assert()
        .success()
        .stdout(predicates::str::contains("run_shell_command"))
        .stdout(predicates::str::contains("cmd=echo hi"));
}

#[test]
fn new_session_loads_agents_md_memory_file() {
    let temp_home = tempfile::tempdir().expect("temp home");
    let project = tempfile::tempdir().expect("project");
    std::fs::write(
        project.path().join("AGENTS.md"),
        "project agent instructions",
    )
    .expect("write agents memory");
    std::fs::write(
        project.path().join("CLAUDE.md"),
        "stale claude instructions",
    )
    .expect("write claude memory");

    let mut cmd = Command::cargo_bin("agent").expect("agent binary");
    cmd.current_dir(project.path())
        .env("HOME", temp_home.path())
        .args(["--model", "mock", "--single", "run echo hi"])
        .assert()
        .success();

    let sessions_dir = temp_home.path().join(".agent-rs").join("sessions");
    let entries = std::fs::read_dir(&sessions_dir)
        .expect("sessions dir")
        .collect::<Result<Vec<_>, _>>()
        .expect("session entries");
    let payload = std::fs::read_to_string(entries[0].path()).expect("session payload");
    let value: serde_json::Value = serde_json::from_str(&payload).expect("session json");
    let system_messages = value["messages"]
        .as_array()
        .expect("messages")
        .iter()
        .filter(|message| message["role"] == "system")
        .map(|message| message["content"].as_str().unwrap_or(""))
        .collect::<Vec<_>>()
        .join("\n");

    assert!(system_messages.contains("project agent instructions"));
    assert!(!system_messages.contains("stale claude instructions"));
}

#[test]
fn command_buffer_flag_is_single_turn_and_prints_only_final_response() {
    let temp_home = tempfile::tempdir().expect("temp home");

    let mut cmd = Command::cargo_bin("agent").expect("agent binary");
    cmd.env("HOME", temp_home.path())
        .args(["--model", "mock", "-c", "run echo hi"])
        .assert()
        .success()
        .stdout("run echo hi\n");

    assert!(!temp_home.path().join(".agent-rs").join("sessions").exists());
}

#[test]
fn command_buffer_flag_is_available_in_help() {
    let mut cmd = Command::cargo_bin("agent").expect("agent binary");
    cmd.env_remove("OPENAI_API_KEY")
        .arg("--help")
        .assert()
        .success()
        .stdout(predicates::str::contains("--command"))
        .stdout(predicates::str::contains("-c"));
}

#[test]
fn help_does_not_require_provider_configuration() {
    let mut cmd = Command::cargo_bin("agent").expect("agent binary");
    cmd.env_remove("OPENAI_API_KEY")
        .arg("--help")
        .assert()
        .success()
        .stdout(predicates::str::contains("--model"))
        .stdout(predicates::str::contains("--single"))
        .stdout(predicates::str::contains("--update-pricing"))
        .stdout(predicates::str::contains("--resume"));
}

#[tokio::test]
async fn update_pricing_flag_downloads_litellm_pricing_without_provider_configuration() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/pricing.json"))
        .respond_with(ResponseTemplate::new(200).set_body_json(pricing_payload()))
        .mount(&server)
        .await;
    let temp_home = tempfile::tempdir().expect("temp home");

    let mut cmd = Command::cargo_bin("agent").expect("agent binary");
    cmd.env("HOME", temp_home.path())
        .env(
            "AGENT_PRICING_URL",
            format!("{}/pricing.json", server.uri()),
        )
        .env_remove("OPENAI_API_KEY")
        .arg("--update-pricing")
        .assert()
        .success()
        .stdout(predicates::str::contains("Pricing updated"));

    assert!(
        temp_home
            .path()
            .join(".agent-rs")
            .join("pricing")
            .join("model_prices_and_context_window.json")
            .exists()
    );
}

#[test]
fn single_without_query_does_not_require_provider_configuration() {
    let temp_home = tempfile::tempdir().expect("temp home");

    let mut cmd = Command::cargo_bin("agent").expect("agent binary");
    cmd.env("HOME", temp_home.path())
        .env_remove("OPENAI_API_KEY")
        .arg("--single")
        .assert()
        .success();
}

#[test]
fn single_without_model_uses_gpt_5_5_default() {
    let temp_home = tempfile::tempdir().expect("temp home");

    let mut cmd = Command::cargo_bin("agent").expect("agent binary");
    cmd.env("HOME", temp_home.path())
        .env("AGENT_MODEL", "")
        .env("OLLAMA_MODEL", "")
        .env("OPENAI_MODEL", "")
        .env_remove("OPENAI_API_KEY")
        .arg("--single")
        .assert()
        .success()
        .stdout(predicates::str::contains("model: gpt-5.5"));
}

#[test]
fn interactive_without_tty_returns_actionable_error() {
    let temp_home = tempfile::tempdir().expect("temp home");

    let mut cmd = Command::cargo_bin("agent").expect("agent binary");
    cmd.env("HOME", temp_home.path())
        .env_remove("OPENAI_API_KEY")
        .args(["--new", "--model", "mock"])
        .assert()
        .failure()
        .stderr(predicates::str::contains("interactive mode requires a TTY"));
}

#[test]
fn slash_help_does_not_require_provider_configuration() {
    let temp_home = tempfile::tempdir().expect("temp home");

    let mut cmd = Command::cargo_bin("agent").expect("agent binary");
    cmd.env("HOME", temp_home.path())
        .env_remove("OPENAI_API_KEY")
        .args(["--single", "/help"])
        .assert()
        .success()
        .stdout(predicates::str::contains("/clear"))
        .stdout(predicates::str::contains("/new"))
        .stdout(predicates::str::contains("/models"))
        .stdout(predicates::str::contains("/pricing refresh"))
        .stdout(predicates::str::contains("/resume"));
}

#[test]
fn slash_new_aliases_clear_without_provider_configuration() {
    let temp_home = tempfile::tempdir().expect("temp home");

    let mut cmd = Command::cargo_bin("agent").expect("agent binary");
    cmd.env("HOME", temp_home.path())
        .env_remove("OPENAI_API_KEY")
        .args(["--single", "/new"])
        .assert()
        .success()
        .stdout(predicates::str::contains("cleared"))
        .stdout(predicates::str::contains("sessionId: "));
}

#[test]
fn slash_completion_includes_models_command_and_model_ids() {
    let temp_home = tempfile::tempdir().expect("temp home");
    let store = SessionStore::with_root(temp_home.path().join(".agent-rs"));

    let command_values = completion_values_for_line(&store, "/mod", 4);
    assert!(command_values.contains(&"/models".to_string()));
    assert!(!command_values.contains(&"/models mock".to_string()));

    let exact_command_values = completion_values_for_line(&store, "/clear", 6);
    assert_eq!(exact_command_values.first(), Some(&"/clear".to_string()));

    let new_command_values = completion_values_for_line(&store, "/new", 4);
    assert_eq!(new_command_values.first(), Some(&"/new".to_string()));

    let model_values = completion_values_for_line(&store, "/models ", 8);
    assert!(!model_values.contains(&"/models mock".to_string()));
    assert!(model_values.contains(&"/models gpt-5.5".to_string()));
    assert!(model_values.contains(&"/models openai:gpt-5.5".to_string()));

    let fuzzy_values = completion_values_for_line(&store, "/models op", 10);
    assert_eq!(
        fuzzy_values.first(),
        Some(&"/models openai:gpt-5.5".to_string())
    );
}

#[test]
fn slash_completion_includes_dynamic_model_ids_but_not_pricing_cache_surface() {
    let temp_home = tempfile::tempdir().expect("temp home");
    let store = SessionStore::with_root(temp_home.path().join(".agent-rs"));
    let pricing_dir = store.root().join("pricing");
    std::fs::create_dir_all(&pricing_dir).expect("pricing dir");
    std::fs::write(
        pricing_dir.join("model_prices_and_context_window.json"),
        serde_json::to_string(&json!({
            "gpt-4o": { "input_cost_per_token": 0.000001 },
            "openai/gpt-4.1": { "input_cost_per_token": 0.000001 },
            "anthropic/claude-sonnet-4": { "input_cost_per_token": 0.000001 },
            "amazon/nova-pro": { "input_cost_per_token": 0.000001 }
        }))
        .expect("pricing json"),
    )
    .expect("write pricing cache");

    let values = completion_values_for_line_with_models(
        &store,
        "/models ",
        8,
        &["openai:gpt-5.2".to_string(), "ollama:llama3.2".to_string()],
    );

    assert!(values.contains(&"/models openai:gpt-5.2".to_string()));
    assert!(values.contains(&"/models ollama:llama3.2".to_string()));
    assert!(!values.contains(&"/models gpt-4o".to_string()));
    assert!(!values.contains(&"/models openai:gpt-4o".to_string()));
    assert!(!values.contains(&"/models openai:gpt-4.1".to_string()));
    assert!(!values.contains(&"/models anthropic:claude-sonnet-4".to_string()));
    assert!(!values.contains(&"/models amazon:nova-pro".to_string()));
}

#[tokio::test]
async fn slash_pricing_refresh_downloads_litellm_pricing_without_provider_configuration() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/pricing.json"))
        .respond_with(ResponseTemplate::new(200).set_body_json(pricing_payload()))
        .mount(&server)
        .await;
    let temp_home = tempfile::tempdir().expect("temp home");

    let mut cmd = Command::cargo_bin("agent").expect("agent binary");
    cmd.env("HOME", temp_home.path())
        .env(
            "AGENT_PRICING_URL",
            format!("{}/pricing.json", server.uri()),
        )
        .env_remove("OPENAI_API_KEY")
        .args(["--single", "/pricing refresh"])
        .assert()
        .success()
        .stdout(predicates::str::contains("Pricing updated"));
}

#[test]
fn slash_model_switch_does_not_initialize_default_provider() {
    let temp_home = tempfile::tempdir().expect("temp home");

    let mut cmd = Command::cargo_bin("agent").expect("agent binary");
    cmd.env("HOME", temp_home.path())
        .env_remove("OPENAI_API_KEY")
        .args(["--single", "/models mock"])
        .assert()
        .success()
        .stdout(predicates::str::contains("model: mock"));
}

fn assert_single_guid_session_id_line(stdout: &str) -> String {
    let session_id_line = stdout
        .lines()
        .find(|line| line.starts_with("sessionId: "))
        .expect("session id line");
    let printed_session_id = session_id_line
        .strip_prefix("sessionId: ")
        .expect("session id prefix");
    uuid::Uuid::parse_str(printed_session_id).expect("session id is a guid");
    assert!(!stdout.contains("session id: "));
    assert_eq!(stdout.matches("sessionId: ").count(), 1);
    printed_session_id.to_string()
}

fn pricing_payload() -> serde_json::Value {
    json!({
        "gpt-5.2": {
            "input_cost_per_token": 0.000001,
            "output_cost_per_token": 0.000003
        }
    })
}
