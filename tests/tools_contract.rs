use std::sync::Mutex;
use std::time::Duration;

use clap::Parser;

use agent_rs::tools::browser::{PlaywrightBrowserSessionArgs, playwright_browser_session};
use agent_rs::tools::fetch::{FetchArgs, fetch};
use agent_rs::tools::files::{
    ReadFileArgs, SearchReplaceArgs, WriteFileArgs, read_file, search_replace, write_file,
};
use agent_rs::tools::image::{GenImageArgs, gen_image};
use agent_rs::tools::registry::ToolRegistry;
use agent_rs::tools::shell::{RunShellCommandArgs, run_shell_command};
use agent_rs::tools::spawn::{SpawnAgent, SpawnArgs, spawn};
use serde_json::json;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

static ENV_LOCK: Mutex<()> = Mutex::new(());

#[tokio::test]
async fn registry_exposes_and_executes_active_tool_surface() {
    let registry = ToolRegistry::new();
    let names = registry
        .definitions()
        .iter()
        .map(|definition| definition.name.as_str())
        .collect::<Vec<_>>();

    assert_eq!(
        names,
        vec![
            "run_shell_command",
            "fetch",
            "read_file",
            "write_file",
            "search_replace",
            "gen_image",
            "communicate",
            "anonymous_browser_session",
            "browser",
            "spawn",
        ]
    );

    let write_description = registry
        .definitions()
        .iter()
        .find(|definition| definition.name == "write_file")
        .expect("write_file definition")
        .description
        .as_str();
    assert!(write_description.contains("Use this only when creating a new file"));
    assert!(write_description.contains("prefer `search_replace`"));

    let search_replace_description = registry
        .definitions()
        .iter()
        .find(|definition| definition.name == "search_replace")
        .expect("search_replace definition")
        .description
        .as_str();
    assert!(search_replace_description.contains("Prefer this over `write_file`"));
    assert!(search_replace_description.contains("original text can be matched exactly"));

    let communicate_description = registry
        .definitions()
        .iter()
        .find(|definition| definition.name == "communicate")
        .expect("communicate definition")
        .description
        .as_str();
    assert!(communicate_description.contains("before starting a multi-step task"));
    assert!(communicate_description.contains("when the plan changes"));
    assert!(communicate_description.contains("before lengthy checks"));
    assert!(communicate_description.contains("Do not use it for every tool call"));

    let browser_definition = registry
        .definitions()
        .iter()
        .find(|definition| definition.name == "anonymous_browser_session")
        .expect("anonymous_browser_session definition");
    assert!(browser_definition.description.contains("anonymous"));
    assert!(browser_definition.description.contains("browser"));
    assert!(
        browser_definition
            .description
            .contains("Never use it for signed-in browsing")
    );
    assert!(
        browser_definition
            .description
            .contains("waitUntil: 'domcontentloaded'")
    );
    assert!(
        browser_definition
            .description
            .contains("does not improve model vision")
    );
    assert!(
        browser_definition
            .description
            .contains("screenshots work headlessly")
    );
    let visible_description = browser_definition.parameters["properties"]["visible"]["description"]
        .as_str()
        .expect("visible parameter description");
    assert!(visible_description.contains("human user"));
    assert!(visible_description.contains("Never set this merely to inspect images"));
    assert!(
        browser_definition
            .parameters
            .pointer("/properties/profile")
            .is_none()
    );
    assert!(
        browser_definition
            .parameters
            .pointer("/properties/signed_in")
            .is_none()
    );

    let live_tab = registry
        .definitions()
        .iter()
        .find(|d| d.name == "browser")
        .unwrap();
    assert!(
        live_tab
            .description
            .contains("Default tool for all browser activity")
    );
    assert!(
        live_tab
            .description
            .contains("use anonymous_browser_session only")
    );
    assert!(
        live_tab
            .description
            .contains("only tabs opened by this tool")
    );
    assert_eq!(
        live_tab.parameters["properties"]["action"]["type"],
        "string"
    );
    assert!(
        live_tab
            .description
            .contains("Only use focus=true when the user explicitly asks")
    );
    let focus = &live_tab.parameters["properties"]["focus"];
    assert!(
        focus["type"]
            .as_array()
            .unwrap()
            .contains(&json!("boolean"))
    );
    assert!(
        focus["description"]
            .as_str()
            .unwrap()
            .contains("Only set true if the user explicitly asks")
    );

    for definition in registry.definitions() {
        let parameters = &definition.parameters;
        let intent = parameters
            .pointer("/properties/intent")
            .expect("intent property exists");
        assert_eq!(intent["type"], "string");
        assert_eq!(intent["maxLength"], 80);
        assert!(
            intent["description"]
                .as_str()
                .expect("intent description")
                .contains("why")
        );
        assert!(
            parameters["required"]
                .as_array()
                .expect("required array")
                .iter()
                .any(|field| field == "intent")
        );
        assert_eq!(
            parameters["properties"]
                .as_object()
                .expect("tool properties")
                .keys()
                .next()
                .map(String::as_str),
            Some("intent"),
            "{} should list intent first in its properties",
            definition.name
        );
        assert_eq!(
            parameters["required"]
                .as_array()
                .expect("required array")
                .first(),
            Some(&json!("intent")),
            "{} should require intent first",
            definition.name
        );
        let serialized = serde_json::to_string(parameters).expect("serialize tool schema");
        let properties = serialized
            .split("\"properties\":{")
            .nth(1)
            .expect("properties object");
        assert!(
            properties.starts_with("\"intent\":"),
            "{} should send intent first to providers",
            definition.name
        );
        assert!(definition.description.contains("intent"));
    }

    let result = registry
        .execute(
            "call_communicate".to_string(),
            "communicate",
            json!({"intent": "share progress", "message": "cutover progress"}),
        )
        .await;
    assert_eq!(result.name, "communicate");
    assert_eq!(result.content, "cutover progress");

    let bad_args = registry
        .execute(
            "call_bad".to_string(),
            "run_shell_command",
            json!({"intent": "run user command", "timeout": 30}),
        )
        .await;
    assert!(bad_args.content.contains("invalid tool arguments"));

    let missing_intent = registry
        .execute(
            "call_missing_intent".to_string(),
            "communicate",
            json!({"message": "cutover progress"}),
        )
        .await;
    assert!(missing_intent.content.contains("missing required intent"));

    let long_intent = registry
        .execute(
            "call_long_intent".to_string(),
            "communicate",
            json!({"intent": "x".repeat(81), "message": "cutover progress"}),
        )
        .await;
    assert!(
        long_intent
            .content
            .contains("intent must be 80 characters or fewer")
    );

    let unknown = registry
        .execute("call_unknown".to_string(), "unknown_tool", json!({}))
        .await;
    assert!(unknown.content.contains("unknown tool"));
}

fn assert_saved_tool_output(output: &str, expected: &str) {
    let encoded_path = output
        .lines()
        .find_map(|line| line.strip_prefix("Full output saved to temporary file: "))
        .expect("saved output path");
    let path: String = serde_json::from_str(encoded_path).expect("JSON-quoted path");
    assert_eq!(
        std::fs::read_to_string(&path).expect("full output"),
        expected
    );
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        assert_eq!(
            std::fs::metadata(&path).unwrap().permissions().mode() & 0o777,
            0o600
        );
    }
    assert!(output.contains("Delete this temporary file when finished with it."));
    std::fs::remove_file(path).expect("remove saved output");
}

#[tokio::test]
async fn registry_preserves_unicode_head_and_tail_and_saves_full_output() {
    let content = format!("{}MIDDLE_ONLY{}", "🙂".repeat(20_000), "終".repeat(20_000));
    let registry = ToolRegistry::without_spawn();
    let mut results = Vec::new();
    for _ in 0..2 {
        let result = registry
            .execute(
                "large".into(),
                "communicate",
                json!({"intent": "test truncation", "message": content}),
            )
            .await;
        assert_eq!(result.status, agent_rs::agent::ToolStatus::Success);
        assert!(result.content.starts_with(&"🙂".repeat(16_000)));
        assert!(result.content.ends_with(&"終".repeat(16_000)));
        assert!(!result.content.contains("MIDDLE_ONLY"));
        assert!(result.content.chars().count() < 33_000);
        results.push(result.content);
    }
    assert_ne!(
        results[0], results[1],
        "each call needs its own output file"
    );
    for output in results {
        assert_saved_tool_output(&output, &content);
    }
}

#[tokio::test]
async fn registry_leaves_output_at_or_below_limit_unchanged() {
    for size in [0, 31_999, 32_000] {
        let content = "🙂".repeat(size);
        let result = ToolRegistry::without_spawn()
            .execute(
                "boundary".into(),
                "communicate",
                json!({"intent": "test boundary", "message": content}),
            )
            .await;
        assert_eq!(result.content, content);
    }
}

#[tokio::test]
async fn registry_truncates_and_saves_error_output_without_changing_status() {
    let name = format!("{}ERROR_AT_END", "x".repeat(32_001));
    let result = ToolRegistry::without_spawn()
        .execute("error".into(), &name, json!({}))
        .await;
    assert_eq!(result.status, agent_rs::agent::ToolStatus::Error);
    assert!(result.content.starts_with("unknown tool: "));
    assert!(result.content.ends_with("ERROR_AT_END"));
    assert!(result.content.contains("The tool finished with an error."));
    assert_saved_tool_output(&result.content, &format!("unknown tool: {name}"));
}

#[tokio::test]
async fn registry_truncates_all_large_tool_results_with_guidance() {
    let _env_lock = ENV_LOCK.lock().expect("env lock");
    let registry = ToolRegistry::without_spawn();
    let result = registry
        .execute(
            "call_shell".to_string(),
            "run_shell_command",
            json!({
                "intent": "produce test output",
                "cmd": "head -c 40000 /dev/zero | tr '\\0' x",
                "timeout": 30
            }),
        )
        .await;

    assert_eq!(result.status, agent_rs::agent::ToolStatus::Success);
    assert!(result.content.starts_with(&"x".repeat(100)));
    assert!(
        result
            .content
            .contains("[Output trimmed by the harness to avoid overwhelming the context.]")
    );
    assert!(result.content.contains("The tool completed successfully."));
    assert!(result.content.contains("make a more selective tool call"));
    assert_saved_tool_output(&result.content, &"x".repeat(40_000));
}

#[tokio::test]
async fn registry_records_elapsed_time_for_tool_results() {
    let registry = ToolRegistry::new();
    let result = registry
        .execute(
            "call_communicate".to_string(),
            "communicate",
            json!({"intent": "share progress", "message": "done"}),
        )
        .await;

    assert_eq!(result.status, agent_rs::agent::ToolStatus::Success);
    assert!(result.elapsed_ms.is_some());
}

#[tokio::test]
async fn run_shell_command_reports_stdout_and_exit_code() {
    let ok = run_shell_command(RunShellCommandArgs {
        cmd: "printf hi".to_string(),
        timeout: 30,
    })
    .await
    .expect("ok command");
    assert_eq!(ok, "hi");

    let err = run_shell_command(RunShellCommandArgs {
        cmd: "printf nope && exit 7".to_string(),
        timeout: 30,
    })
    .await
    .expect("error command output");
    assert!(err.contains("nope"));
    assert!(err.contains("(exit code: 7)"));
}

#[tokio::test]
async fn run_shell_command_truncates_large_completed_output_with_guidance() {
    let _env_lock = ENV_LOCK.lock().expect("env lock");
    let output = ToolRegistry::without_spawn()
        .execute(
            "call_shell".to_string(),
            "run_shell_command",
            json!({
                "intent": "produce test output",
                "cmd": "head -c 40000 /dev/zero | tr '\\0' x",
                "timeout": 30
            }),
        )
        .await
        .content;

    assert!(output.starts_with(&"x".repeat(100)));
    assert!(output.contains("[Output trimmed by the harness to avoid overwhelming the context.]"));
    assert!(output.contains("The tool completed successfully."));
    assert!(output.contains("make a more selective tool call"));
    assert_saved_tool_output(&output, &"x".repeat(40_000));
}

#[tokio::test]
async fn registry_allows_git_writes_only_when_enabled() {
    let temp = tempfile::tempdir().expect("temp dir");
    let repo = temp.path().join("repo");
    let arguments = json!({
        "intent": "initialize test repository",
        "cmd": format!("git init {}", repo.display()),
        "timeout": 30
    });

    let blocked = ToolRegistry::new()
        .execute(
            "call_git".to_string(),
            "run_shell_command",
            arguments.clone(),
        )
        .await;
    assert_eq!(blocked.status, agent_rs::agent::ToolStatus::Error);
    assert!(blocked.content.contains("blocked git write operation"));

    let allowed = ToolRegistry::new_with_git_write_access()
        .execute("call_git".to_string(), "run_shell_command", arguments)
        .await;
    assert_eq!(allowed.status, agent_rs::agent::ToolStatus::Success);
    assert!(repo.join(".git").is_dir());
}

#[test]
fn allow_git_cli_flag_enables_git_write_access() {
    let args = agent_rs::cli::Args::try_parse_from(["agent", "--allow-git"])
        .expect("allow-git should parse");
    assert!(args.allow_git);
}

#[tokio::test]
async fn run_shell_command_blocks_git_write_operations() {
    let blocked_commands = [
        "git commit -m nope",
        "git revert HEAD",
        "git reset --hard HEAD~1",
        "git push origin main",
        "git add src/lib.rs",
        "git checkout -b feature",
        "git --no-pager commit -m nope",
        "git -c user.name=test commit -m nope",
        "cd repo && git commit -m nope",
        "git -C repo commit -m nope",
        "GIT_DIR=.git git update-ref refs/heads/main HEAD",
    ];

    for cmd in blocked_commands {
        let output = run_shell_command(RunShellCommandArgs {
            cmd: cmd.to_string(),
            timeout: 30,
        })
        .await
        .expect_err("git write command should be blocked");

        assert!(
            output.contains("blocked git write operation"),
            "unexpected output for {cmd}: {output}"
        );
    }
}

#[tokio::test]
async fn run_shell_command_allows_git_read_only_operations() {
    let temp = tempfile::tempdir().expect("temp dir");
    let repo = temp.path().join("repo");
    std::fs::create_dir(&repo).expect("repo dir");
    std::fs::create_dir(repo.join(".git")).expect("git dir marker");
    std::fs::write(repo.join(".git/HEAD"), "ref: refs/heads/main\n").expect("head");

    let read_only_commands = [
        format!("git -C {} status --short", repo.display()),
        format!("git -C {} log --oneline -1", repo.display()),
        format!("git -C {} reflog", repo.display()),
        format!("git -C {} diff -- src/lib.rs", repo.display()),
        format!("git -C {} branch --list", repo.display()),
    ];

    for cmd in read_only_commands {
        let output = run_shell_command(RunShellCommandArgs { cmd, timeout: 30 }).await;
        assert!(
            output.is_ok(),
            "read-only git command should reach the shell: {output:?}"
        );
    }
}

#[tokio::test]
async fn run_shell_command_reports_timeout() {
    let timed_out = run_shell_command(RunShellCommandArgs {
        cmd: "/bin/sleep 2".to_string(),
        timeout: 1,
    })
    .await
    .expect("timeout output");

    assert!(timed_out.contains("(exit code: 124)"));
    assert!(timed_out.contains("command timed out after 1s"));
}

#[tokio::test]
async fn run_shell_command_does_not_wait_for_stdin() {
    let output = tokio::time::timeout(
        std::time::Duration::from_secs(1),
        run_shell_command(RunShellCommandArgs {
            cmd: "/bin/cat".to_string(),
            timeout: 30,
        }),
    )
    .await
    .expect("cat should see closed stdin promptly")
    .expect("cat output");

    assert_eq!(output, "");
}

#[tokio::test]
async fn file_tools_read_write_and_search_replace_with_diffs() {
    let temp = tempfile::tempdir().expect("temp dir");
    let path = temp.path().join("sample.txt");
    let path_str = path.to_string_lossy().to_string();

    let write = write_file(WriteFileArgs {
        path: path_str.clone(),
        contents: "one\ntwo\n".to_string(),
    })
    .await
    .expect("write");
    assert!(write.contains("Success"));
    assert!(write.contains("Diff:"));

    let read = read_file(ReadFileArgs {
        path: path_str.clone(),
    })
    .await
    .expect("read");
    assert!(read.contains("[FILE]:"));
    assert!(read.contains("one\ntwo"));

    let replaced = search_replace(SearchReplaceArgs {
        path: path_str,
        old_text: "two".to_string(),
        new_text: "three".to_string(),
    })
    .await
    .expect("replace");
    assert!(replaced.contains("Successfully replaced 1 occurrence(s)"));
    assert!(replaced.contains("-two"));
    assert!(replaced.contains("+three"));
}

#[tokio::test]
async fn read_file_truncates_large_content_with_guidance() {
    let temp = tempfile::tempdir().expect("temp dir");
    let path = temp.path().join("large.txt");
    std::fs::write(&path, "a".repeat(40_000)).expect("large file");

    let output = ToolRegistry::without_spawn()
        .execute(
            "call_read".to_string(),
            "read_file",
            json!({
                "intent": "read large test file",
                "path": path.to_string_lossy()
            }),
        )
        .await
        .content;

    assert!(output.starts_with("[FILE]:"));
    assert!(output.contains("[Output trimmed by the harness to avoid overwhelming the context.]"));
    assert!(output.contains("The tool completed successfully."));
    assert!(output.contains("make a more selective tool call"));
    assert_saved_tool_output(
        &output,
        &format!("[FILE]: {}\n{}", path.display(), "a".repeat(40_000)),
    );
}

#[tokio::test]
#[ignore = "external network smoke for production cutover audits"]
async fn fetch_live_example_dot_com() {
    let output = fetch(FetchArgs {
        url: "https://example.com".to_string(),
    })
    .await
    .expect("fetch");

    assert!(output.contains("[URL]: https://example.com"));
    assert!(output.to_lowercase().contains("example domain"));
}

#[tokio::test]
async fn gen_image_uses_configured_openai_endpoint() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/images/generations"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "data": [
                {
                    "url": "https://example.test/image.png"
                }
            ]
        })))
        .mount(&server)
        .await;
    let _api_key = EnvGuard::set("OPENAI_API_KEY", "test-key");
    let _base_url = EnvGuard::set("AGENT_BASE_URL", &server.uri());

    let output = gen_image(GenImageArgs {
        number: 1,
        model: "dall-e-3".to_string(),
        size: "1024x1024".to_string(),
        prompt: "a test image".to_string(),
    })
    .await
    .expect("image generation");

    assert!(output.contains("https://example.test/image.png"));
}

#[tokio::test]
async fn playwright_browser_session_fails_fast_when_playwright_missing_from_path() {
    let _env_lock = ENV_LOCK.lock().expect("env lock");
    let _path = EnvGuard::set("PATH", "");
    let output = playwright_browser_session(PlaywrightBrowserSessionArgs {
        javascript: "return await page.title();".to_string(),
        url: Some("https://example.com".to_string()),
        timeout: 1,
        close: false,
        reset: false,
        visible: false,
    })
    .await
    .expect_err("missing playwright should fail before Chrome/profile checks");

    assert!(output.contains("playwright is not installed or not found in PATH"));
    assert!(output.contains("anonymous_browser_session cannot work"));
}

#[tokio::test]
async fn playwright_browser_session_rejects_retired_signed_in_argument_before_launching_chrome() {
    let registry = ToolRegistry::new();
    let result = registry
        .execute(
            "browser".into(),
            "anonymous_browser_session",
            json!({
                "intent": "reject profile copy", "javascript": "return 'test';", "signed_in": true
            }),
        )
        .await;
    assert_eq!(result.status, agent_rs::agent::ToolStatus::Error);
    assert!(result.content.contains("signed_in"));
    assert!(result.content.contains("browser"));
}

#[tokio::test]
async fn chrome_tab_rejects_unowned_tab_requests_before_connecting() {
    let registry = ToolRegistry::new();
    let result = registry
        .execute(
            "id".into(),
            "browser",
            json!({
                "intent": "test tab validation", "action": "snapshot"
            }),
        )
        .await;
    assert_eq!(result.status, agent_rs::agent::ToolStatus::Error);
    assert!(result.content.contains("tab_id is required"));

    for old_name in ["chrome_tab", "browser_control"] {
        let result = registry
            .execute(
                "old".into(),
                old_name,
                json!({"intent": "check retired name"}),
            )
            .await;
        assert_eq!(result.status, agent_rs::agent::ToolStatus::Error);
        assert!(result.content.contains("unknown tool"));
    }
}

#[cfg(unix)]
#[tokio::test]
async fn spawn_model_override_must_be_in_models_list() {
    use agent_rs::agent::ToolStatus;
    use std::os::unix::fs::PermissionsExt;

    let _env_lock = ENV_LOCK.lock().expect("env lock");
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/api/tags"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "models": [{"name": "test:latest"}]
        })))
        .mount(&server)
        .await;
    let _url = EnvGuard::set("OLLAMA_URL", &server.uri());
    let _key = EnvGuard::remove("OPENAI_API_KEY");
    let _default = EnvGuard::set("AGENT_SPAWN_MODEL", "mock");
    let directory = tempfile::tempdir().expect("temporary directory");
    let script = directory.path().join("spawn-arguments.sh");
    std::fs::write(&script, "#!/bin/sh\nprintf '%s\\n' \"$@\"\n").expect("write stand-in");
    std::fs::set_permissions(&script, std::fs::Permissions::from_mode(0o755))
        .expect("make stand-in executable");
    let _bin = EnvGuard::set("AGENT_SPAWN_BIN", script.to_str().unwrap());
    let registry = ToolRegistry::new();
    let definition = registry
        .definitions()
        .iter()
        .find(|tool| tool.name == "spawn")
        .unwrap();
    let properties = definition.parameters["properties"].as_object().unwrap();
    assert_eq!(properties.len(), 2);
    assert!(properties.contains_key("agents"));
    assert!(properties.contains_key("intent"));
    assert!(
        definition.parameters["required"]
            .as_array()
            .unwrap()
            .contains(&json!("agents"))
    );

    let result = registry
        .execute(
            "selected".into(),
            "spawn",
            json!({
                "intent": "test selected model", "agents": [{"task": "assigned task", "model": "ollama:test:latest",
                "conversation_id": "existing-session"}]
            }),
        )
        .await;
    assert_eq!(result.status, ToolStatus::Success, "{}", result.content);
    assert_eq!(
        result
            .content
            .matches("--model\nollama:test:latest")
            .count(),
        1
    );
    assert_eq!(
        result.content.matches("--resume\nexisting-session").count(),
        1
    );
    assert!(!result.content.contains("mock"));

    // None of these is an exact entry in /models; mock must not bypass validation.
    for model in [
        "",
        "missing",
        "unknown:test",
        "ollama:missing",
        "mock",
        "test:latest",
    ] {
        let result = registry
            .execute(
                "invalid".into(),
                "spawn",
                json!({
                    "intent": "test unavailable model", "agents": [{"task": "must not launch", "model": model}]
                }),
            )
            .await;
        assert_eq!(result.status, ToolStatus::Error, "model: {model}");
        assert!(result.content.contains("/models"), "{}", result.content);
        assert!(!result.content.contains("[SPAWNED AGENT OUTPUT]"));
    }

    // A discovery failure must fail closed rather than launch the requested model.
    server.reset().await;
    Mock::given(method("GET"))
        .and(path("/api/tags"))
        .respond_with(ResponseTemplate::new(503).set_body_json(json!({
            "models": [{"name": "test:latest"}]
        })))
        .mount(&server)
        .await;
    let result = registry
        .execute(
            "unavailable".into(),
            "spawn",
            json!({
                "intent": "test unavailable model", "agents": [{"task": "must not launch", "model": "ollama:test:latest"}]
            }),
        )
        .await;
    assert_eq!(result.status, ToolStatus::Error);
    assert!(result.content.contains("/models"));
}

#[cfg(unix)]
#[tokio::test]
async fn spawn_agents_run_distinct_tasks_with_individual_options_in_order() {
    use agent_rs::agent::ToolStatus;
    use std::os::unix::fs::PermissionsExt;

    let _env_lock = ENV_LOCK.lock().expect("env lock");
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/api/tags"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "models": [{"name": "test:latest"}]
        })))
        .mount(&server)
        .await;
    let _url = EnvGuard::set("OLLAMA_URL", &server.uri());
    let _key = EnvGuard::remove("OPENAI_API_KEY");
    let _default = EnvGuard::set("AGENT_SPAWN_MODEL", "mock");
    let directory = tempfile::tempdir().expect("temporary directory");
    let script = directory.path().join("spawn-arguments.sh");
    std::fs::write(&script, "#!/bin/sh\nprintf '%s\\n' \"$@\"\n").expect("write stand-in");
    std::fs::set_permissions(&script, std::fs::Permissions::from_mode(0o755))
        .expect("make stand-in executable");
    let _bin = EnvGuard::set("AGENT_SPAWN_BIN", script.to_str().unwrap());
    let registry = ToolRegistry::new();
    let schema = &registry
        .definitions()
        .iter()
        .find(|tool| tool.name == "spawn")
        .unwrap()
        .parameters;
    assert!(schema["properties"].get("agents").is_some());
    assert_eq!(schema["properties"].as_object().unwrap().len(), 2);

    let result = registry.execute("batch".into(), "spawn", json!({
        "intent": "test distinct tasks",
        "agents": [
            {"task": "first instruction", "model": "ollama:test:latest", "conversation_id": "existing-session"},
            {"task": "second instruction"}
        ]
    })).await;
    assert_eq!(result.status, ToolStatus::Success, "{}", result.content);
    let (first, second) = result
        .content
        .split_once("\n\n[SPAWNED AGENT OUTPUT]")
        .expect("two outputs");
    assert!(first.contains("--model\nollama:test:latest"), "{first}");
    assert!(first.contains("--resume\nexisting-session"), "{first}");
    assert!(first.contains("first instruction"), "{first}");
    assert!(!first.contains("second instruction"));
    assert!(second.contains("--model\nmock"), "{second}");
    assert!(second.contains("second instruction"), "{second}");
    assert!(!second.contains("--resume"));
}

#[tokio::test]
async fn spawn_agents_reject_invalid_batches_before_launch() {
    use agent_rs::agent::ToolStatus;
    let registry = ToolRegistry::new();
    for args in [
        json!({"intent": "validate batch", "agents": []}),
        json!({"intent": "validate batch", "agents": [{"task": "one"}], "task": "two"}),
        json!({"intent": "validate batch", "agents": [{"task": "one"}], "num_subagents": 2}),
        json!({"intent": "validate batch", "agents": [{"task": "one"}], "model": "bad"}),
        json!({"intent": "validate batch", "agents": [{"task": "one"}], "conversation_id": "old"}),
        json!({"intent": "validate batch"}),
        json!({"intent": "validate batch", "agents": [{"task": "  "}]}),
        json!({"intent": "validate batch", "agents": [{"task": "one"}, {"task": ""}]}),
        json!({"intent": "validate batch", "agents": [{"task": "one", "unexpected": true}]}),
    ] {
        let result = registry.execute("invalid".into(), "spawn", args).await;
        assert_eq!(result.status, ToolStatus::Error, "{}", result.content);
        assert!(!result.content.contains("[SPAWNED AGENT OUTPUT]"));
    }
}

#[cfg(target_os = "macos")]
#[tokio::test]
async fn spawn_does_not_play_completion_sound() {
    use std::os::unix::fs::PermissionsExt;

    let _env_lock = ENV_LOCK.lock().expect("env lock");
    let temp_home = tempfile::tempdir().expect("temp home");
    let fake_bin = tempfile::tempdir().expect("fake bin");
    let sound_log = temp_home.path().join("sound.log");
    let afplay = fake_bin.path().join("afplay");
    std::fs::write(
        &afplay,
        "#!/bin/sh\nprintf '%s' \"$1\" > \"$AGENT_SOUND_LOG\"\n",
    )
    .expect("write fake afplay");
    std::fs::set_permissions(&afplay, std::fs::Permissions::from_mode(0o755))
        .expect("make fake afplay executable");

    let path = format!(
        "{}:{}",
        fake_bin.path().display(),
        std::env::var("PATH").expect("PATH")
    );
    let _home = EnvGuard::set("HOME", temp_home.path().to_str().expect("UTF-8 home path"));
    let _path = EnvGuard::set("PATH", &path);
    let _sound_log = EnvGuard::set(
        "AGENT_SOUND_LOG",
        sound_log.to_str().expect("UTF-8 sound log path"),
    );
    let _model = EnvGuard::set("AGENT_SPAWN_MODEL", "mock");

    spawn(SpawnArgs {
        agents: vec![SpawnAgent {
            task: "run echo hi".to_string(),
            model: None,
            conversation_id: None,
        }],
    })
    .await
    .expect("spawn");

    tokio::time::sleep(Duration::from_millis(50)).await;
    assert!(
        !sound_log.exists(),
        "subagent completion should not play a sound"
    );
}

#[tokio::test]
async fn spawn_uses_shared_agent_loop_with_mock_provider() {
    let _env_lock = ENV_LOCK.lock().expect("env lock");
    let _guard = EnvGuard::set("AGENT_SPAWN_MODEL", "mock");

    let output = spawn(SpawnArgs {
        agents: vec![SpawnAgent {
            task: "run echo hi".to_string(),
            model: None,
            conversation_id: None,
        }],
    })
    .await
    .expect("spawn");

    assert!(output.contains("[SPAWNED AGENT OUTPUT]"));
    assert!(output.contains("Tool completed: hi"));
    assert!(output.contains("sessionId: "));
    assert!(output.contains("[CONVERSATION ID]"));
}

#[tokio::test]
async fn spawn_agents_execute_distinct_commands_with_mock_provider() {
    use agent_rs::agent::ToolStatus;

    let _env_lock = ENV_LOCK.lock().expect("env lock");
    let _model = EnvGuard::set("AGENT_SPAWN_MODEL", "mock");
    let home = tempfile::tempdir().expect("temporary home");
    let _home = EnvGuard::set("HOME", home.path().to_str().expect("home path"));
    let result = ToolRegistry::new()
        .execute(
            "distinct".into(),
            "spawn",
            json!({
                "intent": "verify distinct mock agent commands",
                "agents": [
                    {"task": "run echo first-batch-result"},
                    {"task": "run echo second-batch-result"}
                ]
            }),
        )
        .await;
    assert_eq!(result.status, ToolStatus::Success, "{}", result.content);
    assert_eq!(result.content.matches("[SPAWNED AGENT OUTPUT]").count(), 2);
    assert_eq!(result.content.matches("Tool completed: hi").count(), 2);
    assert!(result.content.contains("run echo first-batch-result"));
    assert!(result.content.contains("run echo second-batch-result"));
    assert!(result.content.find("first-batch-result") < result.content.find("second-batch-result"));
}

#[cfg(unix)]
#[tokio::test]
async fn spawn_runs_requested_agents_in_parallel_with_distinct_tasks() {
    use std::os::unix::fs::PermissionsExt;
    use tempfile::tempdir;

    let _env_lock = ENV_LOCK.lock().expect("env lock");
    let directory = tempdir().expect("temporary directory");
    let script = directory.path().join("parallel-spawn.sh");
    std::fs::write(&script, "#!/bin/sh\nsleep 0.2\nprintf '%s\\n' \"$@\"\n")
        .expect("write stand-in");
    std::fs::set_permissions(&script, std::fs::Permissions::from_mode(0o755))
        .expect("make stand-in executable");
    let _bin = EnvGuard::set("AGENT_SPAWN_BIN", script.to_str().expect("script path"));

    let started = std::time::Instant::now();
    let output = spawn(SpawnArgs {
        agents: (0..5)
            .map(|index| SpawnAgent {
                task: format!("assigned task {index}"),
                model: None,
                conversation_id: None,
            })
            .collect(),
    })
    .await
    .expect("spawn");

    assert!(started.elapsed() < Duration::from_millis(600));
    for index in 0..5 {
        assert_eq!(
            output.matches(&format!("assigned task {index}")).count(),
            1,
            "{output}"
        );
    }
    assert_eq!(output.matches("--no-subagent").count(), 5);
}

#[cfg(unix)]
#[tokio::test]
async fn spawn_does_not_hang_when_a_descendant_keeps_output_pipes_open() {
    use std::os::unix::fs::PermissionsExt;
    use tempfile::tempdir;
    use tokio::time::timeout;

    let _env_lock = ENV_LOCK.lock().expect("env lock");
    let directory = tempdir().expect("temporary directory");
    let script = directory.path().join("hold-pipes.sh");
    std::fs::write(
        &script,
        "#!/bin/sh\n(sleep 60) &\nprintf 'sessionId: fake\\n'\n",
    )
    .expect("write stand-in");
    std::fs::set_permissions(&script, std::fs::Permissions::from_mode(0o755))
        .expect("make stand-in executable");
    let _bin = EnvGuard::set("AGENT_SPAWN_BIN", script.to_str().expect("script path"));

    let output = timeout(
        Duration::from_secs(1),
        spawn(SpawnArgs {
            agents: vec![SpawnAgent {
                task: "ignored".to_string(),
                model: None,
                conversation_id: None,
            }],
        }),
    )
    .await
    .expect("spawn should not wait for descendant-held pipes")
    .expect("spawn");

    assert!(output.contains("sessionId: fake"));
}

#[tokio::test]
async fn spawn_rejects_removed_num_subagents_argument() {
    use agent_rs::agent::ToolStatus;
    let registry = ToolRegistry::new();
    let schema = &registry
        .definitions()
        .iter()
        .find(|tool| tool.name == "spawn")
        .unwrap()
        .parameters;
    assert!(schema["properties"].get("num_subagents").is_none());
    let result = registry
        .execute(
            "old".into(),
            "spawn",
            json!({
                "intent": "reject removed argument", "agents": [{"task": "must not run"}], "num_subagents": 2
            }),
        )
        .await;
    assert_eq!(result.status, ToolStatus::Error);
    assert!(
        result.content.contains("num_subagents"),
        "{}",
        result.content
    );
}

#[cfg(unix)]
#[tokio::test]
async fn spawn_marks_its_child_as_disallowing_subagents() {
    use std::os::unix::fs::PermissionsExt;
    use tempfile::tempdir;

    let _env_lock = ENV_LOCK.lock().expect("env lock");
    let directory = tempdir().expect("temporary directory");
    let script = directory.path().join("spawn-arguments.sh");
    std::fs::write(&script, "#!/bin/sh\nprintf '%s\\n' \"$@\"\n").expect("write stand-in");
    std::fs::set_permissions(&script, std::fs::Permissions::from_mode(0o755))
        .expect("make stand-in executable");
    let _bin = EnvGuard::set("AGENT_SPAWN_BIN", script.to_str().expect("script path"));

    let output = spawn(SpawnArgs {
        agents: vec![SpawnAgent {
            task: "assigned task".to_string(),
            model: None,
            conversation_id: None,
        }],
    })
    .await
    .expect("spawn");

    assert!(output.contains("--no-subagent"));
}

#[cfg(unix)]
#[tokio::test]
async fn cancelling_spawn_terminates_its_process_group() {
    use agent_rs::agent::{CancellationToken, ToolStatus};
    use std::os::unix::fs::PermissionsExt;
    use tempfile::tempdir;
    use tokio::time::{sleep, timeout};

    let _env_lock = ENV_LOCK.lock().expect("env lock");
    let directory = tempdir().expect("temporary directory");
    let script = directory.path().join("spawn-stand-in.sh");
    let child_pid_file = directory.path().join("child.pid");
    std::fs::write(
        &script,
        format!(
            "#!/bin/sh\nsleep 60 &\necho $! > '{}'\nwait\n",
            child_pid_file.display()
        ),
    )
    .expect("write stand-in");
    std::fs::set_permissions(&script, std::fs::Permissions::from_mode(0o755))
        .expect("make stand-in executable");

    let _bin = EnvGuard::set("AGENT_SPAWN_BIN", script.to_str().expect("script path"));
    let cancellation = CancellationToken::new();
    let registry = ToolRegistry::new();
    let task = tokio::spawn({
        let cancellation = cancellation.clone();
        async move {
            registry
                .execute_cancellable(
                    "spawn-cancel".to_string(),
                    "spawn",
                    json!({ "agents": [{"task": "ignored"}], "intent": "exercise cancellation" }),
                    &cancellation,
                )
                .await
        }
    });

    timeout(Duration::from_secs(1), async {
        while !child_pid_file.exists() {
            sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .expect("spawned process should start");
    let child_pid = std::fs::read_to_string(&child_pid_file)
        .expect("read child pid")
        .trim()
        .parse::<i32>()
        .expect("numeric child pid");

    cancellation.cancel();
    let result = timeout(Duration::from_secs(2), task)
        .await
        .expect("cancelled spawn should finish")
        .expect("spawn task should not panic");
    assert_eq!(result.status, ToolStatus::Error);
    assert_eq!(result.content, "tool call cancelled");

    timeout(Duration::from_secs(1), async {
        loop {
            let output = std::process::Command::new("ps")
                .args(["-o", "stat=", "-p", &child_pid.to_string()])
                .output()
                .expect("inspect child process");
            let status = String::from_utf8_lossy(&output.stdout);
            if status.trim().is_empty() || status.trim_start().starts_with('Z') {
                break;
            }
            sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .expect("spawn child process should be terminated");
}

struct EnvGuard {
    key: &'static str,
    previous: Option<String>,
}

impl EnvGuard {
    fn remove(key: &'static str) -> Self {
        let previous = std::env::var(key).ok();
        unsafe { std::env::remove_var(key) };
        Self { key, previous }
    }

    fn set(key: &'static str, value: &str) -> Self {
        let previous = std::env::var(key).ok();
        unsafe {
            std::env::set_var(key, value);
        }
        Self { key, previous }
    }
}

impl Drop for EnvGuard {
    fn drop(&mut self) {
        unsafe {
            match &self.previous {
                Some(value) => std::env::set_var(self.key, value),
                None => std::env::remove_var(self.key),
            }
        }
    }
}

#[tokio::test]
async fn shell_preserves_full_output_and_times_out_after_child_exits_with_open_pipes() {
    let output = run_shell_command(RunShellCommandArgs {
        cmd: "head -c 200000 /dev/zero | tr '\\0' x; printf TAIL; printf STDERR >&2; exit 7".into(),
        timeout: 5,
    })
    .await
    .expect("output");
    assert_eq!(
        output,
        format!("{}TAILSTDERR\n(exit code: 7)", "x".repeat(200_000))
    );

    let timeout = tokio::time::timeout(
        std::time::Duration::from_secs(2),
        run_shell_command(RunShellCommandArgs {
            cmd: "(sleep 30) & echo done".into(),
            timeout: 1,
        }),
    )
    .await
    .expect("overall timeout must include pipe drain")
    .expect("timeout result");
    assert!(timeout.contains("(exit code: 124)"));
}
