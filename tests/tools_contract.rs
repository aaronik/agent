use std::sync::Mutex;

use agent_rs::tools::browser::{BrowserControlArgs, browser_control};
use agent_rs::tools::fetch::{FetchArgs, fetch};
use agent_rs::tools::files::{
    ReadFileArgs, SearchReplaceArgs, WriteFileArgs, read_file, search_replace, write_file,
};
use agent_rs::tools::image::{GenImageArgs, gen_image};
use agent_rs::tools::registry::ToolRegistry;
use agent_rs::tools::shell::{RunShellCommandArgs, run_shell_command};
use agent_rs::tools::spawn::{SpawnArgs, spawn};
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
            "browser_control",
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
async fn browser_control_fails_fast_when_playwright_missing_from_path() {
    let _env_lock = ENV_LOCK.lock().expect("env lock");
    let _path = EnvGuard::set("PATH", "");
    let output = browser_control(BrowserControlArgs {
        javascript: "return await page.title();".to_string(),
        url: Some("https://example.com".to_string()),
        profile: None,
        timeout: 1,
        close: false,
        reset: false,
        visible: false,
    })
    .await
    .expect_err("missing playwright should fail before Chrome/profile checks");

    assert!(output.contains("playwright is not installed or not found in PATH"));
    assert!(output.contains("browser_control cannot work"));
}

#[tokio::test]
async fn spawn_uses_shared_agent_loop_with_mock_provider() {
    let _guard = EnvGuard::set("AGENT_SPAWN_MODEL", "mock");

    let output = spawn(SpawnArgs {
        task: "run echo hi".to_string(),
        conversation_id: None,
    })
    .await
    .expect("spawn");

    assert!(output.contains("[SPAWNED AGENT OUTPUT]"));
    assert!(output.contains("Tool completed: hi"));
    assert!(output.contains("sessionId: "));
    assert!(output.contains("[CONVERSATION ID]"));
}

struct EnvGuard {
    key: &'static str,
    previous: Option<String>,
}

impl EnvGuard {
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
