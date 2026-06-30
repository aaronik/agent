use agent_rs::agent::{ToolCall, ToolResult, ToolStatus};
use agent_rs::display::TerminalDisplay;
use serde_json::json;

fn strip_ansi(text: &str) -> String {
    let mut out = String::new();
    let mut chars = text.chars().peekable();
    while let Some(ch) = chars.next() {
        if ch == '\x1b' {
            for next in chars.by_ref() {
                if next == 'm' {
                    break;
                }
            }
        } else {
            out.push(ch);
        }
    }
    out
}
#[test]
fn assistant_markdown_renders_common_markdown_features() {
    let display = TerminalDisplay::new();
    let rendered = display.format_assistant_content(
        "# Title\n\nBefore **bold** and `inline`.\n\n- one\n- two\n\n```rust\nfn main() {\n    println!(\"hi\");\n}\n```\nAfter",
    );
    let plain = strip_ansi(&rendered);

    assert!(plain.contains("Title"));
    assert!(plain.contains("Before bold and inline."));
    assert!(plain.contains("one"));
    assert!(plain.contains("two"));
    assert!(plain.contains("fn main"));
    assert!(plain.contains("println!"));
    assert!(plain.contains("After"));
    assert!(!plain.contains("```"));
    assert!(
        rendered.contains("\x1b[38;2;"),
        "fenced code should be syntax-highlighted with syntect true-color ANSI"
    );
    assert_ne!(
        rendered, plain,
        "markdown renderer should apply terminal styling"
    );
}

#[test]
fn assistant_plain_text_content_is_preserved() {
    let display = TerminalDisplay::new();
    let rendered = display.format_assistant_content("Plain text");

    assert_eq!(strip_ansi(&rendered).trim(), "Plain text");
}

#[test]
fn tool_panels_render_without_raw_tool_call_json() {
    let display = TerminalDisplay::new();
    let rendered = display.format_tool_result(&ToolResult {
        tool_call_id: "call_1".to_string(),
        name: "run_shell_command".to_string(),
        status: ToolStatus::Success,
        content: "hi\n".to_string(),
        elapsed_ms: None,
    });

    assert!(rendered.contains("╭─"));
    assert!(rendered.contains("run_shell_command"));
    assert!(rendered.contains("[OK Done]"));
    assert!(rendered.contains("│"));
    assert!(rendered.contains("hi"));
    assert!(rendered.contains("╰"));
    assert!(!rendered.contains("tool_call_id"));
    assert!(!rendered.contains("\"cmd\""));
}

#[test]
fn shell_command_panel_shows_full_command_without_truncation() {
    let display = TerminalDisplay::new();
    let long_command = format!("printf '{}'", "0123456789".repeat(30));
    let rendered = display.format_tool_start(&ToolCall {
        id: "call_1".to_string(),
        name: "run_shell_command".to_string(),
        arguments: json!({"cmd": long_command, "timeout": 30}),
    });

    assert!(rendered.contains("cmd=printf"));
    assert!(rendered.contains("7890123456789"));
    assert!(!rendered.contains('…'));
    assert!(!rendered.contains("..."));
}

#[test]
fn communicate_renders_as_progress_text() {
    let display = TerminalDisplay::new();
    let rendered = display.format_tool_result(&ToolResult {
        tool_call_id: "call_1".to_string(),
        name: "communicate".to_string(),
        status: ToolStatus::Success,
        content: "working".to_string(),
        elapsed_ms: Some(12),
    });

    assert!(rendered.contains("working"));
    assert!(rendered.contains("12ms"));
    assert!(rendered.ends_with('\n'));
}

#[test]
fn tool_start_panel_summarizes_running_call() {
    let display = TerminalDisplay::new();
    let rendered = display.format_tool_start(&ToolCall {
        id: "call_1".to_string(),
        name: "fetch".to_string(),
        arguments: json!({"url": "https://example.com"}),
    });

    assert!(rendered.contains("fetch"));
    assert!(rendered.contains("[> Running]"));
    assert!(rendered.contains("url=https://example.com"));
}

#[test]
fn tool_result_panel_shows_elapsed_time() {
    let display = TerminalDisplay::new();
    let rendered = display.format_tool_result(&ToolResult {
        tool_call_id: "call_1".to_string(),
        name: "fetch".to_string(),
        status: ToolStatus::Success,
        content: "ok".to_string(),
        elapsed_ms: Some(1_234),
    });

    assert!(rendered.contains("[OK Done]"));
    assert!(rendered.contains("1.2s"));
}

#[test]
fn no_live_env_disables_live_mode() {
    let _guard = EnvGuard::set("AGENT_NO_LIVE", "1");

    assert!(!TerminalDisplay::new().live_enabled());
}

#[test]
fn diff_panels_render_with_diff_ansi() {
    let display = TerminalDisplay::new();
    let rendered = display.format_tool_result(&ToolResult {
        tool_call_id: "call_1".to_string(),
        name: "search_replace".to_string(),
        status: ToolStatus::Success,
        content: "Successfully replaced 1 occurrence(s)\n\nDiff:\n--- a\n+++ b\n@@ -1 +1 @@\n-old\n+new\n".to_string(),
        elapsed_ms: None,
    });

    assert!(rendered.contains("\x1b[31m-old\x1b[0m"));
    assert!(rendered.contains("\x1b[32m+new\x1b[0m"));
}

#[test]
fn live_tool_result_can_replace_running_panel_in_place() {
    let display = TerminalDisplay::new();
    let call = ToolCall {
        id: "call_1".to_string(),
        name: "run_shell_command".to_string(),
        arguments: json!({"cmd": "echo hi"}),
    };
    let start = display.format_tool_start(&call);
    let rendered = display.format_tool_result_replacing_start_for_call(
        &ToolResult {
            tool_call_id: "call_1".to_string(),
            name: "run_shell_command".to_string(),
            status: ToolStatus::Success,
            content: "hi\n".to_string(),
            elapsed_ms: None,
        },
        Some(&call),
        start.lines().count(),
    );

    assert!(rendered.starts_with("\x1b[1A\x1b[2K\r"));
    assert_eq!(
        rendered.matches("\x1b[1A\x1b[2K\r").count(),
        start.lines().count()
    );
    assert!(rendered.contains("[OK Done]"));
    assert!(rendered.contains("cmd=echo hi"));
    assert!(rendered.contains("hi"));
}

#[test]
fn read_file_content_with_exit_code_text_still_renders_success() {
    let display = TerminalDisplay::new();
    let rendered = display.format_tool_result(&ToolResult {
        tool_call_id: "call_1".to_string(),
        name: "read_file".to_string(),
        status: ToolStatus::Success,
        content: "[FILE]: ./notes.txt\nread_file  [ERR Done (7)]  0ms\n(exit code: 7)\n"
            .to_string(),
        elapsed_ms: Some(0),
    });

    assert!(rendered.contains("read_file"));
    assert!(rendered.contains("[OK Done]"));
    assert!(rendered.contains("read_file  [ERR Done (7)]  0ms"));
    assert!(rendered.contains("(exit code: 7)"));
}

#[test]
fn shell_command_exit_code_marker_still_renders_error() {
    let display = TerminalDisplay::new();
    let rendered = display.format_tool_result(&ToolResult {
        tool_call_id: "call_1".to_string(),
        name: "run_shell_command".to_string(),
        status: ToolStatus::Success,
        content: "boom\n(exit code: 7)".to_string(),
        elapsed_ms: None,
    });

    assert!(rendered.contains("[ERR Done (7)]"));
    assert!(!rendered.contains("(exit code: 7)"));
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
