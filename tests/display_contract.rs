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
fn streamed_assistant_final_render_only_appends_unstreamed_suffix() {
    assert_eq!(
        TerminalDisplay::assistant_stream_remainder("hello", "hello world"),
        " world"
    );
    assert_eq!(
        TerminalDisplay::assistant_stream_remainder("hello", "hello"),
        ""
    );
    assert_eq!(
        TerminalDisplay::assistant_stream_remainder("hello", "goodbye"),
        "\ngoodbye"
    );
    assert_eq!(
        TerminalDisplay::assistant_stream_remainder("hello worldhello world", "hello world"),
        ""
    );
}

#[test]
fn standalone_footer_keeps_a_blank_separator_above_status() {
    assert_eq!(
        TerminalDisplay::format_standalone_footer("Cost: $0"),
        "\n\nCost: $0"
    );
}

#[test]
fn working_footer_reserves_a_blank_separator_above_status() {
    let rendered = TerminalDisplay::format_working_footer_start("Cost: $0", 24);
    assert!(rendered.contains("\x1b[1;21r"));
    assert!(rendered.contains("\x1b[22;1H\x1b[2K"));
    assert!(rendered.contains("\x1b[23;1H\x1b[2KCost: $0"));
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

#[test]
fn working_footer_reserves_bottom_rows_and_renders_status_and_input() {
    let rendered = TerminalDisplay::format_working_footer_start(
        "cost: $0.01 | context: 90% (900/1,000) | model: mock",
        24,
    );

    assert!(rendered.contains("\x1b[1;21r"));
    assert!(rendered.contains("\x1b[23;1H\x1b[2K"));
    assert!(rendered.contains("cost: $0.01"));
    assert!(rendered.contains("\x1b[24;1H\x1b[2K"));
    assert!(rendered.contains("\x1b[38;5;14m: \x1b[38;5;7m│"));
    assert!(!rendered.contains("Working..."));
    assert!(!rendered.contains("-- INSERT --"));
    assert!(rendered.starts_with("\x1b[?25l\x1b[r\x1b[3S\x1b[1;21r"));
    assert!(rendered.ends_with("\x1b[21;1H\n"));
    assert!(!rendered.contains("\x1b[s\x1b[1;21r"));
}

#[test]
fn working_input_update_renders_multiline_text_on_multiple_footer_rows() {
    let rendered = TerminalDisplay::format_working_input_update(
        "first line\nsecond line",
        22,
        "INSERT",
        "~/projects/agent",
        24,
    );

    assert!(rendered.contains("\x1b[23;1H\x1b[2K"));
    assert!(rendered.contains("first line"));
    assert!(rendered.contains("\x1b[24;1H\x1b[2K"));
    assert!(rendered.contains("second line│"));
    assert!(!rendered.contains("first line↵second line"));
}

#[test]
fn working_footer_resize_grows_reserved_input_rows() {
    let rendered = TerminalDisplay::format_working_footer_resize(1, 2, 24);

    assert!(rendered.contains("\x1b[r\x1b[1S\x1b[1;20r"));
    assert!(rendered.ends_with("\x1b[u\x1b[1A"));
}

#[test]
fn working_footer_resize_shrinks_reserved_input_rows() {
    let rendered = TerminalDisplay::format_working_footer_resize(3, 1, 24);

    assert!(rendered.contains("\x1b[r\x1b[2T\x1b[1;21r"));
    assert!(rendered.ends_with("\x1b[u\x1b[2B"));
}

#[test]
fn working_input_update_renders_typed_text_and_preserves_output_cursor() {
    let insert = TerminalDisplay::format_working_input_update(
        "next question",
        13,
        "INSERT",
        "~/projects/agent",
        24,
    );
    let normal = TerminalDisplay::format_working_input_update(
        "next question",
        13,
        "NORMAL",
        "~/projects/agent",
        24,
    );

    assert!(insert.starts_with("\x1b[s"));
    assert!(insert.contains("\x1b[38;5;14m⠋ \x1b[38;5;10m~/projects/agent"));
    assert!(
        insert.contains("\x1b[38;5;10m~/projects/agent\x1b[38;5;14m: \x1b[38;5;7mnext question│")
    );
    assert!(!insert.contains("-- INSERT --"));
    assert!(
        normal.contains("\x1b[38;5;10m~/projects/agent\x1b[38;5;14m〉\x1b[38;5;7mnext question│")
    );
    assert!(!normal.contains("-- NORMAL --"));
    assert!(insert.ends_with("\x1b[u"));
}

#[test]
fn spinner_update_redraws_only_the_indicator_before_folder() {
    assert_eq!(
        TerminalDisplay::format_spinner_update("⠹", 24),
        "\x1b[s\x1b[24;1H\x1b[38;5;14m⠹\x1b[0m\x1b[u"
    );
}

#[test]
fn working_footer_hides_real_cursor_and_finish_restores_it() {
    let start = TerminalDisplay::format_working_footer_start("status", 24);
    let finish = TerminalDisplay::format_working_footer_finish(24, 1);

    assert!(start.starts_with("\x1b[?25l"));
    assert!(finish.ends_with("\x1b[u\x1b[?25h"));
}

#[test]
fn submitted_prompt_status_can_be_cleared_before_footer_starts() {
    assert_eq!(TerminalDisplay::format_clear_submitted_prompt_status(), "");
}

#[test]
fn working_footer_update_preserves_output_cursor() {
    let rendered = TerminalDisplay::format_working_footer_update("updated", 24);

    assert!(rendered.starts_with("\x1b[s"));
    assert!(rendered.contains("\x1b[23;1H\x1b[2Kupdated"));
    assert!(!rendered.contains("\x1b[24;1H"));
    assert!(!rendered.contains("Working..."));
    assert!(rendered.ends_with("\x1b[u"));
}

#[test]
fn working_footer_finish_restores_scroll_region_and_clears_footer() {
    let rendered = TerminalDisplay::format_working_footer_finish(24, 1);

    assert!(rendered.contains("\x1b[r"));
    assert!(rendered.contains("\x1b[23;1H\x1b[2K"));
    assert!(rendered.contains("\x1b[24;1H\x1b[2K"));
    assert!(rendered.ends_with("\x1b[u\x1b[?25h"));
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
