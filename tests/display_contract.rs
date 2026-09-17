use agent_rs::agent::{ToolCall, ToolResult, ToolStatus};
use agent_rs::display::TerminalDisplay;
use serde_json::json;

fn has_trigger_time(text: &str) -> bool {
    text.split_whitespace().any(|word| {
        let bytes = word.as_bytes();
        bytes.len() == 8
            && bytes[2] == b':'
            && bytes[5] == b':'
            && bytes
                .iter()
                .enumerate()
                .all(|(index, byte)| matches!(index, 2 | 5) || byte.is_ascii_digit())
    })
}

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
fn streamed_fences_render_like_replay_across_chunk_boundaries() {
    use agent_rs::display::terminal::AssistantMarkdownStream;

    let display = TerminalDisplay::new();
    for source in [
        "```python\nprint('hello')\n```",
        "~~~~rust\nfn main() {}\n~~~\n~~~~\n",
        "```unknown\n漢字🙂\n```\n",
        "```python\nprint('unfinished')",
        "```\nliteral ``` inside code\n```\n",
        "   ```rust\n   fn main() {}\n   ```\n",
    ] {
        let mut stream = AssistantMarkdownStream::default();
        let mut rendered = String::new();
        for character in source.chars() {
            let delta = stream.push(&character.to_string());
            assert!(!delta.contains("```python"), "raw fence leaked: {delta}");
            rendered.push_str(&delta);
        }
        rendered.push_str(&stream.finish());
        assert_eq!(
            rendered,
            display.format_assistant_content(source),
            "{source}"
        );
        assert_eq!(stream.push("next response"), "next response");
        assert!(stream.finish().is_empty());
    }
}

#[test]
fn streaming_prose_remains_immediate_and_fence_like_text_is_preserved() {
    use agent_rs::display::terminal::AssistantMarkdownStream;

    for source in [
        "ordinary prose",
        "inline ``` is not a block\n",
        "``not fenced\n",
        "    ```indented\n",
        "```bad`info\n",
    ] {
        let mut stream = AssistantMarkdownStream::default();
        let mut rendered = String::new();
        for character in source.chars() {
            rendered.push_str(&stream.push(&character.to_string()));
        }
        rendered.push_str(&stream.finish());
        assert_eq!(rendered, source);
    }
    let mut stream = AssistantMarkdownStream::default();
    assert_eq!(stream.push("hello"), "hello");
    assert_eq!(stream.push(" world\n"), " world\n");
    assert!(stream.push("``").is_empty());
    assert!(stream.push("`python\nprint('hi')\n").is_empty());
    let rendered = stream.push("```\nAfter");
    assert!(rendered.contains("\x1b[38;2;"));
    assert!(strip_ansi(&rendered).contains("print('hi')"));
    assert!(strip_ansi(&rendered).trim_end().ends_with("After"));
    assert!(!rendered.contains("```"));
    assert!(stream.finish().is_empty());
}

#[test]
fn streamed_code_survives_scrollback_and_interrupted_message_boundaries() {
    use agent_rs::display::terminal::AssistantMarkdownStream;

    let mut stream = AssistantMarkdownStream::default();
    let mut terminal = vt100::Parser::new(10, 40, 1000);
    let mut live = LiveRenderer::new(40, 10);
    feed(&mut terminal, live.start("CODE-STATUS"));
    assert!(stream.push("```python\n").is_empty());
    for index in 0..120 {
        assert!(
            stream
                .push(&format!("print('line-{index:03}')\n"))
                .is_empty()
        );
        feed(&mut terminal, live.spinner(index));
    }
    // Finishing an interrupted/unclosed block must not lose its contents.
    feed(&mut terminal, live.output(&stream.finish()));
    feed(&mut terminal, live.output(&stream.push("NEXT-MESSAGE\n")));
    feed(&mut terminal, live.finish());
    terminal.screen_mut().set_scrollback(1000);
    let mut transcript = String::new();
    loop {
        transcript.push_str(&terminal.screen().contents());
        let offset = terminal.screen().scrollback();
        if offset == 0 {
            break;
        }
        terminal
            .screen_mut()
            .set_scrollback(offset.saturating_sub(10));
    }
    for index in 0..120 {
        assert!(
            transcript.contains(&format!("line-{index:03}")),
            "{transcript}"
        );
    }
    assert!(transcript.contains("NEXT-MESSAGE"));
    assert!(!transcript.contains("```"));
    assert!(!transcript.contains("CODE-STATUS"));

    assert!(stream.push("```unknown\n\x1b[2J\n").is_empty());
    let interrupted = stream.finish();
    assert!(!interrupted.contains("\x1b[2J"));
    assert!(interrupted.contains("␛[2J"));
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
        subagent_usages: Vec::new(),
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
        subagent_usages: Vec::new(),
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
    assert!(has_trigger_time(&strip_ansi(&rendered)));
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
        subagent_usages: Vec::new(),
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
                subagent_usages: Vec::new(),
    });

    assert!(rendered.contains("\x1b[31m-old\x1b[0m"));
    assert!(rendered.contains("\x1b[32m+new\x1b[0m"));
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
        subagent_usages: Vec::new(),
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
        subagent_usages: Vec::new(),
    });

    assert!(rendered.contains("[ERR Done (7)]"));
    assert!(!rendered.contains("(exit code: 7)"));
}

// Exercise the same state machine used by TerminalDisplay against a terminal,
// rather than asserting that potentially incorrect escape strings were emitted.
use agent_rs::display::terminal::LiveRenderer;
use unicode_width::UnicodeWidthStr;

fn feed(parser: &mut vt100::Parser, text: String) {
    parser.process(text.as_bytes());
}

#[test]
fn live_output_preserves_scrollback_and_cleans_up_the_footer() {
    let mut terminal = vt100::Parser::new(12, 40, 1000);
    let mut live = LiveRenderer::new(40, 12);
    terminal.process(b"existing transcript\r\n");
    feed(&mut terminal, live.start("cost: $0"));
    for i in 0..100 {
        feed(&mut terminal, live.output(&format!("OUTPUT-{i:03}\n")));
        feed(&mut terminal, live.spinner(i));
    }
    feed(&mut terminal, live.finish());
    assert!(!terminal.screen().hide_cursor());
    let mut history = terminal.screen().contents();
    for offset in 1..=120 {
        terminal.screen_mut().set_scrollback(offset);
        history.push_str(&terminal.screen().contents());
    }
    assert!(history.contains("existing transcript"));
    for i in 0..100 {
        assert!(
            history.contains(&format!("OUTPUT-{i:03}")),
            "lost output {i}"
        );
    }
    assert!(!history.contains("cost: $0"));
    assert!(!history.contains('⠋'));
    assert!(!history.contains('⠹'));
}

#[test]
fn live_input_wraps_unicode_and_keeps_the_cursor_in_view() {
    let mut live = LiveRenderer::new(24, 10);
    live.start("status");
    let input = "漢字🙂e\u{301}".repeat(40);
    live.input(
        &input,
        input.chars().count(),
        "INSERT",
        "/a/very/long/directory",
    );
    let rows = live.input_lines();
    assert!(rows.len() <= 6);
    assert!(rows.iter().all(|row| row.width() < 24));
    assert!(rows.iter().any(|row| row.contains('│')));
    live.input(&input, 0, "NORMAL", "/a/very/long/directory");
    assert!(live.input_lines().iter().any(|row| row.contains('│')));
}

#[test]
fn resizing_and_multiline_input_do_not_leave_spinner_or_status_remnants() {
    let mut terminal = vt100::Parser::new(24, 80, 1000);
    let mut live = LiveRenderer::new(80, 24);
    feed(&mut terminal, live.start("STATUS-MARKER"));
    feed(&mut terminal, live.output("before resize\n"));
    feed(
        &mut terminal,
        live.input(&"x".repeat(100), 100, "INSERT", "~/agent"),
    );
    for (width, height) in [(40, 12), (100, 35), (8, 4), (2, 2), (80, 24)] {
        terminal.screen_mut().set_size(height, width);
        feed(&mut terminal, live.resize(width, height));
        feed(
            &mut terminal,
            live.input("one\ntwo\nthree", 13, "INSERT", "~/agent"),
        );
        feed(&mut terminal, live.input("", 0, "INSERT", "~/agent"));
        feed(&mut terminal, live.spinner(3));
        feed(&mut terminal, live.output("after resize\n"));
    }
    feed(&mut terminal, live.finish());
    let visible = terminal.screen().contents();
    assert!(!visible.contains("STATUS-MARKER"), "{visible}");
    assert!(!visible.contains('⠸'), "{visible}");
    assert!(!visible.contains("~/agent"), "{visible}");
}

#[test]
fn live_stream_chunks_preserve_partial_lines_and_use_full_screen_scrolling() {
    let mut terminal = vt100::Parser::new(10, 30, 100);
    let mut live = LiveRenderer::new(30, 10);
    feed(&mut terminal, live.start("status"));
    for chunk in [
        "hello",
        " world",
        "\n",
        "漢字",
        "🙂",
        "!\n",
        "x".repeat(60).as_str(),
        "\nEND",
    ] {
        let rendered = live.output(chunk);
        assert!(!rendered.contains(";7r"));
        assert!(!rendered.contains("[1A\x1b[2K"));
        feed(&mut terminal, rendered);
        feed(&mut terminal, live.spinner(0));
    }
    feed(&mut terminal, live.finish());
    let visible = terminal.screen().contents();
    assert!(visible.contains("hello world"), "{visible}");
    assert!(visible.contains("漢字🙂!"), "{visible}");
    assert!(visible.contains("END"), "{visible}");
    assert!(!visible.contains("status"));
}

#[test]
fn tool_results_are_append_only_and_do_not_silently_drop_lines() {
    let display = TerminalDisplay::new();
    let content = (0..100)
        .map(|i| format!("line-{i:03}\n"))
        .collect::<String>();
    let rendered = display.format_tool_result(&ToolResult {
        tool_call_id: "large".into(),
        name: "fetch".into(),
        status: ToolStatus::Success,
        content,
        elapsed_ms: None,
        subagent_usages: Vec::new(),
    });
    assert!(!rendered.contains("\x1b[1A"));
    for i in 0..100 {
        assert!(rendered.contains(&format!("line-{i:03}")));
    }
}

#[test]
fn narrow_panels_fit_terminal_cells_without_losing_unicode_text() {
    use agent_rs::display::terminal::format_panel_at_width;
    for width in [8, 16, 40, 80] {
        let body = vec!["漢字🙂e\u{301}".repeat(10)];
        let rendered = format_panel_at_width("fetch [OK Done]", &body, width);
        let plain = strip_ansi(&rendered);
        assert!(
            plain.lines().all(|line| line.width() < width),
            "width {width}: {plain}"
        );
        assert_eq!(plain.matches('漢').count(), 10);
        assert_eq!(plain.matches('🙂').count(), 10);
    }
}

#[test]
fn multiline_cursor_near_the_start_is_visible_even_with_a_long_tail() {
    let mut live = LiveRenderer::new(30, 12);
    live.start("status");
    let input = format!("one\ntwo\n{}", "tail\n".repeat(50));
    live.input(&input, 5, "INSERT", "~/agent");
    assert!(live.input_lines().iter().any(|row| row.contains("t│wo")));
}

#[test]
fn pending_autowrap_survives_spinner_and_footer_growth() {
    let mut terminal = vt100::Parser::new(10, 20, 100);
    let mut live = LiveRenderer::new(20, 10);
    feed(&mut terminal, live.start("status"));
    feed(&mut terminal, live.output(&"x".repeat(20)));
    feed(
        &mut terminal,
        live.input("one\ntwo\nthree", 13, "INSERT", "~"),
    );
    feed(&mut terminal, live.spinner(5));
    feed(&mut terminal, live.output("Y\nEND"));
    feed(&mut terminal, live.finish());
    let screen = terminal.screen().contents();
    assert!(screen.contains(&"x".repeat(20)), "{screen}");
    assert!(screen.contains("Y\nEND"), "{screen}");
}

#[test]
fn long_input_keeps_an_animated_indicator_when_scrolled_to_the_cursor() {
    let mut live = LiveRenderer::new(20, 10);
    live.start("status");
    live.input(&"x".repeat(300), 300, "INSERT", "~/agent");
    live.spinner(2);
    assert!(live.input_lines()[0].starts_with("⠹ "));
}

#[test]
fn external_cursor_after_resize_is_used_instead_of_old_screen_coordinates() {
    let mut terminal = vt100::Parser::new(12, 40, 100);
    let mut live = LiveRenderer::new(40, 12);
    feed(&mut terminal, live.start("status"));
    feed(&mut terminal, live.output("before\n"));
    // Model an emulator that relocates the output cursor during resize.
    terminal.screen_mut().set_size(24, 80);
    terminal.process(b"\x1b[5;1Hanchor\r\n");
    feed(&mut terminal, live.resize_at(80, 24, Some((0, 5))));
    feed(&mut terminal, live.output("after\n"));
    feed(&mut terminal, live.finish());
    assert!(terminal.screen().contents().contains("anchor\nafter"));
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

#[test]
fn untrusted_content_cannot_emit_terminal_control_sequences() {
    let display = TerminalDisplay::new();
    let rendered = display.format_assistant_content("hello\x1b]52;c;secret\x07\x1b[2Jworld");
    assert!(!rendered.contains("\x1b]52"));
    assert!(!rendered.contains("\x1b[2J"));
    assert!(strip_ansi(&rendered).contains("␛"));
    let tool = display.format_tool_start(&ToolCall {
        id: "x".into(),
        name: "fetch".into(),
        arguments: json!({"url":"x\u{1b}[2J"}),
    });
    assert!(!tool.contains("\x1b[2J"));
}

fn transcript(terminal: &mut vt100::Parser) -> String {
    terminal.screen_mut().set_scrollback(0);
    let mut text = terminal.screen().contents();
    for offset in 1..=500 {
        terminal.screen_mut().set_scrollback(offset);
        text.push('\n');
        text.push_str(&terminal.screen().contents());
    }
    terminal.screen_mut().set_scrollback(0);
    text
}

#[test]
fn running_tools_are_transient_even_across_output_and_footer_growth() {
    let mut terminal = vt100::Parser::new(12, 80, 1000);
    let mut live = LiveRenderer::new(80, 12);
    feed(&mut terminal, live.start("status"));
    feed(&mut terminal, live.output("existing transcript\n"));
    let call = ToolCall {
        id: "one".into(),
        name: "fetch".into(),
        arguments: json!({"url": "https://example.com"}),
    };
    feed(&mut terminal, live.tool_start(&call));
    let visible = terminal.screen().contents();
    assert!(visible.contains("[> Running]"), "{visible}");
    assert!(visible.contains("https://example.com"), "{visible}");
    for i in 0..100 {
        feed(&mut terminal, live.output(&format!("OUTPUT-{i:03}\n")));
        feed(
            &mut terminal,
            live.input("one\ntwo\nthree\nfour", 18, "INSERT", "~"),
        );
        feed(&mut terminal, live.spinner(i));
        feed(&mut terminal, live.input("", 0, "INSERT", "~"));
    }
    let done = TerminalDisplay::new().format_tool_result_for_call(
        &ToolResult {
            tool_call_id: call.id.clone(),
            name: call.name.clone(),
            status: ToolStatus::Success,
            content: "RESULT-MARKER".into(),
            elapsed_ms: Some(123),
            subagent_usages: vec![],
        },
        Some(&call),
    );
    feed(&mut terminal, live.tool_result(&call.id, &done));
    assert!(terminal.screen().contents().contains("[OK Done]"));
    feed(&mut terminal, live.finish());
    let history = transcript(&mut terminal);
    assert!(!history.contains("Running"), "{history}");
    assert!(history.contains("existing transcript"));
    assert!(history.contains("RESULT-MARKER"));
    for i in 0..100 {
        assert!(history.contains(&format!("OUTPUT-{i:03}")));
    }
}

#[test]
fn running_preview_tracks_call_ids_and_cancellation_cleans_up() {
    let mut terminal = vt100::Parser::new(12, 80, 1000);
    let mut live = LiveRenderer::new(80, 12);
    feed(&mut terminal, live.start("status"));
    for (id, name) in [("a", "FIRST"), ("b", "SECOND")] {
        feed(
            &mut terminal,
            live.tool_start(&ToolCall {
                id: id.into(),
                name: name.into(),
                arguments: json!({}),
            }),
        );
    }
    // A repeated notification updates rather than duplicates a running call.
    feed(
        &mut terminal,
        live.tool_start(&ToolCall {
            id: "a".into(),
            name: "FIRST".into(),
            arguments: json!({}),
        }),
    );
    assert_eq!(terminal.screen().contents().matches("FIRST").count(), 1);
    feed(&mut terminal, live.tool_result("b", "SECOND completed\n"));
    let visible = terminal.screen().contents();
    assert!(visible.contains("[> Running] FIRST"), "{visible}");
    assert!(!visible.contains("[> Running] SECOND"), "{visible}");
    feed(&mut terminal, live.finish());
    let history = transcript(&mut terminal);
    assert!(!history.contains("Running"), "{history}");
    assert!(history.contains("SECOND completed"));
    assert!(!terminal.screen().hide_cursor());
}

#[test]
fn running_previews_are_bounded_sanitized_and_resize_safely() {
    let mut terminal = vt100::Parser::new(24, 80, 1000);
    let mut live = LiveRenderer::new(80, 24);
    feed(&mut terminal, live.start("status"));
    feed(&mut terminal, live.output("PRESERVE\n"));
    for i in 0..10 {
        let rendered = live.tool_start(&ToolCall {
            id: i.to_string(),
            name: "fetch".into(),
            arguments: json!({"url": format!("\u{1b}[2J{}", "漢字🙂\n".repeat(100))}),
        });
        assert!(!rendered.contains("\x1b[2J"));
        feed(&mut terminal, rendered);
    }
    assert!(terminal.screen().contents().contains('…'));
    // vt100 truncates screen cells on shrink rather than reflowing them like
    // native terminals. Put the marker into scrollback before a 2-column resize.
    feed(&mut terminal, live.output(&"history\n".repeat(40)));
    for (width, height) in [(40, 12), (100, 35), (8, 4), (2, 2), (80, 24)] {
        terminal.screen_mut().set_size(height, width);
        feed(&mut terminal, live.resize(width, height));
        feed(
            &mut terminal,
            live.input(&"draft".repeat(100), 500, "INSERT", "~"),
        );
        feed(&mut terminal, live.spinner(2));
        feed(&mut terminal, live.input("", 0, "INSERT", "~"));
    }
    feed(&mut terminal, live.finish());
    let history = transcript(&mut terminal);
    assert!(!history.contains("Running"), "{history}");
    assert!(!history.contains("url="), "{history}");
    assert!(history.contains("PRESERVE"));
}
