use std::collections::HashMap;
use std::io::{self, IsTerminal, Write};
use std::sync::Mutex;

use unicode_segmentation::UnicodeSegmentation;
use unicode_width::UnicodeWidthStr;

mod live;
pub use live::LiveRenderer;

use crate::agent::{AgentMessage, ToolCall, ToolResult, ToolStatus};

const RESET: &str = "\x1b[0m";
const DIM: &str = "\x1b[2m";
const ITALIC: &str = "\x1b[3m";
const GREY: &str = "\x1b[90m";
const CYAN: &str = "\x1b[36m";
const GREEN: &str = "\x1b[32m";
const RED: &str = "\x1b[31m";
const PANEL_MIN_WIDTH: usize = 42;
const PANEL_MAX_WIDTH: usize = 120;
const PANEL_PADDING: usize = 2;
const SPINNER_FRAMES: [&str; 10] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];

#[derive(Debug)]
pub struct TerminalDisplay {
    live_enabled: bool,
    active_calls: Mutex<HashMap<String, ToolCall>>,
    // Serialize whole rendering transactions, not just individual print! calls.
    renderer: Mutex<Option<LiveRenderer>>,
}

impl Default for TerminalDisplay {
    fn default() -> Self {
        Self::new()
    }
}

impl TerminalDisplay {
    pub fn new() -> Self {
        Self {
            live_enabled: std::env::var("AGENT_NO_LIVE").ok().as_deref() != Some("1"),
            active_calls: Mutex::new(HashMap::new()),
            renderer: Mutex::new(None),
        }
    }

    pub fn live_enabled(&self) -> bool {
        self.live_enabled
    }

    pub fn assistant_stream_remainder(streamed: &str, final_content: &str) -> String {
        if final_content.is_empty() || streamed.ends_with(final_content) {
            return String::new();
        }
        if let Some(remainder) = final_content.strip_prefix(streamed) {
            return remainder.to_string();
        }

        let overlap = final_content
            .char_indices()
            .map(|(index, _)| index)
            .chain(std::iter::once(final_content.len()))
            .filter(|&length| length > 0 && length <= streamed.len())
            .rev()
            .find(|&length| streamed.ends_with(&final_content[..length]))
            .unwrap_or_default();
        if overlap > 0 {
            return final_content[overlap..].to_string();
        }

        format!("\n{final_content}")
    }

    pub fn format_standalone_footer(status_line: &str) -> String {
        format!("\n\n{status_line}")
    }

    fn transaction(&self, update: impl FnOnce(&mut LiveRenderer) -> String) {
        let Ok(mut renderer) = self.renderer.lock() else {
            return;
        };
        let Some(renderer) = renderer.as_mut() else {
            return;
        };
        let mut rendered = String::new();
        if let Ok((width, height)) = crossterm::terminal::size()
            && renderer.size() != (width.max(1), height.max(1))
        {
            rendered.push_str(&renderer.resize_at(width, height, terminal_cursor_position()));
        }
        rendered.push_str(&update(renderer));
        write_render_transaction(&rendered, true);
    }

    fn output(&self, text: &str) {
        let Ok(mut renderer) = self.renderer.lock() else {
            return;
        };
        let rendered = if let Some(renderer) = renderer.as_mut() {
            let mut rendered = String::new();
            if let Ok((width, height)) = crossterm::terminal::size()
                && renderer.size() != (width.max(1), height.max(1))
            {
                rendered.push_str(&renderer.resize_at(width, height, terminal_cursor_position()));
            }
            rendered.push_str(&renderer.output(text));
            rendered
        } else {
            text.to_string()
        };
        write_render_transaction(&rendered, renderer.is_some());
    }

    pub fn render_turn_submitted(&self, status_line: &str) {
        if self.live_enabled
            && io::stdout().is_terminal()
            && let Ok((width, height)) = crossterm::terminal::size()
            && let Ok(mut renderer) = self.renderer.lock()
        {
            let mut live = LiveRenderer::new(width, height);
            write_render_transaction(&live.start(status_line), true);
            *renderer = Some(live);
        }
    }

    pub fn update_working_footer(&self, status_line: &str) {
        self.transaction(|renderer| renderer.status(status_line));
    }

    pub fn update_working_input(&self, input: &str, cursor: usize, mode: &str) {
        self.transaction(|renderer| {
            renderer.input(input, cursor, mode, &working_directory_prompt())
        });
    }

    pub fn update_spinner(&self, frame_index: usize) {
        self.transaction(|renderer| renderer.spinner(frame_index));
    }

    pub fn finish_turn(&self) {
        let Ok(mut renderer) = self.renderer.lock() else {
            return;
        };
        if let Some(mut live) = renderer.take() {
            let mut rendered = String::new();
            if let Ok((width, height)) = crossterm::terminal::size()
                && live.size() != (width.max(1), height.max(1))
            {
                rendered.push_str(&live.resize_at(width, height, terminal_cursor_position()));
            }
            rendered.push_str(&live.finish());
            write_render_transaction(&rendered, true);
        }
    }

    pub fn render_assistant_delta(&self, text: &str) {
        self.output(&sanitize_terminal_text(text));
    }

    pub fn render_new_message(&self, message: &AgentMessage) {
        match message {
            AgentMessage::System { .. } | AgentMessage::Tool(_) => {}
            AgentMessage::User { content } => {
                self.output(&format!("\n{}\n\n", sanitize_terminal_text(content)));
            }
            AgentMessage::UserWithImages { content, images } => {
                self.output(&format!(
                    "\n{}\n[attached {} image(s)]\n\n",
                    sanitize_terminal_text(content),
                    images.len()
                ));
            }
            AgentMessage::Assistant(assistant) => {
                if !assistant.content.trim().is_empty() {
                    self.output(&format!(
                        "\n{}\n",
                        self.format_assistant_content(&assistant.content)
                    ));
                }
            }
        }
    }

    pub fn render_tool_result(&self, result: &ToolResult) {
        let active = self
            .active_calls
            .lock()
            .ok()
            .and_then(|mut calls| calls.remove(&result.tool_call_id));
        let rendered = self.format_tool_result_with_call(result, active.as_ref());
        // Durable output is append-only. A panel may already be in scrollback,
        // so relative cursor erasure cannot safely replace it.
        self.output(&rendered);
    }

    pub fn render_tool_start(&self, call: &ToolCall) {
        if let Ok(mut calls) = self.active_calls.lock() {
            calls.insert(call.id.clone(), call.clone());
        }
        if self.live_enabled {
            self.output(&self.format_tool_start(call));
        }
    }

    pub fn format_assistant_content(&self, content: &str) -> String {
        format_markdown(&sanitize_terminal_text(content))
    }

    pub fn format_tool_start(&self, call: &ToolCall) -> String {
        let title = tool_title(&call.name, VisualToolStatus::Running);
        let body = tool_call_body(call);
        format_panel(&title, &body)
    }

    pub fn format_tool_result(&self, result: &ToolResult) -> String {
        self.format_tool_result_with_call(result, None)
    }

    pub fn format_tool_result_for_call(
        &self,
        result: &ToolResult,
        call: Option<&ToolCall>,
    ) -> String {
        self.format_tool_result_with_call(result, call)
    }

    fn format_tool_result_with_call(&self, result: &ToolResult, call: Option<&ToolCall>) -> String {
        if result.name == "communicate" {
            let elapsed = result
                .elapsed_ms
                .map(|elapsed_ms| format!(" ({})", format_elapsed(elapsed_ms)))
                .unwrap_or_default();
            return format!(
                "{DIM}{ITALIC}{}{elapsed}{RESET}\n",
                sanitize_terminal_text(result.content.trim())
            );
        }

        let exit_code = extract_shell_exit_code(&result.name, &result.content);
        let visual_status = visual_status(result.status.clone(), exit_code);
        let title = tool_result_title(&result.name, visual_status, result.elapsed_ms);
        let mut body = call.map(tool_call_body).unwrap_or_default();
        let mut result_content = sanitize_terminal_text(&remove_shell_exit_code_marker(
            &result.name,
            &result.content,
        ));
        result_content = format_tool_content(&result.name, &result_content);
        if !result_content.trim().is_empty() {
            body.extend(result_content.lines().map(str::to_string));
        }
        if body.is_empty() {
            body.push(styled(DIM, ""));
        }
        format_panel(&title, &body)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum VisualToolStatus {
    Running,
    Done,
    Error(Option<i32>),
}

fn visual_status(status: ToolStatus, exit_code: Option<i32>) -> VisualToolStatus {
    match status {
        ToolStatus::Error => VisualToolStatus::Error(exit_code),
        ToolStatus::Success if exit_code.is_some_and(|code| code != 0) => {
            VisualToolStatus::Error(exit_code)
        }
        ToolStatus::Success => VisualToolStatus::Done,
    }
}

fn tool_title(name: &str, status: VisualToolStatus) -> String {
    let name = sanitize_terminal_text(name);
    let status_text = match status {
        VisualToolStatus::Running => styled(CYAN, "[> Running]"),
        VisualToolStatus::Done => styled(GREEN, "[OK Done]"),
        VisualToolStatus::Error(Some(code)) => styled(RED, &format!("[ERR Done ({code})]")),
        VisualToolStatus::Error(None) => styled(RED, "[ERR Done]"),
    };
    format!("{}  {status_text}", styled(CYAN, &name))
}

fn tool_result_title(name: &str, status: VisualToolStatus, elapsed_ms: Option<u64>) -> String {
    let mut title = tool_title(name, status);
    if let Some(elapsed_ms) = elapsed_ms {
        title.push_str(&format!("  {}", styled(DIM, &format_elapsed(elapsed_ms))));
    }
    title
}

fn format_elapsed(elapsed_ms: u64) -> String {
    if elapsed_ms < 1_000 {
        format!("{elapsed_ms}ms")
    } else if elapsed_ms < 10_000 {
        format!("{:.1}s", elapsed_ms as f64 / 1_000.0)
    } else {
        format!("{}s", (elapsed_ms + 500) / 1_000)
    }
}

fn tool_call_body(call: &ToolCall) -> Vec<String> {
    format_args_lines(call)
        .into_iter()
        .map(|line| styled(DIM, &line))
        .collect()
}

fn format_args_lines(call: &ToolCall) -> Vec<String> {
    let Some(args) = call.arguments.as_object() else {
        return Vec::new();
    };
    if args.is_empty() {
        return Vec::new();
    }

    let mut lines = Vec::new();
    for (key, value) in args {
        let rendered = match value {
            serde_json::Value::String(value) => sanitize_terminal_text(value),
            other => sanitize_terminal_text(&other.to_string()),
        };
        let mut rendered_lines = rendered.lines();
        if let Some(first) = rendered_lines.next() {
            lines.push(format!("{key}={first}"));
            lines.extend(rendered_lines.map(str::to_string));
        } else {
            lines.push(format!("{key}="));
        }
    }
    lines
}

fn format_tool_content(name: &str, content: &str) -> String {
    if content.contains("\nDiff:\n") {
        return color_diff(content);
    }
    if name == "read_file" {
        return highlight_read_file(content).unwrap_or_else(|| content.to_string());
    }
    content.to_string()
}

fn format_panel(title: &str, body: &[String]) -> String {
    let width = crossterm::terminal::size()
        .map(|(width, _)| usize::from(width))
        .unwrap_or(PANEL_MAX_WIDTH + 1);
    format_panel_at_width(title, body, width)
}

pub fn format_panel_at_width(title: &str, body: &[String], width: usize) -> String {
    let terminal_max = width.saturating_sub(1).clamp(1, PANEL_MAX_WIDTH);
    // On narrow terminals use plain wrapped lines instead of forcing a panel
    // whose borders/title are wider than the screen.
    if terminal_max < PANEL_MIN_WIDTH {
        let mut out = String::from("\n");
        for line in std::iter::once(title).chain(body.iter().map(String::as_str)) {
            for line in wrap_visible(line, terminal_max) {
                out.push_str(&line);
                out.push_str(RESET);
                out.push('\n');
            }
        }
        return out;
    }
    let content_width = panel_content_width(title, body, terminal_max);
    let panel_width = content_width + PANEL_PADDING * 2 + 2;
    let title_prefix = format!("{}╭─ {title} ", GREY);
    let top_fill = panel_width
        .saturating_sub(visible_width(&title_prefix))
        .saturating_sub(1);

    let mut out = String::new();
    out.push('\n');
    out.push_str(&title_prefix);
    out.push_str(&"─".repeat(top_fill));
    out.push_str(&format!("╮{RESET}\n"));

    let body_lines = if body.is_empty() {
        vec![String::new()]
    } else {
        body.to_vec()
    };
    for line in body_lines {
        for line in wrap_visible(&line, content_width) {
            let padding = content_width.saturating_sub(visible_width(&line));
            out.push_str(&format!(
                "{GREY}│{RESET}{}{line}{}{}{GREY}│{RESET}\n",
                " ".repeat(PANEL_PADDING),
                " ".repeat(padding),
                " ".repeat(PANEL_PADDING)
            ));
        }
    }

    out.push_str(&format!(
        "{GREY}╰{}╯{RESET}\n",
        "─".repeat(panel_width.saturating_sub(2))
    ));
    out
}

fn panel_content_width(title: &str, body: &[String], terminal_max: usize) -> usize {
    let title_width = visible_width(title).saturating_add(2);
    let body_width = body
        .iter()
        .map(|line| visible_width(line))
        .max()
        .unwrap_or(0);
    title_width
        .max(body_width)
        .max(PANEL_MIN_WIDTH - PANEL_PADDING * 2 - 2)
        .min(terminal_max - PANEL_PADDING * 2 - 2)
}

fn styled(style: &str, text: &str) -> String {
    format!("{style}{text}{RESET}")
}

fn extract_shell_exit_code(name: &str, content: &str) -> Option<i32> {
    if name == "run_shell_command" {
        extract_exit_code(content)
    } else {
        None
    }
}

fn extract_exit_code(content: &str) -> Option<i32> {
    let marker = "(exit code:";
    let rest = content.split_once(marker)?.1;
    let code = rest.split_once(')')?.0.trim();
    code.parse::<i32>().ok()
}

fn remove_shell_exit_code_marker(name: &str, content: &str) -> String {
    if name == "run_shell_command" {
        remove_exit_code_marker(content)
    } else {
        content.to_string()
    }
}

fn remove_exit_code_marker(content: &str) -> String {
    if let Some((before, _)) = content.split_once("(exit code:") {
        before.trim_end().to_string()
    } else {
        content.to_string()
    }
}

/// Makes externally supplied text inert before it reaches a terminal. Renderer
/// generated SGR sequences are added only after this boundary.
fn sanitize_terminal_text(text: &str) -> String {
    text.chars()
        .map(|character| match character {
            '\x1b' => '␛',
            character if character.is_control() && character != '\n' && character != '\t' => '�',
            character => character,
        })
        .collect()
}

fn visible_width(text: &str) -> usize {
    strip_ansi(text).width()
}

fn wrap_visible(text: &str, max_width: usize) -> Vec<String> {
    let max_width = max_width.max(1);
    let mut out = Vec::new();
    let mut line = String::new();
    let mut visible = 0;
    let mut remaining = text;
    while !remaining.is_empty() {
        if remaining.starts_with("\x1b[")
            && let Some(end) = remaining.find('m')
        {
            line.push_str(&remaining[..=end]);
            remaining = &remaining[end + 1..];
            continue;
        }
        let grapheme = remaining.graphemes(true).next().expect("nonempty text");
        let width = grapheme.width();
        if visible > 0 && visible + width > max_width {
            out.push(std::mem::take(&mut line));
            visible = 0;
        }
        line.push_str(grapheme);
        visible += width;
        remaining = &remaining[grapheme.len()..];
    }
    if !line.is_empty() || out.is_empty() {
        out.push(line);
    }
    out
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

fn working_directory_prompt() -> String {
    let path = std::env::current_dir()
        .map(|path| path.display().to_string())
        .unwrap_or_else(|_| "no path".to_string());
    if let Some(home) = std::env::var_os("HOME") {
        let home = std::path::PathBuf::from(home).display().to_string();
        if path == home {
            return "~".to_string();
        }
        if let Some(suffix) = path.strip_prefix(&format!("{home}/")) {
            return format!("~/{suffix}");
        }
    }
    path
}

fn working_input_preview(text: &str) -> String {
    text.chars()
        .map(|character| match character {
            '\n' => '↵',
            '\r' => '↵',
            '\t' => '→',
            character if character.is_control() => '�',
            character => character,
        })
        .collect()
}

fn char_byte_index(text: &str, character_index: usize) -> usize {
    text.char_indices()
        .nth(character_index)
        .map(|(index, _)| index)
        .unwrap_or(text.len())
}

fn truncate_to_width(text: &str, width: usize) -> String {
    let mut used = 0;
    text.graphemes(true)
        .take_while(|grapheme| {
            used += grapheme.width();
            used <= width
        })
        .collect()
}

fn terminal_cursor_position() -> Option<(u16, u16)> {
    let _input = super::TERMINAL_INPUT.lock().ok()?;
    crossterm::cursor::position().ok()
}

fn write_render_transaction(rendered: &str, synchronized: bool) {
    let mut out = io::stdout().lock();
    // Supporting terminals present footer erasure, output, and redraw as one
    // frame. Unsupported terminals safely ignore synchronized-output mode.
    if synchronized && out.is_terminal() {
        let _ = out.write_all(b"\x1b[?2026h");
    }
    let _ = out.write_all(rendered.as_bytes());
    if synchronized && out.is_terminal() {
        let _ = out.write_all(b"\x1b[?2026l");
    }
    let _ = out.flush();
}

fn format_markdown(content: &str) -> String {
    let code_blocks = markdown_code_blocks(content);
    if code_blocks.is_empty() {
        return render_markdown_with_termimad(content);
    }

    let mut out = String::new();
    let mut cursor = 0;
    for block in code_blocks {
        if cursor < block.source_start {
            out.push_str(&render_markdown_with_termimad(
                &content[cursor..block.source_start],
            ));
        }
        out.push_str(&format_highlighted_code_block(&block.language, &block.code));
        cursor = block.source_end;
    }
    if cursor < content.len() {
        out.push_str(&render_markdown_with_termimad(&content[cursor..]));
    }
    out
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct MarkdownCodeBlock {
    source_start: usize,
    source_end: usize,
    language: String,
    code: String,
}

#[derive(Clone, Debug)]
struct ActiveMarkdownCodeBlock {
    source_start: usize,
    source_end: usize,
    language: String,
    code: String,
}

fn render_markdown_with_termimad(content: &str) -> String {
    format!("{}", termimad::MadSkin::default_dark().term_text(content))
}

fn markdown_code_blocks(content: &str) -> Vec<MarkdownCodeBlock> {
    use pulldown_cmark::{CodeBlockKind, Event, Options, Parser, Tag, TagEnd};

    let mut blocks = Vec::new();
    let mut active: Option<ActiveMarkdownCodeBlock> = None;

    for (event, range) in Parser::new_ext(content, Options::all()).into_offset_iter() {
        match event {
            Event::Start(Tag::CodeBlock(kind)) => {
                let language = match kind {
                    CodeBlockKind::Fenced(info) => info
                        .split_whitespace()
                        .next()
                        .unwrap_or_default()
                        .to_string(),
                    CodeBlockKind::Indented => String::new(),
                };
                active = Some(ActiveMarkdownCodeBlock {
                    source_start: range.start,
                    source_end: range.end,
                    language,
                    code: String::new(),
                });
            }
            Event::Text(text) => {
                if let Some(block) = &mut active {
                    block.code.push_str(&text);
                }
            }
            Event::End(TagEnd::CodeBlock) => {
                if let Some(block) = active.take() {
                    blocks.push(MarkdownCodeBlock {
                        source_start: block.source_start,
                        source_end: block.source_end,
                        language: block.language,
                        code: block.code,
                    });
                }
            }
            _ => {}
        }
    }

    blocks
}

fn format_highlighted_code_block(language: &str, code: &str) -> String {
    let highlighted = highlight_code(code, language).unwrap_or_else(|| code.to_string());
    let mut out = String::new();
    if !out.ends_with('\n') {
        out.push('\n');
    }
    out.push_str(highlighted.trim_end_matches('\n'));
    out.push('\n');
    out
}

fn highlight_code(code: &str, language: &str) -> Option<String> {
    let syntax_set = syntect::parsing::SyntaxSet::load_defaults_newlines();
    let syntax = if language.is_empty() {
        syntax_set.find_syntax_plain_text()
    } else {
        syntax_set
            .find_syntax_by_token(language)
            .or_else(|| syntax_set.find_syntax_by_extension(language))
            .unwrap_or_else(|| syntax_set.find_syntax_plain_text())
    };
    highlight_with_syntax_set(code, syntax, &syntax_set)
}

fn highlight_with_syntax_set(
    code: &str,
    syntax: &syntect::parsing::SyntaxReference,
    syntax_set: &syntect::parsing::SyntaxSet,
) -> Option<String> {
    let theme_set = syntect::highlighting::ThemeSet::load_defaults();
    let theme = theme_set
        .themes
        .get("base16-ocean.dark")
        .or_else(|| theme_set.themes.values().next())?;
    let mut highlighter = syntect::easy::HighlightLines::new(syntax, theme);
    let mut out = String::new();
    for line in syntect::util::LinesWithEndings::from(code) {
        let ranges = highlighter.highlight_line(line, syntax_set).ok()?;
        out.push_str(&syntect::util::as_24_bit_terminal_escaped(
            &ranges[..],
            false,
        ));
    }
    Some(out)
}

fn color_diff(content: &str) -> String {
    content
        .lines()
        .map(|line| {
            if line.starts_with('+') && !line.starts_with("+++") {
                format!("\x1b[32m{line}\x1b[0m")
            } else if line.starts_with('-') && !line.starts_with("---") {
                format!("\x1b[31m{line}\x1b[0m")
            } else if line.starts_with("@@") {
                format!("\x1b[36m{line}\x1b[0m")
            } else {
                line.to_string()
            }
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn highlight_read_file(content: &str) -> Option<String> {
    let (header, body) = content.split_once('\n')?;
    let path = header.strip_prefix("[FILE]: ")?.trim();
    let extension = std::path::Path::new(path)
        .extension()
        .and_then(|extension| extension.to_str());
    let syntax_set = syntect::parsing::SyntaxSet::load_defaults_newlines();
    let syntax = extension
        .and_then(|extension| syntax_set.find_syntax_by_extension(extension))
        .unwrap_or_else(|| syntax_set.find_syntax_plain_text());
    let mut out = String::new();
    out.push_str(header);
    out.push('\n');
    out.push_str(&highlight_with_syntax_set(body, syntax, &syntax_set)?);
    Some(out)
}
