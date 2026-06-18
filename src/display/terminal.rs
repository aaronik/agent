use std::collections::HashMap;
use std::io::{self, IsTerminal, Write};
use std::sync::Mutex;

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

#[derive(Clone, Debug)]
struct ActiveToolCall {
    call: ToolCall,
    rendered_line_count: Option<usize>,
}

#[derive(Debug)]
pub struct TerminalDisplay {
    live_enabled: bool,
    active_calls: Mutex<HashMap<String, ActiveToolCall>>,
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
        }
    }

    pub fn live_enabled(&self) -> bool {
        self.live_enabled
    }

    pub fn render_turn_submitted(&self) {
        println!("{DIM}{ITALIC}Working... Press Esc to abort.{RESET}");
        flush_stdout();
    }

    pub fn render_new_message(&self, message: &AgentMessage) {
        match message {
            AgentMessage::System { .. } | AgentMessage::Tool(_) => {}
            AgentMessage::User { content } => {
                println!("\n{content}\n");
            }
            AgentMessage::Assistant(assistant) => {
                if !assistant.content.trim().is_empty() {
                    println!("\n{}", assistant.content);
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
        let rendered =
            self.format_tool_result_with_call(result, active.as_ref().map(|active| &active.call));
        if self.live_enabled
            && io::stdout().is_terminal()
            && let Some(line_count) = active.and_then(|active| active.rendered_line_count)
        {
            print!("{}", clear_rendered_lines(line_count));
        }
        print!("{rendered}");
        flush_stdout();
    }

    pub fn render_tool_start(&self, call: &ToolCall) {
        let rendered = self.live_enabled.then(|| self.format_tool_start(call));
        let rendered_line_count = rendered.as_deref().map(rendered_line_count);
        if let Ok(mut calls) = self.active_calls.lock() {
            calls.insert(
                call.id.clone(),
                ActiveToolCall {
                    call: call.clone(),
                    rendered_line_count,
                },
            );
        }
        if let Some(rendered) = rendered {
            print!("{rendered}");
            flush_stdout();
        }
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

    pub fn format_tool_result_replacing_start_for_call(
        &self,
        result: &ToolResult,
        call: Option<&ToolCall>,
        start_rendered_line_count: usize,
    ) -> String {
        format!(
            "{}{}",
            clear_rendered_lines(start_rendered_line_count),
            self.format_tool_result_with_call(result, call)
        )
    }

    fn format_tool_result_with_call(&self, result: &ToolResult, call: Option<&ToolCall>) -> String {
        if result.name == "communicate" {
            let elapsed = result
                .elapsed_ms
                .map(|elapsed_ms| format!(" ({})", format_elapsed(elapsed_ms)))
                .unwrap_or_default();
            return format!("{DIM}{ITALIC}{}{elapsed}{RESET}\n", result.content.trim());
        }

        let exit_code = extract_exit_code(&result.content);
        let visual_status = visual_status(result.status.clone(), exit_code);
        let title = tool_result_title(&result.name, visual_status, result.elapsed_ms);
        let mut body = call.map(tool_call_body).unwrap_or_default();
        let mut result_content = remove_exit_code_marker(&result.content);
        result_content = format_tool_content(&result.name, &result_content);
        if !result_content.trim().is_empty() {
            body.extend(preview_lines(&result_content));
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
    let status_text = match status {
        VisualToolStatus::Running => styled(CYAN, "[> Running]"),
        VisualToolStatus::Done => styled(GREEN, "[OK Done]"),
        VisualToolStatus::Error(Some(code)) => styled(RED, &format!("[ERR Done ({code})]")),
        VisualToolStatus::Error(None) => styled(RED, "[ERR Done]"),
    };
    format!("{}  {status_text}", styled(CYAN, name))
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
            serde_json::Value::String(value) => value.clone(),
            other => other.to_string(),
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

fn preview_lines(content: &str) -> Vec<String> {
    let lines = content.lines().collect::<Vec<_>>();
    let mut out = Vec::new();
    for line in lines.iter().take(30) {
        out.push((*line).to_string());
    }
    if lines.len() > 30 {
        out.push(styled(DIM, "..."));
    }
    out
}

fn format_panel(title: &str, body: &[String]) -> String {
    let content_width = panel_content_width(title, body);
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

fn panel_content_width(title: &str, body: &[String]) -> usize {
    let terminal_max = crossterm::terminal::size()
        .ok()
        .map(|(width, _)| (width as usize).saturating_sub(4))
        .unwrap_or(PANEL_MAX_WIDTH)
        .clamp(PANEL_MIN_WIDTH, PANEL_MAX_WIDTH);
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

fn extract_exit_code(content: &str) -> Option<i32> {
    let marker = "(exit code:";
    let rest = content.split_once(marker)?.1;
    let code = rest.split_once(')')?.0.trim();
    code.parse::<i32>().ok()
}

fn remove_exit_code_marker(content: &str) -> String {
    if let Some((before, _)) = content.split_once("(exit code:") {
        before.trim_end().to_string()
    } else {
        content.to_string()
    }
}

fn rendered_line_count(rendered: &str) -> usize {
    rendered.lines().count()
}

fn clear_rendered_lines(line_count: usize) -> String {
    "\x1b[1A\x1b[2K\r".repeat(line_count)
}

fn visible_width(text: &str) -> usize {
    strip_ansi(text).chars().count()
}

fn wrap_visible(text: &str, max_width: usize) -> Vec<String> {
    if max_width == 0 || visible_width(text) <= max_width {
        return vec![text.to_string()];
    }

    let mut out = Vec::new();
    let mut line = String::new();
    let mut visible = 0;
    let mut chars = text.chars().peekable();
    while let Some(ch) = chars.next() {
        if ch == '\x1b' {
            line.push(ch);
            for next in chars.by_ref() {
                line.push(next);
                if next == 'm' {
                    break;
                }
            }
            continue;
        }

        if visible >= max_width {
            out.push(std::mem::take(&mut line));
            visible = 0;
        }
        line.push(ch);
        visible += 1;
    }

    if !line.is_empty() {
        out.push(line);
    }
    if out.is_empty() {
        out.push(String::new());
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

fn flush_stdout() {
    let _ = io::stdout().flush();
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
    let theme_set = syntect::highlighting::ThemeSet::load_defaults();
    let theme = theme_set
        .themes
        .get("base16-ocean.dark")
        .or_else(|| theme_set.themes.values().next())?;
    let mut highlighter = syntect::easy::HighlightLines::new(syntax, theme);
    let mut out = String::new();
    out.push_str(header);
    out.push('\n');
    for line in syntect::util::LinesWithEndings::from(body) {
        let ranges = highlighter.highlight_line(line, &syntax_set).ok()?;
        out.push_str(&syntect::util::as_24_bit_terminal_escaped(
            &ranges[..],
            false,
        ));
    }
    Some(out)
}
