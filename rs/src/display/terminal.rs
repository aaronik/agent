use std::collections::HashMap;
use std::io::{self, Write};
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

#[derive(Debug)]
pub struct TerminalDisplay {
    live_enabled: bool,
    active_calls: Mutex<HashMap<String, ToolCall>>,
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
        let call = self
            .active_calls
            .lock()
            .ok()
            .and_then(|mut calls| calls.remove(&result.tool_call_id));
        print!(
            "{}",
            self.format_tool_result_with_call(result, call.as_ref())
        );
        flush_stdout();
    }

    pub fn render_tool_start(&self, call: &ToolCall) {
        if let Ok(mut calls) = self.active_calls.lock() {
            calls.insert(call.id.clone(), call.clone());
        }
        if self.live_enabled {
            print!("{}", self.format_tool_start(call));
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

    fn format_tool_result_with_call(&self, result: &ToolResult, call: Option<&ToolCall>) -> String {
        if result.name == "communicate" {
            return format!("{DIM}{ITALIC}{}{RESET}\n", result.content.trim());
        }

        let exit_code = extract_exit_code(&result.content);
        let visual_status = visual_status(result.status.clone(), exit_code);
        let title = tool_title(&result.name, visual_status);
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

fn tool_call_body(call: &ToolCall) -> Vec<String> {
    format_args_one_line(call)
        .into_iter()
        .map(|line| styled(DIM, &line))
        .collect()
}

fn format_args_one_line(call: &ToolCall) -> Option<String> {
    let args = call.arguments.as_object()?;
    if args.is_empty() {
        return None;
    }

    let mut parts = Vec::new();
    for (key, value) in args {
        let rendered = match value {
            serde_json::Value::String(value) => value.clone(),
            other => other.to_string(),
        };
        parts.push(format!("{key}={rendered}"));
    }

    let mut line = parts.join(", ");
    if line.chars().count() > 120 {
        line = line.chars().take(117).collect::<String>();
        line.push_str("...");
    }
    Some(line)
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
        let line = truncate_visible(&line, content_width);
        let padding = content_width.saturating_sub(visible_width(&line));
        out.push_str(&format!(
            "{GREY}│{RESET}{}{line}{}{}{GREY}│{RESET}\n",
            " ".repeat(PANEL_PADDING),
            " ".repeat(padding),
            " ".repeat(PANEL_PADDING)
        ));
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

fn visible_width(text: &str) -> usize {
    strip_ansi(text).chars().count()
}

fn truncate_visible(text: &str, max_width: usize) -> String {
    if visible_width(text) <= max_width {
        return text.to_string();
    }

    let target = max_width.saturating_sub(1);
    let mut out = String::new();
    let mut visible = 0;
    let mut chars = text.chars().peekable();
    while let Some(ch) = chars.next() {
        if ch == '\x1b' {
            out.push(ch);
            for next in chars.by_ref() {
                out.push(next);
                if next == 'm' {
                    break;
                }
            }
            continue;
        }
        if visible >= target {
            break;
        }
        out.push(ch);
        visible += 1;
    }
    out.push('…');
    out.push_str(RESET);
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
