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
const SPINNER_FRAMES: [&str; 10] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];

#[derive(Clone, Debug)]
struct ActiveToolCall {
    call: ToolCall,
    rendered_line_count: Option<usize>,
}

#[derive(Debug)]
pub struct TerminalDisplay {
    live_enabled: bool,
    active_calls: Mutex<HashMap<String, ActiveToolCall>>,
    working_footer: Mutex<Option<WorkingFooter>>,
}

#[derive(Clone, Copy, Debug)]
struct WorkingFooter {
    terminal_height: u16,
    input_rows: u16,
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
            working_footer: Mutex::new(None),
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

    pub fn render_turn_submitted(&self, status_line: &str) {
        if self.live_enabled
            && io::stdout().is_terminal()
            && let Ok((width, height)) = crossterm::terminal::size()
            && height >= 3
        {
            let status_line = truncate_to_width(status_line, width as usize);
            print!(
                "{}",
                Self::format_working_footer_start(&status_line, height)
            );
            if let Ok(mut footer) = self.working_footer.lock() {
                *footer = Some(WorkingFooter {
                    terminal_height: height,
                    input_rows: 1,
                });
            }
        }
        flush_stdout();
    }

    pub fn update_working_footer(&self, status_line: &str) {
        let footer = self.working_footer.lock().ok().and_then(|footer| *footer);
        if let Some(footer) = footer {
            let status_line = crossterm::terminal::size()
                .ok()
                .map(|(width, _)| truncate_to_width(status_line, width as usize))
                .unwrap_or_else(|| status_line.to_string());
            print!(
                "{}",
                Self::format_working_footer_update_rows(
                    &status_line,
                    footer.terminal_height,
                    footer.input_rows,
                )
            );
            flush_stdout();
        }
    }

    pub fn update_working_input(&self, input: &str, cursor: usize, mode: &str) {
        let Ok(mut footer_guard) = self.working_footer.lock() else {
            return;
        };
        let Some(mut footer) = *footer_guard else {
            return;
        };
        let max_input_rows = footer.terminal_height.saturating_sub(2).max(1);
        let input_rows = (input.split('\n').count().max(1) as u16).min(max_input_rows);
        if input_rows != footer.input_rows {
            print!(
                "{}",
                Self::format_working_footer_resize(
                    footer.input_rows,
                    input_rows,
                    footer.terminal_height,
                )
            );
            footer.input_rows = input_rows;
            *footer_guard = Some(footer);
        }
        print!(
            "{}",
            Self::format_working_input_update(
                input,
                cursor,
                mode,
                &working_directory_prompt(),
                footer.terminal_height,
            )
        );
        flush_stdout();
        debug_assert!(cursor <= input.chars().count());
    }

    pub fn update_spinner(&self, frame_index: usize) {
        let footer = self.working_footer.lock().ok().and_then(|footer| *footer);
        if let Some(footer) = footer {
            let frame = SPINNER_FRAMES[frame_index % SPINNER_FRAMES.len()];
            let input_start = footer.terminal_height - footer.input_rows + 1;
            print!("{}", Self::format_spinner_update(frame, input_start));
            flush_stdout();
        }
    }

    pub fn finish_turn(&self) {
        let footer = self
            .working_footer
            .lock()
            .ok()
            .and_then(|mut footer| footer.take());
        if let Some(footer) = footer {
            print!(
                "{}",
                Self::format_working_footer_finish(footer.terminal_height, footer.input_rows,)
            );
            flush_stdout();
        }
    }

    pub fn format_working_footer_start(status_line: &str, height: u16) -> String {
        let output_bottom = height.saturating_sub(3).max(1);
        format!(
            "\x1b[?25l\x1b[r\x1b[3S\x1b[1;{output_bottom}r\x1b[{};1H\x1b[2K{}{}\x1b[{output_bottom};1H\n",
            height.saturating_sub(2).max(1),
            Self::format_working_footer_update(status_line, height),
            Self::format_working_input_update("", 0, "INSERT", &working_directory_prompt(), height,)
        )
    }

    pub fn format_working_footer_update(status_line: &str, height: u16) -> String {
        Self::format_working_footer_update_rows(status_line, height, 1)
    }

    fn format_working_footer_update_rows(
        status_line: &str,
        height: u16,
        input_rows: u16,
    ) -> String {
        let status_row = height.saturating_sub(input_rows).max(1);
        format!(
            "\x1b[s\x1b[{};1H\x1b[2K\x1b[{status_row};1H\x1b[2K{status_line}\x1b[u",
            status_row.saturating_sub(1).max(1)
        )
    }

    pub fn format_working_footer_resize(old_rows: u16, new_rows: u16, height: u16) -> String {
        let old_rows = old_rows.max(1);
        let new_rows = new_rows.max(1);
        let output_bottom = height.saturating_sub(new_rows + 2).max(1);
        if new_rows > old_rows {
            let growth = new_rows - old_rows;
            format!("\x1b[s\x1b[r\x1b[{growth}S\x1b[1;{output_bottom}r\x1b[u\x1b[{growth}A")
        } else if old_rows > new_rows {
            let shrink = old_rows - new_rows;
            format!("\x1b[s\x1b[r\x1b[{shrink}T\x1b[1;{output_bottom}r\x1b[u\x1b[{shrink}B")
        } else {
            String::new()
        }
    }

    pub fn format_working_input_update(
        input: &str,
        cursor: usize,
        mode: &str,
        prompt: &str,
        height: u16,
    ) -> String {
        let split = char_byte_index(input, cursor);
        let mut marked = String::with_capacity(input.len() + 3);
        marked.push_str(&input[..split]);
        marked.push('│');
        marked.push_str(&input[split..]);
        let lines = marked.split('\n').collect::<Vec<_>>();
        let max_rows = height.saturating_sub(2).max(1) as usize;
        let first_line = lines.len().saturating_sub(max_rows);
        let visible = &lines[first_line..];
        let start_row = height
            .saturating_sub(visible.len() as u16)
            .saturating_add(1);
        let indicator = if mode == "NORMAL" { "〉" } else { ": " };
        let mut rendered = String::from("\x1b[s");
        for (index, line) in visible.iter().enumerate() {
            let row = start_row + index as u16;
            rendered.push_str(&format!("\x1b[{row};1H\x1b[2K"));
            if index == 0 {
                rendered.push_str(&format!(
                    "\x1b[38;5;14m{} \x1b[38;5;10m{prompt}\x1b[38;5;14m{indicator}\x1b[38;5;7m",
                    SPINNER_FRAMES[0]
                ));
            } else {
                rendered.push_str("\x1b[38;5;14m  · \x1b[38;5;7m");
            }
            rendered.push_str(&working_input_preview(line));
            rendered.push_str(RESET);
        }
        rendered.push_str("\x1b[u");
        rendered
    }

    pub fn format_spinner_update(frame: &str, height: u16) -> String {
        format!("\x1b[s\x1b[{height};1H\x1b[38;5;14m{frame}{RESET}\x1b[u")
    }

    pub fn format_clear_submitted_prompt_status() -> &'static str {
        ""
    }

    pub fn format_working_footer_finish(height: u16, input_rows: u16) -> String {
        let status_row = height.saturating_sub(input_rows).max(1);
        let separator_row = status_row.saturating_sub(1).max(1);
        let mut rendered = String::from("\x1b[s\x1b[r");
        for row in separator_row..=height {
            rendered.push_str(&format!("\x1b[{row};1H\x1b[2K"));
        }
        rendered.push_str("\x1b[u\x1b[?25h");
        rendered
    }

    pub fn render_assistant_delta(&self, text: &str) {
        print!("{text}");
        flush_stdout();
    }

    pub fn render_new_message(&self, message: &AgentMessage) {
        match message {
            AgentMessage::System { .. } | AgentMessage::Tool(_) => {}
            AgentMessage::User { content } => {
                println!("\n{content}\n");
            }
            AgentMessage::UserWithImages { content, images } => {
                println!("\n{content}\n[attached {} image(s)]\n", images.len());
            }
            AgentMessage::Assistant(assistant) => {
                if !assistant.content.trim().is_empty() {
                    println!("\n{}", self.format_assistant_content(&assistant.content));
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

    pub fn format_assistant_content(&self, content: &str) -> String {
        format_markdown(content)
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

        let exit_code = extract_shell_exit_code(&result.name, &result.content);
        let visual_status = visual_status(result.status.clone(), exit_code);
        let title = tool_result_title(&result.name, visual_status, result.elapsed_ms);
        let mut body = call.map(tool_call_body).unwrap_or_default();
        let mut result_content = remove_shell_exit_code_marker(&result.name, &result.content);
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
    text.chars().take(width).collect()
}

fn flush_stdout() {
    let _ = io::stdout().flush();
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
