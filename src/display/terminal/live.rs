//! A transient footer over an append-only, normal-screen transcript.
//!
//! Before writing output we erase the footer, write normally, and reserve space
//! with ordinary full-screen line feeds. No partial scroll region, reverse
//! scrolling, or cursor-up deletion of durable output is used. A small terminal
//! parser tracks cursor/wrap/SGR state across streamed chunks; it is not a second
//! scrollback buffer. The real terminal owns the transcript.
use unicode_segmentation::UnicodeSegmentation;
use unicode_width::UnicodeWidthStr;

use super::{RESET, SPINNER_FRAMES, char_byte_index, truncate_to_width, working_input_preview};

pub struct LiveRenderer {
    terminal: vt100::Parser,
    width: u16,
    height: u16,
    active: bool,
    status: String,
    input: String,
    cursor: usize,
    mode: String,
    prompt: String,
    frame: usize,
    running_tools: Vec<(String, String)>,
}

impl std::fmt::Debug for LiveRenderer {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("LiveRenderer")
            .field("size", &self.size())
            .field("active", &self.active)
            .finish_non_exhaustive()
    }
}

impl LiveRenderer {
    pub fn new(width: u16, height: u16) -> Self {
        let width = width.max(1);
        let height = height.max(1);
        Self {
            terminal: vt100::Parser::new(height, width, 0),
            width,
            height,
            active: false,
            status: String::new(),
            input: String::new(),
            cursor: 0,
            mode: "INSERT".into(),
            prompt: super::working_directory_prompt(),
            frame: 0,
            running_tools: Vec::new(),
        }
    }

    pub fn size(&self) -> (u16, u16) {
        (self.width, self.height)
    }

    pub fn start(&mut self, status: &str) -> String {
        self.active = true;
        self.status = status.into();
        // Establish a known output position without erasing existing output.
        // LF at the physical bottom pushes one row into native scrollback.
        let prefix = format!("\x1b[?25l\x1b[r\x1b[{};1H\r\n", self.height);
        self.terminal.process(prefix.as_bytes());
        format!("{prefix}{}", self.draw())
    }

    fn restore_cursor(&self) -> String {
        let screen = self.terminal.screen();
        let mut out = String::from_utf8_lossy(&screen.cursor_state_formatted()).into_owned();
        out.push_str(&String::from_utf8_lossy(&screen.attributes_formatted()));
        out.push_str("\x1b[?25l");
        out
    }

    fn clear(&self) -> String {
        let row = self.terminal.screen().cursor_position().0;
        if row + 1 >= self.height {
            return String::new();
        }
        format!("{RESET}\x1b[{};1H\x1b[J{}", row + 2, self.restore_cursor())
    }

    pub fn input_lines(&self) -> Vec<String> {
        if self.height < 4 || self.width < 6 {
            return Vec::new();
        }
        // Leave a spare column so drawing the footer never triggers autowrap.
        let width = usize::from(self.width - 3);
        let max_rows = usize::from(self.height.saturating_sub(4).clamp(1, 6));
        let indicator = if self.mode == "NORMAL" { "〉" } else { ": " };
        let prompt = truncate_to_width(&working_input_preview(&self.prompt), width / 2);
        let prefix = format!("{prompt}{indicator}");
        let split = char_byte_index(&self.input, self.cursor);
        // Preserve explicit newlines; sanitize all other terminal controls.
        let clean = |text: &str| {
            text.split('\n')
                .map(working_input_preview)
                .collect::<Vec<_>>()
                .join("\n")
        };
        let marked = format!(
            "{}│{}",
            clean(&self.input[..split]),
            clean(&self.input[split..])
        );
        let mut lines = vec![truncate_to_width(&prefix, width.saturating_sub(1))];
        let mut cursor_row = 0;
        // The marker is tracked by its offset, not by searching for a glyph the
        // user may also have typed. Grapheme segmentation keeps emoji and
        // combining characters together when wrapping.
        let marker_offset = clean(&self.input[..split]).len();
        for (offset, grapheme) in marked.grapheme_indices(true) {
            if grapheme == "\n" {
                lines.push(String::new());
                continue;
            }
            let grapheme_width = grapheme.width();
            if lines.last().expect("one line").width() + grapheme_width > width {
                lines.push(String::new());
            }
            if offset == marker_offset {
                cursor_row = lines.len() - 1;
            }
            if grapheme_width <= width {
                lines.last_mut().expect("one line").push_str(grapheme);
            }
        }
        let first = cursor_row.saturating_sub(max_rows - 1);
        lines
            .into_iter()
            .skip(first)
            .take(max_rows)
            .enumerate()
            .map(|(index, line)| {
                let indicator = if index == 0 {
                    SPINNER_FRAMES[self.frame % SPINNER_FRAMES.len()]
                } else {
                    " "
                };
                format!("{indicator} {line}")
            })
            .collect()
    }

    /// Running calls never enter the terminal's durable transcript. Keep a
    /// stable order and update by ID so overlapping calls complete independently.
    pub fn tool_start(&mut self, call: &crate::agent::ToolCall) -> String {
        let mut out = self.clear();
        let summary = working_input_preview(&format!(
            "[> Running] {}  {}",
            call.name,
            super::format_args_lines(call).join("  ")
        ));
        if let Some((_, preview)) = self.running_tools.iter_mut().find(|(id, _)| id == &call.id) {
            *preview = summary;
        } else {
            self.running_tools.push((call.id.clone(), summary));
        }
        out.push_str(&self.draw());
        out
    }

    pub fn tool_result(&mut self, id: &str, completed: &str) -> String {
        self.running_tools
            .retain(|(running_id, _)| running_id != id);
        // output() clears the old transient area before appending the result,
        // then redraws only the tools that are still running.
        self.output(completed)
    }

    fn tool_lines(&self, input_rows: usize) -> Vec<String> {
        // Preserve the input viewport, status, separator, and an output row.
        let budget = usize::from(self.height)
            .saturating_sub(input_rows + 3)
            .min(3);
        if budget == 0 || self.width < 6 {
            return Vec::new();
        }
        let width = usize::from(self.width - 1);
        let overflow = self.running_tools.len() > budget;
        let shown = if overflow { budget - 1 } else { budget };
        let mut lines: Vec<_> = self
            .running_tools
            .iter()
            .take(shown)
            .map(|(_, text)| {
                if text.width() > width {
                    format!("{}…", truncate_to_width(text, width - 1))
                } else {
                    text.clone()
                }
            })
            .collect();
        if overflow {
            lines.push(truncate_to_width(
                &format!(
                    "[> Running] … {} more tools",
                    self.running_tools.len() - shown
                ),
                width,
            ));
        }
        lines
    }

    fn draw(&mut self) -> String {
        if !self.active {
            return String::new();
        }
        let lines = self.input_lines();
        if lines.is_empty() {
            return self.restore_cursor();
        }
        let tools = self.tool_lines(lines.len());
        let tool_rows = tools.len() as u16;
        let reserve = lines.len() as u16 + tool_rows + 2;
        let (row, col) = self.terminal.screen().cursor_position();
        let available = self.height - row - 1;
        let mut out = String::new();
        if available < reserve {
            let growth = reserve - available;
            // Ordinary LF at the full-screen bottom preserves native scrollback.
            let scroll = format!("\x1b[{};1H{}", self.height, "\r\n".repeat(growth as usize));
            out.push_str(&scroll);
            self.terminal.process(scroll.as_bytes());
            let new_row = row.saturating_sub(growth);
            let reposition = format!("\x1b[{};{}H", new_row + 1, col.min(self.width - 1) + 1);
            self.terminal.process(reposition.as_bytes());
            // CUP cancels pending wrap. Re-draw the final cell to restore it.
            if col >= self.width {
                let mut cell_col = self.width - 1;
                if self
                    .terminal
                    .screen()
                    .cell(new_row, cell_col)
                    .is_some_and(|cell| cell.is_wide_continuation())
                {
                    cell_col = cell_col.saturating_sub(1);
                }
                let cell = self
                    .terminal
                    .screen()
                    .cell(new_row, cell_col)
                    .map(|cell| cell.contents().to_string())
                    .unwrap_or_default();
                let wrap = format!("\x1b[{};{}H{cell}", new_row + 1, cell_col + 1);
                self.terminal.process(wrap.as_bytes());
            }
        }
        let row = self.terminal.screen().cursor_position().0;
        out.push_str(&format!("{RESET}\x1b[{};1H\x1b[J", row + 2));
        for (index, line) in tools.iter().enumerate() {
            out.push_str(&format!(
                "\x1b[{};1H\x1b[36m{line}{RESET}",
                row + 3 + index as u16
            ));
        }
        out.push_str(&format!(
            "\x1b[{};1H\x1b[2m{}",
            row + 3 + tool_rows,
            truncate_to_width(
                &working_input_preview(&self.status),
                usize::from(self.width - 1)
            )
        ));
        for (index, line) in lines.iter().enumerate() {
            out.push_str(&format!(
                "\x1b[{};1H\x1b[0;36m{line}{RESET}",
                row + 4 + tool_rows + index as u16
            ));
        }
        out.push_str(&self.restore_cursor());
        out
    }

    pub fn output(&mut self, text: &str) -> String {
        let mut out = self.clear();
        // The Unix input mode retains OPOST; explicit CRLF also works in raw
        // mode and in the terminal-state parser. Extra CR from ONLCR is harmless.
        let text = text.replace('\n', "\r\n");
        self.terminal.process(text.as_bytes());
        out.push_str(&text);
        out.push_str(&self.draw());
        out
    }

    pub fn status(&mut self, status: &str) -> String {
        let mut out = self.clear();
        self.status = status.into();
        out.push_str(&self.draw());
        out
    }

    pub fn input(&mut self, input: &str, cursor: usize, mode: &str, prompt: &str) -> String {
        let mut out = self.clear();
        self.input = input.into();
        self.cursor = cursor;
        self.mode = mode.into();
        self.prompt = prompt.into();
        out.push_str(&self.draw());
        out
    }

    pub fn spinner(&mut self, frame: usize) -> String {
        self.frame = frame;
        self.draw()
    }

    pub fn resize(&mut self, width: u16, height: u16) -> String {
        self.resize_at(width, height, None)
    }

    /// The runtime supplies the terminal's actual cursor after reflow, rather
    /// than guessing how a particular emulator resized its screen.
    pub fn resize_at(&mut self, width: u16, height: u16, cursor: Option<(u16, u16)>) -> String {
        let (width, height) = (width.max(1), height.max(1));
        if (width, height) == self.size() {
            return String::new();
        }
        self.width = width;
        self.height = height;
        self.terminal.screen_mut().set_size(height, width);
        if let Some((col, row)) = cursor {
            self.terminal.process(
                format!(
                    "\x1b[{};{}H",
                    row.min(height - 1) + 1,
                    col.min(width - 1) + 1
                )
                .as_bytes(),
            );
        }
        let mut out = self.clear();
        out.push_str(&self.draw());
        out
    }

    pub fn finish(&mut self) -> String {
        let mut out = self.clear();
        self.active = false;
        // A streamed final response need not end in LF. Keep the next prompt
        // and session-id line separate from the final assistant text.
        if self.terminal.screen().cursor_position().1 > 0 {
            out.push_str("\r\n");
        }
        out.push_str("\x1b[0m\x1b[?25h");
        out
    }
}
