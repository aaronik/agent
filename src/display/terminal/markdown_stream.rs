//! Render complete prose lines and fenced blocks so Markdown split across provider
//! chunks receives the same treatment as a completed assistant message.
use super::{format_markdown, sanitize_terminal_text};

#[derive(Debug, Default)]
pub struct AssistantMarkdownStream {
    pending_line: String,
    block: String,
    fence: Option<(char, usize)>,
    table: String,
    table_candidate: Option<String>,
}

impl AssistantMarkdownStream {
    pub fn push(&mut self, text: &str) -> String {
        let mut out = String::new();
        for character in sanitize_terminal_text(text).chars() {
            self.pending_line.push(character);
            if character != '\n' {
                continue;
            }
            let line = std::mem::take(&mut self.pending_line);
            if let Some((marker, length)) = self.fence {
                self.block.push_str(&line);
                if closes_fence(&line, marker, length) {
                    out.push_str(&format_markdown(&std::mem::take(&mut self.block)));
                    self.fence = None;
                }
            } else if let Some(fence) = opening_fence(&line) {
                out.push_str(&self.flush_table());
                self.fence = Some(fence);
                self.block = line;
            } else if !self.table.is_empty() {
                if is_table_row(&line) {
                    self.table.push_str(&line);
                } else {
                    out.push_str(&self.flush_table());
                    out.push_str(&self.render_prose_line(&line));
                }
            } else if let Some(candidate) = self.table_candidate.take() {
                if is_table_separator(&line) {
                    self.table = candidate;
                    self.table.push_str(&line);
                } else {
                    out.push_str(&format_markdown(&candidate));
                    out.push_str(&self.render_prose_line(&line));
                }
            } else {
                out.push_str(&self.render_prose_line(&line));
            }
        }
        out
    }

    fn render_prose_line(&mut self, line: &str) -> String {
        if fence_parts(line).is_some_and(|(_, length, _)| length >= 3) {
            return line.to_string();
        }
        if is_table_row(line) {
            self.table_candidate = Some(line.to_string());
            return String::new();
        }
        format_markdown(line)
    }

    fn flush_table(&mut self) -> String {
        let mut out = format_markdown(&std::mem::take(&mut self.table));
        if let Some(candidate) = self.table_candidate.take() {
            out.push_str(&format_markdown(&candidate));
        }
        out
    }

    /// Flush incomplete blocks on message completion, cancellation, or error.
    /// Reset at every message boundary so a fence cannot swallow the next turn.
    pub fn finish(&mut self) -> String {
        let mut state = std::mem::take(self);
        let mut out = state.flush_table();
        if state.fence.is_some() || opening_fence(&state.pending_line).is_some() {
            state.block.push_str(&state.pending_line);
            out.push_str(&format_markdown(&state.block));
        } else if !state.pending_line.is_empty() {
            let mut rendered = format_markdown(&state.pending_line);
            if !state.pending_line.ends_with('\n') && rendered.ends_with('\n') {
                rendered.pop();
            }
            out.push_str(&rendered);
        }
        out
    }
}

fn is_table_row(line: &str) -> bool {
    line.contains('|')
}

fn is_table_separator(line: &str) -> bool {
    if !line.contains('|') {
        return false;
    }
    let trimmed = line.trim().trim_matches('|');
    !trimmed.is_empty()
        && trimmed.split('|').all(|cell| {
            let cell = cell.trim().trim_matches(':');
            cell.len() >= 3 && cell.bytes().all(|byte| byte == b'-')
        })
}

fn fence_parts(line: &str) -> Option<(char, usize, &str)> {
    let trimmed = line.trim_start_matches(' ');
    if line.len() - trimmed.len() > 3 {
        return None;
    }
    let marker = trimmed.chars().next()?;
    if !matches!(marker, '`' | '~') {
        return None;
    }
    let length = trimmed.chars().take_while(|&ch| ch == marker).count();
    Some((marker, length, &trimmed[length..]))
}

fn opening_fence(line: &str) -> Option<(char, usize)> {
    fence_parts(line)
        .filter(|(marker, length, rest)| *length >= 3 && (*marker != '`' || !rest.contains('`')))
        .map(|(marker, length, _)| (marker, length))
}

fn closes_fence(line: &str, marker: char, length: usize) -> bool {
    fence_parts(line).is_some_and(|(closing_marker, closing_length, rest)| {
        closing_marker == marker
            && closing_length >= length
            && rest.trim_matches([' ', '\t', '\n']).is_empty()
    })
}
