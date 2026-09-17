//! Buffer fenced blocks until they close, then use the replay renderer. Prose
//! still streams immediately; only a possible opening fence needs lookahead.
use super::{format_markdown, sanitize_terminal_text};

#[derive(Debug, Default)]
pub struct AssistantMarkdownStream {
    pending_line: String,
    block: String,
    fence: Option<(char, usize)>,
    prose_line: bool,
}

impl AssistantMarkdownStream {
    pub fn push(&mut self, text: &str) -> String {
        let mut out = String::new();
        for character in sanitize_terminal_text(text).chars() {
            if self.prose_line {
                out.push(character);
                if character == '\n' {
                    self.prose_line = false;
                }
                continue;
            }
            self.pending_line.push(character);
            if let Some((marker, length)) = self.fence {
                if character == '\n' {
                    let line = std::mem::take(&mut self.pending_line);
                    self.block.push_str(&line);
                    if closes_fence(&line, marker, length) {
                        out.push_str(&format_markdown(&std::mem::take(&mut self.block)));
                        self.fence = None;
                    }
                }
            } else if character == '\n' {
                if let Some(fence) = opening_fence(&self.pending_line) {
                    self.fence = Some(fence);
                    self.block = std::mem::take(&mut self.pending_line);
                } else {
                    out.push_str(&std::mem::take(&mut self.pending_line));
                }
            } else if !possible_fence(&self.pending_line) {
                out.push_str(&std::mem::take(&mut self.pending_line));
                self.prose_line = true;
            }
        }
        out
    }

    /// Flush incomplete blocks on message completion, cancellation, or error.
    /// Reset at every message boundary so a fence cannot swallow the next turn.
    pub fn finish(&mut self) -> String {
        let mut state = std::mem::take(self);
        if state.fence.is_some() || opening_fence(&state.pending_line).is_some() {
            state.block.push_str(&state.pending_line);
            format_markdown(&state.block)
        } else {
            state.pending_line
        }
    }
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

fn possible_fence(line: &str) -> bool {
    if line.len() <= 3 && line.chars().all(|ch| ch == ' ') {
        return true;
    }
    fence_parts(line).is_some_and(|(marker, length, rest)| {
        (length >= 3 || rest.is_empty()) && (marker != '`' || !rest.contains('`'))
    })
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
