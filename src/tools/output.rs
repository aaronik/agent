use std::io::{self, Write};
use std::path::PathBuf;

use crate::agent::ToolStatus;

// Budget for retained content; the truncation notice is additional.
pub const MAX_TOOL_RESPONSE_LENGTH: usize = 32_000;

pub fn truncate_tool_output(content: String, status: &ToolStatus) -> String {
    truncate_with_saver(content, status, save_full_output)
}

fn save_full_output(content: &str) -> io::Result<PathBuf> {
    // NamedTempFile uses exclusive creation and owner-only permissions on Unix.
    // Persist only after a successful write so failures clean up partial files.
    let mut file = new_output_file()?;
    file.write_all(content.as_bytes())?;
    let (_, path) = file.keep().map_err(|error| error.error)?;
    Ok(path)
}

pub(super) fn new_output_file() -> io::Result<tempfile::NamedTempFile> {
    tempfile::Builder::new()
        .prefix("agent-tool-output-")
        .suffix(".txt")
        .tempfile()
}

fn truncate_with_saver(
    content: String,
    status: &ToolStatus,
    save: impl FnOnce(&str) -> io::Result<PathBuf>,
) -> String {
    let length = content.chars().count();
    if length <= MAX_TOOL_RESPONSE_LENGTH {
        return content;
    }

    let half = MAX_TOOL_RESPONSE_LENGTH / 2;
    let head_end = content.char_indices().nth(half).unwrap().0;
    let tail_start = content.char_indices().rev().nth(half - 1).unwrap().0;
    let head = &content[..head_end];
    let tail = &content[tail_start..];
    let notice = truncation_notice(
        length - MAX_TOOL_RESPONSE_LENGTH,
        half,
        status,
        Some(&save(&content)),
    );
    format!("{head}{notice}{tail}")
}

fn truncation_notice(
    omitted: usize,
    half: usize,
    status: &ToolStatus,
    saved: Option<&io::Result<PathBuf>>,
) -> String {
    let completion = match status {
        ToolStatus::Success => "The tool completed successfully.",
        ToolStatus::Error => "The tool finished with an error.",
    };
    let guidance = match saved {
        Some(Ok(path)) => format!(
            "Full output saved to temporary file: {}\nTo inspect omitted content, make a more selective tool call against this file (e.g. grep or a line range), rather than reading the entire file.\nDelete this temporary file when finished with it.",
            serde_json::to_string(&path.to_string_lossy()).expect("serialize path")
        ),
        Some(Err(error)) => format!(
            "Could not save full output to a temporary file: {error}\nOmitted content was not saved. If needed, make a more selective tool call."
        ),
        None => "Omitted content was not saved.".to_string(),
    };
    format!(
        "\n\n[Output trimmed by the harness to avoid overwhelming the context.]\n[{omitted} characters omitted; showing the first and last {half} characters.]\n{completion}\n{guidance}\n\n[End of output follows]\n"
    )
}

// Keep the first half of the response budget and up to a half-budget suffix,
// without overlap. Length tracks the original text, not the retained text.
#[derive(Default)]
pub(super) struct OutputPreview {
    head: String,
    tail: String,
    tail_len: usize,
    len: usize,
}

impl OutputPreview {
    pub(super) fn len(&self) -> usize {
        self.len
    }

    pub(super) fn ends_with_newline(&self) -> bool {
        if self.tail.is_empty() {
            self.head.ends_with('\n')
        } else {
            self.tail.ends_with('\n')
        }
    }

    pub(super) fn push(&mut self, text: &str) {
        let half = MAX_TOOL_RESPONSE_LENGTH / 2;
        let head_room = half.saturating_sub(self.len);
        let boundary = char_boundary(text, head_room);
        let (head, tail) = text.split_at(boundary);
        let tail_len = tail.chars().count();
        self.head.push_str(head);
        self.len += head.chars().count() + tail_len;
        let drop = (self.tail_len + tail_len).saturating_sub(half);
        if drop <= self.tail_len {
            self.tail.drain(..char_boundary(&self.tail, drop));
            self.tail.push_str(tail);
        } else {
            self.tail.clear();
            self.tail
                .push_str(&tail[char_boundary(tail, drop - self.tail_len)..]);
        }
        self.tail_len += tail_len;
        self.tail_len -= drop;
    }

    pub(super) fn append(&mut self, other: &Self) {
        self.push(&other.head);
        let omitted = other.len.saturating_sub(MAX_TOOL_RESPONSE_LENGTH);
        if omitted > 0 {
            // The skipped middle is outside our head, and other's tail will
            // replace our tail completely.
            self.len += omitted;
            self.tail.clear();
            self.tail_len = 0;
        }
        self.push(&other.tail);
    }

    pub(super) fn retained_parts(&self) -> (&str, &str) {
        (&self.head, &self.tail)
    }

    pub(super) fn render(
        self,
        limit: usize,
        status: &ToolStatus,
        saved: Option<io::Result<PathBuf>>,
    ) -> String {
        if self.len <= limit {
            return self.head + &self.tail;
        }
        // Reserve space for the notice inside the limit: shell and monitor
        // results also pass through the outer tool-output truncator.
        let notice_len = truncation_notice(self.len, limit / 2, status, saved.as_ref())
            .chars()
            .count();
        let half = limit.saturating_sub(notice_len) / 2;
        let head = &self.head[..char_boundary(&self.head, half)];
        let tail_start = if half == 0 {
            self.tail.len()
        } else {
            self.tail
                .char_indices()
                .rev()
                .nth(half - 1)
                .map_or(0, |(i, _)| i)
        };
        let tail = &self.tail[tail_start..];
        let notice = truncation_notice(self.len - 2 * half, half, status, saved.as_ref());
        format!("{head}{notice}{tail}")
    }
}

fn char_boundary(text: &str, chars: usize) -> usize {
    text.char_indices()
        .nth(chars)
        .map_or(text.len(), |(i, _)| i)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn save_failure_still_preserves_head_tail_and_completion_status() {
        let content = format!("{}TAIL", "x".repeat(MAX_TOOL_RESPONSE_LENGTH));
        let output = truncate_with_saver(content, &ToolStatus::Error, |_| {
            Err(io::Error::other("disk full"))
        });
        assert!(output.starts_with(&"x".repeat(16_000)));
        assert!(output.ends_with("TAIL"));
        assert!(output.contains("The tool finished with an error."));
        assert!(output.contains("Could not save full output to a temporary file: disk full"));
        assert!(output.contains("Omitted content was not saved."));
        assert!(!output.contains("Full output saved to temporary file:"));
    }
}
