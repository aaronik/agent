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
    let mut file = tempfile::Builder::new()
        .prefix("agent-tool-output-")
        .suffix(".txt")
        .tempfile()?;
    file.write_all(content.as_bytes())?;
    let (_, path) = file.keep().map_err(|error| error.error)?;
    Ok(path)
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
    let completion = match status {
        ToolStatus::Success => "The tool completed successfully.",
        ToolStatus::Error => "The tool finished with an error.",
    };
    let guidance = match save(&content) {
        Ok(path) => format!(
            "Full output saved to temporary file: {}\nTo inspect omitted content, make a more selective tool call against this file (e.g. grep or a line range), rather than reading the entire file.\nDelete this temporary file when finished with it.",
            serde_json::to_string(&path.to_string_lossy()).expect("serialize path")
        ),
        Err(error) => format!(
            "Could not save full output to a temporary file: {error}\nOmitted content was not saved. If needed, make a more selective tool call."
        ),
    };
    format!(
        "{head}\n\n[Output trimmed by the harness to avoid overwhelming the context.]\n[{} characters omitted; showing the first and last {half} characters.]\n{completion}\n{guidance}\n\n[End of output follows]\n{tail}",
        length - MAX_TOOL_RESPONSE_LENGTH
    )
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
