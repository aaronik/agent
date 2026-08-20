use crate::agent::ToolStatus;

pub const MAX_TOOL_RESPONSE_LENGTH: usize = 32_000;

pub fn truncate_tool_output(content: String, status: &ToolStatus) -> String {
    if content.chars().count() <= MAX_TOOL_RESPONSE_LENGTH {
        return content;
    }

    let retained = content
        .chars()
        .take(MAX_TOOL_RESPONSE_LENGTH)
        .collect::<String>();
    let completion = match status {
        ToolStatus::Success => "The tool completed successfully.",
        ToolStatus::Error => "The tool finished with an error.",
    };
    format!(
        "{retained}\n\n[Output trimmed by the harness to avoid overwhelming the context.]\n{completion} If you need omitted output, make a more selective tool call."
    )
}
