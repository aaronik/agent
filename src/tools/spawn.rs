use std::path::PathBuf;
use std::process::Stdio;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use tokio::process::Command;

use crate::providers::effective_model_name;

#[derive(Clone, Debug, Deserialize, JsonSchema, Serialize)]
pub struct SpawnArgs {
    pub task: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub conversation_id: Option<String>,
}

pub async fn spawn(args: SpawnArgs) -> Result<String, String> {
    let raw_model =
        std::env::var("AGENT_SPAWN_MODEL").unwrap_or_else(|_| effective_model_name(None));

    let mut command = Command::new(agent_executable()?);
    command
        .arg("--model")
        .arg(raw_model)
        .arg("--single")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    if let Some(conversation_id) = args.conversation_id.as_deref() {
        command.arg("--resume").arg(conversation_id);
    }

    let output = command
        .arg(args.task)
        .output()
        .await
        .map_err(|err| format!("Error spawning agent: {err}"))?;

    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();

    if !output.status.success() {
        let mut error = String::new();
        if !stdout.trim().is_empty() {
            error.push_str(stdout.trim_end());
            error.push('\n');
        }
        if !stderr.trim().is_empty() {
            error.push_str(stderr.trim_end());
            error.push('\n');
        }
        error.push_str(&format!(
            "Error spawning agent: exited with status {}",
            output.status
        ));
        return Err(error);
    }

    Ok(format_spawn_output(&stdout))
}

fn agent_executable() -> Result<PathBuf, String> {
    if let Ok(path) = std::env::var("AGENT_SPAWN_BIN") {
        return Ok(PathBuf::from(path));
    }
    if let Ok(path) = std::env::var("CARGO_BIN_EXE_agent") {
        return Ok(PathBuf::from(path));
    }
    std::env::current_exe().map_err(|err| format!("failed to locate agent executable: {err}"))
}

fn format_spawn_output(agent_stdout: &str) -> String {
    let conversation_id = session_id_from_agent_output(agent_stdout);

    let mut output = format!("[SPAWNED AGENT OUTPUT]\n{}", agent_stdout.trim_end());
    if let Some(conversation_id) = conversation_id {
        output.push_str(&format!("\n[CONVERSATION ID]\n{conversation_id}"));
    }
    output
}

fn session_id_from_agent_output(agent_stdout: &str) -> Option<String> {
    const PREFIX: &str = "sessionid:";

    agent_stdout.lines().rev().find_map(|line| {
        let line = line.trim();
        let prefix = line.get(..PREFIX.len())?;
        if !prefix.eq_ignore_ascii_case(PREFIX) {
            return None;
        }

        let id = line[PREFIX.len()..].trim();
        (!id.is_empty()).then(|| id.to_string())
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn format_spawn_output_reads_session_id_from_agent_output_bottom() {
        let output = format_spawn_output("task complete\nsessionId:  abc-123  \n");

        assert_eq!(
            output,
            "[SPAWNED AGENT OUTPUT]\ntask complete\nsessionId:  abc-123\n[CONVERSATION ID]\nabc-123"
        );
    }

    #[test]
    fn session_id_parser_uses_last_session_id_line() {
        assert_eq!(
            session_id_from_agent_output("sessionId: old\nwork\nSESSIONID: wanted"),
            Some("wanted".to_string())
        );
    }

    #[test]
    fn session_id_parser_only_accepts_line_prefix() {
        assert_eq!(
            session_id_from_agent_output("before sessionId: nope\nsession identifier: nope"),
            None
        );
    }
}
