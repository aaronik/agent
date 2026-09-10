use std::path::PathBuf;
use std::process::Stdio;
use std::time::Duration;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use tokio::io::AsyncReadExt;
use tokio::process::{Child, Command};
use tokio::time;

use crate::agent::{AgentMessage, CancellationToken, SubagentUsage};
use crate::providers::effective_model_name;
use crate::session::SessionStore;

#[derive(Clone, Debug, Deserialize, JsonSchema, Serialize)]
pub struct SpawnArgs {
    pub task: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub conversation_id: Option<String>,
}

pub async fn spawn(args: SpawnArgs) -> Result<String, String> {
    spawn_cancellable(args, &CancellationToken::new()).await
}

pub async fn spawn_cancellable(
    args: SpawnArgs,
    cancellation_token: &CancellationToken,
) -> Result<String, String> {
    spawn_cancellable_with_usage(args, cancellation_token)
        .await
        .map(|(output, _)| output)
}

pub async fn spawn_cancellable_with_usage(
    args: SpawnArgs,
    cancellation_token: &CancellationToken,
) -> Result<(String, Vec<SubagentUsage>), String> {
    let raw_model =
        std::env::var("AGENT_SPAWN_MODEL").unwrap_or_else(|_| effective_model_name(None));

    let existing_usage_count = args
        .conversation_id
        .as_deref()
        .and_then(|session_id| SessionStore::new().ok()?.load(Some(session_id)).ok())
        .map(|session| subagent_usages(&session.messages, &raw_model).len())
        .unwrap_or(0);

    let mut command = Command::new(agent_executable()?);
    command
        .arg("--model")
        .arg(&raw_model)
        .arg("--single")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);
    set_process_group(&mut command);

    if let Some(conversation_id) = args.conversation_id.as_deref() {
        command.arg("--resume").arg(conversation_id);
    }

    let mut child = command
        .arg(args.task)
        .spawn()
        .map_err(|err| format!("Error spawning agent: {err}"))?;

    let stdout = child.stdout.take();
    let stderr = child.stderr.take();
    let stdout_task = tokio::spawn(read_pipe(stdout));
    let stderr_task = tokio::spawn(read_pipe(stderr));

    let output = tokio::select! {
        status = child.wait() => {
            let status = status.map_err(|err| format!("Error spawning agent: {err}"))?;
            let stdout = join_pipe_task(stdout_task, "stdout").await?;
            let stderr = join_pipe_task(stderr_task, "stderr").await?;
            std::process::Output { status, stdout, stderr }
        }
        _ = cancellation_token.cancelled() => {
            terminate_child(&mut child).await;
            return Err("tool call cancelled".to_string());
        }
    };

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

    let formatted_output = format_spawn_output(&stdout);
    let usages = session_id_from_agent_output(&stdout)
        .and_then(|session_id| SessionStore::new().ok()?.load(Some(&session_id)).ok())
        .map(|session| subagent_usages(&session.messages, &raw_model))
        .map(|usages| usages.into_iter().skip(existing_usage_count).collect())
        .unwrap_or_default();

    Ok((formatted_output, usages))
}

fn subagent_usages(messages: &[AgentMessage], fallback_model: &str) -> Vec<SubagentUsage> {
    messages
        .iter()
        .filter_map(|message| match message {
            AgentMessage::Assistant(assistant) => {
                assistant.usage.as_ref().map(|usage| SubagentUsage {
                    usage: usage.clone(),
                    model: assistant
                        .model
                        .clone()
                        .unwrap_or_else(|| fallback_model.to_string()),
                })
            }
            _ => None,
        })
        .collect()
}

async fn join_pipe_task(
    task: tokio::task::JoinHandle<Result<Vec<u8>, String>>,
    name: &str,
) -> Result<Vec<u8>, String> {
    task.await
        .map_err(|err| format!("failed to join {name} reader: {err}"))?
}

async fn read_pipe<T>(pipe: Option<T>) -> Result<Vec<u8>, String>
where
    T: tokio::io::AsyncRead + Unpin,
{
    let mut output = Vec::new();
    if let Some(mut pipe) = pipe {
        pipe.read_to_end(&mut output)
            .await
            .map_err(|err| format!("failed to read spawned agent output: {err}"))?;
    }
    Ok(output)
}

async fn terminate_child(child: &mut Child) {
    if child.try_wait().ok().flatten().is_some() {
        return;
    }
    #[cfg(unix)]
    kill_process_group(child);
    let _ = child.start_kill();
    let _ = time::timeout(Duration::from_secs(1), child.wait()).await;
}

#[cfg(unix)]
fn set_process_group(command: &mut Command) {
    command.process_group(0);
}

#[cfg(not(unix))]
fn set_process_group(_command: &mut Command) {}

#[cfg(unix)]
fn kill_process_group(child: &Child) {
    let Some(pid) = child.id() else {
        return;
    };
    let Ok(pid) = i32::try_from(pid) else {
        return;
    };
    // SAFETY: kill is called with a negative pid to signal the child's process group.
    let _ = unsafe { libc::kill(-pid, libc::SIGKILL) };
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
