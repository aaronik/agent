use std::path::PathBuf;
use std::process::Stdio;
use std::time::Duration;

use futures_util::future::join_all;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use tokio::io::{AsyncBufReadExt, AsyncReadExt, BufReader};
use tokio::process::Command;
use tokio::time;

use crate::agent::{AgentMessage, CancellationToken, SubagentUsage};
use crate::providers::{effective_model_name, list_models, parse_model_id};
use crate::session::SessionStore;

pub(crate) const ACTIVITY_ENV: &str = "AGENT_CHILD_ACTIVITY";
pub(crate) const ACTIVITY_PREFIX: &str = "@agent-activity ";
pub(crate) const USAGE_PREFIX: &str = "@agent-usage ";

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SpawnState {
    Started,
    Status(String),
    Usage(SubagentUsage),
    Done,
    Failed,
    Cancelled,
}

#[derive(Clone, Debug)]
pub struct SpawnActivity {
    pub group: String,
    pub index: usize,
    pub total: usize,
    pub task: String,
    pub state: SpawnState,
}

pub(super) type Observer = dyn Fn(&SpawnActivity) + Send + Sync;

#[derive(Clone, Debug, Deserialize, JsonSchema, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SpawnArgs {
    /// One or more subagents, each with its own task and optional model/session.
    pub agents: Vec<SpawnAgent>,
}

#[derive(Clone, Debug, Deserialize, JsonSchema, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SpawnAgent {
    pub task: String,
    /// Exact model ID from /models; omitted to use the configured spawn model.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
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
    let output = spawn_observed(args, cancellation_token, "", false, None).await?;
    output.content.map(|content| (content, output.usages))
}

pub(super) struct SpawnOutput {
    pub content: Result<String, String>,
    pub usages: Vec<SubagentUsage>,
}

pub(super) async fn spawn_observed(
    args: SpawnArgs,
    cancellation_token: &CancellationToken,
    group: &str,
    fast: bool,
    observer: Option<&Observer>,
) -> Result<SpawnOutput, String> {
    let agents = args.agents;
    if agents.is_empty() {
        return Err("agents must contain at least one agent".to_string());
    }
    if agents.iter().any(|agent| agent.task.trim().is_empty()) {
        return Err("each agent must have a non-empty task".to_string());
    }
    if agents.iter().any(|agent| agent.model.is_some()) {
        let available = tokio::select! {
            _ = cancellation_token.cancelled() => return Err("tool call cancelled".to_string()),
            models = list_models() => models,
        };
        for model in agents.iter().filter_map(|agent| agent.model.as_ref()) {
            if !available.contains(model) {
                return Err(format!(
                    "Invalid or unavailable model {model:?}: not present in /models. Use an exact model ID from /models."
                ));
            }
        }
    }

    let total = agents.len();
    let results = join_all(
        agents
            .into_iter()
            .enumerate()
            .map(|(index, agent)| async move {
                let task: String = agent.task.chars().take(160).collect();
                let report = |state| {
                    if let Some(observer) = observer {
                        observer(&SpawnActivity {
                            group: group.to_owned(),
                            index,
                            total,
                            task: task.clone(),
                            state,
                        });
                    }
                };
                let result =
                    spawn_one(agent, cancellation_token, fast, observer.is_some(), &report).await;
                report(match &result {
                    Ok(output) if output.content.is_ok() => SpawnState::Done,
                    _ if cancellation_token.is_cancelled() => SpawnState::Cancelled,
                    _ => SpawnState::Failed,
                });
                result
            }),
    )
    .await;
    let mut outputs = Vec::with_capacity(results.len());
    let mut usages = Vec::new();
    let mut first_error = None;
    for result in results {
        let content = result.and_then(|output| {
            usages.extend(output.usages);
            output.content
        });
        match content {
            Ok(content) => outputs.push(content),
            Err(error) => {
                first_error.get_or_insert(error);
            }
        }
    }
    Ok(SpawnOutput {
        content: first_error.map_or_else(|| Ok(outputs.join("\n\n")), Err),
        usages,
    })
}

async fn spawn_one(
    args: SpawnAgent,
    cancellation_token: &CancellationToken,
    fast: bool,
    observed: bool,
    report: &(dyn Fn(SpawnState) + Sync),
) -> Result<SpawnOutput, String> {
    let raw_model = args.model.clone().unwrap_or_else(|| {
        std::env::var("AGENT_SPAWN_MODEL").unwrap_or_else(|_| effective_model_name(None))
    });

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
        .arg("--no-subagent")
        .arg("--no-completion-sound")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);
    if observed {
        command.env(ACTIVITY_ENV, "1");
    } else {
        command.env_remove(ACTIVITY_ENV);
    }
    set_process_group(&mut command);

    if fast && parse_model_id(&raw_model).provider == "openai" {
        command.arg("--fast");
    }
    if let Some(conversation_id) = args.conversation_id.as_deref() {
        command.arg("--resume").arg(conversation_id);
    }

    let mut child = command
        .arg("--single")
        .arg(args.task)
        .spawn()
        .map_err(|err| format!("Error spawning agent: {err}"))?;

    // Dropping the outer turn (e.g. Escape) must also stop descendants and
    // both pipe readers; none of these futures should outlive the batch.
    let mut group = ProcessGroup(child.id());
    report(SpawnState::Started);
    let stdout = child.stdout.take();
    let stderr = child.stderr.take().expect("piped stderr");
    let output = tokio::select! {
        result = async {
            let wait = async {
                let status = child.wait().await.map_err(|err| format!("Error spawning agent: {err}"))?;
                // Descendants can retain pipes after the agent exits.
                group.stop();
                Ok(status)
            };
            tokio::try_join!(wait, read_pipe(stdout), read_stderr(stderr, observed, report))
        } => {
            let (status, stdout, stderr) = result?;
            std::process::Output {status, stdout, stderr}
        }
        _ = cancellation_token.cancelled() => {
            group.stop();
            let _ = child.start_kill();
            let _ = time::timeout(Duration::from_secs(1), child.wait()).await;
            return Err("tool call cancelled".to_string());
        }
    };

    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();

    // Saved assistant messages are the final accounting source. Exclude usage
    // from before this invocation, including when the worker exits with an error.
    let usages = session_id_from_agent_output(&stdout)
        .and_then(|session_id| SessionStore::new().ok()?.load(Some(&session_id)).ok())
        .map(|session| subagent_usages(&session.messages, &raw_model))
        .map(|usages| usages.into_iter().skip(existing_usage_count).collect())
        .unwrap_or_default();

    let content = if output.status.success() {
        Ok(format_spawn_output(&stdout))
    } else {
        let mut error = String::new();
        for text in [&stdout, &stderr] {
            if !text.trim().is_empty() {
                error.push_str(text.trim_end());
                error.push('\n');
            }
        }
        error.push_str(&format!(
            "Error spawning agent: exited with status {}",
            output.status
        ));
        Err(error)
    };
    Ok(SpawnOutput { content, usages })
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

async fn read_stderr(
    pipe: tokio::process::ChildStderr,
    observed: bool,
    report: &(dyn Fn(SpawnState) + Sync),
) -> Result<Vec<u8>, String> {
    if !observed {
        return read_pipe(Some(pipe)).await;
    }
    let mut reader = BufReader::new(pipe);
    let mut diagnostics = Vec::new();
    loop {
        let start = diagnostics.len();
        if reader
            .read_until(b'\n', &mut diagnostics)
            .await
            .map_err(|e| format!("failed to read spawned agent stderr: {e}"))?
            == 0
        {
            break;
        }
        if let Some(record) = diagnostics[start..].strip_prefix(ACTIVITY_PREFIX.as_bytes())
            && record.len() < 2048
            && let Ok(message) = serde_json::from_slice::<String>(record)
        {
            diagnostics.truncate(start);
            report(SpawnState::Status(message));
        } else if let Some(record) = diagnostics[start..].strip_prefix(USAGE_PREFIX.as_bytes())
            && let Ok(usage) = serde_json::from_slice::<SubagentUsage>(record)
        {
            diagnostics.truncate(start);
            report(SpawnState::Usage(usage));
        }
    }
    Ok(diagnostics)
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

struct ProcessGroup(Option<u32>);

impl ProcessGroup {
    fn stop(&mut self) {
        terminate_process_group(self.0.take());
    }
}

impl Drop for ProcessGroup {
    fn drop(&mut self) {
        self.stop();
    }
}

#[cfg(unix)]
fn set_process_group(command: &mut Command) {
    command.process_group(0);
}

#[cfg(not(unix))]
fn set_process_group(_command: &mut Command) {}

fn terminate_process_group(pid: Option<u32>) {
    #[cfg(unix)]
    kill_process_group(pid);
}

#[cfg(unix)]
fn kill_process_group(pid: Option<u32>) {
    let Some(pid) = pid else {
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
    let agent_stdout = strip_terminal_escape_sequences(agent_stdout);

    let mut output = format!("[SPAWNED AGENT OUTPUT]\n{}", agent_stdout.trim_end());
    if let Some(conversation_id) = conversation_id {
        output.push_str(&format!("\n[CONVERSATION ID]\n{conversation_id}"));
    }
    output
}

fn strip_terminal_escape_sequences(text: &str) -> String {
    let mut output = String::with_capacity(text.len());
    let mut characters = text.chars();

    while let Some(character) = characters.next() {
        if character != '\x1b' {
            output.push(character);
            continue;
        }

        match characters.next() {
            Some('[') => {
                // CSI: consume through its final byte (U+0040..=U+007E).
                for character in characters.by_ref() {
                    if ('@'..='~').contains(&character) {
                        break;
                    }
                }
            }
            Some(']') | Some('P') | Some('^') | Some('_') => {
                // OSC/DCS/PM/APC: terminated by BEL or ST (ESC \\).
                let mut escaped = false;
                for character in characters.by_ref() {
                    if character == '\x07' || (escaped && character == '\\') {
                        break;
                    }
                    escaped = character == '\x1b';
                }
            }
            Some(_) | None => {}
        }
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
    fn format_spawn_output_strips_nested_terminal_escape_sequences() {
        let output = format_spawn_output("\x1b[90m╭─ \x1b[36mtool\x1b[0m\nsessionId: abc-123\n");

        assert_eq!(
            output,
            "[SPAWNED AGENT OUTPUT]\n╭─ tool\nsessionId: abc-123\n[CONVERSATION ID]\nabc-123"
        );
        assert!(!output.contains('\x1b'));
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
