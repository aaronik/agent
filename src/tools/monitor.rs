//! Session-scoped, one-shot background command checks.
use std::collections::HashMap;
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, Ordering},
};
use std::time::Duration;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;
use uuid::Uuid;

use crate::agent::ToolStatus;
use crate::tools::output::truncate_tool_output;
use crate::tools::shell::check_shell_command;

#[derive(Clone, Debug, Deserialize, Serialize, JsonSchema)]
pub struct MonitorArgs {
    /// "start", "stop", or "list".
    pub action: String,
    /// Shell command checked periodically. Exit status 0 triggers a one-shot wake-up.
    pub cmd: Option<String>,
    /// Poll interval in seconds (1 to 3600; default 5).
    pub interval_seconds: Option<u64>,
    /// Context to give the agent when triggered.
    pub message: Option<String>,
    /// Monitor ID, required for stop.
    pub id: Option<String>,
}

#[derive(Clone, Debug)]
pub struct MonitorEvent {
    pub content: String,
}

#[derive(Clone, Debug)]
pub struct MonitorManager {
    tasks: Arc<Mutex<HashMap<String, MonitorTask>>>,
    sender: mpsc::UnboundedSender<MonitorEvent>,
    wake: Arc<AtomicBool>,
}

#[derive(Debug)]
struct MonitorTask {
    handle: tokio::task::JoinHandle<()>,
    cmd: String,
    message: String,
    interval_seconds: u64,
}

impl MonitorManager {
    /// A snapshot of the checks that are still polling in this process.
    pub fn list(&self) -> String {
        let Ok(tasks) = self.tasks.lock() else {
            return "Could not read active monitors.".to_string();
        };
        if tasks.is_empty() {
            return "No active monitors.".to_string();
        }
        let mut active = tasks.iter().collect::<Vec<_>>();
        active.sort_by_key(|(id, _)| *id);
        let mut lines = vec![format!("Active monitors ({}):", active.len())];
        for (id, task) in active {
            lines.push(format!(
                "  {id}\n    Command: {}\n    Every: {}s\n    Message: {}",
                task.cmd, task.interval_seconds, task.message
            ));
        }
        lines.join("\n")
    }

    pub fn new(wake: Arc<AtomicBool>) -> (Self, mpsc::UnboundedReceiver<MonitorEvent>) {
        let (sender, receiver) = mpsc::unbounded_channel();
        (
            Self {
                tasks: Arc::new(Mutex::new(HashMap::new())),
                sender,
                wake,
            },
            receiver,
        )
    }

    pub fn execute(&self, args: MonitorArgs, allow_git_writes: bool) -> Result<String, String> {
        let mut tasks = self.tasks.lock().map_err(|e| e.to_string())?;
        match args.action.as_str() {
            "start" => {
                let cmd = args
                    .cmd
                    .filter(|cmd| !cmd.trim().is_empty())
                    .ok_or("start requires a nonempty cmd")?;
                let interval = args.interval_seconds.unwrap_or(5);
                if !(1..=3600).contains(&interval) {
                    return Err("interval_seconds must be between 1 and 3600".into());
                }
                let id = Uuid::new_v4().to_string();
                let label = args
                    .message
                    .unwrap_or_else(|| format!("Condition met: {cmd}"));
                let manager = self.clone();
                let task_id = id.clone();
                let task_cmd = cmd.clone();
                let task_label = label.clone();
                let handle = tokio::spawn(async move {
                    loop {
                        match check_shell_command(&cmd, Duration::from_secs(30), allow_git_writes)
                            .await
                        {
                            Ok(Some(output)) => {
                                let content = truncate_tool_output(
                                    format!(
                                        "[Monitor {task_id} triggered] {label}\nCommand: {cmd}\nOutput: {output}"
                                    ),
                                    &ToolStatus::Success,
                                );
                                if manager.sender.send(MonitorEvent { content }).is_ok() {
                                    manager.wake.store(true, Ordering::SeqCst);
                                }
                                break;
                            }
                            Ok(None) | Err(_) => {
                                tokio::time::sleep(Duration::from_secs(interval)).await
                            }
                        }
                    }
                    if let Ok(mut tasks) = manager.tasks.lock() {
                        tasks.remove(&task_id);
                    }
                });
                tasks.insert(
                    id.clone(),
                    MonitorTask {
                        handle,
                        cmd: task_cmd,
                        message: task_label,
                        interval_seconds: interval,
                    },
                );
                Ok(format!("Monitor started: {id}"))
            }
            "stop" => {
                let id = args.id.ok_or("stop requires id")?;
                let task = tasks.remove(&id).ok_or("unknown monitor id")?;
                task.handle.abort();
                Ok(format!("Monitor stopped: {id}"))
            }
            "list" => {
                drop(tasks);
                Ok(self.list())
            }
            _ => Err("action must be start, stop, or list".into()),
        }
    }
}

pub fn stop_all(manager: &MonitorManager) {
    if let Ok(mut tasks) = manager.tasks.lock() {
        for (_, task) in tasks.drain() {
            task.handle.abort();
        }
    }
}
