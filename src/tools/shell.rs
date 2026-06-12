use std::process::Stdio;
use std::time::Duration;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use tokio::process::Command;
use tokio::time;

#[derive(Clone, Debug, Deserialize, JsonSchema, Serialize)]
pub struct RunShellCommandArgs {
    pub cmd: String,
    #[serde(default = "default_timeout")]
    pub timeout: u64,
}

pub async fn run_shell_command(args: RunShellCommandArgs) -> Result<String, String> {
    let shell = std::env::var("SHELL").unwrap_or_else(|_| "/bin/sh".to_string());
    let mut command = Command::new(shell);
    command
        .arg("-c")
        .arg(&args.cmd)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);

    let output = match time::timeout(Duration::from_secs(args.timeout), command.output()).await {
        Ok(result) => result.map_err(|err| format!("failed to run command: {err}"))?,
        Err(_) => {
            return Ok(format!(
                "(exit code: 124)\ncommand timed out after {}s",
                args.timeout
            ));
        }
    };

    let mut combined = String::new();
    combined.push_str(&String::from_utf8_lossy(&output.stdout));
    combined.push_str(&String::from_utf8_lossy(&output.stderr));

    if !output.status.success() {
        if !combined.is_empty() && !combined.ends_with('\n') {
            combined.push('\n');
        }
        let code = output.status.code().unwrap_or(1);
        combined.push_str(&format!("(exit code: {code})"));
    }

    Ok(combined)
}

fn default_timeout() -> u64 {
    30
}
