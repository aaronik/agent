use std::io;
use std::path::PathBuf;
use std::process::{ExitStatus, Stdio};
use std::time::Duration;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use tokio::fs::File;
use tokio::io::{AsyncReadExt, AsyncSeekExt, AsyncWriteExt};
use tokio::process::{Child, Command};
use tokio::time;

use crate::agent::{CancellationToken, ToolStatus};
use crate::tools::output::{MAX_TOOL_RESPONSE_LENGTH, OutputPreview, new_output_file};

// Leave room for the monitor event envelope added by the caller.
const MONITOR_OUTPUT_LIMIT: usize = MAX_TOOL_RESPONSE_LENGTH - 1024;

#[derive(Clone, Debug, Deserialize, JsonSchema, Serialize)]
pub struct RunShellCommandArgs {
    pub cmd: String,
    #[serde(default = "default_timeout")]
    pub timeout: u64,
}

pub async fn run_shell_command(args: RunShellCommandArgs) -> Result<String, String> {
    run_shell_command_cancellable(args, &CancellationToken::new(), false).await
}

pub async fn run_shell_command_cancellable(
    args: RunShellCommandArgs,
    cancellation_token: &CancellationToken,
    allow_git_writes: bool,
) -> Result<String, String> {
    if !allow_git_writes && let Some(blocked) = blocked_git_write_operation(&args.cmd) {
        return Err(format!(
            "blocked git write operation: `{blocked}`. Read-only git commands such as log, reflog, status, diff, show, and branch --list are allowed."
        ));
    }
    match run_command_output(
        &args.cmd,
        Duration::from_secs(args.timeout),
        cancellation_token,
        CaptureMode::Shell,
    )
    .await?
    {
        CommandOutput::Completed(output) => Ok(output),
        CommandOutput::CheckFailed => unreachable!("shell commands keep unsuccessful output"),
        CommandOutput::TimedOut => Ok(format!(
            "(exit code: 124)\ncommand timed out after {}s",
            args.timeout
        )),
        CommandOutput::Cancelled => Err("tool call cancelled".to_string()),
    }
}

#[derive(Clone, Copy)]
enum CaptureMode {
    Shell,
    Monitor,
}

async fn run_command_output(
    command_text: &str,
    timeout: Duration,
    cancellation_token: &CancellationToken,
    mode: CaptureMode,
) -> Result<CommandOutput, String> {
    let deadline = time::Instant::now() + timeout;
    let shell = std::env::var("SHELL").unwrap_or_else(|_| "/bin/sh".to_string());
    let mut command = Command::new(shell);
    command
        .arg("-c")
        .arg(command_text)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);
    #[cfg(unix)]
    command.process_group(0);
    let mut child = command
        .spawn()
        .map_err(|err| format!("failed to run command: {err}"))?;
    let mut group = ProcessGroup::new(&child);
    let stdout = child.stdout.take().expect("piped stdout");
    let stderr = child.stderr.take().expect("piped stderr");
    let spill = matches!(mode, CaptureMode::Shell);

    let outcome = tokio::select! {
        // Keep the readers in this future rather than detaching them. On
        // timeout or abort they drop their pipes and any incomplete spills.
        result = async {
            let (status, stdout, stderr) = tokio::try_join!(
                child.wait(), read_pipe(stdout, spill), read_pipe(stderr, spill),
            ).map_err(|err| format!("failed to capture command output: {err}"))?;
            group.disarm();
            if matches!(mode, CaptureMode::Monitor) && !status.success() {
                return Ok(CommandOutput::CheckFailed);
            }
            Ok(CommandOutput::Completed(format_completed_output(status, stdout, stderr, mode).await))
        } => return result,
        _ = time::sleep_until(deadline) => CommandOutput::TimedOut,
        _ = cancellation_token.cancelled() => CommandOutput::Cancelled,
    };
    // child.id() becomes None after wait(), even while descendants still
    // have the pipes open. The guard retains the original group ID.
    drop(group);
    let _ = child.start_kill();
    let _ = time::timeout(Duration::from_secs(1), child.wait()).await;
    Ok(outcome)
}

/// Return Some(output) only when the check exits successfully.
pub(crate) async fn check_shell_command(
    command: &str,
    timeout: Duration,
    allow_git_writes: bool,
) -> Result<Option<String>, String> {
    if !allow_git_writes && let Some(blocked) = blocked_git_write_operation(command) {
        return Err(format!("blocked git write operation: `{blocked}`"));
    }
    match run_command_output(
        command,
        timeout,
        &CancellationToken::new(),
        CaptureMode::Monitor,
    )
    .await?
    {
        CommandOutput::Completed(output) => Ok(Some(output)),
        _ => Ok(None),
    }
}

enum CommandOutput {
    Completed(String),
    CheckFailed,
    TimedOut,
    Cancelled,
}

struct ProcessGroup {
    #[cfg(unix)]
    id: Option<i32>,
}

impl ProcessGroup {
    fn new(_child: &Child) -> Self {
        Self {
            #[cfg(unix)]
            id: _child.id().and_then(|pid| i32::try_from(pid).ok()),
        }
    }

    fn disarm(&mut self) {
        #[cfg(unix)]
        {
            self.id = None;
        }
    }
}

impl Drop for ProcessGroup {
    fn drop(&mut self) {
        #[cfg(unix)]
        if let Some(id) = self.id {
            // A negative ID signals every member of this process group.
            unsafe { libc::kill(-id, libc::SIGKILL) };
        }
    }
}

#[derive(Default)]
struct PipeOutput {
    preview: OutputPreview,
    spill: Option<io::Result<SpillFile>>,
}

impl PipeOutput {
    async fn push(&mut self, text: &str, spill: bool) {
        if text.is_empty() {
            return;
        }
        if spill {
            match &mut self.spill {
                Some(Ok(file)) => {
                    if let Err(err) = file.file.write_all(text.as_bytes()).await {
                        self.spill = Some(Err(err));
                    }
                }
                None if self.preview.len() + text.chars().count() > MAX_TOOL_RESPONSE_LENGTH => {
                    self.spill = Some(
                        async {
                            let mut file = SpillFile::new().await?;
                            file.write_preview(&self.preview).await?;
                            file.file.write_all(text.as_bytes()).await?;
                            Ok(file)
                        }
                        .await,
                    );
                }
                _ => {}
            }
        }
        self.preview.push(text);
    }

    async fn write_to(&mut self, file: &mut SpillFile) -> io::Result<()> {
        if let Some(spill) = self.spill.take() {
            let mut spill = spill?;
            spill.file.rewind().await?;
            tokio::io::copy(&mut spill.file, &mut file.file).await?;
            Ok(())
        } else {
            file.write_preview(&self.preview).await
        }
    }
}

struct SpillFile {
    file: File,
    path: tempfile::TempPath,
}

impl SpillFile {
    async fn new() -> io::Result<Self> {
        let (file, path) = tokio::task::spawn_blocking(new_output_file)
            .await??
            .into_parts();
        Ok(Self {
            file: File::from_std(file),
            path,
        })
    }

    async fn write_preview(&mut self, preview: &OutputPreview) -> io::Result<()> {
        let (head, tail) = preview.retained_parts();
        self.file.write_all(head.as_bytes()).await?;
        self.file.write_all(tail.as_bytes()).await
    }

    async fn keep(mut self) -> io::Result<PathBuf> {
        self.file.flush().await?;
        self.path.keep().map_err(|err| err.error)
    }
}

async fn read_pipe(
    mut pipe: impl tokio::io::AsyncRead + Unpin,
    spill: bool,
) -> io::Result<PipeOutput> {
    let mut output = PipeOutput::default();
    let mut buf = [0; 8192 + 3];
    let mut pending = 0;
    loop {
        let read = pipe.read(&mut buf[pending..]).await?;
        if read == 0 {
            if pending > 0 {
                output.push("�", spill).await;
            }
            // Surface buffered write failures even when this file will be
            // rewound and copied into another spill later.
            if let Some(Ok(file)) = &mut output.spill
                && let Err(err) = file.file.flush().await
            {
                output.spill = Some(Err(err));
            }
            return Ok(output);
        }

        let end = pending + read;
        // Match from_utf8_lossy on the whole pipe: defer a trailing partial
        // sequence until the next read, and write at most one chunk per read.
        let bytes = &buf[..end];
        if let Ok(text) = std::str::from_utf8(bytes) {
            pending = 0;
            output.push(text, spill).await;
            continue;
        }
        pending = {
            let invalid = bytes
                .utf8_chunks()
                .last()
                .expect("nonempty chunk")
                .invalid();
            if std::str::from_utf8(invalid).is_err_and(|err| err.error_len().is_none()) {
                invalid.len()
            } else {
                0
            }
        };
        output
            .push(&String::from_utf8_lossy(&buf[..end - pending]), spill)
            .await;
        buf.copy_within(end - pending..end, 0);
    }
}

async fn format_completed_output(
    status: ExitStatus,
    stdout: PipeOutput,
    mut stderr: PipeOutput,
    mode: CaptureMode,
) -> String {
    let limit = match mode {
        CaptureMode::Shell => MAX_TOOL_RESPONSE_LENGTH,
        CaptureMode::Monitor => MONITOR_OUTPUT_LIMIT,
    };
    let PipeOutput { mut preview, spill } = stdout;
    // Keep stdout before stderr, matching the original tool output contract.
    let ends_with_newline = if stderr.preview.len() > 0 {
        stderr.preview.ends_with_newline()
    } else {
        preview.ends_with_newline()
    };
    let suffix = if status.success() {
        String::new()
    } else {
        let separator = if preview.len() + stderr.preview.len() == 0 || ends_with_newline {
            ""
        } else {
            "\n"
        };
        format!("{separator}(exit code: {})", status.code().unwrap_or(1))
    };
    let saved = if matches!(mode, CaptureMode::Shell)
        && preview.len() + stderr.preview.len() + suffix.len() > limit
    {
        // Reuse the stdout spill, if any, and append stderr in stream order.
        Some(
            async {
                let mut file = match spill {
                    Some(spill) => spill?,
                    None => {
                        let mut file = SpillFile::new().await?;
                        file.write_preview(&preview).await?;
                        file
                    }
                };
                stderr.write_to(&mut file).await?;
                file.file.write_all(suffix.as_bytes()).await?;
                file.keep().await
            }
            .await,
        )
    } else {
        None
    };
    preview.append(&stderr.preview);
    preview.push(&suffix);
    // A command exit code is part of the output, not a tool execution error.
    preview.render(limit, &ToolStatus::Success, saved)
}

fn default_timeout() -> u64 {
    30
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum ShellToken {
    Word(String),
    Boundary,
}

fn blocked_git_write_operation(command: &str) -> Option<String> {
    let tokens = shell_tokens(command);
    let mut current_command = Vec::new();

    for token in tokens {
        match token {
            ShellToken::Word(word) => current_command.push(word),
            ShellToken::Boundary => {
                if let Some(blocked) = blocked_git_simple_command(&current_command) {
                    return Some(blocked);
                }
                current_command.clear();
            }
        }
    }

    blocked_git_simple_command(&current_command)
}

fn blocked_git_simple_command(words: &[String]) -> Option<String> {
    let (git_index, args) = git_invocation(words)?;
    let rendered = words[git_index..].join(" ");
    if git_args_are_read_only(args) {
        None
    } else {
        Some(rendered)
    }
}

fn git_invocation(words: &[String]) -> Option<(usize, &[String])> {
    let mut index = 0;
    while index < words.len() && is_assignment(&words[index]) {
        index += 1;
    }

    while index < words.len() {
        match words[index].as_str() {
            "sudo" | "doas" | "command" | "builtin" | "noglob" | "time" => index += 1,
            "env" => {
                index += 1;
                while index < words.len()
                    && (is_assignment(&words[index]) || words[index].starts_with('-'))
                {
                    if words[index] == "-u" || words[index] == "--unset" || words[index] == "-C" {
                        index += 1;
                    }
                    index += 1;
                }
            }
            _ => break,
        }
    }

    if index < words.len() && is_git_executable(&words[index]) {
        Some((index, &words[index + 1..]))
    } else {
        None
    }
}

fn git_args_are_read_only(args: &[String]) -> bool {
    let Some((subcommand, rest)) = git_subcommand(args) else {
        return true;
    };

    match subcommand.as_str() {
        "--help" | "help" | "version" | "--version" => true,
        "status" | "log" | "reflog" | "diff" | "show" | "grep" | "ls-files" | "ls-tree"
        | "rev-parse" | "rev-list" | "remote" | "tag" | "stash" | "blame" | "bisect"
        | "shortlog" | "describe" | "name-rev" | "merge-base" | "cat-file" | "count-objects"
        | "for-each-ref" | "show-ref" | "symbolic-ref" | "worktree" => {
            git_read_command_args_are_safe(&subcommand, rest)
        }
        "branch" => git_branch_args_are_read_only(rest),
        "config" => git_config_args_are_read_only(rest),
        _ => false,
    }
}

fn git_subcommand(args: &[String]) -> Option<(String, &[String])> {
    let mut index = 0;
    while index < args.len() {
        let arg = &args[index];
        if arg == "--" {
            return None;
        }
        if !arg.starts_with('-') {
            return Some((arg.to_string(), &args[index + 1..]));
        }
        index += 1;
        if git_global_option_takes_value(arg) && index < args.len() {
            index += 1;
        }
    }
    None
}

fn git_global_option_takes_value(option: &str) -> bool {
    matches!(
        option,
        "-C" | "-c"
            | "--config-env"
            | "--exec-path"
            | "--git-dir"
            | "--work-tree"
            | "--namespace"
            | "--paginate"
            | "--no-pager"
    )
}

fn git_read_command_args_are_safe(subcommand: &str, args: &[String]) -> bool {
    if subcommand == "stash" {
        return args
            .first()
            .is_some_and(|arg| matches!(arg.as_str(), "list" | "show"));
    }
    if subcommand == "remote" {
        return args.first().is_none_or(|arg| {
            matches!(arg.as_str(), "" | "show" | "get-url" | "-v" | "--verbose")
        });
    }
    if subcommand == "tag" {
        return !args.iter().any(|arg| {
            matches!(
                arg.as_str(),
                "-a" | "--annotate" | "-s" | "--sign" | "-d" | "--delete" | "-f" | "--force"
            )
        });
    }
    if subcommand == "bisect" {
        return args
            .first()
            .is_some_and(|arg| matches!(arg.as_str(), "log" | "view"));
    }
    if subcommand == "symbolic-ref" {
        return args.iter().any(|arg| arg == "--short") || args.len() == 1;
    }
    if subcommand == "worktree" {
        return args
            .first()
            .is_some_and(|arg| matches!(arg.as_str(), "list"));
    }
    true
}

fn git_branch_args_are_read_only(args: &[String]) -> bool {
    args.is_empty()
        || args.iter().all(|arg| {
            arg.starts_with('-')
                && matches!(
                    arg.as_str(),
                    "--list"
                        | "-l"
                        | "--all"
                        | "-a"
                        | "--remotes"
                        | "-r"
                        | "--verbose"
                        | "-v"
                        | "-vv"
                        | "--no-color"
                        | "--color"
                        | "--contains"
                        | "--no-contains"
                        | "--merged"
                        | "--no-merged"
                        | "--sort"
                        | "--format"
                        | "--show-current"
                )
        })
}

fn git_config_args_are_read_only(args: &[String]) -> bool {
    args.iter().any(|arg| {
        matches!(
            arg.as_str(),
            "--get"
                | "--get-all"
                | "--get-regexp"
                | "--list"
                | "-l"
                | "--show-origin"
                | "--show-scope"
        )
    }) && !args.iter().any(|arg| {
        matches!(
            arg.as_str(),
            "--unset"
                | "--unset-all"
                | "--add"
                | "--replace-all"
                | "--rename-section"
                | "--remove-section"
        )
    })
}

fn is_git_executable(word: &str) -> bool {
    std::path::Path::new(word)
        .file_name()
        .and_then(|name| name.to_str())
        == Some("git")
}

fn is_assignment(word: &str) -> bool {
    let Some((name, _)) = word.split_once('=') else {
        return false;
    };
    let mut chars = name.chars();
    chars
        .next()
        .is_some_and(|ch| ch == '_' || ch.is_ascii_alphabetic())
        && chars.all(|ch| ch == '_' || ch.is_ascii_alphanumeric())
}

fn shell_tokens(command: &str) -> Vec<ShellToken> {
    let mut tokens = Vec::new();
    let mut word = String::new();
    let mut chars = command.chars().peekable();
    let mut single_quoted = false;
    let mut double_quoted = false;

    while let Some(ch) = chars.next() {
        if single_quoted {
            if ch == '\'' {
                single_quoted = false;
            } else {
                word.push(ch);
            }
            continue;
        }

        if double_quoted {
            match ch {
                '"' => double_quoted = false,
                '\\' => {
                    if let Some(next) = chars.next() {
                        word.push(next);
                    }
                }
                _ => word.push(ch),
            }
            continue;
        }

        match ch {
            '\'' => single_quoted = true,
            '"' => double_quoted = true,
            '\\' => {
                if let Some(next) = chars.next() {
                    word.push(next);
                }
            }
            '#' if word.is_empty() => {
                for next in chars.by_ref() {
                    if next == '\n' {
                        tokens.push(ShellToken::Boundary);
                        break;
                    }
                }
            }
            ' ' | '\t' | '\r' => push_word(&mut tokens, &mut word),
            '\n' | ';' | '|' | '&' | '(' | ')' => {
                push_word(&mut tokens, &mut word);
                tokens.push(ShellToken::Boundary);
                if (ch == '|' || ch == '&') && chars.peek() == Some(&ch) {
                    chars.next();
                }
            }
            _ => word.push(ch),
        }
    }

    push_word(&mut tokens, &mut word);
    tokens
}

fn push_word(tokens: &mut Vec<ShellToken>, word: &mut String) {
    if !word.is_empty() {
        tokens.push(ShellToken::Word(std::mem::take(word)));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn pipe_capture_decodes_utf8_across_reads() {
        let first: &[u8] = b"head\xff\xe2";
        let second: &[u8] = b"\x82";
        let third: &[u8] = b"\xac\xf0\x9f";
        let output = read_pipe(first.chain(second).chain(third), false)
            .await
            .unwrap();
        assert_eq!(output.preview.retained_parts(), ("head�€�", ""));
    }

    #[tokio::test]
    async fn monitor_check_bounds_output_and_only_returns_success() {
        let output = check_shell_command(
            "printf HEAD; head -c 200000 /dev/zero | tr '\\0' x; printf TAIL >&2",
            Duration::from_secs(5),
            false,
        )
        .await
        .unwrap()
        .unwrap();
        assert!(output.chars().count() <= 32_000);
        assert!(output.starts_with("HEAD"));
        assert!(output.ends_with("TAIL"));
        assert!(output.contains("characters omitted"));
        assert!(!output.contains("Full output saved"));

        assert!(
            check_shell_command(
                "head -c 200000 /dev/zero; exit 7",
                Duration::from_secs(5),
                false,
            )
            .await
            .unwrap()
            .is_none()
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn aborting_monitor_check_kills_running_descendants() {
        let dir = tempfile::tempdir().unwrap();
        let ready = dir.path().join("ready");
        let survived = dir.path().join("survived");
        let command = format!(
            "(touch '{}'; /bin/sleep 0.7 && touch '{}') & wait",
            ready.display(),
            survived.display(),
        );
        let check = tokio::spawn(async move {
            check_shell_command(&command, Duration::from_secs(5), false).await
        });
        time::timeout(Duration::from_secs(2), async {
            while !ready.exists() {
                time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .expect("descendant started");
        check.abort();
        assert!(check.await.unwrap_err().is_cancelled());
        time::sleep(Duration::from_millis(850)).await;
        assert!(
            !survived.exists(),
            "aborted check left a running descendant"
        );
    }
}
