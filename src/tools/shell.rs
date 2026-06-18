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
    if let Some(blocked) = blocked_git_write_operation(&args.cmd) {
        return Err(format!(
            "blocked git write operation: `{blocked}`. Read-only git commands such as log, reflog, status, diff, show, and branch --list are allowed."
        ));
    }

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
