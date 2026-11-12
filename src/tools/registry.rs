use schemars::{JsonSchema, schema_for};
use serde::Serialize;
use serde_json::{Map, Value, json};
use std::time::Instant;

use crate::agent::{CancellationToken, ImageAttachment, ToolResult, ToolStatus};
use crate::tools::anonymous_browser_session::{
    PlaywrightBrowserSessionArgs, playwright_browser_session,
};
use crate::tools::browser::{BrowserArgs, browser};
use crate::tools::communicate::{CommunicateArgs, communicate};
use crate::tools::fetch::{FetchArgs, fetch};
use crate::tools::files::{
    ReadFileArgs, SearchReplaceArgs, WriteFileArgs, read_file, search_replace, write_file,
};
use crate::tools::image::{GenImageArgs, gen_image};
use crate::tools::monitor::{MonitorArgs, MonitorManager};
use crate::tools::output::truncate_tool_output;
use crate::tools::shell::{RunShellCommandArgs, run_shell_command_cancellable};
use crate::tools::spawn::{SpawnArgs, spawn_cancellable_with_usage};

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    pub parameters: Value,
}

#[derive(Clone, Debug)]
pub struct ToolRegistry {
    definitions: Vec<ToolDefinition>,
    allow_git_writes: bool,
    monitor: Option<MonitorManager>,
}

impl Default for ToolRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl ToolRegistry {
    pub fn new() -> Self {
        Self::with_spawn_and_git_write_access(true, false)
    }

    pub fn new_with_git_write_access() -> Self {
        Self::with_spawn_and_git_write_access(true, true)
    }

    pub fn without_spawn() -> Self {
        Self::with_spawn_and_git_write_access(false, false)
    }

    fn with_spawn_and_git_write_access(include_spawn: bool, allow_git_writes: bool) -> Self {
        let mut definitions = vec![
            definition::<RunShellCommandArgs>(
                "run_shell_command",
                "Run a shell command on the user's machine with a timeout.",
            ),
            definition::<FetchArgs>(
                "fetch",
                "Fetch a public URL as readable text. Pass the original target URL, not a reader or proxy URL.",
            ),
            definition::<ReadFileArgs>("read_file", "Read a UTF-8 file from the filesystem."),
            definition::<WriteFileArgs>(
                "write_file",
                "Write complete UTF-8 file contents to a path and return a unified diff. Use this only when creating a new file or replacing most/all of an existing file. For small edits to existing files, prefer `search_replace`.",
            ),
            definition::<SearchReplaceArgs>(
                "search_replace",
                "Perform a targeted edit by replacing exact text in an existing file. Prefer this over `write_file` for small or localized modifications. Use when the original text can be matched exactly.",
            ),
            definition::<GenImageArgs>("gen_image", "Generate images with the OpenAI image API."),
            definition::<CommunicateArgs>(
                "communicate",
                "Send a brief, user-visible progress update. Use this proactively before starting a multi-step task, at meaningful milestones or when the plan changes, before lengthy checks, and when blocked or waiting. Say what you are doing or what happened. Do not use it for every tool call or a trivial task. This is an intermediate update, not the final answer.",
            ),
            definition::<PlaywrightBrowserSessionArgs>(
                "anonymous_browser_session",
                "Fallback for anonymous, isolated Playwright browsing when live Chrome is unavailable or the task requires Playwright JavaScript. Prefer browser for normal browsing, especially signed-in sites, because it uses the user's existing Chrome profile without copying it. This tool always launches a fresh empty profile with no user cookies, accounts, extensions, or browsing data. Never use it for signed-in browsing. For complex SPAs, do not wait for full page load or networkidle; use waitUntil: 'domcontentloaded' (or 'commit' when only the navigation response is needed), then wait for specific locators or readiness signals. The browser session persists across calls by default; set close=true when done, or reset=true to start fresh. Visibility is only for showing a window to the human user: it does not improve model vision, image loading, or rendering, and screenshots work headlessly. Never set visible=true merely to inspect images or screenshots; set it only if the user directly asks to see the browser window. Requires global playwright in PATH.",
            ),
            definition::<BrowserArgs>(
                "browser",
                "Default tool for all browser activity. Use this first for browsing, including signed-in sites; use anonymous_browser_session only when live Chrome is unavailable or Playwright JavaScript is required. Control only tabs opened by this tool in the user's existing Chrome profile via the locally loaded extension. Call action=status to distinguish connected/on, installed but switched off, and absent; the extension connects for status while off. Actions: open (url, optional focus=true to select tab and focus Chrome window; defaults to background), navigate (tab_id,url), snapshot (tab_id, rendered text and links), screenshot (tab_id, visible viewport image sent to the model; only works when that Agent tab is already active in its window; never activates or focuses automatically), click (tab_id,selector), type (tab_id,selector,text), scroll (tab_id,pixels), close (tab_id). For open, omit focus or set focus=false by default. Only use focus=true when the user explicitly asks to focus the tab or window; do not infer permission from a site's behavior. Never inspect pre-existing tabs. This is distinct from anonymous_browser_session, which launches a separate browser.",
            ),
        ];
        // Monitors are enabled only for interactive sessions with a wake-up receiver.
        if include_spawn {
            definitions.push(definition::<SpawnArgs>(
                "spawn",
                "Spawn focused single-invocation agents. Pass agents: [{task, model?, conversation_id?}, ...] for one or more tasks in parallel (results follow input order). Model must be an exact ID from /models (e.g. openai:gpt-4o); invalid models fail before any agent launches. Omitted models use the configured spawn model.",
            ));
        }

        Self {
            definitions,
            allow_git_writes,
            monitor: None,
        }
    }

    pub fn with_monitor(mut self, monitor: MonitorManager) -> Self {
        self.definitions.push(definition::<MonitorArgs>(
            "monitor",
            "Start a session-scoped background check and return control to the user. The shell command is polled until it exits 0, then the agent wakes once with a distinct monitor message. Failed checks retry; use stop with the returned ID to cancel, or list to see active IDs. Monitors end when the process exits. Do not use for untrusted commands.",
        ));
        self.monitor = Some(monitor);
        self
    }

    pub fn definitions(&self) -> &[ToolDefinition] {
        &self.definitions
    }

    pub async fn execute(&self, tool_call_id: String, name: &str, arguments: Value) -> ToolResult {
        self.execute_cancellable(tool_call_id, name, arguments, &CancellationToken::new())
            .await
    }

    pub async fn execute_cancellable(
        &self,
        tool_call_id: String,
        name: &str,
        arguments: Value,
        cancellation_token: &CancellationToken,
    ) -> ToolResult {
        let started = Instant::now();
        if !self
            .definitions
            .iter()
            .any(|definition| definition.name == name)
        {
            return tool_result(
                tool_call_id,
                name,
                ToolStatus::Error,
                format!("unknown tool: {name}"),
                Vec::new(),
                started,
                false,
            );
        }

        let arguments = match validated_tool_arguments(arguments) {
            Ok(arguments) => arguments,
            Err(content) => {
                return tool_result(
                    tool_call_id,
                    name,
                    ToolStatus::Error,
                    content,
                    Vec::new(),
                    started,
                    false,
                );
            }
        };

        let preserve_browser_snapshot = name == "browser" && arguments["action"] == "snapshot";
        let browser_screenshot = name == "browser" && arguments["action"] == "screenshot";
        let mut subagent_usages = Vec::new();
        let content = match name {
            "run_shell_command" => match serde_json::from_value::<RunShellCommandArgs>(arguments) {
                Ok(args) => {
                    run_shell_command_cancellable(args, cancellation_token, self.allow_git_writes)
                        .await
                }
                Err(err) => Err(format!("invalid tool arguments: {err}")),
            },
            "fetch" => match serde_json::from_value::<FetchArgs>(arguments) {
                Ok(args) => fetch(args).await,
                Err(err) => Err(format!("invalid tool arguments: {err}")),
            },
            "read_file" => match serde_json::from_value::<ReadFileArgs>(arguments) {
                Ok(args) => read_file(args).await,
                Err(err) => Err(format!("invalid tool arguments: {err}")),
            },
            "write_file" => match serde_json::from_value::<WriteFileArgs>(arguments) {
                Ok(args) => write_file(args).await,
                Err(err) => Err(format!("invalid tool arguments: {err}")),
            },
            "search_replace" => match serde_json::from_value::<SearchReplaceArgs>(arguments) {
                Ok(args) => search_replace(args).await,
                Err(err) => Err(format!("invalid tool arguments: {err}")),
            },
            "gen_image" => match serde_json::from_value::<GenImageArgs>(arguments) {
                Ok(args) => gen_image(args).await,
                Err(err) => Err(format!("invalid tool arguments: {err}")),
            },
            "communicate" => match serde_json::from_value::<CommunicateArgs>(arguments) {
                Ok(args) => communicate(args).await,
                Err(err) => Err(format!("invalid tool arguments: {err}")),
            },
            "anonymous_browser_session" => {
                if arguments.get("signed_in").is_some() {
                    Err("anonymous_browser_session no longer accepts signed_in; use browser for signed-in browsing in the user's existing Chrome.".to_string())
                } else {
                    match serde_json::from_value::<PlaywrightBrowserSessionArgs>(arguments) {
                        Ok(args) => playwright_browser_session(args).await,
                        Err(err) => Err(format!("invalid tool arguments: {err}")),
                    }
                }
            }
            "browser" => match serde_json::from_value::<BrowserArgs>(arguments) {
                Ok(args) => browser(args).await,
                Err(err) => Err(format!("invalid tool arguments: {err}")),
            },
            "monitor" => match serde_json::from_value::<MonitorArgs>(arguments) {
                Ok(args) => self.monitor.as_ref().map_or_else(
                    || Err("monitor unavailable".to_string()),
                    |manager| manager.execute(args, self.allow_git_writes),
                ),
                Err(err) => Err(format!("invalid tool arguments: {err}")),
            },
            "spawn" => match serde_json::from_value::<SpawnArgs>(arguments) {
                Ok(args) => {
                    match Box::pin(spawn_cancellable_with_usage(args, cancellation_token)).await {
                        Ok((content, usages)) => {
                            subagent_usages = usages;
                            Ok(content)
                        }
                        Err(error) => Err(error),
                    }
                }
                Err(err) => Err(format!("invalid tool arguments: {err}")),
            },
            _ => Err(format!("unknown tool: {name}")),
        };

        let (status, content, images) = match content {
            Ok(content) if browser_screenshot => match parse_browser_screenshot(&content) {
                Ok((description, image)) => (ToolStatus::Success, description, vec![image]),
                Err(error) => (ToolStatus::Error, error, Vec::new()),
            },
            Ok(content) => (ToolStatus::Success, content, Vec::new()),
            Err(content) => (ToolStatus::Error, content, Vec::new()),
        };
        let mut result = tool_result(
            tool_call_id,
            name,
            status,
            content,
            subagent_usages,
            started,
            preserve_browser_snapshot,
        );
        result.images = images;
        result
    }
}

fn parse_browser_screenshot(content: &str) -> Result<(String, ImageAttachment), String> {
    let response: Value =
        serde_json::from_str(content).map_err(|e| format!("Invalid screenshot response: {e}"))?;
    let data_url = response["image"]
        .as_str()
        .ok_or("Screenshot has no image")?;
    let data = data_url
        .strip_prefix("data:image/png;base64,")
        .ok_or("Screenshot is not a PNG data URL")?;
    if data.is_empty() {
        return Err("Screenshot is empty".into());
    }
    Ok((
        format!(
            "Screenshot of {} (tab {})",
            response["url"].as_str().unwrap_or("page"),
            response["tab_id"]
        ),
        ImageAttachment {
            media_type: "image/png".into(),
            data: data.into(),
        },
    ))
}

fn tool_result(
    tool_call_id: String,
    name: &str,
    status: ToolStatus,
    content: String,
    subagent_usages: Vec<crate::agent::SubagentUsage>,
    started: Instant,
    preserve_browser_snapshot: bool,
) -> ToolResult {
    let content = if preserve_browser_snapshot && status == ToolStatus::Success {
        content
    } else {
        truncate_tool_output(content, &status)
    };
    ToolResult {
        tool_call_id,
        name: name.to_string(),
        status,
        content,
        elapsed_ms: Some(duration_ms(started)),
        subagent_usages,
        images: Vec::new(),
    }
}

#[cfg(test)]
mod browser_output_tests {
    use super::*;

    #[test]
    fn screenshot_response_becomes_image_attachment_not_base64_text() {
        let (description, image) = parse_browser_screenshot(
            r#"{"url":"https://example.org","tab_id":42,"image":"data:image/png;base64,aGVsbG8="}"#,
        )
        .unwrap();
        assert!(description.contains("https://example.org"));
        assert!(!description.contains("aGVsbG8="));
        assert_eq!(image.media_type, "image/png");
        assert_eq!(image.data, "aGVsbG8=");
        assert!(parse_browser_screenshot(r#"{"image":"data:text/html;base64,abc"}"#).is_err());
    }

    #[test]
    fn browser_result_preserves_full_snapshot_but_other_tools_still_trim() {
        let snapshot = format!("{{\"text\":\"{}END\"}}", "a".repeat(40_000));
        let browser = tool_result(
            "browser-call".into(),
            "browser",
            ToolStatus::Success,
            snapshot.clone(),
            Vec::new(),
            Instant::now(),
            true,
        );
        assert_eq!(browser.content, snapshot);
        for (name, status, preserve) in [
            ("fetch", ToolStatus::Success, false),
            ("browser", ToolStatus::Success, false),
            ("browser", ToolStatus::Error, true),
        ] {
            let other = tool_result(
                "other-call".into(),
                name,
                status,
                snapshot.clone(),
                Vec::new(),
                Instant::now(),
                preserve,
            );
            assert!(other.content.contains("[Output trimmed by the harness"));
        }
    }
}

fn duration_ms(started: Instant) -> u64 {
    started.elapsed().as_millis().try_into().unwrap_or(u64::MAX)
}

fn definition<T>(name: &str, description: &str) -> ToolDefinition
where
    T: JsonSchema,
{
    ToolDefinition {
        name: name.to_string(),
        description: format!(
            "{description} Include an `intent` argument explaining why this tool is being called (80 characters or fewer)."
        ),
        parameters: with_intent_parameter(
            serde_json::to_value(schema_for!(T))
                .unwrap_or_else(|_| Value::Object(Default::default())),
        ),
    }
}

fn with_intent_parameter(mut parameters: Value) -> Value {
    let Value::Object(schema) = &mut parameters else {
        return parameters;
    };

    let properties = schema
        .entry("properties")
        .or_insert_with(|| Value::Object(Map::new()));
    if let Value::Object(properties) = properties {
        let previous = std::mem::take(properties);
        properties.insert(
            "intent".to_string(),
            json!({
                "type": "string",
                "maxLength": 80,
                "description": "why this tool is being called; state the intention in 80 characters or fewer."
            }),
        );
        properties.extend(previous);
    }

    let required = schema
        .entry("required")
        .or_insert_with(|| Value::Array(Vec::new()));
    if let Value::Array(required) = required {
        required.retain(|field| field != "intent");
        required.insert(0, Value::String("intent".to_string()));
    }

    parameters
}

fn validated_tool_arguments(mut arguments: Value) -> Result<Value, String> {
    let Value::Object(object) = &mut arguments else {
        return Err(
            "invalid tool arguments: expected JSON object with required intent".to_string(),
        );
    };

    let Some(intent) = object.remove("intent") else {
        return Err("invalid tool arguments: missing required intent".to_string());
    };
    let Some(intent) = intent.as_str() else {
        return Err("invalid tool arguments: intent must be a string".to_string());
    };
    if intent.chars().count() > 80 {
        return Err("invalid tool arguments: intent must be 80 characters or fewer".to_string());
    }

    Ok(arguments)
}
