use schemars::{JsonSchema, schema_for};
use serde::Serialize;
use serde_json::{Map, Value, json};
use std::time::Instant;

use crate::agent::{CancellationToken, ToolResult, ToolStatus};
use crate::tools::browser::{BrowserControlArgs, browser_control};
use crate::tools::communicate::{CommunicateArgs, communicate};
use crate::tools::fetch::{FetchArgs, fetch};
use crate::tools::files::{
    ReadFileArgs, SearchReplaceArgs, WriteFileArgs, read_file, search_replace, write_file,
};
use crate::tools::image::{GenImageArgs, gen_image};
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
                "Communicate progress or intermediate status to the user.",
            ),
            definition::<BrowserControlArgs>(
                "browser_control",
                "Control Chrome with DevTools enabled and run Playwright JavaScript against it. By default, every new session is anonymous and uses an empty, isolated browser profile with no user cookies, accounts, extensions, or browsing data. Set signed_in=true only when the user explicitly requests access to their signed-in Chrome session; this copies the selected local Chrome profile into an isolated temporary directory. For complex SPAs, do not wait for full page load or networkidle; use waitUntil: 'domcontentloaded' (or 'commit' when only the navigation response is needed), then wait for specific locators or readiness signals. The browser session persists across calls by default; set close=true when done, or reset=true to start fresh. Set visible=true only if the user directly asks to see it. Requires global playwright in PATH.",
            ),
        ];
        if include_spawn {
            definitions.push(definition::<SpawnArgs>(
                "spawn",
                "Spawn a focused single-invocation agent using the configured provider.",
            ));
        }

        Self {
            definitions,
            allow_git_writes,
        }
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
                );
            }
        };

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
            "browser_control" => match serde_json::from_value::<BrowserControlArgs>(arguments) {
                Ok(args) => browser_control(args).await,
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

        let (status, content) = match content {
            Ok(content) => (ToolStatus::Success, content),
            Err(content) => (ToolStatus::Error, content),
        };
        tool_result(
            tool_call_id,
            name,
            status,
            content,
            subagent_usages,
            started,
        )
    }
}

fn tool_result(
    tool_call_id: String,
    name: &str,
    status: ToolStatus,
    content: String,
    subagent_usages: Vec<crate::agent::SubagentUsage>,
    started: Instant,
) -> ToolResult {
    let content = truncate_tool_output(content, &status);
    ToolResult {
        tool_call_id,
        name: name.to_string(),
        status,
        content,
        elapsed_ms: Some(duration_ms(started)),
        subagent_usages,
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
        properties.insert(
            "intent".to_string(),
            json!({
                "type": "string",
                "maxLength": 80,
                "description": "why this tool is being called; state the intention in 80 characters or fewer."
            }),
        );
    }

    let required = schema
        .entry("required")
        .or_insert_with(|| Value::Array(Vec::new()));
    if let Value::Array(required) = required
        && !required.iter().any(|field| field == "intent")
    {
        required.push(Value::String("intent".to_string()));
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
