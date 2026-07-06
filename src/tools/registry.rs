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
use crate::tools::shell::{RunShellCommandArgs, run_shell_command_cancellable};
use crate::tools::spawn::{SpawnArgs, spawn};

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    pub parameters: Value,
}

#[derive(Clone, Debug)]
pub struct ToolRegistry {
    definitions: Vec<ToolDefinition>,
}

impl Default for ToolRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl ToolRegistry {
    pub fn new() -> Self {
        Self::with_spawn(true)
    }

    pub fn without_spawn() -> Self {
        Self::with_spawn(false)
    }

    fn with_spawn(include_spawn: bool) -> Self {
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
                "Control a Chrome browser signed in as the user by copying a Chrome profile into a temporary user data directory, launching headless Chrome with DevTools enabled, and running Playwright JavaScript against it. The profile defaults to Default and can be selected with the profile argument, AGENT_BROWSER_CHROME_PROFILE, or AGENT_BROWSER_CHROME_PROFILE_DIR. The browser session persists across calls by default so exploration state can be reused; set close=true when the task is complete, or reset=true to start fresh. Set visible=true only if the user directly asks to see the browser. Requires global playwright in PATH.",
            ),
        ];
        if include_spawn {
            definitions.push(definition::<SpawnArgs>(
                "spawn",
                "Spawn a focused single-invocation agent using the configured provider.",
            ));
        }

        Self { definitions }
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
                started,
            );
        }

        let arguments = match validated_tool_arguments(arguments) {
            Ok(arguments) => arguments,
            Err(content) => {
                return tool_result(tool_call_id, name, ToolStatus::Error, content, started);
            }
        };

        let content = match name {
            "run_shell_command" => match serde_json::from_value::<RunShellCommandArgs>(arguments) {
                Ok(args) => run_shell_command_cancellable(args, cancellation_token).await,
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
                Ok(args) => Box::pin(spawn(args)).await,
                Err(err) => Err(format!("invalid tool arguments: {err}")),
            },
            _ => Err(format!("unknown tool: {name}")),
        };

        match content {
            Ok(content) => tool_result(tool_call_id, name, ToolStatus::Success, content, started),
            Err(content) => tool_result(tool_call_id, name, ToolStatus::Error, content, started),
        }
    }
}

fn tool_result(
    tool_call_id: String,
    name: &str,
    status: ToolStatus,
    content: String,
    started: Instant,
) -> ToolResult {
    ToolResult {
        tool_call_id,
        name: name.to_string(),
        status,
        content,
        elapsed_ms: Some(duration_ms(started)),
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
