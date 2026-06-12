use schemars::{JsonSchema, schema_for};
use serde::Serialize;
use serde_json::Value;

use crate::agent::{ToolResult, ToolStatus};
use crate::tools::communicate::{CommunicateArgs, communicate};
use crate::tools::fetch::{FetchArgs, fetch};
use crate::tools::files::{
    ReadFileArgs, SearchReplaceArgs, WriteFileArgs, read_file, search_replace, write_file,
};
use crate::tools::image::{GenImageArgs, gen_image};
use crate::tools::shell::{RunShellCommandArgs, run_shell_command};
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
                "Write UTF-8 contents to a filesystem path and return a unified diff.",
            ),
            definition::<SearchReplaceArgs>(
                "search_replace",
                "Search for exact text in a file and replace all occurrences.",
            ),
            definition::<GenImageArgs>("gen_image", "Generate images with the OpenAI image API."),
            definition::<CommunicateArgs>(
                "communicate",
                "Communicate progress or intermediate status to the user.",
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
        let content = match name {
            "run_shell_command" => match serde_json::from_value::<RunShellCommandArgs>(arguments) {
                Ok(args) => run_shell_command(args).await,
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
            "spawn" => match serde_json::from_value::<SpawnArgs>(arguments) {
                Ok(args) => Box::pin(spawn(args)).await,
                Err(err) => Err(format!("invalid tool arguments: {err}")),
            },
            _ => Err(format!("unknown tool: {name}")),
        };

        match content {
            Ok(content) => ToolResult {
                tool_call_id,
                name: name.to_string(),
                status: ToolStatus::Success,
                content,
            },
            Err(content) => ToolResult {
                tool_call_id,
                name: name.to_string(),
                status: ToolStatus::Error,
                content,
            },
        }
    }
}

fn definition<T>(name: &str, description: &str) -> ToolDefinition
where
    T: JsonSchema,
{
    ToolDefinition {
        name: name.to_string(),
        description: description.to_string(),
        parameters: serde_json::to_value(schema_for!(T))
            .unwrap_or_else(|_| Value::Object(Default::default())),
    }
}
