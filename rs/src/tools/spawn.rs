use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::agent::{AgentLoop, AgentLoopConfig, AgentMessage};
use crate::providers::{build_provider, context_window_tokens, effective_model_name};
use crate::tools::ToolRegistry;

#[derive(Clone, Debug, Deserialize, JsonSchema, Serialize)]
pub struct SpawnArgs {
    pub task: String,
}

pub async fn spawn(args: SpawnArgs) -> Result<String, String> {
    let system = "You are a single-purpose AI agent spawned to complete one specific task. Focus solely on the given task, be concise, and provide actionable results.";
    let raw_model =
        std::env::var("AGENT_SPAWN_MODEL").unwrap_or_else(|_| effective_model_name(None));
    let provider =
        build_provider(&raw_model).map_err(|err| format!("Error spawning agent: {err}"))?;
    let loop_runner = AgentLoop::new(
        provider,
        ToolRegistry::without_spawn(),
        AgentLoopConfig {
            max_turns: 20,
            max_context_tokens: context_window_tokens(&raw_model),
            model: raw_model.clone(),
        },
    );

    let result = loop_runner
        .run_turn(&[
            AgentMessage::System {
                content: system.to_string(),
            },
            AgentMessage::User { content: args.task },
        ])
        .await
        .map_err(|err| format!("Error spawning agent: {err}"))?;

    Ok(format!("[SPAWNED AGENT OUTPUT]\n{}", result.final_text))
}
