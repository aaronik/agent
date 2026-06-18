use async_trait::async_trait;
use serde_json::json;

use crate::agent::{AgentMessage, AssistantMessage, ToolCall};
use crate::providers::{Provider, ProviderError};
use crate::tools::ToolDefinition;

#[derive(Clone, Debug)]
pub struct MockProvider {
    command: String,
}

impl Default for MockProvider {
    fn default() -> Self {
        Self {
            command: "echo hi".to_string(),
        }
    }
}

impl MockProvider {
    pub fn new(command: impl Into<String>) -> Self {
        Self {
            command: command.into(),
        }
    }
}

#[async_trait]
impl Provider for MockProvider {
    async fn complete(
        &self,
        messages: &[AgentMessage],
        tools: &[ToolDefinition],
    ) -> Result<AssistantMessage, ProviderError> {
        if tools.is_empty() {
            let content = messages
                .iter()
                .rev()
                .find_map(|message| match message {
                    AgentMessage::User { content } => Some(content.clone()),
                    _ => None,
                })
                .unwrap_or_default();
            return Ok(AssistantMessage {
                content,
                tool_calls: Vec::new(),
                usage: None,
                metadata: Default::default(),
            });
        }

        if messages
            .iter()
            .any(|message| matches!(message, AgentMessage::Tool(_)))
        {
            let output = messages
                .iter()
                .rev()
                .find_map(|message| match message {
                    AgentMessage::Tool(result) => Some(result.content.trim().to_string()),
                    _ => None,
                })
                .unwrap_or_default();
            return Ok(AssistantMessage {
                content: format!("Tool completed: {output}"),
                tool_calls: Vec::new(),
                usage: None,
                metadata: Default::default(),
            });
        }

        Ok(AssistantMessage {
            content: String::new(),
            tool_calls: vec![ToolCall {
                id: "call_1".to_string(),
                name: "run_shell_command".to_string(),
                arguments: json!({
                    "intent": "run the configured mock command",
                    "cmd": self.command,
                    "timeout": 30
                }),
            }],
            usage: None,
            metadata: Default::default(),
        })
    }
}
