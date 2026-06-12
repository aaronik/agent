use async_trait::async_trait;
use thiserror::Error;

use crate::agent::{AgentMessage, AssistantMessage, ProviderEvent};
use crate::tools::ToolDefinition;

#[derive(Debug, Error)]
pub enum ProviderError {
    #[error("provider request failed: {0}")]
    Request(String),
    #[error("provider response was invalid: {0}")]
    InvalidResponse(String),
    #[error("agent exceeded max tool turns ({0})")]
    MaxTurnsExceeded(usize),
    #[error("agent turn was cancelled")]
    Cancelled,
}

#[async_trait]
pub trait Provider: Send + Sync {
    async fn complete(
        &self,
        messages: &[AgentMessage],
        tools: &[ToolDefinition],
    ) -> Result<AssistantMessage, ProviderError>;

    async fn events(
        &self,
        messages: &[AgentMessage],
        tools: &[ToolDefinition],
    ) -> Result<Vec<ProviderEvent>, ProviderError> {
        Ok(vec![ProviderEvent::FinalMessage {
            message: self.complete(messages, tools).await?,
        }])
    }
}

#[async_trait]
impl<T> Provider for Box<T>
where
    T: Provider + ?Sized,
{
    async fn complete(
        &self,
        messages: &[AgentMessage],
        tools: &[ToolDefinition],
    ) -> Result<AssistantMessage, ProviderError> {
        (**self).complete(messages, tools).await
    }

    async fn events(
        &self,
        messages: &[AgentMessage],
        tools: &[ToolDefinition],
    ) -> Result<Vec<ProviderEvent>, ProviderError> {
        (**self).events(messages, tools).await
    }
}
