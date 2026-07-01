use crate::agent::{
    AgentMessage, AgentTurnResult, AssistantMessage, CancellationToken, ProviderEvent, ToolCall,
    trim_messages,
};
use crate::providers::{Provider, ProviderError, configuration::DEFAULT_MODEL};
use crate::tools::ToolRegistry;

#[derive(Clone, Debug)]
pub struct AgentLoopConfig {
    pub max_turns: usize,
    pub max_context_tokens: usize,
    pub model: String,
}

impl Default for AgentLoopConfig {
    fn default() -> Self {
        Self {
            max_turns: 200,
            max_context_tokens: 16_384,
            model: DEFAULT_MODEL.to_string(),
        }
    }
}

pub struct AgentLoop<P> {
    provider: P,
    tools: ToolRegistry,
    config: AgentLoopConfig,
}

impl<P> AgentLoop<P>
where
    P: Provider,
{
    pub fn new(provider: P, tools: ToolRegistry, config: AgentLoopConfig) -> Self {
        Self {
            provider,
            tools,
            config,
        }
    }

    pub async fn run_turn(
        &self,
        starting_messages: &[AgentMessage],
    ) -> Result<AgentTurnResult, ProviderError> {
        self.run_turn_cancellable(starting_messages, &CancellationToken::new())
            .await
    }

    pub async fn run_turn_cancellable(
        &self,
        starting_messages: &[AgentMessage],
        cancellation_token: &CancellationToken,
    ) -> Result<AgentTurnResult, ProviderError> {
        self.run_turn_cancellable_with_observer(
            starting_messages,
            cancellation_token,
            |_| {},
            |_| {},
        )
        .await
    }

    pub async fn run_turn_cancellable_with_observer<F, G>(
        &self,
        starting_messages: &[AgentMessage],
        cancellation_token: &CancellationToken,
        mut on_message: F,
        mut on_tool_start: G,
    ) -> Result<AgentTurnResult, ProviderError>
    where
        F: FnMut(&AgentMessage),
        G: FnMut(&ToolCall),
    {
        let mut messages = starting_messages.to_vec();
        let mut new_messages = Vec::new();
        let mut usage = None;

        for _ in 0..self.config.max_turns {
            check_cancelled(cancellation_token)?;
            let provider_messages = trim_messages(
                &messages,
                &self.config.model,
                self.config.max_context_tokens,
            );
            let events = tokio::select! {
                events = self
                    .provider
                    .events(
                        &provider_messages,
                        self.tools.definitions(),
                    ) => events?,
                _ = cancellation_token.cancelled() => return Err(ProviderError::Cancelled),
            };
            check_cancelled(cancellation_token)?;
            let assistant = assistant_from_events(events)?;
            usage = assistant.usage.clone().or(usage);

            let tool_calls = assistant.tool_calls.clone();
            let final_text = assistant.content.clone();
            let assistant_message = AgentMessage::Assistant(assistant);
            on_message(&assistant_message);
            messages.push(assistant_message.clone());
            new_messages.push(assistant_message);

            if tool_calls.is_empty() {
                return Ok(AgentTurnResult {
                    new_messages,
                    final_text,
                    usage,
                });
            }

            for call in tool_calls {
                check_cancelled(cancellation_token)?;
                on_tool_start(&call);
                let result = tokio::select! {
                    result = self
                        .tools
                        .execute_cancellable(
                            call.id.clone(),
                            &call.name,
                            call.arguments.clone(),
                            cancellation_token,
                        ) => result,
                    _ = cancellation_token.cancelled() => return Err(ProviderError::Cancelled),
                };
                check_cancelled(cancellation_token)?;
                let tool_message = AgentMessage::Tool(result);
                on_message(&tool_message);
                messages.push(tool_message.clone());
                new_messages.push(tool_message);
            }
        }

        Err(ProviderError::MaxTurnsExceeded(self.config.max_turns))
    }
}

fn check_cancelled(cancellation_token: &CancellationToken) -> Result<(), ProviderError> {
    if cancellation_token.is_cancelled() {
        Err(ProviderError::Cancelled)
    } else {
        Ok(())
    }
}

fn assistant_from_events(events: Vec<ProviderEvent>) -> Result<AssistantMessage, ProviderError> {
    let mut usage = None;
    let mut text_deltas = Vec::new();
    let mut final_message = None;

    for event in events {
        match event {
            ProviderEvent::TextDelta { text } => text_deltas.push(text),
            ProviderEvent::Usage { usage: event_usage } => usage = Some(event_usage),
            ProviderEvent::FinalMessage { message } => final_message = Some(message),
            ProviderEvent::Error { message } => return Err(ProviderError::Request(message)),
            ProviderEvent::ToolCall { .. } | ProviderEvent::ToolCallDelta { .. } => {}
        }
    }

    let Some(mut message) = final_message else {
        return Err(ProviderError::InvalidResponse(
            "provider did not produce a final message".to_string(),
        ));
    };
    if message.content.is_empty() && !text_deltas.is_empty() {
        message.content = text_deltas.join("");
    }
    if message.usage.is_none() {
        message.usage = usage;
    }
    Ok(message)
}
