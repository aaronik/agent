use crate::agent::{
    AgentMessage, AgentTurnResult, AssistantMessage, CancellationToken, ProviderEvent, ToolCall,
};
use crate::providers::{Provider, ProviderError, configuration::DEFAULT_MODEL};
use crate::tools::ToolRegistry;

#[derive(Clone, Debug)]
pub struct AgentLoopConfig {
    pub max_turns: usize,
    pub model: String,
}

impl Default for AgentLoopConfig {
    fn default() -> Self {
        Self {
            // This is a safety guard for genuinely non-terminating tool loops, not a
            // practical cap on a normal long-running task. A 200-turn limit could
            // end an otherwise healthy run immediately after a tool result.
            max_turns: 1_000,
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
        on_message: F,
        on_tool_start: G,
    ) -> Result<AgentTurnResult, ProviderError>
    where
        F: FnMut(&AgentMessage),
        G: FnMut(&ToolCall),
    {
        self.run_turn_cancellable_with_event_observer(
            starting_messages,
            cancellation_token,
            |_| {},
            on_message,
            on_tool_start,
        )
        .await
    }

    pub async fn run_turn_cancellable_with_event_observer<E, F, G>(
        &self,
        starting_messages: &[AgentMessage],
        cancellation_token: &CancellationToken,
        mut on_event: E,
        mut on_message: F,
        mut on_tool_start: G,
    ) -> Result<AgentTurnResult, ProviderError>
    where
        E: FnMut(&ProviderEvent) + Send,
        F: FnMut(&AgentMessage),
        G: FnMut(&ToolCall),
    {
        let mut messages = starting_messages.to_vec();
        let mut new_messages = Vec::new();
        let mut usage = None;
        // A system message cannot fix a persistently rejected HTTP request.
        // Keep one recovery attempt, but never spend the tool-turn budget on it.
        let mut request_recovery_attempted = false;

        for _ in 0..self.config.max_turns {
            check_cancelled(cancellation_token)?;
            let mut events = Vec::new();
            let mut receive_event = |event| {
                on_event(&event);
                events.push(event);
            };
            let provider_result = tokio::select! {
                result = self.provider.stream_events(
                    &messages,
                    self.tools.definitions(),
                    &mut receive_event,
                ) => result,
                _ = cancellation_token.cancelled() => return Err(ProviderError::Cancelled),
            };
            let events = match provider_result {
                Ok(()) => events,
                Err(error @ ProviderError::ContextLengthExceeded(_)) => return Err(error),
                Err(ProviderError::Request(error)) => {
                    if request_recovery_attempted {
                        return Err(ProviderError::Request(error));
                    }
                    request_recovery_attempted = true;
                    let recovery_message = AgentMessage::System {
                        content: format!(
                            "[HARNESS ERROR] The provider rejected the previous request: {error}. \
                             Recover from this error and continue the task. Do not repeat an invalid \
                             tool call or tool output."
                        ),
                    };
                    on_message(&recovery_message);
                    messages.push(recovery_message.clone());
                    new_messages.push(recovery_message);
                    continue;
                }
                Err(error) => return Err(error),
            };
            check_cancelled(cancellation_token)?;
            let mut assistant = assistant_from_events(events)?;
            assistant.model = Some(self.config.model.clone());
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
            ProviderEvent::Error { message } => return Err(ProviderError::request(message)),
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
