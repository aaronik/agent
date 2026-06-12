use std::collections::BTreeMap;

use async_openai::{Client, config::OpenAIConfig};
use async_trait::async_trait;
use futures_util::StreamExt;
use serde_json::{Map, Value, json};

use crate::agent::{AgentMessage, AssistantMessage, ProviderEvent, ToolCall, Usage};
use crate::providers::{Provider, ProviderError};
use crate::tools::ToolDefinition;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ProviderFlavor {
    OpenAiResponses,
    OpenAiChat,
}

#[derive(Clone, Debug)]
pub struct ProviderConfig {
    pub provider: String,
    pub model: String,
    pub base_url: String,
    pub api_key: String,
    pub flavor: ProviderFlavor,
}

#[derive(Clone, Debug)]
pub struct OpenAiCompatibleProvider {
    config: ProviderConfig,
    client: Client<OpenAIConfig>,
}

impl OpenAiCompatibleProvider {
    pub fn new(config: ProviderConfig) -> Self {
        let openai_config = OpenAIConfig::new()
            .with_api_base(config.base_url.clone())
            .with_api_key(config.api_key.clone());
        Self {
            config,
            client: Client::with_config(openai_config),
        }
    }

    async fn complete_chat(
        &self,
        messages: &[AgentMessage],
        tools: &[ToolDefinition],
    ) -> Result<AssistantMessage, ProviderError> {
        let value: Value = self
            .client
            .chat()
            .create_byot(json!({
                "model": self.config.model,
                "messages": chat_messages(messages),
                "tools": chat_tools(tools),
                "tool_choice": "auto"
            }))
            .await
            .map_err(|err| ProviderError::Request(err.to_string()))?;
        parse_chat_response(value)
    }

    async fn complete_responses(
        &self,
        messages: &[AgentMessage],
        tools: &[ToolDefinition],
    ) -> Result<AssistantMessage, ProviderError> {
        let value: Value = self
            .client
            .responses()
            .create_byot(json!({
                "model": self.config.model,
                "input": responses_input(messages),
                "tools": responses_tools(tools)
            }))
            .await
            .map_err(|err| ProviderError::Request(err.to_string()))?;
        parse_responses_response(value)
    }

    async fn stream_chat_events(
        &self,
        messages: &[AgentMessage],
        tools: &[ToolDefinition],
    ) -> Result<Vec<ProviderEvent>, ProviderError> {
        let mut stream = self
            .client
            .chat()
            .create_stream_byot(json!({
                "model": self.config.model,
                "messages": chat_messages(messages),
                "tools": chat_tools(tools),
                "tool_choice": "auto",
                "stream": true,
                "stream_options": {
                    "include_usage": true
                }
            }))
            .await
            .map_err(|err| ProviderError::Request(err.to_string()))?;

        let mut events = Vec::new();
        let mut content = String::new();
        let mut tool_calls = BTreeMap::new();
        let mut usage = None;

        while let Some(chunk) = stream.next().await {
            let chunk: Value = chunk.map_err(|err| ProviderError::Request(err.to_string()))?;
            if let Some(chunk_usage) = parse_usage(chunk.get("usage")) {
                usage = Some(chunk_usage.clone());
                events.push(ProviderEvent::Usage { usage: chunk_usage });
            }

            let Some(choices) = chunk.get("choices").and_then(Value::as_array) else {
                continue;
            };
            for choice in choices {
                let Some(delta) = choice.get("delta") else {
                    continue;
                };
                if let Some(text) = delta.get("content").and_then(Value::as_str) {
                    content.push_str(text);
                    events.push(ProviderEvent::TextDelta {
                        text: text.to_string(),
                    });
                }
                if let Some(calls) = delta.get("tool_calls").and_then(Value::as_array) {
                    ingest_tool_call_deltas(calls, &mut tool_calls, &mut events);
                }
            }
        }

        let final_tool_calls = finalize_tool_calls(tool_calls);
        for call in &final_tool_calls {
            events.push(ProviderEvent::ToolCall { call: call.clone() });
        }
        events.push(ProviderEvent::FinalMessage {
            message: AssistantMessage {
                content,
                tool_calls: final_tool_calls,
                usage,
                metadata: Map::new(),
            },
        });
        Ok(events)
    }

    async fn stream_responses_events(
        &self,
        messages: &[AgentMessage],
        tools: &[ToolDefinition],
    ) -> Result<Vec<ProviderEvent>, ProviderError> {
        let mut stream = self
            .client
            .responses()
            .create_stream_byot(json!({
                "model": self.config.model,
                "input": responses_input(messages),
                "tools": responses_tools(tools),
                "stream": true
            }))
            .await
            .map_err(|err| ProviderError::Request(err.to_string()))?;

        let mut events = Vec::new();
        let mut content = String::new();
        let mut final_message = None;

        while let Some(chunk) = stream.next().await {
            let chunk: Value = chunk.map_err(|err| ProviderError::Request(err.to_string()))?;
            match chunk.get("type").and_then(Value::as_str) {
                Some("response.output_text.delta") => {
                    if let Some(delta) = chunk.get("delta").and_then(Value::as_str) {
                        content.push_str(delta);
                        events.push(ProviderEvent::TextDelta {
                            text: delta.to_string(),
                        });
                    }
                }
                Some("response.function_call_arguments.delta") => {
                    if let Some(delta) = chunk.get("delta").and_then(Value::as_str) {
                        events.push(ProviderEvent::ToolCallDelta {
                            id: chunk
                                .get("call_id")
                                .or_else(|| chunk.get("item_id"))
                                .and_then(Value::as_str)
                                .unwrap_or("call_0")
                                .to_string(),
                            name: None,
                            arguments_delta: delta.to_string(),
                        });
                    }
                }
                Some("response.completed") => {
                    let Some(response) = chunk.get("response") else {
                        continue;
                    };
                    let message = parse_responses_response(response.clone())?;
                    if let Some(usage) = message.usage.clone() {
                        events.push(ProviderEvent::Usage { usage });
                    }
                    for call in &message.tool_calls {
                        events.push(ProviderEvent::ToolCall { call: call.clone() });
                    }
                    final_message = Some(message);
                }
                Some("response.failed") | Some("response.error") => {
                    return Err(ProviderError::Request(chunk.to_string()));
                }
                _ => {}
            }
        }

        let message = final_message.unwrap_or(AssistantMessage {
            content,
            tool_calls: Vec::new(),
            usage: None,
            metadata: Map::new(),
        });
        events.push(ProviderEvent::FinalMessage { message });
        Ok(events)
    }
}

#[async_trait]
impl Provider for OpenAiCompatibleProvider {
    async fn complete(
        &self,
        messages: &[AgentMessage],
        tools: &[ToolDefinition],
    ) -> Result<AssistantMessage, ProviderError> {
        match self.config.flavor {
            ProviderFlavor::OpenAiResponses => self.complete_responses(messages, tools).await,
            ProviderFlavor::OpenAiChat => self.complete_chat(messages, tools).await,
        }
    }

    async fn events(
        &self,
        messages: &[AgentMessage],
        tools: &[ToolDefinition],
    ) -> Result<Vec<ProviderEvent>, ProviderError> {
        match self.config.flavor {
            ProviderFlavor::OpenAiChat => self.stream_chat_events(messages, tools).await,
            ProviderFlavor::OpenAiResponses => self.stream_responses_events(messages, tools).await,
        }
    }
}

#[derive(Clone, Debug, Default)]
struct StreamingToolCall {
    id: Option<String>,
    name: Option<String>,
    arguments: String,
}

fn chat_messages(messages: &[AgentMessage]) -> Vec<Value> {
    messages
        .iter()
        .map(|message| match message {
            AgentMessage::System { content } => json!({"role": "system", "content": content}),
            AgentMessage::User { content } => json!({"role": "user", "content": content}),
            AgentMessage::Assistant(assistant) => {
                let mut value = json!({"role": "assistant", "content": assistant.content});
                if !assistant.tool_calls.is_empty() {
                    value["tool_calls"] = Value::Array(
                        assistant
                            .tool_calls
                            .iter()
                            .map(|call| {
                                json!({
                                    "id": call.id,
                                    "type": "function",
                                    "function": {
                                        "name": call.name,
                                        "arguments": call.arguments.to_string()
                                    }
                                })
                            })
                            .collect(),
                    );
                }
                value
            }
            AgentMessage::Tool(result) => json!({
                "role": "tool",
                "tool_call_id": result.tool_call_id,
                "content": result.content
            }),
        })
        .collect()
}

fn chat_tools(tools: &[ToolDefinition]) -> Vec<Value> {
    tools
        .iter()
        .map(|tool| {
            json!({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters
                }
            })
        })
        .collect()
}

fn responses_input(messages: &[AgentMessage]) -> Vec<Value> {
    let mut input = Vec::new();
    for message in messages {
        match message {
            AgentMessage::System { content } => {
                input.push(json!({"role": "system", "content": content}));
            }
            AgentMessage::User { content } => {
                input.push(json!({"role": "user", "content": content}));
            }
            AgentMessage::Assistant(assistant) => {
                if !assistant.content.is_empty() {
                    input.push(json!({"role": "assistant", "content": assistant.content}));
                }
                for call in &assistant.tool_calls {
                    input.push(json!({
                        "type": "function_call",
                        "call_id": call.id,
                        "name": call.name,
                        "arguments": call.arguments.to_string()
                    }));
                }
            }
            AgentMessage::Tool(result) => {
                input.push(json!({
                    "type": "function_call_output",
                    "call_id": result.tool_call_id,
                    "output": result.content
                }));
            }
        }
    }
    input
}

fn responses_tools(tools: &[ToolDefinition]) -> Vec<Value> {
    tools
        .iter()
        .map(|tool| {
            json!({
                "type": "function",
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters
            })
        })
        .collect()
}

fn parse_chat_response(value: Value) -> Result<AssistantMessage, ProviderError> {
    let message = value
        .pointer("/choices/0/message")
        .ok_or_else(|| ProviderError::InvalidResponse("missing choices[0].message".to_string()))?;
    let content = message
        .get("content")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string();
    let tool_calls = message
        .get("tool_calls")
        .and_then(Value::as_array)
        .map(|calls| calls.iter().filter_map(parse_chat_tool_call).collect())
        .unwrap_or_default();
    Ok(AssistantMessage {
        content,
        tool_calls,
        usage: parse_usage(value.get("usage")),
        metadata: Map::new(),
    })
}

fn parse_chat_tool_call(value: &Value) -> Option<ToolCall> {
    let id = value.get("id")?.as_str()?.to_string();
    let function = value.get("function")?;
    let name = function.get("name")?.as_str()?.to_string();
    let arguments = function
        .get("arguments")
        .and_then(Value::as_str)
        .and_then(|raw| serde_json::from_str(raw).ok())
        .unwrap_or(Value::Object(Map::new()));
    Some(ToolCall {
        id,
        name,
        arguments,
    })
}

fn ingest_tool_call_deltas(
    calls: &[Value],
    tool_calls: &mut BTreeMap<u64, StreamingToolCall>,
    events: &mut Vec<ProviderEvent>,
) {
    for call in calls {
        let index = call.get("index").and_then(Value::as_u64).unwrap_or(0);
        let entry = tool_calls.entry(index).or_default();
        if let Some(id) = call.get("id").and_then(Value::as_str) {
            entry.id = Some(id.to_string());
        }
        let mut name = None;
        let mut arguments_delta = String::new();
        if let Some(function) = call.get("function") {
            if let Some(function_name) = function.get("name").and_then(Value::as_str) {
                entry.name = Some(function_name.to_string());
                name = Some(function_name.to_string());
            }
            if let Some(arguments) = function.get("arguments").and_then(Value::as_str) {
                entry.arguments.push_str(arguments);
                arguments_delta = arguments.to_string();
            }
        }
        if !arguments_delta.is_empty() || name.is_some() {
            events.push(ProviderEvent::ToolCallDelta {
                id: entry.id.clone().unwrap_or_else(|| format!("call_{index}")),
                name,
                arguments_delta,
            });
        }
    }
}

fn finalize_tool_calls(tool_calls: BTreeMap<u64, StreamingToolCall>) -> Vec<ToolCall> {
    tool_calls
        .into_iter()
        .map(|(index, call)| ToolCall {
            id: call.id.unwrap_or_else(|| format!("call_{index}")),
            name: call.name.unwrap_or_default(),
            arguments: serde_json::from_str(&call.arguments).unwrap_or(Value::Object(Map::new())),
        })
        .collect()
}

fn parse_responses_response(value: Value) -> Result<AssistantMessage, ProviderError> {
    let mut content_parts = Vec::new();
    let mut tool_calls = Vec::new();
    if let Some(output) = value.get("output").and_then(Value::as_array) {
        for item in output {
            match item.get("type").and_then(Value::as_str) {
                Some("message") => {
                    if let Some(content) = item.get("content").and_then(Value::as_array) {
                        for block in content {
                            if let Some(text) = block.get("text").and_then(Value::as_str) {
                                content_parts.push(text.to_string());
                            }
                        }
                    }
                }
                Some("function_call") => {
                    let Some(id) = item
                        .get("call_id")
                        .or_else(|| item.get("id"))
                        .and_then(Value::as_str)
                    else {
                        continue;
                    };
                    let Some(name) = item.get("name").and_then(Value::as_str) else {
                        continue;
                    };
                    let arguments = item
                        .get("arguments")
                        .and_then(Value::as_str)
                        .and_then(|raw| serde_json::from_str(raw).ok())
                        .unwrap_or(Value::Object(Map::new()));
                    tool_calls.push(ToolCall {
                        id: id.to_string(),
                        name: name.to_string(),
                        arguments,
                    });
                }
                _ => {}
            }
        }
    }

    if content_parts.is_empty()
        && let Some(text) = value.get("output_text").and_then(Value::as_str)
    {
        content_parts.push(text.to_string());
    }

    Ok(AssistantMessage {
        content: content_parts.join("\n"),
        tool_calls,
        usage: parse_usage(value.get("usage")),
        metadata: Map::new(),
    })
}

fn parse_usage(value: Option<&Value>) -> Option<Usage> {
    let value = value?;
    let input_tokens = value
        .get("input_tokens")
        .or_else(|| value.get("prompt_tokens"))
        .and_then(Value::as_u64)?;
    let output_tokens = value
        .get("output_tokens")
        .or_else(|| value.get("completion_tokens"))
        .and_then(Value::as_u64)?;
    Some(Usage {
        input_tokens,
        output_tokens,
        raw: Some(value.clone()),
    })
}
