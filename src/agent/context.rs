use crate::agent::AgentMessage;

pub fn count_tokens(messages: &[AgentMessage], model: &str) -> usize {
    let text = messages
        .iter()
        .map(message_to_text)
        .collect::<Vec<_>>()
        .join("\n");

    tiktoken_rs::bpe_for_model(model)
        .map(|bpe| bpe.encode_ordinary(&text).len())
        .unwrap_or_else(|_| approximate_tokens(&text))
}

pub fn trim_messages(
    messages: &[AgentMessage],
    model: &str,
    max_context_tokens: usize,
) -> Vec<AgentMessage> {
    if count_tokens(messages, model) <= max_context_tokens {
        return valid_tool_history(messages.to_vec());
    }

    let system_messages = messages
        .iter()
        .filter(|message| matches!(message, AgentMessage::System { .. }))
        .cloned()
        .collect::<Vec<_>>();
    let mut recent_messages = Vec::new();

    for message in messages
        .iter()
        .rev()
        .filter(|message| !matches!(message, AgentMessage::System { .. }))
    {
        let mut candidate = system_messages.clone();
        candidate.push(message.clone());
        candidate.extend(recent_messages.iter().rev().cloned());
        if count_tokens(&candidate, model) > max_context_tokens && !recent_messages.is_empty() {
            break;
        }
        recent_messages.push(message.clone());
    }

    let mut selected = system_messages;
    selected.extend(recent_messages.into_iter().rev());
    valid_tool_history(selected)
}

fn valid_tool_history(mut messages: Vec<AgentMessage>) -> Vec<AgentMessage> {
    let output_call_ids = messages
        .iter()
        .filter_map(|message| match message {
            AgentMessage::Tool(result) => Some(result.tool_call_id.clone()),
            _ => None,
        })
        .collect::<std::collections::HashSet<_>>();
    for message in &mut messages {
        if let AgentMessage::Assistant(assistant) = message {
            assistant
                .tool_calls
                .retain(|call| output_call_ids.contains(&call.id));
        }
    }

    let retained_call_ids = messages
        .iter()
        .filter_map(|message| match message {
            AgentMessage::Assistant(assistant) => Some(&assistant.tool_calls),
            _ => None,
        })
        .flatten()
        .map(|call| call.id.clone())
        .collect::<std::collections::HashSet<_>>();
    messages.retain(|message| match message {
        AgentMessage::Tool(result) => retained_call_ids.contains(result.tool_call_id.as_str()),
        _ => true,
    });
    messages
}

fn message_to_text(message: &AgentMessage) -> String {
    match message {
        AgentMessage::System { content } => format!("system: {content}"),
        AgentMessage::User { content } => format!("user: {content}"),
        AgentMessage::UserWithImages { content, images } => {
            format!("user: {content} [attachments: {} image(s)]", images.len())
        }
        AgentMessage::Assistant(assistant) => {
            let calls = assistant
                .tool_calls
                .iter()
                .map(|call| format!("{} {}", call.name, call.arguments))
                .collect::<Vec<_>>()
                .join("\n");
            format!("assistant: {}\n{calls}", assistant.content)
        }
        AgentMessage::Tool(result) => format!("tool {}: {}", result.name, result.content),
    }
}

fn approximate_tokens(text: &str) -> usize {
    (text.len() / 4).max(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trim_preserves_system_and_recent_messages() {
        let messages = vec![
            AgentMessage::System {
                content: "system".to_string(),
            },
            AgentMessage::User {
                content: "old ".repeat(200),
            },
            AgentMessage::User {
                content: "new".to_string(),
            },
        ];

        let trimmed = trim_messages(&messages, "unknown-local-model", 20);
        assert!(matches!(trimmed[0], AgentMessage::System { .. }));
        assert!(
            trimmed.iter().any(
                |message| matches!(message, AgentMessage::User { content } if content == "new")
            )
        );
    }

    #[test]
    fn trim_drops_orphaned_tool_outputs() {
        let messages = vec![
            AgentMessage::System {
                content: "system".to_string(),
            },
            AgentMessage::Assistant(crate::agent::AssistantMessage {
                content: String::new(),
                tool_calls: vec![crate::agent::ToolCall {
                    id: "call_old".to_string(),
                    name: "read_file".to_string(),
                    arguments: serde_json::json!({"path": "large ".repeat(200)}),
                }],
                usage: None,
                model: None,
                metadata: Default::default(),
            }),
            AgentMessage::Tool(crate::agent::ToolResult {
                tool_call_id: "call_old".to_string(),
                name: "read_file".to_string(),
                status: crate::agent::ToolStatus::Success,
                content: "ok".to_string(),
                elapsed_ms: None,
            }),
            AgentMessage::User {
                content: "newest".to_string(),
            },
        ];

        let trimmed = trim_messages(&messages, "unknown-local-model", 30);

        assert!(!trimmed.iter().any(|message| matches!(
            message,
            AgentMessage::Tool(result) if result.tool_call_id == "call_old"
        )));
    }

    #[test]
    fn trim_drops_function_calls_without_outputs() {
        let messages = vec![
            AgentMessage::Assistant(crate::agent::AssistantMessage {
                content: "working".to_string(),
                tool_calls: vec![crate::agent::ToolCall {
                    id: "call_incomplete".to_string(),
                    name: "read_file".to_string(),
                    arguments: serde_json::json!({"path": "README.md"}),
                }],
                usage: None,
                model: None,
                metadata: Default::default(),
            }),
            AgentMessage::User {
                content: "continue".to_string(),
            },
        ];

        let trimmed = trim_messages(&messages, "unknown-local-model", 16_384);
        let AgentMessage::Assistant(assistant) = &trimmed[0] else {
            panic!("expected assistant message");
        };
        assert!(assistant.tool_calls.is_empty());
        assert_eq!(assistant.content, "working");
    }

    #[test]
    fn trim_preserves_order_with_duplicate_messages() {
        let duplicate = AgentMessage::User {
            content: "duplicate".to_string(),
        };
        let messages = vec![
            AgentMessage::System {
                content: "system".to_string(),
            },
            duplicate.clone(),
            AgentMessage::User {
                content: "old filler ".repeat(200),
            },
            duplicate.clone(),
            AgentMessage::User {
                content: "newest".to_string(),
            },
        ];

        let trimmed = trim_messages(&messages, "unknown-local-model", 20);

        assert_eq!(
            trimmed,
            vec![
                AgentMessage::System {
                    content: "system".to_string(),
                },
                duplicate,
                AgentMessage::User {
                    content: "newest".to_string(),
                },
            ]
        );
    }
}
