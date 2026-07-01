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
        return messages.to_vec();
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
    selected
}

fn message_to_text(message: &AgentMessage) -> String {
    match message {
        AgentMessage::System { content } => format!("system: {content}"),
        AgentMessage::User { content } => format!("user: {content}"),
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
