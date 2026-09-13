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
