use chrono::{DateTime, Local};
use serde::{Deserialize, Serialize};

use crate::agent::AgentMessage;

pub const SESSION_SCHEMA_VERSION: u32 = 1;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Session {
    pub schema_version: u32,
    pub session_id: String,
    pub created_at: DateTime<Local>,
    pub updated_at: DateTime<Local>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    pub messages: Vec<AgentMessage>,
}

impl Session {
    pub fn new(session_id: String, messages: Vec<AgentMessage>) -> Self {
        let now = Local::now();
        Self {
            schema_version: SESSION_SCHEMA_VERSION,
            session_id,
            created_at: now,
            updated_at: now,
            model: None,
            messages,
        }
    }

    pub fn saved_model(&self) -> Option<&str> {
        self.model
            .as_deref()
            .filter(|model| !model.is_empty())
            .or_else(|| {
                self.messages
                    .iter()
                    .rev()
                    .find_map(|message| match message {
                        AgentMessage::Assistant(assistant) => {
                            assistant.model.as_deref().filter(|model| !model.is_empty())
                        }
                        _ => None,
                    })
            })
    }

    pub fn replace_messages(&mut self, messages: Vec<AgentMessage>) {
        self.updated_at = Local::now();
        self.messages = messages;
    }
}
