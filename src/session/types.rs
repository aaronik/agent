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
            messages,
        }
    }

    pub fn replace_messages(&mut self, messages: Vec<AgentMessage>) {
        self.updated_at = Local::now();
        self.messages = messages;
    }
}
