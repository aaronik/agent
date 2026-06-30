use std::fs;
use std::io;
use std::path::PathBuf;

use uuid::Uuid;

use super::types::{SESSION_SCHEMA_VERSION, Session};

#[derive(Clone, Debug)]
pub struct SessionStore {
    root: PathBuf,
}

impl SessionStore {
    pub fn new() -> io::Result<Self> {
        let home = dirs::home_dir()
            .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "home directory not found"))?;
        Ok(Self {
            root: home.join(".agent"),
        })
    }

    pub fn with_root(root: PathBuf) -> Self {
        Self { root }
    }

    pub fn root(&self) -> &PathBuf {
        &self.root
    }

    pub fn prompt_history_path(&self) -> PathBuf {
        self.root.join("prompt_history")
    }

    pub fn sessions_dir(&self) -> PathBuf {
        self.root.join("sessions")
    }

    pub fn latest_session_path(&self) -> PathBuf {
        self.root.join("latest_session")
    }

    pub fn ensure_dirs(&self) -> io::Result<()> {
        fs::create_dir_all(self.sessions_dir())
    }

    pub fn new_session_id(&self) -> String {
        Uuid::new_v4().to_string()
    }

    pub fn save(&self, session: &Session) -> io::Result<()> {
        self.ensure_dirs()?;
        if session.schema_version != SESSION_SCHEMA_VERSION {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("unsupported session schema: {}", session.schema_version),
            ));
        }

        let session_path = self.session_path(&session.session_id);
        let tmp_path = session_path.with_extension("json.tmp");
        let payload = serde_json::to_string_pretty(session)
            .map_err(|err| io::Error::new(io::ErrorKind::InvalidData, err))?;
        fs::write(&tmp_path, format!("{payload}\n"))?;
        fs::rename(tmp_path, session_path)?;

        let latest = self.latest_session_path();
        let latest_tmp = latest.with_extension("tmp");
        fs::write(&latest_tmp, format!("{}\n", session.session_id))?;
        fs::rename(latest_tmp, latest)?;
        Ok(())
    }

    pub fn load(&self, session_id: Option<&str>) -> io::Result<Session> {
        self.ensure_dirs()?;
        let id = match session_id {
            Some(id) => id.to_string(),
            None => fs::read_to_string(self.latest_session_path())?
                .trim()
                .to_string(),
        };
        if id.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                "latest session pointer was empty",
            ));
        }

        let payload = fs::read_to_string(self.session_path(&id))?;
        let session: Session = serde_json::from_str(&payload)
            .map_err(|err| io::Error::new(io::ErrorKind::InvalidData, err))?;
        if session.schema_version != SESSION_SCHEMA_VERSION {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("unsupported session schema: {}", session.schema_version),
            ));
        }
        Ok(session)
    }

    pub fn list_session_ids(&self) -> io::Result<Vec<String>> {
        self.ensure_dirs()?;
        let mut ids = Vec::new();
        for entry in fs::read_dir(self.sessions_dir())? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().and_then(|ext| ext.to_str()) == Some("json")
                && let Some(stem) = path.file_stem().and_then(|stem| stem.to_str())
            {
                ids.push(stem.to_string());
            }
        }
        ids.sort_by(|a, b| b.cmp(a));
        Ok(ids)
    }

    pub fn list_session_labels(&self, max_preview_len: usize) -> io::Result<Vec<String>> {
        let mut labels = Vec::new();
        for id in self.list_session_ids()? {
            let label = match self.load(Some(&id)) {
                Ok(session) => {
                    let preview = session.messages.iter().find_map(|message| match message {
                        crate::agent::AgentMessage::User { content } => {
                            Some(collapse_preview(content, max_preview_len))
                        }
                        _ => None,
                    });
                    match preview {
                        Some(preview) if !preview.is_empty() => format!("{id}\t{preview}"),
                        _ => id,
                    }
                }
                Err(_) => id,
            };
            labels.push(label);
        }
        Ok(labels)
    }

    fn session_path(&self, session_id: &str) -> PathBuf {
        self.sessions_dir().join(format!("{session_id}.json"))
    }
}

fn collapse_preview(content: &str, max_len: usize) -> String {
    let mut preview = content.split_whitespace().collect::<Vec<_>>().join(" ");
    if preview.len() > max_len {
        preview.truncate(max_len.saturating_sub(1));
        preview.push_str("...");
    }
    preview
}
