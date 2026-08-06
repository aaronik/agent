use std::fs;
use std::io;
use std::path::PathBuf;

use uuid::Uuid;

use super::types::{SESSION_SCHEMA_VERSION, Session};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SessionSearchMatch {
    pub session_id: String,
    pub excerpt: String,
}

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

    pub fn archive_before_compaction(&self, session: &Session) -> io::Result<PathBuf> {
        let archive_dir = self.root.join("compactions").join(&session.session_id);
        fs::create_dir_all(&archive_dir)?;
        let timestamp = chrono::Utc::now().format("%Y%m%dT%H%M%S%6fZ");
        let path = archive_dir.join(format!("{timestamp}.json"));
        let payload = serde_json::to_string_pretty(session)
            .map_err(|err| io::Error::new(io::ErrorKind::InvalidData, err))?;
        fs::write(&path, format!("{payload}\n"))?;
        Ok(path)
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
        let mut sessions = Vec::new();
        for entry in fs::read_dir(self.sessions_dir())? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().and_then(|ext| ext.to_str()) == Some("json")
                && let Some(stem) = path.file_stem().and_then(|stem| stem.to_str())
            {
                let modified = entry
                    .metadata()
                    .and_then(|metadata| metadata.modified())
                    .unwrap_or(std::time::SystemTime::UNIX_EPOCH);
                sessions.push((stem.to_string(), modified));
            }
        }
        sessions.sort_by(|(left_id, left_modified), (right_id, right_modified)| {
            right_modified
                .cmp(left_modified)
                .then_with(|| right_id.cmp(left_id))
        });
        Ok(sessions.into_iter().map(|(id, _)| id).collect())
    }

    pub fn list_session_labels(&self, max_preview_len: usize) -> io::Result<Vec<String>> {
        let mut labels = Vec::new();
        for id in self.list_session_ids()? {
            let label = match self.load(Some(&id)) {
                Ok(session) => {
                    let preview = session.messages.iter().find_map(|message| match message {
                        crate::agent::AgentMessage::User { content }
                        | crate::agent::AgentMessage::UserWithImages { content, .. } => {
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

    pub fn find_sessions(
        &self,
        query: &str,
        max_excerpt_len: usize,
    ) -> io::Result<Vec<SessionSearchMatch>> {
        self.find_sessions_excluding(query, max_excerpt_len, None, usize::MAX)
    }

    pub fn find_sessions_excluding(
        &self,
        query: &str,
        max_excerpt_len: usize,
        excluded_session_id: Option<&str>,
        limit: usize,
    ) -> io::Result<Vec<SessionSearchMatch>> {
        let query_terms = search_terms(query);
        if query_terms.is_empty() || limit == 0 {
            return Ok(Vec::new());
        }

        let mut matches = Vec::new();
        for (recency, id) in self.list_session_ids()?.into_iter().enumerate() {
            if excluded_session_id == Some(id.as_str()) {
                continue;
            }
            let Ok(session) = self.load(Some(&id)) else {
                continue;
            };
            let mut best: Option<(usize, String)> = None;
            for message in &session.messages {
                let content = match message {
                    crate::agent::AgentMessage::User { content }
                    | crate::agent::AgentMessage::UserWithImages { content, .. } => content,
                    crate::agent::AgentMessage::Assistant(message) => &message.content,
                    crate::agent::AgentMessage::System { .. }
                    | crate::agent::AgentMessage::Tool(_) => continue,
                };
                let content_terms = searchable_terms(content);
                let score = query_terms
                    .iter()
                    .filter(|term| content_terms.binary_search(term).is_ok())
                    .count();
                if score == query_terms.len()
                    && best
                        .as_ref()
                        .is_none_or(|(best_score, _)| score > *best_score)
                {
                    best = Some((score, collapse_preview(content, max_excerpt_len)));
                }
            }
            if let Some((score, excerpt)) = best {
                matches.push((
                    score,
                    recency,
                    SessionSearchMatch {
                        session_id: id,
                        excerpt,
                    },
                ));
            }
        }
        matches.sort_by_key(|(score, recency, _)| (std::cmp::Reverse(*score), *recency));
        matches.truncate(limit);
        Ok(matches.into_iter().map(|(_, _, found)| found).collect())
    }

    fn session_path(&self, session_id: &str) -> PathBuf {
        self.sessions_dir().join(format!("{session_id}.json"))
    }
}

fn search_terms(query: &str) -> Vec<String> {
    const STOP_WORDS: &[&str] = &[
        "a", "an", "and", "for", "in", "of", "on", "the", "to", "we", "when", "with", "work",
        "worked", "working", "py",
    ];
    searchable_terms(query)
        .into_iter()
        .filter(|term| !STOP_WORDS.contains(&term.as_str()))
        .collect()
}

fn searchable_terms(content: &str) -> Vec<String> {
    let mut terms: Vec<String> = content
        .split_whitespace()
        .flat_map(|term| term.split(['_', '.', '-']))
        .map(|term| {
            term.trim_matches(|character: char| !character.is_alphanumeric())
                .to_lowercase()
        })
        .filter(|term| !term.is_empty())
        .collect();
    terms.sort();
    terms.dedup();
    terms
}

fn collapse_preview(content: &str, max_len: usize) -> String {
    let mut preview = content.split_whitespace().collect::<Vec<_>>().join(" ");
    if preview.chars().count() > max_len {
        preview = preview.chars().take(max_len.saturating_sub(1)).collect();
        preview.push_str("...");
    }
    preview
}
