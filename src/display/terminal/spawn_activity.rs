use std::collections::{BTreeMap, HashSet};

use std::time::{Duration, Instant};

use unicode_width::UnicodeWidthStr;

use super::{truncate_to_width, working_input_preview};
use crate::tools::{SpawnActivity, SpawnState};

#[derive(Default)]
pub(super) struct SpawnRows {
    groups: Vec<(String, Group)>,
    // Keep only IDs after completion, so late events cannot reopen a group.
    closed: HashSet<String>,
}

struct Group {
    total: usize,
    rows: BTreeMap<usize, Row>,
}

struct Row {
    started: Instant,
    finished_after: Option<Duration>,
    task: String,
    label: String,
    outcome: Option<Outcome>,
}

#[derive(Clone, Copy)]
enum Outcome {
    Done,
    Failed,
    Cancelled,
}

pub(super) enum Change {
    None,
    Redraw,
    Complete(String),
}

impl SpawnRows {
    pub(super) fn update(&mut self, activity: &SpawnActivity, width: usize) -> Change {
        if activity.index >= activity.total || self.closed.contains(&activity.group) {
            return Change::None;
        }
        let (label, outcome) = match &activity.state {
            SpawnState::Usage(_) => return Change::None,
            SpawnState::Started => ("Waiting for model", None),
            SpawnState::Status(status) => (status.trim(), None),
            SpawnState::Done => ("done", Some(Outcome::Done)),
            SpawnState::Failed => ("failed", Some(Outcome::Failed)),
            SpawnState::Cancelled => ("cancelled", Some(Outcome::Cancelled)),
        };
        let position = self
            .groups
            .iter()
            .position(|(id, _)| id == &activity.group)
            .unwrap_or_else(|| {
                self.groups.push((
                    activity.group.clone(),
                    Group {
                        total: activity.total,
                        rows: BTreeMap::new(),
                    },
                ));
                self.groups.len() - 1
            });
        let group = &mut self.groups[position].1;
        if group.total != activity.total
            || group
                .rows
                .get(&activity.index)
                .is_some_and(|row| row.outcome.is_some())
        {
            return Change::None;
        }
        let row = group.rows.entry(activity.index).or_insert_with(|| Row {
            started: Instant::now(),
            finished_after: None,
            task: String::new(),
            label: String::new(),
            outcome: None,
        });
        if !activity.task.trim().is_empty() {
            row.task = working_input_preview(activity.task.trim());
        }
        row.label = working_input_preview(if label.is_empty() { "working" } else { label });
        row.outcome = outcome;
        if outcome.is_some() {
            row.finished_after = Some(row.started.elapsed());
        }
        if outcome.is_none() || group.rows.len() != group.total {
            return Change::Redraw;
        }
        let (mut done, mut failed, mut cancelled) = (0, 0, 0);
        for row in group.rows.values() {
            match row.outcome {
                Some(Outcome::Done) => done += 1,
                Some(Outcome::Failed) => failed += 1,
                Some(Outcome::Cancelled) => cancelled += 1,
                None => return Change::Redraw,
            }
        }
        self.close(&activity.group);
        let summary = format!("Spawn: {done} done, {failed} failed, {cancelled} cancelled");
        if summary.width() <= width {
            Change::Complete(summary)
        } else {
            Change::Complete(clip(
                &format!("Spawn: {done}✓ {failed}✗ {cancelled}−"),
                width,
            ))
        }
    }

    pub(super) fn contains(&self, id: &str) -> bool {
        self.groups.iter().any(|(group, _)| group == id)
    }

    pub(super) fn close(&mut self, id: &str) {
        self.groups.retain(|(group, _)| group != id);
        self.closed.insert(id.to_owned());
    }

    pub(super) fn lines(&self, budget: usize, width: usize) -> Vec<String> {
        let count: usize = self.groups.iter().map(|(_, group)| group.rows.len()).sum();
        let shown = if count > budget {
            budget.saturating_sub(1)
        } else {
            count
        };
        let mut lines: Vec<_> = self
            .groups
            .iter()
            .flat_map(|(_, group)| {
                group
                    .rows
                    .iter()
                    .map(|(index, row)| (index + 1, group.total, row))
            })
            .take(shown)
            .map(|(number, total, row)| row.line(number, total, width))
            .collect();
        if shown < count && budget > 0 {
            lines.push(clip(
                &format!("[Spawn] … {} more agents", count - shown),
                width,
            ));
        }
        lines
    }
}

impl Row {
    fn line(&self, number: usize, total: usize, width: usize) -> String {
        let prefix = format!("[Spawn {number}/{total}] ");
        // On narrow displays prioritize the task and status over the clock.
        let clock = if width >= 40 {
            format!(
                " · {}s",
                self.finished_after
                    .unwrap_or_else(|| self.started.elapsed())
                    .as_secs()
            )
        } else {
            String::new()
        };
        let available = width.saturating_sub(prefix.width() + clock.width());
        if self.task.is_empty() || available < 7 {
            return clip(&format!("{prefix}{}", self.label), width);
        }
        // Keep room for the task even if the latest status is long.
        let body = available - 3; // " · "
        let label_width = body - self.task.width().min(body / 2);
        let label = clip(&self.label, label_width);
        let task = clip(&self.task, body - label.width());
        format!("{prefix}{label} · {task}{clock}")
    }
}

fn clip(text: &str, width: usize) -> String {
    if width == 0 {
        String::new()
    } else if text.width() <= width {
        text.to_owned()
    } else {
        format!("{}…", truncate_to_width(text, width - 1))
    }
}
