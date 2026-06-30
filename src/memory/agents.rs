use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};

const MAX_IMPORT_DEPTH: usize = 5;

pub fn load_all_agents_memory(start_dir: Option<&Path>) -> String {
    find_all_agents_md_files(start_dir)
        .into_iter()
        .filter_map(|path| read_agents_md(&path, 0, &mut HashSet::new()).ok())
        .filter(|content| !content.is_empty())
        .collect::<Vec<_>>()
        .join("\n\n")
}

pub fn find_all_agents_md_files(start_dir: Option<&Path>) -> Vec<PathBuf> {
    let mut files = Vec::new();
    let start = start_dir
        .map(Path::to_path_buf)
        .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")));
    let project = start.join("AGENTS.md");
    if project.is_file() {
        files.push(project);
    }
    if let Some(home) = dirs::home_dir() {
        let user = home.join(".agent").join("AGENTS.md");
        if user.is_file() {
            files.push(user);
        }
    }
    files
}

fn read_agents_md(
    path: &Path,
    current_depth: usize,
    seen: &mut HashSet<PathBuf>,
) -> Result<String, std::io::Error> {
    if current_depth > MAX_IMPORT_DEPTH {
        return Ok(String::new());
    }

    let path = path.canonicalize().unwrap_or_else(|_| path.to_path_buf());
    if seen.contains(&path) {
        return Ok(String::new());
    }
    seen.insert(path.clone());

    let content = fs::read_to_string(&path)?;
    let base_dir = path.parent().unwrap_or_else(|| Path::new("."));
    let mut parts = vec![content.clone()];
    for import_path in parse_import_paths(&content) {
        let resolved = resolve_import(base_dir, &import_path);
        let imported = read_agents_md(&resolved, current_depth + 1, seen)?;
        if !imported.is_empty() {
            parts.push(imported);
        }
    }
    Ok(parts.join("\n"))
}

fn parse_import_paths(text: &str) -> Vec<String> {
    remove_markdown_code(text)
        .lines()
        .filter_map(|line| line.trim().strip_prefix('@').map(str::trim))
        .filter(|path| !path.is_empty())
        .map(ToOwned::to_owned)
        .collect()
}

fn remove_markdown_code(text: &str) -> String {
    let mut out = String::new();
    let mut in_block = false;
    for line in text.lines() {
        if line.trim_start().starts_with("```") {
            in_block = !in_block;
            continue;
        }
        if !in_block {
            out.push_str(&remove_inline_code(line));
            out.push('\n');
        }
    }
    out
}

fn remove_inline_code(line: &str) -> String {
    let mut out = String::new();
    let mut in_code = false;
    for ch in line.chars() {
        if ch == '`' {
            in_code = !in_code;
            continue;
        }
        if !in_code {
            out.push(ch);
        }
    }
    out
}

fn resolve_import(base_dir: &Path, import_path: &str) -> PathBuf {
    if import_path == "~" {
        return dirs::home_dir().unwrap_or_else(|| PathBuf::from(import_path));
    }
    if let Some(rest) = import_path.strip_prefix("~/") {
        return dirs::home_dir()
            .map(|home| home.join(rest))
            .unwrap_or_else(|| PathBuf::from(import_path));
    }
    let path = PathBuf::from(import_path);
    if path.is_absolute() {
        path
    } else {
        base_dir.join(path)
    }
}
