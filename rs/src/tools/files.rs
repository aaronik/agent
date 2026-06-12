use std::fs;
use std::path::{Path, PathBuf};

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use similar::TextDiff;

#[derive(Clone, Debug, Deserialize, JsonSchema, Serialize)]
pub struct ReadFileArgs {
    pub path: String,
}

#[derive(Clone, Debug, Deserialize, JsonSchema, Serialize)]
pub struct WriteFileArgs {
    pub path: String,
    pub contents: String,
}

#[derive(Clone, Debug, Deserialize, JsonSchema, Serialize)]
pub struct SearchReplaceArgs {
    pub path: String,
    pub old_text: String,
    pub new_text: String,
}

pub async fn read_file(args: ReadFileArgs) -> Result<String, String> {
    let path = sanitize_path(&args.path);
    match fs::read_to_string(&path) {
        Ok(content) => Ok(format!("[FILE]: {}\n{}", path.display(), content)),
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => {
            Ok(format!("file not found: {}", path.display()))
        }
        Err(err) => Ok(format!("IOError while reading file: {err}")),
    }
}

pub async fn write_file(args: WriteFileArgs) -> Result<String, String> {
    let path = sanitize_path(&args.path);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|err| format!("Error creating directories: {err}"))?;
    }

    let previous = fs::read_to_string(&path).unwrap_or_default();
    fs::write(&path, &args.contents).map_err(|err| format!("Error writing to file: {err}"))?;

    let diff = unified_diff(
        &previous,
        &args.contents,
        &format!("{} (before)", file_name(&path)),
        &format!("{} (after)", file_name(&path)),
    );

    if diff.is_empty() {
        Ok("Success".to_string())
    } else {
        Ok(format!("Success\n\nDiff:\n{diff}"))
    }
}

pub async fn search_replace(args: SearchReplaceArgs) -> Result<String, String> {
    let path = sanitize_path(&args.path);
    let content = match fs::read_to_string(&path) {
        Ok(content) => content,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => {
            return Ok(format!("file not found: {}", path.display()));
        }
        Err(err) => return Ok(format!("IOError while reading file: {err}")),
    };

    if !content.contains(&args.old_text) {
        return Ok(format!("Text not found in file: {}", path.display()));
    }

    let occurrences = content.matches(&args.old_text).count();
    let new_content = content.replace(&args.old_text, &args.new_text);
    let diff = unified_diff(
        &content,
        &new_content,
        &format!("{} (before)", file_name(&path)),
        &format!("{} (after)", file_name(&path)),
    );

    fs::write(&path, new_content).map_err(|err| format!("Error writing to file: {err}"))?;
    Ok(format!(
        "Successfully replaced {occurrences} occurrence(s)\n\nDiff:\n{diff}"
    ))
}

pub fn sanitize_path(path: &str) -> PathBuf {
    let expanded = if path == "~" {
        dirs::home_dir()
            .map(|home| home.to_string_lossy().into_owned())
            .unwrap_or_else(|| path.to_string())
    } else if let Some(rest) = path.strip_prefix("~/") {
        dirs::home_dir()
            .map(|home| home.join(rest).to_string_lossy().into_owned())
            .unwrap_or_else(|| path.to_string())
    } else {
        path.to_string()
    };

    if expanded.starts_with('/') || expanded.starts_with("./") || expanded.starts_with("../") {
        PathBuf::from(expanded)
    } else {
        PathBuf::from(format!("./{expanded}"))
    }
}

fn unified_diff(old: &str, new: &str, old_name: &str, new_name: &str) -> String {
    TextDiff::from_lines(old, new)
        .unified_diff()
        .header(old_name, new_name)
        .to_string()
}

fn file_name(path: &Path) -> String {
    path.file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("file")
        .to_string()
}
