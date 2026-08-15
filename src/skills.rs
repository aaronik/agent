use std::collections::BTreeMap;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Skill {
    pub name: String,
    pub description: String,
    pub instructions: String,
    pub path: PathBuf,
}

pub fn discover(global_root: &Path, project_dir: &Path) -> io::Result<Vec<Skill>> {
    let mut skills = BTreeMap::new();
    load_dir(&global_root.join("skills"), &mut skills)?;
    load_dir(&project_dir.join(".agent/skills"), &mut skills)?;
    Ok(skills.into_values().collect())
}

pub fn find(global_root: &Path, project_dir: &Path, name: &str) -> io::Result<Option<Skill>> {
    Ok(discover(global_root, project_dir)?
        .into_iter()
        .find(|skill| skill.name == name))
}

fn load_dir(root: &Path, skills: &mut BTreeMap<String, Skill>) -> io::Result<()> {
    let entries = match fs::read_dir(root) {
        Ok(entries) => entries,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(()),
        Err(error) => return Err(error),
    };
    for entry in entries {
        let entry = entry?;
        let path = entry.path().join("SKILL.md");
        if !path.is_file() {
            continue;
        }
        let fallback_name = entry.file_name().to_string_lossy().into_owned();
        if let Ok(skill) = parse_skill(path, &fallback_name) {
            skills.insert(skill.name.clone(), skill);
        }
    }
    Ok(())
}

fn parse_skill(path: PathBuf, fallback_name: &str) -> io::Result<Skill> {
    let contents = fs::read_to_string(&path)?;
    let (metadata, instructions) = if let Some(rest) = contents.strip_prefix("---\n") {
        match rest.split_once("\n---\n") {
            Some(parts) => parts,
            None => ("", contents.as_str()),
        }
    } else {
        ("", contents.as_str())
    };
    let value = |key: &str| {
        metadata.lines().find_map(|line| {
            let (candidate, value) = line.split_once(':')?;
            (candidate.trim() == key).then(|| value.trim().trim_matches(['\'', '"']).to_string())
        })
    };
    let name = value("name").unwrap_or_else(|| fallback_name.to_string());
    if !valid_name(&name) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("invalid skill name: {name}"),
        ));
    }
    Ok(Skill {
        name,
        description: value("description").unwrap_or_default(),
        instructions: instructions.trim().to_string(),
        path,
    })
}

fn valid_name(name: &str) -> bool {
    !name.is_empty()
        && name
            .chars()
            .all(|character| character.is_ascii_alphanumeric() || matches!(character, '-' | '_'))
}

pub fn invocation_prompt(skill: &Skill, arguments: &str) -> String {
    format!(
        "[SKILL: {}]\n{}\n\nArguments:\n{}",
        skill.name,
        skill.instructions,
        if arguments.is_empty() {
            "(none)"
        } else {
            arguments
        }
    )
}
