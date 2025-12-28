import os
import re

MAX_IMPORT_DEPTH = 5


def read_file_content(filepath):
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return ""


def is_import_line(line):
    # Check if the line is a valid import line (starts with @ outside code
    # blocks/spans)
    return line.strip().startswith("@")


def remove_code_spans_and_blocks(text):
    # Remove markdown code blocks ```...```
    text = re.sub(r"```[\s\S]*?```", "", text)
    # Remove inline code spans `...`
    text = re.sub(r"`[^`]*`", "", text)
    return text


def parse_import_paths(text):
    # Cleanup text from code blocks and spans so imports inside those are not
    # evaluated
    cleaned = remove_code_spans_and_blocks(text)

    paths = []
    for line in cleaned.splitlines():
        line = line.strip()
        if line.startswith("@"):
            # path could be after @ and optional space
            path = line[1:].strip()
            if path:
                paths.append(path)
    return paths


def resolve_path(base_dir, path):
    if path.startswith("~"):  # User home dir
        return os.path.expanduser(path)
    elif os.path.isabs(path):
        return path
    else:
        return os.path.normpath(os.path.join(base_dir, path))


def read_claude_md(filepath, current_depth=0, seen_files=None):
    if seen_files is None:
        seen_files = set()

    if current_depth > MAX_IMPORT_DEPTH:
        return ""

    filepath = os.path.abspath(filepath)
    if filepath in seen_files:
        return ""

    seen_files.add(filepath)

    content = read_file_content(filepath)
    if not content:
        return ""

    base_dir = os.path.dirname(filepath)
    memory_texts = [content]

    import_paths = parse_import_paths(content)

    for import_path in import_paths:
        resolved = resolve_path(base_dir, import_path)
        imported_text = read_claude_md(resolved, current_depth + 1, seen_files)
        if imported_text:
            memory_texts.append(imported_text)

    return "\n".join(memory_texts)


def find_project_claude_md_file(start_dir=None):
    """Return the CLAUDE.md path in the *current directory only*.

    This intentionally does NOT search parent directories or recurse into
    subdirectories.
    """

    if start_dir is None:
        start_dir = os.getcwd()

    candidate = os.path.join(os.path.abspath(start_dir), "CLAUDE.md")
    return candidate if os.path.isfile(candidate) else None


def find_user_claude_md_file():
    """Return the user-level CLAUDE.md path at $HOME/.claude/CLAUDE.md."""

    candidate = os.path.join(os.path.expanduser("~"), ".claude", "CLAUDE.md")
    return candidate if os.path.isfile(candidate) else None


def find_all_claude_md_files(start_dir=None):
    """Return memory files in priority order: project-local then user-level."""

    files = []

    project = find_project_claude_md_file(start_dir)
    if project:
        files.append(project)

    user = find_user_claude_md_file()
    if user:
        files.append(user)

    return files


def load_all_claude_memory(start_dir=None):
    files = find_all_claude_md_files(start_dir)
    all_memory = []
    for f in files:
        all_memory.append(read_claude_md(f))
    return "\n\n".join(all_memory)
