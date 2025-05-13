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


def find_upwards_claude_md_files(start_dir=None):
    # Find CLAUDE.md moving upward from start_dir to /, returns list of
    # absolute paths
    if start_dir is None:
        start_dir = os.getcwd()

    current_dir = os.path.abspath(start_dir)
    results = []
    while True:
        candidate = os.path.join(current_dir, "CLAUDE.md")
        if os.path.isfile(candidate):
            results.append(candidate)

        parent = os.path.dirname(current_dir)
        if parent == current_dir:
            break
        current_dir = parent
    return results


def find_nested_claude_md_files(start_dir=None):
    # Recursively find CLAUDE.md files inside start_dir tree
    if start_dir is None:
        start_dir = os.getcwd()
    matches = []
    for root, dirs, files in os.walk(start_dir):
        if "CLAUDE.md" in files:
            matches.append(os.path.join(root, "CLAUDE.md"))
    return matches


def find_all_claude_md_files(start_dir=None):
    # Combine upward and nested subtree CLAUDE.md files
    upwards = set(find_upwards_claude_md_files(start_dir))
    nested = set(find_nested_claude_md_files(start_dir))
    # Remove duplicates
    return list(upwards.union(nested))


def load_all_claude_memory(start_dir=None):
    files = find_all_claude_md_files(start_dir)
    # Read all found CLAUDE.md files and merge
    all_memory = []
    for f in files:
        all_memory.append(read_claude_md(f))
    return "\n\n".join(all_memory)
