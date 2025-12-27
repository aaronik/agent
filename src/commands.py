from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable


@dataclass(frozen=True)
class CommandSpec:
    name: str
    usage: str
    help: str
    run: Callable[[str], bool]
    complete_args: Callable[[str], list[str]] | None = None


def format_help(commands: Iterable[CommandSpec]) -> str:
    lines: list[str] = ["Available commands:"]
    for c in sorted(commands, key=lambda x: x.name):
        lines.append(f"  {c.usage}")
        lines.append(f"      {c.help}")
    return "\n".join(lines) + "\n"


def filter_prefix(candidates: Iterable[str], *, prefix: str) -> list[str]:
    if not prefix:
        return list(candidates)
    return [c for c in candidates if c.startswith(prefix)]


def split_command(text: str) -> tuple[str, str] | None:
    """Split `/cmd rest...` into (cmd, rest).

    Returns None if input doesn't start with `/`.
    """

    s = text.strip()
    if not s.startswith("/"):
        return None

    parts = s.split(maxsplit=1)
    cmd = parts[0]
    rest = parts[1] if len(parts) > 1 else ""
    return cmd, rest
