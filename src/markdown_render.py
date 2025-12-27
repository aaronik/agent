from __future__ import annotations

"""Markdown rendering helpers for terminal output.

We use Rich to render the assistant's final answers as Markdown in the CLI.
We intentionally disable raw HTML support to avoid accidental injection when
content includes snippets of untrusted text.
"""

from rich.console import Console
from rich.markdown import Markdown


def print_markdown(text: str) -> None:
    console = Console()
    # Rich's Markdown supports GitHub-ish basics (lists, bold, code fences, etc.).
    md = Markdown(text, code_theme="monokai", hyperlinks=True, inline_code_lexer="python")
    console.print(md)
