import os
import subprocess
import threading
from functools import wraps
from types import FunctionType
from typing import Any

import aisuite as ai
from bs4 import BeautifulSoup
from langchain_core.messages import BaseMessage
from openai.types.chat.chat_completion import ChatCompletion, Choice
from pydantic import BaseModel

# --- LiteLLM cost map (lazy + optional background preload) -----------------

_LITELLM_COST_MAP: dict[str, Any] | None = None
_LITELLM_COST_MAP_LOCK = threading.Lock()
_LITELLM_COST_MAP_READY = threading.Event()
_LITELLM_PRELOAD_STARTED = False


def preload_litellm_cost_map() -> None:
    """Preload the LiteLLM model cost map in a background thread.

    This keeps CLI startup snappy while still enabling accurate token->cost
    calculations once the user actually needs them.

    Safe to call multiple times.
    """

    global _LITELLM_PRELOAD_STARTED

    # Fast path: already ready.
    if _LITELLM_COST_MAP_READY.is_set():
        return

    with _LITELLM_COST_MAP_LOCK:
        if _LITELLM_PRELOAD_STARTED:
            return
        _LITELLM_PRELOAD_STARTED = True

    def _worker() -> None:
        global _LITELLM_COST_MAP
        try:
            from litellm import get_model_cost_map

            cost_map = get_model_cost_map(url="")
            with _LITELLM_COST_MAP_LOCK:
                _LITELLM_COST_MAP = cost_map
        finally:
            # Always set, even if LiteLLM import fails. Cost calc will just
            # behave like "unknown model" and report $0.
            _LITELLM_COST_MAP_READY.set()

    threading.Thread(target=_worker, name="litellm-cost-map-preload", daemon=True).start()


def _get_litellm_cost_map_sync() -> dict[str, Any]:
    """Get LiteLLM's model cost map, importing LiteLLM only if needed."""

    global _LITELLM_COST_MAP

    if _LITELLM_COST_MAP is not None:
        return _LITELLM_COST_MAP

    # If a preload is in-flight, wait for it rather than double-importing.
    if _LITELLM_PRELOAD_STARTED:
        _LITELLM_COST_MAP_READY.wait(timeout=30)
        if _LITELLM_COST_MAP is not None:
            return _LITELLM_COST_MAP

    # Fallback: do it synchronously.
    from litellm import get_model_cost_map

    with _LITELLM_COST_MAP_LOCK:
        if _LITELLM_COST_MAP is None:
            _LITELLM_COST_MAP = get_model_cost_map(url="")
            _LITELLM_COST_MAP_READY.set()

    return _LITELLM_COST_MAP


class TokenUsage(BaseModel):
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    _cost_map: dict | None = None
    _seen_message_ids: set | None = None

    def model_post_init(self, __context):
        # Avoid importing LiteLLM at initialization time.
        if self._cost_map is None:
            object.__setattr__(self, "_cost_map", {})
        if self._seen_message_ids is None:
            object.__setattr__(self, "_seen_message_ids", set())

    def _ensure_cost_map(self) -> dict[str, Any]:
        # Prefer the shared module-level cache.
        if not self._cost_map:
            try:
                object.__setattr__(self, "_cost_map", _get_litellm_cost_map_sync())
            except Exception:
                # If LiteLLM import fails for any reason, cost calculations
                # should degrade gracefully (treat as unknown model => $0).
                object.__setattr__(self, "_cost_map", {})
        return self._cost_map

    def prompt_cost(self) -> float:
        """Return cost of prompt tokens using LiteLLM pricing data."""

        if self.prompt_tokens == 0:
            return 0.0

        try:
            cost_map = self._ensure_cost_map()
            if self.model not in cost_map:
                if not self.model.startswith("ollama:"):
                    print(f"Warning: Model '{self.model}' not found in LiteLLM cost map")
                return 0.0

            input_cost_per_token = cost_map[self.model].get("input_cost_per_token", 0)
            return round(self.prompt_tokens * input_cost_per_token, 4)
        except Exception as e:
            print(f"Warning: Could not get pricing for model '{self.model}': {e}")
            return 0.0

    def completion_cost(self) -> float:
        """Return cost of completion tokens using LiteLLM pricing data."""

        if self.completion_tokens == 0:
            return 0.0

        try:
            cost_map = self._ensure_cost_map()
            if self.model not in cost_map:
                if not self.model.startswith("ollama:"):
                    print(f"Warning: Model '{self.model}' not found in LiteLLM cost map")
                return 0.0

            output_cost_per_token = cost_map[self.model].get("output_cost_per_token", 0)
            return round(self.completion_tokens * output_cost_per_token, 4)
        except Exception as e:
            print(f"Warning: Could not get pricing for model '{self.model}': {e}")
            return 0.0

    def total_tokens(self) -> int:
        """Return total tokens used."""

        return round(self.prompt_tokens + self.completion_tokens, 4)

    def total_cost(self) -> float:
        """Return total cost as sum of prompt_cost and completion_cost."""

        return round(self.prompt_cost() + self.completion_cost(), 4)

    def ingest_response(self, response: ChatCompletion):
        """Take a response and update internal usage statistics."""

        if hasattr(response, "usage") and response.usage is not None:
            self.prompt_tokens += response.usage.prompt_tokens
            self.completion_tokens += response.usage.completion_tokens

    def ingest_from_messages(self, messages: list[BaseMessage]):
        self.prompt_tokens = 0
        self.completion_tokens = 0

        for message in messages:
            if not message.response_metadata:
                continue

            c = message.response_metadata["token_usage"]["completion_tokens"]
            p = message.response_metadata["token_usage"]["prompt_tokens"]

            self.completion_tokens += c
            self.prompt_tokens += p

    def ingest_messages_incremental(self, messages: list[BaseMessage]):
        """Ingest token usage incrementally, avoiding double-counting."""

        if self._seen_message_ids is None:
            self._seen_message_ids = set()

        for message in messages:
            mid = id(message)
            if mid in self._seen_message_ids:
                continue
            self._seen_message_ids.add(mid)

            if not getattr(message, "response_metadata", None):
                continue

            try:
                c = message.response_metadata["token_usage"]["completion_tokens"]
                p = message.response_metadata["token_usage"]["prompt_tokens"]

                self.completion_tokens += int(c)
                self.prompt_tokens += int(p)
            except Exception:
                continue

    def print_panel(self):
        """Print a Rich panel showing the running cost."""

        from rich.console import Console
        from rich.panel import Panel

        console = Console()
        cost_text = (
            f"[cyan]Input:[/cyan] {self.prompt_tokens} tokens (${self.prompt_cost():.4f}) | "
            f"[yellow]Output:[/yellow] {self.completion_tokens} tokens (${self.completion_cost():.4f}) | "
            f"[green]Total:[/green] ${self.total_cost():.4f}"
        )
        console.print(Panel(cost_text, title="Running Cost", border_style="grey37", padding=(0, 1)))


# Like a memoization, but blanks out the tool call if it's been done before
def refuse_if_duplicate(func: FunctionType):
    cache = {}

    @wraps(func)
    def return_func(*args, **kwargs):
        # Create a hashable key from args and kwargs
        key = args
        if kwargs:
            # Sort kwargs items to ensure consistent order
            key += tuple(sorted(kwargs.items()))
        if key in cache:
            print("tool rejection: " + str(key))
            return ""
        else:
            cache[key] = True
            return func(*args, **kwargs)

    return return_func


def extract_text(html: str):
    soup = BeautifulSoup(html, "html.parser")
    for script in soup(["script", "style"]):
        script.extract()  # Remove these tags
    text = soup.get_text(separator="\n")
    # Optionally clean up whitespace here
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    text = "\n".join(chunk for chunk in chunks if chunk)
    return text


# Simple one to get the uname info of the running machine
def sys_uname():
    return subprocess.run("uname -a", shell=True, capture_output=True, text=True).stdout


def sys_pwd():
    return subprocess.run("pwd", shell=True, capture_output=True, text=True).stdout


def sys_git_ls() -> str:
    """Return `git ls-files` output if we're inside a git worktree.

    Returns an empty string when not in a git repository.
    """

    in_repo = subprocess.run(
        "git rev-parse --is-inside-work-tree",
        shell=True,
        capture_output=True,
        text=True,
    )
    if in_repo.returncode != 0:
        return ""

    return subprocess.run("git ls-files", shell=True, capture_output=True, text=True).stdout


# Simple one to get the uname info of the running machine
def get_current_filetree():
    return subprocess.run("find . -maxdepth 2", shell=True, capture_output=True, text=True).stdout


def message_from_choice(choice: Choice) -> ai.Message:
    return ai.Message(role=choice.message.role, content=choice.message.content)


def message_from_user_input(user_input: str) -> ai.Message:
    return ai.Message(role="user", content=user_input)


# Do some basic path sanitization:
# * Ensure path is prefixed with folder
# * Get shorthands like ~ expanded
def sanitize_path(path: str):
    expanded_path = os.path.expanduser(path)
    if not (
        expanded_path.startswith("/")
        or expanded_path.startswith("./")
        or expanded_path.startswith("../")
    ):
        expanded_path = "./" + expanded_path
    return expanded_path


def format_subproc_result(result: subprocess.CompletedProcess[str]) -> str:
    """Format a subprocess result like a terminal would.

    We want tool output to read like running a command directly:
    - no [STDOUT]/[STDERR] headings
    - preserve stdout/stderr order as best-effort by concatenating
      (we don't have true interleaving when capturing separately)
    - include the exit code only when non-zero

    NOTE: If callers want true streaming / interleaving, they should avoid
    capture_output and stream to the console directly.
    """

    stdout = result.stdout or ""
    stderr = result.stderr or ""

    combined = stdout + stderr

    # Exit code is used by the UI (tool status) rather than printed inline,
    # so we don't append it to the combined output.
    return combined
