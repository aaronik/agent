import subprocess
from bs4 import BeautifulSoup
from types import FunctionType
from functools import wraps
from langchain_core.messages import BaseMessage
from openai.types.chat.chat_completion import ChatCompletion, Choice
import aisuite as ai
from pydantic import BaseModel
import os
from litellm import get_model_cost_map


class TokenUsage(BaseModel):
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    _cost_map: dict = None
    _seen_message_ids: set = None

    def model_post_init(self, __context):
        """Load cost map after model initialization"""
        if self._cost_map is None:
            object.__setattr__(self, '_cost_map', get_model_cost_map(url=''))
        if self._seen_message_ids is None:
            object.__setattr__(self, '_seen_message_ids', set())

    def prompt_cost(self) -> float:
        """
        Return cost of prompt tokens using LiteLLM pricing data
        """
        if self.prompt_tokens == 0:
            return 0.0

        try:
            if self.model not in self._cost_map:
                print(f"⚠️  Warning: Model '{self.model}' not found in LiteLLM cost map")
                return 0.0

            input_cost_per_token = self._cost_map[self.model].get('input_cost_per_token', 0)
            return round(self.prompt_tokens * input_cost_per_token, 4)
        except Exception as e:
            print(f"⚠️  Warning: Could not get pricing for model '{self.model}': {e}")
            return 0.0

    def completion_cost(self) -> float:
        """
        Return cost of completion tokens using LiteLLM pricing data
        """
        if self.completion_tokens == 0:
            return 0.0

        try:
            if self.model not in self._cost_map:
                print(f"⚠️  Warning: Model '{self.model}' not found in LiteLLM cost map")
                return 0.0

            output_cost_per_token = self._cost_map[self.model].get('output_cost_per_token', 0)
            return round(self.completion_tokens * output_cost_per_token, 4)
        except Exception as e:
            print(f"⚠️  Warning: Could not get pricing for model '{self.model}': {e}")
            return 0.0

    def total_tokens(self) -> int:
        """
        Return total tokens used
        """
        return round(self.prompt_tokens + self.completion_tokens, 4)

    def total_cost(self) -> float:
        """
        Return total cost as sum of prompt_cost and completion_cost
        """
        return round(self.prompt_cost() + self.completion_cost(), 4)

    def ingest_response(self, response: ChatCompletion):
        """
        Take a response and update internal usage statistics
        """
        if hasattr(response, 'usage') and response.usage is not None:
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
        """
        Ingest token usage from a list of messages incrementally, avoiding double-counting.
        Uses the object id of message objects to track which messages have been processed.
        """
        if self._seen_message_ids is None:
            self._seen_message_ids = set()

        for message in messages:
            mid = id(message)
            if mid in self._seen_message_ids:
                continue
            self._seen_message_ids.add(mid)

            if not getattr(message, 'response_metadata', None):
                continue

            try:
                c = message.response_metadata["token_usage"]["completion_tokens"]
                p = message.response_metadata["token_usage"]["prompt_tokens"]

                # Be defensive about types
                self.completion_tokens += int(c)
                self.prompt_tokens += int(p)
            except Exception:
                # If structure is unexpected, skip
                continue

    def print(self):
        """
        Print out the usage
        """
        print(
            "\n---\nusage",
            f"\ninput: {self.prompt_tokens} (${self.prompt_cost()})",
            f"\noutput: {self.completion_tokens} (${self.completion_cost()})",
            f"\ntotal: {self.total_tokens()} (${self.total_cost()})"
        )

    def print_panel(self):
        """
        Print a Rich panel showing the running cost
        """
        from rich.console import Console
        from rich.panel import Panel

        console = Console()
        cost_text = (
            f"[cyan]Input:[/cyan] {self.prompt_tokens} tokens (${self.prompt_cost():.4f}) | "
            f"[yellow]Output:[/yellow] {self.completion_tokens} tokens (${self.completion_cost():.4f}) | "
            f"[green]Total:[/green] ${self.total_cost():.4f}"
        )
        console.print(Panel(cost_text, title="💰 Running Cost", border_style="green", padding=(0, 1)))


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
            print("🔧 tool rejection: " + str(key))
            return ""
        else:
            cache[key] = True
            return func(*args, **kwargs)

    return return_func


def extract_text(html: str):
    soup = BeautifulSoup(html, 'html.parser')
    for script in soup(["script", "style"]):
        script.extract()  # Remove these tags
    text = soup.get_text(separator='\n')
    # Optionally clean up whitespace here
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    text = '\n'.join(chunk for chunk in chunks if chunk)
    return text


# Simple one to get the uname info of the running machine
def sys_uname():
    return subprocess.run(
        "uname -a", shell=True, capture_output=True, text=True
    ).stdout


def sys_ls():
    return subprocess.run(
        "ls -l", shell=True, capture_output=True, text=True
    ).stdout


def sys_pwd():
    return subprocess.run(
        "pwd", shell=True, capture_output=True, text=True
    ).stdout


def sys_git_ls():
    return subprocess.run(
        "git ls-files", shell=True, capture_output=True, text=True
    ).stdout or "fatal: not a git repository"


# Simple one to get the uname info of the running machine
def get_current_filetree():
    return subprocess.run(
        "find . -maxdepth 2", shell=True, capture_output=True, text=True
    ).stdout


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
        expanded_path.startswith('/') or
        expanded_path.startswith('./') or
        expanded_path.startswith('../')
    ):
        expanded_path = './' + expanded_path
    return expanded_path


def format_subproc_result(result: subprocess.CompletedProcess[str]) -> str:
    text = (
        "[STDOUT]\n" + result.stdout +
        "\n[STDERR]\n" + result.stderr +
        "\n[CODE]\n" + str(result.returncode)
    )

    return text
