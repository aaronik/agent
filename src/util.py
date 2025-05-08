import subprocess
from bs4 import BeautifulSoup
from types import FunctionType
from functools import wraps
from openai.types.chat.chat_completion import Choice
import aisuite as ai
from pydantic import BaseModel


class TokenUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int


def print_token_usage(token_usage: TokenUsage):
    print(
        "\n---\ntoken usage",
        f"\nprompt: {token_usage.prompt_tokens}",
        f"\ncompletion: {token_usage.completion_tokens}",
        f"\ntotal: {token_usage.completion_tokens + token_usage.prompt_tokens}"
    )


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
def get_sys_info():
    return subprocess.run(
        "uname -a", shell=True, capture_output=True, text=True
    ).stdout


def message_from_choice(choice: Choice) -> ai.Message:
    return ai.Message(role=choice.message.role, content=choice.message.content)


def message_from_user_input(user_input: str) -> ai.Message:
    return ai.Message(role="user", content=user_input)
