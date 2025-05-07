from types import FunctionType
from bs4 import BeautifulSoup
from functools import wraps
import sys
import requests
import aisuite as ai

MAX_LENGTH = 1000000


client = ai.Client()


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


def fetch(url: str):
    """Fetch content from the given URL with a size limit."""

    print(f"\n🔧 [fetch], url: [{url}]")

    response = requests.get(url)
    try:
        response.raise_for_status()
        text = extract_text(response.text)

        if len(text) > MAX_LENGTH:
            print("⚠️ truncating response")
            return (
                f"{text[:MAX_LENGTH]}\n\n"
                "[Content truncated due to size limitations]"
            )

        return (
            f"[URL]: {url}\n\n",
            text
        )

    except Exception as e:
        print(f"⚠️ status: [{response.status_code}]")
        return f"Error fetching URL {url}: {e}"


cli_input = " ".join(sys.argv[1:])
user_request = cli_input if cli_input else input("What'll it be, boss? ")
system_string = (
    "You are a helpful assistant."
    "You cite all sources and you include links in every citation."
)

messages = [{
    "role": "system",
    "content": system_string
}, {
    "role": "user",
    "content": user_request
}]


# Automatic tool execution with max_turns
response = client.chat.completions.create(
    model="openai:gpt-4.1-mini",
    messages=messages,
    tools=[fetch],
    max_turns=20  # Maximum number of back-and-forth tool calls
)

print("\n---\n")
print(response.choices[0].message.content)
