from types import FunctionType
from bs4 import BeautifulSoup
from functools import wraps
import sys
import requests
import pprint
import aisuite as ai
import duckduckgo_search as ddgs

MAX_LENGTH = 1000000


client = ai.Client()


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


def search(string: str, max_results: int = 3):
    """
    Search the web for a given string
    Limit the search to max_results
    """

    print(f"\n🔧 [search], string: [{string}], max_results: [{max_results}]")

    dds = ddgs.duckduckgo_search.DDGS()
    results = dds.text(string, max_results=max_results)

    text = ""

    for i, result in enumerate(results or [], start=1):
        print(f"🔗 {result['href']}")
        text += f"{i}. {result['title']}\n   {result['href']}\n"

    return text


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


response = client.chat.completions.create(
    model="openai:gpt-4.1-mini",
    messages=messages,
    tools=[fetch, search],
    max_turns=20  # Maximum number of back-and-forth tool calls
)

pp = pprint.PrettyPrinter(indent=4, width=80, compact=False)
# pp.pprint(response)

for i, choice in enumerate(response.choices):
    print(f"\n--- Choice {i} ---\n")
    print(choice.message.content)

print(f"\n---\ntotal tokens: {response.usage.total_tokens}")
