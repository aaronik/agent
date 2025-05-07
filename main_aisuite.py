import sys
import pprint
import aisuite as ai
import src.tools as tools

client = ai.Client()


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
    tools=[tools.fetch, tools.search_text, tools.search_images],
    max_turns=20  # Maximum number of back-and-forth tool calls
)

pp = pprint.PrettyPrinter(indent=4, width=80, compact=False)
pp.pprint(response)

for i, choice in enumerate(response.choices):
    print(f"\n--- Choice {i} ---\n")
    print(choice.message.content)

print(f"\n---\ntotal tokens: {response.usage.total_tokens}")
