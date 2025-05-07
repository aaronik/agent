import sys
import pprint
import aisuite as ai
import src.tools as tools
from openai.types.chat.chat_completion import ChatCompletion

client = ai.Client()


cli_input = " ".join(sys.argv[1:])
user_request = cli_input if cli_input else input("What'll it be, boss? ")
system_string = (
    "You are a helpful assistant."
    "Cite all sources and you include links in every citation."
    "Use as many shell commands as needed if it helps meet the user's request."
)

messages = [{
    "role": "system",
    "content": system_string
}, {
    "role": "user",
    "content": user_request
}]

response: ChatCompletion = client.chat.completions.create(
    model="openai:gpt-4.1-mini",
    messages=messages,
    tools=[
        tools.fetch,
        tools.search_text,
        tools.search_images,
        tools.shell_command,
        tools.printz,
    ],
    max_turns=20  # Maximum number of back-and-forth tool calls
)

pp = pprint.PrettyPrinter(indent=4, width=80, compact=False)
# pp.pprint(response)

for i, choice in enumerate(response.choices):
    print("\n---\n")
    print(choice.message.content)

print(f"\n---\ntotal tokens: {response.usage and response.usage.total_tokens}")
