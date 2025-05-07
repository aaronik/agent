import requests
import aisuite as ai

client = ai.Client()


def fetch(url: str, max_length: int = 1000000):
    """Fetch content from the given URL with a size limit."""

    try:
        response = requests.get(url)
        response.raise_for_status()
        text = response.text

        if len(text) > max_length:
            return f"{text[:max_length]}\n\n[Content truncated due to size limitations]"

        return text

    except Exception as e:
        return f"Error fetching URL {url}: {e}"


messages = [{
    "role": "user",
    "content": "I live in San Francisco. Can you check for weather "
               "and plan an outdoor picnic for me at 2pm?"
}]

# Automatic tool execution with max_turns
response = client.chat.completions.create(
    model="openai:gpt-4.1-nano",
    messages=messages,
    tools=[fetch],
    max_turns=2  # Maximum number of back-and-forth tool calls
)
print(response.choices[0].message.content)
