import os
import sys
from functions import fetch  # Simulated tool interface, replace with actual integration

from openai import AzureOpenAI
from azure.identity import DefaultAzureCredential, get_bearer_token_provider

def fetch_content(url):
    # Use the fetch tool to get document content from the web
    result = fetch.fetch(url=url, max_length=10000)
    return result['text'] if 'text' in result else ''

def get_ai_answer(prompt):
    # Setup AzureOpenAI client with environment variables and Azure AD token provider
    token_provider = get_bearer_token_provider(DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default")

    client = AzureOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        azure_ad_token_provider=token_provider,
        api_version="2024-10-21"
    )

    deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
    if not deployment_name:
        raise ValueError("AZURE_OPENAI_DEPLOYMENT_NAME env var not set.")

    response = client.completions.create(
        model=deployment_name,
        prompt=prompt,
        max_tokens=200
    )
    return response.choices[0].text.strip()

def main():
    if len(sys.argv) < 2:
        print("Usage: python ai_cli.py <question>")
        sys.exit(1)

    user_question = sys.argv[1]

    # Example URL of documentation to fetch content from Microsoft docs
    doc_url = "https://learn.microsoft.com/en-us/azure/ai-services/openai/quickstart"

    print(f"Fetching documentation content from: {doc_url}")
    doc_content = fetch_content(doc_url)

    # Compose prompt with fetched doc content and user question
    prompt = f"Using the following document content, answer the question:\n\n{doc_content}\n\nQuestion: {user_question}\nAnswer:"

    print("Querying AI model...")
    answer = get_ai_answer(prompt)

    print("\nAI Answer:")
    print(answer)

if __name__ == "__main__":
    main()
