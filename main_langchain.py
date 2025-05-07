import os
import getpass
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
from langchain.tools import tool
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings

# Ensure you have OPENAI_API_KEY set or enter it at runtime
if "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = getpass.getpass(
        "Enter your OpenAI API key: ")

# A simple fetch tool to get plain text content from a URL


@tool("fetch")
def fetch(url: str) -> str:
    import requests
    resp = requests.get(url)
    resp.raise_for_status()
    text = resp.text
    return text


def main():
    # Initialize chat model
    chat = ChatOpenAI(model="gpt-4o-mini")

    # Get URL to fetch
    url = input("Enter URL to fetch document: ").strip()

    # Fetch document content
    print(f"Fetching document from: {url}")
    document_text = fetch(url)

    # Create documents object for retriever (simulate simple retrieval)
    loader = TextLoader.from_text(document_text, source=url)
    docs = loader.load()

    # Initialize embeddings and vector store from the fetched document
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)

    # Create RetrievalQA chain using the vectorstore and chat model
    qa_chain = RetrievalQA.from_chain_type(
        llm=chat, retriever=vectorstore.as_retriever())

    # CLI loop for asking questions
    print("You can now ask questions about the fetched document. Type 'exit' to quit.")
    while True:
        query = input("Your question: ")
        if query.lower() in ("exit", "quit"):
            break

        answer = qa_chain.run(query)
        print(f"Answer: {answer}\n")


if __name__ == "__main__":
    main()
