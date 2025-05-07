1. Haystack
GitHub: https://github.com/deepset-ai/haystack
Haystack is an open source NLP framework that allows you to build pipelines for LLMs, search, and question answering. It supports multiple provider backends (OpenAI, Cohere, HuggingFace, Anthropic, etc.) via connectors.

PROS:
* Big ecosystem, 20k stars, polished docs
* Easy pipeline creation (RAG, etc), rich set of integrations, fine control over components (retrievers, etc)
    * ex. https://docs.haystack.deepset.ai/docs/get_started#build-your-first-rag-application

CONS:
* No function -> tools sugar

4. Guidance
GitHub: https://github.com/microsoft/guidance
Microsoft Guidance provides an abstraction layer for authoring prompt programs and supports different LLM backends.

PROS:
* 20k stars, polished docs

CONS:
* Super trippy syntax. They're kind of trying to invent a language

5. Instructor
GitHub: https://github.com/jxnl/instructor
Abstraction for LLMs focused on structured output and prompt engineering, with support for multiple models/providers.

AISuite

PROS:

CONS:
* No advanced types in func sigs, including Literal[str1, str2]
