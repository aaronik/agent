# Agent

My personal agent harness - starts up instantly, has talk mode, the tools I need including signed in chrome browser.

## Commands

```sh
cargo fmt --check
cargo clippy -- -D warnings
cargo test
```

Run the mocked offline vertical slice:

```sh
cargo run -- --model mock --single "run echo hi"
```

Run with OpenAI:

```sh
OPENAI_API_KEY=... cargo run -- --model openai:gpt-5.6-terra
cargo run -- --model openai:gpt-5.6-terra --single --image screenshot.png "What is wrong here?"
```

Run with Ollama (using a vision-capable model):

```sh
OLLAMA_URL=http://localhost:11434 cargo run -- --model ollama:llama3.1
cargo run -- --model ollama:llava --single --image photo.jpg "Describe this image"
```

`--image PATH` works in text, `--single`, and `--command` modes and may be repeated.
For a voice conversation, start with `--talk --image photo.jpg "What should I notice?"`,
then discuss the image naturally. In an interactive text chat, drag an image from Finder into the
prompt, optionally add a question, and press Enter. This works on any turn, including resumed
chats. PNG, JPEG, GIF, and WebP files are supported. The selected model must support vision.

Refresh cached LiteLLM pricing data:

```sh
cargo run -- --update-pricing
```

Agent state is stored under `$HOME/.agent`.
Pricing data is cached under `$HOME/.agent/pricing`.
User-level instructions are read from `$HOME/.agent/AGENTS.md`.
