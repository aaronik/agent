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
```

Run with Ollama:

```sh
OLLAMA_URL=http://localhost:11434 cargo run -- --model ollama:llama3.1
```

Refresh cached LiteLLM pricing data:

```sh
cargo run -- --update-pricing
```

Agent state is stored under `$HOME/.agent`.
Pricing data is cached under `$HOME/.agent/pricing`.
User-level instructions are read from `$HOME/.agent/AGENTS.md`.
