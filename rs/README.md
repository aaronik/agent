# agent-rs

Standalone Rust rewrite of the agent CLI.

This crate must not run, embed, import, shell out to, or otherwise depend on Python code.

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
OPENAI_API_KEY=... cargo run -- --model openai:gpt-5.5
```

Run with Ollama:

```sh
OLLAMA_URL=http://localhost:11434 cargo run -- --model ollama:llama3.1
```

Refresh cached LiteLLM pricing data:

```sh
cargo run -- --update-pricing
```

Rust-native state is stored under `$HOME/.agent-rs`.
Pricing data is cached under `$HOME/.agent-rs/pricing`.
