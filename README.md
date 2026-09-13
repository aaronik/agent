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
then discuss the image naturally. To speak one prompt and exit after its response, use
`--talk --single "What should I notice?"`. In an interactive text chat, drag an image from Finder into the
prompt, optionally add a question, and press Enter. This works on any turn, including resumed
chats. PNG, JPEG, GIF, and WebP files are supported. The selected model must support vision.

Refresh cached LiteLLM pricing data:

```sh
cargo run -- --update-pricing
```

Agent state is stored under `$HOME/.agent`.
Pricing data is cached under `$HOME/.agent/pricing`.
User-level instructions are read from `$HOME/.agent/AGENTS.md`.

## Terminal output

Live output uses normal terminal scrollback. Tool-start and completed panels are
append-only, and results are no longer shortened to a 30-line display preview
(the tool layer's existing size limits still apply). Long input wraps by display
width and keeps the editing cursor visible; very small windows hide the footer.

Set `AGENT_NO_LIVE=1` to disable the working footer and animation. Supporting
terminals use synchronized output to avoid presenting half-drawn frames; a resize
that interrupts a frame can still behave differently across terminal emulators.
See [INTERACTIVE_TESTING.md](INTERACTIVE_TESTING.md) for regression checks.

## Skills

Create user skills at `~/.agent/skills/<name>/SKILL.md` or project skills at
`.agent/skills/<name>/SKILL.md`. Project skills override user skills with the same name.
The agent can author these files using its normal file tools. Invoke a skill with
`/<name> [arguments]`; discovered skills also appear in slash completion.

```md
---
name: review
description: Review code for correctness
---
Review the requested code, run relevant tests, and report concrete issues.
```
