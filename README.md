# Agent

My personal agent harness - starts up instantly, has talk mode, the tools I need including signed in chrome browser.

## Live tabs in your existing Chrome (unpacked extension)

1. In your normal Chrome profile open `chrome://extensions`, enable **Developer mode**, click **Load unpacked**, and select this repo's `chrome-extension` directory. No second profile or Chrome restart is needed. Open its toolbar popup and switch **Allow Agent to control its tabs** on (off by default). The choice persists across Chrome restarts.
2. Run `cargo run -- --model openai:gpt-6-sol` (or your usual tool-capable model). The enabled extension automatically connects once an agent calls `browser`; ask it to call `browser` with `action=status` if you want to check. Use `action=open` with a URL; new tabs open in the background without selecting them and join a green tab group labeled **Agent**. Use the returned `tab_id` for snapshot/click/type/scroll/navigate/close; snapshots include the tab's group title and color so grouping can be verified. Multiple harness processes share the bridge; if its host exits, another process takes over the listener on its next browser call. An in-flight action interrupted by host exit has an unknown outcome and is not automatically repeated. The popup refreshes its status while open. On first use, the harness waits up to 35 seconds for Chrome's reconnect alarm; if the extension still shows *Waiting*, reopen its popup to trigger a retry. Chrome may show a connection-refused warning while no agent bridge is running.
3. To stop access, switch **Allow Agent to control its tabs** off in the popup (or disable the extension in `chrome://extensions`). Switching off prevents tab actions but keeps a status-only bridge connection, so the harness can distinguish off from an unloaded extension; tabs remain open while the agent runs, and tabs belonging to the current conversation close on `/clear` or `/new`, or when its agent process exits (including abrupt disconnects if the bridge stays running). If the bridge host and extension disconnect together (for example on crash or Chrome restart), automatic cleanup cannot be guaranteed; check the Agent group manually. The extension can only act on tabs *it opened* since it was installed (ownership survives service-worker sleep); it cannot list or inspect pre-existing tabs. Reload the extension after changing its files.

The local bridge listens on `127.0.0.1:17777` while at least one hosting agent runs. There is no pairing secret: **any process on your machine can call the bridge while the extension is enabled**. Tab IDs are selectors, not authorization; the extension enforces only the narrower boundary that it must have opened a tab before acting on it. The extension requests broad HTTP(S), tabs, tabGroups, scripting, storage and alarms permissions because Chrome does not enforce that boundary for us. Treat other locally running software as trusted. This is a minimal DOM-based browser tool (not Playwright): dynamic sites may require repeated snapshots after navigation, and browser UI/internal pages are inaccessible. For normal browsing, prefer `browser`. `anonymous_browser_session` remains an anonymous Playwright fallback for when live Chrome is unavailable or JavaScript control is needed; it always starts with a fresh empty profile and never copies your Chrome data. Its former `signed_in` argument is removed.

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
OPENAI_API_KEY=... cargo run -- --model openai:gpt-6-sol
cargo run -- --model openai:gpt-6-sol --single --image screenshot.png "What is wrong here?"
```

Run with Ollama (using a vision-capable model):

```sh
OLLAMA_URL=http://localhost:11434 cargo run -- --model ollama:llama3.1
cargo run -- --model ollama:llava --single --image photo.jpg "Describe this image"
```

`--image PATH` works in text, `--single`, and `--command` modes and may be repeated.
For a voice conversation, start with `--talk --image photo.jpg "What should I notice?"`,
then discuss the image naturally. To speak one prompt and exit after its response, use
`--talk --single "What should I notice?"`. To require a spoken wake word at the beginning of each new request, run
`cargo run -- --talk --wake-word Computer` (or `-w Jarvis`). For example, “Computer, what time is it?”
triggers a reply; “What time is it, Computer?” does not. Matching ignores case, and the wake word is
removed from the request. Microphone audio still goes to OpenAI for transcription; non-matching turns
are removed from the realtime conversation. Speaking a new wake-word request during a reply stops
playback and replaces that reply once the new request has been transcribed; unlike ordinary talk
mode, it cannot interrupt at the instant you start speaking. Microphone audio continues to be sent
for transcription while the agent is speaking. An initial typed query still gets an
immediate response.
In an interactive text chat, drag an image from Finder into the
prompt, optionally add a question, and press Enter. This works on any turn, including resumed
chats. PNG, JPEG, GIF, and WebP files are supported. The selected model must support vision.

Refresh cached LiteLLM pricing data:

```sh
cargo run -- --update-pricing
```

Agent state is stored under `$HOME/.agent`.
Pricing data is cached under `$HOME/.agent/pricing`.
User-level instructions are read from `$HOME/.agent/AGENTS.md`.

## Finding saved conversations

Use `/find browser playwright timeout` to search saved user and assistant messages.
Results rank by the number of distinct query words found anywhere in a conversation,
then by how many occur together in one message, then by most recently saved.
Partial matches are included; repeated words do not increase a result's score.
Matching is case-insensitive, splits filename components, and ignores common query
stopwords. It does not infer synonyms or correct typos.

Each result shows an excerpt around a match in its best-matching message and a
`/resume <session_id>` command. Search excludes the current session, tool output,
system messages, and compaction archives, and returns at most ten conversations.
It runs locally without model calls or an index. `/resume` completion remains a
quick picker based on opening-message previews.

## Terminal output

During a running text turn, type a note and press Enter to steer the agent's next model request. Notes are saved in the session; they do not interrupt an in-flight request or tool. Unsubmitted text remains in the next prompt.


Live output uses normal terminal scrollback. Running tools appear as compact,
transient previews above the input footer; only completed panels enter scrollback.
Previews are shortened to fit (at most three rows total, with an overflow count)
and hidden when the window is too small. Completed commands/results are not
shortened to a 30-line display preview (the tool layer's size limits still apply).
Long input wraps by display width and keeps the editing cursor visible.
Piped output and `AGENT_NO_LIVE=1` omit Running previews entirely.

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
