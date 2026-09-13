# Interactive terminal testing

Run from the repository root with `cargo run -- ...`, not an installed `agent`.
Use a temporary `HOME` to avoid modifying real sessions/history; retain the real
`CARGO_HOME` and `RUSTUP_HOME`. Disable completion audio.

## Offline smoke

1. Start `cargo run -- --model mock --no-completion-sound` in a PTY with an
   explicit window size (for example 80 columns × 24 rows).
2. Feed the PTY output into a terminal emulator. Answer cursor-position requests
   (`ESC [ 6 n`) using the emulator's actual cursor, not a constant coordinate.
3. Type `run echo hi`, then Enter. Confirm the running tool panel, completed
   panel, and `Tool completed: hi` appear in order. Confirm a clean new prompt.
4. Repeat a turn; exit with Ctrl-C. Check cursor visibility and terminal modes.
5. Repeat with `AGENT_NO_LIVE=1` and with `--single`.

## Streaming / resize / type-ahead regression

Use a local OpenAI-compatible SSE server (no external provider required), with
`OLLAMA_URL=http://127.0.0.1:<port>` and `--model ollama:test`. Send chat completion
chunks containing sequential numbered lines with a short delay between chunks.

- Stream at least 120 numbered lines; also test long CJK/emoji lines that wrap.
- While streaming, type a long draft; paste multiple lines; move the cursor near
  the start/end and shrink the draft. The input viewport must retain the cursor.
- Resize 80×24 → 40×12 → 100×35. Also test a small window and restore it.
- Check **screen plus scrollback**, not just the raw PTY bytes: all numbered
  output must remain, with no spinner/status/input remnants in the transcript.
- After completion, the draft should be transferred intact to the next prompt.
- Abort a streaming turn using Escape twice; check footer cleanup and that the
  next prompt remains usable.
- Return a tool call whose command/result exceeds one screen. Confirm results
  are appended without erasing previous output and lines beyond 30 are visible.

An xterm-compatible emulator such as `@xterm/headless` plus `node-pty` can drive
this procedure without showing a browser. Install these only in a temporary test
directory. Resize after queued terminal writes have been parsed. Respect
synchronized-output frames (`CSI ? 2026 h/l`) when simulating a visible frame;
resizing between arbitrary transport chunks can intentionally interrupt a draw.

Automated screen-state regressions live in `tests/display_contract.rs` and use
`vt100`. They cover scrollback, partial stream chunks/pending wrap, cursor-based
resizing, Unicode input layout, tiny windows, footer cleanup and long results.
They complement rather than replace native terminal/reflow checks.
