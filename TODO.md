# TODO

## High priority

- [ ] Add a real security boundary and permission model.
  - Scope filesystem access to an approved workspace.
  - Support allow/deny path rules and per-action approval.
  - Offer sandboxed/containerized shell execution.

- [ ] Protect secrets and sensitive user data.
  - Block or require explicit opt-in for sensitive paths such as `~/.ssh`, credential files, and browser-profile data.
  - Redact secrets from tool output, logs, and model context.

- [x] Stream provider events directly to the UI.
  - Render text and tool-call progress as events arrive rather than accumulating an entire `Vec` before display.

- [ ] Add automatic context compaction and durable memory.
  - Compact before context overflow.
  - Persist useful summaries and retrieve relevant prior work automatically.

- [ ] Add task/session budgets and runaway protection.
  - Enforce configurable limits for wall-clock time, tool calls, tokens, and cost.

## Reliability

- [ ] Improve provider resilience.
  - Add request timeouts, retries with backoff, rate-limit handling, and optional provider/model failover.
  - Detect provider/model capabilities instead of relying only on hard-coded context-window heuristics.

- [ ] Add structured observability.
  - Persist run traces with request IDs, timings, tool inputs/results, usage, failures, and retry history.

## Tooling and configuration

- [ ] Introduce a tool policy/capability model.
  - Expose only needed tool groups per project or mode.
  - Add argument-level constraints and confirmation for destructive actions.

- [ ] Add versioned project/runtime configuration.
  - Configure providers, models, tool policy, workspace, budgets, and defaults through a config file as well as environment variables.

- [ ] Add a safer delivery workflow for code changes.
  - Support plan → approve → execute.
  - Add patch review, rollback, and optional git checkpoints.
