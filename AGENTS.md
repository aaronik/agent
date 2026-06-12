# agent - Personal Command Line Agent

Agent is a highly autonomous AI command line agent.

## Running the CLI in this environment

When the user says “run `agent ...`” in this repo/environment, run the Rust CLI from the `rs/` folder via:

```sh
cd rs && cargo run -- ...
```

For example, `agent --model mock --single "run echo hi"` should be run as:

```sh
cd rs && cargo run -- --model mock --single "run echo hi"
```

## Testing Requirements

**CRITICAL: No job is finished until all new functionality is tested and ALL tests are passing.**

When adding or modifying Rust functionality:
1. Write tests FIRST (TDD approach preferred)
2. Implement the functionality
3. Ensure ALL tests pass:
   ```sh
   cd rs && cargo fmt --check && cargo clippy -- -D warnings && cargo test
   ```
4. Only then is the work considered complete

Test guidelines:
- Prefer high-level integration tests over unit tests
- Mock external dependencies (network calls, subprocesses, etc.) to keep tests fast
- Add tests to existing test files when possible (avoid creating new test files unnecessarily)
- Every bug fix should include a test that would have caught the bug
- Tests are more important than the code itself - they guarantee stability

Current test count: 108 tests
Target test execution time: < 2 seconds for full suite

## Note on Interaction with the User

The user is using MacOS native speech recognition. It is not great quality. So the input often ends up sounding similar to what I mean, but being written like something else. I hope you can interpret what is being said beyond the noise.
