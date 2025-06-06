# Agent

A command-line tool to automate arbitrary requests autonomously using AI.
This program gives the AI full control to runs arbitrary code, so don't use
this with AIs you don't totally trust.

Usage: agent "Can we add a feature to this repository whereby such and such does such and such other thing? Include thorough tests, make sure they're all passing, and make sure pyright is giving no type errors."

Agent respects the CLAUDE.md file, CLAUDE.local.md, and the user level CLAUDE files as well.
This makes it easy to use in codebases without committing to doing extra work for this crazy little tool,
that you wouldn't give to your real CLI agent.

## Setup

1. Create and activate a Python virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

You will need an OpenAI API key to use this tool. Set it as an environment variable:

```bash
export OPENAI_API_KEY=your_api_key_here
```

### Installing

I just alias it:

```bash
# ~/.zshrc
alias agent="/<path to the agent folder>/run.sh
```

### Running the CLI

```bash
agent "Let's work on this code together" # <-- starts a loop
agent -s "Research such and such topic" # <-- single agent invocation (including tool uses)
```

---

## Running Tests

To run the tests, use your preferred test runner or execute the following command in the project root directory:

```bash
pytest tests
```

---

## Development
- Pricing data changes over time, and there doesn't seem to be a good API to get at it.
  So periodically we need to scrape good pricing pages and update this repo. To do that:
```sh
./scripts/updating_pricing.sh
```
  It's also possible to use the extract_prices.py script, if their page hasn't changed
  too much since it was vibe coded.

## Notes

* This is not meant for prime time - this is my personal little agent, something I'm hacking on on the side.
I think a big part of becoming good with AI, is building your own AI assistant. Then you see how it's done, and you have full customizability of it. Have it build itself.

* The structure currently is having a bunch of different main_*.py files hanging around top level. Right now this is serving both as this agent that's actually doing a lot, but also multiple ways to make the same agent. It shares tools and some other stuff across the different approaches/libraries
