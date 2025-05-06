# AI CLI Tool

A command-line tool to automate arbitrary flows using AI (OpenAI API).

## Setup

1. Create and activate a Python virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
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

## Running the CLI

```bash
python main.py [your arguments]
```

---

## Features (Planned)
- Automate arbitrary command-line flows with natural language
- Chain shell commands together based on AI instructions
- Integrate with OpenAI GPT models
