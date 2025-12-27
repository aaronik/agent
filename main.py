import argparse
import os
import signal
import sys
from dataclasses import dataclass, field
from typing import List

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage, trim_messages

from src.constants import system_string
from src.markdown_render import print_markdown
from src.util import TokenUsage, preload_litellm_cost_map, sys_git_ls, sys_pwd, sys_uname

HUMAN = ""
ANSWER_HEADER = "\n"


# Default model
MODEL = "gpt-5.2"
MAX_CONTEXT_TOKENS = 150000


def _render_new_output_message(message: BaseMessage | None) -> None:
    if message is None:
        print(ANSWER_HEADER, None)
        return

    # Match existing UX: we print dialogue (user + assistant) but skip system/tool chatter.
    if isinstance(message, HumanMessage):
        # Render similar to the prompt but as part of transcript replay.
        bold_open = "\033[1m"
        bold_close = "\033[0m"
        print(f"\n{bold_open}{message.content}{bold_close}")
        print()
        return

    if isinstance(message, AIMessage):
        # Render assistant output as Markdown (bullets, bold, code fences, etc.)
        print(ANSWER_HEADER, end="")
        print_markdown(str(message.content))


class SimpleTokenCounter:
    """Simple token counter for Ollama models that approximates tokens as characters / 4"""

    def get_num_tokens_from_messages(self, messages: List[BaseMessage]) -> int:
        """Approximate token count using character length / 4"""
        return sum(len(str(msg.content)) // 4 for msg in messages)


@dataclass
class AgentState:
    messages: List[BaseMessage] = field(default_factory=list)
    token_usage: "TokenUsage | None" = None


def _build_agent_and_deps(*, model_override: str | None = None, list_models: bool = False):
    """Create the model, tools, agent, token counter, and token usage.

    Important: this is intentionally done lazily (at runtime) so importing
    `main` stays fast and cheap.
    """

    # Local import to avoid import-time cost for most Python executions.
    from langchain.agents import create_agent
    from langchain_openai import ChatOpenAI

    # Optional: list models and exit early.
    if list_models:
        # OpenAI models
        from openai import OpenAI

        openai_client = OpenAI()
        openai_models = openai_client.models.list()
        openai_ids = [m.id for m in openai_models.data]

        from src.model_registry import list_openai_chat_models

        openai_chat_models = list_openai_chat_models(openai_ids)

        # Ollama models (best-effort). We support the native Ollama endpoint.
        ollama_models: list[str] = []
        ollama_url = os.getenv("OLLAMA_URL") or "http://localhost:11434"  # default Ollama
        if ollama_url:
            import requests

            try:
                resp = requests.get(ollama_url.rstrip("/") + "/api/tags", timeout=5)
                resp.raise_for_status()
                data = resp.json()
                for m in (data.get("models") or []):
                    name = m.get("name")
                    if name:
                        ollama_models.append(str(name))
            except Exception:
                ollama_models = []

        # Print one model id per line for easy piping.
        for mi in openai_chat_models:
            print(f"openai:{mi.id}")
        for mid in sorted(set(ollama_models)):
            print(f"ollama:{mid}")

        raise SystemExit(0)

    # If the user set Ollama configuration, prefer that local model.
    ollama_url = os.getenv("OLLAMA_URL") or "http://localhost:11434"
    ollama_model = os.getenv("OLLAMA_MODEL")

    if ollama_url and ollama_model:
        model_name = ollama_model
        ollama_base_url = ollama_url.rstrip("/") + "/v1"
        model = ChatOpenAI(model=model_name, base_url=ollama_base_url, api_key="dummy")
        token_counter = SimpleTokenCounter()
    else:
        from src.provider_registry import parse_model_id

        raw_model = model_override or os.getenv("AGENT_MODEL") or os.getenv("OPENAI_MODEL") or MODEL
        pm = parse_model_id(raw_model)
        if pm.provider == "openai":
            model_name = pm.model
            model = ChatOpenAI(model=model_name)
        else:
            # For now, non-OpenAI providers are supported via an OpenAI-compatible
            # base_url + model name. Ollama is the primary example.
            if pm.provider == "ollama":
                model_name = pm.model
                ollama_base_url = ollama_url.rstrip("/") + "/v1"
                model = ChatOpenAI(model=model_name, base_url=ollama_base_url, api_key="dummy")
                token_counter = SimpleTokenCounter()
            else:
                raise ValueError(f"Unknown provider '{pm.provider}'. Use --list-models to see options.")

    # Local import: tools module pulls in prompt_toolkit / openai deps.
    import src.tools as tools_module

    tool_list = [
        tools_module.fetch,
        tools_module.read_file,
        tools_module.write_file,
        tools_module.search_replace,
        tools_module.gen_image,
        tools_module.run_shell_command,
        tools_module.communicate,
        tools_module.spawn,
    ]

    agent = create_agent(model, tools=tool_list)

    token_usage = TokenUsage(model=model.model_name)

    return agent, token_counter, token_usage, model_name


def _build_prompt_session():
    # Local import to avoid prompt_toolkit import cost unless we actually run the CLI.
    from prompt_toolkit import PromptSession
    from prompt_toolkit.enums import EditingMode

    return PromptSession(editing_mode=EditingMode.VI)


def _prompt_boxed(session) -> str:
    # Add vertical breathing room so the prompt isn't flush against prior panels.
    # Kept small and theme-independent (just blank lines).
    """Prompt for input using prompt_toolkit with a simple ASCII "box" prefix.

    We intentionally avoid background-color tricks here because they can become
    invisible depending on terminal theme. This stays readable everywhere.

    Visual:
        [ > your input…
    """

    from prompt_toolkit.formatted_text import FormattedText
    from prompt_toolkit.styles import Style

    style = Style.from_dict(
        {
            # Subtle grey for the prompt prefix.
            "prompt.box": "fg:ansibrightblack bold",
            "": "bold",
        }
    )

    # Leading newlines act like a top margin for the input area.
    prompt = FormattedText(
        [
            ("", "\n"),
            ("class:prompt.box", "[ > "),
        ]
    )

    text = session.prompt(prompt, style=style)
    # Small bottom margin so the next rendered output doesn't collide visually.
    print()
    return text


def _build_display():
    # Local import to keep import-time cheap.
    from src.tool_status_display import get_tool_status_display

    return get_tool_status_display()


def _install_signal_handler(graceful_exit, cancel_current_turn):
    def signal_handler(*_):
        # Dual behavior:
        # - If a turn is running, SIGINT cancels that turn *and* interrupts any
        #   in-flight blocking work (model call / tool) by raising KeyboardInterrupt.
        # - If we're idle at the prompt, SIGINT exits the app.
        if cancel_current_turn():
            raise KeyboardInterrupt
        graceful_exit()

    signal.signal(signal.SIGINT, signal_handler)


def main(argv: list[str] | None = None) -> int:
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Personal Command Line Agent")
    parser.add_argument(
        "--model",
        "-m",
        default=None,
        help="Model id to use. Supports 'openai:<id>' or 'ollama:<id>'. Without a prefix defaults to OpenAI.",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available models (OpenAI + any configured local providers) and exit.",
    )
    parser.add_argument(
        "--single",
        "-s",
        action="store_true",
        help="Run single invocation without prompting for more input",
    )
    parser.add_argument(
        "--resume",
        nargs="?",
        const="__LATEST__",
        default=None,
        metavar="ID",
        help="Resume a saved session by id. If no id is provided, resumes the latest session.",
    )
    parser.add_argument("query", nargs="*", help="Query to process")

    args = parser.parse_args(argv)

    agent, token_counter, token_usage, model_name = _build_agent_and_deps(
        model_override=args.model,
        list_models=args.list_models,
    )
    display = _build_display()
    session = _build_prompt_session()

    # Preload LiteLLM's pricing table in the background while the user is typing.
    preload_litellm_cost_map()

    autosaver = None

    def graceful_exit() -> None:
        """Clean up UI and print token usage before exiting."""
        try:
            display.clear()
        except Exception:
            # Fallback: try obtaining a fresh display instance
            try:
                from src.tool_status_display import get_tool_status_display

                get_tool_status_display().clear()
            except Exception:
                pass
        if autosaver is not None:
            autosaver.close()
        print("Goodbye!")
        raise SystemExit(0)

    from src.cancel import CancelToken

    turn_in_progress = False
    cancel_token: CancelToken | None = None
    sigint_during_turn = False

    def cancel_current_turn() -> bool:
        nonlocal cancel_token, sigint_during_turn
        if turn_in_progress and cancel_token is not None:
            cancel_token.cancel("SIGINT")
            sigint_during_turn = True
            return True
        return False

    _install_signal_handler(graceful_exit, cancel_current_turn)

    # Print a tall robot and the model name
    print("\n𜱜\n𜱟 " + model_name)

    # Discover CLAUDE.md memory files and print line counts for each
    from src import claude_memory

    memory_files = claude_memory.find_all_claude_md_files()
    if memory_files:
        print("\nMemory files found:")
        for mf in memory_files:
            try:
                with open(mf, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                print(f"- {mf}: {len(lines)} lines")
            except Exception:
                print(f"- {mf}: (could not read)")
    else:
        print("\nNo CLAUDE.md memory files found")

    # If resuming, load and replay the session *before* collecting new input.
    # This makes the CLI feel continuous (history appears immediately).

    initial_user_input = " ".join(args.query) if args.query else None

    # Initialize or resume state

    session_id = None

    def _autosave() -> None:
        if autosaver is None:
            return
        autosaver.request_save(state.messages)

    if args.resume is not None and not args.single:
        from src.session_store import SessionAutosaver, load_messages

        resume_id = None if args.resume == "__LATEST__" else args.resume
        session_id, loaded_messages = load_messages(resume_id)
        state = AgentState(messages=loaded_messages, token_usage=token_usage)
        autosaver = SessionAutosaver(session_id=session_id)

        # Replay prior assistant outputs so the scrollback matches a continuous session.
        for msg in state.messages:
            _render_new_output_message(msg)

        # Now collect the next user input.
        if initial_user_input is not None:
            user_input = initial_user_input
        else:
            try:
                user_input = _prompt_boxed(session)
            except (KeyboardInterrupt, SystemExit):
                graceful_exit()

        state.messages.append(HumanMessage(content=user_input))
        _autosave()
    else:
        # Initial user input from command line or prompt
        if initial_user_input is not None:
            user_input = initial_user_input
        else:
            try:
                user_input = _prompt_boxed(session)
            except (KeyboardInterrupt, SystemExit):
                graceful_exit()

        state = AgentState(
            messages=[
                SystemMessage(content=system_string),
                SystemMessage(content=f"[SYSTEM INFO] uname -a: {sys_uname()}"),
                SystemMessage(content=f"[SYSTEM INFO] pwd: {sys_pwd()}"),
                *(
                    [SystemMessage(content=f"[SYSTEM INFO] git ls-files: {git_ls}")]
                    if (git_ls := sys_git_ls())
                    else []
                ),
                HumanMessage(content=user_input),
            ],
            token_usage=token_usage,
        )

        if not args.single:
            from src.session_store import SessionAutosaver, new_session_id

            session_id = new_session_id()
            autosaver = SessionAutosaver(session_id=session_id)
            _autosave()


    from src.agent_runner import run_agent_with_display

    while True:
        trimmed_messages = trim_messages(
            state.messages,
            max_tokens=MAX_CONTEXT_TOKENS,
            strategy="last",
            token_counter=token_counter,
            include_system=True,
            start_on="human",
            allow_partial=False,
        )

        trimmed_state = AgentState(messages=trimmed_messages, token_usage=token_usage)

        cancel_token = CancelToken()
        turn_in_progress = True
        sigint_during_turn = False
        try:
            try:
                try:
                    new_messages = run_agent_with_display(agent, trimmed_state, cancel_token=cancel_token)
                except TypeError:
                    # Backwards-compatible for tests/mocks that haven't been updated.
                    new_messages = run_agent_with_display(agent, trimmed_state)
            except KeyboardInterrupt:
                # If SIGINT happened during a turn, treat it as a cancellation and
                # return to the prompt (don't exit the app).
                if sigint_during_turn:
                    new_messages = []
                else:
                    raise
        finally:
            turn_in_progress = False
            cancel_token = None

        state = AgentState(messages=state.messages + new_messages, token_usage=token_usage)
        _autosave()

        output = state.messages[-1] if state.messages else None

        _render_new_output_message(output)

        # Token usage: the bottom box is the single source of truth.
        token_usage.ingest_from_messages(state.messages)
        print()
        token_usage.print_panel()

        if args.single:
            break

        try:
            user_input = _prompt_boxed(session)
        except (KeyboardInterrupt, SystemExit):
            graceful_exit()

        state.messages.append(HumanMessage(content=user_input))
        _autosave()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
