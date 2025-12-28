import pytest


@pytest.mark.parametrize(
    "messages, expected_remaining",
    [
        (["abcd"], 16383),  # 4 chars -> 1 token (//4)
        (["abcdefgh"], 16382),  # 8 chars -> 2 tokens
    ],
)
def test_format_cost_and_context_line_updates_with_messages(messages, expected_remaining):
    from main import AgentState, DEFAULT_MAX_CONTEXT_TOKENS, _format_cost_and_context_line, SimpleTokenCounter

    # Build a minimal state with simple messages to exercise token counting.
    from langchain_core.messages import HumanMessage

    state = AgentState(messages=[HumanMessage(content=m) for m in messages])
    token_counter = SimpleTokenCounter()

    line = _format_cost_and_context_line(
        state=state,
        token_counter=token_counter,
        max_context_tokens=DEFAULT_MAX_CONTEXT_TOKENS,
        model_name="dummy",
    )

    assert f"({expected_remaining:,}/{DEFAULT_MAX_CONTEXT_TOKENS:,} tokens)" in line
    assert "Context" in line
