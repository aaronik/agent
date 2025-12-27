from src.agent_runner import extract_result_preview


def test_extract_result_preview_preserves_blank_lines_between_content():
    s = "a\n\n\nb\n"
    preview = extract_result_preview(s, max_lines=10)
    assert preview == "a\n\n\nb"
