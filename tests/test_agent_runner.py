import unittest
from unittest.mock import Mock, patch
from src.agent_runner import (
    extract_result_preview,
    process_agent_chunk,
    process_tools_chunk,
    run_agent_with_display,
)
from src.tool_status_display import ToolStatus
from langchain_core.messages import AIMessage, ToolMessage, SystemMessage, HumanMessage, trim_messages


class TestExtractResultPreview(unittest.TestCase):
    def test_extract_single_line(self):
        content = "Single line result"
        result = extract_result_preview(content)
        self.assertEqual(result, "Single line result")

    def test_extract_multiline(self):
        content = "Line 1\nLine 2\nLine 3\nLine 4\nLine 5"
        result = extract_result_preview(content, max_lines=3)
        self.assertEqual(result, "Line 1\nLine 2\nLine 3")

    def test_extract_with_empty_lines(self):
        content = "\nLine 1\n\nLine 2\n\nLine 3\n"
        result = extract_result_preview(content, max_lines=3)
        self.assertEqual(result, "Line 1\nLine 2\nLine 3")

    def test_extract_long_line_truncation(self):
        long_line = "x" * 100
        result = extract_result_preview(long_line, max_line_length=80)
        self.assertEqual(len(result), 80)
        self.assertTrue(result.endswith("..."))

    def test_extract_empty_content(self):
        result = extract_result_preview("")
        self.assertEqual(result, "")

    def test_extract_non_string_content(self):
        result = extract_result_preview(None)
        self.assertEqual(result, "")

        result = extract_result_preview(123)
        self.assertEqual(result, "")

    def test_extract_with_control_characters(self):
        content = "Line with \x00control\x01chars"
        result = extract_result_preview(content)
        self.assertNotIn("\x00", result)
        self.assertNotIn("\x01", result)

    def test_extract_strips_whitespace(self):
        content = "  Line 1  \n  Line 2  \n  Line 3  "
        result = extract_result_preview(content, max_lines=3)
        self.assertEqual(result, "Line 1\nLine 2\nLine 3")

    def test_extract_with_mixed_empty_and_content_lines(self):
        content = "\n\nLine 1\nLine 2\n\n\nLine 3\n\n\n"
        result = extract_result_preview(content, max_lines=2)
        self.assertEqual(result, "Line 1\nLine 2")


class TestProcessAgentChunk(unittest.TestCase):
    def setUp(self):
        self.display = Mock()
        self.tool_call_ids_seen = set()

    def test_process_aimessage_with_tool_calls(self):
        tool_calls = [
            {"id": "call_1", "name": "read_file", "args": {"path": "test.txt"}},
            {"id": "call_2", "name": "write_file", "args": {"path": "out.txt", "content": "test"}},
        ]
        msg = AIMessage(content="", tool_calls=tool_calls)

        process_agent_chunk([msg], self.tool_call_ids_seen, self.display)

        self.display.register_calls.assert_called_once()
        registered_calls = self.display.register_calls.call_args[0][0]
        self.assertEqual(len(registered_calls), 2)
        self.assertEqual(registered_calls[0]["id"], "call_1")
        self.assertEqual(registered_calls[1]["id"], "call_2")

        self.assertEqual(self.display.update_status.call_count, 2)
        self.assertIn("call_1", self.tool_call_ids_seen)
        self.assertIn("call_2", self.tool_call_ids_seen)

    def test_process_aimessage_skips_duplicate_ids(self):
        tool_calls = [{"id": "call_1", "name": "read_file", "args": {"path": "test.txt"}}]
        msg = AIMessage(content="", tool_calls=tool_calls)

        self.tool_call_ids_seen.add("call_1")

        process_agent_chunk([msg], self.tool_call_ids_seen, self.display)

        self.display.register_calls.assert_not_called()

    def test_process_toolmessage_updates_display(self):
        msg = ToolMessage(content="File content here\nLine 2\nLine 3", tool_call_id="call_1")

        process_agent_chunk([msg], self.tool_call_ids_seen, self.display)

        self.display.update_status.assert_called_once()
        call_args = self.display.update_status.call_args[0]
        self.assertEqual(call_args[0], "call_1")
        self.assertEqual(call_args[1], ToolStatus.DONE)
        self.assertIn("File content here", call_args[2])


class TestProcessToolsChunk(unittest.TestCase):
    def setUp(self):
        self.display = Mock()

    def test_marks_pending_tools_as_running(self):
        pending_tool = Mock()
        pending_tool.status = ToolStatus.PENDING

        self.display.display_sequence = [
            {
                "type": "tools",
                "tool_calls": {
                    "call_1": pending_tool,
                },
            }
        ]

        process_tools_chunk([], self.display)

        self.display.update_status.assert_called_once_with("call_1", ToolStatus.RUNNING)

    def test_processes_tool_results(self):
        msg = ToolMessage(content="Result content\nSecond line", tool_call_id="call_1")

        self.display.display_sequence = []

        process_tools_chunk([msg], self.display)

        self.display.update_status.assert_called()
        call_args = self.display.update_status.call_args[0]
        self.assertEqual(call_args[0], "call_1")
        self.assertEqual(call_args[1], ToolStatus.DONE)
        self.assertIn("Result content", call_args[2])


class TestRunAgentWithDisplay(unittest.TestCase):
    @patch("src.agent_runner.get_tool_status_display")
    def test_full_agent_execution_flow(self, mock_get_display):
        mock_display = Mock()
        mock_display.display_sequence = []
        mock_get_display.return_value = mock_display

        mock_agent = Mock()

        tool_call = {"id": "call_1", "name": "read_file", "args": {"path": "test.txt"}}
        ai_msg = AIMessage(content="", tool_calls=[tool_call])
        tool_result = ToolMessage(content="File contents", tool_call_id="call_1")

        mock_agent.stream.return_value = [
            {"model": {"messages": [ai_msg]}},
            {"tools": {"messages": [tool_result]}},
        ]

        result = run_agent_with_display(mock_agent, {"messages": []})

        mock_display.clear.assert_called_once()
        mock_display.register_calls.assert_called_once()
        mock_display.finalize.assert_called_once()

        self.assertEqual(len(result), 2)
        self.assertEqual(result[0], ai_msg)
        self.assertEqual(result[1], tool_result)

    @patch("src.agent_runner.get_tool_status_display")
    def test_respects_recursion_limit(self, mock_get_display):
        mock_display = Mock()
        mock_display.display_sequence = []
        mock_get_display.return_value = mock_display

        mock_agent = Mock()
        mock_agent.stream.return_value = []

        run_agent_with_display(mock_agent, {"messages": []}, recursion_limit=150)

        mock_agent.stream.assert_called_once()
        call_args = mock_agent.stream.call_args[0]
        self.assertEqual(call_args[1]["recursion_limit"], 150)

    @patch("src.agent_runner.get_tool_status_display")
    def test_handles_model_chunk_key_from_new_agent_api(self, mock_get_display):
        mock_display = Mock()
        mock_display.display_sequence = []
        mock_get_display.return_value = mock_display

        mock_agent = Mock()

        tool_call = {"id": "call_1", "name": "read_file", "args": {"path": "test.txt"}}
        ai_msg = AIMessage(content="Reading file", tool_calls=[tool_call])
        tool_result = ToolMessage(content="File contents", tool_call_id="call_1")
        final_msg = AIMessage(content="Here are the contents")

        mock_agent.stream.return_value = [
            {"model": {"messages": [ai_msg]}},
            {"tools": {"messages": [tool_result]}},
            {"model": {"messages": [final_msg]}},
        ]

        result = run_agent_with_display(mock_agent, {"messages": []})

        self.assertEqual(len(result), 3)
        self.assertEqual(result[0], ai_msg)
        self.assertEqual(result[1], tool_result)
        self.assertEqual(result[2], final_msg)

        mock_display.clear.assert_called_once()
        mock_display.register_calls.assert_called_once()
        mock_display.finalize.assert_called_once()


class TestMessageTrimming(unittest.TestCase):
    def test_trim_messages_preserves_system_messages(self):
        messages = [
            SystemMessage(content="System context"),
            HumanMessage(content="Question 1"),
            AIMessage(content="Answer 1"),
            HumanMessage(content="Question 2"),
            AIMessage(content="Answer 2"),
        ]

        model = Mock()
        model.get_num_tokens_from_messages = Mock(side_effect=lambda msgs: len(str(msgs)))

        trimmed = trim_messages(
            messages,
            max_tokens=50,
            strategy="last",
            token_counter=model,
            include_system=True,
            start_on="human",
            allow_partial=False,
        )

        self.assertTrue(any(isinstance(msg, SystemMessage) for msg in trimmed))
        self.assertLessEqual(len(trimmed), len(messages))


if __name__ == "__main__":
    unittest.main()
