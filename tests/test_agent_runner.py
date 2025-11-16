import unittest
from unittest.mock import Mock, MagicMock, patch
from src.agent_runner import (
    extract_result_preview,
    process_agent_chunk,
    process_tools_chunk,
    run_agent_with_display
)
from src.tool_status_display import ToolStatus
from langchain_core.messages import AIMessage, ToolMessage


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
        # Control characters should be removed
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
    """Integration tests for process_agent_chunk"""

    def setUp(self):
        self.display = Mock()
        self.tool_call_ids_seen = set()
        self.communicate_calls = set()

    def test_process_aimessage_with_normal_tool_calls(self):
        """Test processing AIMessage with regular tool calls"""
        tool_calls = [
            {"id": "call_1", "name": "read_file", "args": {"path": "test.txt"}},
            {"id": "call_2", "name": "write_file", "args": {"path": "out.txt", "content": "test"}}
        ]
        msg = AIMessage(content="", tool_calls=tool_calls)

        process_agent_chunk([msg], self.tool_call_ids_seen, self.display, self.communicate_calls)

        # Verify tool calls were registered
        self.display.register_calls.assert_called_once()
        registered_calls = self.display.register_calls.call_args[0][0]
        self.assertEqual(len(registered_calls), 2)
        self.assertEqual(registered_calls[0]["id"], "call_1")
        self.assertEqual(registered_calls[1]["id"], "call_2")

        # Verify status updates to RUNNING
        self.assertEqual(self.display.update_status.call_count, 2)

        # Verify IDs tracked
        self.assertIn("call_1", self.tool_call_ids_seen)
        self.assertIn("call_2", self.tool_call_ids_seen)

        # Verify communicate_calls is empty
        self.assertEqual(len(self.communicate_calls), 0)

    def test_process_aimessage_with_communicate_calls(self):
        """Test that communicate calls are tracked separately"""
        tool_calls = [
            {"id": "call_1", "name": "communicate", "args": {"message": "Hello"}},
            {"id": "call_2", "name": "read_file", "args": {"path": "test.txt"}}
        ]
        msg = AIMessage(content="", tool_calls=tool_calls)

        process_agent_chunk([msg], self.tool_call_ids_seen, self.display, self.communicate_calls)

        # Only non-communicate call should be registered
        self.display.register_calls.assert_called_once()
        registered_calls = self.display.register_calls.call_args[0][0]
        self.assertEqual(len(registered_calls), 1)
        self.assertEqual(registered_calls[0]["id"], "call_2")

        # Communicate call should be tracked separately
        self.assertIn("call_1", self.communicate_calls)
        self.assertIn("call_1", self.tool_call_ids_seen)
        self.assertIn("call_2", self.tool_call_ids_seen)

    def test_process_aimessage_skips_duplicate_ids(self):
        """Test that duplicate tool call IDs are skipped"""
        tool_calls = [
            {"id": "call_1", "name": "read_file", "args": {"path": "test.txt"}}
        ]
        msg = AIMessage(content="", tool_calls=tool_calls)

        # Add call_1 to seen IDs
        self.tool_call_ids_seen.add("call_1")

        process_agent_chunk([msg], self.tool_call_ids_seen, self.display, self.communicate_calls)

        # Should not register any calls
        self.display.register_calls.assert_not_called()

    def test_process_toolmessage_updates_display(self):
        """Test processing ToolMessage updates display with result"""
        msg = ToolMessage(
            content="File content here\nLine 2\nLine 3",
            tool_call_id="call_1"
        )

        process_agent_chunk([msg], self.tool_call_ids_seen, self.display, self.communicate_calls)

        # Verify update_status called with preview
        self.display.update_status.assert_called_once()
        call_args = self.display.update_status.call_args[0]
        self.assertEqual(call_args[0], "call_1")
        self.assertEqual(call_args[1], ToolStatus.DONE)
        # Preview should be extracted
        self.assertIn("File content here", call_args[2])

    def test_process_toolmessage_skips_communicate_results(self):
        """Test that ToolMessage from communicate calls is skipped"""
        msg = ToolMessage(content="Communication message", tool_call_id="call_comm")
        self.communicate_calls.add("call_comm")

        process_agent_chunk([msg], self.tool_call_ids_seen, self.display, self.communicate_calls)

        # Should not update display for communicate results
        self.display.update_status.assert_not_called()


class TestProcessToolsChunk(unittest.TestCase):
    """Integration tests for process_tools_chunk"""

    def setUp(self):
        self.display = Mock()
        self.communicate_calls = set()

    def test_marks_pending_tools_as_running(self):
        """Test that pending tools are marked as running"""
        # Setup display with pending tool
        pending_tool = Mock()
        pending_tool.status = ToolStatus.PENDING

        self.display.display_sequence = [
            {
                "type": "table",
                "tool_calls": {
                    "call_1": pending_tool
                }
            }
        ]

        process_tools_chunk([], self.display, self.communicate_calls)

        # Verify status update
        self.display.update_status.assert_called_once_with("call_1", ToolStatus.RUNNING)

    def test_processes_tool_results(self):
        """Test processing regular tool results"""
        msg = ToolMessage(
            content="Result content\nSecond line",
            tool_call_id="call_1"
        )

        self.display.display_sequence = []

        process_tools_chunk([msg], self.display, self.communicate_calls)

        # Verify update_status called with result
        self.display.update_status.assert_called()
        call_args = self.display.update_status.call_args[0]
        self.assertEqual(call_args[0], "call_1")
        self.assertEqual(call_args[1], ToolStatus.DONE)
        self.assertIn("Result content", call_args[2])

    def test_handles_communicate_results_specially(self):
        """Test that communicate tool results go through add_communication"""
        msg = ToolMessage(
            content="Communication message",
            tool_call_id="call_comm"
        )
        self.communicate_calls.add("call_comm")
        self.display.display_sequence = []

        process_tools_chunk([msg], self.display, self.communicate_calls)

        # Should call add_communication instead of update_status
        self.display.add_communication.assert_called_once_with("Communication message")
        # Should NOT call update_status
        self.display.update_status.assert_not_called()


class TestRunAgentWithDisplay(unittest.TestCase):
    """Integration tests for run_agent_with_display"""

    @patch('src.agent_runner.get_tool_status_display')
    def test_full_agent_execution_flow(self, mock_get_display):
        """Test complete agent execution with display integration"""
        # Setup mock display
        mock_display = Mock()
        mock_display.display_sequence = []  # Initialize as empty list
        mock_get_display.return_value = mock_display

        # Setup mock agent
        mock_agent = Mock()

        # Simulate streaming chunks
        tool_call = {"id": "call_1", "name": "read_file", "args": {"path": "test.txt"}}
        ai_msg = AIMessage(content="", tool_calls=[tool_call])
        tool_result = ToolMessage(content="File contents", tool_call_id="call_1")

        mock_agent.stream.return_value = [
            {"agent": {"messages": [ai_msg]}},
            {"tools": {"messages": [tool_result]}}
        ]

        # Run agent
        result = run_agent_with_display(mock_agent, {"messages": []})

        # Verify display lifecycle
        mock_display.clear.assert_called_once()
        mock_display.register_calls.assert_called_once()
        mock_display.finalize.assert_called_once()

        # Verify messages returned
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0], ai_msg)
        self.assertEqual(result[1], tool_result)

    @patch('src.agent_runner.get_tool_status_display')
    def test_handles_communicate_calls_separately(self, mock_get_display):
        """Test that communicate calls are handled separately throughout flow"""
        mock_display = Mock()
        mock_get_display.return_value = mock_display
        mock_display.display_sequence = []

        mock_agent = Mock()

        # Simulate communicate call
        comm_call = {"id": "call_comm", "name": "communicate", "args": {"message": "Hello"}}
        ai_msg = AIMessage(content="", tool_calls=[comm_call])
        comm_result = ToolMessage(content="Communication sent", tool_call_id="call_comm")

        mock_agent.stream.return_value = [
            {"agent": {"messages": [ai_msg]}},
            {"tools": {"messages": [comm_result]}}
        ]

        # Run agent
        run_agent_with_display(mock_agent, {"messages": []})

        # Communicate call should NOT be registered with display
        mock_display.register_calls.assert_not_called()

        # But communication result should be added
        mock_display.add_communication.assert_called_once_with("Communication sent")

    @patch('src.agent_runner.get_tool_status_display')
    def test_respects_recursion_limit(self, mock_get_display):
        """Test that recursion limit is passed to agent.stream"""
        mock_display = Mock()
        mock_display.display_sequence = []
        mock_get_display.return_value = mock_display

        mock_agent = Mock()
        mock_agent.stream.return_value = []

        run_agent_with_display(mock_agent, {"messages": []}, recursion_limit=150)

        # Verify recursion_limit passed as second positional argument
        mock_agent.stream.assert_called_once()
        call_args = mock_agent.stream.call_args[0]
        self.assertEqual(call_args[1]["recursion_limit"], 150)

    @patch('src.agent_runner.get_tool_status_display')
    def test_updates_cost_display_when_token_usage_present(self, mock_get_display):
        """Test that cost display is updated when token_usage is on state"""
        mock_display = Mock()
        mock_display.display_sequence = []
        mock_get_display.return_value = mock_display

        # Create mock token_usage with necessary methods
        mock_token_usage = Mock()
        mock_token_usage.prompt_cost.return_value = 0.001
        mock_token_usage.completion_cost.return_value = 0.002
        mock_token_usage.total_cost.return_value = 0.003

        # Setup state with token_usage
        state = Mock()
        state.token_usage = mock_token_usage

        # Setup mock agent
        mock_agent = Mock()
        tool_call = {"id": "call_1", "name": "read_file", "args": {"path": "test.txt"}}
        ai_msg = AIMessage(content="", tool_calls=[tool_call])
        tool_result = ToolMessage(content="File contents", tool_call_id="call_1")

        mock_agent.stream.return_value = [
            {"agent": {"messages": [ai_msg]}},
            {"tools": {"messages": [tool_result]}}
        ]

        # Run agent
        run_agent_with_display(mock_agent, state)

        # Verify update_cost was called
        self.assertTrue(mock_display.update_cost.called)
        # Should be called at least once (after agent and tools chunks)
        self.assertGreaterEqual(mock_display.update_cost.call_count, 1)


if __name__ == '__main__':
    unittest.main()
