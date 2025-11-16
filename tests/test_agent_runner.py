import unittest
from unittest.mock import Mock, MagicMock, patch
from src.agent_runner import (
    extract_result_preview,
    process_agent_chunk,
    process_tools_chunk,
    run_agent_with_display
)
from src.tool_status_display import ToolStatus
from langchain_core.messages import AIMessage, ToolMessage, SystemMessage, HumanMessage, trim_messages
from langchain_openai import ChatOpenAI


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
            {"model": {"messages": [ai_msg]}},
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
            {"model": {"messages": [ai_msg]}},
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
    def test_handles_model_chunk_key_from_new_agent_api(self, mock_get_display):
        """Test that agent runner correctly processes 'model' chunk key (not 'agent')

        This test captures a regression where the agent runner was checking for
        'agent' chunk keys, but the new LangChain create_agent API returns 'model'
        chunk keys instead. This caused the display and token tracking to break.
        """
        mock_display = Mock()
        mock_display.display_sequence = []
        mock_get_display.return_value = mock_display

        mock_agent = Mock()

        # Simulate chunks with 'model' key (not 'agent')
        tool_call = {"id": "call_1", "name": "read_file", "args": {"path": "test.txt"}}
        ai_msg = AIMessage(content="Reading file", tool_calls=[tool_call])
        tool_result = ToolMessage(content="File contents", tool_call_id="call_1")
        final_msg = AIMessage(content="Here are the contents")

        mock_agent.stream.return_value = [
            {"model": {"messages": [ai_msg]}},  # First model call with tool decision
            {"tools": {"messages": [tool_result]}},  # Tool execution
            {"model": {"messages": [final_msg]}}  # Final model response
        ]

        # Run agent
        result = run_agent_with_display(mock_agent, {"messages": []})

        # Verify messages were processed correctly
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0], ai_msg)
        self.assertEqual(result[1], tool_result)
        self.assertEqual(result[2], final_msg)

        # Verify display was updated (tool registered and updated)
        mock_display.clear.assert_called_once()
        mock_display.register_calls.assert_called_once()
        mock_display.finalize.assert_called_once()

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
        state.messages = []

        # Setup mock agent
        mock_agent = Mock()
        tool_call = {"id": "call_1", "name": "read_file", "args": {"path": "test.txt"}}
        ai_msg = AIMessage(content="", tool_calls=[tool_call])
        tool_result = ToolMessage(content="File contents", tool_call_id="call_1")

        mock_agent.stream.return_value = [
            {"model": {"messages": [ai_msg]}},
            {"tools": {"messages": [tool_result]}}
        ]

        # Run agent
        run_agent_with_display(mock_agent, state)

        # Verify update_cost was called
        self.assertTrue(mock_display.update_cost.called)
        # Should be called at least once (after agent and tools chunks)
        self.assertGreaterEqual(mock_display.update_cost.call_count, 1)


class TestMessageTrimming(unittest.TestCase):
    """Tests for message trimming to prevent token limit errors"""

    def test_trim_messages_preserves_system_messages(self):
        """Test that system messages are preserved when trimming"""
        messages = [
            SystemMessage(content="System context"),
            HumanMessage(content="Question 1"),
            AIMessage(content="Answer 1"),
            HumanMessage(content="Question 2"),
            AIMessage(content="Answer 2"),
        ]

        # Create a mock model for token counting
        model = Mock()
        model.get_num_tokens_from_messages = Mock(side_effect=lambda msgs: len(str(msgs)))

        # Trim to a very small size to force trimming
        trimmed = trim_messages(
            messages,
            max_tokens=50,
            strategy="last",
            token_counter=model,
            include_system=True,
            start_on="human",
            allow_partial=False,
        )

        # System message should always be present
        self.assertTrue(any(isinstance(msg, SystemMessage) for msg in trimmed))
        # Should have fewer messages than original
        self.assertLessEqual(len(trimmed), len(messages))

    def test_trim_messages_with_large_history(self):
        """Test trimming with a large message history"""
        # Create a large message history
        messages = [SystemMessage(content="System context")]
        for i in range(100):
            messages.append(HumanMessage(content=f"Question {i}"))
            messages.append(AIMessage(content=f"Answer {i}" * 100))  # Make answers long

        model = Mock()
        # Simulate realistic token counting (roughly 1 token per 4 chars)
        model.get_num_tokens_from_messages = Mock(
            side_effect=lambda msgs: sum(len(str(m.content)) // 4 for m in msgs)
        )

        # Trim to 1000 tokens
        trimmed = trim_messages(
            messages,
            max_tokens=1000,
            strategy="last",
            token_counter=model,
            include_system=True,
            start_on="human",
            allow_partial=False,
        )

        # Should have significantly fewer messages
        self.assertLess(len(trimmed), len(messages))
        # System message should still be present
        self.assertTrue(any(isinstance(msg, SystemMessage) for msg in trimmed))
        # Should not exceed token limit (approximately)
        trimmed_tokens = sum(len(str(m.content)) // 4 for m in trimmed)
        self.assertLessEqual(trimmed_tokens, 1200)  # Allow some margin

    def test_prevents_token_limit_exceeded_error(self):
        """Integration test: verify trimming prevents token limit errors"""
        # Simulate the scenario that caused the original error
        messages = [
            SystemMessage(content="System prompt here"),
            SystemMessage(content="[SYSTEM INFO] uname -a: Darwin"),
            SystemMessage(content="[SYSTEM INFO] pwd: /Users/test"),
        ]

        # Add many conversation turns
        for i in range(200):
            messages.append(HumanMessage(content=f"User message {i}" * 50))
            messages.append(AIMessage(content=f"AI response {i}" * 100))
            # Simulate tool calls
            messages.append(AIMessage(
                content="",
                tool_calls=[{"id": f"call_{i}", "name": "read_file", "args": {"path": f"file{i}.txt"}}]
            ))
            messages.append(ToolMessage(content=f"File content {i}" * 200, tool_call_id=f"call_{i}"))

        # Create a realistic token counter
        model = Mock()
        model.get_num_tokens_from_messages = Mock(
            side_effect=lambda msgs: sum(len(str(m.content)) // 4 for m in msgs)
        )

        # This would have caused ~800k tokens without trimming
        # Trim to 150k tokens (same as MAX_CONTEXT_TOKENS in main.py)
        trimmed = trim_messages(
            messages,
            max_tokens=150000,
            strategy="last",
            token_counter=model,
            include_system=True,
            start_on="human",
            allow_partial=False,
        )

        # Verify we're well under the limit
        trimmed_tokens = sum(len(str(m.content)) // 4 for m in trimmed)
        self.assertLess(trimmed_tokens, 150000)

        # Verify we kept system messages
        system_msgs = [m for m in trimmed if isinstance(m, SystemMessage)]
        self.assertGreater(len(system_msgs), 0)

        # Verify we have recent conversation history
        self.assertTrue(any("199" in str(m.content) for m in trimmed))


if __name__ == '__main__':
    unittest.main()
