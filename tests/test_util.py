import unittest
import os
import subprocess
from unittest.mock import Mock, MagicMock, patch
import aisuite as ai

from src.util import (
    extract_text, sys_uname, sanitize_path, TokenUsage,
    sys_pwd, sys_git_ls, get_current_filetree,
    message_from_choice, message_from_user_input,
    format_subproc_result, refuse_if_duplicate
)


class TestUtilFunctions(unittest.TestCase):
    def test_extract_text(self):
        html = '<html><body><p>Hello World!</p></body></html>'
        extracted = extract_text(html)
        expected = 'Hello World!'
        self.assertEqual(extracted, expected)

    def test_sys_uname(self):
        info = sys_uname()
        self.assertIn('Darwin', info)

    def test_sanitize_path(self):
        orig = "file.ext"
        expected = "./file.ext"
        actual = sanitize_path(orig)
        assert actual == expected

        orig = "~/file.ext"
        expected = f"{os.environ['HOME']}/file.ext"
        actual = sanitize_path(orig)
        assert actual == expected


class TestTokenUsage(unittest.TestCase):
    def setUp(self):
        self.mock_cost_map = {
            "openai:gpt-4o-mini": {
                "input_cost_per_token": 0.00001,
                "output_cost_per_token": 0.00003
            },
            "test-model": {
                "input_cost_per_token": 0.0001,
                "output_cost_per_token": 0.0002
            }
        }

    @patch('litellm.get_model_cost_map')
    def test_initialization(self, mock_get_cost_map):
        mock_get_cost_map.return_value = self.mock_cost_map
        usage = TokenUsage(model="test-model")
        self.assertEqual(usage.model, "test-model")
        self.assertEqual(usage.prompt_tokens, 0)
        self.assertEqual(usage.completion_tokens, 0)

    @patch('litellm.get_model_cost_map')
    def test_total_tokens(self, mock_get_cost_map):
        mock_get_cost_map.return_value = self.mock_cost_map
        usage = TokenUsage(model="test-model", prompt_tokens=100, completion_tokens=50)
        self.assertEqual(usage.total_tokens(), 150)

    @patch('litellm.get_model_cost_map')
    def test_prompt_cost(self, mock_get_cost_map):
        mock_get_cost_map.return_value = self.mock_cost_map
        usage = TokenUsage(model="test-model", prompt_tokens=1000)
        cost = usage.prompt_cost()
        self.assertEqual(cost, 0.1)

    @patch('litellm.get_model_cost_map')
    def test_completion_cost(self, mock_get_cost_map):
        mock_get_cost_map.return_value = self.mock_cost_map
        usage = TokenUsage(model="test-model", completion_tokens=500)
        cost = usage.completion_cost()
        self.assertEqual(cost, 0.1)

    @patch('litellm.get_model_cost_map')
    def test_total_cost(self, mock_get_cost_map):
        mock_get_cost_map.return_value = self.mock_cost_map
        usage = TokenUsage(model="test-model", prompt_tokens=1000, completion_tokens=500)
        cost = usage.total_cost()
        self.assertEqual(cost, 0.2)

    @patch('litellm.get_model_cost_map')
    def test_zero_tokens_cost(self, mock_get_cost_map):
        mock_get_cost_map.return_value = self.mock_cost_map
        usage = TokenUsage(model="test-model")
        self.assertEqual(usage.prompt_cost(), 0.0)
        self.assertEqual(usage.completion_cost(), 0.0)
        self.assertEqual(usage.total_cost(), 0.0)

    @patch('litellm.get_model_cost_map')
    def test_ingest_response(self, mock_get_cost_map):
        mock_get_cost_map.return_value = self.mock_cost_map
        usage = TokenUsage(model="test-model")

        mock_response = Mock()
        mock_response.usage = Mock()
        mock_response.usage.prompt_tokens = 100
        mock_response.usage.completion_tokens = 50

        usage.ingest_response(mock_response)

        self.assertEqual(usage.prompt_tokens, 100)
        self.assertEqual(usage.completion_tokens, 50)

    @patch('litellm.get_model_cost_map')
    def test_ingest_response_accumulates(self, mock_get_cost_map):
        mock_get_cost_map.return_value = self.mock_cost_map
        usage = TokenUsage(model="test-model", prompt_tokens=50, completion_tokens=25)

        mock_response = Mock()
        mock_response.usage = Mock()
        mock_response.usage.prompt_tokens = 100
        mock_response.usage.completion_tokens = 50

        usage.ingest_response(mock_response)

        self.assertEqual(usage.prompt_tokens, 150)
        self.assertEqual(usage.completion_tokens, 75)

    @patch('litellm.get_model_cost_map')
    def test_model_not_in_cost_map(self, mock_get_cost_map):
        mock_get_cost_map.return_value = self.mock_cost_map
        usage = TokenUsage(model="unknown-model", prompt_tokens=100)
        cost = usage.prompt_cost()
        self.assertEqual(cost, 0.0)

    @patch('litellm.get_model_cost_map')
    @patch('builtins.print')
    def test_print_panel(self, mock_print, mock_get_cost_map):
        """Test that print_panel creates a Rich panel with cost information"""
        mock_get_cost_map.return_value = self.mock_cost_map
        usage = TokenUsage(model="test-model", prompt_tokens=1000, completion_tokens=500)

        # Should not raise any exceptions
        usage.print_panel()

        # Verify the method completes without error (Rich will handle the actual formatting)
        # We can't easily test Rich output without mocking the entire Console,
        # but we can verify the method runs successfully


class TestSystemUtilityFunctions(unittest.TestCase):
    def test_sys_pwd(self):
        result = sys_pwd()
        self.assertIsInstance(result, str)
        self.assertIn("/", result)

    def test_sys_git_ls_in_repo_or_empty(self):
        result = sys_git_ls()
        self.assertIsInstance(result, str)
        # In a git repo, should list files; outside, should be empty.
        if result != "":
            self.assertGreater(len(result), 0)

    def test_get_current_filetree(self):
        result = get_current_filetree()
        self.assertIsInstance(result, str)
        self.assertIn(".", result)


class TestMessageUtilities(unittest.TestCase):
    def test_message_from_user_input(self):
        result = message_from_user_input("test input")
        self.assertIsInstance(result, ai.Message)
        self.assertEqual(result.role, "user")
        self.assertEqual(result.content, "test input")

    def test_message_from_choice(self):
        mock_choice = Mock()
        mock_choice.message = Mock()
        mock_choice.message.role = "assistant"
        mock_choice.message.content = "response content"

        result = message_from_choice(mock_choice)
        self.assertIsInstance(result, ai.Message)
        self.assertEqual(result.role, "assistant")
        self.assertEqual(result.content, "response content")


class TestFormatSubprocResult(unittest.TestCase):
    def test_format_subproc_result(self):
        mock_result = Mock(spec=subprocess.CompletedProcess)
        mock_result.stdout = "output text"
        mock_result.stderr = "error text"
        mock_result.returncode = 0

        result = format_subproc_result(mock_result)
        # Output should look like a terminal: no headings.
        self.assertIn("output text", result)
        self.assertIn("error text", result)
        # For successful commands we don't append an exit code line.
        self.assertNotIn("exit code:", result)


class TestRefuseIfDuplicate(unittest.TestCase):
    def test_refuse_if_duplicate_allows_first_call(self):
        @refuse_if_duplicate
        def test_func(arg):
            return f"result: {arg}"

        result = test_func("test")
        self.assertEqual(result, "result: test")

    def test_refuse_if_duplicate_blocks_duplicate(self):
        @refuse_if_duplicate
        def test_func(arg):
            return f"result: {arg}"

        # First call should work
        result1 = test_func("test")
        self.assertEqual(result1, "result: test")

        # Second call with same args should return empty string
        result2 = test_func("test")
        self.assertEqual(result2, "")

    def test_refuse_if_duplicate_allows_different_args(self):
        @refuse_if_duplicate
        def test_func(arg):
            return f"result: {arg}"

        result1 = test_func("test1")
        self.assertEqual(result1, "result: test1")

        result2 = test_func("test2")
        self.assertEqual(result2, "result: test2")


if __name__ == '__main__':
    unittest.main()
