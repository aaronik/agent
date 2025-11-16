import unittest
import tempfile
import os
import shutil
from unittest.mock import patch, Mock, MagicMock
import aisuite as ai
from src.tools import (
    build_trim_message,
    read_file,
    write_file,
    run_shell_command,
    communicate,
    gen_image,
    search_text,
    search_text_alternative,
    search_images,
    spawn
)


class TestFileOperations(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.test_file = os.path.join(self.test_dir, "test.txt")

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_write_file_success(self):
        result = write_file(self.test_file, "Hello, World!")
        self.assertEqual(result, "Success")
        self.assertTrue(os.path.exists(self.test_file))
        with open(self.test_file, 'r') as f:
            self.assertEqual(f.read(), "Hello, World!")

    def test_write_file_creates_directories(self):
        nested_path = os.path.join(self.test_dir, "subdir", "nested.txt")
        result = write_file(nested_path, "Nested content")
        self.assertEqual(result, "Success")
        self.assertTrue(os.path.exists(nested_path))
        with open(nested_path, 'r') as f:
            self.assertEqual(f.read(), "Nested content")

    def test_write_file_overwrites_existing(self):
        write_file(self.test_file, "Original")
        result = write_file(self.test_file, "Updated")
        self.assertEqual(result, "Success")
        with open(self.test_file, 'r') as f:
            self.assertEqual(f.read(), "Updated")

    def test_read_file_success(self):
        with open(self.test_file, 'w') as f:
            f.write("Test content")
        result = read_file(self.test_file)
        self.assertEqual(result, "Test content")

    def test_read_file_not_found(self):
        result = read_file(os.path.join(self.test_dir, "nonexistent.txt"))
        self.assertIn("file not found", result)

    def test_read_write_integration(self):
        content = "Integration test content\nWith multiple lines\n"
        write_result = write_file(self.test_file, content)
        self.assertEqual(write_result, "Success")
        read_result = read_file(self.test_file)
        self.assertEqual(read_result, content)


class TestRunShellCommand(unittest.TestCase):
    def test_simple_command_success(self):
        result = run_shell_command("echo 'Hello World'")
        self.assertIn("Hello World", result)
        self.assertIn("[STDOUT]", result)
        self.assertIn("[CODE]", result)

    def test_command_with_stderr(self):
        result = run_shell_command("echo 'error message' >&2")
        self.assertIn("error message", result)
        self.assertIn("[STDERR]", result)

    def test_command_exit_code(self):
        result = run_shell_command("sh -c 'exit 42'")
        self.assertIn("42", result)
        self.assertIn("[CODE]", result)

    def test_command_timeout(self):
        result = run_shell_command("sleep 5", timeout=1)
        self.assertIn("124", result)

    def test_pwd_command(self):
        result = run_shell_command("pwd")
        self.assertIn("/", result)
        self.assertIn("[STDOUT]", result)

    def test_large_output(self):
        """Test handling of commands with large output"""
        # Generate a large amount of output (10,000 lines using seq)
        result = run_shell_command("seq 1 10000 | while read i; do echo 'Line '$i; done")
        self.assertIn("[STDOUT]", result)
        self.assertIn("Line 1", result)
        # Verify the output is not truncated (should contain later lines)
        self.assertIn("Line 10000", result)

    def test_special_characters_in_output(self):
        """Test handling of special characters in command output"""
        # Test various special characters
        result = run_shell_command("echo 'Special: !@#$%^&*()[]{}|\\;:,.<>?/~`'")
        self.assertIn("Special:", result)
        self.assertIn("!@#$%^&*", result)

    def test_newlines_in_output(self):
        """Test handling of multiple newlines"""
        result = run_shell_command("printf 'Line1\\n\\n\\nLine2\\n'")
        self.assertIn("Line1", result)
        self.assertIn("Line2", result)

    def test_unicode_characters(self):
        """Test handling of unicode characters"""
        result = run_shell_command("echo '你好世界 🌍 Ñoño'")
        self.assertIn("你好世界", result)
        self.assertIn("🌍", result)
        self.assertIn("Ñoño", result)

    def test_multiline_command(self):
        """Test handling of multiline commands"""
        result = run_shell_command("echo 'First' && echo 'Second' && echo 'Third'")
        self.assertIn("First", result)
        self.assertIn("Second", result)
        self.assertIn("Third", result)

    def test_empty_output(self):
        """Test handling of commands with no output"""
        result = run_shell_command("true")
        self.assertIn("[CODE]", result)
        self.assertIn("0", result)

    def test_command_with_quotes(self):
        """Test handling of commands containing quotes"""
        result = run_shell_command("echo \"Hello 'World'\"")
        self.assertIn("Hello 'World'", result)


class TestBuildTrimMessage(unittest.TestCase):
    def setUp(self):
        self.messages = [
            ai.Message(content="Hello", role="user"),
            ai.Message(content="World", role="user"),
            ai.Message(content="Test message", role="user")
        ]
        self.trim_message = build_trim_message(self.messages)

    def test_delete_message(self):
        self.trim_message(1, "")
        self.assertEqual(len(self.messages), 2)
        self.assertEqual(self.messages[0].content, "Hello")
        self.assertEqual(self.messages[1].content, "Test message")

    def test_update_message(self):
        self.trim_message(0, "Updated")
        self.assertEqual(self.messages[0].content, "Updated")

    def test_invalid_index(self):
        result = self.trim_message(10, "No change")
        self.assertIn("trim_message failed", result)


class TestCommunicate(unittest.TestCase):
    def test_communicate_returns_message(self):
        message = "Test communication message"
        result = communicate(message)
        self.assertEqual(result, message)

    def test_communicate_with_empty_string(self):
        result = communicate("")
        self.assertEqual(result, "")

    def test_communicate_with_multiline(self):
        message = "Line 1\nLine 2\nLine 3"
        result = communicate(message)
        self.assertEqual(result, message)


class TestGenImage(unittest.TestCase):
    """Tests for gen_image tool"""

    @patch('src.tools.requests.post')
    def test_gen_image_success(self, mock_post):
        """Test successful image generation"""
        mock_response = Mock()
        mock_response.text = '{"data": [{"url": "https://example.com/image1.png"}]}'
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response

        result = gen_image(
            number=1,
            model="dall-e-2",
            size="512x512",
            prompt="A beautiful sunset"
        )

        self.assertEqual(result, '{"data": [{"url": "https://example.com/image1.png"}]}')
        mock_post.assert_called_once()

        # Verify API call parameters
        call_args = mock_post.call_args
        self.assertEqual(call_args[1]['json']['n'], 1)
        self.assertEqual(call_args[1]['json']['model'], "dall-e-2")
        self.assertEqual(call_args[1]['json']['size'], "512x512")
        self.assertEqual(call_args[1]['json']['prompt'], "A beautiful sunset")

    @patch('src.tools.requests.post')
    def test_gen_image_connection_error(self, mock_post):
        """Test handling of connection errors - SHOULD be caught and returned as error string"""
        mock_post.side_effect = Exception("API connection failed")

        result = gen_image(
            number=1,
            model="dall-e-3",
            size="1024x1024",
            prompt="Test prompt"
        )

        # Connection errors SHOULD be caught and returned as error strings
        self.assertIn("Error creating images", result)
        self.assertIn("API connection failed", result)

    @patch('src.tools.requests.post')
    def test_gen_image_http_error(self, mock_post):
        """Test handling of HTTP errors"""
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = Exception("HTTP 401: Unauthorized")
        mock_post.return_value = mock_response

        result = gen_image(
            number=1,
            model="dall-e-2",
            size="256x256",
            prompt="Test"
        )

        self.assertIn("Error creating images", result)


class TestSearchText(unittest.TestCase):
    """Tests for search_text tool"""

    @patch('src.tools.ddgs.duckduckgo_search.DDGS')
    def test_search_text_success(self, mock_ddgs_class):
        """Test successful text search"""
        mock_ddgs = Mock()
        mock_ddgs.text.return_value = [
            {"title": "Result 1", "href": "https://example.com/1"},
            {"title": "Result 2", "href": "https://example.com/2"}
        ]
        mock_ddgs_class.return_value = mock_ddgs

        result = search_text("test query", max_results=2)

        self.assertIn("Result 1", result)
        self.assertIn("https://example.com/1", result)
        self.assertIn("Result 2", result)
        self.assertIn("https://example.com/2", result)
        mock_ddgs.text.assert_called_once_with("test query", max_results=2, backend="lite")

    @patch('src.tools.ddgs.duckduckgo_search.DDGS')
    def test_search_text_api_error(self, mock_ddgs_class):
        """Test handling of search API errors"""
        mock_ddgs = Mock()
        mock_ddgs.text.side_effect = Exception("Search API failed")
        mock_ddgs_class.return_value = mock_ddgs

        result = search_text("test query")

        self.assertIn("duck duck go search api error", result)
        self.assertIn("Search API failed", result)

    @patch('src.tools.ddgs.duckduckgo_search.DDGS')
    def test_search_text_empty_results(self, mock_ddgs_class):
        """Test search with no results"""
        mock_ddgs = Mock()
        mock_ddgs.text.return_value = []
        mock_ddgs_class.return_value = mock_ddgs

        result = search_text("nonexistent query")

        self.assertEqual(result, "")


class TestSearchTextAlternative(unittest.TestCase):
    """Tests for search_text_alternative tool"""

    @patch('src.tools.subprocess.run')
    def test_search_text_alternative_success(self, mock_run):
        """Test successful alternative search"""
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "<html><body>Search Results Here</body></html>"
        mock_run.return_value = mock_result

        result = search_text_alternative("test query")

        self.assertIn("Search Results Here", result)
        mock_run.assert_called_once()

    @patch('src.tools.subprocess.run')
    def test_search_text_alternative_curl_error(self, mock_run):
        """Test handling of curl errors"""
        mock_result = Mock()
        mock_result.returncode = 1
        mock_result.stdout = ""
        mock_result.stderr = "curl: (6) Could not resolve host"
        mock_run.return_value = mock_result

        result = search_text_alternative("test query")

        self.assertIn("DuckDuckGo search error", result)


class TestSearchImages(unittest.TestCase):
    """Tests for search_images tool"""

    @patch('src.tools.ddgs.duckduckgo_search.DDGS')
    def test_search_images_success(self, mock_ddgs_class):
        """Test successful image search"""
        mock_ddgs = Mock()
        mock_ddgs.images.return_value = [
            {"title": "Image 1", "image": "https://example.com/img1.jpg"},
            {"title": "Image 2", "image": "https://example.com/img2.jpg"}
        ]
        mock_ddgs_class.return_value = mock_ddgs

        result = search_images("cats", max_results=2)

        self.assertIn("Image 1", result)
        self.assertIn("https://example.com/img1.jpg", result)
        self.assertIn("Image 2", result)
        self.assertIn("https://example.com/img2.jpg", result)

    @patch('src.tools.ddgs.duckduckgo_search.DDGS')
    def test_search_images_empty_results(self, mock_ddgs_class):
        """Test image search with no results"""
        mock_ddgs = Mock()
        mock_ddgs.images.return_value = []
        mock_ddgs_class.return_value = mock_ddgs

        result = search_images("nonexistent query")

        self.assertEqual(result, "")

    @patch('src.tools.ddgs.duckduckgo_search.DDGS')
    def test_search_images_none_results(self, mock_ddgs_class):
        """Test image search when None is returned"""
        mock_ddgs = Mock()
        mock_ddgs.images.return_value = None
        mock_ddgs_class.return_value = mock_ddgs

        result = search_images("test")

        self.assertEqual(result, "")


class TestSpawn(unittest.TestCase):
    """Tests for spawn tool"""

    @patch('aisuite.Client')
    def test_spawn_success(self, mock_client_class):
        """Test successful agent spawning"""
        mock_client = Mock()
        mock_response = Mock()
        mock_choice = Mock()
        mock_message = Mock()
        mock_message.content = "Task completed successfully"
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response
        mock_client_class.return_value = mock_client

        result = spawn("Analyze this data")

        self.assertIn("[SPAWNED AGENT OUTPUT]", result)
        self.assertIn("Task completed successfully", result)
        mock_client.chat.completions.create.assert_called_once()

    @patch('aisuite.Client')
    def test_spawn_error_handling(self, mock_client_class):
        """Test spawn error handling"""
        mock_client_class.side_effect = Exception("Client initialization failed")

        result = spawn("Test task")

        self.assertIn("Error spawning agent", result)
        self.assertIn("Client initialization failed", result)

    @patch('aisuite.Client')
    def test_spawn_includes_system_prompt(self, mock_client_class):
        """Test that spawn includes appropriate system prompt"""
        mock_client = Mock()
        mock_response = Mock()
        mock_choice = Mock()
        mock_message = Mock()
        mock_message.content = "Result"
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response
        mock_client_class.return_value = mock_client

        spawn("Test task")

        # Verify chat.completions.create was called
        call_args = mock_client.chat.completions.create.call_args
        self.assertEqual(call_args[1]['model'], "openai:gpt-4o-mini")
        self.assertEqual(call_args[1]['max_turns'], 10)


if __name__ == '__main__':
    unittest.main()
