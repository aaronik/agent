import unittest
from unittest.mock import patch, MagicMock

from src.tools import fetch


class TestFetch(unittest.TestCase):
    @patch("src.tools.requests.get")
    def test_fetch_plain_url_prefixed_once(self, mock_get):
        """If given a normal URL, fetch should prepend https://r.jina.ai/ exactly once."""
        mock_response = MagicMock()
        mock_response.text = "hello world"
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        result = fetch("https://example.com")

        mock_get.assert_called_once_with(
            "https://r.jina.ai/https://example.com", allow_redirects=True
        )
        self.assertIn("hello world", result)

    @patch("src.tools.requests.get")
    def test_fetch_rjina_url_not_double_prefixed(self, mock_get):
        """If given an r.jina.ai URL, fetch should not double-prefix it."""
        mock_response = MagicMock()
        mock_response.text = "hello reddit"
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        url = "https://r.jina.ai/https://www.reddit.com"
        result = fetch(url)

        mock_get.assert_called_once_with(url, allow_redirects=True)
        self.assertIn("hello reddit", result)

    @patch("src.tools.requests.get")
    def test_fetch_http_error(self, mock_get):
        """Test handling of HTTP errors (404, 500, etc.)"""
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = Exception("HTTP 404: Not Found")
        mock_get.return_value = mock_response

        result = fetch("https://example.com/nonexistent")

        self.assertIn("Error fetching URL", result)
        self.assertIn("404", result)

    @patch("src.tools.requests.get")
    def test_fetch_connection_error(self, mock_get):
        """Test handling of connection errors - SHOULD be caught and returned as error string"""
        mock_get.side_effect = Exception("Connection refused")

        result = fetch("https://unreachable.example.com")

        # Connection errors SHOULD be caught and returned as error strings
        self.assertIn("Error fetching URL", result)
        self.assertIn("Connection refused", result)

    @patch("src.tools.requests.get")
    def test_fetch_timeout_error(self, mock_get):
        """Test handling of timeout errors - SHOULD be caught and returned as error string"""
        mock_get.side_effect = Exception("Read timed out")

        result = fetch("https://slow.example.com")

        # Timeout errors SHOULD be caught and returned as error strings
        self.assertIn("Error fetching URL", result)
        self.assertIn("timed out", result)

    @patch("src.tools.requests.get")
    def test_fetch_content_truncation(self, mock_get):
        """Test that very large responses are truncated"""
        # Create content larger than MAX_RESPONSE_LENGTH (1,000,000)
        large_content = "A" * 1_500_000
        mock_response = MagicMock()
        mock_response.text = large_content
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        result = fetch("https://example.com/large")

        # Should be truncated to MAX_RESPONSE_LENGTH
        self.assertIn("[Content truncated due to size limitations]", result)
        # Result should be around 1,000,000 chars + URL prefix + truncation message
        self.assertLess(len(result), 1_100_000)

    @patch("src.tools.requests.get")
    def test_fetch_empty_response(self, mock_get):
        """Test handling of empty responses"""
        mock_response = MagicMock()
        mock_response.text = ""
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        result = fetch("https://example.com/empty")

        self.assertIn("[URL]: https://example.com/empty", result)

    @patch("src.tools.requests.get")
    def test_fetch_includes_url_in_output(self, mock_get):
        """Test that the original URL is included in the output"""
        mock_response = MagicMock()
        mock_response.text = "Content here"
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        result = fetch("https://example.com/page")

        self.assertIn("[URL]: https://example.com/page", result)
        self.assertIn("Content here", result)

    @patch("src.tools.requests.get")
    def test_fetch_redirects_enabled(self, mock_get):
        """Test that fetch allows redirects"""
        mock_response = MagicMock()
        mock_response.text = "Redirected content"
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        fetch("https://example.com")

        # Verify allow_redirects=True is passed
        call_kwargs = mock_get.call_args[1]
        self.assertTrue(call_kwargs.get("allow_redirects"))


if __name__ == "__main__":
    unittest.main()
