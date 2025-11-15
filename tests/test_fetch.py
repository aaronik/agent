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


if __name__ == "__main__":
    unittest.main()
