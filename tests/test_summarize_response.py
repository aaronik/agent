import unittest
from unittest.mock import patch, MagicMock
from src.tools import summarize_response


class TestSummarizeResponse(unittest.TestCase):

    @patch('subprocess.Popen')
    def test_summarize_response_success(self, mock_popen):
        mock_popen.return_value = MagicMock()
        result = summarize_response("This is a short summary.")
        self.assertEqual(result, "Speech synthesis started")
        mock_popen.assert_called_once_with(['say', "This is a short summary."])

    @patch('subprocess.Popen', side_effect=Exception('Failed'))
    def test_summarize_response_failure(self, mock_popen):
        result = summarize_response("This is a short summary.")
        self.assertIn("Error starting speech synthesis", result)

    def test_summary_input_sanitization(self):
        with patch('subprocess.Popen') as mock_popen:
            summarize_response('Say "hello" and `ls` $PATH')
            mock_popen.assert_called_once()
            args_passed = mock_popen.call_args[0][0]
            # Ensure the summary string has backslashes before dangerous chars
            self.assertIn('\"', args_passed[1])


if __name__ == '__main__':
    unittest.main()
