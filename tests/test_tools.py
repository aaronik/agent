import unittest
import aisuite as ai
from src.tools import build_trim_message


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


if __name__ == '__main__':
    unittest.main()
