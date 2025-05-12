import unittest
import os

from src.util import extract_text, sys_uname, sanitize_path


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


if __name__ == '__main__':
    unittest.main()
