import unittest
import os
import tempfile
from src import claude_memory


class TestClaudeMemory(unittest.TestCase):

    def test_read_file_content_existing_and_missing(self):
        with tempfile.NamedTemporaryFile('w', delete=True) as tf:
            tf.write("content")
            tf.flush()
            self.assertEqual(claude_memory.read_file_content(tf.name), "content")
        self.assertEqual(claude_memory.read_file_content("nonexistent.file"), "")

    def test_remove_code_spans_and_blocks(self):
        text = "Text before ```code block``` more text `inline code` end"
        cleaned = claude_memory.remove_code_spans_and_blocks(text)
        self.assertNotIn("```code block```", cleaned)
        self.assertNotIn("`inline code`", cleaned)
        self.assertIn("Text before", cleaned)
        self.assertIn("more text", cleaned)

    def test_parse_import_paths(self):
        input_text = "@path/to/file\nNot an import\n@ another/path\n`@ignore/inline`\n```\n@ignore/codeblock\n```"
        expected = ["path/to/file", "another/path"]
        result = claude_memory.parse_import_paths(input_text)
        self.assertEqual(result, expected)

    def test_resolve_path(self):
        home_dir = os.path.expanduser("~")
        self.assertEqual(claude_memory.resolve_path("/base", "~/file"), os.path.join(home_dir, "file"))
        self.assertEqual(claude_memory.resolve_path("/base", "/abs/path"), "/abs/path")
        self.assertEqual(claude_memory.resolve_path("/base", "rel/path"), "/base/rel/path")

    def test_read_claude_md_recursive_imports(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root_file = os.path.join(tmpdir, "CLAUDE.md")
            imported_file = os.path.join(tmpdir, "imported.md")
            imported_content = "imported content"
            with open(imported_file, "w") as f:
                f.write(imported_content)

            root_content = f"Root content\n@imported.md"
            with open(root_file, "w") as f:
                f.write(root_content)

            combined = claude_memory.read_claude_md(root_file)
            self.assertIn("Root content", combined)
            self.assertIn(imported_content, combined)

    def test_read_claude_md_max_depth(self):
        # Build chain longer than MAX_IMPORT_DEPTH to test cutoff
        with tempfile.TemporaryDirectory() as tmpdir:
            files = []
            for i in range(claude_memory.MAX_IMPORT_DEPTH + 2):
                f = os.path.join(tmpdir, f"file{i}.md")
                files.append(f)
            for i in range(len(files) - 1):
                with open(files[i], "w") as f:
                    f.write(f"File {i}\n@{os.path.basename(files[i + 1])}")
            with open(files[-1], "w") as f:
                f.write(f"File {len(files) - 1}")

            result = claude_memory.read_claude_md(files[0])
            self.assertIn("File 0", result)
            self.assertNotIn(f"File {claude_memory.MAX_IMPORT_DEPTH + 1}", result)

    def test_find_project_claude_md_file_current_dir_only(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            nested = os.path.join(tmpdir, "a", "b")
            os.makedirs(nested)

            root_file = os.path.join(tmpdir, "CLAUDE.md")
            nested_file = os.path.join(nested, "CLAUDE.md")

            with open(root_file, "w") as f:
                f.write("root")

            # From nested dir, should NOT find root_file
            self.assertIsNone(claude_memory.find_project_claude_md_file(nested))

            # If CLAUDE.md exists in the current dir, it should be found
            with open(nested_file, "w") as f:
                f.write("nested")

            self.assertEqual(os.path.abspath(nested_file), claude_memory.find_project_claude_md_file(nested))

    def test_find_user_claude_md_file_home_claude(self):
        # Hard to test without touching real $HOME; just assert it's a well-formed path
        # and returns None if it doesn't exist.
        path = os.path.join(os.path.expanduser("~"), ".claude", "CLAUDE.md")
        if os.path.isfile(path):
            self.assertEqual(path, claude_memory.find_user_claude_md_file())
        else:
            self.assertIsNone(claude_memory.find_user_claude_md_file())

    def test_load_all_claude_memory_project_only(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            file1 = os.path.join(tmpdir, "CLAUDE.md")
            with open(file1, "w") as f:
                f.write("root content")

            files = claude_memory.find_all_claude_md_files(tmpdir)
            self.assertIn(os.path.abspath(file1), files)

            combined = claude_memory.load_all_claude_memory(tmpdir)
            self.assertIn("root content", combined)


if __name__ == "__main__":
    unittest.main()
