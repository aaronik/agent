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
            prev = None
            files = []
            for i in range(claude_memory.MAX_IMPORT_DEPTH + 2):
                f = os.path.join(tmpdir, f"file{i}.md")
                files.append(f)
            for i in range(len(files)-1):
                with open(files[i], "w") as f:
                    f.write(f"File {i}\n@{os.path.basename(files[i+1])}")
            # Last file empty
            with open(files[-1], "w") as f:
                f.write(f"File {len(files)-1}")

            result = claude_memory.read_claude_md(files[0])
            self.assertIn("File 0", result)
            self.assertNotIn(f"File {claude_memory.MAX_IMPORT_DEPTH + 1}", result)

    def test_find_upwards_claude_md_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create nested directories
            nested = os.path.join(tmpdir, "a/b/c")
            os.makedirs(nested)

            # Place CLAUDE.md in tmpdir and in tmpdir/a
            file1 = os.path.join(tmpdir, "CLAUDE.md")
            file2 = os.path.join(tmpdir, "a", "CLAUDE.md")
            with open(file1, "w") as f:
                f.write("root")
            with open(file2, "w") as f:
                f.write("a")

            found_files = claude_memory.find_upwards_claude_md_files(nested)
            self.assertIn(os.path.abspath(file1), found_files)
            self.assertIn(os.path.abspath(file2), found_files)

    def test_load_all_claude_memory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            file1 = os.path.join(tmpdir, "CLAUDE.md")
            file2 = os.path.join(tmpdir, "b", "CLAUDE.md")
            os.makedirs(os.path.dirname(file2))
            with open(file1, "w") as f:
                f.write("root content")
            with open(file2, "w") as f:
                f.write("nested content")

            files = claude_memory.find_all_claude_md_files(tmpdir)
            print("DEBUG: Found CLAUDE.md files:", files)

            combined = claude_memory.load_all_claude_memory(tmpdir)
            self.assertIn("root content", combined)
            self.assertIn("nested content", combined)


if __name__ == "__main__":
    unittest.main()
