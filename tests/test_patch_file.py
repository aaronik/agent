import os
import tempfile

from src.tools import patch_file

diff_text = ("""
--- original.txt
+++ original.txt
@@ -1,2 +1,2 @@
-This is the ORIGINAL line.
+CHANGED LINE
 Another line.
""")


def test_apply_diff():
    # Create temp file with original content
    tmp_file = tempfile.NamedTemporaryFile(delete=False, mode='w+')
    try:
        tmp_file.write('This is the ORIGINAL line.\nAnother line.\n')
        tmp_file.close()

        # Apply the diff and assert no exception
        try:
            patch_file(tmp_file.name, diff_text)
        except Exception as e:
            assert False, f"apply_diff raised an unexpected exception: {e}"

        # Read back file and verify change
        with open(tmp_file.name, 'r') as f:
            contents = f.read()

        assert 'CHANGED LINE' in contents, (
            f"File contents not changed correctly: {contents}"
        )

    finally:
        os.unlink(tmp_file.name)


if __name__ == '__main__':
    test_apply_diff()
