import os
import tempfile

from src.tools import search_replace


def test_search_replace():
    # Create temp file with original content
    tmp_file = tempfile.NamedTemporaryFile(delete=False, mode='w+')
    try:
        tmp_file.write('This is the ORIGINAL line.\nAnother line.\n')
        tmp_file.close()

        # Apply search and replace
        result = search_replace(
            tmp_file.name,
            'This is the ORIGINAL line.',
            'CHANGED LINE'
        )

        assert 'Successfully' in result, f"search_replace failed: {result}"
        assert 'Diff:' in result, f"No diff in result: {result}"
        assert '-This is the ORIGINAL line.' in result, f"Old text not in diff: {result}"
        assert '+CHANGED LINE' in result, f"New text not in diff: {result}"

        # Read back file and verify change
        with open(tmp_file.name, 'r') as f:
            contents = f.read()

        assert 'CHANGED LINE' in contents, (
            f"File contents not changed correctly: {contents}"
        )
        assert 'Another line.' in contents, (
            f"Other content was affected: {contents}"
        )

    finally:
        os.unlink(tmp_file.name)


def test_search_replace_not_found():
    # Create temp file with content
    tmp_file = tempfile.NamedTemporaryFile(delete=False, mode='w+')
    try:
        tmp_file.write('Some content\n')
        tmp_file.close()

        # Try to replace non-existent text
        result = search_replace(
            tmp_file.name,
            'This text does not exist',
            'New text'
        )

        assert 'Text not found' in result, f"Expected 'Text not found' error, got: {result}"

    finally:
        os.unlink(tmp_file.name)


def test_search_replace_multiple_occurrences():
    # Create temp file with multiple occurrences
    tmp_file = tempfile.NamedTemporaryFile(delete=False, mode='w+')
    try:
        tmp_file.write('foo bar foo\nfoo baz\n')
        tmp_file.close()

        # Replace all occurrences
        result = search_replace(
            tmp_file.name,
            'foo',
            'replaced'
        )

        assert '3 occurrence(s)' in result, f"Expected 3 occurrences, got: {result}"
        assert 'Diff:' in result, f"No diff in result: {result}"

        # Verify all were replaced
        with open(tmp_file.name, 'r') as f:
            contents = f.read()

        assert contents == 'replaced bar replaced\nreplaced baz\n', (
            f"Not all occurrences replaced: {contents}"
        )

    finally:
        os.unlink(tmp_file.name)


if __name__ == '__main__':
    test_search_replace()
    test_search_replace_not_found()
    test_search_replace_multiple_occurrences()
    print("All tests passed!")
