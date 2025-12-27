import time

from src.tools import run_shell_command


def test_run_shell_command_timeout_kills_interactive() -> None:
    # Simulate an interactive-ish command that will block indefinitely.
    # If timeout isn't enforced correctly, this test will hang.
    start = time.time()
    out = run_shell_command("bash -lc 'read -r x'", timeout=1)
    elapsed = time.time() - start

    assert elapsed < 3
    assert "(exit code:" in out
