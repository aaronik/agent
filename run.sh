#!/usr/bin/env sh

# Resolve the directory of this script, even if symlinked, to run from any folder
SCRIPT_DIR=$(cd -- "$(dirname -- "$0")" >/dev/null 2>&1 && pwd)

# Run the python interpreter and script relative to the script directory with passed arguments
"$SCRIPT_DIR/.venv/bin/python" "$SCRIPT_DIR/main_aisuite.py" "$@"
