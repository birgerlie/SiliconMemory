#!/usr/bin/env bash
# Start the Silicon Memory server with the web UI.
#
# Usage:
#   ./start.sh                        # defaults: rest mode, port 8420, qwen3-4b
#   ./start.sh --llm-model qwen3-80b  # override LLM model
#   ./start.sh --mode full             # REST + background reflection
#
# The web UI is served at http://localhost:8420/
# The REST API is at    http://localhost:8420/api/v1/

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV="$SCRIPT_DIR/.venv"

if [ ! -d "$VENV" ]; then
  echo "Error: virtualenv not found at $VENV" >&2
  echo "Create it first: python3 -m venv .venv && .venv/bin/pip install -e ." >&2
  exit 1
fi

exec "$VENV/bin/silicon-memory-server" \
  --mode rest \
  --port 8420 \
  "$@"
