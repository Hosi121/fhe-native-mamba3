#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${PYTHON:-}" ]]; then
  if [[ -x ".venv/bin/python" ]]; then
    PYTHON=".venv/bin/python"
  else
    PYTHON="python3"
  fi
fi

"${PYTHON}" -m ruff format --check .
"${PYTHON}" -m ruff check .
"${PYTHON}" -m pytest

if [[ -x ".venv/bin/pre-commit" ]]; then
  PYTHON="${PYTHON}" .venv/bin/pre-commit run --all-files
else
  PYTHON="${PYTHON}" pre-commit run --all-files
fi
