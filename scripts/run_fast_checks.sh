#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${PYTHON:-}" ]]; then
  if [[ -x ".venv/bin/python" ]]; then
    PYTHON=".venv/bin/python"
  else
    PYTHON="python3"
  fi
fi

PYTEST_DURATIONS="${PYTEST_DURATIONS:-10}"
CHECK_JOBS="${CHECK_JOBS:-}"
PYTEST_PARALLEL=()
if [[ -n "${CHECK_JOBS}" ]]; then
  if "${PYTHON}" -c "import xdist" >/dev/null 2>&1; then
    PYTEST_PARALLEL=(-n "${CHECK_JOBS}")
  else
    echo "pytest-xdist is not installed; running pytest serially"
  fi
fi

"${PYTHON}" -m ruff format --check .
"${PYTHON}" -m ruff check .
"${PYTHON}" -m pytest "${PYTEST_PARALLEL[@]}" --durations="${PYTEST_DURATIONS}" "$@"
