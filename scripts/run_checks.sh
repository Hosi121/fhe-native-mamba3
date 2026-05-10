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
RUN_PRECOMMIT="${RUN_PRECOMMIT:-0}"
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
if "${PYTHON}" -c "import pytest_cov" >/dev/null 2>&1; then
  "${PYTHON}" -m pytest \
    "${PYTEST_PARALLEL[@]}" \
    --cov=fhe_native_mamba3 \
    --cov-report=term-missing \
    --durations="${PYTEST_DURATIONS}"
else
  echo "pytest-cov is not installed; running pytest without coverage"
  "${PYTHON}" -m pytest "${PYTEST_PARALLEL[@]}" --durations="${PYTEST_DURATIONS}"
fi

if [[ "${RUN_PRECOMMIT}" == "1" ]]; then
  if [[ -x ".venv/bin/pre-commit" ]]; then
    PYTHON="${PYTHON}" .venv/bin/pre-commit run --all-files
  else
    PYTHON="${PYTHON}" pre-commit run --all-files
  fi
fi
