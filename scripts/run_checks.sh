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
if "${PYTHON}" -c "import pytest_cov" >/dev/null 2>&1; then
  "${PYTHON}" -m pytest --cov=fhe_native_mamba3 --cov-report=term-missing
else
  echo "pytest-cov is not installed; running pytest without coverage"
  "${PYTHON}" -m pytest
fi

if [[ -x ".venv/bin/pre-commit" ]]; then
  PYTHON="${PYTHON}" .venv/bin/pre-commit run --all-files
else
  PYTHON="${PYTHON}" pre-commit run --all-files
fi
