#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${PYTHON:-}" ]]; then
  if [[ -x ".venv/bin/python" ]]; then
    PYTHON=".venv/bin/python"
  else
    PYTHON="python3"
  fi
fi

"${PYTHON}" -m pytest \
  --cov=fhe_native_mamba3 \
  --cov-report=term-missing \
  --cov-report=xml \
  --durations="${PYTEST_DURATIONS:-10}"
