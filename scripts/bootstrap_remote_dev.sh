#!/usr/bin/env bash
set -euo pipefail

REMOTE="${REMOTE:-high}"
PYTHON="${PYTHON:-\$HOME/miniconda3/envs/nemotron/bin/python}"

ssh "${REMOTE}" "${PYTHON} -m pip install 'coverage[toml]>=7.6' 'pytest>=8.0' 'pytest-cov>=5.0' 'ruff>=0.6' 'pre-commit>=3.7'"
