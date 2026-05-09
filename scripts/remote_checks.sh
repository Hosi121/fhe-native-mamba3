#!/usr/bin/env bash
set -euo pipefail

REMOTE="${REMOTE:-high}"
REMOTE_DIR="${REMOTE_DIR:-~/cipher/fhe-native-mamba3}"
PYTHON="${PYTHON:-\$HOME/miniconda3/envs/nemotron/bin/python}"

ssh "${REMOTE}" "cd ${REMOTE_DIR} && PYTHONPATH=src ${PYTHON} -m ruff format --check ."
ssh "${REMOTE}" "cd ${REMOTE_DIR} && PYTHONPATH=src ${PYTHON} -m ruff check ."
ssh "${REMOTE}" "cd ${REMOTE_DIR} && if PYTHONPATH=src ${PYTHON} -c 'import pytest_cov' >/dev/null 2>&1; then PYTHONPATH=src ${PYTHON} -m pytest --cov=fhe_native_mamba3 --cov-report=term-missing; else echo 'pytest-cov is not installed; running pytest without coverage'; PYTHONPATH=src ${PYTHON} -m pytest; fi"
ssh "${REMOTE}" "cd ${REMOTE_DIR} && PYTHONPATH=src PYTHON=${PYTHON} ${PYTHON} -m pre_commit run --all-files"
ssh "${REMOTE}" "cd ${REMOTE_DIR} && PYTHONPATH=src ${PYTHON} -m fhe_native_mamba3.cli inspect --d-model 16 --d-state 4 --mimo-rank 2 --seq-len 8"
