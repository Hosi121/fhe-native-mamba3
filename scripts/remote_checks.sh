#!/usr/bin/env bash
set -euo pipefail

REMOTE="${REMOTE:-high}"
REMOTE_DIR="${REMOTE_DIR:-~/cipher/fhe-native-mamba3}"
PYTHON="${PYTHON:-\$HOME/miniconda3/envs/nemotron/bin/python}"

ssh "${REMOTE}" "cd ${REMOTE_DIR} && PYTHONPATH=src ${PYTHON} -m ruff format --check ."
ssh "${REMOTE}" "cd ${REMOTE_DIR} && PYTHONPATH=src ${PYTHON} -m ruff check ."
ssh "${REMOTE}" "cd ${REMOTE_DIR} && PYTHONPATH=src ${PYTHON} -m pytest"
ssh "${REMOTE}" "cd ${REMOTE_DIR} && PYTHONPATH=src PYTHON=${PYTHON} ${PYTHON} -m pre_commit run --all-files"
ssh "${REMOTE}" "cd ${REMOTE_DIR} && PYTHONPATH=src ${PYTHON} -m fhe_native_mamba3.cli inspect --d-model 16 --d-state 4 --mimo-rank 2 --seq-len 8"
