#!/usr/bin/env bash
set -euo pipefail

REMOTE="${REMOTE:-high}"
REMOTE_DIR="${REMOTE_DIR:-~/cipher/fhe-native-mamba3}"
PYTHON="${PYTHON:-\$HOME/miniconda3/envs/nemotron/bin/python}"

ssh "${REMOTE}" bash -se -- "${REMOTE_DIR}" "${PYTHON}" <<'REMOTE_CHECKS'
set -euo pipefail

remote_dir="$1"
python="$2"

case "${remote_dir}" in
  "~") remote_dir="${HOME}" ;;
  "~/"*) remote_dir="${HOME}/${remote_dir#"~/"}" ;;
esac

case "${python}" in
  '$HOME/'*) python="${HOME}/${python#\$HOME/}" ;;
  "~/"*) python="${HOME}/${python#"~/"}" ;;
esac

cd "${remote_dir}"

PYTHONPATH=src PYTHON="${python}" CHECK_JOBS=auto RUN_PRECOMMIT=0 scripts/run_checks.sh
PYTHONPATH=src "${python}" -m fhe_native_mamba3.cli inspect \
  --d-model 16 \
  --d-state 4 \
  --mimo-rank 2 \
  --seq-len 8
REMOTE_CHECKS
