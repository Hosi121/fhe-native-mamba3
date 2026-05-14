#!/usr/bin/env bash
set -euo pipefail

REMOTE="${REMOTE:-high}"
REMOTE_DIR="${REMOTE_DIR:-~/cipher/fhe-native-mamba3}"

ssh "${REMOTE}" "mkdir -p ${REMOTE_DIR}"

rsync -az --delete \
  --exclude '.venv/' \
  --exclude '.venv-openfhe/' \
  --exclude '__pycache__/' \
  --exclude '.pytest_cache/' \
  --exclude '.ruff_cache/' \
  --exclude '*.egg-info/' \
  --exclude 'build/' \
  --exclude 'dist/' \
  --exclude 'tmp/' \
  --exclude 'checkpoints/' \
  --exclude 'runs/' \
  --exclude 'logs/' \
  --exclude 'slurm/*.out' \
  --exclude 'slurm/*.err' \
  ./ "${REMOTE}:${REMOTE_DIR}/"

tracked_run_artifacts=()
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  mapfile -t tracked_run_artifacts < <(git ls-files 'runs/*')
fi

if ((${#tracked_run_artifacts[@]} > 0)); then
  printf '%s\0' "${tracked_run_artifacts[@]}" |
    rsync -az --from0 --files-from=- ./ "${REMOTE}:${REMOTE_DIR}/"
fi

ssh "${REMOTE}" "cd ${REMOTE_DIR} && git status --short && git rev-parse --short HEAD"
