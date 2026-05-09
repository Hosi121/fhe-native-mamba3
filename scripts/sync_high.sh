#!/usr/bin/env bash
set -euo pipefail

REMOTE="${REMOTE:-high}"
REMOTE_DIR="${REMOTE_DIR:-~/cipher/fhe-native-mamba3}"

ssh "${REMOTE}" "mkdir -p ${REMOTE_DIR}"

rsync -az --delete \
  --exclude '.venv/' \
  --exclude '__pycache__/' \
  --exclude '.pytest_cache/' \
  --exclude '.ruff_cache/' \
  --exclude '*.egg-info/' \
  --exclude 'build/' \
  --exclude 'dist/' \
  --exclude 'checkpoints/' \
  --exclude 'runs/' \
  --exclude 'logs/' \
  ./ "${REMOTE}:${REMOTE_DIR}/"

ssh "${REMOTE}" "cd ${REMOTE_DIR} && git status --short && git rev-parse --short HEAD"
