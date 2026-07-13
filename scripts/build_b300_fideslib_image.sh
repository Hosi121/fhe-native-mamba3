#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${IMAGE:-fhemamba-b300:cuda13.0-fideslib}"
CUDA_IMAGE="${CUDA_IMAGE:-nvidia/cuda:13.0.1-devel-ubuntu24.04}"

docker build \
  --build-arg "CUDA_IMAGE=${CUDA_IMAGE}" \
  --file "${REPO_DIR}/docker/b300-fideslib.Dockerfile" \
  --tag "${IMAGE}" \
  "${REPO_DIR}"
