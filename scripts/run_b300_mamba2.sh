#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/home/kataiwa/fhemamba-b300}"
IMAGE="${IMAGE:-fhemamba-b300:cuda12.8-fideslib}"
GPU_DEVICE="${GPU_DEVICE:-3}"
FIDESLIB_SM="${FIDESLIB_SM:-100}"
LAYERS="${LAYERS:-24}"
TOKENS="${TOKENS:-1}"
AUTOREGRESSIVE_CLIENT_LOOP="${AUTOREGRESSIVE_CLIENT_LOOP:-0}"
NORMALIZED_STATE_META_BTS="${NORMALIZED_STATE_META_BTS:-0}"
STATE_REFRESH_INTERVAL="${STATE_REFRESH_INTERVAL:-1}"
INPUT_CHAIN="${INPUT_CHAIN:-${ROOT_DIR}/payloads/m2_chain_payload_sqnewton_wiki512_t2}"
RESULTS_DIR="${RESULTS_DIR:-${ROOT_DIR}/results}"
META_BTS_RESIDUAL_LAYERS="${META_BTS_RESIDUAL_LAYERS:-21,22,23}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUTPUT_JSON="${OUTPUT_JSON:-${RESULTS_DIR}/m2_chain_b300-sm${FIDESLIB_SM}-l${LAYERS}-t${TOKENS}-${RUN_ID}.json}"

if [[ "${GPU_DEVICE}" != "2" && "${GPU_DEVICE}" != "3" ]]; then
  echo "GPU_DEVICE must be 2 or 3" >&2
  exit 2
fi
if [[ ! -x "${ROOT_DIR}/build/fideslib-stage0-sm${FIDESLIB_SM}/stage1_mamba2_decode_fideslib" ]]; then
  echo "missing sm${FIDESLIB_SM} stage binary; run launch_b300_fideslib_build.sh first" >&2
  exit 2
fi

mkdir -p "${RESULTS_DIR}"
repo_commit="$(git -C "${ROOT_DIR}/cipher" rev-parse --short HEAD 2>/dev/null || echo working-tree)"
if ! git -C "${ROOT_DIR}/cipher" diff --quiet --ignore-submodules=dirty 2>/dev/null; then
  repo_commit="${repo_commit}-dirty"
fi
binary_sha256="$(sha256sum "${ROOT_DIR}/build/fideslib-stage0-sm${FIDESLIB_SM}/stage1_mamba2_decode_fideslib" | cut -d' ' -f1)"
container_name="fhemamba-b300-${RUN_ID}"

docker run --rm \
  --name "${container_name}" \
  --gpus "device=${GPU_DEVICE}" \
  --ipc=host \
  --shm-size=32g \
  --volume "${ROOT_DIR}:/workspace" \
  --workdir /workspace \
  --env FHEMAMBA_REMOTE_ROOT=/workspace \
  --env BINARY="/workspace/build/fideslib-stage0-sm${FIDESLIB_SM}/stage1_mamba2_decode_fideslib" \
  --env INPUT_CHAIN="${INPUT_CHAIN/#${ROOT_DIR}/\/workspace}" \
  --env RESULTS_DIR=/workspace/results \
  --env META_BTS_RESIDUAL_LAYERS="${META_BTS_RESIDUAL_LAYERS}" \
  --env OUTPUT_JSON="${OUTPUT_JSON/#${ROOT_DIR}/\/workspace}" \
  --env REPO_COMMIT="${repo_commit}" \
  --env LAYERS="${LAYERS}" \
  --env TOKENS="${TOKENS}" \
  --env AUTOREGRESSIVE_CLIENT_LOOP="${AUTOREGRESSIVE_CLIENT_LOOP}" \
  --env NORMALIZED_STATE_META_BTS="${NORMALIZED_STATE_META_BTS}" \
  --env STATE_REFRESH_INTERVAL="${STATE_REFRESH_INTERVAL}" \
  --env BINARY_SHA256="${binary_sha256}" \
  --env LD_LIBRARY_PATH="/workspace/install/fideslib-sm${FIDESLIB_SM}/lib:/workspace/install/fideslib/lib:/workspace/install/openfhe-fides/lib:/workspace/install/openfhe-fides/lib64" \
  "${IMAGE}" \
  bash -lc '
    set -euo pipefail
    unset CUDA_LAUNCH_BLOCKING
    source /workspace/cipher/fhemamba/experiments/dgx_mamba2_common.sh
    init_dgx_mamba2_defaults
    build_dgx_mamba2_args "${LAYERS}" "${TOKENS}"
    "${BINARY}" "${DGX_MAMBA2_ARGS[@]}" \
      --output-json "${OUTPUT_JSON}" \
      --artifact-version "${ARTIFACT_VERSION}" \
      --repo-commit "${REPO_COMMIT}" \
      --binary-sha256 "${BINARY_SHA256}"
  '

echo "output_json=${OUTPUT_JSON}"
