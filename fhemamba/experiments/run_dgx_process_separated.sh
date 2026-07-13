#!/usr/bin/env bash
# Three-process fixed-vector Mamba handoff: client keygen/encrypt, server
# evaluation without a secret key, then client decrypt/correctness check.

set -euo pipefail
umask 077

source "${FHEMAMBA_REMOTE_ROOT:-$HOME/fhemamba}/logs/env.sh"
export CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-1}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/dgx_mamba2_common.sh"
init_dgx_mamba2_defaults

HANDOFF_ROOT="${HANDOFF_ROOT:-$ROOT/handoffs}"
RUN_TAG="${RUN_TAG:-process-separated-smoke}"
LAYERS="${LAYERS:-1}"
TOKENS="${TOKENS:-1}"
GPU_WAIT_TIMEOUT_SECONDS="${GPU_WAIT_TIMEOUT_SECONDS:-14400}"
GPU_POLL_SECONDS="${GPU_POLL_SECONDS:-30}"
BINARY_SHA256="$(sha256sum "$BINARY" | awk '{print $1}')"

waited=0
while [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]]; do
  if (( waited >= GPU_WAIT_TIMEOUT_SECONDS )); then
    echo "ERROR: GPU remained occupied for ${waited}s" >&2
    exit 1
  fi
  echo "GPU occupied; waiting ${GPU_POLL_SECONDS}s (${waited}s elapsed)"
  sleep "$GPU_POLL_SECONDS"
  waited=$((waited + GPU_POLL_SECONDS))
done

HANDOFF_DIR="$HANDOFF_ROOT/$RUN_TAG"
if [[ -e "$HANDOFF_DIR" ]]; then
  echo "ERROR: handoff directory already exists: $HANDOFF_DIR" >&2
  exit 1
fi
mkdir -p "$HANDOFF_ROOT" "$RESULTS_DIR"

build_dgx_mamba2_args "$LAYERS" "$TOKENS"
common_args=(
  "${DGX_MAMBA2_ARGS[@]}"
  --artifact-version "$ARTIFACT_VERSION+$RUN_TAG"
  --repo-commit "$REPO_COMMIT"
  --binary-sha256 "$BINARY_SHA256"
  --handoff-dir "$HANDOFF_DIR"
)

run_phase() {
  local role="$1"
  local artifact="$RESULTS_DIR/${RUN_TAG}_${role}.json"
  "$BINARY" "${common_args[@]}" \
    --process-role "$role" \
    --output-json "$artifact"
  jq -e '.status == "passed" and .passed == true' "$artifact" >/dev/null
}

run_phase client-init
if find "$HANDOFF_DIR/server" -type f -iname '*secret*' -print -quit | grep -q .; then
  echo "ERROR: secret-key-named file found in server directory" >&2
  exit 1
fi
run_phase server-eval
run_phase client-decrypt

echo "process-separated run passed: $RUN_TAG"
