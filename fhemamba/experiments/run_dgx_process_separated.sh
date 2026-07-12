#!/usr/bin/env bash
# Three-process fixed-vector Mamba handoff: client keygen/encrypt, server
# evaluation without a secret key, then client decrypt/correctness check.

set -euo pipefail
umask 077

source "${FHEMAMBA_REMOTE_ROOT:-$HOME/fhemamba}/logs/env.sh"
export CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-1}"

ROOT="${FHEMAMBA_REMOTE_ROOT:-$HOME/fhemamba}"
BINARY="${BINARY:-$ROOT/build_stage0/stage1_mamba2_decode_fideslib}"
INPUT_CHAIN="${INPUT_CHAIN:-$ROOT/m2_chain_payload_sqnewton_wiki512_t8}"
RESULTS_DIR="${RESULTS_DIR:-$ROOT/results}"
HANDOFF_ROOT="${HANDOFF_ROOT:-$ROOT/handoffs}"
RUN_TAG="${RUN_TAG:-process-separated-smoke}"
LAYERS="${LAYERS:-1}"
TOKENS="${TOKENS:-1}"
RING_DIM="${RING_DIM:-65536}"
MULTIPLICATIVE_DEPTH="${MULTIPLICATIVE_DEPTH:-44}"
SCALING_MOD_SIZE="${SCALING_MOD_SIZE:-59}"
FIRST_MOD_SIZE="${FIRST_MOD_SIZE:-60}"
SECURITY="${SECURITY:-not-set}"
ARTIFACT_VERSION="${ARTIFACT_VERSION:-0.4.4}"
REPO_COMMIT="${REPO_COMMIT:-working-tree}"
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

common_args=(
  --input-chain "$INPUT_CHAIN"
  --max-layers "$LAYERS"
  --tokens "$TOKENS"
  --ring-dim "$RING_DIM"
  --multiplicative-depth "$MULTIPLICATIVE_DEPTH"
  --scaling-mod-size "$SCALING_MOD_SIZE"
  --first-mod-size "$FIRST_MOD_SIZE"
  --security "$SECURITY"
  --bsgs-replicas "${BSGS_REPLICAS:-auto}"
  --rotation-keys "${ROTATION_KEYS:-compact}"
  --rotation-key-gib "${ROTATION_KEY_GIB:-45}"
  --pt-cache full
  --pt-cache-gib "${PT_CACHE_GIB:-20}"
  --encode-threads "${ENCODE_THREADS:-8}"
  --auto-bootstrap-headroom "${AUTO_HEADROOM:-4}"
  --residual-bootstrap-headroom "${RESIDUAL_HEADROOM:-0}"
  --carried-bootstrap-headroom "${CARRIED_HEADROOM:-0}"
  --meta-bts "${META_BTS:-1}"
  --meta-bts-alpha "${META_BTS_ALPHA:-5}"
  --state-meta-bts-alpha "${STATE_META_BTS_ALPHA:--1}"
  --refresh-recurrent-state-post "${REFRESH_RECURRENT_STATE_POST:-1}"
  --tolerance "${TOLERANCE:-0.05}"
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
