#!/usr/bin/env bash
# Run the zero-intermediate-decrypt layer-depth ladder without stopping at the
# first tolerance failure. Intended for the single-GPU DGX Spark campaign.

set -u -o pipefail

source "${FHEMAMBA_REMOTE_ROOT:-$HOME/fhemamba}/logs/env.sh"
export CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-1}"

ROOT="${FHEMAMBA_REMOTE_ROOT:-$HOME/fhemamba}"
BINARY="${BINARY:-$ROOT/build_stage0/stage1_mamba2_decode_fideslib}"
INPUT_CHAIN="${INPUT_CHAIN:-$ROOT/m2_chain_payload_sqnewton_wiki512_t8}"
RESULTS_DIR="${RESULTS_DIR:-$ROOT/results}"
LOG_DIR="${LOG_DIR:-$ROOT/logs}"
LAYERS="${LAYERS:-5 8 12 24}"
TOKENS="${TOKENS:-1}"
RUN_TAG="${RUN_TAG:-sqnewton-bts18-h8-c0}"
ARTIFACT_VERSION="${ARTIFACT_VERSION:-0.4.4}"
REPO_COMMIT="${REPO_COMMIT:-working-tree}"
RING_DIM="${RING_DIM:-65536}"
MULTIPLICATIVE_DEPTH="${MULTIPLICATIVE_DEPTH:-44}"
SCALING_MOD_SIZE="${SCALING_MOD_SIZE:-59}"
FIRST_MOD_SIZE="${FIRST_MOD_SIZE:-60}"
SECURITY="${SECURITY:-not-set}"
AUTO_HEADROOM="${AUTO_HEADROOM:-8}"
RESIDUAL_HEADROOM="${RESIDUAL_HEADROOM:-0}"
CARRIED_HEADROOM="${CARRIED_HEADROOM:-0}"
META_BTS="${META_BTS:-1}"
META_BTS_ALPHA="${META_BTS_ALPHA:-12}"
STATE_META_BTS_ALPHA="${STATE_META_BTS_ALPHA:--1}"
META_BTS_RESIDUAL_ALIGN_MODE="${META_BTS_RESIDUAL_ALIGN_MODE:-unity}"
BOOTSTRAP_LEVEL_BUDGET_CTS="${BOOTSTRAP_LEVEL_BUDGET_CTS:-5}"
BOOTSTRAP_LEVEL_BUDGET_STC="${BOOTSTRAP_LEVEL_BUDGET_STC:-5}"
BOOTSTRAP_NORM_MARGIN="${BOOTSTRAP_NORM_MARGIN:-1.1}"
STATE_BOOTSTRAP_MARGIN="${STATE_BOOTSTRAP_MARGIN:-1.1}"
TOLERANCE="${TOLERANCE:-0.05}"
ROTATION_KEYS="${ROTATION_KEYS:-compact}"
ROTATION_KEY_GIB="${ROTATION_KEY_GIB:-45}"
LEVEL_ALIGN_MODE="${LEVEL_ALIGN_MODE:-unity}"
BSGS_REPLICAS="${BSGS_REPLICAS:-auto}"
REPLICATED_TRUE_BSGS="${REPLICATED_TRUE_BSGS:-0}"
REPLICATED_STATE_BLOCKS="${REPLICATED_STATE_BLOCKS:-0}"
PROJECTION_LATE_LEVEL="${PROJECTION_LATE_LEVEL:-0}"
PT_CACHE_LEVEL="${PT_CACHE_LEVEL:-0}"
PT_CACHE_WEIGHT_LEVEL="${PT_CACHE_WEIGHT_LEVEL:-0}"
PT_MISS_CONSUMPTION_LEVEL="${PT_MISS_CONSUMPTION_LEVEL:-0}"
BOOTSTRAP_BEFORE_TOKEN="${BOOTSTRAP_BEFORE_TOKEN:-}"
DEBUG_LAYER_ERRORS="${DEBUG_LAYER_ERRORS:-0}"
DEBUG_CLIENT_REENCRYPT_BEFORE_TOKEN="${DEBUG_CLIENT_REENCRYPT_BEFORE_TOKEN:-}"
AUTOREGRESSIVE_CLIENT_LOOP="${AUTOREGRESSIVE_CLIENT_LOOP:-0}"
REFRESH_RECURRENT_STATE_POST="${REFRESH_RECURRENT_STATE_POST:-0}"
REFRESH_RECURRENT_STATE_POST_LAYERS="${REFRESH_RECURRENT_STATE_POST_LAYERS:-}"
STATE_REFRESH_INTERVAL="${STATE_REFRESH_INTERVAL:-0}"

mkdir -p "$RESULTS_DIR" "$LOG_DIR"
BINARY_SHA256="$(sha256sum "$BINARY" | awk '{print $1}')"

overall=0
for layers in $LAYERS; do
  name="m2_chain_${RUN_TAG}_l${layers}_t${TOKENS}"
  result="$RESULTS_DIR/$name.json"
  rm -f "$result"
  echo "=== $name start $(date --iso-8601=seconds) ==="
  "$BINARY" \
    --input-chain "$INPUT_CHAIN" \
    --max-layers "$layers" \
    --tokens "$TOKENS" \
    --ring-dim "$RING_DIM" \
    --multiplicative-depth "$MULTIPLICATIVE_DEPTH" \
    --scaling-mod-size "$SCALING_MOD_SIZE" \
    --first-mod-size "$FIRST_MOD_SIZE" \
    --bootstrap-level-budget-cts "$BOOTSTRAP_LEVEL_BUDGET_CTS" \
    --bootstrap-level-budget-stc "$BOOTSTRAP_LEVEL_BUDGET_STC" \
    --security "$SECURITY" \
    --bsgs-replicas "$BSGS_REPLICAS" \
    --replicated-true-bsgs "$REPLICATED_TRUE_BSGS" \
    --replicated-state-blocks "$REPLICATED_STATE_BLOCKS" \
    --projection-late-level "$PROJECTION_LATE_LEVEL" \
    --rotation-keys "$ROTATION_KEYS" \
    --rotation-key-gib "$ROTATION_KEY_GIB" \
    --level-align-mode "$LEVEL_ALIGN_MODE" \
    --pt-cache full \
    --pt-cache-gib "${PT_CACHE_GIB:-20}" \
    --pt-cache-level "$PT_CACHE_LEVEL" \
    --pt-cache-weight-level "$PT_CACHE_WEIGHT_LEVEL" \
    --pt-miss-consumption-level "$PT_MISS_CONSUMPTION_LEVEL" \
    --encode-threads "${ENCODE_THREADS:-8}" \
    --bootstrap-norm-margin "$BOOTSTRAP_NORM_MARGIN" \
    --state-bootstrap-margin "$STATE_BOOTSTRAP_MARGIN" \
    --auto-bootstrap-headroom "$AUTO_HEADROOM" \
    --residual-bootstrap-headroom "$RESIDUAL_HEADROOM" \
    --carried-bootstrap-headroom "$CARRIED_HEADROOM" \
    --meta-bts "$META_BTS" \
    --meta-bts-alpha "$META_BTS_ALPHA" \
    --state-meta-bts-alpha "$STATE_META_BTS_ALPHA" \
    --meta-bts-residual-align-mode "$META_BTS_RESIDUAL_ALIGN_MODE" \
    --bootstrap-before-token "$BOOTSTRAP_BEFORE_TOKEN" \
    --debug-layer-errors "$DEBUG_LAYER_ERRORS" \
    --debug-client-reencrypt-before-token "$DEBUG_CLIENT_REENCRYPT_BEFORE_TOKEN" \
    --autoregressive-client-loop "$AUTOREGRESSIVE_CLIENT_LOOP" \
    --refresh-recurrent-state-post "$REFRESH_RECURRENT_STATE_POST" \
    --refresh-recurrent-state-post-layers "$REFRESH_RECURRENT_STATE_POST_LAYERS" \
    --state-refresh-interval "$STATE_REFRESH_INTERVAL" \
    --tolerance "$TOLERANCE" \
    --artifact-version "$ARTIFACT_VERSION+$name" \
    --repo-commit "$REPO_COMMIT" \
    --binary-sha256 "$BINARY_SHA256" \
    --output-json "$result" \
    2>&1 | tee "$LOG_DIR/$name.log"
  rc=${PIPESTATUS[0]}
  if [[ ! -s "$result" ]]; then
    echo "ERROR: backend produced no artifact: $result"
    rc=1
  elif ! jq -e 'type == "object" and has("status") and has("passed")' \
      "$result" >/dev/null; then
    echo "ERROR: backend produced an invalid artifact: $result"
    rc=1
  elif [[ $rc -eq 0 ]] && ! jq -e '.status == "passed" and .passed == true' \
      "$result" >/dev/null; then
    echo "ERROR: backend exit code and artifact success status disagree: $result"
    rc=1
  fi
  echo "=== $name exit=$rc $(date --iso-8601=seconds) ==="
  if [[ $rc -ne 0 ]]; then
    overall=1
  fi
done

exit "$overall"
