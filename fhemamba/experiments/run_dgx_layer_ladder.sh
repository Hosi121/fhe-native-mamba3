#!/usr/bin/env bash
# Run the zero-intermediate-decrypt layer-depth ladder without stopping at the
# first tolerance failure. Intended for the single-GPU DGX Spark campaign.

set -u -o pipefail

source "${FHEMAMBA_REMOTE_ROOT:-$HOME/fhemamba}/logs/env.sh"
export CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-1}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/dgx_mamba2_common.sh"
init_dgx_mamba2_defaults

LOG_DIR="${LOG_DIR:-$ROOT/logs}"
LAYERS="${LAYERS:-5 8 12 24}"
TOKENS="${TOKENS:-1}"
RUN_TAG="${RUN_TAG:-structural-metabts-i1-cache5}"

mkdir -p "$RESULTS_DIR" "$LOG_DIR"
BINARY_SHA256="$(sha256sum "$BINARY" | awk '{print $1}')"

overall=0
for layers in $LAYERS; do
  name="m2_chain_${RUN_TAG}_l${layers}_t${TOKENS}"
  result="$RESULTS_DIR/$name.json"
  rm -f "$result"
  echo "=== $name start $(date --iso-8601=seconds) ==="
  build_dgx_mamba2_args "$layers" "$TOKENS"
  "$BINARY" "${DGX_MAMBA2_ARGS[@]}" \
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
