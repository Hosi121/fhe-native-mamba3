#!/usr/bin/env bash
# Submit the Stage 0 high/SLURM measurement suite.

set -euo pipefail

cd "${REPO_DIR:-$(pwd)}"

PYTHON="${PYTHON:-${PWD}/.venv/bin/python}"
CHECKPOINT="${CHECKPOINT:-checkpoints/mamba-130m-hf}"
export PYTHONPATH="${PWD}/src:${PYTHONPATH:-}"
RUN_PREFIX="${RUN_PREFIX:-stage0-$("${PYTHON}" - <<'PY'
from fhe_native_mamba3 import __version__

print(f"v{__version__}")
PY
)}"
DRY_RUN="${DRY_RUN:-0}"

SUBMIT_FULL_LAYER_GATE="${SUBMIT_FULL_LAYER_GATE:-1}"
SUBMIT_RECURRENCE_CHAIN="${SUBMIT_RECURRENCE_CHAIN:-1}"
SUBMIT_BOOTSTRAP_LATENCY="${SUBMIT_BOOTSTRAP_LATENCY:-1}"
SUBMIT_FULL_LAYER_SWEEP="${SUBMIT_FULL_LAYER_SWEEP:-1}"
SUBMIT_VISIBLE_PROJECTION_SWEEP="${SUBMIT_VISIBLE_PROJECTION_SWEEP:-1}"
SUBMIT_ALL_LAYER_RECURRENCE="${SUBMIT_ALL_LAYER_RECURRENCE:-1}"
SUBMIT_SOURCE_PROFILE="${SUBMIT_SOURCE_PROFILE:-1}"

if [[ "${DRY_RUN}" != "1" ]] && ! command -v sbatch >/dev/null 2>&1; then
  echo "sbatch is not available; run this script on the high SLURM login node" >&2
  exit 1
fi

mkdir -p runs slurm

run_cmd() {
  local label="$1"
  shift
  printf '\n==== submit %s ====\n' "${label}"
  if [[ "${DRY_RUN}" == "1" ]]; then
    printf '%q ' "$@"
    printf '\n'
    return 0
  fi
  "$@"
}

if [[ "${SUBMIT_FULL_LAYER_GATE}" == "1" ]]; then
  run_cmd "OpenFHE full-layer gate" \
    env \
      PYTHON="${PYTHON}" \
      CHECKPOINT="${CHECKPOINT}" \
      RUN_NAME="${RUN_PREFIX}-openfhe-full-layer-gate-l0-vis8" \
      BACKEND=openfhe \
      LAYER_INDEX=0 \
      PROMPT=1 \
      VISIBLE_DIM_LIMIT=8 \
      RING_DIM=65536 \
      MULTIPLICATIVE_DEPTH=16 \
      SCALING_MOD_SIZE=40 \
      MAX_ROTATION_KEYS=256 \
      sbatch slurm/mamba_checkpoint_full_layer_gate.sbatch
fi

if [[ "${SUBMIT_RECURRENCE_CHAIN}" == "1" ]]; then
  run_cmd "OpenFHE recurrence ciphertext chain" \
    env \
      PYTHON="${PYTHON}" \
      RUN_NAME="${RUN_PREFIX}-openfhe-rec-chain-small" \
      LAYERS=4 \
      SEQ_LEN=2 \
      D_STATE=2 \
      RANK=2 \
      INPUT_MODE=server-bx \
      BOOTSTRAP_AFTER_LAYERS=2 \
      RING_DIM=65536 \
      sbatch slurm/openfhe_recurrence_chain.sbatch
fi

if [[ "${SUBMIT_BOOTSTRAP_LATENCY}" == "1" ]]; then
  run_cmd "OpenFHE bootstrap latency" \
    env \
      PYTHON="${PYTHON}" \
      RUN_NAME="${RUN_PREFIX}-bootstrap-latency-b16" \
      BATCH_SIZE=16 \
      RING_DIM=65536 \
      ITERATIONS=1 \
      WARMUPS=0 \
      sbatch slurm/openfhe_bootstrap_latency.sbatch
fi

if [[ "${SUBMIT_FULL_LAYER_SWEEP}" == "1" ]]; then
  run_cmd "tracking full-layer sweep" \
    env \
      PYTHON="${PYTHON}" \
      CHECKPOINT="${CHECKPOINT}" \
      RUN_NAME="${RUN_PREFIX}-full-layer-sweep-tracking-vis16" \
      BACKEND=tracking \
      LAYER_COUNT=24 \
      PROMPT=1 \
      VISIBLE_DIM_LIMIT=16 \
      MAX_ROTATION_KEYS=512 \
      sbatch slurm/mamba_checkpoint_full_layer_sweep.sbatch
fi

if [[ "${SUBMIT_VISIBLE_PROJECTION_SWEEP}" == "1" ]]; then
  run_cmd "OpenFHE visible projection sweep" \
    env \
      PYTHON="${PYTHON}" \
      CHECKPOINT="${CHECKPOINT}" \
      RUN_NAME="${RUN_PREFIX}-visible-proj-openfhe-8-16" \
      BACKEND=openfhe \
      VISIBLE_DIM_LIMITS=8,16 \
      LAYER_INDEX=0 \
      PROMPT=1 \
      MAX_ROTATION_KEYS=256 \
      RING_DIM=65536 \
      sbatch slurm/mamba_checkpoint_visible_projection_sweep.sbatch
fi

if [[ "${SUBMIT_ALL_LAYER_RECURRENCE}" == "1" ]]; then
  run_cmd "OpenFHE all-layer recurrence" \
    env \
      PYTHON="${PYTHON}" \
      CHECKPOINT="${CHECKPOINT}" \
      RUN_NAME="${RUN_PREFIX}-openfhe-all-layer-recurrence-24" \
      N_LAYERS=24 \
      PROMPT=1,2,3,4 \
      EXECUTE_SCHEDULED_BOOTSTRAP=0 \
      sbatch slurm/openfhe_all_layer_recurrence.sbatch
fi

if [[ "${SUBMIT_SOURCE_PROFILE}" == "1" ]]; then
  run_cmd "checkpoint source profile" \
    env \
      PYTHON="${PYTHON}" \
      CHECKPOINT="${CHECKPOINT}" \
      RUN_NAME="${RUN_PREFIX}-source-profile-24" \
      PROMPT=1,2,3,4 \
      PROFILE_ALL_LAYERS=1 \
      sbatch slurm/mamba_checkpoint_source_profile.sbatch
fi

if [[ "${DRY_RUN}" != "1" ]]; then
  printf '\n==== queue ====\n'
  squeue -u "${USER}" -o "%.18i %.24j %.8T %.10M %.6D %R"
fi
