#!/usr/bin/env bash
#SBATCH --job-name=mamba-source-profile
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:08:00
#SBATCH --output=slurm/mamba_source_profile_%j.out
#SBATCH --error=slurm/mamba_source_profile_%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

PYTHON="${PYTHON:-${PWD}/.venv/bin/python}"
CHECKPOINT="${CHECKPOINT:-checkpoints/mamba-130m-hf}"
RUN_NAME="${RUN_NAME:-mamba-source-profile-${SLURM_JOB_ID:-manual}}"
OUTPUT_JSON="${OUTPUT_JSON:-runs/${RUN_NAME}.json}"
STDOUT_JSON="${STDOUT_JSON:-runs/${RUN_NAME}.stdout.json}"
PROMPT="${PROMPT:-1,2,3,4}"
LAYER_COUNT="${LAYER_COUNT:-24}"
PROFILE_ALL_LAYERS="${PROFILE_ALL_LAYERS:-1}"
D_STATE="${D_STATE:-0}"
MIMO_RANK="${MIMO_RANK:-0}"
NORM_EPS="${NORM_EPS:-1e-5}"
POSITION_BUCKETS="${POSITION_BUCKETS:-4}"
HIGH_DECAY_THRESHOLD="${HIGH_DECAY_THRESHOLD:-0.95}"
TOP_K_EXAMPLES="${TOP_K_EXAMPLES:-5}"
MAX_PLAN_LAYERS="${MAX_PLAN_LAYERS:-8}"

mkdir -p runs slurm
export PYTHONPATH="${PWD}/src:${PYTHONPATH:-}"

section() {
  printf '\n==== %s ====\n' "$1"
}

section "Environment"
echo "job_id=${SLURM_JOB_ID:-manual}"
echo "node_list=${SLURM_JOB_NODELIST:-local}"
echo "date_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
"${PYTHON}" --version
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi -L || true
fi

section "Run checkpoint source profile"
layer_args=(--layer-count "${LAYER_COUNT}")
if [[ "${PROFILE_ALL_LAYERS}" == "1" ]]; then
  layer_args=(--all-layers)
fi
shape_args=()
if [[ "${D_STATE}" != "0" ]]; then
  shape_args+=(--d-state "${D_STATE}")
fi
if [[ "${MIMO_RANK}" != "0" ]]; then
  shape_args+=(--mimo-rank "${MIMO_RANK}")
fi

"${PYTHON}" scripts/run_checkpoint_source_profile.py "${CHECKPOINT}" \
  "${layer_args[@]}" \
  "${shape_args[@]}" \
  --prompt "${PROMPT}" \
  --norm-eps "${NORM_EPS}" \
  --position-buckets "${POSITION_BUCKETS}" \
  --high-decay-threshold "${HIGH_DECAY_THRESHOLD}" \
  --top-k-examples "${TOP_K_EXAMPLES}" \
  --max-plan-layers "${MAX_PLAN_LAYERS}" \
  --output-json "${OUTPUT_JSON}" > "${STDOUT_JSON}"

section "Summary"
"${PYTHON}" - "${OUTPUT_JSON}" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
result = payload["result"]
layers = result["layers"]
summary = {
    "version": payload["version"],
    "passed": payload["passed"],
    "layer_count": result["layer_count"],
    "seq_len": len(result["token_ids"]),
    "d_model": result["d_model"],
    "d_state": result["d_state"],
    "mimo_rank": result["mimo_rank"],
    "elapsed_sec": result["elapsed_sec"],
    "top1_token": result["top1_token"],
    "top1_top2_gap": result["top1_top2_gap"],
    "global_maxima": result["global_maxima"],
    "first_layer_range_score": layers[0]["range_score"] if layers else None,
    "first_layer_range_score_stage": layers[0]["range_score_stage"] if layers else None,
}
print(json.dumps(summary, indent=2, sort_keys=True))
PY
