#!/usr/bin/env bash
#SBATCH --job-name=checkpoint-sketch-trace
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=slurm/checkpoint_source_sketch_trace_%j.out
#SBATCH --error=slurm/checkpoint_source_sketch_trace_%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

PYTHON="${PYTHON:-${PWD}/.venv/bin/python}"
RUN_NAME="${RUN_NAME:-checkpoint-source-sketch-trace-${SLURM_JOB_ID:-manual}}"
OUTPUT_JSON="${OUTPUT_JSON:-runs/${RUN_NAME}.json}"
CHECKPOINT="${CHECKPOINT:?set CHECKPOINT to a checkpoint path}"
STATE_DICT_KEY="${STATE_DICT_KEY:-}"
MAP_LOCATION="${MAP_LOCATION:-cpu}"
D_STATE="${D_STATE:-0}"
MIMO_RANK="${MIMO_RANK:-0}"
LAYER_INDEX="${LAYER_INDEX:-0}"
PROMPT="${PROMPT:-1}"
RANK_LIMIT="${RANK_LIMIT:-8}"
RANK_INDICES="${RANK_INDICES:-}"
ALL_RANKS="${ALL_RANKS:-0}"
NORM_EPS="${NORM_EPS:-1e-5}"
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

section "Run checkpoint source sketch trace"
state_dict_args=()
if [[ -n "${STATE_DICT_KEY}" ]]; then
  state_dict_args=(--state-dict-key "${STATE_DICT_KEY}")
fi
rank_args=()
if [[ -n "${RANK_INDICES}" ]]; then
  rank_args=(--rank-indices "${RANK_INDICES}")
elif [[ "${ALL_RANKS}" == "1" ]]; then
  rank_args=(--all-ranks)
else
  rank_args=(--rank-limit "${RANK_LIMIT}")
fi

"${PYTHON}" scripts/run_checkpoint_source_sketch_trace.py "${CHECKPOINT}" \
  "${state_dict_args[@]}" \
  --map-location "${MAP_LOCATION}" \
  --d-state "${D_STATE}" \
  --mimo-rank "${MIMO_RANK}" \
  --layer-index "${LAYER_INDEX}" \
  --prompt "${PROMPT}" \
  "${rank_args[@]}" \
  --norm-eps "${NORM_EPS}" \
  --max-plan-layers "${MAX_PLAN_LAYERS}" \
  --output-json "${OUTPUT_JSON}"

section "Summary"
"${PYTHON}" - "${OUTPUT_JSON}" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
result = payload["result"]
summary = {
    "version": payload["version"],
    "passed": payload["passed"],
    "layer_index": result["layer_index"],
    "d_state": result["d_state"],
    "mimo_rank": result["mimo_rank"],
    "rank_indices": result["rank_indices"],
    "trajectory_count": result["trajectory_count"],
    "seq_len": result["seq_len"],
    "decay_kind": result["decay_kind"],
}
print(json.dumps(summary, indent=2, sort_keys=True))
PY
