#!/usr/bin/env bash
#SBATCH --job-name=checkpoint-sketch-matrix
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=slurm/checkpoint_sketch_matrix_%j.out
#SBATCH --error=slurm/checkpoint_sketch_matrix_%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

PYTHON="${PYTHON:-${PWD}/.venv/bin/python}"
RUN_NAME="${RUN_NAME:-checkpoint-sketch-matrix-${SLURM_JOB_ID:-manual}}"
OUTPUT_JSON="${OUTPUT_JSON:-runs/${RUN_NAME}.json}"
CHECKPOINT="${CHECKPOINT:?set CHECKPOINT to a checkpoint path}"
STATE_DICT_KEY="${STATE_DICT_KEY:-}"
MAP_LOCATION="${MAP_LOCATION:-cpu}"
D_STATE="${D_STATE:-0}"
MIMO_RANK="${MIMO_RANK:-0}"
LAYER_INDICES="${LAYER_INDICES:-0}"
PROMPT_SETS="${PROMPT_SETS:-default:1}"
RANK_STRATEGIES="${RANK_STRATEGIES:-first:8}"
SKETCH_SIZES="${SKETCH_SIZES:-4,8,16}"
SEEDS="${SEEDS:-0,1,2,3,4}"
MAX_PAIRNORM_L2_ERROR="${MAX_PAIRNORM_L2_ERROR:-0.25}"
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

section "Run checkpoint sketch matrix"
state_dict_args=()
if [[ -n "${STATE_DICT_KEY}" ]]; then
  state_dict_args=(--state-dict-key "${STATE_DICT_KEY}")
fi

prompt_args=()
IFS=';' read -r -a prompt_items <<< "${PROMPT_SETS}"
for prompt_item in "${prompt_items[@]}"; do
  if [[ -n "${prompt_item}" ]]; then
    prompt_args+=(--prompt-set "${prompt_item}")
  fi
done

"${PYTHON}" scripts/run_checkpoint_sketch_matrix.py "${CHECKPOINT}" \
  "${state_dict_args[@]}" \
  --map-location "${MAP_LOCATION}" \
  --d-state "${D_STATE}" \
  --mimo-rank "${MIMO_RANK}" \
  --layer-indices "${LAYER_INDICES}" \
  "${prompt_args[@]}" \
  --rank-strategies "${RANK_STRATEGIES}" \
  --sketch-sizes "${SKETCH_SIZES}" \
  --seeds "${SEEDS}" \
  --max-pairnorm-l2-error "${MAX_PAIRNORM_L2_ERROR}" \
  --norm-eps "${NORM_EPS}" \
  --max-plan-layers "${MAX_PLAN_LAYERS}" \
  --output-json "${OUTPUT_JSON}"

section "Summary"
"${PYTHON}" - "${OUTPUT_JSON}" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
rows = payload["rows"]
summary_rows = [
    {
        "layer": row["layer_index"],
        "prompt": row["prompt_name"],
        "rank_strategy": row["rank_strategy"],
        "recommended": row["seed_sweep"]["recommended_sketch_size"],
        "passed": row["passed"],
        "worst_error": max(
            sweep_row["max_pairnorm_l2_error"] for sweep_row in row["seed_sweep"]["rows"]
        ),
    }
    for row in rows
]
summary = {
    "version": payload["version"],
    "passed": payload["passed"],
    "row_count": payload["row_count"],
    "layers": payload["layer_indices"],
    "prompts": payload["prompt_names"],
    "rank_strategies": payload["rank_strategies"],
    "rows": summary_rows,
}
print(json.dumps(summary, indent=2, sort_keys=True))
PY
