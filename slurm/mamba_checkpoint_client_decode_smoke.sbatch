#!/usr/bin/env bash
#SBATCH --job-name=mamba-client-decode
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:08:00
#SBATCH --output=slurm/mamba_client_decode_%j.out
#SBATCH --error=slurm/mamba_client_decode_%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

PYTHON="${PYTHON:-${PWD}/.venv/bin/python}"
CHECKPOINT="${CHECKPOINT:-checkpoints/mamba-130m-hf}"
RUN_NAME="${RUN_NAME:-mamba-client-decode-${SLURM_JOB_ID:-manual}}"
OUTPUT_JSON="${OUTPUT_JSON:-runs/${RUN_NAME}.json}"
STDOUT_JSON="${STDOUT_JSON:-runs/${RUN_NAME}.stdout.json}"
PROMPT="${PROMPT:-1}"
STEPS="${STEPS:-1}"
LAYER_COUNT="${LAYER_COUNT:-24}"
DECODE_ALL_LAYERS="${DECODE_ALL_LAYERS:-1}"
D_STATE="${D_STATE:-0}"
MIMO_RANK="${MIMO_RANK:-0}"
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

section "Run checkpoint client decode smoke"
layer_args=(--layer-count "${LAYER_COUNT}")
if [[ "${DECODE_ALL_LAYERS}" == "1" ]]; then
  layer_args=(--all-layers)
fi
shape_args=()
if [[ "${D_STATE}" != "0" ]]; then
  shape_args+=(--d-state "${D_STATE}")
fi
if [[ "${MIMO_RANK}" != "0" ]]; then
  shape_args+=(--mimo-rank "${MIMO_RANK}")
fi

"${PYTHON}" scripts/run_checkpoint_client_decode_smoke.py "${CHECKPOINT}" \
  "${layer_args[@]}" \
  "${shape_args[@]}" \
  --prompt "${PROMPT}" \
  --steps "${STEPS}" \
  --norm-eps "${NORM_EPS}" \
  --max-plan-layers "${MAX_PLAN_LAYERS}" \
  --output-json "${OUTPUT_JSON}" > "${STDOUT_JSON}"

section "Summary"
"${PYTHON}" - "${OUTPUT_JSON}" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
result = payload["result"]
first_step = result["decode_steps"][0] if result["decode_steps"] else {}
summary = {
    "version": payload["version"],
    "passed": payload["passed"],
    "layer_count": result["layer_count"],
    "prompt_token_ids": result["prompt_token_ids"],
    "new_token_ids": result["new_token_ids"],
    "steps": len(result["decode_steps"]),
    "d_model": result["d_model"],
    "d_state": result["d_state"],
    "mimo_rank": result["mimo_rank"],
    "vocab_size": result["vocab_size"],
    "elapsed_sec": result["elapsed_sec"],
    "hidden_abs_max": result["hidden_abs_max"],
    "logits_abs_max": result["logits_abs_max"],
    "top1_top2_gap": first_step.get("top1_top2_gap"),
    "client_side_lm_head": result["client_side_lm_head"],
    "client_side_argmax": result["client_side_argmax"],
    "encrypted_argmax": result["encrypted_argmax"],
}
print(json.dumps(summary, indent=2, sort_keys=True))
PY
