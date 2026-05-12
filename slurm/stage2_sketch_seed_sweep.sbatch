#!/usr/bin/env bash
#SBATCH --job-name=stage2-sketch-seeds
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:08:00
#SBATCH --output=slurm/stage2_sketch_seed_sweep_%j.out
#SBATCH --error=slurm/stage2_sketch_seed_sweep_%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

PYTHON="${PYTHON:-${PWD}/.venv/bin/python}"
RUN_NAME="${RUN_NAME:-stage2-sketch-seed-sweep-${SLURM_JOB_ID:-manual}}"
OUTPUT_JSON="${OUTPUT_JSON:-runs/${RUN_NAME}.json}"
STATE_WIDTH="${STATE_WIDTH:-64}"
SEQ_LEN="${SEQ_LEN:-64}"
TRAJECTORY_COUNT="${TRAJECTORY_COUNT:-8}"
SKETCH_SIZES="${SKETCH_SIZES:-8,16,32,64}"
SEEDS="${SEEDS:-0,1,2,3,4}"
DECAY_CENTER="${DECAY_CENTER:-0.92}"
DECAY_JITTER="${DECAY_JITTER:-0.04}"
UPDATE_SCALE="${UPDATE_SCALE:-0.05}"
READOUT_SCALE="${READOUT_SCALE:-0.05}"
MAX_PAIRNORM_L2_ERROR="${MAX_PAIRNORM_L2_ERROR:-0.25}"
TRAJECTORY_JSON="${TRAJECTORY_JSON:-}"

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

section "Run Stage 2 multi-seed sketch sweep"
trajectory_args=()
if [[ -n "${TRAJECTORY_JSON}" ]]; then
  trajectory_args=(--trajectory-json "${TRAJECTORY_JSON}")
fi

"${PYTHON}" scripts/run_stage2_sketch_seed_sweep.py \
  --state-width "${STATE_WIDTH}" \
  --seq-len "${SEQ_LEN}" \
  --trajectory-count "${TRAJECTORY_COUNT}" \
  --sketch-sizes "${SKETCH_SIZES}" \
  --seeds "${SEEDS}" \
  --decay-center "${DECAY_CENTER}" \
  --decay-jitter "${DECAY_JITTER}" \
  --update-scale "${UPDATE_SCALE}" \
  --readout-scale "${READOUT_SCALE}" \
  --max-pairnorm-l2-error "${MAX_PAIRNORM_L2_ERROR}" \
  "${trajectory_args[@]}" \
  --output-json "${OUTPUT_JSON}"

section "Summary"
"${PYTHON}" - "${OUTPUT_JSON}" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
summary = {
    "version": payload["version"],
    "passed": payload["passed"],
    "seeds": payload["seeds"],
    "recommended_sketch_size": payload["recommended_sketch_size"],
    "rows": [
        {
            "sketch_size": row["sketch_size"],
            "compression_ratio": row["compression_ratio"],
            "pass_rate": row["pass_rate"],
            "all_passed": row["all_passed"],
            "median_pairnorm_l2_error": row["median_pairnorm_l2_error"],
            "max_pairnorm_l2_error": row["max_pairnorm_l2_error"],
        }
        for row in payload["rows"]
    ],
}
print(json.dumps(summary, indent=2, sort_keys=True))
PY
