#!/usr/bin/env bash
#SBATCH --job-name=openfhe-segment-samples
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:10:00
#SBATCH --output=slurm/openfhe_segment_samples_%j.out
#SBATCH --error=slurm/openfhe_segment_samples_%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

PYTHON="${PYTHON:-${PWD}/.venv/bin/python}"
SWEEP_JSON="${SWEEP_JSON:-runs/real-mamba-130m-source-scale-plan-bootstrap-segments-sweep-v057.json}"
CHECKPOINT="${CHECKPOINT:-checkpoints/mamba-130m-hf}"
SCALE_PLAN_JSON="${SCALE_PLAN_JSON:-runs/real-mamba-130m-hf-source-stack-scale-plan-v047.json}"
RUN_NAME="${RUN_NAME:-openfhe-segment-samples-${SLURM_JOB_ID:-manual}}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/${RUN_NAME}}"
OUTPUT_JSON="${OUTPUT_JSON:-runs/${RUN_NAME}.json}"
RUN_PREFIX="${RUN_PREFIX:-${RUN_NAME}}"
PROMPT="${PROMPT:-1}"
N_LAYERS="${N_LAYERS:-24}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-8}"
INPUT_PROPAGATION="${INPUT_PROPAGATION:-source}"
SCALING_MOD_SIZE="${SCALING_MOD_SIZE:-40}"
MULTIPLICATIVE_DEPTH_OVERRIDE="${MULTIPLICATIVE_DEPTH_OVERRIDE:-28}"
LIMIT="${LIMIT:-1}"
OFFSET="${OFFSET:-0}"
BOOTSTRAP_AFTER_TOKENS="${BOOTSTRAP_AFTER_TOKENS:-}"
BOOTSTRAP_EVERY_TOKENS="${BOOTSTRAP_EVERY_TOKENS:-0}"
RING_DIM="${RING_DIM:-65536}"
BOOTSTRAP_CORRECTION_FACTOR="${BOOTSTRAP_CORRECTION_FACTOR:-20}"
INSTALL_OPENFHE="${INSTALL_OPENFHE:-0}"

mkdir -p runs slurm
export PYTHONPATH="${PWD}/src:${PYTHONPATH:-}"

section() {
  printf '\n==== %s ====\n' "$1"
}

write_skip_json() {
  local reason="$1"
  "${PYTHON}" - "${OUTPUT_JSON}" "${reason}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = {
    "stage": "openfhe-segment-samples",
    "status": "skipped",
    "reason": sys.argv[2],
}
path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
print(json.dumps(payload, indent=2, sort_keys=True))
PY
}

section "Environment"
echo "job_id=${SLURM_JOB_ID:-manual}"
echo "node_list=${SLURM_JOB_NODELIST:-local}"
echo "date_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
"${PYTHON}" --version
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi -L || true
fi

if ! "${PYTHON}" - <<'PY'
import openfhe  # noqa: F401
PY
then
  if [[ "${INSTALL_OPENFHE}" == "1" ]]; then
    section "Install OpenFHE Python wheel"
    "${PYTHON}" -m pip install 'openfhe==1.5.1.0.24.4'
  else
    write_skip_json "openfhe Python binding is not installed; set INSTALL_OPENFHE=1 to install the wheel"
    exit 0
  fi
fi

section "Run OpenFHE segment samples"
args=()
if [[ -f "${SCALE_PLAN_JSON}" ]]; then
  args+=(--scale-plan-json "${SCALE_PLAN_JSON}")
fi
if [[ -n "${BOOTSTRAP_AFTER_TOKENS}" ]]; then
  args+=(--bootstrap-after-tokens "${BOOTSTRAP_AFTER_TOKENS}")
fi
if [[ "${BOOTSTRAP_EVERY_TOKENS}" != "0" ]]; then
  args+=(--bootstrap-every-tokens "${BOOTSTRAP_EVERY_TOKENS}")
fi

"${PYTHON}" scripts/run_openfhe_segment_samples.py "${SWEEP_JSON}" \
  --checkpoint "${CHECKPOINT}" \
  "${args[@]}" \
  --output-dir "${OUTPUT_DIR}" \
  --output-json "${OUTPUT_JSON}" \
  --run-prefix "${RUN_PREFIX}" \
  --prompt "${PROMPT}" \
  --n-layers "${N_LAYERS}" \
  --max-seq-len "${MAX_SEQ_LEN}" \
  --input-propagation "${INPUT_PROPAGATION}" \
  --scaling-mod-size "${SCALING_MOD_SIZE}" \
  --multiplicative-depth-override "${MULTIPLICATIVE_DEPTH_OVERRIDE}" \
  --ring-dim "${RING_DIM}" \
  --bootstrap-correction-factor "${BOOTSTRAP_CORRECTION_FACTOR}" \
  --limit "${LIMIT}" \
  --offset "${OFFSET}"

section "Summary"
"${PYTHON}" - "${OUTPUT_JSON}" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
summary = {
    "stage": payload.get("stage"),
    "sample_count": payload.get("sample_count"),
    "success_count": payload.get("success_count"),
    "results": [
        {
            "layer_index": result.get("layer_index"),
            "segment_index": result.get("segment_index"),
            "returncode": result.get("returncode"),
            "latency_sec_per_token": result.get("latency_sec_per_token"),
            "max_abs_error": result.get("max_abs_error"),
            "bootstraps": result.get("operation_counts", {}).get("bootstraps"),
        }
        for result in payload.get("results", [])
    ],
}
print(json.dumps(summary, indent=2, sort_keys=True))
PY
