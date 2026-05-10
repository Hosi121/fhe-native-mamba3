#!/usr/bin/env bash
#SBATCH --job-name=openfhe-rec-chain
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:08:00
#SBATCH --output=slurm/openfhe_recurrence_chain_%j.out
#SBATCH --error=slurm/openfhe_recurrence_chain_%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

PYTHON="${PYTHON:-${PWD}/.venv/bin/python}"
RUN_NAME="${RUN_NAME:-openfhe-recurrence-chain-${SLURM_JOB_ID:-manual}}"
OUTPUT_JSON="${OUTPUT_JSON:-runs/${RUN_NAME}.json}"
LAYERS="${LAYERS:-4}"
SEQ_LEN="${SEQ_LEN:-2}"
D_STATE="${D_STATE:-2}"
RANK="${RANK:-2}"
INPUT_MODE="${INPUT_MODE:-server-bx}"
READOUT_STRATEGY="${READOUT_STRATEGY:-rank-local}"
RING_DIM="${RING_DIM:-65536}"
MULTIPLICATIVE_DEPTH="${MULTIPLICATIVE_DEPTH:-28}"
SCALING_MOD_SIZE="${SCALING_MOD_SIZE:-40}"
BOOTSTRAP_AFTER_LAYERS="${BOOTSTRAP_AFTER_LAYERS:-2}"
BOOTSTRAP_LEVEL_BUDGET="${BOOTSTRAP_LEVEL_BUDGET:-5,4}"
BOOTSTRAP_DIM1="${BOOTSTRAP_DIM1:-0,0}"
BOOTSTRAP_SLOTS="${BOOTSTRAP_SLOTS:-}"
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
    "stage": "openfhe-recurrence-ciphertext-chain-smoke",
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

section "Run recurrence ciphertext chain smoke"
optional_args=()
if [[ -n "${BOOTSTRAP_SLOTS}" ]]; then
  optional_args+=(--bootstrap-slots "${BOOTSTRAP_SLOTS}")
fi

"${PYTHON}" scripts/run_openfhe_recurrence_chain_smoke.py \
  --backend openfhe \
  --layers "${LAYERS}" \
  --seq-len "${SEQ_LEN}" \
  --d-state "${D_STATE}" \
  --rank "${RANK}" \
  --input-mode "${INPUT_MODE}" \
  --readout-strategy "${READOUT_STRATEGY}" \
  --ring-dim "${RING_DIM}" \
  --multiplicative-depth "${MULTIPLICATIVE_DEPTH}" \
  --scaling-mod-size "${SCALING_MOD_SIZE}" \
  --bootstrap-after-layers "${BOOTSTRAP_AFTER_LAYERS}" \
  --bootstrap-level-budget "${BOOTSTRAP_LEVEL_BUDGET}" \
  --bootstrap-dim1 "${BOOTSTRAP_DIM1}" \
  --bootstrap-correction-factor "${BOOTSTRAP_CORRECTION_FACTOR}" \
  "${optional_args[@]}" \
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
    "backend": payload["backend"],
    "encrypted": payload["encrypted"],
    "layers": payload["config"]["layers"],
    "seq_len": payload["config"]["seq_len"],
    "bootstrap_after_layers": result["bootstrap_after_layers"],
    "no_intermediate_decrypt": payload["no_intermediate_decrypt"],
    "ciphertext_chain": result["ciphertext_chain"],
    "encrypted_chain": result["encrypted_chain"],
    "max_abs_error": result["max_abs_error"],
    "latency_sec_per_token": result["latency_sec_per_token"],
    "operation_counts": result["backend_stats"],
}
print(json.dumps(summary, indent=2, sort_keys=True))
PY
