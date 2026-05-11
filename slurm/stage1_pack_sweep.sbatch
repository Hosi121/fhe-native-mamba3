#!/usr/bin/env bash
#SBATCH --job-name=stage1-pack-sweep
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:12:00
#SBATCH --output=slurm/stage1_pack_sweep_%j.out
#SBATCH --error=slurm/stage1_pack_sweep_%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

PYTHON="${PYTHON:-${PWD}/.venv/bin/python}"
RUN_NAME="${RUN_NAME:-stage1-pack-sweep-${SLURM_JOB_ID:-manual}}"
OUTPUT_JSON="${OUTPUT_JSON:-runs/${RUN_NAME}.json}"
BACKEND="${BACKEND:-tracking}"
HEAD_COUNT="${HEAD_COUNT:-32}"
D_STATE="${D_STATE:-64}"
D_MODEL="${D_MODEL:-768}"
SEQ_LEN="${SEQ_LEN:-5}"
SCAN_LEN="${SCAN_LEN:-256}"
SLOT_COUNT="${SLOT_COUNT:-32768}"
CANDIDATE_PACK_SIZES="${CANDIDATE_PACK_SIZES:-4,8,16,32}"
READOUT_STRATEGY="${READOUT_STRATEGY:-rank-local}"
KEY_SIZE_MB="${KEY_SIZE_MB:-200}"
MAX_KEY_MEMORY_GIB="${MAX_KEY_MEMORY_GIB:-80}"
BOOTSTRAP_LATENCY_JSON="${BOOTSTRAP_LATENCY_JSON:-}"
MULTIPLICATIVE_DEPTH="${MULTIPLICATIVE_DEPTH:-16}"
SCALING_MOD_SIZE="${SCALING_MOD_SIZE:-50}"
RING_DIMENSION="${RING_DIMENSION:-65536}"
ATOL="${ATOL:-1e-10}"
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
    "stage": "stage1-head-pack-readout-sweep",
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

if [[ "${BACKEND}" == "openfhe" ]]; then
  if ! "${PYTHON}" - <<'PY'
import openfhe  # noqa: F401
PY
  then
    if [[ "${INSTALL_OPENFHE}" == "1" ]]; then
      section "Install OpenFHE Python wheel"
      "${PYTHON}" -m pip install 'openfhe==1.5.1.0.24.4'
    else
      write_skip_json "openfhe Python binding is not installed; set INSTALL_OPENFHE=1"
      exit 0
    fi
  fi
fi

section "Run Stage 1 pack sweep"
bootstrap_latency_args=()
if [[ -n "${BOOTSTRAP_LATENCY_JSON}" ]]; then
  bootstrap_latency_args=(--bootstrap-latency-json "${BOOTSTRAP_LATENCY_JSON}")
fi

"${PYTHON}" scripts/run_stage1_pack_sweep.py \
  --backend "${BACKEND}" \
  --head-count "${HEAD_COUNT}" \
  --d-state "${D_STATE}" \
  --d-model "${D_MODEL}" \
  --seq-len "${SEQ_LEN}" \
  --scan-len "${SCAN_LEN}" \
  --slot-count "${SLOT_COUNT}" \
  --candidate-pack-sizes "${CANDIDATE_PACK_SIZES}" \
  --readout-strategy "${READOUT_STRATEGY}" \
  --key-size-mb "${KEY_SIZE_MB}" \
  --max-key-memory-gib "${MAX_KEY_MEMORY_GIB}" \
  "${bootstrap_latency_args[@]}" \
  --multiplicative-depth "${MULTIPLICATIVE_DEPTH}" \
  --scaling-mod-size "${SCALING_MOD_SIZE}" \
  --ring-dimension "${RING_DIMENSION}" \
  --atol "${ATOL}" \
  --output-json "${OUTPUT_JSON}"

section "Summary"
"${PYTHON}" - "${OUTPUT_JSON}" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
summary = {
    "version": payload["version"],
    "backend": payload["measurement_scope"]["backend"],
    "encrypted": payload["measurement_scope"]["encrypted"],
    "passed": payload["passed"],
    "recommended_pack_size": payload["recommended_pack_size"],
    "rows": [
        {
            "pack_size": row["pack_size"],
            "passed": row["passed"],
            "max_abs_error": row["max_abs_error"],
            "eval_seconds": row["eval_seconds"],
            "estimated_total_scan_depth": row["estimated_total_scan_depth"],
            "full_inventory_rotation_key_count": row[
                "full_inventory_rotation_key_count"
            ],
            "estimated_key_memory_gib": row["estimated_key_memory_gib"],
            "amortized_bootstrap_latency_sec": row[
                "amortized_bootstrap_latency_sec"
            ],
        }
        for row in payload["rows"]
    ],
}
print(json.dumps(summary, indent=2, sort_keys=True))
PY
