#!/usr/bin/env bash
#SBATCH --job-name=fideslib-s1-rgate
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=00:20:00
#SBATCH --output=slurm/fideslib_stage1_rank_gate_projection_%j.out
#SBATCH --error=slurm/fideslib_stage1_rank_gate_projection_%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

CUDA_HOME="${CUDA_HOME:-$HOME/cuda12.8}"
export CUDA_HOME
export CUDA_PATH="${CUDA_PATH:-${CUDA_HOME}}"
export PATH="${CUDA_HOME}/bin:${PATH}"

OPENFHE_PREFIX="${OPENFHE_PREFIX:-$HOME/fhe-native-mamba3-deps/openfhe-fides}"
FIDESLIB_PREFIX="${FIDESLIB_PREFIX:-$HOME/fhe-native-mamba3-deps/fideslib-install}"
PYTHON="${PYTHON:-${SLURM_SUBMIT_DIR}/.venv/bin/python}"
BUILD_DIR="${BUILD_DIR:-${SLURM_SUBMIT_DIR}/build/fideslib_stage1_rank_gate_${SLURM_JOB_ID:-manual}}"
BUILD_JOBS="${BUILD_JOBS:-${SLURM_CPUS_PER_TASK:-8}}"
RUN_NAME="${RUN_NAME:-stage1-s045-rank-gate-fideslib-${SLURM_JOB_ID:-manual}}"
CHECKPOINT="${CHECKPOINT:-runs/stage1-s014-tiny-mamba.pt}"
STATE_DICT_KEY="${STATE_DICT_KEY:-}"
OUTPUT_JSON="${OUTPUT_JSON:-runs/${RUN_NAME}.json}"
RANK_GATE_BINARY="${RANK_GATE_BINARY:-runs/${RUN_NAME}.bin}"
RANK_GATE_MANIFEST="${RANK_GATE_MANIFEST:-runs/${RUN_NAME}-payload-export.json}"

D_STATE="${D_STATE:-2}"
MIMO_RANK="${MIMO_RANK:-6}"
D_MODEL_PAD="${D_MODEL_PAD:-8}"
RANK_PAD="${RANK_PAD:-8}"
MODEL_BABY_STEP="${MODEL_BABY_STEP:-4}"
RANK_BABY_STEP="${RANK_BABY_STEP:-4}"
PROMPT_TOKEN="${PROMPT_TOKEN:-1}"
LAYER_INDEX="${LAYER_INDEX:-0}"
POLYNOMIAL_DEGREE="${POLYNOMIAL_DEGREE:-15}"
GATE_POLYNOMIAL_DEGREE="${GATE_POLYNOMIAL_DEGREE:-9}"
POLYNOMIAL_RANGE="${POLYNOMIAL_RANGE:-8.0}"
DECAY_POLYNOMIAL_DEGREE="${DECAY_POLYNOMIAL_DEGREE:-5}"
DECAY_POLYNOMIAL_RANGE_LOWER="${DECAY_POLYNOMIAL_RANGE_LOWER:--0.5}"
DECAY_POLYNOMIAL_RANGE_UPPER="${DECAY_POLYNOMIAL_RANGE_UPPER:-0.5}"
PREVIOUS_STATE_SCALE="${PREVIOUS_STATE_SCALE:-0.0}"
PREVIOUS_STATE_SEED="${PREVIOUS_STATE_SEED:-0}"

RING_DIM="${RING_DIM:-4096}"
MULTIPLICATIVE_DEPTH="${MULTIPLICATIVE_DEPTH:-24}"
SCALING_MOD_SIZE="${SCALING_MOD_SIZE:-40}"
FIRST_MOD_SIZE="${FIRST_MOD_SIZE:-60}"
ATOL="${ATOL:-1e-4}"
RANK_PROJECTION_SCALE="${RANK_PROJECTION_SCALE:-1.0}"
DT_PROJECTION_SCALE="${DT_PROJECTION_SCALE:-1.0}"
SECURITY="${SECURITY:-not-set}"
SECRET_KEY_DIST="${SECRET_KEY_DIST:-uniform-ternary}"

export OUTPUT_JSON
export LD_LIBRARY_PATH="${FIDESLIB_PREFIX}/lib:${OPENFHE_PREFIX}/lib:${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"

section() {
  printf '\n==== %s ====\n' "$1"
}

section "Environment"
echo "job_id=${SLURM_JOB_ID:-unknown}"
echo "node_list=${SLURM_JOB_NODELIST:-unknown}"
echo "date_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "checkpoint=${CHECKPOINT}"
echo "output_json=${OUTPUT_JSON}"
nvidia-smi -L
nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv
test -f "${FIDESLIB_PREFIX}/lib/fideslib.a"
test -f "${OPENFHE_PREFIX}/lib/OpenFHE/OpenFHEConfig.cmake"

section "Configure"
rm -rf "${BUILD_DIR}"
cmake \
  -S native/fideslib_stage0 \
  -B "${BUILD_DIR}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="${FIDESLIB_PREFIX};${OPENFHE_PREFIX}" \
  -Dfideslib_DIR="${FIDESLIB_PREFIX}/share/fideslib/cmake" \
  -DOpenFHE_DIR="${OPENFHE_PREFIX}/lib/OpenFHE"

section "Build"
cmake --build "${BUILD_DIR}" --target stage1_rank_gate_fideslib -j "${BUILD_JOBS}"

section "Export rank/gate payload"
mkdir -p "$(dirname "${OUTPUT_JSON}")"
EXPORT_ARGS=()
if [[ -n "${STATE_DICT_KEY}" ]]; then
  EXPORT_ARGS+=(--state-dict-key "${STATE_DICT_KEY}")
fi
"${PYTHON}" scripts/export_stage1_rank_gate_payload.py \
  "${CHECKPOINT}" \
  "${EXPORT_ARGS[@]}" \
  --layer-index "${LAYER_INDEX}" \
  --prompt-token "${PROMPT_TOKEN}" \
  --d-state "${D_STATE}" \
  --mimo-rank "${MIMO_RANK}" \
  --d-model-pad "${D_MODEL_PAD}" \
  --rank-pad "${RANK_PAD}" \
  --model-baby-step "${MODEL_BABY_STEP}" \
  --rank-baby-step "${RANK_BABY_STEP}" \
  --polynomial-degree "${POLYNOMIAL_DEGREE}" \
  --gate-polynomial-degree "${GATE_POLYNOMIAL_DEGREE}" \
  --polynomial-range "${POLYNOMIAL_RANGE}" \
  --decay-polynomial-degree "${DECAY_POLYNOMIAL_DEGREE}" \
  --decay-polynomial-range "${DECAY_POLYNOMIAL_RANGE_LOWER}" "${DECAY_POLYNOMIAL_RANGE_UPPER}" \
  --previous-state-scale "${PREVIOUS_STATE_SCALE}" \
  --previous-state-seed "${PREVIOUS_STATE_SEED}" \
  --output-binary "${RANK_GATE_BINARY}" \
  --output-json "${RANK_GATE_MANIFEST}"

section "Run FIDESlib rank/gate projection"
"${BUILD_DIR}/stage1_rank_gate_fideslib" \
  --input "${RANK_GATE_BINARY}" \
  --ring-dim "${RING_DIM}" \
  --multiplicative-depth "${MULTIPLICATIVE_DEPTH}" \
  --scaling-mod-size "${SCALING_MOD_SIZE}" \
  --first-mod-size "${FIRST_MOD_SIZE}" \
  --security "${SECURITY}" \
  --secret-key-dist "${SECRET_KEY_DIST}" \
  --atol "${ATOL}" \
  --rank-projection-scale "${RANK_PROJECTION_SCALE}" \
  --dt-projection-scale "${DT_PROJECTION_SCALE}" \
  --output-json "${OUTPUT_JSON}"

if [[ ! -s "${OUTPUT_JSON}" ]]; then
  echo "missing output_json=${OUTPUT_JSON}" >&2
  exit 1
fi

section "Artifact summary"
"${PYTHON}" - <<'PY'
import json
import os

path = os.environ["OUTPUT_JSON"]
payload = json.load(open(path, encoding="utf-8"))
summary = {
    "path": path,
    "passed": payload.get("passed"),
    "max_abs_error": payload.get("measurements", {}).get("max_abs_error"),
    "required_application_rotation_key_count": payload.get("measurements", {}).get(
        "required_application_rotation_key_count"
    ),
    "eval_seconds": payload.get("timing", {}).get("eval_seconds"),
    "peak_rss_gib": payload.get("measurements", {}).get("peak_rss_gib"),
}
print(json.dumps(summary, indent=2, sort_keys=True))
PY
