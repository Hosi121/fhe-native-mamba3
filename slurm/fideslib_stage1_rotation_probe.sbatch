#!/usr/bin/env bash
#SBATCH --job-name=fideslib-s1-rotation
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=00:35:00
#SBATCH --output=slurm/fideslib_stage1_rotation_probe_%j.out
#SBATCH --error=slurm/fideslib_stage1_rotation_probe_%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

CUDA_HOME="${CUDA_HOME:-$HOME/cuda12.8}"
export CUDA_HOME
export CUDA_PATH="${CUDA_PATH:-${CUDA_HOME}}"
export PATH="${CUDA_HOME}/bin:${PATH}"

OPENFHE_PREFIX="${OPENFHE_PREFIX:-$HOME/fhe-native-mamba3-deps/openfhe-fides}"
FIDESLIB_PREFIX="${FIDESLIB_PREFIX:-$HOME/fhe-native-mamba3-deps/fideslib-install}"
PYTHON="${PYTHON:-${SLURM_SUBMIT_DIR}/.venv/bin/python}"
BUILD_DIR="${BUILD_DIR:-${SLURM_SUBMIT_DIR}/build/fideslib_stage1_rotation_probe_${SLURM_JOB_ID:-manual}}"
BUILD_JOBS="${BUILD_JOBS:-${SLURM_CPUS_PER_TASK:-8}}"
RUN_NAME="${RUN_NAME:-stage1-s043-fideslib-rotation-${SLURM_JOB_ID:-manual}}"
OUTPUT_JSON="${OUTPUT_JSON:-runs/${RUN_NAME}.json}"
ROTATION_ARTIFACT="${ROTATION_ARTIFACT:-runs/stage1-s041-mamba130m-layer-openfhe-v0384.json}"
ROTATIONS_CSV="${ROTATIONS_CSV:-}"

RING_DIM="${RING_DIM:-131072}"
NUM_SLOTS="${NUM_SLOTS:-32768}"
MULTIPLICATIVE_DEPTH="${MULTIPLICATIVE_DEPTH:-48}"
SCALING_MOD_SIZE="${SCALING_MOD_SIZE:-40}"
FIRST_MOD_SIZE="${FIRST_MOD_SIZE:-60}"
ITERATIONS="${ITERATIONS:-1}"
ROTATION_LIMIT="${ROTATION_LIMIT:-0}"
SECURITY="${SECURITY:-128-classic}"
SECRET_KEY_DIST="${SECRET_KEY_DIST:-sparse-ternary}"
SKIP_DECRYPT="${SKIP_DECRYPT:-1}"

export OUTPUT_JSON
export LD_LIBRARY_PATH="${FIDESLIB_PREFIX}/lib:${OPENFHE_PREFIX}/lib:${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"

section() {
  printf '\n==== %s ====\n' "$1"
}

section "Environment"
echo "job_id=${SLURM_JOB_ID:-unknown}"
echo "node_list=${SLURM_JOB_NODELIST:-unknown}"
echo "date_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
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
cmake --build "${BUILD_DIR}" --target stage1_rotation_probe -j "${BUILD_JOBS}"

section "Run Stage 1 rotation probe"
mkdir -p "$(dirname "${OUTPUT_JSON}")"
NATIVE_ARGS=(
  --first-mod-size "${FIRST_MOD_SIZE}"
  --iterations "${ITERATIONS}"
  --rotation-limit "${ROTATION_LIMIT}"
  --security "${SECURITY}"
  --secret-key-dist "${SECRET_KEY_DIST}"
)
case "${SKIP_DECRYPT}" in
  0|false|FALSE|no|NO) NATIVE_ARGS+=(--check-decrypt) ;;
  *) NATIVE_ARGS+=(--skip-decrypt) ;;
esac

WRAPPER_ARGS=()
if [[ -n "${ROTATIONS_CSV}" ]]; then
  WRAPPER_ARGS+=(--rotations-csv "${ROTATIONS_CSV}")
else
  WRAPPER_ARGS+=(--rotation-artifact "${ROTATION_ARTIFACT}")
fi

"${PYTHON}" scripts/run_fideslib_stage1_rotation_probe.py \
  --binary "${BUILD_DIR}/stage1_rotation_probe" \
  --output-json "${OUTPUT_JSON}" \
  "${WRAPPER_ARGS[@]}" \
  --ring-dimension "${RING_DIM}" \
  --num-slots "${NUM_SLOTS}" \
  --multiplicative-depth "${MULTIPLICATIVE_DEPTH}" \
  --scaling-mod-size "${SCALING_MOD_SIZE}" \
  "${NATIVE_ARGS[@]}"

section "Artifact summary"
"${PYTHON}" - <<'PY'
import json
import os

path = os.environ["OUTPUT_JSON"]
payload = json.load(open(path, encoding="utf-8"))
summary = {
    "path": path,
    "passed": payload.get("passed"),
    "available": payload.get("available"),
    "requested_rotation_key_count": payload.get("measurements", {}).get(
        "requested_rotation_key_count"
    ),
    "mean_latency_sec": payload.get("mean_latency_sec"),
    "peak_rss_gib": payload.get("measurements", {}).get("peak_rss_gib"),
    "stage1_state_major_target_compatible": payload.get("measurement_scope", {}).get(
        "stage1_state_major_target_compatible"
    ),
}
print(json.dumps(summary, indent=2, sort_keys=True))
PY
