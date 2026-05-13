#!/usr/bin/env bash
#SBATCH --job-name=fideslib-s1-bootstrap
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=00:25:00
#SBATCH --output=slurm/fideslib_stage1_bootstrap_probe_%j.out
#SBATCH --error=slurm/fideslib_stage1_bootstrap_probe_%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

CUDA_HOME="${CUDA_HOME:-$HOME/cuda12.8}"
export CUDA_HOME
export CUDA_PATH="${CUDA_PATH:-${CUDA_HOME}}"
export PATH="${CUDA_HOME}/bin:${PATH}"

OPENFHE_PREFIX="${OPENFHE_PREFIX:-$HOME/fhe-native-mamba3-deps/openfhe-fides}"
FIDESLIB_PREFIX="${FIDESLIB_PREFIX:-$HOME/fhe-native-mamba3-deps/fideslib-install}"
PYTHON="${PYTHON:-${SLURM_SUBMIT_DIR}/.venv/bin/python}"
BUILD_DIR="${BUILD_DIR:-${SLURM_SUBMIT_DIR}/build/fideslib_stage1_bootstrap_probe_${SLURM_JOB_ID:-manual}}"
BUILD_JOBS="${BUILD_JOBS:-${SLURM_CPUS_PER_TASK:-8}}"
RUN_NAME="${RUN_NAME:-stage1-s007-fideslib-bootstrap-${SLURM_JOB_ID:-manual}}"
OUTPUT_JSON="${OUTPUT_JSON:-runs/${RUN_NAME}.json}"

RING_DIM="${RING_DIM:-65536}"
NUM_SLOTS="${NUM_SLOTS:-32768}"
MULTIPLICATIVE_DEPTH="${MULTIPLICATIVE_DEPTH:-28}"
SCALING_MOD_SIZE="${SCALING_MOD_SIZE:-40}"
FIRST_MOD_SIZE="${FIRST_MOD_SIZE:-60}"
LEVEL_BUDGET_CTS="${LEVEL_BUDGET_CTS:-5}"
LEVEL_BUDGET_STC="${LEVEL_BUDGET_STC:-4}"
BSGS_DIM_CTS="${BSGS_DIM_CTS:-0}"
BSGS_DIM_STC="${BSGS_DIM_STC:-0}"
ITERATIONS="${ITERATIONS:-3}"
SECURITY="${SECURITY:-128-classic}"
SECRET_KEY_DIST="${SECRET_KEY_DIST:-sparse-ternary}"
SKIP_DECRYPT="${SKIP_DECRYPT:-1}"

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
cmake --build "${BUILD_DIR}" --target stage1_bootstrap_probe -j "${BUILD_JOBS}"

section "Run Stage 1 bootstrap probe"
mkdir -p "$(dirname "${OUTPUT_JSON}")"
NATIVE_ARGS=(
  --ring-dim "${RING_DIM}"
  --num-slots "${NUM_SLOTS}"
  --multiplicative-depth "${MULTIPLICATIVE_DEPTH}"
  --scaling-mod-size "${SCALING_MOD_SIZE}"
  --first-mod-size "${FIRST_MOD_SIZE}"
  --level-budget-cts "${LEVEL_BUDGET_CTS}"
  --level-budget-stc "${LEVEL_BUDGET_STC}"
  --bsgs-dim-cts "${BSGS_DIM_CTS}"
  --bsgs-dim-stc "${BSGS_DIM_STC}"
  --iterations "${ITERATIONS}"
  --security "${SECURITY}"
  --secret-key-dist "${SECRET_KEY_DIST}"
)
case "${SKIP_DECRYPT}" in
  1|true|TRUE|yes|YES) NATIVE_ARGS+=(--skip-decrypt) ;;
esac

"${PYTHON}" scripts/run_fideslib_stage1_bootstrap_probe.py \
  --binary "${BUILD_DIR}/stage1_bootstrap_probe" \
  --output-json "${OUTPUT_JSON}" \
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
    "mean_latency_sec": payload.get("mean_latency_sec"),
    "stage1_target_compatible": payload.get("measurement_scope", {}).get(
        "stage1_target_compatible"
    ),
}
print(json.dumps(summary, indent=2, sort_keys=True))
PY
