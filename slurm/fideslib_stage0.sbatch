#!/usr/bin/env bash
#SBATCH --job-name=fideslib-stage0
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:08:00
#SBATCH --output=slurm/fideslib_stage0_%j.out
#SBATCH --error=slurm/fideslib_stage0_%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

CUDA_HOME="${CUDA_HOME:-$HOME/cuda12.8}"
export CUDA_HOME
export CUDA_PATH="${CUDA_PATH:-${CUDA_HOME}}"
export PATH="${CUDA_HOME}/bin:${PATH}"

OPENFHE_PREFIX="${OPENFHE_PREFIX:-$HOME/fhe-native-mamba3-deps/openfhe-fides}"
FIDESLIB_PREFIX="${FIDESLIB_PREFIX:-$HOME/fhe-native-mamba3-deps/fideslib-install}"
BUILD_DIR="${BUILD_DIR:-${SLURM_SUBMIT_DIR}/build/fideslib_stage0_${SLURM_JOB_ID:-manual}}"
BUILD_JOBS="${BUILD_JOBS:-${SLURM_CPUS_PER_TASK:-8}}"

export LD_LIBRARY_PATH="${FIDESLIB_PREFIX}/lib:${OPENFHE_PREFIX}/lib:${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"

section() {
  printf '\n==== %s ====\n' "$1"
}

section "Environment"
echo "job_id=${SLURM_JOB_ID:-unknown}"
echo "node_list=${SLURM_JOB_NODELIST:-unknown}"
echo "date_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
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
cmake --build "${BUILD_DIR}" -j "${BUILD_JOBS}"

section "Run"
"${BUILD_DIR}/stage0_static_mimo" \
  --seq-len "${SEQ_LEN:-8}" \
  --d-state "${D_STATE:-4}" \
  --mimo-rank "${MIMO_RANK:-2}" \
  --multiplicative-depth "${MULTIPLICATIVE_DEPTH:-24}" \
  --scaling-mod-size "${SCALING_MOD_SIZE:-59}" \
  --first-mod-size "${FIRST_MOD_SIZE:-60}" \
  --ring-dim "${RING_DIM:-4096}" \
  --input-mode "${INPUT_MODE:-client-update}" \
  --update-level-offset "${UPDATE_LEVEL_OFFSET:-0}"
