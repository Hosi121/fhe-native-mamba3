#!/usr/bin/env bash
#SBATCH --job-name=fideslib-build
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:55:00
#SBATCH --output=slurm/fideslib_build_%j.out
#SBATCH --error=slurm/fideslib_build_%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

CUDA_HOME="${CUDA_HOME:-$HOME/cuda12.8}"
export CUDA_HOME
export CUDA_PATH="${CUDA_PATH:-${CUDA_HOME}}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"

FIDESLIB_DIR="${FIDESLIB_DIR:-$HOME/fhe-native-mamba3-deps/FIDESlib}"
OPENFHE_PREFIX="${OPENFHE_PREFIX:-$HOME/fhe-native-mamba3-deps/openfhe-fides}"
FIDESLIB_PREFIX="${FIDESLIB_PREFIX:-$HOME/fhe-native-mamba3-deps/fideslib-install}"
BUILD_DIR="${BUILD_DIR:-${FIDESLIB_DIR}/build_b200}"
FIDESLIB_ARCH="${FIDESLIB_ARCH:-100-real}"
BUILD_JOBS="${BUILD_JOBS:-${SLURM_CPUS_PER_TASK:-8}}"

section() {
  printf '\n==== %s ====\n' "$1"
}

section "Environment"
echo "job_id=${SLURM_JOB_ID:-unknown}"
echo "node_list=${SLURM_JOB_NODELIST:-unknown}"
echo "date_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
nvidia-smi -L
nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv
nvcc --version
cmake --version
g++ --version | head -1

section "Source"
test -d "${FIDESLIB_DIR}/.git"
cd "${FIDESLIB_DIR}"
git rev-parse --short HEAD

section "Configure"
rm -rf "${BUILD_DIR}"
cmake \
  -S . \
  -B "${BUILD_DIR}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCUDA_PATH="${CUDA_PATH}" \
  -DFIDESLIB_ARCH="${FIDESLIB_ARCH}" \
  -DOPENFHE_INSTALL_PREFIX="${OPENFHE_PREFIX}" \
  -DFIDESLIB_INSTALL_PREFIX="${FIDESLIB_PREFIX}" \
  -DFIDESLIB_INSTALL_OPENFHE=ON \
  -DFIDESLIB_COMPILE_TESTS=OFF \
  -DFIDESLIB_COMPILE_BENCHMARKS=OFF

section "Build"
cmake --build "${BUILD_DIR}" --target fideslib gpu-test -j "${BUILD_JOBS}"

section "Install"
cmake --build "${BUILD_DIR}" --target install -j "${BUILD_JOBS}"

section "GPU Test"
"${BUILD_DIR}/gpu-test"

section "Artifacts"
find "${FIDESLIB_PREFIX}" -maxdepth 3 -type f | sort
