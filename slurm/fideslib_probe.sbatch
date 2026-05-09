#!/usr/bin/env bash
#SBATCH --job-name=fideslib-probe
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:10:00
#SBATCH --output=slurm/fideslib_probe_%j.out
#SBATCH --error=slurm/fideslib_probe_%j.err

set -u

cd "${SLURM_SUBMIT_DIR}"

CUDA_HOME="${CUDA_HOME:-$HOME/cuda12.8}"
if [[ -d "${CUDA_HOME}" ]]; then
  export CUDA_HOME
  export CUDA_PATH="${CUDA_PATH:-${CUDA_HOME}}"
  export PATH="${CUDA_HOME}/bin:${PATH}"
  export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
fi

FIDESLIB_DIR="${FIDESLIB_DIR:-$HOME/fhe-native-mamba3-deps/FIDESlib}"
OPENFHE_PREFIX="${OPENFHE_PREFIX:-$HOME/fhe-native-mamba3-deps/openfhe-fides}"
BUILD_DIR="${BUILD_DIR:-${FIDESLIB_DIR}/build_probe_b200}"
FIDESLIB_ARCH="${FIDESLIB_ARCH:-100-real}"
CONFIGURE_TIMEOUT_SEC="${CONFIGURE_TIMEOUT_SEC:-420}"

section() {
  printf '\n==== %s ====\n' "$1"
}

run_optional() {
  printf '+ %s\n' "$*"
  "$@"
  local status=$?
  printf 'exit_code=%s\n' "${status}"
  return 0
}

section "SLURM"
echo "job_id=${SLURM_JOB_ID:-unknown}"
echo "node_list=${SLURM_JOB_NODELIST:-unknown}"
echo "submit_dir=${SLURM_SUBMIT_DIR}"
echo "date_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"

section "GPU"
run_optional hostname
run_optional nvidia-smi -L
run_optional nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv

section "Toolchain"
run_optional which nvcc
run_optional nvcc --version
run_optional which cmake
run_optional cmake --version
run_optional gcc --version
run_optional g++ --version
run_optional python3 --version

section "Project Probe"
FIDESLIB_DIR="${FIDESLIB_DIR}" scripts/probe_fideslib.sh

section "FIDESlib Source"
if [[ -d "${FIDESLIB_DIR}/.git" ]]; then
  (
    cd "${FIDESLIB_DIR}" || exit 1
    run_optional git rev-parse --short HEAD
    run_optional git status --short
    run_optional find . -maxdepth 2 -type f
  )
else
  echo "missing_fideslib_dir=${FIDESLIB_DIR}"
fi

section "Bootstrap Mentions"
if [[ -d "${FIDESLIB_DIR}" ]]; then
  (
    cd "${FIDESLIB_DIR}" || exit 1
    grep -Rni "bootstr" README.md api src bench test examples 2>/dev/null | head -60
  )
fi

section "CMake Configure Probe"
if [[ -d "${FIDESLIB_DIR}" ]]; then
  rm -rf "${BUILD_DIR}"
  mkdir -p "${BUILD_DIR}"
  (
    cd "${FIDESLIB_DIR}" || exit 1
    set +e
    timeout "${CONFIGURE_TIMEOUT_SEC}" cmake \
      -S . \
      -B "${BUILD_DIR}" \
      -DCMAKE_BUILD_TYPE=Release \
      -DCUDA_PATH="${CUDA_PATH:-${CUDA_HOME}}" \
      -DFIDESLIB_ARCH="${FIDESLIB_ARCH}" \
      -DOPENFHE_INSTALL_PREFIX="${OPENFHE_PREFIX}" \
      -DFIDESLIB_INSTALL_OPENFHE=OFF \
      -DFIDESLIB_COMPILE_TESTS=OFF \
      -DFIDESLIB_COMPILE_BENCHMARKS=OFF
    status=$?
    set -e
    echo "cmake_configure_exit_code=${status}"
    if [[ -f "${BUILD_DIR}/CMakeCache.txt" ]]; then
      grep -E '^(CMAKE_CUDA_COMPILER|CMAKE_CUDA_ARCHITECTURES|CUDAToolkit_|OPENFHE_INSTALL_PREFIX|FIDESLIB_ARCH):' \
        "${BUILD_DIR}/CMakeCache.txt" || true
    fi
    if [[ ${status} -ne 0 ]]; then
      echo "configure_note=nonzero is expected until patched OpenFHE 1.4.2 is installed at OPENFHE_PREFIX"
    fi
  )
fi
