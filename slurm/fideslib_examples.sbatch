#!/usr/bin/env bash
#SBATCH --job-name=fideslib-examples
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:12:00
#SBATCH --output=slurm/fideslib_examples_%j.out
#SBATCH --error=slurm/fideslib_examples_%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

CUDA_HOME="${CUDA_HOME:-$HOME/cuda12.8}"
export CUDA_HOME
export CUDA_PATH="${CUDA_PATH:-${CUDA_HOME}}"
export PATH="${CUDA_HOME}/bin:${PATH}"

FIDESLIB_DIR="${FIDESLIB_DIR:-$HOME/fhe-native-mamba3-deps/FIDESlib}"
OPENFHE_PREFIX="${OPENFHE_PREFIX:-$HOME/fhe-native-mamba3-deps/openfhe-fides}"
FIDESLIB_PREFIX="${FIDESLIB_PREFIX:-$HOME/fhe-native-mamba3-deps/fideslib-install}"
BUILD_JOBS="${BUILD_JOBS:-${SLURM_CPUS_PER_TASK:-8}}"
RUN_TIMEOUT_SEC="${RUN_TIMEOUT_SEC:-420}"

export LD_LIBRARY_PATH="${FIDESLIB_PREFIX}/lib:${OPENFHE_PREFIX}/lib:${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"

section() {
  printf '\n==== %s ====\n' "$1"
}

build_and_run_example() {
  local name="$1"
  local source_dir="${2:-${FIDESLIB_DIR}/examples/${name}}"
  local build_dir="${source_dir}/build_b200"

  section "Configure ${name}"
  rm -rf "${build_dir}"
  cmake \
    -S "${source_dir}" \
    -B "${build_dir}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_PREFIX_PATH="${FIDESLIB_PREFIX};${OPENFHE_PREFIX}" \
    -Dfideslib_DIR="${FIDESLIB_PREFIX}/share/fideslib/cmake" \
    -DOpenFHE_DIR="${OPENFHE_PREFIX}/lib/OpenFHE"

  section "Build ${name}"
  cmake --build "${build_dir}" -j "${BUILD_JOBS}"

  section "Run ${name}"
  timeout "${RUN_TIMEOUT_SEC}" "${build_dir}/${name}"
}

section "Environment"
echo "job_id=${SLURM_JOB_ID:-unknown}"
echo "node_list=${SLURM_JOB_NODELIST:-unknown}"
echo "date_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
nvidia-smi -L
nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv
test -f "${FIDESLIB_PREFIX}/lib/fideslib.a"
test -f "${OPENFHE_PREFIX}/lib/OpenFHE/OpenFHEConfig.cmake"

build_and_run_example simple

BOOTSTRAP_SOURCE="${SLURM_TMPDIR:-/tmp}/fideslib_bootstrap_${SLURM_JOB_ID:-manual}"
rm -rf "${BOOTSTRAP_SOURCE}"
cp -a "${FIDESLIB_DIR}/examples/bootstrap" "${BOOTSTRAP_SOURCE}"
perl -0pi -e 's/EvalBootstrapKeyGen\(keyPair,/EvalBootstrapKeyGen(keyPair.secretKey,/g' \
  "${BOOTSTRAP_SOURCE}/src/bootstrap.cpp"
build_and_run_example bootstrap "${BOOTSTRAP_SOURCE}"
