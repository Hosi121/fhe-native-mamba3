#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/workspace}"
FIDESLIB_DIR="${FIDESLIB_DIR:-${ROOT_DIR}/src/FIDESlib}"
OPENFHE_PREFIX="${OPENFHE_PREFIX:-${ROOT_DIR}/install/openfhe-fides}"
FIDESLIB_ARCH="${FIDESLIB_ARCH:-103-real}"
FIDESLIB_SM="${FIDESLIB_ARCH%%-*}"
FIDESLIB_PREFIX="${FIDESLIB_PREFIX:-${ROOT_DIR}/install/fideslib-sm${FIDESLIB_SM}}"
FIDESLIB_BUILD_DIR="${FIDESLIB_BUILD_DIR:-${ROOT_DIR}/build/fideslib-sm${FIDESLIB_SM}}"
STAGE_BUILD_DIR="${STAGE_BUILD_DIR:-${ROOT_DIR}/build/fideslib-stage0-sm${FIDESLIB_SM}}"
BUILD_JOBS="${BUILD_JOBS:-32}"
LOG_FILE="${LOG_FILE:-${ROOT_DIR}/logs/fideslib-build-sm${FIDESLIB_SM}.log}"

mkdir -p "$(dirname "${LOG_FILE}")" "${ROOT_DIR}/build" "${ROOT_DIR}/install"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "started_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "fideslib_arch=${FIDESLIB_ARCH}"
echo "build_jobs=${BUILD_JOBS}"
nvidia-smi --query-gpu=index,name,compute_cap,memory.total --format=csv,noheader
nvcc --version

test -d "${FIDESLIB_DIR}/.git"
git config --global --add safe.directory "${FIDESLIB_DIR}"
git -C "${FIDESLIB_DIR}" rev-parse HEAD

apply_patch_once() {
  local patch_path="$1"
  if git -C "${FIDESLIB_DIR}" apply --reverse --check "${patch_path}" 2>/dev/null; then
    echo "patch_already_applied=${patch_path}"
  elif git -C "${FIDESLIB_DIR}" apply --check "${patch_path}" 2>/dev/null; then
    git -C "${FIDESLIB_DIR}" apply "${patch_path}"
    echo "patch_applied=${patch_path}"
  else
    echo "patch_cannot_be_applied=${patch_path}" >&2
    return 1
  fi
}

apply_patch_once \
  "${ROOT_DIR}/cipher/native/fideslib_stage0/patches/fideslib-v2.1.0-bootstrap-stage-sync.patch"
apply_patch_once \
  "${ROOT_DIR}/cipher/native/fideslib_stage0/patches/fideslib-v2.1.0-b300-ciphertext-lifetime-sync.patch"
apply_patch_once \
  "${ROOT_DIR}/cipher/native/fideslib_stage0/patches/fideslib-v2.1.0-b300-keyswitch-stage-sync.patch"

cuda_major="$(nvcc --version | sed -n 's/.*release \([0-9][0-9]*\)\..*/\1/p' | tail -1)"
if [[ -z "${cuda_major}" ]]; then
  echo "cannot determine CUDA major version" >&2
  exit 1
fi
if ((cuda_major >= 13)); then
  apply_patch_once \
    "${ROOT_DIR}/cipher/native/fideslib_stage0/patches/fideslib-v2.1.0-cuda13-graph-api.patch"
  apply_patch_once \
    "${ROOT_DIR}/cipher/native/fideslib_stage0/patches/fideslib-v2.1.0-cuda13-cccl-include.patch"
fi

if [[ ! -f "${OPENFHE_PREFIX}/lib/OpenFHE/OpenFHEConfig.cmake" && \
      ! -f "${OPENFHE_PREFIX}/lib/cmake/OpenFHE/OpenFHEConfig.cmake" ]]; then
  (
    cd "${FIDESLIB_DIR}/deps"
    ./build.sh "${OPENFHE_PREFIX}"
  )
else
  echo "using_existing_openfhe=${OPENFHE_PREFIX}"
fi

cmake \
  -S "${FIDESLIB_DIR}" \
  -B "${FIDESLIB_BUILD_DIR}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCUDA_PATH=/usr/local/cuda \
  -DFIDESLIB_ARCH="${FIDESLIB_ARCH}" \
  -DOPENFHE_INSTALL_PREFIX="${OPENFHE_PREFIX}" \
  -DFIDESLIB_INSTALL_PREFIX="${FIDESLIB_PREFIX}" \
  -DFIDESLIB_INSTALL_OPENFHE=OFF \
  -DFIDESLIB_COMPILE_TESTS=OFF \
  -DFIDESLIB_COMPILE_BENCHMARKS=OFF
cmake --build "${FIDESLIB_BUILD_DIR}" --target fideslib gpu-test -j "${BUILD_JOBS}"
cmake --build "${FIDESLIB_BUILD_DIR}" --target install -j "${BUILD_JOBS}"

"${FIDESLIB_BUILD_DIR}/gpu-test"

cmake \
  -S "${ROOT_DIR}/cipher/native/fideslib_stage0" \
  -B "${STAGE_BUILD_DIR}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="${FIDESLIB_PREFIX};${OPENFHE_PREFIX}"
cmake --build "${STAGE_BUILD_DIR}" \
  --target stage1_mamba2_decode_fideslib stage1_bootstrap_probe fideslib_client_server_probe \
  -j "${BUILD_JOBS}"

echo "completed_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
