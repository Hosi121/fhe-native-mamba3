#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/workspace}"
FIDESLIB_DIR="${FIDESLIB_DIR:-${ROOT_DIR}/src/FIDESlib}"
OPENFHE_PREFIX="${OPENFHE_PREFIX:-${ROOT_DIR}/install/openfhe-fides}"
FIDESLIB_ARCH="${FIDESLIB_ARCH:-103-real}"
FIDESLIB_SM="${FIDESLIB_ARCH%%-*}"
B300_SYNC_PROFILE="${B300_SYNC_PROFILE:-full}"
if [[ "${B300_SYNC_PROFILE}" == "full" ]]; then
  BUILD_VARIANT="sm${FIDESLIB_SM}"
else
  BUILD_VARIANT="sm${FIDESLIB_SM}-${B300_SYNC_PROFILE}"
fi
FIDESLIB_PREFIX="${FIDESLIB_PREFIX:-${ROOT_DIR}/install/fideslib-${BUILD_VARIANT}}"
FIDESLIB_BUILD_DIR="${FIDESLIB_BUILD_DIR:-${ROOT_DIR}/build/fideslib-${BUILD_VARIANT}}"
STAGE_BUILD_DIR="${STAGE_BUILD_DIR:-${ROOT_DIR}/build/fideslib-stage0-${BUILD_VARIANT}}"
BUILD_JOBS="${BUILD_JOBS:-32}"
LOG_FILE="${LOG_FILE:-${ROOT_DIR}/logs/fideslib-build-${BUILD_VARIANT}.log}"

mkdir -p "$(dirname "${LOG_FILE}")" "${ROOT_DIR}/build" "${ROOT_DIR}/install"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "started_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "fideslib_arch=${FIDESLIB_ARCH}"
echo "b300_sync_profile=${B300_SYNC_PROFILE}"
echo "build_variant=${BUILD_VARIANT}"
echo "build_jobs=${BUILD_JOBS}"
nvidia-smi --query-gpu=index,name,compute_cap,memory.total --format=csv,noheader
nvcc --version

test -e "${FIDESLIB_DIR}/.git"
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

remove_patch_if_applied() {
  local patch_path="$1"
  if git -C "${FIDESLIB_DIR}" apply --reverse --check "${patch_path}" 2>/dev/null; then
    git -C "${FIDESLIB_DIR}" apply --reverse "${patch_path}"
    echo "patch_removed=${patch_path}"
  elif git -C "${FIDESLIB_DIR}" apply --check "${patch_path}" 2>/dev/null; then
    echo "patch_already_absent=${patch_path}"
  else
    echo "patch_cannot_be_removed=${patch_path}" >&2
    return 1
  fi
}

bootstrap_sync_patch="${ROOT_DIR}/cipher/native/fideslib_stage0/patches/fideslib-v2.1.0-bootstrap-stage-sync.patch"
lifetime_sync_patch="${ROOT_DIR}/cipher/native/fideslib_stage0/patches/fideslib-v2.1.0-b300-ciphertext-lifetime-sync.patch"
keyswitch_sync_patch="${ROOT_DIR}/cipher/native/fideslib_stage0/patches/fideslib-v2.1.0-b300-keyswitch-stage-sync.patch"

case "${B300_SYNC_PROFILE}" in
  full)
    apply_patch_once "${bootstrap_sync_patch}"
    apply_patch_once "${lifetime_sync_patch}"
    apply_patch_once "${keyswitch_sync_patch}"
    ;;
  bootstrap-lifetime)
    apply_patch_once "${bootstrap_sync_patch}"
    apply_patch_once "${lifetime_sync_patch}"
    remove_patch_if_applied "${keyswitch_sync_patch}"
    ;;
  lifetime)
    remove_patch_if_applied "${bootstrap_sync_patch}"
    apply_patch_once "${lifetime_sync_patch}"
    remove_patch_if_applied "${keyswitch_sync_patch}"
    ;;
  none)
    remove_patch_if_applied "${bootstrap_sync_patch}"
    remove_patch_if_applied "${lifetime_sync_patch}"
    remove_patch_if_applied "${keyswitch_sync_patch}"
    ;;
  *)
    echo "B300_SYNC_PROFILE must be full, bootstrap-lifetime, lifetime, or none" >&2
    exit 2
    ;;
esac
apply_patch_once \
  "${ROOT_DIR}/cipher/native/fideslib_stage0/patches/fideslib-v2.1.0-linear-transform-api.patch"
apply_patch_once \
  "${ROOT_DIR}/cipher/native/fideslib_stage0/patches/fideslib-v2.1.0-conjugate-api.patch"
apply_patch_once \
  "${ROOT_DIR}/cipher/native/fideslib_stage0/patches/fideslib-v2.1.0-ckks-data-type-api.patch"

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
else
  remove_patch_if_applied \
    "${ROOT_DIR}/cipher/native/fideslib_stage0/patches/fideslib-v2.1.0-cuda13-cccl-include.patch"
  remove_patch_if_applied \
    "${ROOT_DIR}/cipher/native/fideslib_stage0/patches/fideslib-v2.1.0-cuda13-graph-api.patch"
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
  -Dfideslib_DIR="${FIDESLIB_PREFIX}/share/fideslib/cmake" \
  -DCMAKE_PREFIX_PATH="${FIDESLIB_PREFIX};${OPENFHE_PREFIX}"
cmake --build "${STAGE_BUILD_DIR}" \
  --target stage1_mamba2_decode_fideslib stage1_bootstrap_probe fideslib_client_server_probe \
  -j "${BUILD_JOBS}"

echo "completed_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
