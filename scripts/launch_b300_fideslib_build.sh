#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/home/kataiwa/fhemamba-b300}"
IMAGE="${IMAGE:-fhemamba-b300:cuda13.0-fideslib}"
CONTAINER_NAME="${CONTAINER_NAME:-fhemamba-b300-build}"
GPU_DEVICE="${GPU_DEVICE:-3}"
FIDESLIB_ARCH="${FIDESLIB_ARCH:-103-real}"
FIDESLIB_SM="${FIDESLIB_ARCH%%-*}"

if [[ "${GPU_DEVICE}" != "2" && "${GPU_DEVICE}" != "3" ]]; then
  echo "GPU_DEVICE must be 2 or 3" >&2
  exit 2
fi

mkdir -p "${ROOT_DIR}/build" "${ROOT_DIR}/install" "${ROOT_DIR}/logs"

if docker container inspect "${CONTAINER_NAME}" >/dev/null 2>&1; then
  state="$(docker inspect --format '{{.State.Status}}' "${CONTAINER_NAME}")"
  if [[ "${state}" == "running" ]]; then
    echo "container_already_running=${CONTAINER_NAME}"
    exit 0
  fi
  docker rm "${CONTAINER_NAME}" >/dev/null
fi

container_id="$({
  docker run \
    --detach \
    --name "${CONTAINER_NAME}" \
    --gpus "device=${GPU_DEVICE}" \
    --ipc=host \
    --shm-size=32g \
    --volume "${ROOT_DIR}:/workspace" \
    --workdir /workspace \
    --env ROOT_DIR=/workspace \
    --env FIDESLIB_ARCH="${FIDESLIB_ARCH}" \
    --env BUILD_JOBS="${BUILD_JOBS:-32}" \
    "${IMAGE}" \
    bash /workspace/cipher/scripts/build_b300_fideslib.sh
})"

echo "container_id=${container_id}"
echo "container_name=${CONTAINER_NAME}"
echo "host_gpu=${GPU_DEVICE}"
echo "fideslib_arch=${FIDESLIB_ARCH}"
echo "log=${ROOT_DIR}/logs/fideslib-build-sm${FIDESLIB_SM}.log"
