#!/usr/bin/env bash
set -euo pipefail

FIDESLIB_DIR="${FIDESLIB_DIR:-$HOME/FIDESlib}"

json_bool() {
  if "$@" >/dev/null 2>&1; then
    printf 'true'
  else
    printf 'false'
  fi
}

command_path() {
  command -v "$1" 2>/dev/null || true
}

printf '{\n'
printf '  "fideslib_dir": "%s",\n' "${FIDESLIB_DIR}"
printf '  "tools": {\n'
printf '    "git": "%s",\n' "$(command_path git)"
printf '    "cmake": "%s",\n' "$(command_path cmake)"
printf '    "nvcc": "%s",\n' "$(command_path nvcc)"
printf '    "nvidia_smi": "%s"\n' "$(command_path nvidia-smi)"
printf '  },\n'
printf '  "environment": {\n'
printf '    "has_cuda_visible_devices": %s,\n' "$(test -n "${CUDA_VISIBLE_DEVICES:-}" && echo true || echo false)"
printf '    "cuda_visible_devices": "%s"\n' "${CUDA_VISIBLE_DEVICES:-}"
printf '  },\n'
printf '  "repository": {\n'
printf '    "exists": %s,\n' "$(test -d "${FIDESLIB_DIR}" && echo true || echo false)"
printf '    "has_cmake_lists": %s,\n' "$(test -f "${FIDESLIB_DIR}/CMakeLists.txt" && echo true || echo false)"
printf '    "mentions_bootstrap": %s\n' "$(test -d "${FIDESLIB_DIR}" && grep -Rqi 'Bootstrap' "${FIDESLIB_DIR}" 2>/dev/null && echo true || echo false)"
printf '  },\n'
printf '  "can_probe": %s\n' "$(json_bool test -d "${FIDESLIB_DIR}")"
printf '}\n'
