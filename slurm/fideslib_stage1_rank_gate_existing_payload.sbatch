#!/usr/bin/env bash
#SBATCH --job-name=fideslib-s1-existing
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=00:40:00
#SBATCH --output=slurm/fideslib_stage1_rank_gate_existing_payload_%j.out
#SBATCH --error=slurm/fideslib_stage1_rank_gate_existing_payload_%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

CUDA_HOME="${CUDA_HOME:-$HOME/cuda12.8}"
export CUDA_HOME
export CUDA_PATH="${CUDA_PATH:-${CUDA_HOME}}"
export PATH="${CUDA_HOME}/bin:${PATH}"

OPENFHE_PREFIX="${OPENFHE_PREFIX:-$HOME/fhe-native-mamba3-deps/openfhe-fides}"
FIDESLIB_PREFIX="${FIDESLIB_PREFIX:-$HOME/fhe-native-mamba3-deps/fideslib-install}"
PYTHON="${PYTHON:-${PWD}/.venv/bin/python}"
BUILD_DIR="${BUILD_DIR:-${PWD}/build/fideslib_stage1_rank_gate_existing_${SLURM_JOB_ID:-manual}}"
BUILD_JOBS="${BUILD_JOBS:-${SLURM_CPUS_PER_TASK:-8}}"

RANK_GATE_INPUT="${RANK_GATE_INPUT:?set RANK_GATE_INPUT to an existing payload binary}"
RUN_NAME="${RUN_NAME:-stage1-rank-gate-existing-${SLURM_JOB_ID:-manual}}"
OUTPUT_JSON="${OUTPUT_JSON:-runs/${RUN_NAME}.json}"

RING_DIM="${RING_DIM:-131072}"
MULTIPLICATIVE_DEPTH="${MULTIPLICATIVE_DEPTH:-48}"
SCALING_MOD_SIZE="${SCALING_MOD_SIZE:-40}"
FIRST_MOD_SIZE="${FIRST_MOD_SIZE:-60}"
ATOL="${ATOL:-1e-4}"
RANK_PROJECTION_SCALE="${RANK_PROJECTION_SCALE:-1.0}"
DT_PROJECTION_SCALE="${DT_PROJECTION_SCALE:-1.0}"
CHAIN_STEPS="${CHAIN_STEPS:-1}"
BOOTSTRAP_BEFORE_CHAIN_STEPS="${BOOTSTRAP_BEFORE_CHAIN_STEPS:-}"
SECURITY="${SECURITY:-not-set}"
SECRET_KEY_DIST="${SECRET_KEY_DIST:-uniform-ternary}"

export LD_LIBRARY_PATH="${FIDESLIB_PREFIX}/lib:${OPENFHE_PREFIX}/lib:${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
ARTIFACT_VERSION="$("${PYTHON}" - <<'PY'
from fhe_native_mamba3 import __version__

print(__version__)
PY
)"
REPO_COMMIT="$(git rev-parse HEAD 2>/dev/null || true)"
REPO_COMMIT="${REPO_COMMIT:-unknown}"

mkdir -p "$(dirname "${OUTPUT_JSON}")" slurm

cmake \
  -S native/fideslib_stage0 \
  -B "${BUILD_DIR}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="${FIDESLIB_PREFIX};${OPENFHE_PREFIX}" \
  -Dfideslib_DIR="${FIDESLIB_PREFIX}/share/fideslib/cmake" \
  -DOpenFHE_DIR="${OPENFHE_PREFIX}/lib/OpenFHE"
cmake --build "${BUILD_DIR}" --target stage1_rank_gate_fideslib -j "${BUILD_JOBS}"

"${BUILD_DIR}/stage1_rank_gate_fideslib" \
  --input "${RANK_GATE_INPUT}" \
  --artifact-version "${ARTIFACT_VERSION}" \
  --repo-commit "${REPO_COMMIT}" \
  --ring-dim "${RING_DIM}" \
  --multiplicative-depth "${MULTIPLICATIVE_DEPTH}" \
  --scaling-mod-size "${SCALING_MOD_SIZE}" \
  --first-mod-size "${FIRST_MOD_SIZE}" \
  --security "${SECURITY}" \
  --secret-key-dist "${SECRET_KEY_DIST}" \
  --atol "${ATOL}" \
  --rank-projection-scale "${RANK_PROJECTION_SCALE}" \
  --dt-projection-scale "${DT_PROJECTION_SCALE}" \
  --chain-steps "${CHAIN_STEPS}" \
  --bootstrap-before-chain-steps "${BOOTSTRAP_BEFORE_CHAIN_STEPS}" \
  --output-json "${OUTPUT_JSON}"

"${PYTHON}" - "${OUTPUT_JSON}" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(
    json.dumps(
        {
            "path": sys.argv[1],
            "passed": payload.get("passed"),
            "max_abs_error": payload.get("measurements", {}).get("max_abs_error"),
            "diagnostic_max_abs_error": payload.get("measurements", {}).get(
                "diagnostic_max_abs_error"
            ),
            "eval_seconds": payload.get("timing", {}).get("eval_seconds"),
            "peak_rss_gib": payload.get("measurements", {}).get("peak_rss_gib"),
        },
        indent=2,
        sort_keys=True,
    )
)
PY
