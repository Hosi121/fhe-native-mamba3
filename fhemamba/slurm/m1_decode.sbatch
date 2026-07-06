#!/usr/bin/env bash
#SBATCH --job-name=m1-decode
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=02:00:00
#SBATCH --output=fhemamba/slurm/m1_decode_%j.out
#SBATCH --error=fhemamba/slurm/m1_decode_%j.err

# Full-width Mamba-2 decode step (layer 0) on FIDESlib: builds the ct/pt op
# probe and the decode kernel, runs the probe at both ring dimensions, then
# runs the kernel on the exported m1 payload with 4 tokens of ciphertext state
# carry. Modeled on slurm/fideslib_stage1_rank_gate_existing_payload.sbatch.
#
# dgx runbook (FIDESlib v2.1.0, fhemamba/results/dgx/*.json):
# - MAXP=64 constant-memory tower cap: empirical max depth 50 at scale 40,
#   44 at scale 59. The uncut circuit (61+ levels) can never run; the kernel
#   uses mid-circuit auto-bootstrap checkpoints instead. Defaults here follow
#   the working geometry: scale 59, depth 44.
# - EvalBootstrap is numerically dead below scaling-mod 54 (5/5 decrypt
#   failures at 40/50/52); at 59 the error is ~1e-5 and a warm GPU bootstrap
#   costs 12-14 ms. Bootstrap output sits at GetLevel 18 at depth 44/scale 59,
#   leaving a 26-level segment budget (all circuit segments fit; worst is the
#   dt->decay path of squarings-heavy layers at ~21).
# - Memory: ring 131072 at ~194 rotation keys OOMed 119 GB unified on GB10
#   during keygen+LoadContext; ring 65536 peaked 44-57 GiB. Use RING_DIM=65536
#   on memory-constrained nodes (packing regroups automatically).

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

CUDA_HOME="${CUDA_HOME:-$HOME/cuda12.8}"
export CUDA_HOME
export CUDA_PATH="${CUDA_PATH:-${CUDA_HOME}}"
export PATH="${CUDA_HOME}/bin:${PATH}"

OPENFHE_PREFIX="${OPENFHE_PREFIX:-$HOME/fhe-native-mamba3-deps/openfhe-fides}"
FIDESLIB_PREFIX="${FIDESLIB_PREFIX:-$HOME/fhe-native-mamba3-deps/fideslib-install}"
PYTHON="${PYTHON:-${PWD}/.venv/bin/python}"
if [ ! -x "${PYTHON}" ]; then
  PYTHON="$(command -v python3)"
fi
BUILD_DIR="${BUILD_DIR:-${PWD}/build/m1_decode_${SLURM_JOB_ID:-manual}}"
BUILD_JOBS="${BUILD_JOBS:-${SLURM_CPUS_PER_TASK:-8}}"

M1_INPUT="${M1_INPUT:-$HOME/fhemamba-campaign/m1_payload_l0}"
RESULTS_DIR="${RESULTS_DIR:-results_b200}"
OUTPUT_JSON="${OUTPUT_JSON:-${RESULTS_DIR}/m1_decode_l0.json}"

RING_DIM="${RING_DIM:-131072}"
MULTIPLICATIVE_DEPTH="${MULTIPLICATIVE_DEPTH:-44}"
SCALING_MOD_SIZE="${SCALING_MOD_SIZE:-59}"
FIRST_MOD_SIZE="${FIRST_MOD_SIZE:-60}"
TOKENS="${TOKENS:-4}"
TOLERANCE="${TOLERANCE:-5e-2}"
BOOTSTRAP_BEFORE_TOKEN="${BOOTSTRAP_BEFORE_TOKEN:-}"
SECURITY="${SECURITY:-not-set}"
SECRET_KEY_DIST="${SECRET_KEY_DIST:-uniform-ternary}"
PROBE_ITERATIONS="${PROBE_ITERATIONS:-200}"

# M2 chain smoke: 4 layers x 2 tokens with ciphertext residual handoff. With
# mid-circuit bootstrap checkpoints every segment fits inside the depth-44 /
# scale-59 geometry (simulated bootstraps: ~31-34 per token for 4 layers, at
# 12-14 ms each).
M2_INPUT="${M2_INPUT:-$HOME/fhemamba-campaign/m2_chain_payload}"
CHAIN_OUTPUT_JSON="${CHAIN_OUTPUT_JSON:-${RESULTS_DIR}/m2_chain_smoke.json}"
CHAIN_DEPTH="${CHAIN_DEPTH:-44}"
CHAIN_MAX_LAYERS="${CHAIN_MAX_LAYERS:-4}"
CHAIN_TOKENS="${CHAIN_TOKENS:-2}"
CHAIN_TOLERANCE="${CHAIN_TOLERANCE:-1e-1}"
CHAIN_HEADROOM="${CHAIN_HEADROOM:-12}"

export LD_LIBRARY_PATH="${FIDESLIB_PREFIX}/lib:${OPENFHE_PREFIX}/lib:${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
ARTIFACT_VERSION="$("${PYTHON}" - <<'PY' 2>/dev/null || echo "0.0.0+unknown"
from fhe_native_mamba3 import __version__

print(__version__)
PY
)"
REPO_COMMIT="$(git rev-parse HEAD 2>/dev/null || true)"
REPO_COMMIT="${REPO_COMMIT:-unknown}"

mkdir -p "$(dirname "${OUTPUT_JSON}")" "${RESULTS_DIR}" fhemamba/slurm

cmake \
  -S native/fideslib_stage0 \
  -B "${BUILD_DIR}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="${FIDESLIB_PREFIX};${OPENFHE_PREFIX}" \
  -Dfideslib_DIR="${FIDESLIB_PREFIX}/share/fideslib/cmake" \
  -DOpenFHE_DIR="${OPENFHE_PREFIX}/lib/OpenFHE"
cmake --build "${BUILD_DIR}" --target fideslib_ctpt_probe -j "${BUILD_JOBS}"
cmake --build "${BUILD_DIR}" --target stage1_mamba2_decode_fideslib -j "${BUILD_JOBS}"

echo "== ct/pt op probe =="
for PROBE_RING in 65536 131072; do
  "${BUILD_DIR}/fideslib_ctpt_probe" \
    --ring-dim "${PROBE_RING}" \
    --iterations "${PROBE_ITERATIONS}" \
    --security "${SECURITY}" \
    --secret-key-dist "${SECRET_KEY_DIST}" \
    --output-json "${RESULTS_DIR}/ctpt_probe_${PROBE_RING}.json"
done

# Run the single-layer kernel and the chain smoke independently: a failure of
# one must not prevent the other; the job exits nonzero at the end if any run
# failed.
JOB_STATUS=0

echo "== mamba2 decode kernel (single layer) =="
set +e
"${BUILD_DIR}/stage1_mamba2_decode_fideslib" \
  --input "${M1_INPUT}" \
  --artifact-version "${ARTIFACT_VERSION}" \
  --repo-commit "${REPO_COMMIT}" \
  --ring-dim "${RING_DIM}" \
  --multiplicative-depth "${MULTIPLICATIVE_DEPTH}" \
  --scaling-mod-size "${SCALING_MOD_SIZE}" \
  --first-mod-size "${FIRST_MOD_SIZE}" \
  --security "${SECURITY}" \
  --secret-key-dist "${SECRET_KEY_DIST}" \
  --tokens "${TOKENS}" \
  --tolerance "${TOLERANCE}" \
  ${BOOTSTRAP_BEFORE_TOKEN:+--bootstrap-before-token "${BOOTSTRAP_BEFORE_TOKEN}"} \
  --output-json "${OUTPUT_JSON}"
SINGLE_RC=$?
set -e
if [ "${SINGLE_RC}" -ne 0 ]; then
  echo "single-layer run failed rc=${SINGLE_RC}"
  JOB_STATUS=1
fi

echo "== mamba2 chain smoke (${CHAIN_MAX_LAYERS} layers x ${CHAIN_TOKENS} tokens) =="
set +e
"${BUILD_DIR}/stage1_mamba2_decode_fideslib" \
  --input-chain "${M2_INPUT}" \
  --artifact-version "${ARTIFACT_VERSION}" \
  --repo-commit "${REPO_COMMIT}" \
  --ring-dim "${RING_DIM}" \
  --multiplicative-depth "${CHAIN_DEPTH}" \
  --scaling-mod-size "${SCALING_MOD_SIZE}" \
  --first-mod-size "${FIRST_MOD_SIZE}" \
  --security "${SECURITY}" \
  --secret-key-dist "${SECRET_KEY_DIST}" \
  --max-layers "${CHAIN_MAX_LAYERS}" \
  --tokens "${CHAIN_TOKENS}" \
  --tolerance "${CHAIN_TOLERANCE}" \
  --auto-bootstrap-headroom "${CHAIN_HEADROOM}" \
  --output-json "${CHAIN_OUTPUT_JSON}"
CHAIN_RC=$?
set -e
if [ "${CHAIN_RC}" -ne 0 ]; then
  echo "chain smoke failed rc=${CHAIN_RC}"
  JOB_STATUS=1
fi

echo "== summaries =="
set +e
"${PYTHON}" - "${OUTPUT_JSON}" "${CHAIN_OUTPUT_JSON}" "${RESULTS_DIR}/ctpt_probe_65536.json" \
  "${RESULTS_DIR}/ctpt_probe_131072.json" <<'PY'
import json
import sys
from pathlib import Path

for kernel_path in sys.argv[1:3]:
    path = Path(kernel_path)
    if not path.exists():
        print(f"missing: {kernel_path}")
        continue
    kernel = json.loads(path.read_text(encoding="utf-8"))
    summary = {
        "path": kernel_path,
        "passed": kernel.get("passed"),
        "max_abs_error": kernel.get("measurements", {}).get("max_abs_error"),
        "per_token_max_abs_error": kernel.get("measurements", {}).get(
            "per_token_max_abs_error"
        ),
        "per_token_bootstrap_count": kernel.get("measurements", {}).get(
            "per_token_bootstrap_count"
        ),
        "rotation_keys": kernel.get("measurements", {}).get(
            "required_application_rotation_key_count"
        ),
        "eval_seconds": kernel.get("timing", {}).get("eval_seconds"),
        "rotate_keygen_seconds": kernel.get("timing", {}).get("rotate_keygen_seconds"),
        "peak_rss_gib": kernel.get("measurements", {}).get("peak_rss_gib"),
        "layer_token_summary": kernel.get("layer_token_summary"),
        "token_output_levels": {
            key: value
            for key, value in (kernel.get("ckks_levels") or {}).items()
            if key.endswith(".output") or key.endswith(".final_norm")
        },
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
for probe_path in sys.argv[3:]:
    path = Path(probe_path)
    if path.exists():
        print(json.dumps(json.loads(path.read_text(encoding="utf-8")), indent=2, sort_keys=True))
PY
set -e

exit "${JOB_STATUS}"
