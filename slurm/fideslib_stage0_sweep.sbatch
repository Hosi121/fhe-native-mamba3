#!/usr/bin/env bash
#SBATCH --job-name=fideslib-stage0-sweep
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:12:00
#SBATCH --output=slurm/fideslib_stage0_sweep_%j.out
#SBATCH --error=slurm/fideslib_stage0_sweep_%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

CUDA_HOME="${CUDA_HOME:-$HOME/cuda12.8}"
export CUDA_HOME
export CUDA_PATH="${CUDA_PATH:-${CUDA_HOME}}"
export PATH="${CUDA_HOME}/bin:${PATH}"

OPENFHE_PREFIX="${OPENFHE_PREFIX:-$HOME/fhe-native-mamba3-deps/openfhe-fides}"
FIDESLIB_PREFIX="${FIDESLIB_PREFIX:-$HOME/fhe-native-mamba3-deps/fideslib-install}"
BUILD_DIR="${BUILD_DIR:-${SLURM_SUBMIT_DIR}/build/fideslib_stage0_sweep_${SLURM_JOB_ID:-manual}}"
BUILD_JOBS="${BUILD_JOBS:-${SLURM_CPUS_PER_TASK:-8}}"

SEQ_LENS="${SEQ_LENS:-8}"
D_STATES="${D_STATES:-4}"
MIMO_RANKS="${MIMO_RANKS:-1 2 4 8}"
READOUT_MODES="${READOUT_MODES:-rank-reduce}"
INPUT_MODE="${INPUT_MODE:-client-update}"
UPDATE_LEVEL_OFFSET="${UPDATE_LEVEL_OFFSET:-0}"
MULTIPLICATIVE_DEPTH="${MULTIPLICATIVE_DEPTH:-24}"
SCALING_MOD_SIZE="${SCALING_MOD_SIZE:-59}"
FIRST_MOD_SIZE="${FIRST_MOD_SIZE:-60}"
RING_DIM="${RING_DIM:-4096}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-1}"

export LD_LIBRARY_PATH="${FIDESLIB_PREFIX}/lib:${OPENFHE_PREFIX}/lib:${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"

section() {
  printf '\n==== %s ====\n' "$1"
}

section "Environment"
echo "job_id=${SLURM_JOB_ID:-unknown}"
echo "node_list=${SLURM_JOB_NODELIST:-unknown}"
echo "date_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "seq_lens=${SEQ_LENS}"
echo "d_states=${D_STATES}"
echo "mimo_ranks=${MIMO_RANKS}"
echo "readout_modes=${READOUT_MODES}"
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

section "Run sweep"
for seq_len in ${SEQ_LENS}; do
  for d_state in ${D_STATES}; do
    for mimo_rank in ${MIMO_RANKS}; do
      for readout_mode in ${READOUT_MODES}; do
        echo "config seq_len=${seq_len} d_state=${d_state} mimo_rank=${mimo_rank} readout_mode=${readout_mode}"
        if "${BUILD_DIR}/stage0_static_mimo" \
          --seq-len "${seq_len}" \
          --d-state "${d_state}" \
          --mimo-rank "${mimo_rank}" \
          --multiplicative-depth "${MULTIPLICATIVE_DEPTH}" \
          --scaling-mod-size "${SCALING_MOD_SIZE}" \
          --first-mod-size "${FIRST_MOD_SIZE}" \
          --ring-dim "${RING_DIM}" \
          --input-mode "${INPUT_MODE}" \
          --update-level-offset "${UPDATE_LEVEL_OFFSET}" \
          --readout-mode "${readout_mode}"; then
          :
        else
          status=$?
          printf '{"backend":"fideslib-gpu","encrypted":true,"stage":"0","name":"fideslib-static-mimo-recurrence","error":true,"exit_code":%s,"model":{"seq_len":%s,"d_state":%s,"mimo_rank":%s,"input_mode":"%s","readout_mode":"%s"},"next_bottleneck":"configuration failed; inspect stderr and retry with larger CKKS parameters or smaller packing group"}\n' \
            "${status}" \
            "${seq_len}" \
            "${d_state}" \
            "${mimo_rank}" \
            "${INPUT_MODE}" \
            "${readout_mode}"
          if [[ "${CONTINUE_ON_ERROR}" != "1" ]]; then
            exit "${status}"
          fi
        fi
      done
    done
  done
done
