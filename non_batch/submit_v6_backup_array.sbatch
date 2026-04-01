#!/bin/bash
#SBATCH --job-name=mhd_v6_backup
#SBATCH --output=slurm/%x_%A_%a.out
#SBATCH --error=slurm/%x_%A_%a.err
#SBATCH --array=0-99%96
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G

set -euo pipefail

# Keep full environment for this batch job and any nested Slurm steps.
export SLURM_EXPORT_ENV="${SLURM_EXPORT_ENV:-ALL}"

export PATH="${PATH:-/usr/local/bin:/usr/bin:/bin}"

REPO_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$REPO_DIR"

mkdir -p slurm results

PYTHON_BIN="${PYTHON_BIN:-./.venv_jit/bin/python}"
if [ ! -x "$PYTHON_BIN" ]; then
    echo "Python interpreter not found or not executable: $PYTHON_BIN" >&2
    exit 1
fi

SHARD_INDEX="${SLURM_ARRAY_TASK_ID:-0}"
SHARD_COUNT="${SLURM_ARRAY_TASK_COUNT:-1}"

N_TOTAL="${N_TOTAL:-16000}"
SEED="${SEED:-0}"
TOP_K="${TOP_K:-50}"
TP_IN="${TP_IN:-1000.0}"
B_FIELD_VAL="${B_FIELD_VAL:-0.02}"
CHANNEL_LENGTH="${CHANNEL_LENGTH:-0.039}"
NP_MIN="${NP_MIN:-1e21}"
NP_MAX="${NP_MAX:-1e24}"
Z_MIN="${Z_MIN:-1.0}"
Z_MAX="${Z_MAX:-120.0}"
TE_MIN="${TE_MIN:-1500.0}"
TE_MAX="${TE_MAX:-3500.0}"
SEEDF_MIN="${SEEDF_MIN:-1e-4}"
SEEDF_MAX="${SEEDF_MAX:-5e-2}"

OUT_DIR="${OUT_DIR:-results}"
OUT_FILE="${OUT_DIR}/sweep_shard_${SHARD_INDEX}.jsonl"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-1}}"
export NUMBA_NUM_THREADS="${NUMBA_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-1}}"

cmd=(
    "$PYTHON_BIN"
    non_batch/cluster_sweep_worker.py
    --n-total "$N_TOTAL"
    --seed "$SEED"
    --shard-index "$SHARD_INDEX"
    --shard-count "$SHARD_COUNT"
    --top-k "$TOP_K"
    --out "$OUT_FILE"
    --B "$B_FIELD_VAL"
    --L "$CHANNEL_LENGTH"
    --tp-in "$TP_IN"
    --np-min "$NP_MIN"
    --np-max "$NP_MAX"
    --z-min "$Z_MIN"
    --z-max "$Z_MAX"
    --te-min "$TE_MIN"
    --te-max "$TE_MAX"
    --seedf-min "$SEEDF_MIN"
    --seedf-max "$SEEDF_MAX"
)

if [ "${NO_VELIKHOV_CHECK:-0}" = "1" ]; then
    cmd+=(--no-velikhov-check)
fi

echo "Running backup shard ${SHARD_INDEX}/${SHARD_COUNT} with output ${OUT_FILE}"
printf 'Command:'
printf ' %q' "${cmd[@]}"
printf '\n'

"${cmd[@]}"
