#!/bin/bash
#SBATCH --job-name=mhd_v6_batch
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

SLURM_WORKER_INDEX="${SLURM_ARRAY_TASK_ID:-0}"
SLURM_WORKER_COUNT="${SLURM_ARRAY_TASK_COUNT:-1}"

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
BATCH_SIZE="${BATCH_SIZE:-256}"
DX="${DX:-2e-5}"

OUT_DIR="${OUT_DIR:-results}"
OUT_FILE="${OUT_DIR}/sweep_shard_${SLURM_WORKER_INDEX}.jsonl"

PULL_POOL_MODE="${PULL_POOL_MODE:-0}"
TASK_SHARD_COUNT="${TASK_SHARD_COUNT:-100}"
TASK_POOL_ROOT="${TASK_POOL_ROOT:-${OUT_DIR}}"
TASK_POLL_INTERVAL_S="${TASK_POLL_INTERVAL_S:-1.0}"
TASK_STALE_TIMEOUT_S="${TASK_STALE_TIMEOUT_S:-900}"
TASK_HEARTBEAT_INTERVAL_S="${TASK_HEARTBEAT_INTERVAL_S:-15}"
TASK_MAX_REQUEUE_PER_SCAN="${TASK_MAX_REQUEUE_PER_SCAN:-4}"
TASK_MAX_ATTEMPTS="${TASK_MAX_ATTEMPTS:-3}"
TASK_FAIL_FAST="${TASK_FAIL_FAST:-0}"

if [ "$PULL_POOL_MODE" = "1" ]; then
    SHARD_COUNT="$TASK_SHARD_COUNT"
else
    SHARD_COUNT="$SLURM_WORKER_COUNT"
fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-1}}"
export NUMBA_NUM_THREADS="${NUMBA_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-1}}"

cmd=(
    "$PYTHON_BIN"
    cluster_sweep_worker_batch.py
    --n-total "$N_TOTAL"
    --seed "$SEED"
    --shard-index "$SLURM_WORKER_INDEX"
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
    --batch-size "$BATCH_SIZE"
    --dx "$DX"
)

if [ "$PULL_POOL_MODE" = "1" ]; then
    mkdir -p \
        "${TASK_POOL_ROOT%/}/todo" \
        "${TASK_POOL_ROOT%/}/processing" \
        "${TASK_POOL_ROOT%/}/done" \
        "${TASK_POOL_ROOT%/}/failed" \
        "${TASK_POOL_ROOT%/}/attempts"
    cmd+=(
        --task-pool-root "$TASK_POOL_ROOT"
        --task-poll-interval-s "$TASK_POLL_INTERVAL_S"
        --task-stale-timeout-s "$TASK_STALE_TIMEOUT_S"
        --task-heartbeat-interval-s "$TASK_HEARTBEAT_INTERVAL_S"
        --task-max-requeue-per-scan "$TASK_MAX_REQUEUE_PER_SCAN"
        --task-max-attempts "$TASK_MAX_ATTEMPTS"
        --task-worker-id "${SLURM_JOB_ID:-0}_${SLURM_ARRAY_TASK_ID:-0}_${HOSTNAME:-unknown}"
    )
    if [ "$TASK_FAIL_FAST" = "1" ]; then
        cmd+=(--task-fail-fast)
    fi
fi

if [ "${NO_VELIKHOV_CHECK:-0}" = "1" ]; then
    cmd+=(--no-velikhov-check)
fi

if [ "${NO_PREFILTER_DTE_REL:-0}" = "1" ]; then
    cmd+=(--no-prefilter-dte-rel)
elif [ -n "${PREFILTER_DTE_REL_MIN:-}" ]; then
    cmd+=(--prefilter-dte-rel-min "$PREFILTER_DTE_REL_MIN")
fi

if [ -n "${PREFILTER_MACH_INLET_MIN:-}" ]; then
    cmd+=(--prefilter-mach-inlet-min "$PREFILTER_MACH_INLET_MIN")
fi

if [ "${PREFILTER_CHECK_INLET_VELIKHOV:-0}" = "1" ]; then
    cmd+=(--prefilter-check-inlet-velikhov)
fi

if [ "$PULL_POOL_MODE" = "1" ]; then
    echo "Running pull worker ${SLURM_WORKER_INDEX}/${SLURM_WORKER_COUNT} (shards=${SHARD_COUNT}, pool=${TASK_POOL_ROOT})"
else
    echo "Running shard ${SLURM_WORKER_INDEX}/${SHARD_COUNT} with output ${OUT_FILE}"
fi
printf 'Command:'
printf ' %q' "${cmd[@]}"
printf '\n'

"${cmd[@]}"
