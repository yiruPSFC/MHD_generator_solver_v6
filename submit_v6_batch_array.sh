#!/bin/bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_DIR"

SBATCH_BIN="${SBATCH_BIN:-sbatch}"
PARTITION="${PARTITION:-mit_normal}"
ARRAY_SPEC="${ARRAY_SPEC:-}"
JOB_NAME="${JOB_NAME:-mhd_v6_batch}"
TIME_LIMIT="${TIME_LIMIT:-02:00:00}"
MEM_PER_JOB="${MEM_PER_JOB:-1G}"
EXCLUDE_NODES="${EXCLUDE_NODES:-}"

PYTHON_BIN="${PYTHON_BIN:-./.venv_jit/bin/python}"
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
NO_REQUEUE="${NO_REQUEUE:-1}"

PULL_POOL_MODE="${PULL_POOL_MODE:-1}"
SHARD_COUNT="${SHARD_COUNT:-100}"
WORKER_COUNT="${WORKER_COUNT:-115}"
ARRAY_CONCURRENCY="${ARRAY_CONCURRENCY:-96}"
TASK_POOL_ROOT="${TASK_POOL_ROOT:-${OUT_DIR}}"
INIT_TASK_POOL="${INIT_TASK_POOL:-1}"
RESET_TASK_POOL="${RESET_TASK_POOL:-0}"
TASK_POLL_INTERVAL_S="${TASK_POLL_INTERVAL_S:-1.0}"
TASK_STALE_TIMEOUT_S="${TASK_STALE_TIMEOUT_S:-900}"
TASK_HEARTBEAT_INTERVAL_S="${TASK_HEARTBEAT_INTERVAL_S:-15}"
TASK_MAX_REQUEUE_PER_SCAN="${TASK_MAX_REQUEUE_PER_SCAN:-4}"
TASK_MAX_ATTEMPTS="${TASK_MAX_ATTEMPTS:-3}"
TASK_FAIL_FAST="${TASK_FAIL_FAST:-0}"
ALLOW_TASK_POOL_RESET="${ALLOW_TASK_POOL_RESET:-0}"
FORCE_RESET_TASK_POOL="${FORCE_RESET_TASK_POOL:-0}"

NO_VELIKHOV_CHECK="${NO_VELIKHOV_CHECK:-0}"
NO_PREFILTER_DTE_REL="${NO_PREFILTER_DTE_REL:-0}"
PREFILTER_DTE_REL_MIN="${PREFILTER_DTE_REL_MIN:-}"
PREFILTER_MACH_INLET_MIN="${PREFILTER_MACH_INLET_MIN:-}"
PREFILTER_CHECK_INLET_VELIKHOV="${PREFILTER_CHECK_INLET_VELIKHOV:-0}"

require_positive_int() {
    local name="$1"
    local value="$2"
    if ! [[ "$value" =~ ^[0-9]+$ ]] || [ "$value" -le 0 ]; then
        echo "Invalid ${name}: ${value}" >&2
        exit 1
    fi
}

if [ "$PULL_POOL_MODE" != "0" ] && [ "$PULL_POOL_MODE" != "1" ]; then
    echo "Invalid PULL_POOL_MODE: $PULL_POOL_MODE (expected 0 or 1)" >&2
    exit 1
fi

require_positive_int "SHARD_COUNT" "$SHARD_COUNT"

if [ "$PULL_POOL_MODE" = "1" ]; then
    require_positive_int "WORKER_COUNT" "$WORKER_COUNT"
    require_positive_int "ARRAY_CONCURRENCY" "$ARRAY_CONCURRENCY"
    require_positive_int "TASK_MAX_ATTEMPTS" "$TASK_MAX_ATTEMPTS"
    if [ -z "$ARRAY_SPEC" ]; then
        ARRAY_SPEC="0-$((WORKER_COUNT - 1))%${ARRAY_CONCURRENCY}"
    fi
else
    if [ -z "$ARRAY_SPEC" ]; then
        ARRAY_SPEC="0-99%96"
    fi
fi

if [ "$PULL_POOL_MODE" = "1" ]; then
    TODO_DIR="${TASK_POOL_ROOT%/}/todo"
    PROCESSING_DIR="${TASK_POOL_ROOT%/}/processing"
    DONE_DIR="${TASK_POOL_ROOT%/}/done"
    FAILED_DIR="${TASK_POOL_ROOT%/}/failed"
    ATTEMPTS_DIR="${TASK_POOL_ROOT%/}/attempts"
    mkdir -p "$OUT_DIR" "$TODO_DIR" "$PROCESSING_DIR" "$DONE_DIR" "$FAILED_DIR" "$ATTEMPTS_DIR"

    if [ "$RESET_TASK_POOL" = "1" ]; then
        if [ "$ALLOW_TASK_POOL_RESET" != "1" ]; then
            echo "RESET_TASK_POOL=1 is destructive. Set ALLOW_TASK_POOL_RESET=1 to confirm." >&2
            exit 1
        fi
        processing_live_n=$(find "$PROCESSING_DIR" -maxdepth 1 -type f -name 'shard_*.task' | wc -l | tr -d ' ')
        if [ "$processing_live_n" -gt 0 ] && [ "$FORCE_RESET_TASK_POOL" != "1" ]; then
            echo "Detected $processing_live_n processing tasks. Refusing reset while work may be active." >&2
            echo "If you are sure this run is dead, set FORCE_RESET_TASK_POOL=1." >&2
            exit 1
        fi
        rm -f "$TODO_DIR"/shard_*.task
        rm -f "$PROCESSING_DIR"/shard_*.task
        rm -f "$DONE_DIR"/shard_*.task
        rm -f "$FAILED_DIR"/shard_*.task
        rm -f "$ATTEMPTS_DIR"/shard_*.attempt
    fi

    if [ "$INIT_TASK_POOL" = "1" ]; then
        for ((i = 0; i < SHARD_COUNT; i++)); do
            task_name="shard_${i}.task"
            out_file="${OUT_DIR%/}/sweep_shard_${i}.jsonl"
            todo_task="${TODO_DIR}/${task_name}"
            done_task="${DONE_DIR}/${task_name}"

            if [ -f "$out_file" ]; then
                : >"$done_task"
                rm -f "$todo_task"
                rm -f "$PROCESSING_DIR"/"shard_${i}"*.task
                rm -f "$FAILED_DIR"/"shard_${i}.task"
                rm -f "$ATTEMPTS_DIR"/"shard_${i}.attempt"
                continue
            fi

            rm -f "$done_task"
            rm -f "$FAILED_DIR"/"shard_${i}.task"
            rm -f "$ATTEMPTS_DIR"/"shard_${i}.attempt"
            if [ -f "$todo_task" ]; then
                continue
            fi
            if compgen -G "$PROCESSING_DIR/shard_${i}"'*.task' >/dev/null; then
                continue
            fi
            : >"$todo_task"
        done
    fi

    todo_n=$(find "$TODO_DIR" -maxdepth 1 -type f -name 'shard_*.task' | wc -l | tr -d ' ')
    processing_n=$(find "$PROCESSING_DIR" -maxdepth 1 -type f -name 'shard_*.task' | wc -l | tr -d ' ')
    done_n=$(find "$DONE_DIR" -maxdepth 1 -type f -name 'shard_*.task' | wc -l | tr -d ' ')
    failed_n=$(find "$FAILED_DIR" -maxdepth 1 -type f -name 'shard_*.task' | wc -l | tr -d ' ')
    echo "Task pool prepared: root=$TASK_POOL_ROOT todo=$todo_n processing=$processing_n done=$done_n failed=$failed_n"
fi

export_items=(
    "PYTHON_BIN=${PYTHON_BIN}"
    "N_TOTAL=${N_TOTAL}"
    "SEED=${SEED}"
    "TOP_K=${TOP_K}"
    "TP_IN=${TP_IN}"
    "B_FIELD_VAL=${B_FIELD_VAL}"
    "CHANNEL_LENGTH=${CHANNEL_LENGTH}"
    "NP_MIN=${NP_MIN}"
    "NP_MAX=${NP_MAX}"
    "Z_MIN=${Z_MIN}"
    "Z_MAX=${Z_MAX}"
    "TE_MIN=${TE_MIN}"
    "TE_MAX=${TE_MAX}"
    "SEEDF_MIN=${SEEDF_MIN}"
    "SEEDF_MAX=${SEEDF_MAX}"
    "BATCH_SIZE=${BATCH_SIZE}"
    "DX=${DX}"
    "OUT_DIR=${OUT_DIR}"
    "NO_VELIKHOV_CHECK=${NO_VELIKHOV_CHECK}"
    "NO_PREFILTER_DTE_REL=${NO_PREFILTER_DTE_REL}"
    "PREFILTER_CHECK_INLET_VELIKHOV=${PREFILTER_CHECK_INLET_VELIKHOV}"
    "PULL_POOL_MODE=${PULL_POOL_MODE}"
    "TASK_SHARD_COUNT=${SHARD_COUNT}"
)

if [ "$PULL_POOL_MODE" = "1" ]; then
    export_items+=(
        "TASK_POOL_ROOT=${TASK_POOL_ROOT}"
        "TASK_POLL_INTERVAL_S=${TASK_POLL_INTERVAL_S}"
        "TASK_STALE_TIMEOUT_S=${TASK_STALE_TIMEOUT_S}"
        "TASK_HEARTBEAT_INTERVAL_S=${TASK_HEARTBEAT_INTERVAL_S}"
        "TASK_MAX_REQUEUE_PER_SCAN=${TASK_MAX_REQUEUE_PER_SCAN}"
        "TASK_MAX_ATTEMPTS=${TASK_MAX_ATTEMPTS}"
        "TASK_FAIL_FAST=${TASK_FAIL_FAST}"
    )
fi

if [ -n "${PREFILTER_DTE_REL_MIN}" ]; then
    export_items+=("PREFILTER_DTE_REL_MIN=${PREFILTER_DTE_REL_MIN}")
fi

if [ -n "${PREFILTER_MACH_INLET_MIN}" ]; then
    export_items+=("PREFILTER_MACH_INLET_MIN=${PREFILTER_MACH_INLET_MIN}")
fi

cmd=(
    "$SBATCH_BIN"
    --partition="$PARTITION"
    --array="$ARRAY_SPEC"
    --job-name="$JOB_NAME"
    --time="$TIME_LIMIT"
    --cpus-per-task=1
    --mem="$MEM_PER_JOB"
)

if [ "$NO_REQUEUE" = "1" ]; then
    cmd+=(--no-requeue)
fi

if [ -n "$EXCLUDE_NODES" ]; then
    cmd+=(--exclude="$EXCLUDE_NODES")
fi

cmd+=(submit_v6_batch_array.sbatch)

# Avoid sbatch --export=... to reduce chances of hitting get-user-env failures.
export SLURM_EXPORT_ENV=ALL
for item in "${export_items[@]}"; do
    export "$item"
done

printf 'Submitting V6 array job: env'
printf ' %q' "${export_items[@]}"
printf ' --'
printf ' %q' "${cmd[@]}"
printf '\n'

"${cmd[@]}"
