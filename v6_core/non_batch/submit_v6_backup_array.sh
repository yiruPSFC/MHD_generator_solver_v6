#!/bin/bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_DIR"

SBATCH_BIN="${SBATCH_BIN:-sbatch}"
PARTITION="${PARTITION:-mit_normal}"
ARRAY_SPEC="${ARRAY_SPEC:-0-99%96}"
JOB_NAME="${JOB_NAME:-mhd_v6_backup}"
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
OUT_DIR="${OUT_DIR:-v6_core/non_batch/outputs/results}"
NO_REQUEUE="${NO_REQUEUE:-1}"
NO_VELIKHOV_CHECK="${NO_VELIKHOV_CHECK:-0}"

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
    "OUT_DIR=${OUT_DIR}"
    "NO_VELIKHOV_CHECK=${NO_VELIKHOV_CHECK}"
)

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

cmd+=(v6_core/non_batch/submit_v6_backup_array.sbatch)

# Avoid sbatch --export=... to reduce chances of hitting get-user-env failures.
export SLURM_EXPORT_ENV=ALL
for item in "${export_items[@]}"; do
    export "$item"
done

printf 'Submitting V6 backup array job: env'
printf ' %q' "${export_items[@]}"
printf ' --'
printf ' %q' "${cmd[@]}"
printf '\n'

"${cmd[@]}"
