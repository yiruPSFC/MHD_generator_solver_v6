# V6 Cluster Environment Notes

This note records how the V6 environment should be set up on the cluster and what was inherited from V5.

Important repo fact:

- `./.venv_jit` is intentionally local-only and is ignored by git.
- After `git clone`, it is normal for `./.venv_jit` to be missing.
- If Codex or a shell session says the environment is missing, the correct action is to create `./.venv_jit` for this repo, not to mutate some unrelated global env by default.

## What V5 was using

V5 had two Python environment patterns:

- Preferred interpreter in batch helpers:
  - `/home/yiruxiao/miniconda3/envs/mhd_search/bin/python`
- Fallback interpreter in Slurm scripts:
  - `./.venv/bin/python`

Relevant references in V5:

- `feasibility_search/engaging/submit_stageA_array.sh`
- `feasibility_search/engaging/stageA_array_100x1k.sbatch`
- `WORKFLOW.md`

Important lesson from V5 logs:

- If `PYTHON_BIN` is left at the default `./.venv/bin/python` and `.venv` does not exist on the cluster node, jobs fail immediately.
- V5 notes also mentioned a JIT-focused environment (`.venv_jit`) for numba-enabled runs.

## Recommended V6 choice

For V6, use a repo-local virtual environment:

- Path: `./.venv_jit`
- Reason:
  - does not mutate the shared/global `mhd_search` env
  - keeps numba compatibility pinned for this repo
  - is easy to point Slurm jobs at with `PYTHON_BIN=./.venv_jit/bin/python`

Verified working versions in this repo at the time of the latest smoke test:

- Python `3.12.7`
- `numpy==2.0.2`
- `scipy==1.15.3`
- `numba==0.61.2`

## Local setup

From repo root after cloning:

```bash
cd /home/yiruxiao/MHD_generator_solver/MHD_generator_solver_v6
python3 -m venv .venv_jit
./.venv_jit/bin/python -m pip install --upgrade pip setuptools wheel
./.venv_jit/bin/python -m pip install numpy==2.0.2 scipy==1.15.3 numba==0.61.2
```

Quick version check:

```bash
./.venv_jit/bin/python -c "import numpy, scipy, numba; print(numpy.__version__, scipy.__version__, numba.__version__)"
```

Expected output:

- `2.0.2 1.15.3 0.61.2`

## Local verification

Smoke test used for this setup:

```bash
./.venv_jit/bin/python v6_batch/smoke_test_batch_v6.py
```

Expected result:

- `SMOKE TEST PASSED`
- The run may also print an OpenMP info line from the runtime before the smoke test output. That line is not a failure.

## Local batch benchmark

We also benchmarked `cluster_sweep_worker_batch.py` locally on this Mac under AC power with:

- fixed `seed=12345`
- fixed `n_total=100000`
- default inlet prefilter enabled
- thread counts `1, 2, 4, 8`
- batch sizes `32, 64, 96, 128, 192, 256`

Top local configurations from that sweep:

- `8 threads + batch_size=256` -> `5.296 s`, about `247 MiB` max RSS
- `4 threads + batch_size=192` -> `5.399 s`, about `215 MiB` max RSS
- `4 threads + batch_size=256` -> `5.405 s`, about `220 MiB` max RSS

Practical local default:

- use `OMP_NUM_THREADS=4`
- use `NUMBA_NUM_THREADS=4`
- use `BATCH_SIZE=256`

Reason:

- it is effectively tied with the fastest local configurations
- it uses fewer threads than the absolute-fastest `8 x 256` setting
- it keeps memory lower while preserving nearly all of the speed

Example local run:

```bash
OMP_NUM_THREADS=4 NUMBA_NUM_THREADS=4 \
./.venv_jit/bin/python v6_batch/cluster_sweep_worker_batch.py \
  --n-total 100000 --seed 12345 --batch-size 256 --top-k 50 --out sweep_results.jsonl
```

Important caveat:

- these local-optimal settings should not be assumed to be cluster-optimal
- on Engaging, you observed a different winner: `1 CPU per task + batch_size=128`
- that is plausible because cluster nodes have different per-core speed, threading overhead, cache behavior, and scheduling characteristics
- always re-benchmark on the target platform before choosing production defaults

## How to run on cluster next time

After cloning the repo on the cluster login node:

```bash
cd /path/to/MHD_generator_solver_v6
python3 -m venv .venv_jit
./.venv_jit/bin/python -m pip install --upgrade pip setuptools wheel
./.venv_jit/bin/python -m pip install numpy==2.0.2 scipy==1.15.3 numba==0.61.2
./.venv_jit/bin/python v6_batch/smoke_test_batch_v6.py
```

For direct script runs:

```bash
PYTHON_BIN=./.venv_jit/bin/python
$PYTHON_BIN v6_core/non_batch/cluster_sweep_worker.py --n-total 1000 --top-k 50 --out sweep_results_scalar.jsonl
$PYTHON_BIN v6_batch/cluster_sweep_worker_batch.py --n-total 1000 --top-k 50 --out sweep_results.jsonl
```

Current layout note:

- V6 explicit local-closure core lives under `v6_core/`
- V6 batch/vectorized sweep pipeline lives under `v6_batch/`
- variable-area marginal-stability solver lives under `v6_global_marginal/`
- CasADi / NLP optimization workflows live under `v6_casadi/`

For Slurm jobs on this cluster, avoid `sbatch --export=VAR1=...` style submissions. We observed `user env retrieval failed -> held` around Slurm's get-user-env path, and explicit `--export=...` submissions are the pattern that repeatedly correlated with those failures.

Preferred V6 submission pattern:

```bash
./v6_batch/submit_v6_batch_array.sh
```

The wrapper now exports run variables in the submit shell, then calls `sbatch` without any `--export` flag (default environment propagation).

Default scheduler mode is now pull-based worker pool:

- `PULL_POOL_MODE=1`
- `SHARD_COUNT=100` (physics shards)
- `WORKER_COUNT=115` (redundant workers to absorb node loss)
- `ARRAY_CONCURRENCY=96`
- `TASK_MAX_ATTEMPTS=3`
- task pool root: `${OUT_DIR}/todo|processing|done|failed|attempts` (or override via `TASK_POOL_ROOT`)

Workers atomically claim `todo/shard_*.task` using rename, write `sweep_shard_<i>.jsonl`, mark `done/`, and idle workers can requeue stale `processing/` tasks after `TASK_STALE_TIMEOUT_S` to recover from node kills. Repeated shard exceptions are capped by `TASK_MAX_ATTEMPTS`; over-limit shards are moved to `failed/` instead of infinite requeue.

That `--no-requeue` default is intentional on this cluster. The observed `user env retrieval failed requeued held` failures only appeared after Slurm requeued array tasks (`Restarts=1`). Disabling requeue avoids that failure mode. The tradeoff is that if a shard is interrupted by the scheduler or node issues, it will stay failed and should be resubmitted explicitly instead of being silently retried.

If you want to override run parameters, set them explicitly:

```bash
N_TOTAL=100000000 \
OUT_DIR=results/run_1e8_example \
TIME_LIMIT=04:00:00 \
./v6_batch/submit_v6_batch_array.sh
```
To tune pull-pool redundancy:

```bash
SHARD_COUNT=100 \
WORKER_COUNT=130 \
ARRAY_CONCURRENCY=96 \
TASK_STALE_TIMEOUT_S=1200 \
TASK_MAX_ATTEMPTS=3 \
./v6_batch/submit_v6_batch_array.sh
```

If you identify flaky nodes, exclude them at submit time:

```bash
EXCLUDE_NODES=node1618,node2705 ./v6_batch/submit_v6_batch_array.sh
```

If you need to call `sbatch` directly, set variables in your shell first, then submit without `--export`:

```bash
export PYTHON_BIN=./.venv_jit/bin/python
export N_TOTAL=100000000
export TOP_K=100
export OUT_DIR=results/run_1e8_example
export BATCH_SIZE=256
export SLURM_EXPORT_ENV=ALL

sbatch -p mit_normal --array=0-99%96 \
  --job-name=mhd_v6_batch \
  --time=04:00:00 \
  --cpus-per-task=1 \
  --mem=1G \
  --no-requeue \
  v6_batch/submit_v6_batch_array.sbatch
```

Backup sweeper (non-batch worker) is available if you need a fallback lane:

```bash
./v6_core/non_batch/submit_v6_backup_array.sh
```

To resume an existing run and only fill missing shards, reuse the same `OUT_DIR` and `SHARD_COUNT`:

```bash
OUT_DIR=results/run_xxx SHARD_COUNT=100 ./v6_batch/submit_v6_batch_array.sh
```

`RESET_TASK_POOL=1` is guarded because it clears queue state. To confirm reset, set:

```bash
RESET_TASK_POOL=1 ALLOW_TASK_POOL_RESET=1 ./v6_batch/submit_v6_batch_array.sh
```

If `processing/` is non-empty and you are sure the old run is dead, additionally set `FORCE_RESET_TASK_POOL=1`.

## Batch tuning notes

- Current Engaging default for `submit_v6_batch_array.sbatch`:
  - `--cpus-per-task=1`
  - `BATCH_SIZE=256`
- Suggested first-pass Engaging comparison sweep:
  - keep `--cpus-per-task=1` fixed
  - compare `BATCH_SIZE=96, 128, 192, 256`
  - do not assume that increasing `cpus-per-task` improves single-job throughput; on Engaging it may reduce throughput
- Local MacBook tuning note:
  - use `4` CPU threads and `batch_size=256` for local runs/benchmarks

## If you want to reuse the old global env

Old V5 helpers preferred:

```bash
/home/yiruxiao/miniconda3/envs/mhd_search/bin/python
```

Right now that env has:

- Python `3.10.20`
- `numpy 2.2.5`
- `scipy 1.15.3`
- no `numba`

So it is not the recommended V6 JIT environment unless you deliberately add numba there.

## Troubleshooting

- `ModuleNotFoundError: No module named 'numpy'`
  - You are using the wrong interpreter. Use `./.venv_jit/bin/python`.
- `./.venv_jit/bin/python: No such file or directory`
  - This is expected immediately after cloning because `.venv_jit` is not committed. Create it with the setup commands above.
- `Python interpreter not found: ./.venv/bin/python`
  - Old V5 Slurm default. Override with `PYTHON_BIN=./.venv_jit/bin/python`.
- Numba import/version issues
  - Recreate the env with the pinned versions above instead of installing into `mhd_search`.
