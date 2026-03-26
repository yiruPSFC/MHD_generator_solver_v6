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
./.venv_jit/bin/python smoke_test_batch_v6.py
```

Expected result:

- `SMOKE TEST PASSED`
- The run may also print an OpenMP info line from the runtime before the smoke test output. That line is not a failure.

## How to run on cluster next time

After cloning the repo on the cluster login node:

```bash
cd /path/to/MHD_generator_solver_v6
python3 -m venv .venv_jit
./.venv_jit/bin/python -m pip install --upgrade pip setuptools wheel
./.venv_jit/bin/python -m pip install numpy==2.0.2 scipy==1.15.3 numba==0.61.2
./.venv_jit/bin/python smoke_test_batch_v6.py
```

For direct script runs:

```bash
PYTHON_BIN=./.venv_jit/bin/python
$PYTHON_BIN benchmark_batch_vs_lsoda.py --mode both
$PYTHON_BIN cluster_sweep_worker_batch.py --n-total 1000 --top-k 50 --out sweep_results.jsonl
```

If V6 later gets Slurm scripts, keep the same pattern as V5 and export:

```bash
PYTHON_BIN=./.venv_jit/bin/python
```

Example Slurm invocation pattern:

```bash
sbatch --export=ALL,PYTHON_BIN=./.venv_jit/bin/python your_job.sbatch
```

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
