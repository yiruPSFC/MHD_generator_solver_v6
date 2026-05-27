# v6_firedrake_reduced

This package is a sandbox for a Firedrake + pyadjoint reduced-functional
workflow for the quasi-1D MHD generator model.

The goal is to test a different optimization contract from the existing
CasADi/IPOPT direct transcription and the MAiNGO reduced-space search:

```text
design m = inlet/load variables + low-dimensional logA spline controls
state u(m) = Firedrake nonlinear forward solve of the implicit MHD residuals
score Jhat(m) = outlet enthalpy extraction / plasma quality diagnostics
gradient dJhat/dm = pyadjoint reduced-functional derivative
```

The optimizer does not directly move every intermediate profile node.  A design
is accepted for scoring only after the forward solve has produced a state
profile.  Failed forward solves are experiment results and must be written to
`failure_log.jsonl`; they are not hidden behind fake objective values.

## What This Is Not

- Not a replacement for `v6_maingo_casadi`.
- Not a MAiNGO branch-and-bound global certificate.
- Not a CasADi full-space collocation/transcription.
- Not an RK4 rollout.  RK4 remains useful only as a post-check or comparison
  path because it reintroduces explicit-RHS determinant division risk.

## Forward Solver Choice

The v0 forward solver is an implicit residual solve on a 1D Firedrake interval
mesh:

```text
state fields: log(n_p)(x), log(T_e)(x)
area control: A(x) = A_in * exp(logA_spline(x))
residuals: momentum residual = 0, energy residual = 0
solver: Firedrake NonlinearVariationalProblem + PETSc/SNES Newton
```

The residual is written in implicit form.  We do not solve explicitly for
`dn_p/dx = f(...)` and `dT_e/dx = g(...)`, because that route recreates the
singular chart problem seen in the explicit RK4/MAiNGO prototypes.

The weak residual rows are scaled by default with one inlet characteristic
scale per equation.  For each row, the scale is
`max(|coefficient * inlet_state / L|, |rhs|, 1)` evaluated at the inlet.  This
keeps the momentum and energy test-function rows in comparable numerical
units without adding extra nonlinear UFL expressions.  The older nodal
reference scaling remains available as `--residual-scaling characteristic`;
the raw dimensional form remains available as `--residual-scaling dimensional`.

## Design Variables

The default design vector mirrors the current reduced MAiNGO contract:

```text
log_n_p_in, T_e_in, Z_in, I_0, log_seed_fraction, a1, a2, a3
```

The area spline has knots at `x/L = [0, 1/3, 2/3, 1]` and values
`logA = [0, a1, a2, a3]`.  This keeps the inlet area fixed at `A_in` while
allowing the outlet area to vary independently.

## Case Data And Legacy Physics

Yamasaki-specific parameter data lives inside
`v6_firedrake_reduced/cases/yamasaki2004/`.  The canonical definitions are not
generated JSON artifacts under `outputs/`; they are local case data split into
paper values and model-seed windows.

The remaining compatibility layer, `legacy_physics.py`, is only for equations
and working-fluid helper functions that still come from `v6_maingo_casadi`.
It no longer supplies Yamasaki case parameters.

## Environment

Use a separate environment from the current `.venv_jit`:

```bash
python3 -m venv .venv_firedrake
# Install Firedrake according to the official Firedrake documentation for your
# platform.  Firedrake brings pyadjoint through its adjoint integration.
```

Local macOS note for this repo: PETSc rejects paths containing spaces, while
the workspace path contains `MHD Generator`.  The working local setup therefore
keeps PETSc and the real venv under:

```text
~/.cache/mhd_generator_solver_v6_firedrake/
```

and uses `.venv_firedrake` in the repo as a symlink to that venv.  Run Firedrake
commands with a clean Homebrew-oriented environment so Conda compiler variables
do not leak into CMake or JIT builds:

```bash
env -i \
  HOME="$HOME" USER="$USER" TMPDIR="${TMPDIR:-/tmp}" \
  PATH="/opt/homebrew/bin:/opt/homebrew/sbin:/usr/bin:/bin:/usr/sbin:/sbin" \
  OMP_NUM_THREADS=1 \
  PETSC_DIR="$HOME/.cache/mhd_generator_solver_v6_firedrake/petsc" \
  PETSC_ARCH=arch-firedrake-default \
  HDF5_MPI=ON HDF5_DIR=/opt/homebrew \
  ./.venv_firedrake/bin/python -m v6_firedrake_reduced.run_firedrake_reduced ...
```

Pure Python tests do not require Firedrake.  Firedrake smoke tests skip with a
clear message when Firedrake is unavailable.

## Example Calls

Baseline evaluation:

```bash
python -m v6_firedrake_reduced.run_firedrake_reduced \
  --case yamasaki2004 \
  --mode evaluate \
  --objective enthalpy_extraction \
  --n-area-controls 3 \
  --out-dir v6_firedrake_reduced/outputs/yamasaki2004_baseline
```

Local reduced-functional optimization:

```bash
python -m v6_firedrake_reduced.run_firedrake_reduced \
  --case yamasaki2004 \
  --mode optimize \
  --objective enthalpy_extraction \
  --n-area-controls 3 \
  --multistart 8 \
  --out-dir v6_firedrake_reduced/outputs/yamasaki2004_opt
```

By default, `--design-json` must lie inside the active case bounds; otherwise
the CLI exits with the offending variables.  Use
`--allow-out-of-bounds-design-json` only for explicit expanded-window
replay/audit runs where the saved design should be evaluated exactly.

`--geometry-length-mode` selects the Yamasaki streamwise length convention:
`radial` uses `r_exit-r_throat`, and `inferred_swirl` infers `ds/dr` from the
paper-reported swirl-flow cross sections.

`--electron-transport` selects the electron-heavy collision model.  `e-He`
uses the LXCat e-He elastic momentum-transfer median at 4300 K; `e-Argon`
uses the legacy cross-section value.  Arbitrary numeric `sigma_ep` overrides
are intentionally not part of the main interface.

Freidberg area-only benchmark:

```bash
env -u PETSC_DIR -u PETSC_ARCH OMP_NUM_THREADS=1 \
  ./.venv_firedrake/bin/python -m v6_firedrake_reduced.run_freidberg_area_only_benchmark \
  --mode optimize \
  --optimizer coordinate_search \
  --n-intervals 8 \
  --max-iterations 1 \
  --coordinate-initial-step 0.02 \
  --out-dir v6_firedrake_reduced/outputs/freidberg_area_only_smoke
```

The `freidberg_reference` case freezes all inlet/load controls at the
slide-recovered reference values and leaves only `a1/a2/a3` movable.  The runner
writes `reference_profile_metrics.json` and `benchmark_summary.json`; the latter
compares the optimized candidate against the recovered Freidberg profile using
the same metric evaluator.  Use `--mode reference` to regenerate only the
baseline metrics without invoking Firedrake.

Choking compatibility experiments are deliberately evaluate-only.  Use
`--reference-initial sonic_freidberg_hl` to run the pure-Python H/L sonic
compatibility matcher and stop before Firedrake if no smooth matched reference
profile is found.  Use `--reference-initial front_loaded_area` to create a
fixed area-ratio bracket curve that expands near the inlet and is then passed
to the Firedrake forward solve as a fixed area profile.  This is an initial
guess / bracketing tool, not an optimizer parameterization.

Example e-He bracketing run:

```bash
env -u PETSC_DIR -u PETSC_ARCH OMP_NUM_THREADS=1 \
  ./.venv_firedrake/bin/python -m v6_firedrake_reduced.run_firedrake_reduced \
  --case yamasaki2004 \
  --mode evaluate \
  --n-intervals 40 \
  --equation-form primitive \
  --electron-transport e-He \
  --reference-initial front_loaded_area \
  --front-loaded-area-ratio 6 \
  --front-loaded-area-width-fraction 0.05 \
  --out-dir v6_firedrake_reduced/outputs/yamasaki_front_loaded_area_bracket
```

`--freidberg-branch-audit` is a diagnostic add-on for evaluate/optimize runs.
It writes `freidberg_branch_audit.json` for a successful final profile, or
`freidberg_branch_audit_failed.json` when SNES returns a partial failed
profile.  The audit maps the primitive profile to the current H/L/T_e
coordinates, tries both subsonic and supersonic algebraic closure branches,
and records the selected branch residuals without changing the forward solve.

The v0 optimizer is a small projected-gradient/backtracking smoke driver, not
a global optimizer.  It optimizes normalized design coordinates internally and
accepts a trial only after the Firedrake forward replay succeeds.  Failed
trials are written to `failure_log.jsonl`; they are not reported as successful
objective values.

For a node-only hard Velikhov path-constraint experiment, use the constrained
SLSQP wrapper and keep the Velikhov objective mode diagnostic:

```bash
python -m v6_firedrake_reduced.run_firedrake_reduced \
  --case yamasaki2004 \
  --mode optimize \
  --objective enthalpy_extraction \
  --n-area-controls 3 \
  --optimizer constrained_slsqp \
  --velikhov-constraint-mode hard \
  --velikhov-hard-floor 0.0 \
  --velikhov-mode diagnostic \
  --out-dir v6_firedrake_reduced/outputs/yamasaki2004_hard_G_opt
```

This enforces only the forward-mesh node constraints `G_node - floor >= 0`.
It deliberately does not add midpoint constraints; spatial resolution should
be checked by replaying the resulting design on a finer mesh.

After a successful run, the reduced-space KKT diagnostic can be generated with:

```bash
python -m v6_firedrake_reduced.analyze_kkt \
  v6_firedrake_reduced/outputs/yamasaki2004_hard_G_opt/run_summary.json
```

The KKT report decomposes the full 8D reduced-control stationarity balance into
objective gradient, active node `G` constraints, active box bounds, and residual
terms.  It is a local diagnostic recovered by least squares, not a global
optimality certificate.

## Outputs

Each run directory should contain:

```text
run_summary.json
best_design.json
objective_history.csv
profile.npz
freidberg_branch_audit.json
failure_log.jsonl
README_snapshot.md
```

`profile.npz` is written only after a successful forward solve.  Failed starts
are appended to `failure_log.jsonl` with the design vector and error message.
The Freidberg branch audit file is present only when requested.

## Current Checkpoint, 2026-05-19

- The taped pyadjoint objective is now `enthalpy_extraction`, built from
  `100 * integral(-A J_x E_x dx) / H_in`, not the earlier `Te/Tp` proxy.
- Design controls are scalar Firedrake `Function`s on an `R` space.  Plain
  Firedrake `Constant` controls replayed as constants in assembled forms and
  produced zero reduced-functional variation.
- State variables are inlet-relative `delta_log(n_p)` and `delta_log(T_e)`.
  This keeps inlet Dirichlet values fixed at zero and avoids differentiating
  scalar controls through DirichletBC blocks.
- The default weak residual row scaling is now inlet-based and dimensionless:
  one fixed inlet characteristic scale for the momentum row and one for the
  energy row.  Run summaries also record the actual solver row scales.
- Velikhov margin can now be part of the reduced objective, not just a
  post-check.  `--velikhov-mode penalty` subtracts
  `weight * mean(max(floor - G, 0) / scale)^2` from the enthalpy score, with
  defaults matching the MAiNGO-side soft margin scale.
- Run metrics include raw enthalpy extraction, the applied Velikhov penalty,
  minimum Velikhov margin, Hall voltage, and electric power from Hall voltage.
- Verified smoke artifacts:
  `outputs/enthalpy_tape_baseline_20260519/` and
  `outputs/enthalpy_tape_opt2_pg_20260519/`.

## Transport-Corrected Yamasaki Status, 2026-05-20

The earlier 2026-05-19 recovery numbers in
`outputs/yamasaki_recovery_20260519/` are legacy/transport-ambiguous artifacts
and should not be cited as the current model status.  In particular,
`bounds_n40_eval_best_with_hall/` reported about `9.93%` extraction and
`expanded_I_Te_Z_n40_eval_best_with_hall/` reported about `28.22%`, but those
summaries do not record the working-fluid `sigma_ep` and predate the
artifact-visible LXCat e-He correction.  The likely source of the old mismatch
is the legacy electron-heavy collision cross section:
`3.942573033087758e-21 m^2`, whereas the current He/Cs default is the LXCat
e-He elastic momentum-transfer median at 4300 K,
`6.450369478794445e-20 m^2`.

Current corrected-transport artifacts do not yet support a full Yamasaki-window
recovery claim:

- Direct replay of the old best design at the full LXCat value fails in
  `outputs/lxcat_transport_20260519/replay_old_best_sigma_lxcat_median4300_n20/`
  with `DIVERGED_DTOL`.
- A historical arbitrary-`sigma_ep` continuation branch in
  `outputs/lxcat_transport_20260519/sigma_continuation_low_Z_I_seed3e-4_n20/`
  reaches `sigma_ep = 3.111285390715702e-20 m^2` with about `30.58%`
  extraction, `1.289 MW`, and `1612 V`, but then fails at
  `sigma_ep = 3.513289788531344e-20 m^2`, still below the full LXCat target.
  The current interface no longer exposes arbitrary intermediate cross-section
  values; use only `--electron-transport e-He` or `--electron-transport e-Argon`.
- Zero-delta direct solves at the full LXCat default in
  `outputs/direct_physics_20260520/low_Z_I_seed3e-4_lxcat_default_n20/` also
  fail with `DIVERGED_DTOL`.

The safe current statement is therefore narrower: the old recovery snapshot is
stale after the transport correction, and the corrected-transport model needs a
new continuation/recovery pass before quoting a best physical extraction number
or diagnosing the paper-window gap.
