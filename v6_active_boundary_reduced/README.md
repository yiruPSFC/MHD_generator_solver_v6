# v6_active_boundary_reduced

Prototype for a reverse-only active-boundary preparation solver.

The only intended workflow in this folder is:

```text
given:  a hard-to-prepare target state
return: an upstream preparation profile that could produce it
```

The target state is passed as an anchor, either from JSON or from a profile
node.  The solver then marches upstream only.  The physical coordinate is `x`,
but the preparation marching coordinate is upstream from the target:

```text
x_next = x_current - dx
```

At each step, the current state is `(log_n, log_Te, logA)`.  The only local
pedal is `sigma = d(log A)/dx`.  Area is not an independent control:

```text
logA_next = logA - dx * sigma
```

The default step solver treats the common `G`-limited case as a local
three-equation algebraic solve:

```text
unknowns:  log_n_next, log_Te_next, sigma
equations: primitive momentum residual = 0
           primitive energy residual   = 0
           G_next - G_floor             = 0
```

The hot physical closure and dynamic-term formulas are implemented in
`numba_physics.py` with `numba.njit(cache=True)`.  The older scan-and-refine
policy is kept as a fallback for non-`G` active sets or failed local solves.
For reverse preparation, the `G`-boundary solve is warm-started by linearly
extrapolating the last two accepted generated states and the last two accepted
`sigma` values.  This does not change the pedal-direction test; it only changes
the nonlinear solver initial guess.

Current prototype constraints:

```text
G_next >= G_floor
Tp_next >= Tp_floor
sigma_min <= sigma <= sigma_max
logA_min <= logA_next <= logA_max
|sigma - sigma_prev| <= curvature_max   optional
```

This folder is intentionally independent of IPOPT/CasADi.  It is a direct
reverse preparation rollout prototype that can later feed inlet-condition
optimization and Firedrake/pyadjoint validation once the active-set semantics
are clear.

Example Freidberg-inlet preparation recovery:

```bash
./.venv_jit/bin/python -m v6_active_boundary_reduced.run_preparation_recovery \
  --case freidberg_reference \
  --anchor-profile-index 0 \
  --dx 0.01 \
  --n-steps 60 \
  --out-dir outputs/preparation_recovery_freidberg_inlet
```

## Anchor-design scan and optimization

The optimization layer uses the same design vocabulary as
`v6_firedrake_reduced.design.DesignVector`, except the area spline variables
`a1`, `a2`, and `a3` are fixed to the case baseline.  The active-boundary
rollout generates area from `sigma = dlogA/dx`, so those spline controls are
not search variables here.

Search variables:

```text
log_n_p_in, T_e_in, Z_in, I_0, log_seed_fraction, B_T
```

For a candidate design, the target anchor is parameterized as:

```text
log_n = log_n_p_in
log_Te = log(T_e_in)
logA = --anchor-logA          default 0
sigma = --anchor-sigma        default Freidberg profile index 0 when available
```

The score is a soft-penalty objective:

```text
score =
  delta_improvement_weight * (Delta_outlet - Delta_preparation_inlet)
  - inlet_delta_weight * Delta_preparation_inlet
  - inlet_te_shortfall_weight * max(6000 K - Te_preparation_inlet, 0)^2 / scale^2
  - inlet_tp_shortfall_weight * max(3000 K - Tp_preparation_inlet, 0)^2 / scale^2
```

Parallel grid scan:

```bash
./.venv_jit/bin/python -m v6_active_boundary_reduced.run_anchor_scan \
  --case freidberg_reference \
  --dx 0.01 \
  --n-steps 60 \
  --range T_e_in 5800 6800 5 \
  --range B_T 8 14 7 \
  --workers 4 \
  --out-dir outputs/anchor_scan_example
```

Two-stage coarse/refined scan.  This scans all candidates with the coarse mesh,
then re-evaluates only the top 20 coarse candidates with a finer mesh over the
same physical preparation length:

```bash
./.venv_jit/bin/python -m v6_active_boundary_reduced.run_anchor_scan \
  --case freidberg_reference \
  --dx 0.02 \
  --n-steps 270 \
  --refine-top-k 20 \
  --refine-dx 0.005 \
  --range T_e_in 5800 6800 5 \
  --range B_T 8 14 7 \
  --workers 32 \
  --out-dir outputs/anchor_scan_two_stage
```

The coarse pass writes `scan_results.csv/jsonl`; the refined pass writes
`refined_results.csv/jsonl`.  When refinement is enabled, `scan_summary.json`
reports the final `best` and `top10` from the refined pass, while also keeping
the coarse ranking under the `coarse` key.

Early-stage SciPy optimization:

```bash
./.venv_jit/bin/python -m v6_active_boundary_reduced.run_anchor_optimize \
  --case freidberg_reference \
  --dx 0.01 \
  --n-steps 60 \
  --bound T_e_in 5500 7500 \
  --bound B_T 6 16 \
  --bound I_0 1000 2500 \
  --maxiter 8 \
  --popsize 6 \
  --out-dir outputs/anchor_optimize_example
```

## Outer L-BFGS-B reduced-model optimizer

The outer reduced-model optimizer lives under `outer_solver/`.  It optimizes
the five primary inlet/anchor controls

```text
log_n_p_in, T_e_in, Z_in, I_0, log_seed_fraction
```

where `log_n_p_in` is the internal representation of physical `np_in`.  The
CLI accepts `--bound np_in ...` or `--bound n_p_in ...` and log-transforms the
range.  `B_T` is fixed by default and can be set with `--fixed B_T VALUE`.

The prescreening stage samples candidates and keeps states with non-negative
`G`, low anchor `T_e/T_p`, and positive local `d(T_e/T_p)/dx` near the anchor.
The selected seeds are then improved with SciPy `L-BFGS-B` in normalized
`[0, 1]^5` space.

Example:

```bash
./.venv_jit/bin/python -m v6_active_boundary_reduced.outer_solver.lbfgsb \
  --case yamasaki2004 \
  --dx 0.01 \
  --n-steps 60 \
  --bound np_in 2.0e25 5.0e25 \
  --bound T_e_in 5000 7000 \
  --bound Z_in 70 110 \
  --bound I_0 1000 1800 \
  --bound log_seed_fraction -13.8 -9.2 \
  --fixed B_T 12.0 \
  --prescreen-candidates 128 \
  --prescreen-top-k 8 \
  --maxiter 24 \
  --out-dir outputs/active_boundary_outer_lbfgsb_example
```

The reward is a soft-penalty objective built from the active-boundary rollout:
`Delta` improvement, magnetic-field range, `Amax/Amin` range, too-low minimum
`T_p`, too-high maximum `T_e`, `G` shortfall, choking/Mach excess, and incomplete
rollout penalties.
