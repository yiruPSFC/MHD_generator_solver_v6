# v6_active_boundary_reduced

Prototype for a reverse-only active-boundary preparation solver.

## Code organization

The package is organized by functionality:

```text
core/          stable-ish solver logic, physics kernels, policy, objectives
diagnostics/   rollout summaries, diagnostic plots, postprocessing assets
runners/       command-line entrypoints and file-oriented workflow glue
outer_solvers/ early-stage outer optimization prototypes
validation/    smoke and behavior-regression tests
```

`outer_solvers/` is intentionally separated from `core/`: the current
low-dimensional outer optimization path is useful for experiments, but its
results are not yet stable or easy to interpret as solver behavior.

The main active-boundary solver workflow in this folder is:

```text
given:  a hard-to-prepare target state
return: an upstream preparation profile that could produce it
```

The target state is passed as an anchor, either from JSON, from a profile node,
or from design variables in the scan/optimization runners.  The solver then
marches upstream only.  The physical coordinate is `x`, but the preparation
marching coordinate is upstream from the target:

```text
x_next = x_current - dx
```

At each step, the current state is `(log_n, log_Te, logA)`.  The only local
pedal is `sigma = d(log A)/dx`.  Area is not an independent control:

```text
logA_next = logA - dx * sigma
```

Ordinary non-sonic finite steps use explicit RK4 for local state advancement:

```text
given:   current state, sigma, dx
return:  next log_n, log_Te, logA
accept:  RK4 error estimate, G_next, and Tp_next satisfy local gates
```

The hot physical closure and dynamic-term formulas are implemented in
`core/numba_physics.py` with `numba.njit(cache=True)`.  The reverse sign-aware
path first tries the objective-preferred affine endpoint.  If finite-step
validation fails because the `G` floor was crossed, it can try a sign-aware
Brent `G`-boundary fallback before falling back to scan-and-refine endpoint
selection.  There is no longer an implicit backward-Euler step backend, and
the active-boundary reduced tree is no longer connected to IPOPT/CasADi.

Current prototype constraints:

```text
G_next >= G_floor
Tp_next >= Tp_floor
sigma_min <= sigma <= sigma_max
logA_min <= logA_next <= logA_max
|sigma - sigma_prev| <= curvature_max   optional sigma-slew bound
```

The `runners/` directory still contains diagnostic and benchmark entrypoints,
but the intended solver architecture is the direct reverse preparation rollout.
That rollout feeds the early-stage inlet-condition optimization prototypes and
can later be compared against Firedrake/pyadjoint validation once the active-set
semantics are clear.

Example Freidberg-inlet preparation recovery:

```bash
./.venv_jit/bin/python -m v6_active_boundary_reduced.runners.run_preparation_recovery \
  --case freidberg_reference \
  --anchor-profile-index 0 \
  --dx 0.01 \
  --n-steps 60 \
  --out-dir outputs/preparation_recovery_freidberg_inlet
```

## Sonic-compatible reverse-policy branch

Near `M=1`, the primitive momentum/energy matrix becomes singular.  The
reverse preparation policy therefore does not divide through that matrix when
the current state is near choking.  Instead it computes the left-null
compatibility condition

```text
ell^T (f0 + A * sigma_* * f1) = 0
```

For reverse `delta_drop` steps, `sonic_mode=auto` switches from the ordinary
sign-aware affine endpoint to the primitive left-null solve when the current
state is near `M=1` or has small `det_D`.  In that branch the area pedal is no
longer scanned as a free interval; it is set by

```text
A_prime_sonic = - ell^T f0 / ell^T f1
sigma_sonic = A_prime_sonic / A
```

and the finite step is then accepted only if the usual residual, `G`, and
`T_p` checks pass.  The old standalone sonic-delta profile CLI has been
removed; sonic handling now lives in the main policy path.  The relevant
settings, exposed on the rollout/scan/outer optimizer CLIs as matching dashed
flags, are `sonic_mode`, `sonic_mach_tol`, `sonic_det_abs_tol`,
`sonic_compatibility_tol`, and `sonic_residual_tol`.

## Reverse objective support

For reverse preparation, use `--objective delta_drop`.  Some CLIs still expose
`power_next` as a visible legacy/diagnostic option, but the main reverse policy
only supports reverse `delta_drop`; reverse non-`delta_drop` objectives terminate
with `unsupported_reverse_objective` instead of silently using old scan
semantics.  Forward diagnostics and benchmark scripts may still use their own
objective choices where they implement them directly.

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
sigma = --anchor-sigma        default None
```

When `--anchor-sigma` is omitted, the first rollout step has no slope history,
so the optional `curvature_max` slew bound is disabled until a real previous
`sigma` exists.  The design-anchor path does not automatically inherit the
Freidberg profile's inlet sigma.

The score is a soft-penalty objective:

```text
score =
  delta_improvement_weight * (Delta_outlet - Delta_preparation_inlet)
  + mhd_output_power_weight * mhd_output_power_MW                 optional
  + enthalpy_extraction_weight * raw_enthalpy_extraction_percent  optional
  - inlet_delta_weight * Delta_preparation_inlet
  - inlet_te_shortfall_weight * max(6000 K - Te_preparation_inlet, 0)^2 / scale^2
  - inlet_tp_shortfall_weight * max(3000 K - Tp_preparation_inlet, 0)^2 / scale^2
```

The rollout must also pass its own gates, target-anchor `G/T_p` checks, and any
profile-metric availability checks required by nonzero optional reward weights;
otherwise the score is replaced by the configured failure penalty.

Parallel grid scan:

```bash
./.venv_jit/bin/python -m v6_active_boundary_reduced.runners.run_anchor_scan \
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
./.venv_jit/bin/python -m v6_active_boundary_reduced.runners.run_anchor_scan \
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
./.venv_jit/bin/python -m v6_active_boundary_reduced.runners.run_anchor_optimize \
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

The outer reduced-model optimizer lives under `outer_solvers/`.  It optimizes
the five primary inlet/anchor controls

```text
log_n_p_in, T_e_in, Z_in, I_0, log_seed_fraction
```

where `log_n_p_in` is the internal representation of physical `np_in`.  The
CLI accepts `--bound np_in ...` or `--bound n_p_in ...` and log-transforms the
range.  `B_T` is fixed by default and can be set with `--fixed B_T VALUE`.

The prescreening stage samples candidates and keeps states with non-negative
`G`, low anchor `T_e/T_p`, and positive local `d(T_e/T_p)/dx` near the anchor.
The selected seeds are not handed directly to SciPy.  Each seed is first
certified with normalized `+/- --neighborhood-eps` perturbations in every
control direction.  `L-BFGS-B` only runs from seeds whose whole local
neighborhood completes the rollout.  The final `best` is also chosen only from
top feasible evaluations that pass the same neighborhood certification.

This keeps accept/fail boundary probes out of the final optimizer result:
`best_seen_uncertified` records the highest feasible point encountered during
raw function evaluation, while `best` is reserved for a locally robust feasible
profile.  If no robust seed exists, the run writes diagnostics and skips
`L-BFGS-B` by default.

Example:

```bash
./.venv_jit/bin/python -m v6_active_boundary_reduced.outer_solvers.lbfgsb \
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
  --neighborhood-eps 0.001 \
  --certify-top-k 8 \
  --maxiter 24 \
  --out-dir outputs/active_boundary_outer_lbfgsb_example
```

Use `--allow-nonrobust-lbfgsb` only for diagnostics.  That option still requires
post-run neighborhood certification before a point appears as `best`.

The reward is a soft-penalty objective built from the active-boundary rollout:
`Delta` improvement, optional MHD output power, magnetic-field range,
`Amax/Amin` range, too-low minimum `T_p`, too-high maximum `T_e`, `G` shortfall,
choking/Mach excess, incomplete rollout penalties, and a failure penalty for
unsuccessful rollouts.
