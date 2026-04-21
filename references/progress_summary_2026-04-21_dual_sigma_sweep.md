# Progress Summary 2026-04-21: v6 CasADi Dual Diagnostics and Sigma Sweep

## Context

The v6 CasADi workflow has produced plausible active-segment solutions, but the
best earlier shots were difficult to interpret physically. On 2026-04-21 we added
IPOPT/CasADi constraint-dual diagnostics to identify which constraints shape the
optimized profiles.

The immediate finding was that the apparent best relaxed-inlet-search solutions
were dominated by the nozzle geometry rate bound:

```text
sigma = d log(A) / dx
```

In the earlier best shot, `sigma_upper_interval` was the dominant multiplier and
`sigma_bound_hit_fraction` was 1.0. The profile was therefore close to a trivial
maximum-rate exponential nozzle expansion.

## Code Changes

- Added constraint-dual extraction to `v6_casadi/optimize_area_profile_casadi_v6.py`.
- Added dual arrays to continuation `.npz` artifacts and dual summary / dual plot
  paths to stage `.json` artifacts.
- Added `*_duals.png` plots for path constraints, area / sigma bounds, and Mach
  bounds in `v6_casadi/run_casadi_continuation_v6.py`.
- Added scoring gates in `run_relaxed_inlet_search_v6.py`:
  - reject shortcut stages that did not inherit an accepted continuation warm start
  - reject stage scores with `sigma_bound_hit_fraction > 0.98` by default
  - add a configurable sigma-bound-hit score penalty
- Added `run_sigma_max_sweep_v6.py` to sweep final-stage `max_abs_dlogA_dx` for a
  fixed inlet candidate and summarize performance plus dual bottlenecks.

## Strict-Gate Search Result

Re-running the `coal_mhd_htah` search with shortcut and sigma-saturation gates
changed the selected result. The previous final-stage shortcut was rejected. The
best strict-gate result was:

```text
candidate_index: 22
stage: stage_3_hs_release__bridge_1_of_3
L: 1.0 m
dTe: 600.94 K
sigma_bound_hit_fraction: 0.2875
warm_start_input_source: continuation:stage_2_hs_anchor
```

This result is more honest, but lower performance. Later stages for the same
candidate were rejected because they saturated the sigma upper bound.

## Sigma-Max Sweep for Candidate 022

A fixed-inlet sweep was then run for candidate 022 using:

```text
sigma_max = 0.5, 0.75, 1.0, 1.5, 2.0
L = 1.0 m
```

Summary:

| sigma_max | acceptable | solver status | dTe K | sigma hit fraction | dominant dual |
|---:|---|---|---:|---:|---|
| 0.5 | yes | Solve_Succeeded | 1884.82 | 1.0 | sigma_upper |
| 0.75 | yes | Solve_Succeeded | 3097.20 | 1.0 | sigma_upper |
| 1.0 | yes | Solve_Succeeded | 4681.34 | 1.0 | sigma_upper |
| 1.5 | no | Maximum_Iterations_Exceeded | 3881.46 | 0.025 | invalid / constraint failure |
| 2.0 | yes by diagnostics | Maximum_Iterations_Exceeded | 4203.00 | 0.0125 | G_lower_node |

The most important result is `sigma_max = 1.0`:

```text
T_e,in ~= 3193 K
dTe ~= 4681 K
T_e,out ~= 7874 K
L = 1.0 m
A_out / A_in ~= exp(1) ~= 2.718
return_status = Solve_Succeeded
acceptable = true
max_constraint_violation = 0
```

This is a strong lab-scale signal: within the current model, a 1 m active segment
with an aggressive but not absurd area expansion can raise electron temperature
from roughly 3.2 kK to 7.9 kK.

## Interpretation

The dual diagnostics support the engineering interpretation that the conservative
`sigma_max = 0.5` solution was geometry-rate limited, not physics-limited. Lifting
the sigma upper bound to 1.0 produces a large and clean performance gain while
remaining converged and feasible.

At `sigma_max = 2.0`, the top dual shifts from `sigma_upper_interval` to
`G_lower_node`, suggesting a bottleneck transition toward the inlet Velikhov
margin. However, that case ended with `Maximum_Iterations_Exceeded`, so it should
be treated as a qualitative signal rather than a trusted reference shot.

## Recommended Next Steps

1. Refine the sweep around the transition:

   ```text
   sigma_max = 1.0, 1.1, 1.2, 1.3, 1.4
   ```

2. Improve continuation robustness between `sigma_max = 1.0` and `1.5`.

3. Repeat the `sigma_max = 1.0` test across multiple inlet candidates to verify
   that the result is not a single-candidate artifact.

4. Run grid refinement for the `sigma_max = 1.0` candidate:

   ```text
   n_intervals = 80, 120, 160
   ```

5. Evaluate engineering quantities for the `sigma_max = 1.0` reference candidate:

   ```text
   E_x, J_x, J dot E, wall loading, seed fraction, magnetic energy, B-field feasibility
   ```
