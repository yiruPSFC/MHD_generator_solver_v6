# v6_maingo_casadi

Hybrid MAiNGO + CasADi workflow:

1. solve an 8D design problem over inlet conditions plus a 3-parameter `logA`
   spline,
2. evaluate each design through a reduced implicit fixed-Newton
   backward-Euler rollout,
3. repair the accepted coarse path when needed with a local CasADi Newton
   projection and baseline-to-incumbent homotopy,
4. export the best MAiNGO profile as a warm-start NPZ,
5. hand that profile into the area-scale-aware `v6_casadi_v2.run_continuation(...)`
   for local refinement when requested.

## Decision Variables

- `log_n_p_in`
- `T_e_in`
- `Z_in`
- `I_0`
- `log_seed_fraction`
- `a1`
- `a2`
- `a3`

The spline control points are fixed at `x/L = [0, 1/3, 2/3, 1]` with
`logA = [0, a1, a2, a3]`. This keeps the inlet shape factor normalized to one
while allowing the outlet area to remain an independent third area degree of
freedom. The physical inlet area is separate: `A_in = baseline.area_scale_m2`.
For old `v6_casadi_v2`-derived baselines this is `1.0`; for physical benchmark
seeds, such as Yamasaki 2004, it is the throat area in square meters.

`a1/a2/a3` are always direct spline coordinates in this module. Historical
Yamasaki seed files that used `area_reference_mode` or `area_reference` as
reference-geometry perturbations are intentionally rejected; regenerate those
seeds with `cases/yamasaki2004/build_seed.py` so the fitted paper geometry is
encoded directly in `aligned_area_window`.

The production MAiNGO problem keeps the global search low-dimensional: five
inlet/load variables plus three area variables. It does not expose every
coarse-mesh state as an independent global decision variable. Instead, for each
8D design point, the model internally reconstructs the coarse trajectory with a
fixed number of Gauss-Newton iterations on the backward-Euler implicit step:

- `n_p[k]`
- `T_e[k]`
- `dn_p_dx[k]`
- `dT_e_dx[k]`

This keeps the expression graph factorable for MAiNGO while avoiding the
explicit RHS division by the local Jacobian determinant that caused the previous
`Inverse with zero in range` failure during global preprocessing. The fixed
Newton rollout is intentionally not the final feasibility authority; accepted
candidates are rechecked in numeric post-processing before handoff.

## Formulation Rationale

The 8D reduced formulation is a practical compromise between global-search
coverage and deterministic solver cost. A full direct transcription would add
`n_p`, `T_e`, and derivative variables at every coarse mesh point, plus their
path constraints. That larger search space is still factorable in principle, but
it makes branch-and-bound pruning much harder: interval relaxations get wider,
many boxes survive longer, and MAiNGO can spend most of its time proving little
rather than improving the incumbent. Keeping MAiNGO focused on the low-dimensional
inlet/area design lets it search the physically meaningful controls while the
fixed-Newton rollout supplies a factorable trajectory map for objective and
constraint evaluation.

## Default Baseline Seed

By default, the workflow reads:

`v6_casadi_v2/outputs/continuation/baseline_release_from_v6_candidate_022_sigma_0p5/continuation_summary.json`

It uses that summary to recover:

- inlet nominal values and aligned box bounds,
- the warm profile NPZ for projecting the initial spline coefficients,
- the three-stage continuation schedule,
- the adaptive-bridge settings for the CasADi handoff.

## Dependency

This module requires the real `maingopy` Python binding. If `maingopy` is not
importable, the CLI fails immediately and does not fall back to random search or
another optimizer.

Typical installation:

```bash
pip install maingopy
```

If you built MAiNGO from source, make sure the resulting `maingopy` package is
visible to the same interpreter used for this repository.

## Entry Point

```bash
./.venv_jit/bin/python -m v6_maingo_casadi.run_hybrid_maingo_casadi \
  --out-dir v6_maingo_casadi/outputs/hybrid_default
```

To optimize MAiNGO directly for outlet enthalpy extraction and skip the CasADi
continuation handoff:

```bash
./.venv_jit/bin/python -m v6_maingo_casadi.run_hybrid_maingo_casadi \
  --objective-profile enthalpy_extraction \
  --skip-casadi-handoff \
  --out-dir v6_maingo_casadi/outputs/maingo_enthalpy
```

To start from a different search basin, pass an absolute search-window override
instead of relying on multiplicative bound factors:

```bash
./.venv_jit/bin/python -m v6_maingo_casadi.run_hybrid_maingo_casadi \
  --baseline-summary v6_maingo_casadi/outputs/cases/yamasaki2004/seeds/yamasaki2004_hecs_disk_geometry_reference_seed_summary.json \
  --search-window-json path/to/search_window.json \
  --out-dir v6_maingo_casadi/outputs/hybrid_new_basin
```

The search-window JSON can use either the direct package keys or the
`source_alignment.aligned_*` layout used by case seed summaries:

```json
{
  "inlet_windows": {
    "n_p_in": {"guess": 4.2e24, "min": 3.8e24, "max": 4.8e24},
    "T_e_in": {"guess": 5200.0, "min": 4700.0, "max": 6100.0},
    "Z_in": {"guess": 75.0, "min": 55.0, "max": 105.0},
    "I_0": {"guess": 950.0, "min": 700.0, "max": 1200.0},
    "seed_fraction": {"guess": 4.0e-4, "min": 1.5e-4, "max": 7.0e-4}
  },
  "area_design_windows": {
    "a1": {"guess": 0.05, "min": -0.2, "max": 0.2},
    "a2": {"guess": -0.03, "min": -0.25, "max": 0.15},
    "a3": {"guess": 0.08, "min": -0.1, "max": 0.3}
  }
}
```

The override is applied after loading the baseline summary and before any
`--*-lower-factor` or `--*-upper-factor` expansion. Use the JSON override when
the nominal point and basin should change; use the factor flags only to widen or
shrink a known baseline-aligned window.

Settings files live under `v6_maingo_casadi/settings/`. The workflow still
accepts the old top-level settings filenames for compatibility and resolves
them to the new folder when needed.

Main outputs:

- `maingo_summary.json`
- `maingo_coarse_profile.npz`: the MAiNGO coarse-grid profile actually checked by the reduced problem
- `maingo_handoff_profile.npz`: the dense exported profile used for handoff/postcheck
- `hybrid_summary.json`
- `continuation/`

`maingo_summary.json` records the MAiNGO retcode, incumbent objective, lower
bound, wall/CPU time, and the 8D design point passed into the CasADi handoff.
If the time-limited MAiNGO incumbent is only solver-tolerance feasible, the
wrapper records a feasibility-restoration step that projects the path back onto
the implicit dynamics and falls back along a baseline-to-incumbent homotopy.

The baseline-to-incumbent `alpha` fallback is a pragmatic repair mechanism, not
a trusted optimization strategy. It linearly interpolates the 8D design vector
from the baseline reference (`alpha = 0`) toward the MAiNGO incumbent
(`alpha = 1`) and keeps the largest projected point that passes the tightened
coarse feasibility checks. This is useful for salvaging a time-limited incumbent,
but it assumes the straight line between two designs crosses a usable feasible
region. Future replacements should prefer a real feasibility-restoration NLP, a
structured continuation over grouped variables or constraints, or a tighter
MAiNGO reduced formulation that makes this fallback rare.

## Area And Current Conventions

`I_0` is the total Hall current in amperes. The local current density is
computed from the area profile:

```text
J_x(x) = I_0 / A(x)
```

This distinction is invisible only in normalized runs where `A_in = 1`. In a
physical-area case, for example `A_in = 0.0069 m^2`, `I_0 = 1000 A` implies
`J_x_in ~= 1.45e5 A/m^2`. The summary dictionaries now report both:

- `I_0`, `I_0_A`, `total_current_A`: total current.
- `J_x_in`, `J_x_in_A_m2`, `current_density_in_A_m2`: inlet current density.
- `A_in`: physical inlet area used by the coarse MAiNGO model.

Search-window JSON should use `I_0` for the total-current box. The loader still
accepts legacy `jx_in`/`J_x_in` keys because older normalized summaries used
that name, but in this package those aliases are interpreted as `I_0`, not as a
physical current-density window.

The `v6_casadi_v2` continuation handoff now carries `objective_profile`,
`area_scale_m2`, and working-fluid constants through the warm-profile loader and
stage solves. Physical-area handoffs should preserve `A[0] = area_scale_m2` and
infer total current as `I_0 = J_x[0] * A[0]`.

When a result looks suspicious, check these fields first:

- `hybrid_summary.json -> baseline_seed.area_scale_m2`
- `maingo_summary.json -> coarse_best.inlet_design.A_in`
- `maingo_summary.json -> coarse_best.inlet_design.I_0_A`
- `maingo_summary.json -> coarse_best.inlet_design.J_x_in_A_m2`
- `maingo_coarse_profile.npz -> A[0]` and `J_x[0]`
- `maingo_handoff_profile.npz -> A[0]` and `J_x[0]`
- `maingo_summary.json -> objective_profile`
- `hybrid_summary.json -> continuation.skipped` and `continuation.skip_reason`

If a downstream artifact says `A_in = 1.0`, it is in the normalized-area
convention and should not be compared directly against physical-area power,
mass-flow, or enthalpy-flux numbers without an explicit conversion.

## Code Layout

`core.py` is now only a compatibility facade for older imports. The implementation
is split by role:

- `constants.py`, `profiles.py`, `numerics.py`: shared constants, working-fluid
  profiles, objective-profile helpers, and backend-agnostic math operations.
- `geometry.py`, `models.py`: area spline geometry and result/seed dataclasses.
- `physics.py`: closure equations, inlet construction, dynamics, RK4 rollout,
  and coarse objective evaluation.
- `casadi_evaluator.py`: explicit CasADi rollout function and numeric evaluator.
- `implicit.py`: lifted implicit trajectory variables, scaling, reference
  construction, projection, and feasibility restoration.
- `maingo_models.py`, `workflow.py`: MAiNGO model adapters and the full
  MAiNGO-to-CasADi handoff workflow.
- `settings/`: MAiNGO runtime settings.
- `cases/`: benchmark-specific mappings and scripts that are not generic solver
  engine code.

## Yamasaki 2004 Case

The Yamasaki 2004 CCMHD benchmark mapping lives in
`cases/yamasaki2004/`:

- `parameters.py`: paper values, disk geometry, and model-neighborhood seed.
- `build_seed.py`: projects the paper geometry into the same 3-parameter
  `SplineAreaDesign`, then writes case warm-profile and summary artifacts.
- `README.md`: records the case-specific artifact layout.

Use the case package directly for new code and commands:

```bash
./.venv_jit/bin/python -m v6_maingo_casadi.cases.yamasaki2004.build_seed
```

New seed artifacts default to:

```text
v6_maingo_casadi/outputs/cases/yamasaki2004/seeds/
```

Older milestone outputs under `outputs/maingo_yamasaki2004_neighborhood/` are
left in place because their JSON files record those paths.

## Validation

- `validation/test_hybrid_components.py`
- `validation/smoke_test_hybrid_maingo_casadi.py`
