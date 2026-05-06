# v6_maingo_casadi

Hybrid MAiNGO + CasADi workflow:

1. solve an 8D design problem over inlet conditions plus a 3-parameter `logA`
   spline,
2. lift the coarse path into an implicit full-space MAiNGO model with state and
   derivative variables on a backward-Euler mesh,
3. export the best MAiNGO profile as a warm-start NPZ,
4. hand that profile into `v6_casadi_v2.run_continuation(...)` for local
   refinement.

## Decision Variables

- `log_n_p_in`
- `T_e_in`
- `Z_in`
- `I_0`
- `log_seed_fraction`
- `a1`
- `a2`
- `a3`

The spline control points are fixed at `x/L = [0, 0.25, 0.5, 0.75, 1.0]` with
`logA = [0, a1, a2, a3, a3]`, so `A_in = 1` is enforced by construction.

The MAiNGO model also carries lifted trajectory variables on the coarse mesh:

- `n_p[k]`
- `T_e[k]`
- `dn_p_dx[k]`
- `dT_e_dx[k]`

Those auxiliary variables let the coarse model impose the active-segment
physics as implicit residual constraints instead of explicitly dividing by the
local Jacobian determinant. This is the key change that avoids the previous
`Inverse with zero in range` failure during global preprocessing.

The lifted trajectory variables are stored in MAiNGO as centered, dimensionless
`hat` variables around the implicit baseline-connected reference trajectory.
That scaling removes the previous badly-scaled variable domains and gives the
local NLPs a usable coordinate system.

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

Settings files live under `v6_maingo_casadi/settings/`. The workflow still
accepts the old top-level settings filenames for compatibility and resolves
them to the new folder when needed.

Main outputs:

- `maingo_summary.json`
- `maingo_best_profile.npz`
- `hybrid_summary.json`
- `continuation/`

`maingo_summary.json` records the MAiNGO retcode, incumbent objective, lower
bound, wall/CPU time, and the 8D design point passed into the CasADi handoff.
If the time-limited MAiNGO incumbent is only solver-tolerance feasible, the
wrapper records a feasibility-restoration step that projects the path back onto
the implicit dynamics and falls back along a baseline-to-incumbent homotopy.

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
- `build_seed.py`: writes case warm-profile and summary artifacts.
- `README.md`: records the case-specific artifact layout.

Compatibility shims remain at `yamasaki2004_parameters.py`,
`yamasaki2004_geometry.py`, and `build_yamasaki2004_disk_seed.py` for older
imports and command snippets.

New seed artifacts default to:

```text
v6_maingo_casadi/outputs/cases/yamasaki2004/seeds/
```

Older milestone outputs under `outputs/maingo_yamasaki2004_neighborhood/` are
left in place because their JSON files record those paths.

## Validation

- `validation/test_hybrid_components.py`
- `validation/smoke_test_hybrid_maingo_casadi.py`
