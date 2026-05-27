# v6_casadi_v2

Experimental active-segment CasADi workflows where inlet design variables are
optimized inside the NLP instead of being searched by an outer random loop.

Current scope:

- single-segment solver with inlet decision variables
- continuation driver for a single bounded inlet-design problem
- relaxed continuation runner and smoke test
- baseline-release runner that reconstructs a `v2` continuation from a `v6`
  sigma sweep warm profile
- staged-release IPOPT workflow: baseline anchor -> `n_p/T_e` -> `Z` ->
  `I_0/seed_fraction` -> geometry/objective
- experiment JSON support for inlet windows, bound factors, and schedule
  overrides

Key modeling choices:

- `A_in = area_scale_m2`; the old normalized workflow is the default
  `area_scale_m2 = 1.0`
- `I_0` is total current; local current density is `J_x(x) = I_0 / A(x)`
- inlet decision variables are `n_p_in`, `T_e_in`, `Z_in`, `I_0`, and
  `seed_fraction`
- `T_p_in` is still derived from the inlet closure
- downstream stage schedule remains close to `v6_casadi`
- `objective_profile=lab_poc_v2` is the lab heating proxy;
  `objective_profile=enthalpy_extraction` is the Yamasaki-style enthalpy
  extraction score

Useful entry points:

- `run_casadi_continuation_v2.py`
- `runners/run_relaxed_continuation_v2.py`
- `runners/run_baseline_release_from_v6_sweep_v2.py`

The baseline-release runner now writes:

- `experiment_definition.json`: source summary, inlet windows, bound factors,
  selected objective/area/working-fluid semantics, and the exact stage schedule
- `aligned_release_schedule.json`: the schedule actually sent to IPOPT
- `continuation_summary.json`: trusted/failing stages, active-set summaries,
  objective breakdowns, and energy-budget audits for each stage

Example staged run:

```bash
./.venv_jit/bin/python v6_casadi_v2/runners/run_baseline_release_from_v6_sweep_v2.py \
  --schedule-mode staged-release \
  --search-window-json v6_casadi_v2/experiments/windows/example.json \
  --objective-profile lab_poc_v2 \
  --out-dir v6_casadi_v2/outputs/continuation/<run-name>
```

Latest progress note:

- `PROGRESS_2026-04-22.md`
