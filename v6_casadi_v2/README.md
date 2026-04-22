# v6_casadi_v2

Experimental active-segment CasADi workflows where inlet design variables are
optimized inside the NLP instead of being searched by an outer random loop.

Phase-1 scope:

- single-segment solver with inlet decision variables
- continuation driver for a single bounded inlet-design problem
- relaxed continuation runner and smoke test
- baseline-release runner that reconstructs a `v2` continuation from a `v6`
  sigma sweep warm profile

Key modeling choices:

- `A_in = 1.0` is fixed and normalized
- inlet decision variables are `n_p_in`, `T_e_in`, `Z_in`, `J_x_in`
- `T_p_in` and `seed_fraction` are solved implicitly through inlet relations
- downstream stage schedule remains close to `v6_casadi`

Useful entry points:

- `run_casadi_continuation_v2.py`
- `runners/run_relaxed_continuation_v2.py`
- `runners/run_baseline_release_from_v6_sweep_v2.py`

Latest progress note:

- `PROGRESS_2026-04-22.md`
