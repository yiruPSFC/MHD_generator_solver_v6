# v6_casadi

Stable active-segment CasADi workflows.

Recommended mental model:

- [`optimize_area_profile_casadi_v6.py`](./optimize_area_profile_casadi_v6.py)
  Core NLP solver for the active segment.
- [`run_casadi_continuation_v6.py`](./run_casadi_continuation_v6.py)
  General continuation driver shared by the stable workflows.
- [`runners/`](./runners)
  Daily entry points for reference and relaxed continuation runs.
- [`validation/`](./validation)
  Smoke tests and one-off validation / refinement scripts.
- [`outputs/continuation`](./outputs/continuation)
  Saved continuation artifacts.

Practical rule:

- if you want reliable active-segment studies, start from `v6_casadi`
- if you want the more physical but currently unstable multi-stage prototype, use [`v6_piecewise`](../v6_piecewise)
