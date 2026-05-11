# v6_maingo_casadi research plan, 2026-05-11

## Scope

This plan maps the recent idea notes onto the current `v6_maingo_casadi`
implementation. The goal is not to tune blindly. Each item should produce an
artifact that separates:

- raw MAiNGO incumbent,
- reduced-implicit feasibility restoration,
- dense handoff profile,
- optional `v6_casadi_v2` continuation result,
- post-hoc physical/KKT diagnostics.

The current production coarse model is `reduced_implicit_fixed_newton_backward_euler`.
It is an 8-variable search by default:

```text
log_n_p_in, T_e_in, Z_in, I_0, log_seed_fraction, a1, a2, a3
```

`--critical-mode` adds a ninth variable, `x_sonic`, but this is only a sonic
compatibility residual. It is not yet a full transonic branch construction.

## Current implementation map

| Topic | Current state in `v6_maingo_casadi` | Trust level |
| --- | --- | --- |
| `lab_poc_v2` objective | Implemented in `profiles.py` and `physics.py`. It now rewards outlet `Te`, outlet `Te/Tp`, and outlet ionization; magnetic-field and device-length penalties are zero. | Usable, but still a reward-shaped proxy. |
| `enthalpy_extraction` objective | Implemented as `--objective-profile enthalpy_extraction`; reports percent of inlet stagnation enthalpy flux. | Usable for Yamasaki-style comparisons. |
| Inlet-parameter relaxation | Implemented through `--search-window-json` and per-variable lower/upper factor flags. | Usable for positive inlet windows. |
| Physical area/current convention | Implemented: `I_0` is total current and `J_x(x) = I_0 / A(x)`. Summaries include both. | Usable; must check `A_in`. |
| He/Cs Yamasaki case | Implemented under `cases/yamasaki2004/`; paper constants and model seed are separated. | Usable, but paper-window and expanded-neighborhood claims must stay separate. |
| Reduced implicit path | Implemented and now the production MAiNGO path. RK4 is only a post-hoc benchmark. | Usable; accepted points still require numeric recheck. |
| CasADi handoff bridge/fallback | Implemented in `v6_casadi_v2`: adaptive bridges, `bridge_stop`, `final_trusted`, `failed_attempts`, area-scale-aware warm profile loading. | Usable; summaries must report trusted vs failed stages separately. |
| KKT-like analysis | Implemented in `analyze_hybrid_solution.py` by rebuilding a fixed-curve CasADi operator and recovering active-set multipliers. | Useful diagnostic; not a MAiNGO certificate. |
| Sonic point / `M=1` | Partially implemented as opt-in `--critical-mode`; not yet a true crossing or domain split. | Experimental only. |
| Higher B search | `B` is loaded from the baseline summary; no first-class CLI sweep/design variable yet. | Possible by seed editing; needs cleaner workflow. |
| Negative `Z` / denominator singularity | The equations can evaluate signed `Z`, but search-window validation currently treats inlet windows as positive. | Not explored cleanly yet. |
| Piecewise current | Not implemented. Current is total constant `I_0`; local `J_x` only changes through `A(x)`. | New model branch required. |
| Swirl as a generic design knob | Yamasaki disk geometry is implemented as a fixed effective area/length mapping. A free swirl ratio is not a generic optimization variable. | Case-specific only. |

## Work plan sorted by effort

| Effort | Idea | What to implement or call | Code touchpoints | Example call | Required artifact / decision gate |
| --- | --- | --- | --- | --- | --- |
| XS | Confirm `v6_casadi_v2` bridge/fallback is active in current MAiNGO handoffs | No new code. For any run with handoff, inspect `hybrid_summary.json` and `continuation/continuation_summary.json` for `bridge_stop`, `final_trusted`, `failed_attempts`, `stopped_after_failed_bridge`. | `v6_maingo_casadi/workflow.py`, `v6_casadi_v2/run_casadi_continuation_v2.py` | `jq '.continuation' <out>/hybrid_summary.json` and `jq '{bridge_stop, final_trusted_return_status, failed_attempts}' <out>/continuation/continuation_summary.json` | A run is trusted only if the reported final trusted stage is acceptable. Failed final attempts must remain visible. |
| XS | Compare existing reward profiles: lab proxy vs enthalpy extraction | No new code. Run the same seed/window twice with `--objective-profile lab_poc_v2` and `--objective-profile enthalpy_extraction`, preferably with `--skip-casadi-handoff` first. | `profiles.py`, `physics.py`, `workflow.py` | `./.venv_jit/bin/python -m v6_maingo_casadi.run_hybrid_maingo_casadi --baseline-summary <seed.json> --objective-profile enthalpy_extraction --skip-casadi-handoff --maingo-settings maingo_branch_and_bound_yamasaki.txt --out-dir <out>` | Compare `value_profile`, `outlet_enthalpy_extraction_percent`, `outlet_delta_Te_K`, `I_0`, `A_in`, `min_path_G_all`, and restoration alpha. |
| XS | Relax inlet parameter constraints | Already supported. Use `--search-window-json` for deliberate basin changes, or factor flags for widening a known basin. | `models.py`, `run_hybrid_maingo_casadi.py` | `--search-window-json v6_maingo_casadi/experiments/windows/<name>.json` | Save the exact window JSON next to the output. Treat the window as part of the experiment definition. |
| S | Higher magnetic-field scan | Add a first-class `--B-override` CLI option or generate B-specific seed summaries. Prefer CLI override for scans, but record the override in `baseline_seed` and `maingo_summary.json`. | `run_hybrid_maingo_casadi.py`, `workflow.py`, `models.py` | `./.venv_jit/bin/python -m v6_maingo_casadi.run_hybrid_maingo_casadi --baseline-summary <seed.json> --B-override 4.0 --objective-profile enthalpy_extraction --skip-casadi-handoff --out-dir <out>` | Plot score, feasibility restoration alpha, min `G`, min `Tp`, min Mach, and integrated power vs B. Reject runs that improve only by restoration fallback. |
| S | Baseline-sampling sensitivity | Add a small experiment runner that reads a seed summary plus a grid/list of window JSON files, launches MAiNGO runs, and writes one CSV/JSON summary. This is not a physics change. | New `experiments/run_window_sweep.py`, maybe `experiments/README.md` | `./.venv_jit/bin/python -m v6_maingo_casadi.experiments.run_window_sweep --baseline-summary <seed.json> --window-dir <windows> --objective-profile enthalpy_extraction --max-time 300` | Decide whether results are basin-dependent by comparing best score, active bounds, restoration alpha, and KKT active set across windows. |
| S-M | `JyB` dominance diagnosis | First add diagnostics, not a reward. Compute path summaries for `J_y * B`, `A * J_y * B`, `J_x * E_x`, and their correlation with `dTe/dx`, `dTp/dx`, and objective contributions. | `physics.py`, `models.py`, `audit_maingo_profile.py`, `analyze_hybrid_solution.py` | `./.venv_jit/bin/python -m v6_maingo_casadi.audit_maingo_profile <out>/maingo_summary.json` | If high-score runs are dominated by large `J_y B` while constraints are healthy, then add a dedicated `jyB_proxy` objective profile. |
| M | Alternative reward: lab availability | Add a new objective profile such as `lab_availability` or `enthalpy_with_lab_limits`. It should score enthalpy extraction or `Te` gain while penalizing unavailable lab conditions: B, total current, device length, thermal input, mass flow, and maybe Hall voltage mismatch. | `constants.py`, `profiles.py`, `physics.py`, `v6_casadi_v2/optimize_area_profile_casadi_v2.py` if handoff must optimize the same profile | `--objective-profile lab_availability` | Summary must report every reward term separately. Do not accept a result whose gain comes mostly from hidden scaling or constraint relaxation. |
| M | `1 - C partial_Te F` singularity probe | Expose `dTp_dTe = 1 - C * dF_dTe`, `dTp_dnp`, `dTp_dA`, determinant, and critical numerators as saved arrays. Then add a diagnostic scan/objective that searches near small `abs(dTp_dTe)` without violating `Tp`, `G`, or Mach constraints. | `physics.py`, `models.py`, `reduced_implicit.py`, `audit_maingo_profile.py` | `./.venv_jit/bin/python -m v6_maingo_casadi.audit_maingo_profile <out>/maingo_summary.json --include-jacobian-terms` | Gate: find whether small `abs(1 - C partial_Te F)` is a real feasible mechanism or only a numerical cliff. |
| M | Negative `Z` / `beta^2 + 1 + Z` denominator exploration | Allow signed `Z_in` windows while keeping positive constraints on `n_p`, `T_e`, `I_0`, and `seed_fraction`. Add explicit diagnostics for inlet and path denominator margins. | `models.py`, `physics.py`, `audit_maingo_profile.py`, tests | Search-window JSON with `"Z_in": {"guess": -0.5, "min": -5.0, "max": 5.0}` after code support | Gate: require positive `Tp`, stable Velikhov margin, bounded denominator margin, and no MAiNGO safe-denominator artifact. |
| M | Make KKT analysis part of every serious run | Wrap `analyze_hybrid_solution.py` into an optional post-run step, or provide a `make_analysis_report.py` that reads `hybrid_summary.json`, `maingo_summary.json`, and profile NPZ. | `analyze_hybrid_solution.py`, new report script | `./.venv_jit/bin/python -m v6_maingo_casadi.analyze_hybrid_solution <out>/hybrid_summary.json` | Every candidate should list active bounds, active path constraints, recovered multipliers, reward contributions, and whether the result is box-limited. |
| L | True `M=1` crossing / sonic branch | Audit whether current implicit residuals can cross Mach 1 without relying on clipped/floored quantities. If not, build a two-domain or event-located formulation with `x_sonic`, left/right constraints, and compatibility conditions. | `reduced_implicit.py`, `implicit.py`, `physics.py`, `v6_casadi_v2/optimize_area_profile_casadi_v2.py` | Start with `--critical-mode`; only then add a branch-specific runner. | Gate: crossing must be shown by profile arrays, not only by `critical_max_abs_residual`. Need a before/after Mach plot and equality residual report. |
| L | Generic swirl ratio / longer streamline in smaller device | Promote swirl from case-specific geometry interpretation to a model parameter. Map radial device length to streamline length with `ds/dr = sqrt(1 + S^2)`, and keep area-volume consistency explicit. | `geometry.py`, `models.py`, `cases/yamasaki2004/parameters.py`, seed builder | Future: `--swirl-ratio S` or seed field `swirl_ratio` | Gate: compare same physical disk area under radial vs swirl-equivalent length. Report whether gain comes from real residence length or area/current rescaling. |
| XL | Piecewise current `Jx = I0 * step(x - x0)` | Create a new current-profile formulation. Use a smooth tanh/logistic step first for factorability, then consider sharper steps. This changes closure, profile exports, handoff, diagnostics, and possibly current conservation assumptions. | `physics.py`, `models.py`, `reduced_implicit.py`, `implicit.py`, `workflow.py`, `v6_casadi_v2`, tests | Future: `--current-profile tanh-step --current-step-x0 ... --current-step-width ...` | Gate: conserve total current convention explicitly, expose `I(x)` and `J_x(x)`, and compare against constant-`I_0` baseline before claiming a new branch. |

## Recommended near-term sequence

1. Run the no-new-code checks first: enthalpy vs lab objective, inlet-window
   widening, and KKT analysis on the same best-shot seed.
2. Add `--B-override` and a small B-sweep runner. This is the smallest code
   change that directly tests the current "try higher B field" idea.
3. Add `JyB` and singularity diagnostics before changing objectives. If the
   diagnostic says `JyB` or `1 - C partial_Te F` is genuinely organizing the
   solution, then add a reward profile.
4. Only after the above, spend time on the large topology changes:
   true sonic crossing, generic swirl, and piecewise current.

## Output layout convention

For each experiment family, use:

```text
v6_maingo_casadi/outputs/<family>/<run_name>/
  maingo_summary.json
  maingo_coarse_profile.npz
  maingo_handoff_profile.npz
  hybrid_summary.json
  hybrid_analysis.json
  maingo_profile_audit.json
  continuation/
```

For experiment definitions, use:

```text
v6_maingo_casadi/experiments/
  windows/
  sweeps/
  reports/
```

Do not compare results without recording the baseline summary path, search
window JSON, objective profile, B value, working-fluid profile, area scale,
MAiNGO settings file, and whether CasADi handoff was skipped.
