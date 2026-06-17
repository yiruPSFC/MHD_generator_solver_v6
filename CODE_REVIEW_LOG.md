## 2026-06-16 - Local Affine Singular Matrix Marker

Scope:
`v6_active_boundary_reduced/local_affine.py`, `compute_forward_affine_coefficients`.

Question:
During review, the singular primitive matrix branch was identified as a risk point that should not live only in chat.

Before:
When `np.linalg.solve(D, ...)` raised `LinAlgError`, the code returned `NaN` values for the affine decompositions `y0` and `y1` without an explicit marker.

Change:
Added a short `# RISK` comment at the singular-matrix branch.

Current Behavior:
The function still returns `ForwardAffineCoefficients` with `NaN` affine values when the local primitive matrix is singular. Downstream policy code is expected to reject these coefficients through interval construction, finite-step validation, or diagnostics before trusting an endpoint.

Rationale:
Near sonic or degenerate primitive states, singular local linearization is a numerical condition rather than a trustworthy affine model. Keeping the marker at the origin of the `NaN` makes the caller responsibility visible during future refactors.

Open Risks:
Need to verify in `policy.py` that all non-finite affine coefficients are rejected before endpoint selection and finite-step acceptance.

Verification:
Not run; comment-only marker and review-log update.

## 2026-06-16 - Affine-Predicted G Boundary Fallback

Scope:
`v6_active_boundary_reduced/policy.py`, `_sign_aware_g_boundary_fallback`; preparation entry scripts.

Question:
The review question was whether the already-computed affine `G` boundary can seed the expensive finite-step fallback instead of immediately trying only the interval endpoints.

Before:
When the sign-aware reverse endpoint failed on `G`, `_sign_aware_g_boundary_fallback` evaluated the two interval ends and used `brentq` only if a real finite-step `G` sign change was found.

Change:
Added `g_boundary_fallback_mode="affine_expand"` behind a default-legacy switch. In the new mode, the fallback first checks whether `interval.sigma_G_bound` is finite, in-range, and actually active as `G_lower` or `G_upper`; if so, it probes from that predicted boundary into the affine-feasible side with exponentially growing offsets and still accepts only a real finite-step `G` bracket.

Current Behavior:
Default behavior remains `legacy`. With `affine_expand`, an unusable predicted boundary, inactive `G` source, singular/non-finite value, or missing real sign bracket falls through to the previous endpoint-bracket fallback. The accepted segment diagnostics include the affine-expand probe count and predicted `sigma_G_bound` so later A/B runs can see whether the initializer is helping or adding probes.

Rationale:
The affine `G` gradient is already computed on the main reverse path to decide which side of the interval is G-limited. Reusing that boundary as a probe initializer can reduce unnecessary finite-step evaluations when the affine predictor is locally accurate, while preserving the actual finite-step validation as the acceptance criterion.

Open Risks:
Speedup depends on how often endpoint failure is caused by a nearby finite-step `G` boundary. If the affine predictor is poor or the active failure is dominated by another constraint, the mode should behave like legacy plus a few extra probes. A too-small initial expansion step was observed to make the mode slower by requiring about six probes per step; the current initial offset is scaled to `1e-4` of the active interval width to avoid that local pathology.

Verification:
`python -m compileall -q v6_active_boundary_reduced/policy.py v6_active_boundary_reduced/run_preparation_recovery.py v6_active_boundary_reduced/run_anchor_scan.py v6_active_boundary_reduced/run_anchor_optimize.py v6_active_boundary_reduced/outer_solver/lbfgsb.py v6_active_boundary_reduced/run_short_channel_reachability.py v6_active_boundary_reduced/validation/test_policy_behavior_guards.py`; `git diff --check`; `PYTHONPATH=. python v6_active_boundary_reduced/validation/test_policy_behavior_guards.py`; Freidberg sign-aware smoke tests. A five-repeat Freidberg 60-step local A/B gave legacy mean wall/process about 0.714/0.675 s and affine-expand mean wall/process about 0.537/0.530 s with mean probe count 1.

## 2026-06-16 - Preparation Recovery Diagnostics Module Name

Scope:
`v6_active_boundary_reduced/preparation_recovery_diagnostics.py`; callers in `run_preparation_recovery.py`, `run_anchor_optimize.py`, and `run_short_channel_reachability.py`.

Question:
The old module name suggested the file was only a plotting script, but review showed it recomputes closure quantities, writes diagnostic tables, computes Freidberg-form H/L residual diagnostics, and then writes plots.

Before:
The module was named `plot_preparation_recovery.py`.

Change:
Renamed it to `preparation_recovery_diagnostics.py` and updated all imports.

Current Behavior:
The public function remains `write_preparation_diagnostics`, and the generated artifacts are unchanged. The module CLI now describes the task as writing diagnostic tables and plots.

Rationale:
The new filename makes the file's role clearer: it is a post-processing diagnostics writer for preparation recovery summaries, not just a plotting helper.

Open Risks:
External commands that directly import or run `v6_active_boundary_reduced.plot_preparation_recovery` must be updated to the new module name.

Verification:
`python -m compileall -q v6_active_boundary_reduced/preparation_recovery_diagnostics.py v6_active_boundary_reduced/run_preparation_recovery.py v6_active_boundary_reduced/run_anchor_optimize.py v6_active_boundary_reduced/run_short_channel_reachability.py`; import smoke for `write_preparation_diagnostics` and the three callers; `python -m v6_active_boundary_reduced.preparation_recovery_diagnostics --help`; `git diff --check`.

## 2026-06-16 - Analytical G Gradient A/B

Scope:
`v6_active_boundary_reduced/numba_physics.py`, `g_state_and_gradients_numba`; `v6_active_boundary_reduced/local_affine.py`, `compute_forward_affine_coefficients`; `v6_active_boundary_reduced/benchmark_local_affine_algebraic.py`.

Question:
The finite-difference `G` gradient calls the local closure evaluator six times per state. The review question was whether an analytical gradient can reduce local-affine cost, and whether the benchmark should report wall time, process CPU time, and gradient-only timing.

Before:
`_closure_G_gradients` computed `dG/dn`, `dG/dTe`, and `dG/dA` by centered log-space finite differences.

Change:
Added an analytical `G` gradient path behind `g_gradient_mode="analytic"` and extended the benchmark to compare rollout-level, operator-only, and gradient-only timing with both wall and process CPU clocks.

Current Behavior:
Default production local-affine behavior remains finite-difference. The benchmark can run the analytical path through `compute_forward_affine_coefficients_analytic_g`.

Rationale:
The analytical gradient isolates the cost of differentiating `G` from the rest of the rollout. The A/B result showed the gradient kernel itself was much faster, but full rollout timing changed only a few percent because local-affine gradient work is a small fraction of the whole reverse rollout.

Open Risks:
The analytical formula is piecewise around clips/floors for `T_p`, `Delta`, `f_I`, and Saha limits. It should remain behind the benchmark path until broader cases, especially near clipping and sonic boundaries, are checked.

Verification:
`git diff --check`; `python -m compileall -q v6_active_boundary_reduced/numba_physics.py v6_active_boundary_reduced/local_affine.py v6_active_boundary_reduced/benchmark_local_affine_algebraic.py`; A/B benchmark summaries under `outputs/local_affine_analytic_g_ab_20260616/`.

## 2026-06-16 - Remove One-Off Local Affine Benchmark

Scope:
`v6_active_boundary_reduced/benchmark_local_affine_algebraic.py`.

Question:
After using the local-affine analytical-`G` benchmark for A/B timing, the review asked whether the script was still part of the maintained active-boundary-reduced flow.

Before:
The standalone benchmark script remained in the package even though no production module or CLI imported it.

Change:
Deleted `benchmark_local_affine_algebraic.py`.

Current Behavior:
The analytical `G` gradient implementation remains in `numba_physics.py` and the selectable local-affine path remains in `local_affine.py`, but the one-off benchmark entrypoint is gone.

Rationale:
The script served its purpose as an A/B measurement tool and was not part of the main solver, diagnostics, or validation chain. Keeping it would make the module inventory noisier while not improving the maintained workflow.

Open Risks:
Historical documentation still mentions the benchmark in old verification notes; those references are archival, not active entrypoints.

Verification:
`rg` found no active source references to the deleted benchmark module; `python -m compileall -q v6_active_boundary_reduced/numba_physics.py v6_active_boundary_reduced/local_affine.py`; import smoke for `closure_state_numba`, `dynamic_terms_numba`, `g_state_and_gradients_numba`, and `compute_forward_affine_coefficients`; `git diff --check`.

## 2026-06-16 - Sonic Route Before Local Affine

Scope:
`v6_active_boundary_reduced/policy.py`, `_reverse_sign_policy_step`; `v6_active_boundary_reduced/local_affine.py`, `compute_forward_affine_coefficients`.

Question:
During review, the singular `D` case was clarified as a routing problem rather than a local-affine problem.

Before:
The risk marker said downstream callers must reject `NaN` affine coefficients when `D` is singular.

Change:
Updated the `local_affine.py` marker and added a `policy.py` review marker before the sonic gate.

Current Behavior:
The main reverse policy computes primitive sonic compatibility before calling `compute_forward_affine_coefficients`. If `sonic_mode=auto` sees `M` near 1 or small `det_D`, it routes to the sonic-compatible step, where `A_prime` is selected from the left-null condition `ell^T (f0 + A_prime f1) = 0`.

Rationale:
At singular or near-singular primitive matrix states, inverting `D` to build local affine coefficients is the wrong mathematical object. The admissible local question is whether the area-slope control can cancel the left-null forcing component, so the branch should solve for `A_prime` directly from the compatibility condition.

Open Risks:
Non-main callers of `compute_forward_affine_coefficients`, such as forward greedy diagnostics or tests, may still call local affine near singular states unless they add their own sonic gate.

Verification:
Not run; comment-only marker and review-log update.

## 2026-06-16 - Local Active-Boundary Physics Kernels

Scope:
`v6_active_boundary_reduced/physics_constants.py`; `v6_active_boundary_reduced/numba_physics.py`; callers in `policy.py`, `preparation_recovery_diagnostics.py`, and `run_yamasaki_power_benchmark.py`.

Question:
The active-boundary-reduced package still used legacy wrapper formulas for inlet `dot_N`, inlet `T_p`/Mach/G, closure quantities, and Freidberg-form H/L diagnostic balances. The review question was whether these physics kernels should live in the local package so the reduced solver is easier to read and maintain.

Before:
`policy._physics_params` and the Yamasaki benchmark called `v6_firedrake_reduced.legacy_physics.inlet_design_generic`; the preparation recovery diagnostics called legacy inlet/closure helpers and `v6_firedrake_reduced.forward._freidberg_balance_terms`; `numba_physics.py` imported constants from `v6_maingo_casadi.constants`.

Change:
Added local `physics_constants.py`, added `inlet_design_numba` and `freidberg_balance_terms_numba`, and updated active-boundary callers to use the local numba physics kernels.

Current Behavior:
The active-boundary-reduced main path computes `dot_N` from `inlet_design_numba`. Diagnostics compute closure fields, `T_p`, Mach, G, and Freidberg H/L residual terms through `numba_physics.py`. The package no longer imports `v6_maingo_casadi.constants` or the legacy inlet/closure helpers in active source files.

Rationale:
`dot_N`, `T_p`, Mach, G, and H/L balances are all core local physics quantities for this reduced solver. Keeping their numeric implementations in one local module makes the codebase easier to audit, keeps the analytical G-gradient work next to the closure it differentiates, and reduces the chance that future readers must jump through the legacy wrapper layer to understand active-boundary behavior.

Open Risks:
The Freidberg H/L kernel is local now, but it is still a diagnostic balance rather than the primitive finite-step solve used by the policy. The tuple-return API in `numba_physics.py` is fast and numba-friendly but less self-documenting than dictionaries; call sites must keep index mappings correct.

Verification:
Numeric parity against the old numeric legacy path for Freidberg and Yamasaki inlet, closure, dynamic terms, and H/L balances passed with maximum relative error 0.000e+00; `python -m compileall -q v6_active_boundary_reduced/physics_constants.py v6_active_boundary_reduced/numba_physics.py v6_active_boundary_reduced/policy.py v6_active_boundary_reduced/preparation_recovery_diagnostics.py v6_active_boundary_reduced/run_yamasaki_power_benchmark.py`; `PYTHONPATH=. python v6_active_boundary_reduced/validation/test_policy_behavior_guards.py`; Freidberg sign-aware smoke tests; `python -m v6_active_boundary_reduced.preparation_recovery_diagnostics --help`; diagnostics smoke on `outputs/active_boundary_sign_aware_freidberg_smoke_20260531/preparation_recovery_summary.json`; `git diff --check`.

## 2026-06-16 - Preparation Recovery Wall-Time Breakdown

Scope:
`v6_active_boundary_reduced/policy.py`, reverse sign-aware preparation recovery; `v6_active_boundary_reduced/finite_step.py`, RK4 finite-step RHS; `v6_active_boundary_reduced/local_affine.py`, affine coefficients.

Question:
Which parts of the main preparation-recovery flow consume enough wall time that numba replacement or lower-overhead numerical kernels could matter?

Before:
The recent local physics-kernel work made closure and dynamic terms local numba functions, but the main flow still spent time in Python/NumPy layers such as RK4 wrappers, `np.linalg.solve`, `brentq`, scan/fallback control flow, and diagnostics dictionaries.

Change:
No source-code behavior change. Ran a temporary wall-time profiling harness on Freidberg reference, anchor index 60 (`x=0.6`), `n_steps=60`, `dx=0.01`, `scan_points=41`, `refine_iterations=24`.

Current Behavior:
With `g_boundary_fallback_mode="legacy"`, the 60-step rollout averaged about 0.51 s and every step used `sign_aware_brentq_G_boundary_fallback`. With `affine_expand`, the same rollout averaged about 0.41 s and every step used `sign_aware_affine_expand_G_boundary_fallback` with mean probe count 1. In the affine-expand timed run, 60 steps produced about 300 `_evaluate_sigma` calls and about 3600 RK4 RHS stages. Inclusive time was dominated by `_evaluate_sigma` and RK4; leaf timing showed the numba dynamic kernel itself was only about 0.002 s per rollout, while small NumPy diagnostics in `primitive_log_rhs_with_diagnostics` were roughly 0.28 s per rollout.

Rationale:
The main remaining opportunity is not moving more closure physics into numba. The high-probability optimization is to split `primitive_log_rhs_with_diagnostics` into a fast RHS path used when `rk4_stage_diagnostics` and `rk4_stage_gate` are both off, and a diagnostic path used only when those diagnostics are requested. A temporary monkeypatch that skipped stage diagnostics reduced the affine-expand rollout to about 0.14 s; a temporary fused numba raw-RHS kernel reduced it to about 0.08 s while matching baseline sigmas and node `n_p`/`T_e`/`A` with max absolute difference 0.0 on this case.

Open Risks:
The temporary fused raw-RHS benchmark only covered the default raw RHS mode and a non-sonic Freidberg segment. Production changes must preserve the diagnostic mode, `log`/`nondim` RHS modes, stage gates, near-singular handling, and finite-step validation semantics.

Verification:
Temporary profiling harness after JIT warmup; uninstrumented legacy and affine-expand timing runs; temporary fast-RHS and fused-numba RHS timing runs; baseline-versus-fused output parity for sigmas and node `n_p`/`T_e`/`A` on the profiled case.

## 2026-06-16 - Default RK4 Stage Fast Path

Scope:
`v6_active_boundary_reduced/finite_step.py`, `primitive_log_rhs_fast`, `primitive_log_rhs_with_diagnostics`, and `rk4_integrate_state`.

Question:
Stage diagnostics are mainly a post-failure inspection toolbox. The main policy decisions use endpoint constraints, RK4 step-doubling error, finite-state checks, and optional stage gate margins. Should the default path pay the full per-stage diagnostic cost when both `rk4_stage_diagnostics` and `rk4_stage_gate` are disabled?

Before:
Every RK4 RHS stage called `primitive_log_rhs_with_diagnostics`, which computed condition numbers, determinants, row-normalized matrices, singular values, replay residuals, and a diagnostics dictionary even when `collect_diagnostics=False` meant the result would be discarded.

Change:
Added `primitive_log_rhs_fast` and changed `rk4_integrate_state` so default non-diagnostic stages compute only the RHS needed for RK4. The full diagnostics path is still used whenever `collect_diagnostics=True`, which happens when `rk4_stage_diagnostics=True` or `rk4_stage_gate=True`.

Current Behavior:
Default preparation recovery leaves stage summaries empty (`rk4_stage_count=0`) and avoids per-stage diagnostic work. Enabling stage diagnostics or stage gate restores the full stage summary (`rk4_stage_count>0`) and all stage gate margins. The public `primitive_log_rhs` now uses the same fast RHS because it returns only RHS values.

Rationale:
This matches the intended semantics: stage diagnostics are for inspecting or gating internal RK4 stages, not for default endpoint policy decisions. Avoiding discarded diagnostics removes the largest measured Python/NumPy overhead while keeping the explicit diagnostic/gate mode unchanged.

Open Risks:
The fast path implements the 2x2 solve directly and falls back to `solve_linear_rhs` only for singular/non-finite determinants. It matched the full diagnostic path on tested Freidberg states and `raw`/`log`/`nondim` modes, but near-singular pathological cases should still be exercised when sonic-boundary tests expand.

Verification:
`python -m compileall -q v6_active_boundary_reduced/finite_step.py v6_active_boundary_reduced/policy.py`; `PYTHONPATH=. python v6_active_boundary_reduced/validation/test_policy_behavior_guards.py`; Freidberg sign-aware smoke tests; RHS fast/full parity for `raw`, `log`, and `nondim` modes with relative differences below `1e-12`; 60-step Freidberg `affine_expand` rollout parity between default fast path and `rk4_stage_diagnostics=True` with zero difference in `sigma`, node `n_p`, `T_e`, and `A`; `rk4_stage_gate=True` smoke confirmed stage counts are collected. Timing on the same 60-step Freidberg case after warmup: default fast path median about 0.105 s, full stage diagnostics median about 0.409 s.

## 2026-06-16 - Scalar RK4 Fast Path And Params Downlink

Scope:
`v6_active_boundary_reduced/finite_step.py`, default non-diagnostic `rk4_integrate_state` and fast RHS helpers.

Question:
After removing default stage diagnostics, the remaining hot path still spent time rebuilding physics-parameter cache keys, constructing `State` objects for every RK4 stage, using scalar NumPy clipping in state properties, and allocating small RHS arrays. The review question was whether low-risk scalar cleanup should be done before considering a fused numba raw-RHS implementation.

Before:
Default RK4 stages called `primitive_log_rhs_fast` through temporary `State` objects. That wrapper called `physics_params_fn(config)` at stage frequency and returned a new 3-element NumPy array for each RHS stage.

Change:
`solve_next_state_rk4` now resolves physics parameters once per candidate finite step and passes them into both coarse and fine RK4 integrations. The non-diagnostic RK4 branch now advances scalar `log_n`, `log_Te`, and `logA` values directly through `_primitive_log_rhs_fast_values`, avoiding per-stage `State` construction and 3-element RHS arrays. The public `primitive_log_rhs_fast` remains as a wrapper for external callers. Added a code marker deferring fused-numba RHS because the 2x2 solve is no longer the bottleneck, the expected gain is limited, and a separate `raw`/`log`/`nondim` numba path would increase maintenance drift risk.

Current Behavior:
Default non-diagnostic rollouts use scalar RK4/RHS and keep stage summaries empty. Enabling stage diagnostics or stage gate still uses the full diagnostics path and collects stage summaries. `raw`, `log`, and `nondim` fast RHS modes remain supported.

Rationale:
This keeps the numerical implementation readable while removing Python overhead in the actual hot loop. It also preserves one source of truth for `raw`/`log`/`nondim` semantics instead of introducing a fused numba path that would need to mirror all three modes.

Open Risks:
The scalar fast path must remain in parity with the full diagnostics path as RHS modes evolve. The direct 2x2 solve still falls back to `solve_linear_rhs` only for singular or non-finite determinants, so near-singular cases should stay covered by sonic-routing and future edge-case tests.

Verification:
`python -m compileall -q v6_active_boundary_reduced/finite_step.py v6_active_boundary_reduced/policy.py`; RHS fast/full parity for `raw`, `log`, and `nondim` modes with relative differences below `1e-12`; `PYTHONPATH=. python v6_active_boundary_reduced/validation/test_policy_behavior_guards.py`; Freidberg sign-aware smoke tests; 60-step Freidberg `affine_expand` rollout parity between default scalar fast path and `rk4_stage_diagnostics=True` with zero difference in `sigma`, node `n_p`, `T_e`, and `A`; `rk4_stage_gate=True` smoke confirmed diagnostics are still collected. Timing on the same 60-step Freidberg case after warmup: median about 0.053 s uninstrumented; instrumented breakdown showed `_primitive_log_rhs_fast_values` about 0.012 s/rollout, `_solve_2x2_fast` about 0.003 s/rollout, and `dynamic_terms_numba` about 0.002 s/rollout.

## 2026-06-16 - Active Objective Profile Metrics Localized

Scope:
`v6_active_boundary_reduced/objective.py`, `_score_rollout`, `_profile_metric_terms`, and local `evaluate_profile_metrics`; `v6_active_boundary_reduced/run_yamasaki_power_benchmark.py`.

Question:
`objective.py` still imported `evaluate_profile_metrics` from `v6_firedrake_reduced.objective`, and profile-metric failures were silently converted into missing reward terms.

Before:
Active-boundary scoring depended on the legacy objective module for MHD power, enthalpy extraction, Hall voltage, and profile extrema. If profile arrays were missing or metric evaluation failed, `_profile_metric_terms` returned `{}`, so MHD and enthalpy rewards became zero without an explicit signal.

Change:
Moved the active-boundary profile metric calculation into `v6_active_boundary_reduced/objective.py` using local `inlet_design_numba`, `closure_state_numba`, and `_physics_params`. Updated the Yamasaki benchmark to import the local metrics function. `_profile_metric_terms` now emits `profile_metrics_ok` and `profile_metrics_failure`; `_score_rollout` keeps metrics optional for delta-only scans but gates the result as failed when MHD power or enthalpy reward weights require unavailable metrics.

Current Behavior:
Default delta-improvement objectives remain tolerant of missing profile arrays, but the missing metrics are visible in `objective_terms` and CSV output. When `mhd_output_power_MW` or `enthalpy_extraction_percent` weights are nonzero, unavailable profile metrics produce `ok=False`, a `profile_metrics_unavailable` failure, and a `profile_metrics` blocker.

Rationale:
The objective module should own the reward quantities it uses, and the active-boundary scorer should not depend on the older Firedrake objective implementation. Metrics are diagnostics for the default objective, but they become required data once they participate in the scalar score.

Open Risks:
The local metrics intentionally cover the active-boundary fields currently used by this scorer and benchmark, not the full legacy Velikhov and thermal-window penalty surface. If those older penalties are reintroduced as active rewards, they should be added explicitly here rather than reimporting the legacy objective.

Verification:
`PYTHONPATH=. ./.venv_jit/bin/python -m compileall -q v6_active_boundary_reduced/objective.py v6_active_boundary_reduced/run_yamasaki_power_benchmark.py`; legacy-versus-local metric parity smoke on a Freidberg constant profile for MHD power, enthalpy extraction, inlet enthalpy flux, Hall voltage, electric power, min `T_p`, max `T_e/T_p`, and min Mach; score-gate smoke for missing profile arrays with zero and nonzero metric weights; `PYTHONPATH=. ./.venv_jit/bin/python v6_active_boundary_reduced/validation/test_policy_behavior_guards.py`; `PYTHONPATH=. ./.venv_jit/bin/python -m v6_active_boundary_reduced.run_yamasaki_power_benchmark --help`; `git diff --check -- v6_active_boundary_reduced/objective.py v6_active_boundary_reduced/run_yamasaki_power_benchmark.py`.

## 2026-06-16 - G-Boundary Fallback Mode Names And Validation Notes

Scope:
`v6_active_boundary_reduced/policy.py`, preparation entry scripts, and the two validation files under `v6_active_boundary_reduced/validation`.

Question:
The names `legacy` and `affine_expand` hid what the reverse G-boundary fallback modes actually do, and the Freidberg sign-aware smoke test looked more like a historical diagnostic than a default representative test case.

Before:
`g_boundary_fallback_mode` defaulted to `legacy`, while `affine_expand` meant "try the affine-predicted G boundary first, then fall back to endpoint brentq." The validation files did not explain that one was synthetic branch guarding and the Freidberg one was written to understand the x~=0 forward/reverse policy asymmetry.

Change:
Renamed the canonical modes to `endpoint_brentq` and `affine_expand_then_endpoint_brentq`. Kept `legacy` and `affine_expand` as accepted aliases in the policy normalizer for old commands. Updated CLI defaults/help and validation test names to use the clearer names. Added module-level notes to the validation files: the policy guard file is synthetic control-flow coverage, and the Freidberg x~=0 smoke is a historical diagnostic for the "forward pushes objective, reverse stays on G boundary" asymmetry rather than a broad default regression case.

Current Behavior:
New scripts default to `endpoint_brentq`. Passing old aliases still maps to the same behavior. The Freidberg smoke remains executable, but its header now makes clear why it exists and why it should not be overinterpreted as a representative active-boundary test.

Rationale:
The endpoint-brentq fallback is not obsolete; it is the concrete G-boundary recovery used before scan fallback. The new names describe the actual algorithm and make it easier to reason about the difference between direct endpoint bracketing and affine-predicted initialization.

Open Risks:
Existing saved configs may still show the old alias strings if users pass them explicitly. That is acceptable while the alias layer remains.

Verification:
See command results from the current review turn after the rename.

## 2026-06-16 - Preparation Recovery CLI Diagnostics Switch

Scope:
`v6_active_boundary_reduced/run_preparation_recovery.py`.

Question:
During review, `--objective power_next` looked misleading for a reverse-only preparation recovery entry point, and the script generated full diagnostic plots/tables unconditionally.

Before:
The CLI accepted `--objective power_next` even though reverse preparation is only supported for `delta_drop`, and every run called `write_preparation_diagnostics`, which recomputes closure diagnostics and writes plots even for batch or timing use.

Change:
Added a `REVIEW` marker above the objective option explaining that `power_next` is a visible legacy CLI option while reverse preparation currently supports `delta_drop`. Added `--write-diagnostics`; diagnostic tables and plots are now generated only when that flag is passed.

Current Behavior:
The script always writes the core summary artifacts: `preparation_recovery_summary.json`, `nodes.csv`, `segments.csv`, and `profile.npz`. The printed short summary sets `diagnostics` to `null` unless `--write-diagnostics` is requested, in which case it contains the diagnostic manifest.

Rationale:
The preparation recovery script is useful both as an interactive diagnostic driver and as a batch/baseline runner. Making diagnostics explicit avoids unnecessary plotting and closure post-processing cost on batch runs while preserving the manual diagnostic workflow behind a clear flag.

Open Risks:
Existing scripts that expected diagnostic plots from every `run_preparation_recovery.py` invocation must add `--write-diagnostics`.

Verification:
See command results from the current review turn.

## 2026-06-17 - Runner Common Anchor IO Review Notes

Scope:
`v6_active_boundary_reduced/runners/common.py`, with call-site checks in `select_profile_anchor.py`, `run_short_channel_reachability.py`, `run_forward_phi_greedy.py`, `run_ipopt_endpoint_reachability.py`, and local duplicate profile-array code in `outer_solvers/lbfgsb.py`.

Question:
During the file-by-file active-boundary review, two notes came up from reading `common.py` and searching its call sites: whether JSON serialization is strict-JSON safe for non-finite array elements, and whether common profile/node helpers are actually used consistently across runner and outer-solver outputs.

Before:
These notes existed only in the chat review summary. `common.json_default` converted numpy arrays with `tolist()` immediately while converting numpy scalar floats and Python floats through a finite check. `common.node_payload_to_state` and `common.profile_arrays_from_nodes` had no package call sites, while `outer_solvers/lbfgsb.py` kept a separate `_profile_arrays_from_nodes` implementation.

Change:
No code change. Logged the review notes so they are preserved for the later `lbfgsb.py` and output-format review.

Current Behavior:
`write_json` uses `json.dumps(..., default=json_default)` with Python's default `allow_nan=True`. Scalar non-finite floats handled by `json_default` become `null`, but non-finite values inside arrays can pass through the array `tolist()` path and may be emitted as `NaN` or `Infinity`, which Python can read back but strict JSON parsers may reject. The shared common helpers are used by reachability, forward-greedy, IPOPT reachability, and Yamasaki benchmark runners for anchor/profile IO, but not all scripts have converged on them yet.

Rationale:
This is not an immediate solver correctness bug because current internal readers are Python-based and the unused helpers do not affect active execution paths. It is still worth tracking because these files define artifact formats used across scripts, and duplicated profile-array writers can drift as output fields expand.

Open Risks:
Strict external JSON consumers may reject runner summaries containing non-finite values nested inside arrays. Profile NPZ schemas may diverge between `common.profile_arrays_from_nodes`, rollout payloads, and `outer_solvers/lbfgsb.py` unless the later outer-solver review either accepts the duplication intentionally or consolidates it.

Verification:
Read `common.py` with line numbers; searched package call sites for `json_default`, `write_json`, `write_csv`, `load_profile`, `anchor_payload`, `anchor_from_node_payload`, `load_anchor_json`, `load_profile_anchor`, `node_payload_to_state`, `profile_arrays_from_nodes`, and `save_profile_npz`; inspected the duplicate `_profile_arrays_from_nodes` in `outer_solvers/lbfgsb.py`. Tests not run because this was a review-log-only marker.

## 2026-06-17 - Select Profile Anchor Rename

Scope:
`v6_active_boundary_reduced/runners/select_profile_anchor.py`, renamed from `v6_active_boundary_reduced/runners/extract_reachability_anchor.py`.

Question:
During review, the script name looked misleading because the file is a small CLI adapter/helper rather than a reachability solver or reachability-analysis implementation.

Before:
The old script name emphasized the downstream consumer (`reachability`) and the artifact (`anchor`), but the script actually selects one node from a built-in/reference profile, profile NPZ, or preparation summary and exports the standard anchor JSON payload.

Change:
Renamed `extract_reachability_anchor.py` to `select_profile_anchor.py` and updated repository references in the dependency map and review log.

Current Behavior:
The script remains a helper CLI under the clearer module name `v6_active_boundary_reduced.runners.select_profile_anchor`. It accepts either `--summary-json` or `--profile-npz`/built-in profile input, selects one node by index, computes the common `anchor_payload`, optionally writes `--out-json`, and prints the payload.

Rationale:
The old name was not technically wrong because the generated artifact is used by reachability scripts as a fixed endpoint. It was still misleading when read in isolation because it sounded like it extracted anchors from a reachability result or participated in reachability computation. `select_profile_anchor.py` names the actual action more directly.

Open Risks:
No compatibility alias was added. Existing commands that invoke `v6_active_boundary_reduced.runners.extract_reachability_anchor` must switch to `v6_active_boundary_reduced.runners.select_profile_anchor`.

Verification:
Read `select_profile_anchor.py` and searched for downstream `source_anchor_json`, `target_anchor_json`, and `load_anchor_json` usage in reachability and forward-greedy runners. Ran a repository search for the old module name after the rename; remaining occurrences are historical review-log notes. Verified `PYTHONPATH=. ./.venv_jit/bin/python -m v6_active_boundary_reduced.runners.select_profile_anchor --help` and `PYTHONPATH=. ./.venv_jit/bin/python -m compileall -q v6_active_boundary_reduced/runners/select_profile_anchor.py`.

## 2026-06-17 - Objective And Outer Reward Layering

Scope:
`v6_active_boundary_reduced/core/objective.py`, `v6_active_boundary_reduced/outer_solvers/reward.py`, and new shared helper module `v6_active_boundary_reduced/core/scoring.py`.

Question:
During review, the distinction between `core/objective.py` and `outer_solvers/reward.py` was unclear because both expose weights, scores, reward terms, and failure penalties.

Before:
`core/objective.py` was easy to misread as the greedy policy objective. In current code, the per-step greedy objective is actually in `core/policy.py`; `core/objective.py` evaluates a full preparation design rollout and produces `objective_terms`, while `outer_solvers/reward.py` re-scores that result for prescreening and L-BFGS-B.

Change:
Kept the two score layers separate, but extracted repeated scalar helper logic into `core/scoring.py`. `core/objective.py` now imports `soft_square` for inlet temperature shortfall penalties. `outer_solvers/reward.py` imports `finite_float`, `profile_stat`, `area_ratio_from_nodes`, and `soft_square` instead of carrying local copies.

Current Behavior:
`core/objective.evaluate_preparation_design` applies design overrides, builds an anchor from the design, runs `recover_preparation_profile`, and scores the whole rollout with preparation-level terms such as delta improvement, optional MHD power/enthalpy rewards, inlet Delta penalty, inlet `T_e`/`T_p` floors, target-anchor feasibility, and profile-metric availability. `outer_solvers.reward.score_outer_result` consumes that result, prefers `objective_terms` where available, and adds outer-solver-specific scoring for profile min/max temperature, area ratio, magnetic-field bounds, `G` floor, Mach ceiling, incomplete rollout, and failure handling. Shared numeric helpers live in `core/scoring.py` and do not own either layer's semantics.

Rationale:
The overlap in `delta_improvement`, optional MHD power, temperature scaling, and failure penalty is intentional because both layers need comparable scalar diagnostics. The outer reward is not the reduced-model greedy policy objective; it is the optimizer-facing scalarization of an already evaluated rollout.

Open Risks:
Because the layers still intentionally duplicate some scalar names and weight concepts, defaults can drift or users can confuse `PreparationObjectiveWeights` with `OuterRewardWeights`. Later review of `lbfgsb.py` should verify which score is actually optimized and whether CSV/output names make the distinction visible enough.

Verification:
Inspected function lists and call sites for `evaluate_preparation_design`, `_score_rollout`, `evaluate_profile_metrics`, `score_outer_result`, `PreparationObjectiveWeights`, and `OuterRewardWeights`; inspected `lbfgsb.py` where `evaluate_preparation_design` is followed by `score_outer_result`. Ran `PYTHONPATH=. ./.venv_jit/bin/python -m compileall -q v6_active_boundary_reduced/core/scoring.py v6_active_boundary_reduced/core/objective.py v6_active_boundary_reduced/outer_solvers/reward.py v6_active_boundary_reduced/outer_solvers/prescreen.py v6_active_boundary_reduced/validation/test_outer_solver.py`; `git diff --check -- v6_active_boundary_reduced/core/scoring.py v6_active_boundary_reduced/core/objective.py v6_active_boundary_reduced/outer_solvers/reward.py ACTIVE_BOUNDARY_DEPENDENCY_CODE_MAP.md CODE_REVIEW_LOG.md`; direct invocation of all four `test_outer_solver.py` test functions in `.venv_jit`. `pytest` was not available in `.venv_jit`.

## 2026-06-17 - Prescreen Direction-Specific Heuristic Marker

Scope:
`v6_active_boundary_reduced/outer_solvers/prescreen.py`.

Question:
During review, the current prescreen logic was compared with the intended future idea of a cheaper local derivative prescreen. The current code samples complete candidate designs, solves full preparation profiles, and filters the resulting nodes; it does not yet implement a one-dx local `partial_x(T_e/T_p)` screening model.

Before:
`passes_prescreen` looked like a generic physical admissibility gate, but its current hard gates are tuned to the reverse-preparation workflow where the design anchor is an inlet-like condition and low anchor `T_e/T_p` plus positive local `d(T_e/T_p)/dx` are desirable. That interpretation may not transfer unchanged to a future formalized forward workflow.

Change:
Added a short `REVIEW` marker above `passes_prescreen` noting that these gates are reverse-preparation seed heuristics and future forward-self-consistent prescreening needs direction-specific metrics.

Current Behavior:
`prescreen_candidates` evaluates the center point and random uniform samples over the selected control bounds. Each candidate is converted into design overrides, passed through `evaluate_preparation_design` with `return_payload=True`, re-scored by `score_outer_result`, and then filtered by `passes_prescreen`. Accepted candidates are ranked by `te_over_tp_gradient - ratio_penalty - g_penalty + 0.001 * outer_score`, with optional fallback to rejected rows if allowed.

Rationale:
The current implementation is a multistart seed finder for the robust L-BFGS-B outer prototype, not a cheap local-derivative prescreen. Preserving this distinction matters because a forward approach may want opposite or different `T_e/T_p` trend semantics depending on the anchored endpoint and integration direction.

Open Risks:
If the same `prescreen.py` gates are reused for a forward formalization without revisiting direction semantics, good forward candidates may be filtered out or ranked incorrectly. A future implementation may need a separate local derivative prescreen that evaluates only a single dx or analytic local quantities before full-profile solves.

Verification:
Ran `PYTHONPATH=. ./.venv_jit/bin/python -m compileall -q v6_active_boundary_reduced/outer_solvers/prescreen.py v6_active_boundary_reduced/validation/test_outer_solver.py`; direct invocation of all four `test_outer_solver.py` test functions in `.venv_jit`; `git diff --check -- v6_active_boundary_reduced/outer_solvers/prescreen.py CODE_REVIEW_LOG.md`.

## 2026-06-17 - Prescreen Multistart Rationale Question

Scope:
`v6_active_boundary_reduced/outer_solvers/prescreen.py` and `v6_active_boundary_reduced/outer_solvers/lbfgsb.py`.

Question:
During review, the value of multistart seeding before a gradient-based outer optimizer was questioned. If the outer shell is already using L-BFGS-B, it is not obvious why random multistart seeds should be needed.

Before:
The code treated prescreening as center-point plus random-candidate sampling, full-profile evaluation, hard-gate filtering, ranking, and robust-neighborhood certification before L-BFGS-B. The rationale was implicit in the implementation rather than stated: the outer objective is a black-box rollout score with failure boundaries, active-set switches, and finite-difference optimizer gradients, not a clean smooth objective.

Change:
No code change. Logged the design question for the later `lbfgsb.py` review.

Current Behavior:
Every prescreen seed currently pays the cost of a full `evaluate_preparation_design(..., return_payload=True)` call. L-BFGS-B then starts only from ranked prescreen seeds that pass robust-neighborhood certification unless explicitly configured otherwise.

Rationale:
Multistart can be justified only if the outer objective has multiple feasible basins or hard failure regions where a single local start is unreliable. It is not a substitute for a cheaper physics-informed local derivative prescreen, and it may be wasteful if most candidate profiles are expensive and the objective landscape is not actually multi-basin.

Open Risks:
Current prescreening may be more expensive than necessary and may obscure the intended local-derivative screening idea. During `lbfgsb.py` review, verify whether multistart is still needed, whether a deterministic center/known-good seed is enough, or whether a cheap one-dx derivative screen should replace random full-profile sampling.

Verification:
Review-log-only entry. No tests run.

## 2026-06-17 - Outer L-BFGS-B Main Flow Review

Scope:
`v6_active_boundary_reduced/outer_solvers/lbfgsb.py`.

Question:
During review, the role of `lbfgsb.py` needed to be separated from `prescreen.py`, `core/objective.py`, and `outer_solvers/reward.py`.

Before:
The file looked like "the L-BFGS-B optimizer," but it actually owns more of the outer-shell prototype: prescreen invocation, normalized-space objective evaluation, full rollout evaluation, outer reward scoring, local neighborhood certification, SciPy run orchestration, post-run best certification, and artifact writing.

Change:
Added a `REVIEW` marker above the CLI `--objective` option noting that `PreparationSettings` is converted to reverse rollout; `delta_drop` is the supported production objective, while `power_next` remains diagnostic. No algorithmic behavior changed.

Current Behavior:
`run_outer_lbfgsb` first calls `prescreen_candidates` to produce ranked seeds. It then evaluates normalized candidates through `evaluate_preparation_design(..., return_payload=True)` and re-scores with `score_outer_result`. Seeds must pass neighborhood certification before L-BFGS-B unless the robust gate is disabled or nonrobust fallback is explicitly allowed. SciPy minimizes `-outer_score` over normalized `[0, 1]` bounds. The final `best` is selected only from feasible evaluations that pass post-run neighborhood certification; `best_seen_uncertified` records the highest feasible raw evaluation separately.

Rationale:
The robust-neighborhood gate is a guardrail around finite-difference gradients on a black-box rollout objective with failure boundaries. Keeping `best` reserved for locally certified profiles prevents boundary probes from becoming the reported optimizer result.

Open Risks:
The file still contains local JSON/CSV/profile-output helpers instead of importing a shared output layer. `certify_top_k=0` would prevent any certified `best` from being selected. The multistart/full-profile prescreen remains expensive and should be revisited when a cheaper local derivative prescreen is formalized.

Verification:
Ran `PYTHONPATH=. ./.venv_jit/bin/python -m compileall -q v6_active_boundary_reduced/outer_solvers/lbfgsb.py`; `PYTHONPATH=. ./.venv_jit/bin/python -m v6_active_boundary_reduced.outer_solvers.lbfgsb --help`; `git diff --check -- v6_active_boundary_reduced/outer_solvers/lbfgsb.py CODE_REVIEW_LOG.md`.

## 2026-06-17 - Outer Shell Robustness Workaround Assessment

Scope:
`v6_active_boundary_reduced/outer_solvers/lbfgsb.py`, `prescreen.py`, and their dependency on the reduced active-boundary rollout.

Question:
During review, the file's control flow looked simple but the algorithmic treatment looked heavy: full-profile multistart prescreening, normalized-space finite-difference L-BFGS-B, and full-rollout neighborhood certification around seeds and candidate best points.

Before:
The code treated local feasibility certification and multistart as necessary outer-shell guardrails, but it did not explicitly state that much of this machinery compensates for inner rollout fragility rather than solving the underlying numerical robustness problem.

Change:
No code change. Logged the architectural assessment.

Current Behavior:
The outer shell discards failed rollout evaluations, uses multistart to find feasible basins, and certifies local neighborhoods by re-running complete profiles under small normalized perturbations. This can be computationally acceptable after recent forward-solver speedups, but it remains expensive and indirect.

Rationale:
The heavy outer-shell mechanics are best understood as defensive wrappers around a non-robust or discontinuous inner forward/reverse rollout. They can make experiments run and avoid reporting boundary probes as final optima, but they are not a first-principles solution to solver failure, active-set discontinuity, or poor local model continuity.

Open Risks:
If effort continues to accumulate in outer-shell tricks, the project may optimize around inner solver failures instead of fixing them. Future work should prioritize forward solver robustness, local continuity, sonic/active-boundary handling, and a cheaper local derivative prescreen before treating the outer L-BFGS-B shell as a production optimizer.

Verification:
Review-log-only assessment. No tests run.

## 2026-06-17 - Sonic Delta Profile Review

Scope:
`v6_active_boundary_reduced/core/sonic_delta_profile.py`, with call-site checks in `v6_active_boundary_reduced/runners/run_sonic_delta_profile.py`, `v6_active_boundary_reduced/validation/test_sonic_delta_profile.py`, shared sonic helpers in `v6_active_boundary_reduced/core/sonic.py`, and the README sonic-profile example.

Question:
The file was reviewed as a standalone local profile builder through `M=1`, now relocated under `core/`, to check whether its branch selection, Delta objective semantics, and sonic helper usage still match the current active-boundary design.

Before:
The module documentation says each step selects the admissible root with the steepest requested `Delta` change while preserving `G >= g_floor`. The default settings are `branch_mode="fixed"` and `selection_mode="continuation"`, while the README example uses `--selection-mode steepest` but leaves `branch_mode` at its default.

Change:
No algorithmic code change. Logged the review findings so the semantic mismatch is visible before deciding whether to change defaults, gates, or documentation.

Current Behavior:
`build_sonic_delta_profile` builds a seed from `v6_firedrake_reduced.solve_local_sonic_match`, computes the primitive left-null sonic `sigma`, marches left and right with trapezoidal `sigma`, solves primitive finite-step residuals by least squares, and emits nodes, segments, arrays, and an active summary. In fixed branch mode, the left side is forced to `supersonic` and the right side to `subsonic`; candidate selection first filters by that Mach branch before applying the objective. As a result, even `objective="pedal"` with `selection_mode="steepest"` can return `ok=True` with nonzero direction-violation counts. With the README-scale `dx=1e-5`, fixed/steepest produced one reverse direction violation; agnostic/steepest produced no direction violations in the same smoke check.

Rationale:
This is not a crash-path bug because residual, `G`, `T_p`, and Mach-crossing checks still pass in the tested Freidberg local profile. It is a semantic risk: "sonic-compatible profile exists", "fixed Mach branch profile exists", and "pedal objective direction was obeyed" are currently separate facts, but the public `ok` flag and docstring can be read as if all three were satisfied together.

Open Risks:
The `ok` flag does not gate on `reverse_direction_violation_count` or `forward_direction_violation_count`, so callers may treat a profile as objective-consistent when it is only physically/admissibly solved. `support_type="sonic_compatible_steepest_delta"` is emitted even in continuation mode, where the selected candidate is continuity-ranked rather than the steepest-Delta candidate. `sonic_delta_profile.primitive_sonic_compatibility` duplicates the shared `core.sonic.primitive_sonic_compatibility` logic and lacks the shared helper's explicit `ok`/error handling and scalar singular-value schema, so future sonic-helper fixes can drift unless this module reuses the shared implementation.

Verification:
Read `sonic_delta_profile.py` with line numbers and traced callers/tests. Ran `PYTHONPATH=. ./.venv_jit/bin/python v6_active_boundary_reduced/validation/test_sonic_delta_profile.py`; `PYTHONPATH=. ./.venv_jit/bin/python -m compileall -q v6_active_boundary_reduced/core/sonic_delta_profile.py v6_active_boundary_reduced/runners/run_sonic_delta_profile.py v6_active_boundary_reduced/validation/test_sonic_delta_profile.py`; and `PYTHONPATH=. ./.venv_jit/bin/python -m v6_active_boundary_reduced.runners.run_sonic_delta_profile --case freidberg_reference --out-dir outputs/tmp_sonic_delta_profile_review --n-steps-each-side 1 --scan-points 15 --no-plot`. The temporary output directory was removed after inspection.
