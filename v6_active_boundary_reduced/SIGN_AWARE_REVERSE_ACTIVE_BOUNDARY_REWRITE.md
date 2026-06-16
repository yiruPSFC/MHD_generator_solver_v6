# Sign-Aware Reverse Active-Boundary Rewrite Plan

This note records the sign-aware rewrite plan for `v6_active_boundary_reduced`.
The solver should stay focused on the reverse preparation problem:

```text
given:  a downstream target anchor
return: an upstream preparation profile that can reach it
```

Do not implement a forward solver for this rewrite.  The algorithmic issue is
only the local active-set decision inside the reverse step.

## Coordinate And Step Convention

Use the physical forward coordinate `x`, but march upstream:

```text
dx > 0
x_upstream = x_current - dx
Y_upstream ~= Y_current - dx * Y_prime_forward
```

Do not encode the reverse step by making `dx` negative.  Keep `dx` positive and
write the reverse minus sign explicitly; otherwise the G sign rules are
inverted.

The area slope variable is:

```text
sigma = d(log A)/dx
A_prime = dA/dx = A_current * sigma
logA_upstream ~= logA_current - dx*sigma
```

## Core Rule

For the reverse solver, define the local one-step objective and constraint
directly in reverse-step quantities:

```text
A_prime = A * sigma

objective_drop(A_prime)
  = Phi_current - Phi_upstream
  ~= p0 + p1 * A_prime

G_margin_upstream(A_prime)
  = G_upstream - G_floor
  ~= q0 + q1 * A_prime
```

where:

```text
Phi = T_e / T_p - 1
```

With this convention, `p1` is the slope of the reverse objective and `q1` is
the slope of the reverse `G` feasibility margin.  Then the limiter test is the
usual geometric test:

```text
p1*q1 < 0  -> G opposes the reverse objective direction; G can be the limiter.
p1*q1 > 0  -> G permits the reverse objective direction; G is not the limiter.
p1*q1 ~= 0 -> singular/flat case; classify separately.
```

This is the rule the code should implement.  Do not use `q1` to mean the
coefficient of physical forward `G_prime` in the final active-set code or the
sign test will appear reversed.

## Why This Convention

The current equations are usually easiest to derive in the physical forward
coordinate `x`:

```text
Phi_prime_forward = a0 + a1 * A_prime
G_prime_forward   = b0 + b1 * A_prime
```

But the solver marches upstream:

```text
Phi_upstream ~= Phi_current - dx * Phi_prime_forward
G_upstream   ~= G_current   - dx * G_prime_forward
```

Therefore the reverse-step coefficients used by the active-set policy are:

```text
p0 = dx * a0
p1 = dx * a1

q0 = G_current - G_floor - dx * b0
q1 = -dx * b1
```

After this conversion, the code should reason only with `p1` and `q1` as
reverse-step quantities.

## Current Problem

The existing reverse rollout can over-select `G=0` because it asks whether a
candidate is close to a `G` boundary before asking whether that boundary
actually blocks the objective-preferred `A_prime` direction.

Being near `G=0` is not enough.  The solver must first determine whether the
reverse `G` margin increases or decreases along the objective direction.

Examples:

```text
p1 > 0, q1 < 0:
  objective wants larger A_prime
  G margin decreases with larger A_prime
  => G can be an upper-bound limiter

p1 > 0, q1 > 0:
  objective wants larger A_prime
  G margin increases with larger A_prime
  => G is permissive, not the limiter

p1 < 0, q1 > 0:
  objective wants smaller A_prime
  G margin decreases when moving smaller
  => G can be a lower-bound limiter

p1 < 0, q1 < 0:
  objective wants smaller A_prime
  G margin increases when moving smaller
  => G is permissive, not the limiter
```

So the sign test is exactly:

```text
p1*q1 < 0 -> candidate G limiter
p1*q1 > 0 -> G permissive
```

## Numerical Anchors From This Discussion

Freidberg reference near `x=0.01 m`, computed first in forward-coordinate
coefficients:

```text
a1 = d(Phi_prime_forward)/dA_prime = +1.7079e2
b1 = d(G_prime_forward)/dA_prime   = +6.6555e5
```

For the reverse solver:

```text
p1 = dx*a1      > 0
q1 = -dx*b1     < 0
p1*q1           < 0
```

Reverse interpretation: the objective wants larger `A_prime`, while upstream
`G` margin decreases with larger `A_prime`.  Therefore `G` can genuinely limit
the Freidberg-near reverse recovery.  The solver should still verify that the
selected endpoint is actually the `G` bound rather than a tighter geometry,
curvature, temperature, or sonic bound.

Broad-scan opposite-sign case in forward-coordinate coefficients:

```text
A = 0.447 m^2
log_n_p_in = 60.4542531912
n_p = 1.7986578958e26 m^-3
T_e_in = 12639.3496714709 K
Z_in = 167.1096772810
I_0 = 2004.2295307617 A
log_seed_fraction = -10.0718696001
seed_fraction = 4.2251546549e-5
B_T = 1.3004131441 T

T_p = 10378.3028840064 K
Phi = 0.2178628638
G = 1.6348532785e8
Mach = 1.2037234338

a1 = +6.28945418
b1 = -4.05845845e9
```

For the reverse solver:

```text
p1 = dx*a1      > 0
q1 = -dx*b1     > 0
p1*q1           > 0
```

Reverse interpretation: the objective wants larger `A_prime`, and upstream
`G` margin also increases with larger `A_prime`.  `G` is permissive here.  A
reverse policy that chooses a `G` boundary in this local sign regime is probably
using the wrong active-set logic, unless the finite-`dx` solve reveals a tighter
nonlinear constraint.

## Rewrite Algorithm

Rewrite `_policy_step` around explicit reverse interval construction.

1. Compute current closure and reject already-invalid states:

```text
G_current >= G_floor
T_p_current >= T_p_floor
finite closure
```

2. Compute forward-coordinate affine coefficients from the local dynamics:

```text
Phi_prime_forward = a0 + a1*A_prime
G_prime_forward   = b0 + b1*A_prime
det_D
```

3. Convert them immediately into reverse-step coefficients:

```text
p0 = dx*a0
p1 = dx*a1
q0 = G_current - G_floor - dx*b0
q1 = -dx*b1
```

After this point, the active-set code should use only `p0,p1,q0,q1`.

4. Build the reverse candidate interval from geometry:

```text
sigma_min <= sigma <= sigma_max
logA_min <= logA_current - dx*sigma <= logA_max
|sigma - sigma_prev| <= curvature_max, if enabled
A_prime = A_current * sigma
```

The log-area interval is:

```text
(logA_current - logA_max)/dx <= sigma <= (logA_current - logA_min)/dx
```

The lower bound contains `logA_max`, and the upper bound contains `logA_min`
because the reverse update is `logA_upstream = logA_current - dx*sigma`.

5. Intersect the interval with reverse `G` admissibility:

```text
q0 + q1*A_prime >= 0
```

Use the sign of reverse `q1`:

```text
q1 > 0 -> G gives a lower bound on A_prime
q1 < 0 -> G gives an upper bound on A_prime
q1 ~= 0 -> either all feasible or all infeasible for G at this order
```

6. Choose the objective-preferred endpoint:

```text
p1 > p1_tol  -> choose interval upper endpoint
p1 < -p1_tol -> choose interval lower endpoint
otherwise    -> singular/flat; use a secondary regularizer
```

7. Classify support:

```text
G_limited_reverse
  selected endpoint is the G bound and p1*q1 < 0

G_permissive_reverse
  p1*q1 > 0 and selected endpoint is not the G bound

geometry_limited
curvature_limited
area_limited
temperature_limited
sonic_compat_limited
singular_flat
```

8. Solve the actual finite-`dx` upstream state with the selected `sigma`.
The current implementation advances this state with RK4; the selected sigma must
come from the sign-aware reverse interval logic.

9. Validate the finite-`dx` result:

```text
G_upstream >= G_floor
T_p_upstream >= T_p_floor
RK4 error estimate <= rk4_error_tol
```

If validation fails, refine or backtrack against the actual finite step.  The
affine model is the active-set predictor, not a substitute for the finite-step
residual solve.

## Sonic/Choking Branch

The affine analysis assumes:

```text
det(D) != 0
```

Near choking, switch branches when:

```text
abs(det_D) < det_tol
```

Then solve a sonic compatibility condition instead of using the generic
endpoint rule:

```text
D * y_prime = f0 + A_prime*f1
ell^T * (f0 + A_prime*f1) = 0
```

where `ell` is the left null vector of `D`.  This produces a compatibility
value of `A_prime`.  Keep the scalar G coefficients `b0,b1` separate from the
vector sonic RHS coefficients `f0,f1`; they are not interchangeable.  Do not
allow the ordinary reverse endpoint policy to cross this region without the
compatibility check.

## File-Level Implementation Plan

Suggested additions:

```text
v6_active_boundary_reduced/local_affine.py
v6_active_boundary_reduced/reverse_sign_policy.py
```

Suggested modifications:

```text
v6_active_boundary_reduced/policy.py
v6_active_boundary_reduced/run_preparation_recovery.py
v6_active_boundary_reduced/run_anchor_scan.py
v6_active_boundary_reduced/run_anchor_optimize.py
```

Keep `recover_preparation_profile` as the main public API.  The scan and
optimization entrypoints can remain reverse-target-anchor workflows, but their
segment outputs must expose sign diagnostics.

## Required Diagnostics

Every segment row should include:

```text
# physical-forward affine coefficients
a0
a1
b0
b1

# reverse-step coefficients
p0
p1
q0
q1
p1q1_reverse

# geometry / interval state
det_D
sonic_branch_used
A_current
logA_current
dx
sigma_min
sigma_max
sigma_logA_lower
sigma_logA_upper
sigma_curvature_lower
sigma_curvature_upper
sigma_interval_lower
sigma_interval_upper
sigma_selected
lower_source
upper_source

# G margin state
G_margin_current
G_margin_upstream_predicted
G_margin_upstream
reverse_G_bound_kind   # lower, upper, none, infeasible_flat
sigma_G_bound

# objective and validation state
Phi_current
Phi_upstream_predicted
objective_drop_predicted
objective_bound_kind   # lower, upper, flat
T_p_upstream
T_p_margin_upstream
residual
validation_status
validation_failure_reason
support_type
sigma_selected
```

Without these fields, it is impossible to distinguish real reverse
`G`-limited recovery from an accidental boundary-selection artifact.

## Tests And Smoke Checks

1. Freidberg local sign regression:

```text
At the Freidberg reference anchor or x=0.01 m:
a1 > 0
b1 > 0
p1 > 0
q1 < 0
p1*q1 < 0
```

The reverse policy should classify `G` as a possible upper-bound limiter and,
near the marginal reference state, should recover the `G` boundary unless a
tighter geometry, curvature, temperature, or sonic constraint intervenes.

2. Opposite-sign/permissive regression:

Use the broad-scan point listed above and verify:

```text
a1 > 0
b1 < 0
p1 > 0
q1 > 0
p1*q1 > 0
```

The reverse policy should classify `G` as permissive for the larger-`A_prime`
objective direction.  It should not choose the `G` lower bound unless another
finite-step effect makes the preferred endpoint invalid.

3. Interval algebra regression:

Construct synthetic reverse-coefficient cases and verify:

```text
q1 > 0 -> reverse G lower bound
q1 < 0 -> reverse G upper bound
p1*q1 < 0 -> reverse G can oppose objective
p1*q1 > 0 -> reverse G permits objective
```

4. Short reverse rollout smoke:

Run a Freidberg anchor recovery and require segment diagnostics to show why
`G` is selected.  The expected explanation is `p1*q1_reverse < 0`, not merely
"the profile is close to marginal stability."

Run with:

```bash
./.venv_jit/bin/python -m pytest ...
```

Use `.venv_jit`; the default system Python may have incompatible NumPy/Numba or
SciPy dynamic-library issues.

## Bottom Line

Keep the algorithm reverse-focused.  The correct rewrite is:

```text
compute forward local coefficients a0,a1,b0,b1;
convert immediately to reverse coefficients p0,p1,q0,q1;
construct the reverse G interval using q1;
check whether p1*q1 makes G oppose or permit the reverse objective direction;
choose the objective-preferred feasible endpoint;
switch to sonic compatibility when det(D) approaches zero.
```

With the reverse-step convention used by the solver:

```text
p1*q1 < 0 -> G can be the limiter
p1*q1 > 0 -> G is permissive
```

## 2026-06-11 Review Log - Policy Rollout Cleanup

Scope:
`policy.py`, `reverse_sign_policy.py`, and the reverse preparation rollout path from
`PolicySettings`/`PreparationSettings` through `_reverse_sign_policy_step`,
finite-step RK4 validation, fallback, and active summary diagnostics.

Question:
The review focused on which pieces were still real policy logic versus legacy
plumbing, why sign-aware local-affine reverse selection is preferred over the
legacy scan path, and which design assumptions should be visible outside the
chat transcript.

Before:
Earlier revisions still carried implicit backward-Euler solver plumbing,
mandatory/invented anchor `sigma_logA`, a redundant profile-window rollout
entrypoint, and state warm-start extrapolation left over from nonlinear solves.
The fallback path also did not expose enough diagnostics to tell when an affine
endpoint had been replaced by a finite-step feasible sigma.

Change:
The current code has removed the implicit backward-Euler backend, made anchor
`sigma_logA` optional so the first rollout step disables curvature when no real
slope history exists, removed the redundant profile-window rollout entrypoint,
and removed state extrapolation warm-start plumbing.  Fallback status and
endpoint diagnostics are now returned and flattened for downstream inspection.

Current Behavior:
Reverse `delta_drop` uses the sign-aware local-affine path: compute forward
local coefficients, convert them immediately to reverse `p0,p1,q0,q1`, intersect
the sigma interval with reverse `G` admissibility, choose the
objective-preferred endpoint, and validate the finite step with RK4.  If the
affine endpoint fails finite-step validation, scan/backtrack fallback searches
the same interval and selects the most aggressive RK4-feasible sigma in the
objective direction.

Rationale:
The local-affine path makes the active-set decision explicit.  In reverse
preparation, `G` should be treated as a limiter only when `p1*q1 < 0`; being
near marginal stability is not enough.  RK4 remains the finite-step validator
because the affine model is only a one-step predictor.

Open Risks:
Only reverse `delta_drop` currently uses sign-aware local-affine semantics; other
directions/objectives still fall through to the legacy scan policy.  The
scan/backtrack fallback also treats `p1 == 0` as the upper-side case, whereas
endpoint selection has a flat tolerance and regularizer.

Verification:
Validation was run during the cleanup with compile/smoke checks after the code
edits.  This review-log update only records those decisions and adds narrow
risk comments; it does not change rollout behavior.

## 2026-06-11 Review Log - Sonic Left-Null Degeneracy

Scope:
`policy.py::_primitive_sonic_compatibility` and
`policy.py::_sonic_compatible_policy_step`.

Question:
The review asked whether sonic singular behavior is only `det(D) -> 0`, or
whether the compatibility denominator `ell^T f1` becoming small can create a
separate failure mode that should warn or hard-fail.

Before:
The sonic branch computes `sigma_sonic = -ell^T f0 / (A * ell^T f1)`.  It only
hard-fails immediately when the denominator is numerically zero, and later
rejects nonfinite or out-of-interval `sigma_sonic`.  It does not currently
classify a small-but-finite `ell^T f1` as an explicit loss of sonic control
authority.

Change:
The sonic branch now evaluates the interval residual
`r(sigma)=ell^T f0 + A*sigma*ell^T f1` after slope/area/curvature bounds are
known.  It classifies the compatibility solve as `root_in_interval`,
`boundary_compatible`, `flat_compatible`, `unreachable_interval`,
`unreachable_flat_forcing`, or `invalid_left_null_data`.

Current Behavior:
If `r(sigma)=0` has a root inside the active interval, that root is selected.
If the compatibility residual is flat and already small over the interval, the
branch selects a clipped continuation/reference sigma.  If the residual is flat
but nonzero, or if the root is outside the interval and no endpoint is within
tolerance, the step fails with explicit sonic compatibility diagnostics instead
of reporting only a nonfinite/out-of-range sigma.

Rationale:
`det(D) -> 0` is the primitive matrix singularity.  `ell^T f1 -> 0` is different:
it means the area-slope control has little or no projection onto the left-null
compatibility direction, so changing `A'` cannot efficiently cancel the singular
forcing component `ell^T f0`.

In the current primitive system:

```text
D = [[M11, M12],
     [E11, E12]]

f0 = [J_y B,
      1.5 * nu_E * n_e * (T_e - T_p) / v_p]

f1 = [-M13,
      -E13]
```

where the full two-equation differential form is equivalently:

```text
M11*n' + M12*Te' + M13*A' = J_y*B
E11*n' + E12*Te' + E13*A' = 1.5*nu_E*n_e*(T_e - T_p)/v_p
```

Thus `det(D)=M11*E12-M12*E11 -> 0` says the `n'` and `Te'`
columns lose rank.  At exact rank loss, a left-null vector can be written,
up to scale, as `[E11, -M11]` or `[E12, -M12]`.  Then
`ell^T f1 -> 0` means the area column `[M13,E13]` also lies in the same
rank-one column space, so `A'` does not provide a transverse control direction
through the sonic compatibility condition.

Open Risks:
The flat-compatible branch uses the local continuation/reference sigma rather
than an objective scan.  That is conservative for continuity, but it is not a
global optimization over the flat sonic manifold.

Verification:
Added a direct algebraic test for the four main compatibility cases and kept the
existing sonic branch smoke test.

## 2026-06-11 Review Log - Sonic Residual Gate

Scope:
`policy.py::_apply_sonic_residual_gate`.

Question:
The review noted that sonic finite-step acceptance should be based on the actual
scaled momentum/energy residual and active constraints, not only on the
optimizer's success flag.

Before:
The finite-step solver records the raw `least_squares` success flag, but the
sonic branch applies its own residual tolerance afterward.

Change:
No behavior change.  A short code comment now marks that this is intentional:
low scaled residual can override a conservative `least_squares` flag.

Current Behavior:
The sonic residual gate accepts the step when `G`, `T_p`, and scaled residual
margins are all within tolerance.  Otherwise it marks the gate as failed and
keeps the constraint violation for diagnostics.

Rationale:
Near the sonic point, optimizer status can be conservative even when the
physical finite-step residual is already below the explicit sonic tolerance.

Open Risks:
This relies on the residual scaling in `_solve_sonic_finite_step`; that scaling
should be reviewed with the finite-step solve itself.

Verification:
`policy.py` compile check passed after the comment-only change.

## 2026-06-11 Review Log - Sonic Delta Direction Gate

Scope:
`policy.py::_sonic_compatible_policy_step`.

Question:
The review questioned whether reverse `delta_drop` should veto a sonic-compatible
step when the selected sonic step does not locally decrease `Delta`.

Before:
After selecting a sonic-compatible sigma and solving the finite step, reverse
`delta_drop` marked the step as failed with `delta_not_dropping_upstream` when
`delta_gain > active_tol`.

Change:
The delta-direction veto was removed for sonic-compatible steps.  `delta_gain`
and `sonic_objective_score` remain diagnostic fields, and the segment now marks
`sonic_direction_gate = not_applied`.

Current Behavior:
Sonic branch acceptance is controlled by compatibility, finite-step residual,
`G`, and `T_p`.  The local `Delta` direction near choking is reported but does
not decide feasibility.

Rationale:
Near choking, the area slope is constrained by the compatibility condition.  A
local objective-direction veto can reject a physically admissible sonic crossing
that the policy cannot freely steer.

Open Risks:
This can allow short local deviations from the reverse `delta_drop` objective
near sonic points.  Those should be interpreted as sonic compatibility costs,
not ordinary policy choices.

Verification:
`policy.py`/`objective.py` compile checks passed, and the direct sonic delta
profile smoke passed.

## 2026-06-11 Review Log - Sonic Seed Plumbing Cleanup

Scope:
`policy.py::_policy_step`, `_legacy_policy_step`,
`_forward_sonic_fallback_payload`, `_reverse_sign_policy_step`, and
`_sonic_compatible_policy_step`.

Question:
The review noted that `seed_next` looked like stale state extrapolation plumbing
after the generic implicit backend and state warm-start path were removed.

Before:
`seed_next` was passed from rollout into policy dispatch and then through the
sonic branch, but no current code read it.  Sonic finite-step solves now build
their own deterministic multi-start guesses around the current state and selected
next area.

Change:
Removed the unused `seed_next` parameter and all pass-through call arguments.

Current Behavior:
Policy steps depend on the current state, slope history, optional sigma warm
start, step size, direction, config, and settings.  Sonic finite-step initial
guesses are produced only by `_sonic_initial_guesses`.

Rationale:
Keeping an unused state seed made the sonic branch look like it had two separate
next-state initialization mechanisms.  Removing it clarifies that the only
remaining multi-start source is the local sonic guess generator.

Open Risks:
None noted.

Verification:
`python -m compileall -q v6_active_boundary_reduced/policy.py` passed, and the
direct sonic delta profile smoke passed with `.venv_jit`.

## 2026-06-11 Review Log - Sonic Implicit Step Check

Scope:
`policy.py::_evaluate_sonic_sigma`, `policy.py::_solve_sonic_finite_step`, and
the generic RK4 comparison path through `_evaluate_sigma`.

Question:
The review asked whether the sonic branch truly needs a finite-step nonlinear
solve, or whether an explicit RK4 step should be enough once `sigma_sonic` is
known.

Before:
The sonic branch used a multi-start `least_squares` finite-step solve for
`(log_n_next, log_Te_next)` after selecting `sigma_sonic`, but the need for this
was not demonstrated in the review notes.

Change:
No code behavior change for this topic.  A local validation compared, at the
Freidberg sonic seed and nearby log-density perturbations, the sonic finite-step
solve against generic RK4 using the same `sigma_sonic`.

Current Behavior:
At the exact sonic seed, RK4 produced NaN/inf or failed error checks, while the
sonic finite-step solve produced scaled momentum/energy residuals around
`1e-12` to `1e-7` depending on step direction and size.  Near but not exactly at
sonic, RK4 recovered only after moving sufficiently far from the rank-loss
region; the finite-step solve remained low-residual across the tested offsets.

Rationale:
`sigma_sonic` fixes the area slope and therefore `logA_next`, but it does not
uniquely determine `(n_next, T_e_next)`.  At `det(D) ~= 0`, the explicit RHS
requires an ill-conditioned inverse of `D`; the finite-step solve instead asks
for a next state satisfying the discrete momentum/energy equations at the next
area.

Open Risks:
The finite-step solve can still have multiple roots.  Current root selection is
objective-based among feasible candidates rather than explicitly
continuation-based.

Verification:
One-off local comparison with `.venv_jit` on `freidberg_reference`; no persistent
benchmark artifact was added.

## 2026-06-11 Review Log - Sonic Multi-Start Roots

Scope:
`policy.py::_sonic_initial_guesses` and `policy.py::_evaluate_sonic_sigma`.

Question:
The review asked why the sonic finite-step solve uses multiple initial guesses
for one selected sigma.

Before:
The code tried a deterministic set of log-space perturbations around the current
state but did not document the branch-selection risk.

Change:
No behavior change.  A short risk comment now marks that multiple starts may
find different finite-step roots, and the current selection is objective-based
rather than continuity-based.

Current Behavior:
All starts target the same next `logA` and same selected sigma.  If multiple
finite-step candidates are feasible, the candidate with largest objective value
is selected; if none are feasible, the least-violating candidate is returned.

Rationale:
The multi-start solve improves robustness of the nonlinear sonic finite-step
solve near singular dynamics, but different converged roots may represent
different local branches.

Open Risks:
A continuation-based selector may be more appropriate than objective-based
selection when multiple feasible sonic roots exist.

Verification:
`policy.py`/`objective.py` compile checks passed, and the direct sonic delta
profile smoke passed.

## 2026-06-12 Review Log - Sonic Finite-Step Residual Payload

Scope:
`policy.py::_solve_sonic_finite_step`, `_evaluate_sonic_sigma`,
`_apply_sonic_residual_gate`, and the matching residual solve in
`sonic_delta_profile.py`.

Question:
The resumed review picked up around `policy.py` line 1400 and checked whether
the sonic finite-step residual, scaling, and `residual_ok` fields were part of
the acceptance policy or only diagnostic plumbing.

Before:
The sonic finite-step solve looked suspicious because `_solve_sonic_finite_step`
sets `residual_ok` with a hard-coded `1e-7`, while `PolicySettings` exposes
`sonic_residual_tol`.

Change:
No code behavior change.  Added a short `RISK` comment marking that the hard-coded
`1e-7` is solver-local diagnostics, not the final policy gate.

Current Behavior:
The finite-step solve evaluates momentum and energy residuals at the next state,
using backward/forward finite differences from the current primitive state and
the selected next `logA`.  This matches the standalone sonic delta profile
solver.  `_evaluate_sonic_sigma` builds candidate margins with
`settings.sonic_residual_tol`, and `_apply_sonic_residual_gate` rewrites the
candidate acceptance state from the configurable residual margin.  The local
`residual_ok <= 1e-7` inside `_solve_sonic_finite_step` is therefore an
intermediate diagnostic, not the final acceptance criterion.

Rationale:
At the sonic singular point the explicit RK4 RHS is not reliable because it
requires inverting the nearly rank-deficient primitive matrix.  The finite-step
residual solve instead asks whether a next primitive state satisfies the
discrete momentum and energy equations at the chosen area slope.

Open Risks:
The duplicated `1e-7` diagnostic can still confuse readers because it differs
from the configurable `sonic_residual_tol`.  It is not currently a behavior bug
because the later residual gate overwrites acceptance fields.

Verification:
`python -m compileall -q v6_active_boundary_reduced/policy.py` passed after the
comment-only code change.

## 2026-06-12 Review Log - Sign-Aware Scan Fallback Flat-P1 Side

Scope:
`policy.py::_sign_aware_scan_fallback`, `reverse_sign_policy.py::choose_objective_endpoint`,
and the reverse endpoint dispatch in `policy.py::_reverse_sign_policy_step`.

Question:
The review checked whether the scan fallback uses the same near-flat objective
logic as the primary sign-aware endpoint selector.

Before:
`choose_objective_endpoint` treats `|p1| <= p1_tol` as flat and clips to a
regularizer sigma.  The scan fallback instead chooses the upper feasible side
when `p1 >= 0` and the lower feasible side otherwise.

Change:
No code behavior change.  The existing `RISK` comment in
`_sign_aware_scan_fallback` remains the source marker.

Current Behavior:
The primary sign-aware endpoint path has an explicit flat-objective branch.  If
that endpoint fails finite-step validation, the scan fallback evaluates the
whole admissible interval and picks an extreme feasible sigma based only on the
sign of `p1`; exact or near-zero `p1` falls to the upper side.

Rationale:
The fallback is meant as a recovery path after endpoint validation fails, not as
the canonical affine endpoint selector.  That makes the simpler side choice
understandable, but it is still a real policy difference from the primary
endpoint logic.

Open Risks:
When `p1` is numerically flat, scan fallback can introduce an arbitrary
upper-side preference instead of preserving the regularizer/warm-start bias.
This may matter in low-gradient regions where many sigmas are nearly equivalent.

Verification:
Source review only.  No tests run because this entry documents existing behavior
and an already-marked risk.

## 2026-06-12 Review Log - Sign-Aware Diagnostic Attribution

Scope:
`policy.py::_sign_aware_base_diagnostics`,
`policy.py::_sign_aware_finite_diagnostics`, and the return payload from
`policy.py::_reverse_sign_policy_step`.

Question:
The review continued below the sonic finite-step solve into the diagnostic
payload builders and checked whether any reported fields could be mistaken for
the actual accepted support.

Before:
`_sign_aware_finite_diagnostics` reported `objective_bound_kind` and
`selected_endpoint_source` from the original affine endpoint decision even when
finite-step validation failed and scan/backtrack fallback replaced the chosen
sigma.

Change:
Added explicit selected-sigma diagnostics:
`selected_sigma_origin`, `selected_sigma_source`, and `selected_support_type`.
The old endpoint fields remain as backward-compatible affine-proposal fields,
and new `affine_objective_bound_kind` / `affine_selected_endpoint_source` fields
make that meaning explicit.  Termination summaries, objective CSV flattening,
the local affine benchmark comparison, and the sign-aware smoke test now include
the new fields.

Current Behavior:
The final payload now separates original affine intent from the actual finite
step.  If fallback is used, `selected_sigma_origin = scan_fallback`,
`selected_sigma_source` reports the recovery method, and
`selected_support_type` classifies the actual chosen sigma.  The older
`objective_bound_kind` and `selected_endpoint_source` still describe the
original affine endpoint proposal for compatibility.

Rationale:
Keeping the original endpoint attribution is useful for explaining why the
affine policy tried that side first, but the actual accepted finite step needs
its own fields so downstream readers do not have to infer fallback recovery from
several separate diagnostics.

Open Risks:
Existing consumers that treat `selected_endpoint_source` as the actual accepted
source should switch to `selected_sigma_source` / `selected_support_type`.

Verification:
`python -m compileall -q v6_active_boundary_reduced/policy.py
v6_active_boundary_reduced/objective.py
v6_active_boundary_reduced/benchmark_local_affine_algebraic.py
v6_active_boundary_reduced/validation/test_freidberg_sign_aware_smoke.py` passed.
`env -u PETSC_DIR -u PETSC_ARCH ./.venv_jit/bin/python
v6_active_boundary_reduced/validation/test_freidberg_sign_aware_smoke.py` passed.

## 2026-06-12 Review Log - Legacy Policy Deletion Marker

Scope:
`policy.py::_policy_step`, `policy.py::_legacy_policy_step`, and
`policy.py::_sign_aware_scan_fallback`.

Question:
The review noticed that `_sign_aware_scan_fallback` repeats the old scan /
feasible / boundary-refine structure from `_legacy_policy_step`, and the active
workflow no longer needs the legacy policy route.

Before:
`_policy_step` still routes non reverse-`delta_drop` objectives into
`_legacy_policy_step`, and `_legacy_policy_step` remains a large parallel scan
implementation next to the sign-aware route.

Change:
No behavior change.  Added a `REVIEW` marker at `_legacy_policy_step` noting that
active development has moved to the sign-aware reverse path and that the legacy
scan route should be deleted after callers are migrated.

Current Behavior:
The legacy route still exists for compatibility with current dispatch behavior,
but it is now explicitly marked as a cleanup/deletion target.

Rationale:
Leaving the marker near the function definition makes the future refactor target
visible when revisiting duplicated scan fallback logic.

Open Risks:
Deleting `_legacy_policy_step` requires either migrating or removing the
remaining non reverse-`delta_drop` callers in `_policy_step`.

Verification:
`python -m compileall -q v6_active_boundary_reduced/policy.py` passed, and
`git diff --check -- v6_active_boundary_reduced/policy.py
v6_active_boundary_reduced/SIGN_AWARE_REVERSE_ACTIVE_BOUNDARY_REWRITE.md`
passed.

## 2026-06-12 Review Log - Sign-Aware G Boundary First

Scope:
`policy.py::_reverse_sign_policy_step`,
`policy.py::_sign_aware_g_boundary_fallback`,
`policy.py::_sign_aware_scan_fallback`, and
`validation/test_policy_behavior_guards.py`.

Question:
The review questioned whether a `G`-limited sign-aware endpoint should scan the
whole reverse interval first, or solve the true finite-step `G_next = G_floor`
boundary first.

Before:
If the affine sign-aware endpoint failed finite-step validation, the reverse
path immediately scanned the admissible sigma interval and only refined a
boundary after finding a feasible scan point next to an infeasible neighbor.

Change:
For endpoint failures that include `G`, the reverse path now first tries a
finite-step brentq solve on the true evaluated margin
`constraint_margins["G"]`.  Scan fallback remains the recovery path when the
endpoint/opposite endpoint do not bracket a feasible `G` boundary.

Current Behavior:
Freidberg sign-aware smoke now reports `selected_sigma_origin =
g_boundary_fallback` and `solver_method =
sign_aware_brentq_G_boundary_fallback` for the first `G`-limited recovery.  The
scan fallback path is still covered separately and still refines bracketed
`G` boundaries.

Rationale:
When the failure attribution is specifically `G`, the marginal `G` boundary is
the physical/numerical target.  Scanning first is useful as a fallback, but it
should not be the primary recovery path for a directly bracketed finite-step
`G` margin.

Open Risks:
The brentq-first path still requires a finite sign-changing bracket.  If both
interval endpoints are infeasible or do not bracket `G`, recovery falls back to
scan/backtrack.

Verification:
`env -u PETSC_DIR -u PETSC_ARCH ./.venv_jit/bin/python
v6_active_boundary_reduced/validation/test_policy_behavior_guards.py` passed.
`env -u PETSC_DIR -u PETSC_ARCH ./.venv_jit/bin/python
v6_active_boundary_reduced/validation/test_freidberg_sign_aware_smoke.py`
passed.
`env -u PETSC_DIR -u PETSC_ARCH ./.venv_jit/bin/python
v6_active_boundary_reduced/validation/test_sonic_delta_profile.py` passed.

## 2026-06-12 Review Log - Diagnostics And Solver Module Split

Scope:
`policy.py`, `diagnostics.py`, `finite_step.py`, `sonic.py`,
`sigma_interval.py`, `reverse_sign_policy.py`, and
`sonic_delta_profile.py`.

Question:
The review found that `policy.py` mixed policy choice, sigma bounds, finite-step
RK4 execution, sonic compatibility, sonic finite-step solves, and public
diagnostic payload formatting in one file.

Before:
`policy.py` owned `_sigma_interval`, `_evaluate_sigma`, RK4 RHS/linear solves,
stage diagnostics, sonic compatibility choice, sonic least-squares solves, scan
summary payloads, active summaries, and termination summaries.  RK4
step-doubling error was also exposed through generic
`max_abs_scaled_residual` / `residual_ok` names, which could be confused with a
physical residual.

Change:
Moved sigma interval bounds to `sigma_interval.py`, public payload/scan/active
summary formatting to `diagnostics.py`, RK4 finite-step logic and stage
diagnostics to `finite_step.py`, and sonic compatibility/finite-step helpers to
`sonic.py`.  `policy.py` now keeps thin wrappers where callback injection is
needed for `State`, `_physics_params`, `_closure_metrics`, and objective
payloads.  `sonic_delta_profile.py` now uses the shared sonic finite-step
residual vector.  RK4 diagnostics now expose `rk4_error_estimate`,
`rk4_error_ok`, and `rk4_error_margin`; physical sonic residuals expose
`physical_residual_scaled` and `physical_residual_ok`.

Current Behavior:
Compatibility fields `max_abs_scaled_residual` and `residual_ok` are still
present for existing payload consumers, but RK4 and physical residual meanings
are now separated by `step_error_kind` and the explicit RK4/physical names.

Rationale:
The split makes it harder for policy selection changes to accidentally edit
low-level finite-step numerics or public reporting schemas.  It also removes the
diagnostic naming ambiguity found during review.

Open Risks:
`policy.py` still carries wrapper functions because `State` and config-derived
physics parameters have not yet been moved to a shared type/context module.
Private helper aliases are retained for compatibility with current tests and
review-time references.

Verification:
`python -m compileall -q v6_active_boundary_reduced/policy.py
v6_active_boundary_reduced/finite_step.py v6_active_boundary_reduced/sonic.py
v6_active_boundary_reduced/sonic_delta_profile.py
v6_active_boundary_reduced/diagnostics.py
v6_active_boundary_reduced/sigma_interval.py
v6_active_boundary_reduced/reverse_sign_policy.py
v6_active_boundary_reduced/validation/test_policy_behavior_guards.py` passed.
The three smoke commands listed in the previous entry passed.

## 2026-06-12 Review Log - Legacy Route Removal

Scope:
`policy.py::_policy_step`, `policy.py::_forward_scan_policy_step`,
`policy.py::_direct_boundary_choice`, and
`validation/test_policy_behavior_guards.py`.

Question:
The review concluded that `_legacy_policy_step` should not remain as a hidden
catch-all after the sign-aware reverse path became the active reverse policy.

Before:
`_policy_step` sent reverse `delta_drop` to the sign-aware route, but every
other direction/objective combination fell through to `_legacy_policy_step`.
That meant unsupported reverse objectives could silently use old scan semantics.

Change:
Deleted the `_legacy_policy_step` entry point by renaming the remaining scan
implementation to `_forward_scan_policy_step`.  `_policy_step` now explicitly
routes reverse `delta_drop` to sign-aware, routes forward cases to forward scan,
and returns `unsupported_reverse_objective` for reverse non-`delta_drop`
objectives.  Added a behavior guard for that unsupported reverse path.

Current Behavior:
Forward benchmark-style runs still use scan semantics through the explicitly
named forward builder.  Reverse preparation no longer has a hidden legacy scan
fallback for non-`delta_drop` objectives.

Rationale:
Making unsupported reverse objectives fail explicitly is preferable to keeping a
stale route whose diagnostics and active-boundary assumptions differ from the
sign-aware reverse path.

Open Risks:
If a real caller still needs reverse `delta_gain` or reverse `power_next`, it now
needs a deliberate policy builder rather than accidental scan semantics.

Verification:
`env -u PETSC_DIR -u PETSC_ARCH ./.venv_jit/bin/python
v6_active_boundary_reduced/validation/test_policy_behavior_guards.py` passed.
`env -u PETSC_DIR -u PETSC_ARCH ./.venv_jit/bin/python
v6_active_boundary_reduced/validation/test_freidberg_sign_aware_smoke.py`
passed.
`env -u PETSC_DIR -u PETSC_ARCH ./.venv_jit/bin/python
v6_active_boundary_reduced/validation/test_sonic_delta_profile.py` passed.

## 2026-06-12 Review Log - Sign-Aware G Boundary Recovery Order

Scope:
`policy.py::_reverse_sign_policy_step`, `policy.py::_sign_aware_scan_fallback`,
and `policy.py::_refine_boundary`.

Question:
The review questioned whether sign-aware endpoint validation failures should
scan the full reverse interval before attempting the true finite-step
`G_next = G_floor` boundary.

Before:
When the sign-aware affine endpoint failed finite-step validation,
`_reverse_sign_policy_step` immediately called `_sign_aware_scan_fallback`.
That fallback scanned the whole interval, picked a feasible candidate, and only
then called `_refine_boundary`, which uses `brentq` on the true finite-step
`G` margin if the infeasible neighbor is blocked by `G`.

Change:
No behavior change.  Added a `RISK` marker at the sign-aware fallback call site
recording that `G`-limited endpoint failures should try a sign-aware direct
finite-step `G` boundary brentq before falling back to full scan.

Current Behavior:
The true `G_next = G_floor` solve is currently a refinement after scan has
found a feasible/infeasible bracket.  If the scan finds no feasible point, no
`G` root is attempted.  This avoids accepting a meaningless marginal `G` point
when other constraints fail, but it also means a `G`-limited endpoint failure
does not get the most direct boundary recovery first.

Rationale:
For a sign-aware reverse interval whose affine endpoint is `G`-limited, the
natural recovery target is the real finite-step `G` boundary.  A future cleanup
should evaluate a safe-side bracket, solve `G(next_state(sigma)) - G_floor = 0`
with `brentq` when the bracket is valid, and only use scan as bracket discovery
or recovery for non-`G` blockers.

Open Risks:
The current scan-first fallback can be less precise and more expensive than a
direct `G` boundary recovery for `G`-limited endpoint failures.  A direct
recovery still must verify full finite-step feasibility because `G=G_floor` does
not guarantee residual, `T_p`, or stage constraints.

Verification:
Not yet run after this comment-only change.

## 2026-06-12 Review Log - Shared Sigma Interval Cleanup Marker

Scope:
`policy.py::_sigma_interval` and `reverse_sign_policy.py::build_reverse_sigma_interval`.

Question:
The review checked whether `_sigma_interval` is legacy-only and whether the new
approach still needs a sigma interval builder.

Before:
`_sigma_interval` was a shared helper for the legacy scan route and
sonic-compatible branch.  It mixed `direction` into the same area-bound algebra,
while the sign-aware reverse path used a separate `build_reverse_sigma_interval`
that also intersects the affine reverse `G` bound.

Change:
No behavior change.  Added a `REVIEW` marker to `_sigma_interval` recording that
it should remain temporarily until a forward sigma builder exists and the common
slope, curvature, and area-bound functionality can be shared with the reverse
builder.

Current Behavior:
`_sigma_interval` still supplies `sigma_box`, curvature, and log-area endpoint
bounds for legacy and sonic-compatible paths.  Sign-aware reverse `delta_drop`
continues to use `build_reverse_sigma_interval`, which carries reverse-specific
`G_lower/G_upper` bound semantics and richer interval diagnostics.

Rationale:
The duplicate builders are acceptable during transition because the reverse
builder has different sign-aware semantics.  The cleanup should first introduce
a forward builder, then factor the common geometry/slope interval machinery
under both forward and reverse builders.

Open Risks:
Until that cleanup happens, interval-source naming and direction conventions are
split across two implementations.

Verification:
Not yet run after this comment-only change.

## 2026-06-12 Review Log - Legacy Pedal Direction Marker

Scope:
`policy.py::_pedal_direction`, `policy.py::_legacy_policy_step`, and
`policy.py::_direct_boundary_choice`.

Question:
The review checked whether `_pedal_direction` only affects the legacy route and
whether it can be removed immediately.

Before:
`_pedal_direction` had a single caller in `_legacy_policy_step`, but that
legacy-only status was not documented at the function itself.

Change:
No behavior change.  Added a `REVIEW` marker to `_pedal_direction` noting that
it should be removed or replaced together with `_legacy_policy_step` and
`_direct_boundary_choice`.

Current Behavior:
`_pedal_direction` remains a local objective probe used only by the legacy scan
route.  The sign-aware reverse `delta_drop` path does not use it.  The legacy
route is still reachable for forward direction and non-`delta_drop` objectives,
so deleting `_pedal_direction` alone would break those paths.

Rationale:
The cleanup should be done as a grouped route migration or removal, not as an
isolated helper deletion.

Open Risks:
Legacy forward/non-delta objective callers still depend on this route until a
replacement or explicit unsupported-mode decision is made.

Verification:
Not yet run after this comment-only change.

## 2026-06-12 Review Log - RK4 Fixed-Sigma Diagnostics Marker

Scope:
`policy.py::_rk4_integrate_state` and `policy.py::_solve_next_state_rk4`.

Question:
The review questioned how RK4 substeps choose `sigma` and `logA`, and noted
that `residual_ok` / `max_abs_scaled_residual` are misleading names on the RK4
path.

Before:
The RK4 path used a constant candidate `sigma` through all substeps, but this
piecewise-constant area-slope assumption was not marked near the RK4 loop.  The
diagnostic payload reused solver-compatibility names even though the values were
the coarse-vs-fine RK4 step-doubling error, not a physical residual.

Change:
No behavior change.  Added review markers documenting that RK4 substeps hold
`sigma` fixed over the candidate segment, and that the compatibility residual
fields should be renamed or split during diagnostic cleanup.

Current Behavior:
For each candidate sigma, `_evaluate_sigma` sets the segment endpoint
`logA_next = current.logA + direction * dx * sigma`.  `_rk4_integrate_state`
then integrates `[log_n, log_Te, logA]` with `dlogA/dx = sigma` fixed across
substeps, so substeps sample the linear log-area path implied by that candidate
slope.  `_solve_next_state_rk4` compares coarse and fine integrations in
`log_n/log_Te` to produce `rk4_error_estimate`; the legacy
`max_abs_scaled_residual` field carries that same error estimate for payload
compatibility.

Rationale:
This RK4 route evaluates one finite-step candidate slope, not a per-substep
area reoptimization.  Making that assumption explicit reduces the chance that
stage diagnostics or endpoint forcing are mistaken for adaptive sigma selection.

Open Risks:
If the intended area model should vary sigma inside the segment, the current RK4
candidate evaluator is not doing that.  The diagnostic field names still risk
confusion until the payload is cleaned up.

Verification:
Not yet run after this comment-only change.

## 2026-06-12 Review Log - Evaluate Sigma Nonfinite Margins

Scope:
`policy.py::_evaluate_sigma` and `policy.py::_constraint_blockers`.

Question:
The review of lines 1872-1921 checked how a single candidate sigma is converted
into `ok`, `feasible`, constraint margins, and constraint violation.

Before:
`_evaluate_sigma` correctly made `nan` margins fail the `feasible` comparison,
but `constraint_violation` could itself become `nan` because it summed
`max(-value, 0.0)` over raw margin values.  Fallback paths rank failed
candidates by `constraint_violation`, so a nonfinite violation is a poor
diagnostic/ranking value.  `_constraint_blockers` also did not name nonfinite
margins as blockers.

Change:
Nonfinite margins are now explicitly infeasible, contribute infinite constraint
violation, and are named as blockers by `_constraint_blockers`.

Current Behavior:
A candidate with nonfinite `G`, `Tp`, residual, or stage margin cannot be
selected as feasible and cannot be preferred by lowest finite violation in
fallback ranking.

Rationale:
Failed-candidate diagnostics should be ordered by meaningful finite violations.
Nonfinite margins are not admissible physical margins and should be treated as
hard blockers.

Open Risks:
None noted.

Verification:
`python -m compileall -q v6_active_boundary_reduced/policy.py` passed.
`env -u PETSC_DIR -u PETSC_ARCH ./.venv_jit/bin/python
v6_active_boundary_reduced/validation/test_freidberg_sign_aware_smoke.py` passed.
`git diff --check -- v6_active_boundary_reduced/policy.py
v6_active_boundary_reduced/SIGN_AWARE_REVERSE_ACTIVE_BOUNDARY_REWRITE.md` passed.

## 2026-06-12 Review Log - RK4 Log-State RHS Comment

Scope:
`policy.py::_primitive_log_rhs_with_diagnostics` and
`policy.py::_empty_rk4_stage_summary`.

Question:
The review asked what `log_matrix = matrix @ diag([n, Te])` means and whether
the stage solve method/rank diagnostics survive into segment-level summaries.

Before:
The code solved raw, log-column-scaled, or nondimensionalized RHS systems, but
the log-state column scaling was not documented at the point where it appears.
The per-stage `rk4_stage_linear_solve_method` and
`rk4_stage_linear_solve_rank` fields were emitted by the stage RHS diagnostic,
but the summary accumulator did not aggregate them.

Change:
No behavior change.  Added a comment explaining that column scaling rewrites
`M*[n', Te'] = rhs` as
`(M*diag(n, Te))*[dlogn/dx, dlogTe/dx] = rhs`.  Added a marker noting that
stage solve method/rank should be aggregated if least-squares fallback frequency
becomes important.

Current Behavior:
RK4 can solve the primitive RHS in raw, log-column-scaled, or row-normalized log
form.  Segment-level diagnostics still aggregate matrix conditioning,
singularity, replay residual, and RHS magnitude, but not solve-method/rank
counts.

Rationale:
The log-state matrix is a chain-rule rewrite, not a derivative of the primitive
coefficients.  The solve-method/rank marker prevents the `lstsq` fallback signal
from being forgotten during future diagnostic cleanup.

Open Risks:
If `lstsq` fallback frequency matters operationally, current segment summaries
do not expose it directly.

Verification:
`python -m compileall -q v6_active_boundary_reduced/policy.py` passed, and
`git diff --check -- v6_active_boundary_reduced/policy.py
v6_active_boundary_reduced/SIGN_AWARE_REVERSE_ACTIVE_BOUNDARY_REWRITE.md`
passed.

## 2026-06-12 Review Log - Direct G Boundary Feasibility Guard

Scope:
`policy.py::_direct_boundary_choice`, the legacy direct endpoint path in
`policy.py::_legacy_policy_step`, and `policy.py::_refine_boundary`.

Question:
The review continued into `_direct_boundary_choice` and compared it with
`_refine_boundary`, because both can use `brentq` to recover a `G=0` boundary.

Before:
`_direct_boundary_choice` returned the brentq root after solving only the `G`
margin sign change.  Unlike `_refine_boundary`, it did not verify that the root
candidate was still fully finite-step feasible.  In the legacy caller, direct
payload `ok` was based on solver `ok`, so a root that fixed `G` but failed
another margin could be reported too optimistically.

Change:
After evaluating the brentq root, `_direct_boundary_choice` now returns `None`
unless the root candidate is fully `feasible`.  If the direct shortcut cannot
produce a feasible root, legacy scan/backtrack remains responsible for recovery.

Current Behavior:
The direct shortcut still accepts a feasible preferred endpoint, a feasible
opposite endpoint exactly on the boundary, or a fully feasible `G=0` root.  It no
longer returns a candidate that only solves `G` while leaving other constraints
failed.

Rationale:
`G=0` is only one active-boundary condition.  A direct root should not bypass the
same finite-step feasibility gate used by scan/backtrack refinement.

Open Risks:
This is still part of the legacy route, which is marked for deletion after
callers are migrated to sign-aware semantics.  `_direct_boundary_choice` is only
called by `_legacy_policy_step`, so it should be removed with that legacy route
rather than extended as sign-aware behavior.

Verification:
`python -m compileall -q v6_active_boundary_reduced/policy.py` passed.
`env -u PETSC_DIR -u PETSC_ARCH ./.venv_jit/bin/python
v6_active_boundary_reduced/validation/test_freidberg_sign_aware_smoke.py` passed.
`git diff --check -- v6_active_boundary_reduced/policy.py
v6_active_boundary_reduced/SIGN_AWARE_REVERSE_ACTIVE_BOUNDARY_REWRITE.md` passed.

## 2026-06-12 Review Log - Enthalpy Objective Marker

Scope:
`policy.py::_normalized_objective` and `policy.py::_step_objective_payload`.

Question:
The review noted that the current objective set only covers `delta_gain`,
`delta_drop`, and `power_next`, but future reverse preparation may need a score
that prefers the upstream step with the slowest enthalpy rise.

Before:
Objective normalization only accepts Delta and power aliases.

Change:
No behavior change.  Added a `REVIEW` marker next to the objective alias table to
record the future enthalpy objective.

Current Behavior:
The accepted objective names remain unchanged.  The marker only records the next
scoring extension target.

Rationale:
If reverse preparation is judged by avoiding excessive upstream thermodynamic
growth, the local objective should be able to score enthalpy change directly
instead of using only Delta or power proxies.

Open Risks:
The enthalpy definition and sign convention still need to be specified before
implementation, especially whether the score should minimize upstream total
enthalpy rise, static thermal enthalpy rise, or a closure-derived proxy.

Verification:
`python -m compileall -q v6_active_boundary_reduced/policy.py` passed, and
`git diff --check -- v6_active_boundary_reduced/policy.py
v6_active_boundary_reduced/SIGN_AWARE_REVERSE_ACTIVE_BOUNDARY_REWRITE.md`
passed.
