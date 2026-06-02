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
The existing implicit residual solve can be reused, but the selected sigma must
come from the sign-aware reverse interval logic.

9. Validate the finite-`dx` result:

```text
G_upstream >= G_floor
T_p_upstream >= T_p_floor
residual <= residual_tol
```

If validation fails, refine or backtrack against the actual implicit step.  The
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
