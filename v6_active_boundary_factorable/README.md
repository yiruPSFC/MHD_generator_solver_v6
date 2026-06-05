# v6_active_boundary_factorable

Prototype for a reduced-space MAiNGO-style active-boundary rollout.

The intent is to test whether the active-boundary idea can be rewritten as a
fixed factorable expression while keeping only a small external design vector.
This is not wired into the production MAiNGO workflow.

The prototype uses:

- a soft endpoint selector instead of variable-dependent `if p1 > 0`;
- explicit Euler/RK-style state updates instead of fixed Newton correction;
- a 2x2 left-null sonic chart instead of SVD;
- a continuous sonic gate that blends ordinary greedy sigma with sonic-compatible sigma;
- diagnostic outputs for det, Mach, sigma bound margins, and sonic compatibility.

Known limitations:

- it is a soft surrogate for the hard greedy policy, not an exact transcription;
- `G` is currently diagnosed after the step, not used as an active sigma-bound source;
- the sonic gate can blend a compatible sonic sigma with a nonsonic greedy sigma, so
  compatibility is only exact when the gate is close to one;
- the regularized inverse avoids det division blow-up but can bias the RHS near choking.

Run the focused smoke tests with:

```bash
./.venv_jit/bin/python v6_active_boundary_factorable/validation/test_soft_greedy_rk.py
```
