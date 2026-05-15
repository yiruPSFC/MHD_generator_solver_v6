# v6_maingo_mach_spline

This is a shadow prototype for replacing the reduced-space area spline with a
Mach-number spline.

The intended coordinate change is:

```text
current:  inlet variables + log-area spline coefficients
prototype: inlet variables + log-Mach-ratio spline coefficients
```

For fixed `dot_N`, `I_0`, `seed_fraction`, and `B`, the local Hall factor is a
function of `n_p` and `T_e`.  Prescribing Mach therefore gives an explicit local
reconstruction:

```text
T_p = 9 T_e / (9 + 5 M^2 F)
v_p = sqrt(5 k T_p M^2 / (3 m_p))
A = dot_N / (n_p v_p)
```

This keeps the reduced decision dimension comparable to the current
MAiNGO-CasADi flow while moving sonic behavior into the Mach chart instead of
forcing it through an area chart.

Current scope:

- fit a 3-parameter log-Mach spline from existing profiles;
- reconstruct primitive variables from `n_p`, `T_e`, and Mach without a root
  solve;
- compare exact-Mach and fitted-Mach reconstruction against the stored 41% and
  147% artifacts;
- reuse Freidberg interval-defect diagnostics as a physical audit.
- run a reduced fixed-Newton shadow rollout with the current 8D outer structure,
  replacing `a1/a2/a3` with `m1/m2/m3`.
- expose that reduced Mach rollout through a MAiNGO-style model adapter and the
  explicit `--coarse-model mach_spline` workflow branch.
- expose a faster `--coarse-model mach_spline_rk4_soft` candidate generator that
  removes fixed-Newton equality residuals, uses RK4 with a finite-difference
  chain-rule `A(n_p,T_e,M(x))` RHS, and records derived-sigma /
  Freidberg-defect diagnostics as post-hoc physical acceptance checks.

The first reduced rollout uses backward-difference derived geometry:

```text
A_i = A(n_i, T_e_i, M_i)
sigma_i = (A_i - A_{i-1}) / (dx A_i)
```

This mirrors the current reduced-implicit workflow while making area a derived
quantity instead of an outer decision spline.

CLI smoke entrypoint:

```bash
./.venv_jit/bin/python -m v6_maingo_casadi.run_hybrid_maingo_casadi \
  --coarse-model mach_spline \
  --mach-reference-profile path/to/maingo_best_profile.npz \
  --skip-casadi-handoff
```

Fast candidate-generator smoke entrypoint:

```bash
./.venv_jit/bin/python -m v6_maingo_casadi.run_hybrid_maingo_casadi \
  --coarse-model mach_spline_rk4_soft \
  --mach-reference-profile path/to/maingo_best_profile.npz \
  --skip-casadi-handoff
```

If `--mach-reference-profile` is omitted, the workflow tries to infer a profile
from `--initial-solution-json` by checking the same output directory for
`maingo_best_profile.npz`, `maingo_handoff_profile.npz`, or
`maingo_coarse_profile.npz`.
