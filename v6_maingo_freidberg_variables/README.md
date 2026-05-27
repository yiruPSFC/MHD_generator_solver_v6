# v6_maingo_freidberg_variables

This package is an isolated first step toward a Freidberg-variable solver.  It
does not modify the existing `v6_maingo_casadi` workflow.

Current scope:

- map stored primitive profiles into Freidberg slide-38 variables `H_p` and
  `L_p`;
- reconstruct primitive variables from `(H_p, L_p, T_e)` using the same local
  algebraic closure as the current solver;
- audit the Freidberg slide-39 `dH_p/dx` and `dL_p/dx` conservation balances;
- regression-test both the plausible 41% profile and the nonphysical 147%
  profile before any optimization rewrite.
- evaluate direct-transcription interval defects for Freidberg's `dH_p/dx`
  and `dL_p/dx` equations.

The important bridge contract is:

```text
old primitive profile -> H_p/L_p/T_e -> reconstructed primitive profile
```

This must pass for old artifacts before a new optimizer is allowed to use
`H_p` and `L_p` as trajectory variables.

The next contract is interval-level, not only endpoint-level:

```text
H_p[i+1] - H_p[i] = trapezoid(RHS_H[i], RHS_H[i+1])
L_p[i+1] - L_p[i] = trapezoid(RHS_L[i], RHS_L[i+1])
```

The 41% profile is the positive regression case for this contract.  The 147%
profile is the negative regression case: it remains locally recoverable but
has large Freidberg interval and terminal defects.
