# Yamasaki 2004 Case

This folder contains the case-specific benchmark mapping for the Murakami,
Okuno, and Yamasaki 2004 CCMHD paper. It is intentionally separate from the
generic MAiNGO/CasADi engine in the package root.

- `parameters.py` holds the paper values and the model-neighborhood seed.
- `build_seed.py` projects the paper geometry into the package's direct
  3-parameter area spline, then writes warm-profile and summary artifacts.
- `geometry.py` preserves the old geometry-only import surface.

New seed artifacts default to:

```text
v6_maingo_casadi/outputs/cases/yamasaki2004/seeds/
```

Older milestone artifacts are still under:

```text
v6_maingo_casadi/outputs/maingo_yamasaki2004_neighborhood/
```

Those old output folders are left in place to preserve their recorded paths.
