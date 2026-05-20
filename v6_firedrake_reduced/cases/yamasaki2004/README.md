# Yamasaki 2004 Case Data

This folder is the local source of truth for Yamasaki 2004 case data used by
`v6_firedrake_reduced`.

`parameters.py` intentionally separates two categories:

- `Yamasaki2004PaperParameters`: values reported by, or directly derived from,
  the paper text, including geometry endpoints, magnetic field, operating
  ranges, and reported performance diagnostics.
- `Yamasaki2004ModelSeed`: solver initialization windows for quantities the
  paper does not directly report, such as inlet plasma density, electron
  temperature, ionization level, and area-control neighborhood.

Generated JSON summaries under `outputs/` are artifacts, not the canonical
parameter source.
