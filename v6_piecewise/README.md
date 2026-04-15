# v6_piecewise

Experimental piecewise-current prototype workflows.

This folder is intentionally separate from [`v6_casadi`](../v6_casadi), because it is a different modeling problem:

- upstream passive nozzle proxy
- activation / current turn-on bridge
- downstream active segment solved with the CasADi continuation machinery

Current status:

- this line is closer to the intended physical device layout
- the formulation is not yet stable enough to treat as a production solver
- the downstream active segment can produce acceptable profiles
- the activation bridge has not been solved robustly yet, so the full piecewise workflow should still be treated as a research prototype

Practical rule:

- use [`v6_casadi`](../v6_casadi) for stable active-segment studies and regression runs
- use `v6_piecewise` only for restructuring / redesign work on the higher-fidelity device model
