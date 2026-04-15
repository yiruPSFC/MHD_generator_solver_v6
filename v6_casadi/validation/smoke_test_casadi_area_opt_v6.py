from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

REPO_DIR = Path(__file__).resolve().parents[2]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from v6_casadi.optimize_area_profile_casadi_v6 import (
    _make_stage_function,
    _prepare_inlet_constants,
    optimize_area_profile,
)


def main() -> int:
    inlet = _prepare_inlet_constants(
        n_p_in=2.9418007053766115e21,
        Z_in=56.32891972495766,
        T_p_in=1000.0,
        T_e_in=5727.0800768822055,
        A_in=1.0,
        B=0.02,
        seed_fraction=None,
    )
    stage = _make_stage_function(
        dot_N=inlet.dot_N,
        I_0=inlet.I_0,
        seed_fraction=inlet.seed_fraction,
        B=0.02,
    )

    stage_cases = [
        np.array([2.9418007053766115e21, 100.0, 1.0], dtype=float),
        np.array([2.9418007053766115e21, 5727.0800768822055, 1.0e-6], dtype=float),
    ]
    failures: list[str] = []
    for idx, state in enumerate(stage_cases):
        out = np.asarray(stage(state, 0.0)).reshape(-1)
        if not np.all(np.isfinite(out)):
            failures.append(f"stage case {idx} returned non-finite outputs")

    result = optimize_area_profile(
        n_p_in=2.9418007053766115e21,
        Z_in=56.32891972495766,
        T_p_in=1000.0,
        T_e_in=5727.0800768822055,
        A_in=1.0,
        B=0.02,
        length=1e-8,
        n_intervals=4,
        A_min_ratio=0.95,
        A_max_ratio=1.05,
        max_abs_dlogA_dx=1e7,
        ipopt_max_iter=200,
    )

    if not result.success:
        failures.append(f"optimizer did not report success: {result.return_status}")
    if not result.acceptable:
        failures.append(f"optimizer iterate not acceptable: diagnostics={result.diagnostics}")
    if not np.all(np.isfinite(result.T_e)):
        failures.append("non-finite T_e in optimized profile")
    if not np.all(np.isfinite(result.A)):
        failures.append("non-finite A in optimized profile")
    if float(np.nanmin(result.velikhov_margin)) < -1e-5:
        failures.append(f"Velikhov constraint violated: min G={float(np.nanmin(result.velikhov_margin)):.3e}")
    if result.objective_delta_Te <= 0.0:
        failures.append(f"non-positive outlet heating objective: dTe={result.objective_delta_Te:.3e}")

    if failures:
        print("SMOKE TEST FAILED")
        for item in failures:
            print(item)
        return 1

    print("SMOKE TEST PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
