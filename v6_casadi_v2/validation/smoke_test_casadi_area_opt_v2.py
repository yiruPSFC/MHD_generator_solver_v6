#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

REPO_DIR = Path(__file__).resolve().parents[2]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from v6_casadi_v2.optimize_area_profile_casadi_v2 import optimize_area_profile


def main() -> int:
    result = optimize_area_profile(
        n_p_in_guess=3.05e25,
        n_p_in_min=2.9e25,
        n_p_in_max=3.2e25,
        T_e_in_guess=4420.0,
        T_e_in_min=4300.0,
        T_e_in_max=4550.0,
        Z_in_guess=75.954994,
        Z_in_min=72.0,
        Z_in_max=80.0,
        J_x_in_guess=32000.0,
        J_x_in_min=1000.0,
        J_x_in_max=100000.0,
        seed_fraction_guess=1e-4,
        seed_fraction_min=1e-7,
        seed_fraction_max=2e-3,
        B=10.2,
        length=2.0,
        n_intervals=16,
        transcription="hermite-simpson",
        min_margin=0.0,
        A_min_ratio=0.9,
        A_max_ratio=2.2,
        max_abs_dlogA_dx=0.25,
        np_min_ratio=1e-4,
        np_max_ratio=30.0,
        te_min=100.0,
        te_max_ratio=8.0,
        tp_min=1.0,
        mach_min=None,
        margin_slack_max=0.0,
        margin_slack_weight=0.0,
        smooth_weight=0.01,
        control_slew_weight=0.2,
        control_curvature_weight=0.0,
        state_curvature_weight=0.0,
        warm_profile_track_weight=5.0,
        warm_control_track_weight=2.0,
        ipopt_max_iter=500,
        ipopt_tol=1e-6,
        objective_weight=0.0,
    )

    assert result.x.size == 17
    assert result.sigma_logA.size == 16
    assert np.all(np.isfinite(result.x))
    assert np.all(np.isfinite(result.n_p))
    assert np.all(np.isfinite(result.T_e))
    assert np.all(np.isfinite(result.A))
    assert np.isfinite(result.inlet.n_p)
    assert np.isfinite(result.inlet.T_e)
    assert np.isfinite(result.inlet.T_p)
    assert np.isfinite(result.inlet.Z)
    assert np.isfinite(result.inlet.J_x)
    assert np.isfinite(result.inlet.seed_fraction)
    assert bool(result.acceptable)
    assert abs(float(result.A[0]) - 1.0) < 1e-8
    assert "inlet_design" in result.diagnostics

    payload = {
        "success": bool(result.success),
        "acceptable": bool(result.acceptable),
        "return_status": str(result.return_status),
        "inlet_design": result.diagnostics["inlet_design"],
        "dynamic_defect_inf": float(result.diagnostics["dynamic_defect_inf"]),
        "max_constraint_violation": float(result.diagnostics["max_constraint_violation"]),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
