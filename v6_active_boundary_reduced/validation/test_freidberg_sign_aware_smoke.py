"""Historical Freidberg x~=0 sign-aware diagnostic.

This file was written to explain why behavior near the inlet looked surprising:
forward policy tends to push the objective aggressively, while reverse
preparation often stays on the G boundary. That asymmetry made the x~=0
Freidberg case useful as a debugging smoke test, but it is not a representative
default regression case for the active-boundary solver.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from v6_firedrake_reduced.cases.freidberg_reference import load_reference_profile
from v6_firedrake_reduced.design import load_case_config

from v6_active_boundary_reduced.core.local_affine import compute_forward_affine_coefficients
from v6_active_boundary_reduced.core.policy import (
    PreparationSettings,
    _physics_params,
    anchor_from_profile,
    recover_preparation_profile,
)
from v6_active_boundary_reduced.core.reverse_sign_policy import reverse_coefficients_from_forward


def _freidberg_anchor_near_x(x_target: float = 0.01):
    config = load_case_config(case="freidberg_reference")
    profile = load_reference_profile()
    index = int(np.argmin(np.abs(np.asarray(profile["x"], dtype=float) - float(x_target))))
    anchor = anchor_from_profile(profile, index=index, config=config, source="freidberg_sign_aware_smoke")
    return config, profile, index, anchor


def test_freidberg_near_x001_reverse_signs() -> None:
    config, _, _, anchor = _freidberg_anchor_near_x()
    params = _physics_params(config)
    coeffs = compute_forward_affine_coefficients(
        n_p=anchor.state.n_p,
        T_e=anchor.state.T_e,
        A=anchor.state.area(config),
        logA=anchor.state.logA,
        params=params,
    )
    reverse = reverse_coefficients_from_forward(
        dx=0.01,
        G_current=coeffs.G_current,
        G_floor=0.0,
        a0=coeffs.a0,
        a1=coeffs.a1,
        b0=coeffs.b0,
        b1=coeffs.b1,
    )

    assert coeffs.a1 > 0.0
    assert coeffs.b1 > 0.0
    assert reverse.p1 > 0.0
    assert reverse.q1 < 0.0
    assert reverse.p1q1 < 0.0


def test_freidberg_near_x001_reverse_rollout_smoke() -> None:
    config, profile, index, anchor = _freidberg_anchor_near_x()
    assert abs(float(profile["x"][index]) - 0.01) <= 0.01

    settings = PreparationSettings(
        n_steps=1,
        dx=0.01,
        scan_points=11,
        refine_iterations=8,
    )
    payload = recover_preparation_profile(config=config, anchor=anchor, settings=settings)
    assert payload["ok"]
    assert len(payload["segments"]) == 1

    segment = payload["segments"][0]
    for field in (
        "a0",
        "a1",
        "b0",
        "b1",
        "p0",
        "p1",
        "q0",
        "q1",
        "p1q1_reverse",
        "reverse_G_bound_kind",
        "selected_support_type",
        "selected_sigma_origin",
        "selected_sigma_source",
        "affine_selected_endpoint_source",
        "affine_support_type",
        "selected_endpoint_source",
        "objective_bound_kind",
        "sign_aware_fallback_used",
        "sign_aware_fallback_recovered",
        "validation_status",
        "step_error_kind",
        "rk4_error_ok",
        "rk4_error_estimate",
        "rk4_error_margin",
        "physical_residual_scaled",
    ):
        assert field in segment

    assert segment["a1"] > 0.0
    assert segment["b1"] > 0.0
    assert segment["p1"] > 0.0
    assert segment["q1"] < 0.0
    assert segment["p1q1_reverse"] < 0.0
    assert segment["reverse_G_bound_kind"] == "upper"
    assert segment["selected_support_type"] == "G_supported"
    assert segment["selected_sigma_origin"] == "g_boundary_fallback"
    assert segment["selected_sigma_source"] == "sign_aware_brentq_G_boundary_fallback"
    assert segment["affine_support_type"] == "G_limited_reverse"
    assert segment["affine_selected_endpoint_source"] == "G_upper"
    assert segment["selected_endpoint_source"] == "G_upper"
    assert segment["support_type"] == "G_supported"
    assert segment["sign_aware_fallback_used"]
    assert segment["sign_aware_fallback_recovered"]
    assert segment["sign_aware_endpoint_validation_failure"] == "G"
    assert segment["sign_aware_fallback_solver_method"] == "sign_aware_brentq_G_boundary_fallback"
    assert segment["boundary_blockers"] == ["G"]
    assert "scan_diagnostics" not in segment
    assert segment["step_error_kind"] == "rk4_step_doubling"
    assert segment["rk4_error_ok"]
    assert float(segment["rk4_error_margin"]) >= -float(settings.active_tol)
    assert np.isnan(float(segment["physical_residual_scaled"]))
    assert segment["validation_status"] == "ok"


if __name__ == "__main__":
    test_freidberg_near_x001_reverse_signs()
    test_freidberg_near_x001_reverse_rollout_smoke()
    print("freidberg sign-aware active-boundary smoke passed")
