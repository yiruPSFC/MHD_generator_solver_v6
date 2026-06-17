"""Synthetic policy-control guards, not broad physics validation.

Historically this file protected fallback routing while sign-aware reverse
policy was being split out of the older scan path. The tests use fake
``_evaluate_sigma`` data, so they are useful for preserving branch semantics
and diagnostics names, but they should not be treated as default physical
correctness cases.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import v6_active_boundary_reduced.core.policy as policy
from v6_active_boundary_reduced.core.policy import PolicySettings


def _fake_eval_for_g_boundary(*, sigma: float, **_kwargs):
    g_margin = 0.5 - float(sigma)
    feasible = g_margin >= -1.0e-12
    return {
        "ok": True,
        "feasible": feasible,
        "sigma": float(sigma),
        "next_state": None,
        "objective_kind": "delta_drop",
        "objective_value": -float(sigma),
        "delta_gain": float(sigma),
        "constraint_margins": {"G": g_margin, "Tp": 1.0, "residual": 1.0},
        "constraint_violation": max(-g_margin, 0.0),
        "max_abs_scaled_residual": 0.0,
        "rk4_error_estimate": 0.0,
    }


def test_endpoint_brentq_g_boundary_fallback_runs_before_scan() -> None:
    original = policy._evaluate_sigma
    try:
        policy._evaluate_sigma = _fake_eval_for_g_boundary  # type: ignore[assignment]
        endpoint = _fake_eval_for_g_boundary(sigma=1.0)
        endpoint = {**endpoint, "solver_method": "sign_aware_reverse_endpoint"}
        result = policy._sign_aware_g_boundary_fallback(
            current=None,  # type: ignore[arg-type]
            interval=SimpleNamespace(sigma_lo=0.0, sigma_hi=1.0),
            endpoint=endpoint,
            dx=1.0,
            config=None,  # type: ignore[arg-type]
            settings=PolicySettings(refine_iterations=12),
        )
    finally:
        policy._evaluate_sigma = original  # type: ignore[assignment]

    assert result is not None
    assert result["solver_method"] == "sign_aware_brentq_G_boundary_fallback"
    assert abs(float(result["sigma"]) - 0.5) <= 1.0e-10
    assert result["boundary_blockers"] == ["G"]
    assert bool(result["feasible"])


def test_affine_expand_then_endpoint_brentq_uses_predicted_bound() -> None:
    original = policy._evaluate_sigma
    try:
        policy._evaluate_sigma = _fake_eval_for_g_boundary  # type: ignore[assignment]
        endpoint = _fake_eval_for_g_boundary(sigma=1.0)
        endpoint = {**endpoint, "solver_method": "sign_aware_reverse_endpoint"}
        result = policy._sign_aware_g_boundary_fallback(
            current=None,  # type: ignore[arg-type]
            interval=SimpleNamespace(
                sigma_lo=0.0,
                sigma_hi=1.0,
                lower_source="slope_min",
                upper_source="G_upper",
                reverse_G_bound_kind="upper",
                sigma_G_bound=0.5,
            ),
            endpoint=endpoint,
            dx=1.0,
            config=None,  # type: ignore[arg-type]
            settings=PolicySettings(
                refine_iterations=12,
                g_boundary_fallback_mode="affine_expand_then_endpoint_brentq",
            ),
        )
    finally:
        policy._evaluate_sigma = original  # type: ignore[assignment]

    assert result is not None
    assert result["solver_method"] == "sign_aware_affine_expand_G_boundary_fallback"
    assert abs(float(result["sigma"]) - 0.5) <= 1.0e-10
    assert result["boundary_blockers"] == ["G"]
    assert result["affine_expand_reverse_G_bound_kind"] == "upper"
    assert bool(result["feasible"])


def test_affine_expand_then_endpoint_brentq_falls_back_when_bound_unusable() -> None:
    original = policy._evaluate_sigma
    try:
        policy._evaluate_sigma = _fake_eval_for_g_boundary  # type: ignore[assignment]
        endpoint = _fake_eval_for_g_boundary(sigma=1.0)
        endpoint = {**endpoint, "solver_method": "sign_aware_reverse_endpoint"}
        result = policy._sign_aware_g_boundary_fallback(
            current=None,  # type: ignore[arg-type]
            interval=SimpleNamespace(
                sigma_lo=0.0,
                sigma_hi=1.0,
                lower_source="slope_min",
                upper_source="slope_max",
                reverse_G_bound_kind="upper",
                sigma_G_bound=2.0,
            ),
            endpoint=endpoint,
            dx=1.0,
            config=None,  # type: ignore[arg-type]
            settings=PolicySettings(
                refine_iterations=12,
                g_boundary_fallback_mode="affine_expand_then_endpoint_brentq",
            ),
        )
    finally:
        policy._evaluate_sigma = original  # type: ignore[assignment]

    assert result is not None
    assert result["solver_method"] == "sign_aware_brentq_G_boundary_fallback"
    assert abs(float(result["sigma"]) - 0.5) <= 1.0e-10
    assert bool(result["feasible"])


def test_scan_fallback_refines_g_boundary_when_bracketed() -> None:
    original = policy._evaluate_sigma
    try:
        policy._evaluate_sigma = _fake_eval_for_g_boundary  # type: ignore[assignment]
        result = policy._sign_aware_scan_fallback(
            current=None,  # type: ignore[arg-type]
            interval=SimpleNamespace(sigma_lo=0.0, sigma_hi=1.0),
            p1=1.0,
            dx=1.0,
            config=None,  # type: ignore[arg-type]
            settings=PolicySettings(scan_points=5, refine_iterations=12),
        )
    finally:
        policy._evaluate_sigma = original  # type: ignore[assignment]

    assert result is not None
    assert result["solver_method"] == "brentq_G_boundary_refine"
    assert abs(float(result["sigma"]) - 0.5) <= 1.0e-10
    assert result["boundary_blockers"] == ["G"]
    assert bool(result["feasible"])


def test_rk4_stage_gate_margin_signs() -> None:
    settings = PolicySettings(
        active_tol=1.0e-6,
        tp_floor_K=300.0,
        g_floor=0.0,
        rk4_stage_cond_max=100.0,
        rk4_stage_mach_max=1.0,
        rk4_stage_replay_tol=1.0e-4,
    )
    margins = policy._rk4_stage_gate_margins(
        summary={
            "rk4_stage_min_Tp_K": 290.0,
            "rk4_stage_min_G": -0.2,
            "rk4_stage_max_cond_row_norm_log": 120.0,
            "rk4_stage_max_mach": 1.1,
            "rk4_stage_max_differential_replay_residual": 2.0e-4,
        },
        settings=settings,
    )

    assert margins["stage_Tp"] < 0.0
    assert margins["stage_G"] < 0.0
    assert margins["stage_cond"] < 0.0
    assert margins["stage_mach"] < 0.0
    assert margins["stage_replay"] < 0.0


def test_reverse_non_delta_objective_is_explicitly_unsupported() -> None:
    result = policy._policy_step(
        current=policy.State(log_n=0.0, log_Te=0.0, logA=0.0),
        sigma_prev=None,
        sigma_warm_start=None,
        dx=1.0,
        direction=-1,
        config=None,  # type: ignore[arg-type]
        settings=PolicySettings(direction="reverse", objective="delta_gain"),
    )

    assert not bool(result["ok"])
    assert result["support_type"] == "unsupported_reverse_objective"
    assert result["termination_reason"] == "unsupported_reverse_objective"


if __name__ == "__main__":
    test_endpoint_brentq_g_boundary_fallback_runs_before_scan()
    test_affine_expand_then_endpoint_brentq_uses_predicted_bound()
    test_affine_expand_then_endpoint_brentq_falls_back_when_bound_unusable()
    test_scan_fallback_refines_g_boundary_when_bracketed()
    test_rk4_stage_gate_margin_signs()
    test_reverse_non_delta_objective_is_explicitly_unsupported()
    print("policy behavior guards passed")
