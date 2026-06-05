from __future__ import annotations

import numpy as np

from v6_active_boundary_reduced.outer_solver.prescreen import (
    PrescreenSettings,
    from_normalized,
    passes_prescreen,
    prescreen_metrics,
    to_normalized,
)
from v6_active_boundary_reduced.outer_solver.reward import OuterRewardWeights, score_outer_result


def _sample_result(*, ok: bool = True) -> dict:
    nodes = [
        {
            "x": 0.0,
            "n_p": 3.0e25,
            "T_e": 5000.0,
            "T_p": 4500.0,
            "A": 0.45,
            "logA": 0.0,
            "sigma_logA": 0.0,
            "G": 2.0e5,
            "Delta": 5000.0 / 4500.0 - 1.0,
            "mach": 0.4,
            "beta": 1.0,
            "Z": 80.0,
        },
        {
            "x": -0.01,
            "n_p": 3.1e25,
            "T_e": 4970.0,
            "T_p": 4520.0,
            "A": 0.47,
            "logA": 0.04,
            "sigma_logA": -0.1,
            "G": 1.5e5,
            "Delta": 4970.0 / 4520.0 - 1.0,
            "mach": 0.42,
            "beta": 1.0,
            "Z": 79.0,
        },
    ]
    return {
        "ok": ok,
        "design": {"B_T": 12.0},
        "objective_terms": {"delta_improvement": 0.012},
        "active_summary": {
            "n_steps_requested": 1,
            "n_steps_completed": 1,
            "Tp_min_K": 4500.0,
            "Te_max_K": 5000.0,
            "G_min_excluding_anchor": 1.5e5,
            "mach_max": 0.42,
        },
        "payload": {"nodes": nodes, "segments": [{"ok": True}]},
    }


def test_outer_reward_penalizes_failed_rollout() -> None:
    weights = OuterRewardWeights(failure_penalty=100.0)
    ok_score = score_outer_result(_sample_result(ok=True), weights=weights)["score"]
    failed_score = score_outer_result(_sample_result(ok=False), weights=weights)["score"]

    assert failed_score <= -100.0
    assert failed_score < ok_score


def test_outer_reward_penalizes_temperature_and_area_limits() -> None:
    result = _sample_result(ok=True)
    result["active_summary"]["Tp_min_K"] = 1000.0
    result["active_summary"]["Te_max_K"] = 20000.0
    result["payload"]["nodes"][1]["A"] = 100.0

    scored = score_outer_result(
        result,
        weights=OuterRewardWeights(
            min_tp_floor_K=3000.0,
            max_te_ceiling_K=10000.0,
            area_ratio_max=10.0,
        ),
    )

    assert scored["terms"]["min_tp_shortfall_penalty"] > 0.0
    assert scored["terms"]["max_te_excess_penalty"] > 0.0
    assert scored["terms"]["area_ratio_penalty"] > 0.0


def test_prescreen_metrics_detect_low_ratio_fast_growth_state() -> None:
    result = _sample_result(ok=True)
    metrics = prescreen_metrics(result)

    assert metrics["anchor_G"] > 0.0
    assert metrics["anchor_te_over_tp"] < 1.2
    assert metrics["te_over_tp_gradient"] > 0.0
    assert passes_prescreen(
        metrics,
        settings=PrescreenSettings(
            te_over_tp_ceiling=1.2,
            min_te_over_tp_gradient=0.0,
            g_floor=0.0,
        ),
    )


def test_normalized_variable_roundtrip() -> None:
    lower = np.array([1.0, 10.0, -5.0], dtype=float)
    upper = np.array([3.0, 20.0, 5.0], dtype=float)
    values = np.array([2.0, 13.0, -1.0], dtype=float)

    normalized = to_normalized(values, lower=lower, upper=upper)
    recovered = from_normalized(normalized, lower=lower, upper=upper)

    assert np.allclose(recovered, values)
