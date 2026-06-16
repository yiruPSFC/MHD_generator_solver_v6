from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class OuterRewardWeights:
    delta_improvement: float = 1.0
    mhd_output_power_MW: float = 0.0
    min_tp_floor_K: float = 3000.0
    min_tp_shortfall: float = 1.0
    max_te_ceiling_K: float = 10000.0
    max_te_excess: float = 1.0
    temperature_scale_K: float = 1000.0
    area_ratio_min: float = 1.0
    area_ratio_max: float = 25.0
    area_ratio_penalty: float = 0.25
    magnetic_field_min_T: float = 1.0
    magnetic_field_max_T: float = 20.0
    magnetic_field_penalty: float = 1.0
    g_floor: float = 0.0
    g_shortfall: float = 1.0
    g_scale: float = 1.0e5
    mach_ceiling: float = 0.98
    mach_excess: float = 10.0
    incomplete_rollout: float = 100.0
    failure_penalty: float = 1.0e6


def _finite(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return out if np.isfinite(out) else float(default)


def _nodes_from_result(result: dict[str, Any]) -> list[dict[str, Any]]:
    payload = dict(result.get("payload", {}) or {})
    nodes = payload.get("nodes", [])
    if isinstance(nodes, list) and nodes:
        return [dict(node) for node in nodes]
    fallback = []
    for key in ("outlet", "preparation_inlet"):
        node = dict(result.get(key, {}) or {})
        if node:
            fallback.append(node)
    return fallback


def _profile_stat(nodes: list[dict[str, Any]], name: str, reducer, default: float = float("nan")) -> float:
    values = [_finite(node.get(name)) for node in nodes]
    finite = [value for value in values if np.isfinite(value)]
    if not finite:
        return float(default)
    return float(reducer(finite))


def _area_ratio(nodes: list[dict[str, Any]]) -> float:
    a_min = _profile_stat(nodes, "A", min)
    a_max = _profile_stat(nodes, "A", max)
    if not np.isfinite(a_min) or not np.isfinite(a_max) or a_min <= 0.0:
        return float("nan")
    return float(a_max / a_min)


def _soft_square(shortfall: float, scale: float) -> float:
    denom = max(float(scale), 1e-300)
    return float(max(float(shortfall), 0.0) / denom) ** 2


def score_outer_result(result: dict[str, Any], *, weights: OuterRewardWeights) -> dict[str, Any]:
    """Score one active-boundary rollout for the low-dimensional outer optimizer.

    The returned ``score`` is maximized by the outer layer.  L-BFGS-B minimizes
    ``-score``.
    """

    nodes = _nodes_from_result(result)
    active_summary = dict(result.get("active_summary", {}) or {})
    objective_terms = dict(result.get("objective_terms", {}) or {})
    design = dict(result.get("design", {}) or {})

    delta_improvement = _finite(
        objective_terms.get("delta_improvement"),
        _finite(active_summary.get("Delta_start")) - _finite(active_summary.get("Delta_end")),
    )
    if not np.isfinite(delta_improvement):
        delta_improvement = 0.0
    mhd_output_power_MW = _finite(
        active_summary.get("mhd_output_power_MW"),
        _finite(objective_terms.get("mhd_output_power_MW"), 0.0),
    )
    if not np.isfinite(mhd_output_power_MW):
        mhd_output_power_MW = 0.0
    min_tp = _finite(
        active_summary.get("Tp_min_K"),
        _finite(_profile_stat(nodes, "T_p", min), float(weights.min_tp_floor_K)),
    )
    max_te = _finite(
        active_summary.get("Te_max_K"),
        _finite(_profile_stat(nodes, "T_e", max), float(weights.max_te_ceiling_K)),
    )
    min_g = _finite(
        active_summary.get("G_min_excluding_anchor"),
        _finite(_profile_stat(nodes, "G", min), float(weights.g_floor)),
    )
    mach_max = _finite(
        active_summary.get("mach_max"),
        _finite(_profile_stat(nodes, "mach", max), 0.0),
    )
    area_ratio = _area_ratio(nodes)
    b_t = _finite(design.get("B_T"))
    n_requested = max(int(active_summary.get("n_steps_requested", result.get("n_steps_completed", 0)) or 0), 0)
    n_completed = max(int(active_summary.get("n_steps_completed", result.get("n_steps_completed", 0)) or 0), 0)

    temperature_scale = max(float(weights.temperature_scale_K), 1e-300)
    tp_penalty = float(weights.min_tp_shortfall) * _soft_square(float(weights.min_tp_floor_K) - min_tp, temperature_scale)
    te_penalty = float(weights.max_te_excess) * _soft_square(max_te - float(weights.max_te_ceiling_K), temperature_scale)
    g_penalty = float(weights.g_shortfall) * _soft_square(float(weights.g_floor) - min_g, float(weights.g_scale))
    mach_penalty = float(weights.mach_excess) * _soft_square(mach_max - float(weights.mach_ceiling), 1.0)

    area_penalty = 0.0
    if np.isfinite(area_ratio):
        area_penalty += _soft_square(float(weights.area_ratio_min) - area_ratio, max(float(weights.area_ratio_min), 1e-12))
        area_penalty += _soft_square(area_ratio - float(weights.area_ratio_max), max(float(weights.area_ratio_max), 1e-12))
        area_penalty *= float(weights.area_ratio_penalty)
    else:
        area_penalty = float(weights.area_ratio_penalty)

    magnetic_penalty = 0.0
    if np.isfinite(b_t):
        b_scale = max(float(weights.magnetic_field_max_T) - float(weights.magnetic_field_min_T), 1e-12)
        magnetic_penalty += _soft_square(float(weights.magnetic_field_min_T) - b_t, b_scale)
        magnetic_penalty += _soft_square(b_t - float(weights.magnetic_field_max_T), b_scale)
        magnetic_penalty *= float(weights.magnetic_field_penalty)

    incomplete_penalty = 0.0
    if n_requested > 0 and n_completed < n_requested:
        incomplete_penalty = float(weights.incomplete_rollout) * float(n_requested - n_completed) / float(n_requested)

    rollout_ok = bool(result.get("ok", False))
    failure_penalty = 0.0 if rollout_ok else float(weights.failure_penalty)
    score_before_failure = (
        float(weights.delta_improvement) * delta_improvement
        + float(weights.mhd_output_power_MW) * mhd_output_power_MW
        - tp_penalty
        - te_penalty
        - area_penalty
        - magnetic_penalty
        - g_penalty
        - mach_penalty
        - incomplete_penalty
    )
    if rollout_ok:
        score = score_before_failure
    else:
        score = -(
            failure_penalty
            + tp_penalty
            + te_penalty
            + area_penalty
            + magnetic_penalty
            + g_penalty
            + mach_penalty
            + incomplete_penalty
        )
    if not np.isfinite(score):
        score = -float(weights.failure_penalty)

    return {
        "score": float(score),
        "ok": rollout_ok and np.isfinite(score),
        "terms": {
            "delta_improvement": float(delta_improvement),
            "delta_improvement_reward": float(weights.delta_improvement) * delta_improvement,
            "mhd_output_power_MW": float(mhd_output_power_MW),
            "mhd_output_power_reward": float(weights.mhd_output_power_MW) * mhd_output_power_MW,
            "min_tp_K": float(min_tp),
            "min_tp_shortfall_penalty": float(tp_penalty),
            "max_te_K": float(max_te),
            "max_te_excess_penalty": float(te_penalty),
            "area_ratio": float(area_ratio) if np.isfinite(area_ratio) else 0.0,
            "area_ratio_missing": float(0.0 if np.isfinite(area_ratio) else 1.0),
            "area_ratio_penalty": float(area_penalty),
            "B_T": float(b_t) if np.isfinite(b_t) else 0.0,
            "B_T_missing": float(0.0 if np.isfinite(b_t) else 1.0),
            "magnetic_field_penalty": float(magnetic_penalty),
            "min_G": float(min_g),
            "G_shortfall_penalty": float(g_penalty),
            "mach_max": float(mach_max),
            "mach_excess_penalty": float(mach_penalty),
            "incomplete_rollout_penalty": float(incomplete_penalty),
            "failure_penalty": float(failure_penalty),
            "score_before_failure": float(score_before_failure),
        },
    }
