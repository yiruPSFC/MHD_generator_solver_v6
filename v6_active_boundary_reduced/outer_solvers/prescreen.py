from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from v6_firedrake_reduced.design import CaseConfig

from ..core.objective import AnchorOptions, PreparationObjectiveWeights, evaluate_preparation_design
from ..core.policy import PreparationSettings
from .reward import OuterRewardWeights, score_outer_result


CONTROL_VARIABLE_NAMES = (
    "log_n_p_in",
    "T_e_in",
    "Z_in",
    "I_0",
    "log_seed_fraction",
)


@dataclass(frozen=True)
class PrescreenSettings:
    candidates: int = 64
    top_k: int = 8
    seed: int = 1
    te_over_tp_ceiling: float = 1.2
    min_te_over_tp_gradient: float = 0.0
    g_floor: float = 0.0
    require_rollout_ok: bool = True
    allow_fallback: bool = False


def variable_bounds(
    config: CaseConfig,
    *,
    variable_names: tuple[str, ...] = CONTROL_VARIABLE_NAMES,
    explicit_bounds: dict[str, tuple[float, float]] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    explicit_bounds = dict(explicit_bounds or {})
    lower = []
    upper = []
    for name in variable_names:
        if name in explicit_bounds:
            lo, hi = explicit_bounds[name]
        else:
            lo = float(getattr(config.bounds.lower, name))
            hi = float(getattr(config.bounds.upper, name))
        if hi < lo:
            raise ValueError(f"upper bound is smaller than lower bound for {name}.")
        lower.append(float(lo))
        upper.append(float(hi))
    return np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)


def to_normalized(values: np.ndarray, *, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    width = np.maximum(np.asarray(upper, dtype=float) - np.asarray(lower, dtype=float), 1e-300)
    return np.clip((np.asarray(values, dtype=float) - np.asarray(lower, dtype=float)) / width, 0.0, 1.0)


def from_normalized(values: np.ndarray, *, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    return np.asarray(lower, dtype=float) + np.clip(np.asarray(values, dtype=float), 0.0, 1.0) * (
        np.asarray(upper, dtype=float) - np.asarray(lower, dtype=float)
    )


def overrides_from_values(
    values: np.ndarray,
    *,
    variable_names: tuple[str, ...] = CONTROL_VARIABLE_NAMES,
    fixed_overrides: dict[str, float] | None = None,
) -> dict[str, float]:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size != len(variable_names):
        raise ValueError(f"expected {len(variable_names)} values, got {arr.size}.")
    overrides = {name: float(value) for name, value in zip(variable_names, arr, strict=True)}
    overrides.update({str(k): float(v) for k, v in dict(fixed_overrides or {}).items()})
    return overrides


def _nodes(result: dict[str, Any]) -> list[dict[str, Any]]:
    payload = dict(result.get("payload", {}) or {})
    raw_nodes = payload.get("nodes", [])
    if isinstance(raw_nodes, list):
        return [dict(node) for node in raw_nodes if isinstance(node, dict)]
    return []


def prescreen_metrics(result: dict[str, Any]) -> dict[str, Any]:
    nodes = _nodes(result)
    if not nodes:
        return {
            "has_profile": False,
            "anchor_G": float("nan"),
            "anchor_te_over_tp": float("nan"),
            "te_over_tp_gradient": float("nan"),
        }
    anchor = dict(nodes[0])
    anchor_tp = max(float(anchor.get("T_p", float("nan"))), 1e-300)
    anchor_ratio = float(anchor.get("T_e", float("nan"))) / anchor_tp
    anchor_g = float(anchor.get("G", float("nan")))
    gradient = float("nan")
    if len(nodes) >= 2:
        ordered = sorted(nodes, key=lambda node: float(node["x"]))
        x = np.asarray([float(node["x"]) for node in ordered], dtype=float)
        ratio = np.asarray(
            [float(node["T_e"]) / max(float(node["T_p"]), 1e-300) for node in ordered],
            dtype=float,
        )
        if x.size >= 2 and abs(float(x[-1] - x[-2])) > 0.0:
            gradient = float((ratio[-1] - ratio[-2]) / (x[-1] - x[-2]))
    return {
        "has_profile": True,
        "anchor_G": float(anchor_g),
        "anchor_te_over_tp": float(anchor_ratio),
        "te_over_tp_gradient": float(gradient),
    }


def passes_prescreen(metrics: dict[str, Any], *, settings: PrescreenSettings) -> bool:
    if not bool(metrics.get("has_profile", False)):
        return False
    # REVIEW: These gates are reverse-preparation seed heuristics; a future
    # forward-self-consistent prescreen will need direction-specific metrics.
    return bool(
        float(metrics.get("anchor_G", float("nan"))) >= float(settings.g_floor)
        and float(metrics.get("anchor_te_over_tp", float("nan"))) <= float(settings.te_over_tp_ceiling)
        and float(metrics.get("te_over_tp_gradient", float("nan"))) >= float(settings.min_te_over_tp_gradient)
    )


def _prescreen_rank_score(
    *,
    metrics: dict[str, Any],
    outer_score: float,
    settings: PrescreenSettings,
) -> float:
    gradient = float(metrics.get("te_over_tp_gradient", -1e300))
    ratio = float(metrics.get("anchor_te_over_tp", 1e300))
    g = float(metrics.get("anchor_G", -1e300))
    if not np.isfinite(gradient):
        gradient = -1e6
    if not np.isfinite(ratio):
        ratio = 1e6
    if not np.isfinite(g):
        g = -1e6
    ratio_penalty = max(ratio - float(settings.te_over_tp_ceiling), 0.0)
    g_penalty = max(float(settings.g_floor) - g, 0.0) / 1.0e5
    return float(gradient - ratio_penalty - g_penalty + 1e-3 * float(outer_score))


def prescreen_candidates(
    *,
    base_config: CaseConfig,
    lower: np.ndarray,
    upper: np.ndarray,
    variable_names: tuple[str, ...],
    settings: PrescreenSettings,
    rollout_settings: PreparationSettings,
    rollout_weights: PreparationObjectiveWeights,
    reward_weights: OuterRewardWeights,
    anchor_options: AnchorOptions,
    fixed_overrides: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(int(settings.seed))
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    if lower.shape != upper.shape:
        raise ValueError("lower and upper bounds must have matching shape.")
    rows: list[dict[str, Any]] = []
    centers = [0.5 * (lower + upper)]
    for idx, center in enumerate(centers):
        values = np.asarray(center, dtype=float)
        result = evaluate_preparation_design(
            base_config=base_config,
            design_overrides=overrides_from_values(
                values,
                variable_names=variable_names,
                fixed_overrides=fixed_overrides,
            ),
            settings=rollout_settings,
            weights=rollout_weights,
            anchor_options=anchor_options,
            case_id=idx,
            return_payload=True,
        )
        scored = score_outer_result(result, weights=reward_weights)
        metrics = prescreen_metrics(result)
        accepted = passes_prescreen(metrics, settings=settings)
        if bool(settings.require_rollout_ok):
            accepted = accepted and bool(result.get("ok", False))
        rows.append(
            {
                "case_id": int(idx),
                "values": [float(v) for v in values],
                "normalized": [float(v) for v in to_normalized(values, lower=lower, upper=upper)],
                "design_overrides": overrides_from_values(
                    values,
                    variable_names=variable_names,
                    fixed_overrides=fixed_overrides,
                ),
                "accepted": bool(accepted),
                "outer_score": float(scored["score"]),
                "prescreen_rank_score": _prescreen_rank_score(
                    metrics=metrics,
                    outer_score=float(scored["score"]),
                    settings=settings,
                ),
                "prescreen_metrics": metrics,
                "rollout_ok": bool(result.get("ok", False)),
                "failure": result.get("failure"),
                "reward_terms": dict(scored["terms"]),
                "result": result,
            }
        )
    for offset in range(max(int(settings.candidates) - len(centers), 0)):
        idx = offset + len(centers)
        values = lower + rng.random(lower.shape) * (upper - lower)
        result = evaluate_preparation_design(
            base_config=base_config,
            design_overrides=overrides_from_values(
                values,
                variable_names=variable_names,
                fixed_overrides=fixed_overrides,
            ),
            settings=rollout_settings,
            weights=rollout_weights,
            anchor_options=anchor_options,
            case_id=idx,
            return_payload=True,
        )
        scored = score_outer_result(result, weights=reward_weights)
        metrics = prescreen_metrics(result)
        accepted = passes_prescreen(metrics, settings=settings)
        if bool(settings.require_rollout_ok):
            accepted = accepted and bool(result.get("ok", False))
        rows.append(
            {
                "case_id": int(idx),
                "values": [float(v) for v in values],
                "normalized": [float(v) for v in to_normalized(values, lower=lower, upper=upper)],
                "design_overrides": overrides_from_values(
                    values,
                    variable_names=variable_names,
                    fixed_overrides=fixed_overrides,
                ),
                "accepted": bool(accepted),
                "outer_score": float(scored["score"]),
                "prescreen_rank_score": _prescreen_rank_score(
                    metrics=metrics,
                    outer_score=float(scored["score"]),
                    settings=settings,
                ),
                "prescreen_metrics": metrics,
                "rollout_ok": bool(result.get("ok", False)),
                "failure": result.get("failure"),
                "reward_terms": dict(scored["terms"]),
                "result": result,
            }
        )
    accepted_rows = [row for row in rows if bool(row["accepted"])]
    source = accepted_rows
    if not source and bool(settings.allow_fallback):
        source = rows
    ranked = sorted(source, key=lambda row: float(row["prescreen_rank_score"]), reverse=True)
    return ranked[: max(int(settings.top_k), 1)]
