from __future__ import annotations

import time
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from v6_firedrake_reduced.cases.freidberg_reference import load_reference_profile
from v6_firedrake_reduced.design import (
    DESIGN_VARIABLE_NAMES,
    CaseConfig,
    DesignVector,
    load_case_config,
)

from .policy import AnchorState, PreparationSettings, State, recover_preparation_profile


AREA_DESIGN_VARIABLE_NAMES = ("a1", "a2", "a3")
SEARCH_DESIGN_VARIABLE_NAMES = tuple(
    name for name in DESIGN_VARIABLE_NAMES if name not in AREA_DESIGN_VARIABLE_NAMES
)


@dataclass(frozen=True)
class AnchorOptions:
    x: float = 0.0
    logA: float = 0.0
    sigma_logA: float | None = None
    source: str = "design_anchor"


@dataclass(frozen=True)
class PreparationObjectiveWeights:
    delta_improvement: float = 1.0
    inlet_delta: float = 0.05
    inlet_te_floor_K: float = 6000.0
    inlet_tp_floor_K: float = 3000.0
    inlet_te_shortfall: float = 1.0
    inlet_tp_shortfall: float = 1.0
    temperature_scale_K: float = 1000.0
    failure_penalty: float = 1.0e6


def load_base_config(*, case: str, n_intervals: int | None = None) -> CaseConfig:
    return load_case_config(case=case, n_intervals=n_intervals)


def default_anchor_sigma(config: CaseConfig) -> float:
    if str(config.case) == "freidberg_reference":
        profile = load_reference_profile()
        sigma = np.asarray(profile["sigma_logA"], dtype=float).reshape(-1)
        if sigma.size:
            return float(sigma[0])
    return 0.0


def design_from_overrides(
    base_design: DesignVector,
    overrides: dict[str, float],
    *,
    allow_area_overrides: bool = False,
) -> DesignVector:
    unknown = sorted(set(overrides) - set(DESIGN_VARIABLE_NAMES))
    if unknown:
        raise ValueError(f"unknown design variable(s): {unknown}")
    if not allow_area_overrides:
        area = sorted(set(overrides) & set(AREA_DESIGN_VARIABLE_NAMES))
        if area:
            raise ValueError(
                "area spline variables are not search variables for the active-boundary solver: "
                + ", ".join(area)
            )
    values = base_design.to_dict()
    for name, value in overrides.items():
        values[str(name)] = float(value)
    return DesignVector.from_dict(values)


def config_with_design(config: CaseConfig, design: DesignVector) -> CaseConfig:
    return replace(config, design=design, B_T=float(design.B_T))


def anchor_from_design(
    config: CaseConfig,
    *,
    options: AnchorOptions | None = None,
) -> AnchorState:
    opts = options or AnchorOptions()
    design = config.design
    if float(design.T_e_in) <= 0.0:
        raise ValueError("design.T_e_in must be positive.")
    sigma = default_anchor_sigma(config) if opts.sigma_logA is None else float(opts.sigma_logA)
    return AnchorState(
        state=State(
            log_n=float(design.log_n_p_in),
            log_Te=float(np.log(float(design.T_e_in))),
            logA=float(opts.logA),
        ),
        sigma_logA=float(sigma),
        x=float(opts.x),
        source=str(opts.source),
        source_index=-1,
    )


def evaluate_preparation_design(
    *,
    base_config: CaseConfig,
    design_overrides: dict[str, float],
    settings: PreparationSettings,
    weights: PreparationObjectiveWeights | None = None,
    anchor_options: AnchorOptions | None = None,
    case_id: int | None = None,
    return_payload: bool = False,
) -> dict[str, Any]:
    objective_weights = weights or PreparationObjectiveWeights()
    started = time.perf_counter()
    design_payload: dict[str, float]
    try:
        design = design_from_overrides(
            base_config.design,
            {str(k): float(v) for k, v in design_overrides.items()},
            allow_area_overrides=False,
        )
        config = config_with_design(base_config, design)
        anchor = anchor_from_design(config, options=anchor_options)
        design_payload = design.to_dict()
        payload = recover_preparation_profile(config=config, anchor=anchor, settings=settings)
        result = _score_rollout(
            payload=payload,
            design=design,
            weights=objective_weights,
            elapsed_s=time.perf_counter() - started,
            case_id=case_id,
            failure=None,
        )
        if return_payload:
            result["payload"] = payload
        return result
    except Exception as exc:
        try:
            design_payload = design_from_overrides(
                base_config.design,
                {str(k): float(v) for k, v in design_overrides.items()},
                allow_area_overrides=False,
            ).to_dict()
        except Exception:
            design_payload = dict(base_config.design.to_dict())
            for key, value in design_overrides.items():
                design_payload[str(key)] = float(value)
        return _failure_result(
            design=design_payload,
            weights=objective_weights,
            elapsed_s=time.perf_counter() - started,
            case_id=case_id,
            failure=str(exc),
        )


def _score_rollout(
    *,
    payload: dict[str, Any],
    design: DesignVector,
    weights: PreparationObjectiveWeights,
    elapsed_s: float,
    case_id: int | None,
    failure: str | None,
) -> dict[str, Any]:
    nodes = list(payload.get("nodes", []))
    if len(nodes) < 2:
        return _failure_result(
            design=design.to_dict(),
            weights=weights,
            elapsed_s=elapsed_s,
            case_id=case_id,
            failure=failure or "rollout produced fewer than two nodes",
        )
    outlet = dict(nodes[0])
    inlet = dict(nodes[-1])
    active_summary = dict(payload.get("active_summary", {}) or {})
    support_counts = dict(active_summary.get("support_counts", {}) or {})
    settings_payload = dict(payload.get("settings", {}) or {})
    g_floor = float(settings_payload.get("g_floor", 0.0))
    tp_floor = float(settings_payload.get("tp_floor_K", 300.0))
    active_tol = float(settings_payload.get("active_tol", 1e-6))
    delta_improvement = float(outlet["Delta"]) - float(inlet["Delta"])
    inlet_delta_penalty = float(weights.inlet_delta) * max(float(inlet["Delta"]), 0.0)
    temperature_scale = max(float(weights.temperature_scale_K), 1e-300)
    te_shortfall_penalty = float(weights.inlet_te_shortfall) * (
        max(float(weights.inlet_te_floor_K) - float(inlet["T_e"]), 0.0) / temperature_scale
    ) ** 2
    tp_shortfall_penalty = float(weights.inlet_tp_shortfall) * (
        max(float(weights.inlet_tp_floor_K) - float(inlet["T_p"]), 0.0) / temperature_scale
    ) ** 2
    target_g_margin = float(outlet["G"]) - g_floor
    target_tp_margin = float(outlet["T_p"]) - tp_floor
    target_ok = bool(target_g_margin >= -active_tol and target_tp_margin >= -active_tol)
    rollout_ok = bool(payload.get("ok", False)) and target_ok
    failure_reason = failure
    if failure_reason is None and not target_ok:
        failure_reason = (
            "target_anchor_infeasible: "
            f"G_margin={target_g_margin:.6g}, Tp_margin={target_tp_margin:.6g}"
        )
    if failure_reason is None and not bool(payload.get("ok", False)):
        failure_reason = "reverse_rollout_failed"
    failure_penalty = 0.0 if rollout_ok else float(weights.failure_penalty)
    raw_score = float(weights.delta_improvement) * delta_improvement
    raw_score -= inlet_delta_penalty + te_shortfall_penalty + tp_shortfall_penalty
    score = raw_score if rollout_ok and np.isfinite(raw_score) else -float(weights.failure_penalty)
    return {
        "case_id": None if case_id is None else int(case_id),
        "ok": rollout_ok,
        "score": float(score),
        "failure": failure_reason,
        "elapsed_s": float(elapsed_s),
        "design": design.to_dict(),
        "search_variables": {name: float(getattr(design, name)) for name in SEARCH_DESIGN_VARIABLE_NAMES},
        "fixed_area_variables": {name: float(getattr(design, name)) for name in AREA_DESIGN_VARIABLE_NAMES},
        "objective_terms": {
            "delta_improvement": float(delta_improvement),
            "delta_improvement_reward": float(weights.delta_improvement) * delta_improvement,
            "inlet_delta_penalty": inlet_delta_penalty,
            "inlet_te_shortfall_penalty": te_shortfall_penalty,
            "inlet_tp_shortfall_penalty": tp_shortfall_penalty,
            "failure_penalty": failure_penalty,
            "raw_score_before_failure_gate": float(raw_score),
            "target_g_margin": target_g_margin,
            "target_tp_margin": target_tp_margin,
            "target_ok": float(1.0 if target_ok else 0.0),
        },
        "outlet": _node_summary(outlet),
        "preparation_inlet": _node_summary(inlet),
        "active_summary": active_summary,
        "support_counts": support_counts,
        "max_abs_scaled_residual": float(active_summary.get("max_abs_scaled_residual", np.nan)),
        "n_steps_completed": max(len(nodes) - 1, 0),
    }


def _failure_result(
    *,
    design: dict[str, float],
    weights: PreparationObjectiveWeights,
    elapsed_s: float,
    case_id: int | None,
    failure: str,
) -> dict[str, Any]:
    return {
        "case_id": None if case_id is None else int(case_id),
        "ok": False,
        "score": -float(weights.failure_penalty),
        "failure": str(failure),
        "elapsed_s": float(elapsed_s),
        "design": {str(k): float(v) for k, v in design.items()},
        "search_variables": {
            name: float(design[name]) for name in SEARCH_DESIGN_VARIABLE_NAMES if name in design
        },
        "fixed_area_variables": {
            name: float(design[name]) for name in AREA_DESIGN_VARIABLE_NAMES if name in design
        },
        "objective_terms": {
            "delta_improvement": float("nan"),
            "delta_improvement_reward": float("nan"),
            "inlet_delta_penalty": float("nan"),
            "inlet_te_shortfall_penalty": float("nan"),
            "inlet_tp_shortfall_penalty": float("nan"),
            "failure_penalty": float(weights.failure_penalty),
        },
        "outlet": {},
        "preparation_inlet": {},
        "active_summary": {},
        "support_counts": {},
        "max_abs_scaled_residual": float("nan"),
        "n_steps_completed": 0,
    }


def _node_summary(node: dict[str, Any]) -> dict[str, float]:
    fields = (
        "x",
        "n_p",
        "T_e",
        "T_p",
        "Delta",
        "mach",
        "A",
        "logA",
        "sigma_logA",
        "G",
        "beta",
        "Z",
    )
    return {name: float(node[name]) for name in fields if name in node}


def flatten_result_for_csv(result: dict[str, Any]) -> dict[str, Any]:
    active_summary = dict(result.get("active_summary", {}) or {})
    termination = dict(active_summary.get("termination", {}) or {})
    bound_sources = dict(termination.get("bound_sources", {}) or {})
    constraint_margins = dict(termination.get("constraint_margins", {}) or {})
    scan_diagnostics = dict(termination.get("scan_diagnostics", {}) or {})
    row: dict[str, Any] = {
        "case_id": result.get("case_id"),
        "ok": bool(result.get("ok", False)),
        "score": float(result.get("score", np.nan)),
        "failure": result.get("failure") or "",
        "elapsed_s": float(result.get("elapsed_s", np.nan)),
        "max_abs_scaled_residual": float(result.get("max_abs_scaled_residual", np.nan)),
        "n_steps_completed": int(result.get("n_steps_completed", 0)),
    }
    for name in (
        "n_steps_requested",
        "n_steps_completed",
        "logA_min",
        "logA_max",
        "A_min",
        "A_max",
        "Te_min_K",
        "Te_max_K",
        "Tp_min_K",
        "Tp_max_K",
        "mach_min",
        "mach_max",
        "sigma_min",
        "sigma_max",
        "G_min_excluding_anchor",
        "G_active_count_excluding_anchor",
        "G_margin_near_count_excluding_anchor",
        "Tp_min_excluding_anchor_K",
        "Tp_floor_active_count_excluding_anchor",
        "Tp_floor_margin_near_count_excluding_anchor",
    ):
        if name in active_summary:
            row[f"active_{name}"] = active_summary[name]
    for name in (
        "ok",
        "reason",
        "step",
        "support_type",
        "solver_method",
        "sigma",
        "sigma_interval_lower",
        "sigma_interval_upper",
        "max_abs_scaled_residual",
        "least_squares_nfev",
        "error",
    ):
        if name in termination:
            row[f"termination_{name}"] = termination[name]
    for side, source in bound_sources.items():
        row[f"termination_bound_source_{side}"] = source
    for name, value in constraint_margins.items():
        row[f"termination_margin_{name}"] = value
    for name in (
        "n_scan",
        "feasible_count",
        "sigma_min",
        "sigma_max",
        "feasible_sigma_min",
        "feasible_sigma_max",
    ):
        if name in scan_diagnostics:
            row[f"scan_{name}"] = scan_diagnostics[name]
    for label in (
        "best_violation",
        "best_residual",
        "left_endpoint",
        "right_endpoint",
        "best_objective_feasible",
    ):
        item = dict(scan_diagnostics.get(label, {}) or {})
        if not item:
            continue
        for name in (
            "sigma",
            "ok",
            "feasible",
            "objective_value",
            "delta_gain",
            "constraint_violation",
            "max_abs_scaled_residual",
            "least_squares_nfev",
        ):
            if name in item:
                row[f"scan_{label}_{name}"] = item[name]
        for name, value in dict(item.get("constraint_margins", {}) or {}).items():
            row[f"scan_{label}_margin_{name}"] = value
    for name, value in dict(result.get("design", {}) or {}).items():
        row[f"design_{name}"] = float(value)
    for name, value in dict(result.get("objective_terms", {}) or {}).items():
        row[f"objective_{name}"] = float(value)
    for prefix in ("outlet", "preparation_inlet"):
        for name, value in dict(result.get(prefix, {}) or {}).items():
            row[f"{prefix}_{name}"] = float(value)
    for name, value in dict(result.get("support_counts", {}) or {}).items():
        row[f"support_{name}"] = int(value)
    return row
