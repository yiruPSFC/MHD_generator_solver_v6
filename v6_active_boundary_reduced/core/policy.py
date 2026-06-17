from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Callable

import numpy as np
from scipy.optimize import brentq

from v6_firedrake_reduced.design import CaseConfig
from v6_firedrake_reduced.geometry import LogAreaSplineControl
from v6_firedrake_reduced.transport import working_fluid_for_config

from ..diagnostics.summary import (
    active_summary as _active_summary,
    eval_public as _eval_public,
    scan_diagnostics as _scan_diagnostics,
)
from .finite_step import (
    empty_rk4_stage_summary as _empty_rk4_stage_summary,
    evaluate_sigma as _evaluate_sigma_impl,
    finite_or_default as _finite_or_default,
    finalize_rk4_stage_summary as _finalize_rk4_stage_summary,
    primitive_log_rhs as _primitive_log_rhs_impl,
    primitive_log_rhs_with_diagnostics as _primitive_log_rhs_with_diagnostics_impl,
    rk4_integrate_state as _rk4_integrate_state_impl,
    rk4_rhs_mode as _rk4_rhs_mode,
    rk4_stage_gate_margins as _rk4_stage_gate_margins,
    row_cosine as _row_cosine,
    row_normalized_matrix as _row_normalized_matrix,
    row_norms as _row_norms,
    safe_matrix_cond as _safe_matrix_cond,
    safe_matrix_det as _safe_matrix_det,
    safe_singular_values as _safe_singular_values,
    solve_linear_rhs as _solve_linear_rhs,
    solve_next_state_rk4 as _solve_next_state_rk4_impl,
    update_rk4_stage_summary as _update_rk4_stage_summary,
)
from .local_affine import ForwardAffineCoefficients, compute_forward_affine_coefficients
from .numba_physics import closure_state_numba, inlet_design_numba
from .reverse_sign_policy import (
    build_reverse_sigma_interval,
    choose_objective_endpoint,
    classify_endpoint_support,
    interval_diagnostics,
    reverse_coefficients_from_forward,
)
from .sigma_interval import build_base_sigma_interval
from .sonic import (
    apply_sonic_residual_gate as _apply_sonic_residual_gate,
    choose_sonic_sigma as _choose_sonic_sigma,
    primitive_sonic_compatibility as _primitive_sonic_compatibility_impl,
    should_use_sonic_branch as _should_use_sonic_branch,
    solve_sonic_finite_step as _solve_sonic_finite_step_impl,
    sonic_compatibility_choice_diagnostics as _sonic_compatibility_choice_diagnostics,
    sonic_compatibility_residual as _sonic_compatibility_residual,
    sonic_initial_guesses as _sonic_initial_guesses_impl,
)


G_BOUNDARY_FALLBACK_ENDPOINT_BRENTQ = "endpoint_brentq"
G_BOUNDARY_FALLBACK_AFFINE_EXPAND_THEN_ENDPOINT_BRENTQ = "affine_expand_then_endpoint_brentq"


@dataclass(frozen=True)
class PolicySettings:
    direction: str = "forward"
    objective: str = "delta_gain"
    n_steps: int = 12
    sigma_min: float = -0.5
    sigma_max: float = 0.5
    curvature_max: float | None = 8.0
    g_floor: float = 0.0
    tp_floor_K: float = 300.0
    scan_points: int = 41
    refine_iterations: int = 24
    active_tol: float = 1e-6
    sonic_mode: str = "auto"
    sonic_mach_tol: float = 1.0e-3
    sonic_det_abs_tol: float = 1.0e-2
    sonic_compatibility_tol: float = 1.0e-7
    sonic_residual_tol: float = 1.0e-6
    rk4_substeps: int = 1
    rk4_error_tol: float = 1.0e-6
    rk4_rhs_mode: str = "raw"
    rk4_stage_replay_tol: float = float("inf")
    rk4_stage_diagnostics: bool = False
    rk4_stage_gate: bool = False
    rk4_stage_cond_max: float = float("inf")
    rk4_stage_mach_max: float = float("inf")
    rk4_stage_tp_floor_K: float | None = None
    rk4_stage_g_floor: float | None = None
    g_boundary_fallback_mode: str = G_BOUNDARY_FALLBACK_ENDPOINT_BRENTQ


@dataclass(frozen=True)
class PreparationSettings:
    n_steps: int = 60
    dx: float = 0.01
    objective: str = "delta_drop"
    sigma_min: float = -0.5
    sigma_max: float = 0.5
    curvature_max: float | None = 8.0
    g_floor: float = 0.0
    tp_floor_K: float = 300.0
    scan_points: int = 41
    refine_iterations: int = 24
    active_tol: float = 1e-6
    sonic_mode: str = "auto"
    sonic_mach_tol: float = 1.0e-3
    sonic_det_abs_tol: float = 1.0e-2
    sonic_compatibility_tol: float = 1.0e-7
    sonic_residual_tol: float = 1.0e-6
    rk4_substeps: int = 1
    rk4_error_tol: float = 1.0e-6
    rk4_rhs_mode: str = "raw"
    rk4_stage_replay_tol: float = float("inf")
    rk4_stage_diagnostics: bool = False
    rk4_stage_gate: bool = False
    rk4_stage_cond_max: float = float("inf")
    rk4_stage_mach_max: float = float("inf")
    rk4_stage_tp_floor_K: float | None = None
    rk4_stage_g_floor: float | None = None
    g_boundary_fallback_mode: str = G_BOUNDARY_FALLBACK_ENDPOINT_BRENTQ


_POLICY_SETTING_FIELD_NAMES = {field.name for field in fields(PolicySettings)}


def _preparation_policy_settings(settings: PreparationSettings) -> PolicySettings:
    values = {name: getattr(settings, name) for name in _POLICY_SETTING_FIELD_NAMES if name != "direction"}
    values["direction"] = "reverse"
    return PolicySettings(**values)


@dataclass(frozen=True)
class State:
    log_n: float
    log_Te: float
    logA: float

    @property
    def n_p(self) -> float:
        return float(np.exp(np.clip(self.log_n, -700.0, 700.0)))

    @property
    def T_e(self) -> float:
        return float(np.exp(np.clip(self.log_Te, -700.0, 700.0)))

    def area(self, config: CaseConfig) -> float:
        return float(config.area_scale_m2) * float(np.exp(np.clip(self.logA, -700.0, 700.0)))


@dataclass(frozen=True)
class AnchorState:
    state: State
    # REVIEW: External anchors may not have a physical slope history; rollout disables curvature on that first step.
    sigma_logA: float | None = None
    x: float = 0.0
    source: str = "manual"
    source_index: int = -1


@dataclass(frozen=True)
class PhysicsParams:
    dot_N: float
    I_0: float
    seed_fraction: float
    B: float
    area_scale_m2: float
    heavy_particle_mass_kg: float
    seed_ionization_energy_J: float
    sigma_ep: float


_PHYSICS_PARAMS_CACHE: dict[tuple[Any, ...], PhysicsParams] = {}


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    out = float(value)
    return out if np.isfinite(out) else None


def anchor_from_dict(payload: dict[str, Any], *, config: CaseConfig) -> AnchorState:
    if "log_n" in payload:
        log_n = float(payload["log_n"])
    elif "n_p" in payload:
        log_n = float(np.log(max(float(payload["n_p"]), 1e-300)))
    else:
        raise ValueError("anchor JSON must contain log_n or n_p.")
    if "log_Te" in payload:
        log_Te = float(payload["log_Te"])
    elif "T_e" in payload:
        log_Te = float(np.log(max(float(payload["T_e"]), 1.0)))
    else:
        raise ValueError("anchor JSON must contain log_Te or T_e.")
    if "logA" in payload:
        logA = float(payload["logA"])
    elif "A" in payload:
        logA = float(np.log(max(float(payload["A"]) / max(float(config.area_scale_m2), 1e-300), 1e-300)))
    else:
        raise ValueError("anchor JSON must contain logA or A.")
    return AnchorState(
        state=State(log_n=log_n, log_Te=log_Te, logA=logA),
        sigma_logA=_optional_float(payload.get("sigma_logA")),
        x=float(payload.get("x", 0.0)),
        source=str(payload.get("source", "manual")),
        source_index=int(payload.get("source_index", -1)),
    )


def anchor_from_profile(
    profile: dict[str, Any],
    *,
    index: int,
    config: CaseConfig,
    source: str = "profile",
) -> AnchorState:
    x = np.asarray(profile["x"], dtype=float).reshape(-1)
    n_p = np.asarray(profile["n_p"], dtype=float).reshape(-1)
    T_e = np.asarray(profile["T_e"], dtype=float).reshape(-1)
    A = np.asarray(profile["A"], dtype=float).reshape(-1)
    if "sigma_logA" in profile:
        sigma = np.asarray(profile["sigma_logA"], dtype=float).reshape(-1)
    else:
        sigma = np.gradient(np.log(np.maximum(A / max(float(A[0]), 1e-300), 1e-300)), x, edge_order=1)
    idx = int(index)
    if idx < 0:
        idx = int(x.size + idx)
    if idx < 0 or idx >= x.size:
        raise IndexError(f"anchor profile index out of range: {index}")
    return AnchorState(
        state=State(
            log_n=float(np.log(max(float(n_p[idx]), 1e-300))),
            log_Te=float(np.log(max(float(T_e[idx]), 1.0))),
            logA=float(np.log(max(float(A[idx]) / max(float(config.area_scale_m2), 1e-300), 1e-300))),
        ),
        sigma_logA=float(sigma[idx]),
        x=float(x[idx]),
        source=source,
        source_index=idx,
    )


def _physics_params(config: CaseConfig) -> PhysicsParams:
    key = (
        str(config.case),
        float(config.area_scale_m2),
        str(config.working_fluid_profile),
        tuple(float(v) for v in config.design.as_array()),
        tuple(sorted((str(k), str(v)) for k, v in dict(config.metadata).items())),
    )
    cached = _PHYSICS_PARAMS_CACHE.get(key)
    if cached is not None:
        return cached
    design = config.design
    fluid = working_fluid_for_config(config)
    inlet = inlet_design_numba(
        float(design.n_p_in),
        float(design.T_e_in),
        float(design.Z_in),
        float(design.I_0),
        float(design.seed_fraction),
        float(design.B_T),
        float(config.area_scale_m2),
        float(fluid.heavy_particle_mass_kg),
        float(fluid.seed_ionization_energy_J),
        float(fluid.sigma_ep),
    )
    params = PhysicsParams(
        dot_N=float(inlet[6]),
        I_0=float(design.I_0),
        seed_fraction=float(design.seed_fraction),
        B=float(design.B_T),
        area_scale_m2=float(config.area_scale_m2),
        heavy_particle_mass_kg=float(fluid.heavy_particle_mass_kg),
        seed_ionization_energy_J=float(fluid.seed_ionization_energy_J),
        sigma_ep=float(fluid.sigma_ep),
    )
    _PHYSICS_PARAMS_CACHE[key] = params
    return params


def recover_preparation_profile(
    *,
    config: CaseConfig,
    anchor: AnchorState,
    settings: PreparationSettings,
) -> dict[str, Any]:
    """Recover an upstream preparation profile from a target anchor.

    This is the public reverse-only API.  It never marches downstream and never
    changes objective semantics: every step asks which admissible sigma drops
    Delta = Te/Tp - 1 fastest while satisfying the local physics constraints.
    """

    payload = rollout_policy_from_anchor(
        config=config,
        anchor=anchor,
        settings=_preparation_policy_settings(settings),
        dx=float(settings.dx),
    )
    payload = dict(payload)
    payload.pop("window", None)
    payload["mode"] = "reverse_preparation_recovery"
    payload["settings"] = settings.__dict__
    return payload


def rollout_policy_from_anchor(
    *,
    config: CaseConfig,
    anchor: AnchorState,
    settings: PolicySettings,
    dx: float,
) -> dict[str, Any]:
    direction = _direction_sign(settings.direction)
    if float(dx) <= 0.0:
        raise ValueError("dx must be positive for anchor rollout.")
    states = [anchor.state]
    sigma_prev: float | None = None
    sigma_history: list[float] = []
    segments = []
    nodes = [
        _node_payload(
            0,
            states[0],
            config=config,
            sigma=anchor.sigma_logA,
            seed_index=int(anchor.source_index),
            x=float(anchor.x),
        )
    ]
    for k in range(int(settings.n_steps)):
        sigma_warm_start = _extrapolated_sigma(sigma_history)
        step = _policy_step(
            current=states[-1],
            sigma_prev=sigma_prev,
            sigma_warm_start=sigma_warm_start,
            dx=float(dx),
            direction=direction,
            config=config,
            settings=settings,
        )
        states.append(step["next_state"])
        sigma_prev = float(step["sigma"])
        sigma_history.append(sigma_prev)
        x_value = float(anchor.x) + float(direction) * float(dx) * float(k + 1)
        nodes.append(
            _node_payload(
                k + 1,
                step["next_state"],
                config=config,
                sigma=sigma_prev,
                seed_index=-1,
                x=x_value,
            )
        )
        segments.append({**{key: value for key, value in step.items() if key != "next_state"}, "k": int(k)})
        if not bool(step.get("ok", False)):
            break
    active_summary = _active_summary(
        nodes=nodes,
        segments=segments,
        settings=settings,
        objective=_normalized_objective(settings),
    )
    return {
        "ok": bool(all(bool(item["ok"]) for item in segments)),
        "settings": settings.__dict__,
        "case_config": config.to_dict(),
        "anchor": {
            "source": str(anchor.source),
            "source_index": int(anchor.source_index),
            "x": float(anchor.x),
            "n_p": float(anchor.state.n_p),
            "T_e": float(anchor.state.T_e),
            "A": float(anchor.state.area(config)),
            "log_n": float(anchor.state.log_n),
            "log_Te": float(anchor.state.log_Te),
            "logA": float(anchor.state.logA),
            "sigma_logA": _optional_float(anchor.sigma_logA),
        },
        "window": {
            "indices": [int(anchor.source_index)],
            "x": [float(anchor.x)],
            "dx": float(dx),
            "mode": "anchor",
        },
        "active_summary": active_summary,
        "nodes": nodes,
        "segments": segments,
        "profile_arrays": {
            "x": np.asarray([float(node["x"]) for node in nodes], dtype=float),
            "n_p": np.asarray([float(node["n_p"]) for node in nodes], dtype=float),
            "T_e": np.asarray([float(node["T_e"]) for node in nodes], dtype=float),
            "A": np.asarray([float(node["A"]) for node in nodes], dtype=float),
            "sigma_logA": np.asarray([float(node["sigma_logA"]) for node in nodes], dtype=float),
        },
    }


def _extrapolated_sigma(sigmas: list[float]) -> float | None:
    finite = [float(sigma) for sigma in sigmas if np.isfinite(float(sigma))]
    if not finite:
        return None
    if len(finite) < 2:
        return float(finite[-1])
    return float(finite[-1] + (finite[-1] - finite[-2]))


def _sigma_reference(*values: float | None, default: float = 0.0) -> float:
    for value in values:
        if value is not None and np.isfinite(float(value)):
            return float(value)
    return float(default)


def _policy_step(
    *,
    current: State,
    sigma_prev: float | None,
    sigma_warm_start: float | None = None,
    dx: float,
    direction: int,
    config: CaseConfig,
    settings: PolicySettings,
) -> dict[str, Any]:
    objective = _normalized_objective(settings)
    if int(direction) == -1 and objective == "delta_drop":
        return _reverse_sign_policy_step(
            current=current,
            sigma_prev=sigma_prev,
            sigma_warm_start=sigma_warm_start,
            dx=dx,
            config=config,
            settings=settings,
        )
    if int(direction) == -1:
        return {
            **_invalid_step_payload(
                current=current,
                support_type="unsupported_reverse_objective",
                error=f"reverse active-boundary policy only supports delta_drop, got {objective}",
            ),
            "objective_kind": objective,
            "termination_reason": "unsupported_reverse_objective",
        }
    return _forward_scan_policy_step(
        current=current,
        sigma_prev=sigma_prev,
        sigma_warm_start=sigma_warm_start,
        dx=dx,
        config=config,
        settings=settings,
    )


def _forward_scan_policy_step(
    *,
    current: State,
    sigma_prev: float | None,
    sigma_warm_start: float | None = None,
    dx: float,
    config: CaseConfig,
    settings: PolicySettings,
) -> dict[str, Any]:
    direction = 1
    interval = build_base_sigma_interval(
        logA_current=current.logA,
        sigma_prev=sigma_prev,
        dx=dx,
        direction=direction,
        sigma_min=float(settings.sigma_min),
        sigma_max=float(settings.sigma_max),
        curvature_max=settings.curvature_max,
        logA_min=LogAreaSplineControl.lower_bound(),
        logA_max=LogAreaSplineControl.upper_bound(),
    )
    lo = float(interval.sigma_lo)
    hi = float(interval.sigma_hi)
    bound_sources = {"lower": str(interval.lower_source), "upper": str(interval.upper_source)}
    if lo > hi:
        return {
            "ok": False,
            "sigma": float("nan"),
            "next_state": current,
            "support_type": "empty_sigma_interval",
            "error": "sigma interval is empty before physics constraints",
            "sigma_interval_lower": float(lo),
            "sigma_interval_upper": float(hi),
            "bound_sources": bound_sources,
            "termination_reason": "empty_sigma_interval",
        }

    base_sigma = float(np.clip(_sigma_reference(sigma_prev, sigma_warm_start), lo, hi))
    pedal = _pedal_direction(
        current=current,
        base_sigma=base_sigma,
        lo=lo,
        hi=hi,
        dx=dx,
        direction=direction,
        config=config,
        settings=settings,
    )
    direct = _direct_boundary_choice(
        current=current,
        lo=lo,
        hi=hi,
        pedal=pedal,
        dx=dx,
        direction=direction,
        config=config,
        settings=settings,
    )
    if direct is not None:
        chosen = direct
        support = _classify_support(
            chosen=chosen,
            lo=lo,
            hi=hi,
            bound_sources=bound_sources,
            settings=settings,
        )
        payload = {
            "ok": bool(chosen["ok"]),
            "sigma": float(chosen["sigma"]),
            "next_state": chosen["next_state"],
            "support_type": support,
            "sigma_interval_lower": float(lo),
            "sigma_interval_upper": float(hi),
            "pedal_direction": float(pedal),
            "bound_sources": bound_sources,
            **_eval_public(chosen),
        }
        sonic_payload = _forward_sonic_fallback_payload(
            current=current,
            sigma_prev=sigma_prev,
            sigma_warm_start=sigma_warm_start,
            dx=dx,
            direction=direction,
            config=config,
            settings=settings,
            chosen=chosen,
            scan=[],
        )
        return sonic_payload if sonic_payload is not None else payload

    scan = []
    for sigma in np.linspace(lo, hi, max(int(settings.scan_points), 3)):
        scan.append(
            _evaluate_sigma(
                current=current,
                sigma=float(sigma),
                dx=dx,
                direction=direction,
                config=config,
                settings=settings,
            )
        )
    feasible = [item for item in scan if bool(item["feasible"])]
    if not feasible:
        best_failed = min(scan, key=lambda item: float(item.get("constraint_violation", 1e300)))
        sonic_payload = _forward_sonic_fallback_payload(
            current=current,
            sigma_prev=sigma_prev,
            sigma_warm_start=sigma_warm_start,
            dx=dx,
            direction=direction,
            config=config,
            settings=settings,
            chosen=best_failed,
            scan=scan,
        )
        if sonic_payload is not None:
            return sonic_payload
        return {
            "ok": False,
            "sigma": float(best_failed["sigma"]),
            "next_state": best_failed.get("next_state", current),
            "support_type": "no_feasible_sigma",
            "error": "no feasible sigma on scan grid",
            "sigma_interval_lower": float(lo),
            "sigma_interval_upper": float(hi),
            "pedal_direction": pedal,
            "bound_sources": bound_sources,
            "termination_reason": "no_feasible_sigma",
            "scan_diagnostics": _scan_diagnostics(scan),
            **_eval_public(best_failed),
        }

    if pedal >= 0.0:
        chosen = max(feasible, key=lambda item: float(item["sigma"]))
        neighbor = _first_infeasible_neighbor(scan, chosen_sigma=float(chosen["sigma"]), side="upper")
    else:
        chosen = min(feasible, key=lambda item: float(item["sigma"]))
        neighbor = _first_infeasible_neighbor(scan, chosen_sigma=float(chosen["sigma"]), side="lower")
    if neighbor is not None:
        chosen = _refine_boundary(
            feasible_eval=chosen,
            infeasible_eval=neighbor,
            current=current,
            dx=dx,
            direction=direction,
            config=config,
            settings=settings,
        )
    support = _classify_support(
        chosen=chosen,
        lo=lo,
        hi=hi,
        bound_sources=bound_sources,
        settings=settings,
    )
    payload = {
        "ok": bool(chosen["ok"]),
        "sigma": float(chosen["sigma"]),
        "next_state": chosen["next_state"],
        "support_type": support,
        "sigma_interval_lower": float(lo),
        "sigma_interval_upper": float(hi),
        "pedal_direction": float(pedal),
        "bound_sources": bound_sources,
        "scan_diagnostics": _scan_diagnostics(scan),
        **_eval_public(chosen),
    }
    sonic_payload = _forward_sonic_fallback_payload(
        current=current,
        sigma_prev=sigma_prev,
        sigma_warm_start=sigma_warm_start,
        dx=dx,
        direction=direction,
        config=config,
        settings=settings,
        chosen=chosen,
        scan=scan,
    )
    return sonic_payload if sonic_payload is not None else payload


def _forward_sonic_trigger_reason(
    *,
    sonic: dict[str, Any],
    chosen: dict[str, Any],
    scan: list[dict[str, Any]],
    settings: PolicySettings,
) -> str:
    if _should_use_sonic_branch(sonic=sonic, settings=settings):
        return "current_state_sonic_trigger"

    candidates = [chosen, *scan]
    for item in candidates:
        stage_mach = float(item.get("rk4_stage_max_mach", float("nan")))
        if np.isfinite(stage_mach) and stage_mach >= 1.0 - float(settings.active_tol):
            return "rk4_stage_mach_crossing"

    cond_max = float(getattr(settings, "rk4_stage_cond_max", float("inf")))
    if np.isfinite(cond_max):
        for item in candidates:
            stage_cond = float(item.get("rk4_stage_max_cond_row_norm_log", float("nan")))
            if np.isfinite(stage_cond) and stage_cond >= cond_max:
                return "rk4_stage_condition_limit"

    det_tol = float(getattr(settings, "sonic_det_abs_tol", 1.0e-2))
    if np.isfinite(det_tol) and det_tol > 0.0:
        for item in candidates:
            stage_det = float(item.get("rk4_stage_min_abs_det_raw", float("nan")))
            if np.isfinite(stage_det) and stage_det <= det_tol:
                return "rk4_stage_raw_det_trigger"

    return ""


def _forward_sonic_fallback_payload(
    *,
    current: State,
    sigma_prev: float | None,
    sigma_warm_start: float | None,
    dx: float,
    direction: int,
    config: CaseConfig,
    settings: PolicySettings,
    chosen: dict[str, Any],
    scan: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if int(direction) != 1:
        return None
    mode = str(getattr(settings, "sonic_mode", "auto")).strip().lower()
    if mode in {"off", "none", "disabled", "false", "0"}:
        return None

    # REVIEW: Near-singular D belongs to the sonic left-null branch; local affine assumes D can be inverted.
    sonic = _primitive_sonic_compatibility(current, config=config)
    trigger = _forward_sonic_trigger_reason(sonic=sonic, chosen=chosen, scan=scan, settings=settings)
    if not trigger:
        return None

    candidate = _sonic_compatible_policy_step(
        current=current,
        sigma_prev=sigma_prev,
        sigma_warm_start=sigma_warm_start,
        dx=dx,
        direction=direction,
        config=config,
        settings=settings,
        sonic=sonic,
    )
    candidate = {
        **candidate,
        "sonic_fallback_trigger": trigger,
        "legacy_sigma": float(chosen.get("sigma", float("nan"))),
        "legacy_objective_value": float(chosen.get("objective_value", float("nan"))),
        "legacy_constraint_violation": float(chosen.get("constraint_violation", float("nan"))),
        "legacy_max_abs_scaled_residual": float(chosen.get("max_abs_scaled_residual", float("nan"))),
    }
    if bool(candidate.get("ok", False)) and bool(candidate.get("feasible", True)):
        return candidate
    return None


def _reverse_sign_policy_step(
    *,
    current: State,
    sigma_prev: float | None,
    sigma_warm_start: float | None,
    dx: float,
    config: CaseConfig,
    settings: PolicySettings,
) -> dict[str, Any]:
    step = float(dx)
    if step <= 0.0:
        return _invalid_step_payload(
            current=current,
            support_type="invalid_reverse_step",
            error="dx must be positive for reverse sign-aware policy",
        )

    current_metrics = _closure_metrics(current, config=config)
    A_current = float(current.area(config))
    g_margin_current = float(current_metrics["G"] - float(settings.g_floor))
    tp_margin_current = float(current_metrics["T_p"] - float(settings.tp_floor_K))
    if not all(np.isfinite(value) for value in [A_current, g_margin_current, tp_margin_current]):
        return _invalid_step_payload(
            current=current,
            support_type="invalid_current_state",
            error="current closure is not finite",
        )
    if A_current <= 0.0:
        return _invalid_step_payload(
            current=current,
            support_type="invalid_current_state",
            error="current area is not positive",
        )
    if g_margin_current < -float(settings.active_tol) or tp_margin_current < -float(settings.active_tol):
        return {
            **_invalid_step_payload(
                current=current,
                support_type="invalid_current_state",
                error="current state violates reverse admissibility",
            ),
            "G_current": float(current_metrics["G"]),
            "G_floor": float(settings.g_floor),
            "G_margin_current": g_margin_current,
            "T_p_current": float(current_metrics["T_p"]),
            "T_p_floor": float(settings.tp_floor_K),
            "T_p_margin_current": tp_margin_current,
        }

    # REVIEW: Near-singular D belongs to the sonic left-null branch; local affine assumes D can be inverted.
    sonic = _primitive_sonic_compatibility(current, config=config)
    if _should_use_sonic_branch(sonic=sonic, settings=settings):
        return _sonic_compatible_policy_step(
            current=current,
            sigma_prev=sigma_prev,
            sigma_warm_start=sigma_warm_start,
            dx=step,
            config=config,
            settings=settings,
            sonic=sonic,
        )

    params = _physics_params(config)
    try:
        affine = compute_forward_affine_coefficients(
            n_p=current.n_p,
            T_e=current.T_e,
            A=A_current,
            logA=current.logA,
            params=params,
        )
    except Exception as exc:
        return {
            **_invalid_step_payload(
                current=current,
                support_type="local_affine_failed",
                error=f"local affine coefficient calculation failed: {exc}",
            ),
            "G_current": float(current_metrics["G"]),
            "T_p_current": float(current_metrics["T_p"]),
        }

    reverse = reverse_coefficients_from_forward(
        dx=step,
        G_current=affine.G_current,
        G_floor=float(settings.g_floor),
        a0=affine.a0,
        a1=affine.a1,
        b0=affine.b0,
        b1=affine.b1,
    )
    interval = build_reverse_sigma_interval(
        A_current=A_current,
        logA_current=current.logA,
        dx=step,
        sigma_prev=sigma_prev,
        sigma_min=float(settings.sigma_min),
        sigma_max=float(settings.sigma_max),
        curvature_max=settings.curvature_max,
        logA_min=LogAreaSplineControl.lower_bound(),
        logA_max=LogAreaSplineControl.upper_bound(),
        q0=reverse.q0,
        q1=reverse.q1,
        g_margin_tol=float(settings.active_tol),
    )
    base_diagnostics = _sign_aware_base_diagnostics(
        affine=affine,
        reverse=reverse,
        interval=interval,
        dx=step,
        settings=settings,
    )
    if not interval.ok:
        return {
            **_invalid_step_payload(
                current=current,
                support_type="empty_reverse_sigma_interval",
                error=interval.error,
            ),
            **base_diagnostics,
        }

    decision = choose_objective_endpoint(
        interval=interval,
        p1=reverse.p1,
        regularizer_sigma=_sigma_reference(sigma_warm_start, sigma_prev),
    )
    endpoint = _evaluate_sigma(
        current=current,
        sigma=float(decision.sigma),
        dx=step,
        direction=-1,
        config=config,
        settings=settings,
    )
    endpoint = {**endpoint, "solver_method": "sign_aware_reverse_endpoint"}
    endpoint_validation_failure = _validation_failure_reason(endpoint, settings=settings)
    chosen = endpoint
    validation_failure = endpoint_validation_failure
    fallback_attempted = bool(endpoint_validation_failure)
    fallback_used = False
    fallback_origin = ""

    if endpoint_validation_failure:
        fallback = None
        if "G" in set(endpoint_validation_failure.split(",")):
            fallback = _sign_aware_g_boundary_fallback(
                current=current,
                interval=interval,
                endpoint=endpoint,
                dx=step,
                config=config,
                settings=settings,
            )
            fallback_origin = "g_boundary_fallback" if fallback is not None else ""
        if fallback is None:
            fallback = _sign_aware_scan_fallback(
                current=current,
                interval=interval,
                p1=reverse.p1,
                dx=step,
                config=config,
                settings=settings,
            )
            fallback_origin = "scan_fallback" if fallback is not None else fallback_origin
        if fallback is not None:
            chosen = fallback
            fallback_used = True
            validation_failure = _validation_failure_reason(chosen, settings=settings)

    if not fallback_attempted:
        fallback_status = "not_needed"
    elif not fallback_used:
        fallback_status = "attempted_no_candidate"
    elif validation_failure:
        fallback_status = "used_but_failed"
    else:
        fallback_status = "used"

    bound_sources = {
        "lower": str(interval.lower_source),
        "upper": str(interval.upper_source),
    }
    affine_support = classify_endpoint_support(
        endpoint_source=str(decision.endpoint_source),
        objective_bound_kind=str(decision.objective_bound_kind),
        p1=reverse.p1,
        q1=reverse.q1,
    )
    selected_support = (
        _classify_support(
            chosen=chosen,
            lo=float(interval.sigma_lo),
            hi=float(interval.sigma_hi),
            bound_sources=bound_sources,
            settings=settings,
        )
        if fallback_used
        else affine_support
    )
    support = selected_support if not validation_failure else "finite_step_validation_failed"
    finite_diagnostics = _sign_aware_finite_diagnostics(
        affine=affine,
        reverse=reverse,
        chosen=chosen,
        decision=decision,
        validation_failure=validation_failure,
        affine_support=affine_support,
        selected_support=selected_support,
        fallback_used=fallback_used,
        fallback_origin=fallback_origin,
        settings=settings,
    )
    return {
        "ok": bool(chosen["ok"] and chosen["feasible"] and not validation_failure),
        "sigma": float(chosen["sigma"]),
        "next_state": chosen["next_state"],
        "support_type": support,
        "affine_support_type": affine_support,
        "bound_sources": bound_sources,
        "sign_aware_fallback_status": fallback_status,
        "sign_aware_fallback_attempted": fallback_attempted,
        "sign_aware_fallback_used": fallback_used,
        "sign_aware_fallback_recovered": bool(fallback_used and not validation_failure),
        "sign_aware_fallback_solver_method": str(chosen.get("solver_method", "")) if fallback_used else "",
        "sign_aware_fallback_validation_failure": str(validation_failure) if fallback_used else "",
        "sign_aware_endpoint_sigma": float(endpoint.get("sigma", float("nan"))),
        "sign_aware_endpoint_ok": bool(endpoint.get("ok", False)),
        "sign_aware_endpoint_feasible": bool(endpoint.get("feasible", False)),
        "sign_aware_endpoint_solver_method": str(endpoint.get("solver_method", "")),
        "sign_aware_endpoint_validation_failure": str(endpoint_validation_failure),
        "sign_aware_endpoint_constraint_violation": float(endpoint.get("constraint_violation", float("nan"))),
        "sign_aware_endpoint_constraint_margins": dict(endpoint.get("constraint_margins", {}) or {}),
        **base_diagnostics,
        **finite_diagnostics,
        **_eval_public(chosen),
    }


def _primitive_sonic_compatibility(state: State, *, config: CaseConfig) -> dict[str, Any]:
    return _primitive_sonic_compatibility_impl(
        state,
        config=config,
        physics_params_fn=_physics_params,
        closure_metrics_fn=_closure_metrics,
    )


def _sonic_compatible_policy_step(
    *,
    current: State,
    sigma_prev: float | None,
    sigma_warm_start: float | None,
    dx: float,
    config: CaseConfig,
    settings: PolicySettings,
    sonic: dict[str, Any],
    direction: int = -1,
) -> dict[str, Any]:
    area = float(current.area(config))
    interval = build_base_sigma_interval(
        logA_current=current.logA,
        sigma_prev=sigma_prev,
        dx=dx,
        direction=direction,
        sigma_min=float(settings.sigma_min),
        sigma_max=float(settings.sigma_max),
        curvature_max=settings.curvature_max,
        logA_min=LogAreaSplineControl.lower_bound(),
        logA_max=LogAreaSplineControl.upper_bound(),
    )
    lo = float(interval.sigma_lo)
    hi = float(interval.sigma_hi)
    bound_sources = {"lower": str(interval.lower_source), "upper": str(interval.upper_source)}
    base = _sonic_diagnostics(
        sonic=sonic,
        lo=lo,
        hi=hi,
        bound_sources=bound_sources,
        settings=settings,
    )
    if lo > hi:
        return {
            **_invalid_step_payload(
                current=current,
                support_type="empty_sonic_sigma_interval",
                error="sigma interval is empty before sonic compatibility",
            ),
            "termination_reason": "empty_sonic_sigma_interval",
            **base,
        }

    choice = _choose_sonic_sigma(
        sonic=sonic,
        area=area,
        lo=lo,
        hi=hi,
        sigma_reference=_sigma_reference(sigma_warm_start, sigma_prev),
        settings=settings,
    )
    choice_diagnostics = _sonic_compatibility_choice_diagnostics(choice)
    base = {**base, **choice_diagnostics}
    sigma = float(choice.get("sigma", float("nan")))
    if not bool(choice.get("ok", False)) or not np.isfinite(sigma):
        status = str(choice.get("status", "failed"))
        return {
            **_invalid_step_payload(
                current=current,
                support_type=f"sonic_compatibility_{status}",
                error=str(choice.get("error") or "sonic compatibility did not produce an admissible sigma"),
            ),
            "sigma": sigma,
            "termination_reason": f"sonic_compatibility_{status}",
            **base,
        }

    chosen = _evaluate_sonic_sigma(
        current=current,
        sigma=sigma,
        dx=dx,
        direction=direction,
        config=config,
        settings=settings,
    )
    chosen = {**chosen, "solver_method": str(choice.get("solver_method", "sonic_left_null_explicit_A_prime"))}
    chosen = _apply_sonic_residual_gate(chosen, settings=settings)
    validation_failure = _validation_failure_reason(chosen, settings=settings)
    delta_gain = float(chosen.get("delta_gain", float("nan")))
    # REVIEW: Sonic compatibility is a feasibility branch; Delta direction remains diagnostic, not a veto.
    support = "sonic_compatible_left_null"
    if validation_failure:
        support = "sonic_finite_step_validation_failed"
    return {
        "ok": bool(chosen["ok"] and chosen["feasible"] and not validation_failure),
        "sigma": float(chosen["sigma"]),
        "next_state": chosen["next_state"],
        "support_type": support,
        "affine_support_type": support,
        "bound_sources": bound_sources,
        "validation_status": "ok" if not validation_failure else "failed",
        "validation_failure_reason": str(validation_failure),
        "sonic_objective_score": float(-delta_gain) if np.isfinite(delta_gain) else float("nan"),
        "sonic_direction_ok": True,
        "sonic_direction_gate": "not_applied",
        "sonic_residual_gate": str(chosen.get("sonic_residual_gate", "")),
        "sonic_step_residual_tol": float(chosen.get("sonic_residual_tol", float("nan"))),
        **base,
        **_eval_public(chosen),
    }


def _evaluate_sonic_sigma(
    *,
    current: State,
    sigma: float,
    dx: float,
    direction: int,
    config: CaseConfig,
    settings: PolicySettings,
) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    logA_next = float(current.logA + float(direction) * float(dx) * float(sigma))
    for initial in _sonic_initial_guesses(current=current, logA_next=logA_next):
        next_state, residual = _solve_sonic_finite_step(
            current=current,
            logA_next=logA_next,
            sigma=float(sigma),
            dx=dx,
            direction=direction,
            config=config,
            initial=initial,
        )
        metrics = _closure_metrics(next_state, config=config)
        objective = _step_objective_payload(
            current_metrics=_closure_metrics(current, config=config),
            next_metrics=metrics,
            dx=dx,
            settings=settings,
        )
        residual_tol = float(getattr(settings, "sonic_residual_tol", 1.0e-6))
        margins = {
            "G": float(metrics["G"] - float(settings.g_floor)),
            "Tp": float(metrics["T_p"] - float(settings.tp_floor_K)),
            "residual": float(residual_tol - float(residual["max_abs_scaled_residual"])),
        }
        feasible = bool(
            bool(residual["ok"]) and all(value >= -float(settings.active_tol) for value in margins.values())
        )
        violation = float(sum(max(-float(value), 0.0) for value in margins.values()))
        candidates.append(
            {
                "ok": bool(residual["ok"]),
                "feasible": feasible,
                "sigma": float(sigma),
                "next_state": next_state,
                **objective,
                "constraint_margins": margins,
                "constraint_violation": violation,
                **metrics,
                **residual,
            }
        )
    feasible_candidates = [item for item in candidates if bool(item.get("feasible", False))]
    if feasible_candidates:
        # RISK: Multi-start sonic solves may find different roots; selection is objective-based, not continuity-based.
        return max(feasible_candidates, key=lambda item: float(item.get("objective_value", -1.0e300)))
    if candidates:
        return min(candidates, key=lambda item: float(item.get("constraint_violation", 1.0e300)))
    return {
        "ok": False,
        "feasible": False,
        "sigma": float(sigma),
        "next_state": current,
        "objective_kind": _normalized_objective(settings),
        "objective_value": float("nan"),
        "delta_gain": float("nan"),
        "constraint_margins": {"residual": -float("inf")},
        "constraint_violation": float("inf"),
        "max_abs_scaled_residual": float("inf"),
        "error": "sonic finite-step solve produced no candidates",
        **_closure_metrics(current, config=config),
    }


def _sonic_initial_guesses(*, current: State, logA_next: float) -> list[State]:
    return _sonic_initial_guesses_impl(current=current, logA_next=logA_next, state_factory=State)


def _solve_sonic_finite_step(
    *,
    current: State,
    logA_next: float,
    sigma: float,
    dx: float,
    direction: int,
    config: CaseConfig,
    initial: State,
) -> tuple[State, dict[str, Any]]:
    return _solve_sonic_finite_step_impl(
        current=current,
        logA_next=logA_next,
        sigma=sigma,
        dx=dx,
        direction=direction,
        config=config,
        initial=initial,
        physics_params_fn=_physics_params,
        state_factory=State,
    )


def _sonic_diagnostics(
    *,
    sonic: dict[str, Any],
    lo: float,
    hi: float,
    bound_sources: dict[str, str],
    settings: PolicySettings,
) -> dict[str, Any]:
    return {
        "sonic_branch_used": True,
        "sonic_mode": str(getattr(settings, "sonic_mode", "auto")),
        "sonic_mach_tol": float(getattr(settings, "sonic_mach_tol", 1.0e-3)),
        "sonic_det_abs_tol": float(getattr(settings, "sonic_det_abs_tol", 1.0e-2)),
        "sonic_compatibility_tol": float(getattr(settings, "sonic_compatibility_tol", 1.0e-7)),
        "sonic_residual_tol": float(getattr(settings, "sonic_residual_tol", 1.0e-6)),
        "sonic_mach": float(sonic.get("mach", float("nan"))),
        "det_D": float(sonic.get("det_D", float("nan"))),
        "sonic_singular_value_min": float(sonic.get("singular_value_min", float("nan"))),
        "sonic_singular_value_max": float(sonic.get("singular_value_max", float("nan"))),
        "ellTf0": float(sonic.get("ellTf0", float("nan"))),
        "ellTf1": float(sonic.get("ellTf1", float("nan"))),
        "A_prime_sonic": float(sonic.get("A_prime_sonic", float("nan"))),
        "sigma_sonic": float(sonic.get("sigma_sonic", float("nan"))),
        "sonic_compatibility_residual": float(sonic.get("compatibility_residual", float("nan"))),
        "sonic_compatibility_scaled_residual": float(
            sonic.get("compatibility_scaled_residual", float("nan"))
        ),
        "sigma_interval_lower": float(lo),
        "sigma_interval_upper": float(hi),
        "lower_source": str(bound_sources.get("lower", "")),
        "upper_source": str(bound_sources.get("upper", "")),
    }


def _invalid_step_payload(*, current: State, support_type: str, error: str) -> dict[str, Any]:
    return {
        "ok": False,
        "sigma": float("nan"),
        "next_state": current,
        "support_type": str(support_type),
        "error": str(error),
        "objective_value": float("nan"),
        "delta_gain": float("nan"),
        "constraint_margins": {},
        "boundary_blockers": [],
        "boundary_bracket_width": float("nan"),
        "boundary_infeasible_sigma": float("nan"),
        "boundary_infeasible_margins": {},
        "solver_method": str(support_type),
        "max_abs_scaled_residual": float("nan"),
    }


def _reverse_interval_conflict_diagnostics(*, affine: ForwardAffineCoefficients, interval) -> dict[str, Any]:
    lower = float(interval.sigma_lo)
    upper = float(interval.sigma_hi)
    area = float(affine.A_current)
    conflict = bool(
        not bool(interval.ok)
        and np.isfinite(lower)
        and np.isfinite(upper)
        and lower > upper
    )
    sigma_gap = float(lower - upper) if conflict else 0.0
    aprime_lower = float(area * lower) if np.isfinite(area) and np.isfinite(lower) else float("nan")
    aprime_upper = float(area * upper) if np.isfinite(area) and np.isfinite(upper) else float("nan")
    aprime_gap = float(area * sigma_gap) if conflict and np.isfinite(area) else float("nan")
    lower_source = str(interval.lower_source)
    upper_source = str(interval.upper_source)
    if conflict:
        summary = (
            f"{lower_source} requires sigma >= {lower:.6g} "
            f"(A_prime >= {aprime_lower:.6g}), but {upper_source} requires "
            f"sigma <= {upper:.6g} (A_prime <= {aprime_upper:.6g}); "
            f"sigma_gap={sigma_gap:.6g}."
        )
    elif not bool(interval.ok):
        summary = f"reverse interval failed: {interval.error}"
    else:
        summary = ""
    return {
        "reverse_interval_conflict": conflict,
        "reverse_interval_conflict_kind": f"{lower_source}_vs_{upper_source}" if conflict else "",
        "reverse_interval_conflict_summary": summary,
        "reverse_interval_conflict_lower_source": lower_source if conflict else "",
        "reverse_interval_conflict_upper_source": upper_source if conflict else "",
        "reverse_interval_conflict_sigma_lower": lower if conflict else float("nan"),
        "reverse_interval_conflict_sigma_upper": upper if conflict else float("nan"),
        "reverse_interval_conflict_sigma_gap": sigma_gap,
        "reverse_interval_conflict_Aprime_lower": aprime_lower if conflict else float("nan"),
        "reverse_interval_conflict_Aprime_upper": aprime_upper if conflict else float("nan"),
        "reverse_interval_conflict_Aprime_gap": aprime_gap,
    }


def _sign_aware_base_diagnostics(
    *,
    affine: ForwardAffineCoefficients,
    reverse,
    interval,
    dx: float,
    settings: PolicySettings,
) -> dict[str, Any]:
    return {
        "a0": float(affine.a0),
        "a1": float(affine.a1),
        "b0": float(affine.b0),
        "b1": float(affine.b1),
        "p0": float(reverse.p0),
        "p1": float(reverse.p1),
        "q0": float(reverse.q0),
        "q1": float(reverse.q1),
        "p1q1_reverse": float(reverse.p1q1),
        "det_D": float(affine.det_D),
        "A_current": float(affine.A_current),
        "logA_current": float(affine.logA_current),
        "dx": float(dx),
        "sigma_min": float(settings.sigma_min),
        "sigma_max": float(settings.sigma_max),
        "G_current": float(affine.G_current),
        "G_floor": float(settings.g_floor),
        "G_margin_current": float(affine.G_current - float(settings.g_floor)),
        "Phi_current": float(affine.phi_current),
        "T_p_current": float(affine.T_p_current),
        "T_p_floor": float(settings.tp_floor_K),
        "T_p_margin_current": float(affine.T_p_current - float(settings.tp_floor_K)),
        "sonic_branch_used": False,
        "ellTf0": float("nan"),
        "ellTf1": float("nan"),
        "A_prime_sonic": float("nan"),
        "sigma_sonic": float("nan"),
        **interval_diagnostics(interval),
        **_reverse_interval_conflict_diagnostics(affine=affine, interval=interval),
    }


def _sign_aware_finite_diagnostics(
    *,
    affine: ForwardAffineCoefficients,
    reverse,
    chosen: dict[str, Any],
    decision,
    validation_failure: str,
    affine_support: str,
    selected_support: str,
    fallback_used: bool,
    fallback_origin: str,
    settings: PolicySettings,
) -> dict[str, Any]:
    sigma_selected = float(chosen.get("sigma", decision.sigma))
    A_prime = float(affine.A_current * sigma_selected)
    objective_drop_predicted = float(reverse.p0 + reverse.p1 * A_prime)
    g_margin_predicted = float(reverse.q0 + reverse.q1 * A_prime)
    phi_upstream_predicted = float(affine.phi_current - objective_drop_predicted)
    g_margin_upstream = float(dict(chosen.get("constraint_margins", {})).get("G", float("nan")))
    tp_margin_upstream = float(dict(chosen.get("constraint_margins", {})).get("Tp", float("nan")))
    residual_margin = float(dict(chosen.get("constraint_margins", {})).get("residual", float("nan")))
    return {
        "sigma_selected": sigma_selected,
        "A_prime_selected": A_prime,
        "selected_sigma_origin": str(fallback_origin or "finite_step_fallback") if fallback_used else "affine_endpoint",
        "selected_sigma_source": (
            str(chosen.get("solver_method", "sign_aware_scan_fallback"))
            if fallback_used
            else str(decision.endpoint_source)
        ),
        "selected_support_type": str(selected_support),
        "affine_objective_bound_kind": str(decision.objective_bound_kind),
        "affine_selected_endpoint_source": str(decision.endpoint_source),
        # Backward-compatible endpoint fields: these describe the affine proposal, not necessarily fallback recovery.
        "objective_bound_kind": str(decision.objective_bound_kind),
        "selected_endpoint_source": str(decision.endpoint_source),
        "objective_drop_predicted": objective_drop_predicted,
        "Phi_upstream_predicted": phi_upstream_predicted,
        "G_margin_upstream_predicted": g_margin_predicted,
        "G_margin_upstream": g_margin_upstream,
        "T_p_upstream": float(chosen.get("T_p", float("nan"))),
        "T_p_margin_upstream": tp_margin_upstream,
        "residual": float(chosen.get("max_abs_scaled_residual", float("nan"))),
        "residual_margin": residual_margin,
        "validation_status": "ok" if not validation_failure else "failed",
        "validation_failure_reason": str(validation_failure),
        "affine_support_type": str(affine_support),
        "finite_step_solver_method": str(chosen.get("solver_method", "")),
        "affine_expand_probe_count": int(chosen.get("affine_expand_probe_count", -1)),
        "affine_expand_sigma_G_bound": float(chosen.get("affine_expand_sigma_G_bound", float("nan"))),
        "affine_expand_reverse_G_bound_kind": str(chosen.get("affine_expand_reverse_G_bound_kind", "")),
        "G_limiter_sign_opposes_objective": bool(float(reverse.p1q1) < 0.0),
    }


def _validation_failure_reason(item: dict[str, Any], *, settings: PolicySettings) -> str:
    blockers = _constraint_blockers(item, settings=settings)
    if blockers:
        return ",".join(blockers)
    if not bool(item.get("feasible", False)):
        return "infeasible"
    return ""


def _normalized_g_boundary_fallback_mode(settings: PolicySettings) -> str:
    raw_mode = getattr(settings, "g_boundary_fallback_mode", G_BOUNDARY_FALLBACK_ENDPOINT_BRENTQ)
    mode = str(raw_mode).strip().lower().replace("-", "_")
    if mode in {
        G_BOUNDARY_FALLBACK_ENDPOINT_BRENTQ,
        "legacy",
        "off",
        "none",
        "false",
        "0",
    }:
        return G_BOUNDARY_FALLBACK_ENDPOINT_BRENTQ
    if mode in {
        G_BOUNDARY_FALLBACK_AFFINE_EXPAND_THEN_ENDPOINT_BRENTQ,
        "affine",
        "expand",
        "affine_expand",
        "affine_expand_then_legacy",
    }:
        return G_BOUNDARY_FALLBACK_AFFINE_EXPAND_THEN_ENDPOINT_BRENTQ
    raise ValueError(
        "g_boundary_fallback_mode must be 'endpoint_brentq' or "
        "'affine_expand_then_endpoint_brentq' "
        f"(got {raw_mode!r})."
    )


def _sign_aware_g_boundary_fallback(
    *,
    current: State,
    interval,
    endpoint: dict[str, Any],
    dx: float,
    config: CaseConfig,
    settings: PolicySettings,
) -> dict[str, Any] | None:
    endpoint_sigma = float(endpoint.get("sigma", float("nan")))
    endpoint_g = float(dict(endpoint.get("constraint_margins", {}) or {}).get("G", float("nan")))
    if not np.isfinite(endpoint_sigma) or not np.isfinite(endpoint_g):
        return None
    lo = float(interval.sigma_lo)
    hi = float(interval.sigma_hi)
    if not np.isfinite(lo) or not np.isfinite(hi) or lo > hi:
        return None

    cache: dict[float, dict[str, Any]] = {endpoint_sigma: endpoint}

    def evaluate(sigma: float) -> dict[str, Any]:
        key = float(sigma)
        if key not in cache:
            cache[key] = _evaluate_sigma(
                current=current,
                sigma=key,
                dx=dx,
                direction=-1,
                config=config,
                settings=settings,
            )
        return cache[key]

    if _normalized_g_boundary_fallback_mode(settings) == G_BOUNDARY_FALLBACK_AFFINE_EXPAND_THEN_ENDPOINT_BRENTQ:
        affine = _sign_aware_affine_expand_g_boundary_fallback(
            interval=interval,
            endpoint=endpoint,
            endpoint_sigma=endpoint_sigma,
            endpoint_g=endpoint_g,
            lo=lo,
            hi=hi,
            evaluate=evaluate,
            settings=settings,
        )
        if affine is not None:
            return affine

    candidates = [lo, hi]
    candidates = [sigma for sigma in candidates if abs(float(sigma) - endpoint_sigma) > float(settings.active_tol)]
    candidates.sort(key=lambda sigma: abs(float(sigma) - endpoint_sigma), reverse=True)
    for safe_sigma in candidates:
        safe = evaluate(float(safe_sigma))
        safe_g = float(dict(safe.get("constraint_margins", {}) or {}).get("G", float("nan")))
        if not np.isfinite(safe_g) or safe_g * endpoint_g > 0.0:
            continue
        try:
            sigma_root = float(
                brentq(
                    lambda sigma: float(evaluate(float(sigma))["constraint_margins"]["G"]),
                    endpoint_sigma,
                    float(safe_sigma),
                    xtol=1e-10,
                    rtol=1e-10,
                    maxiter=max(8, int(settings.refine_iterations)),
                )
            )
        except ValueError:
            continue
        root = evaluate(sigma_root)
        if bool(root.get("feasible", False)):
            return {
                **root,
                "boundary_blockers": ["G"],
                "boundary_bracket_width": abs(float(endpoint_sigma) - float(safe_sigma)),
                "boundary_infeasible_sigma": endpoint_sigma,
                "boundary_infeasible_margins": dict(endpoint.get("constraint_margins", {})),
                "solver_method": "sign_aware_brentq_G_boundary_fallback",
            }
    return None


def _sign_aware_affine_expand_g_boundary_fallback(
    *,
    interval,
    endpoint: dict[str, Any],
    endpoint_sigma: float,
    endpoint_g: float,
    lo: float,
    hi: float,
    evaluate: Callable[[float], dict[str, Any]],
    settings: PolicySettings,
) -> dict[str, Any] | None:
    kind = str(getattr(interval, "reverse_G_bound_kind", "")).strip().lower()
    if kind == "lower":
        if str(getattr(interval, "lower_source", "")) != "G_lower":
            return None
        side = 1.0
        limit = float(hi)
    elif kind == "upper":
        if str(getattr(interval, "upper_source", "")) != "G_upper":
            return None
        side = -1.0
        limit = float(lo)
    else:
        return None

    boundary = float(getattr(interval, "sigma_G_bound", float("nan")))
    tol = max(float(settings.active_tol), 0.0)
    if not np.isfinite(boundary) or boundary < lo - tol or boundary > hi + tol:
        return None
    boundary = float(np.clip(boundary, lo, hi))

    width = max(abs(float(hi) - float(lo)), tol, 1.0e-12)
    initial_step = max(width * 1.0e-4, tol * 2.0, 1.0e-12)
    max_probes = max(24, min(64, int(settings.refine_iterations) + 8))

    candidates: list[float] = [boundary]
    step = initial_step
    for _ in range(max_probes - 1):
        sigma = boundary + side * step
        if side > 0.0:
            if sigma >= limit:
                candidates.append(float(limit))
                break
        else:
            if sigma <= limit:
                candidates.append(float(limit))
                break
        candidates.append(float(sigma))
        step *= 2.0

    seen: set[float] = set()
    for safe_sigma in candidates:
        safe_sigma = float(np.clip(float(safe_sigma), lo, hi))
        if abs(safe_sigma - endpoint_sigma) <= tol:
            continue
        key = round(safe_sigma, 15)
        if key in seen:
            continue
        seen.add(key)
        safe = evaluate(safe_sigma)
        safe_g = float(dict(safe.get("constraint_margins", {}) or {}).get("G", float("nan")))
        if not np.isfinite(safe_g) or safe_g * endpoint_g > 0.0:
            continue
        try:
            sigma_root = float(
                brentq(
                    lambda sigma: float(evaluate(float(sigma))["constraint_margins"]["G"]),
                    endpoint_sigma,
                    safe_sigma,
                    xtol=1e-10,
                    rtol=1e-10,
                    maxiter=max(8, int(settings.refine_iterations)),
                )
            )
        except ValueError:
            continue
        root = evaluate(sigma_root)
        if bool(root.get("feasible", False)):
            return {
                **root,
                "boundary_blockers": ["G"],
                "boundary_bracket_width": abs(float(endpoint_sigma) - float(safe_sigma)),
                "boundary_infeasible_sigma": endpoint_sigma,
                "boundary_infeasible_margins": dict(endpoint.get("constraint_margins", {})),
                "solver_method": "sign_aware_affine_expand_G_boundary_fallback",
                "affine_expand_probe_count": len(seen),
                "affine_expand_sigma_G_bound": boundary,
                "affine_expand_reverse_G_bound_kind": kind,
            }
    return None


def _sign_aware_scan_fallback(
    *,
    current: State,
    interval,
    p1: float,
    dx: float,
    config: CaseConfig,
    settings: PolicySettings,
) -> dict[str, Any] | None:
    scan = []
    lo = float(interval.sigma_lo)
    hi = float(interval.sigma_hi)
    for sigma in np.linspace(lo, hi, max(int(settings.scan_points), 3)):
        scan.append(
            _evaluate_sigma(
                current=current,
                sigma=float(sigma),
                dx=dx,
                direction=-1,
                config=config,
                settings=settings,
            )
        )
    feasible = [item for item in scan if bool(item.get("feasible", False))]
    if not feasible:
        if not scan:
            return None
        best_failed = min(scan, key=lambda item: float(item.get("constraint_violation", 1e300)))
        return {**best_failed, "solver_method": "sign_aware_scan_no_feasible_sigma"}

    # RISK: Unlike endpoint selection, near-flat p1 falls to the upper side here; no tolerance/regularizer is applied.
    if float(p1) >= 0.0:
        chosen = max(feasible, key=lambda item: float(item["sigma"]))
        neighbor = _first_infeasible_neighbor(scan, chosen_sigma=float(chosen["sigma"]), side="upper")
    else:
        chosen = min(feasible, key=lambda item: float(item["sigma"]))
        neighbor = _first_infeasible_neighbor(scan, chosen_sigma=float(chosen["sigma"]), side="lower")
    if neighbor is not None:
        chosen = _refine_boundary(
            feasible_eval=chosen,
            infeasible_eval=neighbor,
            current=current,
            dx=dx,
            direction=-1,
            config=config,
            settings=settings,
        )
    return {**chosen, "solver_method": str(chosen.get("solver_method") or "sign_aware_scan_backtrack")}


def _direct_boundary_choice(
    *,
    current: State,
    lo: float,
    hi: float,
    pedal: float,
    dx: float,
    direction: int,
    config: CaseConfig,
    settings: PolicySettings,
) -> dict[str, Any] | None:
    """Choose a boundary without scanning when the preferred side is G-limited."""

    # REVIEW: Forward-scan shortcut only; sign-aware reverse uses its own endpoint validation/fallback path.
    cache: dict[float, dict[str, Any]] = {}

    def evaluate(sigma: float) -> dict[str, Any]:
        key = float(sigma)
        if key not in cache:
            item = _evaluate_sigma(
                current=current,
                sigma=key,
                dx=dx,
                direction=direction,
                config=config,
                settings=settings,
            )
            cache[key] = item
        return cache[key]

    if float(pedal) >= 0.0:
        preferred_sigma = float(hi)
        opposite_sigma = float(lo)
    else:
        preferred_sigma = float(lo)
        opposite_sigma = float(hi)

    preferred = evaluate(preferred_sigma)
    if bool(preferred["feasible"]):
        return {**preferred, "solver_method": "direct_preferred_endpoint"}

    preferred_blockers = _constraint_blockers(preferred, settings=settings)
    if "G" not in preferred_blockers:
        return None

    opposite = evaluate(opposite_sigma)
    if not bool(opposite["feasible"]):
        return None

    g_preferred = float(preferred["constraint_margins"]["G"])
    g_opposite = float(opposite["constraint_margins"]["G"])
    if g_preferred == 0.0:
        return {**preferred, "solver_method": "direct_preferred_endpoint"}
    if g_opposite == 0.0:
        return {**opposite, "solver_method": "direct_opposite_endpoint"}
    if g_preferred * g_opposite > 0.0:
        return None

    try:
        sigma_root = float(
            brentq(
                lambda sigma: float(evaluate(float(sigma))["constraint_margins"]["G"]),
                float(opposite_sigma),
                float(preferred_sigma),
                xtol=1e-10,
                rtol=1e-10,
                maxiter=max(8, int(settings.refine_iterations)),
            )
        )
    except ValueError:
        return None

    chosen = evaluate(sigma_root)
    if not bool(chosen.get("feasible", False)):
        return None
    return {
        **chosen,
        "boundary_blockers": ["G"],
        "boundary_bracket_width": abs(float(preferred_sigma) - float(opposite_sigma)),
        "boundary_infeasible_sigma": float(preferred_sigma),
        "boundary_infeasible_margins": dict(preferred.get("constraint_margins", {})),
        "solver_method": "brentq_G_boundary_fallback",
    }


def _evaluate_sigma(
    *,
    current: State,
    sigma: float,
    dx: float,
    direction: int,
    config: CaseConfig,
    settings: PolicySettings,
) -> dict[str, Any]:
    return _evaluate_sigma_impl(
        current=current,
        sigma=float(sigma),
        dx=dx,
        direction=direction,
        config=config,
        settings=settings,
        solve_next_state_fn=_solve_next_state_rk4,
        closure_metrics_fn=_closure_metrics,
        step_objective_payload_fn=_step_objective_payload,
    )


def _primitive_log_rhs(
    *,
    state: State,
    sigma: float,
    config: CaseConfig,
    rhs_mode: str = "raw",
) -> np.ndarray:
    return _primitive_log_rhs_impl(
        state=state,
        sigma=sigma,
        config=config,
        rhs_mode=rhs_mode,
        physics_params_fn=_physics_params,
    )


def _primitive_log_rhs_with_diagnostics(
    *,
    state: State,
    sigma: float,
    config: CaseConfig,
    rhs_mode: str,
) -> tuple[np.ndarray, dict[str, float | bool | str]]:
    return _primitive_log_rhs_with_diagnostics_impl(
        state=state,
        sigma=sigma,
        config=config,
        rhs_mode=rhs_mode,
        physics_params_fn=_physics_params,
    )


def _rk4_integrate_state(
    *,
    current: State,
    sigma: float,
    dx_signed: float,
    config: CaseConfig,
    substeps: int,
    rhs_mode: str,
    collect_diagnostics: bool,
) -> tuple[State, dict[str, float | int | bool | str]]:
    return _rk4_integrate_state_impl(
        current=current,
        sigma=sigma,
        dx_signed=dx_signed,
        config=config,
        substeps=substeps,
        rhs_mode=rhs_mode,
        collect_diagnostics=collect_diagnostics,
        physics_params_fn=_physics_params,
        state_factory=State,
    )


def _solve_next_state_rk4(
    *,
    current: State,
    logA_next: float,
    sigma: float,
    dx: float,
    direction: int,
    config: CaseConfig,
    settings: PolicySettings,
) -> tuple[State, dict[str, float | bool | int | str]]:
    return _solve_next_state_rk4_impl(
        current=current,
        logA_next=logA_next,
        sigma=sigma,
        dx=dx,
        direction=direction,
        config=config,
        settings=settings,
        physics_params_fn=_physics_params,
        state_factory=State,
    )


def _pedal_direction(
    *,
    current: State,
    base_sigma: float,
    lo: float,
    hi: float,
    dx: float,
    direction: int,
    config: CaseConfig,
    settings: PolicySettings,
) -> float:
    # REVIEW: Forward-scan objective probe; sign-aware reverse gets its direction from affine coefficients.
    eps = max(1e-5, 1e-4 * max(abs(hi - lo), 1.0))
    left = max(float(lo), float(base_sigma) - eps)
    right = min(float(hi), float(base_sigma) + eps)
    if right <= left:
        return 0.0
    left_eval = _evaluate_sigma(
        current=current,
        sigma=left,
        dx=dx,
        direction=direction,
        config=config,
        settings=settings,
    )
    right_eval = _evaluate_sigma(
        current=current,
        sigma=right,
        dx=dx,
        direction=direction,
        config=config,
        settings=settings,
    )
    return float(right_eval["objective_value"] - left_eval["objective_value"])


def _closure_metrics(state: State, *, config: CaseConfig) -> dict[str, float]:
    params = _physics_params(config)
    closure = closure_state_numba(
        float(state.n_p),
        float(state.T_e),
        float(state.area(config)),
        float(params.dot_N),
        float(params.I_0),
        float(params.seed_fraction),
        float(params.B),
        float(params.heavy_particle_mass_kg),
        float(params.seed_ionization_energy_J),
        float(params.sigma_ep),
    )
    T_p = float(closure[10])
    area = float(state.area(config))
    J_x = float(closure[14])
    E_x = float(closure[16])
    power_density = float(-area * J_x * E_x)
    return {
        "T_p": T_p,
        "Delta": float(state.T_e / max(T_p, 1e-300) - 1.0),
        "G": float(closure[18]),
        "mach": float(closure[17]),
        "beta": float(closure[5]),
        "Z": float(closure[7]),
        "n_e": float(closure[4]),
        "v_p": float(closure[9]),
        "J_x": J_x,
        "J_y": float(closure[15]),
        "E_x": E_x,
        "hall_field_V_per_m": float(-E_x),
        "power_density_W_per_m": power_density,
    }


def _normalized_objective(settings: PolicySettings) -> str:
    objective = str(getattr(settings, "objective", "delta_gain")).strip().lower().replace("-", "_")
    # REVIEW: Add an enthalpy objective for reverse preparation that minimizes
    # upstream enthalpy rise.
    aliases = {
        "delta": "delta_gain",
        "delta_gain": "delta_gain",
        "delta_rise": "delta_gain",
        "delta_drop": "delta_drop",
        "power": "power_next",
        "power_next": "power_next",
        "power_output": "power_next",
        "power_output_next": "power_next",
        "mhd_power": "power_next",
        "mhd_power_next": "power_next",
    }
    if objective not in aliases:
        raise ValueError(
            "policy objective must be one of delta_gain, delta_drop, or power_next "
            f"(got {getattr(settings, 'objective', None)!r})."
        )
    return aliases[objective]


def _step_objective_payload(
    *,
    current_metrics: dict[str, float],
    next_metrics: dict[str, float],
    dx: float,
    settings: PolicySettings,
) -> dict[str, float | str]:
    objective_kind = _normalized_objective(settings)
    delta_gain = float(next_metrics["Delta"] - current_metrics["Delta"])
    power_current = float(current_metrics.get("power_density_W_per_m", float("nan")))
    power_next = float(next_metrics.get("power_density_W_per_m", float("nan")))
    power_gain = float(power_next - power_current)
    step_power = float(0.5 * abs(float(dx)) * (power_current + power_next))
    if objective_kind == "delta_drop":
        objective_value = -delta_gain
    elif objective_kind == "power_next":
        objective_value = step_power
    else:
        objective_value = delta_gain
    return {
        "objective_kind": objective_kind,
        "objective_value": float(objective_value),
        "delta_gain": delta_gain,
        "power_next_W_per_m": power_next,
        "power_density_gain_W_per_m": power_gain,
        "step_power_W": step_power,
    }


def _node_payload(
    k: int,
    state: State,
    *,
    config: CaseConfig,
    sigma: float | None,
    seed_index: int,
    x: float,
) -> dict[str, float | int]:
    metrics = _closure_metrics(state, config=config)
    sigma_value = float("nan") if sigma is None else float(sigma)
    return {
        "k": int(k),
        "x": float(x),
        "seed_index": int(seed_index),
        "n_p": float(state.n_p),
        "T_e": float(state.T_e),
        "A": float(state.area(config)),
        "logA": float(state.logA),
        "sigma_logA": sigma_value,
        **metrics,
    }


def _first_infeasible_neighbor(scan: list[dict[str, Any]], *, chosen_sigma: float, side: str) -> dict[str, Any] | None:
    if side == "upper":
        candidates = [item for item in scan if float(item["sigma"]) > float(chosen_sigma) and not bool(item["feasible"])]
        return min(candidates, key=lambda item: float(item["sigma"])) if candidates else None
    candidates = [item for item in scan if float(item["sigma"]) < float(chosen_sigma) and not bool(item["feasible"])]
    return max(candidates, key=lambda item: float(item["sigma"])) if candidates else None


def _refine_boundary(
    *,
    feasible_eval: dict[str, Any],
    infeasible_eval: dict[str, Any],
    current: State,
    dx: float,
    direction: int,
    config: CaseConfig,
    settings: PolicySettings,
) -> dict[str, Any]:
    feasible = feasible_eval
    infeasible = infeasible_eval
    blockers = _constraint_blockers(infeasible, settings=settings)
    if "G" in blockers:
        g_feasible = float(dict(feasible.get("constraint_margins", {}) or {}).get("G", float("nan")))
        g_infeasible = float(dict(infeasible.get("constraint_margins", {}) or {}).get("G", float("nan")))
        if np.isfinite(g_feasible) and np.isfinite(g_infeasible) and g_feasible * g_infeasible <= 0.0:
            cache: dict[float, dict[str, Any]] = {
                float(feasible["sigma"]): feasible,
                float(infeasible["sigma"]): infeasible,
            }

            def evaluate(sigma: float) -> dict[str, Any]:
                key = float(sigma)
                if key not in cache:
                    cache[key] = _evaluate_sigma(
                        current=current,
                        sigma=key,
                        dx=dx,
                        direction=direction,
                        config=config,
                        settings=settings,
                    )
                return cache[key]

            try:
                sigma_root = float(
                    brentq(
                        lambda sigma: float(evaluate(float(sigma))["constraint_margins"]["G"]),
                        float(feasible["sigma"]),
                        float(infeasible["sigma"]),
                        xtol=1e-10,
                        rtol=1e-10,
                        maxiter=max(8, int(settings.refine_iterations)),
                    )
                )
                root = evaluate(sigma_root)
                if bool(root.get("feasible", False)):
                    return {
                        **root,
                        "boundary_blockers": ["G"],
                        "boundary_bracket_width": abs(float(feasible["sigma"]) - float(infeasible["sigma"])),
                        "boundary_infeasible_sigma": float(infeasible["sigma"]),
                        "boundary_infeasible_margins": dict(infeasible.get("constraint_margins", {})),
                        "solver_method": "brentq_G_boundary_refine",
                    }
            except ValueError:
                pass

    for _ in range(int(settings.refine_iterations)):
        mid_sigma = 0.5 * (float(feasible["sigma"]) + float(infeasible["sigma"]))
        mid = _evaluate_sigma(
            current=current,
            sigma=mid_sigma,
            dx=dx,
            direction=direction,
            config=config,
            settings=settings,
        )
        if bool(mid["feasible"]):
            feasible = mid
        else:
            infeasible = mid
    return {
        **feasible,
        "boundary_blockers": _constraint_blockers(infeasible, settings=settings),
        "boundary_bracket_width": abs(float(feasible["sigma"]) - float(infeasible["sigma"])),
        "boundary_infeasible_sigma": float(infeasible["sigma"]),
        "boundary_infeasible_margins": dict(infeasible.get("constraint_margins", {})),
    }


def _constraint_blockers(item: dict[str, Any], *, settings: PolicySettings) -> list[str]:
    blockers: list[str] = []
    margins = dict(item.get("constraint_margins", {}))
    tol = float(settings.active_tol)
    for name, value in margins.items():
        margin = float(value)
        if not np.isfinite(margin) or margin < -tol:
            blockers.append(str(name))
    if not bool(item.get("ok", False)):
        blockers.append("solver")
    return blockers


def _classify_support(
    *,
    chosen: dict[str, Any],
    lo: float,
    hi: float,
    bound_sources: dict[str, str],
    settings: PolicySettings,
) -> str:
    sigma = float(chosen["sigma"])
    tol = float(settings.active_tol)
    margins = dict(chosen.get("constraint_margins", {}))
    boundary_blockers = set(str(name) for name in chosen.get("boundary_blockers", []))
    if abs(sigma - lo) <= tol:
        return f"{bound_sources.get('lower', 'lower')}_lower_supported"
    if abs(sigma - hi) <= tol:
        return f"{bound_sources.get('upper', 'upper')}_upper_supported"
    if "G" in boundary_blockers:
        return "G_supported"
    if "Tp" in boundary_blockers:
        return "Tp_floor_supported"
    if "residual" in boundary_blockers:
        return "residual_supported"
    if "solver" in boundary_blockers:
        return "local_solver_supported"
    if float(margins.get("G", 1.0)) <= tol:
        return "G_supported"
    if float(margins.get("Tp", 1.0)) <= tol:
        return "Tp_floor_supported"
    return "interior"


def _direction_sign(direction: str) -> int:
    normalized = str(direction).strip().lower()
    if normalized in {"forward", "downstream", "+", "+1"}:
        return 1
    if normalized in {"reverse", "backward", "upstream", "-", "-1"}:
        return -1
    raise ValueError("direction must be forward or reverse.")
