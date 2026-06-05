from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import least_squares

from v6_firedrake_reduced.design import CaseConfig
from v6_firedrake_reduced.sonic_compatibility import solve_local_sonic_match

from .numba_physics import dynamic_terms_numba
from .policy import State, _closure_metrics, _physics_params


@dataclass(frozen=True)
class SonicDeltaSettings:
    dx: float = 1.0e-4
    n_steps_each_side: int = 2
    sigma_min: float = -0.5
    sigma_max: float = 0.5
    curvature_max: float | None = None
    g_floor: float = 0.0
    tp_floor_K: float = 300.0
    scan_points: int = 41
    residual_tol: float = 1.0e-6
    active_tol: float = 1.0e-7
    branch_mach_tol: float = 1.0e-7
    branch_mode: str = "fixed"
    objective: str = "pedal"
    selection_mode: str = "continuation"
    target_mach_slope_1_per_m: float = 20.0
    target_mach_offset_max: float = 0.015


def build_sonic_delta_profile(
    *,
    config: CaseConfig,
    settings: SonicDeltaSettings | None = None,
) -> dict[str, Any]:
    """Build a local profile through M=1 with the primitive sonic condition.

    The active-boundary state equation is singular at M=1.  At the sonic node
    this routine fixes dlogA/dx from the left-null compatibility condition,
    then marches away from that node using a trapezoidal sigma update.  Each
    step selects the admissible root with the steepest Delta change while
    preserving G >= g_floor.
    """

    opts = settings or SonicDeltaSettings()
    if float(opts.dx) <= 0.0:
        raise ValueError("settings.dx must be positive.")
    seed = _default_sonic_seed(config=config)
    sonic_state = seed["state"]
    sonic = primitive_sonic_compatibility(sonic_state, config=config)
    sigma_star = float(sonic["sigma_sonic"])
    sonic_metrics = _closure_metrics(sonic_state, config=config)
    sonic_node = _node_payload(
        k=0,
        x=0.0,
        state=sonic_state,
        sigma=sigma_star,
        side="sonic",
        branch="sonic",
        segment_index=-1,
        config=config,
    )

    branch_mode = _branch_mode(opts)
    left_branch = "agnostic" if branch_mode == "agnostic" else "supersonic"
    right_branch = "agnostic" if branch_mode == "agnostic" else "subsonic"

    left_nodes, left_segments = _march_side(
        config=config,
        settings=opts,
        start=sonic_state,
        sigma_start=sigma_star,
        direction=-1,
        branch=left_branch,
        side="left",
    )
    right_nodes, right_segments = _march_side(
        config=config,
        settings=opts,
        start=sonic_state,
        sigma_start=sigma_star,
        direction=1,
        branch=right_branch,
        side="right",
    )

    nodes = list(reversed(left_nodes)) + [sonic_node] + right_nodes
    for idx, node in enumerate(nodes):
        node["k"] = int(idx)
    segments = list(reversed(left_segments)) + right_segments
    for idx, segment in enumerate(segments):
        segment["segment_index"] = int(idx)

    mach_values = np.asarray([float(node["mach"]) for node in nodes], dtype=float)
    g_margins = np.asarray([float(node["G"]) - float(opts.g_floor) for node in nodes], dtype=float)
    delta_values = np.asarray([float(node["Delta"]) for node in nodes], dtype=float)
    mach_crosses_sonic = bool(np.nanmin(mach_values) < 1.0 and np.nanmax(mach_values) > 1.0)
    direction_violations = _direction_violation_counts(segments=segments, active_tol=float(opts.active_tol))
    ok = bool(
        len(nodes) >= 3
        and (branch_mode == "agnostic" or mach_crosses_sonic)
        and np.nanmin(g_margins) >= -float(opts.active_tol)
        and all(bool(segment.get("ok", False)) for segment in segments)
    )
    return {
        "ok": ok,
        "mode": "sonic_delta_profile",
        "settings": opts.__dict__,
        "case_config": config.to_dict(),
        "sonic_seed": _json_clean(seed["summary"]),
        "sonic_primitive_compatibility": _json_clean(sonic),
        "active_summary": {
            "mach_min": float(np.nanmin(mach_values)),
            "mach_max": float(np.nanmax(mach_values)),
            "G_min": float(np.nanmin([float(node["G"]) for node in nodes])),
            "G_margin_min": float(np.nanmin(g_margins)),
            "Delta_min": float(np.nanmin(delta_values)),
            "Delta_max": float(np.nanmax(delta_values)),
            "max_abs_delta_change_per_m": float(
                max((abs(float(seg["delta_change"])) / max(float(seg["dx"]), 1.0e-300)) for seg in segments)
            )
            if segments
            else 0.0,
            "sonic_mach": float(sonic_metrics["mach"]),
            "sonic_G": float(sonic_metrics["G"]),
            "sonic_Delta": float(sonic_metrics["Delta"]),
            "selection_mode": str(opts.selection_mode),
            "branch_mode": branch_mode,
            "mach_crosses_sonic": mach_crosses_sonic,
            **direction_violations,
        },
        "nodes": nodes,
        "segments": segments,
        "profile_arrays": {
            "x": np.asarray([float(node["x"]) for node in nodes], dtype=float),
            "n_p": np.asarray([float(node["n_p"]) for node in nodes], dtype=float),
            "T_e": np.asarray([float(node["T_e"]) for node in nodes], dtype=float),
            "A": np.asarray([float(node["A"]) for node in nodes], dtype=float),
            "sigma_logA": np.asarray([float(node["sigma_logA"]) for node in nodes], dtype=float),
            "mach": mach_values,
            "G": np.asarray([float(node["G"]) for node in nodes], dtype=float),
            "Delta": delta_values,
            "T_p": np.asarray([float(node["T_p"]) for node in nodes], dtype=float),
        },
    }


def primitive_sonic_compatibility(state: State, *, config: CaseConfig) -> dict[str, Any]:
    params = _physics_params(config)
    area = float(state.area(config))
    terms = dynamic_terms_numba(
        float(state.n_p),
        float(state.T_e),
        area,
        0.0,
        float(params.dot_N),
        float(params.I_0),
        float(params.seed_fraction),
        float(params.B),
        float(params.heavy_particle_mass_kg),
        float(params.seed_ionization_energy_J),
        float(params.sigma_ep),
    )
    matrix = np.array([[float(terms[0]), float(terms[1])], [float(terms[3]), float(terms[4])]], dtype=float)
    f0 = np.array([float(terms[7]), float(terms[8])], dtype=float)
    f1 = np.array([-float(terms[2]), -float(terms[5])], dtype=float)
    u, singular_values, _ = np.linalg.svd(matrix)
    left_null = np.asarray(u[:, -1], dtype=float)
    numerator = float(left_null @ f0)
    denominator = float(left_null @ f1)
    if abs(denominator) <= 1.0e-300 or area <= 0.0:
        sigma = float("nan")
    else:
        sigma = float(-numerator / (area * denominator))
    residual = float(left_null @ (f0 + area * sigma * f1)) if np.isfinite(sigma) else float("nan")
    metrics = _closure_metrics(state, config=config)
    return {
        "mach": float(metrics["mach"]),
        "det_D": float(np.linalg.det(matrix)),
        "singular_values": singular_values,
        "left_null": left_null,
        "ellTf0": numerator,
        "ellTf1": denominator,
        "A": area,
        "A_prime_sonic": float(area * sigma) if np.isfinite(sigma) else float("nan"),
        "sigma_sonic": sigma,
        "compatibility_residual": residual,
        "compatibility_scaled_residual": residual / max(1.0, abs(numerator), abs(area * sigma * denominator))
        if np.isfinite(residual) and np.isfinite(sigma)
        else float("nan"),
        "formula": "ell^T(f0 + A*sigma*f1) = 0 at M=1",
    }


def _default_sonic_seed(*, config: CaseConfig) -> dict[str, Any]:
    match = solve_local_sonic_match(design=config.design, config=config)
    sonic = dict(match.get("sonic_point") or {})
    if not sonic:
        raise RuntimeError("solve_local_sonic_match did not return a sonic_point.")
    state = State(
        log_n=float(np.log(max(float(sonic["n_p"]), 1.0e-300))),
        log_Te=float(np.log(max(float(sonic["T_e"]), 1.0))),
        logA=float(sonic["logA"]),
    )
    return {
        "state": state,
        "summary": {
            "source": "v6_firedrake_reduced.solve_local_sonic_match",
            "source_ok": bool(match.get("ok", False)),
            "source_reference_ok": bool(match.get("reference_ok", False)),
            "source_reference_error": match.get("reference_error"),
            "source_scaled_residual_inf": match.get("selected_solution", {}).get("scaled_residual_inf"),
            "x_m": sonic.get("x_m"),
            "x_fraction": sonic.get("x_fraction"),
            "source_sigma_logA": sonic.get("sigma_required_1_per_m"),
            "n_p": sonic.get("n_p"),
            "T_e": sonic.get("T_e"),
            "T_p": sonic.get("T_p_K"),
            "A": sonic.get("A_m2"),
            "logA": sonic.get("logA"),
            "mach": sonic.get("mach"),
        },
    }


def _march_side(
    *,
    config: CaseConfig,
    settings: SonicDeltaSettings,
    start: State,
    sigma_start: float,
    direction: int,
    branch: str,
    side: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    nodes: list[dict[str, Any]] = []
    segments: list[dict[str, Any]] = []
    current = start
    sigma_current = float(sigma_start)
    x_current = 0.0
    for step_index in range(int(settings.n_steps_each_side)):
        step = _select_step(
            current=current,
            sigma_current=sigma_current,
            x_current=x_current,
            direction=int(direction),
            branch=branch,
            side=side,
            config=config,
            settings=settings,
        )
        segments.append({key: value for key, value in step.items() if key != "next_state"})
        if not bool(step.get("ok", False)):
            break
        current = step["next_state"]
        sigma_current = float(step["sigma_next"])
        x_current = float(step["x_next"])
        nodes.append(
            _node_payload(
                k=step_index + 1,
                x=x_current,
                state=current,
                sigma=sigma_current,
                side=side,
                branch=branch,
                segment_index=step_index,
                config=config,
            )
        )
    return nodes, segments


def _select_step(
    *,
    current: State,
    sigma_current: float,
    x_current: float,
    direction: int,
    branch: str,
    side: str,
    config: CaseConfig,
    settings: SonicDeltaSettings,
) -> dict[str, Any]:
    sigma_values = _sigma_grid(sigma_current=sigma_current, settings=settings)
    candidates: list[dict[str, Any]] = []
    for sigma_next in sigma_values:
        candidates.extend(
            _candidate_roots(
                current=current,
                sigma_current=float(sigma_current),
                sigma_next=float(sigma_next),
                dx=float(settings.dx),
                direction=int(direction),
                branch=branch,
                config=config,
                settings=settings,
            )
        )
    feasible = [item for item in candidates if bool(item["feasible"])]
    branch_filter_enabled = bool(_branch_mode(settings) == "fixed" and str(branch) in {"subsonic", "supersonic"})
    branched = [item for item in feasible if _branch_ok(float(item["mach"]), branch=branch, settings=settings)]
    selectable = (branched if branched else feasible) if branch_filter_enabled else feasible
    if not selectable:
        best_failed = min(candidates, key=lambda item: float(item.get("constraint_violation", 1.0e300)), default=None)
        return {
            "ok": False,
            "side": side,
            "branch": branch,
            "x_current": float(x_current),
            "x_next": float(x_current + int(direction) * float(settings.dx)),
            "dx": float(settings.dx),
            "sigma_current": float(sigma_current),
            "sigma_next": float("nan"),
            "support_type": "no_admissible_sonic_delta_step",
            "candidate_count": int(len(candidates)),
            "feasible_candidate_count": int(len(feasible)),
            "branch_candidate_count": int(len(branched)),
            "branch_filter_enabled": branch_filter_enabled,
            "error": "no root satisfied residual and active constraints",
            "best_failed": _candidate_public(best_failed) if best_failed is not None else {},
        }
    steepest = max(selectable, key=lambda item: _objective_score(item, settings=settings, direction=direction))
    if str(settings.selection_mode).strip().lower() == "steepest":
        selected = steepest
    elif str(settings.selection_mode).strip().lower() == "continuation":
        selected = max(
            selectable,
            key=lambda item: _continuation_score(
                item,
                current=current,
                sigma_current=float(sigma_current),
                x_next=float(x_current + int(direction) * float(settings.dx)),
                branch=branch,
                settings=settings,
            ),
        )
    else:
        raise ValueError("selection_mode must be continuation or steepest.")
    branch_selected = bool(_branch_ok(float(selected["mach"]), branch=branch, settings=settings))
    selected_mach_branch = _mach_branch(float(selected["mach"]), settings=settings)
    return {
        "ok": True,
        "side": side,
        "branch": branch,
        "x_current": float(x_current),
        "x_next": float(x_current + int(direction) * float(settings.dx)),
        "dx": float(settings.dx),
        "sigma_current": float(sigma_current),
        "sigma_next": float(selected["sigma_next"]),
        "sigma_average": float(0.5 * (float(sigma_current) + float(selected["sigma_next"]))),
        "next_state": selected["next_state"],
        "support_type": "sonic_compatible_steepest_delta",
        "candidate_count": int(len(candidates)),
        "feasible_candidate_count": int(len(feasible)),
        "branch_candidate_count": int(len(branched)),
        "branch_filter_enabled": branch_filter_enabled,
        "branch_selected": branch_selected,
        "selected_mach_branch": selected_mach_branch,
        "objective": str(settings.objective),
        "objective_score": float(_objective_score(selected, settings=settings, direction=direction)),
        "steepest_candidate": _candidate_public(steepest),
        **_candidate_public(selected),
    }


def _sigma_grid(*, sigma_current: float, settings: SonicDeltaSettings) -> np.ndarray:
    lo = float(settings.sigma_min)
    hi = float(settings.sigma_max)
    if settings.curvature_max is not None and np.isfinite(float(settings.curvature_max)):
        width = abs(float(settings.curvature_max)) * float(settings.dx)
        lo = max(lo, float(sigma_current) - width)
        hi = min(hi, float(sigma_current) + width)
    if lo > hi:
        return np.asarray([], dtype=float)
    values = np.linspace(lo, hi, max(int(settings.scan_points), 3), dtype=float)
    return np.unique(np.concatenate([values, np.asarray([float(sigma_current)], dtype=float)]))


def _candidate_roots(
    *,
    current: State,
    sigma_current: float,
    sigma_next: float,
    dx: float,
    direction: int,
    branch: str,
    config: CaseConfig,
    settings: SonicDeltaSettings,
) -> list[dict[str, Any]]:
    roots: list[dict[str, Any]] = []
    for initial in _initial_guesses(
        current=current,
        sigma_current=sigma_current,
        sigma_next=sigma_next,
        dx=dx,
        direction=direction,
        branch=branch,
    ):
        next_state, residual = _solve_next_state_trapezoid(
            current=current,
            sigma_current=sigma_current,
            sigma_next=sigma_next,
            dx=dx,
            direction=direction,
            config=config,
            initial=initial,
        )
        if any(
            abs(float(next_state.log_n) - float(item["next_state"].log_n)) <= 1.0e-8
            and abs(float(next_state.log_Te) - float(item["next_state"].log_Te)) <= 1.0e-8
            for item in roots
        ):
            continue
        metrics = _closure_metrics(next_state, config=config)
        current_metrics = _closure_metrics(current, config=config)
        margins = {
            "G": float(metrics["G"] - float(settings.g_floor)),
            "Tp": float(metrics["T_p"] - float(settings.tp_floor_K)),
            "residual": float(settings.residual_tol - float(residual["max_abs_scaled_residual"])),
        }
        feasible = bool(bool(residual["ok"]) and all(value >= -float(settings.active_tol) for value in margins.values()))
        violation = float(sum(max(-float(value), 0.0) for value in margins.values()))
        roots.append(
            {
                "next_state": next_state,
                "sigma_next": float(sigma_next),
                "sigma_current": float(sigma_current),
                "feasible": feasible,
                "constraint_margins": margins,
                "constraint_violation": violation,
                "delta_change": float(metrics["Delta"] - current_metrics["Delta"]),
                **metrics,
                **residual,
            }
        )
    return roots


def _solve_next_state_trapezoid(
    *,
    current: State,
    sigma_current: float,
    sigma_next: float,
    dx: float,
    direction: int,
    config: CaseConfig,
    initial: State,
) -> tuple[State, dict[str, Any]]:
    params = _physics_params(config)
    dx_signed = float(direction) * float(dx)
    logA_next = float(current.logA + dx_signed * 0.5 * (float(sigma_current) + float(sigma_next)))
    area_next = float(params.area_scale_m2) * float(np.exp(np.clip(logA_next, -700.0, 700.0)))

    def residual(y: np.ndarray) -> np.ndarray:
        log_n_next = float(y[0])
        log_Te_next = float(y[1])
        n_next = float(np.exp(np.clip(log_n_next, -700.0, 700.0)))
        te_next = float(np.exp(np.clip(log_Te_next, -700.0, 700.0)))
        terms = dynamic_terms_numba(
            n_next,
            te_next,
            area_next,
            float(sigma_next),
            float(params.dot_N),
            float(params.I_0),
            float(params.seed_fraction),
            float(params.B),
            float(params.heavy_particle_mass_kg),
            float(params.seed_ionization_energy_J),
            float(params.sigma_ep),
        )
        dn_dx = (n_next - current.n_p) / dx_signed
        dte_dx = (te_next - current.T_e) / dx_signed
        momentum = float(terms[0]) * dn_dx + float(terms[1]) * dte_dx - float(terms[7])
        energy = float(terms[3]) * dn_dx + float(terms[4]) * dte_dx - float(terms[8])
        m_scale = max(
            1.0,
            abs(float(terms[0]) * max(n_next, 1.0) / max(abs(dx_signed), 1.0e-300)),
            abs(float(terms[1]) * max(te_next, 1.0) / max(abs(dx_signed), 1.0e-300)),
            abs(float(terms[7])),
        )
        e_scale = max(
            1.0,
            abs(float(terms[3]) * max(n_next, 1.0) / max(abs(dx_signed), 1.0e-300)),
            abs(float(terms[4]) * max(te_next, 1.0) / max(abs(dx_signed), 1.0e-300)),
            abs(float(terms[8])),
        )
        return np.array([momentum / m_scale, energy / e_scale], dtype=float)

    guess = np.array([float(initial.log_n), float(initial.log_Te)], dtype=float)
    sol = least_squares(residual, guess, xtol=1.0e-11, ftol=1.0e-11, gtol=1.0e-11, max_nfev=80)
    res = residual(np.asarray(sol.x, dtype=float))
    max_abs = float(np.max(np.abs(res))) if res.size else float("inf")
    return (
        State(log_n=float(sol.x[0]), log_Te=float(sol.x[1]), logA=logA_next),
        {
            "ok": bool(sol.success and np.isfinite(max_abs)),
            "residual_ok": bool(sol.success and max_abs <= 1.0e-7),
            "max_abs_scaled_residual": max_abs,
            "least_squares_cost": float(sol.cost),
            "least_squares_nfev": int(sol.nfev),
        },
    )


def _initial_guesses(
    *,
    current: State,
    sigma_current: float,
    sigma_next: float,
    dx: float,
    direction: int,
    branch: str,
) -> list[State]:
    logA_next = float(current.logA + float(direction) * float(dx) * 0.5 * (float(sigma_current) + float(sigma_next)))
    pairs = [(0.0, 0.0)]
    signs = (-1.0, 1.0) if str(branch) == "agnostic" else (-1.0 if str(branch) == "supersonic" else 1.0,)
    for sign in signs:
        for magnitude in (1.0e-6, 1.0e-4, 1.0e-3, 5.0e-3, 1.0e-2):
            pairs.append((sign * magnitude, -sign * magnitude))
            pairs.append((-sign * magnitude, sign * magnitude))
    for magnitude in (1.0e-3, 5.0e-3):
        pairs.extend(
            [
                (magnitude, magnitude),
                (-magnitude, -magnitude),
                (magnitude, -magnitude),
                (-magnitude, magnitude),
            ]
        )
    out: list[State] = []
    seen: set[tuple[float, float]] = set()
    for dn, dte in pairs:
        key = (round(float(dn), 12), round(float(dte), 12))
        if key in seen:
            continue
        seen.add(key)
        out.append(State(log_n=float(current.log_n + dn), log_Te=float(current.log_Te + dte), logA=logA_next))
    return out


def _branch_ok(mach: float, *, branch: str, settings: SonicDeltaSettings) -> bool:
    tol = float(settings.branch_mach_tol)
    if str(branch) == "subsonic":
        return float(mach) <= 1.0 - tol
    if str(branch) == "supersonic":
        return float(mach) >= 1.0 + tol
    return True


def _branch_mode(settings: SonicDeltaSettings) -> str:
    mode = str(settings.branch_mode).strip().lower()
    if mode not in {"fixed", "agnostic"}:
        raise ValueError("branch_mode must be fixed or agnostic.")
    return mode


def _mach_branch(mach: float, *, settings: SonicDeltaSettings) -> str:
    tol = float(settings.branch_mach_tol)
    if float(mach) <= 1.0 - tol:
        return "subsonic"
    if float(mach) >= 1.0 + tol:
        return "supersonic"
    return "sonic_near"


def _objective_score(item: dict[str, Any], *, settings: SonicDeltaSettings, direction: int = 0) -> float:
    change = float(item["delta_change"])
    objective = str(settings.objective).strip().lower()
    if objective in {"pedal", "directional", "reverse_drop_forward_rise"}:
        if int(direction) < 0:
            return -change
        if int(direction) > 0:
            return change
        return abs(change)
    if objective == "drop":
        return -change
    if objective == "rise":
        return change
    if objective == "abs":
        return abs(change)
    raise ValueError("objective must be one of: pedal, abs, drop, rise.")


def _continuation_score(
    item: dict[str, Any],
    *,
    current: State,
    sigma_current: float,
    x_next: float,
    branch: str,
    settings: SonicDeltaSettings,
) -> float:
    next_state = item["next_state"]
    if str(branch) == "agnostic":
        mach_penalty = abs(float(item["mach"]) - 1.0)
    else:
        branch_sign = -1.0 if str(branch) == "subsonic" else 1.0
        target_offset = min(
            float(settings.target_mach_offset_max),
            max(float(settings.branch_mach_tol) * 10.0, abs(float(x_next)) * float(settings.target_mach_slope_1_per_m)),
        )
        target_mach = 1.0 + branch_sign * target_offset
        mach_penalty = abs(float(item["mach"]) - target_mach)
    sigma_penalty = 0.1 * abs(float(item["sigma_next"]) - float(sigma_current))
    state_penalty = 2.0 * float(
        np.hypot(
            float(next_state.log_n) - float(current.log_n),
            float(next_state.log_Te) - float(current.log_Te),
        )
    )
    return -float(mach_penalty + sigma_penalty + state_penalty)


def _direction_violation_counts(*, segments: list[dict[str, Any]], active_tol: float) -> dict[str, int]:
    reverse_bad = 0
    forward_bad = 0
    for segment in segments:
        if not bool(segment.get("ok", False)):
            continue
        change = float(segment["delta_change"])
        x_current = float(segment["x_current"])
        x_next = float(segment["x_next"])
        if x_next < x_current and change > float(active_tol):
            reverse_bad += 1
        elif x_next > x_current and change < -float(active_tol):
            forward_bad += 1
    return {
        "reverse_direction_violation_count": int(reverse_bad),
        "forward_direction_violation_count": int(forward_bad),
    }


def _candidate_public(item: dict[str, Any] | None) -> dict[str, Any]:
    if item is None:
        return {}
    return {
        "sigma_next": float(item.get("sigma_next", float("nan"))),
        "delta_change": float(item.get("delta_change", float("nan"))),
        "Delta": float(item.get("Delta", float("nan"))),
        "G": float(item.get("G", float("nan"))),
        "T_p": float(item.get("T_p", float("nan"))),
        "mach": float(item.get("mach", float("nan"))),
        "constraint_margins": dict(item.get("constraint_margins", {}) or {}),
        "constraint_violation": float(item.get("constraint_violation", float("nan"))),
        "max_abs_scaled_residual": float(item.get("max_abs_scaled_residual", float("nan"))),
        "least_squares_nfev": int(item.get("least_squares_nfev", -1)),
    }


def _node_payload(
    *,
    k: int,
    x: float,
    state: State,
    sigma: float,
    side: str,
    branch: str,
    segment_index: int,
    config: CaseConfig,
) -> dict[str, Any]:
    metrics = _closure_metrics(state, config=config)
    return {
        "k": int(k),
        "x": float(x),
        "side": str(side),
        "branch": str(branch),
        "segment_index": int(segment_index),
        "n_p": float(state.n_p),
        "T_e": float(state.T_e),
        "A": float(state.area(config)),
        "log_n": float(state.log_n),
        "log_Te": float(state.log_Te),
        "logA": float(state.logA),
        "sigma_logA": float(sigma),
        **metrics,
    }


def _json_clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_clean(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_clean(item) for item in value.reshape(-1).tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        out = float(value)
        return out if np.isfinite(out) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value
