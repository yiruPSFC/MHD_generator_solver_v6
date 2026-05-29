from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import brentq, least_squares

from v6_firedrake_reduced.design import CaseConfig
from v6_firedrake_reduced.geometry import LogAreaSplineControl
from v6_firedrake_reduced.legacy_physics import inlet_design_generic, ops_for_numeric
from v6_firedrake_reduced.transport import working_fluid_for_config

from .numba_physics import closure_state_numba, dynamic_terms_numba, g_boundary_residual_numba


@dataclass(frozen=True)
class PolicySettings:
    direction: str = "forward"
    objective: str = "delta_gain"
    n_steps: int = 12
    stride: int = 5
    start_index: int | None = None
    sigma_min: float = -0.5
    sigma_max: float = 0.5
    curvature_max: float | None = 0.05
    g_floor: float = 0.0
    tp_floor_K: float = 300.0
    scan_points: int = 41
    refine_iterations: int = 24
    active_tol: float = 1e-6
    residual_tol: float = 1e-8


@dataclass(frozen=True)
class PreparationSettings:
    n_steps: int = 60
    dx: float = 0.01
    sigma_min: float = -0.5
    sigma_max: float = 0.5
    curvature_max: float | None = 0.05
    g_floor: float = 0.0
    tp_floor_K: float = 300.0
    scan_points: int = 41
    refine_iterations: int = 24
    active_tol: float = 1e-6
    residual_tol: float = 1e-8

    @property
    def objective(self) -> str:
        return "delta_drop"


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
    sigma_logA: float
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
    if "sigma_logA" not in payload:
        raise ValueError("anchor JSON must contain sigma_logA for curvature initialization.")
    return AnchorState(
        state=State(log_n=log_n, log_Te=log_Te, logA=logA),
        sigma_logA=float(payload["sigma_logA"]),
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
    ops = ops_for_numeric()
    fluid = working_fluid_for_config(config)
    inlet = inlet_design_generic(
        ops=ops,
        n_p_in=design.n_p_in,
        T_e_in=design.T_e_in,
        Z_in=design.Z_in,
        I_0=design.I_0,
        seed_fraction=design.seed_fraction,
        B=float(design.B_T),
        inlet_A=float(config.area_scale_m2),
        working_fluid=fluid,
    )
    params = PhysicsParams(
        dot_N=float(inlet["dot_N"]),
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


def rollout_policy(
    *,
    config: CaseConfig,
    profile: dict[str, Any],
    settings: PolicySettings,
) -> dict[str, Any]:
    window = _window_from_profile(profile, settings=settings)
    direction = _direction_sign(settings.direction)
    dx = float(window["dx"])
    seed_states = [
        State(
            log_n=float(np.log(max(float(n), 1e-300))),
            log_Te=float(np.log(max(float(te), 1.0))),
            logA=float(np.log(max(float(area) / max(float(config.area_scale_m2), 1e-300), 1e-300))),
        )
        for n, te, area in zip(window["n_p"], window["T_e"], window["A"], strict=True)
    ]
    states = [seed_states[0]]
    sigma_prev = float(window["sigma_logA"][0])
    segments = []
    nodes = [_node_payload(0, states[0], config=config, sigma=sigma_prev, seed_index=int(window["indices"][0]), x=float(window["x"][0]))]
    for k in range(int(settings.n_steps)):
        seed_next = seed_states[min(k + 1, len(seed_states) - 1)]
        step = _policy_step(
            current=states[-1],
            seed_next=seed_next,
            sigma_prev=sigma_prev,
            dx=dx,
            direction=direction,
            config=config,
            settings=settings,
        )
        states.append(step["next_state"])
        sigma_prev = float(step["sigma"])
        x_value = float(window["x"][0]) + float(direction) * dx * float(k + 1)
        nodes.append(
            _node_payload(
                k + 1,
                step["next_state"],
                config=config,
                sigma=sigma_prev,
                seed_index=int(window["indices"][min(k + 1, len(window["indices"]) - 1)]),
                x=x_value,
            )
        )
        segments.append({**{key: value for key, value in step.items() if key != "next_state"}, "k": int(k)})
    active_summary = _active_summary(nodes=nodes, segments=segments, settings=settings)
    return {
        "ok": bool(all(bool(item["ok"]) for item in segments)),
        "settings": settings.__dict__,
        "case_config": config.to_dict(),
        "window": {
            "indices": [int(v) for v in window["indices"]],
            "x": [float(v) for v in window["x"]],
            "dx": dx,
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

    dx = float(settings.dx)
    if dx <= 0.0:
        raise ValueError("settings.dx must be positive.")
    states = [anchor.state]
    sigma_prev = float(anchor.sigma_logA)
    sigma_history = [sigma_prev]
    segments = []
    nodes = [
        _node_payload(
            0,
            states[0],
            config=config,
            sigma=sigma_prev,
            seed_index=int(anchor.source_index),
            x=float(anchor.x),
        )
    ]
    for k in range(int(settings.n_steps)):
        warm_start = _extrapolated_state(states)
        sigma_warm_start = _extrapolated_sigma(sigma_history)
        step = _policy_step(
            current=states[-1],
            seed_next=states[-1],
            sigma_prev=sigma_prev,
            warm_start=warm_start,
            sigma_warm_start=sigma_warm_start,
            dx=dx,
            direction=-1,
            config=config,
            settings=settings,  # type: ignore[arg-type]
        )
        states.append(step["next_state"])
        sigma_prev = float(step["sigma"])
        sigma_history.append(sigma_prev)
        x_value = float(anchor.x) - dx * float(k + 1)
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
    active_summary = _active_summary(nodes=nodes, segments=segments, settings=settings)  # type: ignore[arg-type]
    return {
        "ok": bool(all(bool(item["ok"]) for item in segments)),
        "mode": "reverse_preparation_recovery",
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
            "sigma_logA": float(anchor.sigma_logA),
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
    sigma_prev = float(anchor.sigma_logA)
    sigma_history = [sigma_prev]
    segments = []
    nodes = [
        _node_payload(
            0,
            states[0],
            config=config,
            sigma=sigma_prev,
            seed_index=int(anchor.source_index),
            x=float(anchor.x),
        )
    ]
    for k in range(int(settings.n_steps)):
        warm_start = _extrapolated_state(states)
        sigma_warm_start = _extrapolated_sigma(sigma_history)
        step = _policy_step(
            current=states[-1],
            seed_next=states[-1],
            sigma_prev=sigma_prev,
            warm_start=warm_start,
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
    active_summary = _active_summary(nodes=nodes, segments=segments, settings=settings)
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
            "sigma_logA": float(anchor.sigma_logA),
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


def _extrapolated_state(states: list[State]) -> State:
    if len(states) < 2:
        return states[-1]
    previous = states[-2]
    current = states[-1]
    return State(
        log_n=float(current.log_n + (current.log_n - previous.log_n)),
        log_Te=float(current.log_Te + (current.log_Te - previous.log_Te)),
        logA=float(current.logA + (current.logA - previous.logA)),
    )


def _extrapolated_sigma(sigmas: list[float]) -> float:
    if len(sigmas) < 2:
        return float(sigmas[-1])
    return float(sigmas[-1] + (sigmas[-1] - sigmas[-2]))


def _policy_step(
    *,
    current: State,
    seed_next: State,
    sigma_prev: float,
    warm_start: State | None = None,
    sigma_warm_start: float | None = None,
    dx: float,
    direction: int,
    config: CaseConfig,
    settings: PolicySettings,
) -> dict[str, Any]:
    lo, hi, bound_sources = _sigma_interval(
        current=current,
        sigma_prev=sigma_prev,
        dx=dx,
        direction=direction,
        settings=settings,
    )
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
        }

    base_sigma = float(np.clip(sigma_prev, lo, hi))
    g_sigma_initial = float(np.clip(base_sigma if sigma_warm_start is None else float(sigma_warm_start), lo, hi))
    pedal = _pedal_direction(
        current=current,
        base_sigma=base_sigma,
        lo=lo,
        hi=hi,
        dx=dx,
        direction=direction,
        config=config,
        settings=settings,
        initial=seed_next,
    )
    direct = _direct_boundary_choice(
        current=current,
        seed_next=seed_next,
        g_initial=warm_start if warm_start is not None else seed_next,
        sigma_initial=g_sigma_initial,
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
        return {
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
                initial=seed_next,
            )
        )
    feasible = [item for item in scan if bool(item["feasible"])]
    if not feasible:
        best_failed = min(scan, key=lambda item: float(item.get("constraint_violation", 1e300)))
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
            initial=chosen["next_state"],
        )
    support = _classify_support(
        chosen=chosen,
        lo=lo,
        hi=hi,
        bound_sources=bound_sources,
        settings=settings,
    )
    return {
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


def _direct_boundary_choice(
    *,
    current: State,
    seed_next: State,
    g_initial: State,
    sigma_initial: float,
    lo: float,
    hi: float,
    pedal: float,
    dx: float,
    direction: int,
    config: CaseConfig,
    settings: PolicySettings,
) -> dict[str, Any] | None:
    """Choose a boundary without scanning when the preferred side is G-limited."""

    cache: dict[float, dict[str, Any]] = {}
    warm = seed_next

    def evaluate(sigma: float) -> dict[str, Any]:
        nonlocal warm
        key = float(sigma)
        if key not in cache:
            item = _evaluate_sigma(
                current=current,
                sigma=key,
                dx=dx,
                direction=direction,
                config=config,
                settings=settings,
                initial=warm,
            )
            cache[key] = item
            if bool(item.get("ok", False)):
                warm = item["next_state"]
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

    g_step = _solve_g_boundary_step(
        current=current,
        initial=g_initial,
        sigma_initial=float(np.clip(sigma_initial, lo, hi)),
        lo=lo,
        hi=hi,
        dx=dx,
        direction=direction,
        config=config,
        settings=settings,
        preferred=preferred,
    )
    if g_step is not None:
        return g_step

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
    return {
        **chosen,
        "boundary_blockers": ["G"],
        "boundary_bracket_width": abs(float(preferred_sigma) - float(opposite_sigma)),
        "boundary_infeasible_sigma": float(preferred_sigma),
        "boundary_infeasible_margins": dict(preferred.get("constraint_margins", {})),
        "solver_method": "brentq_G_boundary_fallback",
    }


def _solve_g_boundary_step(
    *,
    current: State,
    initial: State,
    sigma_initial: float,
    lo: float,
    hi: float,
    dx: float,
    direction: int,
    config: CaseConfig,
    settings: PolicySettings,
    preferred: dict[str, Any],
) -> dict[str, Any] | None:
    params = _physics_params(config)
    dx_signed = float(direction) * float(dx)

    def residual(y: np.ndarray) -> np.ndarray:
        r = g_boundary_residual_numba(
            float(y[0]),
            float(y[1]),
            float(y[2]),
            float(current.n_p),
            float(current.T_e),
            float(current.logA),
            dx_signed,
            float(params.area_scale_m2),
            float(params.dot_N),
            float(params.I_0),
            float(params.seed_fraction),
            float(params.B),
            float(params.heavy_particle_mass_kg),
            float(params.seed_ionization_energy_J),
            float(params.sigma_ep),
            float(settings.g_floor),
        )
        return np.array([float(r[0]), float(r[1]), float(r[2])], dtype=float)

    guess = np.array(
        [float(initial.log_n), float(initial.log_Te), float(np.clip(sigma_initial, lo, hi))],
        dtype=float,
    )
    lower = np.array([-np.inf, np.log(1.0), float(lo)], dtype=float)
    upper = np.array([np.inf, np.inf, float(hi)], dtype=float)
    sol = least_squares(
        residual,
        guess,
        bounds=(lower, upper),
        xtol=1e-11,
        ftol=1e-11,
        gtol=1e-11,
        max_nfev=max(20, int(settings.refine_iterations) + 8),
        x_scale=np.array([1.0, 1.0, max(abs(float(hi) - float(lo)), 1e-3)], dtype=float),
    )
    r = residual(sol.x)
    if not bool(sol.success):
        return None
    if float(np.max(np.abs(r))) > 1e-5:
        return None
    sigma = float(sol.x[2])
    if sigma < float(lo) - float(settings.active_tol) or sigma > float(hi) + float(settings.active_tol):
        return None

    logA_next = float(current.logA + dx_signed * sigma)
    next_state = State(log_n=float(sol.x[0]), log_Te=float(sol.x[1]), logA=logA_next)
    metrics = _closure_metrics(next_state, config=config)
    current_delta = _closure_metrics(current, config=config)["Delta"]
    delta_gain = float(metrics["Delta"] - current_delta)
    objective_value = -delta_gain if settings.objective == "delta_drop" else delta_gain
    max_abs_scaled_residual = float(max(abs(float(r[0])), abs(float(r[1]))))
    margins = {
        "G": float(metrics["G"] - float(settings.g_floor)),
        "Tp": float(metrics["T_p"] - float(settings.tp_floor_K)),
        "residual": float(settings.residual_tol - max_abs_scaled_residual),
    }
    feasible = bool(all(value >= -float(settings.active_tol) for value in margins.values()))
    if not feasible:
        return None
    return {
        "ok": True,
        "residual_ok": bool(max_abs_scaled_residual <= 1e-7),
        "feasible": True,
        "sigma": sigma,
        "next_state": next_state,
        "objective_value": float(objective_value),
        "delta_gain": delta_gain,
        "constraint_margins": margins,
        "constraint_violation": 0.0,
        "max_abs_scaled_residual": max_abs_scaled_residual,
        "least_squares_cost": float(sol.cost),
        "least_squares_nfev": int(sol.nfev),
        "boundary_blockers": ["G"],
        "boundary_bracket_width": abs(float(preferred["sigma"]) - sigma),
        "boundary_infeasible_sigma": float(preferred["sigma"]),
        "boundary_infeasible_margins": dict(preferred.get("constraint_margins", {})),
        "solver_method": "least_squares_G_boundary",
        **metrics,
    }


def _evaluate_sigma(
    *,
    current: State,
    sigma: float,
    dx: float,
    direction: int,
    config: CaseConfig,
    settings: PolicySettings,
    initial: State,
) -> dict[str, Any]:
    logA_next = float(current.logA + float(direction) * float(dx) * float(sigma))
    next_state, residual = _solve_next_state(
        current=current,
        logA_next=logA_next,
        sigma=float(sigma),
        dx=dx,
        direction=direction,
        config=config,
        initial=initial,
    )
    metrics = _closure_metrics(next_state, config=config)
    current_delta = _closure_metrics(current, config=config)["Delta"]
    delta_gain = float(metrics["Delta"] - current_delta)
    if settings.objective == "delta_drop":
        objective_value = -delta_gain
    else:
        objective_value = delta_gain
    margins = {
        "G": float(metrics["G"] - float(settings.g_floor)),
        "Tp": float(metrics["T_p"] - float(settings.tp_floor_K)),
        "residual": float(settings.residual_tol - residual["max_abs_scaled_residual"]),
    }
    feasible = bool(all(value >= -float(settings.active_tol) for value in margins.values()))
    violation = float(sum(max(-value, 0.0) for value in margins.values()))
    return {
        "ok": bool(residual["ok"]),
        "feasible": bool(feasible and residual["ok"]),
        "sigma": float(sigma),
        "next_state": next_state,
        "objective_value": float(objective_value),
        "delta_gain": delta_gain,
        "constraint_margins": margins,
        "constraint_violation": violation,
        **metrics,
        **residual,
    }


def _solve_next_state(
    *,
    current: State,
    logA_next: float,
    sigma: float,
    dx: float,
    direction: int,
    config: CaseConfig,
    initial: State,
) -> tuple[State, dict[str, float | bool]]:
    params = _physics_params(config)
    dx_signed = float(direction) * float(dx)
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
            float(sigma),
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
            abs(float(terms[0]) * max(n_next, 1.0) / max(abs(dx_signed), 1e-300)),
            abs(float(terms[1]) * max(te_next, 1.0) / max(abs(dx_signed), 1e-300)),
            abs(float(terms[7])),
        )
        e_scale = max(
            1.0,
            abs(float(terms[3]) * max(n_next, 1.0) / max(abs(dx_signed), 1e-300)),
            abs(float(terms[4]) * max(te_next, 1.0) / max(abs(dx_signed), 1e-300)),
            abs(float(terms[8])),
        )
        return np.array([momentum / m_scale, energy / e_scale], dtype=float)

    guess = np.array([float(initial.log_n), float(initial.log_Te)], dtype=float)
    sol = least_squares(residual, guess, xtol=1e-11, ftol=1e-11, gtol=1e-11, max_nfev=80)
    next_state = State(log_n=float(sol.x[0]), log_Te=float(sol.x[1]), logA=float(logA_next))
    res = residual(sol.x)
    max_abs = float(np.max(np.abs(res)))
    return next_state, {
        "residual_ok": bool(sol.success and max_abs <= 1e-7),
        "ok": bool(sol.success),
        "max_abs_scaled_residual": max_abs,
        "least_squares_cost": float(sol.cost),
        "least_squares_nfev": int(sol.nfev),
    }


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
    initial: State,
) -> float:
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
        initial=initial,
    )
    right_eval = _evaluate_sigma(
        current=current,
        sigma=right,
        dx=dx,
        direction=direction,
        config=config,
        settings=settings,
        initial=initial,
    )
    return float(right_eval["objective_value"] - left_eval["objective_value"])


def _sigma_interval(
    *,
    current: State,
    sigma_prev: float,
    dx: float,
    direction: int,
    settings: PolicySettings,
) -> tuple[float, float, dict[str, str]]:
    intervals: list[tuple[str, float, float]] = [
        ("sigma_box", float(settings.sigma_min), float(settings.sigma_max)),
    ]
    if settings.curvature_max is not None and np.isfinite(float(settings.curvature_max)):
        width = abs(float(settings.curvature_max))
        intervals.append(("curvature", float(sigma_prev) - width, float(sigma_prev) + width))
    area_candidates = [
        (LogAreaSplineControl.lower_bound() - current.logA) / (float(direction) * float(dx)),
        (LogAreaSplineControl.upper_bound() - current.logA) / (float(direction) * float(dx)),
    ]
    intervals.append(("area", min(area_candidates), max(area_candidates)))
    lo = max(item[1] for item in intervals)
    hi = min(item[2] for item in intervals)
    sources = {
        "lower": ",".join(name for name, low, _ in intervals if abs(low - lo) <= 1e-12),
        "upper": ",".join(name for name, _, high in intervals if abs(high - hi) <= 1e-12),
    }
    return float(lo), float(hi), sources


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
    return {
        "T_p": T_p,
        "Delta": float(state.T_e / max(T_p, 1e-300) - 1.0),
        "G": float(closure[18]),
        "mach": float(closure[17]),
        "beta": float(closure[5]),
        "Z": float(closure[7]),
    }


def _node_payload(k: int, state: State, *, config: CaseConfig, sigma: float, seed_index: int, x: float) -> dict[str, float | int]:
    metrics = _closure_metrics(state, config=config)
    return {
        "k": int(k),
        "x": float(x),
        "seed_index": int(seed_index),
        "n_p": float(state.n_p),
        "T_e": float(state.T_e),
        "A": float(state.area(config)),
        "logA": float(state.logA),
        "sigma_logA": float(sigma),
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
    initial: State,
) -> dict[str, Any]:
    feasible = feasible_eval
    infeasible = infeasible_eval
    for _ in range(int(settings.refine_iterations)):
        mid_sigma = 0.5 * (float(feasible["sigma"]) + float(infeasible["sigma"]))
        mid = _evaluate_sigma(
            current=current,
            sigma=mid_sigma,
            dx=dx,
            direction=direction,
            config=config,
            settings=settings,
            initial=initial,
        )
        if bool(mid["feasible"]):
            feasible = mid
            initial = mid["next_state"]
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
        if float(value) < -tol:
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


def _eval_public(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "objective_value": float(item.get("objective_value", float("nan"))),
        "delta_gain": float(item.get("delta_gain", float("nan"))),
        "G": float(item.get("G", float("nan"))),
        "T_p": float(item.get("T_p", float("nan"))),
        "Delta": float(item.get("Delta", float("nan"))),
        "mach": float(item.get("mach", float("nan"))),
        "constraint_margins": dict(item.get("constraint_margins", {})),
        "boundary_blockers": list(item.get("boundary_blockers", [])),
        "boundary_bracket_width": float(item.get("boundary_bracket_width", float("nan"))),
        "boundary_infeasible_sigma": float(item.get("boundary_infeasible_sigma", float("nan"))),
        "boundary_infeasible_margins": dict(item.get("boundary_infeasible_margins", {})),
        "solver_method": str(item.get("solver_method", "legacy_scan")),
        "max_abs_scaled_residual": float(item.get("max_abs_scaled_residual", float("nan"))),
        "least_squares_nfev": int(item.get("least_squares_nfev", -1)),
    }


def _window_from_profile(profile: dict[str, Any], *, settings: PolicySettings) -> dict[str, Any]:
    x = np.asarray(profile["x"], dtype=float).reshape(-1)
    n_p = np.asarray(profile["n_p"], dtype=float).reshape(-1)
    T_e = np.asarray(profile["T_e"], dtype=float).reshape(-1)
    A = np.asarray(profile["A"], dtype=float).reshape(-1)
    if "sigma_logA" in profile:
        sigma = np.asarray(profile["sigma_logA"], dtype=float).reshape(-1)
    else:
        sigma = np.gradient(np.log(np.maximum(A / max(float(A[0]), 1e-300), 1e-300)), x, edge_order=1)
    if not (x.size == n_p.size == T_e.size == A.size == sigma.size):
        raise ValueError("profile x/n_p/T_e/A/sigma arrays must have matching sizes.")
    direction = _direction_sign(settings.direction)
    start_index = int(settings.start_index) if settings.start_index is not None else (0 if direction > 0 else x.size - 1)
    offsets = direction * int(settings.stride) * np.arange(int(settings.n_steps) + 1, dtype=int)
    indices = start_index + offsets
    if np.any(indices < 0) or np.any(indices >= x.size):
        raise ValueError("requested policy window is outside profile bounds.")
    dx_values = np.abs(np.diff(x[indices]))
    if not np.all(dx_values > 0.0):
        raise ValueError("selected policy window has non-positive dx.")
    return {
        "indices": indices,
        "x": x[indices],
        "n_p": n_p[indices],
        "T_e": T_e[indices],
        "A": A[indices],
        "sigma_logA": sigma[indices],
        "dx": float(np.mean(dx_values)),
    }


def _direction_sign(direction: str) -> int:
    normalized = str(direction).strip().lower()
    if normalized in {"forward", "downstream", "+", "+1"}:
        return 1
    if normalized in {"reverse", "backward", "upstream", "-", "-1"}:
        return -1
    raise ValueError("direction must be forward or reverse.")


def _active_summary(*, nodes: list[dict[str, Any]], segments: list[dict[str, Any]], settings: PolicySettings) -> dict[str, Any]:
    support_counts: dict[str, int] = {}
    for seg in segments:
        key = str(seg.get("support_type", "unknown"))
        support_counts[key] = support_counts.get(key, 0) + 1
    g_margin_near_count = int(
        sum(1 for node in nodes[1:] if float(node["G"]) - float(settings.g_floor) <= float(settings.active_tol))
    )
    tp_margin_near_count = int(
        sum(1 for node in nodes[1:] if float(node["T_p"]) - float(settings.tp_floor_K) <= float(settings.active_tol))
    )
    return {
        "support_counts": support_counts,
        "Delta_start": float(nodes[0]["Delta"]),
        "Delta_end": float(nodes[-1]["Delta"]),
        "Delta_gain": float(nodes[-1]["Delta"] - nodes[0]["Delta"]),
        "G_min_excluding_anchor": float(min(float(node["G"]) for node in nodes[1:])) if len(nodes) > 1 else None,
        "Tp_min_excluding_anchor_K": float(min(float(node["T_p"]) for node in nodes[1:])) if len(nodes) > 1 else None,
        "max_abs_scaled_residual": float(max(float(seg["max_abs_scaled_residual"]) for seg in segments)) if segments else 0.0,
        "G_active_count_excluding_anchor": int(support_counts.get("G_supported", 0)),
        "G_margin_near_count_excluding_anchor": g_margin_near_count,
        "Tp_floor_active_count_excluding_anchor": int(support_counts.get("Tp_floor_supported", 0)),
        "Tp_floor_margin_near_count_excluding_anchor": tp_margin_near_count,
    }
