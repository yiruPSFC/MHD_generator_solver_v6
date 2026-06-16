from __future__ import annotations

from typing import Any, Callable

import numpy as np
from scipy.optimize import least_squares

from v6_firedrake_reduced.design import CaseConfig

from .numba_physics import dynamic_terms_numba


PhysicsParamsFn = Callable[[CaseConfig], Any]
ClosureMetricsFn = Callable[..., dict[str, float]]
StateFactory = Callable[..., Any]


def primitive_sonic_compatibility(
    state: Any,
    *,
    config: CaseConfig,
    physics_params_fn: PhysicsParamsFn,
    closure_metrics_fn: ClosureMetricsFn,
) -> dict[str, Any]:
    params = physics_params_fn(config)
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
    metrics = closure_metrics_fn(state, config=config)
    if not (np.all(np.isfinite(matrix)) and np.all(np.isfinite(f0)) and np.all(np.isfinite(f1))):
        return {
            "ok": False,
            "mach": float(metrics["mach"]),
            "det_D": float("nan"),
            "singular_value_min": float("nan"),
            "singular_value_max": float("nan"),
            "ellTf0": float("nan"),
            "ellTf1": float("nan"),
            "A_prime_sonic": float("nan"),
            "sigma_sonic": float("nan"),
            "compatibility_residual": float("nan"),
            "compatibility_scaled_residual": float("nan"),
            "error": "non-finite primitive sonic matrix",
        }
    try:
        u, singular_values, _ = np.linalg.svd(matrix)
    except np.linalg.LinAlgError as exc:
        return {
            "ok": False,
            "mach": float(metrics["mach"]),
            "det_D": float(np.linalg.det(matrix)),
            "singular_value_min": float("nan"),
            "singular_value_max": float("nan"),
            "ellTf0": float("nan"),
            "ellTf1": float("nan"),
            "A_prime_sonic": float("nan"),
            "sigma_sonic": float("nan"),
            "compatibility_residual": float("nan"),
            "compatibility_scaled_residual": float("nan"),
            "error": f"primitive sonic SVD failed: {exc}",
        }
    left_null = np.asarray(u[:, -1], dtype=float)
    numerator = float(left_null @ f0)
    denominator = float(left_null @ f1)
    # REVIEW: ell^T f1 is the area-control projection; the step selector classifies small-denominator cases with the interval.
    if abs(denominator) <= 1.0e-300 or area <= 0.0:
        sigma = float("nan")
    else:
        sigma = float(-numerator / (area * denominator))
    residual = float(left_null @ (f0 + area * sigma * f1)) if np.isfinite(sigma) else float("nan")
    scaled = (
        residual / max(1.0, abs(numerator), abs(area * sigma * denominator))
        if np.isfinite(residual) and np.isfinite(sigma)
        else float("nan")
    )
    return {
        "ok": bool(np.isfinite(sigma) and np.isfinite(scaled)),
        "mach": float(metrics["mach"]),
        "det_D": float(np.linalg.det(matrix)),
        "singular_value_min": float(np.min(singular_values)) if singular_values.size else float("nan"),
        "singular_value_max": float(np.max(singular_values)) if singular_values.size else float("nan"),
        "ellTf0": numerator,
        "ellTf1": denominator,
        "A_prime_sonic": float(area * sigma) if np.isfinite(sigma) else float("nan"),
        "sigma_sonic": sigma,
        "compatibility_residual": residual,
        "compatibility_scaled_residual": scaled,
        "error": "",
    }


def should_use_sonic_branch(*, sonic: dict[str, Any], settings: Any) -> bool:
    mode = str(getattr(settings, "sonic_mode", "auto")).strip().lower()
    if mode in {"off", "none", "disabled", "false", "0"}:
        return False
    if mode in {"on", "always", "enabled", "true", "1"}:
        return True
    if mode != "auto":
        raise ValueError("sonic_mode must be auto, off, or on.")
    mach = float(sonic.get("mach", float("nan")))
    det_D = float(sonic.get("det_D", float("nan")))
    mach_near = np.isfinite(mach) and abs(mach - 1.0) <= float(getattr(settings, "sonic_mach_tol", 1.0e-3))
    det_near = np.isfinite(det_D) and abs(det_D) <= float(getattr(settings, "sonic_det_abs_tol", 1.0e-2))
    return bool(mach_near or det_near)


def sonic_compatibility_residual(*, sigma: float, area: float, ellTf0: float, ellTf1: float) -> float:
    return float(float(ellTf0) + float(area) * float(sigma) * float(ellTf1))


def choose_sonic_sigma(
    *,
    sonic: dict[str, Any],
    area: float,
    lo: float,
    hi: float,
    sigma_reference: float,
    settings: Any,
) -> dict[str, Any]:
    ellTf0 = float(sonic.get("ellTf0", float("nan")))
    ellTf1 = float(sonic.get("ellTf1", float("nan")))
    area = float(area)
    lo = float(lo)
    hi = float(hi)
    tolerance = float(getattr(settings, "sonic_compatibility_tol", 1.0e-7))
    if not all(np.isfinite(value) for value in [ellTf0, ellTf1, area, lo, hi]) or area <= 0.0:
        return {
            "ok": False,
            "status": "invalid_left_null_data",
            "sigma": float("nan"),
            "solver_method": "sonic_left_null_invalid",
            "error": str(sonic.get("error") or "sonic left-null data are not finite"),
        }

    r_lo = sonic_compatibility_residual(sigma=lo, area=area, ellTf0=ellTf0, ellTf1=ellTf1)
    r_hi = sonic_compatibility_residual(sigma=hi, area=area, ellTf0=ellTf0, ellTf1=ellTf1)
    scale = max(1.0, abs(ellTf0), abs(area * lo * ellTf1), abs(area * hi * ellTf1))
    r_lo_scaled = float(r_lo / scale)
    r_hi_scaled = float(r_hi / scale)
    variation_scaled = float(abs(r_hi - r_lo) / scale)
    if abs(r_lo_scaled) <= abs(r_hi_scaled):
        best_sigma = lo
        best_residual = r_lo
        best_scaled = r_lo_scaled
    else:
        best_sigma = hi
        best_residual = r_hi
        best_scaled = r_hi_scaled

    flat_control = bool(variation_scaled <= tolerance)
    flat_compatible = bool(flat_control and max(abs(r_lo_scaled), abs(r_hi_scaled)) <= tolerance)
    if flat_compatible:
        sigma = float(np.clip(float(sigma_reference), lo, hi))
        residual = sonic_compatibility_residual(sigma=sigma, area=area, ellTf0=ellTf0, ellTf1=ellTf1)
        return {
            "ok": True,
            "status": "flat_compatible",
            "sigma": sigma,
            "solver_method": "sonic_left_null_flat_reference",
            "residual": residual,
            "scaled_residual": float(residual / scale),
            "r_lo": r_lo,
            "r_hi": r_hi,
            "r_lo_scaled": r_lo_scaled,
            "r_hi_scaled": r_hi_scaled,
            "best_interval_sigma": best_sigma,
            "best_interval_residual": best_residual,
            "best_interval_scaled_residual": best_scaled,
            "variation_scaled": variation_scaled,
            "scale": scale,
            "error": "",
        }

    if flat_control:
        return {
            "ok": False,
            "status": "unreachable_flat_forcing",
            "sigma": best_sigma,
            "solver_method": "sonic_left_null_unreachable",
            "residual": best_residual,
            "scaled_residual": best_scaled,
            "r_lo": r_lo,
            "r_hi": r_hi,
            "r_lo_scaled": r_lo_scaled,
            "r_hi_scaled": r_hi_scaled,
            "best_interval_sigma": best_sigma,
            "best_interval_residual": best_residual,
            "best_interval_scaled_residual": best_scaled,
            "variation_scaled": variation_scaled,
            "scale": scale,
            "error": "sonic compatibility forcing is not controllable by area slope",
        }

    root = float("nan")
    if abs(ellTf1) > 0.0:
        root = float(-ellTf0 / (area * ellTf1))
    active_tol = float(settings.active_tol)
    if np.isfinite(root) and root >= lo - active_tol and root <= hi + active_tol:
        sigma = float(np.clip(root, lo, hi))
        residual = sonic_compatibility_residual(sigma=sigma, area=area, ellTf0=ellTf0, ellTf1=ellTf1)
        return {
            "ok": True,
            "status": "root_in_interval",
            "sigma": sigma,
            "solver_method": "sonic_left_null_explicit_A_prime",
            "residual": residual,
            "scaled_residual": float(residual / scale),
            "r_lo": r_lo,
            "r_hi": r_hi,
            "r_lo_scaled": r_lo_scaled,
            "r_hi_scaled": r_hi_scaled,
            "best_interval_sigma": best_sigma,
            "best_interval_residual": best_residual,
            "best_interval_scaled_residual": best_scaled,
            "variation_scaled": variation_scaled,
            "scale": scale,
            "root_sigma": root,
            "error": "",
        }

    if abs(best_scaled) <= tolerance:
        return {
            "ok": True,
            "status": "boundary_compatible",
            "sigma": best_sigma,
            "solver_method": "sonic_left_null_boundary_compatible",
            "residual": best_residual,
            "scaled_residual": best_scaled,
            "r_lo": r_lo,
            "r_hi": r_hi,
            "r_lo_scaled": r_lo_scaled,
            "r_hi_scaled": r_hi_scaled,
            "best_interval_sigma": best_sigma,
            "best_interval_residual": best_residual,
            "best_interval_scaled_residual": best_scaled,
            "variation_scaled": variation_scaled,
            "scale": scale,
            "root_sigma": root,
            "error": "",
        }

    return {
        "ok": False,
        "status": "unreachable_interval",
        "sigma": best_sigma,
        "solver_method": "sonic_left_null_unreachable",
        "residual": best_residual,
        "scaled_residual": best_scaled,
        "r_lo": r_lo,
        "r_hi": r_hi,
        "r_lo_scaled": r_lo_scaled,
        "r_hi_scaled": r_hi_scaled,
        "best_interval_sigma": best_sigma,
        "best_interval_residual": best_residual,
        "best_interval_scaled_residual": best_scaled,
        "variation_scaled": variation_scaled,
        "scale": scale,
        "root_sigma": root,
        "error": "no sonic-compatible sigma exists inside slope/area/curvature bounds",
    }


def sonic_compatibility_choice_diagnostics(choice: dict[str, Any]) -> dict[str, Any]:
    return {
        "sonic_compatibility_status": str(choice.get("status", "")),
        "sonic_compatibility_selected_sigma": float(choice.get("sigma", float("nan"))),
        "sonic_compatibility_selected_residual": float(choice.get("residual", float("nan"))),
        "sonic_compatibility_selected_scaled_residual": float(choice.get("scaled_residual", float("nan"))),
        "sonic_compatibility_r_lower": float(choice.get("r_lo", float("nan"))),
        "sonic_compatibility_r_upper": float(choice.get("r_hi", float("nan"))),
        "sonic_compatibility_r_lower_scaled": float(choice.get("r_lo_scaled", float("nan"))),
        "sonic_compatibility_r_upper_scaled": float(choice.get("r_hi_scaled", float("nan"))),
        "sonic_compatibility_best_interval_sigma": float(choice.get("best_interval_sigma", float("nan"))),
        "sonic_compatibility_best_interval_residual": float(choice.get("best_interval_residual", float("nan"))),
        "sonic_compatibility_best_interval_scaled_residual": float(
            choice.get("best_interval_scaled_residual", float("nan"))
        ),
        "sonic_compatibility_variation_scaled": float(choice.get("variation_scaled", float("nan"))),
        "sonic_compatibility_scale": float(choice.get("scale", float("nan"))),
        "sonic_compatibility_root_sigma": float(choice.get("root_sigma", float("nan"))),
    }


def apply_sonic_residual_gate(item: dict[str, Any], *, settings: Any) -> dict[str, Any]:
    out = dict(item)
    residual_tol = float(getattr(settings, "sonic_residual_tol", 1.0e-6))
    max_residual = float(out.get("max_abs_scaled_residual", float("inf")))
    out["step_error_kind"] = "physical_residual"
    out["physical_residual_scaled"] = max_residual
    margins = dict(out.get("constraint_margins", {}) or {})
    margins["residual"] = float(residual_tol - max_residual)
    out["constraint_margins"] = margins
    active_tol = float(settings.active_tol)
    active_feasible = bool(
        float(margins.get("G", 0.0)) >= -active_tol
        and float(margins.get("Tp", 0.0)) >= -active_tol
        and float(margins.get("residual", 0.0)) >= -active_tol
    )
    # REVIEW: Sonic steps trust the scaled residual gate; a low-residual solve can override a conservative least_squares flag.
    if active_feasible and np.isfinite(max_residual):
        out["ok"] = True
        out["feasible"] = True
        out["residual_ok"] = bool(max_residual <= residual_tol)
        out["physical_residual_ok"] = bool(max_residual <= residual_tol)
        out["sonic_residual_gate"] = "accepted"
        out["sonic_residual_tol"] = residual_tol
        out["constraint_violation"] = 0.0
    else:
        out["sonic_residual_gate"] = "failed"
        out["sonic_residual_tol"] = residual_tol
        out["physical_residual_ok"] = bool(np.isfinite(max_residual) and max_residual <= residual_tol)
        out["constraint_violation"] = float(sum(max(-float(value), 0.0) for value in margins.values()))
    return out


def sonic_initial_guesses(*, current: Any, logA_next: float, state_factory: StateFactory) -> list[Any]:
    pairs = [(0.0, 0.0)]
    for sign in (-1.0, 1.0):
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
    out: list[Any] = []
    seen: set[tuple[float, float]] = set()
    for dn, dte in pairs:
        key = (round(float(dn), 12), round(float(dte), 12))
        if key in seen:
            continue
        seen.add(key)
        out.append(state_factory(log_n=float(current.log_n + dn), log_Te=float(current.log_Te + dte), logA=logA_next))
    return out


def sonic_finite_residual_vector(
    *,
    current: Any,
    log_n_next: float,
    log_Te_next: float,
    area_next: float,
    sigma: float,
    dx_signed: float,
    params: Any,
) -> np.ndarray:
    n_next = float(np.exp(np.clip(float(log_n_next), -700.0, 700.0)))
    te_next = float(np.exp(np.clip(float(log_Te_next), -700.0, 700.0)))
    terms = dynamic_terms_numba(
        n_next,
        te_next,
        float(area_next),
        float(sigma),
        float(params.dot_N),
        float(params.I_0),
        float(params.seed_fraction),
        float(params.B),
        float(params.heavy_particle_mass_kg),
        float(params.seed_ionization_energy_J),
        float(params.sigma_ep),
    )
    dn_dx = (n_next - current.n_p) / float(dx_signed)
    dte_dx = (te_next - current.T_e) / float(dx_signed)
    momentum = float(terms[0]) * dn_dx + float(terms[1]) * dte_dx - float(terms[7])
    energy = float(terms[3]) * dn_dx + float(terms[4]) * dte_dx - float(terms[8])
    m_scale = max(
        1.0,
        abs(float(terms[0]) * max(n_next, 1.0) / max(abs(float(dx_signed)), 1.0e-300)),
        abs(float(terms[1]) * max(te_next, 1.0) / max(abs(float(dx_signed)), 1.0e-300)),
        abs(float(terms[7])),
    )
    e_scale = max(
        1.0,
        abs(float(terms[3]) * max(n_next, 1.0) / max(abs(float(dx_signed)), 1.0e-300)),
        abs(float(terms[4]) * max(te_next, 1.0) / max(abs(float(dx_signed)), 1.0e-300)),
        abs(float(terms[8])),
    )
    return np.array([momentum / m_scale, energy / e_scale], dtype=float)


def solve_sonic_finite_step(
    *,
    current: Any,
    logA_next: float,
    sigma: float,
    dx: float,
    direction: int,
    config: CaseConfig,
    initial: Any,
    physics_params_fn: PhysicsParamsFn,
    state_factory: StateFactory,
) -> tuple[Any, dict[str, Any]]:
    params = physics_params_fn(config)
    dx_signed = float(direction) * float(dx)
    area_next = float(params.area_scale_m2) * float(np.exp(np.clip(logA_next, -700.0, 700.0)))

    def residual(y: np.ndarray) -> np.ndarray:
        return sonic_finite_residual_vector(
            current=current,
            log_n_next=float(y[0]),
            log_Te_next=float(y[1]),
            area_next=area_next,
            sigma=float(sigma),
            dx_signed=dx_signed,
            params=params,
        )

    guess = np.array([float(initial.log_n), float(initial.log_Te)], dtype=float)
    try:
        sol = least_squares(residual, guess, xtol=1.0e-11, ftol=1.0e-11, gtol=1.0e-11, max_nfev=80)
        res = residual(np.asarray(sol.x, dtype=float))
        max_abs = float(np.max(np.abs(res))) if res.size else float("inf")
        return (
            state_factory(log_n=float(sol.x[0]), log_Te=float(sol.x[1]), logA=float(logA_next)),
            {
                "ok": bool(sol.success and np.isfinite(max_abs)),
                # RISK: Solver-local diagnostic only; final sonic acceptance uses sonic_residual_tol downstream.
                "residual_ok": bool(sol.success and max_abs <= 1.0e-7),
                "max_abs_scaled_residual": max_abs,
                "step_error_kind": "physical_residual",
                "physical_residual_scaled": max_abs,
                "physical_residual_ok": bool(sol.success and max_abs <= 1.0e-7),
                "sonic_step_cost": float(sol.cost),
                "sonic_step_nfev": int(sol.nfev),
            },
        )
    except Exception as exc:
        return (
            current,
            {
                "ok": False,
                "residual_ok": False,
                "max_abs_scaled_residual": float("inf"),
                "step_error_kind": "physical_residual",
                "physical_residual_scaled": float("inf"),
                "physical_residual_ok": False,
                "sonic_step_cost": float("inf"),
                "sonic_step_nfev": -1,
                "error": f"sonic finite-step solve failed: {exc}",
            },
        )
