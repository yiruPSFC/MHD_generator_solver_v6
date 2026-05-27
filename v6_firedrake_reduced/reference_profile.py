from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

from .design import CaseConfig, DesignVector
from .legacy_physics import dynamic_system_terms, inlet_design_generic, ops_for_numeric
from .transport import working_fluid_for_config


K_B = 1.380649e-23


@dataclass(frozen=True)
class ReferenceProfileResult:
    ok: bool
    profile: dict[str, np.ndarray] | None
    diagnostics: dict[str, Any]
    error: str | None = None


def _area_profile_arrays(
    area_profile: dict[str, Any],
    *,
    config: CaseConfig,
    n_intervals: int | None = None,
) -> dict[str, np.ndarray]:
    if "x" in area_profile:
        x = np.asarray(area_profile["x"], dtype=float).reshape(-1)
    elif "x_norm" in area_profile:
        x = np.asarray(area_profile["x_norm"], dtype=float).reshape(-1) * float(config.length_m)
    else:
        if "A" not in area_profile:
            raise ValueError("area_profile must contain x/x_norm and A/logA arrays.")
        x = np.linspace(
            0.0,
            float(config.length_m),
            np.asarray(area_profile["A"], dtype=float).reshape(-1).size,
            dtype=float,
        )
    order = np.argsort(x)
    if "A" in area_profile:
        A = np.asarray(area_profile["A"], dtype=float).reshape(-1)
        if np.any(A <= 0.0) or not np.all(np.isfinite(A)):
            raise ValueError("area_profile A values must be finite and positive.")
        logA = np.log(A / max(float(config.area_scale_m2), 1e-300))
    elif "logA" in area_profile:
        logA = np.asarray(area_profile["logA"], dtype=float).reshape(-1)
        A = float(config.area_scale_m2) * np.exp(np.clip(logA, -700.0, 700.0))
    else:
        raise ValueError("area_profile must contain A or logA.")
    if x.size != A.size or x.size < 2:
        raise ValueError("area_profile x and A/logA arrays must have matching length >= 2.")
    needs_sort = bool(not np.all(np.isfinite(x)) or not np.all(np.diff(x) > 0.0))
    if needs_sort:
        x = x[order]
        A = A[order]
        logA = logA[order]
        if not np.all(np.isfinite(x)) or not np.all(np.diff(x) > 0.0):
            raise ValueError("area_profile x values must be finite and strictly increasing.")
    if "sigma_logA" in area_profile:
        sigma = np.asarray(area_profile["sigma_logA"], dtype=float).reshape(-1)
        if sigma.size != x.size:
            raise ValueError("area_profile sigma_logA must match x size.")
        if needs_sort:
            sigma = sigma[order]
    else:
        sigma = np.gradient(logA, x, edge_order=1)
    if n_intervals is not None:
        x_target = np.linspace(0.0, float(config.length_m), int(n_intervals) + 1, dtype=float)
        logA_target = np.interp(x_target, x, logA)
        sigma_target = np.interp(x_target, x, sigma)
        return {
            "x": x_target,
            "x_norm": x_target / max(float(config.length_m), 1e-300),
            "logA": logA_target,
            "A": float(config.area_scale_m2) * np.exp(np.clip(logA_target, -700.0, 700.0)),
            "sigma_logA": sigma_target,
        }
    return {
        "x": x,
        "x_norm": x / max(float(config.length_m), 1e-300),
        "logA": logA,
        "A": A,
        "sigma_logA": sigma,
    }


def _area_at_x(
    *,
    design: DesignVector,
    config: CaseConfig,
    x: float,
    area_profile: dict[str, Any] | None = None,
) -> tuple[float, float]:
    if area_profile is not None:
        area = _area_profile_arrays(area_profile, config=config)
        x_value = float(np.clip(float(x), 0.0, float(config.length_m)))
        logA = float(np.interp(x_value, area["x"], area["logA"]))
        sigma = float(np.interp(x_value, area["x"], area["sigma_logA"]))
        return float(config.area_scale_m2) * float(np.exp(np.clip(logA, -700.0, 700.0))), sigma
    x_norm = np.array([float(x) / max(float(config.length_m), 1e-300)], dtype=float)
    x_norm = np.clip(x_norm, 0.0, 1.0)
    basis, slopes = design.area_control.basis_matrices(x_norm)
    params = design.area_control.as_array()
    logA = float(basis[0, :] @ params)
    sigma_logA = float(slopes[0, :] @ params) / max(float(config.length_m), 1e-300)
    return float(config.area_scale_m2) * float(np.exp(logA)), sigma_logA


def _safe_exp(value: float) -> float:
    return float(np.exp(np.clip(float(value), -700.0, 700.0)))


def _freidberg_terms_for_log_state(
    *,
    ops,
    fluid,
    design: DesignVector,
    config: CaseConfig,
    inlet: dict[str, Any],
    x: float,
    z: np.ndarray,
    area_profile: dict[str, Any] | None = None,
) -> dict[str, Any]:
    log_n = float(design.log_n_p_in) + float(z[0])
    log_Te = float(np.log(max(float(design.T_e_in), 1.0))) + float(z[1])
    n_p = _safe_exp(log_n)
    T_e = _safe_exp(log_Te)
    A, sigma_logA = _area_at_x(design=design, config=config, x=float(x), area_profile=area_profile)
    closure, terms = dynamic_system_terms(
        ops=ops,
        n_p=n_p,
        T_e=T_e,
        A=A,
        sigma=sigma_logA,
        dot_N=float(inlet["dot_N"]),
        I_0=float(design.I_0),
        seed_fraction=design.seed_fraction,
        B=float(design.B_T),
        working_fluid=fluid,
    )
    J2 = float(closure["J_x"]) * float(closure["J_x"]) + float(closure["J_y"]) * float(closure["J_y"])
    v_p = float(closure["v_p"])
    T_p = float(closure["T_p"])
    T_p_safe = max(T_p, 1.0)
    mach = float(closure["mach"])
    M2 = mach * mach
    A0 = max(float(config.area_scale_m2), 1e-300)
    H_p = (A * n_p * v_p / A0) * (
        2.5 * K_B * T_p_safe + 0.5 * float(fluid.heavy_particle_mass_kg) * v_p * v_p
    )
    L_p = mach * (A / A0) / max((M2 + 3.0) * (M2 + 3.0), 1e-300)
    rhs_H = (A / A0) * (v_p * float(closure["J_y"]) * float(design.B_T) + float(closure["eta"]) * J2)
    p_p = max(n_p * K_B * T_p_safe, 1e-300)
    rhs_L = (
        -(12.0 / 5.0)
        * L_p
        / max((M2 + 3.0) * p_p * v_p, 1e-300)
        * (v_p * float(closure["J_y"]) * float(design.B_T) - ((5.0 * M2 + 3.0) / 12.0) * float(closure["eta"]) * J2)
    )
    return {
        "x": float(x),
        "z": np.asarray(z, dtype=float).copy(),
        "n_p": float(n_p),
        "T_e": float(T_e),
        "A": float(A),
        "sigma_logA": float(sigma_logA),
        "closure": closure,
        "primitive_terms": terms,
        "H_p": float(H_p),
        "L_p": float(L_p),
        "rhs_H": float(rhs_H),
        "rhs_L": float(rhs_L),
        "mach": float(mach),
        "T_p": float(T_p),
        "det": float(terms["det"]),
        "J2": float(J2),
    }


def _freidberg_step_residual(
    *,
    ops,
    fluid,
    design: DesignVector,
    config: CaseConfig,
    inlet: dict[str, Any],
    x_prev: float,
    z_prev: np.ndarray,
    x_next: float,
    z_next: np.ndarray,
    h: float,
    area_profile: dict[str, Any] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    prev = _freidberg_terms_for_log_state(
        ops=ops,
        fluid=fluid,
        design=design,
        config=config,
        inlet=inlet,
        x=float(x_prev),
        z=z_prev,
        area_profile=area_profile,
    )
    nxt = _freidberg_terms_for_log_state(
        ops=ops,
        fluid=fluid,
        design=design,
        config=config,
        inlet=inlet,
        x=float(x_next),
        z=z_next,
        area_profile=area_profile,
    )
    h_safe = max(float(h), 1e-300)
    H_res = (nxt["H_p"] - prev["H_p"]) / h_safe - nxt["rhs_H"]
    L_res = (nxt["L_p"] - prev["L_p"]) / h_safe - nxt["rhs_L"]
    inv_l = 1.0 / max(float(config.length_m), 1e-300)
    H_scale = max(1.0, abs(nxt["H_p"] * inv_l), abs(prev["H_p"] * inv_l), abs(nxt["rhs_H"]))
    L_scale = max(inv_l, abs(nxt["L_p"] * inv_l), abs(prev["L_p"] * inv_l), abs(nxt["rhs_L"]))
    detail = {
        "H_residual": float(H_res),
        "L_residual": float(L_res),
        "scaled_H_residual": float(H_res / H_scale),
        "scaled_L_residual": float(L_res / L_scale),
        "H_scale": float(H_scale),
        "L_scale": float(L_scale),
        "H_p": float(nxt["H_p"]),
        "L_p": float(nxt["L_p"]),
        "rhs_H": float(nxt["rhs_H"]),
        "rhs_L": float(nxt["rhs_L"]),
        "det": float(nxt["det"]),
        "mach": float(nxt["mach"]),
        "T_p": float(nxt["T_p"]),
        "n_p": float(nxt["n_p"]),
        "T_e": float(nxt["T_e"]),
        "A": float(nxt["A"]),
        "sigma_logA": float(nxt["sigma_logA"]),
    }
    return np.array([detail["scaled_H_residual"], detail["scaled_L_residual"]], dtype=float), detail


def build_reference_profile(
    *,
    design: DesignVector,
    config: CaseConfig,
    n_intervals: int | None = None,
    method: str = "BDF",
    rtol: float = 1e-6,
    atol: float = 1e-8,
    max_step_fraction: float = 0.25,
) -> ReferenceProfileResult:
    """Generate a forward-marched log-space profile for Firedrake initialization."""

    ops = ops_for_numeric()
    fluid = working_fluid_for_config(config)
    n_nodes = int(config.n_intervals if n_intervals is None else n_intervals) + 1
    x_eval = np.linspace(0.0, float(config.length_m), n_nodes, dtype=float)
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
    log_n_in = float(design.log_n_p_in)
    log_Te_in = float(np.log(max(float(design.T_e_in), 1.0)))
    rhs_calls = 0

    def rhs(x: float, z: np.ndarray) -> np.ndarray:
        nonlocal rhs_calls
        rhs_calls += 1
        if not np.all(np.isfinite(z)):
            raise FloatingPointError("non-finite log-state in reference profile RHS")
        n_p = _safe_exp(log_n_in + float(z[0]))
        T_e = _safe_exp(log_Te_in + float(z[1]))
        A, sigma_logA = _area_at_x(design=design, config=config, x=float(x))
        _, terms = dynamic_system_terms(
            ops=ops,
            n_p=n_p,
            T_e=T_e,
            A=A,
            sigma=sigma_logA,
            dot_N=float(inlet["dot_N"]),
            I_0=float(design.I_0),
            seed_fraction=design.seed_fraction,
            B=float(design.B_T),
            working_fluid=fluid,
        )
        det = float(terms["det"])
        if abs(det) < 1e-300:
            det = 1e-300 if det >= 0.0 else -1e-300
        dn_dx = (float(terms["rhs_m"]) * float(terms["E12"]) - float(terms["M12"]) * float(terms["rhs_e"])) / det
        dTe_dx = (float(terms["M11"]) * float(terms["rhs_e"]) - float(terms["rhs_m"]) * float(terms["E11"])) / det
        return np.array([dn_dx / max(n_p, 1e-300), dTe_dx / max(T_e, 1e-300)], dtype=float)

    try:
        sol = solve_ivp(
            rhs,
            t_span=(0.0, float(config.length_m)),
            y0=np.array([0.0, 0.0], dtype=float),
            method=str(method),
            t_eval=x_eval,
            rtol=float(rtol),
            atol=float(atol),
            max_step=max(float(config.length_m) * float(max_step_fraction) / max(n_nodes - 1, 1), 1e-12),
        )
    except Exception as exc:
        return ReferenceProfileResult(
            ok=False,
            profile=None,
            diagnostics={
                "method": str(method),
                "rhs_calls": int(rhs_calls),
                "working_fluid": fluid.to_dict(),
                "inlet": {key: float(value) for key, value in inlet.items()},
            },
            error=f"{type(exc).__name__}: {exc}",
        )

    diagnostics = {
        "method": str(method),
        "success": bool(sol.success),
        "status": int(sol.status),
        "message": str(sol.message),
        "rhs_calls": int(rhs_calls),
        "working_fluid": fluid.to_dict(),
        "inlet": {key: float(value) for key, value in inlet.items()},
    }
    if sol.t.size:
        diagnostics.update(
            {
                "x_end": float(sol.t[-1]),
                "x_end_fraction": float(sol.t[-1] / max(float(config.length_m), 1e-300)),
                "partial_delta_log_n_end": float(sol.y[0, -1]),
                "partial_delta_log_Te_end": float(sol.y[1, -1]),
                "partial_n_p_end": _safe_exp(log_n_in + float(sol.y[0, -1])),
                "partial_T_e_end": _safe_exp(log_Te_in + float(sol.y[1, -1])),
            }
        )
    if not bool(sol.success) or sol.y.shape[1] != x_eval.size:
        return ReferenceProfileResult(
            ok=False,
            profile=None,
            diagnostics=diagnostics,
            error=str(sol.message),
        )

    delta_log_n = np.asarray(sol.y[0, :], dtype=float)
    delta_log_Te = np.asarray(sol.y[1, :], dtype=float)
    n_p = np.exp(log_n_in + delta_log_n)
    T_e = np.exp(log_Te_in + delta_log_Te)
    area = design.area_control.evaluate_profile(
        length=float(config.length_m),
        n_intervals=n_nodes - 1,
        area_scale=float(config.area_scale_m2),
    )
    profile = {
        "x": x_eval,
        "x_norm": x_eval / max(float(config.length_m), 1e-300),
        "n_p": n_p,
        "T_e": T_e,
        "A": np.asarray(area["A"], dtype=float),
        "sigma_logA": np.asarray(area["sigma_logA"], dtype=float),
    }
    diagnostics.update(
        {
            "delta_log_n_min": float(np.nanmin(delta_log_n)),
            "delta_log_n_max": float(np.nanmax(delta_log_n)),
            "delta_log_Te_min": float(np.nanmin(delta_log_Te)),
            "delta_log_Te_max": float(np.nanmax(delta_log_Te)),
            "n_p_min": float(np.nanmin(n_p)),
            "n_p_max": float(np.nanmax(n_p)),
            "T_e_min": float(np.nanmin(T_e)),
            "T_e_max": float(np.nanmax(T_e)),
        }
    )
    return ReferenceProfileResult(ok=True, profile=profile, diagnostics=diagnostics)


def _local_terms_for_log_state(
    *,
    ops,
    fluid,
    design: DesignVector,
    config: CaseConfig,
    inlet: dict[str, Any],
    x: float,
    z: np.ndarray,
) -> tuple[dict[str, Any], dict[str, Any], float, float, float, float]:
    log_n = float(design.log_n_p_in) + float(z[0])
    log_Te = float(np.log(max(float(design.T_e_in), 1.0))) + float(z[1])
    n_p = _safe_exp(log_n)
    T_e = _safe_exp(log_Te)
    A, sigma_logA = _area_at_x(design=design, config=config, x=float(x))
    closure, terms = dynamic_system_terms(
        ops=ops,
        n_p=n_p,
        T_e=T_e,
        A=A,
        sigma=sigma_logA,
        dot_N=float(inlet["dot_N"]),
        I_0=float(design.I_0),
        seed_fraction=design.seed_fraction,
        B=float(design.B_T),
        working_fluid=fluid,
    )
    return closure, terms, n_p, T_e, A, sigma_logA


def _scaled_step_residual(
    *,
    ops,
    fluid,
    design: DesignVector,
    config: CaseConfig,
    inlet: dict[str, Any],
    x_next: float,
    z_prev: np.ndarray,
    z_next: np.ndarray,
    h: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    closure, terms, n_next, T_next, _, _ = _local_terms_for_log_state(
        ops=ops,
        fluid=fluid,
        design=design,
        config=config,
        inlet=inlet,
        x=float(x_next),
        z=z_next,
    )
    n_prev = _safe_exp(float(design.log_n_p_in) + float(z_prev[0]))
    T_prev = _safe_exp(float(np.log(max(float(design.T_e_in), 1.0))) + float(z_prev[1]))
    dn_dx = (n_next - n_prev) / max(float(h), 1e-300)
    dT_dx = (T_next - T_prev) / max(float(h), 1e-300)
    momentum = float(terms["M11"]) * dn_dx + float(terms["M12"]) * dT_dx - float(terms["rhs_m"])
    energy = float(terms["E11"]) * dn_dx + float(terms["E12"]) * dT_dx - float(terms["rhs_e"])
    inv_l = 1.0 / max(float(config.length_m), 1e-300)
    momentum_scale = max(
        1.0,
        abs(float(terms["M11"]) * n_next * inv_l),
        abs(float(terms["M12"]) * T_next * inv_l),
        abs(float(terms["rhs_m"])),
    )
    energy_scale = max(
        1.0,
        abs(float(terms["E11"]) * n_next * inv_l),
        abs(float(terms["E12"]) * T_next * inv_l),
        abs(float(terms["rhs_e"])),
    )
    detail = {
        "momentum": float(momentum),
        "energy": float(energy),
        "scaled_momentum": float(momentum / momentum_scale),
        "scaled_energy": float(energy / energy_scale),
        "momentum_scale": float(momentum_scale),
        "energy_scale": float(energy_scale),
        "det": float(terms["det"]),
        "mach": float(closure["mach"]),
        "T_p": float(closure["T_p"]),
        "n_p": float(n_next),
        "T_e": float(T_next),
    }
    return np.array([detail["scaled_momentum"], detail["scaled_energy"]], dtype=float), detail


def _explicit_log_rhs_for_predictor(
    *,
    ops,
    fluid,
    design: DesignVector,
    config: CaseConfig,
    inlet: dict[str, Any],
    x: float,
    z: np.ndarray,
) -> np.ndarray | None:
    try:
        _, terms, n_p, T_e, _, _ = _local_terms_for_log_state(
            ops=ops,
            fluid=fluid,
            design=design,
            config=config,
            inlet=inlet,
            x=float(x),
            z=z,
        )
        det = float(terms["det"])
        if not np.isfinite(det) or abs(det) < 1e-300:
            return None
        dn_dx = (float(terms["rhs_m"]) * float(terms["E12"]) - float(terms["M12"]) * float(terms["rhs_e"])) / det
        dT_dx = (float(terms["M11"]) * float(terms["rhs_e"]) - float(terms["rhs_m"]) * float(terms["E11"])) / det
        rhs = np.array([dn_dx / max(n_p, 1e-300), dT_dx / max(T_e, 1e-300)], dtype=float)
        if not np.all(np.isfinite(rhs)):
            return None
        return rhs
    except Exception:
        return None


def build_implicit_reference_profile(
    *,
    design: DesignVector,
    config: CaseConfig,
    n_intervals: int | None = None,
    residual_tol: float = 1e-7,
    initial_substeps_per_interval: int = 10,
    max_log_step: float = 0.25,
    min_step_fraction: float = 1e-8,
    max_steps: int = 20000,
) -> ReferenceProfileResult:
    """Generate a profile with adaptive backward-Euler steps without dividing by det."""

    ops = ops_for_numeric()
    fluid = working_fluid_for_config(config)
    n_out = int(config.n_intervals if n_intervals is None else n_intervals) + 1
    x_eval = np.linspace(0.0, float(config.length_m), n_out, dtype=float)
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
    x = 0.0
    z = np.array([0.0, 0.0], dtype=float)
    h_base = float(config.length_m) / max((n_out - 1) * int(initial_substeps_per_interval), 1)
    h_min = float(config.length_m) * float(min_step_fraction)
    h = h_base
    xs = [float(x)]
    zs = [z.copy()]
    step_rows: list[dict[str, Any]] = []
    rejected_steps = 0

    for step_index in range(int(max_steps)):
        if x >= float(config.length_m) - 1e-14:
            break
        h = min(float(h), float(config.length_m) - float(x))
        rhs = _explicit_log_rhs_for_predictor(
            ops=ops,
            fluid=fluid,
            design=design,
            config=config,
            inlet=inlet,
            x=float(x),
            z=z,
        )
        if rhs is None:
            guess = z.copy()
        else:
            dz = h * rhs
            dz_scale = max(1.0, float(np.max(np.abs(dz))) / max(float(max_log_step), 1e-12))
            guess = z + dz / dz_scale

        lower = z - float(max_log_step)
        upper = z + float(max_log_step)
        guess = np.minimum(np.maximum(guess, lower), upper)
        x_next = float(x + h)

        def residual(candidate: np.ndarray) -> np.ndarray:
            try:
                values, _ = _scaled_step_residual(
                    ops=ops,
                    fluid=fluid,
                    design=design,
                    config=config,
                    inlet=inlet,
                    x_next=x_next,
                    z_prev=z,
                    z_next=np.asarray(candidate, dtype=float),
                    h=h,
                )
                if not np.all(np.isfinite(values)):
                    return np.array([1e30, 1e30], dtype=float)
                return values
            except Exception:
                return np.array([1e30, 1e30], dtype=float)

        sol = least_squares(
            residual,
            guess,
            bounds=(lower, upper),
            xtol=1e-10,
            ftol=1e-10,
            gtol=1e-10,
            max_nfev=80,
        )
        values, detail = _scaled_step_residual(
            ops=ops,
            fluid=fluid,
            design=design,
            config=config,
            inlet=inlet,
            x_next=x_next,
            z_prev=z,
            z_next=np.asarray(sol.x, dtype=float),
            h=h,
        )
        residual_inf = float(np.max(np.abs(values)))
        if bool(sol.success) and np.isfinite(residual_inf) and residual_inf <= float(residual_tol):
            x = x_next
            z = np.asarray(sol.x, dtype=float)
            xs.append(float(x))
            zs.append(z.copy())
            step_rows.append(
                {
                    "step": int(step_index),
                    "x": float(x),
                    "h": float(h),
                    "residual_inf": residual_inf,
                    **detail,
                }
            )
            h = min(float(h) * 1.25, h_base)
            continue

        rejected_steps += 1
        if h <= h_min:
            diagnostics = {
                "method": "adaptive_backward_euler_log_state",
                "ok": False,
                "step": int(step_index),
                "x": float(x),
                "x_fraction": float(x / max(float(config.length_m), 1e-300)),
                "attempted_h": float(h),
                "rejected_steps": int(rejected_steps),
                "residual_inf": residual_inf,
                "least_squares_success": bool(sol.success),
                "least_squares_message": str(sol.message),
                "candidate_z": np.asarray(sol.x, dtype=float).tolist(),
                "current_z": z.tolist(),
                "detail": detail,
                "working_fluid": fluid.to_dict(),
                "inlet": {key: float(value) for key, value in inlet.items()},
                "accepted_tail": step_rows[-5:],
            }
            return ReferenceProfileResult(
                ok=False,
                profile=None,
                diagnostics=diagnostics,
                error="implicit backward-Euler step failed at minimum step",
            )
        h *= 0.5

    else:
        return ReferenceProfileResult(
            ok=False,
            profile=None,
            diagnostics={
                "method": "adaptive_backward_euler_log_state",
                "ok": False,
                "max_steps": int(max_steps),
                "x": float(x),
                "x_fraction": float(x / max(float(config.length_m), 1e-300)),
                "working_fluid": fluid.to_dict(),
                "inlet": {key: float(value) for key, value in inlet.items()},
            },
            error="maximum implicit marching steps exceeded",
        )

    xs_arr = np.asarray(xs, dtype=float)
    zs_arr = np.vstack(zs)
    delta_log_n = np.interp(x_eval, xs_arr, zs_arr[:, 0])
    delta_log_Te = np.interp(x_eval, xs_arr, zs_arr[:, 1])
    log_n_in = float(design.log_n_p_in)
    log_Te_in = float(np.log(max(float(design.T_e_in), 1.0)))
    n_p = np.exp(log_n_in + delta_log_n)
    T_e = np.exp(log_Te_in + delta_log_Te)
    area = design.area_control.evaluate_profile(
        length=float(config.length_m),
        n_intervals=n_out - 1,
        area_scale=float(config.area_scale_m2),
    )
    profile = {
        "x": x_eval,
        "x_norm": x_eval / max(float(config.length_m), 1e-300),
        "n_p": n_p,
        "T_e": T_e,
        "A": np.asarray(area["A"], dtype=float),
        "sigma_logA": np.asarray(area["sigma_logA"], dtype=float),
    }
    diagnostics = {
        "method": "adaptive_backward_euler_log_state",
        "ok": True,
        "accepted_steps": int(len(step_rows)),
        "rejected_steps": int(rejected_steps),
        "min_h": float(np.min([row["h"] for row in step_rows])) if step_rows else 0.0,
        "max_h": float(np.max([row["h"] for row in step_rows])) if step_rows else 0.0,
        "max_residual_inf": float(np.max([row["residual_inf"] for row in step_rows])) if step_rows else 0.0,
        "min_det": float(np.min([row["det"] for row in step_rows])) if step_rows else 0.0,
        "max_det": float(np.max([row["det"] for row in step_rows])) if step_rows else 0.0,
        "min_mach": float(np.min([row["mach"] for row in step_rows])) if step_rows else 0.0,
        "max_mach": float(np.max([row["mach"] for row in step_rows])) if step_rows else 0.0,
        "delta_log_n_min": float(np.nanmin(delta_log_n)),
        "delta_log_n_max": float(np.nanmax(delta_log_n)),
        "delta_log_Te_min": float(np.nanmin(delta_log_Te)),
        "delta_log_Te_max": float(np.nanmax(delta_log_Te)),
        "n_p_min": float(np.nanmin(n_p)),
        "n_p_max": float(np.nanmax(n_p)),
        "T_e_min": float(np.nanmin(T_e)),
        "T_e_max": float(np.nanmax(T_e)),
        "working_fluid": fluid.to_dict(),
        "inlet": {key: float(value) for key, value in inlet.items()},
        "accepted_tail": step_rows[-5:],
    }
    return ReferenceProfileResult(ok=True, profile=profile, diagnostics=diagnostics)


def build_freidberg_reference_profile(
    *,
    design: DesignVector,
    config: CaseConfig,
    n_intervals: int | None = None,
    residual_tol: float = 1e-7,
    initial_substeps_per_interval: int = 10,
    max_log_step: float = 0.25,
    min_step_fraction: float = 1e-8,
    max_steps: int = 20000,
    area_profile: dict[str, Any] | None = None,
) -> ReferenceProfileResult:
    """Generate a profile with adaptive Freidberg H/L balance steps."""

    ops = ops_for_numeric()
    fluid = working_fluid_for_config(config)
    n_out = int(config.n_intervals if n_intervals is None else n_intervals) + 1
    x_eval = np.linspace(0.0, float(config.length_m), n_out, dtype=float)
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
    x = 0.0
    z = np.array([0.0, 0.0], dtype=float)
    h_base = float(config.length_m) / max((n_out - 1) * int(initial_substeps_per_interval), 1)
    h_min = float(config.length_m) * float(min_step_fraction)
    h = h_base
    xs = [float(x)]
    zs = [z.copy()]
    step_rows: list[dict[str, Any]] = []
    rejected_steps = 0

    for step_index in range(int(max_steps)):
        if x >= float(config.length_m) - 1e-14:
            break
        h = min(float(h), float(config.length_m) - float(x))
        x_next = float(x + h)
        guess = z.copy()
        lower = z - float(max_log_step)
        upper = z + float(max_log_step)

        def residual(candidate: np.ndarray) -> np.ndarray:
            try:
                values, _ = _freidberg_step_residual(
                    ops=ops,
                    fluid=fluid,
                    design=design,
                    config=config,
                    inlet=inlet,
                    x_prev=float(x),
                    z_prev=z,
                    x_next=x_next,
                    z_next=np.asarray(candidate, dtype=float),
                    h=h,
                    area_profile=area_profile,
                )
                if not np.all(np.isfinite(values)):
                    return np.array([1e30, 1e30], dtype=float)
                return values
            except Exception:
                return np.array([1e30, 1e30], dtype=float)

        sol = least_squares(
            residual,
            guess,
            bounds=(lower, upper),
            xtol=1e-10,
            ftol=1e-10,
            gtol=1e-10,
            max_nfev=100,
        )
        values, detail = _freidberg_step_residual(
            ops=ops,
            fluid=fluid,
            design=design,
            config=config,
            inlet=inlet,
            x_prev=float(x),
            z_prev=z,
            x_next=x_next,
            z_next=np.asarray(sol.x, dtype=float),
            h=h,
            area_profile=area_profile,
        )
        residual_inf = float(np.max(np.abs(values)))
        if bool(sol.success) and np.isfinite(residual_inf) and residual_inf <= float(residual_tol):
            x = x_next
            z = np.asarray(sol.x, dtype=float)
            xs.append(float(x))
            zs.append(z.copy())
            step_rows.append(
                {
                    "step": int(step_index),
                    "x": float(x),
                    "h": float(h),
                    "residual_inf": residual_inf,
                    **detail,
                }
            )
            h = min(float(h) * 1.25, h_base)
            continue

        rejected_steps += 1
        if h <= h_min:
            diagnostics = {
                "method": "adaptive_backward_euler_freidberg_hl",
                "ok": False,
                "step": int(step_index),
                "x": float(x),
                "x_fraction": float(x / max(float(config.length_m), 1e-300)),
                "attempted_h": float(h),
                "rejected_steps": int(rejected_steps),
                "residual_inf": residual_inf,
                "least_squares_success": bool(sol.success),
                "least_squares_message": str(sol.message),
                "candidate_z": np.asarray(sol.x, dtype=float).tolist(),
                "current_z": z.tolist(),
                "detail": detail,
                "working_fluid": fluid.to_dict(),
                "inlet": {key: float(value) for key, value in inlet.items()},
                "area_profile_source": "fixed_profile" if area_profile is not None else "design_spline",
                "accepted_tail": step_rows[-5:],
            }
            return ReferenceProfileResult(
                ok=False,
                profile=None,
                diagnostics=diagnostics,
                error="Freidberg H/L backward-Euler step failed at minimum step",
            )
        h *= 0.5

    else:
        return ReferenceProfileResult(
            ok=False,
            profile=None,
            diagnostics={
                "method": "adaptive_backward_euler_freidberg_hl",
                "ok": False,
                "max_steps": int(max_steps),
                "x": float(x),
                "x_fraction": float(x / max(float(config.length_m), 1e-300)),
                "working_fluid": fluid.to_dict(),
                "inlet": {key: float(value) for key, value in inlet.items()},
            },
            error="maximum Freidberg H/L marching steps exceeded",
        )

    xs_arr = np.asarray(xs, dtype=float)
    zs_arr = np.vstack(zs)
    delta_log_n = np.interp(x_eval, xs_arr, zs_arr[:, 0])
    delta_log_Te = np.interp(x_eval, xs_arr, zs_arr[:, 1])
    log_n_in = float(design.log_n_p_in)
    log_Te_in = float(np.log(max(float(design.T_e_in), 1.0)))
    n_p = np.exp(log_n_in + delta_log_n)
    T_e = np.exp(log_Te_in + delta_log_Te)
    area = design.area_control.evaluate_profile(
        length=float(config.length_m),
        n_intervals=n_out - 1,
        area_scale=float(config.area_scale_m2),
    ) if area_profile is None else _area_profile_arrays(area_profile, config=config, n_intervals=n_out - 1)
    profile = {
        "x": x_eval,
        "x_norm": x_eval / max(float(config.length_m), 1e-300),
        "n_p": n_p,
        "T_e": T_e,
        "A": np.asarray(area["A"], dtype=float),
        "sigma_logA": np.asarray(area["sigma_logA"], dtype=float),
    }
    diagnostics = {
        "method": "adaptive_backward_euler_freidberg_hl",
        "ok": True,
        "accepted_steps": int(len(step_rows)),
        "rejected_steps": int(rejected_steps),
        "min_h": float(np.min([row["h"] for row in step_rows])) if step_rows else 0.0,
        "max_h": float(np.max([row["h"] for row in step_rows])) if step_rows else 0.0,
        "max_residual_inf": float(np.max([row["residual_inf"] for row in step_rows])) if step_rows else 0.0,
        "min_det": float(np.min([row["det"] for row in step_rows])) if step_rows else 0.0,
        "max_det": float(np.max([row["det"] for row in step_rows])) if step_rows else 0.0,
        "min_mach": float(np.min([row["mach"] for row in step_rows])) if step_rows else 0.0,
        "max_mach": float(np.max([row["mach"] for row in step_rows])) if step_rows else 0.0,
        "min_H_p": float(np.min([row["H_p"] for row in step_rows])) if step_rows else 0.0,
        "max_H_p": float(np.max([row["H_p"] for row in step_rows])) if step_rows else 0.0,
        "min_L_p": float(np.min([row["L_p"] for row in step_rows])) if step_rows else 0.0,
        "max_L_p": float(np.max([row["L_p"] for row in step_rows])) if step_rows else 0.0,
        "delta_log_n_min": float(np.nanmin(delta_log_n)),
        "delta_log_n_max": float(np.nanmax(delta_log_n)),
        "delta_log_Te_min": float(np.nanmin(delta_log_Te)),
        "delta_log_Te_max": float(np.nanmax(delta_log_Te)),
        "n_p_min": float(np.nanmin(n_p)),
        "n_p_max": float(np.nanmax(n_p)),
        "T_e_min": float(np.nanmin(T_e)),
        "T_e_max": float(np.nanmax(T_e)),
        "working_fluid": fluid.to_dict(),
        "inlet": {key: float(value) for key, value in inlet.items()},
        "area_profile_source": "fixed_profile" if area_profile is not None else "design_spline",
        "accepted_tail": step_rows[-5:],
    }
    return ReferenceProfileResult(ok=True, profile=profile, diagnostics=diagnostics)
