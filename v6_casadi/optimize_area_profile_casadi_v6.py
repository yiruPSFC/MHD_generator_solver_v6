#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
import sys

import casadi as ca
import numpy as np

REPO_DIR = Path(__file__).resolve().parents[1]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from v6_global_marginal.global_postprocess_v6 import compute_global_metrics
from v6_global_marginal.global_postprocess_v6 import (
    compute_design_value_terms,
    design_value_weights_delta_te_only,
    design_value_weights_lab_poc,
    evaluate_design_value,
)
from v6_core.local_algebraic_closure import (
    B_FIELD,
    E_CHARGE,
    E_I,
    H_P,
    K_B,
    M_E,
    M_P,
    SIGMA_EP,
    beta_from_np_te,
    ne_from_np_te,
)
from v6_global_marginal.local_algebraic_closure_global import (
    compute_currents_fields_global,
    local_closure_global_with_partials,
)
from v6_batch.pde_solver_v6_batch import _inlet_velocity_from_eq8_prime
from v6_global_marginal.pde_solver_v6_batch_global import (
    ForwardPDESolverV6BatchGlobal,
    event_name_from_code,
    project_seed_fraction_to_marginal_inlet,
)


_EPS = 1e-30
_TP_MIN = 1.0
_DELTA_MIN = 1e-12
_FION_MIN = 1e-12
_FION_MAX = 1.0 - 1e-12
_SAHA_K_MIN = 1e-100


@dataclass(frozen=True)
class InletConstants:
    seed_fraction: float
    seed_mode: str
    dot_N: float
    I_0: float
    v_in: float


@dataclass(frozen=True)
class WarmStartProfile:
    x: np.ndarray
    n_p: np.ndarray
    T_e: np.ndarray
    A: np.ndarray
    sigma_logA: np.ndarray
    source: str


@dataclass(frozen=True)
class OptimizedAreaProfile:
    success: bool
    acceptable: bool
    return_status: str
    objective_delta_Te: float
    objective_value: float
    inlet: InletConstants
    transcription: str
    x: np.ndarray
    n_p: np.ndarray
    T_e: np.ndarray
    T_p: np.ndarray
    A: np.ndarray
    v_p: np.ndarray
    n_e: np.ndarray
    beta: np.ndarray
    eta: np.ndarray
    Z: np.ndarray
    J_x: np.ndarray
    J_y: np.ndarray
    E_x: np.ndarray
    mach: np.ndarray
    velikhov_margin: np.ndarray
    sigma_logA: np.ndarray
    warm_start_source: str
    stats: dict
    diagnostics: dict


@dataclass(frozen=True)
class FeasibilityThresholds:
    defect_inf_tol: float = 1e-4
    defect_rms_tol: float = 1e-5
    boundary_inf_tol: float = 1e-8
    path_slack_tol: float = 1e-6


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Optimize A(x) with CasADi + IPOPT under Velikhov margin constraints"
    )
    p.add_argument("--np-in", type=float, required=True)
    p.add_argument("--z-in", type=float, required=True)
    p.add_argument("--tp-in", type=float, required=True)
    p.add_argument("--te-in", type=float, required=True)
    p.add_argument("--A-in", type=float, required=True, help="inlet cross-section area [m^2]")
    p.add_argument(
        "--seed-fraction",
        type=float,
        default=None,
        help="if omitted, project the inlet onto the marginal Velikhov boundary",
    )
    p.add_argument("--B", type=float, default=B_FIELD)
    p.add_argument("--L", type=float, default=5.4)
    p.add_argument("--n-intervals", type=int, default=80)
    p.add_argument(
        "--transcription",
        type=str,
        choices=("trapezoid", "hermite-simpson"),
        default="hermite-simpson",
    )
    p.add_argument("--min-margin", type=float, default=0.0)
    p.add_argument("--A-min-ratio", type=float, default=0.25)
    p.add_argument("--A-max-ratio", type=float, default=4.0)
    p.add_argument(
        "--max-abs-dlogA-dx",
        type=float,
        default=2.0,
        help="bound on d/dx log(A), in 1/m",
    )
    p.add_argument("--np-min-ratio", type=float, default=1e-6)
    p.add_argument("--np-max-ratio", type=float, default=100.0)
    p.add_argument("--te-min", type=float, default=100.0)
    p.add_argument(
        "--te-max-ratio",
        type=float,
        default=20.0,
        help="upper bound T_e(x) <= te_max_ratio * T_e_in for IPOPT robustness",
    )
    p.add_argument("--tp-min", type=float, default=1.0)
    p.add_argument("--mach-min", type=float, default=None)
    p.add_argument("--mach-max", type=float, default=None)
    p.add_argument(
        "--margin-slack-max",
        type=float,
        default=0.0,
        help="optional nonnegative slack cap for the G >= min_margin path constraint",
    )
    p.add_argument(
        "--margin-slack-weight",
        type=float,
        default=0.0,
        help="L1 exact-penalty weight applied to the Velikhov-margin slack variables",
    )
    p.add_argument(
        "--smooth-weight",
        type=float,
        default=0.0,
        help="quadratic regularization weight on d/dx log(A)",
    )
    p.add_argument(
        "--control-slew-weight",
        type=float,
        default=0.0,
        help="quadratic regularization weight on d/dx sigma_logA",
    )
    p.add_argument(
        "--control-curvature-weight",
        type=float,
        default=0.0,
        help="quadratic regularization weight on d^2/dx^2 sigma_logA",
    )
    p.add_argument(
        "--state-curvature-weight",
        type=float,
        default=0.0,
        help="quadratic regularization weight on d^2/dx^2 of the scaled states",
    )
    p.add_argument(
        "--warm-profile-track-weight",
        type=float,
        default=0.0,
        help="quadratic tracking weight that keeps the solution close to the warm-start state profile",
    )
    p.add_argument(
        "--warm-control-track-weight",
        type=float,
        default=0.0,
        help="quadratic tracking weight that keeps dlogA/dx close to the warm-start control",
    )
    p.add_argument(
        "--warm-start",
        type=str,
        choices=("marginal", "constant"),
        default="marginal",
    )
    p.add_argument(
        "--warm-start-dx",
        type=float,
        default=None,
        help="dx for the current G=0 global solver when warm-starting from the marginal profile",
    )
    p.add_argument("--ipopt-max-iter", type=int, default=1000)
    p.add_argument("--ipopt-tol", type=float, default=1e-7)
    p.add_argument(
        "--objective-weight",
        type=float,
        default=1.0,
        help="weight on the Te(out)-Te(in) objective; use <1 for feasibility-first homotopy stages",
    )
    p.add_argument("--out-json", type=str, default="")
    p.add_argument("--out-npz", type=str, default="")
    return p


def _safe_signed_scalar(x: float) -> float:
    if abs(x) >= _EPS:
        return x
    return _EPS if x >= 0.0 else -_EPS


def _jsonify_stats(stats: dict) -> dict:
    out: dict[str, object] = {}
    for key, value in stats.items():
        if str(key) == "iterations":
            continue
        if isinstance(value, (bool, int, str)):
            out[str(key)] = value
            continue
        if isinstance(value, float):
            out[str(key)] = float(value)
            continue
        if isinstance(value, np.ndarray):
            out[str(key)] = value.tolist()
            continue
        try:
            out[str(key)] = float(value)
        except Exception:
            out[str(key)] = str(value)
    return out


def _count_sign_changes(values: np.ndarray, *, zero_tol: float = 1e-12) -> int:
    out = 0
    prev = 0
    for value in np.asarray(values, dtype=float).reshape(-1):
        if not np.isfinite(value) or abs(float(value)) <= zero_tol:
            continue
        sign = 1 if value > 0.0 else -1
        if prev != 0 and sign != prev:
            out += 1
        prev = sign
    return out


def _total_variation(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size < 2:
        return 0.0
    return float(np.sum(np.abs(np.diff(arr))))


def _total_variation_ratio(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size < 2:
        return 0.0
    net = float(abs(arr[-1] - arr[0]))
    scale = float(np.nanmax(np.abs(arr))) if arr.size else 1.0
    denom = max(net, 1e-12 * max(scale, 1.0))
    return _total_variation(arr) / denom


def _fraction_near_bounds(values: np.ndarray, *, lower: float, upper: float) -> tuple[float, float, float]:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size == 0:
        return 0.0, 0.0, 0.0
    span = max(abs(float(upper) - float(lower)), _EPS)
    atol = max(0.02 * span, 1e-8)
    near_lower = float(np.mean(arr <= float(lower) + atol))
    near_upper = float(np.mean(arr >= float(upper) - atol))
    return near_lower, near_upper, min(1.0, near_lower + near_upper)


def _prepare_inlet_constants(
    *,
    n_p_in: float,
    Z_in: float,
    T_p_in: float,
    T_e_in: float,
    A_in: float,
    B: float,
    seed_fraction: float | None,
) -> InletConstants:
    if A_in <= 0.0:
        raise ValueError("A_in must be positive.")

    if seed_fraction is None:
        status, projected = project_seed_fraction_to_marginal_inlet(
            n_p_in=float(n_p_in),
            T_p_in=float(T_p_in),
            T_e_in=float(T_e_in),
            B=float(B),
        )
        if status != 0:
            raise ValueError(
                f"marginal seed projection failed with event={event_name_from_code(int(status))}"
            )
        seed_mode = "projected_marginal"
        seed_fraction_final = float(projected)
    else:
        seed_fraction_final = float(seed_fraction)
        if seed_fraction_final <= 0.0 or seed_fraction_final >= 1.0:
            raise ValueError("seed_fraction must satisfy 0 < seed_fraction < 1.")
        seed_mode = "specified"

    v_in, inlet_status = _inlet_velocity_from_eq8_prime(
        float(T_e_in), float(T_p_in), float(n_p_in), float(Z_in), float(B)
    )
    if inlet_status != 0:
        raise ValueError(
            f"inlet velocity evaluation failed with event={event_name_from_code(int(inlet_status))}"
        )

    n_e_in = ne_from_np_te(float(n_p_in), float(T_e_in), seed_fraction_final)
    beta_in = beta_from_np_te(float(n_p_in), float(T_e_in), B=float(B), sigma_ep=SIGMA_EP)
    b2 = beta_in * beta_in
    den = _safe_signed_scalar(b2 + 1.0 + float(Z_in))
    J_x_in = b2 / den * E_CHARGE * n_e_in * v_in
    dot_N = float(n_p_in) * float(v_in) * float(A_in)
    I_0 = float(J_x_in) * float(A_in)

    return InletConstants(
        seed_fraction=seed_fraction_final,
        seed_mode=seed_mode,
        dot_N=dot_N,
        I_0=I_0,
        v_in=float(v_in),
    )


def _make_stage_function(*, dot_N: float, I_0: float, seed_fraction: float, B: float) -> ca.Function:
    x = ca.SX.sym("x", 3)
    sigma = ca.SX.sym("sigma")

    n_p = x[0]
    T_e = x[1]
    A = x[2]

    n_p_safe = ca.fmax(n_p, 1.0)
    T_e_safe = ca.fmax(T_e, 1.0)
    A_safe = ca.fmax(A, 1e-12)

    v_te = ca.sqrt(2.0 * K_B * T_e_safe / M_E)
    beta = E_CHARGE * B / (M_E * n_p_safe * SIGMA_EP * v_te + _EPS)

    saha_a = 2.0 * math.pi * M_E * K_B * T_e_safe / (H_P * H_P)
    saha_k = (saha_a ** 1.5) * ca.exp(-E_I / (K_B * T_e_safe))
    saha_k_safe = ca.fmax(saha_k, _SAHA_K_MIN)
    n_s = seed_fraction * n_p_safe
    n_e = 2.0 * n_s / (1.0 + ca.sqrt(1.0 + 4.0 * n_s / saha_k_safe))

    eta = M_E * n_p_safe * SIGMA_EP * v_te / (E_CHARGE * E_CHARGE * n_e + _EPS)
    q = E_CHARGE * n_e * dot_N / (I_0 * n_p_safe + _EPS)
    b2 = beta * beta
    Z = b2 * (q - 1.0) - 1.0

    one_plus_z = 1.0 + Z
    den = b2 + one_plus_z
    F = b2 * (b2 + one_plus_z * one_plus_z) / (den * den + _EPS)

    v_p = dot_N / (n_p_safe * A_safe)
    T_p = T_e_safe - M_P * v_p * v_p * F / (3.0 * K_B)
    T_p_safe_for_math = ca.fmax(T_p, _TP_MIN)
    dTp_dnp = ca.gradient(T_p, n_p)
    dTp_dTe = ca.gradient(T_p, T_e)
    dTp_dA = ca.gradient(T_p, A)

    jfac = E_CHARGE * n_e * v_p
    J_x = I_0 / A_safe
    J_y = -beta * one_plus_z / (den + _EPS) * jfac
    E_x = -b2 * Z / (den + _EPS) * eta * jfac
    nu_E = eta * 2.0 * E_CHARGE * E_CHARGE * n_e / M_P

    dA_dx = sigma * A_safe

    M11 = (-M_P * v_p * v_p + K_B * T_p) + K_B * n_p_safe * dTp_dnp
    M12 = K_B * n_p_safe * dTp_dTe
    M13 = K_B * n_p_safe * dTp_dA - M_P * n_p_safe * v_p * v_p / A_safe

    E11 = -T_p + 1.5 * n_p_safe * dTp_dnp
    E12 = 1.5 * n_p_safe * dTp_dTe
    E13 = 1.5 * n_p_safe * dTp_dA

    rhs_m = J_y * B - M13 * dA_dx
    rhs_e = 1.5 * nu_E * n_e * (T_e_safe - T_p) / (v_p + _EPS) - E13 * dA_dx
    det = M11 * E12 - M12 * E11
    det_safe = det + _EPS

    dn_dx = (rhs_m * E12 - M12 * rhs_e) / det_safe
    dTe_dx = (M11 * rhs_e - rhs_m * E11) / det_safe

    c_s = ca.sqrt((5.0 / 3.0) * K_B * T_p_safe_for_math / M_P + _EPS)
    mach = v_p / c_s

    f_I_raw = n_e / (n_s + _EPS)
    f_I = ca.fmin(ca.fmax(f_I_raw, _FION_MIN), _FION_MAX)
    delta_raw = T_e_safe / T_p_safe_for_math - 1.0
    delta = ca.fmax(delta_raw, _DELTA_MIN)
    alpha = (K_B * T_e_safe / (2.0 * E_I)) * (2.0 - f_I) / (1.0 - f_I + _EPS)
    G = 4.0 * alpha * (2.0 + 1.0 / delta) * (
        1.0 + alpha * (1.0 + 1.0 / delta)
    ) - b2

    out = ca.vertcat(dn_dx, dTe_dx, dA_dx, T_p, v_p, n_e, beta, eta, Z, J_x, J_y, E_x, mach, G)
    return ca.Function("stage_fun", [x, sigma], [out])


def _build_constant_warm_start(
    *,
    x: np.ndarray,
    n_p_in: float,
    T_e_in: float,
    A_in: float,
) -> WarmStartProfile:
    n_points = x.size
    return WarmStartProfile(
        x=x.copy(),
        n_p=np.full(n_points, float(n_p_in), dtype=float),
        T_e=np.full(n_points, float(T_e_in), dtype=float),
        A=np.full(n_points, float(A_in), dtype=float),
        sigma_logA=np.zeros(n_points - 1, dtype=float),
        source="constant",
    )


def _build_marginal_warm_start(
    *,
    x: np.ndarray,
    n_p_in: float,
    Z_in: float,
    T_p_in: float,
    T_e_in: float,
    A_in: float,
    B: float,
    length: float,
    warm_start_dx: float | None,
) -> WarmStartProfile | None:
    dx = warm_start_dx
    if dx is None:
        dx = length / max(8 * (x.size - 1), 200)
    dx = min(max(float(dx), length / 5000.0), length / max(x.size - 1, 1))

    solver = ForwardPDESolverV6BatchGlobal(B=float(B), length=float(length))
    out = solver.solve_batch(
        n_p_in=np.array([n_p_in], dtype=float),
        Z_in=np.array([Z_in], dtype=float),
        T_p_in=np.array([T_p_in], dtype=float),
        T_e_in=np.array([T_e_in], dtype=float),
        A_in=np.array([A_in], dtype=float),
        dx=dx,
        store_profiles=True,
    )
    if (not bool(out.success[0])) or (not bool(out.reached_end[0])):
        return None

    idx_last = int(out.valid_points[0]) - 1
    if idx_last < 1:
        return None

    x_ref = np.asarray(out.x[: idx_last + 1], dtype=float)
    n_p_ref = np.asarray(out.n_p[0, : idx_last + 1], dtype=float)
    T_e_ref = np.asarray(out.T_e[0, : idx_last + 1], dtype=float)
    A_ref = np.asarray(out.A[0, : idx_last + 1], dtype=float)

    n_p_guess = np.interp(x, x_ref, n_p_ref)
    T_e_guess = np.interp(x, x_ref, T_e_ref)
    A_guess = np.interp(x, x_ref, A_ref)
    sigma_guess = np.diff(np.log(np.maximum(A_guess, 1e-20))) / np.diff(x)

    return WarmStartProfile(
        x=x.copy(),
        n_p=n_p_guess,
        T_e=T_e_guess,
        A=A_guess,
        sigma_logA=sigma_guess,
        source="marginal_global_solver",
    )


def _evaluate_profile_numeric(
    *,
    x: np.ndarray,
    n_p: np.ndarray,
    T_e: np.ndarray,
    A: np.ndarray,
    inlet: InletConstants,
    B: float,
    sigma_logA: np.ndarray,
) -> dict[str, np.ndarray]:
    n_points = x.size

    T_p = np.full(n_points, np.nan, dtype=float)
    v_p = np.full(n_points, np.nan, dtype=float)
    n_e = np.full(n_points, np.nan, dtype=float)
    beta = np.full(n_points, np.nan, dtype=float)
    eta = np.full(n_points, np.nan, dtype=float)
    Z = np.full(n_points, np.nan, dtype=float)
    J_x = np.full(n_points, np.nan, dtype=float)
    J_y = np.full(n_points, np.nan, dtype=float)
    E_x = np.full(n_points, np.nan, dtype=float)
    mach = np.full(n_points, np.nan, dtype=float)
    G = np.full(n_points, np.nan, dtype=float)

    for i in range(n_points):
        vals = local_closure_global_with_partials(
            n_p=float(n_p[i]),
            T_e=float(T_e[i]),
            A=float(A[i]),
            dot_N=inlet.dot_N,
            I_0=inlet.I_0,
            seed_fraction=inlet.seed_fraction,
            B=float(B),
            sigma_ep=SIGMA_EP,
        )
        v_p[i] = float(vals[0])
        n_e[i] = float(vals[1])
        beta[i] = float(vals[2])
        eta[i] = float(vals[3])
        Z[i] = float(vals[4])
        T_p[i] = float(vals[5])
        G[i] = float(vals[13])
        J_x[i], J_y[i], E_x[i], _ = compute_currents_fields_global(
            v_p=float(v_p[i]),
            n_e=float(n_e[i]),
            beta=float(beta[i]),
            eta=float(eta[i]),
            Z=float(Z[i]),
            I_0=float(inlet.I_0),
            A=float(A[i]),
        )
        c_s = math.sqrt((5.0 / 3.0) * K_B * max(T_p[i], _TP_MIN) / M_P)
        mach[i] = v_p[i] / max(c_s, _EPS)

    return {
        "x": x,
        "n_p": n_p,
        "T_e": T_e,
        "T_p": T_p,
        "A": A,
        "v_p": v_p,
        "n_e": n_e,
        "beta": beta,
        "eta": eta,
        "Z": Z,
        "J_x": J_x,
        "J_y": J_y,
        "E_x": E_x,
        "mach": mach,
        "velikhov_margin": G,
        "sigma_logA": sigma_logA,
    }


def _resample_warm_profile(warm: WarmStartProfile, x_new: np.ndarray) -> WarmStartProfile:
    x_old = np.asarray(warm.x, dtype=float)
    if x_old.ndim != 1 or x_old.size < 2:
        raise ValueError("warm profile x must be a 1D array with at least two points.")
    if np.any(np.diff(x_old) <= 0.0):
        raise ValueError("warm profile x must be strictly increasing.")

    n_p_new = np.interp(x_new, x_old, np.asarray(warm.n_p, dtype=float))
    T_e_new = np.interp(x_new, x_old, np.asarray(warm.T_e, dtype=float))
    A_new = np.interp(x_new, x_old, np.asarray(warm.A, dtype=float))
    sigma_new = np.diff(np.log(np.maximum(A_new, 1e-20))) / np.diff(x_new)

    return WarmStartProfile(
        x=x_new.copy(),
        n_p=n_p_new,
        T_e=T_e_new,
        A=A_new,
        sigma_logA=sigma_new,
        source=warm.source,
    )


def _project_warm_profile_to_bounds(
    warm: WarmStartProfile,
    *,
    n_p_floor: float,
    n_p_ceil: float,
    T_e_floor: float,
    T_e_ceil: float,
    A_floor: float,
    A_ceil: float,
    sigma_floor: float,
    sigma_ceil: float,
    inlet_target: tuple[float, float, float],
) -> WarmStartProfile:
    n_p = np.clip(np.asarray(warm.n_p, dtype=float), n_p_floor, n_p_ceil)
    T_e = np.clip(np.asarray(warm.T_e, dtype=float), T_e_floor, T_e_ceil)
    A = np.clip(np.asarray(warm.A, dtype=float), A_floor, A_ceil)
    n_p[0] = float(inlet_target[0])
    T_e[0] = float(inlet_target[1])
    A[0] = float(inlet_target[2])
    sigma = np.diff(np.log(np.maximum(A, 1e-20))) / np.diff(np.asarray(warm.x, dtype=float))
    sigma = np.clip(sigma, sigma_floor, sigma_ceil)
    return WarmStartProfile(
        x=np.asarray(warm.x, dtype=float).copy(),
        n_p=n_p,
        T_e=T_e,
        A=A,
        sigma_logA=sigma,
        source=warm.source,
    )


def _compute_regularity_diagnostics(
    *,
    x_nodes: np.ndarray,
    A: np.ndarray,
    sigma_logA: np.ndarray,
    sigma_bounds: tuple[float, float],
) -> dict[str, object]:
    dx = np.diff(np.asarray(x_nodes, dtype=float))
    dx_mean = float(np.mean(dx)) if dx.size else 1.0
    sigma = np.asarray(sigma_logA, dtype=float).reshape(-1)
    area_steps = np.diff(np.asarray(A, dtype=float).reshape(-1))

    if sigma.size > 1:
        sigma_slew = np.diff(sigma) / dx_mean
        sigma_slew_rms = float(np.sqrt(np.mean(sigma_slew * sigma_slew)))
    else:
        sigma_slew_rms = 0.0

    if sigma.size > 2:
        sigma_curvature = (sigma[2:] - 2.0 * sigma[1:-1] + sigma[:-2]) / (dx_mean * dx_mean)
        sigma_curvature_rms = float(np.sqrt(np.mean(sigma_curvature * sigma_curvature)))
    else:
        sigma_curvature_rms = 0.0

    sigma_lower, sigma_upper = sigma_bounds
    sigma_near_lower, sigma_near_upper, sigma_bound_hit_fraction = _fraction_near_bounds(
        sigma,
        lower=float(sigma_lower),
        upper=float(sigma_upper),
    )

    n_intervals = int(sigma.size)
    max_turns = max(4, int(math.ceil(max(n_intervals, 1) / 20.0)))
    area_tv_ratio = _total_variation_ratio(A)
    regularity_ok = bool(
        _count_sign_changes(area_steps) <= max_turns
        and _count_sign_changes(sigma) <= max_turns
        and area_tv_ratio <= 1.10
        and sigma_bound_hit_fraction <= 0.35
    )

    return {
        "area_step_sign_changes": _count_sign_changes(area_steps),
        "area_total_variation": _total_variation(A),
        "area_total_variation_ratio": area_tv_ratio,
        "sigma_sign_changes": _count_sign_changes(sigma),
        "sigma_total_variation": _total_variation(sigma),
        "sigma_slew_rms": sigma_slew_rms,
        "sigma_curvature_rms": sigma_curvature_rms,
        "sigma_near_lower_bound_fraction": sigma_near_lower,
        "sigma_near_upper_bound_fraction": sigma_near_upper,
        "sigma_bound_hit_fraction": sigma_bound_hit_fraction,
        "regularity_ok": regularity_ok,
        "regularity_thresholds": {
            "max_turns": max_turns,
            "max_area_total_variation_ratio": 1.10,
            "max_sigma_bound_hit_fraction": 0.35,
        },
    }


def _compute_interval_defects(
    *,
    transcription: str,
    x_nodes: np.ndarray,
    n_p: np.ndarray,
    T_e: np.ndarray,
    A: np.ndarray,
    sigma_logA: np.ndarray,
    stage_fun: ca.Function,
    scales: tuple[float, float, float],
) -> tuple[np.ndarray, np.ndarray]:
    np_scale, te_scale, A_scale = scales
    n_intervals = sigma_logA.size
    defects = np.zeros((n_intervals, 3), dtype=float)
    mid_defects = np.zeros((n_intervals, 3), dtype=float)

    for k in range(n_intervals):
        dx = float(x_nodes[k + 1] - x_nodes[k])
        xk = np.array([n_p[k], T_e[k], A[k]], dtype=float)
        xkp1 = np.array([n_p[k + 1], T_e[k + 1], A[k + 1]], dtype=float)
        uk = float(sigma_logA[k])
        out_k = np.asarray(stage_fun(xk, uk), dtype=float).reshape(-1)
        out_kp1 = np.asarray(stage_fun(xkp1, uk), dtype=float).reshape(-1)
        fk = np.array([out_k[0] / np_scale, out_k[1] / te_scale, out_k[2] / A_scale], dtype=float)
        fkp1 = np.array(
            [out_kp1[0] / np_scale, out_kp1[1] / te_scale, out_kp1[2] / A_scale],
            dtype=float,
        )
        xk_hat = np.array([n_p[k] / np_scale, T_e[k] / te_scale, A[k] / A_scale], dtype=float)
        xkp1_hat = np.array(
            [n_p[k + 1] / np_scale, T_e[k + 1] / te_scale, A[k + 1] / A_scale],
            dtype=float,
        )

        if transcription == "trapezoid":
            defects[k, :] = xkp1_hat - xk_hat - 0.5 * dx * (fk + fkp1)
            mid_defects[k, :] = 0.0
            continue

        xmid_phys = 0.5 * (xk + xkp1) + 0.125 * dx * np.array(
            [out_k[0] - out_kp1[0], out_k[1] - out_kp1[1], out_k[2] - out_kp1[2]],
            dtype=float,
        )
        out_mid = np.asarray(stage_fun(xmid_phys, uk), dtype=float).reshape(-1)
        fmid = np.array(
            [out_mid[0] / np_scale, out_mid[1] / te_scale, out_mid[2] / A_scale],
            dtype=float,
        )
        xmid_hat_target = 0.5 * (xk_hat + xkp1_hat) + 0.125 * dx * (fk - fkp1)
        xmid_hat_actual = np.array(
            [xmid_phys[0] / np_scale, xmid_phys[1] / te_scale, xmid_phys[2] / A_scale],
            dtype=float,
        )
        mid_defects[k, :] = xmid_hat_actual - xmid_hat_target
        defects[k, :] = xkp1_hat - xk_hat - dx / 6.0 * (fk + 4.0 * fmid + fkp1)

    return defects, mid_defects


def _compute_feasibility_diagnostics(
    *,
    transcription: str,
    x_nodes: np.ndarray,
    n_p: np.ndarray,
    T_e: np.ndarray,
    A: np.ndarray,
    sigma_logA: np.ndarray,
    T_p: np.ndarray,
    mach: np.ndarray,
    velikhov_margin: np.ndarray,
    stage_fun: ca.Function,
    inlet_target: tuple[float, float, float],
    state_bounds: dict[str, float | None],
    sigma_bounds: tuple[float, float],
    thresholds: FeasibilityThresholds,
    margin_slack_nodes: np.ndarray | None = None,
    margin_slack_mid: np.ndarray | None = None,
) -> dict:
    np_scale = float(inlet_target[0])
    te_scale = float(max(inlet_target[1], 1.0))
    A_scale = float(inlet_target[2])
    defects, mid_defects = _compute_interval_defects(
        transcription=transcription,
        x_nodes=x_nodes,
        n_p=n_p,
        T_e=T_e,
        A=A,
        sigma_logA=sigma_logA,
        stage_fun=stage_fun,
        scales=(np_scale, te_scale, A_scale),
    )

    initial_residual = np.array(
        [n_p[0] - inlet_target[0], T_e[0] - inlet_target[1], A[0] - inlet_target[2]],
        dtype=float,
    )
    initial_residual_scaled = np.array(
        [initial_residual[0] / np_scale, initial_residual[1] / te_scale, initial_residual[2] / A_scale],
        dtype=float,
    )

    finite_profile = bool(
        np.all(np.isfinite(n_p))
        and np.all(np.isfinite(T_e))
        and np.all(np.isfinite(A))
        and np.all(np.isfinite(T_p))
        and np.all(np.isfinite(mach))
        and np.all(np.isfinite(velikhov_margin))
        and np.all(np.isfinite(defects))
        and np.all(np.isfinite(mid_defects))
    )

    violations = []
    path_slack_tol = float(thresholds.path_slack_tol)
    tp_min = float(np.nanmin(T_p))
    margin_min = float(np.nanmin(velikhov_margin))
    mach_min_val = float(np.nanmin(mach))
    mach_max_val = float(np.nanmax(mach))

    state_np_min = state_bounds["np_floor"]
    state_np_max = state_bounds["np_ceil"]
    te_min = state_bounds["te_floor"]
    te_max = state_bounds["te_ceil"]
    A_min = state_bounds["A_floor"]
    A_max = state_bounds["A_ceil"]
    tp_floor = state_bounds["tp_floor"]
    margin_floor = state_bounds["margin_floor"]
    mach_floor = state_bounds["mach_floor"]
    mach_ceil = state_bounds["mach_ceil"]

    violations.append(max(0.0, float(np.nanmax(state_np_min - n_p))))
    violations.append(max(0.0, float(np.nanmax(n_p - state_np_max))))
    violations.append(max(0.0, float(np.nanmax(te_min - T_e))))
    violations.append(max(0.0, float(np.nanmax(T_e - te_max))))
    violations.append(max(0.0, float(np.nanmax(A_min - A))))
    violations.append(max(0.0, float(np.nanmax(A - A_max))))
    violations.append(max(0.0, tp_floor - tp_min))
    violations.append(max(0.0, margin_floor - margin_min))
    if mach_floor is not None:
        violations.append(max(0.0, mach_floor - mach_min_val))
    if mach_ceil is not None:
        violations.append(max(0.0, mach_max_val - mach_ceil))

    defect_inf = float(np.nanmax(np.abs(defects))) if defects.size else 0.0
    defect_rms = float(np.sqrt(np.nanmean(defects * defects))) if defects.size else 0.0
    mid_defect_inf = float(np.nanmax(np.abs(mid_defects))) if mid_defects.size else 0.0
    mid_defect_rms = float(np.sqrt(np.nanmean(mid_defects * mid_defects))) if mid_defects.size else 0.0
    boundary_inf = float(np.nanmax(np.abs(initial_residual_scaled)))
    max_constraint_violation = float(max(violations)) if violations else 0.0
    regularity = _compute_regularity_diagnostics(
        x_nodes=x_nodes,
        A=A,
        sigma_logA=sigma_logA,
        sigma_bounds=sigma_bounds,
    )
    slack_nodes = np.maximum(
        (
            np.zeros_like(np.asarray(velikhov_margin, dtype=float))
            if margin_slack_nodes is None
            else np.asarray(margin_slack_nodes, dtype=float).reshape(-1)
        ),
        0.0,
    )
    slack_mid = np.maximum(
        (
            np.zeros(max(x_nodes.size - 1, 0), dtype=float)
            if margin_slack_mid is None
            else np.asarray(margin_slack_mid, dtype=float).reshape(-1)
        ),
        0.0,
    )
    slack_all = np.concatenate([slack_nodes, slack_mid]) if (slack_nodes.size or slack_mid.size) else np.zeros(0, dtype=float)

    acceptable = bool(
        finite_profile
        and defect_inf <= float(thresholds.defect_inf_tol)
        and defect_rms <= float(thresholds.defect_rms_tol)
        and mid_defect_inf <= float(thresholds.defect_inf_tol)
        and mid_defect_rms <= float(thresholds.defect_rms_tol)
        and boundary_inf <= float(thresholds.boundary_inf_tol)
        and max_constraint_violation <= path_slack_tol
    )

    return {
        "finite_profile": finite_profile,
        "dynamic_defect_inf": defect_inf,
        "dynamic_defect_rms": defect_rms,
        "midpoint_defect_inf": mid_defect_inf,
        "midpoint_defect_rms": mid_defect_rms,
        "boundary_residual_inf": boundary_inf,
        "initial_state_residual": initial_residual.tolist(),
        "tp_min": tp_min,
        "velikhov_margin_min": margin_min,
        "mach_min": mach_min_val,
        "mach_max": mach_max_val,
        "max_constraint_violation": max_constraint_violation,
        "acceptable": acceptable,
        "margin_slack_max": float(np.nanmax(slack_all)) if slack_all.size else 0.0,
        "margin_slack_l1": float(np.nansum(slack_all)) if slack_all.size else 0.0,
        "margin_slack_active_fraction": float(np.mean(slack_all > 1e-12)) if slack_all.size else 0.0,
        **regularity,
        "thresholds": {
            "defect_inf_tol": float(thresholds.defect_inf_tol),
            "defect_rms_tol": float(thresholds.defect_rms_tol),
            "boundary_inf_tol": float(thresholds.boundary_inf_tol),
            "path_slack_tol": float(thresholds.path_slack_tol),
        },
    }


def optimize_area_profile(
    *,
    n_p_in: float,
    Z_in: float,
    T_p_in: float,
    T_e_in: float,
    A_in: float,
    B: float = B_FIELD,
    length: float = 5.4,
    n_intervals: int = 80,
    transcription: str = "hermite-simpson",
    min_margin: float = 0.0,
    A_min_ratio: float = 0.25,
    A_max_ratio: float = 4.0,
    max_abs_dlogA_dx: float = 2.0,
    np_min_ratio: float = 1e-6,
    np_max_ratio: float = 100.0,
    te_min: float = 100.0,
    te_max_ratio: float = 20.0,
    tp_min: float = 1.0,
    mach_min: float | None = None,
    mach_max: float | None = None,
    margin_slack_max: float = 0.0,
    margin_slack_weight: float = 0.0,
    smooth_weight: float = 0.0,
    control_slew_weight: float = 0.0,
    control_curvature_weight: float = 0.0,
    state_curvature_weight: float = 0.0,
    warm_profile_track_weight: float = 0.0,
    warm_control_track_weight: float = 0.0,
    seed_fraction: float | None = None,
    warm_start: str = "marginal",
    warm_start_dx: float | None = None,
    warm_profile: WarmStartProfile | None = None,
    feasibility_thresholds: FeasibilityThresholds | None = None,
    ipopt_max_iter: int = 1000,
    ipopt_tol: float = 1e-7,
    objective_weight: float = 1.0,
) -> OptimizedAreaProfile:
    if length <= 0.0:
        raise ValueError("length must be positive.")
    if n_intervals < 2:
        raise ValueError("n_intervals must be at least 2.")
    if transcription not in ("trapezoid", "hermite-simpson"):
        raise ValueError("unsupported transcription.")
    if A_min_ratio <= 0.0 or A_max_ratio <= 0.0 or A_max_ratio < A_min_ratio:
        raise ValueError("invalid area bounds.")
    if np_min_ratio <= 0.0:
        raise ValueError("np_min_ratio must be positive.")
    if np_max_ratio < 1.0:
        raise ValueError("np_max_ratio must be at least 1.")
    if te_min <= 0.0 or tp_min <= 0.0:
        raise ValueError("temperature lower bounds must be positive.")
    if margin_slack_max < 0.0 or margin_slack_weight < 0.0:
        raise ValueError("margin slack parameters must be nonnegative.")
    if margin_slack_max > 0.0 and margin_slack_weight <= 0.0:
        raise ValueError("margin_slack_weight must be positive when margin_slack_max > 0.")
    if te_max_ratio <= 1.0:
        raise ValueError("te_max_ratio must be greater than 1.")
    if feasibility_thresholds is None:
        feasibility_thresholds = FeasibilityThresholds()

    inlet = _prepare_inlet_constants(
        n_p_in=float(n_p_in),
        Z_in=float(Z_in),
        T_p_in=float(T_p_in),
        T_e_in=float(T_e_in),
        A_in=float(A_in),
        B=float(B),
        seed_fraction=seed_fraction,
    )

    x_nodes = np.linspace(0.0, float(length), int(n_intervals) + 1, dtype=float)
    dx = float(length) / int(n_intervals)
    stage = _make_stage_function(
        dot_N=inlet.dot_N,
        I_0=inlet.I_0,
        seed_fraction=inlet.seed_fraction,
        B=float(B),
    )

    warm: WarmStartProfile
    if warm_profile is not None:
        warm = _resample_warm_profile(warm_profile, x_nodes)
    else:
        if warm_start == "marginal":
            maybe_warm = _build_marginal_warm_start(
                x=x_nodes,
                n_p_in=float(n_p_in),
                Z_in=float(Z_in),
                T_p_in=float(T_p_in),
                T_e_in=float(T_e_in),
                A_in=float(A_in),
                B=float(B),
                length=float(length),
                warm_start_dx=warm_start_dx,
            )
            if maybe_warm is None:
                warm = _build_constant_warm_start(
                    x=x_nodes,
                    n_p_in=float(n_p_in),
                    T_e_in=float(T_e_in),
                    A_in=float(A_in),
                )
            else:
                warm = maybe_warm
        else:
            warm = _build_constant_warm_start(
                x=x_nodes,
                n_p_in=float(n_p_in),
                T_e_in=float(T_e_in),
                A_in=float(A_in),
            )

    opti = ca.Opti()
    X = opti.variable(3, n_intervals + 1)
    U = opti.variable(1, n_intervals)
    use_margin_slack = bool(margin_slack_max > 0.0)
    S_node = opti.variable(1, n_intervals + 1) if use_margin_slack else None
    S_mid = opti.variable(1, n_intervals) if use_margin_slack else None

    np_scale = float(n_p_in)
    te_scale = float(max(T_e_in, te_min))
    A_scale = float(A_in)

    n_p_hat = X[0, :]
    T_e_hat = X[1, :]
    A_hat = X[2, :]

    np_floor = max(float(np_min_ratio) * float(n_p_in), 1e10)
    np_ceil = float(np_max_ratio) * float(n_p_in)
    A_floor = float(A_min_ratio) * float(A_in)
    A_ceil = float(A_max_ratio) * float(A_in)
    te_ceil = float(te_max_ratio) * float(T_e_in)
    warm = _project_warm_profile_to_bounds(
        warm,
        n_p_floor=np_floor,
        n_p_ceil=np_ceil,
        T_e_floor=float(te_min),
        T_e_ceil=te_ceil,
        A_floor=A_floor,
        A_ceil=A_ceil,
        sigma_floor=-float(max_abs_dlogA_dx),
        sigma_ceil=float(max_abs_dlogA_dx),
        inlet_target=(float(n_p_in), float(T_e_in), float(A_in)),
    )

    opti.subject_to(X[:, 0] == ca.DM([1.0, float(T_e_in) / te_scale, 1.0]))
    opti.subject_to(n_p_hat >= np_floor / np_scale)
    opti.subject_to(n_p_hat <= np_ceil / np_scale)
    opti.subject_to(T_e_hat >= float(te_min) / te_scale)
    opti.subject_to(T_e_hat <= te_ceil / te_scale)
    opti.subject_to(A_hat >= A_floor / A_scale)
    opti.subject_to(A_hat <= A_ceil / A_scale)
    opti.subject_to(opti.bounded(-float(max_abs_dlogA_dx), U, float(max_abs_dlogA_dx)))
    if use_margin_slack:
        opti.subject_to(opti.bounded(0.0, S_node, float(margin_slack_max)))
        opti.subject_to(opti.bounded(0.0, S_mid, float(margin_slack_max)))

    objective = -float(objective_weight) * (te_scale * T_e_hat[-1] - float(T_e_in))
    if use_margin_slack:
        objective += float(margin_slack_weight) * dx * (ca.sum2(S_node) + ca.sum2(S_mid))
    if smooth_weight > 0.0:
        objective += float(smooth_weight) * dx * ca.sumsqr(U)
        if n_intervals > 1:
            dU = U[:, 1:] - U[:, :-1]
            objective += float(smooth_weight) * dx * ca.sumsqr(dU)
    if control_slew_weight > 0.0 and n_intervals > 1:
        dU_dx = (U[:, 1:] - U[:, :-1]) / dx
        objective += float(control_slew_weight) * dx * ca.sumsqr(dU_dx)
    if control_curvature_weight > 0.0 and n_intervals > 2:
        d2U_dx2 = (U[:, 2:] - 2.0 * U[:, 1:-1] + U[:, :-2]) / (dx * dx)
        objective += float(control_curvature_weight) * dx * ca.sumsqr(d2U_dx2)
    if state_curvature_weight > 0.0 and n_intervals > 2:
        d2np_dx2 = (n_p_hat[2:] - 2.0 * n_p_hat[1:-1] + n_p_hat[:-2]) / (dx * dx)
        d2te_dx2 = (T_e_hat[2:] - 2.0 * T_e_hat[1:-1] + T_e_hat[:-2]) / (dx * dx)
        d2A_dx2 = (A_hat[2:] - 2.0 * A_hat[1:-1] + A_hat[:-2]) / (dx * dx)
        objective += float(state_curvature_weight) * dx * (
            ca.sumsqr(d2np_dx2) + ca.sumsqr(d2te_dx2) + ca.sumsqr(d2A_dx2)
        )
    if warm_profile_track_weight > 0.0:
        warm_np_hat = ca.DM((warm.n_p / np_scale).reshape(1, -1))
        warm_te_hat = ca.DM((warm.T_e / te_scale).reshape(1, -1))
        warm_A_hat = ca.DM((warm.A / A_scale).reshape(1, -1))
        objective += float(warm_profile_track_weight) * dx * (
            ca.sumsqr(n_p_hat - warm_np_hat)
            + ca.sumsqr(T_e_hat - warm_te_hat)
            + ca.sumsqr(A_hat - warm_A_hat)
        )
    if warm_control_track_weight > 0.0:
        warm_u = ca.DM(warm.sigma_logA.reshape(1, -1))
        objective += float(warm_control_track_weight) * dx * ca.sumsqr(U - warm_u)

    for k in range(n_intervals):
        xk_phys = ca.vertcat(np_scale * n_p_hat[k], te_scale * T_e_hat[k], A_scale * A_hat[k])
        xkp1_phys = ca.vertcat(
            np_scale * n_p_hat[k + 1],
            te_scale * T_e_hat[k + 1],
            A_scale * A_hat[k + 1],
        )
        out_k = stage(xk_phys, U[0, k])
        out_kp1 = stage(xkp1_phys, U[0, k])
        f_k = ca.vertcat(out_k[0] / np_scale, out_k[1] / te_scale, out_k[2] / A_scale)
        f_kp1 = ca.vertcat(
            out_kp1[0] / np_scale,
            out_kp1[1] / te_scale,
            out_kp1[2] / A_scale,
        )
        if transcription == "trapezoid":
            opti.subject_to(X[:, k + 1] == X[:, k] + 0.5 * dx * (f_k + f_kp1))
            mid_state = 0.5 * (xk_phys + xkp1_phys)
        else:
            mid_state = 0.5 * (xk_phys + xkp1_phys) + 0.125 * dx * ca.vertcat(
                out_k[0] - out_kp1[0],
                out_k[1] - out_kp1[1],
                out_k[2] - out_kp1[2],
            )
            out_mid_hs = stage(mid_state, U[0, k])
            f_mid = ca.vertcat(
                out_mid_hs[0] / np_scale,
                out_mid_hs[1] / te_scale,
                out_mid_hs[2] / A_scale,
            )
            opti.subject_to(X[:, k + 1] == X[:, k] + dx / 6.0 * (f_k + 4.0 * f_mid + f_kp1))

        opti.subject_to(out_k[3] >= float(tp_min))
        if use_margin_slack:
            opti.subject_to(out_k[13] + S_node[0, k] >= float(min_margin))
        else:
            opti.subject_to(out_k[13] >= float(min_margin))
        if mach_min is not None:
            opti.subject_to(out_k[12] >= float(mach_min))
        if mach_max is not None:
            opti.subject_to(out_k[12] <= float(mach_max))

        out_mid = stage(mid_state, U[0, k])
        opti.subject_to(out_mid[3] >= float(tp_min))
        if use_margin_slack:
            opti.subject_to(out_mid[13] + S_mid[0, k] >= float(min_margin))
        else:
            opti.subject_to(out_mid[13] >= float(min_margin))
        if mach_min is not None:
            opti.subject_to(out_mid[12] >= float(mach_min))
        if mach_max is not None:
            opti.subject_to(out_mid[12] <= float(mach_max))

    out_end = stage(
        ca.vertcat(np_scale * n_p_hat[-1], te_scale * T_e_hat[-1], A_scale * A_hat[-1]),
        U[0, -1],
    )
    opti.subject_to(out_end[3] >= float(tp_min))
    if use_margin_slack:
        opti.subject_to(out_end[13] + S_node[0, -1] >= float(min_margin))
    else:
        opti.subject_to(out_end[13] >= float(min_margin))
    if mach_min is not None:
        opti.subject_to(out_end[12] >= float(mach_min))
    if mach_max is not None:
        opti.subject_to(out_end[12] <= float(mach_max))

    opti.minimize(objective)

    opti.set_initial(X[0, :], warm.n_p / np_scale)
    opti.set_initial(X[1, :], warm.T_e / te_scale)
    opti.set_initial(X[2, :], warm.A / A_scale)
    opti.set_initial(U, warm.sigma_logA.reshape(1, -1))
    if use_margin_slack:
        opti.set_initial(S_node, 0.0)
        opti.set_initial(S_mid, 0.0)

    opti.solver(
        "ipopt",
        {
            "expand": True,
            "print_time": 0,
            "ipopt.print_level": 0,
            "ipopt.max_iter": int(ipopt_max_iter),
            "ipopt.tol": float(ipopt_tol),
            "ipopt.acceptable_tol": max(float(ipopt_tol) * 10.0, 1e-6),
            "ipopt.sb": "yes",
        },
        {},
    )

    sol = None
    value_fn = None
    try:
        sol = opti.solve_limited()
        value_fn = sol.value
    except RuntimeError as exc:
        # For IPOPT failures (including Invalid_Number_Detected), try to recover
        # the latest iterate through opti.debug.value so continuation can proceed
        # with diagnostics instead of crashing the whole schedule.
        stats = opti.stats()
        try:
            value_fn = opti.debug.value
        except Exception as debug_exc:
            raise RuntimeError(f"IPOPT failed: {exc}") from debug_exc

    stats = opti.stats()
    if value_fn is None:
        value_fn = opti.debug.value
    X_sol = np.asarray(value_fn(X), dtype=float)
    U_sol = np.asarray(value_fn(U), dtype=float).reshape(-1)
    S_node_sol = np.asarray(value_fn(S_node), dtype=float).reshape(-1) if use_margin_slack else np.zeros(n_intervals + 1, dtype=float)
    S_mid_sol = np.asarray(value_fn(S_mid), dtype=float).reshape(-1) if use_margin_slack else np.zeros(n_intervals, dtype=float)
    n_p_sol = np_scale * X_sol[0, :]
    T_e_sol = te_scale * X_sol[1, :]
    A_sol = A_scale * X_sol[2, :]

    profile = _evaluate_profile_numeric(
        x=x_nodes,
        n_p=n_p_sol,
        T_e=T_e_sol,
        A=A_sol,
        inlet=inlet,
        B=float(B),
        sigma_logA=U_sol,
    )
    diagnostics = _compute_feasibility_diagnostics(
        transcription=transcription,
        x_nodes=x_nodes,
        n_p=profile["n_p"],
        T_e=profile["T_e"],
        A=profile["A"],
        sigma_logA=profile["sigma_logA"],
        T_p=profile["T_p"],
        mach=profile["mach"],
        velikhov_margin=profile["velikhov_margin"],
        stage_fun=stage,
        inlet_target=(float(n_p_in), float(T_e_in), float(A_in)),
        state_bounds={
            "np_floor": np_floor,
            "np_ceil": np_ceil,
            "te_floor": float(te_min),
            "te_ceil": te_ceil,
            "A_floor": A_floor,
            "A_ceil": A_ceil,
            "tp_floor": float(tp_min),
            "margin_floor": float(min_margin),
            "mach_floor": None if mach_min is None else float(mach_min),
            "mach_ceil": None if mach_max is None else float(mach_max),
        },
        sigma_bounds=(-float(max_abs_dlogA_dx), float(max_abs_dlogA_dx)),
        thresholds=feasibility_thresholds,
        margin_slack_nodes=S_node_sol,
        margin_slack_mid=S_mid_sol,
    )
    objective_value = float(value_fn(objective))

    return OptimizedAreaProfile(
        success=bool(stats.get("success", False)),
        acceptable=bool(diagnostics["acceptable"]),
        return_status=str(stats.get("return_status", "")),
        objective_delta_Te=float(T_e_sol[-1] - T_e_sol[0]),
        objective_value=objective_value,
        inlet=inlet,
        transcription=transcription,
        x=profile["x"],
        n_p=profile["n_p"],
        T_e=profile["T_e"],
        T_p=profile["T_p"],
        A=profile["A"],
        v_p=profile["v_p"],
        n_e=profile["n_e"],
        beta=profile["beta"],
        eta=profile["eta"],
        Z=profile["Z"],
        J_x=profile["J_x"],
        J_y=profile["J_y"],
        E_x=profile["E_x"],
        mach=profile["mach"],
        velikhov_margin=profile["velikhov_margin"],
        sigma_logA=profile["sigma_logA"],
        warm_start_source=warm.source,
        stats=_jsonify_stats(stats),
        diagnostics=diagnostics,
    )


def _payload_from_result(result: OptimizedAreaProfile, B: float) -> dict[str, object]:
    metrics = compute_global_metrics(
        x=result.x,
        A=result.A,
        v_p=result.v_p,
        eta=result.eta,
        J_x=result.J_x,
        J_y=result.J_y,
        E_x=result.E_x,
        B=float(B),
        velikhov_margin=result.velikhov_margin,
    )
    value_terms = compute_design_value_terms(
        x=result.x,
        T_e=result.T_e,
        T_p=result.T_p,
        n_p=result.n_p,
        n_e=result.n_e,
        mach=result.mach,
        A=result.A,
        J_x=result.J_x,
        E_x=result.E_x,
        B=float(B),
        seed_fraction=result.inlet.seed_fraction,
    )
    value_profiles = {
        "delta_te_only": evaluate_design_value(
            value_terms,
            weights=design_value_weights_delta_te_only(),
            profile_name="delta_te_only",
        ).to_dict(),
        "lab_poc": evaluate_design_value(
            value_terms,
            weights=design_value_weights_lab_poc(),
            profile_name="lab_poc",
        ).to_dict(),
    }
    return {
        "ok": bool(result.acceptable),
        "solver_success": bool(result.success),
        "acceptable": bool(result.acceptable),
        "solver": f"casadi-ipopt-{result.transcription}",
        "return_status": result.return_status,
        "objective": "maximize_Te_out_minus_Te_in",
        "objective_delta_Te_K": float(result.objective_delta_Te),
        "nlp_objective_value": float(result.objective_value),
        "seed_fraction": float(result.inlet.seed_fraction),
        "seed_mode": result.inlet.seed_mode,
        "dot_N": float(result.inlet.dot_N),
        "I_0": float(result.inlet.I_0),
        "v_in": float(result.inlet.v_in),
        "warm_start_source": result.warm_start_source,
        "min_velikhov_margin": float(np.nanmin(result.velikhov_margin)),
        "max_velikhov_margin": float(np.nanmax(result.velikhov_margin)),
        "min_mach": float(np.nanmin(result.mach)),
        "max_mach": float(np.nanmax(result.mach)),
        "metrics": metrics.to_dict(),
        "value_terms": value_terms.to_dict(),
        "value_profiles": value_profiles,
        "solver_stats": result.stats,
        "diagnostics": result.diagnostics,
    }


def main() -> int:
    args = _build_parser().parse_args()

    result = optimize_area_profile(
        n_p_in=float(args.np_in),
        Z_in=float(args.z_in),
        T_p_in=float(args.tp_in),
        T_e_in=float(args.te_in),
        A_in=float(args.A_in),
        B=float(args.B),
        length=float(args.L),
        n_intervals=int(args.n_intervals),
        transcription=str(args.transcription),
        min_margin=float(args.min_margin),
        A_min_ratio=float(args.A_min_ratio),
        A_max_ratio=float(args.A_max_ratio),
        max_abs_dlogA_dx=float(args.max_abs_dlogA_dx),
        np_min_ratio=float(args.np_min_ratio),
        np_max_ratio=float(args.np_max_ratio),
        te_min=float(args.te_min),
        te_max_ratio=float(args.te_max_ratio),
        tp_min=float(args.tp_min),
        mach_min=None if args.mach_min is None else float(args.mach_min),
        mach_max=None if args.mach_max is None else float(args.mach_max),
        margin_slack_max=float(args.margin_slack_max),
        margin_slack_weight=float(args.margin_slack_weight),
        smooth_weight=float(args.smooth_weight),
        control_slew_weight=float(args.control_slew_weight),
        control_curvature_weight=float(args.control_curvature_weight),
        state_curvature_weight=float(args.state_curvature_weight),
        warm_profile_track_weight=float(args.warm_profile_track_weight),
        warm_control_track_weight=float(args.warm_control_track_weight),
        seed_fraction=None if args.seed_fraction is None else float(args.seed_fraction),
        warm_start=str(args.warm_start),
        warm_start_dx=None if args.warm_start_dx is None else float(args.warm_start_dx),
        ipopt_max_iter=int(args.ipopt_max_iter),
        ipopt_tol=float(args.ipopt_tol),
        objective_weight=float(args.objective_weight),
    )

    payload = _payload_from_result(result, float(args.B))
    print(json.dumps(payload, ensure_ascii=False, indent=2))

    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.out_npz:
        out_path = Path(args.out_npz)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            out_path,
            x=result.x,
            n_p=result.n_p,
            T_e=result.T_e,
            T_p=result.T_p,
            A=result.A,
            v_p=result.v_p,
            n_e=result.n_e,
            beta=result.beta,
            eta=result.eta,
            Z=result.Z,
            J_x=result.J_x,
            J_y=result.J_y,
            E_x=result.E_x,
            mach=result.mach,
            velikhov_margin=result.velikhov_margin,
            sigma_logA=result.sigma_logA,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
