#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
import sys
from types import SimpleNamespace

import casadi as ca
import numpy as np

REPO_DIR = Path(__file__).resolve().parents[1]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from v6_core.local_algebraic_closure import B_FIELD, E_CHARGE, E_I, H_P, K_B, M_E, M_P, SIGMA_EP
from v6_core.local_algebraic_closure import beta_from_np_te, ne_from_np_te
from v6_casadi.optimize_area_profile_casadi_v6 import (
    _payload_from_result,
    _prepare_inlet_constants,
    optimize_area_profile,
)
from v6_casadi.run_casadi_continuation_v6 import run_continuation
from v6_casadi.runners.run_relaxed_continuation_v6 import _relaxed_stage_schedule

_EPS = 1e-30
_TP_MIN = 1.0
_FION_MIN = 1e-12
_FION_MAX = 1.0 - 1e-12
_DELTA_MIN = 1e-12
_GAMMA = 5.0 / 3.0


@dataclass(frozen=True)
class SegmentProfile:
    x: np.ndarray
    n_p: np.ndarray
    T_e: np.ndarray
    T_p: np.ndarray
    A: np.ndarray
    v_p: np.ndarray
    n_e: np.ndarray
    beta: np.ndarray
    eta: np.ndarray
    J_x: np.ndarray
    J_y: np.ndarray
    E_x: np.ndarray
    mach: np.ndarray
    velikhov_margin: np.ndarray
    segment_name: str


@dataclass(frozen=True)
class PiecewisePrototypeResult:
    passive: SegmentProfile
    activation: SegmentProfile
    active: object
    activation_payload: dict[str, object]
    active_payload: dict[str, object]
    x: np.ndarray
    n_p: np.ndarray
    T_e: np.ndarray
    T_p: np.ndarray
    A: np.ndarray
    v_p: np.ndarray
    n_e: np.ndarray
    beta: np.ndarray
    eta: np.ndarray
    J_x: np.ndarray
    J_y: np.ndarray
    E_x: np.ndarray
    mach: np.ndarray
    velikhov_margin: np.ndarray
    current_turn_on_x: float
    activation_start_x: float
    active_start_x: float
    nozzle_stagnation_temperature: float
    nozzle_stagnation_pressure: float
    passive_inlet_mach: float
    passive_exit_mach: float
    total_volume: float


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Prototype piecewise-current device builder: passive de Laval nozzle proxy + "
            "short activation ramp + active post-Jeffrey CasADi/IPOPT profile optimizer."
        )
    )
    p.add_argument("--np-in", type=float, default=3.05e25)
    p.add_argument("--z-in", type=float, default=75.954994)
    p.add_argument("--tp-in", type=float, default=429.0)
    p.add_argument("--te-in", type=float, default=4420.0)
    p.add_argument("--A-in", type=float, default=0.447)
    p.add_argument("--B", type=float, default=B_FIELD)
    p.add_argument("--L-total", type=float, default=6.6)
    p.add_argument(
        "--x0",
        type=float,
        default=1.0,
        help="location where the external circuit starts to turn on [m]",
    )
    p.add_argument(
        "--activation-width",
        type=float,
        default=0.15,
        help="short transition region that ramps JxA from passive to active [m]",
    )
    p.add_argument(
        "--activation-current-sharpness",
        type=float,
        default=3.0,
        help="dimensionless tanh sharpness controlling how abruptly JxA turns on in Stage B",
    )
    p.add_argument(
        "--activation-current-floor-frac",
        type=float,
        default=1e-3,
        help="small positive current floor used inside Stage B closure to avoid singular I=0 algebra",
    )
    p.add_argument(
        "--activation-closure-blend-current-frac",
        type=float,
        default=0.05,
        help="fraction of I_on over which Stage B blends from passive closure into active closure",
    )
    p.add_argument(
        "--passive-mach-in",
        type=float,
        default=0.15,
        help="subsonic Mach number used at the upstream end of the passive nozzle proxy",
    )
    p.add_argument(
        "--passive-mach-out",
        type=float,
        default=1.8,
        help="Mach number reached at the end of the passive nozzle proxy, before activation",
    )
    p.add_argument(
        "--nozzle-stagnation-temp",
        type=float,
        default=1000.0,
        help="stagnation temperature at the upstream nozzle reservoir [K]",
    )
    p.add_argument(
        "--nozzle-stagnation-pressure",
        type=float,
        default=5.0e6,
        help="stagnation pressure at the upstream nozzle reservoir [Pa]",
    )
    p.add_argument(
        "--passive-te-ratio",
        type=float,
        default=1.0,
        help="proxy electron temperature in the passive nozzle, as Te/Tp",
    )
    p.add_argument("--seed-fraction", type=float, default=None)
    p.add_argument(
        "--active-solver-mode",
        type=str,
        choices=("single", "continuation"),
        default="continuation",
        help="single IPOPT solve or the compact relaxed continuation workflow for the active segment",
    )
    p.add_argument("--passive-points", type=int, default=121)
    p.add_argument("--activation-points", type=int, default=25)
    p.add_argument("--n-intervals", type=int, default=80)
    p.add_argument(
        "--transcription",
        type=str,
        choices=("trapezoid", "hermite-simpson"),
        default="hermite-simpson",
    )
    p.add_argument("--warm-start", type=str, choices=("marginal", "constant"), default="marginal")
    p.add_argument("--warm-start-dx", type=float, default=0.01)
    p.add_argument("--min-margin", type=float, default=0.0)
    p.add_argument("--A-min-ratio", type=float, default=0.40)
    p.add_argument("--A-max-ratio", type=float, default=5.0)
    p.add_argument("--max-abs-dlogA-dx", type=float, default=0.50)
    p.add_argument("--np-min-ratio", type=float, default=1e-8)
    p.add_argument("--np-max-ratio", type=float, default=150.0)
    p.add_argument("--te-min", type=float, default=100.0)
    p.add_argument("--te-max-ratio", type=float, default=24.0)
    p.add_argument("--tp-min", type=float, default=1.0)
    p.add_argument("--mach-min", type=float, default=1.0)
    p.add_argument("--mach-max", type=float, default=None)
    p.add_argument("--margin-slack-max", type=float, default=0.0)
    p.add_argument("--margin-slack-weight", type=float, default=0.0)
    p.add_argument("--smooth-weight", type=float, default=0.004)
    p.add_argument("--control-slew-weight", type=float, default=0.06)
    p.add_argument("--control-curvature-weight", type=float, default=0.015)
    p.add_argument("--state-curvature-weight", type=float, default=0.003)
    p.add_argument("--warm-profile-track-weight", type=float, default=1.0)
    p.add_argument("--warm-control-track-weight", type=float, default=0.25)
    p.add_argument("--objective-weight", type=float, default=0.08)
    p.add_argument("--ipopt-max-iter", type=int, default=2600)
    p.add_argument("--ipopt-tol", type=float, default=1e-7)
    p.add_argument(
        "--out-dir",
        type=str,
        default=str(Path(__file__).resolve().parent / "outputs" / "prototype_case"),
    )
    return p


def _safe_pos(x: float, floor: float = _EPS) -> float:
    return x if x > floor else floor


def _smoothstep(s: np.ndarray) -> np.ndarray:
    s_clip = np.clip(np.asarray(s, dtype=float), 0.0, 1.0)
    return s_clip * s_clip * (3.0 - 2.0 * s_clip)


def _smooth_tanh_ramp(s: np.ndarray, *, sharpness: float) -> np.ndarray:
    s_arr = np.clip(np.asarray(s, dtype=float), 0.0, 1.0)
    k = max(float(sharpness), 1e-6)
    num = np.tanh(k * (2.0 * s_arr - 1.0)) + np.tanh(k)
    den = 2.0 * np.tanh(k)
    return num / max(den, _EPS)


def _eta_from_np_ne_te(n_p: np.ndarray, n_e: np.ndarray, T_e: np.ndarray) -> np.ndarray:
    v_te = np.sqrt(2.0 * K_B * np.maximum(T_e, 1.0) / M_E)
    return M_E * np.maximum(n_p, 1.0) * SIGMA_EP * v_te / (E_CHARGE * E_CHARGE * np.maximum(n_e, _EPS))


def _velikhov_margin_array(
    *,
    n_p: np.ndarray,
    T_e: np.ndarray,
    T_p: np.ndarray,
    seed_fraction: float,
    B: float,
) -> np.ndarray:
    out = np.full_like(n_p, np.nan, dtype=float)
    if seed_fraction <= 0.0:
        return out
    n_s = seed_fraction * np.maximum(n_p, 1.0)
    n_e = np.array([ne_from_np_te(float(np_i), float(te_i), float(seed_fraction)) for np_i, te_i in zip(n_p, T_e)])
    beta = np.array([beta_from_np_te(float(np_i), float(te_i), B=float(B), sigma_ep=SIGMA_EP) for np_i, te_i in zip(n_p, T_e)])
    f_I = np.clip(n_e / np.maximum(n_s, _EPS), _FION_MIN, _FION_MAX)
    delta = np.maximum(T_e / np.maximum(T_p, _TP_MIN) - 1.0, _DELTA_MIN)
    alpha = (K_B * T_e / (2.0 * E_I)) * (2.0 - f_I) / np.maximum(1.0 - f_I, _EPS)
    out[:] = 4.0 * alpha * (2.0 + 1.0 / delta) * (1.0 + alpha * (1.0 + 1.0 / delta)) - beta * beta
    return out


def _isentropic_area_ratio(mach: np.ndarray, *, gamma: float = _GAMMA) -> np.ndarray:
    mach_safe = np.maximum(np.asarray(mach, dtype=float), 1e-6)
    factor = (2.0 / (gamma + 1.0)) * (1.0 + 0.5 * (gamma - 1.0) * mach_safe * mach_safe)
    exponent = (gamma + 1.0) / (2.0 * (gamma - 1.0))
    return np.power(factor, exponent) / mach_safe


def _isentropic_state_from_mach(
    *,
    stagnation_temperature: float,
    stagnation_pressure: float,
    mach: float,
) -> tuple[float, float, float, float]:
    fac = 1.0 + 0.5 * (_GAMMA - 1.0) * float(mach) * float(mach)
    T_p = float(stagnation_temperature) / fac
    p = float(stagnation_pressure) / fac ** (_GAMMA / (_GAMMA - 1.0))
    n_p = p / max(K_B * T_p, _EPS)
    v_p = float(mach) * math.sqrt(_GAMMA * K_B * T_p / M_P)
    return T_p, p, n_p, v_p


def _build_passive_nozzle_profile(
    *,
    A_exit: float,
    seed_fraction: float,
    B: float,
    length: float,
    n_points: int,
    stagnation_temperature: float,
    stagnation_pressure: float,
    mach_in: float,
    mach_out: float,
    passive_te_ratio: float,
) -> SegmentProfile:
    if length <= 0.0:
        raise ValueError("passive length must be positive for the prototype builder.")
    if n_points < 2:
        raise ValueError("passive-points must be at least 2.")
    if not (0.0 < mach_in < 1.0):
        raise ValueError("passive-mach-in must satisfy 0 < M < 1.")
    if mach_out <= 1.0:
        raise ValueError("passive-mach-out must be supersonic.")
    if stagnation_temperature <= 0.0 or stagnation_pressure <= 0.0:
        raise ValueError("nozzle stagnation conditions must be positive.")

    x = np.linspace(0.0, float(length), int(n_points), dtype=float)
    s = _smoothstep(x / max(float(length), _EPS))
    mach = float(mach_in) + (float(mach_out) - float(mach_in)) * s
    fac = 1.0 + 0.5 * (_GAMMA - 1.0) * mach * mach
    T_p = float(stagnation_temperature) / fac
    p = float(stagnation_pressure) / np.power(fac, _GAMMA / (_GAMMA - 1.0))
    n_p = p / np.maximum(K_B * T_p, _EPS)
    v_p = mach * np.sqrt(_GAMMA * K_B * T_p / M_P)

    area_ratio = _isentropic_area_ratio(mach)
    area_ratio_exit = float(_isentropic_area_ratio(np.array([float(mach_out)]))[0])
    A_star = float(A_exit) / max(area_ratio_exit, _EPS)
    A = A_star * area_ratio

    T_e = np.maximum(float(passive_te_ratio), 1.0) * T_p
    n_e = np.array([ne_from_np_te(float(np_i), float(te_i), float(seed_fraction)) for np_i, te_i in zip(n_p, T_e)])
    beta = np.array([beta_from_np_te(float(np_i), float(te_i), B=float(B), sigma_ep=SIGMA_EP) for np_i, te_i in zip(n_p, T_e)])
    eta = _eta_from_np_ne_te(n_p, n_e, T_e)

    passive = SegmentProfile(
        x=x,
        n_p=n_p,
        T_e=T_e,
        T_p=T_p,
        A=A,
        v_p=v_p,
        n_e=n_e,
        beta=beta,
        eta=eta,
        J_x=np.zeros_like(x),
        J_y=np.zeros_like(x),
        E_x=np.zeros_like(x),
        mach=mach,
        velikhov_margin=np.full_like(x, np.nan, dtype=float),
        segment_name="passive_nozzle",
    )
    return passive


def _make_activation_stage_function(
    *,
    dot_N: float,
    seed_fraction: float,
    B: float,
    I_floor: float,
    I_blend: float,
) -> ca.Function:
    x = ca.SX.sym("x", 3)
    sigma = ca.SX.sym("sigma")
    I_local = ca.SX.sym("I_local")

    n_p = x[0]
    T_e = x[1]
    A = x[2]

    n_p_safe = ca.fmax(n_p, 1.0)
    T_e_safe = ca.fmax(T_e, 1.0)
    A_safe = ca.fmax(A, 1e-12)
    I_eff = ca.fmax(I_local, float(I_floor))
    blend_s = ca.fmin(ca.fmax(I_local / max(float(I_blend), _EPS), 0.0), 1.0)
    g = blend_s * blend_s * (3.0 - 2.0 * blend_s)

    v_te = ca.sqrt(2.0 * K_B * T_e_safe / M_E)
    beta = E_CHARGE * B / (M_E * n_p_safe * SIGMA_EP * v_te + _EPS)

    saha_a = 2.0 * math.pi * M_E * K_B * T_e_safe / (H_P * H_P)
    saha_k = (saha_a ** 1.5) * ca.exp(-E_I / (K_B * T_e_safe))
    saha_k_safe = ca.fmax(saha_k, 1e-100)
    n_s = seed_fraction * n_p_safe
    n_e = 2.0 * n_s / (1.0 + ca.sqrt(1.0 + 4.0 * n_s / saha_k_safe))

    eta = M_E * n_p_safe * SIGMA_EP * v_te / (E_CHARGE * E_CHARGE * n_e + _EPS)
    q = E_CHARGE * n_e * dot_N / (I_eff * n_p_safe + _EPS)
    b2 = beta * beta
    Z = b2 * (q - 1.0) - 1.0

    one_plus_z = 1.0 + Z
    den = b2 + one_plus_z
    F = b2 * (b2 + one_plus_z * one_plus_z) / (den * den + _EPS)

    v_p = dot_N / (n_p_safe * A_safe)
    T_p_active = T_e_safe - M_P * v_p * v_p * F / (3.0 * K_B)
    dTp_active_dnp = ca.gradient(T_p_active, n_p)
    dTp_active_dTe = ca.gradient(T_p_active, T_e)
    dTp_active_dA = ca.gradient(T_p_active, A)

    # Regularize the low-current edge of Stage B by blending from passive
    # single-temperature behavior into the active Hall-branch closure.
    T_p = (1.0 - g) * T_e_safe + g * T_p_active
    T_p_safe = ca.fmax(T_p, _TP_MIN)
    dTp_dnp = g * dTp_active_dnp
    dTp_dTe = (1.0 - g) + g * dTp_active_dTe
    dTp_dA = g * dTp_active_dA

    jfac = E_CHARGE * n_e * v_p
    J_x = I_local / A_safe
    J_y_active = -beta * one_plus_z / (den + _EPS) * jfac
    E_x_active = -b2 * Z / (den + _EPS) * eta * jfac
    J_y = g * J_y_active
    E_x = g * E_x_active
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

    c_s = ca.sqrt((5.0 / 3.0) * K_B * T_p_safe / M_P + _EPS)
    mach = v_p / c_s
    f_I_raw = n_e / (n_s + _EPS)
    f_I = ca.fmin(ca.fmax(f_I_raw, _FION_MIN), _FION_MAX)
    delta_raw = T_e_safe / T_p_safe - 1.0
    delta = ca.fmax(delta_raw, _DELTA_MIN)
    alpha = (K_B * T_e_safe / (2.0 * E_I)) * (2.0 - f_I) / (1.0 - f_I + _EPS)
    G = 4.0 * alpha * (2.0 + 1.0 / delta) * (
        1.0 + alpha * (1.0 + 1.0 / delta)
    ) - b2

    out = ca.vertcat(dn_dx, dTe_dx, dA_dx, T_p, v_p, n_e, beta, eta, Z, J_x, J_y, E_x, mach, G)
    return ca.Function("activation_stage_fun", [x, sigma, I_local], [out])


def _build_activation_current_profile(
    *,
    width: float,
    n_intervals: int,
    I_on: float,
    sharpness: float,
) -> tuple[np.ndarray, np.ndarray]:
    x_nodes = np.linspace(0.0, float(width), int(n_intervals) + 1, dtype=float)
    x_mid = 0.5 * (x_nodes[:-1] + x_nodes[1:])
    s_nodes = x_nodes / max(float(width), _EPS)
    s_mid = x_mid / max(float(width), _EPS)
    I_nodes = float(I_on) * _smooth_tanh_ramp(s_nodes, sharpness=float(sharpness))
    I_mid = float(I_on) * _smooth_tanh_ramp(s_mid, sharpness=float(sharpness))
    return I_nodes, I_mid


def _build_activation_warm_profile(
    *,
    x_nodes: np.ndarray,
    start_state: tuple[float, float, float],
    end_state: tuple[float, float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    s = x_nodes / max(float(x_nodes[-1]), _EPS)
    n_p = float(start_state[0]) + (float(end_state[0]) - float(start_state[0])) * s
    T_e = float(start_state[1]) + (float(end_state[1]) - float(start_state[1])) * s
    A = float(start_state[2]) + (float(end_state[2]) - float(start_state[2])) * s
    sigma = np.diff(np.log(np.maximum(A, 1e-20))) / np.diff(x_nodes)
    return n_p, T_e, A, sigma


def _evaluate_activation_numeric(
    *,
    x_nodes: np.ndarray,
    n_p: np.ndarray,
    T_e: np.ndarray,
    A: np.ndarray,
    I_nodes: np.ndarray,
    stage_fun: ca.Function,
) -> SegmentProfile:
    n_points = x_nodes.size
    T_p = np.full(n_points, np.nan, dtype=float)
    v_p = np.full(n_points, np.nan, dtype=float)
    n_e = np.full(n_points, np.nan, dtype=float)
    beta = np.full(n_points, np.nan, dtype=float)
    eta = np.full(n_points, np.nan, dtype=float)
    J_x = np.full(n_points, np.nan, dtype=float)
    J_y = np.full(n_points, np.nan, dtype=float)
    E_x = np.full(n_points, np.nan, dtype=float)
    mach = np.full(n_points, np.nan, dtype=float)
    G = np.full(n_points, np.nan, dtype=float)

    for i in range(n_points):
        current = float(I_nodes[min(i, I_nodes.size - 1)])
        sigma = 0.0
        out = np.asarray(stage_fun(np.array([n_p[i], T_e[i], A[i]], dtype=float), sigma, current), dtype=float).reshape(-1)
        T_p[i] = out[3]
        v_p[i] = out[4]
        n_e[i] = out[5]
        beta[i] = out[6]
        eta[i] = out[7]
        J_x[i] = out[9]
        J_y[i] = out[10]
        E_x[i] = out[11]
        mach[i] = out[12]
        G[i] = out[13]

    return SegmentProfile(
        x=np.asarray(x_nodes, dtype=float),
        n_p=np.asarray(n_p, dtype=float),
        T_e=np.asarray(T_e, dtype=float),
        T_p=T_p,
        A=np.asarray(A, dtype=float),
        v_p=v_p,
        n_e=n_e,
        beta=beta,
        eta=eta,
        J_x=J_x,
        J_y=J_y,
        E_x=E_x,
        mach=mach,
        velikhov_margin=G,
        segment_name="activation_nlp",
    )


def _solve_activation_segment(
    *,
    x_start: float,
    width: float,
    n_points: int,
    passive_end: SegmentProfile,
    active_result,
    B: float,
    seed_fraction: float,
    tp_min: float,
    min_margin: float,
    mach_min: float | None,
    mach_max: float | None,
    max_abs_dlogA_dx: float,
    smooth_weight: float,
    control_slew_weight: float,
    control_curvature_weight: float,
    state_curvature_weight: float,
    warm_profile_track_weight: float,
    warm_control_track_weight: float,
    ipopt_max_iter: int,
    ipopt_tol: float,
    current_sharpness: float,
    current_floor_frac: float,
    closure_blend_current_frac: float,
) -> tuple[SegmentProfile, dict[str, object]]:
    if width <= 0.0 or n_points < 2:
        raise ValueError("activation NLP requires positive width and at least two points.")

    terminal_track_weight = 0.2
    terminal_te_reward_weight = 1e-4
    start_np = float(passive_end.n_p[-1])
    start_te = float(passive_end.T_e[-1])
    start_A = float(passive_end.A[-1])
    end_np = float(active_result.n_p[0])
    end_te = float(active_result.T_e[0])
    end_A = float(active_result.A[0])
    dot_N = float(active_result.n_p[0] * active_result.v_p[0] * active_result.A[0])
    I_on = float(active_result.J_x[0] * active_result.A[0])
    I_floor = max(float(current_floor_frac) * abs(I_on), 1e-12)
    I_blend = max(float(closure_blend_current_frac) * abs(I_on), I_floor)

    n_intervals = int(n_points) - 1
    x_local = np.linspace(0.0, float(width), int(n_points), dtype=float)
    dx = float(width) / float(n_intervals)
    I_nodes, I_mid = _build_activation_current_profile(
        width=float(width),
        n_intervals=n_intervals,
        I_on=float(I_on),
        sharpness=float(current_sharpness),
    )
    stage_fun = _make_activation_stage_function(
        dot_N=float(dot_N),
        seed_fraction=float(seed_fraction),
        B=float(B),
        I_floor=float(I_floor),
        I_blend=float(I_blend),
    )

    warm_np, warm_te, warm_A, warm_sigma = _build_activation_warm_profile(
        x_nodes=x_local,
        start_state=(start_np, start_te, start_A),
        end_state=(end_np, end_te, end_A),
    )

    opti = ca.Opti()
    X = opti.variable(3, n_intervals + 1)
    U = opti.variable(1, n_intervals)

    np_scale = max(abs(start_np), abs(end_np), 1.0)
    te_scale = max(abs(start_te), abs(end_te), _TP_MIN)
    A_scale = max(abs(start_A), abs(end_A), 1e-8)

    n_p_hat = X[0, :]
    T_e_hat = X[1, :]
    A_hat = X[2, :]

    opti.subject_to(X[:, 0] == ca.DM([start_np / np_scale, start_te / te_scale, start_A / A_scale]))
    opti.subject_to(n_p_hat >= 1e-8)
    opti.subject_to(T_e_hat >= 1e-8)
    opti.subject_to(A_hat >= 1e-8)
    opti.subject_to(opti.bounded(-float(max_abs_dlogA_dx), U, float(max_abs_dlogA_dx)))
    opti.subject_to(opti.bounded(0.05 * end_np / np_scale, n_p_hat[-1], 20.0 * end_np / np_scale))
    opti.subject_to(opti.bounded(0.25 * end_te / te_scale, T_e_hat[-1], 8.0 * end_te / te_scale))
    opti.subject_to(opti.bounded(0.1 * end_A / A_scale, A_hat[-1], 8.0 * end_A / A_scale))

    objective = 0.0
    if smooth_weight > 0.0:
        objective += float(smooth_weight) * dx * ca.sumsqr(U)
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
        objective += float(warm_profile_track_weight) * dx * (
            ca.sumsqr(n_p_hat - ca.DM((warm_np / np_scale).reshape(1, -1)))
            + ca.sumsqr(T_e_hat - ca.DM((warm_te / te_scale).reshape(1, -1)))
            + ca.sumsqr(A_hat - ca.DM((warm_A / A_scale).reshape(1, -1)))
        )
    if warm_control_track_weight > 0.0:
        objective += float(warm_control_track_weight) * dx * ca.sumsqr(U - ca.DM(warm_sigma.reshape(1, -1)))
    objective += terminal_track_weight * (
        ca.sumsqr(n_p_hat[-1] - float(end_np / np_scale))
        + ca.sumsqr(T_e_hat[-1] - float(end_te / te_scale))
        + ca.sumsqr(A_hat[-1] - float(end_A / A_scale))
    )
    objective += -terminal_te_reward_weight * te_scale * T_e_hat[-1]

    for k in range(n_intervals):
        xk_phys = ca.vertcat(np_scale * n_p_hat[k], te_scale * T_e_hat[k], A_scale * A_hat[k])
        xkp1_phys = ca.vertcat(np_scale * n_p_hat[k + 1], te_scale * T_e_hat[k + 1], A_scale * A_hat[k + 1])
        out_k = stage_fun(xk_phys, U[0, k], float(I_nodes[k]))
        out_kp1 = stage_fun(xkp1_phys, U[0, k], float(I_nodes[k + 1]))
        f_k = ca.vertcat(out_k[0] / np_scale, out_k[1] / te_scale, out_k[2] / A_scale)
        f_kp1 = ca.vertcat(out_kp1[0] / np_scale, out_kp1[1] / te_scale, out_kp1[2] / A_scale)

        mid_state = 0.5 * (xk_phys + xkp1_phys) + 0.125 * dx * ca.vertcat(
            out_k[0] - out_kp1[0],
            out_k[1] - out_kp1[1],
            out_k[2] - out_kp1[2],
        )
        out_mid = stage_fun(mid_state, U[0, k], float(I_mid[k]))
        f_mid = ca.vertcat(out_mid[0] / np_scale, out_mid[1] / te_scale, out_mid[2] / A_scale)
        opti.subject_to(X[:, k + 1] == X[:, k] + dx / 6.0 * (f_k + 4.0 * f_mid + f_kp1))

        opti.subject_to(out_k[3] >= float(tp_min))
        opti.subject_to(out_k[13] >= float(min_margin))
        opti.subject_to(out_mid[3] >= float(tp_min))
        opti.subject_to(out_mid[13] >= float(min_margin))
        if mach_min is not None:
            opti.subject_to(out_k[12] >= float(mach_min))
            opti.subject_to(out_mid[12] >= float(mach_min))
        if mach_max is not None:
            opti.subject_to(out_k[12] <= float(mach_max))
            opti.subject_to(out_mid[12] <= float(mach_max))

    out_end = stage_fun(
        ca.vertcat(np_scale * n_p_hat[-1], te_scale * T_e_hat[-1], A_scale * A_hat[-1]),
        U[0, -1],
        float(I_nodes[-1]),
    )
    opti.subject_to(out_end[3] >= float(tp_min))
    opti.subject_to(out_end[13] >= float(min_margin))
    if mach_min is not None:
        opti.subject_to(out_end[12] >= float(mach_min))
    if mach_max is not None:
        opti.subject_to(out_end[12] <= float(mach_max))

    opti.minimize(objective)
    opti.set_initial(X[0, :], warm_np / np_scale)
    opti.set_initial(X[1, :], warm_te / te_scale)
    opti.set_initial(X[2, :], warm_A / A_scale)
    opti.set_initial(U, warm_sigma.reshape(1, -1))
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
    except RuntimeError:
        value_fn = opti.debug.value

    stats = opti.stats()
    if value_fn is None:
        value_fn = opti.debug.value

    X_sol = np.asarray(value_fn(X), dtype=float)
    n_p_sol = np_scale * X_sol[0, :]
    T_e_sol = te_scale * X_sol[1, :]
    A_sol = A_scale * X_sol[2, :]

    activation = _evaluate_activation_numeric(
        x_nodes=x_local + float(x_start),
        n_p=n_p_sol,
        T_e=T_e_sol,
        A=A_sol,
        I_nodes=I_nodes,
        stage_fun=stage_fun,
    )
    acceptable = bool(
        np.all(np.isfinite(activation.n_p))
        and np.all(np.isfinite(activation.T_e))
        and np.all(np.isfinite(activation.T_p))
        and float(np.nanmin(activation.T_p)) >= float(tp_min)
        and float(np.nanmin(activation.velikhov_margin)) >= float(min_margin)
    )
    payload = {
        "solver_success": bool(stats.get("success", False)),
        "acceptable": acceptable,
        "return_status": str(stats.get("return_status", "")),
        "dot_N": float(dot_N),
        "I_on": float(I_on),
        "I_floor": float(I_floor),
        "I_blend": float(I_blend),
        "current_profile": {
            "kind": "tanh",
            "sharpness": float(current_sharpness),
            "floor_fraction": float(current_floor_frac),
            "closure_blend_current_frac": float(closure_blend_current_frac),
        },
        "min_tp_K": float(np.nanmin(activation.T_p)),
        "min_margin": float(np.nanmin(activation.velikhov_margin)),
        "max_margin": float(np.nanmax(activation.velikhov_margin)),
        "terminal_state": {
            "n_p": float(activation.n_p[-1]),
            "T_e": float(activation.T_e[-1]),
            "T_p": float(activation.T_p[-1]),
            "A": float(activation.A[-1]),
        },
        "target_state": {
            "n_p": float(end_np),
            "T_e": float(end_te),
            "A": float(end_A),
        },
    }
    return activation, payload


def _concat_segments(passive: SegmentProfile, activation: SegmentProfile, active_result, active_shift: float) -> dict[str, np.ndarray]:
    active_x = np.asarray(active_result.x, dtype=float) + float(active_shift)

    def join(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
        parts = [np.asarray(a, dtype=float)]
        if b.size > 1:
            parts.append(np.asarray(b[1:], dtype=float))
        if c.size > 1:
            parts.append(np.asarray(c[1:], dtype=float))
        return np.concatenate(parts)

    return {
        "x": join(passive.x, activation.x, active_x),
        "n_p": join(passive.n_p, activation.n_p, active_result.n_p),
        "T_e": join(passive.T_e, activation.T_e, active_result.T_e),
        "T_p": join(passive.T_p, activation.T_p, active_result.T_p),
        "A": join(passive.A, activation.A, active_result.A),
        "v_p": join(passive.v_p, activation.v_p, active_result.v_p),
        "n_e": join(passive.n_e, activation.n_e, active_result.n_e),
        "beta": join(passive.beta, activation.beta, active_result.beta),
        "eta": join(passive.eta, activation.eta, active_result.eta),
        "J_x": join(passive.J_x, activation.J_x, active_result.J_x),
        "J_y": join(passive.J_y, activation.J_y, active_result.J_y),
        "E_x": join(passive.E_x, activation.E_x, active_result.E_x),
        "mach": join(passive.mach, activation.mach, active_result.mach),
        "velikhov_margin": join(passive.velikhov_margin, activation.velikhov_margin, active_result.velikhov_margin),
    }


def _load_active_result_from_npz(npz_path: Path, *, payload: dict[str, object]) -> object:
    arrays = np.load(npz_path)
    return SimpleNamespace(
        success=bool(payload.get("solver_success", False)),
        acceptable=bool(payload.get("ok", False)),
        return_status=str(payload.get("final_return_status", "")),
        objective_delta_Te=float(payload.get("final_objective_delta_Te_K", math.nan)),
        x=np.asarray(arrays["x"], dtype=float),
        n_p=np.asarray(arrays["n_p"], dtype=float),
        T_e=np.asarray(arrays["T_e"], dtype=float),
        T_p=np.asarray(arrays["T_p"], dtype=float),
        A=np.asarray(arrays["A"], dtype=float),
        v_p=np.asarray(arrays["v_p"], dtype=float),
        n_e=np.asarray(arrays["n_e"], dtype=float),
        beta=np.asarray(arrays["beta"], dtype=float),
        eta=np.asarray(arrays["eta"], dtype=float),
        Z=np.asarray(arrays["Z"], dtype=float),
        J_x=np.asarray(arrays["J_x"], dtype=float),
        J_y=np.asarray(arrays["J_y"], dtype=float),
        E_x=np.asarray(arrays["E_x"], dtype=float),
        mach=np.asarray(arrays["mach"], dtype=float),
        velikhov_margin=np.asarray(arrays["velikhov_margin"], dtype=float),
        sigma_logA=np.asarray(arrays["sigma_logA"], dtype=float),
    )


def _solve_active_segment(
    *,
    solver_mode: str,
    active_out_dir: Path,
    n_p_in: float,
    Z_in: float,
    T_p_in: float,
    T_e_in: float,
    A_in: float,
    B: float,
    active_length: float,
    seed_fraction: float,
    warm_start_dx: float | None,
    n_intervals: int,
    transcription: str,
    min_margin: float,
    A_min_ratio: float,
    A_max_ratio: float,
    max_abs_dlogA_dx: float,
    np_min_ratio: float,
    np_max_ratio: float,
    te_min: float,
    te_max_ratio: float,
    tp_min: float,
    mach_min: float | None,
    mach_max: float | None,
    margin_slack_max: float,
    margin_slack_weight: float,
    smooth_weight: float,
    control_slew_weight: float,
    control_curvature_weight: float,
    state_curvature_weight: float,
    warm_profile_track_weight: float,
    warm_control_track_weight: float,
    warm_start: str,
    ipopt_max_iter: int,
    ipopt_tol: float,
    objective_weight: float,
) -> tuple[object, dict[str, object]]:
    if solver_mode == "single":
        active_result = optimize_area_profile(
            n_p_in=float(n_p_in),
            Z_in=float(Z_in),
            T_p_in=float(T_p_in),
            T_e_in=float(T_e_in),
            A_in=float(A_in),
            B=float(B),
            length=float(active_length),
            n_intervals=int(n_intervals),
            transcription=str(transcription),
            min_margin=float(min_margin),
            A_min_ratio=float(A_min_ratio),
            A_max_ratio=float(A_max_ratio),
            max_abs_dlogA_dx=float(max_abs_dlogA_dx),
            np_min_ratio=float(np_min_ratio),
            np_max_ratio=float(np_max_ratio),
            te_min=float(te_min),
            te_max_ratio=float(te_max_ratio),
            tp_min=float(tp_min),
            mach_min=None if mach_min is None else float(mach_min),
            mach_max=None if mach_max is None else float(mach_max),
            margin_slack_max=float(margin_slack_max),
            margin_slack_weight=float(margin_slack_weight),
            smooth_weight=float(smooth_weight),
            control_slew_weight=float(control_slew_weight),
            control_curvature_weight=float(control_curvature_weight),
            state_curvature_weight=float(state_curvature_weight),
            warm_profile_track_weight=float(warm_profile_track_weight),
            warm_control_track_weight=float(warm_control_track_weight),
            seed_fraction=float(seed_fraction),
            warm_start=str(warm_start),
            warm_start_dx=warm_start_dx,
            ipopt_max_iter=int(ipopt_max_iter),
            ipopt_tol=float(ipopt_tol),
            objective_weight=float(objective_weight),
        )
        return active_result, _payload_from_result(active_result, float(B))

    schedule = _relaxed_stage_schedule(
        profile="balanced",
        n_intervals=int(n_intervals),
        final_mach_min=None if mach_min is None else float(mach_min),
        final_a_min_ratio=float(A_min_ratio),
        final_a_max_ratio=float(A_max_ratio),
        final_max_abs_dlogA_dx=float(max_abs_dlogA_dx),
        final_np_min_ratio=float(np_min_ratio),
        final_np_max_ratio=float(np_max_ratio),
        final_te_min=float(te_min),
        final_te_max_ratio=float(te_max_ratio),
        final_smooth_weight=float(smooth_weight),
        final_control_slew_weight=float(control_slew_weight),
        final_control_curvature_weight=float(control_curvature_weight),
        final_state_curvature_weight=float(state_curvature_weight),
        final_warm_profile_track_weight=float(warm_profile_track_weight),
        final_warm_control_track_weight=float(warm_control_track_weight),
        final_objective_weight=float(objective_weight),
        final_ipopt_max_iter=int(ipopt_max_iter),
        include_stage1_assess=False,
        transition_margin_slack_max=float(margin_slack_max),
        transition_margin_slack_weight=float(margin_slack_weight),
    )
    payload = run_continuation(
        n_p_in=float(n_p_in),
        Z_in=float(Z_in),
        T_p_in=float(T_p_in),
        T_e_in=float(T_e_in),
        A_in=float(A_in),
        B=float(B),
        L=float(active_length),
        seed_fraction=float(seed_fraction),
        warm_start_dx=0.01 if warm_start_dx is None else float(warm_start_dx),
        stage_schedule=schedule,
        out_dir=active_out_dir,
        stop_on_unacceptable=False,
        warm_start_policy="regular",
        adaptive_bridge_count=3,
        adaptive_bridge_max_count=12,
    )
    npz_path = active_out_dir / "final_acceptable.npz"
    if not npz_path.exists():
        raise RuntimeError(f"continuation run did not produce {npz_path}")
    active_result = _load_active_result_from_npz(npz_path, payload=payload)
    active_payload = {
        "ok": bool(payload.get("ok", False)),
        "solver_success": bool(payload.get("solver_success", False)),
        "acceptable": bool(payload.get("ok", False)),
        "solver": "piecewise-active-continuation",
        "return_status": str(payload.get("final_return_status", "")),
        "objective_delta_Te_K": float(payload.get("final_objective_delta_Te_K", math.nan)),
        "seed_fraction": float(seed_fraction),
        "continuation_summary": payload,
    }
    return active_result, active_payload


def optimize_piecewise_current_profile(
    *,
    active_solver_mode: str,
    active_out_dir: Path,
    n_p_in: float,
    Z_in: float,
    T_p_in: float,
    T_e_in: float,
    A_in: float,
    B: float,
    L_total: float,
    x0: float,
    activation_width: float,
    nozzle_stagnation_temperature: float,
    nozzle_stagnation_pressure: float,
    passive_mach_in: float,
    passive_mach_out: float,
    passive_te_ratio: float,
    activation_current_sharpness: float,
    activation_current_floor_frac: float,
    activation_closure_blend_current_frac: float,
    passive_points: int,
    activation_points: int,
    n_intervals: int,
    transcription: str,
    min_margin: float,
    A_min_ratio: float,
    A_max_ratio: float,
    max_abs_dlogA_dx: float,
    np_min_ratio: float,
    np_max_ratio: float,
    te_min: float,
    te_max_ratio: float,
    tp_min: float,
    mach_min: float | None,
    mach_max: float | None,
    margin_slack_max: float,
    margin_slack_weight: float,
    smooth_weight: float,
    control_slew_weight: float,
    control_curvature_weight: float,
    state_curvature_weight: float,
    warm_profile_track_weight: float,
    warm_control_track_weight: float,
    seed_fraction: float | None,
    warm_start: str,
    warm_start_dx: float | None,
    ipopt_max_iter: int,
    ipopt_tol: float,
    objective_weight: float,
) -> PiecewisePrototypeResult:
    if L_total <= 0.0:
        raise ValueError("L_total must be positive.")
    if x0 <= 0.0:
        raise ValueError("x0 must be positive.")
    if activation_width < 0.0:
        raise ValueError("activation_width must be nonnegative.")

    active_length = float(L_total) - float(x0) - float(activation_width)
    if active_length <= 0.0:
        raise ValueError("L_total must exceed x0 + activation_width.")

    inlet = _prepare_inlet_constants(
        n_p_in=float(n_p_in),
        Z_in=float(Z_in),
        T_p_in=float(T_p_in),
        T_e_in=float(T_e_in),
        A_in=float(A_in),
        B=float(B),
        seed_fraction=seed_fraction,
    )

    active_result, active_payload = _solve_active_segment(
        solver_mode=str(active_solver_mode),
        active_out_dir=Path(active_out_dir),
        n_p_in=float(n_p_in),
        Z_in=float(Z_in),
        T_p_in=float(T_p_in),
        T_e_in=float(T_e_in),
        A_in=float(A_in),
        B=float(B),
        active_length=float(active_length),
        seed_fraction=float(inlet.seed_fraction),
        warm_start_dx=warm_start_dx,
        n_intervals=int(n_intervals),
        transcription=str(transcription),
        min_margin=float(min_margin),
        A_min_ratio=float(A_min_ratio),
        A_max_ratio=float(A_max_ratio),
        max_abs_dlogA_dx=float(max_abs_dlogA_dx),
        np_min_ratio=float(np_min_ratio),
        np_max_ratio=float(np_max_ratio),
        te_min=float(te_min),
        te_max_ratio=float(te_max_ratio),
        tp_min=float(tp_min),
        mach_min=None if mach_min is None else float(mach_min),
        mach_max=None if mach_max is None else float(mach_max),
        margin_slack_max=float(margin_slack_max),
        margin_slack_weight=float(margin_slack_weight),
        smooth_weight=float(smooth_weight),
        control_slew_weight=float(control_slew_weight),
        control_curvature_weight=float(control_curvature_weight),
        state_curvature_weight=float(state_curvature_weight),
        warm_profile_track_weight=float(warm_profile_track_weight),
        warm_control_track_weight=float(warm_control_track_weight),
        warm_start=str(warm_start),
        ipopt_max_iter=int(ipopt_max_iter),
        ipopt_tol=float(ipopt_tol),
        objective_weight=float(objective_weight),
    )

    _, _, passive_exit_n_p, passive_exit_v_p = _isentropic_state_from_mach(
        stagnation_temperature=float(nozzle_stagnation_temperature),
        stagnation_pressure=float(nozzle_stagnation_pressure),
        mach=float(passive_mach_out),
    )
    target_dot_N = float(active_result.n_p[0] * active_result.v_p[0] * active_result.A[0])
    passive_A_exit = target_dot_N / max(passive_exit_n_p * passive_exit_v_p, _EPS)
    passive = _build_passive_nozzle_profile(
        A_exit=float(passive_A_exit),
        seed_fraction=float(inlet.seed_fraction),
        B=float(B),
        length=float(x0),
        n_points=int(passive_points),
        stagnation_temperature=float(nozzle_stagnation_temperature),
        stagnation_pressure=float(nozzle_stagnation_pressure),
        mach_in=float(passive_mach_in),
        mach_out=float(passive_mach_out),
        passive_te_ratio=float(passive_te_ratio),
    )

    activation, activation_payload = _solve_activation_segment(
        x_start=float(x0),
        width=float(activation_width),
        n_points=int(activation_points),
        passive_end=passive,
        active_result=active_result,
        B=float(B),
        seed_fraction=float(inlet.seed_fraction),
        tp_min=float(tp_min),
        min_margin=float(min_margin),
        mach_min=mach_min,
        mach_max=mach_max,
        max_abs_dlogA_dx=float(max_abs_dlogA_dx),
        smooth_weight=float(smooth_weight),
        control_slew_weight=float(control_slew_weight),
        control_curvature_weight=float(control_curvature_weight),
        state_curvature_weight=float(state_curvature_weight),
        warm_profile_track_weight=float(warm_profile_track_weight),
        warm_control_track_weight=float(warm_control_track_weight),
        ipopt_max_iter=int(ipopt_max_iter),
        ipopt_tol=float(ipopt_tol),
        current_sharpness=float(activation_current_sharpness),
        current_floor_frac=float(activation_current_floor_frac),
        closure_blend_current_frac=float(activation_closure_blend_current_frac),
    )
    stitched = _concat_segments(passive, activation, active_result, float(x0 + activation_width))

    total_volume = float(np.trapezoid(stitched["A"], stitched["x"]))
    return PiecewisePrototypeResult(
        passive=passive,
        activation=activation,
        active=active_result,
        activation_payload=activation_payload,
        active_payload=active_payload,
        x=stitched["x"],
        n_p=stitched["n_p"],
        T_e=stitched["T_e"],
        T_p=stitched["T_p"],
        A=stitched["A"],
        v_p=stitched["v_p"],
        n_e=stitched["n_e"],
        beta=stitched["beta"],
        eta=stitched["eta"],
        J_x=stitched["J_x"],
        J_y=stitched["J_y"],
        E_x=stitched["E_x"],
        mach=stitched["mach"],
        velikhov_margin=stitched["velikhov_margin"],
        current_turn_on_x=float(x0),
        activation_start_x=float(x0),
        active_start_x=float(x0 + activation_width),
        nozzle_stagnation_temperature=float(nozzle_stagnation_temperature),
        nozzle_stagnation_pressure=float(nozzle_stagnation_pressure),
        passive_inlet_mach=float(passive_mach_in),
        passive_exit_mach=float(passive_mach_out),
        total_volume=total_volume,
    )


def _save_plot(result: PiecewisePrototypeResult, save_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    fig, axes = plt.subplots(3, 2, figsize=(12, 10), sharex=True)
    axes = axes.reshape(-1)
    x = result.x
    panels = [
        ("A (m^2)", result.A),
        ("T_e (K)", result.T_e),
        ("T_p (K)", result.T_p),
        ("J_x (A/m^2)", result.J_x),
        ("Mach", result.mach),
        ("-J_x E_x (MW/m^3)", -(result.J_x * result.E_x) / 1e6),
    ]

    for ax, (label, y) in zip(axes, panels):
        ax.plot(x, np.asarray(y, dtype=float), lw=2)
        ax.axvline(result.current_turn_on_x, color="tab:orange", ls="--", lw=1.5)
        ax.axvline(result.active_start_x, color="tab:red", ls="--", lw=1.5)
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
    axes[-2].set_xlabel("x (m)")
    axes[-1].set_xlabel("x (m)")
    fig.suptitle("Piecewise-current prototype: passive nozzle + activation ramp + active NLP")
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=180)
    plt.close(fig)


def _summary_payload(result: PiecewisePrototypeResult, *, B: float) -> dict[str, object]:
    return {
        "model": "piecewise_current_prototype_v1",
        "ok": bool(result.active.acceptable),
        "solver_success": bool(result.active.success),
        "acceptable": bool(result.active.acceptable),
        "prototype_limitations": [
            "Passive nozzle is an isentropic proxy, not a solved plasma/nozzle NLP.",
            "Stage B uses a prescribed tanh current turn-on profile and solves the state/area bridge with NLP.",
            "Stage C still uses the downstream post-Jeffrey active solver rather than a fully coupled 3-stage NLP.",
        ],
        "current_turn_on_x_m": float(result.current_turn_on_x),
        "activation_start_x_m": float(result.activation_start_x),
        "active_start_x_m": float(result.active_start_x),
        "passive_length_m": float(result.current_turn_on_x),
        "activation_width_m": float(result.active_start_x - result.activation_start_x),
        "active_length_m": float(result.active.x[-1]),
        "passive_inlet_mach": float(result.passive_inlet_mach),
        "passive_exit_mach": float(result.passive_exit_mach),
        "nozzle_stagnation_temperature_K": float(result.nozzle_stagnation_temperature),
        "nozzle_stagnation_pressure_Pa": float(result.nozzle_stagnation_pressure),
        "total_device_volume_m3": float(result.total_volume),
        "activation_payload": result.activation_payload,
        "active_payload": result.active_payload,
    }


def main() -> int:
    args = _build_parser().parse_args()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    result = optimize_piecewise_current_profile(
        active_solver_mode=str(args.active_solver_mode),
        active_out_dir=out_dir / "active_segment",
        n_p_in=float(args.np_in),
        Z_in=float(args.z_in),
        T_p_in=float(args.tp_in),
        T_e_in=float(args.te_in),
        A_in=float(args.A_in),
        B=float(args.B),
        L_total=float(args.L_total),
        x0=float(args.x0),
        activation_width=float(args.activation_width),
        nozzle_stagnation_temperature=float(args.nozzle_stagnation_temp),
        nozzle_stagnation_pressure=float(args.nozzle_stagnation_pressure),
        passive_mach_in=float(args.passive_mach_in),
        passive_mach_out=float(args.passive_mach_out),
        passive_te_ratio=float(args.passive_te_ratio),
        activation_current_sharpness=float(args.activation_current_sharpness),
        activation_current_floor_frac=float(args.activation_current_floor_frac),
        activation_closure_blend_current_frac=float(args.activation_closure_blend_current_frac),
        passive_points=int(args.passive_points),
        activation_points=int(args.activation_points),
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

    payload = _summary_payload(result, B=float(args.B))
    (out_dir / "piecewise_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    np.savez_compressed(
        out_dir / "piecewise_profile.npz",
        x=result.x,
        n_p=result.n_p,
        T_e=result.T_e,
        T_p=result.T_p,
        A=result.A,
        v_p=result.v_p,
        n_e=result.n_e,
        beta=result.beta,
        eta=result.eta,
        J_x=result.J_x,
        J_y=result.J_y,
        E_x=result.E_x,
        mach=result.mach,
        velikhov_margin=result.velikhov_margin,
    )
    _save_plot(result, out_dir / "piecewise_profile.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
