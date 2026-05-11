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
    DesignValueWeights,
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
)
from v6_global_marginal.local_algebraic_closure_global import (
    compute_currents_fields_global,
    local_closure_global_with_partials,
)
from v6_global_marginal.pde_solver_v6_batch_global import ForwardPDESolverV6BatchGlobal


_EPS = 1e-30
_TP_MIN = 1.0
_DELTA_MIN = 1e-12
_FION_MIN = 1e-12
_FION_MAX = 1.0 - 1e-12
_SAHA_K_MIN = 1e-100
_DEFAULT_SEED_FRACTION_GUESS = 1e-4
_A_IN = 1.0
OBJECTIVE_PROFILE_LAB_POC_V2 = "lab_poc_v2"
OBJECTIVE_PROFILE_ENTHALPY_EXTRACTION = "enthalpy_extraction"
OBJECTIVE_PROFILES = (OBJECTIVE_PROFILE_LAB_POC_V2, OBJECTIVE_PROFILE_ENTHALPY_EXTRACTION)


def _normalize_objective_profile(objective_profile: str) -> str:
    profile = str(objective_profile or OBJECTIVE_PROFILE_LAB_POC_V2).strip().lower()
    aliases = {
        "lab": OBJECTIVE_PROFILE_LAB_POC_V2,
        "lab_poc": OBJECTIVE_PROFILE_LAB_POC_V2,
        "lab_poc_v2_objective": OBJECTIVE_PROFILE_LAB_POC_V2,
        "enthalpy": OBJECTIVE_PROFILE_ENTHALPY_EXTRACTION,
        "enthalpy_extraction_percent": OBJECTIVE_PROFILE_ENTHALPY_EXTRACTION,
        "enthalpy_extraction_objective": OBJECTIVE_PROFILE_ENTHALPY_EXTRACTION,
    }
    profile = aliases.get(profile, profile)
    if profile not in OBJECTIVE_PROFILES:
        raise ValueError(f"unknown objective_profile={objective_profile!r}; expected one of {OBJECTIVE_PROFILES!r}")
    return profile


@dataclass(frozen=True)
class InletDesign:
    n_p: float
    T_e: float
    T_p: float
    Z: float
    J_x: float
    I_0: float
    dot_N: float
    v_in: float
    seed_fraction: float
    mach: float
    velikhov_margin: float


@dataclass(frozen=True)
class WarmStartProfile:
    x: np.ndarray
    n_p: np.ndarray
    T_e: np.ndarray
    A: np.ndarray
    sigma_logA: np.ndarray
    inlet_n_p: float
    inlet_T_e: float
    inlet_Z: float
    inlet_I0: float
    inlet_seed_fraction: float
    source: str


@dataclass(frozen=True)
class OptimizedAreaProfile:
    success: bool
    acceptable: bool
    return_status: str
    objective_delta_Te: float
    objective_value: float
    inlet: InletDesign
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
    duals: dict


@dataclass(frozen=True)
class FeasibilityThresholds:
    defect_inf_tol: float = 1e-4
    defect_rms_tol: float = 1e-5
    boundary_inf_tol: float = 1e-8
    path_slack_tol: float = 1e-6


def _design_value_weights_lab_poc_v2_objective() -> DesignValueWeights:
    # Keep the lab-poc structure, but remove terms that are constant for a fixed run
    # or not aligned with the current v2 objective emphasis.
    return DesignValueWeights(
        outlet_delta_te_per_kK=1.0,
        outlet_delta_ratio=0.35,
        outlet_f_ion=0.35,
        outlet_mhd_output_per_100MWe=0.0,
        inlet_delta_ratio_penalty=0.75,
        inlet_mach_penalty=0.10,
        magnetic_field_T_penalty=0.0,
        device_length_per_5m_penalty=0.0,
    )


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Optimize A(x) with CasADi + IPOPT, with inlet variables inside the NLP."
    )
    p.add_argument("--np-in-guess", type=float, required=True)
    p.add_argument("--np-in-min", type=float, required=True)
    p.add_argument("--np-in-max", type=float, required=True)
    p.add_argument("--te-in-guess", type=float, required=True)
    p.add_argument("--te-in-min", type=float, required=True)
    p.add_argument("--te-in-max", type=float, required=True)
    p.add_argument("--z-in-guess", type=float, required=True)
    p.add_argument("--z-in-min", type=float, required=True)
    p.add_argument("--z-in-max", type=float, required=True)
    p.add_argument("--jx-in-guess", type=float, required=True)
    p.add_argument("--jx-in-min", type=float, required=True)
    p.add_argument("--jx-in-max", type=float, required=True)
    p.add_argument("--seed-fraction-guess", type=float, default=_DEFAULT_SEED_FRACTION_GUESS)
    p.add_argument("--seed-fraction-min", type=float, default=1e-13)
    p.add_argument("--seed-fraction-max", type=float, default=5e-2)
    p.add_argument(
        "--inlet-margin-mode",
        type=str,
        choices=("equality", "lower-bound"),
        default="equality",
        help="enforce G_in == min_margin or only G_in >= min_margin",
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
    p.add_argument("--max-abs-dlogA-dx", type=float, default=2.0)
    p.add_argument("--np-min-ratio", type=float, default=1e-6)
    p.add_argument("--np-max-ratio", type=float, default=100.0)
    p.add_argument("--te-min", type=float, default=100.0)
    p.add_argument("--te-max-ratio", type=float, default=20.0)
    p.add_argument("--tp-min", type=float, default=1.0)
    p.add_argument("--mach-min", type=float, default=None)
    p.add_argument("--mach-max", type=float, default=None)
    p.add_argument("--margin-slack-max", type=float, default=0.0)
    p.add_argument("--margin-slack-weight", type=float, default=0.0)
    p.add_argument("--smooth-weight", type=float, default=0.0)
    p.add_argument("--control-slew-weight", type=float, default=0.0)
    p.add_argument("--control-curvature-weight", type=float, default=0.0)
    p.add_argument("--state-curvature-weight", type=float, default=0.0)
    p.add_argument("--warm-profile-track-weight", type=float, default=0.0)
    p.add_argument("--warm-control-track-weight", type=float, default=0.0)
    p.add_argument("--ipopt-max-iter", type=int, default=1000)
    p.add_argument("--ipopt-tol", type=float, default=1e-7)
    p.add_argument("--objective-weight", type=float, default=1.0)
    p.add_argument(
        "--objective-profile",
        type=str,
        default=OBJECTIVE_PROFILE_LAB_POC_V2,
        choices=OBJECTIVE_PROFILES,
        help="stage objective profile; enthalpy_extraction maximizes percent inlet stagnation enthalpy extracted",
    )
    p.add_argument(
        "--area-scale-m2",
        type=float,
        default=_A_IN,
        help="physical inlet area used by A[0], dot_N, current density, and enthalpy-flux semantics",
    )
    p.add_argument("--heavy-particle-mass-kg", type=float, default=M_P)
    p.add_argument("--seed-ionization-energy-J", type=float, default=E_I)
    p.add_argument("--electron-particle-sigma-m2", type=float, default=SIGMA_EP)
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


def _saha_n_e_symbolic(n_p, T_e, seed_fraction, *, seed_ionization_energy_J: float = E_I):
    n_p_safe = ca.fmax(n_p, 1.0)
    T_e_safe = ca.fmax(T_e, 1.0)
    seed_safe = ca.fmax(seed_fraction, 1e-12)
    saha_a = 2.0 * math.pi * M_E * K_B * T_e_safe / (H_P * H_P)
    saha_k = (saha_a ** 1.5) * ca.exp(-float(seed_ionization_energy_J) / (K_B * T_e_safe))
    saha_k_safe = ca.fmax(saha_k, _SAHA_K_MIN)
    n_s = seed_safe * n_p_safe
    return 2.0 * n_s / (1.0 + ca.sqrt(1.0 + 4.0 * n_s / saha_k_safe))


def _beta_symbolic(n_p, T_e, *, B: float, sigma_ep: float = SIGMA_EP):
    n_p_safe = ca.fmax(n_p, 1.0)
    T_e_safe = ca.fmax(T_e, 1.0)
    v_te = ca.sqrt(2.0 * K_B * T_e_safe / M_E)
    return E_CHARGE * float(B) / (M_E * n_p_safe * float(sigma_ep) * v_te + _EPS)


def _make_stage_function(
    *,
    B: float,
    heavy_particle_mass_kg: float = M_P,
    seed_ionization_energy_J: float = E_I,
    sigma_ep: float = SIGMA_EP,
) -> ca.Function:
    x = ca.SX.sym("x", 3)
    sigma = ca.SX.sym("sigma")
    params = ca.SX.sym("params", 3)
    dot_N = params[0]
    I_0 = params[1]
    seed_fraction = params[2]

    n_p = x[0]
    T_e = x[1]
    A = x[2]

    n_p_safe = ca.fmax(n_p, 1.0)
    T_e_safe = ca.fmax(T_e, 1.0)
    A_safe = ca.fmax(A, 1e-12)

    mass = float(heavy_particle_mass_kg)
    ionization_energy = float(seed_ionization_energy_J)
    sigma_ep_value = float(sigma_ep)
    beta = _beta_symbolic(n_p_safe, T_e_safe, B=float(B), sigma_ep=sigma_ep_value)
    n_e = _saha_n_e_symbolic(
        n_p_safe,
        T_e_safe,
        seed_fraction,
        seed_ionization_energy_J=ionization_energy,
    )
    eta = M_E * n_p_safe * sigma_ep_value * ca.sqrt(2.0 * K_B * T_e_safe / M_E) / (
        E_CHARGE * E_CHARGE * n_e + _EPS
    )
    q = E_CHARGE * n_e * dot_N / (I_0 * n_p_safe + _EPS)
    b2 = beta * beta
    Z = b2 * (q - 1.0) - 1.0

    one_plus_z = 1.0 + Z
    den = b2 + one_plus_z
    F = b2 * (b2 + one_plus_z * one_plus_z) / (den * den + _EPS)

    v_p = dot_N / (n_p_safe * A_safe)
    T_p = T_e_safe - mass * v_p * v_p * F / (3.0 * K_B)
    T_p_safe_for_math = ca.fmax(T_p, _TP_MIN)
    dTp_dnp = ca.gradient(T_p, n_p)
    dTp_dTe = ca.gradient(T_p, T_e)
    dTp_dA = ca.gradient(T_p, A)

    jfac = E_CHARGE * n_e * v_p
    J_x = I_0 / A_safe
    J_y = -beta * one_plus_z / (den + _EPS) * jfac
    E_x = -b2 * Z / (den + _EPS) * eta * jfac
    nu_E = eta * 2.0 * E_CHARGE * E_CHARGE * n_e / mass

    dA_dx = sigma * A_safe

    M11 = (-mass * v_p * v_p + K_B * T_p) + K_B * n_p_safe * dTp_dnp
    M12 = K_B * n_p_safe * dTp_dTe
    M13 = K_B * n_p_safe * dTp_dA - mass * n_p_safe * v_p * v_p / A_safe

    E11 = -T_p + 1.5 * n_p_safe * dTp_dnp
    E12 = 1.5 * n_p_safe * dTp_dTe
    E13 = 1.5 * n_p_safe * dTp_dA

    rhs_m = J_y * float(B) - M13 * dA_dx
    rhs_e = 1.5 * nu_E * n_e * (T_e_safe - T_p) / (v_p + _EPS) - E13 * dA_dx
    det = M11 * E12 - M12 * E11
    det_safe = det + _EPS

    dn_dx = (rhs_m * E12 - M12 * rhs_e) / det_safe
    dTe_dx = (M11 * rhs_e - rhs_m * E11) / det_safe

    c_s = ca.sqrt((5.0 / 3.0) * K_B * T_p_safe_for_math / mass + _EPS)
    mach = v_p / c_s

    n_s = ca.fmax(seed_fraction, 1e-12) * n_p_safe
    f_I_raw = n_e / (n_s + _EPS)
    f_I = ca.fmin(ca.fmax(f_I_raw, _FION_MIN), _FION_MAX)
    delta_raw = T_e_safe / T_p_safe_for_math - 1.0
    delta = ca.fmax(delta_raw, _DELTA_MIN)
    alpha = (K_B * T_e_safe / (2.0 * ionization_energy)) * (2.0 - f_I) / (1.0 - f_I + _EPS)
    G = 4.0 * alpha * (2.0 + 1.0 / delta) * (
        1.0 + alpha * (1.0 + 1.0 / delta)
    ) - b2

    out = ca.vertcat(dn_dx, dTe_dx, dA_dx, T_p, v_p, n_e, beta, eta, Z, J_x, J_y, E_x, mach, G)
    return ca.Function("stage_fun_v2", [x, sigma, params], [out])


def _evaluate_inlet_design_numeric(
    *,
    n_p_in: float,
    T_e_in: float,
    Z_in: float,
    I_0: float,
    seed_fraction: float,
    B: float,
    area_scale: float = _A_IN,
    heavy_particle_mass_kg: float = M_P,
    seed_ionization_energy_J: float = E_I,
    sigma_ep: float = SIGMA_EP,
) -> InletDesign:
    n_e_in = float(
        _saha_n_e_numeric(
            n_p_in,
            T_e_in,
            seed_fraction,
            seed_ionization_energy_J=float(seed_ionization_energy_J),
        )
    )
    beta_in = float(_beta_numeric(n_p_in, T_e_in, B=float(B), sigma_ep=float(sigma_ep)))
    b2 = beta_in * beta_in
    den = _safe_signed_scalar(b2 + 1.0 + float(Z_in))
    v_in = float(I_0) * den / (b2 * E_CHARGE * n_e_in + _EPS)
    dot_N = float(n_p_in) * v_in * float(area_scale)
    one_plus_z = 1.0 + float(Z_in)
    F = b2 * (b2 + one_plus_z * one_plus_z) / (den * den + _EPS)
    mass = float(heavy_particle_mass_kg)
    T_p_in = float(T_e_in) - mass * v_in * v_in * F / (3.0 * K_B)
    c_s = math.sqrt((5.0 / 3.0) * K_B * max(T_p_in, _TP_MIN) / mass + _EPS)
    mach = v_in / max(c_s, _EPS)
    n_s = max(float(seed_fraction) * float(n_p_in), 1e-30)
    f_I = min(max(n_e_in / n_s, _FION_MIN), _FION_MAX)
    delta = max(float(T_e_in) / max(T_p_in, _TP_MIN) - 1.0, _DELTA_MIN)
    alpha = (K_B * float(T_e_in) / (2.0 * float(seed_ionization_energy_J))) * (2.0 - f_I) / (
        1.0 - f_I + _EPS
    )
    G = 4.0 * alpha * (2.0 + 1.0 / delta) * (1.0 + alpha * (1.0 + 1.0 / delta)) - b2
    return InletDesign(
        n_p=float(n_p_in),
        T_e=float(T_e_in),
        T_p=float(T_p_in),
        Z=float(Z_in),
        J_x=float(I_0) / max(float(area_scale), _EPS),
        I_0=float(I_0),
        dot_N=float(dot_N),
        v_in=float(v_in),
        seed_fraction=float(seed_fraction),
        mach=float(mach),
        velikhov_margin=float(G),
    )


def _saha_n_e_numeric(
    n_p: float,
    T_e: float,
    seed_fraction: float,
    *,
    seed_ionization_energy_J: float = E_I,
) -> float:
    n_p_safe = max(float(n_p), 1.0)
    T_e_safe = max(float(T_e), 1.0)
    seed_safe = max(float(seed_fraction), 1e-12)
    saha_a = 2.0 * math.pi * M_E * K_B * T_e_safe / (H_P * H_P)
    saha_k = (saha_a ** 1.5) * math.exp(-float(seed_ionization_energy_J) / (K_B * T_e_safe))
    saha_k_safe = max(saha_k, _SAHA_K_MIN)
    n_s = seed_safe * n_p_safe
    return 2.0 * n_s / (1.0 + math.sqrt(1.0 + 4.0 * n_s / saha_k_safe))


def _beta_numeric(n_p: float, T_e: float, *, B: float, sigma_ep: float = SIGMA_EP) -> float:
    n_p_safe = max(float(n_p), 1.0)
    T_e_safe = max(float(T_e), 1.0)
    v_te = math.sqrt(2.0 * K_B * T_e_safe / M_E)
    return E_CHARGE * float(B) / (M_E * n_p_safe * float(sigma_ep) * v_te + _EPS)


def _project_inlet_z_guess_to_margin(
    *,
    n_p_in: float,
    T_e_in: float,
    Z_in_guess: float,
    Z_in_min: float,
    Z_in_max: float,
    I_0: float,
    seed_fraction: float,
    B: float,
    area_scale: float = _A_IN,
    heavy_particle_mass_kg: float = M_P,
    seed_ionization_energy_J: float = E_I,
    sigma_ep: float = SIGMA_EP,
    sample_count: int = 33,
    bisection_steps: int = 80,
) -> float:
    lower = float(min(Z_in_min, Z_in_max))
    upper = float(max(Z_in_min, Z_in_max))
    if upper <= lower:
        return float(np.clip(Z_in_guess, lower, upper))

    grid = np.unique(
        np.concatenate(
            [
                np.linspace(lower, upper, max(int(sample_count), 3), dtype=float),
                np.array([float(np.clip(Z_in_guess, lower, upper))], dtype=float),
            ]
        )
    )

    best_z = float(np.clip(Z_in_guess, lower, upper))
    best_abs_G = float("inf")
    values: list[tuple[float, float]] = []
    for z in grid:
        try:
            inlet = _evaluate_inlet_design_numeric(
                n_p_in=float(n_p_in),
                T_e_in=float(T_e_in),
                Z_in=float(z),
                I_0=float(I_0),
                seed_fraction=float(seed_fraction),
                B=float(B),
                area_scale=float(area_scale),
                heavy_particle_mass_kg=float(heavy_particle_mass_kg),
                seed_ionization_energy_J=float(seed_ionization_energy_J),
                sigma_ep=float(sigma_ep),
            )
        except Exception:
            continue
        G = float(inlet.velikhov_margin)
        if not np.isfinite(G):
            continue
        values.append((float(z), G))
        if abs(G) < best_abs_G:
            best_abs_G = abs(G)
            best_z = float(z)

    if not values:
        return float(np.clip(Z_in_guess, lower, upper))

    values.sort(key=lambda item: item[0])
    bracket: tuple[tuple[float, float], tuple[float, float]] | None = None
    bracket_distance = float("inf")
    for lhs, rhs in zip(values[:-1], values[1:]):
        z_l, g_l = lhs
        z_r, g_r = rhs
        if g_l == 0.0:
            return float(z_l)
        if g_r == 0.0:
            return float(z_r)
        if g_l * g_r > 0.0:
            continue
        distance = abs(float(np.clip(Z_in_guess, z_l, z_r)) - float(Z_in_guess))
        if distance < bracket_distance:
            bracket_distance = distance
            bracket = (lhs, rhs)

    if bracket is None:
        return best_z

    (z_l, g_l), (z_r, g_r) = bracket
    for _ in range(max(int(bisection_steps), 1)):
        z_m = 0.5 * (z_l + z_r)
        inlet_m = _evaluate_inlet_design_numeric(
            n_p_in=float(n_p_in),
            T_e_in=float(T_e_in),
            Z_in=float(z_m),
            I_0=float(I_0),
            seed_fraction=float(seed_fraction),
            B=float(B),
            area_scale=float(area_scale),
            heavy_particle_mass_kg=float(heavy_particle_mass_kg),
            seed_ionization_energy_J=float(seed_ionization_energy_J),
            sigma_ep=float(sigma_ep),
        )
        g_m = float(inlet_m.velikhov_margin)
        if not np.isfinite(g_m):
            break
        if abs(g_m) < 1e-10:
            return float(z_m)
        if g_l * g_m <= 0.0:
            z_r, g_r = z_m, g_m
        else:
            z_l, g_l = z_m, g_m
    return float(0.5 * (z_l + z_r))


def _build_integrated_warm_start(
    *,
    x: np.ndarray,
    n_p_in_guess: float,
    T_e_in_guess: float,
    Z_in_guess: float,
    Z_in_min: float,
    Z_in_max: float,
    I_0_guess: float,
    seed_fraction_guess: float,
    B: float,
    area_scale: float = _A_IN,
    heavy_particle_mass_kg: float = M_P,
    seed_ionization_energy_J: float = E_I,
    sigma_ep: float = SIGMA_EP,
) -> WarmStartProfile | None:
    try:
        projected_Z = _project_inlet_z_guess_to_margin(
            n_p_in=float(n_p_in_guess),
            T_e_in=float(T_e_in_guess),
            Z_in_guess=float(Z_in_guess),
            Z_in_min=float(Z_in_min),
            Z_in_max=float(Z_in_max),
            I_0=float(I_0_guess),
            seed_fraction=float(seed_fraction_guess),
            B=float(B),
            area_scale=float(area_scale),
            heavy_particle_mass_kg=float(heavy_particle_mass_kg),
            seed_ionization_energy_J=float(seed_ionization_energy_J),
            sigma_ep=float(sigma_ep),
        )
        inlet = _evaluate_inlet_design_numeric(
            n_p_in=float(n_p_in_guess),
            T_e_in=float(T_e_in_guess),
            Z_in=float(projected_Z),
            I_0=float(I_0_guess),
            seed_fraction=float(seed_fraction_guess),
            B=float(B),
            area_scale=float(area_scale),
            heavy_particle_mass_kg=float(heavy_particle_mass_kg),
            seed_ionization_energy_J=float(seed_ionization_energy_J),
            sigma_ep=float(sigma_ep),
        )
        x_nodes = np.asarray(x, dtype=float)
        if x_nodes.ndim != 1 or x_nodes.size < 2 or np.any(np.diff(x_nodes) <= 0.0):
            return None

        stage = _make_stage_function(
            B=float(B),
            heavy_particle_mass_kg=float(heavy_particle_mass_kg),
            seed_ionization_energy_J=float(seed_ionization_energy_J),
            sigma_ep=float(sigma_ep),
        )
        params = np.array([inlet.dot_N, inlet.I_0, inlet.seed_fraction], dtype=float)
        n_points = x_nodes.size
        n_p = np.full(n_points, np.nan, dtype=float)
        T_e = np.full(n_points, np.nan, dtype=float)
        A = np.full(n_points, float(area_scale), dtype=float)
        sigma = np.zeros(n_points - 1, dtype=float)
        n_p[0] = float(inlet.n_p)
        T_e[0] = float(inlet.T_e)

        def rhs(state: np.ndarray) -> np.ndarray:
            out = np.asarray(stage(state, 0.0, params), dtype=float).reshape(-1)
            return out[:3]

        for k in range(n_points - 1):
            dx = float(x_nodes[k + 1] - x_nodes[k])
            state_k = np.array([n_p[k], T_e[k], float(area_scale)], dtype=float)
            k1 = rhs(state_k)
            k2 = rhs(state_k + 0.5 * dx * k1)
            k3 = rhs(state_k + 0.5 * dx * k2)
            k4 = rhs(state_k + dx * k3)
            state_kp1 = state_k + dx * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
            n_p[k + 1] = float(state_kp1[0])
            T_e[k + 1] = float(state_kp1[1])

        if not np.all(np.isfinite(n_p)) or not np.all(np.isfinite(T_e)):
            return None

        return WarmStartProfile(
            x=x_nodes.copy(),
            n_p=n_p,
            T_e=T_e,
            A=A,
            sigma_logA=sigma,
            inlet_n_p=float(inlet.n_p),
            inlet_T_e=float(inlet.T_e),
            inlet_Z=float(inlet.Z),
            inlet_I0=float(inlet.I_0),
            inlet_seed_fraction=float(inlet.seed_fraction),
            source="integrated_margin_projected",
        )
    except Exception:
        return None


def _build_global_marginal_warm_start(
    *,
    x: np.ndarray,
    n_p_in_guess: float,
    T_e_in_guess: float,
    Z_in_guess: float,
    Z_in_min: float,
    Z_in_max: float,
    I_0_guess: float,
    seed_fraction_guess: float,
    B: float,
    length: float,
    area_scale: float = _A_IN,
    heavy_particle_mass_kg: float = M_P,
    seed_ionization_energy_J: float = E_I,
    sigma_ep: float = SIGMA_EP,
) -> WarmStartProfile | None:
    try:
        if (
            not np.isclose(float(area_scale), float(_A_IN), rtol=1e-12, atol=1e-15)
            or not np.isclose(float(heavy_particle_mass_kg), float(M_P), rtol=1e-12, atol=0.0)
            or not np.isclose(float(seed_ionization_energy_J), float(E_I), rtol=1e-12, atol=0.0)
            or not np.isclose(float(sigma_ep), float(SIGMA_EP), rtol=1e-12, atol=0.0)
        ):
            return None
        projected_Z = _project_inlet_z_guess_to_margin(
            n_p_in=float(n_p_in_guess),
            T_e_in=float(T_e_in_guess),
            Z_in_guess=float(Z_in_guess),
            Z_in_min=float(Z_in_min),
            Z_in_max=float(Z_in_max),
            I_0=float(I_0_guess),
            seed_fraction=float(seed_fraction_guess),
            B=float(B),
            area_scale=float(area_scale),
            heavy_particle_mass_kg=float(heavy_particle_mass_kg),
            seed_ionization_energy_J=float(seed_ionization_energy_J),
            sigma_ep=float(sigma_ep),
        )
        inlet = _evaluate_inlet_design_numeric(
            n_p_in=float(n_p_in_guess),
            T_e_in=float(T_e_in_guess),
            Z_in=float(projected_Z),
            I_0=float(I_0_guess),
            seed_fraction=float(seed_fraction_guess),
            B=float(B),
            area_scale=float(area_scale),
            heavy_particle_mass_kg=float(heavy_particle_mass_kg),
            seed_ionization_energy_J=float(seed_ionization_energy_J),
            sigma_ep=float(sigma_ep),
        )

        x_nodes = np.asarray(x, dtype=float)
        if x_nodes.ndim != 1 or x_nodes.size < 2 or np.any(np.diff(x_nodes) <= 0.0):
            return None

        dx = float(length) / max(8 * (x_nodes.size - 1), 200)
        dx = min(max(dx, float(length) / 5000.0), float(length) / max(x_nodes.size - 1, 1))
        solver = ForwardPDESolverV6BatchGlobal(B=float(B), length=float(length))
        out = solver.solve_batch(
            n_p_in=np.array([inlet.n_p], dtype=float),
            Z_in=np.array([inlet.Z], dtype=float),
            T_p_in=np.array([inlet.T_p], dtype=float),
            T_e_in=np.array([inlet.T_e], dtype=float),
            A_in=np.array([float(area_scale)], dtype=float),
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

        n_p_guess = np.interp(x_nodes, x_ref, n_p_ref)
        T_e_guess = np.interp(x_nodes, x_ref, T_e_ref)
        A_guess = np.interp(x_nodes, x_ref, A_ref)
        sigma_guess = np.diff(np.log(np.maximum(A_guess, 1e-20))) / np.diff(x_nodes)

        return WarmStartProfile(
            x=x_nodes.copy(),
            n_p=n_p_guess,
            T_e=T_e_guess,
            A=A_guess,
            sigma_logA=sigma_guess,
            inlet_n_p=float(inlet.n_p),
            inlet_T_e=float(inlet.T_e),
            inlet_Z=float(inlet.Z),
            inlet_I0=float(inlet.I_0),
            inlet_seed_fraction=float(inlet.seed_fraction),
            source="marginal_global_solver_v6",
        )
    except Exception:
        return None


def _build_constant_warm_start(
    *,
    x: np.ndarray,
    n_p_in_guess: float,
    T_e_in_guess: float,
    Z_in_guess: float,
    I_0_guess: float,
    seed_fraction_guess: float,
    area_scale: float = _A_IN,
) -> WarmStartProfile:
    n_points = x.size
    return WarmStartProfile(
        x=x.copy(),
        n_p=np.full(n_points, float(n_p_in_guess), dtype=float),
        T_e=np.full(n_points, float(T_e_in_guess), dtype=float),
        A=np.full(n_points, float(area_scale), dtype=float),
        sigma_logA=np.zeros(n_points - 1, dtype=float),
        inlet_n_p=float(n_p_in_guess),
        inlet_T_e=float(T_e_in_guess),
        inlet_Z=float(Z_in_guess),
        inlet_I0=float(I_0_guess),
        inlet_seed_fraction=float(seed_fraction_guess),
        source="constant",
    )


def _evaluate_profile_numeric(
    *,
    x: np.ndarray,
    n_p: np.ndarray,
    T_e: np.ndarray,
    A: np.ndarray,
    inlet: InletDesign,
    B: float,
    sigma_logA: np.ndarray,
    stage_fun: ca.Function | None = None,
    heavy_particle_mass_kg: float = M_P,
    seed_ionization_energy_J: float = E_I,
    sigma_ep: float = SIGMA_EP,
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
        if stage_fun is not None:
            sigma_idx = min(max(i, 0), max(sigma_logA.size - 1, 0))
            sigma_i = float(sigma_logA[sigma_idx]) if sigma_logA.size else 0.0
            out = np.asarray(
                stage_fun(
                    np.array([float(n_p[i]), float(T_e[i]), float(A[i])], dtype=float),
                    sigma_i,
                    np.array([float(inlet.dot_N), float(inlet.I_0), float(inlet.seed_fraction)], dtype=float),
                ),
                dtype=float,
            ).reshape(-1)
            T_p[i] = float(out[3])
            v_p[i] = float(out[4])
            n_e[i] = float(out[5])
            beta[i] = float(out[6])
            eta[i] = float(out[7])
            Z[i] = float(out[8])
            J_x[i] = float(out[9])
            J_y[i] = float(out[10])
            E_x[i] = float(out[11])
            mach[i] = float(out[12])
            G[i] = float(out[13])
        else:
            vals = local_closure_global_with_partials(
                n_p=float(n_p[i]),
                T_e=float(T_e[i]),
                A=float(A[i]),
                dot_N=inlet.dot_N,
                I_0=inlet.I_0,
                seed_fraction=inlet.seed_fraction,
                B=float(B),
                sigma_ep=float(sigma_ep),
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
            c_s = math.sqrt((5.0 / 3.0) * K_B * max(T_p[i], _TP_MIN) / float(heavy_particle_mass_kg))
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
        inlet_n_p=float(warm.inlet_n_p),
        inlet_T_e=float(warm.inlet_T_e),
        inlet_Z=float(warm.inlet_Z),
        inlet_I0=float(warm.inlet_I0),
        inlet_seed_fraction=float(warm.inlet_seed_fraction),
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
    inlet_n_p_bounds: tuple[float, float],
    inlet_T_e_bounds: tuple[float, float],
    inlet_Z_bounds: tuple[float, float],
    inlet_I0_bounds: tuple[float, float],
    inlet_seed_fraction_bounds: tuple[float, float],
    area_scale: float = _A_IN,
) -> WarmStartProfile:
    n_p = np.clip(np.asarray(warm.n_p, dtype=float), n_p_floor, n_p_ceil)
    T_e = np.clip(np.asarray(warm.T_e, dtype=float), T_e_floor, T_e_ceil)
    A = np.clip(np.asarray(warm.A, dtype=float), A_floor, A_ceil)
    n_p[0] = float(np.clip(warm.inlet_n_p, *inlet_n_p_bounds))
    T_e[0] = float(np.clip(warm.inlet_T_e, *inlet_T_e_bounds))
    A[0] = float(area_scale)
    sigma = np.diff(np.log(np.maximum(A, 1e-20))) / np.diff(np.asarray(warm.x, dtype=float))
    sigma = np.clip(sigma, sigma_floor, sigma_ceil)
    return WarmStartProfile(
        x=np.asarray(warm.x, dtype=float).copy(),
        n_p=n_p,
        T_e=T_e,
        A=A,
        sigma_logA=sigma,
        inlet_n_p=float(np.clip(warm.inlet_n_p, *inlet_n_p_bounds)),
        inlet_T_e=float(np.clip(warm.inlet_T_e, *inlet_T_e_bounds)),
        inlet_Z=float(np.clip(warm.inlet_Z, *inlet_Z_bounds)),
        inlet_I0=float(np.clip(warm.inlet_I0, *inlet_I0_bounds)),
        inlet_seed_fraction=float(np.clip(warm.inlet_seed_fraction, *inlet_seed_fraction_bounds)),
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
    max_turns = max(8, int(math.ceil(max(n_intervals, 1) / 20.0)))
    area_tv_ratio = _total_variation_ratio(A)
    regularity_ok = bool(
        _count_sign_changes(area_steps) <= max_turns
        and _count_sign_changes(sigma) <= max_turns
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
    stage_params: np.ndarray,
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
        out_k = np.asarray(stage_fun(xk, uk, stage_params), dtype=float).reshape(-1)
        out_kp1 = np.asarray(stage_fun(xkp1, uk, stage_params), dtype=float).reshape(-1)
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
        out_mid = np.asarray(stage_fun(xmid_phys, uk, stage_params), dtype=float).reshape(-1)
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
    stage_params: np.ndarray,
    inlet_target: tuple[float, float, float],
    state_bounds: dict[str, float | None],
    sigma_bounds: tuple[float, float],
    thresholds: FeasibilityThresholds,
    margin_slack_nodes: np.ndarray | None = None,
    margin_slack_mid: np.ndarray | None = None,
) -> dict:
    velikhov_margin_activity_threshold = 1e-3
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
        stage_params=stage_params,
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

    finite_margin_mask = np.isfinite(velikhov_margin)
    if np.any(finite_margin_mask):
        margin_values = np.asarray(velikhov_margin, dtype=float)
        margin_indices = np.flatnonzero(finite_margin_mask)
        min_margin_local_idx = int(np.argmin(margin_values[finite_margin_mask]))
        margin_min_index = int(margin_indices[min_margin_local_idx])
        margin_min_x = float(np.asarray(x_nodes, dtype=float)[margin_min_index])
        margin_lt_threshold_fraction = float(
            np.mean(margin_values[finite_margin_mask] < float(velikhov_margin_activity_threshold))
        )
    else:
        margin_min_index = -1
        margin_min_x = float("nan")
        margin_lt_threshold_fraction = float("nan")

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
        "velikhov_margin_min_index": margin_min_index,
        "velikhov_margin_min_x_m": margin_min_x,
        "velikhov_margin_lt_1e_3_fraction": margin_lt_threshold_fraction,
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
            "velikhov_margin_activity_threshold": float(velikhov_margin_activity_threshold),
        },
    }


def optimize_area_profile(
    *,
    n_p_in_guess: float,
    n_p_in_min: float,
    n_p_in_max: float,
    T_e_in_guess: float,
    T_e_in_min: float,
    T_e_in_max: float,
    Z_in_guess: float,
    Z_in_min: float,
    Z_in_max: float,
    J_x_in_guess: float,
    J_x_in_min: float,
    J_x_in_max: float,
    seed_fraction_guess: float = _DEFAULT_SEED_FRACTION_GUESS,
    seed_fraction_min: float = 1e-13,
    seed_fraction_max: float = 5e-2,
    inlet_margin_mode: str = "equality",
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
    warm_profile: WarmStartProfile | None = None,
    feasibility_thresholds: FeasibilityThresholds | None = None,
    ipopt_max_iter: int = 1000,
    ipopt_tol: float = 1e-7,
    objective_weight: float = 1.0,
    objective_profile: str = OBJECTIVE_PROFILE_LAB_POC_V2,
    area_scale: float = _A_IN,
    heavy_particle_mass_kg: float = M_P,
    seed_ionization_energy_J: float = E_I,
    sigma_ep: float = SIGMA_EP,
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
    if not (n_p_in_min <= n_p_in_guess <= n_p_in_max):
        raise ValueError("n_p_in_guess must lie within [n_p_in_min, n_p_in_max].")
    if not (T_e_in_min <= T_e_in_guess <= T_e_in_max):
        raise ValueError("T_e_in_guess must lie within [T_e_in_min, T_e_in_max].")
    if not (Z_in_min <= Z_in_guess <= Z_in_max):
        raise ValueError("Z_in_guess must lie within [Z_in_min, Z_in_max].")
    if not (J_x_in_min <= J_x_in_guess <= J_x_in_max):
        raise ValueError("J_x_in_guess must lie within [J_x_in_min, J_x_in_max].")
    if not (seed_fraction_min < seed_fraction_guess < seed_fraction_max):
        raise ValueError("seed_fraction_guess must lie within (seed_fraction_min, seed_fraction_max).")
    if inlet_margin_mode not in ("equality", "lower-bound"):
        raise ValueError("inlet_margin_mode must be either 'equality' or 'lower-bound'.")
    objective_profile = _normalize_objective_profile(objective_profile)
    if area_scale <= 0.0:
        raise ValueError("area_scale must be positive.")
    if heavy_particle_mass_kg <= 0.0:
        raise ValueError("heavy_particle_mass_kg must be positive.")
    if seed_ionization_energy_J <= 0.0:
        raise ValueError("seed_ionization_energy_J must be positive.")
    if sigma_ep <= 0.0:
        raise ValueError("sigma_ep must be positive.")
    if feasibility_thresholds is None:
        feasibility_thresholds = FeasibilityThresholds()

    x_nodes = np.linspace(0.0, float(length), int(n_intervals) + 1, dtype=float)
    dx = float(length) / int(n_intervals)
    stage = _make_stage_function(
        B=float(B),
        heavy_particle_mass_kg=float(heavy_particle_mass_kg),
        seed_ionization_energy_J=float(seed_ionization_energy_J),
        sigma_ep=float(sigma_ep),
    )

    if warm_profile is not None:
        warm = _resample_warm_profile(warm_profile, x_nodes)
    else:
        marginal_warm = _build_global_marginal_warm_start(
            x=x_nodes,
            n_p_in_guess=float(n_p_in_guess),
            T_e_in_guess=float(T_e_in_guess),
            Z_in_guess=float(Z_in_guess),
            Z_in_min=float(Z_in_min),
            Z_in_max=float(Z_in_max),
            I_0_guess=float(J_x_in_guess),
            seed_fraction_guess=float(seed_fraction_guess),
            B=float(B),
            length=float(length),
            area_scale=float(area_scale),
            heavy_particle_mass_kg=float(heavy_particle_mass_kg),
            seed_ionization_energy_J=float(seed_ionization_energy_J),
            sigma_ep=float(sigma_ep),
        )
        integrated_warm = _build_integrated_warm_start(
            x=x_nodes,
            n_p_in_guess=float(n_p_in_guess),
            T_e_in_guess=float(T_e_in_guess),
            Z_in_guess=float(Z_in_guess),
            Z_in_min=float(Z_in_min),
            Z_in_max=float(Z_in_max),
            I_0_guess=float(J_x_in_guess),
            seed_fraction_guess=float(seed_fraction_guess),
            B=float(B),
            area_scale=float(area_scale),
            heavy_particle_mass_kg=float(heavy_particle_mass_kg),
            seed_ionization_energy_J=float(seed_ionization_energy_J),
            sigma_ep=float(sigma_ep),
        )
        warm = (
            marginal_warm
            if marginal_warm is not None
            else integrated_warm
            if integrated_warm is not None
            else _build_constant_warm_start(
                x=x_nodes,
                n_p_in_guess=float(n_p_in_guess),
                T_e_in_guess=float(T_e_in_guess),
                Z_in_guess=float(Z_in_guess),
                I_0_guess=float(J_x_in_guess),
                seed_fraction_guess=float(seed_fraction_guess),
                area_scale=float(area_scale),
            )
        )

    np_scale = float(max(n_p_in_guess, 1e10))
    te_scale = float(max(T_e_in_guess, te_min))
    A_scale = float(area_scale)
    A_floor = float(A_min_ratio) * float(area_scale)
    A_ceil = float(A_max_ratio) * float(area_scale)
    global_np_floor = float(np_min_ratio) * float(n_p_in_min)
    global_np_ceil = float(np_max_ratio) * float(n_p_in_max)
    global_te_ceil = float(te_max_ratio) * float(T_e_in_max)
    warm = _project_warm_profile_to_bounds(
        warm,
        n_p_floor=global_np_floor,
        n_p_ceil=global_np_ceil,
        T_e_floor=float(te_min),
        T_e_ceil=global_te_ceil,
        A_floor=A_floor,
        A_ceil=A_ceil,
        sigma_floor=-float(max_abs_dlogA_dx),
        sigma_ceil=float(max_abs_dlogA_dx),
        inlet_n_p_bounds=(float(n_p_in_min), float(n_p_in_max)),
        inlet_T_e_bounds=(float(T_e_in_min), float(T_e_in_max)),
        inlet_Z_bounds=(float(Z_in_min), float(Z_in_max)),
        inlet_I0_bounds=(float(J_x_in_min), float(J_x_in_max)),
        inlet_seed_fraction_bounds=(float(seed_fraction_min), float(seed_fraction_max)),
        area_scale=float(area_scale),
    )

    opti = ca.Opti()
    X = opti.variable(3, n_intervals + 1)
    U = opti.variable(1, n_intervals)
    use_margin_slack = bool(margin_slack_max > 0.0)
    S_node = opti.variable(1, n_intervals + 1) if use_margin_slack else None
    S_mid = opti.variable(1, n_intervals) if use_margin_slack else None
    dual_handles: dict[str, list[ca.MX]] = {
        "A_lower_node": [],
        "A_upper_node": [],
        "sigma_lower_interval": [],
        "sigma_upper_interval": [],
        "Tp_lower_node": [],
        "Tp_lower_mid": [],
        "G_lower_node": [],
        "G_lower_mid": [],
        "Mach_lower_node": [],
        "Mach_lower_mid": [],
        "Mach_upper_node": [],
        "Mach_upper_mid": [],
        "Inlet_Tp_lower": [],
        "Inlet_G_equal_zero": [],
        "Inlet_G_lower_bound": [],
        "Inlet_Mach_lower": [],
        "Inlet_Mach_upper": [],
    }

    def _subject_to(name: str | None, expr):
        opti.subject_to(expr)
        if name is not None:
            dual_handles[name].append(expr)

    inlet_n_p_scale = float(max(abs(n_p_in_guess), abs(n_p_in_min), abs(n_p_in_max), 1.0))
    inlet_T_e_scale = float(max(abs(T_e_in_guess), abs(T_e_in_min), abs(T_e_in_max), 1.0))
    inlet_Z_scale = float(max(abs(Z_in_guess), abs(Z_in_min), abs(Z_in_max), 1.0))
    inlet_I0_scale = float(max(abs(J_x_in_guess), abs(J_x_in_min), abs(J_x_in_max), 1.0))
    inlet_n_p_hat_var = opti.variable()
    inlet_T_e_hat_var = opti.variable()
    inlet_Z_hat_var = opti.variable()
    inlet_I0_hat_var = opti.variable()
    inlet_log_seed_var = opti.variable()

    inlet_n_p = inlet_n_p_scale * inlet_n_p_hat_var
    inlet_T_e = inlet_T_e_scale * inlet_T_e_hat_var
    inlet_Z = inlet_Z_scale * inlet_Z_hat_var
    inlet_I0 = inlet_I0_scale * inlet_I0_hat_var
    inlet_seed_fraction = ca.exp(inlet_log_seed_var)

    n_p_hat = X[0, :]
    T_e_hat = X[1, :]
    A_hat = X[2, :]

    _subject_to(
        None,
        opti.bounded(float(n_p_in_min) / inlet_n_p_scale, inlet_n_p_hat_var, float(n_p_in_max) / inlet_n_p_scale)
    )
    _subject_to(
        None,
        opti.bounded(float(T_e_in_min) / inlet_T_e_scale, inlet_T_e_hat_var, float(T_e_in_max) / inlet_T_e_scale)
    )
    _subject_to(
        None,
        opti.bounded(float(Z_in_min) / inlet_Z_scale, inlet_Z_hat_var, float(Z_in_max) / inlet_Z_scale)
    )
    _subject_to(
        None,
        opti.bounded(float(J_x_in_min) / inlet_I0_scale, inlet_I0_hat_var, float(J_x_in_max) / inlet_I0_scale)
    )
    _subject_to(
        None,
        opti.bounded(math.log(float(seed_fraction_min)), inlet_log_seed_var, math.log(float(seed_fraction_max)))
    )
    _subject_to(None, X[:, 0] == ca.vertcat(inlet_n_p / np_scale, inlet_T_e / te_scale, 1.0))
    _subject_to(None, n_p_hat >= float(np_min_ratio) * inlet_n_p / np_scale)
    _subject_to(None, n_p_hat <= float(np_max_ratio) * inlet_n_p / np_scale)
    _subject_to(None, T_e_hat >= float(te_min) / te_scale)
    _subject_to(None, T_e_hat <= float(te_max_ratio) * inlet_T_e / te_scale)
    _subject_to("A_lower_node", A_hat >= A_floor / A_scale)
    _subject_to("A_upper_node", A_hat <= A_ceil / A_scale)
    _subject_to("sigma_lower_interval", U >= -float(max_abs_dlogA_dx))
    _subject_to("sigma_upper_interval", U <= float(max_abs_dlogA_dx))
    if use_margin_slack:
        _subject_to(None, opti.bounded(0.0, S_node, float(margin_slack_max)))
        _subject_to(None, opti.bounded(0.0, S_mid, float(margin_slack_max)))

    beta_in = _beta_symbolic(inlet_n_p, inlet_T_e, B=float(B), sigma_ep=float(sigma_ep))
    n_e_in = _saha_n_e_symbolic(
        inlet_n_p,
        inlet_T_e,
        inlet_seed_fraction,
        seed_ionization_energy_J=float(seed_ionization_energy_J),
    )
    b2_in = beta_in * beta_in
    one_plus_z_in = 1.0 + inlet_Z
    den_in = b2_in + one_plus_z_in
    v_in = inlet_I0 * den_in / (b2_in * E_CHARGE * n_e_in + _EPS)
    dot_N = inlet_n_p * v_in * float(area_scale)
    F_in = b2_in * (b2_in + one_plus_z_in * one_plus_z_in) / (den_in * den_in + _EPS)
    T_p_in = inlet_T_e - float(heavy_particle_mass_kg) * v_in * v_in * F_in / (3.0 * K_B)
    c_s_in = ca.sqrt(
        (5.0 / 3.0) * K_B * ca.fmax(T_p_in, _TP_MIN) / float(heavy_particle_mass_kg) + _EPS
    )
    mach_in = v_in / c_s_in
    seed_density_in = ca.fmax(inlet_seed_fraction * inlet_n_p, 1e-30)
    f_I_in = ca.fmin(ca.fmax(n_e_in / seed_density_in, _FION_MIN), _FION_MAX)
    delta_in = ca.fmax(inlet_T_e / ca.fmax(T_p_in, _TP_MIN) - 1.0, _DELTA_MIN)
    alpha_in = (K_B * inlet_T_e / (2.0 * float(seed_ionization_energy_J))) * (2.0 - f_I_in) / (
        1.0 - f_I_in + _EPS
    )
    G_in = 4.0 * alpha_in * (2.0 + 1.0 / delta_in) * (1.0 + alpha_in * (1.0 + 1.0 / delta_in)) - b2_in

    _subject_to("Inlet_Tp_lower", T_p_in >= float(tp_min))
    if inlet_margin_mode == "equality":
        _subject_to("Inlet_G_equal_zero", G_in == float(min_margin))
    else:
        _subject_to("Inlet_G_lower_bound", G_in >= float(min_margin))
    if mach_min is not None:
        _subject_to("Inlet_Mach_lower", mach_in >= float(mach_min))
    if mach_max is not None:
        _subject_to("Inlet_Mach_upper", mach_in <= float(mach_max))

    params = ca.vertcat(dot_N, inlet_I0, inlet_seed_fraction)

    objective = 0.0
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
        objective += float(warm_profile_track_weight) * (
            (inlet_n_p - float(warm.inlet_n_p)) ** 2 / max(np_scale * np_scale, 1.0)
            + (inlet_T_e - float(warm.inlet_T_e)) ** 2 / max(te_scale * te_scale, 1.0)
            + (inlet_Z - float(warm.inlet_Z)) ** 2 / max(float(Z_in_guess) * float(Z_in_guess), 1.0)
            + (inlet_I0 - float(warm.inlet_I0)) ** 2 / max(float(J_x_in_guess) * float(J_x_in_guess), 1.0)
            + (inlet_log_seed_var - math.log(float(warm.inlet_seed_fraction))) ** 2
        )
    if warm_control_track_weight > 0.0:
        warm_u = ca.DM(warm.sigma_logA.reshape(1, -1))
        objective += float(warm_control_track_weight) * dx * ca.sumsqr(U - warm_u)

    node_outputs = []
    for k in range(n_intervals):
        xk_phys = ca.vertcat(np_scale * n_p_hat[k], te_scale * T_e_hat[k], A_scale * A_hat[k])
        xkp1_phys = ca.vertcat(
            np_scale * n_p_hat[k + 1],
            te_scale * T_e_hat[k + 1],
            A_scale * A_hat[k + 1],
        )
        out_k = stage(xk_phys, U[0, k], params)
        out_kp1 = stage(xkp1_phys, U[0, k], params)
        node_outputs.append(out_k)
        f_k = ca.vertcat(out_k[0] / np_scale, out_k[1] / te_scale, out_k[2] / A_scale)
        f_kp1 = ca.vertcat(out_kp1[0] / np_scale, out_kp1[1] / te_scale, out_kp1[2] / A_scale)
        if transcription == "trapezoid":
            _subject_to(None, X[:, k + 1] == X[:, k] + 0.5 * dx * (f_k + f_kp1))
            mid_state = 0.5 * (xk_phys + xkp1_phys)
        else:
            mid_state = 0.5 * (xk_phys + xkp1_phys) + 0.125 * dx * ca.vertcat(
                out_k[0] - out_kp1[0],
                out_k[1] - out_kp1[1],
                out_k[2] - out_kp1[2],
            )
            out_mid_hs = stage(mid_state, U[0, k], params)
            f_mid = ca.vertcat(out_mid_hs[0] / np_scale, out_mid_hs[1] / te_scale, out_mid_hs[2] / A_scale)
            _subject_to(None, X[:, k + 1] == X[:, k] + dx / 6.0 * (f_k + 4.0 * f_mid + f_kp1))

        _subject_to("Tp_lower_node", out_k[3] >= float(tp_min))
        if use_margin_slack:
            _subject_to("G_lower_node", out_k[13] + S_node[0, k] >= float(min_margin))
        else:
            _subject_to("G_lower_node", out_k[13] >= float(min_margin))
        if mach_min is not None:
            _subject_to("Mach_lower_node", out_k[12] >= float(mach_min))
        if mach_max is not None:
            _subject_to("Mach_upper_node", out_k[12] <= float(mach_max))

        out_mid = stage(mid_state, U[0, k], params)
        _subject_to("Tp_lower_mid", out_mid[3] >= float(tp_min))
        if use_margin_slack:
            _subject_to("G_lower_mid", out_mid[13] + S_mid[0, k] >= float(min_margin))
        else:
            _subject_to("G_lower_mid", out_mid[13] >= float(min_margin))
        if mach_min is not None:
            _subject_to("Mach_lower_mid", out_mid[12] >= float(mach_min))
        if mach_max is not None:
            _subject_to("Mach_upper_mid", out_mid[12] <= float(mach_max))

    out_end = stage(
        ca.vertcat(np_scale * n_p_hat[-1], te_scale * T_e_hat[-1], A_scale * A_hat[-1]),
        U[0, -1],
        params,
    )
    node_outputs.append(out_end)
    _subject_to("Tp_lower_node", out_end[3] >= float(tp_min))
    if use_margin_slack:
        _subject_to("G_lower_node", out_end[13] + S_node[0, -1] >= float(min_margin))
    else:
        _subject_to("G_lower_node", out_end[13] >= float(min_margin))
    if mach_min is not None:
        _subject_to("Mach_lower_node", out_end[12] >= float(mach_min))
    if mach_max is not None:
        _subject_to("Mach_upper_node", out_end[12] <= float(mach_max))

    outlet_T_e = te_scale * T_e_hat[-1]
    outlet_T_p = out_end[3]
    outlet_n_p = np_scale * n_p_hat[-1]
    outlet_n_e = out_end[5]
    power_density_samples = []
    for j, out_j in enumerate(node_outputs):
        A_j = A_scale * A_hat[j]
        power_density_samples.append(-A_j * out_j[9] * out_j[11] / 1e8)
    outlet_mhd_output_per_100MWe = 0.0
    for j in range(n_intervals):
        outlet_mhd_output_per_100MWe += 0.5 * dx * (power_density_samples[j] + power_density_samples[j + 1])
    if objective_profile == OBJECTIVE_PROFILE_ENTHALPY_EXTRACTION:
        inlet_thermal_density = 2.5 * K_B * (inlet_n_p * T_p_in + n_e_in * inlet_T_e)
        inlet_kinetic_density = 0.5 * float(heavy_particle_mass_kg) * inlet_n_p * v_in * v_in
        inlet_enthalpy_flux_W = float(area_scale) * v_in * (inlet_thermal_density + inlet_kinetic_density)
        design_score = 100.0 * (outlet_mhd_output_per_100MWe * 1e8) / ca.fmax(inlet_enthalpy_flux_W, _EPS)
    else:
        weights = _design_value_weights_lab_poc_v2_objective()
        outlet_delta_te_per_kK = (outlet_T_e - inlet_T_e) / 1e3
        outlet_delta_ratio = outlet_T_e / ca.fmax(outlet_T_p, _TP_MIN) - 1.0
        outlet_f_ion = outlet_n_e / ca.fmax(inlet_seed_fraction * outlet_n_p, 1e-30)
        inlet_delta_ratio = inlet_T_e / ca.fmax(T_p_in, _TP_MIN) - 1.0
        device_length_per_5m = float(length) / 5.0
        design_score = (
            float(weights.outlet_delta_te_per_kK) * outlet_delta_te_per_kK
            + float(weights.outlet_delta_ratio) * outlet_delta_ratio
            + float(weights.outlet_f_ion) * outlet_f_ion
            + float(weights.outlet_mhd_output_per_100MWe) * outlet_mhd_output_per_100MWe
            - float(weights.inlet_delta_ratio_penalty) * inlet_delta_ratio
            - float(weights.inlet_mach_penalty) * mach_in
            - float(weights.magnetic_field_T_penalty) * abs(float(B))
            - float(weights.device_length_per_5m_penalty) * device_length_per_5m
        )
    objective += -float(objective_weight) * design_score

    opti.minimize(objective)

    opti.set_initial(X[0, :], warm.n_p / np_scale)
    opti.set_initial(X[1, :], warm.T_e / te_scale)
    opti.set_initial(X[2, :], warm.A / A_scale)
    opti.set_initial(U, warm.sigma_logA.reshape(1, -1))
    opti.set_initial(inlet_n_p_hat_var, float(warm.inlet_n_p) / inlet_n_p_scale)
    opti.set_initial(inlet_T_e_hat_var, float(warm.inlet_T_e) / inlet_T_e_scale)
    opti.set_initial(inlet_Z_hat_var, float(warm.inlet_Z) / inlet_Z_scale)
    opti.set_initial(inlet_I0_hat_var, float(warm.inlet_I0) / inlet_I0_scale)
    opti.set_initial(inlet_log_seed_var, math.log(float(warm.inlet_seed_fraction)))
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

    value_fn = None
    try:
        opti.solve_limited()
        value_fn = opti.value
    except RuntimeError as exc:
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
    inlet_n_p_sol = float(value_fn(inlet_n_p))
    inlet_T_e_sol = float(value_fn(inlet_T_e))
    inlet_Z_sol = float(value_fn(inlet_Z))
    inlet_I0_sol = float(value_fn(inlet_I0))
    inlet_seed_fraction_sol = float(value_fn(inlet_seed_fraction))
    inlet = _evaluate_inlet_design_numeric(
        n_p_in=inlet_n_p_sol,
        T_e_in=inlet_T_e_sol,
        Z_in=inlet_Z_sol,
        I_0=inlet_I0_sol,
        seed_fraction=inlet_seed_fraction_sol,
        B=float(B),
        area_scale=float(area_scale),
        heavy_particle_mass_kg=float(heavy_particle_mass_kg),
        seed_ionization_energy_J=float(seed_ionization_energy_J),
        sigma_ep=float(sigma_ep),
    )

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
        stage_fun=stage,
        heavy_particle_mass_kg=float(heavy_particle_mass_kg),
        seed_ionization_energy_J=float(seed_ionization_energy_J),
        sigma_ep=float(sigma_ep),
    )
    stage_params_numeric = np.array([inlet.dot_N, inlet.I_0, inlet.seed_fraction], dtype=float)
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
        stage_params=stage_params_numeric,
        inlet_target=(float(inlet.n_p), float(inlet.T_e), float(area_scale)),
        state_bounds={
            "np_floor": float(np_min_ratio) * float(inlet.n_p),
            "np_ceil": float(np_max_ratio) * float(inlet.n_p),
            "te_floor": float(te_min),
            "te_ceil": float(te_max_ratio) * float(inlet.T_e),
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
    inlet_margin_violation = (
        abs(float(inlet.velikhov_margin) - float(min_margin))
        if inlet_margin_mode == "equality"
        else max(0.0, float(min_margin) - float(inlet.velikhov_margin))
    )
    diagnostics["inlet_margin_mode"] = str(inlet_margin_mode)
    diagnostics["inlet_velikhov_target"] = float(min_margin)
    diagnostics["inlet_velikhov_violation"] = float(inlet_margin_violation)
    diagnostics["inlet_velikhov_equality_abs"] = abs(float(inlet.velikhov_margin) - float(min_margin))
    diagnostics["objective_profile"] = objective_profile
    diagnostics["area_scale_m2"] = float(area_scale)
    diagnostics["heavy_particle_mass_kg"] = float(heavy_particle_mass_kg)
    diagnostics["seed_ionization_energy_J"] = float(seed_ionization_energy_J)
    diagnostics["electron_particle_sigma_m2"] = float(sigma_ep)
    diagnostics["max_constraint_violation"] = max(
        float(diagnostics["max_constraint_violation"]),
        float(inlet_margin_violation),
    )
    diagnostics["acceptable"] = bool(
        bool(diagnostics["finite_profile"])
        and float(diagnostics["dynamic_defect_inf"]) <= float(feasibility_thresholds.defect_inf_tol)
        and float(diagnostics["dynamic_defect_rms"]) <= float(feasibility_thresholds.defect_rms_tol)
        and float(diagnostics["midpoint_defect_inf"]) <= float(feasibility_thresholds.defect_inf_tol)
        and float(diagnostics["midpoint_defect_rms"]) <= float(feasibility_thresholds.defect_rms_tol)
        and float(diagnostics["boundary_residual_inf"]) <= float(feasibility_thresholds.boundary_inf_tol)
        and float(diagnostics["max_constraint_violation"]) <= float(feasibility_thresholds.path_slack_tol)
    )
    diagnostics["inlet_design"] = {
        "n_p_in": float(inlet.n_p),
        "T_e_in": float(inlet.T_e),
        "T_p_in": float(inlet.T_p),
        "Z_in": float(inlet.Z),
        "J_x_in": float(inlet.J_x),
        "I_0": float(inlet.I_0),
        "seed_fraction": float(inlet.seed_fraction),
        "v_in": float(inlet.v_in),
        "dot_N": float(inlet.dot_N),
        "mach_in": float(inlet.mach),
        "velikhov_margin_in": float(inlet.velikhov_margin),
        "A_in": float(area_scale),
        "objective_profile": objective_profile,
        "area_scale_m2": float(area_scale),
        "heavy_particle_mass_kg": float(heavy_particle_mass_kg),
        "seed_ionization_energy_J": float(seed_ionization_energy_J),
        "electron_particle_sigma_m2": float(sigma_ep),
    }
    objective_value = float(value_fn(objective))
    dual_status = "converged" if bool(stats.get("success", False)) else "debug_last_iterate"

    def _value_dual_array(handles: list[ca.MX]) -> np.ndarray:
        if not handles:
            return np.zeros(0, dtype=float)
        exprs = []
        for handle in handles:
            dual_expr = opti.dual(handle)
            exprs.append(ca.reshape(dual_expr, dual_expr.numel(), 1))
        return np.asarray(value_fn(ca.vertcat(*exprs)), dtype=float).reshape(-1)

    dual_arrays: dict[str, np.ndarray] = {}
    dual_errors: dict[str, str] = {}
    for name, handles in dual_handles.items():
        try:
            dual_arrays[name] = _value_dual_array(handles)
        except Exception as exc:
            dual_arrays[name] = np.zeros(0, dtype=float)
            dual_errors[name] = str(exc)

    def _summarize_dual(values: np.ndarray) -> dict[str, object]:
        arr = np.asarray(values, dtype=float).reshape(-1)
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            return {
                "size": int(arr.size),
                "finite_count": 0,
                "max_abs": float("nan"),
                "max": float("nan"),
                "l1": float("nan"),
                "active_fraction_gt_1e_8": float("nan"),
            }
        max_abs_idx = int(np.nanargmax(np.abs(arr)))
        return {
            "size": int(arr.size),
            "finite_count": int(finite.size),
            "max_abs": float(np.nanmax(np.abs(arr))),
            "max_abs_index": max_abs_idx,
            "max": float(np.nanmax(arr)),
            "l1": float(np.nansum(np.abs(arr))),
            "active_fraction_gt_1e_8": float(np.mean(np.abs(finite) > 1e-8)),
        }

    duals = {
        "status": dual_status,
        "sign_convention": (
            "Values are from CasADi opti.dual(saved_constraint). For single-sided "
            "bounds, positive values indicate active KKT pressure on that named bound."
        ),
        "arrays": dual_arrays,
        "summary": {name: _summarize_dual(values) for name, values in dual_arrays.items()},
        "errors": dual_errors,
    }

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
        duals=duals,
    )


def _enthalpy_extraction_value_profile(value_terms) -> dict[str, object]:
    score = float(value_terms.outlet_enthalpy_extraction_percent)
    return {
        "profile_name": "enthalpy_extraction_objective",
        "total_score": score,
        "score_units": "percent_of_inlet_stagnation_enthalpy_flux",
        "contributions": {"reward_outlet_enthalpy_extraction_percent": score},
        "terms": value_terms.to_dict(),
        "weights": {"outlet_enthalpy_extraction_percent": 1.0},
    }


def _payload_from_result(
    result: OptimizedAreaProfile,
    B: float,
    *,
    objective_profile: str = OBJECTIVE_PROFILE_LAB_POC_V2,
    heavy_particle_mass_kg: float = M_P,
    seed_ionization_energy_J: float = E_I,
    sigma_ep: float = SIGMA_EP,
    area_scale: float = _A_IN,
) -> dict[str, object]:
    objective_profile = _normalize_objective_profile(objective_profile)
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
        v_p=result.v_p,
        heavy_particle_mass_kg=float(heavy_particle_mass_kg),
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
        "lab_poc_v2_objective": evaluate_design_value(
            value_terms,
            weights=_design_value_weights_lab_poc_v2_objective(),
            profile_name="lab_poc_v2_objective",
        ).to_dict(),
        "enthalpy_extraction_objective": _enthalpy_extraction_value_profile(value_terms),
    }
    return {
        "ok": bool(result.acceptable),
        "solver_success": bool(result.success),
        "acceptable": bool(result.acceptable),
        "solver": f"casadi-ipopt-{result.transcription}",
        "return_status": result.return_status,
        "objective": f"maximize_{objective_profile}",
        "objective_profile": objective_profile,
        "objective_delta_Te_K": float(result.objective_delta_Te),
        "nlp_objective_value": float(result.objective_value),
        "objective_score": float(value_profiles["enthalpy_extraction_objective"]["total_score"])
        if objective_profile == OBJECTIVE_PROFILE_ENTHALPY_EXTRACTION
        else float(value_profiles["lab_poc_v2_objective"]["total_score"]),
        "area_scale_m2": float(area_scale),
        "working_fluid_parameters": {
            "heavy_particle_mass_kg": float(heavy_particle_mass_kg),
            "seed_ionization_energy_J": float(seed_ionization_energy_J),
            "electron_particle_sigma_m2": float(sigma_ep),
        },
        "warm_start_source": result.warm_start_source,
        "inlet_design": {
            "n_p_in": float(result.inlet.n_p),
            "T_e_in": float(result.inlet.T_e),
            "T_p_in": float(result.inlet.T_p),
            "Z_in": float(result.inlet.Z),
            "J_x_in": float(result.inlet.J_x),
            "I_0": float(result.inlet.I_0),
            "dot_N": float(result.inlet.dot_N),
            "v_in": float(result.inlet.v_in),
            "seed_fraction": float(result.inlet.seed_fraction),
            "mach_in": float(result.inlet.mach),
            "velikhov_margin_in": float(result.inlet.velikhov_margin),
            "A_in": float(area_scale),
            "inlet_margin_mode": str(result.diagnostics.get("inlet_margin_mode", "")),
        },
        "min_velikhov_margin": float(np.nanmin(result.velikhov_margin)),
        "max_velikhov_margin": float(np.nanmax(result.velikhov_margin)),
        "min_mach": float(np.nanmin(result.mach)),
        "max_mach": float(np.nanmax(result.mach)),
        "metrics": metrics.to_dict(),
        "value_terms": value_terms.to_dict(),
        "value_profiles": value_profiles,
        "solver_stats": result.stats,
        "diagnostics": result.diagnostics,
        "dual_summary": result.duals.get("summary", {}),
        "dual_status": result.duals.get("status", ""),
        "dual_errors": result.duals.get("errors", {}),
    }


def main() -> int:
    args = _build_parser().parse_args()
    result = optimize_area_profile(
        n_p_in_guess=float(args.np_in_guess),
        n_p_in_min=float(args.np_in_min),
        n_p_in_max=float(args.np_in_max),
        T_e_in_guess=float(args.te_in_guess),
        T_e_in_min=float(args.te_in_min),
        T_e_in_max=float(args.te_in_max),
        Z_in_guess=float(args.z_in_guess),
        Z_in_min=float(args.z_in_min),
        Z_in_max=float(args.z_in_max),
        J_x_in_guess=float(args.jx_in_guess),
        J_x_in_min=float(args.jx_in_min),
        J_x_in_max=float(args.jx_in_max),
        seed_fraction_guess=float(args.seed_fraction_guess),
        seed_fraction_min=float(args.seed_fraction_min),
        seed_fraction_max=float(args.seed_fraction_max),
        inlet_margin_mode=str(args.inlet_margin_mode),
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
        ipopt_max_iter=int(args.ipopt_max_iter),
        ipopt_tol=float(args.ipopt_tol),
        objective_weight=float(args.objective_weight),
        objective_profile=str(args.objective_profile),
        area_scale=float(args.area_scale_m2),
        heavy_particle_mass_kg=float(args.heavy_particle_mass_kg),
        seed_ionization_energy_J=float(args.seed_ionization_energy_J),
        sigma_ep=float(args.electron_particle_sigma_m2),
    )
    payload = _payload_from_result(
        result,
        float(args.B),
        objective_profile=str(args.objective_profile),
        area_scale=float(args.area_scale_m2),
        heavy_particle_mass_kg=float(args.heavy_particle_mass_kg),
        seed_ionization_energy_J=float(args.seed_ionization_energy_J),
        sigma_ep=float(args.electron_particle_sigma_m2),
    )
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
            **{
                f"dual_{name}": np.asarray(values, dtype=float)
                for name, values in (result.duals.get("arrays", {}) or {}).items()
            },
        )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
