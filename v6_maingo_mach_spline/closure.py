from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from v6_core.local_algebraic_closure import E_CHARGE, H_P, K_B, M_E

from v6_maingo_freidberg_variables.models import FreidbergConfig, PrimitivePoint


_EPS = 1e-300
_T_MIN = 1.0
_FION_MIN = 1e-12
_FION_MAX = 1.0 - 1e-12
_DELTA_MIN = 1e-12
_SAHA_LOG_K_MIN = math.log(1e-100)
_SAHA_LOG_K_MAX = math.log(1e60)
_SAHA_PREFAC = 2.0 * math.pi * M_E * K_B / (H_P * H_P)


def _safe_pos(value: float, floor: float = _EPS) -> float:
    if not math.isfinite(float(value)):
        raise ValueError(f"expected finite positive value, got {value!r}")
    return float(value) if float(value) > floor else floor


def _clip(value: float, lower: float, upper: float) -> float:
    return min(max(float(value), float(lower)), float(upper))


def _saha_electron_density(n_p: float, T_e: float, seed_fraction: float, config: FreidbergConfig) -> float:
    n_p_safe = _safe_pos(n_p, 1.0)
    T_e_safe = _safe_pos(T_e, _T_MIN)
    seed_safe = _safe_pos(seed_fraction, 1e-12)
    n_s = seed_safe * n_p_safe
    log_K = 1.5 * math.log(max(_SAHA_PREFAC * T_e_safe, _EPS))
    log_K -= float(config.working_fluid.seed_ionization_energy_J) / (K_B * T_e_safe)
    K = math.exp(min(max(log_K, _SAHA_LOG_K_MIN), _SAHA_LOG_K_MAX))
    sqrt_term = math.sqrt(1.0 + 4.0 * n_s / max(K, _EPS))
    n_e = 2.0 * n_s / (1.0 + sqrt_term)
    return min(max(n_e, 0.0), n_s * (1.0 - 1e-12))


def mach_closure_from_state(
    *,
    n_p: float,
    T_e: float,
    mach: float,
    config: FreidbergConfig,
    x: float = 0.0,
) -> PrimitivePoint:
    """Recover primitive variables from n_p, T_e, and prescribed Mach.

    This is the key replacement for area-spline closure.  For fixed dot_N and
    I_0, the Hall/ionization factor F depends on n_p and T_e but not on A.
    Therefore T_p, v_p, and A are explicit functions of n_p, T_e, and Mach.
    """

    n_p_safe = _safe_pos(n_p, 1.0)
    T_e_safe = _safe_pos(T_e, _T_MIN)
    M = _safe_pos(mach, 1e-12)
    M2 = M * M
    fluid = config.working_fluid

    n_e = _saha_electron_density(n_p_safe, T_e_safe, config.seed_fraction, config)
    v_te = math.sqrt(2.0 * K_B * T_e_safe / M_E)
    beta = E_CHARGE * config.B_T / max(M_E * n_p_safe * fluid.sigma_ep * v_te, _EPS)
    eta = M_E * n_p_safe * fluid.sigma_ep * v_te / max(E_CHARGE * E_CHARGE * n_e, _EPS)
    q = E_CHARGE * config.dot_N * n_e / max(config.I_0 * n_p_safe, _EPS)
    b2 = beta * beta
    Z = b2 * (q - 1.0) - 1.0
    one_plus_z = 1.0 + Z
    den_q = max(b2 * q, _EPS)
    F = b2 * (b2 + one_plus_z * one_plus_z) / (den_q * den_q)

    T_p = 9.0 * T_e_safe / max(9.0 + 5.0 * M2 * F, _EPS)
    v_p = math.sqrt(max(5.0 * K_B * T_p * M2 / (3.0 * fluid.heavy_particle_mass_kg), 0.0))
    A = config.dot_N / max(n_p_safe * v_p, _EPS)

    den = b2 + one_plus_z
    if abs(den) < _EPS:
        den = _EPS if den >= 0.0 else -_EPS
    jfac = E_CHARGE * n_e * v_p
    J_x = config.I_0 / max(A, _EPS)
    J_y = -beta * one_plus_z * jfac / den
    E_x = -b2 * Z * eta * jfac / den

    T_p_floor = max(T_p, _T_MIN)
    f_I = _clip(n_e / max(config.seed_fraction * n_p_safe, _EPS), _FION_MIN, _FION_MAX)
    delta = max(T_e_safe / T_p_floor - 1.0, _DELTA_MIN)
    alpha = (K_B * T_e_safe / (2.0 * fluid.seed_ionization_energy_J)) * (2.0 - f_I) / max(1.0 - f_I, _EPS)
    velikhov_margin = 4.0 * alpha * (2.0 + 1.0 / delta) * (1.0 + alpha * (1.0 + 1.0 / delta)) - b2

    return PrimitivePoint(
        x=float(x),
        n_p=float(n_p_safe),
        T_e=float(T_e_safe),
        T_p=float(T_p),
        A=float(A),
        v_p=float(v_p),
        n_e=float(n_e),
        beta=float(beta),
        eta=float(eta),
        Z=float(Z),
        J_x=float(J_x),
        J_y=float(J_y),
        E_x=float(E_x),
        mach=float(M),
        velikhov_margin=float(velikhov_margin),
        seed_fraction=float(config.seed_fraction),
    )


def reconstruct_points_from_mach(
    *,
    x: np.ndarray,
    n_p: np.ndarray,
    T_e: np.ndarray,
    mach: np.ndarray,
    config: FreidbergConfig,
) -> list[PrimitivePoint]:
    x_arr = np.asarray(x, dtype=float)
    n_arr = np.asarray(n_p, dtype=float)
    te_arr = np.asarray(T_e, dtype=float)
    mach_arr = np.asarray(mach, dtype=float)
    if not (x_arr.shape == n_arr.shape == te_arr.shape == mach_arr.shape):
        raise ValueError("x, n_p, T_e, and mach arrays must have identical shape")
    return [
        mach_closure_from_state(
            x=float(x_arr[idx]),
            n_p=float(n_arr[idx]),
            T_e=float(te_arr[idx]),
            mach=float(mach_arr[idx]),
            config=config,
        )
        for idx in range(x_arr.size)
    ]


def reconstruct_points_from_profile(profile_path: str | Path, config: FreidbergConfig) -> list[PrimitivePoint]:
    with np.load(profile_path) as data:
        return reconstruct_points_from_mach(
            x=np.asarray(data["x"], dtype=float),
            n_p=np.asarray(data["n_p"], dtype=float),
            T_e=np.asarray(data["T_e"], dtype=float),
            mach=np.asarray(data["mach"], dtype=float),
            config=config,
        )


def primitive_arrays(points: list[PrimitivePoint]) -> dict[str, np.ndarray]:
    fields = ("x", "n_p", "T_e", "T_p", "A", "v_p", "n_e", "beta", "eta", "Z", "J_x", "J_y", "E_x", "mach")
    return {field: np.asarray([getattr(point, field) for point in points], dtype=float) for field in fields}
