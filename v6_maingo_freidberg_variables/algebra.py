from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from v6_core.local_algebraic_closure import E_CHARGE, H_P, K_B, M_E

from .models import FreidbergConfig, FreidbergState, PrimitivePoint


_EPS = 1e-300
_T_MIN = 1.0
_SAHA_LOG_K_MIN = math.log(1e-100)
_SAHA_LOG_K_MAX = math.log(1e60)
_SAHA_PREFAC = 2.0 * math.pi * M_E * K_B / (H_P * H_P)


@dataclass(frozen=True)
class ReconstructionDiagnostics:
    max_abs_closure_residual_K: float
    max_abs_mach_error: float
    max_abs_T_p_error_K: float
    max_rel_A_error: float
    max_rel_n_p_error: float
    max_rel_v_p_error: float
    max_abs_Z_error: float

    def to_dict(self) -> dict[str, float]:
        return {
            "max_abs_closure_residual_K": float(self.max_abs_closure_residual_K),
            "max_abs_mach_error": float(self.max_abs_mach_error),
            "max_abs_T_p_error_K": float(self.max_abs_T_p_error_K),
            "max_rel_A_error": float(self.max_rel_A_error),
            "max_rel_n_p_error": float(self.max_rel_n_p_error),
            "max_rel_v_p_error": float(self.max_rel_v_p_error),
            "max_abs_Z_error": float(self.max_abs_Z_error),
        }


def _safe_pos(value: float, floor: float = _EPS) -> float:
    if not math.isfinite(float(value)):
        raise ValueError(f"expected finite positive value, got {value!r}")
    return float(value) if float(value) > floor else floor


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


def _transport_terms(n_p: float, T_e: float, v_p: float, config: FreidbergConfig) -> tuple[float, float, float, float, float]:
    fluid = config.working_fluid
    n_p_safe = _safe_pos(n_p, 1.0)
    T_e_safe = _safe_pos(T_e, _T_MIN)
    n_e = _saha_electron_density(n_p_safe, T_e_safe, config.seed_fraction, config)
    v_te = math.sqrt(2.0 * K_B * T_e_safe / M_E)
    beta = E_CHARGE * config.B_T / max(M_E * n_p_safe * fluid.sigma_ep * v_te, _EPS)
    eta = M_E * n_p_safe * fluid.sigma_ep * v_te / max(E_CHARGE * E_CHARGE * n_e, _EPS)
    q = E_CHARGE * config.dot_N * n_e / max(config.I_0 * n_p_safe, _EPS)
    b2 = beta * beta
    Z = b2 * (q - 1.0) - 1.0
    one_plus_z = 1.0 + Z
    den = max(b2 * q, _EPS)
    F = b2 * (b2 + one_plus_z * one_plus_z) / (den * den)
    T_p_closure = T_e_safe - fluid.heavy_particle_mass_kg * v_p * v_p * F / (3.0 * K_B)
    return n_e, beta, eta, Z, T_p_closure


def primitive_to_freidberg(point: PrimitivePoint, config: FreidbergConfig) -> FreidbergState:
    fluid = config.working_fluid
    A0 = config.inlet_area_m2
    M2 = 3.0 * point.v_p * point.v_p / max(5.0 * K_B * point.T_p / fluid.heavy_particle_mass_kg, _EPS)
    M = math.sqrt(max(M2, 0.0))
    H_p = (point.A * point.n_p * point.v_p / A0) * (
        2.5 * K_B * point.T_p + 0.5 * fluid.heavy_particle_mass_kg * point.v_p * point.v_p
    )
    L_p = M * (point.A / A0) / max((M2 + 3.0) ** 2, _EPS)
    return FreidbergState(H_p=float(H_p), L_p=float(L_p), T_e=float(point.T_e), x=float(point.x))


def _primitive_from_mach(
    *,
    H_p: float,
    L_p: float,
    T_e: float,
    mach: float,
    x: float,
    config: FreidbergConfig,
) -> tuple[PrimitivePoint, float]:
    M = _safe_pos(mach, 1e-12)
    M2 = M * M
    fluid = config.working_fluid
    A0 = config.inlet_area_m2
    denom = max(5.0 * K_B * config.dot_N * (M2 + 3.0), _EPS)
    T_p_from_H = 6.0 * H_p * A0 / denom
    v_p = math.sqrt(max(5.0 * K_B * T_p_from_H * M2 / (3.0 * fluid.heavy_particle_mass_kg), 0.0))
    A = A0 * L_p * (M2 + 3.0) ** 2 / M
    n_p = config.dot_N / max(A * v_p, _EPS)
    n_e, beta, eta, Z, T_p_closure = _transport_terms(n_p, T_e, v_p, config)
    residual = T_p_from_H - T_p_closure
    b2 = beta * beta
    den = b2 + 1.0 + Z
    if abs(den) < _EPS:
        den = _EPS if den >= 0.0 else -_EPS
    jfac = E_CHARGE * n_e * v_p
    J_x = config.I_0 / max(A, _EPS)
    J_y = -beta * (1.0 + Z) * jfac / den
    E_x = -b2 * Z * eta * jfac / den
    return (
        PrimitivePoint(
            x=float(x),
            n_p=float(n_p),
            T_e=float(T_e),
            T_p=float(T_p_from_H),
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
            seed_fraction=float(config.seed_fraction),
        ),
        float(residual),
    )


def _branch_bounds(branch: str) -> tuple[float, float]:
    if branch == "subsonic":
        return 1e-5, 1.0 - 1e-10
    if branch == "supersonic":
        return 1.0 + 1e-10, 50.0
    if branch == "any":
        return 1e-5, 50.0
    raise ValueError(f"unknown branch {branch!r}; expected 'subsonic', 'supersonic', or 'any'")


def _bisect_root(fn, lower: float, upper: float, *, iterations: int = 90) -> float:
    f_lower = fn(lower)
    f_upper = fn(upper)
    if not math.isfinite(f_lower) or not math.isfinite(f_upper) or f_lower * f_upper > 0.0:
        raise ValueError("invalid bisection bracket")
    lo = float(lower)
    hi = float(upper)
    flo = float(f_lower)
    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        fmid = float(fn(mid))
        if not math.isfinite(fmid):
            break
        if abs(fmid) <= 1e-10:
            return mid
        if flo * fmid <= 0.0:
            hi = mid
        else:
            lo = mid
            flo = fmid
    return 0.5 * (lo + hi)


def _refine_min_abs(fn, center: float, lower: float, upper: float) -> float:
    lo = max(float(lower), float(center) / 1.8)
    hi = min(float(upper), float(center) * 1.8)
    if not lo < hi:
        return float(center)
    log_lo = math.log(lo)
    log_hi = math.log(hi)
    gr = (math.sqrt(5.0) - 1.0) / 2.0
    a = log_lo
    b = log_hi
    c = b - gr * (b - a)
    d = a + gr * (b - a)

    def score(log_m: float) -> float:
        val = fn(math.exp(log_m))
        if not math.isfinite(val):
            return float("inf")
        return val * val

    fc = score(c)
    fd = score(d)
    for _ in range(100):
        if fc < fd:
            b = d
            d = c
            fd = fc
            c = b - gr * (b - a)
            fc = score(c)
        else:
            a = c
            c = d
            fc = fd
            d = a + gr * (b - a)
            fd = score(d)
    return math.exp(0.5 * (a + b))


def solve_primitive_from_hlt(
    *,
    H_p: float,
    L_p: float,
    T_e: float,
    config: FreidbergConfig,
    x: float = 0.0,
    branch: str = "any",
    mach_hint: float | None = None,
    tolerance_K: float = 1e-5,
) -> tuple[PrimitivePoint, float]:
    lower, upper = _branch_bounds(branch)
    if H_p <= 0.0 or L_p <= 0.0 or T_e <= 0.0:
        raise ValueError("H_p, L_p, and T_e must be positive")

    def residual(mach: float) -> float:
        _, r = _primitive_from_mach(H_p=H_p, L_p=L_p, T_e=T_e, mach=mach, x=x, config=config)
        return r

    candidates: list[float] = []
    if mach_hint is not None and math.isfinite(float(mach_hint)) and float(mach_hint) > 0.0:
        hint = min(max(float(mach_hint), lower), upper)
        candidates.append(hint)
        candidates.append(_refine_min_abs(residual, hint, lower, upper))
        lo = max(lower, hint / 2.0)
        hi = min(upper, hint * 2.0)
        try:
            if lo < hi and residual(lo) * residual(hi) <= 0.0:
                candidates.append(_bisect_root(residual, lo, hi))
        except ValueError:
            pass

    grid_lower = math.log(lower)
    grid_upper = math.log(upper)
    grid = np.exp(np.linspace(grid_lower, grid_upper, 180))
    vals = np.asarray([residual(float(m)) for m in grid], dtype=float)
    finite = np.isfinite(vals)
    if np.any(finite):
        best_idx = int(np.nanargmin(np.where(finite, np.abs(vals), np.inf)))
        candidates.append(float(grid[best_idx]))
        candidates.append(_refine_min_abs(residual, float(grid[best_idx]), lower, upper))
    for i in range(grid.size - 1):
        if not (finite[i] and finite[i + 1]):
            continue
        if vals[i] == 0.0:
            candidates.append(float(grid[i]))
        elif vals[i] * vals[i + 1] < 0.0:
            try:
                candidates.append(_bisect_root(residual, float(grid[i]), float(grid[i + 1])))
            except ValueError:
                pass

    if not candidates:
        raise ValueError("could not find any finite Mach candidate for H/L/T_e closure")

    best_mach = None
    best_abs = float("inf")
    best_residual = float("inf")
    for mach in candidates:
        if not math.isfinite(mach) or mach <= 0.0:
            continue
        try:
            value = residual(float(mach))
        except (ValueError, OverflowError):
            continue
        if not math.isfinite(value):
            continue
        abs_value = abs(value)
        if abs_value < best_abs:
            best_abs = abs_value
            best_residual = value
            best_mach = float(mach)
    if best_mach is None:
        raise ValueError("could not find a finite Mach root for H/L/T_e closure")
    point, final_residual = _primitive_from_mach(
        H_p=H_p,
        L_p=L_p,
        T_e=T_e,
        mach=best_mach,
        x=x,
        config=config,
    )
    if abs(final_residual) > tolerance_K:
        raise ValueError(
            "H/L/T_e closure did not solve to tolerance: "
            f"best Mach={best_mach:.12g}, residual_K={best_residual:.6g}, tolerance_K={tolerance_K:.6g}"
        )
    return point, float(final_residual)


def profile_to_freidberg_arrays(profile_path: str | Path, config: FreidbergConfig) -> dict[str, np.ndarray]:
    profile_path = Path(profile_path)
    with np.load(profile_path) as data:
        n = np.asarray(data["x"], dtype=float).size
        states = [primitive_to_freidberg(PrimitivePoint.from_npz(data, idx), config) for idx in range(n)]
        return {
            "x": np.asarray([state.x for state in states], dtype=float),
            "H_p": np.asarray([state.H_p for state in states], dtype=float),
            "L_p": np.asarray([state.L_p for state in states], dtype=float),
            "T_e": np.asarray(data["T_e"], dtype=float),
            "mach": np.asarray(data["mach"], dtype=float),
        }


def reconstruct_profile_from_hl_arrays(
    *,
    H_p: np.ndarray,
    L_p: np.ndarray,
    T_e: np.ndarray,
    x: np.ndarray,
    config: FreidbergConfig,
    mach_hints: np.ndarray | None = None,
    branch: str = "any",
    tolerance_K: float = 1e-5,
) -> tuple[list[PrimitivePoint], np.ndarray]:
    H_arr = np.asarray(H_p, dtype=float)
    L_arr = np.asarray(L_p, dtype=float)
    Te_arr = np.asarray(T_e, dtype=float)
    x_arr = np.asarray(x, dtype=float)
    if not (H_arr.shape == L_arr.shape == Te_arr.shape == x_arr.shape):
        raise ValueError("H_p, L_p, T_e, and x arrays must have identical shape")
    hints = None if mach_hints is None else np.asarray(mach_hints, dtype=float)
    if hints is not None and hints.shape != H_arr.shape:
        raise ValueError("mach_hints must match H_p shape")
    points: list[PrimitivePoint] = []
    residuals: list[float] = []
    for idx in range(H_arr.size):
        point, residual = solve_primitive_from_hlt(
            H_p=float(H_arr[idx]),
            L_p=float(L_arr[idx]),
            T_e=float(Te_arr[idx]),
            x=float(x_arr[idx]),
            config=config,
            branch=branch,
            mach_hint=None if hints is None else float(hints[idx]),
            tolerance_K=tolerance_K,
        )
        points.append(point)
        residuals.append(float(residual))
    return points, np.asarray(residuals, dtype=float)


def bridge_diagnostics(original: list[PrimitivePoint], reconstructed: list[PrimitivePoint], residuals: np.ndarray) -> ReconstructionDiagnostics:
    if len(original) != len(reconstructed):
        raise ValueError("original and reconstructed point counts differ")

    def max_abs(name: str) -> float:
        return float(max(abs(getattr(a, name) - getattr(b, name)) for a, b in zip(original, reconstructed)))

    def max_rel(name: str) -> float:
        values = []
        for a, b in zip(original, reconstructed):
            denom = max(abs(float(getattr(a, name))), 1e-300)
            values.append(abs(float(getattr(a, name)) - float(getattr(b, name))) / denom)
        return float(max(values))

    return ReconstructionDiagnostics(
        max_abs_closure_residual_K=float(np.max(np.abs(np.asarray(residuals, dtype=float)))),
        max_abs_mach_error=max_abs("mach"),
        max_abs_T_p_error_K=max_abs("T_p"),
        max_rel_A_error=max_rel("A"),
        max_rel_n_p_error=max_rel("n_p"),
        max_rel_v_p_error=max_rel("v_p"),
        max_abs_Z_error=max_abs("Z"),
    )
