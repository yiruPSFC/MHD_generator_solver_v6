from __future__ import annotations

"""
V6 global local algebraic closure (explicit, no along-channel root-finding).

State variables:
- n_p, T_e, A

Conserved constants per trajectory:
- dot_N, I_0

Derived variables:
- v_p, n_e, beta, eta, Z, T_p, J_x, J_y, E_x

All core kernels are explicit algebraic forms and numba-jittable.
"""

import math

try:
    from numba import njit
except Exception:  # pragma: no cover
    def njit(*args, **kwargs):
        def deco(fn):
            return fn
        return deco


from v6_core.local_algebraic_closure import (
    B_FIELD,
    E_CHARGE,
    E_I,
    K_B,
    M_P,
    SIGMA_EP,
    F_beta_Z,
    _saha_partials,
    beta_from_np_te,
    dF_dbeta,
    dF_dZ,
    eta_from_np_ne_te,
    ne_from_np_te,
)


_EPS = 1e-300
_TMIN = 1.0


@njit(cache=True)
def _safe_pos(x: float, floor: float = _EPS) -> float:
    return x if x > floor else floor


@njit(cache=True)
def _safe_signed(x: float, floor: float = _EPS) -> float:
    if abs(x) >= floor:
        return x
    return floor if x >= 0.0 else -floor


@njit(cache=True)
def _q_from_state(n_p: float, n_e: float, dot_N: float, I_0: float) -> float:
    return E_CHARGE * n_e * dot_N / _safe_signed(I_0) / _safe_pos(n_p)


@njit(cache=True)
def _clip_fraction_with_zero_slope(x: float, dx_dTe: float, dx_dnp: float):
    lo = 1e-12
    hi = 1.0 - 1e-12
    if x <= lo:
        return lo, 0.0, 0.0
    if x >= hi:
        return hi, 0.0, 0.0
    return x, dx_dTe, dx_dnp


@njit(cache=True)
def _clip_positive_with_zero_slope(x: float, dx_dTe: float, dx_dnp: float, dx_dA: float):
    lo = 1e-12
    if x <= lo:
        return lo, 0.0, 0.0, 0.0
    return x, dx_dTe, dx_dnp, dx_dA


@njit(cache=True)
def _velikhov_margin_value_from_state(
    n_p: float,
    T_e: float,
    T_p: float,
    n_e: float,
    beta: float,
    seed_fraction: float,
) -> float:
    np_safe = _safe_pos(n_p)
    te_safe = _safe_pos(T_e, _TMIN)
    if T_p <= 0.0 or (not math.isfinite(T_p)):
        return math.nan

    ns = seed_fraction * np_safe
    if ns <= 0.0:
        return math.nan

    fI_raw = n_e / _safe_pos(ns)
    fI, _, _ = _clip_fraction_with_zero_slope(fI_raw, 0.0, 0.0)
    delta_raw = te_safe / _safe_pos(T_p, _TMIN) - 1.0
    delta, _, _, _ = _clip_positive_with_zero_slope(delta_raw, 0.0, 0.0, 0.0)

    alpha = (K_B * te_safe / (2.0 * E_I)) * (2.0 - fI) / _safe_pos(1.0 - fI, 1e-12)
    rhs = 4.0 * alpha * (2.0 + 1.0 / delta) * (1.0 + alpha * (1.0 + 1.0 / delta))
    lhs = beta * beta
    return rhs - lhs


@njit(cache=True)
def compute_currents_fields_global(
    v_p: float,
    n_e: float,
    beta: float,
    eta: float,
    Z: float,
    I_0: float,
    A: float,
):
    a_safe = _safe_pos(A)
    J_x = I_0 / a_safe

    b2 = beta * beta
    den = _safe_signed(b2 + 1.0 + Z)
    jfac = E_CHARGE * n_e * v_p
    J_y = -beta * (1.0 + Z) / den * jfac
    E_x = -b2 * Z / den * eta * jfac
    E_y = 0.0
    return J_x, J_y, E_x, E_y


@njit(cache=True)
def velikhov_margin_global(
    n_p: float,
    T_e: float,
    T_p: float,
    n_e: float,
    beta: float,
    seed_fraction: float,
) -> float:
    return _velikhov_margin_value_from_state(n_p, T_e, T_p, n_e, beta, seed_fraction)


@njit(cache=True)
def local_closure_global_with_partials(
    n_p: float,
    T_e: float,
    A: float,
    dot_N: float,
    I_0: float,
    seed_fraction: float,
    B: float = B_FIELD,
    sigma_ep: float = SIGMA_EP,
):
    """
    Explicit global closure + analytical Jacobian ingredients.

    Returns:
      v_p, n_e, beta, eta, Z, T_p,
      dTp_dTe, dTp_dnp, dTp_dA,
      dZ_dTe, dZ_dnp,
      dne_dTe, dne_dnp,
      G, G_np, G_Te, G_A
    """
    np_safe = _safe_pos(n_p)
    te_safe = _safe_pos(T_e, _TMIN)
    a_safe = _safe_pos(A)

    v_p = dot_N / (_safe_pos(np_safe) * a_safe)
    n_e = ne_from_np_te(np_safe, te_safe, seed_fraction)
    beta = beta_from_np_te(np_safe, te_safe, B=B, sigma_ep=sigma_ep)
    eta = eta_from_np_ne_te(np_safe, n_e, te_safe, sigma_ep=sigma_ep)

    q = _q_from_state(np_safe, n_e, dot_N, I_0)
    b2 = beta * beta
    Z = b2 * (q - 1.0) - 1.0

    F = F_beta_Z(beta, Z)
    C = M_P * v_p * v_p / (3.0 * K_B)
    T_p = te_safe - C * F

    dbeta_dTe = -0.5 * beta / te_safe
    dbeta_dnp = -beta / np_safe

    dne_dTe, dne_dnp = _saha_partials(np_safe, te_safe, seed_fraction, n_e)

    q_pref = E_CHARGE * dot_N / _safe_signed(I_0)
    q_Te = q_pref * dne_dTe / np_safe
    q_np = q_pref * (dne_dnp / np_safe - n_e / (np_safe * np_safe))

    dZ_dTe = 2.0 * beta * dbeta_dTe * (q - 1.0) + b2 * q_Te
    dZ_dnp = 2.0 * beta * dbeta_dnp * (q - 1.0) + b2 * q_np

    dF_db = dF_dbeta(beta, Z)
    dF_dz = dF_dZ(beta, Z)

    dF_dTe = dF_db * dbeta_dTe + dF_dz * dZ_dTe
    dF_dnp = dF_db * dbeta_dnp + dF_dz * dZ_dnp

    dTp_dTe = 1.0 - C * dF_dTe
    dTp_dnp = (2.0 * C / np_safe) * F - C * dF_dnp
    dTp_dA = (2.0 * C / a_safe) * F

    if (not math.isfinite(T_p)) or T_p <= 0.0:
        G = math.nan
        G_np = math.nan
        G_Te = math.nan
        G_A = math.nan
    else:
        ns = seed_fraction * np_safe
        ns_safe = _safe_pos(ns)
        fI_raw = n_e / ns_safe
        dfI_dTe_raw = dne_dTe / ns_safe
        dfI_dnp_raw = (np_safe * dne_dnp - n_e) / _safe_pos(seed_fraction * np_safe * np_safe)
        fI, dfI_dTe, dfI_dnp = _clip_fraction_with_zero_slope(fI_raw, dfI_dTe_raw, dfI_dnp_raw)

        one_minus_fI = _safe_pos(1.0 - fI, 1e-12)
        R_fI = (2.0 - fI) / one_minus_fI
        alpha_pref = K_B * te_safe / (2.0 * E_I)
        alpha = alpha_pref * R_fI

        dalpha_dTe = (K_B / (2.0 * E_I)) * R_fI + alpha_pref * dfI_dTe / (one_minus_fI * one_minus_fI)
        dalpha_dnp = alpha_pref * dfI_dnp / (one_minus_fI * one_minus_fI)
        dalpha_dA = 0.0

        tp_safe = _safe_pos(T_p, _TMIN)
        delta_raw = te_safe / tp_safe - 1.0
        ddelta_dTe_raw = 1.0 / tp_safe - te_safe * dTp_dTe / (tp_safe * tp_safe)
        ddelta_dnp_raw = -te_safe * dTp_dnp / (tp_safe * tp_safe)
        ddelta_dA_raw = -te_safe * dTp_dA / (tp_safe * tp_safe)
        delta, ddelta_dTe, ddelta_dnp, ddelta_dA = _clip_positive_with_zero_slope(
            delta_raw,
            ddelta_dTe_raw,
            ddelta_dnp_raw,
            ddelta_dA_raw,
        )

        inv_delta = 1.0 / delta
        inv_delta2 = inv_delta * inv_delta
        U = 2.0 + inv_delta
        V = 1.0 + alpha * (1.0 + inv_delta)

        dU_dTe = -ddelta_dTe * inv_delta2
        dU_dnp = -ddelta_dnp * inv_delta2
        dU_dA = -ddelta_dA * inv_delta2

        common_V = 1.0 + inv_delta
        dV_dTe = dalpha_dTe * common_V - alpha * ddelta_dTe * inv_delta2
        dV_dnp = dalpha_dnp * common_V - alpha * ddelta_dnp * inv_delta2
        dV_dA = dalpha_dA * common_V - alpha * ddelta_dA * inv_delta2

        G = 4.0 * alpha * U * V - b2
        G_Te = 4.0 * (dalpha_dTe * U * V + alpha * dU_dTe * V + alpha * U * dV_dTe) - 2.0 * beta * dbeta_dTe
        G_np = 4.0 * (dalpha_dnp * U * V + alpha * dU_dnp * V + alpha * U * dV_dnp) - 2.0 * beta * dbeta_dnp
        G_A = 4.0 * (dalpha_dA * U * V + alpha * dU_dA * V + alpha * U * dV_dA)

    return (
        v_p,
        n_e,
        beta,
        eta,
        Z,
        T_p,
        dTp_dTe,
        dTp_dnp,
        dTp_dA,
        dZ_dTe,
        dZ_dnp,
        dne_dTe,
        dne_dnp,
        G,
        G_np,
        G_Te,
        G_A,
    )
