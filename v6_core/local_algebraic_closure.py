from __future__ import annotations

"""
V6 local algebraic closure (explicit, no root-finding).

State variables:
- n_p, T_e

Conserved constants per trajectory:
- m_0, J_x0

Derived variables:
- v_p, n_e, beta, eta, Z, T_p

All core kernels are explicit algebraic forms and numba-jittable.
"""

import math
from dataclasses import dataclass

try:
    from numba import njit
except Exception:  # pragma: no cover
    def njit(*args, **kwargs):
        def deco(fn):
            return fn
        return deco


# Physical constants
B_FIELD = 0.02  # Tesla
E_CHARGE = 1.602176634e-19
K_B = 1.380649e-23
H_P = 6.62607015e-34
M_E = 9.10938356e-31
M_P = 39.948 * 1.66053906660e-27
SIGMA_EP = 3.942573033087758e-21
E_I = 4.3407 * E_CHARGE

_EPS = 1e-300
_TMIN = 1.0


@njit(cache=True)
def _safe_pos(x: float, floor: float = _EPS) -> float:
    return x if x > floor else floor


@njit(cache=True)
def beta_from_np_te(n_p: float, T_e: float, B: float = B_FIELD, sigma_ep: float = SIGMA_EP) -> float:
    np_safe = _safe_pos(n_p)
    te_safe = _safe_pos(T_e, _TMIN)
    v_te = math.sqrt(2.0 * K_B * te_safe / M_E)
    return E_CHARGE * B / _safe_pos(M_E * np_safe * sigma_ep * v_te)


@njit(cache=True)
def saha_K(T_e: float) -> float:
    te_safe = _safe_pos(T_e, _TMIN)
    A = 2.0 * math.pi * M_E * K_B * te_safe / (H_P * H_P)
    logK = 1.5 * math.log(_safe_pos(A)) - E_I / (K_B * te_safe)
    if logK < -700.0:
        return 0.0
    if logK > 700.0:
        return math.exp(700.0)
    return math.exp(logK)


@njit(cache=True)
def ne_from_np_te(n_p: float, T_e: float, seed_fraction: float) -> float:
    np_safe = _safe_pos(n_p)
    ns = seed_fraction * np_safe
    if ns <= 0.0:
        return 0.0

    K = saha_K(T_e)
    if K <= 0.0:
        return 0.0

    # Stable quadratic positive root of ne^2 + K*ne - K*ns = 0
    sqrt_term = K * math.sqrt(1.0 + 4.0 * ns / _safe_pos(K))
    denom = K + sqrt_term
    ne = (2.0 * K * ns) / _safe_pos(denom)

    upper = ns * (1.0 - 1e-12)
    if ne < 0.0:
        return 0.0
    if ne > upper:
        return upper
    return ne


@njit(cache=True)
def eta_from_np_ne_te(n_p: float, n_e: float, T_e: float, sigma_ep: float = SIGMA_EP) -> float:
    np_safe = _safe_pos(n_p)
    ne_safe = _safe_pos(n_e)
    te_safe = _safe_pos(T_e, _TMIN)
    v_te = math.sqrt(2.0 * K_B * te_safe / M_E)
    return M_E * np_safe * sigma_ep * v_te / _safe_pos(E_CHARGE * E_CHARGE * ne_safe)


@njit(cache=True)
def F_beta_Z(beta: float, Z: float) -> float:
    b2 = beta * beta
    d = 1.0 + Z
    den = _safe_pos((b2 + d) * (b2 + d))
    return b2 * (b2 + d * d) / den


@njit(cache=True)
def dF_dbeta(beta: float, Z: float) -> float:
    b2 = beta * beta
    d = 1.0 + Z
    N = b2 * (b2 + d * d)
    C = b2 + d
    D = _safe_pos(C * C)
    dN_db2 = 2.0 * b2 + d * d
    dD_db2 = 2.0 * C
    dF_db2 = (dN_db2 * D - N * dD_db2) / _safe_pos(D * D)
    return dF_db2 * 2.0 * beta


@njit(cache=True)
def dF_dZ(beta: float, Z: float) -> float:
    b2 = beta * beta
    d = 1.0 + Z
    N = b2 * (b2 + d * d)
    C = b2 + d
    D = _safe_pos(C * C)
    dN_dd = 2.0 * b2 * d
    dD_dd = 2.0 * C
    return (dN_dd * D - N * dD_dd) / _safe_pos(D * D)


@njit(cache=True)
def _saha_partials(n_p: float, T_e: float, seed_fraction: float, n_e: float):
    """
    Return (dne_dTe, dne_dnp) from implicit differentiation of:
      ne^2 / (ns - ne) = K(Te), ns = sf * np.
    """
    np_safe = _safe_pos(n_p)
    te_safe = _safe_pos(T_e, _TMIN)
    ns = seed_fraction * np_safe
    if ns <= 0.0:
        return 0.0, 0.0

    ne = n_e
    if ne <= 0.0:
        return 0.0, 0.0
    if ne >= ns:
        ne = ns * (1.0 - 1e-12)

    den_ns = _safe_pos(ns - ne)
    den2 = den_ns * den_ns
    df_dne = ne * (2.0 * ns - ne) / _safe_pos(den2)

    K = saha_K(te_safe)
    if K <= 0.0 or abs(df_dne) < 1e-300:
        return 0.0, 0.0

    dK_dTe = K * (1.5 / te_safe + E_I / (K_B * te_safe * te_safe))
    dne_dTe = dK_dTe / df_dne

    # partial f / partial np at fixed ne, Te
    df_dnp = -(ne * ne) * seed_fraction / _safe_pos(den2)
    dne_dnp = -df_dnp / df_dne

    return dne_dTe, dne_dnp


@njit(cache=True)
def local_closure_with_partials(
    n_p: float,
    T_e: float,
    m_0: float,
    J_x0: float,
    seed_fraction: float,
    B: float = B_FIELD,
    sigma_ep: float = SIGMA_EP,
):
    """
    Explicit closure + analytical Jacobian ingredients.

    Returns:
      v_p, n_e, beta, eta, Z, T_p,
      dTp_dTe, dTp_dnp,
      dZ_dTe, dZ_dnp,
      dne_dTe, dne_dnp
    """
    np_safe = _safe_pos(n_p)
    te_safe = _safe_pos(T_e, _TMIN)
    jx_safe = J_x0
    if abs(jx_safe) < 1e-300:
        jx_safe = 1e-300 if jx_safe >= 0.0 else -1e-300

    v_p = m_0 / np_safe
    n_e = ne_from_np_te(np_safe, te_safe, seed_fraction)
    beta = beta_from_np_te(np_safe, te_safe, B=B, sigma_ep=sigma_ep)
    eta = eta_from_np_ne_te(np_safe, n_e, te_safe, sigma_ep=sigma_ep)

    q = E_CHARGE * n_e * v_p / jx_safe
    b2 = beta * beta
    Z = b2 * (q - 1.0) - 1.0

    F = F_beta_Z(beta, Z)
    C = M_P * v_p * v_p / (3.0 * K_B)
    T_p = te_safe - C * F

    # Partials
    dbeta_dTe = -0.5 * beta / te_safe
    dbeta_dnp = -beta / np_safe

    dne_dTe, dne_dnp = _saha_partials(np_safe, te_safe, seed_fraction, n_e)

    dq_dTe = E_CHARGE * v_p * dne_dTe / jx_safe
    dv_dnp = -v_p / np_safe
    dq_dnp = E_CHARGE * (v_p * dne_dnp + n_e * dv_dnp) / jx_safe

    dZ_dTe = 2.0 * beta * dbeta_dTe * (q - 1.0) + b2 * dq_dTe
    dZ_dnp = 2.0 * beta * dbeta_dnp * (q - 1.0) + b2 * dq_dnp

    dF_db = dF_dbeta(beta, Z)
    dF_dz = dF_dZ(beta, Z)

    dF_dTe = dF_db * dbeta_dTe + dF_dz * dZ_dTe
    dF_dnp = dF_db * dbeta_dnp + dF_dz * dZ_dnp

    dTp_dTe = 1.0 - C * dF_dTe
    dTp_dnp = (2.0 * C / np_safe) * F - C * dF_dnp

    return (
        v_p,
        n_e,
        beta,
        eta,
        Z,
        T_p,
        dTp_dTe,
        dTp_dnp,
        dZ_dTe,
        dZ_dnp,
        dne_dTe,
        dne_dnp,
    )


@njit(cache=True)
def compute_currents_fields(v_p: float, n_e: float, beta: float, eta: float, Z: float):
    b2 = beta * beta
    den = b2 + 1.0 + Z
    if abs(den) < 1e-300:
        den = 1e-300 if den >= 0.0 else -1e-300
    jfac = E_CHARGE * n_e * v_p
    J_x = b2 / den * jfac
    J_y = -beta * (1.0 + Z) / den * jfac
    E_x = -b2 * Z / den * eta * E_CHARGE * n_e * v_p
    E_y = 0.0
    return J_x, J_y, E_x, E_y


@dataclass
class LocalClosureOutput:
    v_p: float
    n_e: float
    beta: float
    eta: float
    Z: float
    T_p: float
    dTp_dTe: float
    dTp_dnp: float
    dZ_dTe: float
    dZ_dnp: float
    dne_dTe: float
    dne_dnp: float


def evaluate_local_closure(
    n_p: float,
    T_e: float,
    m_0: float,
    J_x0: float,
    seed_fraction: float,
    B: float = B_FIELD,
    sigma_ep: float = SIGMA_EP,
) -> LocalClosureOutput:
    vals = local_closure_with_partials(
        n_p=n_p,
        T_e=T_e,
        m_0=m_0,
        J_x0=J_x0,
        seed_fraction=seed_fraction,
        B=B,
        sigma_ep=sigma_ep,
    )
    return LocalClosureOutput(*[float(v) for v in vals])
