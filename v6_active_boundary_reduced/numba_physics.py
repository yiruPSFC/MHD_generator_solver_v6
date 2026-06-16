from __future__ import annotations

import math

from numba import njit

from .physics_constants import (
    E_CHARGE,
    H_P,
    K_B,
    M_E,
    _DELTA_MIN,
    _EPS,
    _FION_MAX,
    _FION_MIN,
    _SAHA_LOG_K_MAX,
    _SAHA_LOG_K_MIN,
    _SAHA_PREFAC,
    _TP_MIN,
)


@njit(cache=True)
def _exp_clip(value: float) -> float:
    return math.exp(min(max(value, -700.0), 700.0))


@njit(cache=True)
def _clip_range(value: float, lower: float, upper: float) -> float:
    return min(max(value, lower), upper)


@njit(cache=True)
def _df_dbeta_numeric(beta: float, Z: float) -> float:
    b2 = beta * beta
    d = 1.0 + Z
    num = b2 * (b2 + d * d)
    c = b2 + d
    den = c * c + _EPS
    dnum_db2 = 2.0 * b2 + d * d
    dden_db2 = 2.0 * c
    dF_db2 = (dnum_db2 * den - num * dden_db2) / (den * den + _EPS)
    return dF_db2 * 2.0 * beta


@njit(cache=True)
def _df_dz_numeric(beta: float, Z: float) -> float:
    b2 = beta * beta
    d = 1.0 + Z
    num = b2 * (b2 + d * d)
    c = b2 + d
    den = c * c + _EPS
    dnum_dd = 2.0 * b2 * d
    dden_dd = 2.0 * c
    return (dnum_dd * den - num * dden_dd) / (den * den + _EPS)


@njit(cache=True)
def saha_terms_numba(
    n_p: float,
    T_e: float,
    seed_fraction: float,
    seed_ionization_energy_J: float,
) -> tuple[float, float, float, float, float, float]:
    n_p_safe = max(n_p, 1.0)
    T_e_safe = max(T_e, 1.0)
    seed_safe = max(seed_fraction, 1e-12)
    saha_a = _SAHA_PREFAC * T_e_safe
    log_K = 1.5 * math.log(max(saha_a, _EPS)) - seed_ionization_energy_J / (K_B * T_e_safe)
    K = math.exp(_clip_range(log_K, _SAHA_LOG_K_MIN, _SAHA_LOG_K_MAX))
    n_s = seed_safe * n_p_safe
    sqrt_term = math.sqrt(1.0 + 4.0 * n_s / (K + _EPS))
    n_e = 2.0 * n_s / (1.0 + sqrt_term)
    n_e = min(max(n_e, 0.0), n_s * (1.0 - 1e-12))
    return n_p_safe, T_e_safe, seed_safe, K, n_s, n_e


@njit(cache=True)
def inlet_design_numba(
    n_p_in: float,
    T_e_in: float,
    Z_in: float,
    I_0: float,
    seed_fraction: float,
    B: float,
    inlet_A: float,
    heavy_particle_mass_kg: float,
    seed_ionization_energy_J: float,
    sigma_ep: float,
) -> tuple[float, ...]:
    n_p_safe, T_e_safe, seed_safe, _K, n_s, n_e = saha_terms_numba(
        n_p_in,
        T_e_in,
        seed_fraction,
        seed_ionization_energy_J,
    )
    v_te = math.sqrt(2.0 * K_B * T_e_safe / M_E)
    beta = E_CHARGE * B / (M_E * n_p_safe * sigma_ep * v_te + _EPS)
    b2 = beta * beta
    den = b2 + 1.0 + Z_in
    inlet_A_safe = max(float(inlet_A), _EPS)
    v_in = (I_0 / inlet_A_safe) * den / (b2 * E_CHARGE * n_e + _EPS)
    dot_N = n_p_safe * v_in * inlet_A_safe
    F = b2 * (b2 + (1.0 + Z_in) * (1.0 + Z_in)) / (den * den + _EPS)
    T_p_in = T_e_safe - heavy_particle_mass_kg * v_in * v_in * F / (3.0 * K_B)
    T_p_floor = max(T_p_in, _TP_MIN)
    c_s = math.sqrt((5.0 / 3.0) * K_B * T_p_floor / heavy_particle_mass_kg + _EPS)
    mach = v_in / (c_s + _EPS)
    f_I_raw = n_e / (n_s + _EPS)
    f_I = _clip_range(f_I_raw, _FION_MIN, _FION_MAX)
    delta = max(T_e_safe / T_p_floor - 1.0, _DELTA_MIN)
    alpha = (K_B * T_e_safe / (2.0 * seed_ionization_energy_J)) * (2.0 - f_I) / (1.0 - f_I + _EPS)
    G = 4.0 * alpha * (2.0 + 1.0 / delta) * (1.0 + alpha * (1.0 + 1.0 / delta)) - b2
    return (
        n_p_safe,
        n_e,
        T_e_safe,
        T_p_in,
        Z_in,
        I_0,
        dot_N,
        v_in,
        seed_safe,
        mach,
        G,
        inlet_A_safe,
    )


@njit(cache=True)
def closure_state_numba(
    n_p: float,
    T_e: float,
    A: float,
    dot_N: float,
    I_0: float,
    seed_fraction: float,
    B: float,
    heavy_particle_mass_kg: float,
    seed_ionization_energy_J: float,
    sigma_ep: float,
) -> tuple[float, ...]:
    n_p_safe, T_e_safe, seed_safe, K, n_s, n_e = saha_terms_numba(
        n_p, T_e, seed_fraction, seed_ionization_energy_J
    )
    A_safe = max(A, 1e-12)
    v_te = math.sqrt(2.0 * K_B * T_e_safe / M_E)
    beta = E_CHARGE * B / (M_E * n_p_safe * sigma_ep * v_te + _EPS)
    eta = M_E * n_p_safe * sigma_ep * v_te / (E_CHARGE * E_CHARGE * n_e + _EPS)

    q_factor = E_CHARGE * dot_N / (I_0 + _EPS)
    q = q_factor * n_e / (n_p_safe + _EPS)
    b2 = beta * beta
    Z = b2 * (q - 1.0) - 1.0
    one_plus_z = 1.0 + Z
    den = b2 + one_plus_z
    F = b2 * (b2 + one_plus_z * one_plus_z) / (den * den + _EPS)

    v_p = dot_N / (n_p_safe * A_safe + _EPS)
    C = heavy_particle_mass_kg * v_p * v_p / (3.0 * K_B)
    T_p = T_e_safe - C * F

    dbeta_dTe = -0.5 * beta / T_e_safe
    dbeta_dnp = -beta / n_p_safe

    den_ns = n_s - n_e
    den_ns_sq = den_ns * den_ns + _EPS
    df_dne = n_e * (2.0 * n_s - n_e) / den_ns_sq
    dK_dTe = K * (1.5 / T_e_safe + seed_ionization_energy_J / (K_B * T_e_safe * T_e_safe))
    df_dne_safe = df_dne + _EPS
    dne_dTe = dK_dTe / df_dne_safe
    df_dnp = -(n_e * n_e) * seed_safe / den_ns_sq
    dne_dnp = -df_dnp / df_dne_safe

    dq_dTe = q_factor * dne_dTe / (n_p_safe + _EPS)
    dq_dnp = q_factor * (dne_dnp * n_p_safe - n_e) / (n_p_safe * n_p_safe + _EPS)
    dZ_dTe = 2.0 * beta * dbeta_dTe * (q - 1.0) + b2 * dq_dTe
    dZ_dnp = 2.0 * beta * dbeta_dnp * (q - 1.0) + b2 * dq_dnp

    dF_dTe = _df_dbeta_numeric(beta, Z) * dbeta_dTe + _df_dz_numeric(beta, Z) * dZ_dTe
    dF_dnp = _df_dbeta_numeric(beta, Z) * dbeta_dnp + _df_dz_numeric(beta, Z) * dZ_dnp

    dTp_dTe = 1.0 - C * dF_dTe
    dTp_dnp = 2.0 * C * F / (n_p_safe + _EPS) - C * dF_dnp
    dTp_dA = 2.0 * C * F / (A_safe + _EPS)

    jfac = E_CHARGE * n_e * v_p
    J_x = I_0 / A_safe
    J_y = -beta * one_plus_z / (den + _EPS) * jfac
    E_x = -b2 * Z / (den + _EPS) * eta * jfac
    nu_E = eta * 2.0 * E_CHARGE * E_CHARGE * n_e / heavy_particle_mass_kg

    T_p_floor = max(T_p, _TP_MIN)
    c_s = math.sqrt((5.0 / 3.0) * K_B * T_p_floor / heavy_particle_mass_kg + _EPS)
    mach = v_p / (c_s + _EPS)

    seed_density = max(seed_fraction, 1e-12) * n_p_safe
    f_I_raw = n_e / (seed_density + _EPS)
    f_I = _clip_range(f_I_raw, _FION_MIN, _FION_MAX)
    delta_raw = T_e_safe / T_p_floor - 1.0
    delta = max(delta_raw, _DELTA_MIN)
    alpha = (K_B * T_e_safe / (2.0 * seed_ionization_energy_J)) * (2.0 - f_I) / (1.0 - f_I + _EPS)
    G = 4.0 * alpha * (2.0 + 1.0 / delta) * (1.0 + alpha * (1.0 + 1.0 / delta)) - b2

    return (
        n_p_safe,
        T_e_safe,
        A_safe,
        n_s,
        n_e,
        beta,
        eta,
        Z,
        F,
        v_p,
        T_p,
        dTp_dTe,
        dTp_dnp,
        dTp_dA,
        J_x,
        J_y,
        E_x,
        mach,
        G,
        nu_E,
    )


@njit(cache=True)
def g_state_and_gradients_numba(
    n_p: float,
    T_e: float,
    A: float,
    dot_N: float,
    I_0: float,
    seed_fraction: float,
    B: float,
    heavy_particle_mass_kg: float,
    seed_ionization_energy_J: float,
    sigma_ep: float,
) -> tuple[float, float, float, float]:
    """Return G and analytical derivatives with respect to n_p, T_e, and A."""

    n_p_safe, T_e_safe, seed_safe, K, n_s, n_e = saha_terms_numba(
        n_p, T_e, seed_fraction, seed_ionization_energy_J
    )
    A_safe = max(A, 1e-12)
    v_te = math.sqrt(2.0 * K_B * T_e_safe / M_E)
    beta = E_CHARGE * B / (M_E * n_p_safe * sigma_ep * v_te + _EPS)

    q_factor = E_CHARGE * dot_N / (I_0 + _EPS)
    q = q_factor * n_e / (n_p_safe + _EPS)
    b2 = beta * beta
    Z = b2 * (q - 1.0) - 1.0
    one_plus_z = 1.0 + Z
    den = b2 + one_plus_z
    F = b2 * (b2 + one_plus_z * one_plus_z) / (den * den + _EPS)

    v_p = dot_N / (n_p_safe * A_safe + _EPS)
    C = heavy_particle_mass_kg * v_p * v_p / (3.0 * K_B)
    T_p = T_e_safe - C * F

    dn_safe_dn = 1.0 if n_p > 1.0 else 0.0
    dte_safe_dte = 1.0 if T_e > 1.0 else 0.0
    dA_safe_dA = 1.0 if A > 1e-12 else 0.0

    dbeta_dTe = -0.5 * beta / T_e_safe * dte_safe_dte
    dbeta_dn = -beta / n_p_safe * dn_safe_dn

    den_ns = n_s - n_e
    den_ns_sq = den_ns * den_ns + _EPS
    df_dne = n_e * (2.0 * n_s - n_e) / den_ns_sq
    log_K_raw = 1.5 * math.log(max(_SAHA_PREFAC * T_e_safe, _EPS)) - seed_ionization_energy_J / (
        K_B * T_e_safe
    )
    if log_K_raw <= _SAHA_LOG_K_MIN or log_K_raw >= _SAHA_LOG_K_MAX:
        dK_dTe = 0.0
    else:
        dK_dTe = K * (1.5 / T_e_safe + seed_ionization_energy_J / (K_B * T_e_safe * T_e_safe))
    dK_dTe *= dte_safe_dte
    df_dne_safe = df_dne + _EPS
    dne_dTe = dK_dTe / df_dne_safe
    df_dnp = -(n_e * n_e) * seed_safe / den_ns_sq
    dne_dn = -df_dnp / df_dne_safe * dn_safe_dn

    seed_density = seed_safe * n_p_safe
    n_e_upper = seed_density * (1.0 - 1e-12)
    if n_e >= n_e_upper:
        dne_dn = seed_safe * (1.0 - 1e-12) * dn_safe_dn
        dne_dTe = 0.0

    dq_dTe = q_factor * dne_dTe / (n_p_safe + _EPS)
    dq_dn = q_factor * (dne_dn * n_p_safe - n_e * dn_safe_dn) / (n_p_safe * n_p_safe + _EPS)
    dZ_dTe = 2.0 * beta * dbeta_dTe * (q - 1.0) + b2 * dq_dTe
    dZ_dn = 2.0 * beta * dbeta_dn * (q - 1.0) + b2 * dq_dn

    dF_dTe = _df_dbeta_numeric(beta, Z) * dbeta_dTe + _df_dz_numeric(beta, Z) * dZ_dTe
    dF_dn = _df_dbeta_numeric(beta, Z) * dbeta_dn + _df_dz_numeric(beta, Z) * dZ_dn

    dTp_dTe = dte_safe_dte - C * dF_dTe
    dTp_dn = 2.0 * C * F / (n_p_safe + _EPS) * dn_safe_dn - C * dF_dn
    dTp_dA = 2.0 * C * F / (A_safe + _EPS) * dA_safe_dA

    T_p_floor = max(T_p, _TP_MIN)
    if T_p > _TP_MIN:
        dTp_floor_dn = dTp_dn
        dTp_floor_dTe = dTp_dTe
        dTp_floor_dA = dTp_dA
    else:
        dTp_floor_dn = 0.0
        dTp_floor_dTe = 0.0
        dTp_floor_dA = 0.0

    seed_denominator = seed_density + _EPS
    f_I_raw = n_e / seed_denominator
    f_I = _clip_range(f_I_raw, _FION_MIN, _FION_MAX)
    if f_I_raw <= _FION_MIN or f_I_raw >= _FION_MAX:
        dfI_dn = 0.0
        dfI_dTe = 0.0
    else:
        dfI_dn = (dne_dn * seed_denominator - n_e * seed_safe * dn_safe_dn) / (
            seed_denominator * seed_denominator + _EPS
        )
        dfI_dTe = dne_dTe / seed_denominator

    delta_raw = T_e_safe / T_p_floor - 1.0
    delta = max(delta_raw, _DELTA_MIN)
    if delta_raw > _DELTA_MIN:
        inv_Tp_floor = 1.0 / max(T_p_floor, _EPS)
        inv_Tp_floor2 = 1.0 / max(T_p_floor * T_p_floor, _EPS)
        ddelta_dn = -T_e_safe * inv_Tp_floor2 * dTp_floor_dn
        ddelta_dTe = dte_safe_dte * inv_Tp_floor - T_e_safe * inv_Tp_floor2 * dTp_floor_dTe
        ddelta_dA = -T_e_safe * inv_Tp_floor2 * dTp_floor_dA
    else:
        ddelta_dn = 0.0
        ddelta_dTe = 0.0
        ddelta_dA = 0.0

    c_te = K_B * T_e_safe / (2.0 * seed_ionization_energy_J)
    ratio_den = 1.0 - f_I + _EPS
    ratio = (2.0 - f_I) / ratio_den
    alpha = c_te * ratio
    dratio_df = (1.0 - _EPS) / (ratio_den * ratio_den + _EPS)
    dc_te_dTe = K_B / (2.0 * seed_ionization_energy_J) * dte_safe_dte
    dalpha_dn = c_te * dratio_df * dfI_dn
    dalpha_dTe = dc_te_dTe * ratio + c_te * dratio_df * dfI_dTe
    dalpha_dA = 0.0

    inv_delta = 1.0 / delta
    inv_delta2 = inv_delta * inv_delta
    U = 2.0 + inv_delta
    W = 1.0 + inv_delta
    V = 1.0 + alpha * W
    G = 4.0 * alpha * U * V - b2

    common_alpha = 4.0 * U * (V + alpha * W)
    common_delta = -4.0 * alpha * inv_delta2 * (V + alpha * U)
    dG_dn = common_alpha * dalpha_dn + common_delta * ddelta_dn - 2.0 * beta * dbeta_dn
    dG_dTe = common_alpha * dalpha_dTe + common_delta * ddelta_dTe - 2.0 * beta * dbeta_dTe
    dG_dA = common_alpha * dalpha_dA + common_delta * ddelta_dA
    return G, dG_dn, dG_dTe, dG_dA


@njit(cache=True)
def _max_abs3(a: float, b: float, floor: float) -> float:
    return max(max(abs(a), abs(b)), floor)


@njit(cache=True)
def freidberg_balance_terms_numba(
    n_p: float,
    T_e: float,
    A: float,
    dot_N: float,
    I_0: float,
    seed_fraction: float,
    B: float,
    area_scale_m2: float,
    heavy_particle_mass_kg: float,
    seed_ionization_energy_J: float,
    sigma_ep: float,
    length_m: float,
) -> tuple[float, float, float, float, float, float]:
    closure = closure_state_numba(
        n_p,
        T_e,
        A,
        dot_N,
        I_0,
        seed_fraction,
        B,
        heavy_particle_mass_kg,
        seed_ionization_energy_J,
        sigma_ep,
    )
    n_p_safe = closure[0]
    A_safe = closure[2]
    eta = closure[6]
    v_p = closure[9]
    T_p = closure[10]
    J_x = closure[14]
    J_y = closure[15]
    mach = closure[17]

    A0 = max(float(area_scale_m2), 1.0e-300)
    T_p_safe = max(T_p, _TP_MIN)
    M2 = mach * mach
    J2 = J_x * J_x + J_y * J_y
    H_p = (A_safe * n_p_safe * v_p / A0) * (2.5 * K_B * T_p_safe + 0.5 * heavy_particle_mass_kg * v_p * v_p)
    L_p = mach * (A_safe / A0) / ((M2 + 3.0) * (M2 + 3.0))
    rhs_H = (A_safe / A0) * (v_p * J_y * B + eta * J2)
    p_p = n_p_safe * K_B * T_p_safe
    denom = max((M2 + 3.0) * p_p * v_p, 1.0e-300)
    rhs_L = (
        -(12.0 / 5.0)
        * L_p
        / denom
        * (v_p * J_y * B - ((5.0 * M2 + 3.0) / 12.0) * eta * J2)
    )
    inverse_length = 1.0 / max(float(length_m), 1.0e-30)
    H_scale = _max_abs3(H_p * inverse_length, rhs_H, 1.0)
    L_scale = _max_abs3(L_p * inverse_length, rhs_L, inverse_length)
    return H_p, L_p, rhs_H, rhs_L, H_scale, L_scale


@njit(cache=True)
def dynamic_terms_numba(
    n_p: float,
    T_e: float,
    A: float,
    sigma: float,
    dot_N: float,
    I_0: float,
    seed_fraction: float,
    B: float,
    heavy_particle_mass_kg: float,
    seed_ionization_energy_J: float,
    sigma_ep: float,
) -> tuple[float, ...]:
    closure = closure_state_numba(
        n_p,
        T_e,
        A,
        dot_N,
        I_0,
        seed_fraction,
        B,
        heavy_particle_mass_kg,
        seed_ionization_energy_J,
        sigma_ep,
    )
    n_p_safe = closure[0]
    T_e_safe = closure[1]
    A_safe = closure[2]
    n_e = closure[4]
    v_p = closure[9]
    T_p = closure[10]
    dTp_dTe = closure[11]
    dTp_dnp = closure[12]
    dTp_dA = closure[13]
    J_y = closure[15]
    nu_E = closure[19]
    v_p_safe = max(v_p, _EPS)

    M11 = (-heavy_particle_mass_kg * v_p * v_p + K_B * T_p) + K_B * n_p_safe * dTp_dnp
    M12 = K_B * n_p_safe * dTp_dTe
    M13 = K_B * n_p_safe * dTp_dA - heavy_particle_mass_kg * n_p_safe * v_p * v_p / (A_safe + _EPS)
    E11 = -T_p + 1.5 * n_p_safe * dTp_dnp
    E12 = 1.5 * n_p_safe * dTp_dTe
    E13 = 1.5 * n_p_safe * dTp_dA

    dA_dx = sigma * A_safe
    rhs_m = J_y * B - M13 * dA_dx
    rhs_e = 1.5 * nu_E * n_e * (T_e_safe - T_p) / v_p_safe - E13 * dA_dx
    det = M11 * E12 - M12 * E11
    return (
        M11,
        M12,
        M13,
        E11,
        E12,
        E13,
        dA_dx,
        rhs_m,
        rhs_e,
        det,
        closure[5],
        closure[7],
        closure[9],
        closure[10],
        closure[17],
        closure[18],
    )
