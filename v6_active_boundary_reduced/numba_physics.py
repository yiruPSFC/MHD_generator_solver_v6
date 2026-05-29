from __future__ import annotations

import math

from numba import njit

from v6_maingo_casadi.constants import (
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


@njit(cache=True)
def g_boundary_residual_numba(
    log_n_next: float,
    log_Te_next: float,
    sigma: float,
    current_n_p: float,
    current_T_e: float,
    current_logA: float,
    dx_signed: float,
    area_scale_m2: float,
    dot_N: float,
    I_0: float,
    seed_fraction: float,
    B: float,
    heavy_particle_mass_kg: float,
    seed_ionization_energy_J: float,
    sigma_ep: float,
    g_floor: float,
) -> tuple[float, float, float, float, float, float, float, float]:
    n_next = _exp_clip(log_n_next)
    te_next = _exp_clip(log_Te_next)
    logA_next = current_logA + dx_signed * sigma
    area_next = area_scale_m2 * _exp_clip(logA_next)
    terms = dynamic_terms_numba(
        n_next,
        te_next,
        area_next,
        sigma,
        dot_N,
        I_0,
        seed_fraction,
        B,
        heavy_particle_mass_kg,
        seed_ionization_energy_J,
        sigma_ep,
    )
    dn_dx = (n_next - current_n_p) / dx_signed
    dte_dx = (te_next - current_T_e) / dx_signed
    momentum = terms[0] * dn_dx + terms[1] * dte_dx - terms[7]
    energy = terms[3] * dn_dx + terms[4] * dte_dx - terms[8]
    m_scale = max(
        1.0,
        abs(terms[0] * max(n_next, 1.0) / max(abs(dx_signed), 1e-300)),
        abs(terms[1] * max(te_next, 1.0) / max(abs(dx_signed), 1e-300)),
        abs(terms[7]),
    )
    e_scale = max(
        1.0,
        abs(terms[3] * max(n_next, 1.0) / max(abs(dx_signed), 1e-300)),
        abs(terms[4] * max(te_next, 1.0) / max(abs(dx_signed), 1e-300)),
        abs(terms[8]),
    )
    G = terms[15]
    return (
        momentum / m_scale,
        energy / e_scale,
        G - g_floor,
        n_next,
        te_next,
        logA_next,
        area_next,
        terms[13],
    )
