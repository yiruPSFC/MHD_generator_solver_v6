from __future__ import annotations

import math

import numpy as np

from .constants import (
    E_CHARGE,
    H_P,
    K_B,
    M_E,
    _A_IN,
    _DELTA_MIN,
    _EPS,
    _FION_MAX,
    _FION_MIN,
    _G_HARD_MARGIN,
    _SAHA_K_MAX,
    _SAHA_K_MIN,
    _SAHA_LOG_K_MAX,
    _SAHA_LOG_K_MIN,
    _SAHA_PREFAC,
    _TP_MIN,
    OBJECTIVE_PROFILE_ENTHALPY_EXTRACTION,
    OBJECTIVE_PROFILE_LAB_POC_V2,
)
from .geometry import SplineAreaDesign, _evaluate_area_design_samples, _sample_area_reference
from .models import InletDesign
from .numerics import (
    _clip_range,
    _floored_pos,
    _max_op,
    _min_op,
    _ops_for_numeric,
    _safe_pos,
    _safe_signed_denom,
)
from .profiles import (
    WorkingFluidProfile,
    _DEFAULT_WORKING_FLUID_PROFILE,
    _design_value_weights_lab_poc_v2_objective,
    _normalize_objective_profile,
    _normalize_working_fluid_profile,
)

def _f_beta_z(beta, Z):
    b2 = beta * beta
    d = 1.0 + Z
    den = (b2 + d) * (b2 + d) + _EPS
    return b2 * (b2 + d * d) / den


def _df_dbeta(beta, Z):
    b2 = beta * beta
    d = 1.0 + Z
    num = b2 * (b2 + d * d)
    c = b2 + d
    den = c * c + _EPS
    dnum_db2 = 2.0 * b2 + d * d
    dden_db2 = 2.0 * c
    dF_db2 = (dnum_db2 * den - num * dden_db2) / (den * den + _EPS)
    return dF_db2 * 2.0 * beta


def _df_dz(beta, Z):
    b2 = beta * beta
    d = 1.0 + Z
    num = b2 * (b2 + d * d)
    c = b2 + d
    den = c * c + _EPS
    dnum_dd = 2.0 * b2 * d
    dden_dd = 2.0 * c
    return (dnum_dd * den - num * dden_dd) / (den * den + _EPS)


def _saha_terms(
    *,
    ops,
    n_p,
    T_e,
    seed_fraction,
    working_fluid: WorkingFluidProfile = _DEFAULT_WORKING_FLUID_PROFILE,
):
    fluid = _normalize_working_fluid_profile(working_fluid)
    n_p_safe = _safe_pos(ops, n_p, 1.0)
    T_e_safe = _safe_pos(ops, T_e, 1.0)
    seed_safe = _safe_pos(ops, seed_fraction, 1e-12)
    # Group the Saha prefactor into one moderate-sized constant. With maingopy's
    # FFVar on macOS arm64, dividing by h^2 (~4.39e-67) directly produces NaN,
    # while the equivalent grouped product below evaluates correctly.
    saha_a = _SAHA_PREFAC * T_e_safe
    log_K = 1.5 * ops.log(_safe_pos(ops, saha_a, _EPS)) - float(fluid.seed_ionization_energy_J) / (
        K_B * T_e_safe
    )
    K = ops.exp(_clip_range(ops, log_K, _SAHA_LOG_K_MIN, _SAHA_LOG_K_MAX))
    n_s = seed_safe * n_p_safe
    sqrt_term = ops.sqrt(1.0 + 4.0 * n_s / (K + _EPS))
    n_e = 2.0 * n_s / (1.0 + sqrt_term)
    n_e = _min_op(ops, _max_op(ops, n_e, 0.0), n_s * (1.0 - 1e-12))
    return n_p_safe, T_e_safe, seed_safe, K, n_s, n_e


def _closure_state(
    *,
    ops,
    n_p,
    T_e,
    A,
    dot_N,
    I_0,
    seed_fraction,
    B,
    working_fluid: WorkingFluidProfile = _DEFAULT_WORKING_FLUID_PROFILE,
):
    fluid = _normalize_working_fluid_profile(working_fluid)
    is_maingo = getattr(ops, "lb_func", None) is not None
    n_p_safe, T_e_safe, seed_safe, K, n_s, n_e = _saha_terms(
        ops=ops,
        n_p=n_p,
        T_e=T_e,
        seed_fraction=seed_fraction,
        working_fluid=fluid,
    )
    A_safe = _safe_pos(ops, A, 1e-12)
    v_te = ops.sqrt(2.0 * K_B * T_e_safe / M_E)
    if is_maingo:
        beta = E_CHARGE * float(B) / _safe_pos(ops, M_E * n_p_safe * float(fluid.sigma_ep) * v_te, _EPS)
        eta = M_E * n_p_safe * float(fluid.sigma_ep) * v_te / _safe_pos(ops, E_CHARGE * E_CHARGE * n_e, _EPS)
    else:
        beta = E_CHARGE * float(B) / (M_E * n_p_safe * float(fluid.sigma_ep) * v_te + _EPS)
        eta = M_E * n_p_safe * float(fluid.sigma_ep) * v_te / (E_CHARGE * E_CHARGE * n_e + _EPS)

    if is_maingo:
        q_factor = E_CHARGE * dot_N / _safe_pos(ops, I_0, 1e-12)
        q = q_factor * n_e / n_p_safe
    else:
        q_factor = E_CHARGE * dot_N / (I_0 + _EPS)
        q = q_factor * n_e / (n_p_safe + _EPS)
    b2 = beta * beta
    Z = b2 * (q - 1.0) - 1.0
    one_plus_z = 1.0 + Z
    if is_maingo:
        den = _safe_pos(ops, b2 * q, _EPS)
        F = b2 * (b2 + one_plus_z * one_plus_z) / (den * den)
    else:
        den = b2 + one_plus_z
        F = b2 * (b2 + one_plus_z * one_plus_z) / (den * den + _EPS)

    if is_maingo:
        v_p = dot_N / _safe_pos(ops, n_p_safe * A_safe, _EPS)
    else:
        v_p = dot_N / (n_p_safe * A_safe + _EPS)
    C = float(fluid.heavy_particle_mass_kg) * v_p * v_p / (3.0 * K_B)
    T_p = T_e_safe - C * F

    dbeta_dTe = -0.5 * beta / T_e_safe
    dbeta_dnp = -beta / n_p_safe

    den_ns = n_s - n_e
    den_ns_sq = den_ns * den_ns + _EPS
    df_dne = n_e * (2.0 * n_s - n_e) / den_ns_sq
    dK_dTe = K * (1.5 / T_e_safe + float(fluid.seed_ionization_energy_J) / (K_B * T_e_safe * T_e_safe))
    df_dne_safe = _safe_pos(ops, df_dne, _EPS) if is_maingo else (df_dne + _EPS)
    dne_dTe = dK_dTe / df_dne_safe
    df_dnp = -(n_e * n_e) * seed_safe / den_ns_sq
    dne_dnp = -df_dnp / df_dne_safe

    if is_maingo:
        dq_dTe = q_factor * dne_dTe / n_p_safe
        dq_dnp = q_factor * (dne_dnp * n_p_safe - n_e) / (n_p_safe * n_p_safe)
    else:
        dq_dTe = q_factor * dne_dTe / (n_p_safe + _EPS)
        dq_dnp = q_factor * (dne_dnp * n_p_safe - n_e) / (n_p_safe * n_p_safe + _EPS)
    dZ_dTe = 2.0 * beta * dbeta_dTe * (q - 1.0) + b2 * dq_dTe
    dZ_dnp = 2.0 * beta * dbeta_dnp * (q - 1.0) + b2 * dq_dnp

    dF_dTe = _df_dbeta(beta, Z) * dbeta_dTe + _df_dz(beta, Z) * dZ_dTe
    dF_dnp = _df_dbeta(beta, Z) * dbeta_dnp + _df_dz(beta, Z) * dZ_dnp

    dTp_dTe = 1.0 - C * dF_dTe
    if is_maingo:
        dTp_dnp = 2.0 * C * F / n_p_safe - C * dF_dnp
        dTp_dA = 2.0 * C * F / A_safe
    else:
        dTp_dnp = 2.0 * C * F / (n_p_safe + _EPS) - C * dF_dnp
        dTp_dA = 2.0 * C * F / (A_safe + _EPS)

    jfac = E_CHARGE * n_e * v_p
    J_x = I_0 / A_safe
    if is_maingo:
        J_y = -beta * one_plus_z / den * jfac
        E_x = -b2 * Z / den * eta * jfac
    else:
        J_y = -beta * one_plus_z / (den + _EPS) * jfac
        E_x = -b2 * Z / (den + _EPS) * eta * jfac
    nu_E = eta * 2.0 * E_CHARGE * E_CHARGE * n_e / float(fluid.heavy_particle_mass_kg)

    T_p_floor = _floored_pos(ops, T_p, _TP_MIN)
    c_s = ops.sqrt((5.0 / 3.0) * K_B * T_p_floor / float(fluid.heavy_particle_mass_kg) + _EPS)
    mach = v_p / (_safe_pos(ops, c_s, _EPS) if is_maingo else (c_s + _EPS))

    seed_density = _safe_pos(ops, seed_fraction, 1e-12) * n_p_safe
    f_I_raw = n_e / (seed_density if is_maingo else (seed_density + _EPS))
    f_I = _clip_range(ops, f_I_raw, _FION_MIN, _FION_MAX)
    delta_raw = T_e_safe / T_p_floor - 1.0
    delta = _floored_pos(ops, delta_raw, _DELTA_MIN)
    alpha = (K_B * T_e_safe / (2.0 * float(fluid.seed_ionization_energy_J))) * (2.0 - f_I) / (
        _safe_pos(ops, 1.0 - f_I, _EPS) if is_maingo else (1.0 - f_I + _EPS)
    )
    G = 4.0 * alpha * (2.0 + 1.0 / delta) * (1.0 + alpha * (1.0 + 1.0 / delta)) - b2

    return {
        "n_p_safe": n_p_safe,
        "T_e_safe": T_e_safe,
        "A_safe": A_safe,
        "seed_safe": seed_safe,
        "K": K,
        "n_s": n_s,
        "n_e": n_e,
        "beta": beta,
        "eta": eta,
        "Z": Z,
        "F": F,
        "v_p": v_p,
        "T_p": T_p,
        "dTp_dTe": dTp_dTe,
        "dTp_dnp": dTp_dnp,
        "dTp_dA": dTp_dA,
        "J_x": J_x,
        "J_y": J_y,
        "E_x": E_x,
        "mach": mach,
        "G": G,
        "nu_E": nu_E,
    }


def _inlet_design_generic(
    *,
    ops,
    n_p_in,
    T_e_in,
    Z_in,
    I_0,
    seed_fraction,
    B,
    inlet_A: float = _A_IN,
    working_fluid: WorkingFluidProfile = _DEFAULT_WORKING_FLUID_PROFILE,
):
    fluid = _normalize_working_fluid_profile(working_fluid)
    is_maingo = getattr(ops, "lb_func", None) is not None
    n_p_safe, T_e_safe, _, _, n_s, n_e = _saha_terms(
        ops=ops,
        n_p=n_p_in,
        T_e=T_e_in,
        seed_fraction=seed_fraction,
        working_fluid=fluid,
    )
    v_te = ops.sqrt(2.0 * K_B * T_e_safe / M_E)
    beta = E_CHARGE * float(B) / (
        _safe_pos(ops, M_E * n_p_safe * float(fluid.sigma_ep) * v_te, _EPS)
        if is_maingo
        else (M_E * n_p_safe * float(fluid.sigma_ep) * v_te + _EPS)
    )
    b2 = beta * beta
    den = _safe_pos(ops, b2 + 1.0 + Z_in, _EPS) if is_maingo else (b2 + 1.0 + Z_in)
    inlet_A_safe = float(max(float(inlet_A), _EPS))
    v_in = (I_0 / inlet_A_safe) * den / (
        _safe_pos(ops, b2 * E_CHARGE * n_e, _EPS)
        if is_maingo
        else (b2 * E_CHARGE * n_e + _EPS)
    )
    dot_N = n_p_safe * v_in * inlet_A_safe
    F = b2 * (b2 + (1.0 + Z_in) * (1.0 + Z_in)) / (
        (den * den) if is_maingo else (den * den + _EPS)
    )
    T_p_in = T_e_safe - float(fluid.heavy_particle_mass_kg) * v_in * v_in * F / (3.0 * K_B)
    T_p_floor = _floored_pos(ops, T_p_in, _TP_MIN)
    c_s = ops.sqrt((5.0 / 3.0) * K_B * T_p_floor / float(fluid.heavy_particle_mass_kg) + _EPS)
    mach = v_in / (_safe_pos(ops, c_s, _EPS) if is_maingo else (c_s + _EPS))
    f_I_raw = n_e / (_safe_pos(ops, n_s, _EPS) if is_maingo else (n_s + _EPS))
    f_I = _clip_range(ops, f_I_raw, _FION_MIN, _FION_MAX)
    delta = _floored_pos(ops, T_e_safe / T_p_floor - 1.0, _DELTA_MIN)
    alpha = (K_B * T_e_safe / (2.0 * float(fluid.seed_ionization_energy_J))) * (2.0 - f_I) / (
        _safe_pos(ops, 1.0 - f_I, _EPS) if is_maingo else (1.0 - f_I + _EPS)
    )
    G = 4.0 * alpha * (2.0 + 1.0 / delta) * (1.0 + alpha * (1.0 + 1.0 / delta)) - b2
    return {
        "n_p": n_p_safe,
        "n_e": n_e,
        "T_e": T_e_safe,
        "T_p": T_p_in,
        "Z": Z_in,
        "I_0": I_0,
        "dot_N": dot_N,
        "v_in": v_in,
        "seed_fraction": seed_fraction,
        "mach": mach,
        "G": G,
        "A_in": inlet_A_safe,
    }


def _dynamic_system_terms(
    *,
    ops,
    n_p,
    T_e,
    A,
    sigma,
    dot_N,
    I_0,
    seed_fraction,
    B,
    working_fluid: WorkingFluidProfile = _DEFAULT_WORKING_FLUID_PROFILE,
):
    fluid = _normalize_working_fluid_profile(working_fluid)
    closure = _closure_state(
        ops=ops,
        n_p=n_p,
        T_e=T_e,
        A=A,
        dot_N=dot_N,
        I_0=I_0,
        seed_fraction=seed_fraction,
        B=B,
        working_fluid=fluid,
    )
    v_p = closure["v_p"]
    T_p = closure["T_p"]
    n_p_safe = closure["n_p_safe"]
    A_safe = closure["A_safe"]
    v_p_safe = _safe_pos(ops, v_p, _EPS)
    M11 = (-float(fluid.heavy_particle_mass_kg) * v_p * v_p + K_B * T_p) + K_B * n_p_safe * closure["dTp_dnp"]
    M12 = K_B * n_p_safe * closure["dTp_dTe"]
    M13 = K_B * n_p_safe * closure["dTp_dA"] - float(fluid.heavy_particle_mass_kg) * n_p_safe * v_p * v_p / (A_safe + _EPS)
    E11 = -T_p + 1.5 * n_p_safe * closure["dTp_dnp"]
    E12 = 1.5 * n_p_safe * closure["dTp_dTe"]
    E13 = 1.5 * n_p_safe * closure["dTp_dA"]

    dA_dx = sigma * A_safe
    rhs_m = closure["J_y"] * float(B) - M13 * dA_dx
    rhs_e = (
        1.5 * closure["nu_E"] * closure["n_e"] * (closure["T_e_safe"] - T_p) / v_p_safe
        - E13 * dA_dx
    )
    det = M11 * E12 - M12 * E11
    return closure, {
        "M11": M11,
        "M12": M12,
        "M13": M13,
        "E11": E11,
        "E12": E12,
        "E13": E13,
        "dA_dx": dA_dx,
        "rhs_m": rhs_m,
        "rhs_e": rhs_e,
        "det": det,
    }


def _evaluate_midpoint_closures(
    *,
    ops,
    area_design: SplineAreaDesign,
    length: float,
    n_intervals: int,
    n_p_nodes,
    T_e_nodes,
    dot_N,
    I_0,
    seed_fraction,
    B,
    area_scale: float = 1.0,
    area_reference_x_norm: np.ndarray | None = None,
    area_reference_factor: np.ndarray | None = None,
    area_reference_sigma_logA: np.ndarray | None = None,
    working_fluid: WorkingFluidProfile = _DEFAULT_WORKING_FLUID_PROFILE,
):
    fluid = _normalize_working_fluid_profile(working_fluid)
    x_mid_norm = (np.arange(int(n_intervals), dtype=float) + 0.5) / float(int(n_intervals))
    area_mid = _evaluate_area_design_samples(
        ops=ops,
        area_design=area_design,
        length=float(length),
        x_norm=x_mid_norm,
        area_scale=float(area_scale),
        area_reference_x_norm=area_reference_x_norm,
        area_reference_factor=area_reference_factor,
        area_reference_sigma_logA=area_reference_sigma_logA,
    )
    closures_mid = []
    for k in range(int(n_intervals)):
        n_mid = 0.5 * (n_p_nodes[k] + n_p_nodes[k + 1])
        T_mid = 0.5 * (T_e_nodes[k] + T_e_nodes[k + 1])
        closure_mid, _ = _dynamic_system_terms(
            ops=ops,
            n_p=n_mid,
            T_e=T_mid,
            A=area_mid["A"][k],
            sigma=area_mid["sigma_logA"][k],
            dot_N=dot_N,
            I_0=I_0,
            seed_fraction=seed_fraction,
            B=B,
            working_fluid=fluid,
        )
        closures_mid.append(closure_mid)
    return area_mid, closures_mid


def _implicit_step_residuals(
    *,
    ops,
    n_prev,
    T_e_prev,
    n_next,
    T_e_next,
    dn_dx,
    dTe_dx,
    A_next,
    sigma_next,
    dot_N,
    I_0,
    seed_fraction,
    B,
    dx,
    working_fluid: WorkingFluidProfile = _DEFAULT_WORKING_FLUID_PROFILE,
):
    closure, terms = _dynamic_system_terms(
        ops=ops,
        n_p=n_next,
        T_e=T_e_next,
        A=A_next,
        sigma=sigma_next,
        dot_N=dot_N,
        I_0=I_0,
        seed_fraction=seed_fraction,
        B=B,
        working_fluid=working_fluid,
    )
    step_n = n_next - n_prev - float(dx) * dn_dx
    step_Te = T_e_next - T_e_prev - float(dx) * dTe_dx
    momentum = terms["M11"] * dn_dx + terms["M12"] * dTe_dx - terms["rhs_m"]
    energy = terms["E11"] * dn_dx + terms["E12"] * dTe_dx - terms["rhs_e"]
    return step_n, step_Te, momentum, energy, closure, terms


def _state_rhs(
    *,
    ops,
    n_p,
    T_e,
    A,
    sigma,
    dot_N,
    I_0,
    seed_fraction,
    B,
    working_fluid: WorkingFluidProfile = _DEFAULT_WORKING_FLUID_PROFILE,
):
    closure, terms = _dynamic_system_terms(
        ops=ops,
        n_p=n_p,
        T_e=T_e,
        A=A,
        sigma=sigma,
        dot_N=dot_N,
        I_0=I_0,
        seed_fraction=seed_fraction,
        B=B,
        working_fluid=working_fluid,
    )
    det_denom = _safe_signed_denom(ops, terms["det"], sign_hint="negative")
    dn_dx = (terms["rhs_m"] * terms["E12"] - terms["M12"] * terms["rhs_e"]) / det_denom
    dTe_dx = (terms["M11"] * terms["rhs_e"] - terms["rhs_m"] * terms["E11"]) / det_denom
    return dn_dx, dTe_dx, closure


def _rk4_rollout_generic(
    *,
    ops,
    n_p_in,
    T_e_in,
    Z_in,
    I_0,
    seed_fraction,
    area_design: SplineAreaDesign,
    B: float,
    length: float,
    n_intervals: int,
    area_scale: float = 1.0,
    area_reference_x_norm: np.ndarray | None = None,
    area_reference_factor: np.ndarray | None = None,
    area_reference_sigma_logA: np.ndarray | None = None,
    working_fluid: WorkingFluidProfile = _DEFAULT_WORKING_FLUID_PROFILE,
):
    fluid = _normalize_working_fluid_profile(working_fluid)
    inlet = _inlet_design_generic(
        ops=ops,
        n_p_in=n_p_in,
        T_e_in=T_e_in,
        Z_in=Z_in,
        I_0=I_0,
        seed_fraction=seed_fraction,
        B=B,
        inlet_A=float(area_scale),
        working_fluid=fluid,
    )
    x_norm = np.linspace(0.0, 1.0, int(n_intervals) + 1, dtype=float)
    basis_nodes, slopes_nodes = SplineAreaDesign.basis_matrices(x_norm)
    x_mid_norm = 0.5 * (x_norm[:-1] + x_norm[1:])
    basis_mid, slopes_mid = SplineAreaDesign.basis_matrices(x_mid_norm)
    params = [area_design.a1, area_design.a2, area_design.a3]
    ref_nodes, ref_sigma_nodes = _sample_area_reference(
        x_norm,
        area_reference_x_norm=area_reference_x_norm,
        area_reference_factor=area_reference_factor,
        area_reference_sigma_logA=area_reference_sigma_logA,
    )
    ref_mid, ref_sigma_mid = _sample_area_reference(
        x_mid_norm,
        area_reference_x_norm=area_reference_x_norm,
        area_reference_factor=area_reference_factor,
        area_reference_sigma_logA=area_reference_sigma_logA,
    )

    def _apply_basis(row):
        return row[0] * params[0] + row[1] * params[1] + row[2] * params[2]

    n_p_nodes = [n_p_in]
    T_e_nodes = [T_e_in]
    closures = []
    sigma_nodes = []
    A_nodes = []
    dx = float(length) / int(n_intervals)
    for k in range(int(n_intervals)):
        logA_k = _apply_basis(basis_nodes[k, :])
        sigma_k = float(ref_sigma_nodes[k]) + _apply_basis(slopes_nodes[k, :]) / float(length)
        A_k = float(area_scale) * float(ref_nodes[k]) * ops.exp(logA_k)
        dn1, dTe1, closure_k = _state_rhs(
            ops=ops,
            n_p=n_p_nodes[-1],
            T_e=T_e_nodes[-1],
            A=A_k,
            sigma=sigma_k,
            dot_N=inlet["dot_N"],
            I_0=I_0,
            seed_fraction=seed_fraction,
            B=B,
            working_fluid=fluid,
        )
        logA_mid = _apply_basis(basis_mid[k, :])
        sigma_mid = float(ref_sigma_mid[k]) + _apply_basis(slopes_mid[k, :]) / float(length)
        A_mid = float(area_scale) * float(ref_mid[k]) * ops.exp(logA_mid)
        n_mid_1 = n_p_nodes[-1] + 0.5 * dx * dn1
        Te_mid_1 = T_e_nodes[-1] + 0.5 * dx * dTe1
        dn2, dTe2, _ = _state_rhs(
            ops=ops,
            n_p=n_mid_1,
            T_e=Te_mid_1,
            A=A_mid,
            sigma=sigma_mid,
            dot_N=inlet["dot_N"],
            I_0=I_0,
            seed_fraction=seed_fraction,
            B=B,
            working_fluid=fluid,
        )
        n_mid_2 = n_p_nodes[-1] + 0.5 * dx * dn2
        Te_mid_2 = T_e_nodes[-1] + 0.5 * dx * dTe2
        dn3, dTe3, _ = _state_rhs(
            ops=ops,
            n_p=n_mid_2,
            T_e=Te_mid_2,
            A=A_mid,
            sigma=sigma_mid,
            dot_N=inlet["dot_N"],
            I_0=I_0,
            seed_fraction=seed_fraction,
            B=B,
            working_fluid=fluid,
        )
        logA_kp1 = _apply_basis(basis_nodes[k + 1, :])
        sigma_kp1 = float(ref_sigma_nodes[k + 1]) + _apply_basis(slopes_nodes[k + 1, :]) / float(length)
        A_kp1 = float(area_scale) * float(ref_nodes[k + 1]) * ops.exp(logA_kp1)
        n_end = n_p_nodes[-1] + dx * dn3
        Te_end = T_e_nodes[-1] + dx * dTe3
        dn4, dTe4, _ = _state_rhs(
            ops=ops,
            n_p=n_end,
            T_e=Te_end,
            A=A_kp1,
            sigma=sigma_kp1,
            dot_N=inlet["dot_N"],
            I_0=I_0,
            seed_fraction=seed_fraction,
            B=B,
            working_fluid=fluid,
        )
        n_next = n_p_nodes[-1] + dx * (dn1 + 2.0 * dn2 + 2.0 * dn3 + dn4) / 6.0
        Te_next = T_e_nodes[-1] + dx * (dTe1 + 2.0 * dTe2 + 2.0 * dTe3 + dTe4) / 6.0
        n_p_nodes.append(n_next)
        T_e_nodes.append(Te_next)
        closures.append(closure_k)
        sigma_nodes.append(sigma_k)
        A_nodes.append(A_k)

    logA_end = _apply_basis(basis_nodes[-1, :])
    sigma_end = float(ref_sigma_nodes[-1]) + _apply_basis(slopes_nodes[-1, :]) / float(length)
    A_end = float(area_scale) * float(ref_nodes[-1]) * ops.exp(logA_end)
    _, _, closure_end = _state_rhs(
        ops=ops,
        n_p=n_p_nodes[-1],
        T_e=T_e_nodes[-1],
        A=A_end,
        sigma=sigma_end,
        dot_N=inlet["dot_N"],
        I_0=I_0,
        seed_fraction=seed_fraction,
        B=B,
        working_fluid=fluid,
    )
    closures.append(closure_end)
    sigma_nodes.append(sigma_end)
    A_nodes.append(A_end)

    return {
        "inlet": inlet,
        "x": x_norm * float(length),
        "x_norm": x_norm,
        "n_p": n_p_nodes,
        "T_e": T_e_nodes,
        "A": A_nodes,
        "sigma_logA": sigma_nodes,
        "closures": closures,
    }


def _inlet_enthalpy_flux_generic(
    *,
    ops,
    inlet_n_p,
    inlet_T_p,
    inlet_T_e,
    inlet_n_e,
    inlet_v,
    inlet_A,
    working_fluid: WorkingFluidProfile = _DEFAULT_WORKING_FLUID_PROFILE,
):
    fluid = _normalize_working_fluid_profile(working_fluid)
    n_p_safe = _floored_pos(ops, inlet_n_p, 1.0)
    T_p_safe = _floored_pos(ops, inlet_T_p, _TP_MIN)
    T_e_safe = _floored_pos(ops, inlet_T_e, 1.0)
    n_e_safe = _max_op(ops, inlet_n_e, 0.0)
    v_safe = _floored_pos(ops, inlet_v, _EPS)
    A_safe = _floored_pos(ops, inlet_A, 1e-12)
    thermal_density = 2.5 * K_B * (n_p_safe * T_p_safe + n_e_safe * T_e_safe)
    kinetic_density = 0.5 * float(fluid.heavy_particle_mass_kg) * n_p_safe * v_safe * v_safe
    return A_safe * v_safe * (thermal_density + kinetic_density)


def _design_score_generic(
    *,
    ops,
    outlet_T_e,
    outlet_T_p,
    outlet_n_p,
    outlet_n_e,
    inlet_T_e,
    inlet_T_p,
    inlet_mach,
    power_density_nodes,
    x_nodes: np.ndarray,
    seed_fraction,
    B: float,
    length: float,
    objective_profile: str = OBJECTIVE_PROFILE_LAB_POC_V2,
    inlet_n_p=None,
    inlet_n_e=None,
    inlet_v=None,
    inlet_A: float = _A_IN,
    working_fluid: WorkingFluidProfile = _DEFAULT_WORKING_FLUID_PROFILE,
):
    objective_profile = _normalize_objective_profile(objective_profile)
    fluid = _normalize_working_fluid_profile(working_fluid)
    outlet_delta_te_per_kK = (outlet_T_e - inlet_T_e) / 1e3
    outlet_delta_ratio = outlet_T_e / _floored_pos(ops, outlet_T_p, _TP_MIN) - 1.0
    outlet_f_ion = outlet_n_e / (seed_fraction * outlet_n_p + _EPS)
    outlet_mhd_output_per_100MWe = 0.0
    for j in range(x_nodes.size - 1):
        outlet_mhd_output_per_100MWe += 0.5 * (x_nodes[j + 1] - x_nodes[j]) * (
            power_density_nodes[j] + power_density_nodes[j + 1]
        )
    if objective_profile == OBJECTIVE_PROFILE_ENTHALPY_EXTRACTION:
        if inlet_n_p is None or inlet_n_e is None or inlet_v is None:
            raise ValueError("enthalpy_extraction objective requires inlet_n_p, inlet_n_e, and inlet_v")
        inlet_enthalpy_flux_W = _inlet_enthalpy_flux_generic(
            ops=ops,
            inlet_n_p=inlet_n_p,
            inlet_T_p=inlet_T_p,
            inlet_T_e=inlet_T_e,
            inlet_n_e=inlet_n_e,
            inlet_v=inlet_v,
            inlet_A=float(inlet_A),
            working_fluid=fluid,
        )
        mhd_output_power_W = outlet_mhd_output_per_100MWe * 1e8
        return 100.0 * mhd_output_power_W / _floored_pos(ops, inlet_enthalpy_flux_W, _EPS)

    weights = _design_value_weights_lab_poc_v2_objective()
    inlet_delta_ratio = inlet_T_e / _floored_pos(ops, inlet_T_p, _TP_MIN) - 1.0
    device_length_per_5m = float(length) / 5.0
    return (
        float(weights.outlet_delta_te_per_kK) * outlet_delta_te_per_kK
        + float(weights.outlet_delta_ratio) * outlet_delta_ratio
        + float(weights.outlet_f_ion) * outlet_f_ion
        + float(weights.outlet_mhd_output_per_100MWe) * outlet_mhd_output_per_100MWe
        - float(weights.inlet_delta_ratio_penalty) * inlet_delta_ratio
        - float(weights.inlet_mach_penalty) * inlet_mach
        - float(weights.magnetic_field_T_penalty) * abs(float(B))
        - float(weights.device_length_per_5m_penalty) * device_length_per_5m
    )


def evaluate_inlet_design_numeric(
    *,
    n_p_in: float,
    T_e_in: float,
    Z_in: float,
    I_0: float,
    seed_fraction: float,
    B: float,
    inlet_area: float = _A_IN,
    working_fluid_profile: str | WorkingFluidProfile | None = None,
) -> InletDesign:
    ops = _ops_for_numeric()
    fluid = _normalize_working_fluid_profile(working_fluid_profile)
    inlet = _inlet_design_generic(
        ops=ops,
        n_p_in=float(n_p_in),
        T_e_in=float(T_e_in),
        Z_in=float(Z_in),
        I_0=float(I_0),
        seed_fraction=float(seed_fraction),
        B=float(B),
        inlet_A=float(inlet_area),
        working_fluid=fluid,
    )
    return InletDesign(
        n_p=float(inlet["n_p"]),
        T_e=float(inlet["T_e"]),
        T_p=float(inlet["T_p"]),
        Z=float(inlet["Z"]),
        I_0=float(inlet["I_0"]),
        dot_N=float(inlet["dot_N"]),
        v_in=float(inlet["v_in"]),
        seed_fraction=float(inlet["seed_fraction"]),
        mach=float(inlet["mach"]),
        velikhov_margin=float(inlet["G"]),
        A_in=float(inlet["A_in"]),
    )
