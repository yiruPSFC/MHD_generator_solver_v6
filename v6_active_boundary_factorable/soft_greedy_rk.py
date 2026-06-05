from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from v6_firedrake_reduced.design import CaseConfig, DesignVector
from v6_firedrake_reduced.geometry import LogAreaSplineControl
from v6_firedrake_reduced.transport import working_fluid_for_config
from v6_maingo_casadi.constants import _EPS
from v6_maingo_casadi.numerics import _clip_range, _max_op, _min_op, _ops_for_numeric
from v6_maingo_casadi.physics import _dynamic_system_terms, _inlet_design_generic
from v6_maingo_casadi.profiles import WorkingFluidProfile


@dataclass(frozen=True)
class FactorableState:
    log_n: Any
    log_Te: Any
    logA: Any


@dataclass(frozen=True)
class FactorableParams:
    dot_N: Any
    I_0: Any
    seed_fraction: Any
    B: Any
    area_scale_m2: float
    working_fluid: Any


@dataclass(frozen=True)
class SoftGreedySettings:
    dx: float = 1.0e-3
    n_steps: int = 8
    direction: int = 1
    sigma_min: float = -0.5
    sigma_max: float = 0.5
    curvature_max: float | None = 8.0
    logA_min: float = LogAreaSplineControl.lower_bound()
    logA_max: float = LogAreaSplineControl.upper_bound()
    selector_eps: float = 1.0e-4
    det_gate_eps: float = 1.0e-2
    mach_gate_eps: float = 1.0e-3
    sonic_den_eps: float = 1.0e-12
    rhs_det_eps: float = 1.0e-6
    row_norm_eps: float = 1.0e-30
    eigenvector_eps: float = 1.0e-30
    use_mach_gate: bool = True
    use_null_continuity: bool = True
    clip_final_sigma: bool = False


def factorable_params_from_config(
    config: CaseConfig,
    *,
    ops=None,
    design: DesignVector | None = None,
) -> FactorableParams:
    """Build the physical constants that are fixed during a factorable rollout."""

    ops = _ops_for_numeric() if ops is None else ops
    design = config.design if design is None else design
    fluid = _maingo_working_fluid_profile(working_fluid_for_config(config))
    inlet = _inlet_design_generic(
        ops=ops,
        n_p_in=ops.exp(design.log_n_p_in),
        T_e_in=design.T_e_in,
        Z_in=design.Z_in,
        I_0=design.I_0,
        seed_fraction=ops.exp(design.log_seed_fraction),
        B=design.B_T,
        inlet_A=float(config.area_scale_m2),
        working_fluid=fluid,
    )
    return FactorableParams(
        dot_N=inlet["dot_N"],
        I_0=design.I_0,
        seed_fraction=ops.exp(design.log_seed_fraction),
        B=design.B_T,
        area_scale_m2=float(config.area_scale_m2),
        working_fluid=fluid,
    )


def _maingo_working_fluid_profile(profile: Any) -> WorkingFluidProfile:
    """Convert compatible legacy profile objects into the MAiNGO profile class."""

    if isinstance(profile, WorkingFluidProfile):
        return profile
    return WorkingFluidProfile(
        key=str(profile.key),
        working_gas=str(profile.working_gas),
        seed_species=str(profile.seed_species),
        heavy_particle_mass_amu=float(profile.heavy_particle_mass_amu),
        seed_ionization_energy_eV=float(profile.seed_ionization_energy_eV),
        sigma_ep=float(profile.sigma_ep),
        sigma_ep_note=str(profile.sigma_ep_note),
    )


def initial_state_from_design(design: DesignVector) -> FactorableState:
    return FactorableState(
        log_n=float(design.log_n_p_in),
        log_Te=float(math.log(max(float(design.T_e_in), 1.0))),
        logA=0.0,
    )


def state_values(*, ops, state: FactorableState, params: FactorableParams) -> dict[str, Any]:
    n_p = ops.exp(state.log_n)
    T_e = ops.exp(state.log_Te)
    A = float(params.area_scale_m2) * ops.exp(state.logA)
    return {"n_p": n_p, "T_e": T_e, "A": A}


def _soft_sign(ops, value, eps: float):
    return value / ops.sqrt(value * value + float(eps) * float(eps))


def _soft_select_by_preference(ops, *, preference, lower, upper, eps: float):
    weight_upper = 0.5 * (1.0 + _soft_sign(ops, preference, eps))
    return weight_upper * upper + (1.0 - weight_upper) * lower, weight_upper


def sigma_interval(
    *,
    ops,
    state: FactorableState,
    sigma_prev,
    settings: SoftGreedySettings,
) -> dict[str, Any]:
    step = float(settings.direction) * float(settings.dx)
    sigma_lo = float(settings.sigma_min)
    sigma_hi = float(settings.sigma_max)
    if settings.curvature_max is not None and np.isfinite(float(settings.curvature_max)):
        width = abs(float(settings.curvature_max))
        sigma_lo = _max_op(ops, sigma_lo, sigma_prev - width)
        sigma_hi = _min_op(ops, sigma_hi, sigma_prev + width)
    area_low = (float(settings.logA_min) - state.logA) / step
    area_high = (float(settings.logA_max) - state.logA) / step
    sigma_lo = _max_op(ops, sigma_lo, _min_op(ops, area_low, area_high))
    sigma_hi = _min_op(ops, sigma_hi, _max_op(ops, area_low, area_high))
    return {"lo": sigma_lo, "hi": sigma_hi}


def _primitive_components(
    *,
    ops,
    state: FactorableState,
    params: FactorableParams,
) -> dict[str, Any]:
    values = state_values(ops=ops, state=state, params=params)
    closure, terms = _dynamic_system_terms(
        ops=ops,
        n_p=values["n_p"],
        T_e=values["T_e"],
        A=values["A"],
        sigma=0.0,
        dot_N=params.dot_N,
        I_0=params.I_0,
        seed_fraction=params.seed_fraction,
        B=params.B,
        working_fluid=params.working_fluid,
    )
    return {
        "values": values,
        "closure": closure,
        "terms": terms,
        "f0_m": terms["rhs_m"],
        "f0_e": terms["rhs_e"],
        "f1_m": -terms["M13"],
        "f1_e": -terms["E13"],
    }


def _regularized_ratio(ops, numerator, denominator, eps: float):
    return numerator * denominator / (denominator * denominator + float(eps) * float(eps))


def sonic_sigma_chart(
    *,
    ops=None,
    state: FactorableState,
    params: FactorableParams,
    settings: SoftGreedySettings = SoftGreedySettings(),
) -> dict[str, Any]:
    """Return a factorable 2x2 left-null approximation to sonic-compatible sigma."""

    ops = _ops_for_numeric() if ops is None else ops
    comp = _primitive_components(ops=ops, state=state, params=params)
    terms = comp["terms"]
    A = comp["values"]["A"]
    f0_m = comp["f0_m"]
    f0_e = comp["f0_e"]
    f1_m = comp["f1_m"]
    f1_e = comp["f1_e"]

    s11 = terms["M11"] * terms["M11"] + terms["M12"] * terms["M12"]
    s12 = terms["M11"] * terms["E11"] + terms["M12"] * terms["E12"]
    s22 = terms["E11"] * terms["E11"] + terms["E12"] * terms["E12"]
    eig_gap = ops.sqrt((s11 - s22) * (s11 - s22) + 4.0 * s12 * s12)
    lambda_min = 0.5 * (s11 + s22 - eig_gap)

    # Two algebraic charts for the smallest left-singular vector of D.  This is
    # factorable and avoids the row-null approximation failing when det is small
    # but nonzero.
    ell1_m = s12
    ell1_e = lambda_min - s11
    ell2_m = lambda_min - s22
    ell2_e = s12

    ell1_f0 = ell1_m * f0_m + ell1_e * f0_e
    ell1_f1 = ell1_m * f1_m + ell1_e * f1_e
    ell2_f0 = ell2_m * f0_m + ell2_e * f0_e
    ell2_f1 = ell2_m * f1_m + ell2_e * f1_e

    denom1 = A * ell1_f1
    denom2 = A * ell2_f1
    sigma1 = -_regularized_ratio(ops, ell1_f0, denom1, settings.sonic_den_eps)
    sigma2 = -_regularized_ratio(ops, ell2_f0, denom2, settings.sonic_den_eps)
    weight1 = (ell1_m * ell1_m + ell1_e * ell1_e) * ell1_f1 * ell1_f1
    weight2 = (ell2_m * ell2_m + ell2_e * ell2_e) * ell2_f1 * ell2_f1
    sigma = (weight1 * sigma1 + weight2 * sigma2) / (weight1 + weight2 + float(settings.sonic_den_eps))

    compat1 = ell1_f0 + A * sigma * ell1_f1
    compat2 = ell2_f0 + A * sigma * ell2_f1
    scale1 = _max_op(ops, 1.0, _max_op(ops, ops.fabs(ell1_f0), ops.fabs(A * sigma * ell1_f1)))
    scale2 = _max_op(ops, 1.0, _max_op(ops, ops.fabs(ell2_f0), ops.fabs(A * sigma * ell2_f1)))
    det = terms["M11"] * terms["E12"] - terms["M12"] * terms["E11"]
    return {
        **comp,
        "sigma_sonic": sigma,
        "sigma_chart_1": sigma1,
        "sigma_chart_2": sigma2,
        "chart_weight_1": weight1,
        "chart_weight_2": weight2,
        "det": det,
        "lambda_min_DDt": lambda_min,
        "eig_gap_DDt": eig_gap,
        "ell1_f0": ell1_f0,
        "ell1_f1": ell1_f1,
        "ell2_f0": ell2_f0,
        "ell2_f1": ell2_f1,
        "compat1_scaled": compat1 / scale1,
        "compat2_scaled": compat2 / scale2,
    }


def _delta_gradients(closure: dict[str, Any]) -> tuple[Any, Any, Any, Any]:
    T_e = closure["T_e_safe"]
    T_p = closure["T_p"]
    inv_tp = 1.0 / (T_p + _EPS)
    inv_tp2 = 1.0 / (T_p * T_p + _EPS)
    d_delta_dn = -T_e * inv_tp2 * closure["dTp_dnp"]
    d_delta_dTe = inv_tp - T_e * inv_tp2 * closure["dTp_dTe"]
    d_delta_dA = -T_e * inv_tp2 * closure["dTp_dA"]
    delta = T_e * inv_tp - 1.0
    return delta, d_delta_dn, d_delta_dTe, d_delta_dA


def _rhs_from_components(
    *,
    ops,
    terms: dict[str, Any],
    A,
    sigma,
    f0_m,
    f0_e,
    f1_m,
    f1_e,
    det_eps: float,
) -> tuple[Any, Any, Any, Any, Any]:
    rhs_m = f0_m + A * sigma * f1_m
    rhs_e = f0_e + A * sigma * f1_e
    det = terms["M11"] * terms["E12"] - terms["M12"] * terms["E11"]
    inv_det = det / (det * det + float(det_eps) * float(det_eps))
    dn_dx = (rhs_m * terms["E12"] - terms["M12"] * rhs_e) * inv_det
    dTe_dx = (terms["M11"] * rhs_e - rhs_m * terms["E11"]) * inv_det
    return dn_dx, dTe_dx, rhs_m, rhs_e, det


def _sonic_rhs(
    *,
    ops,
    terms: dict[str, Any],
    rhs_m,
    rhs_e,
    previous_rhs: tuple[Any, Any] | None,
    settings: SoftGreedySettings,
) -> tuple[Any, Any]:
    row1_norm = terms["M11"] * terms["M11"] + terms["M12"] * terms["M12"]
    row2_norm = terms["E11"] * terms["E11"] + terms["E12"] * terms["E12"]
    dn1 = terms["M11"] * rhs_m / (row1_norm + float(settings.row_norm_eps))
    dTe1 = terms["M12"] * rhs_m / (row1_norm + float(settings.row_norm_eps))
    dn2 = terms["E11"] * rhs_e / (row2_norm + float(settings.row_norm_eps))
    dTe2 = terms["E12"] * rhs_e / (row2_norm + float(settings.row_norm_eps))
    dn_base = (row1_norm * dn1 + row2_norm * dn2) / (row1_norm + row2_norm + float(settings.row_norm_eps))
    dTe_base = (row1_norm * dTe1 + row2_norm * dTe2) / (row1_norm + row2_norm + float(settings.row_norm_eps))
    if previous_rhs is None or not bool(settings.use_null_continuity):
        return dn_base, dTe_base

    z1_n = -terms["M12"]
    z1_t = terms["M11"]
    z2_n = -terms["E12"]
    z2_t = terms["E11"]
    z_n = (row1_norm * z1_n + row2_norm * z2_n) / (row1_norm + row2_norm + float(settings.row_norm_eps))
    z_t = (row1_norm * z1_t + row2_norm * z2_t) / (row1_norm + row2_norm + float(settings.row_norm_eps))
    z_norm = z_n * z_n + z_t * z_t
    alpha = ((previous_rhs[0] - dn_base) * z_n + (previous_rhs[1] - dTe_base) * z_t) / (
        z_norm + float(settings.row_norm_eps)
    )
    return dn_base + alpha * z_n, dTe_base + alpha * z_t


def soft_greedy_step(
    *,
    ops=None,
    state: FactorableState,
    sigma_prev,
    params: FactorableParams,
    settings: SoftGreedySettings = SoftGreedySettings(),
    previous_rhs: tuple[Any, Any] | None = None,
) -> dict[str, Any]:
    ops = _ops_for_numeric() if ops is None else ops
    interval = sigma_interval(ops=ops, state=state, sigma_prev=sigma_prev, settings=settings)
    chart = sonic_sigma_chart(ops=ops, state=state, params=params, settings=settings)
    values = chart["values"]
    closure = chart["closure"]
    terms = chart["terms"]
    _, d_delta_dn, d_delta_dTe, d_delta_dA = _delta_gradients(closure)

    dn_dsigma, dTe_dsigma, _, _, _ = _rhs_from_components(
        ops=ops,
        terms=terms,
        A=values["A"],
        sigma=1.0,
        f0_m=0.0,
        f0_e=0.0,
        f1_m=chart["f1_m"],
        f1_e=chart["f1_e"],
        det_eps=settings.rhs_det_eps,
    )
    preference = d_delta_dn * dn_dsigma + d_delta_dTe * dTe_dsigma + d_delta_dA * values["A"]
    sigma_greedy, endpoint_weight = _soft_select_by_preference(
        ops,
        preference=preference,
        lower=interval["lo"],
        upper=interval["hi"],
        eps=float(settings.selector_eps),
    )

    det_gate = float(settings.det_gate_eps) * float(settings.det_gate_eps) / (
        chart["det"] * chart["det"] + float(settings.det_gate_eps) * float(settings.det_gate_eps)
    )
    if settings.use_mach_gate:
        mach_gate = float(settings.mach_gate_eps) * float(settings.mach_gate_eps) / (
            (closure["mach"] - 1.0) * (closure["mach"] - 1.0)
            + float(settings.mach_gate_eps) * float(settings.mach_gate_eps)
        )
        sonic_gate = 1.0 - (1.0 - det_gate) * (1.0 - mach_gate)
    else:
        mach_gate = 0.0
        sonic_gate = det_gate

    sigma_blended = (1.0 - sonic_gate) * sigma_greedy + sonic_gate * chart["sigma_sonic"]
    sigma_final = (
        _clip_range(ops, sigma_blended, interval["lo"], interval["hi"])
        if bool(settings.clip_final_sigma)
        else sigma_blended
    )
    dn_reg, dTe_reg, rhs_m, rhs_e, _ = _rhs_from_components(
        ops=ops,
        terms=terms,
        A=values["A"],
        sigma=sigma_final,
        f0_m=chart["f0_m"],
        f0_e=chart["f0_e"],
        f1_m=chart["f1_m"],
        f1_e=chart["f1_e"],
        det_eps=settings.rhs_det_eps,
    )
    dn_sonic, dTe_sonic = _sonic_rhs(
        ops=ops,
        terms=terms,
        rhs_m=rhs_m,
        rhs_e=rhs_e,
        previous_rhs=previous_rhs,
        settings=settings,
    )
    dn_dx = (1.0 - sonic_gate) * dn_reg + sonic_gate * dn_sonic
    dTe_dx = (1.0 - sonic_gate) * dTe_reg + sonic_gate * dTe_sonic

    step = float(settings.direction) * float(settings.dx)
    n_safe = values["n_p"]
    Te_safe = values["T_e"]
    next_state = FactorableState(
        log_n=state.log_n + step * dn_dx / (n_safe + _EPS),
        log_Te=state.log_Te + step * dTe_dx / (Te_safe + _EPS),
        logA=state.logA + step * sigma_final,
    )
    next_values = state_values(ops=ops, state=next_state, params=params)
    next_closure, next_terms = _dynamic_system_terms(
        ops=ops,
        n_p=next_values["n_p"],
        T_e=next_values["T_e"],
        A=next_values["A"],
        sigma=sigma_final,
        dot_N=params.dot_N,
        I_0=params.I_0,
        seed_fraction=params.seed_fraction,
        B=params.B,
        working_fluid=params.working_fluid,
    )
    step_momentum = next_terms["M11"] * dn_dx + next_terms["M12"] * dTe_dx - next_terms["rhs_m"]
    step_energy = next_terms["E11"] * dn_dx + next_terms["E12"] * dTe_dx - next_terms["rhs_e"]
    m_scale = _max_op(
        ops,
        1.0,
        _max_op(
            ops,
            ops.fabs(next_terms["M11"] * next_values["n_p"] / (abs(step) + _EPS)),
            _max_op(
                ops,
                ops.fabs(next_terms["M12"] * next_values["T_e"] / (abs(step) + _EPS)),
                ops.fabs(next_terms["rhs_m"]),
            ),
        ),
    )
    e_scale = _max_op(
        ops,
        1.0,
        _max_op(
            ops,
            ops.fabs(next_terms["E11"] * next_values["n_p"] / (abs(step) + _EPS)),
            _max_op(
                ops,
                ops.fabs(next_terms["E12"] * next_values["T_e"] / (abs(step) + _EPS)),
                ops.fabs(next_terms["rhs_e"]),
            ),
        ),
    )
    return {
        "state": state,
        "next_state": next_state,
        "sigma": sigma_final,
        "sigma_greedy": sigma_greedy,
        "sigma_sonic": chart["sigma_sonic"],
        "sigma_lo": interval["lo"],
        "sigma_hi": interval["hi"],
        "sigma_lower_margin": sigma_final - interval["lo"],
        "sigma_upper_margin": interval["hi"] - sigma_final,
        "endpoint_weight_upper": endpoint_weight,
        "preference": preference,
        "sonic_gate": sonic_gate,
        "det_gate": det_gate,
        "mach_gate": mach_gate,
        "det": chart["det"],
        "mach": closure["mach"],
        "G": closure["G"],
        "T_p": closure["T_p"],
        "next_G": next_closure["G"],
        "next_T_p": next_closure["T_p"],
        "dn_dx": dn_dx,
        "dTe_dx": dTe_dx,
        "dn_reg": dn_reg,
        "dTe_reg": dTe_reg,
        "dn_sonic": dn_sonic,
        "dTe_sonic": dTe_sonic,
        "compat1_scaled": chart["compat1_scaled"],
        "compat2_scaled": chart["compat2_scaled"],
        "step_momentum": step_momentum,
        "step_energy": step_energy,
        "step_momentum_scaled": step_momentum / m_scale,
        "step_energy_scaled": step_energy / e_scale,
    }


def rollout_soft_greedy(
    *,
    ops=None,
    state: FactorableState,
    sigma_initial,
    params: FactorableParams,
    settings: SoftGreedySettings = SoftGreedySettings(),
) -> dict[str, Any]:
    ops = _ops_for_numeric() if ops is None else ops
    states = [state]
    segments = []
    sigma_prev = sigma_initial
    previous_rhs = None
    for k in range(int(settings.n_steps)):
        segment = soft_greedy_step(
            ops=ops,
            state=states[-1],
            sigma_prev=sigma_prev,
            params=params,
            settings=settings,
            previous_rhs=previous_rhs,
        )
        segments.append({"k": k, **segment})
        states.append(segment["next_state"])
        sigma_prev = segment["sigma"]
        previous_rhs = (segment["dn_dx"], segment["dTe_dx"])
    return {"states": states, "segments": segments, "settings": settings}


def finite_numeric(value: Any) -> bool:
    return bool(np.isfinite(float(value)))
