from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from v6_maingo_casadi.constants import (
    E_CHARGE,
    K_B,
    M_E,
    _DELTA_MIN,
    _EPS,
    _FION_MAX,
    _FION_MIN,
    _G_HARD_MARGIN,
    _TP_MIN,
)
from v6_maingo_casadi.geometry import SplineAreaDesign
from v6_maingo_casadi.implicit import ImplicitResidualScales
from v6_maingo_casadi.numerics import _clip_range, _floored_pos, _max_op, _ops_for_numeric, _reduce_min, _safe_pos
from v6_maingo_casadi.physics import (
    _dynamic_system_terms,
    _implicit_step_residuals,
    _inlet_design_generic,
    _saha_terms,
)
from v6_maingo_casadi.profiles import WorkingFluidProfile, _normalize_working_fluid_profile
from v6_maingo_casadi.reduced_implicit import _fixed_gauss_newton_log_state_2d

from .geometry import MachSplineDesign


MACH_DECISION_NAMES = (
    "log_n_p_in",
    "T_e_in",
    "Z_in",
    "I_0",
    "log_seed_fraction",
    "m1",
    "m2",
    "m3",
)


@dataclass(frozen=True)
class MachReducedConfig:
    B: float
    length: float
    area_scale_m2: float
    working_fluid: WorkingFluidProfile

    @classmethod
    def from_summary_and_profile(cls, summary_path: str | Path, profile_path: str | Path) -> "MachReducedConfig":
        summary_path = Path(summary_path)
        profile_path = Path(profile_path)
        summary = json.loads(summary_path.read_text())
        baseline = dict(summary.get("baseline_seed", {}) or {})
        fluid_payload = summary.get("working_fluid_profile") or summary.get("working_fluid") or baseline.get("working_fluid")
        with np.load(profile_path) as data:
            x = np.asarray(data["x"], dtype=float)
            A = np.asarray(data["A"], dtype=float)
        B = summary.get("B", baseline.get("B"))
        if B is None:
            raise ValueError(f"cannot infer B from {summary_path}")
        length = summary.get("L", baseline.get("L", float(x[-1] - x[0])))
        area_scale = summary.get("area_scale_m2", baseline.get("area_scale_m2", float(A[0])))
        return cls(
            B=float(B),
            length=float(length),
            area_scale_m2=float(area_scale),
            working_fluid=_normalize_working_fluid_profile(fluid_payload),
        )


@dataclass(frozen=True)
class MachReducedRollout:
    decision_vector: dict[str, Any]
    inlet: dict[str, Any]
    mach_nodes: dict[str, Any]
    area_nodes: dict[str, Any]
    n_p_nodes: list[Any]
    T_e_nodes: list[Any]
    dn_dx: list[Any]
    dTe_dx: list[Any]
    scaled_momentum_residuals: list[Any]
    scaled_energy_residuals: list[Any]
    closures: list[dict[str, Any]]
    terms_by_node: list[dict[str, Any]]
    max_abs_scaled_residual: Any
    min_abs_det: Any


def _reduce_max(ops, values: list[Any]):
    if not values:
        raise ValueError("cannot reduce an empty list.")
    acc = values[0]
    for value in values[1:]:
        acc = _max_op(ops, acc, value)
    return acc


def _positive_denom(ops, value, floor: float):
    return _safe_pos(ops, value, float(floor))


def _mach_design_nodes_generic(
    *,
    ops,
    mach_design: MachSplineDesign,
    length: float,
    n_intervals: int,
    mach_in,
) -> dict[str, Any]:
    x_norm = np.linspace(0.0, 1.0, int(n_intervals) + 1, dtype=float)
    basis, slopes = SplineAreaDesign.basis_matrices(x_norm)
    params = [mach_design.m1, mach_design.m2, mach_design.m3]
    mach_nodes = []
    dlogM_dx = []
    log_ratios = []
    for idx in range(x_norm.size):
        log_ratio = basis[idx, 0] * params[0] + basis[idx, 1] * params[1] + basis[idx, 2] * params[2]
        slope = (slopes[idx, 0] * params[0] + slopes[idx, 1] * params[1] + slopes[idx, 2] * params[2]) / float(length)
        log_ratios.append(log_ratio)
        dlogM_dx.append(slope)
        mach_nodes.append(mach_in * ops.exp(log_ratio))
    return {
        "x_norm": x_norm,
        "x": x_norm * float(length),
        "log_mach_ratio": log_ratios,
        "mach": mach_nodes,
        "dlogM_dx": dlogM_dx,
    }


def _mach_area_closure_generic(
    *,
    ops,
    n_p,
    T_e,
    mach,
    dot_N,
    I_0,
    seed_fraction,
    B: float,
    working_fluid: WorkingFluidProfile,
) -> dict[str, Any]:
    fluid = _normalize_working_fluid_profile(working_fluid)
    is_maingo = getattr(ops, "lb_func", None) is not None
    n_p_safe, T_e_safe, seed_safe, _, n_s, n_e = _saha_terms(
        ops=ops,
        n_p=n_p,
        T_e=T_e,
        seed_fraction=seed_fraction,
        working_fluid=fluid,
    )
    M = _safe_pos(ops, mach, 1e-12)
    M2 = M * M
    v_te = ops.sqrt(2.0 * K_B * T_e_safe / M_E)
    beta = E_CHARGE * float(B) / (
        _positive_denom(ops, M_E * n_p_safe * float(fluid.sigma_ep) * v_te, _EPS)
        if is_maingo
        else (M_E * n_p_safe * float(fluid.sigma_ep) * v_te + _EPS)
    )
    eta = M_E * n_p_safe * float(fluid.sigma_ep) * v_te / (
        _positive_denom(ops, E_CHARGE * E_CHARGE * n_e, _EPS)
        if is_maingo
        else (E_CHARGE * E_CHARGE * n_e + _EPS)
    )
    q_factor = E_CHARGE * dot_N / (_positive_denom(ops, I_0, 1e-12) if is_maingo else (I_0 + _EPS))
    q = q_factor * n_e / (_positive_denom(ops, n_p_safe, 1.0) if is_maingo else (n_p_safe + _EPS))
    b2 = beta * beta
    Z = b2 * (q - 1.0) - 1.0
    one_plus_z = 1.0 + Z
    den = _positive_denom(ops, b2 * q, _EPS) if is_maingo else (b2 * q + _EPS)
    F = b2 * (b2 + one_plus_z * one_plus_z) / (den * den)
    tp_denom = 9.0 + (_max_op(ops, 5.0 * M2 * F, 0.0) if is_maingo else 5.0 * M2 * F)
    T_p = 9.0 * T_e_safe / tp_denom
    v_p = ops.sqrt(5.0 * K_B * T_p * M2 / (3.0 * float(fluid.heavy_particle_mass_kg)) + _EPS)
    A = dot_N / (_positive_denom(ops, n_p_safe * v_p, _EPS) if is_maingo else (n_p_safe * v_p + _EPS))
    J_x = I_0 / (_positive_denom(ops, A, _EPS) if is_maingo else (A + _EPS))
    den_current = den
    jfac = E_CHARGE * n_e * v_p
    J_y = -beta * one_plus_z / den_current * jfac
    E_x = -b2 * Z / den_current * eta * jfac
    nu_E = eta * 2.0 * E_CHARGE * E_CHARGE * n_e / float(fluid.heavy_particle_mass_kg)

    T_p_floor = _floored_pos(ops, T_p, _TP_MIN)
    f_I_raw = n_e / (_positive_denom(ops, n_s, _EPS) if is_maingo else (n_s + _EPS))
    f_I = _clip_range(ops, f_I_raw, _FION_MIN, _FION_MAX)
    T_p_denom = _positive_denom(ops, T_p_floor, _TP_MIN) if is_maingo else T_p_floor
    delta = _floored_pos(ops, T_e_safe / T_p_denom - 1.0, _DELTA_MIN)
    alpha = (K_B * T_e_safe / (2.0 * float(fluid.seed_ionization_energy_J))) * (2.0 - f_I) / (
        _positive_denom(ops, 1.0 - f_I, _EPS) if is_maingo else (1.0 - f_I + _EPS)
    )
    delta_denom = _positive_denom(ops, delta, _DELTA_MIN) if is_maingo else delta
    G = 4.0 * alpha * (2.0 + 1.0 / delta_denom) * (1.0 + alpha * (1.0 + 1.0 / delta_denom)) - b2
    return {
        "n_p_safe": n_p_safe,
        "T_e_safe": T_e_safe,
        "A_safe": A,
        "seed_safe": seed_safe,
        "n_s": n_s,
        "n_e": n_e,
        "beta": beta,
        "eta": eta,
        "Z": Z,
        "F": F,
        "v_p": v_p,
        "T_p": T_p,
        "J_x": J_x,
        "J_y": J_y,
        "E_x": E_x,
        "mach": M,
        "G": G,
        "nu_E": nu_E,
    }


def _scaled_momentum_energy_residuals_mach(
    *,
    ops,
    n_prev,
    T_e_prev,
    A_prev,
    n_next,
    T_e_next,
    mach_next,
    dot_N,
    I_0,
    seed_fraction,
    B: float,
    dx: float,
    momentum_scale: float,
    energy_scale: float,
    working_fluid: WorkingFluidProfile,
) -> tuple[Any, Any, dict[str, Any], dict[str, Any], Any, Any, Any, Any]:
    mach_area = _mach_area_closure_generic(
        ops=ops,
        n_p=n_next,
        T_e=T_e_next,
        mach=mach_next,
        dot_N=dot_N,
        I_0=I_0,
        seed_fraction=seed_fraction,
        B=float(B),
        working_fluid=working_fluid,
    )
    A_next = mach_area["A_safe"]
    sigma_next = (A_next - A_prev) / (float(dx) * (_safe_pos(ops, A_next, _EPS)))
    dn_dx = (n_next - n_prev) / float(dx)
    dTe_dx = (T_e_next - T_e_prev) / float(dx)
    _, _, momentum, energy, closure, terms = _implicit_step_residuals(
        ops=ops,
        n_prev=n_prev,
        T_e_prev=T_e_prev,
        n_next=n_next,
        T_e_next=T_e_next,
        dn_dx=dn_dx,
        dTe_dx=dTe_dx,
        A_next=A_next,
        sigma_next=sigma_next,
        dot_N=dot_N,
        I_0=I_0,
        seed_fraction=seed_fraction,
        B=float(B),
        dx=float(dx),
        working_fluid=working_fluid,
    )
    return (
        momentum / float(momentum_scale),
        energy / float(energy_scale),
        closure,
        terms,
        dn_dx,
        dTe_dx,
        A_next,
        sigma_next,
    )


def rollout_reduced_mach_generic(
    *,
    ops,
    config: MachReducedConfig,
    n_intervals: int,
    decision_vector: dict[str, Any],
    residual_scales: ImplicitResidualScales,
    newton_steps: int = 10,
    finite_difference_step: float = 1e-4,
    regularization: float = 1e-8,
    max_log_step: float = 0.5,
) -> MachReducedRollout:
    n_intervals = int(n_intervals)
    dx = float(config.length) / n_intervals
    log_n_p_in = decision_vector["log_n_p_in"]
    T_e_in = decision_vector["T_e_in"]
    Z_in = decision_vector["Z_in"]
    I_0 = decision_vector["I_0"]
    seed_fraction = ops.exp(decision_vector["log_seed_fraction"])
    mach_design = MachSplineDesign(
        m1=decision_vector["m1"],
        m2=decision_vector["m2"],
        m3=decision_vector["m3"],
    )
    inlet = _inlet_design_generic(
        ops=ops,
        n_p_in=ops.exp(log_n_p_in),
        T_e_in=T_e_in,
        Z_in=Z_in,
        I_0=I_0,
        seed_fraction=seed_fraction,
        B=float(config.B),
        inlet_A=float(config.area_scale_m2),
        working_fluid=config.working_fluid,
    )
    mach_nodes = _mach_design_nodes_generic(
        ops=ops,
        mach_design=mach_design,
        length=float(config.length),
        n_intervals=n_intervals,
        mach_in=inlet["mach"],
    )
    n_nodes = [inlet["n_p"]]
    T_nodes = [inlet["T_e"]]
    dn_dx_values = []
    dTe_dx_values = []
    scaled_momentum_residuals = []
    scaled_energy_residuals = []
    area_values = [float(config.area_scale_m2)]
    sigma_values = [0.0]
    previous_dn_dx = 0.0
    previous_dTe_dx = 0.0

    for k in range(n_intervals):
        n_prev = n_nodes[-1]
        T_prev = T_nodes[-1]
        A_prev = area_values[-1]
        mach_next = mach_nodes["mach"][k + 1]
        momentum_scale = float(residual_scales.momentum[k])
        energy_scale = float(residual_scales.energy[k])
        n_initial = _safe_pos(ops, n_prev + dx * previous_dn_dx, 1.0)
        T_initial = _safe_pos(ops, T_prev + dx * previous_dTe_dx, 1.0)
        log_n = ops.log(n_initial)
        log_Te = ops.log(T_initial)

        def residual_pair(candidate_log_n, candidate_log_Te):
            n_next_candidate = ops.exp(candidate_log_n)
            T_next_candidate = ops.exp(candidate_log_Te)
            r_m, r_e, _, _, _, _, _, _ = _scaled_momentum_energy_residuals_mach(
                ops=ops,
                n_prev=n_prev,
                T_e_prev=T_prev,
                A_prev=A_prev,
                n_next=n_next_candidate,
                T_e_next=T_next_candidate,
                mach_next=mach_next,
                dot_N=inlet["dot_N"],
                I_0=I_0,
                seed_fraction=seed_fraction,
                B=float(config.B),
                dx=dx,
                momentum_scale=momentum_scale,
                energy_scale=energy_scale,
                working_fluid=config.working_fluid,
            )
            return r_m, r_e

        log_n, log_Te = _fixed_gauss_newton_log_state_2d(
            ops=ops,
            initial_log_n=log_n,
            initial_log_Te=log_Te,
            residual_fn=residual_pair,
            steps=int(newton_steps),
            finite_difference_step=float(finite_difference_step),
            regularization=float(regularization),
            max_log_step=float(max_log_step),
        )
        n_next = ops.exp(log_n)
        T_next = ops.exp(log_Te)
        r_m, r_e, closure, terms, dn_dx, dTe_dx, A_next, sigma_next = _scaled_momentum_energy_residuals_mach(
            ops=ops,
            n_prev=n_prev,
            T_e_prev=T_prev,
            A_prev=A_prev,
            n_next=n_next,
            T_e_next=T_next,
            mach_next=mach_next,
            dot_N=inlet["dot_N"],
            I_0=I_0,
            seed_fraction=seed_fraction,
            B=float(config.B),
            dx=dx,
            momentum_scale=momentum_scale,
            energy_scale=energy_scale,
            working_fluid=config.working_fluid,
        )
        n_nodes.append(n_next)
        T_nodes.append(T_next)
        dn_dx_values.append(dn_dx)
        dTe_dx_values.append(dTe_dx)
        scaled_momentum_residuals.append(r_m)
        scaled_energy_residuals.append(r_e)
        area_values.append(A_next)
        sigma_values.append(sigma_next)
        previous_dn_dx = dn_dx
        previous_dTe_dx = dTe_dx

    closures = []
    terms_by_node = []
    for idx in range(n_intervals + 1):
        sigma = sigma_values[idx]
        closure, terms = _dynamic_system_terms(
            ops=ops,
            n_p=n_nodes[idx],
            T_e=T_nodes[idx],
            A=area_values[idx],
            sigma=sigma,
            dot_N=inlet["dot_N"],
            I_0=I_0,
            seed_fraction=seed_fraction,
            B=float(config.B),
            working_fluid=config.working_fluid,
        )
        closures.append(closure)
        terms_by_node.append(terms)

    abs_residuals = [ops.fabs(item) for item in [*scaled_momentum_residuals, *scaled_energy_residuals]]
    det_abs = [ops.fabs(item["det"]) for item in terms_by_node]
    return MachReducedRollout(
        decision_vector=dict(decision_vector),
        inlet=inlet,
        mach_nodes=mach_nodes,
        area_nodes={
            "x_norm": mach_nodes["x_norm"],
            "x": mach_nodes["x"],
            "A": area_values,
            "sigma_logA": sigma_values,
            "mach": mach_nodes["mach"],
        },
        n_p_nodes=n_nodes,
        T_e_nodes=T_nodes,
        dn_dx=dn_dx_values,
        dTe_dx=dTe_dx_values,
        scaled_momentum_residuals=scaled_momentum_residuals,
        scaled_energy_residuals=scaled_energy_residuals,
        closures=closures,
        terms_by_node=terms_by_node,
        max_abs_scaled_residual=_reduce_max(ops, abs_residuals),
        min_abs_det=_reduce_min(ops, det_abs),
    )


def decision_from_profile(profile_path: str | Path) -> dict[str, float]:
    with np.load(profile_path) as data:
        x = np.asarray(data["x"], dtype=float)
        mach = np.asarray(data["mach"], dtype=float)
        design = MachSplineDesign.project_from_profile(x=x, mach=mach)
        A = np.asarray(data["A"], dtype=float)
        J_x = np.asarray(data["J_x"], dtype=float)
        seed = np.asarray(data["seed_fraction"], dtype=float)
        return {
            "log_n_p_in": math.log(float(np.asarray(data["n_p"], dtype=float)[0])),
            "T_e_in": float(np.asarray(data["T_e"], dtype=float)[0]),
            "Z_in": float(np.asarray(data["Z"], dtype=float)[0]),
            "I_0": float(np.median(J_x * A)),
            "log_seed_fraction": math.log(float(np.median(seed))),
            "m1": float(design.m1),
            "m2": float(design.m2),
            "m3": float(design.m3),
        }


def residual_scales_from_profile(
    *,
    profile_path: str | Path,
    config: MachReducedConfig,
) -> ImplicitResidualScales:
    ops = _ops_for_numeric()
    decision = decision_from_profile(profile_path)
    with np.load(profile_path) as data:
        x = np.asarray(data["x"], dtype=float)
        n_p = np.asarray(data["n_p"], dtype=float)
        T_e = np.asarray(data["T_e"], dtype=float)
        n_intervals = x.size - 1
    dx = float(config.length) / int(n_intervals)
    inlet = _inlet_design_generic(
        ops=ops,
        n_p_in=math.exp(decision["log_n_p_in"]),
        T_e_in=decision["T_e_in"],
        Z_in=decision["Z_in"],
        I_0=decision["I_0"],
        seed_fraction=math.exp(decision["log_seed_fraction"]),
        B=float(config.B),
        inlet_A=float(config.area_scale_m2),
        working_fluid=config.working_fluid,
    )
    mach_nodes = _mach_design_nodes_generic(
        ops=ops,
        mach_design=MachSplineDesign(m1=decision["m1"], m2=decision["m2"], m3=decision["m3"]),
        length=float(config.length),
        n_intervals=int(n_intervals),
        mach_in=float(inlet["mach"]),
    )
    step_n_scales = []
    step_Te_scales = []
    momentum_scales = []
    energy_scales = []
    first_area = _mach_area_closure_generic(
        ops=ops,
        n_p=float(n_p[0]),
        T_e=float(T_e[0]),
        mach=float(mach_nodes["mach"][0]),
        dot_N=float(inlet["dot_N"]),
        I_0=float(inlet["I_0"]),
        seed_fraction=float(inlet["seed_fraction"]),
        B=float(config.B),
        working_fluid=config.working_fluid,
    )["A_safe"]
    A_prev = float(first_area)
    for k in range(int(n_intervals)):
        dn_guess = float((n_p[k + 1] - n_p[k]) / dx)
        dTe_guess = float((T_e[k + 1] - T_e[k]) / dx)
        r_m, r_e, _, terms, _, _, A_next, _ = _scaled_momentum_energy_residuals_mach(
            ops=ops,
            n_prev=float(n_p[k]),
            T_e_prev=float(T_e[k]),
            A_prev=float(A_prev),
            n_next=float(n_p[k + 1]),
            T_e_next=float(T_e[k + 1]),
            mach_next=float(mach_nodes["mach"][k + 1]),
            dot_N=float(inlet["dot_N"]),
            I_0=float(inlet["I_0"]),
            seed_fraction=float(inlet["seed_fraction"]),
            B=float(config.B),
            dx=dx,
            momentum_scale=1.0,
            energy_scale=1.0,
            working_fluid=config.working_fluid,
        )
        step_n_scales.append(max(abs(float(n_p[k + 1] - n_p[k])), abs(dx * dn_guess), abs(float(n_p[k + 1])), 1.0))
        step_Te_scales.append(max(abs(float(T_e[k + 1] - T_e[k])), abs(dx * dTe_guess), abs(float(T_e[k + 1])), 1.0))
        momentum_scales.append(
            max(
                abs(float(terms["rhs_m"])),
                abs(float(terms["M11"] * dn_guess)),
                abs(float(terms["M12"] * dTe_guess)),
                abs(float(r_m)),
                1.0,
            )
        )
        energy_scales.append(
            max(
                abs(float(terms["rhs_e"])),
                abs(float(terms["E11"] * dn_guess)),
                abs(float(terms["E12"] * dTe_guess)),
                abs(float(r_e)),
                1.0,
            )
        )
        A_prev = float(A_next)
    return ImplicitResidualScales(
        step_n=np.asarray(step_n_scales, dtype=float),
        step_Te=np.asarray(step_Te_scales, dtype=float),
        momentum=np.asarray(momentum_scales, dtype=float),
        energy=np.asarray(energy_scales, dtype=float),
    )


def rollout_summary_from_profile(
    *,
    profile_path: str | Path,
    summary_path: str | Path,
    newton_steps: int = 10,
) -> dict[str, Any]:
    config = MachReducedConfig.from_summary_and_profile(summary_path, profile_path)
    decision = decision_from_profile(profile_path)
    scales = residual_scales_from_profile(profile_path=profile_path, config=config)
    with np.load(profile_path) as data:
        x = np.asarray(data["x"], dtype=float)
        n_ref = np.asarray(data["n_p"], dtype=float)
        te_ref = np.asarray(data["T_e"], dtype=float)
        A_ref = np.asarray(data["A"], dtype=float)
        mach_ref = np.asarray(data["mach"], dtype=float)
    rollout = rollout_reduced_mach_generic(
        ops=_ops_for_numeric(),
        config=config,
        n_intervals=x.size - 1,
        decision_vector=decision,
        residual_scales=scales,
        newton_steps=int(newton_steps),
    )
    n = np.asarray(rollout.n_p_nodes, dtype=float)
    te = np.asarray(rollout.T_e_nodes, dtype=float)
    A = np.asarray(rollout.area_nodes["A"], dtype=float)
    mach = np.asarray([closure["mach"] for closure in rollout.closures], dtype=float)
    max_residual = float(rollout.max_abs_scaled_residual)
    return {
        "profile_path": str(Path(profile_path).resolve()),
        "summary_path": str(Path(summary_path).resolve()),
        "decision_vector": {key: float(value) for key, value in decision.items()},
        "newton_steps": int(newton_steps),
        "max_abs_scaled_residual": max_residual,
        "min_abs_det": float(rollout.min_abs_det),
        "max_rel_n_p_error": float(np.nanmax(np.abs(n - n_ref) / np.maximum(np.abs(n_ref), 1e-300))),
        "max_abs_T_e_error_K": float(np.nanmax(np.abs(te - te_ref))),
        "max_rel_A_error": float(np.nanmax(np.abs(A - A_ref) / np.maximum(np.abs(A_ref), 1e-300))),
        "max_rel_mach_error": float(np.nanmax(np.abs(mach - mach_ref) / np.maximum(np.abs(mach_ref), 1e-300))),
        "outlet": {
            "n_p": float(n[-1]),
            "T_e": float(te[-1]),
            "A": float(A[-1]),
            "mach": float(mach[-1]),
        },
        "reference_outlet": {
            "n_p": float(n_ref[-1]),
            "T_e": float(te_ref[-1]),
            "A": float(A_ref[-1]),
            "mach": float(mach_ref[-1]),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the reduced Mach-spline fixed-Newton shadow rollout for a stored profile."
    )
    parser.add_argument("profile", type=Path)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--newton-steps", type=int, default=10)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    report = rollout_summary_from_profile(
        profile_path=args.profile,
        summary_path=args.summary,
        newton_steps=int(args.newton_steps),
    )
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.out is None:
        print(text)
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")
        print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
