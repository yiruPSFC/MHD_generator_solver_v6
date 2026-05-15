from __future__ import annotations

import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from v6_global_marginal.global_postprocess_v6 import compute_design_value_terms

from v6_maingo_casadi.constants import _EPS, _G_HARD_MARGIN, _TP_MIN, OBJECTIVE_PROFILE_LAB_POC_V2
from v6_maingo_casadi.geometry import SplineAreaDesign
from v6_maingo_casadi.implicit import ImplicitResidualScales, _scaled_interval
from v6_maingo_casadi.models import BaselineSeed, CoarseProfileResult, InletDesign
from v6_maingo_casadi.numerics import (
    _max_op,
    _min_op,
    _ops_for_maingo,
    _ops_for_numeric,
    _reduce_min,
    _safe_pos,
    _velikhov_margin_penalty,
)
from v6_maingo_casadi.physics import _design_score_generic, _dynamic_system_terms, _inlet_design_generic
from v6_maingo_casadi.profiles import (
    _augment_value_terms_with_hall_diagnostics,
    _normalize_objective_profile,
    _value_profile_dict,
)
from v6_maingo_casadi.reduced_implicit import _model_function

from .geometry import MachSplineDesign
from .reduced_implicit import (
    MACH_DECISION_NAMES,
    MachReducedConfig,
    MachReducedRollout,
    decision_from_profile,
    _mach_area_closure_generic,
    _mach_design_nodes_generic,
    _positive_denom,
    _reduce_max,
    residual_scales_from_profile,
    rollout_reduced_mach_generic,
)


@dataclass(frozen=True)
class MachRK4SoftRollout:
    decision_vector: dict[str, Any]
    inlet: dict[str, Any]
    mach_nodes: dict[str, Any]
    area_nodes: dict[str, Any]
    n_p_nodes: list[Any]
    T_e_nodes: list[Any]
    closures: list[dict[str, Any]]
    terms_by_node: list[dict[str, Any]]
    terms_by_stage: list[dict[str, Any]]
    det_values: list[Any]
    min_abs_det: Any
    min_signed_det: Any


@dataclass(frozen=True)
class MachTrapezoidRollout:
    decision_vector: dict[str, Any]
    inlet: dict[str, Any]
    mach_nodes: dict[str, Any]
    area_nodes: dict[str, Any]
    n_p_nodes: list[Any]
    T_e_nodes: list[Any]
    dn_dx: list[Any]
    dTe_dx: list[Any]
    scaled_step_n_residuals: list[Any]
    scaled_step_Te_residuals: list[Any]
    scaled_momentum_residuals: list[Any]
    scaled_energy_residuals: list[Any]
    closures: list[dict[str, Any]]
    terms_by_node: list[dict[str, Any]]
    max_abs_scaled_residual: Any
    min_abs_det: Any
    min_signed_det: Any


def _normalize_det_branch_sign(det_sign: str | None) -> int:
    sign = str(det_sign or "none").strip().replace("-", "_").lower()
    if sign in {"none", "off", "disabled", "0"}:
        return 0
    if sign in {"positive", "pos", "+", "+1", "det_positive"}:
        return 1
    if sign in {"negative", "neg", "-", "-1", "det_negative"}:
        return -1
    raise ValueError(
        f"unsupported det branch sign {det_sign!r}; expected positive, negative, or none."
    )


def _det_branch_name(sign: int) -> str:
    if sign > 0:
        return "positive"
    if sign < 0:
        return "negative"
    return "none"


def _positive_part(ops, value):
    if getattr(ops, "pos", None) is not None:
        return ops.pos(value)
    return _max_op(ops, value, 0.0)


def _negative_part(ops, value):
    if getattr(ops, "neg", None) is not None:
        return ops.neg(value)
    return _min_op(ops, value, 0.0)


def _rk4_positive_state(ops, value, floor: float):
    return _safe_pos(ops, value, float(floor))


def _chain_det_value(terms: dict[str, Any]):
    return terms.get("chain_scaled_det", terms.get("chain_det", terms["det"]))


def _det_branch_denom(ops, det_value, *, sign: int, floor: float):
    if sign > 0:
        return float(floor) + _positive_part(ops, det_value - float(floor))
    if sign < 0:
        return -float(floor) + _negative_part(ops, det_value + float(floor))
    return _safe_pos(ops, det_value, 1e-12)


def _signed_det_values(det_values: list[Any], *, sign: int) -> list[Any]:
    if sign > 0:
        return list(det_values)
    if sign < 0:
        return [-item for item in det_values]
    return [0.0]


def _mach_stage_value_generic(
    *,
    ops,
    mach_design: MachSplineDesign,
    length: float,
    x_norm: float,
    mach_in,
) -> dict[str, Any]:
    basis, slopes = SplineAreaDesign.basis_matrices(np.asarray([float(x_norm)], dtype=float))
    params = [mach_design.m1, mach_design.m2, mach_design.m3]
    log_ratio = basis[0, 0] * params[0] + basis[0, 1] * params[1] + basis[0, 2] * params[2]
    dlogM_dx = (slopes[0, 0] * params[0] + slopes[0, 1] * params[1] + slopes[0, 2] * params[2]) / float(length)
    return {
        "x_norm": float(x_norm),
        "x": float(x_norm) * float(length),
        "log_mach_ratio": log_ratio,
        "mach": mach_in * ops.exp(log_ratio),
        "dlogM_dx": dlogM_dx,
    }


def _mach_rk4_stage_rhs(
    *,
    ops,
    config: MachReducedConfig,
    mach_design: MachSplineDesign,
    x_norm: float,
    n_p,
    T_e,
    mach_in,
    dot_N,
    I_0,
    seed_fraction,
    det_branch_sign: int = 1,
    det_floor: float = 1e-6,
) -> tuple[Any, Any, dict[str, Any], dict[str, Any], Any]:
    fd = 1e-4
    mach_stage = _mach_stage_value_generic(
        ops=ops,
        mach_design=mach_design,
        length=float(config.length),
        x_norm=float(x_norm),
        mach_in=mach_in,
    )
    mach_area = _mach_area_closure_generic(
        ops=ops,
        n_p=n_p,
        T_e=T_e,
        mach=mach_stage["mach"],
        dot_N=dot_N,
        I_0=I_0,
        seed_fraction=seed_fraction,
        B=float(config.B),
        working_fluid=config.working_fluid,
    )
    n_safe = _rk4_positive_state(ops, n_p, 1.0)
    T_safe = _rk4_positive_state(ops, T_e, 1.0)
    M_safe = _rk4_positive_state(ops, mach_stage["mach"], 1e-12)
    exp_fd = math.exp(fd)
    exp_mfd = math.exp(-fd)
    log_delta = exp_fd - exp_mfd

    def area_at(*, n_value, T_value, M_value):
        return _mach_area_closure_generic(
            ops=ops,
            n_p=n_value,
            T_e=T_value,
            mach=M_value,
            dot_N=dot_N,
            I_0=I_0,
            seed_fraction=seed_fraction,
            B=float(config.B),
            working_fluid=config.working_fluid,
        )["A_safe"]

    # Chain-consistent Mach-coordinate RHS:
    # A = A(n, T, M(x)), so dA/dx = A_n n' + A_T T' + A_M M'.
    # The local partials are finite-difference approximations in log coordinates.
    dA_dn = (area_at(n_value=n_safe * exp_fd, T_value=T_safe, M_value=M_safe) - area_at(
        n_value=n_safe * exp_mfd,
        T_value=T_safe,
        M_value=M_safe,
    )) / _positive_denom(ops, n_safe * log_delta, log_delta)
    dA_dT = (area_at(n_value=n_safe, T_value=T_safe * exp_fd, M_value=M_safe) - area_at(
        n_value=n_safe,
        T_value=T_safe * exp_mfd,
        M_value=M_safe,
    )) / _positive_denom(ops, T_safe * log_delta, log_delta)
    dA_dM = (area_at(n_value=n_safe, T_value=T_safe, M_value=M_safe * exp_fd) - area_at(
        n_value=n_safe,
        T_value=T_safe,
        M_value=M_safe * exp_mfd,
    )) / _positive_denom(ops, M_safe * log_delta, 1e-12 * log_delta)
    dM_dx = M_safe * mach_stage["dlogM_dx"]
    closure, terms = _dynamic_system_terms(
        ops=ops,
        n_p=n_p,
        T_e=T_e,
        A=mach_area["A_safe"],
        sigma=0.0,
        dot_N=dot_N,
        I_0=I_0,
        seed_fraction=seed_fraction,
        B=float(config.B),
        working_fluid=config.working_fluid,
    )
    M11_eff = terms["M11"] + terms["M13"] * dA_dn
    M12_eff = terms["M12"] + terms["M13"] * dA_dT
    E11_eff = terms["E11"] + terms["E13"] * dA_dn
    E12_eff = terms["E12"] + terms["E13"] * dA_dT
    rhs_m_eff = terms["rhs_m"] - terms["M13"] * dA_dM * dM_dx
    rhs_e_eff = terms["rhs_e"] - terms["E13"] * dA_dM * dM_dx
    # Solve for logarithmic derivatives u = dlog(n)/dx and v = dlog(T_e)/dx.
    # The unscaled primitive system has enormous column scales and suffers
    # catastrophic determinant cancellation in the Yamasaki neighborhood.
    a11 = M11_eff * n_safe
    a12 = M12_eff * T_safe
    a21 = E11_eff * n_safe
    a22 = E12_eff * T_safe
    b1 = rhs_m_eff
    b2 = rhs_e_eff
    row1_scale = _max_op(ops, _max_op(ops, ops.fabs(a11), ops.fabs(a12)), _max_op(ops, ops.fabs(b1), 1.0))
    row2_scale = _max_op(ops, _max_op(ops, ops.fabs(a21), ops.fabs(a22)), _max_op(ops, ops.fabs(b2), 1.0))
    row1_denom = _positive_denom(ops, row1_scale, 1.0)
    row2_denom = _positive_denom(ops, row2_scale, 1.0)
    a11s = a11 / row1_denom
    a12s = a12 / row1_denom
    b1s = b1 / row1_denom
    a21s = a21 / row2_denom
    a22s = a22 / row2_denom
    b2s = b2 / row2_denom
    det_scaled = a11s * a22s - a12s * a21s
    det_denom = _det_branch_denom(
        ops,
        det_scaled,
        sign=int(det_branch_sign),
        floor=float(det_floor),
    )
    dlogn_dx = (b1s * a22s - a12s * b2s) / det_denom
    dlogT_dx = (a11s * b2s - b1s * a21s) / det_denom
    dn_dx = n_safe * dlogn_dx
    dTe_dx = T_safe * dlogT_dx
    terms = dict(terms)
    terms.update(
        {
            "chain_dA_dn": dA_dn,
            "chain_dA_dT": dA_dT,
            "chain_dA_dM": dA_dM,
            "chain_dM_dx": dM_dx,
            "chain_M11": M11_eff,
            "chain_M12": M12_eff,
            "chain_E11": E11_eff,
            "chain_E12": E12_eff,
            "chain_rhs_m": rhs_m_eff,
            "chain_rhs_e": rhs_e_eff,
            "chain_det": M11_eff * E12_eff - M12_eff * E11_eff,
            "chain_scaled_det": det_scaled,
            "chain_det_denom": det_denom,
            "chain_dlogn_dx": dlogn_dx,
            "chain_dlogT_dx": dlogT_dx,
            "chain_row1_scale": row1_denom,
            "chain_row2_scale": row2_denom,
        }
    )
    return dn_dx, dTe_dx, closure, terms, mach_area["A_safe"]


def rollout_mach_spline_rk4_soft_generic(
    *,
    ops,
    config: MachReducedConfig,
    n_intervals: int,
    decision_vector: dict[str, Any],
    det_branch_sign: int = 1,
    det_floor: float = 1e-6,
) -> MachRK4SoftRollout:
    n_intervals = int(n_intervals)
    dx = float(config.length) / n_intervals
    x_norm = np.linspace(0.0, 1.0, n_intervals + 1, dtype=float)
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
    terms_by_stage = []
    for k in range(n_intervals):
        n0 = n_nodes[-1]
        T0 = T_nodes[-1]
        x0 = float(x_norm[k])
        x_mid = 0.5 * (float(x_norm[k]) + float(x_norm[k + 1]))
        x1 = float(x_norm[k + 1])
        dn1, dT1, _, terms1, _ = _mach_rk4_stage_rhs(
            ops=ops,
            config=config,
            mach_design=mach_design,
            x_norm=x0,
            n_p=n0,
            T_e=T0,
            mach_in=inlet["mach"],
            dot_N=inlet["dot_N"],
            I_0=I_0,
            seed_fraction=seed_fraction,
            det_branch_sign=int(det_branch_sign),
            det_floor=float(det_floor),
        )
        terms_by_stage.append(terms1)
        dn2, dT2, _, terms2, _ = _mach_rk4_stage_rhs(
            ops=ops,
            config=config,
            mach_design=mach_design,
            x_norm=x_mid,
            n_p=_rk4_positive_state(ops, n0 + 0.5 * dx * dn1, 1.0),
            T_e=_rk4_positive_state(ops, T0 + 0.5 * dx * dT1, 1.0),
            mach_in=inlet["mach"],
            dot_N=inlet["dot_N"],
            I_0=I_0,
            seed_fraction=seed_fraction,
            det_branch_sign=int(det_branch_sign),
            det_floor=float(det_floor),
        )
        terms_by_stage.append(terms2)
        dn3, dT3, _, terms3, _ = _mach_rk4_stage_rhs(
            ops=ops,
            config=config,
            mach_design=mach_design,
            x_norm=x_mid,
            n_p=_rk4_positive_state(ops, n0 + 0.5 * dx * dn2, 1.0),
            T_e=_rk4_positive_state(ops, T0 + 0.5 * dx * dT2, 1.0),
            mach_in=inlet["mach"],
            dot_N=inlet["dot_N"],
            I_0=I_0,
            seed_fraction=seed_fraction,
            det_branch_sign=int(det_branch_sign),
            det_floor=float(det_floor),
        )
        terms_by_stage.append(terms3)
        dn4, dT4, _, terms4, _ = _mach_rk4_stage_rhs(
            ops=ops,
            config=config,
            mach_design=mach_design,
            x_norm=x1,
            n_p=_rk4_positive_state(ops, n0 + dx * dn3, 1.0),
            T_e=_rk4_positive_state(ops, T0 + dx * dT3, 1.0),
            mach_in=inlet["mach"],
            dot_N=inlet["dot_N"],
            I_0=I_0,
            seed_fraction=seed_fraction,
            det_branch_sign=int(det_branch_sign),
            det_floor=float(det_floor),
        )
        terms_by_stage.append(terms4)
        n_nodes.append(_rk4_positive_state(ops, n0 + dx * (dn1 + 2.0 * dn2 + 2.0 * dn3 + dn4) / 6.0, 1.0))
        T_nodes.append(_rk4_positive_state(ops, T0 + dx * (dT1 + 2.0 * dT2 + 2.0 * dT3 + dT4) / 6.0, 1.0))

    closures = []
    terms_by_node = []
    area_values = []
    for idx in range(n_intervals + 1):
        _, _, closure, terms, area = _mach_rk4_stage_rhs(
            ops=ops,
            config=config,
            mach_design=mach_design,
            x_norm=float(x_norm[idx]),
            n_p=n_nodes[idx],
            T_e=T_nodes[idx],
            mach_in=inlet["mach"],
            dot_N=inlet["dot_N"],
            I_0=I_0,
            seed_fraction=seed_fraction,
            det_branch_sign=int(det_branch_sign),
            det_floor=float(det_floor),
        )
        closures.append(closure)
        terms_by_node.append(terms)
        area_values.append(area)
    sigma_values = [0.0]
    for idx in range(1, n_intervals + 1):
        sigma_values.append(
            (area_values[idx] - area_values[idx - 1]) / (dx * _positive_denom(ops, area_values[idx], _EPS))
        )
    det_values = [_chain_det_value(item) for item in [*terms_by_stage, *terms_by_node]]
    det_abs = [ops.fabs(item) for item in det_values]
    signed_det = _signed_det_values(det_values, sign=int(det_branch_sign))
    return MachRK4SoftRollout(
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
        closures=closures,
        terms_by_node=terms_by_node,
        terms_by_stage=terms_by_stage,
        det_values=det_values,
        min_abs_det=_reduce_min(ops, det_abs),
        min_signed_det=_reduce_min(ops, signed_det),
    )


def _soft_average_square_shortfall(ops, values: list[Any], *, scale: float) -> Any:
    if not values:
        return 0.0
    acc = 0.0
    denom = max(float(scale), 1e-12)
    for value in values:
        shortfall = _max_op(ops, value, 0.0) / denom
        acc = acc + shortfall * shortfall
    return acc / float(len(values))


def _mach_midpoint_closures(
    *,
    ops,
    config: MachReducedConfig,
    rollout,
    decision: dict[str, Any],
    n_intervals: int,
) -> list[dict[str, Any]]:
    x_mid_norm = (np.arange(int(n_intervals), dtype=float) + 0.5) / float(int(n_intervals))
    from v6_maingo_casadi.geometry import SplineAreaDesign

    basis_mid, _ = SplineAreaDesign.basis_matrices(x_mid_norm)
    params = [decision["m1"], decision["m2"], decision["m3"]]
    closures = []
    for idx in range(int(n_intervals)):
        log_ratio = basis_mid[idx, 0] * params[0] + basis_mid[idx, 1] * params[1] + basis_mid[idx, 2] * params[2]
        mach_mid = rollout.inlet["mach"] * ops.exp(log_ratio)
        n_mid = 0.5 * (rollout.n_p_nodes[idx] + rollout.n_p_nodes[idx + 1])
        T_mid = 0.5 * (rollout.T_e_nodes[idx] + rollout.T_e_nodes[idx + 1])
        closures.append(
            _mach_area_closure_generic(
                ops=ops,
                n_p=n_mid,
                T_e=T_mid,
                mach=mach_mid,
                dot_N=rollout.inlet["dot_N"],
                I_0=decision["I_0"],
                seed_fraction=ops.exp(decision["log_seed_fraction"]),
                B=float(config.B),
                working_fluid=config.working_fluid,
            )
        )
    return closures


def _numeric_midpoint_closures(
    *,
    config: MachReducedConfig,
    decision: dict[str, float],
    inlet: dict[str, float],
    n_p_nodes: np.ndarray,
    T_e_nodes: np.ndarray,
    n_intervals: int,
) -> list[dict[str, float]]:
    ops = _ops_for_numeric()
    x_mid_norm = (np.arange(int(n_intervals), dtype=float) + 0.5) / float(int(n_intervals))
    basis_mid, _ = SplineAreaDesign.basis_matrices(x_mid_norm)
    params = np.asarray([decision["m1"], decision["m2"], decision["m3"]], dtype=float)
    closures = []
    for idx in range(int(n_intervals)):
        log_ratio = float(basis_mid[idx, :] @ params)
        mach_mid = float(inlet["mach"]) * math.exp(log_ratio)
        closures.append(
            _mach_area_closure_generic(
                ops=ops,
                n_p=float(0.5 * (n_p_nodes[idx] + n_p_nodes[idx + 1])),
                T_e=float(0.5 * (T_e_nodes[idx] + T_e_nodes[idx + 1])),
                mach=mach_mid,
                dot_N=float(inlet["dot_N"]),
                I_0=float(decision["I_0"]),
                seed_fraction=math.exp(float(decision["log_seed_fraction"])),
                B=float(config.B),
                working_fluid=config.working_fluid,
            )
        )
    return closures


class MachSplineReducedImplicitModelBase:
    formulation = "mach_spline_reduced_implicit_fixed_newton"

    def __init__(
        self,
        *,
        baseline: BaselineSeed,
        reference_profile_path: str | Path | None,
        n_intervals: int,
        maingopy_module,
        objective_profile: str = OBJECTIVE_PROFILE_LAB_POC_V2,
        newton_steps: int = 10,
        residual_tolerance: float = 1e-5,
        mach_window_radius: float = 1.0,
    ):
        self._baseline = baseline
        self._reference_profile_path = Path(reference_profile_path or baseline.warm_profile_npz_path).resolve()
        self._n_intervals = int(n_intervals)
        self._maingopy = maingopy_module
        self._ops = _ops_for_maingo(maingopy_module)
        self._objective_profile = _normalize_objective_profile(objective_profile)
        self._newton_steps = int(newton_steps)
        self._residual_tolerance = float(residual_tolerance)
        self._mach_window_radius = float(mach_window_radius)
        self._config = MachReducedConfig(
            B=float(baseline.B),
            length=float(baseline.L),
            area_scale_m2=float(baseline.area_scale_m2),
            working_fluid=baseline.working_fluid,
        )
        self._mach_design_nominal = self._project_reference_mach_design(self._reference_profile_path)
        self._residual_scales = residual_scales_from_profile(
            profile_path=self._reference_profile_path,
            config=self._config,
        )
        self._variable_specs = self._build_variable_specs()
        self._initial_point = self._build_initial_point()

    @staticmethod
    def _project_reference_mach_design(path: Path) -> MachSplineDesign:
        with np.load(path) as data:
            return MachSplineDesign.project_from_profile(
                x=np.asarray(data["x"], dtype=float),
                mach=np.asarray(data["mach"], dtype=float),
            )

    def _build_variable_specs(self) -> list[tuple[float, float, str]]:
        inlet = self._baseline.inlet_windows
        nominal = self._mach_design_nominal.as_array()
        radius = float(self._mach_window_radius)
        mach_specs = []
        for name, value in zip(("m1", "m2", "m3"), nominal, strict=True):
            lower = max(MachSplineDesign.lower_bound(), float(value) - radius)
            upper = min(MachSplineDesign.upper_bound(), float(value) + radius)
            if lower >= upper:
                lower = MachSplineDesign.lower_bound()
                upper = MachSplineDesign.upper_bound()
            mach_specs.append((float(lower), float(upper), name))
        return [
            (
                math.log(float(inlet["n_p_in"]["min"])),
                math.log(float(inlet["n_p_in"]["max"])),
                "log_n_p_in",
            ),
            (float(inlet["T_e_in"]["min"]), float(inlet["T_e_in"]["max"]), "T_e_in"),
            (float(inlet["Z_in"]["min"]), float(inlet["Z_in"]["max"]), "Z_in"),
            (float(inlet["I_0"]["min"]), float(inlet["I_0"]["max"]), "I_0"),
            (
                math.log(float(inlet["seed_fraction"]["min"])),
                math.log(float(inlet["seed_fraction"]["max"])),
                "log_seed_fraction",
            ),
            *mach_specs,
        ]

    def _build_initial_point(self) -> list[float]:
        design = self._mach_design_nominal
        return [
            math.log(float(self._baseline.n_p_in_nominal)),
            float(self._baseline.T_e_in_nominal),
            float(self._baseline.Z_in_nominal),
            float(self._baseline.I_0_nominal),
            math.log(float(self._baseline.seed_fraction_nominal)),
            float(design.m1),
            float(design.m2),
            float(design.m3),
        ]

    @property
    def total_variables(self) -> int:
        return len(self._variable_specs)

    def summary_metadata(self) -> dict[str, Any]:
        return {
            "formulation": self.formulation,
            "newton_steps": int(self._newton_steps),
            "residual_tolerance": float(self._residual_tolerance),
            "mach_window_radius": float(self._mach_window_radius),
            "reference_profile_path": str(self._reference_profile_path),
            "mach_design_nominal": self._mach_design_nominal.to_dict(),
            "residual_scale_momentum_min": float(np.min(self._residual_scales.momentum)),
            "residual_scale_momentum_max": float(np.max(self._residual_scales.momentum)),
            "residual_scale_energy_min": float(np.min(self._residual_scales.energy)),
            "residual_scale_energy_max": float(np.max(self._residual_scales.energy)),
        }

    def get_variables(self):
        variables = []
        for lower, upper, name in self._variable_specs:
            variables.append(
                self._maingopy.OptimizationVariable(
                    self._maingopy.Bounds(float(lower), float(upper)),
                    self._maingopy.VT_CONTINUOUS,
                    str(name),
                )
            )
        return variables

    def get_initial_point(self) -> list[float]:
        return list(self._initial_point)

    def _decision_from_values(self, values) -> dict[str, Any]:
        values = list(values)
        if len(values) != len(MACH_DECISION_NAMES):
            raise ValueError(
                f"mach-spline reduced solution size mismatch: got {len(values)}, expected {len(MACH_DECISION_NAMES)}."
            )
        return {name: value for name, value in zip(MACH_DECISION_NAMES, values, strict=True)}

    def set_initial_point_from_decision(self, decision: dict[str, Any]) -> dict[str, float]:
        decision = dict(decision)
        if "log_n_p_in" not in decision and "n_p_in" in decision:
            decision["log_n_p_in"] = math.log(float(decision["n_p_in"]))
        if "log_seed_fraction" not in decision and "seed_fraction" in decision:
            decision["log_seed_fraction"] = math.log(float(decision["seed_fraction"]))
        normalized: dict[str, float] = {}
        values: list[float] = []
        for idx, (lower, upper, name) in enumerate(self._variable_specs):
            value = float(decision.get(name, self._initial_point[idx]))
            if not math.isfinite(value):
                raise ValueError(f"non-finite initial decision value for {name!r}: {value!r}.")
            tol = 1e-8 * max(1.0, abs(float(lower)), abs(float(upper)))
            if value < float(lower) - tol or value > float(upper) + tol:
                raise ValueError(
                    f"initial decision value for {name!r}={value:.16g} is outside "
                    f"bounds [{float(lower):.16g}, {float(upper):.16g}]."
                )
            values.append(value)
            normalized[str(name)] = value
        self._initial_point = values
        return normalized

    def _numeric_rollout(self, decision: dict[str, Any]) -> MachReducedRollout:
        numeric_decision = {key: float(value) for key, value in dict(decision).items() if key in MACH_DECISION_NAMES}
        return rollout_reduced_mach_generic(
            ops=_ops_for_numeric(),
            config=self._config,
            n_intervals=self._n_intervals,
            decision_vector=numeric_decision,
            residual_scales=self._residual_scales,
            newton_steps=self._newton_steps,
        )

    def decode_solution_point(self, values) -> MachReducedRollout:
        decision = {key: float(value) for key, value in self._decision_from_values(values).items()}
        return self._numeric_rollout(decision)

    def _coarse_result_from_nodes(
        self,
        *,
        decision: dict[str, float],
        n_p_nodes: np.ndarray,
        T_e_nodes: np.ndarray,
        n_intervals: int,
        rollout: MachReducedRollout | None = None,
        check_equalities: bool = True,
    ) -> CoarseProfileResult:
        ops = _ops_for_numeric()
        n_intervals = int(n_intervals)
        dx = float(self._config.length) / float(n_intervals)
        decision = {key: float(decision[key]) for key in MACH_DECISION_NAMES}
        n_p_nodes = np.asarray(n_p_nodes, dtype=float).reshape(n_intervals + 1)
        T_e_nodes = np.asarray(T_e_nodes, dtype=float).reshape(n_intervals + 1)
        seed_fraction = math.exp(float(decision["log_seed_fraction"]))
        inlet = _inlet_design_generic(
            ops=ops,
            n_p_in=math.exp(float(decision["log_n_p_in"])),
            T_e_in=float(decision["T_e_in"]),
            Z_in=float(decision["Z_in"]),
            I_0=float(decision["I_0"]),
            seed_fraction=seed_fraction,
            B=float(self._config.B),
            inlet_A=float(self._config.area_scale_m2),
            working_fluid=self._config.working_fluid,
        )
        mach_nodes = _mach_design_nodes_generic(
            ops=ops,
            mach_design=MachSplineDesign(m1=decision["m1"], m2=decision["m2"], m3=decision["m3"]),
            length=float(self._config.length),
            n_intervals=n_intervals,
            mach_in=float(inlet["mach"]),
        )
        x_nodes = np.asarray(mach_nodes["x"], dtype=float)
        area_nodes = [float(self._config.area_scale_m2)]
        sigma_nodes = [0.0]
        for idx in range(1, n_intervals + 1):
            mach_area = _mach_area_closure_generic(
                ops=ops,
                n_p=float(n_p_nodes[idx]),
                T_e=float(T_e_nodes[idx]),
                mach=float(mach_nodes["mach"][idx]),
                dot_N=float(inlet["dot_N"]),
                I_0=float(decision["I_0"]),
                seed_fraction=seed_fraction,
                B=float(self._config.B),
                working_fluid=self._config.working_fluid,
            )
            A_next = float(mach_area["A_safe"])
            sigma_next = (A_next - float(area_nodes[-1])) / (dx * max(A_next, _EPS))
            area_nodes.append(A_next)
            sigma_nodes.append(sigma_next)
        A_nodes = np.asarray(area_nodes, dtype=float)
        sigma_nodes_arr = np.asarray(sigma_nodes, dtype=float)

        closures = []
        terms_by_node = []
        power_density_nodes = []
        det_nodes = []
        for idx in range(n_intervals + 1):
            closure, terms = _dynamic_system_terms(
                ops=ops,
                n_p=float(n_p_nodes[idx]),
                T_e=float(T_e_nodes[idx]),
                A=float(A_nodes[idx]),
                sigma=float(sigma_nodes_arr[idx]),
                dot_N=float(inlet["dot_N"]),
                I_0=float(decision["I_0"]),
                seed_fraction=seed_fraction,
                B=float(self._config.B),
                working_fluid=self._config.working_fluid,
            )
            closures.append(closure)
            terms_by_node.append(terms)
            power_density_nodes.append(-float(A_nodes[idx]) * float(closure["J_x"]) * float(closure["E_x"]) / 1e8)
            det_nodes.append(float(terms["det"]))

        midpoint_closures = _numeric_midpoint_closures(
            config=self._config,
            decision=decision,
            inlet={key: float(value) for key, value in inlet.items()},
            n_p_nodes=n_p_nodes,
            T_e_nodes=T_e_nodes,
            n_intervals=n_intervals,
        )
        min_g_nodes = float(np.min(np.asarray([item["G"] for item in closures], dtype=float)))
        min_g_midpoints = float(np.min(np.asarray([item["G"] for item in midpoint_closures], dtype=float)))
        min_g_all = min(min_g_nodes, min_g_midpoints)
        min_tp_nodes = float(np.min(np.asarray([item["T_p"] for item in closures], dtype=float)))
        min_tp_midpoints = float(np.min(np.asarray([item["T_p"] for item in midpoint_closures], dtype=float)))
        min_tp_all = min(min_tp_nodes, min_tp_midpoints)
        raw_design_score = float(
            _design_score_generic(
                ops=ops,
                outlet_T_e=float(T_e_nodes[-1]),
                outlet_T_p=float(closures[-1]["T_p"]),
                outlet_n_p=float(n_p_nodes[-1]),
                outlet_n_e=float(closures[-1]["n_e"]),
                inlet_T_e=float(inlet["T_e"]),
                inlet_T_p=float(inlet["T_p"]),
                inlet_mach=float(inlet["mach"]),
                power_density_nodes=power_density_nodes,
                x_nodes=x_nodes,
                seed_fraction=seed_fraction,
                B=float(self._config.B),
                length=float(self._config.length),
                objective_profile=self._objective_profile,
                inlet_n_p=float(inlet["n_p"]),
                inlet_n_e=float(closures[0]["n_e"]),
                inlet_v=float(inlet["v_in"]),
                inlet_A=float(self._config.area_scale_m2),
                working_fluid=self._config.working_fluid,
            )
        )
        velikhov_penalty = float(_velikhov_margin_penalty(ops, min_g_all))
        design_score = float(raw_design_score - velikhov_penalty)
        arrays = {
            "x": x_nodes,
            "n_p": n_p_nodes,
            "T_e": T_e_nodes,
            "A": A_nodes,
            "sigma_logA": sigma_nodes_arr,
            "T_p": np.asarray([item["T_p"] for item in closures], dtype=float),
            "T_p_midpoint": np.asarray([item["T_p"] for item in midpoint_closures], dtype=float),
            "v_p": np.asarray([item["v_p"] for item in closures], dtype=float),
            "n_e": np.asarray([item["n_e"] for item in closures], dtype=float),
            "beta": np.asarray([item["beta"] for item in closures], dtype=float),
            "eta": np.asarray([item["eta"] for item in closures], dtype=float),
            "Z": np.asarray([item["Z"] for item in closures], dtype=float),
            "J_x": np.asarray([item["J_x"] for item in closures], dtype=float),
            "J_y": np.asarray([item["J_y"] for item in closures], dtype=float),
            "E_x": np.asarray([item["E_x"] for item in closures], dtype=float),
            "mach": np.asarray([item["mach"] for item in closures], dtype=float),
            "velikhov_margin": np.asarray([item["G"] for item in closures], dtype=float),
            "velikhov_margin_midpoint": np.asarray([item["G"] for item in midpoint_closures], dtype=float),
        }
        ineq = []
        sigma_max = float(self._baseline.schedule[0]["max_abs_dlogA_dx"])
        tp_min = float(self._baseline.schedule[0].get("tp_min", _TP_MIN))
        mach_min = float(self._baseline.schedule[0].get("mach_min", 0.0) or 0.0)
        for sigma in arrays["sigma_logA"]:
            ineq.append(float(abs(sigma) - sigma_max))
        for closure in closures:
            ineq.append(float(tp_min - closure["T_p"]))
            ineq.append(float(_G_HARD_MARGIN - closure["G"]))
            if mach_min > 0.0:
                ineq.append(float(mach_min - closure["mach"]))
        for closure_mid in midpoint_closures:
            ineq.append(float(_G_HARD_MARGIN - closure_mid["G"]))
        if rollout is not None and check_equalities:
            eq_arr = np.asarray(
                [*rollout.scaled_momentum_residuals, *rollout.scaled_energy_residuals],
                dtype=float,
            )
            max_eq_residual = float(rollout.max_abs_scaled_residual)
            min_abs_det = float(rollout.min_abs_det)
        else:
            eq_arr = np.zeros(0, dtype=float)
            max_eq_residual = 0.0
            min_abs_det = float(np.min(np.abs(np.asarray(det_nodes, dtype=float))))
        ineq_arr = np.asarray(ineq, dtype=float)
        value_terms = compute_design_value_terms(
            x=arrays["x"],
            T_e=arrays["T_e"],
            T_p=arrays["T_p"],
            n_p=arrays["n_p"],
            n_e=arrays["n_e"],
            mach=arrays["mach"],
            A=arrays["A"],
            J_x=arrays["J_x"],
            E_x=arrays["E_x"],
            B=float(self._config.B),
            seed_fraction=seed_fraction,
            v_p=arrays["v_p"],
            heavy_particle_mass_kg=float(self._config.working_fluid.heavy_particle_mass_kg),
        )
        value_terms_dict = value_terms.to_dict()
        _augment_value_terms_with_hall_diagnostics(
            value_terms_dict,
            x=arrays["x"],
            E_x=arrays["E_x"],
            I_0=float(inlet["I_0"]),
        )
        value_terms_dict["mass_flow_rate_kg_s"] = float(inlet["dot_N"]) * float(
            self._config.working_fluid.heavy_particle_mass_kg
        )
        value_terms_dict["inlet_area_m2"] = float(arrays["A"][0])
        value_terms_dict["outlet_area_m2"] = float(arrays["A"][-1])
        value_terms_dict["outlet_to_inlet_area_ratio"] = float(arrays["A"][-1]) / max(float(arrays["A"][0]), _EPS)
        value_terms_dict["velikhov_margin_penalty"] = float(velikhov_penalty)
        value_terms_dict["raw_design_score"] = float(raw_design_score)
        value_terms_dict["min_T_p_midpoint"] = float(min_tp_midpoints)
        value_terms_dict["min_T_p_all_checks"] = float(min_tp_all)
        value_terms_dict["min_velikhov_margin_midpoint"] = float(min_g_midpoints)
        value_profile = _value_profile_dict(value_terms, objective_profile=self._objective_profile)
        value_profile["terms"] = dict(value_terms_dict)
        diagnostics = {
            "n_intervals": int(n_intervals),
            "finite_profile": bool(
                all(np.all(np.isfinite(arr)) for arr in arrays.values())
                and np.all(np.isfinite(ineq_arr))
                and np.all(np.isfinite(eq_arr))
                and np.isfinite(design_score)
            ),
            "min_T_p": float(min_tp_nodes),
            "min_T_p_midpoint": float(min_tp_midpoints),
            "min_T_p_all_checks": float(min_tp_all),
            "min_velikhov_margin": float(min_g_nodes),
            "min_velikhov_margin_midpoint": float(min_g_midpoints),
            "min_velikhov_margin_all_checks": float(min_g_all),
            "min_mach": float(np.min(arrays["mach"])),
            "max_abs_sigma_logA": float(np.max(np.abs(arrays["sigma_logA"]))),
            "max_ineq_residual": float(np.max(ineq_arr)) if ineq_arr.size else 0.0,
            "constraint_count": int(ineq_arr.size),
            "max_eq_residual": float(max_eq_residual),
            "equality_count": int(eq_arr.size),
            "det_min_abs": float(min_abs_det),
            "mach_spline_max_scaled_residual": float(max_eq_residual),
            "mach_spline_min_abs_det": float(min_abs_det),
            "raw_design_score": float(raw_design_score),
            "velikhov_margin_penalty": float(velikhov_penalty),
            "formulation": self.formulation,
            "objective_profile": self._objective_profile,
            "acceptable": bool(
                (float(np.max(ineq_arr)) <= 1e-7 if ineq_arr.size else True)
                and (max_eq_residual <= float(self._residual_tolerance) if check_equalities else True)
            ),
        }
        return CoarseProfileResult(
            decision_vector=dict(decision),
            inlet_design=InletDesign(
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
            ),
            area_design=SplineAreaDesign.project_from_profile(x=arrays["x"], A=arrays["A"]),
            objective_score=design_score,
            objective_to_minimize=float(-design_score),
            diagnostics=diagnostics,
            x=arrays["x"],
            n_p=arrays["n_p"],
            T_e=arrays["T_e"],
            T_p=arrays["T_p"],
            A=arrays["A"],
            sigma_logA=arrays["sigma_logA"],
            v_p=arrays["v_p"],
            n_e=arrays["n_e"],
            beta=arrays["beta"],
            eta=arrays["eta"],
            Z=arrays["Z"],
            J_x=arrays["J_x"],
            J_y=arrays["J_y"],
            E_x=arrays["E_x"],
            mach=arrays["mach"],
            velikhov_margin=arrays["velikhov_margin"],
            value_terms=value_terms_dict,
            value_profile=value_profile,
        )

    def evaluate_solution(self, solution: MachReducedRollout) -> CoarseProfileResult:
        return self._coarse_result_from_nodes(
            decision={key: float(value) for key, value in solution.decision_vector.items()},
            n_p_nodes=np.asarray(solution.n_p_nodes, dtype=float),
            T_e_nodes=np.asarray(solution.T_e_nodes, dtype=float),
            n_intervals=self._n_intervals,
            rollout=solution,
            check_equalities=True,
        )

    def resample_solution_result(self, result: CoarseProfileResult, *, n_intervals: int) -> CoarseProfileResult:
        x_new = np.linspace(0.0, float(self._config.length), int(n_intervals) + 1, dtype=float)
        n_p_nodes = np.interp(x_new, np.asarray(result.x, dtype=float), np.asarray(result.n_p, dtype=float))
        T_e_nodes = np.interp(x_new, np.asarray(result.x, dtype=float), np.asarray(result.T_e, dtype=float))
        return self._coarse_result_from_nodes(
            decision={key: float(value) for key, value in result.decision_vector.items()},
            n_p_nodes=n_p_nodes,
            T_e_nodes=T_e_nodes,
            n_intervals=int(n_intervals),
            rollout=None,
            check_equalities=False,
        )

    def evaluate(self, vars):
        result = self._maingopy.EvaluationContainer()
        decision = self._decision_from_values(vars)
        rollout = rollout_reduced_mach_generic(
            ops=self._ops,
            config=self._config,
            n_intervals=self._n_intervals,
            decision_vector=decision,
            residual_scales=self._residual_scales,
            newton_steps=self._newton_steps,
        )
        midpoint_closures = _mach_midpoint_closures(
            ops=self._ops,
            config=self._config,
            rollout=rollout,
            decision=decision,
            n_intervals=self._n_intervals,
        )
        power_density_nodes = [
            -rollout.area_nodes["A"][idx] * closure["J_x"] * closure["E_x"] / 1e8
            for idx, closure in enumerate(rollout.closures)
        ]
        raw_design_score = _design_score_generic(
            ops=self._ops,
            outlet_T_e=rollout.T_e_nodes[-1],
            outlet_T_p=rollout.closures[-1]["T_p"],
            outlet_n_p=rollout.n_p_nodes[-1],
            outlet_n_e=rollout.closures[-1]["n_e"],
            inlet_T_e=rollout.inlet["T_e"],
            inlet_T_p=rollout.inlet["T_p"],
            inlet_mach=rollout.inlet["mach"],
            power_density_nodes=power_density_nodes,
            x_nodes=np.asarray(rollout.area_nodes["x"], dtype=float),
            seed_fraction=self._ops.exp(decision["log_seed_fraction"]),
            B=float(self._baseline.B),
            length=float(self._baseline.L),
            objective_profile=self._objective_profile,
            inlet_n_p=rollout.inlet["n_p"],
            inlet_n_e=rollout.inlet["n_e"],
            inlet_v=rollout.inlet["v_in"],
            inlet_A=float(self._baseline.area_scale_m2),
            working_fluid=self._baseline.working_fluid,
        )
        min_g_nodes = _reduce_min(self._ops, [item["G"] for item in rollout.closures])
        min_g_midpoints = _reduce_min(self._ops, [item["G"] for item in midpoint_closures])
        min_g_all = _min_op(self._ops, min_g_nodes, min_g_midpoints)
        min_tp_nodes = _reduce_min(self._ops, [item["T_p"] for item in rollout.closures])
        min_tp_midpoints = _reduce_min(self._ops, [item["T_p"] for item in midpoint_closures])
        min_tp_all = _min_op(self._ops, min_tp_nodes, min_tp_midpoints)
        velikhov_penalty = _velikhov_margin_penalty(self._ops, min_g_all)
        design_score = raw_design_score - velikhov_penalty

        ineq = []
        sigma_max = float(self._baseline.schedule[0]["max_abs_dlogA_dx"])
        tp_min = float(self._baseline.schedule[0].get("tp_min", _TP_MIN))
        mach_min = float(self._baseline.schedule[0].get("mach_min", 0.0) or 0.0)
        for sigma in rollout.area_nodes["sigma_logA"]:
            ineq.append(self._ops.fabs(sigma) - sigma_max)
        for closure in rollout.closures:
            ineq.append(tp_min - closure["T_p"])
            ineq.append(float(_G_HARD_MARGIN) - closure["G"])
            if mach_min > 0.0:
                ineq.append(mach_min - closure["mach"])
        for closure_mid in midpoint_closures:
            ineq.append(float(_G_HARD_MARGIN) - closure_mid["G"])
        ineq.append(rollout.max_abs_scaled_residual - float(self._residual_tolerance))

        result.objective = -design_score
        result.ineq = _model_function(self._maingopy, ineq)
        result.output = [
            self._maingopy.OutputVariable("design_score", design_score),
            self._maingopy.OutputVariable("raw_design_score", raw_design_score),
            self._maingopy.OutputVariable("velikhov_penalty", velikhov_penalty),
            self._maingopy.OutputVariable("mach_spline_max_scaled_residual", rollout.max_abs_scaled_residual),
            self._maingopy.OutputVariable("mach_spline_min_abs_det", rollout.min_abs_det),
            self._maingopy.OutputVariable("inlet_G", rollout.inlet["G"]),
            self._maingopy.OutputVariable("inlet_mach", rollout.inlet["mach"]),
            self._maingopy.OutputVariable("outlet_mach", rollout.closures[-1]["mach"]),
            self._maingopy.OutputVariable("derived_area_outlet", rollout.area_nodes["A"][-1]),
            self._maingopy.OutputVariable(
                "mach_spline_max_abs_sigma",
                _reduce_max(self._ops, [self._ops.fabs(sigma) for sigma in rollout.area_nodes["sigma_logA"]]),
            ),
            self._maingopy.OutputVariable("min_path_Tp_nodes", min_tp_nodes),
            self._maingopy.OutputVariable("min_path_Tp_midpoints", min_tp_midpoints),
            self._maingopy.OutputVariable("min_path_Tp_all", min_tp_all),
            self._maingopy.OutputVariable("min_path_G_nodes", min_g_nodes),
            self._maingopy.OutputVariable("min_path_G_midpoints", min_g_midpoints),
            self._maingopy.OutputVariable("min_path_G_all", min_g_all),
        ]
        return result


class MachSplineLiftedTrapezoidModelBase(MachSplineReducedImplicitModelBase):
    formulation = "mach_spline_trapezoid_lifted"

    def __init__(
        self,
        *,
        baseline: BaselineSeed,
        reference_profile_path: str | Path | None,
        n_intervals: int,
        maingopy_module,
        objective_profile: str = OBJECTIVE_PROFILE_LAB_POC_V2,
        newton_steps: int = 10,
        residual_tolerance: float = 1e-5,
        mach_window_radius: float = 1.0,
        det_floor: float = 1.0,
    ):
        if int(n_intervals) < 1:
            raise ValueError(f"n_intervals must be positive, got {n_intervals!r}.")
        super().__init__(
            baseline=baseline,
            reference_profile_path=reference_profile_path,
            n_intervals=n_intervals,
            maingopy_module=maingopy_module,
            objective_profile=objective_profile,
            newton_steps=newton_steps,
            residual_tolerance=residual_tolerance,
            mach_window_radius=mach_window_radius,
        )
        self._det_floor = float(det_floor)
        if self._det_floor <= 0.0:
            raise ValueError(f"det_floor must be positive, got {self._det_floor!r}.")
        self._decision_variable_specs = list(self._variable_specs)
        self._reference_decision = self._load_reference_decision()
        self._reference_x, self._reference_n_p, self._reference_T_e = self._load_reference_nodes()
        self._reference_dn_dx = self._fit_trapezoid_derivatives(self._reference_n_p)
        self._reference_dTe_dx = self._fit_trapezoid_derivatives(self._reference_T_e)
        trap_specs = self._build_trapezoid_state_specs()
        self._residual_scales = self._build_trapezoid_residual_scales()
        self._variable_specs = [*self._decision_variable_specs, *trap_specs]
        self._initial_point = self._build_trapezoid_initial_point()

    def _load_reference_decision(self) -> dict[str, float]:
        try:
            decision = decision_from_profile(self._reference_profile_path)
        except Exception:
            decision = {
                name: float(value)
                for name, value in zip(MACH_DECISION_NAMES, super()._build_initial_point(), strict=True)
            }
        return {key: float(decision[key]) for key in MACH_DECISION_NAMES}

    def _load_reference_nodes(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        with np.load(self._reference_profile_path) as data:
            x_ref = np.asarray(data["x"], dtype=float)
            n_ref = np.asarray(data["n_p"], dtype=float)
            T_ref = np.asarray(data["T_e"], dtype=float)
        x_nodes = np.linspace(0.0, float(self._config.length), int(self._n_intervals) + 1, dtype=float)
        return x_nodes, np.interp(x_nodes, x_ref, n_ref), np.interp(x_nodes, x_ref, T_ref)

    def _fit_trapezoid_derivatives(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=float).reshape(int(self._n_intervals) + 1)
        dx = float(self._config.length) / float(self._n_intervals)
        slopes = np.diff(values) / dx
        mat = np.zeros((int(self._n_intervals), int(self._n_intervals) + 1), dtype=float)
        for k in range(int(self._n_intervals)):
            mat[k, k] = 1.0
            mat[k, k + 1] = 1.0
        deriv, *_ = np.linalg.lstsq(mat, 2.0 * slopes, rcond=None)
        return np.asarray(deriv, dtype=float)

    def _build_trapezoid_state_specs(self) -> list[tuple[float, float, str]]:
        ref_n = np.asarray(self._reference_n_p, dtype=float)
        ref_T = np.asarray(self._reference_T_e, dtype=float)
        ref_dn = np.asarray(self._reference_dn_dx, dtype=float)
        ref_dT = np.asarray(self._reference_dTe_dx, dtype=float)
        dx = float(self._config.length) / float(self._n_intervals)
        n_global_lb = max(1.0, 0.25 * float(np.min(ref_n)))
        n_global_ub = max(n_global_lb * 1.01, 4.0 * float(np.max(ref_n)))
        T_global_lb = max(1.0, 0.25 * float(np.min(ref_T)))
        T_global_ub = max(T_global_lb * 1.01, 2.5 * float(np.max(ref_T)))
        dn_global_span = max((n_global_ub - n_global_lb) / max(dx, _EPS), 1.0)
        dT_global_span = max((T_global_ub - T_global_lb) / max(dx, _EPS), 1.0)
        self._n_tail_center: list[float] = []
        self._n_tail_scale: list[float] = []
        self._T_tail_center: list[float] = []
        self._T_tail_scale: list[float] = []
        self._dn_center: list[float] = []
        self._dn_scale: list[float] = []
        self._dT_center: list[float] = []
        self._dT_scale: list[float] = []
        n_specs: list[tuple[float, float, str]] = []
        T_specs: list[tuple[float, float, str]] = []
        dn_specs: list[tuple[float, float, str]] = []
        dT_specs: list[tuple[float, float, str]] = []
        for idx in range(1, int(self._n_intervals) + 1):
            n_lb = max(n_global_lb, 0.5 * float(ref_n[idx]))
            n_ub = min(n_global_ub, 2.0 * float(ref_n[idx]))
            if n_ub <= n_lb:
                n_ub = max(n_global_ub, n_lb * 1.01)
            T_lb = max(T_global_lb, 0.5 * float(ref_T[idx]))
            T_ub = min(T_global_ub, 1.75 * float(ref_T[idx]))
            if T_ub <= T_lb:
                T_ub = max(T_global_ub, T_lb * 1.01)
            n_center, n_scale, n_scaled_lb, n_scaled_ub = _scaled_interval(
                n_lb,
                n_ub,
                float(ref_n[idx]),
                min_scale=max(1.0, 0.05 * abs(float(ref_n[idx]))),
            )
            T_center, T_scale, T_scaled_lb, T_scaled_ub = _scaled_interval(
                T_lb,
                T_ub,
                float(ref_T[idx]),
                min_scale=max(1.0, 0.05 * abs(float(ref_T[idx]))),
            )
            self._n_tail_center.append(float(n_center))
            self._n_tail_scale.append(float(n_scale))
            self._T_tail_center.append(float(T_center))
            self._T_tail_scale.append(float(T_scale))
            n_specs.append((float(n_scaled_lb), float(n_scaled_ub), f"n_p_hat_{idx}"))
            T_specs.append((float(T_scaled_lb), float(T_scaled_ub), f"T_e_hat_{idx}"))
        for idx in range(int(self._n_intervals) + 1):
            dn_span = max(4.0 * abs(float(ref_dn[idx])), dn_global_span)
            dT_span = max(4.0 * abs(float(ref_dT[idx])), dT_global_span)
            dn_center, dn_scale, dn_scaled_lb, dn_scaled_ub = _scaled_interval(
                float(ref_dn[idx]) - dn_span,
                float(ref_dn[idx]) + dn_span,
                float(ref_dn[idx]),
                min_scale=max(1.0, 0.05 * dn_span),
            )
            dT_center, dT_scale, dT_scaled_lb, dT_scaled_ub = _scaled_interval(
                float(ref_dT[idx]) - dT_span,
                float(ref_dT[idx]) + dT_span,
                float(ref_dT[idx]),
                min_scale=max(1.0, 0.05 * dT_span),
            )
            self._dn_center.append(float(dn_center))
            self._dn_scale.append(float(dn_scale))
            self._dT_center.append(float(dT_center))
            self._dT_scale.append(float(dT_scale))
            dn_specs.append((float(dn_scaled_lb), float(dn_scaled_ub), f"dn_dx_hat_{idx}"))
            dT_specs.append((float(dT_scaled_lb), float(dT_scaled_ub), f"dTe_dx_hat_{idx}"))
        return [*n_specs, *T_specs, *dn_specs, *dT_specs]

    @staticmethod
    def _encode_scaled(values: np.ndarray, centers: list[float], scales: list[float]) -> list[float]:
        return [
            float((float(value) - float(center)) / float(scale))
            for value, center, scale in zip(np.asarray(values, dtype=float), centers, scales, strict=True)
        ]

    @staticmethod
    def _decode_scaled(raw_values, centers: list[float], scales: list[float]) -> list[Any]:
        return [float(center) + float(scale) * value for value, center, scale in zip(raw_values, centers, scales, strict=True)]

    def _build_trapezoid_initial_point(self) -> list[float]:
        decision_values = []
        for lower, upper, name in self._decision_variable_specs:
            value = float(self._reference_decision.get(str(name), 0.5 * (float(lower) + float(upper))))
            value = min(max(value, float(lower)), float(upper))
            decision_values.append(value)
        return [
            *decision_values,
            *self._encode_scaled(self._reference_n_p[1:], self._n_tail_center, self._n_tail_scale),
            *self._encode_scaled(self._reference_T_e[1:], self._T_tail_center, self._T_tail_scale),
            *self._encode_scaled(self._reference_dn_dx, self._dn_center, self._dn_scale),
            *self._encode_scaled(self._reference_dTe_dx, self._dT_center, self._dT_scale),
        ]

    def _split_raw(self, values):
        values = list(values)
        expected = 8 + 2 * int(self._n_intervals) + 2 * (int(self._n_intervals) + 1)
        if len(values) != expected:
            raise ValueError(f"mach-spline trapezoid solution size mismatch: got {len(values)}, expected {expected}.")
        idx = 0
        decision_raw = values[idx : idx + 8]
        idx += 8
        n_tail_raw = values[idx : idx + int(self._n_intervals)]
        idx += int(self._n_intervals)
        T_tail_raw = values[idx : idx + int(self._n_intervals)]
        idx += int(self._n_intervals)
        dn_raw = values[idx : idx + int(self._n_intervals) + 1]
        idx += int(self._n_intervals) + 1
        dT_raw = values[idx : idx + int(self._n_intervals) + 1]
        return decision_raw, n_tail_raw, T_tail_raw, dn_raw, dT_raw

    def _decision_from_values(self, values) -> dict[str, Any]:
        values = list(values)
        if len(values) == len(MACH_DECISION_NAMES):
            decision_raw = values
        elif len(values) == self.total_variables:
            decision_raw = values[: len(MACH_DECISION_NAMES)]
        else:
            raise ValueError(
                f"mach-spline trapezoid solution size mismatch: got {len(values)}, "
                f"expected {len(MACH_DECISION_NAMES)} or {self.total_variables}."
            )
        return {name: value for name, value in zip(MACH_DECISION_NAMES, decision_raw, strict=True)}

    def set_initial_point_from_decision(self, decision: dict[str, Any]) -> dict[str, float]:
        decision = dict(decision)
        if "log_n_p_in" not in decision and "n_p_in" in decision:
            decision["log_n_p_in"] = math.log(float(decision["n_p_in"]))
        if "log_seed_fraction" not in decision and "seed_fraction" in decision:
            decision["log_seed_fraction"] = math.log(float(decision["seed_fraction"]))
        values = list(self._initial_point)
        normalized: dict[str, float] = {}
        for idx, (lower, upper, name) in enumerate(self._decision_variable_specs):
            value = float(decision.get(name, values[idx]))
            if not math.isfinite(value):
                raise ValueError(f"non-finite initial decision value for {name!r}: {value!r}.")
            tol = 1e-8 * max(1.0, abs(float(lower)), abs(float(upper)))
            if value < float(lower) - tol or value > float(upper) + tol:
                raise ValueError(
                    f"initial decision value for {name!r}={value:.16g} is outside "
                    f"bounds [{float(lower):.16g}, {float(upper):.16g}]."
                )
            values[idx] = value
            normalized[str(name)] = value
        self._initial_point = values
        return normalized

    def _sigma_from_areas(self, ops, area_values: list[Any]) -> list[Any]:
        dx = float(self._config.length) / float(self._n_intervals)
        sigma_values = []
        for idx in range(int(self._n_intervals) + 1):
            denom = float(dx) * _positive_denom(ops, area_values[idx], _EPS)
            if idx == 0:
                sigma_values.append((area_values[1] - area_values[0]) / denom)
            elif idx == int(self._n_intervals):
                sigma_values.append((area_values[idx] - area_values[idx - 1]) / denom)
            else:
                sigma_values.append((area_values[idx + 1] - area_values[idx - 1]) / (2.0 * denom))
        return sigma_values

    def _build_trapezoid_residual_scales(self) -> ImplicitResidualScales:
        ops = _ops_for_numeric()
        decision = {key: float(value) for key, value in self._reference_decision.items()}
        seed_fraction = math.exp(float(decision["log_seed_fraction"]))
        inlet = _inlet_design_generic(
            ops=ops,
            n_p_in=math.exp(float(decision["log_n_p_in"])),
            T_e_in=float(decision["T_e_in"]),
            Z_in=float(decision["Z_in"]),
            I_0=float(decision["I_0"]),
            seed_fraction=seed_fraction,
            B=float(self._config.B),
            inlet_A=float(self._config.area_scale_m2),
            working_fluid=self._config.working_fluid,
        )
        mach_nodes = _mach_design_nodes_generic(
            ops=ops,
            mach_design=MachSplineDesign(m1=decision["m1"], m2=decision["m2"], m3=decision["m3"]),
            length=float(self._config.length),
            n_intervals=int(self._n_intervals),
            mach_in=float(inlet["mach"]),
        )
        area_values = []
        for idx in range(int(self._n_intervals) + 1):
            area_values.append(
                float(
                    _mach_area_closure_generic(
                        ops=ops,
                        n_p=float(self._reference_n_p[idx]),
                        T_e=float(self._reference_T_e[idx]),
                        mach=float(mach_nodes["mach"][idx]),
                        dot_N=float(inlet["dot_N"]),
                        I_0=float(decision["I_0"]),
                        seed_fraction=seed_fraction,
                        B=float(self._config.B),
                        working_fluid=self._config.working_fluid,
                    )["A_safe"]
                )
            )
        sigma_values = [float(item) for item in self._sigma_from_areas(ops, area_values)]
        step_n = []
        step_Te = []
        momentum = []
        energy = []
        dx = float(self._config.length) / float(self._n_intervals)
        for idx in range(int(self._n_intervals)):
            step_n.append(
                max(
                    abs(float(self._reference_n_p[idx + 1] - self._reference_n_p[idx])),
                    abs(0.5 * dx * float(self._reference_dn_dx[idx] + self._reference_dn_dx[idx + 1])),
                    abs(float(self._reference_n_p[idx + 1])),
                    1.0,
                )
            )
            step_Te.append(
                max(
                    abs(float(self._reference_T_e[idx + 1] - self._reference_T_e[idx])),
                    abs(0.5 * dx * float(self._reference_dTe_dx[idx] + self._reference_dTe_dx[idx + 1])),
                    abs(float(self._reference_T_e[idx + 1])),
                    1.0,
                )
            )
        for idx in range(int(self._n_intervals) + 1):
            _, terms = _dynamic_system_terms(
                ops=ops,
                n_p=float(self._reference_n_p[idx]),
                T_e=float(self._reference_T_e[idx]),
                A=float(area_values[idx]),
                sigma=float(sigma_values[idx]),
                dot_N=float(inlet["dot_N"]),
                I_0=float(decision["I_0"]),
                seed_fraction=seed_fraction,
                B=float(self._config.B),
                working_fluid=self._config.working_fluid,
            )
            momentum.append(
                max(
                    abs(float(terms["rhs_m"])),
                    abs(float(terms["M11"]) * float(self._reference_dn_dx[idx])),
                    abs(float(terms["M12"]) * float(self._reference_dTe_dx[idx])),
                    1.0,
                )
            )
            energy.append(
                max(
                    abs(float(terms["rhs_e"])),
                    abs(float(terms["E11"]) * float(self._reference_dn_dx[idx])),
                    abs(float(terms["E12"]) * float(self._reference_dTe_dx[idx])),
                    1.0,
                )
            )
        return ImplicitResidualScales(
            step_n=np.asarray(step_n, dtype=float),
            step_Te=np.asarray(step_Te, dtype=float),
            momentum=np.asarray(momentum, dtype=float),
            energy=np.asarray(energy, dtype=float),
        )

    def _rollout_from_values(self, values, *, ops) -> MachTrapezoidRollout:
        decision_raw, n_tail_raw, T_tail_raw, dn_raw, dT_raw = self._split_raw(values)
        decision = {name: value for name, value in zip(MACH_DECISION_NAMES, decision_raw, strict=True)}
        seed_fraction = ops.exp(decision["log_seed_fraction"])
        inlet = _inlet_design_generic(
            ops=ops,
            n_p_in=ops.exp(decision["log_n_p_in"]),
            T_e_in=decision["T_e_in"],
            Z_in=decision["Z_in"],
            I_0=decision["I_0"],
            seed_fraction=seed_fraction,
            B=float(self._config.B),
            inlet_A=float(self._config.area_scale_m2),
            working_fluid=self._config.working_fluid,
        )
        mach_nodes = _mach_design_nodes_generic(
            ops=ops,
            mach_design=MachSplineDesign(m1=decision["m1"], m2=decision["m2"], m3=decision["m3"]),
            length=float(self._config.length),
            n_intervals=int(self._n_intervals),
            mach_in=inlet["mach"],
        )
        n_nodes = [inlet["n_p"], *self._decode_scaled(n_tail_raw, self._n_tail_center, self._n_tail_scale)]
        T_nodes = [inlet["T_e"], *self._decode_scaled(T_tail_raw, self._T_tail_center, self._T_tail_scale)]
        dn_dx = self._decode_scaled(dn_raw, self._dn_center, self._dn_scale)
        dTe_dx = self._decode_scaled(dT_raw, self._dT_center, self._dT_scale)
        area_values = []
        for idx in range(int(self._n_intervals) + 1):
            area_values.append(
                _mach_area_closure_generic(
                    ops=ops,
                    n_p=n_nodes[idx],
                    T_e=T_nodes[idx],
                    mach=mach_nodes["mach"][idx],
                    dot_N=inlet["dot_N"],
                    I_0=decision["I_0"],
                    seed_fraction=seed_fraction,
                    B=float(self._config.B),
                    working_fluid=self._config.working_fluid,
                )["A_safe"]
            )
        sigma_values = self._sigma_from_areas(ops, area_values)
        closures = []
        terms_by_node = []
        for idx in range(int(self._n_intervals) + 1):
            closure, terms = _dynamic_system_terms(
                ops=ops,
                n_p=n_nodes[idx],
                T_e=T_nodes[idx],
                A=area_values[idx],
                sigma=sigma_values[idx],
                dot_N=inlet["dot_N"],
                I_0=decision["I_0"],
                seed_fraction=seed_fraction,
                B=float(self._config.B),
                working_fluid=self._config.working_fluid,
            )
            closures.append(closure)
            terms_by_node.append(terms)
        dx = float(self._config.length) / float(self._n_intervals)
        scaled_step_n = []
        scaled_step_Te = []
        for idx in range(int(self._n_intervals)):
            scaled_step_n.append(
                (n_nodes[idx + 1] - n_nodes[idx] - 0.5 * dx * (dn_dx[idx] + dn_dx[idx + 1]))
                / float(self._residual_scales.step_n[idx])
            )
            scaled_step_Te.append(
                (T_nodes[idx + 1] - T_nodes[idx] - 0.5 * dx * (dTe_dx[idx] + dTe_dx[idx + 1]))
                / float(self._residual_scales.step_Te[idx])
            )
        scaled_momentum = []
        scaled_energy = []
        for idx, terms in enumerate(terms_by_node):
            scaled_momentum.append(
                (terms["M11"] * dn_dx[idx] + terms["M12"] * dTe_dx[idx] - terms["rhs_m"])
                / float(self._residual_scales.momentum[idx])
            )
            scaled_energy.append(
                (terms["E11"] * dn_dx[idx] + terms["E12"] * dTe_dx[idx] - terms["rhs_e"])
                / float(self._residual_scales.energy[idx])
            )
        residuals = [*scaled_step_n, *scaled_step_Te, *scaled_momentum, *scaled_energy]
        abs_residuals = [ops.fabs(item) for item in residuals]
        det_values = [item["det"] for item in terms_by_node]
        det_abs = [ops.fabs(item) for item in det_values]
        return MachTrapezoidRollout(
            decision_vector=dict(decision),
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
            dn_dx=dn_dx,
            dTe_dx=dTe_dx,
            scaled_step_n_residuals=scaled_step_n,
            scaled_step_Te_residuals=scaled_step_Te,
            scaled_momentum_residuals=scaled_momentum,
            scaled_energy_residuals=scaled_energy,
            closures=closures,
            terms_by_node=terms_by_node,
            max_abs_scaled_residual=_reduce_max(ops, abs_residuals),
            min_abs_det=_reduce_min(ops, det_abs),
            min_signed_det=_reduce_min(ops, det_values),
        )

    def decode_solution_point(self, values) -> MachTrapezoidRollout:
        rollout = self._rollout_from_values(values, ops=_ops_for_numeric())
        return MachTrapezoidRollout(
            decision_vector={key: float(value) for key, value in rollout.decision_vector.items()},
            inlet={key: float(value) for key, value in rollout.inlet.items()},
            mach_nodes={
                key: np.asarray(value, dtype=float) if key in {"x_norm", "x"} else [float(item) for item in value]
                for key, value in rollout.mach_nodes.items()
            },
            area_nodes={
                key: np.asarray(value, dtype=float) if key in {"x_norm", "x"} else [float(item) for item in value]
                for key, value in rollout.area_nodes.items()
            },
            n_p_nodes=[float(item) for item in rollout.n_p_nodes],
            T_e_nodes=[float(item) for item in rollout.T_e_nodes],
            dn_dx=[float(item) for item in rollout.dn_dx],
            dTe_dx=[float(item) for item in rollout.dTe_dx],
            scaled_step_n_residuals=[float(item) for item in rollout.scaled_step_n_residuals],
            scaled_step_Te_residuals=[float(item) for item in rollout.scaled_step_Te_residuals],
            scaled_momentum_residuals=[float(item) for item in rollout.scaled_momentum_residuals],
            scaled_energy_residuals=[float(item) for item in rollout.scaled_energy_residuals],
            closures=[{key: float(value) for key, value in item.items()} for item in rollout.closures],
            terms_by_node=[{key: float(value) for key, value in item.items()} for item in rollout.terms_by_node],
            max_abs_scaled_residual=float(rollout.max_abs_scaled_residual),
            min_abs_det=float(rollout.min_abs_det),
            min_signed_det=float(rollout.min_signed_det),
        )

    def summary_metadata(self) -> dict[str, Any]:
        return {
            "formulation": self.formulation,
            "transcription": "trapezoid_direct_lifted_node_derivatives",
            "reference_profile_path": str(self._reference_profile_path),
            "mach_window_radius": float(self._mach_window_radius),
            "mach_design_nominal": self._mach_design_nominal.to_dict(),
            "state_tail_variable_count": int(2 * self._n_intervals),
            "derivative_node_variable_count": int(2 * (self._n_intervals + 1)),
            "equality_count": int(4 * self._n_intervals + 2),
            "det_branch_sign": "positive",
            "det_floor": float(self._det_floor),
            "det_constraint_scope": "profile_nodes",
            "residual_scale_momentum_min": float(np.min(self._residual_scales.momentum)),
            "residual_scale_momentum_max": float(np.max(self._residual_scales.momentum)),
            "residual_scale_energy_min": float(np.min(self._residual_scales.energy)),
            "residual_scale_energy_max": float(np.max(self._residual_scales.energy)),
        }

    def evaluate_solution(self, solution: MachTrapezoidRollout) -> CoarseProfileResult:
        result = self._coarse_result_from_nodes(
            decision={key: float(value) for key, value in solution.decision_vector.items()},
            n_p_nodes=np.asarray(solution.n_p_nodes, dtype=float),
            T_e_nodes=np.asarray(solution.T_e_nodes, dtype=float),
            n_intervals=self._n_intervals,
            rollout=None,
            check_equalities=False,
        )
        eq = np.asarray(
            [
                *solution.scaled_step_n_residuals,
                *solution.scaled_step_Te_residuals,
                *solution.scaled_momentum_residuals,
                *solution.scaled_energy_residuals,
            ],
            dtype=float,
        )
        diagnostics = dict(result.diagnostics)
        max_eq_residual = float(np.max(np.abs(eq))) if eq.size else 0.0
        det_branch_acceptable = bool(float(solution.min_signed_det) >= float(self._det_floor))
        diagnostics.update(
            {
                "formulation": self.formulation,
                "coarse_model": "mach_spline_trapezoid",
                "transcription": "trapezoid_direct_lifted_node_derivatives",
                "max_eq_residual": max_eq_residual,
                "equality_count": int(eq.size),
                "mach_spline_max_scaled_residual": max_eq_residual,
                "mach_spline_min_abs_det": float(solution.min_abs_det),
                "mach_spline_min_signed_det": float(solution.min_signed_det),
                "det_branch_sign": "positive",
                "det_floor": float(self._det_floor),
                "det_branch_acceptable": det_branch_acceptable,
                "acceptable": bool(
                    result.diagnostics.get("finite_profile", False)
                    and float(result.diagnostics.get("max_ineq_residual", 0.0)) <= 1e-7
                    and max_eq_residual <= float(self._residual_tolerance)
                    and det_branch_acceptable
                ),
            }
        )
        return replace(result, diagnostics=diagnostics)

    def resample_solution_result(self, result: CoarseProfileResult, *, n_intervals: int) -> CoarseProfileResult:
        x_new = np.linspace(0.0, float(self._config.length), int(n_intervals) + 1, dtype=float)
        n_p_nodes = np.interp(x_new, np.asarray(result.x, dtype=float), np.asarray(result.n_p, dtype=float))
        T_e_nodes = np.interp(x_new, np.asarray(result.x, dtype=float), np.asarray(result.T_e, dtype=float))
        coarse = self._coarse_result_from_nodes(
            decision={key: float(value) for key, value in result.decision_vector.items()},
            n_p_nodes=n_p_nodes,
            T_e_nodes=T_e_nodes,
            n_intervals=int(n_intervals),
            rollout=None,
            check_equalities=False,
        )
        diagnostics = dict(coarse.diagnostics)
        diagnostics.update(
            {
                "formulation": self.formulation,
                "coarse_model": "mach_spline_trapezoid",
                "transcription": "trapezoid_resampled_profile_posthoc",
            }
        )
        return replace(coarse, diagnostics=diagnostics)

    def evaluate(self, vars):
        result = self._maingopy.EvaluationContainer()
        rollout = self._rollout_from_values(vars, ops=self._ops)
        decision = rollout.decision_vector
        midpoint_closures = _mach_midpoint_closures(
            ops=self._ops,
            config=self._config,
            rollout=rollout,
            decision=decision,
            n_intervals=self._n_intervals,
        )
        power_density_nodes = [
            -rollout.area_nodes["A"][idx] * closure["J_x"] * closure["E_x"] / 1e8
            for idx, closure in enumerate(rollout.closures)
        ]
        raw_design_score = _design_score_generic(
            ops=self._ops,
            outlet_T_e=rollout.T_e_nodes[-1],
            outlet_T_p=rollout.closures[-1]["T_p"],
            outlet_n_p=rollout.n_p_nodes[-1],
            outlet_n_e=rollout.closures[-1]["n_e"],
            inlet_T_e=rollout.inlet["T_e"],
            inlet_T_p=rollout.inlet["T_p"],
            inlet_mach=rollout.inlet["mach"],
            power_density_nodes=power_density_nodes,
            x_nodes=np.asarray(rollout.area_nodes["x"], dtype=float),
            seed_fraction=self._ops.exp(decision["log_seed_fraction"]),
            B=float(self._baseline.B),
            length=float(self._baseline.L),
            objective_profile=self._objective_profile,
            inlet_n_p=rollout.inlet["n_p"],
            inlet_n_e=rollout.inlet["n_e"],
            inlet_v=rollout.inlet["v_in"],
            inlet_A=float(self._baseline.area_scale_m2),
            working_fluid=self._baseline.working_fluid,
        )
        min_g_nodes = _reduce_min(self._ops, [item["G"] for item in rollout.closures])
        min_g_midpoints = _reduce_min(self._ops, [item["G"] for item in midpoint_closures])
        min_g_all = _min_op(self._ops, min_g_nodes, min_g_midpoints)
        min_tp_nodes = _reduce_min(self._ops, [item["T_p"] for item in rollout.closures])
        min_tp_midpoints = _reduce_min(self._ops, [item["T_p"] for item in midpoint_closures])
        min_tp_all = _min_op(self._ops, min_tp_nodes, min_tp_midpoints)
        velikhov_penalty = _velikhov_margin_penalty(self._ops, min_g_all)
        design_score = raw_design_score - velikhov_penalty

        schedule0 = self._baseline.schedule[0]
        sigma_max = float(schedule0["max_abs_dlogA_dx"])
        tp_min = float(schedule0.get("tp_min", _TP_MIN))
        mach_min = float(schedule0.get("mach_min", 0.0) or 0.0)
        ineq = []
        for sigma in rollout.area_nodes["sigma_logA"]:
            ineq.append(self._ops.fabs(sigma) - sigma_max)
        for closure in rollout.closures:
            ineq.append(tp_min - closure["T_p"])
            ineq.append(float(_G_HARD_MARGIN) - closure["G"])
            if mach_min > 0.0:
                ineq.append(mach_min - closure["mach"])
        for closure_mid in midpoint_closures:
            ineq.append(float(_G_HARD_MARGIN) - closure_mid["G"])
        ineq.extend(float(self._det_floor) - terms["det"] for terms in rollout.terms_by_node)
        eq = [
            *rollout.scaled_step_n_residuals,
            *rollout.scaled_step_Te_residuals,
            *rollout.scaled_momentum_residuals,
            *rollout.scaled_energy_residuals,
        ]

        result.objective = -design_score
        result.ineq = _model_function(self._maingopy, ineq)
        result.eq = _model_function(self._maingopy, eq)
        result.output = [
            self._maingopy.OutputVariable("design_score", design_score),
            self._maingopy.OutputVariable("raw_design_score", raw_design_score),
            self._maingopy.OutputVariable("velikhov_penalty", velikhov_penalty),
            self._maingopy.OutputVariable("mach_spline_max_scaled_residual", rollout.max_abs_scaled_residual),
            self._maingopy.OutputVariable("mach_spline_min_abs_det", rollout.min_abs_det),
            self._maingopy.OutputVariable("mach_spline_min_signed_det", rollout.min_signed_det),
            self._maingopy.OutputVariable("mach_spline_det_floor", float(self._det_floor)),
            self._maingopy.OutputVariable("inlet_G", rollout.inlet["G"]),
            self._maingopy.OutputVariable("inlet_mach", rollout.inlet["mach"]),
            self._maingopy.OutputVariable("outlet_mach", rollout.closures[-1]["mach"]),
            self._maingopy.OutputVariable("derived_area_outlet", rollout.area_nodes["A"][-1]),
            self._maingopy.OutputVariable(
                "mach_spline_max_abs_sigma",
                _reduce_max(self._ops, [self._ops.fabs(sigma) for sigma in rollout.area_nodes["sigma_logA"]]),
            ),
            self._maingopy.OutputVariable("min_path_Tp_nodes", min_tp_nodes),
            self._maingopy.OutputVariable("min_path_Tp_midpoints", min_tp_midpoints),
            self._maingopy.OutputVariable("min_path_Tp_all", min_tp_all),
            self._maingopy.OutputVariable("min_path_G_nodes", min_g_nodes),
            self._maingopy.OutputVariable("min_path_G_midpoints", min_g_midpoints),
            self._maingopy.OutputVariable("min_path_G_all", min_g_all),
        ]
        return result


class MachSplineTrapezoidModelBase(MachSplineReducedImplicitModelBase):
    formulation = "mach_spline_trapezoid_parametric"
    _INLET_DECISION_NAMES = MACH_DECISION_NAMES[:5]
    _STATE_PARAM_NAMES = (
        "log_n_ratio_1",
        "log_n_ratio_2",
        "log_n_ratio_3",
        "log_Te_ratio_1",
        "log_Te_ratio_2",
        "log_Te_ratio_3",
    )

    def __init__(
        self,
        *,
        baseline: BaselineSeed,
        reference_profile_path: str | Path | None,
        n_intervals: int,
        maingopy_module,
        objective_profile: str = OBJECTIVE_PROFILE_LAB_POC_V2,
        newton_steps: int = 10,
        residual_tolerance: float = 5e-2,
        mach_window_radius: float = 1.0,
        state_window_radius: float = 0.5,
        residual_penalty_weight: float = 10.0,
        det_floor: float = 1.0,
        fixed_inlet: bool = False,
    ):
        super().__init__(
            baseline=baseline,
            reference_profile_path=reference_profile_path,
            n_intervals=n_intervals,
            maingopy_module=maingopy_module,
            objective_profile=objective_profile,
            newton_steps=newton_steps,
            residual_tolerance=residual_tolerance,
            mach_window_radius=mach_window_radius,
        )
        self._state_window_radius = float(state_window_radius)
        self._residual_penalty_weight = float(residual_penalty_weight)
        self._det_floor = float(det_floor)
        self._fixed_inlet = bool(fixed_inlet)
        if self._det_floor <= 0.0:
            raise ValueError(f"det_floor must be positive, got {self._det_floor!r}.")
        self._full_decision_variable_specs = list(self._variable_specs)
        self._reference_decision = self._load_parametric_reference_decision()
        self._fixed_inlet_decision = {
            name: float(self._reference_decision[name])
            for name in self._INLET_DECISION_NAMES
        }
        if self._fixed_inlet:
            self._decision_variable_specs = [
                spec for spec in self._full_decision_variable_specs
                if str(spec[2]) not in self._INLET_DECISION_NAMES
            ]
        else:
            self._decision_variable_specs = list(self._full_decision_variable_specs)
        self._state_nominal = self._project_reference_state_splines()
        self._variable_specs = [*self._decision_variable_specs, *self._build_state_param_specs()]
        self._initial_point = self._build_parametric_initial_point()
        self._residual_scales = self._build_parametric_residual_scales()

    def _load_parametric_reference_decision(self) -> dict[str, float]:
        try:
            decision = decision_from_profile(self._reference_profile_path)
        except Exception:
            decision = {
                name: float(value)
                for name, value in zip(MACH_DECISION_NAMES, super()._build_initial_point(), strict=True)
            }
        return {key: float(decision[key]) for key in MACH_DECISION_NAMES}

    def _project_reference_state_splines(self) -> dict[str, float]:
        with np.load(self._reference_profile_path) as data:
            x = np.asarray(data["x"], dtype=float)
            n_p = np.asarray(data["n_p"], dtype=float)
            T_e = np.asarray(data["T_e"], dtype=float)
        n_design = SplineAreaDesign.project_from_profile(x=x, A=n_p / max(float(n_p[0]), _EPS))
        T_design = SplineAreaDesign.project_from_profile(x=x, A=T_e / max(float(T_e[0]), _EPS))
        return {
            "log_n_ratio_1": float(n_design.a1),
            "log_n_ratio_2": float(n_design.a2),
            "log_n_ratio_3": float(n_design.a3),
            "log_Te_ratio_1": float(T_design.a1),
            "log_Te_ratio_2": float(T_design.a2),
            "log_Te_ratio_3": float(T_design.a3),
        }

    def _build_state_param_specs(self) -> list[tuple[float, float, str]]:
        specs = []
        radius = float(self._state_window_radius)
        lower_bound = SplineAreaDesign.lower_bound()
        upper_bound = SplineAreaDesign.upper_bound()
        for name in self._STATE_PARAM_NAMES:
            value = float(self._state_nominal[name])
            lower = max(lower_bound, value - radius)
            upper = min(upper_bound, value + radius)
            if lower >= upper:
                lower = lower_bound
                upper = upper_bound
            specs.append((float(lower), float(upper), str(name)))
        return specs

    def _build_parametric_initial_point(self) -> list[float]:
        decision_values = []
        for lower, upper, name in self._decision_variable_specs:
            value = float(self._reference_decision.get(str(name), 0.5 * (float(lower) + float(upper))))
            decision_values.append(min(max(value, float(lower)), float(upper)))
        state_values = []
        for lower, upper, name in self._build_state_param_specs():
            value = float(self._state_nominal[str(name)])
            state_values.append(min(max(value, float(lower)), float(upper)))
        return [*decision_values, *state_values]

    def _decision_from_values(self, values) -> dict[str, Any]:
        values = list(values)
        exposed_count = len(self._decision_variable_specs)
        if len(values) == len(MACH_DECISION_NAMES):
            return {name: value for name, value in zip(MACH_DECISION_NAMES, values, strict=True)}
        if len(values) == self.total_variables:
            values = values[:exposed_count]
        if len(values) != exposed_count:
            raise ValueError(
                f"mach-spline trapezoid solution size mismatch: got {len(values)}, "
                f"expected {exposed_count}, {len(MACH_DECISION_NAMES)}, or {self.total_variables}."
            )
        decision = dict(self._fixed_inlet_decision)
        for idx, (_, _, name) in enumerate(self._decision_variable_specs):
            decision[str(name)] = values[idx]
        return decision

    def _split_parametric_values(self, values):
        values = list(values)
        exposed_count = len(self._decision_variable_specs)
        expected = exposed_count + len(self._STATE_PARAM_NAMES)
        full_expected = len(MACH_DECISION_NAMES) + len(self._STATE_PARAM_NAMES)
        if len(values) == full_expected:
            decision = {name: values[idx] for idx, name in enumerate(MACH_DECISION_NAMES)}
            offset = len(MACH_DECISION_NAMES)
        elif len(values) == expected:
            decision = dict(self._fixed_inlet_decision)
            for idx, (_, _, name) in enumerate(self._decision_variable_specs):
                decision[str(name)] = values[idx]
            offset = exposed_count
        else:
            raise ValueError(
                f"mach-spline parametric trapezoid solution size mismatch: got {len(values)}, "
                f"expected {expected} exposed values or {full_expected} full values."
            )
        state = {name: values[offset + idx] for idx, name in enumerate(self._STATE_PARAM_NAMES)}
        return decision, state

    def set_initial_point_from_decision(self, decision: dict[str, Any]) -> dict[str, float]:
        decision = dict(decision)
        if "log_n_p_in" not in decision and "n_p_in" in decision:
            decision["log_n_p_in"] = math.log(float(decision["n_p_in"]))
        if "log_seed_fraction" not in decision and "seed_fraction" in decision:
            decision["log_seed_fraction"] = math.log(float(decision["seed_fraction"]))
        if self._fixed_inlet:
            for name in self._INLET_DECISION_NAMES:
                if name not in decision:
                    continue
                value = float(decision[name])
                if not math.isfinite(value):
                    raise ValueError(f"non-finite fixed inlet decision value for {name!r}: {value!r}.")
                self._fixed_inlet_decision[name] = value
        values = list(self._initial_point)
        normalized: dict[str, float] = {}
        for idx, (lower, upper, name) in enumerate(self._decision_variable_specs):
            value = float(decision.get(name, values[idx]))
            if not math.isfinite(value):
                raise ValueError(f"non-finite initial decision value for {name!r}: {value!r}.")
            tol = 1e-8 * max(1.0, abs(float(lower)), abs(float(upper)))
            if value < float(lower) - tol or value > float(upper) + tol:
                raise ValueError(
                    f"initial decision value for {name!r}={value:.16g} is outside "
                    f"bounds [{float(lower):.16g}, {float(upper):.16g}]."
                )
            values[idx] = value
            normalized[str(name)] = value
        state_offset = len(self._decision_variable_specs)
        for local_idx, (lower, upper, name) in enumerate(self._variable_specs[state_offset:]):
            if name not in decision:
                continue
            idx = state_offset + local_idx
            value = float(decision[name])
            if not math.isfinite(value):
                raise ValueError(f"non-finite initial decision value for {name!r}: {value!r}.")
            tol = 1e-8 * max(1.0, abs(float(lower)), abs(float(upper)))
            if value < float(lower) - tol or value > float(upper) + tol:
                raise ValueError(
                    f"initial decision value for {name!r}={value:.16g} is outside "
                    f"bounds [{float(lower):.16g}, {float(upper):.16g}]."
                )
            values[idx] = value
            normalized[str(name)] = value
        self._initial_point = values
        self._residual_scales = self._build_parametric_residual_scales()
        return normalized

    def _evaluate_state_splines(self, *, ops, inlet: dict[str, Any], state: dict[str, Any]) -> tuple[list[Any], list[Any], list[Any], list[Any]]:
        x_norm = np.linspace(0.0, 1.0, int(self._n_intervals) + 1, dtype=float)
        basis, slopes = SplineAreaDesign.basis_matrices(x_norm)
        n_params = [state["log_n_ratio_1"], state["log_n_ratio_2"], state["log_n_ratio_3"]]
        T_params = [state["log_Te_ratio_1"], state["log_Te_ratio_2"], state["log_Te_ratio_3"]]
        n_nodes = []
        T_nodes = []
        dn_dx = []
        dTe_dx = []
        for idx in range(x_norm.size):
            log_n_ratio = basis[idx, 0] * n_params[0] + basis[idx, 1] * n_params[1] + basis[idx, 2] * n_params[2]
            log_T_ratio = basis[idx, 0] * T_params[0] + basis[idx, 1] * T_params[1] + basis[idx, 2] * T_params[2]
            dlogn_dx = (slopes[idx, 0] * n_params[0] + slopes[idx, 1] * n_params[1] + slopes[idx, 2] * n_params[2]) / float(self._config.length)
            dlogT_dx = (slopes[idx, 0] * T_params[0] + slopes[idx, 1] * T_params[1] + slopes[idx, 2] * T_params[2]) / float(self._config.length)
            n_value = inlet["n_p"] * ops.exp(log_n_ratio)
            T_value = inlet["T_e"] * ops.exp(log_T_ratio)
            n_nodes.append(n_value)
            T_nodes.append(T_value)
            dn_dx.append(n_value * dlogn_dx)
            dTe_dx.append(T_value * dlogT_dx)
        return n_nodes, T_nodes, dn_dx, dTe_dx

    def _sigma_from_areas_parametric(self, ops, area_values: list[Any]) -> list[Any]:
        dx = float(self._config.length) / float(self._n_intervals)
        sigma_values = []
        for idx in range(int(self._n_intervals) + 1):
            denom = float(dx) * _positive_denom(ops, area_values[idx], _EPS)
            if idx == 0:
                sigma_values.append((area_values[1] - area_values[0]) / denom)
            elif idx == int(self._n_intervals):
                sigma_values.append((area_values[idx] - area_values[idx - 1]) / denom)
            else:
                sigma_values.append((area_values[idx + 1] - area_values[idx - 1]) / (2.0 * denom))
        return sigma_values

    def _rollout_from_parametric_values(self, values, *, ops) -> MachTrapezoidRollout:
        decision, state = self._split_parametric_values(values)
        seed_fraction = ops.exp(decision["log_seed_fraction"])
        inlet = _inlet_design_generic(
            ops=ops,
            n_p_in=ops.exp(decision["log_n_p_in"]),
            T_e_in=decision["T_e_in"],
            Z_in=decision["Z_in"],
            I_0=decision["I_0"],
            seed_fraction=seed_fraction,
            B=float(self._config.B),
            inlet_A=float(self._config.area_scale_m2),
            working_fluid=self._config.working_fluid,
        )
        mach_nodes = _mach_design_nodes_generic(
            ops=ops,
            mach_design=MachSplineDesign(m1=decision["m1"], m2=decision["m2"], m3=decision["m3"]),
            length=float(self._config.length),
            n_intervals=int(self._n_intervals),
            mach_in=inlet["mach"],
        )
        n_nodes, T_nodes, dn_dx, dTe_dx = self._evaluate_state_splines(ops=ops, inlet=inlet, state=state)
        area_values = []
        for idx in range(int(self._n_intervals) + 1):
            area_values.append(
                _mach_area_closure_generic(
                    ops=ops,
                    n_p=n_nodes[idx],
                    T_e=T_nodes[idx],
                    mach=mach_nodes["mach"][idx],
                    dot_N=inlet["dot_N"],
                    I_0=decision["I_0"],
                    seed_fraction=seed_fraction,
                    B=float(self._config.B),
                    working_fluid=self._config.working_fluid,
                )["A_safe"]
            )
        sigma_values = self._sigma_from_areas_parametric(ops, area_values)
        closures = []
        terms_by_node = []
        for idx in range(int(self._n_intervals) + 1):
            closure, terms = _dynamic_system_terms(
                ops=ops,
                n_p=n_nodes[idx],
                T_e=T_nodes[idx],
                A=area_values[idx],
                sigma=sigma_values[idx],
                dot_N=inlet["dot_N"],
                I_0=decision["I_0"],
                seed_fraction=seed_fraction,
                B=float(self._config.B),
                working_fluid=self._config.working_fluid,
            )
            closures.append(closure)
            terms_by_node.append(terms)
        dx = float(self._config.length) / float(self._n_intervals)
        momentum = []
        energy = []
        for idx, terms in enumerate(terms_by_node):
            scale_idx = min(idx, len(self._residual_scales.momentum) - 1)
            momentum.append(
                (terms["M11"] * dn_dx[idx] + terms["M12"] * dTe_dx[idx] - terms["rhs_m"])
                / float(self._residual_scales.momentum[scale_idx])
            )
            energy.append(
                (terms["E11"] * dn_dx[idx] + terms["E12"] * dTe_dx[idx] - terms["rhs_e"])
                / float(self._residual_scales.energy[scale_idx])
            )
        residuals = [*momentum, *energy]
        abs_residuals = [ops.fabs(item) for item in residuals]
        det_values = [item["det"] for item in terms_by_node]
        return MachTrapezoidRollout(
            decision_vector={**dict(decision), **dict(state)},
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
            dn_dx=dn_dx,
            dTe_dx=dTe_dx,
            scaled_step_n_residuals=[],
            scaled_step_Te_residuals=[],
            scaled_momentum_residuals=momentum,
            scaled_energy_residuals=energy,
            closures=closures,
            terms_by_node=terms_by_node,
            max_abs_scaled_residual=_reduce_max(ops, abs_residuals),
            min_abs_det=_reduce_min(ops, [ops.fabs(item) for item in det_values]),
            min_signed_det=_reduce_min(ops, det_values),
        )

    def _build_parametric_residual_scales(self) -> ImplicitResidualScales:
        rollout = self._rollout_from_parametric_values(self._initial_point, ops=_ops_for_numeric())
        dx = float(self._config.length) / float(self._n_intervals)
        step_n = []
        step_Te = []
        for idx in range(int(self._n_intervals)):
            step_n.append(
                max(
                    abs(float(rollout.n_p_nodes[idx + 1] - rollout.n_p_nodes[idx])),
                    abs(0.5 * dx * float(rollout.dn_dx[idx] + rollout.dn_dx[idx + 1])),
                    abs(float(rollout.n_p_nodes[idx + 1])),
                    1.0,
                )
            )
            step_Te.append(
                max(
                    abs(float(rollout.T_e_nodes[idx + 1] - rollout.T_e_nodes[idx])),
                    abs(0.5 * dx * float(rollout.dTe_dx[idx] + rollout.dTe_dx[idx + 1])),
                    abs(float(rollout.T_e_nodes[idx + 1])),
                    1.0,
                )
            )
        momentum = []
        energy = []
        for idx, terms in enumerate(rollout.terms_by_node):
            momentum.append(
                max(
                    abs(float(terms["rhs_m"])),
                    abs(float(terms["M11"]) * float(rollout.dn_dx[idx])),
                    abs(float(terms["M12"]) * float(rollout.dTe_dx[idx])),
                    1.0,
                )
            )
            energy.append(
                max(
                    abs(float(terms["rhs_e"])),
                    abs(float(terms["E11"]) * float(rollout.dn_dx[idx])),
                    abs(float(terms["E12"]) * float(rollout.dTe_dx[idx])),
                    1.0,
                )
            )
        return ImplicitResidualScales(
            step_n=np.asarray(step_n, dtype=float),
            step_Te=np.asarray(step_Te, dtype=float),
            momentum=np.asarray(momentum, dtype=float),
            energy=np.asarray(energy, dtype=float),
        )

    def decode_solution_point(self, values) -> MachTrapezoidRollout:
        rollout = self._rollout_from_parametric_values(values, ops=_ops_for_numeric())
        return MachTrapezoidRollout(
            decision_vector={key: float(value) for key, value in rollout.decision_vector.items()},
            inlet={key: float(value) for key, value in rollout.inlet.items()},
            mach_nodes={
                key: np.asarray(value, dtype=float) if key in {"x_norm", "x"} else [float(item) for item in value]
                for key, value in rollout.mach_nodes.items()
            },
            area_nodes={
                key: np.asarray(value, dtype=float) if key in {"x_norm", "x"} else [float(item) for item in value]
                for key, value in rollout.area_nodes.items()
            },
            n_p_nodes=[float(item) for item in rollout.n_p_nodes],
            T_e_nodes=[float(item) for item in rollout.T_e_nodes],
            dn_dx=[float(item) for item in rollout.dn_dx],
            dTe_dx=[float(item) for item in rollout.dTe_dx],
            scaled_step_n_residuals=[float(item) for item in rollout.scaled_step_n_residuals],
            scaled_step_Te_residuals=[float(item) for item in rollout.scaled_step_Te_residuals],
            scaled_momentum_residuals=[float(item) for item in rollout.scaled_momentum_residuals],
            scaled_energy_residuals=[float(item) for item in rollout.scaled_energy_residuals],
            closures=[{key: float(value) for key, value in item.items()} for item in rollout.closures],
            terms_by_node=[{key: float(value) for key, value in item.items()} for item in rollout.terms_by_node],
            max_abs_scaled_residual=float(rollout.max_abs_scaled_residual),
            min_abs_det=float(rollout.min_abs_det),
            min_signed_det=float(rollout.min_signed_det),
        )

    def summary_metadata(self) -> dict[str, Any]:
        payload = {
            "formulation": self.formulation,
            "transcription": "trapezoid_parametric_state_splines",
            "reference_profile_path": str(self._reference_profile_path),
            "mach_window_radius": float(self._mach_window_radius),
            "state_window_radius": float(self._state_window_radius),
            "mach_design_nominal": self._mach_design_nominal.to_dict(),
            "fixed_inlet": bool(self._fixed_inlet),
            "exposed_decision_names": [str(name) for _, _, name in self._decision_variable_specs],
            "state_param_count": int(len(self._STATE_PARAM_NAMES)),
            "total_variables": int(self.total_variables),
            "residual_constraint_count": int(2 * (self._n_intervals + 1)),
            "residual_soft_limit": float(self._residual_tolerance),
            "residual_penalty_weight": float(self._residual_penalty_weight),
            "det_branch_sign": "positive",
            "det_floor": float(self._det_floor),
            "det_constraint_scope": "profile_nodes",
        }
        if self._fixed_inlet:
            payload["fixed_inlet_decision"] = dict(self._fixed_inlet_decision)
        return payload

    def evaluate_solution(self, solution: MachTrapezoidRollout) -> CoarseProfileResult:
        result = self._coarse_result_from_nodes(
            decision={key: float(value) for key, value in solution.decision_vector.items()},
            n_p_nodes=np.asarray(solution.n_p_nodes, dtype=float),
            T_e_nodes=np.asarray(solution.T_e_nodes, dtype=float),
            n_intervals=self._n_intervals,
            rollout=None,
            check_equalities=False,
        )
        residuals = np.asarray(
            [
                *solution.scaled_momentum_residuals,
                *solution.scaled_energy_residuals,
            ],
            dtype=float,
        )
        max_residual = float(np.max(np.abs(residuals))) if residuals.size else 0.0
        residual_excess = np.maximum(np.abs(residuals) - float(self._residual_tolerance), 0.0)
        residual_penalty = float(self._residual_penalty_weight) * float(
            np.mean((residual_excess / max(float(self._residual_tolerance), 1e-12)) ** 2)
        ) if residual_excess.size else 0.0
        soft_score = float(result.objective_score) - residual_penalty
        diagnostics = dict(result.diagnostics)
        diagnostics.update(
            {
                "formulation": self.formulation,
                "coarse_model": "mach_spline_trapezoid",
                "transcription": "trapezoid_parametric_state_splines",
                "max_eq_residual": 0.0,
                "equality_count": 0,
                "max_scaled_residual": max_residual,
                "residual_constraint_count": int(residuals.size),
                "residual_soft_limit": float(self._residual_tolerance),
                "residual_soft_acceptable": bool(max_residual <= float(self._residual_tolerance)),
                "residual_penalty_weight": float(self._residual_penalty_weight),
                "residual_soft_penalty": float(residual_penalty),
                "fixed_inlet": bool(self._fixed_inlet),
                "mach_spline_max_scaled_residual": max_residual,
                "mach_spline_min_abs_det": float(solution.min_abs_det),
                "mach_spline_min_signed_det": float(solution.min_signed_det),
                "det_branch_sign": "positive",
                "det_floor": float(self._det_floor),
                "det_branch_acceptable": bool(float(solution.min_signed_det) >= float(self._det_floor)),
                "acceptable": bool(
                    result.diagnostics.get("finite_profile", False)
                    and float(result.diagnostics.get("max_ineq_residual", 0.0)) <= 1e-7
                    and float(solution.min_signed_det) >= float(self._det_floor)
                ),
            }
        )
        value_terms = dict(result.value_terms)
        value_terms.update(
            {
                "residual_soft_penalty": float(residual_penalty),
                "soft_objective_score": float(soft_score),
                "max_scaled_residual": max_residual,
            }
        )
        value_profile = dict(result.value_profile)
        value_profile["terms"] = dict(value_terms)
        return replace(
            result,
            objective_score=float(soft_score),
            objective_to_minimize=float(-soft_score),
            diagnostics=diagnostics,
            value_terms=value_terms,
            value_profile=value_profile,
        )

    def resample_solution_result(self, result: CoarseProfileResult, *, n_intervals: int) -> CoarseProfileResult:
        x_new = np.linspace(0.0, float(self._config.length), int(n_intervals) + 1, dtype=float)
        n_p_nodes = np.interp(x_new, np.asarray(result.x, dtype=float), np.asarray(result.n_p, dtype=float))
        T_e_nodes = np.interp(x_new, np.asarray(result.x, dtype=float), np.asarray(result.T_e, dtype=float))
        coarse = self._coarse_result_from_nodes(
            decision={key: float(value) for key, value in result.decision_vector.items()},
            n_p_nodes=n_p_nodes,
            T_e_nodes=T_e_nodes,
            n_intervals=int(n_intervals),
            rollout=None,
            check_equalities=False,
        )
        diagnostics = dict(coarse.diagnostics)
        diagnostics.update(
            {
                "formulation": self.formulation,
                "coarse_model": "mach_spline_trapezoid",
                "transcription": "trapezoid_resampled_profile_posthoc",
                "fixed_inlet": bool(self._fixed_inlet),
            }
        )
        return replace(coarse, diagnostics=diagnostics)

    def evaluate(self, vars):
        result = self._maingopy.EvaluationContainer()
        rollout = self._rollout_from_parametric_values(vars, ops=self._ops)
        decision = rollout.decision_vector
        midpoint_closures = _mach_midpoint_closures(
            ops=self._ops,
            config=self._config,
            rollout=rollout,
            decision=decision,
            n_intervals=self._n_intervals,
        )
        power_density_nodes = [
            -rollout.area_nodes["A"][idx] * closure["J_x"] * closure["E_x"] / 1e8
            for idx, closure in enumerate(rollout.closures)
        ]
        raw_design_score = _design_score_generic(
            ops=self._ops,
            outlet_T_e=rollout.T_e_nodes[-1],
            outlet_T_p=rollout.closures[-1]["T_p"],
            outlet_n_p=rollout.n_p_nodes[-1],
            outlet_n_e=rollout.closures[-1]["n_e"],
            inlet_T_e=rollout.inlet["T_e"],
            inlet_T_p=rollout.inlet["T_p"],
            inlet_mach=rollout.inlet["mach"],
            power_density_nodes=power_density_nodes,
            x_nodes=np.asarray(rollout.area_nodes["x"], dtype=float),
            seed_fraction=self._ops.exp(decision["log_seed_fraction"]),
            B=float(self._baseline.B),
            length=float(self._baseline.L),
            objective_profile=self._objective_profile,
            inlet_n_p=rollout.inlet["n_p"],
            inlet_n_e=rollout.inlet["n_e"],
            inlet_v=rollout.inlet["v_in"],
            inlet_A=float(self._baseline.area_scale_m2),
            working_fluid=self._baseline.working_fluid,
        )
        min_g_nodes = _reduce_min(self._ops, [item["G"] for item in rollout.closures])
        min_g_midpoints = _reduce_min(self._ops, [item["G"] for item in midpoint_closures])
        min_g_all = _min_op(self._ops, min_g_nodes, min_g_midpoints)
        min_tp_nodes = _reduce_min(self._ops, [item["T_p"] for item in rollout.closures])
        min_tp_midpoints = _reduce_min(self._ops, [item["T_p"] for item in midpoint_closures])
        min_tp_all = _min_op(self._ops, min_tp_nodes, min_tp_midpoints)
        velikhov_penalty = _velikhov_margin_penalty(self._ops, min_g_all)
        residual_soft_penalty = float(self._residual_penalty_weight) * _soft_average_square_shortfall(
            self._ops,
            [
                self._ops.fabs(item) - float(self._residual_tolerance)
                for item in [*rollout.scaled_momentum_residuals, *rollout.scaled_energy_residuals]
            ],
            scale=max(float(self._residual_tolerance), 1e-12),
        )
        design_score = raw_design_score - velikhov_penalty - residual_soft_penalty

        schedule0 = self._baseline.schedule[0]
        sigma_max = float(schedule0["max_abs_dlogA_dx"])
        tp_min = float(schedule0.get("tp_min", _TP_MIN))
        mach_min = float(schedule0.get("mach_min", 0.0) or 0.0)
        ineq = []
        for sigma in rollout.area_nodes["sigma_logA"]:
            ineq.append(self._ops.fabs(sigma) - sigma_max)
        for closure in rollout.closures:
            ineq.append(tp_min - closure["T_p"])
            ineq.append(float(_G_HARD_MARGIN) - closure["G"])
            if mach_min > 0.0:
                ineq.append(mach_min - closure["mach"])
        for closure_mid in midpoint_closures:
            ineq.append(float(_G_HARD_MARGIN) - closure_mid["G"])
        ineq.extend(float(self._det_floor) - terms["det"] for terms in rollout.terms_by_node)

        result.objective = -design_score
        result.ineq = _model_function(self._maingopy, ineq)
        result.eq = _model_function(self._maingopy, [])
        result.output = [
            self._maingopy.OutputVariable("design_score", design_score),
            self._maingopy.OutputVariable("raw_design_score", raw_design_score),
            self._maingopy.OutputVariable("velikhov_penalty", velikhov_penalty),
            self._maingopy.OutputVariable("residual_soft_penalty", residual_soft_penalty),
            self._maingopy.OutputVariable("mach_spline_max_scaled_residual", rollout.max_abs_scaled_residual),
            self._maingopy.OutputVariable("mach_spline_min_abs_det", rollout.min_abs_det),
            self._maingopy.OutputVariable("mach_spline_min_signed_det", rollout.min_signed_det),
            self._maingopy.OutputVariable("mach_spline_det_floor", float(self._det_floor)),
            self._maingopy.OutputVariable("inlet_G", rollout.inlet["G"]),
            self._maingopy.OutputVariable("inlet_mach", rollout.inlet["mach"]),
            self._maingopy.OutputVariable("outlet_mach", rollout.closures[-1]["mach"]),
            self._maingopy.OutputVariable("derived_area_outlet", rollout.area_nodes["A"][-1]),
            self._maingopy.OutputVariable(
                "mach_spline_max_abs_sigma",
                _reduce_max(self._ops, [self._ops.fabs(sigma) for sigma in rollout.area_nodes["sigma_logA"]]),
            ),
            self._maingopy.OutputVariable("min_path_Tp_nodes", min_tp_nodes),
            self._maingopy.OutputVariable("min_path_Tp_midpoints", min_tp_midpoints),
            self._maingopy.OutputVariable("min_path_Tp_all", min_tp_all),
            self._maingopy.OutputVariable("min_path_G_nodes", min_g_nodes),
            self._maingopy.OutputVariable("min_path_G_midpoints", min_g_midpoints),
            self._maingopy.OutputVariable("min_path_G_all", min_g_all),
        ]
        return result


class MachSplineRK4SoftModelBase(MachSplineReducedImplicitModelBase):
    formulation = "mach_spline_rk4_soft"

    def __init__(
        self,
        *,
        baseline: BaselineSeed,
        reference_profile_path: str | Path | None,
        n_intervals: int,
        maingopy_module,
        objective_profile: str = OBJECTIVE_PROFILE_LAB_POC_V2,
        newton_steps: int = 10,
        residual_tolerance: float = 1e-5,
        mach_window_radius: float = 1.0,
        sigma_penalty_weight: float = 5.0,
        tp_penalty_weight: float = 25.0,
        det_branch_sign: str | None = "positive",
        det_floor: float = 1e-3,
    ):
        super().__init__(
            baseline=baseline,
            reference_profile_path=reference_profile_path,
            n_intervals=n_intervals,
            maingopy_module=maingopy_module,
            objective_profile=objective_profile,
            newton_steps=newton_steps,
            residual_tolerance=residual_tolerance,
            mach_window_radius=mach_window_radius,
        )
        self._sigma_penalty_weight = float(sigma_penalty_weight)
        self._tp_penalty_weight = float(tp_penalty_weight)
        self._det_branch_sign = _normalize_det_branch_sign(det_branch_sign)
        self._det_floor = float(det_floor)
        if self._det_floor <= 0.0:
            raise ValueError(f"det_floor must be positive, got {self._det_floor!r}.")

    def summary_metadata(self) -> dict[str, Any]:
        return {
            "formulation": self.formulation,
            "reference_profile_path": str(self._reference_profile_path),
            "mach_window_radius": float(self._mach_window_radius),
            "mach_design_nominal": self._mach_design_nominal.to_dict(),
            "rhs_sigma_mode": "mach_chain_rule_finite_difference",
            "sigma_penalty_weight": float(self._sigma_penalty_weight),
            "tp_penalty_weight": float(self._tp_penalty_weight),
            "freidberg_defects": "posthoc_numeric_diagnostic",
            "det_branch_sign": _det_branch_name(self._det_branch_sign),
            "det_floor": float(self._det_floor),
            "det_constraint_scope": "rk4_stages_and_profile_nodes",
        }

    def _numeric_rollout(self, decision: dict[str, Any]) -> MachRK4SoftRollout:
        numeric_decision = {key: float(value) for key, value in dict(decision).items() if key in MACH_DECISION_NAMES}
        return rollout_mach_spline_rk4_soft_generic(
            ops=_ops_for_numeric(),
            config=self._config,
            n_intervals=self._n_intervals,
            decision_vector=numeric_decision,
            det_branch_sign=int(self._det_branch_sign),
            det_floor=float(self._det_floor),
        )

    def decode_solution_point(self, values) -> MachRK4SoftRollout:
        decision = {key: float(value) for key, value in self._decision_from_values(values).items()}
        return self._numeric_rollout(decision)

    def _numeric_soft_penalties(self, result: CoarseProfileResult) -> dict[str, float]:
        schedule0 = self._baseline.schedule[0]
        sigma_max = float(schedule0["max_abs_dlogA_dx"])
        tp_min = float(schedule0.get("tp_min", _TP_MIN))
        sigma = np.asarray(result.sigma_logA, dtype=float)
        sigma_excess = np.maximum(np.abs(sigma) - sigma_max, 0.0)
        sigma_penalty = float(self._sigma_penalty_weight) * float(
            np.mean((sigma_excess / max(sigma_max, 1e-12)) ** 2)
        )
        min_tp_all = float(result.diagnostics.get("min_T_p_all_checks", np.min(np.asarray(result.T_p, dtype=float))))
        tp_shortfall = max(tp_min - min_tp_all, 0.0)
        tp_penalty = float(self._tp_penalty_weight) * (tp_shortfall / max(tp_min, 1.0)) ** 2
        return {
            "sigma_smoothness_penalty": float(sigma_penalty),
            "tp_shortfall_penalty": float(tp_penalty),
            "soft_path_penalty": float(sigma_penalty + tp_penalty),
            "max_abs_sigma_excess": float(np.max(sigma_excess)) if sigma_excess.size else 0.0,
        }

    def _freidberg_defect_summary(self, result: CoarseProfileResult) -> dict[str, Any]:
        try:
            from v6_maingo_freidberg_variables.models import FreidbergConfig, PrimitivePoint
            from v6_maingo_freidberg_variables.transcription import interval_defects_from_points

            config = FreidbergConfig(
                B_T=float(self._config.B),
                inlet_area_m2=float(self._config.area_scale_m2),
                dot_N=float(result.inlet_design.dot_N),
                I_0=float(result.inlet_design.I_0),
                seed_fraction=float(result.inlet_design.seed_fraction),
                working_fluid=self._config.working_fluid,
            )
            points = [
                PrimitivePoint(
                    x=float(result.x[idx]),
                    n_p=float(result.n_p[idx]),
                    T_e=float(result.T_e[idx]),
                    T_p=float(result.T_p[idx]),
                    A=float(result.A[idx]),
                    v_p=float(result.v_p[idx]),
                    n_e=float(result.n_e[idx]),
                    beta=float(result.beta[idx]),
                    eta=float(result.eta[idx]),
                    Z=float(result.Z[idx]),
                    J_x=float(result.J_x[idx]),
                    J_y=float(result.J_y[idx]),
                    E_x=float(result.E_x[idx]),
                    mach=float(result.mach[idx]),
                    velikhov_margin=float(result.velikhov_margin[idx]),
                    seed_fraction=float(result.inlet_design.seed_fraction),
                )
                for idx in range(np.asarray(result.x).size)
            ]
            return interval_defects_from_points(points, config).summary()
        except Exception as exc:
            return {"error": str(exc)}

    def _with_soft_diagnostics(self, result: CoarseProfileResult) -> CoarseProfileResult:
        penalties = self._numeric_soft_penalties(result)
        freidberg = self._freidberg_defect_summary(result)
        raw_score = float(result.diagnostics.get("raw_design_score", result.objective_score))
        velikhov_penalty = float(result.value_terms.get("velikhov_margin_penalty", 0.0))
        soft_score = raw_score - velikhov_penalty - float(penalties["soft_path_penalty"])
        schedule0 = self._baseline.schedule[0]
        tp_min = float(schedule0.get("tp_min", _TP_MIN))
        mach_min = float(schedule0.get("mach_min", 0.0) or 0.0)
        state_path_acceptable = bool(
            result.diagnostics.get("finite_profile", False)
            and float(result.diagnostics.get("min_T_p_all_checks", np.min(result.T_p))) >= tp_min - 1e-7
            and float(result.diagnostics.get("min_velikhov_margin_all_checks", np.min(result.velikhov_margin)))
            >= float(_G_HARD_MARGIN) - 1e-7
            and float(result.diagnostics.get("min_mach", np.min(result.mach))) >= mach_min - 1e-7
        )
        freidberg_acceptable = False
        if "error" not in freidberg:
            freidberg_acceptable = bool(
                abs(float(freidberg.get("terminal_H_defect_MW", float("inf")))) <= 0.2
                and float(freidberg.get("max_abs_H_defect_MW", float("inf"))) <= 0.1
                and abs(float(freidberg.get("terminal_L_defect", float("inf")))) <= 0.01
                and float(freidberg.get("max_abs_L_defect", float("inf"))) <= 0.01
            )
        diagnostics = dict(result.diagnostics)
        diagnostics.update(
            {
                "formulation": self.formulation,
                "coarse_model": "mach_spline_rk4_soft",
                "rhs_sigma_mode": "mach_chain_rule_finite_difference",
                "candidate_generator_acceptable": bool(result.diagnostics.get("finite_profile", False)),
                "acceptable": bool(result.diagnostics.get("finite_profile", False)),
                "physical_state_path_acceptable": state_path_acceptable,
                "freidberg_defect_acceptable": freidberg_acceptable,
                "physical_acceptable": bool(state_path_acceptable and freidberg_acceptable),
                "freidberg_interval_defects": freidberg,
                **penalties,
                "raw_design_score": raw_score,
                "velikhov_margin_penalty": velikhov_penalty,
                "soft_objective_score": float(soft_score),
            }
        )
        value_terms = dict(result.value_terms)
        value_terms.update(
            {
                **penalties,
                "raw_design_score": raw_score,
                "velikhov_margin_penalty": velikhov_penalty,
                "soft_objective_score": float(soft_score),
            }
        )
        value_profile = dict(result.value_profile)
        value_profile["terms"] = dict(value_terms)
        return replace(
            result,
            objective_score=float(soft_score),
            objective_to_minimize=float(-soft_score),
            diagnostics=diagnostics,
            value_terms=value_terms,
            value_profile=value_profile,
        )

    def evaluate_solution(self, solution: MachRK4SoftRollout) -> CoarseProfileResult:
        result = self._coarse_result_from_nodes(
            decision={key: float(value) for key, value in solution.decision_vector.items()},
            n_p_nodes=np.asarray(solution.n_p_nodes, dtype=float),
            T_e_nodes=np.asarray(solution.T_e_nodes, dtype=float),
            n_intervals=self._n_intervals,
            rollout=None,
            check_equalities=False,
        )
        result = self._with_soft_diagnostics(result)
        diagnostics = dict(result.diagnostics)
        diagnostics.update(
            {
                "det_branch_sign": _det_branch_name(self._det_branch_sign),
                "det_floor": float(self._det_floor),
                "det_constraint_scope": "rk4_stages_and_profile_nodes",
                "mach_spline_min_abs_det": float(solution.min_abs_det),
                "mach_spline_min_signed_det": float(solution.min_signed_det),
                "det_branch_acceptable": bool(
                    int(self._det_branch_sign) == 0 or float(solution.min_signed_det) >= float(self._det_floor)
                ),
            }
        )
        return replace(result, diagnostics=diagnostics)

    def resample_solution_result(self, result: CoarseProfileResult, *, n_intervals: int) -> CoarseProfileResult:
        x_new = np.linspace(0.0, float(self._config.length), int(n_intervals) + 1, dtype=float)
        n_p_nodes = np.interp(x_new, np.asarray(result.x, dtype=float), np.asarray(result.n_p, dtype=float))
        T_e_nodes = np.interp(x_new, np.asarray(result.x, dtype=float), np.asarray(result.T_e, dtype=float))
        coarse = self._coarse_result_from_nodes(
            decision={key: float(value) for key, value in result.decision_vector.items()},
            n_p_nodes=n_p_nodes,
            T_e_nodes=T_e_nodes,
            n_intervals=int(n_intervals),
            rollout=None,
            check_equalities=False,
        )
        return self._with_soft_diagnostics(coarse)

    def evaluate(self, vars):
        result = self._maingopy.EvaluationContainer()
        decision = self._decision_from_values(vars)
        rollout = rollout_mach_spline_rk4_soft_generic(
            ops=self._ops,
            config=self._config,
            n_intervals=self._n_intervals,
            decision_vector=decision,
            det_branch_sign=int(self._det_branch_sign),
            det_floor=float(self._det_floor),
        )
        midpoint_closures = _mach_midpoint_closures(
            ops=self._ops,
            config=self._config,
            rollout=rollout,
            decision=decision,
            n_intervals=self._n_intervals,
        )
        power_density_nodes = [
            -rollout.area_nodes["A"][idx] * closure["J_x"] * closure["E_x"] / 1e8
            for idx, closure in enumerate(rollout.closures)
        ]
        raw_design_score = _design_score_generic(
            ops=self._ops,
            outlet_T_e=rollout.T_e_nodes[-1],
            outlet_T_p=rollout.closures[-1]["T_p"],
            outlet_n_p=rollout.n_p_nodes[-1],
            outlet_n_e=rollout.closures[-1]["n_e"],
            inlet_T_e=rollout.inlet["T_e"],
            inlet_T_p=rollout.inlet["T_p"],
            inlet_mach=rollout.inlet["mach"],
            power_density_nodes=power_density_nodes,
            x_nodes=np.asarray(rollout.area_nodes["x"], dtype=float),
            seed_fraction=self._ops.exp(decision["log_seed_fraction"]),
            B=float(self._baseline.B),
            length=float(self._baseline.L),
            objective_profile=self._objective_profile,
            inlet_n_p=rollout.inlet["n_p"],
            inlet_n_e=rollout.inlet["n_e"],
            inlet_v=rollout.inlet["v_in"],
            inlet_A=float(self._baseline.area_scale_m2),
            working_fluid=self._baseline.working_fluid,
        )
        min_g_nodes = _reduce_min(self._ops, [item["G"] for item in rollout.closures])
        min_g_midpoints = _reduce_min(self._ops, [item["G"] for item in midpoint_closures])
        min_g_all = _min_op(self._ops, min_g_nodes, min_g_midpoints)
        min_tp_nodes = _reduce_min(self._ops, [item["T_p"] for item in rollout.closures])
        min_tp_midpoints = _reduce_min(self._ops, [item["T_p"] for item in midpoint_closures])
        min_tp_all = _min_op(self._ops, min_tp_nodes, min_tp_midpoints)
        schedule0 = self._baseline.schedule[0]
        sigma_max = float(schedule0["max_abs_dlogA_dx"])
        tp_min = float(schedule0.get("tp_min", _TP_MIN))
        velikhov_penalty = _velikhov_margin_penalty(self._ops, min_g_all)
        sigma_penalty = float(self._sigma_penalty_weight) * _soft_average_square_shortfall(
            self._ops,
            [self._ops.fabs(sigma) - sigma_max for sigma in rollout.area_nodes["sigma_logA"]],
            scale=max(sigma_max, 1e-12),
        )
        tp_penalty = float(self._tp_penalty_weight) * _soft_average_square_shortfall(
            self._ops,
            [tp_min - min_tp_all],
            scale=max(tp_min, 1.0),
        )
        design_score = raw_design_score - velikhov_penalty - sigma_penalty - tp_penalty

        ineq = []
        if int(self._det_branch_sign) != 0:
            ineq.extend(float(self._det_floor) - item for item in _signed_det_values(
                rollout.det_values,
                sign=int(self._det_branch_sign),
            ))

        result.objective = -design_score
        result.ineq = _model_function(self._maingopy, ineq)
        result.output = [
            self._maingopy.OutputVariable("design_score", design_score),
            self._maingopy.OutputVariable("raw_design_score", raw_design_score),
            self._maingopy.OutputVariable("velikhov_penalty", velikhov_penalty),
            self._maingopy.OutputVariable("sigma_smoothness_penalty", sigma_penalty),
            self._maingopy.OutputVariable("tp_shortfall_penalty", tp_penalty),
            self._maingopy.OutputVariable("mach_spline_min_abs_det", rollout.min_abs_det),
            self._maingopy.OutputVariable("mach_spline_min_signed_det", rollout.min_signed_det),
            self._maingopy.OutputVariable("mach_spline_det_floor", float(self._det_floor)),
            self._maingopy.OutputVariable("inlet_G", rollout.inlet["G"]),
            self._maingopy.OutputVariable("inlet_mach", rollout.inlet["mach"]),
            self._maingopy.OutputVariable("outlet_mach", rollout.closures[-1]["mach"]),
            self._maingopy.OutputVariable("derived_area_outlet", rollout.area_nodes["A"][-1]),
            self._maingopy.OutputVariable(
                "mach_spline_max_abs_sigma",
                _reduce_max(self._ops, [self._ops.fabs(sigma) for sigma in rollout.area_nodes["sigma_logA"]]),
            ),
            self._maingopy.OutputVariable("min_path_Tp_nodes", min_tp_nodes),
            self._maingopy.OutputVariable("min_path_Tp_midpoints", min_tp_midpoints),
            self._maingopy.OutputVariable("min_path_Tp_all", min_tp_all),
            self._maingopy.OutputVariable("min_path_G_nodes", min_g_nodes),
            self._maingopy.OutputVariable("min_path_G_midpoints", min_g_midpoints),
            self._maingopy.OutputVariable("min_path_G_all", min_g_all),
        ]
        return result


def make_mach_spline_reduced_maingo_model(*, maingopy_module, **kwargs):
    base = MachSplineReducedImplicitModelBase(maingopy_module=maingopy_module, **kwargs)

    class MachSplineReducedMAiNGOModel(maingopy_module.MAiNGOmodel):
        def __init__(self):
            maingopy_module.MAiNGOmodel.__init__(self)

        def get_variables(self):
            return base.get_variables()

        def get_initial_point(self):
            return base.get_initial_point()

        def evaluate(self, vars):
            return base.evaluate(vars)

        def summary_metadata(self):
            return base.summary_metadata()

        def set_initial_point_from_decision(self, decision: dict[str, Any]) -> dict[str, float]:
            return base.set_initial_point_from_decision(decision)

    return MachSplineReducedMAiNGOModel()
