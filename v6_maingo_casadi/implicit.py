from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any

import casadi as ca
import numpy as np

from v6_global_marginal.global_postprocess_v6 import compute_design_value_terms

from .casadi_evaluator import CasadiCoarseEvaluator
from .constants import _EPS, _G_HARD_MARGIN, _TP_MIN, OBJECTIVE_PROFILE_LAB_POC_V2
from .geometry import SplineAreaDesign, _evaluate_area_design_nodes
from .models import BaselineSeed, CoarseProfileResult
from .numerics import _ops_for_casadi, _ops_for_numeric, _velikhov_margin_penalty
from .physics import (
    _design_score_generic,
    _dynamic_system_terms,
    _evaluate_midpoint_closures,
    _implicit_step_residuals,
    evaluate_inlet_design_numeric,
)
from .profiles import _augment_value_terms_with_hall_diagnostics, _normalize_objective_profile, _value_profile_dict

_LOGGER = logging.getLogger(__name__)

@dataclass(frozen=True)
class ImplicitTrajectoryVariables:
    decision_vector: dict[str, float]
    n_p_nodes: np.ndarray
    T_e_nodes: np.ndarray
    dn_dx: np.ndarray
    dTe_dx: np.ndarray


@dataclass(frozen=True)
class ImplicitResidualScales:
    step_n: np.ndarray
    step_Te: np.ndarray
    momentum: np.ndarray
    energy: np.ndarray


@dataclass(frozen=True)
class ImplicitTrajectoryScaling:
    n_p_center: np.ndarray
    n_p_scale: np.ndarray
    T_e_center: np.ndarray
    T_e_scale: np.ndarray
    dn_dx_center: np.ndarray
    dn_dx_scale: np.ndarray
    dTe_dx_center: np.ndarray
    dTe_dx_scale: np.ndarray

    def decode_n_p_tail(self, values):
        return [float(c) + float(s) * value for c, s, value in zip(self.n_p_center, self.n_p_scale, values, strict=True)]

    def decode_T_e_tail(self, values):
        return [float(c) + float(s) * value for c, s, value in zip(self.T_e_center, self.T_e_scale, values, strict=True)]

    def decode_dn_dx(self, values):
        return [float(c) + float(s) * value for c, s, value in zip(self.dn_dx_center, self.dn_dx_scale, values, strict=True)]

    def decode_dTe_dx(self, values):
        return [float(c) + float(s) * value for c, s, value in zip(self.dTe_dx_center, self.dTe_dx_scale, values, strict=True)]

    def encode_n_p_tail(self, values: np.ndarray) -> list[float]:
        return [
            float((float(value) - float(c)) / float(s))
            for value, c, s in zip(np.asarray(values, dtype=float), self.n_p_center, self.n_p_scale, strict=True)
        ]

    def encode_T_e_tail(self, values: np.ndarray) -> list[float]:
        return [
            float((float(value) - float(c)) / float(s))
            for value, c, s in zip(np.asarray(values, dtype=float), self.T_e_center, self.T_e_scale, strict=True)
        ]

    def encode_dn_dx(self, values: np.ndarray) -> list[float]:
        return [
            float((float(value) - float(c)) / float(s))
            for value, c, s in zip(np.asarray(values, dtype=float), self.dn_dx_center, self.dn_dx_scale, strict=True)
        ]

    def encode_dTe_dx(self, values: np.ndarray) -> list[float]:
        return [
            float((float(value) - float(c)) / float(s))
            for value, c, s in zip(np.asarray(values, dtype=float), self.dTe_dx_center, self.dTe_dx_scale, strict=True)
        ]

    def to_dict(self) -> dict[str, float]:
        return {
            "n_p_scale_min": float(np.min(self.n_p_scale)),
            "n_p_scale_max": float(np.max(self.n_p_scale)),
            "T_e_scale_min": float(np.min(self.T_e_scale)),
            "T_e_scale_max": float(np.max(self.T_e_scale)),
            "dn_dx_scale_min": float(np.min(self.dn_dx_scale)),
            "dn_dx_scale_max": float(np.max(self.dn_dx_scale)),
            "dTe_dx_scale_min": float(np.min(self.dTe_dx_scale)),
            "dTe_dx_scale_max": float(np.max(self.dTe_dx_scale)),
        }


class _ImplicitVariableLayout:
    _DECISION_NAMES = (
        "log_n_p_in",
        "T_e_in",
        "Z_in",
        "I_0",
        "log_seed_fraction",
        "a1",
        "a2",
        "a3",
    )

    def __init__(self, n_intervals: int):
        self.n_intervals = int(n_intervals)

    @property
    def total_variables(self) -> int:
        return 8 + 4 * self.n_intervals

    def split_raw(self, values):
        values = list(values)
        if len(values) != self.total_variables:
            raise ValueError(
                f"implicit full-space solution size mismatch: got {len(values)}, expected {self.total_variables}."
            )
        idx = 0
        decision_raw = values[idx : idx + 8]
        idx += 8
        n_p_tail = values[idx : idx + self.n_intervals]
        idx += self.n_intervals
        T_e_tail = values[idx : idx + self.n_intervals]
        idx += self.n_intervals
        dn_dx = values[idx : idx + self.n_intervals]
        idx += self.n_intervals
        dTe_dx = values[idx : idx + self.n_intervals]
        return decision_raw, n_p_tail, T_e_tail, dn_dx, dTe_dx

    def decode_numeric(self, values, scaling: ImplicitTrajectoryScaling) -> ImplicitTrajectoryVariables:
        decision_raw, n_p_tail_raw, T_e_tail_raw, dn_dx_raw, dTe_dx_raw = self.split_raw(values)
        decision = {
            name: float(value)
            for name, value in zip(self._DECISION_NAMES, decision_raw, strict=True)
        }
        n_p_tail = scaling.decode_n_p_tail(n_p_tail_raw)
        T_e_tail = scaling.decode_T_e_tail(T_e_tail_raw)
        dn_dx = scaling.decode_dn_dx(dn_dx_raw)
        dTe_dx = scaling.decode_dTe_dx(dTe_dx_raw)
        n_p_nodes = np.concatenate(
            (
                np.array([math.exp(decision["log_n_p_in"])], dtype=float),
                np.asarray(n_p_tail, dtype=float).reshape(-1),
            )
        )
        T_e_nodes = np.concatenate(
            (
                np.array([decision["T_e_in"]], dtype=float),
                np.asarray(T_e_tail, dtype=float).reshape(-1),
            )
        )
        return ImplicitTrajectoryVariables(
            decision_vector=decision,
            n_p_nodes=n_p_nodes,
            T_e_nodes=T_e_nodes,
            dn_dx=np.asarray(dn_dx, dtype=float).reshape(-1),
            dTe_dx=np.asarray(dTe_dx, dtype=float).reshape(-1),
        )


def _scaled_interval(lower: float, upper: float, reference: float, *, min_scale: float = 1.0) -> tuple[float, float, float]:
    ref = float(min(max(reference, lower), upper))
    scale = max(abs(float(upper) - ref), abs(ref - float(lower)), float(min_scale))
    scaled_lower = (float(lower) - ref) / scale
    scaled_upper = (float(upper) - ref) / scale
    if scaled_upper <= scaled_lower:
        scaled_upper = scaled_lower + 1e-6
    return ref, scale, scaled_lower, scaled_upper


def _build_coarse_result_from_state_trajectory(
    *,
    baseline: BaselineSeed,
    n_intervals: int,
    variables: ImplicitTrajectoryVariables,
    check_equalities: bool,
    residual_scales: ImplicitResidualScales | None = None,
    objective_profile: str = OBJECTIVE_PROFILE_LAB_POC_V2,
) -> CoarseProfileResult:
    objective_profile = _normalize_objective_profile(objective_profile)
    fluid = baseline.working_fluid
    ops = _ops_for_numeric()
    area_design = SplineAreaDesign(
        a1=float(variables.decision_vector["a1"]),
        a2=float(variables.decision_vector["a2"]),
        a3=float(variables.decision_vector["a3"]),
    )
    inlet = evaluate_inlet_design_numeric(
        n_p_in=math.exp(float(variables.decision_vector["log_n_p_in"])),
        T_e_in=float(variables.decision_vector["T_e_in"]),
        Z_in=float(variables.decision_vector["Z_in"]),
        I_0=float(variables.decision_vector["I_0"]),
        seed_fraction=math.exp(float(variables.decision_vector["log_seed_fraction"])),
        B=float(baseline.B),
        inlet_area=float(baseline.area_scale_m2),
        working_fluid_profile=fluid,
    )
    area_nodes = _evaluate_area_design_nodes(
        ops=ops,
        area_design=area_design,
        length=float(baseline.L),
        n_intervals=int(n_intervals),
        area_scale=float(baseline.area_scale_m2),
    )
    x_nodes = np.asarray(area_nodes["x"], dtype=float)
    A_nodes = np.asarray(area_nodes["A"], dtype=float)
    sigma_nodes = np.asarray(area_nodes["sigma_logA"], dtype=float)
    area_mid, midpoint_closures = _evaluate_midpoint_closures(
        ops=ops,
        area_design=area_design,
        length=float(baseline.L),
        n_intervals=int(n_intervals),
        n_p_nodes=variables.n_p_nodes,
        T_e_nodes=variables.T_e_nodes,
        dot_N=float(inlet.dot_N),
        I_0=float(inlet.I_0),
        seed_fraction=float(inlet.seed_fraction),
        B=float(baseline.B),
        area_scale=float(baseline.area_scale_m2),
        working_fluid=fluid,
    )

    closures = []
    terms_by_node = []
    power_density_nodes = []
    det_nodes = []
    for idx in range(int(n_intervals) + 1):
        closure, terms = _dynamic_system_terms(
            ops=ops,
            n_p=float(variables.n_p_nodes[idx]),
            T_e=float(variables.T_e_nodes[idx]),
            A=float(A_nodes[idx]),
            sigma=float(sigma_nodes[idx]),
            dot_N=float(inlet.dot_N),
            I_0=float(inlet.I_0),
            seed_fraction=float(inlet.seed_fraction),
            B=float(baseline.B),
            working_fluid=fluid,
        )
        closures.append(closure)
        terms_by_node.append(terms)
        power_density_nodes.append(-A_nodes[idx] * closure["J_x"] * closure["E_x"] / 1e8)
        det_nodes.append(float(terms["det"]))

    ineq = []
    sigma_max = float(baseline.schedule[0]["max_abs_dlogA_dx"])
    tp_min = float(baseline.schedule[0].get("tp_min", _TP_MIN))
    mach_min = float(baseline.schedule[0].get("mach_min", 0.0) or 0.0)
    for sigma in sigma_nodes:
        ineq.append(float(abs(sigma) - sigma_max))
    for closure in closures:
        ineq.append(float(tp_min - closure["T_p"]))
        ineq.append(float(_G_HARD_MARGIN - closure["G"]))
        if mach_min > 0.0:
            ineq.append(float(mach_min - closure["mach"]))
    for closure_mid in midpoint_closures:
        ineq.append(float(_G_HARD_MARGIN - closure_mid["G"]))

    eq = []
    dx = float(baseline.L) / int(n_intervals)
    if check_equalities:
        for k in range(int(n_intervals)):
            step_n_scale = float(residual_scales.step_n[k]) if residual_scales is not None else 1.0
            step_Te_scale = float(residual_scales.step_Te[k]) if residual_scales is not None else 1.0
            momentum_scale = float(residual_scales.momentum[k]) if residual_scales is not None else 1.0
            energy_scale = float(residual_scales.energy[k]) if residual_scales is not None else 1.0
            eq.append(
                float(variables.n_p_nodes[k + 1] - variables.n_p_nodes[k] - dx * variables.dn_dx[k])
                / step_n_scale
            )
            eq.append(
                float(variables.T_e_nodes[k + 1] - variables.T_e_nodes[k] - dx * variables.dTe_dx[k])
                / step_Te_scale
            )
            terms = terms_by_node[k + 1]
            eq.append(
                float(terms["M11"] * variables.dn_dx[k] + terms["M12"] * variables.dTe_dx[k] - terms["rhs_m"])
                / momentum_scale
            )
            eq.append(
                float(terms["E11"] * variables.dn_dx[k] + terms["E12"] * variables.dTe_dx[k] - terms["rhs_e"])
                / energy_scale
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
            outlet_T_e=float(variables.T_e_nodes[-1]),
            outlet_T_p=float(closures[-1]["T_p"]),
            outlet_n_p=float(variables.n_p_nodes[-1]),
            outlet_n_e=float(closures[-1]["n_e"]),
            inlet_T_e=float(inlet.T_e),
            inlet_T_p=float(inlet.T_p),
            inlet_mach=float(inlet.mach),
            power_density_nodes=power_density_nodes,
            x_nodes=x_nodes,
            seed_fraction=float(inlet.seed_fraction),
            B=float(baseline.B),
            length=float(baseline.L),
            objective_profile=objective_profile,
            inlet_n_p=float(inlet.n_p),
            inlet_n_e=float(closures[0]["n_e"]),
            inlet_v=float(inlet.v_in),
            inlet_A=float(baseline.area_scale_m2),
            working_fluid=fluid,
        )
    )
    velikhov_penalty = float(_velikhov_margin_penalty(ops, min_g_all))
    design_score = float(raw_design_score - velikhov_penalty)
    arrays = {
        "x": x_nodes,
        "n_p": np.asarray(variables.n_p_nodes, dtype=float),
        "T_e": np.asarray(variables.T_e_nodes, dtype=float),
        "A": A_nodes,
        "sigma_logA": sigma_nodes,
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
        "x_midpoint": np.asarray(area_mid["x"], dtype=float),
    }
    ineq_arr = np.asarray(ineq, dtype=float)
    eq_arr = np.asarray(eq, dtype=float) if eq else np.zeros(0, dtype=float)
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
        B=float(baseline.B),
        seed_fraction=float(inlet.seed_fraction),
        v_p=arrays["v_p"],
        heavy_particle_mass_kg=float(fluid.heavy_particle_mass_kg),
    )
    value_terms_dict = value_terms.to_dict()
    _augment_value_terms_with_hall_diagnostics(
        value_terms_dict,
        x=arrays["x"],
        E_x=arrays["E_x"],
        I_0=float(inlet.I_0),
    )
    value_terms_dict["mass_flow_rate_kg_s"] = float(inlet.dot_N) * float(fluid.heavy_particle_mass_kg)
    value_terms_dict["inlet_area_m2"] = float(arrays["A"][0])
    value_terms_dict["outlet_area_m2"] = float(arrays["A"][-1])
    value_terms_dict["outlet_to_inlet_area_ratio"] = float(arrays["A"][-1]) / max(float(arrays["A"][0]), _EPS)
    value_terms_dict["velikhov_margin_penalty"] = float(velikhov_penalty)
    value_terms_dict["raw_design_score"] = float(raw_design_score)
    value_terms_dict["min_T_p_midpoint"] = float(min_tp_midpoints)
    value_terms_dict["min_T_p_all_checks"] = float(min_tp_all)
    value_terms_dict["min_velikhov_margin_midpoint"] = float(min_g_midpoints)
    value_profile = _value_profile_dict(
        value_terms,
        objective_profile=objective_profile,
    )
    value_profile["terms"] = dict(value_terms_dict)
    max_eq_residual = float(np.max(np.abs(eq_arr))) if eq_arr.size else 0.0
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
        "min_velikhov_margin": float(np.min(arrays["velikhov_margin"])),
        "min_velikhov_margin_midpoint": float(np.min(arrays["velikhov_margin_midpoint"])),
        "min_velikhov_margin_all_checks": float(min_g_all),
        "min_mach": float(np.min(arrays["mach"])),
        "max_abs_sigma_logA": float(np.max(np.abs(arrays["sigma_logA"]))),
        "max_ineq_residual": float(np.max(ineq_arr)) if ineq_arr.size else 0.0,
        "constraint_count": int(ineq_arr.size),
        "max_eq_residual": max_eq_residual,
        "equality_count": int(eq_arr.size),
        "det_min_abs": float(np.min(np.abs(np.asarray(det_nodes, dtype=float)))),
        "raw_design_score": float(raw_design_score),
        "velikhov_margin_penalty": float(velikhov_penalty),
        "formulation": "implicit_fullspace_backward_euler_scaled_variables",
        "objective_profile": objective_profile,
        "acceptable": bool(
            (float(np.max(ineq_arr)) <= 1e-7 if ineq_arr.size else True)
            and (max_eq_residual <= 1e-6 if check_equalities else True)
        ),
    }
    return CoarseProfileResult(
        decision_vector=dict(variables.decision_vector),
        inlet_design=inlet,
        area_design=area_design,
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


def _resample_profile_result(
    *,
    baseline: BaselineSeed,
    result: CoarseProfileResult,
    n_intervals: int,
    objective_profile: str = OBJECTIVE_PROFILE_LAB_POC_V2,
) -> CoarseProfileResult:
    x_new = np.linspace(0.0, float(baseline.L), int(n_intervals) + 1, dtype=float)
    n_p_nodes = np.interp(x_new, np.asarray(result.x, dtype=float), np.asarray(result.n_p, dtype=float))
    T_e_nodes = np.interp(x_new, np.asarray(result.x, dtype=float), np.asarray(result.T_e, dtype=float))
    dx = float(baseline.L) / int(n_intervals)
    return _build_coarse_result_from_state_trajectory(
        baseline=baseline,
        n_intervals=int(n_intervals),
        variables=ImplicitTrajectoryVariables(
            decision_vector=dict(result.decision_vector),
            n_p_nodes=n_p_nodes,
            T_e_nodes=T_e_nodes,
            dn_dx=np.diff(n_p_nodes) / dx,
            dTe_dx=np.diff(T_e_nodes) / dx,
        ),
        check_equalities=False,
        objective_profile=objective_profile,
    )


def _json_float(value: Any) -> float | None:
    numeric = float(np.asarray(value, dtype=float))
    return numeric if math.isfinite(numeric) else None


def _json_float_list(values: Any) -> list[float | None]:
    return [_json_float(value) for value in np.asarray(values, dtype=float).reshape(-1)]


def _exception_diagnostic(exc: BaseException) -> dict[str, str]:
    return {
        "type": type(exc).__name__,
        "message": str(exc),
    }


def _implicit_reference_step_failure_diagnostic(
    *,
    interval_index: int,
    dx: float,
    length: float,
    z0: np.ndarray,
    params: np.ndarray,
    residual_function,
    exc: RuntimeError,
) -> dict[str, Any]:
    param_names = (
        "n_prev",
        "T_e_prev",
        "A_next",
        "sigma_next",
        "dot_N",
        "I_0",
        "seed_fraction",
        "step_n_scale",
        "step_Te_scale",
        "momentum_scale",
        "energy_scale",
    )
    diagnostic: dict[str, Any] = {
        "interval_index": int(interval_index),
        "x": float((int(interval_index) + 1) * float(dx)),
        "x_over_L": float((int(interval_index) + 1) * float(dx) / max(float(length), _EPS)),
        "exception": _exception_diagnostic(exc),
        "initial_guess": {
            "n_next": _json_float(z0[0]),
            "T_e_next": _json_float(z0[1]),
            "dn_dx": _json_float(z0[2]),
            "dTe_dx": _json_float(z0[3]),
        },
        "parameters": {
            name: _json_float(value)
            for name, value in zip(param_names, np.asarray(params, dtype=float).reshape(-1), strict=True)
        },
    }
    try:
        residual = np.asarray(residual_function(z0, params), dtype=float).reshape(-1)
        diagnostic["scaled_residual_at_initial_guess"] = {
            "step_n": _json_float(residual[0]),
            "step_Te": _json_float(residual[1]),
            "momentum": _json_float(residual[2]),
            "energy": _json_float(residual[3]),
            "max_abs": (
                float(np.max(np.abs(residual)))
                if residual.size and np.all(np.isfinite(residual))
                else None
            ),
        }
    except RuntimeError as residual_exc:
        diagnostic["scaled_residual_at_initial_guess_error"] = _exception_diagnostic(residual_exc)
    return diagnostic


def _implicit_reference_is_reasonable(*, baseline: BaselineSeed, variables: ImplicitTrajectoryVariables) -> bool:
    return bool(_implicit_reference_reasonableness_diagnostics(baseline=baseline, variables=variables)["reasonable"])


def _implicit_reference_reasonableness_diagnostics(
    *,
    baseline: BaselineSeed,
    variables: ImplicitTrajectoryVariables,
) -> dict[str, Any]:
    n_nodes = np.asarray(variables.n_p_nodes, dtype=float)
    te_nodes = np.asarray(variables.T_e_nodes, dtype=float)
    dn_dx = np.asarray(variables.dn_dx, dtype=float)
    dte_dx = np.asarray(variables.dTe_dx, dtype=float)
    finite = bool(
        np.all(np.isfinite(n_nodes))
        and np.all(np.isfinite(te_nodes))
        and np.all(np.isfinite(dn_dx))
        and np.all(np.isfinite(dte_dx))
    )
    positive_state = bool(np.all(n_nodes > 0.0) and np.all(te_nodes > 0.0))
    n_upper = max(float(baseline.inlet_windows["n_p_in"]["max"]) * 100.0, 1e28)
    te_upper = max(float(baseline.inlet_windows["T_e_in"]["max"]) * 20.0, 5e4)
    max_n = _json_float(np.max(n_nodes)) if n_nodes.size else None
    max_te = _json_float(np.max(te_nodes)) if te_nodes.size else None
    within_upper = bool(
        max_n is not None
        and max_te is not None
        and float(max_n) <= n_upper
        and float(max_te) <= te_upper
    )
    return {
        "reasonable": bool(finite and positive_state and within_upper),
        "finite": finite,
        "positive_state": positive_state,
        "within_upper_bounds": within_upper,
        "max_n_p": max_n,
        "max_T_e": max_te,
        "n_p_upper_reasonable": float(n_upper),
        "T_e_upper_reasonable": float(te_upper),
        "first_n_p_nodes": _json_float_list(n_nodes[: min(3, n_nodes.size)]),
        "first_T_e_nodes": _json_float_list(te_nodes[: min(3, te_nodes.size)]),
        "last_n_p_nodes": _json_float_list(n_nodes[max(0, n_nodes.size - 3) :]),
        "last_T_e_nodes": _json_float_list(te_nodes[max(0, te_nodes.size - 3) :]),
    }


def _constant_implicit_reference(
    *,
    baseline: BaselineSeed,
    n_intervals: int,
    decision_vector: dict[str, float],
) -> tuple[ImplicitTrajectoryVariables, ImplicitResidualScales]:
    n0 = math.exp(float(decision_vector["log_n_p_in"]))
    te0 = float(decision_vector["T_e_in"])
    n_nodes = np.full(int(n_intervals) + 1, n0, dtype=float)
    te_nodes = np.full(int(n_intervals) + 1, te0, dtype=float)
    dn_dx = np.zeros(int(n_intervals), dtype=float)
    dte_dx = np.zeros(int(n_intervals), dtype=float)
    variables = ImplicitTrajectoryVariables(
        decision_vector=dict(decision_vector),
        n_p_nodes=n_nodes,
        T_e_nodes=te_nodes,
        dn_dx=dn_dx,
        dTe_dx=dte_dx,
    )

    area_design = SplineAreaDesign(
        a1=float(decision_vector["a1"]),
        a2=float(decision_vector["a2"]),
        a3=float(decision_vector["a3"]),
    )
    area_nodes = _evaluate_area_design_nodes(
        ops=_ops_for_numeric(),
        area_design=area_design,
        length=float(baseline.L),
        n_intervals=int(n_intervals),
        area_scale=float(baseline.area_scale_m2),
    )
    inlet = evaluate_inlet_design_numeric(
        n_p_in=n0,
        T_e_in=te0,
        Z_in=float(decision_vector["Z_in"]),
        I_0=float(decision_vector["I_0"]),
        seed_fraction=math.exp(float(decision_vector["log_seed_fraction"])),
        B=float(baseline.B),
        inlet_area=float(baseline.area_scale_m2),
        working_fluid_profile=baseline.working_fluid,
    )
    step_n_scales = np.full(int(n_intervals), max(abs(n0), 1.0), dtype=float)
    step_te_scales = np.full(int(n_intervals), max(abs(te0), 1.0), dtype=float)
    momentum_scales = []
    energy_scales = []
    for k in range(int(n_intervals)):
        _, terms = _dynamic_system_terms(
            ops=_ops_for_numeric(),
            n_p=n0,
            T_e=te0,
            A=float(np.asarray(area_nodes["A"], dtype=float)[k + 1]),
            sigma=float(np.asarray(area_nodes["sigma_logA"], dtype=float)[k + 1]),
            dot_N=float(inlet.dot_N),
            I_0=float(inlet.I_0),
            seed_fraction=float(inlet.seed_fraction),
            B=float(baseline.B),
            working_fluid=baseline.working_fluid,
        )
        momentum_scales.append(max(abs(float(terms["rhs_m"])), 1.0))
        energy_scales.append(max(abs(float(terms["rhs_e"])), 1.0))
    scales = ImplicitResidualScales(
        step_n=step_n_scales,
        step_Te=step_te_scales,
        momentum=np.asarray(momentum_scales, dtype=float),
        energy=np.asarray(energy_scales, dtype=float),
    )
    return variables, scales


def _build_implicit_reference(
    *,
    baseline: BaselineSeed,
    n_intervals: int,
    objective_profile: str = OBJECTIVE_PROFILE_LAB_POC_V2,
) -> tuple[ImplicitTrajectoryVariables, CoarseProfileResult, ImplicitResidualScales]:
    objective_profile = _normalize_objective_profile(objective_profile)
    fluid = baseline.working_fluid
    decision_vector = {
        "log_n_p_in": math.log(float(baseline.n_p_in_nominal)),
        "T_e_in": float(baseline.T_e_in_nominal),
        "Z_in": float(baseline.Z_in_nominal),
        "I_0": float(baseline.I_0_nominal),
        "log_seed_fraction": math.log(float(baseline.seed_fraction_nominal)),
        "a1": float(baseline.area_design_nominal.a1),
        "a2": float(baseline.area_design_nominal.a2),
        "a3": float(baseline.area_design_nominal.a3),
    }
    explicit = CasadiCoarseEvaluator(
        baseline=baseline,
        n_intervals=int(n_intervals),
        objective_profile=objective_profile,
    ).evaluate(decision_vector)
    area_nodes = _evaluate_area_design_nodes(
        ops=_ops_for_numeric(),
        area_design=explicit.area_design,
        length=float(baseline.L),
        n_intervals=int(n_intervals),
        area_scale=float(baseline.area_scale_m2),
    )
    dx = float(baseline.L) / int(n_intervals)
    step_n_scales = []
    step_Te_scales = []
    momentum_scales = []
    energy_scales = []
    for k in range(int(n_intervals)):
        dn_guess = float((explicit.n_p[k + 1] - explicit.n_p[k]) / dx)
        dTe_guess = float((explicit.T_e[k + 1] - explicit.T_e[k]) / dx)
        closure, terms = _dynamic_system_terms(
            ops=_ops_for_numeric(),
            n_p=float(explicit.n_p[k + 1]),
            T_e=float(explicit.T_e[k + 1]),
            A=float(np.asarray(area_nodes["A"], dtype=float)[k + 1]),
            sigma=float(np.asarray(area_nodes["sigma_logA"], dtype=float)[k + 1]),
            dot_N=float(explicit.inlet_design.dot_N),
            I_0=float(explicit.inlet_design.I_0),
            seed_fraction=float(explicit.inlet_design.seed_fraction),
            B=float(baseline.B),
            working_fluid=fluid,
        )
        step_n_scales.append(
            max(abs(float(explicit.n_p[k + 1] - explicit.n_p[k])), abs(dx * dn_guess), abs(float(explicit.n_p[k + 1])), 1.0)
        )
        step_Te_scales.append(
            max(abs(float(explicit.T_e[k + 1] - explicit.T_e[k])), abs(dx * dTe_guess), abs(float(explicit.T_e[k + 1])), 1.0)
        )
        momentum_scales.append(
            max(
                abs(float(terms["rhs_m"])),
                abs(float(terms["M11"] * dn_guess)),
                abs(float(terms["M12"] * dTe_guess)),
                1.0,
            )
        )
        energy_scales.append(
            max(
                abs(float(terms["rhs_e"])),
                abs(float(terms["E11"] * dn_guess)),
                abs(float(terms["E12"] * dTe_guess)),
                1.0,
            )
        )
    residual_scales = ImplicitResidualScales(
        step_n=np.asarray(step_n_scales, dtype=float),
        step_Te=np.asarray(step_Te_scales, dtype=float),
        momentum=np.asarray(momentum_scales, dtype=float),
        energy=np.asarray(energy_scales, dtype=float),
    )

    z = ca.MX.sym("z", 4)
    p = ca.MX.sym("p", 11)
    step_n, step_Te, momentum, energy, _, _ = _implicit_step_residuals(
        ops=_ops_for_casadi(),
        n_prev=p[0],
        T_e_prev=p[1],
        n_next=z[0],
        T_e_next=z[1],
        dn_dx=z[2],
        dTe_dx=z[3],
        A_next=p[2],
        sigma_next=p[3],
        dot_N=p[4],
        I_0=p[5],
        seed_fraction=p[6],
        B=float(baseline.B),
        dx=dx,
        working_fluid=fluid,
    )
    residual_function = ca.Function(
        "implicit_reference_residual",
        [z, p],
        [
            ca.vertcat(
                step_n / p[7],
                step_Te / p[8],
                momentum / p[9],
                energy / p[10],
            )
        ],
    )
    rootfinder = ca.rootfinder("implicit_reference_step", "newton", residual_function)

    n_p_nodes = [float(explicit.n_p[0])]
    T_e_nodes = [float(explicit.T_e[0])]
    dn_dx = []
    dTe_dx = []
    seed_fraction = float(explicit.inlet_design.seed_fraction)
    newton_failures: list[dict[str, Any]] = []
    for k in range(int(n_intervals)):
        z0 = np.array(
            [
                float(explicit.n_p[k + 1]),
                float(explicit.T_e[k + 1]),
                float((explicit.n_p[k + 1] - n_p_nodes[-1]) / dx),
                float((explicit.T_e[k + 1] - T_e_nodes[-1]) / dx),
            ],
            dtype=float,
        )
        params = np.array(
            [
                float(n_p_nodes[-1]),
                float(T_e_nodes[-1]),
                float(np.asarray(area_nodes["A"], dtype=float)[k + 1]),
                float(np.asarray(area_nodes["sigma_logA"], dtype=float)[k + 1]),
                float(explicit.inlet_design.dot_N),
                float(explicit.inlet_design.I_0),
                seed_fraction,
                float(residual_scales.step_n[k]),
                float(residual_scales.step_Te[k]),
                float(residual_scales.momentum[k]),
                float(residual_scales.energy[k]),
            ],
            dtype=float,
        )
        try:
            z_sol = np.asarray(rootfinder(z0, params), dtype=float).reshape(-1)
        except RuntimeError as exc:
            failure = _implicit_reference_step_failure_diagnostic(
                interval_index=k,
                dx=dx,
                length=float(baseline.L),
                z0=z0,
                params=params,
                residual_function=residual_function,
                exc=exc,
            )
            newton_failures.append(failure)
            _LOGGER.warning(
                "implicit reference Newton failed at interval %d (x/L=%.6g); using constant implicit reference fallback",
                int(k),
                float(failure["x_over_L"]),
            )
            break
        n_p_nodes.append(float(z_sol[0]))
        T_e_nodes.append(float(z_sol[1]))
        dn_dx.append(float(z_sol[2]))
        dTe_dx.append(float(z_sol[3]))

    reference_variables = ImplicitTrajectoryVariables(
        decision_vector=decision_vector,
        n_p_nodes=np.asarray(n_p_nodes, dtype=float),
        T_e_nodes=np.asarray(T_e_nodes, dtype=float),
        dn_dx=np.asarray(dn_dx, dtype=float),
        dTe_dx=np.asarray(dTe_dx, dtype=float),
    )
    reference_diagnostics: dict[str, Any] = {
        "fallback_used": False,
        "fallback_reason": None,
        "newton_failure_count": int(len(newton_failures)),
        "first_newton_failure": newton_failures[0] if newton_failures else None,
    }
    reasonableness = _implicit_reference_reasonableness_diagnostics(
        baseline=baseline,
        variables=reference_variables,
    )
    reference_diagnostics["pre_fallback_reasonableness"] = reasonableness
    if newton_failures:
        reference_variables, residual_scales = _constant_implicit_reference(
            baseline=baseline,
            n_intervals=int(n_intervals),
            decision_vector=decision_vector,
        )
        reference_diagnostics["fallback_used"] = True
        reference_diagnostics["fallback_reason"] = "implicit_reference_newton_runtime_error"
    elif not bool(reasonableness["reasonable"]):
        reference_variables, residual_scales = _constant_implicit_reference(
            baseline=baseline,
            n_intervals=int(n_intervals),
            decision_vector=decision_vector,
        )
        reference_diagnostics["fallback_used"] = True
        reference_diagnostics["fallback_reason"] = "implicit_reference_unreasonable"
    reference_result = _build_coarse_result_from_state_trajectory(
        baseline=baseline,
        n_intervals=int(n_intervals),
        variables=reference_variables,
        check_equalities=True,
        residual_scales=residual_scales,
        objective_profile=objective_profile,
    )
    reference_diagnostics["post_fallback_reasonableness"] = _implicit_reference_reasonableness_diagnostics(
        baseline=baseline,
        variables=reference_variables,
    )
    reference_result.diagnostics["implicit_reference"] = reference_diagnostics
    return reference_variables, reference_result, residual_scales


def _build_implicit_step_rootfinder(*, baseline: BaselineSeed, dx: float):
    fluid = baseline.working_fluid
    z = ca.MX.sym("z", 4)
    p = ca.MX.sym("p", 11)
    step_n, step_Te, momentum, energy, _, _ = _implicit_step_residuals(
        ops=_ops_for_casadi(),
        n_prev=p[0],
        T_e_prev=p[1],
        n_next=z[0],
        T_e_next=z[1],
        dn_dx=z[2],
        dTe_dx=z[3],
        A_next=p[2],
        sigma_next=p[3],
        dot_N=p[4],
        I_0=p[5],
        seed_fraction=p[6],
        B=float(baseline.B),
        dx=float(dx),
        working_fluid=fluid,
    )
    return ca.rootfinder(
        "implicit_trajectory_step",
        "newton",
        ca.Function(
            "implicit_trajectory_residual",
            [z, p],
            [
                ca.vertcat(
                    step_n / p[7],
                    step_Te / p[8],
                    momentum / p[9],
                    energy / p[10],
                )
            ],
        ),
    )


def _project_implicit_trajectory(
    *,
    baseline: BaselineSeed,
    n_intervals: int,
    decision_vector: dict[str, float],
    residual_scales: ImplicitResidualScales,
    initial_guess: ImplicitTrajectoryVariables | None = None,
) -> ImplicitTrajectoryVariables:
    dx = float(baseline.L) / int(n_intervals)
    rootfinder = _build_implicit_step_rootfinder(baseline=baseline, dx=dx)
    area_design = SplineAreaDesign(
        a1=float(decision_vector["a1"]),
        a2=float(decision_vector["a2"]),
        a3=float(decision_vector["a3"]),
    )
    area_nodes = _evaluate_area_design_nodes(
        ops=_ops_for_numeric(),
        area_design=area_design,
        length=float(baseline.L),
        n_intervals=int(n_intervals),
        area_scale=float(baseline.area_scale_m2),
    )
    inlet = evaluate_inlet_design_numeric(
        n_p_in=math.exp(float(decision_vector["log_n_p_in"])),
        T_e_in=float(decision_vector["T_e_in"]),
        Z_in=float(decision_vector["Z_in"]),
        I_0=float(decision_vector["I_0"]),
        seed_fraction=math.exp(float(decision_vector["log_seed_fraction"])),
        B=float(baseline.B),
        inlet_area=float(baseline.area_scale_m2),
        working_fluid_profile=baseline.working_fluid,
    )
    n_p_nodes = [float(inlet.n_p)]
    T_e_nodes = [float(inlet.T_e)]
    dn_dx = []
    dTe_dx = []
    guess_n = None if initial_guess is None else np.asarray(initial_guess.n_p_nodes, dtype=float)
    guess_T = None if initial_guess is None else np.asarray(initial_guess.T_e_nodes, dtype=float)
    guess_dn = None if initial_guess is None else np.asarray(initial_guess.dn_dx, dtype=float)
    guess_dTe = None if initial_guess is None else np.asarray(initial_guess.dTe_dx, dtype=float)
    A_nodes = np.asarray(area_nodes["A"], dtype=float)
    sigma_nodes = np.asarray(area_nodes["sigma_logA"], dtype=float)
    for k in range(int(n_intervals)):
        z0 = np.array(
            [
                float(guess_n[k + 1]) if guess_n is not None else float(n_p_nodes[-1]),
                float(guess_T[k + 1]) if guess_T is not None else float(T_e_nodes[-1]),
                float(guess_dn[k]) if guess_dn is not None else 0.0,
                float(guess_dTe[k]) if guess_dTe is not None else 0.0,
            ],
            dtype=float,
        )
        params = np.array(
            [
                float(n_p_nodes[-1]),
                float(T_e_nodes[-1]),
                float(A_nodes[k + 1]),
                float(sigma_nodes[k + 1]),
                float(inlet.dot_N),
                float(inlet.I_0),
                float(inlet.seed_fraction),
                float(residual_scales.step_n[k]),
                float(residual_scales.step_Te[k]),
                float(residual_scales.momentum[k]),
                float(residual_scales.energy[k]),
            ],
            dtype=float,
        )
        z_sol = np.asarray(rootfinder(z0, params), dtype=float).reshape(-1)
        n_p_nodes.append(float(z_sol[0]))
        T_e_nodes.append(float(z_sol[1]))
        dn_dx.append(float(z_sol[2]))
        dTe_dx.append(float(z_sol[3]))
    return ImplicitTrajectoryVariables(
        decision_vector=dict(decision_vector),
        n_p_nodes=np.asarray(n_p_nodes, dtype=float),
        T_e_nodes=np.asarray(T_e_nodes, dtype=float),
        dn_dx=np.asarray(dn_dx, dtype=float),
        dTe_dx=np.asarray(dTe_dx, dtype=float),
    )


def _interpolate_decision_vector(
    low: dict[str, float],
    high: dict[str, float],
    alpha: float,
) -> dict[str, float]:
    return {
        key: float((1.0 - alpha) * float(low[key]) + alpha * float(high[key]))
        for key in low.keys()
    }


def _interpolate_implicit_variables(
    low: ImplicitTrajectoryVariables,
    high: ImplicitTrajectoryVariables,
    alpha: float,
) -> ImplicitTrajectoryVariables:
    decision_vector = _interpolate_decision_vector(low.decision_vector, high.decision_vector, alpha)
    return ImplicitTrajectoryVariables(
        decision_vector=decision_vector,
        n_p_nodes=(1.0 - alpha) * np.asarray(low.n_p_nodes, dtype=float) + alpha * np.asarray(high.n_p_nodes, dtype=float),
        T_e_nodes=(1.0 - alpha) * np.asarray(low.T_e_nodes, dtype=float) + alpha * np.asarray(high.T_e_nodes, dtype=float),
        dn_dx=(1.0 - alpha) * np.asarray(low.dn_dx, dtype=float) + alpha * np.asarray(high.dn_dx, dtype=float),
        dTe_dx=(1.0 - alpha) * np.asarray(low.dTe_dx, dtype=float) + alpha * np.asarray(high.dTe_dx, dtype=float),
    )


def _restore_feasible_implicit_solution(
    *,
    baseline: BaselineSeed,
    n_intervals: int,
    reference_variables: ImplicitTrajectoryVariables,
    reference_result: CoarseProfileResult,
    residual_scales: ImplicitResidualScales,
    candidate_variables: ImplicitTrajectoryVariables,
    candidate_result: CoarseProfileResult,
    objective_profile: str = OBJECTIVE_PROFILE_LAB_POC_V2,
    max_bisection_steps: int = 10,
) -> tuple[ImplicitTrajectoryVariables, CoarseProfileResult, dict[str, Any]]:
    objective_profile = _normalize_objective_profile(objective_profile)
    if bool(candidate_result.diagnostics.get("acceptable", False)):
        return candidate_variables, candidate_result, {
            "used": False,
            "reason": None,
            "alpha": 1.0,
        }

    try:
        projected_candidate = _project_implicit_trajectory(
            baseline=baseline,
            n_intervals=int(n_intervals),
            decision_vector=dict(candidate_variables.decision_vector),
            residual_scales=residual_scales,
            initial_guess=candidate_variables,
        )
        projected_candidate_result = _build_coarse_result_from_state_trajectory(
            baseline=baseline,
            n_intervals=int(n_intervals),
            variables=projected_candidate,
            check_equalities=True,
            residual_scales=residual_scales,
            objective_profile=objective_profile,
        )
        if bool(projected_candidate_result.diagnostics.get("acceptable", False)):
            return projected_candidate, projected_candidate_result, {
                "used": True,
                "reason": (
                    "MAiNGO incumbent was projected back onto the implicit trajectory manifold "
                    "to remove residual drift before handoff."
                ),
                "alpha": 1.0,
            }
    except Exception:
        pass

    best_variables = reference_variables
    best_result = reference_result
    lo = 0.0
    hi = 1.0
    repair_reason = (
        "MAiNGO incumbent violated the tightened coarse-path feasibility check; "
        "restored the nearest feasible point along the baseline-to-incumbent homotopy."
    )
    for _ in range(int(max_bisection_steps)):
        alpha = 0.5 * (lo + hi)
        guess = _interpolate_implicit_variables(reference_variables, candidate_variables, alpha)
        decision = dict(guess.decision_vector)
        try:
            projected = _project_implicit_trajectory(
                baseline=baseline,
                n_intervals=int(n_intervals),
                decision_vector=decision,
                residual_scales=residual_scales,
                initial_guess=guess,
            )
        except Exception:
            hi = alpha
            continue
        projected_result = _build_coarse_result_from_state_trajectory(
            baseline=baseline,
            n_intervals=int(n_intervals),
            variables=projected,
            check_equalities=True,
            residual_scales=residual_scales,
            objective_profile=objective_profile,
        )
        if bool(projected_result.diagnostics.get("acceptable", False)):
            lo = alpha
            best_variables = projected
            best_result = projected_result
        else:
            hi = alpha
    return best_variables, best_result, {
        "used": True,
        "reason": repair_reason,
        "alpha": float(lo),
    }
