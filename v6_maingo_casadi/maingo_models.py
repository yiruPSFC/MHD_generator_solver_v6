from __future__ import annotations

import importlib
from typing import Any

import numpy as np

from .casadi_evaluator import CasadiCoarseEvaluator
from .constants import _EPS, _G_HARD_MARGIN, _TP_MIN, OBJECTIVE_PROFILE_LAB_POC_V2
from .geometry import SplineAreaDesign, _evaluate_area_design_nodes
from .implicit import (
    ImplicitTrajectoryScaling,
    ImplicitTrajectoryVariables,
    _ImplicitVariableLayout,
    _build_coarse_result_from_state_trajectory,
    _build_implicit_reference,
    _scaled_interval,
)
from .models import BaselineSeed, CoarseProfileResult
from .numerics import _min_op, _ops_for_maingo, _reduce_min, _velikhov_margin_penalty
from .physics import (
    _design_score_generic,
    _dynamic_system_terms,
    _evaluate_midpoint_closures,
    _inlet_design_generic,
    _rk4_rollout_generic,
)
from .profiles import _normalize_objective_profile

class _MAiNGOHybridModelBase:
    formulation = "rk4_reduced_benchmark"
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

    def __init__(
        self,
        *,
        baseline: BaselineSeed,
        n_intervals: int,
        maingopy_module,
        objective_profile: str = OBJECTIVE_PROFILE_LAB_POC_V2,
    ):
        self._baseline = baseline
        self._n_intervals = int(n_intervals)
        self._maingopy = maingopy_module
        self._ops = _ops_for_maingo(maingopy_module)
        self._objective_profile = _normalize_objective_profile(objective_profile)
        self._working_fluid = baseline.working_fluid
        self._numeric_evaluator = CasadiCoarseEvaluator(
            baseline=baseline,
            n_intervals=self._n_intervals,
            objective_profile=self._objective_profile,
        )

    @property
    def total_variables(self) -> int:
        return 8

    def summary_metadata(self) -> dict[str, Any]:
        return {"benchmark_model": "rk4_reduced"}

    def get_variables(self):
        variables = []
        for lower, upper, name in self._baseline.optimization_variable_bounds():
            variables.append(
                self._maingopy.OptimizationVariable(
                    self._maingopy.Bounds(float(lower), float(upper)),
                    self._maingopy.VT_CONTINUOUS,
                    str(name),
                )
            )
        return variables

    def get_initial_point(self):
        return self._baseline.initial_point()

    def decode_solution_point(self, values) -> dict[str, float]:
        values = list(values)
        if len(values) != len(self._DECISION_NAMES):
            raise ValueError(f"RK4 reduced solution size mismatch: got {len(values)}, expected {len(self._DECISION_NAMES)}.")
        return {name: float(value) for name, value in zip(self._DECISION_NAMES, values, strict=True)}

    def evaluate_solution(self, solution: dict[str, float]) -> CoarseProfileResult:
        return self._numeric_evaluator.evaluate(solution)

    def evaluate(self, vars):
        result = self._maingopy.EvaluationContainer()
        log_n_p_in, T_e_in, Z_in, I_0, log_seed_fraction, a1, a2, a3 = vars
        area_design = SplineAreaDesign(a1=a1, a2=a2, a3=a3)
        rollout = _rk4_rollout_generic(
            ops=self._ops,
            n_p_in=self._ops.exp(log_n_p_in),
            T_e_in=T_e_in,
            Z_in=Z_in,
            I_0=I_0,
            seed_fraction=self._ops.exp(log_seed_fraction),
            area_design=area_design,
            B=float(self._baseline.B),
            length=float(self._baseline.L),
            n_intervals=self._n_intervals,
            area_scale=float(self._baseline.area_scale_m2),
            working_fluid=self._working_fluid,
        )
        closures = rollout["closures"]
        x_nodes = np.asarray(rollout["x"], dtype=float)
        power_density_nodes = []
        for idx, closure in enumerate(closures):
            power_density_nodes.append(-rollout["A"][idx] * closure["J_x"] * closure["E_x"] / 1e8)
        design_score = _design_score_generic(
            ops=self._ops,
            outlet_T_e=rollout["T_e"][-1],
            outlet_T_p=closures[-1]["T_p"],
            outlet_n_p=rollout["n_p"][-1],
            outlet_n_e=closures[-1]["n_e"],
            inlet_T_e=rollout["inlet"]["T_e"],
            inlet_T_p=rollout["inlet"]["T_p"],
            inlet_mach=rollout["inlet"]["mach"],
            power_density_nodes=power_density_nodes,
            x_nodes=x_nodes,
            seed_fraction=self._ops.exp(log_seed_fraction),
            B=float(self._baseline.B),
            length=float(self._baseline.L),
            objective_profile=self._objective_profile,
            inlet_n_p=rollout["inlet"]["n_p"],
            inlet_n_e=rollout["inlet"]["n_e"],
            inlet_v=rollout["inlet"]["v_in"],
            inlet_A=float(self._baseline.area_scale_m2),
            working_fluid=self._working_fluid,
        )
        ineq = []
        sigma_max = float(self._baseline.schedule[0]["max_abs_dlogA_dx"])
        tp_min = float(self._baseline.schedule[0].get("tp_min", _TP_MIN))
        mach_min = float(self._baseline.schedule[0].get("mach_min", 0.0) or 0.0)
        for sigma in rollout["sigma_logA"]:
            ineq.append(self._ops.fabs(sigma) - sigma_max)
        for closure in closures:
            ineq.append(tp_min - closure["T_p"])
            ineq.append(0.0 - closure["G"])
            if mach_min > 0.0:
                ineq.append(mach_min - closure["mach"])
        result.objective = -design_score
        result.ineq = ineq
        result.output = [
            self._maingopy.OutputVariable("design_score", design_score),
            self._maingopy.OutputVariable("inlet_G", rollout["inlet"]["G"]),
            self._maingopy.OutputVariable("inlet_mach", rollout["inlet"]["mach"]),
        ]
        return result


class _MAiNGOHybridImplicitModelBase:
    def __init__(
        self,
        *,
        baseline: BaselineSeed,
        n_intervals: int,
        maingopy_module,
        objective_profile: str = OBJECTIVE_PROFILE_LAB_POC_V2,
    ):
        self._baseline = baseline
        self._n_intervals = int(n_intervals)
        self._maingopy = maingopy_module
        self._ops = _ops_for_maingo(maingopy_module)
        self._layout = _ImplicitVariableLayout(self._n_intervals)
        self._objective_profile = _normalize_objective_profile(objective_profile)
        self._working_fluid = baseline.working_fluid
        self._reference_variables, self._reference_profile, self._residual_scales = _build_implicit_reference(
            baseline=baseline,
            n_intervals=self._n_intervals,
            objective_profile=self._objective_profile,
        )
        self._trajectory_scaling = self._build_trajectory_scaling()
        self._variable_specs = self._build_variable_specs()
        self._initial_point = self._build_initial_point()

    @property
    def total_variables(self) -> int:
        return self._layout.total_variables

    def _build_trajectory_scaling(self) -> ImplicitTrajectoryScaling:
        ref = self._reference_profile
        n_global_lb = max(1.0, 0.25 * float(np.min(ref.n_p)))
        n_global_ub = max(n_global_lb * 1.01, 4.0 * float(np.max(ref.n_p)))
        Te_global_lb = max(1.0, 0.25 * float(np.min(ref.T_e)))
        Te_global_ub = max(Te_global_lb * 1.01, 2.5 * float(np.max(ref.T_e)))
        dx = float(self._baseline.L) / int(self._n_intervals)
        dn_global_span = max((n_global_ub - n_global_lb) / max(dx, _EPS), 1.0)
        dTe_global_span = max((Te_global_ub - Te_global_lb) / max(dx, _EPS), 1.0)
        n_center = []
        n_scale = []
        Te_center = []
        Te_scale = []
        dn_center = []
        dn_scale = []
        dTe_center = []
        dTe_scale = []
        for k in range(int(self._n_intervals)):
            n_ref = float(self._reference_variables.n_p_nodes[k + 1])
            Te_ref = float(self._reference_variables.T_e_nodes[k + 1])
            dn_ref = float(self._reference_variables.dn_dx[k])
            dTe_ref = float(self._reference_variables.dTe_dx[k])
            n_lb = max(n_global_lb, 0.5 * n_ref)
            n_ub = min(n_global_ub, 2.0 * n_ref)
            if n_ub <= n_lb:
                n_ub = max(n_global_ub, n_lb * 1.01)
            Te_lb = max(Te_global_lb, 0.5 * Te_ref)
            Te_ub = min(Te_global_ub, 1.75 * Te_ref)
            if Te_ub <= Te_lb:
                Te_ub = max(Te_global_ub, Te_lb * 1.01)
            dn_span = max(4.0 * abs(dn_ref), dn_global_span)
            dTe_span = max(4.0 * abs(dTe_ref), dTe_global_span)
            n_ref_center, n_ref_scale, _, _ = _scaled_interval(n_lb, n_ub, n_ref, min_scale=max(1.0, 0.05 * abs(n_ref)))
            Te_ref_center, Te_ref_scale, _, _ = _scaled_interval(
                Te_lb,
                Te_ub,
                Te_ref,
                min_scale=max(1.0, 0.05 * abs(Te_ref)),
            )
            dn_ref_center, dn_ref_scale, _, _ = _scaled_interval(
                dn_ref - dn_span,
                dn_ref + dn_span,
                dn_ref,
                min_scale=max(1.0, 0.05 * dn_span),
            )
            dTe_ref_center, dTe_ref_scale, _, _ = _scaled_interval(
                dTe_ref - dTe_span,
                dTe_ref + dTe_span,
                dTe_ref,
                min_scale=max(1.0, 0.05 * dTe_span),
            )
            n_center.append(float(n_ref_center))
            n_scale.append(float(n_ref_scale))
            Te_center.append(float(Te_ref_center))
            Te_scale.append(float(Te_ref_scale))
            dn_center.append(float(dn_ref_center))
            dn_scale.append(float(dn_ref_scale))
            dTe_center.append(float(dTe_ref_center))
            dTe_scale.append(float(dTe_ref_scale))
        return ImplicitTrajectoryScaling(
            n_p_center=np.asarray(n_center, dtype=float),
            n_p_scale=np.asarray(n_scale, dtype=float),
            T_e_center=np.asarray(Te_center, dtype=float),
            T_e_scale=np.asarray(Te_scale, dtype=float),
            dn_dx_center=np.asarray(dn_center, dtype=float),
            dn_dx_scale=np.asarray(dn_scale, dtype=float),
            dTe_dx_center=np.asarray(dTe_center, dtype=float),
            dTe_dx_scale=np.asarray(dTe_scale, dtype=float),
        )

    def _build_variable_specs(self) -> list[tuple[float, float, str]]:
        specs = list(self._baseline.optimization_variable_bounds())
        n_specs = []
        Te_specs = []
        dn_specs = []
        dTe_specs = []
        ref = self._reference_profile
        n_global_lb = max(1.0, 0.25 * float(np.min(ref.n_p)))
        n_global_ub = max(n_global_lb * 1.01, 4.0 * float(np.max(ref.n_p)))
        Te_global_lb = max(1.0, 0.25 * float(np.min(ref.T_e)))
        Te_global_ub = max(Te_global_lb * 1.01, 2.5 * float(np.max(ref.T_e)))
        dx = float(self._baseline.L) / int(self._n_intervals)
        dn_global_span = max((n_global_ub - n_global_lb) / max(dx, _EPS), 1.0)
        dTe_global_span = max((Te_global_ub - Te_global_lb) / max(dx, _EPS), 1.0)
        for k in range(int(self._n_intervals)):
            n_ref = float(self._reference_variables.n_p_nodes[k + 1])
            Te_ref = float(self._reference_variables.T_e_nodes[k + 1])
            dn_ref = float(self._reference_variables.dn_dx[k])
            dTe_ref = float(self._reference_variables.dTe_dx[k])
            n_lb = max(n_global_lb, 0.5 * n_ref)
            n_ub = min(n_global_ub, 2.0 * n_ref)
            if n_ub <= n_lb:
                n_ub = max(n_global_ub, n_lb * 1.01)
            Te_lb = max(Te_global_lb, 0.5 * Te_ref)
            Te_ub = min(Te_global_ub, 1.75 * Te_ref)
            if Te_ub <= Te_lb:
                Te_ub = max(Te_global_ub, Te_lb * 1.01)
            dn_span = max(4.0 * abs(dn_ref), dn_global_span)
            dTe_span = max(4.0 * abs(dTe_ref), dTe_global_span)

            _, _, n_scaled_lb, n_scaled_ub = _scaled_interval(
                n_lb,
                n_ub,
                float(self._trajectory_scaling.n_p_center[k]),
                min_scale=float(self._trajectory_scaling.n_p_scale[k]),
            )
            _, _, Te_scaled_lb, Te_scaled_ub = _scaled_interval(
                Te_lb,
                Te_ub,
                float(self._trajectory_scaling.T_e_center[k]),
                min_scale=float(self._trajectory_scaling.T_e_scale[k]),
            )
            _, _, dn_scaled_lb, dn_scaled_ub = _scaled_interval(
                dn_ref - dn_span,
                dn_ref + dn_span,
                float(self._trajectory_scaling.dn_dx_center[k]),
                min_scale=float(self._trajectory_scaling.dn_dx_scale[k]),
            )
            _, _, dTe_scaled_lb, dTe_scaled_ub = _scaled_interval(
                dTe_ref - dTe_span,
                dTe_ref + dTe_span,
                float(self._trajectory_scaling.dTe_dx_center[k]),
                min_scale=float(self._trajectory_scaling.dTe_dx_scale[k]),
            )

            n_specs.append((float(n_scaled_lb), float(n_scaled_ub), f"n_p_hat_{k + 1}"))
            Te_specs.append((float(Te_scaled_lb), float(Te_scaled_ub), f"T_e_hat_{k + 1}"))
            dn_specs.append((float(dn_scaled_lb), float(dn_scaled_ub), f"dn_dx_hat_{k + 1}"))
            dTe_specs.append((float(dTe_scaled_lb), float(dTe_scaled_ub), f"dTe_dx_hat_{k + 1}"))
        return specs + n_specs + Te_specs + dn_specs + dTe_specs

    def _build_initial_point(self) -> list[float]:
        decision = [
            float(self._reference_variables.decision_vector["log_n_p_in"]),
            float(self._reference_variables.decision_vector["T_e_in"]),
            float(self._reference_variables.decision_vector["Z_in"]),
            float(self._reference_variables.decision_vector["I_0"]),
            float(self._reference_variables.decision_vector["log_seed_fraction"]),
            float(self._reference_variables.decision_vector["a1"]),
            float(self._reference_variables.decision_vector["a2"]),
            float(self._reference_variables.decision_vector["a3"]),
        ]
        n_tail = self._trajectory_scaling.encode_n_p_tail(self._reference_variables.n_p_nodes[1:])
        T_tail = self._trajectory_scaling.encode_T_e_tail(self._reference_variables.T_e_nodes[1:])
        dn_tail = self._trajectory_scaling.encode_dn_dx(self._reference_variables.dn_dx)
        dTe_tail = self._trajectory_scaling.encode_dTe_dx(self._reference_variables.dTe_dx)
        return decision + n_tail + T_tail + dn_tail + dTe_tail

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

    def get_initial_point(self):
        return list(self._initial_point)

    def decode_solution_point(self, values) -> ImplicitTrajectoryVariables:
        return self._layout.decode_numeric(values, self._trajectory_scaling)

    def evaluate_solution(self, solution: ImplicitTrajectoryVariables) -> CoarseProfileResult:
        return _build_coarse_result_from_state_trajectory(
            baseline=self._baseline,
            n_intervals=self._n_intervals,
            variables=solution,
            check_equalities=True,
            residual_scales=self._residual_scales,
            objective_profile=self._objective_profile,
        )

    def evaluate(self, vars):
        result = self._maingopy.EvaluationContainer()
        decision_raw, n_p_tail_raw, T_e_tail_raw, dn_dx_raw, dTe_dx_raw = self._layout.split_raw(vars)
        log_n_p_in, T_e_in, Z_in, I_0, log_seed_fraction, a1, a2, a3 = decision_raw
        area_design = SplineAreaDesign(a1=a1, a2=a2, a3=a3)
        inlet = _inlet_design_generic(
            ops=self._ops,
            n_p_in=self._ops.exp(log_n_p_in),
            T_e_in=T_e_in,
            Z_in=Z_in,
            I_0=I_0,
            seed_fraction=self._ops.exp(log_seed_fraction),
            B=float(self._baseline.B),
            inlet_A=float(self._baseline.area_scale_m2),
            working_fluid=self._working_fluid,
        )
        area_nodes = _evaluate_area_design_nodes(
            ops=self._ops,
            area_design=area_design,
            length=float(self._baseline.L),
            n_intervals=self._n_intervals,
            area_scale=float(self._baseline.area_scale_m2),
        )
        n_p_tail = self._trajectory_scaling.decode_n_p_tail(n_p_tail_raw)
        T_e_tail = self._trajectory_scaling.decode_T_e_tail(T_e_tail_raw)
        dn_dx = self._trajectory_scaling.decode_dn_dx(dn_dx_raw)
        dTe_dx = self._trajectory_scaling.decode_dTe_dx(dTe_dx_raw)
        n_nodes = [self._ops.exp(log_n_p_in), *n_p_tail]
        T_nodes = [T_e_in, *T_e_tail]
        _, midpoint_closures = _evaluate_midpoint_closures(
            ops=self._ops,
            area_design=area_design,
            length=float(self._baseline.L),
            n_intervals=self._n_intervals,
            n_p_nodes=n_nodes,
            T_e_nodes=T_nodes,
            dot_N=inlet["dot_N"],
            I_0=I_0,
            seed_fraction=self._ops.exp(log_seed_fraction),
            B=float(self._baseline.B),
            area_scale=float(self._baseline.area_scale_m2),
            working_fluid=self._working_fluid,
        )
        closures = []
        terms_by_node = []
        power_density_nodes = []
        ineq = []
        eq = []
        sigma_max = float(self._baseline.schedule[0]["max_abs_dlogA_dx"])
        tp_min = float(self._baseline.schedule[0].get("tp_min", _TP_MIN))
        mach_min = float(self._baseline.schedule[0].get("mach_min", 0.0) or 0.0)
        for idx in range(int(self._n_intervals) + 1):
            sigma = area_nodes["sigma_logA"][idx]
            A = area_nodes["A"][idx]
            ineq.append(self._ops.fabs(sigma) - sigma_max)
            closure, terms = _dynamic_system_terms(
                ops=self._ops,
                n_p=n_nodes[idx],
                T_e=T_nodes[idx],
                A=A,
                sigma=sigma,
                dot_N=inlet["dot_N"],
                I_0=I_0,
                seed_fraction=self._ops.exp(log_seed_fraction),
                B=float(self._baseline.B),
                working_fluid=self._working_fluid,
            )
            closures.append(closure)
            terms_by_node.append(terms)
            power_density_nodes.append(-A * closure["J_x"] * closure["E_x"] / 1e8)
            ineq.append(tp_min - closure["T_p"])
            ineq.append(float(_G_HARD_MARGIN) - closure["G"])
            if mach_min > 0.0:
                ineq.append(mach_min - closure["mach"])
        for closure_mid in midpoint_closures:
            ineq.append(float(_G_HARD_MARGIN) - closure_mid["G"])
        dx = float(self._baseline.L) / int(self._n_intervals)
        for k in range(int(self._n_intervals)):
            eq.append((n_nodes[k + 1] - n_nodes[k] - dx * dn_dx[k]) / float(self._residual_scales.step_n[k]))
            eq.append((T_nodes[k + 1] - T_nodes[k] - dx * dTe_dx[k]) / float(self._residual_scales.step_Te[k]))
            terms = terms_by_node[k + 1]
            eq.append(
                (terms["M11"] * dn_dx[k] + terms["M12"] * dTe_dx[k] - terms["rhs_m"])
                / float(self._residual_scales.momentum[k])
            )
            eq.append(
                (terms["E11"] * dn_dx[k] + terms["E12"] * dTe_dx[k] - terms["rhs_e"])
                / float(self._residual_scales.energy[k])
            )
        raw_design_score = _design_score_generic(
            ops=self._ops,
            outlet_T_e=T_nodes[-1],
            outlet_T_p=closures[-1]["T_p"],
            outlet_n_p=n_nodes[-1],
            outlet_n_e=closures[-1]["n_e"],
            inlet_T_e=inlet["T_e"],
            inlet_T_p=inlet["T_p"],
            inlet_mach=inlet["mach"],
            power_density_nodes=power_density_nodes,
            x_nodes=np.asarray(area_nodes["x"], dtype=float),
            seed_fraction=self._ops.exp(log_seed_fraction),
            B=float(self._baseline.B),
            length=float(self._baseline.L),
            objective_profile=self._objective_profile,
            inlet_n_p=inlet["n_p"],
            inlet_n_e=inlet["n_e"],
            inlet_v=inlet["v_in"],
            inlet_A=float(self._baseline.area_scale_m2),
            working_fluid=self._working_fluid,
        )
        min_g_nodes = _reduce_min(self._ops, [item["G"] for item in closures])
        min_g_midpoints = _reduce_min(self._ops, [item["G"] for item in midpoint_closures])
        min_g_all = _min_op(self._ops, min_g_nodes, min_g_midpoints)
        velikhov_penalty = _velikhov_margin_penalty(self._ops, min_g_all)
        design_score = raw_design_score - velikhov_penalty
        def _model_function(items):
            model_function = self._maingopy.ModelFunction()
            for item in items:
                model_function.push_back(item)
            return model_function

        result.objective = -design_score
        result.ineq = _model_function(ineq)
        result.eq = _model_function(eq)
        result.output = [
            self._maingopy.OutputVariable("design_score", design_score),
            self._maingopy.OutputVariable("raw_design_score", raw_design_score),
            self._maingopy.OutputVariable("velikhov_penalty", velikhov_penalty),
            self._maingopy.OutputVariable("inlet_G", inlet["G"]),
            self._maingopy.OutputVariable("inlet_mach", inlet["mach"]),
            self._maingopy.OutputVariable("min_path_G_nodes", min_g_nodes),
            self._maingopy.OutputVariable("min_path_G_midpoints", min_g_midpoints),
            self._maingopy.OutputVariable("min_path_G_all", min_g_all),
        ]
        return result


def _import_maingopy():
    try:
        return importlib.import_module("maingopy")
    except Exception as exc:
        raise ImportError(
            "maingopy is required for v6_maingo_casadi. Install the official Python "
            "binding first, e.g. `pip install maingopy`, or make a source-built "
            "maingopy package visible to this interpreter. No fallback solver is implemented."
        ) from exc


def _retcode_name(status: Any) -> str:
    if hasattr(status, "name"):
        return str(status.name)
    text = str(status)
    if "." in text:
        return text.split(".")[-1]
    return text


def _safe_solver_metric(getter, default: Any = None):
    try:
        return getter()
    except Exception:
        return default
