from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from .constants import _EPS, _G_HARD_MARGIN, _TP_MIN, OBJECTIVE_PROFILE_LAB_POC_V2
from .geometry import SplineAreaDesign, _evaluate_area_design_nodes
from .implicit import (
    ImplicitResidualScales,
    ImplicitTrajectoryVariables,
    _build_coarse_result_from_state_trajectory,
    _build_implicit_reference,
)
from .models import BaselineSeed, CoarseProfileResult
from .numerics import _max_op, _min_op, _ops_for_maingo, _ops_for_numeric, _reduce_min, _safe_pos, _velikhov_margin_penalty
from .physics import (
    _design_score_generic,
    _dynamic_system_terms,
    _evaluate_midpoint_closures,
    _implicit_step_residuals,
    _inlet_design_generic,
)
from .profiles import _normalize_objective_profile


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
_CRITICAL_DECISION_NAME = "x_sonic"


def _reduce_max(ops, values: list[Any]):
    if not values:
        raise ValueError("cannot reduce an empty list.")
    acc = values[0]
    for value in values[1:]:
        acc = _max_op(ops, acc, value)
    return acc


def _clip_value(ops, value, lower: float, upper: float):
    return _min_op(ops, float(upper), _max_op(ops, float(lower), value))


def _safe_positive_denom(ops, value, floor: float):
    return _safe_pos(ops, value + float(floor), float(floor))


def _model_function(maingopy_module, items):
    if getattr(maingopy_module, "ModelFunction", None) is None:
        return list(items)
    model_function = maingopy_module.ModelFunction()
    for item in items:
        model_function.push_back(item)
    return model_function


@dataclass(frozen=True)
class ReducedImplicitRollout:
    decision_vector: dict[str, Any]
    inlet: dict[str, Any]
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


@dataclass(frozen=True)
class CriticalPointScales:
    det: float
    numerator_n: float
    numerator_Te: float
    x_initial: float


@dataclass(frozen=True)
class CriticalPointResiduals:
    x_sonic: Any
    max_abs_residual: Any
    max_gate: Any
    det_residuals: list[Any]
    numerator_n_residuals: list[Any]
    numerator_Te_residuals: list[Any]


def _scaled_momentum_energy_residuals(
    *,
    ops,
    n_prev,
    T_e_prev,
    n_next,
    T_e_next,
    A_next,
    sigma_next,
    dot_N,
    I_0,
    seed_fraction,
    B: float,
    dx: float,
    momentum_scale: float,
    energy_scale: float,
    working_fluid,
):
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
    return momentum / float(momentum_scale), energy / float(energy_scale), closure, terms, dn_dx, dTe_dx


def _critical_numerators(terms: dict[str, Any]) -> tuple[Any, Any]:
    numerator_n = terms["rhs_m"] * terms["E12"] - terms["M12"] * terms["rhs_e"]
    numerator_Te = terms["M11"] * terms["rhs_e"] - terms["rhs_m"] * terms["E11"]
    return numerator_n, numerator_Te


def _critical_scales_from_reference(
    *,
    baseline: BaselineSeed,
    n_intervals: int,
    reference_variables: ImplicitTrajectoryVariables,
) -> CriticalPointScales:
    ops = _ops_for_numeric()
    area_design = SplineAreaDesign(
        a1=float(reference_variables.decision_vector["a1"]),
        a2=float(reference_variables.decision_vector["a2"]),
        a3=float(reference_variables.decision_vector["a3"]),
    )
    area_nodes = _evaluate_area_design_nodes(
        ops=ops,
        area_design=area_design,
        length=float(baseline.L),
        n_intervals=int(n_intervals),
        area_scale=float(baseline.area_scale_m2),
        **baseline.area_reference_kwargs(),
    )
    inlet = _inlet_design_generic(
        ops=ops,
        n_p_in=math.exp(float(reference_variables.decision_vector["log_n_p_in"])),
        T_e_in=float(reference_variables.decision_vector["T_e_in"]),
        Z_in=float(reference_variables.decision_vector["Z_in"]),
        I_0=float(reference_variables.decision_vector["I_0"]),
        seed_fraction=math.exp(float(reference_variables.decision_vector["log_seed_fraction"])),
        B=float(baseline.B),
        inlet_A=float(baseline.area_scale_m2),
        working_fluid=baseline.working_fluid,
    )
    det_values = []
    numerator_n_values = []
    numerator_Te_values = []
    for idx in range(int(n_intervals) + 1):
        _, terms = _dynamic_system_terms(
            ops=ops,
            n_p=float(reference_variables.n_p_nodes[idx]),
            T_e=float(reference_variables.T_e_nodes[idx]),
            A=float(np.asarray(area_nodes["A"], dtype=float)[idx]),
            sigma=float(np.asarray(area_nodes["sigma_logA"], dtype=float)[idx]),
            dot_N=float(inlet["dot_N"]),
            I_0=float(inlet["I_0"]),
            seed_fraction=float(inlet["seed_fraction"]),
            B=float(baseline.B),
            working_fluid=baseline.working_fluid,
        )
        numerator_n, numerator_Te = _critical_numerators(terms)
        det_values.append(float(terms["det"]))
        numerator_n_values.append(float(numerator_n))
        numerator_Te_values.append(float(numerator_Te))
    det_abs = np.abs(np.asarray(det_values, dtype=float))
    idx_min = int(np.argmin(det_abs))
    x_nodes = np.asarray(area_nodes["x"], dtype=float)
    return CriticalPointScales(
        det=max(float(np.max(det_abs)), 1.0),
        numerator_n=max(float(np.max(np.abs(np.asarray(numerator_n_values, dtype=float)))), 1.0),
        numerator_Te=max(float(np.max(np.abs(np.asarray(numerator_Te_values, dtype=float)))), 1.0),
        x_initial=float(x_nodes[idx_min]),
    )


def critical_point_residuals(
    *,
    ops,
    baseline: BaselineSeed,
    rollout: ReducedImplicitRollout,
    x_sonic,
    scales: CriticalPointScales,
    gate_padding_fraction: float = 0.25,
) -> CriticalPointResiduals:
    x_nodes = np.asarray(rollout.area_nodes["x"], dtype=float)
    dx = float(baseline.L) / (len(x_nodes) - 1)
    padding = max(float(gate_padding_fraction) * dx, 1e-12)
    det_residuals = []
    numerator_n_residuals = []
    numerator_Te_residuals = []
    gates = []
    for k in range(len(x_nodes) - 1):
        x_left = float(x_nodes[k])
        x_right = float(x_nodes[k + 1])
        theta = _clip_value(ops, (x_sonic - x_left) / dx, 0.0, 1.0)
        left_gate = _clip_value(ops, (x_sonic - x_left + padding) / padding, 0.0, 1.0)
        right_gate = _clip_value(ops, (x_right - x_sonic + padding) / padding, 0.0, 1.0)
        gate = left_gate * right_gate
        gates.append(gate)

        n_sonic = (1.0 - theta) * rollout.n_p_nodes[k] + theta * rollout.n_p_nodes[k + 1]
        T_sonic = (1.0 - theta) * rollout.T_e_nodes[k] + theta * rollout.T_e_nodes[k + 1]
        A_sonic = (1.0 - theta) * rollout.area_nodes["A"][k] + theta * rollout.area_nodes["A"][k + 1]
        sigma_sonic = (1.0 - theta) * rollout.area_nodes["sigma_logA"][k] + theta * rollout.area_nodes["sigma_logA"][k + 1]
        _, terms = _dynamic_system_terms(
            ops=ops,
            n_p=n_sonic,
            T_e=T_sonic,
            A=A_sonic,
            sigma=sigma_sonic,
            dot_N=rollout.inlet["dot_N"],
            I_0=rollout.inlet["I_0"],
            seed_fraction=rollout.inlet["seed_fraction"],
            B=float(baseline.B),
            working_fluid=baseline.working_fluid,
        )
        numerator_n, numerator_Te = _critical_numerators(terms)
        det_residuals.append(gate * terms["det"] / float(scales.det))
        numerator_n_residuals.append(gate * numerator_n / float(scales.numerator_n))
        numerator_Te_residuals.append(gate * numerator_Te / float(scales.numerator_Te))
    all_residuals = [*det_residuals, *numerator_n_residuals, *numerator_Te_residuals]
    return CriticalPointResiduals(
        x_sonic=x_sonic,
        max_abs_residual=_reduce_max(ops, [ops.fabs(item) for item in all_residuals]),
        max_gate=_reduce_max(ops, gates),
        det_residuals=det_residuals,
        numerator_n_residuals=numerator_n_residuals,
        numerator_Te_residuals=numerator_Te_residuals,
    )


def _fixed_gauss_newton_log_state_2d(
    *,
    ops,
    initial_log_n,
    initial_log_Te,
    residual_fn,
    steps: int,
    finite_difference_step: float,
    regularization: float,
    max_log_step: float,
):
    log_n = initial_log_n
    log_Te = initial_log_Te
    fd = float(finite_difference_step)
    for _ in range(int(steps)):
        r_m, r_e = residual_fn(log_n, log_Te)
        r_m_np, r_e_np = residual_fn(log_n + fd, log_Te)
        r_m_nm, r_e_nm = residual_fn(log_n - fd, log_Te)
        r_m_tp, r_e_tp = residual_fn(log_n, log_Te + fd)
        r_m_tm, r_e_tm = residual_fn(log_n, log_Te - fd)

        j11 = (r_m_np - r_m_nm) / (2.0 * fd)
        j21 = (r_e_np - r_e_nm) / (2.0 * fd)
        j12 = (r_m_tp - r_m_tm) / (2.0 * fd)
        j22 = (r_e_tp - r_e_tm) / (2.0 * fd)

        a11 = j11 * j11 + j21 * j21 + float(regularization)
        a12 = j11 * j12 + j21 * j22
        a22 = j12 * j12 + j22 * j22 + float(regularization)
        g1 = j11 * r_m + j21 * r_e
        g2 = j12 * r_m + j22 * r_e
        denom = _safe_positive_denom(ops, a11 * a22 - a12 * a12, float(regularization))
        delta_log_n = (a22 * g1 - a12 * g2) / denom
        delta_log_Te = (-a12 * g1 + a11 * g2) / denom
        delta_log_n = _clip_value(ops, delta_log_n, -float(max_log_step), float(max_log_step))
        delta_log_Te = _clip_value(ops, delta_log_Te, -float(max_log_step), float(max_log_step))
        log_n = log_n - delta_log_n
        log_Te = log_Te - delta_log_Te
    return log_n, log_Te


def rollout_reduced_implicit_generic(
    *,
    ops,
    baseline: BaselineSeed,
    n_intervals: int,
    decision_vector: dict[str, Any],
    residual_scales: ImplicitResidualScales,
    newton_steps: int = 10,
    finite_difference_step: float = 1e-4,
    regularization: float = 1e-8,
    max_log_step: float = 0.5,
) -> ReducedImplicitRollout:
    fluid = baseline.working_fluid
    n_intervals = int(n_intervals)
    dx = float(baseline.L) / n_intervals
    log_n_p_in = decision_vector["log_n_p_in"]
    T_e_in = decision_vector["T_e_in"]
    Z_in = decision_vector["Z_in"]
    I_0 = decision_vector["I_0"]
    log_seed_fraction = decision_vector["log_seed_fraction"]
    seed_fraction = ops.exp(log_seed_fraction)
    area_design = SplineAreaDesign(
        a1=decision_vector["a1"],
        a2=decision_vector["a2"],
        a3=decision_vector["a3"],
    )
    inlet = _inlet_design_generic(
        ops=ops,
        n_p_in=ops.exp(log_n_p_in),
        T_e_in=T_e_in,
        Z_in=Z_in,
        I_0=I_0,
        seed_fraction=seed_fraction,
        B=float(baseline.B),
        inlet_A=float(baseline.area_scale_m2),
        working_fluid=fluid,
    )
    area_nodes = _evaluate_area_design_nodes(
        ops=ops,
        area_design=area_design,
        length=float(baseline.L),
        n_intervals=n_intervals,
        area_scale=float(baseline.area_scale_m2),
        **baseline.area_reference_kwargs(),
    )

    n_nodes = [inlet["n_p"]]
    T_nodes = [inlet["T_e"]]
    dn_dx_values = []
    dTe_dx_values = []
    scaled_momentum_residuals = []
    scaled_energy_residuals = []
    previous_dn_dx = 0.0
    previous_dTe_dx = 0.0
    for k in range(n_intervals):
        n_prev = n_nodes[-1]
        T_prev = T_nodes[-1]
        A_next = area_nodes["A"][k + 1]
        sigma_next = area_nodes["sigma_logA"][k + 1]
        momentum_scale = float(residual_scales.momentum[k])
        energy_scale = float(residual_scales.energy[k])
        n_initial = _safe_pos(ops, n_prev + dx * previous_dn_dx, 1.0)
        T_initial = _safe_pos(ops, T_prev + dx * previous_dTe_dx, 1.0)
        log_n = ops.log(n_initial)
        log_Te = ops.log(T_initial)

        def residual_pair(candidate_log_n, candidate_log_Te):
            n_next_candidate = ops.exp(candidate_log_n)
            T_next_candidate = ops.exp(candidate_log_Te)
            r_m, r_e, _, _, _, _ = _scaled_momentum_energy_residuals(
                ops=ops,
                n_prev=n_prev,
                T_e_prev=T_prev,
                n_next=n_next_candidate,
                T_e_next=T_next_candidate,
                A_next=A_next,
                sigma_next=sigma_next,
                dot_N=inlet["dot_N"],
                I_0=I_0,
                seed_fraction=seed_fraction,
                B=float(baseline.B),
                dx=dx,
                momentum_scale=momentum_scale,
                energy_scale=energy_scale,
                working_fluid=fluid,
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
        r_m, r_e, _, _, dn_dx, dTe_dx = _scaled_momentum_energy_residuals(
            ops=ops,
            n_prev=n_prev,
            T_e_prev=T_prev,
            n_next=n_next,
            T_e_next=T_next,
            A_next=A_next,
            sigma_next=sigma_next,
            dot_N=inlet["dot_N"],
            I_0=I_0,
            seed_fraction=seed_fraction,
            B=float(baseline.B),
            dx=dx,
            momentum_scale=momentum_scale,
            energy_scale=energy_scale,
            working_fluid=fluid,
        )
        n_nodes.append(n_next)
        T_nodes.append(T_next)
        dn_dx_values.append(dn_dx)
        dTe_dx_values.append(dTe_dx)
        scaled_momentum_residuals.append(r_m)
        scaled_energy_residuals.append(r_e)
        previous_dn_dx = dn_dx
        previous_dTe_dx = dTe_dx

    closures = []
    terms_by_node = []
    for idx in range(n_intervals + 1):
        closure, terms = _dynamic_system_terms(
            ops=ops,
            n_p=n_nodes[idx],
            T_e=T_nodes[idx],
            A=area_nodes["A"][idx],
            sigma=area_nodes["sigma_logA"][idx],
            dot_N=inlet["dot_N"],
            I_0=I_0,
            seed_fraction=seed_fraction,
            B=float(baseline.B),
            working_fluid=fluid,
        )
        closures.append(closure)
        terms_by_node.append(terms)

    abs_residuals = [ops.fabs(item) for item in [*scaled_momentum_residuals, *scaled_energy_residuals]]
    det_abs = [ops.fabs(item["det"]) for item in terms_by_node]
    return ReducedImplicitRollout(
        decision_vector=dict(decision_vector),
        inlet=inlet,
        area_nodes=area_nodes,
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


class _MAiNGOHybridReducedImplicitModelBase:
    formulation = "reduced_implicit_fixed_newton_backward_euler"

    def __init__(
        self,
        *,
        baseline: BaselineSeed,
        n_intervals: int,
        maingopy_module,
        objective_profile: str = OBJECTIVE_PROFILE_LAB_POC_V2,
        newton_steps: int = 10,
        residual_tolerance: float = 1e-5,
        critical_mode: bool = False,
        critical_residual_tolerance: float = 1e-4,
    ):
        self._baseline = baseline
        self._n_intervals = int(n_intervals)
        self._maingopy = maingopy_module
        self._ops = _ops_for_maingo(maingopy_module)
        self._objective_profile = _normalize_objective_profile(objective_profile)
        self._working_fluid = baseline.working_fluid
        self._newton_steps = int(newton_steps)
        self._residual_tolerance = float(residual_tolerance)
        self._critical_mode = bool(critical_mode)
        self._critical_residual_tolerance = float(critical_residual_tolerance)
        self._reference_variables, self._reference_profile, self._residual_scales = _build_implicit_reference(
            baseline=baseline,
            n_intervals=self._n_intervals,
            objective_profile=self._objective_profile,
        )
        self._critical_scales = _critical_scales_from_reference(
            baseline=baseline,
            n_intervals=self._n_intervals,
            reference_variables=self._reference_variables,
        )
        if self._critical_mode:
            reference_decision = dict(self._reference_variables.decision_vector)
            reference_decision[_CRITICAL_DECISION_NAME] = float(self._critical_scales.x_initial)
            self._reference_variables = ImplicitTrajectoryVariables(
                decision_vector=reference_decision,
                n_p_nodes=np.asarray(self._reference_variables.n_p_nodes, dtype=float),
                T_e_nodes=np.asarray(self._reference_variables.T_e_nodes, dtype=float),
                dn_dx=np.asarray(self._reference_variables.dn_dx, dtype=float),
                dTe_dx=np.asarray(self._reference_variables.dTe_dx, dtype=float),
            )
        self._variable_specs = list(self._baseline.optimization_variable_bounds())
        self._initial_point = list(self._baseline.initial_point())
        if self._critical_mode:
            self._variable_specs.append((0.0, float(self._baseline.L), _CRITICAL_DECISION_NAME))
            self._initial_point.append(float(self._critical_scales.x_initial))

    @property
    def total_variables(self) -> int:
        return len(self._variable_specs)

    def summary_metadata(self) -> dict[str, Any]:
        return {
            "newton_steps": int(self._newton_steps),
            "residual_tolerance": float(self._residual_tolerance),
            "residual_scale_momentum_min": float(np.min(self._residual_scales.momentum)),
            "residual_scale_momentum_max": float(np.max(self._residual_scales.momentum)),
            "residual_scale_energy_min": float(np.min(self._residual_scales.energy)),
            "residual_scale_energy_max": float(np.max(self._residual_scales.energy)),
            "critical_mode": bool(self._critical_mode),
            "critical_residual_tolerance": float(self._critical_residual_tolerance),
            "critical_x_initial": float(self._critical_scales.x_initial),
            "critical_scale_det": float(self._critical_scales.det),
            "critical_scale_numerator_n": float(self._critical_scales.numerator_n),
            "critical_scale_numerator_Te": float(self._critical_scales.numerator_Te),
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

    def get_initial_point(self):
        return list(self._initial_point)

    def _decision_from_values(self, values) -> dict[str, Any]:
        values = list(values)
        expected_names = list(_DECISION_NAMES)
        if self._critical_mode:
            expected_names.append(_CRITICAL_DECISION_NAME)
        if len(values) != len(expected_names):
            raise ValueError(
                f"reduced implicit solution size mismatch: got {len(values)}, expected {len(expected_names)}."
            )
        return {name: value for name, value in zip(expected_names, values, strict=True)}

    def _numeric_rollout_to_trajectory(self, decision: dict[str, float]) -> ImplicitTrajectoryVariables:
        rollout = rollout_reduced_implicit_generic(
            ops=_ops_for_numeric(),
            baseline=self._baseline,
            n_intervals=self._n_intervals,
            decision_vector=decision,
            residual_scales=self._residual_scales,
            newton_steps=self._newton_steps,
        )
        return ImplicitTrajectoryVariables(
            decision_vector={key: float(value) for key, value in decision.items()},
            n_p_nodes=np.asarray(rollout.n_p_nodes, dtype=float),
            T_e_nodes=np.asarray(rollout.T_e_nodes, dtype=float),
            dn_dx=np.asarray(rollout.dn_dx, dtype=float),
            dTe_dx=np.asarray(rollout.dTe_dx, dtype=float),
        )

    def decode_solution_point(self, values) -> ImplicitTrajectoryVariables:
        decision = {key: float(value) for key, value in self._decision_from_values(values).items()}
        return self._numeric_rollout_to_trajectory(decision)

    def evaluate_solution(self, solution: ImplicitTrajectoryVariables) -> CoarseProfileResult:
        result = _build_coarse_result_from_state_trajectory(
            baseline=self._baseline,
            n_intervals=self._n_intervals,
            variables=solution,
            check_equalities=True,
            residual_scales=self._residual_scales,
            objective_profile=self._objective_profile,
        )
        result.diagnostics["formulation"] = self.formulation
        result.diagnostics["reduced_implicit_newton_steps"] = int(self._newton_steps)
        result.diagnostics["critical_mode"] = bool(self._critical_mode)
        if self._critical_mode:
            decision = dict(solution.decision_vector)
            x_sonic = float(decision.get(_CRITICAL_DECISION_NAME, self._critical_scales.x_initial))
            rollout = rollout_reduced_implicit_generic(
                ops=_ops_for_numeric(),
                baseline=self._baseline,
                n_intervals=self._n_intervals,
                decision_vector=decision,
                residual_scales=self._residual_scales,
                newton_steps=self._newton_steps,
            )
            critical = critical_point_residuals(
                ops=_ops_for_numeric(),
                baseline=self._baseline,
                rollout=rollout,
                x_sonic=x_sonic,
                scales=self._critical_scales,
            )
            critical_max_abs_residual = float(critical.max_abs_residual)
            result.diagnostics["critical_x_sonic"] = x_sonic
            result.diagnostics["critical_max_abs_residual"] = critical_max_abs_residual
            result.diagnostics["critical_max_gate"] = float(critical.max_gate)
            result.diagnostics["critical_residual_tolerance"] = float(self._critical_residual_tolerance)
            result.diagnostics["acceptable"] = bool(result.diagnostics.get("acceptable", False)) and (
                critical_max_abs_residual <= float(self._critical_residual_tolerance)
            )
        return result

    def evaluate(self, vars):
        result = self._maingopy.EvaluationContainer()
        decision = self._decision_from_values(vars)
        area_design = SplineAreaDesign(
            a1=decision["a1"],
            a2=decision["a2"],
            a3=decision["a3"],
        )
        rollout = rollout_reduced_implicit_generic(
            ops=self._ops,
            baseline=self._baseline,
            n_intervals=self._n_intervals,
            decision_vector=decision,
            residual_scales=self._residual_scales,
            newton_steps=self._newton_steps,
        )
        _, midpoint_closures = _evaluate_midpoint_closures(
            ops=self._ops,
            area_design=area_design,
            length=float(self._baseline.L),
            n_intervals=self._n_intervals,
            n_p_nodes=rollout.n_p_nodes,
            T_e_nodes=rollout.T_e_nodes,
            dot_N=rollout.inlet["dot_N"],
            I_0=decision["I_0"],
            seed_fraction=self._ops.exp(decision["log_seed_fraction"]),
            B=float(self._baseline.B),
            area_scale=float(self._baseline.area_scale_m2),
            working_fluid=self._working_fluid,
            **self._baseline.area_reference_kwargs(),
        )
        power_density_nodes = []
        for idx, closure in enumerate(rollout.closures):
            power_density_nodes.append(-rollout.area_nodes["A"][idx] * closure["J_x"] * closure["E_x"] / 1e8)
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
            working_fluid=self._working_fluid,
        )
        min_g_nodes = _reduce_min(self._ops, [item["G"] for item in rollout.closures])
        min_g_midpoints = _reduce_min(self._ops, [item["G"] for item in midpoint_closures])
        min_g_all = _min_op(self._ops, min_g_nodes, min_g_midpoints)
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
        critical = None
        if self._critical_mode:
            critical = critical_point_residuals(
                ops=self._ops,
                baseline=self._baseline,
                rollout=rollout,
                x_sonic=decision[_CRITICAL_DECISION_NAME],
                scales=self._critical_scales,
            )
            ineq.append(critical.max_abs_residual - float(self._critical_residual_tolerance))

        result.objective = -design_score
        result.ineq = _model_function(self._maingopy, ineq)
        result.output = [
            self._maingopy.OutputVariable("design_score", design_score),
            self._maingopy.OutputVariable("raw_design_score", raw_design_score),
            self._maingopy.OutputVariable("velikhov_penalty", velikhov_penalty),
            self._maingopy.OutputVariable("reduced_implicit_max_scaled_residual", rollout.max_abs_scaled_residual),
            self._maingopy.OutputVariable("reduced_implicit_min_abs_det", rollout.min_abs_det),
            self._maingopy.OutputVariable("inlet_G", rollout.inlet["G"]),
            self._maingopy.OutputVariable("inlet_mach", rollout.inlet["mach"]),
            self._maingopy.OutputVariable("min_path_G_nodes", min_g_nodes),
            self._maingopy.OutputVariable("min_path_G_midpoints", min_g_midpoints),
            self._maingopy.OutputVariable("min_path_G_all", min_g_all),
        ]
        if critical is not None:
            result.output.extend(
                [
                    self._maingopy.OutputVariable("critical_x_sonic", decision[_CRITICAL_DECISION_NAME]),
                    self._maingopy.OutputVariable("critical_max_abs_residual", critical.max_abs_residual),
                    self._maingopy.OutputVariable("critical_max_gate", critical.max_gate),
                ]
            )
        return result
