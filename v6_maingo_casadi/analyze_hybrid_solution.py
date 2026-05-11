#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import casadi as ca
import numpy as np

REPO_DIR = Path(__file__).resolve().parents[1]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from v6_maingo_casadi.core import (
    BaselineSeed,
    ImplicitResidualScales,
    ImplicitTrajectoryVariables,
    SplineAreaDesign,
    _G_HARD_MARGIN,
    _TP_MIN,
    _build_coarse_result_from_state_trajectory,
    _design_score_generic,
    _dynamic_system_terms,
    _evaluate_area_design_nodes,
    _evaluate_midpoint_closures,
    _implicit_step_residuals,
    _inlet_design_generic,
    _ops_for_casadi,
    _ops_for_numeric,
    _velikhov_margin_penalty,
)
from v6_maingo_casadi.profiles import _normalize_objective_profile, _normalize_working_fluid_profile


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


@dataclass(frozen=True)
class CurveSnapshot:
    source: str
    decision_vector: dict[str, float]
    n_p_nodes: np.ndarray
    T_e_nodes: np.ndarray
    dn_dx: np.ndarray
    dTe_dx: np.ndarray
    x_nodes: np.ndarray

    @property
    def n_intervals(self) -> int:
        return int(self.n_p_nodes.size - 1)


@dataclass(frozen=True)
class StateScaling:
    n_p_center: np.ndarray
    n_p_scale: np.ndarray
    T_e_center: np.ndarray
    T_e_scale: np.ndarray
    dn_dx_center: np.ndarray
    dn_dx_scale: np.ndarray
    dTe_dx_center: np.ndarray
    dTe_dx_scale: np.ndarray


@dataclass(frozen=True)
class OperatorBundle:
    x: ca.MX
    objective: ca.MX
    ineq: ca.MX
    eq: ca.MX
    measured_ineq: ca.MX
    ineq_meta: list[dict[str, Any]]
    eq_meta: list[dict[str, Any]]
    variable_names: list[str]
    lower_bounds: np.ndarray
    upper_bounds: np.ndarray


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _jsonify(value: Any):
    if isinstance(value, dict):
        return {str(key): _jsonify(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_jsonify(item) for item in value.tolist()]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        out = float(value)
        return out if math.isfinite(out) else None
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _seed_from_hybrid_payload(hybrid: dict[str, Any]) -> BaselineSeed:
    baseline_payload = dict(hybrid.get("baseline_seed", {}) or {})
    baseline_summary = Path(str(baseline_payload["summary_path"]))
    seed = BaselineSeed.from_summary(baseline_summary)
    overrides: dict[str, Any] = {}

    inlet_windows = baseline_payload.get("inlet_windows")
    if inlet_windows:
        overrides["inlet_windows"] = {
            str(key): {
                "guess": float(value["guess"]),
                "min": float(value["min"]),
                "max": float(value["max"]),
            }
            for key, value in dict(inlet_windows).items()
        }
        overrides["n_p_in_nominal"] = float(overrides["inlet_windows"]["n_p_in"]["guess"])
        overrides["T_e_in_nominal"] = float(overrides["inlet_windows"]["T_e_in"]["guess"])
        overrides["Z_in_nominal"] = float(overrides["inlet_windows"]["Z_in"]["guess"])
        overrides["I_0_nominal"] = float(overrides["inlet_windows"]["I_0"]["guess"])
        overrides["seed_fraction_nominal"] = float(overrides["inlet_windows"]["seed_fraction"]["guess"])

    if "area_reference" in baseline_payload:
        raise ValueError(
            "legacy baseline_seed.area_reference payload is no longer supported; "
            "regenerate the seed/run with direct area_design_nominal spline coordinates."
        )
    area_design = baseline_payload.get("area_design_nominal")
    if area_design:
        overrides["area_design_nominal"] = SplineAreaDesign(
            a1=float(area_design["a1"]),
            a2=float(area_design["a2"]),
            a3=float(area_design["a3"]),
        )
    area_windows = baseline_payload.get("area_design_windows")
    if area_windows:
        overrides["area_design_windows"] = {
            str(key): {
                "guess": float(value["guess"]),
                "min": float(value["min"]),
                "max": float(value["max"]),
            }
            for key, value in dict(area_windows).items()
        }

    for key in ("B", "L", "area_scale_m2", "adaptive_bridge_count", "adaptive_bridge_max_count"):
        if key in baseline_payload:
            overrides[key] = baseline_payload[key]
    if "working_fluid" in baseline_payload:
        overrides["working_fluid"] = _normalize_working_fluid_profile(baseline_payload["working_fluid"])
    elif "working_fluid_profile" in hybrid:
        overrides["working_fluid"] = _normalize_working_fluid_profile(hybrid["working_fluid_profile"])

    return replace(seed, **overrides) if overrides else seed


def _objective_profile_from_summary(hybrid: dict[str, Any]) -> str:
    candidates = [
        hybrid.get("objective_profile"),
        dict(hybrid.get("maingo_best", {}) or {}).get("objective_profile"),
        dict(dict(hybrid.get("maingo_best", {}) or {}).get("diagnostics", {}) or {}).get("objective_profile"),
    ]
    maingo_summary_path = dict(hybrid.get("artifacts", {}) or {}).get("maingo_summary_json")
    if maingo_summary_path and Path(str(maingo_summary_path)).exists():
        candidates.append(dict(_load_json(maingo_summary_path)).get("objective_profile"))
    for candidate in candidates:
        if candidate:
            return _normalize_objective_profile(str(candidate))
    return _normalize_objective_profile("lab_poc_v2")


def _decision_from_hybrid(hybrid: dict[str, Any]) -> dict[str, float]:
    solver_status = dict(hybrid.get("maingo_status", {}) or {})
    decision = (
        solver_status.get("handoff_solution_point")
        or solver_status.get("solution_point")
        or dict(hybrid.get("maingo_best", {}) or {}).get("decision_vector")
        or {}
    )
    missing = [name for name in _DECISION_NAMES if name not in decision]
    if missing:
        raise ValueError(f"hybrid summary is missing decision entries: {', '.join(missing)}")
    return {name: float(decision[name]) for name in _DECISION_NAMES}


def _resolve_profile_path(summary_path: str | Path, hybrid: dict[str, Any]) -> Path | None:
    artifacts = dict(hybrid.get("artifacts", {}) or {})
    candidates = [
        artifacts.get("maingo_coarse_profile_npz"),
        artifacts.get("maingo_handoff_profile_npz"),
        artifacts.get("maingo_best_profile_npz"),
        hybrid.get("maingo_best_profile_path"),
        Path(summary_path).with_name("maingo_coarse_profile.npz"),
        Path(summary_path).with_name("maingo_handoff_profile.npz"),
        Path(summary_path).with_name("maingo_best_profile.npz"),
    ]
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(str(candidate))
        if not path.is_absolute():
            path = Path(summary_path).resolve().parent / path
        if path.exists():
            return path.resolve()
    return None


def _load_curve_snapshot(
    *,
    summary_path: str | Path,
    hybrid: dict[str, Any],
    seed: BaselineSeed,
    decision_vector: dict[str, float],
) -> CurveSnapshot:
    profile_path = _resolve_profile_path(summary_path, hybrid)
    if profile_path is None:
        raise FileNotFoundError(
            "could not find a MAiNGO profile NPZ from hybrid summary artifacts or next to the summary"
        )
    with np.load(profile_path) as data:
        n_p = np.asarray(data["n_p"], dtype=float).reshape(-1)
        T_e = np.asarray(data["T_e"], dtype=float).reshape(-1)
        if "x" in data:
            x = np.asarray(data["x"], dtype=float).reshape(-1)
        else:
            x = np.linspace(0.0, float(seed.L), n_p.size, dtype=float)

    if n_p.size != T_e.size or n_p.size < 2:
        raise ValueError("profile NPZ must contain matching n_p and T_e arrays with at least two nodes")
    if x.size != n_p.size:
        x = np.linspace(0.0, float(seed.L), n_p.size, dtype=float)
    n_intervals = int(n_p.size - 1)
    dx = float(seed.L) / n_intervals
    return CurveSnapshot(
        source=str(profile_path),
        decision_vector=dict(decision_vector),
        n_p_nodes=n_p,
        T_e_nodes=T_e,
        dn_dx=np.diff(n_p) / dx,
        dTe_dx=np.diff(T_e) / dx,
        x_nodes=x,
    )


def _state_scaling_from_curve(curve: CurveSnapshot) -> StateScaling:
    def scale(values: np.ndarray, *, floor: float = 1.0) -> np.ndarray:
        return np.maximum(np.abs(np.asarray(values, dtype=float)), float(floor))

    return StateScaling(
        n_p_center=np.asarray(curve.n_p_nodes[1:], dtype=float),
        n_p_scale=scale(curve.n_p_nodes[1:]),
        T_e_center=np.asarray(curve.T_e_nodes[1:], dtype=float),
        T_e_scale=scale(curve.T_e_nodes[1:]),
        dn_dx_center=np.asarray(curve.dn_dx, dtype=float),
        dn_dx_scale=scale(curve.dn_dx),
        dTe_dx_center=np.asarray(curve.dTe_dx, dtype=float),
        dTe_dx_scale=scale(curve.dTe_dx),
    )


def _residual_scales_from_curve(seed: BaselineSeed, curve: CurveSnapshot) -> ImplicitResidualScales:
    ops = _ops_for_numeric()
    area_design = SplineAreaDesign(
        a1=float(curve.decision_vector["a1"]),
        a2=float(curve.decision_vector["a2"]),
        a3=float(curve.decision_vector["a3"]),
    )
    area_nodes = _evaluate_area_design_nodes(
        ops=ops,
        area_design=area_design,
        length=float(seed.L),
        n_intervals=curve.n_intervals,
        area_scale=float(seed.area_scale_m2),
    )
    inlet = _inlet_design_generic(
        ops=ops,
        n_p_in=math.exp(float(curve.decision_vector["log_n_p_in"])),
        T_e_in=float(curve.decision_vector["T_e_in"]),
        Z_in=float(curve.decision_vector["Z_in"]),
        I_0=float(curve.decision_vector["I_0"]),
        seed_fraction=math.exp(float(curve.decision_vector["log_seed_fraction"])),
        B=float(seed.B),
        inlet_A=float(seed.area_scale_m2),
        working_fluid=seed.working_fluid,
    )
    dx = float(seed.L) / curve.n_intervals
    step_n = []
    step_Te = []
    momentum = []
    energy = []
    for k in range(curve.n_intervals):
        _, terms = _dynamic_system_terms(
            ops=ops,
            n_p=float(curve.n_p_nodes[k + 1]),
            T_e=float(curve.T_e_nodes[k + 1]),
            A=float(np.asarray(area_nodes["A"], dtype=float)[k + 1]),
            sigma=float(np.asarray(area_nodes["sigma_logA"], dtype=float)[k + 1]),
            dot_N=float(inlet["dot_N"]),
            I_0=float(inlet["I_0"]),
            seed_fraction=float(inlet["seed_fraction"]),
            B=float(seed.B),
            working_fluid=seed.working_fluid,
        )
        dn = float(curve.dn_dx[k])
        dte = float(curve.dTe_dx[k])
        step_n.append(
            max(
                abs(float(curve.n_p_nodes[k + 1] - curve.n_p_nodes[k])),
                abs(float(dx * dn)),
                abs(float(curve.n_p_nodes[k + 1])),
                1.0,
            )
        )
        step_Te.append(
            max(
                abs(float(curve.T_e_nodes[k + 1] - curve.T_e_nodes[k])),
                abs(float(dx * dte)),
                abs(float(curve.T_e_nodes[k + 1])),
                1.0,
            )
        )
        momentum.append(max(abs(float(terms["rhs_m"])), abs(float(terms["M11"] * dn)), abs(float(terms["M12"] * dte)), 1.0))
        energy.append(max(abs(float(terms["rhs_e"])), abs(float(terms["E11"] * dn)), abs(float(terms["E12"] * dte)), 1.0))

    return ImplicitResidualScales(
        step_n=np.asarray(step_n, dtype=float),
        step_Te=np.asarray(step_Te, dtype=float),
        momentum=np.asarray(momentum, dtype=float),
        energy=np.asarray(energy, dtype=float),
    )


def _trajectory_variables_from_curve(curve: CurveSnapshot) -> ImplicitTrajectoryVariables:
    return ImplicitTrajectoryVariables(
        decision_vector=dict(curve.decision_vector),
        n_p_nodes=np.asarray(curve.n_p_nodes, dtype=float),
        T_e_nodes=np.asarray(curve.T_e_nodes, dtype=float),
        dn_dx=np.asarray(curve.dn_dx, dtype=float),
        dTe_dx=np.asarray(curve.dTe_dx, dtype=float),
    )


def _decision_bounds(seed: BaselineSeed) -> dict[str, tuple[float, float]]:
    return {
        name: (float(lower), float(upper))
        for lower, upper, name in seed.optimization_variable_bounds()
        if name in _DECISION_NAMES
    }


def _fraction_to_box(value: float, lower: float, upper: float) -> dict[str, float]:
    width = max(float(upper) - float(lower), 1e-30)
    return {
        "fraction": float((float(value) - float(lower)) / width),
        "dist_lo": float(float(value) - float(lower)),
        "dist_hi": float(float(upper) - float(value)),
    }


def _build_operator_bundle(
    *,
    seed: BaselineSeed,
    curve: CurveSnapshot,
    scaling: StateScaling,
    residual_scales: ImplicitResidualScales,
    objective_profile: str,
) -> OperatorBundle:
    ni = int(curve.n_intervals)
    total_variables = 8 + 4 * ni
    x = ca.MX.sym("x", total_variables)
    ops = _ops_for_casadi()
    idx = 0
    decision_raw = [x[idx + i] for i in range(8)]
    idx += 8
    n_hat = [x[idx + i] for i in range(ni)]
    idx += ni
    te_hat = [x[idx + i] for i in range(ni)]
    idx += ni
    dn_hat = [x[idx + i] for i in range(ni)]
    idx += ni
    dte_hat = [x[idx + i] for i in range(ni)]

    log_n_p_in, T_e_in, Z_in, I_0, log_seed_fraction, a1, a2, a3 = decision_raw
    n_tail = [
        float(scaling.n_p_center[k]) + float(scaling.n_p_scale[k]) * n_hat[k]
        for k in range(ni)
    ]
    te_tail = [
        float(scaling.T_e_center[k]) + float(scaling.T_e_scale[k]) * te_hat[k]
        for k in range(ni)
    ]
    dn_dx = [
        float(scaling.dn_dx_center[k]) + float(scaling.dn_dx_scale[k]) * dn_hat[k]
        for k in range(ni)
    ]
    dte_dx = [
        float(scaling.dTe_dx_center[k]) + float(scaling.dTe_dx_scale[k]) * dte_hat[k]
        for k in range(ni)
    ]

    area_design = SplineAreaDesign(a1=a1, a2=a2, a3=a3)
    inlet = _inlet_design_generic(
        ops=ops,
        n_p_in=ops.exp(log_n_p_in),
        T_e_in=T_e_in,
        Z_in=Z_in,
        I_0=I_0,
        seed_fraction=ops.exp(log_seed_fraction),
        B=float(seed.B),
        inlet_A=float(seed.area_scale_m2),
        working_fluid=seed.working_fluid,
    )
    area_nodes = _evaluate_area_design_nodes(
        ops=ops,
        area_design=area_design,
        length=float(seed.L),
        n_intervals=ni,
        area_scale=float(seed.area_scale_m2),
    )
    n_nodes = [ops.exp(log_n_p_in), *n_tail]
    te_nodes = [T_e_in, *te_tail]
    _, midpoint_closures = _evaluate_midpoint_closures(
        ops=ops,
        area_design=area_design,
        length=float(seed.L),
        n_intervals=ni,
        n_p_nodes=n_nodes,
        T_e_nodes=te_nodes,
        dot_N=inlet["dot_N"],
        I_0=I_0,
        seed_fraction=ops.exp(log_seed_fraction),
        B=float(seed.B),
        area_scale=float(seed.area_scale_m2),
        working_fluid=seed.working_fluid,
    )

    x_nodes = np.asarray(area_nodes["x"], dtype=float)
    x_mid = 0.5 * (x_nodes[:-1] + x_nodes[1:])
    closures = []
    terms_by_node = []
    power_density_nodes = []
    ineq = []
    ineq_measured = []
    ineq_meta = []
    eq = []
    eq_meta = []
    sigma_max = float(seed.schedule[0]["max_abs_dlogA_dx"])
    tp_min = float(seed.schedule[0].get("tp_min", _TP_MIN))
    mach_min = float(seed.schedule[0].get("mach_min", 0.0) or 0.0)

    for i in range(ni + 1):
        sigma = area_nodes["sigma_logA"][i]
        A = area_nodes["A"][i]
        closure, terms = _dynamic_system_terms(
            ops=ops,
            n_p=n_nodes[i],
            T_e=te_nodes[i],
            A=A,
            sigma=sigma,
            dot_N=inlet["dot_N"],
            I_0=I_0,
            seed_fraction=ops.exp(log_seed_fraction),
            B=float(seed.B),
            working_fluid=seed.working_fluid,
        )
        closures.append(closure)
        terms_by_node.append(terms)
        power_density_nodes.append(-A * closure["J_x"] * closure["E_x"] / 1e8)

        ineq.append(ca.fabs(sigma) - sigma_max)
        ineq_measured.append(ca.fabs(sigma))
        ineq_meta.append(
            {
                "name": f"sigma_node_{i}",
                "family": "sigma_upper",
                "index": i,
                "x": float(x_nodes[i]),
                "sense": "abs(sigma_logA) <= max_abs_dlogA_dx",
                "measured_name": "abs_sigma_logA",
                "bound": sigma_max,
            }
        )
        ineq.append(tp_min - closure["T_p"])
        ineq_measured.append(closure["T_p"])
        ineq_meta.append(
            {
                "name": f"Tp_node_{i}",
                "family": "T_p_lower",
                "index": i,
                "x": float(x_nodes[i]),
                "sense": "T_p >= tp_min",
                "measured_name": "T_p",
                "bound": tp_min,
            }
        )
        ineq.append(float(_G_HARD_MARGIN) - closure["G"])
        ineq_measured.append(closure["G"])
        ineq_meta.append(
            {
                "name": f"G_node_{i}",
                "family": "velikhov_lower",
                "index": i,
                "x": float(x_nodes[i]),
                "sense": "G >= G_hard_margin",
                "measured_name": "G",
                "bound": float(_G_HARD_MARGIN),
            }
        )
        if mach_min > 0.0:
            ineq.append(mach_min - closure["mach"])
            ineq_measured.append(closure["mach"])
            ineq_meta.append(
                {
                    "name": f"Mach_node_{i}",
                    "family": "mach_lower",
                    "index": i,
                    "x": float(x_nodes[i]),
                    "sense": "mach >= mach_min",
                    "measured_name": "mach",
                    "bound": mach_min,
                }
            )

    for i, closure_mid in enumerate(midpoint_closures):
        ineq.append(float(_G_HARD_MARGIN) - closure_mid["G"])
        ineq_measured.append(closure_mid["G"])
        ineq_meta.append(
            {
                "name": f"G_mid_{i}",
                "family": "velikhov_lower_midpoint",
                "index": i,
                "x": float(x_mid[i]),
                "sense": "G_midpoint >= G_hard_margin",
                "measured_name": "G_midpoint",
                "bound": float(_G_HARD_MARGIN),
            }
        )

    dx = float(seed.L) / ni
    for k in range(ni):
        terms = terms_by_node[k + 1]
        step_n, step_Te, momentum, energy, _, _ = _implicit_step_residuals(
            ops=ops,
            n_prev=n_nodes[k],
            T_e_prev=te_nodes[k],
            n_next=n_nodes[k + 1],
            T_e_next=te_nodes[k + 1],
            dn_dx=dn_dx[k],
            dTe_dx=dte_dx[k],
            A_next=area_nodes["A"][k + 1],
            sigma_next=area_nodes["sigma_logA"][k + 1],
            dot_N=inlet["dot_N"],
            I_0=I_0,
            seed_fraction=ops.exp(log_seed_fraction),
            B=float(seed.B),
            dx=dx,
            working_fluid=seed.working_fluid,
        )
        scaled_eqs = (
            (step_n / float(residual_scales.step_n[k]), "step_n"),
            (step_Te / float(residual_scales.step_Te[k]), "step_Te"),
            (momentum / float(residual_scales.momentum[k]), "momentum"),
            (energy / float(residual_scales.energy[k]), "energy"),
        )
        for expr, family in scaled_eqs:
            eq.append(expr)
            eq_meta.append(
                {
                    "name": f"{family}_{k}",
                    "family": family,
                    "index": k,
                    "x_left": float(x_nodes[k]),
                    "x_right": float(x_nodes[k + 1]),
                    "sense": "scaled residual == 0",
                }
            )

    raw_score = _design_score_generic(
        ops=ops,
        outlet_T_e=te_nodes[-1],
        outlet_T_p=closures[-1]["T_p"],
        outlet_n_p=n_nodes[-1],
        outlet_n_e=closures[-1]["n_e"],
        inlet_T_e=inlet["T_e"],
        inlet_T_p=inlet["T_p"],
        inlet_mach=inlet["mach"],
        power_density_nodes=power_density_nodes,
        x_nodes=np.asarray(area_nodes["x"], dtype=float),
        seed_fraction=ops.exp(log_seed_fraction),
        B=float(seed.B),
        length=float(seed.L),
        objective_profile=objective_profile,
        inlet_n_p=inlet["n_p"],
        inlet_n_e=closures[0]["n_e"],
        inlet_v=inlet["v_in"],
        inlet_A=float(seed.area_scale_m2),
        working_fluid=seed.working_fluid,
    )
    min_g_nodes = ca.mmin(ca.vertcat(*[item["G"] for item in closures]))
    min_g_midpoints = ca.mmin(ca.vertcat(*[item["G"] for item in midpoint_closures]))
    min_g_all = ca.fmin(min_g_nodes, min_g_midpoints)
    objective = -(raw_score - _velikhov_margin_penalty(ops, min_g_all))

    decision_bounds = _decision_bounds(seed)
    lower_bounds = []
    upper_bounds = []
    variable_names = list(_DECISION_NAMES)
    for name in _DECISION_NAMES:
        lower, upper = decision_bounds[name]
        lower_bounds.append(lower)
        upper_bounds.append(upper)
    for prefix in ("n_p_hat", "T_e_hat", "dn_dx_hat", "dTe_dx_hat"):
        for k in range(ni):
            variable_names.append(f"{prefix}_{k + 1}")
            lower_bounds.append(-np.inf)
            upper_bounds.append(np.inf)

    return OperatorBundle(
        x=x,
        objective=objective,
        ineq=ca.vertcat(*ineq) if ineq else ca.MX.zeros(0, 1),
        eq=ca.vertcat(*eq) if eq else ca.MX.zeros(0, 1),
        measured_ineq=ca.vertcat(*ineq_measured) if ineq_measured else ca.MX.zeros(0, 1),
        ineq_meta=ineq_meta,
        eq_meta=eq_meta,
        variable_names=variable_names,
        lower_bounds=np.asarray(lower_bounds, dtype=float),
        upper_bounds=np.asarray(upper_bounds, dtype=float),
    )


def _curve_x0(curve: CurveSnapshot) -> np.ndarray:
    return np.asarray(
        [float(curve.decision_vector[name]) for name in _DECISION_NAMES]
        + [0.0 for _ in range(4 * curve.n_intervals)],
        dtype=float,
    )


def _evaluate_operator_bundle(bundle: OperatorBundle, x0: np.ndarray) -> dict[str, np.ndarray | float]:
    fun = ca.Function(
        "fixed_curve_kkt_ops",
        [bundle.x],
        [
            bundle.objective,
            bundle.ineq,
            bundle.eq,
            bundle.measured_ineq,
            ca.gradient(bundle.objective, bundle.x),
            ca.jacobian(bundle.ineq, bundle.x),
            ca.jacobian(bundle.eq, bundle.x),
        ],
    )
    objective, ineq, eq, measured, grad, jac_ineq, jac_eq = fun(x0)
    return {
        "objective_to_minimize": float(objective),
        "ineq": np.asarray(ineq, dtype=float).reshape(-1),
        "eq": np.asarray(eq, dtype=float).reshape(-1),
        "measured_ineq": np.asarray(measured, dtype=float).reshape(-1),
        "grad_objective": np.asarray(grad, dtype=float).reshape(-1),
        "jac_ineq": np.asarray(jac_ineq, dtype=float),
        "jac_eq": np.asarray(jac_eq, dtype=float),
    }


def _operator_reason(meta: dict[str, Any], residual: float, active: bool, violated: bool) -> str:
    if violated:
        status = "violated"
    elif active:
        status = "active_or_near_active"
    else:
        status = "inactive"
    family = str(meta["family"])
    if family == "sigma_upper":
        reason = "area log-slope is close to or above the schedule slope cap"
    elif family in {"velikhov_lower", "velikhov_lower_midpoint"}:
        reason = "Velikhov margin is close to or below the hard stability margin"
    elif family == "T_p_lower":
        reason = "heavy-particle temperature is close to or below its lower bound"
    elif family == "mach_lower":
        reason = "Mach number is close to or below its lower bound"
    else:
        reason = "operator residual is close to its active threshold"
    return f"{status}: {reason}; residual={residual:.6e}"


def _format_inequality_entries(
    *,
    meta: list[dict[str, Any]],
    residuals: np.ndarray,
    measured: np.ndarray,
    multipliers: np.ndarray,
    active_tol: float,
    violation_tol: float,
) -> list[dict[str, Any]]:
    entries = []
    for i, item in enumerate(meta):
        residual = float(residuals[i])
        active = bool(residual >= -float(active_tol))
        violated = bool(residual > float(violation_tol))
        entry = dict(item)
        entry.update(
            {
                "residual": residual,
                "margin": float(-residual),
                "measured_value": float(measured[i]),
                "active": active,
                "violated": violated,
                "recovered_multiplier": float(multipliers[i]),
                "why": _operator_reason(item, residual, active, violated),
            }
        )
        entries.append(entry)
    return entries


def _format_equality_entries(
    *,
    meta: list[dict[str, Any]],
    residuals: np.ndarray,
    multipliers: np.ndarray,
) -> list[dict[str, Any]]:
    entries = []
    for i, item in enumerate(meta):
        residual = float(residuals[i])
        entry = dict(item)
        entry.update(
            {
                "residual": residual,
                "abs_residual": float(abs(residual)),
                "recovered_multiplier": float(multipliers[i]),
                "why": f"equality residual should vanish; scaled residual={residual:.6e}",
            }
        )
        entries.append(entry)
    return entries


def _active_bounds(
    *,
    x0: np.ndarray,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
    variable_names: list[str],
    abs_tol: float,
    rel_tol: float,
) -> list[dict[str, Any]]:
    active = []
    for i, name in enumerate(variable_names):
        value = float(x0[i])
        lower = float(lower_bounds[i])
        upper = float(upper_bounds[i])
        if math.isfinite(lower):
            width = abs(upper - lower) if math.isfinite(upper) else max(abs(value), 1.0)
            tol = max(float(abs_tol), float(rel_tol) * max(width, 1.0))
            dist = value - lower
            if dist <= tol:
                active.append(
                    {
                        "variable": name,
                        "index": i,
                        "side": "lower",
                        "value": value,
                        "bound": lower,
                        "distance": float(dist),
                        "constraint_value": float(lower - value),
                    }
                )
        if math.isfinite(upper):
            width = abs(upper - lower) if math.isfinite(lower) else max(abs(value), 1.0)
            tol = max(float(abs_tol), float(rel_tol) * max(width, 1.0))
            dist = upper - value
            if dist <= tol:
                active.append(
                    {
                        "variable": name,
                        "index": i,
                        "side": "upper",
                        "value": value,
                        "bound": upper,
                        "distance": float(dist),
                        "constraint_value": float(value - upper),
                    }
                )
    return active


def _recover_multipliers(
    *,
    values: dict[str, np.ndarray | float],
    bundle: OperatorBundle,
    x0: np.ndarray,
    active_tol: float,
    bound_abs_tol: float,
    bound_rel_tol: float,
) -> dict[str, Any]:
    grad = np.asarray(values["grad_objective"], dtype=float).reshape(-1)
    jac_eq = np.asarray(values["jac_eq"], dtype=float)
    jac_ineq = np.asarray(values["jac_ineq"], dtype=float)
    eq_values = np.asarray(values["eq"], dtype=float).reshape(-1)
    ineq_values = np.asarray(values["ineq"], dtype=float).reshape(-1)
    active_ineq_indices = [int(i) for i, value in enumerate(ineq_values) if float(value) >= -float(active_tol)]
    active_bounds = _active_bounds(
        x0=x0,
        lower_bounds=bundle.lower_bounds,
        upper_bounds=bundle.upper_bounds,
        variable_names=bundle.variable_names,
        abs_tol=bound_abs_tol,
        rel_tol=bound_rel_tol,
    )

    columns = []
    labels = []
    for i in range(eq_values.size):
        columns.append(jac_eq[i, :])
        labels.append({"type": "eq", "index": int(i), "name": bundle.eq_meta[i]["name"]})
    for i in active_ineq_indices:
        columns.append(jac_ineq[i, :])
        labels.append({"type": "ineq", "index": int(i), "name": bundle.ineq_meta[i]["name"]})
    for bound in active_bounds:
        column = np.zeros_like(grad)
        if bound["side"] == "lower":
            column[int(bound["index"])] = -1.0
        else:
            column[int(bound["index"])] = 1.0
        columns.append(column)
        labels.append(
            {
                "type": "bound",
                "index": int(bound["index"]),
                "name": f"{bound['variable']}_{bound['side']}",
                "bound": bound,
            }
        )

    eq_multipliers = np.zeros(eq_values.size, dtype=float)
    ineq_multipliers = np.zeros(ineq_values.size, dtype=float)
    bound_entries = [dict(item, recovered_multiplier=0.0) for item in active_bounds]
    if not columns:
        return {
            "eq_multipliers": eq_multipliers,
            "ineq_multipliers": ineq_multipliers,
            "active_bounds": bound_entries,
            "quality": {
                "status": "no_active_operators",
                "stationarity_l2": float(np.linalg.norm(grad)),
                "stationarity_inf": float(np.max(np.abs(grad))) if grad.size else 0.0,
            },
            "top_eq": [],
            "top_ineq": [],
            "top_bounds": [],
        }

    A = np.column_stack(columns)
    column_norms = np.maximum(np.linalg.norm(A, axis=0), 1.0)
    A_scaled = A / column_norms.reshape(1, -1)
    try:
        solution_scaled, residuals, rank, singular_values = np.linalg.lstsq(A_scaled, -grad, rcond=None)
    except np.linalg.LinAlgError:
        solution_scaled = np.zeros(A.shape[1], dtype=float)
        residuals = np.asarray([], dtype=float)
        rank = 0
        singular_values = np.asarray([], dtype=float)
    multipliers = solution_scaled / column_norms
    stationarity = grad + A @ multipliers

    negative_duals = []
    for value, label in zip(multipliers, labels, strict=True):
        if label["type"] in {"ineq", "bound"} and float(value) < -1e-8:
            negative_duals.append({"name": label["name"], "type": label["type"], "value": float(value)})
        if label["type"] == "eq":
            eq_multipliers[int(label["index"])] = float(value)
        elif label["type"] == "ineq":
            ineq_multipliers[int(label["index"])] = float(value)
        else:
            for entry in bound_entries:
                if entry["variable"] == label["bound"]["variable"] and entry["side"] == label["bound"]["side"]:
                    entry["recovered_multiplier"] = float(value)
                    break

    active_ineq_complementarity = [
        abs(float(ineq_multipliers[i]) * float(ineq_values[i]))
        for i in active_ineq_indices
    ]
    bound_complementarity = [
        abs(float(entry["recovered_multiplier"]) * float(entry["constraint_value"]))
        for entry in bound_entries
    ]
    condition_number = (
        float(singular_values[0] / singular_values[-1])
        if len(singular_values) > 0 and float(singular_values[-1]) > 0.0
        else float("inf")
    )

    top_eq = sorted(
        [
            {
                "name": bundle.eq_meta[i]["name"],
                "family": bundle.eq_meta[i]["family"],
                "residual": float(eq_values[i]),
                "lambda": float(eq_multipliers[i]),
            }
            for i in range(eq_values.size)
        ],
        key=lambda item: abs(item["lambda"]),
        reverse=True,
    )[:25]
    top_ineq = sorted(
        [
            {
                "name": bundle.ineq_meta[i]["name"],
                "family": bundle.ineq_meta[i]["family"],
                "residual": float(ineq_values[i]),
                "lambda": float(ineq_multipliers[i]),
            }
            for i in active_ineq_indices
        ],
        key=lambda item: abs(item["lambda"]),
        reverse=True,
    )[:25]
    top_bounds = sorted(
        bound_entries,
        key=lambda item: abs(float(item["recovered_multiplier"])),
        reverse=True,
    )[:25]

    return {
        "eq_multipliers": eq_multipliers,
        "ineq_multipliers": ineq_multipliers,
        "active_bounds": bound_entries,
        "quality": {
            "status": "least_squares_recovered_at_fixed_curve",
            "stationarity_l2": float(np.linalg.norm(stationarity)),
            "stationarity_inf": float(np.max(np.abs(stationarity))) if stationarity.size else 0.0,
            "gradient_l2": float(np.linalg.norm(grad)),
            "gradient_inf": float(np.max(np.abs(grad))) if grad.size else 0.0,
            "least_squares_rank": int(rank),
            "least_squares_residual_sum": float(np.sum(residuals)) if np.asarray(residuals).size else 0.0,
            "condition_number_column_scaled": condition_number,
            "active_inequality_count": int(len(active_ineq_indices)),
            "active_bound_count": int(len(bound_entries)),
            "equality_count": int(eq_values.size),
            "dual_sign_violation_count": int(len(negative_duals)),
            "dual_sign_violations": negative_duals[:25],
            "max_active_ineq_complementarity": float(max(active_ineq_complementarity, default=0.0)),
            "max_bound_complementarity": float(max(bound_complementarity, default=0.0)),
            "note": (
                "Multipliers are recovered by fixed-curve stationarity least squares; "
                "the script does not re-optimize the curve."
            ),
        },
        "top_eq": top_eq,
        "top_ineq": top_ineq,
        "top_bounds": top_bounds,
    }


def _top_by_abs(entries: list[dict[str, Any]], key: str, limit: int) -> list[dict[str, Any]]:
    return sorted(entries, key=lambda item: abs(float(item[key])), reverse=True)[: int(limit)]


def _top_by_residual(entries: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    return sorted(entries, key=lambda item: float(item["residual"]), reverse=True)[: int(limit)]


def analyze_hybrid_summary(
    summary_path: str | Path,
    *,
    active_tol: float = 1e-6,
    violation_tol: float = 1e-9,
    bound_abs_tol: float = 1e-8,
    bound_rel_tol: float = 1e-6,
) -> dict[str, Any]:
    hybrid = _load_json(summary_path)
    seed = _seed_from_hybrid_payload(hybrid)
    objective_profile = _objective_profile_from_summary(hybrid)
    decision = _decision_from_hybrid(hybrid)
    curve = _load_curve_snapshot(
        summary_path=summary_path,
        hybrid=hybrid,
        seed=seed,
        decision_vector=decision,
    )
    scaling = _state_scaling_from_curve(curve)
    residual_scales = _residual_scales_from_curve(seed, curve)
    variables = _trajectory_variables_from_curve(curve)
    curve_result = _build_coarse_result_from_state_trajectory(
        baseline=seed,
        n_intervals=curve.n_intervals,
        variables=variables,
        check_equalities=True,
        residual_scales=residual_scales,
        objective_profile=objective_profile,
    )
    bundle = _build_operator_bundle(
        seed=seed,
        curve=curve,
        scaling=scaling,
        residual_scales=residual_scales,
        objective_profile=objective_profile,
    )
    x0 = _curve_x0(curve)
    values = _evaluate_operator_bundle(bundle, x0)
    recovered = _recover_multipliers(
        values=values,
        bundle=bundle,
        x0=x0,
        active_tol=active_tol,
        bound_abs_tol=bound_abs_tol,
        bound_rel_tol=bound_rel_tol,
    )

    ineq_entries = _format_inequality_entries(
        meta=bundle.ineq_meta,
        residuals=np.asarray(values["ineq"], dtype=float),
        measured=np.asarray(values["measured_ineq"], dtype=float),
        multipliers=np.asarray(recovered["ineq_multipliers"], dtype=float),
        active_tol=active_tol,
        violation_tol=violation_tol,
    )
    eq_entries = _format_equality_entries(
        meta=bundle.eq_meta,
        residuals=np.asarray(values["eq"], dtype=float),
        multipliers=np.asarray(recovered["eq_multipliers"], dtype=float),
    )
    decision_box = _decision_bounds(seed)
    decision_position = {
        key: _fraction_to_box(float(decision[key]), *decision_box[key])
        for key in _DECISION_NAMES
    }

    initial_n_from_decision = math.exp(float(decision["log_n_p_in"]))
    initial_Te_from_decision = float(decision["T_e_in"])
    curve_consistency = {
        "initial_n_p_from_decision": float(initial_n_from_decision),
        "initial_n_p_from_curve": float(curve.n_p_nodes[0]),
        "initial_n_p_relative_mismatch": float(
            abs(initial_n_from_decision - float(curve.n_p_nodes[0])) / max(abs(initial_n_from_decision), 1.0)
        ),
        "initial_T_e_from_decision": float(initial_Te_from_decision),
        "initial_T_e_from_curve": float(curve.T_e_nodes[0]),
        "initial_T_e_relative_mismatch": float(
            abs(initial_Te_from_decision - float(curve.T_e_nodes[0])) / max(abs(initial_Te_from_decision), 1.0)
        ),
        "x_end": float(curve.x_nodes[-1]),
        "baseline_L": float(seed.L),
    }

    return _jsonify(
        {
            "summary_path": str(Path(summary_path).resolve()),
            "analysis_mode": "fixed_curve_operator_kkt_recovery",
            "curve_source": curve.source,
            "objective_profile": objective_profile,
            "working_fluid": seed.working_fluid.to_dict(),
            "operator_formulation": "fullspace_backward_euler_operators_evaluated_at_fixed_curve",
            "optimizer_rerun": False,
            "n_intervals": int(curve.n_intervals),
            "objective": {
                "score": float(curve_result.objective_score),
                "objective_to_minimize": float(values["objective_to_minimize"]),
                "raw_design_score": float(curve_result.diagnostics.get("raw_design_score", float("nan"))),
                "velikhov_margin_penalty": float(curve_result.diagnostics.get("velikhov_margin_penalty", float("nan"))),
            },
            "curve_diagnostics": curve_result.diagnostics,
            "curve_consistency": curve_consistency,
            "decision_position_in_current_box": decision_position,
            "residual_summary": {
                "max_ineq_residual": float(np.max(np.asarray(values["ineq"], dtype=float))) if len(ineq_entries) else 0.0,
                "max_eq_abs_residual": float(np.max(np.abs(np.asarray(values["eq"], dtype=float)))) if len(eq_entries) else 0.0,
                "active_inequality_count": int(sum(bool(item["active"]) for item in ineq_entries)),
                "violated_inequality_count": int(sum(bool(item["violated"]) for item in ineq_entries)),
                "equality_count": int(len(eq_entries)),
                "inequality_count": int(len(ineq_entries)),
                "active_tolerance": float(active_tol),
                "violation_tolerance": float(violation_tol),
                "bound_abs_tolerance": float(bound_abs_tol),
                "bound_rel_tolerance": float(bound_rel_tol),
            },
            "active_inequality_operators": [item for item in ineq_entries if item["active"]],
            "most_loaded_inequality_operators": _top_by_abs(ineq_entries, "recovered_multiplier", 25),
            "closest_inequality_operators": _top_by_residual(ineq_entries, 25),
            "largest_equality_residuals": _top_by_abs(eq_entries, "residual", 25),
            "most_loaded_equality_operators": _top_by_abs(eq_entries, "recovered_multiplier", 25),
            "all_inequality_operators": ineq_entries,
            "all_equality_operators": eq_entries,
            "recovered_multipliers": {
                "quality": recovered["quality"],
                "top_equality_multipliers": recovered["top_eq"],
                "top_inequality_multipliers": recovered["top_ineq"],
                "top_bound_multipliers": recovered["top_bounds"],
                "active_bounds": recovered["active_bounds"],
            },
            "notes": {
                "kkt_semantics": (
                    "This is fixed-curve operator analysis. Multipliers are recovered from stationarity "
                    "least squares at the supplied curve; no optimizer is called and the curve is not moved."
                ),
                "operator_goal": (
                    "Use active_inequality_operators, largest_equality_residuals, and recovered_multipliers "
                    "to see which local KKT operators explain the curve."
                ),
            },
        }
    )


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Analyze fixed-curve KKT operators for a v6_maingo_casadi hybrid summary."
    )
    p.add_argument("summary", type=str, help="path to hybrid_summary.json")
    p.add_argument(
        "--out",
        type=str,
        default="",
        help="optional output path for the analysis JSON; defaults next to the input summary",
    )
    p.add_argument(
        "--active-tol",
        type=float,
        default=1e-6,
        help="inequality residual tolerance for active-set selection, using g(x) >= -tol",
    )
    p.add_argument(
        "--violation-tol",
        type=float,
        default=1e-9,
        help="positive inequality residual tolerance for reporting violations",
    )
    p.add_argument(
        "--bound-abs-tol",
        type=float,
        default=1e-8,
        help="absolute tolerance for decision-bound activity",
    )
    p.add_argument(
        "--bound-rel-tol",
        type=float,
        default=1e-6,
        help="relative tolerance for decision-bound activity",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    analysis = analyze_hybrid_summary(
        args.summary,
        active_tol=float(args.active_tol),
        violation_tol=float(args.violation_tol),
        bound_abs_tol=float(args.bound_abs_tol),
        bound_rel_tol=float(args.bound_rel_tol),
    )
    out_path = Path(args.out) if args.out else Path(args.summary).with_name("hybrid_analysis.json")
    out_path.write_text(json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(analysis, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
