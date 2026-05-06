#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from dataclasses import replace
from pathlib import Path
from typing import Any

import casadi as ca
import numpy as np

from v6_maingo_casadi.core import (
    BaselineSeed,
    SplineAreaDesign,
    _G_HARD_MARGIN,
    _MAiNGOHybridImplicitModelBase,
    _build_implicit_reference,
    _design_score_generic,
    _dynamic_system_terms,
    _evaluate_area_design_nodes,
    _evaluate_midpoint_closures,
    _import_maingopy,
    _inlet_design_generic,
    _ops_for_casadi,
    _project_implicit_trajectory,
    _velikhov_margin_penalty,
    _build_coarse_result_from_state_trajectory,
)


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _seed_from_hybrid_payload(hybrid: dict[str, Any]) -> BaselineSeed:
    baseline_payload = dict(hybrid.get("baseline_seed", {}) or {})
    baseline_summary = Path(str(baseline_payload["summary_path"]))
    seed = BaselineSeed.from_summary(baseline_summary)
    inlet_windows = baseline_payload.get("inlet_windows")
    area_design = baseline_payload.get("area_design_nominal")
    overrides = {}
    if inlet_windows:
        overrides["inlet_windows"] = {
            str(key): {
                "guess": float(value["guess"]),
                "min": float(value["min"]),
                "max": float(value["max"]),
            }
            for key, value in dict(inlet_windows).items()
        }
    if area_design:
        overrides["area_design_nominal"] = SplineAreaDesign(
            a1=float(area_design["a1"]),
            a2=float(area_design["a2"]),
            a3=float(area_design["a3"]),
        )
    for key in ("B", "L", "adaptive_bridge_count", "adaptive_bridge_max_count"):
        if key in baseline_payload:
            overrides[key] = baseline_payload[key]
    if not overrides:
        return seed
    return replace(seed, **overrides)


def _decision_bounds(seed: BaselineSeed) -> dict[str, tuple[float, float]]:
    return {
        "log_n_p_in": (
            math.log(float(seed.inlet_windows["n_p_in"]["min"])),
            math.log(float(seed.inlet_windows["n_p_in"]["max"])),
        ),
        "T_e_in": (
            float(seed.inlet_windows["T_e_in"]["min"]),
            float(seed.inlet_windows["T_e_in"]["max"]),
        ),
        "Z_in": (
            float(seed.inlet_windows["Z_in"]["min"]),
            float(seed.inlet_windows["Z_in"]["max"]),
        ),
        "I_0": (
            float(seed.inlet_windows["I_0"]["min"]),
            float(seed.inlet_windows["I_0"]["max"]),
        ),
        "log_seed_fraction": (
            math.log(float(seed.inlet_windows["seed_fraction"]["min"])),
            math.log(float(seed.inlet_windows["seed_fraction"]["max"])),
        ),
        "a1": (SplineAreaDesign.lower_bound(), SplineAreaDesign.upper_bound()),
        "a2": (SplineAreaDesign.lower_bound(), SplineAreaDesign.upper_bound()),
        "a3": (SplineAreaDesign.lower_bound(), SplineAreaDesign.upper_bound()),
    }


def _fraction_to_box(value: float, lower: float, upper: float) -> dict[str, float]:
    width = max(float(upper) - float(lower), 1e-30)
    return {
        "fraction": float((float(value) - float(lower)) / width),
        "dist_lo": float(float(value) - float(lower)),
        "dist_hi": float(float(upper) - float(value)),
    }


def _build_local_kkt_summary(
    *,
    seed: BaselineSeed,
    model: _MAiNGOHybridImplicitModelBase,
    decision_vector: dict[str, float],
) -> dict[str, Any]:
    projected = _project_implicit_trajectory(
        baseline=seed,
        n_intervals=model._n_intervals,
        decision_vector=decision_vector,
        residual_scales=model._residual_scales,
        initial_guess=model._reference_variables,
    )
    x0 = [
        float(decision_vector["log_n_p_in"]),
        float(decision_vector["T_e_in"]),
        float(decision_vector["Z_in"]),
        float(decision_vector["I_0"]),
        float(decision_vector["log_seed_fraction"]),
        float(decision_vector["a1"]),
        float(decision_vector["a2"]),
        float(decision_vector["a3"]),
    ]
    x0 += model._trajectory_scaling.encode_n_p_tail(projected.n_p_nodes[1:])
    x0 += model._trajectory_scaling.encode_T_e_tail(projected.T_e_nodes[1:])
    x0 += model._trajectory_scaling.encode_dn_dx(projected.dn_dx)
    x0 += model._trajectory_scaling.encode_dTe_dx(projected.dTe_dx)

    ni = int(model._n_intervals)
    x = ca.MX.sym("x", model.total_variables)
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
        float(model._trajectory_scaling.n_p_center[k]) + float(model._trajectory_scaling.n_p_scale[k]) * n_hat[k]
        for k in range(ni)
    ]
    te_tail = [
        float(model._trajectory_scaling.T_e_center[k]) + float(model._trajectory_scaling.T_e_scale[k]) * te_hat[k]
        for k in range(ni)
    ]
    dn_dx = [
        float(model._trajectory_scaling.dn_dx_center[k]) + float(model._trajectory_scaling.dn_dx_scale[k]) * dn_hat[k]
        for k in range(ni)
    ]
    dte_dx = [
        float(model._trajectory_scaling.dTe_dx_center[k]) + float(model._trajectory_scaling.dTe_dx_scale[k]) * dte_hat[k]
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
    )
    area_nodes = _evaluate_area_design_nodes(
        ops=ops,
        area_design=area_design,
        length=float(seed.L),
        n_intervals=ni,
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
    )

    closures = []
    terms_by_node = []
    power_density_nodes = []
    ineq = []
    ineq_names = []
    eq = []
    eq_names = []
    sigma_max = float(seed.schedule[0]["max_abs_dlogA_dx"])
    tp_min = float(seed.schedule[0].get("tp_min", 1.0))
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
        )
        closures.append(closure)
        terms_by_node.append(terms)
        power_density_nodes.append(-A * closure["J_x"] * closure["E_x"] / 1e8)
        ineq.append(ca.fabs(sigma) - sigma_max)
        ineq_names.append(f"sigma_node_{i}")
        ineq.append(tp_min - closure["T_p"])
        ineq_names.append(f"Tp_node_{i}")
        ineq.append(float(_G_HARD_MARGIN) - closure["G"])
        ineq_names.append(f"G_node_{i}")
        if mach_min > 0.0:
            ineq.append(mach_min - closure["mach"])
            ineq_names.append(f"Mach_node_{i}")
    for i, closure_mid in enumerate(midpoint_closures):
        ineq.append(float(_G_HARD_MARGIN) - closure_mid["G"])
        ineq_names.append(f"G_mid_{i}")

    dx = float(seed.L) / ni
    for k in range(ni):
        terms = terms_by_node[k + 1]
        eq.append((n_nodes[k + 1] - n_nodes[k] - dx * dn_dx[k]) / float(model._residual_scales.step_n[k]))
        eq_names.append(f"step_n_{k}")
        eq.append((te_nodes[k + 1] - te_nodes[k] - dx * dte_dx[k]) / float(model._residual_scales.step_Te[k]))
        eq_names.append(f"step_Te_{k}")
        eq.append(
            (terms["M11"] * dn_dx[k] + terms["M12"] * dte_dx[k] - terms["rhs_m"])
            / float(model._residual_scales.momentum[k])
        )
        eq_names.append(f"momentum_{k}")
        eq.append(
            (terms["E11"] * dn_dx[k] + terms["E12"] * dte_dx[k] - terms["rhs_e"])
            / float(model._residual_scales.energy[k])
        )
        eq_names.append(f"energy_{k}")

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
    )
    min_g_nodes = ca.mmin(ca.vertcat(*[item["G"] for item in closures]))
    min_g_midpoints = ca.mmin(ca.vertcat(*[item["G"] for item in midpoint_closures]))
    min_g_all = ca.fmin(min_g_nodes, min_g_midpoints)
    objective = -(raw_score - _velikhov_margin_penalty(ops, min_g_all))

    g = ca.vertcat(*(ineq + eq))
    lbg = [-ca.inf] * len(ineq) + [0.0] * len(eq)
    ubg = [0.0] * len(ineq) + [0.0] * len(eq)
    lbx = [spec[0] for spec in model._variable_specs]
    ubx = [spec[1] for spec in model._variable_specs]
    solver = ca.nlpsol(
        "coarse_kkt",
        "ipopt",
        {"x": x, "f": objective, "g": g},
        {
            "ipopt.print_level": 0,
            "print_time": 0,
            "ipopt.sb": "yes",
            "ipopt.max_iter": 400,
            "ipopt.tol": 1e-10,
            "ipopt.acceptable_tol": 1e-8,
        },
    )
    sol = solver(x0=np.asarray(x0, dtype=float), lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
    x_sol = np.asarray(sol["x"], dtype=float).reshape(-1)
    lam_g = np.asarray(sol["lam_g"], dtype=float).reshape(-1)
    lam_x = np.asarray(sol["lam_x"], dtype=float).reshape(-1)
    g_val = np.asarray(ca.Function("g_eval", [x], [g])(x_sol), dtype=float).reshape(-1)
    local_solution = model.decode_solution_point(x_sol)
    local_result = _build_coarse_result_from_state_trajectory(
        baseline=seed,
        n_intervals=ni,
        variables=local_solution,
        check_equalities=True,
        residual_scales=model._residual_scales,
    )

    top_ineq = sorted(
        [
            {
                "name": name,
                "lambda": float(lam_g[i]),
                "residual": float(g_val[i]),
            }
            for i, name in enumerate(ineq_names)
        ],
        key=lambda item: abs(item["lambda"]),
        reverse=True,
    )[:20]
    top_bounds = sorted(
        [
            {
                "name": model._variable_specs[i][2],
                "value": float(x_sol[i]),
                "lambda": float(lam_x[i]),
                "dist_lo": float(x_sol[i] - lbx[i]),
                "dist_hi": float(ubx[i] - x_sol[i]),
            }
            for i in range(len(model._variable_specs))
        ],
        key=lambda item: abs(item["lambda"]),
        reverse=True,
    )[:20]
    return {
        "return_status": str(solver.stats().get("return_status", "")),
        "objective_score": float(local_result.objective_score),
        "diagnostics": local_result.diagnostics,
        "decision_vector": local_solution.decision_vector,
        "top_inequality_multipliers": top_ineq,
        "top_bound_multipliers": top_bounds,
    }


def analyze_hybrid_summary(summary_path: str | Path) -> dict[str, Any]:
    hybrid = _load_json(summary_path)
    seed = _seed_from_hybrid_payload(hybrid)
    model = _MAiNGOHybridImplicitModelBase(
        baseline=seed,
        n_intervals=40,
        maingopy_module=_import_maingopy(),
    )
    reference_variables, reference_result, _ = _build_implicit_reference(baseline=seed, n_intervals=40)
    _ = reference_variables
    solver_status = dict(hybrid.get("maingo_status", {}))
    handoff_decision = dict(
        solver_status.get("handoff_solution_point")
        or hybrid.get("maingo_best", {}).get("decision_vector", {})
    )
    maingo_best = dict(hybrid.get("maingo_best", {}))
    maingo_value_profile = dict(maingo_best.get("value_profile", {}))
    baseline_profile = dict(reference_result.value_profile)

    contribution_delta = {}
    best_contrib = dict(maingo_value_profile.get("contributions", {}))
    baseline_contrib = dict(baseline_profile.get("contributions", {}))
    for key in sorted(set(best_contrib) | set(baseline_contrib)):
        contribution_delta[key] = float(best_contrib.get(key, 0.0) - baseline_contrib.get(key, 0.0))

    decision_box = _decision_bounds(seed)
    decision_position = {
        key: _fraction_to_box(float(handoff_decision[key]), *decision_box[key])
        for key in decision_box.keys()
    }

    continuation_position = {}
    continuation = dict(hybrid.get("continuation", {}))
    final_trusted = dict(continuation.get("final_trusted_inlet_design", {}))
    handoff_bounds = dict(hybrid.get("handoff_bounds", {}))
    mapping = [
        ("n_p_in", "n_p_in"),
        ("T_e_in", "T_e_in"),
        ("Z_in", "Z_in"),
        ("I_0", "J_x_in"),
        ("seed_fraction", "seed_fraction"),
    ]
    for key_bounds, key_final in mapping:
        if key_bounds in handoff_bounds and key_final in final_trusted:
            continuation_position[key_bounds] = _fraction_to_box(
                float(final_trusted[key_final]),
                float(handoff_bounds[key_bounds]["min"]),
                float(handoff_bounds[key_bounds]["max"]),
            )

    return {
        "summary_path": str(Path(summary_path).resolve()),
        "baseline_objective_score": float(reference_result.objective_score),
        "maingo_objective_score": float(maingo_best.get("objective_score", float("nan"))),
        "objective_score_gain_vs_baseline": float(
            maingo_best.get("objective_score", 0.0) - reference_result.objective_score
        ),
        "baseline_contributions": baseline_contrib,
        "maingo_contributions": best_contrib,
        "contribution_delta_vs_baseline": contribution_delta,
        "decision_position_in_current_box": decision_position,
        "continuation_position_in_handoff_box": continuation_position,
        "kkt_local_analysis": _build_local_kkt_summary(
            seed=seed,
            model=model,
            decision_vector=handoff_decision,
        ),
        "notes": {
            "maingopy_duals_available": False,
            "comment": (
                "MAiNGO's Python API does not expose KKT multipliers directly. "
                "The KKT section is from a local CasADi/IPOPT solve of the same coarse implicit NLP."
            ),
        },
    }


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Analyze a v6_maingo_casadi hybrid summary: contributions, box activity, and local KKT."
    )
    p.add_argument("summary", type=str, help="path to hybrid_summary.json")
    p.add_argument(
        "--out",
        type=str,
        default="",
        help="optional output path for the analysis JSON; defaults next to the input summary",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    analysis = analyze_hybrid_summary(args.summary)
    out_path = Path(args.out) if args.out else Path(args.summary).with_name("hybrid_analysis.json")
    out_path.write_text(json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(analysis, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
