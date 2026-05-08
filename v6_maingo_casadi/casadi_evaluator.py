from __future__ import annotations

import casadi as ca
import numpy as np

from v6_global_marginal.global_postprocess_v6 import compute_design_value_terms

from .constants import _EPS, _G_HARD_MARGIN, _TP_MIN, OBJECTIVE_PROFILE_LAB_POC_V2
from .geometry import SplineAreaDesign
from .models import BaselineSeed, CoarseProfileResult, InletDesign
from .numerics import _min_op, _ops_for_casadi, _reduce_min, _velikhov_margin_penalty
from .physics import _design_score_generic, _evaluate_midpoint_closures, _rk4_rollout_generic
from .profiles import _augment_value_terms_with_hall_diagnostics, _normalize_objective_profile, _value_profile_dict

def _make_casadi_rollout_function(
    *,
    baseline: BaselineSeed,
    n_intervals: int,
    objective_profile: str = OBJECTIVE_PROFILE_LAB_POC_V2,
) -> ca.Function:
    objective_profile = _normalize_objective_profile(objective_profile)
    fluid = baseline.working_fluid
    ops = _ops_for_casadi()
    vars_sym = ca.MX.sym("vars", 8)
    log_n_p_in, T_e_in, Z_in, I_0, log_seed_fraction, a1, a2, a3 = ca.vertsplit(vars_sym)
    area_design = SplineAreaDesign(a1=a1, a2=a2, a3=a3)
    rollout = _rk4_rollout_generic(
        ops=ops,
        n_p_in=ca.exp(log_n_p_in),
        T_e_in=T_e_in,
        Z_in=Z_in,
        I_0=I_0,
        seed_fraction=ca.exp(log_seed_fraction),
        area_design=area_design,
        B=float(baseline.B),
        length=float(baseline.L),
        n_intervals=int(n_intervals),
        area_scale=float(baseline.area_scale_m2),
        working_fluid=fluid,
    )
    closures = rollout["closures"]
    x_nodes = np.asarray(rollout["x"], dtype=float)
    power_density_nodes = []
    for idx, closure in enumerate(closures):
        power_density_nodes.append(-rollout["A"][idx] * closure["J_x"] * closure["E_x"] / 1e8)
    raw_design_score = _design_score_generic(
        ops=ops,
        outlet_T_e=rollout["T_e"][-1],
        outlet_T_p=closures[-1]["T_p"],
        outlet_n_p=rollout["n_p"][-1],
        outlet_n_e=closures[-1]["n_e"],
        inlet_T_e=rollout["inlet"]["T_e"],
        inlet_T_p=rollout["inlet"]["T_p"],
        inlet_mach=rollout["inlet"]["mach"],
        power_density_nodes=power_density_nodes,
        x_nodes=x_nodes,
        seed_fraction=ca.exp(log_seed_fraction),
        B=float(baseline.B),
        length=float(baseline.L),
        objective_profile=objective_profile,
        inlet_n_p=rollout["inlet"]["n_p"],
        inlet_n_e=rollout["inlet"]["n_e"],
        inlet_v=rollout["inlet"]["v_in"],
        inlet_A=float(baseline.area_scale_m2),
        working_fluid=fluid,
    )
    _, midpoint_closures = _evaluate_midpoint_closures(
        ops=ops,
        area_design=area_design,
        length=float(baseline.L),
        n_intervals=int(n_intervals),
        n_p_nodes=rollout["n_p"],
        T_e_nodes=rollout["T_e"],
        dot_N=rollout["inlet"]["dot_N"],
        I_0=I_0,
        seed_fraction=ca.exp(log_seed_fraction),
        B=float(baseline.B),
        area_scale=float(baseline.area_scale_m2),
        working_fluid=fluid,
    )
    min_g_nodes = _reduce_min(ops, [item["G"] for item in closures])
    min_g_midpoints = _reduce_min(ops, [item["G"] for item in midpoint_closures])
    min_g_all = _min_op(ops, min_g_nodes, min_g_midpoints)
    velikhov_penalty = _velikhov_margin_penalty(ops, min_g_all)
    design_score = raw_design_score - velikhov_penalty
    g_list = []
    for sigma in rollout["sigma_logA"]:
        g_list.append(ca.fabs(sigma) - float(baseline.schedule[0]["max_abs_dlogA_dx"]))
    tp_min = float(baseline.schedule[0].get("tp_min", _TP_MIN))
    mach_min = float(baseline.schedule[0].get("mach_min", 0.0) or 0.0)
    for closure in closures:
        g_list.append(tp_min - closure["T_p"])
        g_list.append(float(_G_HARD_MARGIN) - closure["G"])
        if mach_min > 0.0:
            g_list.append(mach_min - closure["mach"])
    for closure_mid in midpoint_closures:
        g_list.append(float(_G_HARD_MARGIN) - closure_mid["G"])
    return ca.Function(
        "hybrid_coarse_rollout",
        [vars_sym],
        [
            ca.vertcat(*rollout["n_p"]),
            ca.vertcat(*rollout["T_e"]),
            ca.vertcat(*rollout["A"]),
            ca.vertcat(*rollout["sigma_logA"]),
            ca.vertcat(*[item["T_p"] for item in closures]),
            ca.vertcat(*[item["v_p"] for item in closures]),
            ca.vertcat(*[item["n_e"] for item in closures]),
            ca.vertcat(*[item["beta"] for item in closures]),
            ca.vertcat(*[item["eta"] for item in closures]),
            ca.vertcat(*[item["Z"] for item in closures]),
            ca.vertcat(*[item["J_x"] for item in closures]),
            ca.vertcat(*[item["J_y"] for item in closures]),
            ca.vertcat(*[item["E_x"] for item in closures]),
            ca.vertcat(*[item["mach"] for item in closures]),
            ca.vertcat(*[item["G"] for item in closures]),
            ca.vertcat(*[item["G"] for item in midpoint_closures]),
            design_score,
            raw_design_score,
            velikhov_penalty,
            ca.vertcat(*g_list),
            ca.vertcat(
                rollout["inlet"]["n_p"],
                rollout["inlet"]["T_e"],
                rollout["inlet"]["T_p"],
                rollout["inlet"]["Z"],
                rollout["inlet"]["I_0"],
                rollout["inlet"]["dot_N"],
                rollout["inlet"]["v_in"],
                ca.exp(log_seed_fraction),
                rollout["inlet"]["mach"],
                rollout["inlet"]["G"],
                min_g_all,
            ),
        ],
    )


class CasadiCoarseEvaluator:
    def __init__(
        self,
        *,
        baseline: BaselineSeed,
        n_intervals: int,
        objective_profile: str = OBJECTIVE_PROFILE_LAB_POC_V2,
    ):
        self.baseline = baseline
        self.n_intervals = int(n_intervals)
        self.objective_profile = _normalize_objective_profile(objective_profile)
        self._rollout_fn = _make_casadi_rollout_function(
            baseline=baseline,
            n_intervals=int(n_intervals),
            objective_profile=self.objective_profile,
        )

    def evaluate(self, decision_vector: dict[str, float]) -> CoarseProfileResult:
        vars_array = np.array(
            [
                float(decision_vector["log_n_p_in"]),
                float(decision_vector["T_e_in"]),
                float(decision_vector["Z_in"]),
                float(decision_vector["I_0"]),
                float(decision_vector["log_seed_fraction"]),
                float(decision_vector["a1"]),
                float(decision_vector["a2"]),
                float(decision_vector["a3"]),
            ],
            dtype=float,
        )
        (
            n_p,
            T_e,
            A,
            sigma,
            T_p,
            v_p,
            n_e,
            beta,
            eta,
            Z,
            J_x,
            J_y,
            E_x,
            mach,
            G,
            G_mid,
            design_score,
            raw_design_score,
            velikhov_penalty,
            g_vals,
            inlet_vals,
        ) = self._rollout_fn(vars_array)
        x = np.linspace(0.0, float(self.baseline.L), self.n_intervals + 1, dtype=float)
        decision = dict(decision_vector)
        inlet = InletDesign(
            n_p=float(inlet_vals[0]),
            T_e=float(inlet_vals[1]),
            T_p=float(inlet_vals[2]),
            Z=float(inlet_vals[3]),
            I_0=float(inlet_vals[4]),
            dot_N=float(inlet_vals[5]),
            v_in=float(inlet_vals[6]),
            seed_fraction=float(inlet_vals[7]),
            mach=float(inlet_vals[8]),
            velikhov_margin=float(inlet_vals[9]),
            A_in=float(self.baseline.area_scale_m2),
        )
        arrays = {
            "x": x,
            "n_p": np.asarray(n_p, dtype=float).reshape(-1),
            "T_e": np.asarray(T_e, dtype=float).reshape(-1),
            "A": np.asarray(A, dtype=float).reshape(-1),
            "sigma_logA": np.asarray(sigma, dtype=float).reshape(-1),
            "T_p": np.asarray(T_p, dtype=float).reshape(-1),
            "v_p": np.asarray(v_p, dtype=float).reshape(-1),
            "n_e": np.asarray(n_e, dtype=float).reshape(-1),
            "beta": np.asarray(beta, dtype=float).reshape(-1),
            "eta": np.asarray(eta, dtype=float).reshape(-1),
            "Z": np.asarray(Z, dtype=float).reshape(-1),
            "J_x": np.asarray(J_x, dtype=float).reshape(-1),
            "J_y": np.asarray(J_y, dtype=float).reshape(-1),
            "E_x": np.asarray(E_x, dtype=float).reshape(-1),
            "mach": np.asarray(mach, dtype=float).reshape(-1),
            "velikhov_margin": np.asarray(G, dtype=float).reshape(-1),
            "velikhov_margin_midpoint": np.asarray(G_mid, dtype=float).reshape(-1),
        }
        g_arr = np.asarray(g_vals, dtype=float).reshape(-1)
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
            B=float(self.baseline.B),
            seed_fraction=float(inlet.seed_fraction),
            v_p=arrays["v_p"],
            heavy_particle_mass_kg=float(self.baseline.working_fluid.heavy_particle_mass_kg),
        )
        value_terms_dict = value_terms.to_dict()
        _augment_value_terms_with_hall_diagnostics(
            value_terms_dict,
            x=arrays["x"],
            E_x=arrays["E_x"],
            I_0=float(inlet.I_0),
        )
        value_terms_dict["mass_flow_rate_kg_s"] = float(inlet.dot_N) * float(
            self.baseline.working_fluid.heavy_particle_mass_kg
        )
        value_terms_dict["inlet_area_m2"] = float(arrays["A"][0])
        value_terms_dict["outlet_area_m2"] = float(arrays["A"][-1])
        value_terms_dict["outlet_to_inlet_area_ratio"] = float(arrays["A"][-1]) / max(float(arrays["A"][0]), _EPS)
        value_terms_dict["velikhov_margin_penalty"] = float(velikhov_penalty)
        value_terms_dict["raw_design_score"] = float(raw_design_score)
        value_terms_dict["min_velikhov_margin_midpoint"] = float(np.min(arrays["velikhov_margin_midpoint"]))
        value_profile = _value_profile_dict(
            value_terms,
            objective_profile=self.objective_profile,
        )
        value_profile["terms"] = dict(value_terms_dict)
        diagnostics = {
            "n_intervals": int(self.n_intervals),
            "finite_profile": bool(
                all(np.all(np.isfinite(arr)) for arr in arrays.values())
                and np.all(np.isfinite(g_arr))
                and np.isfinite(float(design_score))
            ),
            "min_T_p": float(np.min(arrays["T_p"])),
            "min_velikhov_margin": float(np.min(arrays["velikhov_margin"])),
            "min_velikhov_margin_midpoint": float(np.min(arrays["velikhov_margin_midpoint"])),
            "min_velikhov_margin_all_checks": float(
                min(
                    float(np.min(arrays["velikhov_margin"])),
                    float(np.min(arrays["velikhov_margin_midpoint"])),
                )
            ),
            "min_mach": float(np.min(arrays["mach"])),
            "max_abs_sigma_logA": float(np.max(np.abs(arrays["sigma_logA"]))),
            "max_ineq_residual": float(np.max(g_arr)) if g_arr.size else 0.0,
            "constraint_count": int(g_arr.size),
            "raw_design_score": float(raw_design_score),
            "velikhov_margin_penalty": float(velikhov_penalty),
            "objective_profile": self.objective_profile,
            "acceptable": bool(np.max(g_arr) <= 1e-7) if g_arr.size else True,
        }
        return CoarseProfileResult(
            decision_vector=decision,
            inlet_design=inlet,
            area_design=SplineAreaDesign(
                a1=float(decision["a1"]),
                a2=float(decision["a2"]),
                a3=float(decision["a3"]),
            ),
            objective_score=float(design_score),
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
