from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from v6_global_marginal.global_postprocess_v6 import compute_design_value_terms

from v6_maingo_casadi.constants import _EPS, _G_HARD_MARGIN, _TP_MIN, OBJECTIVE_PROFILE_LAB_POC_V2
from v6_maingo_casadi.geometry import SplineAreaDesign
from v6_maingo_casadi.models import BaselineSeed, CoarseProfileResult, InletDesign
from v6_maingo_casadi.numerics import (
    _min_op,
    _ops_for_maingo,
    _ops_for_numeric,
    _reduce_min,
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
    _mach_area_closure_generic,
    _mach_design_nodes_generic,
    _reduce_max,
    residual_scales_from_profile,
    rollout_reduced_mach_generic,
)


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
        jacobian_mode: str = "analytic",
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
        self._jacobian_mode = str(jacobian_mode)
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
            "jacobian_mode": self._jacobian_mode,
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
            jacobian_mode=self._jacobian_mode,
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
            jacobian_mode=self._jacobian_mode,
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
