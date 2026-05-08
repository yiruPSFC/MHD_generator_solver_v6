from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from v6_casadi_v2.run_casadi_continuation_v2 import load_warm_profile_npz, run_continuation

from .casadi_evaluator import CasadiCoarseEvaluator
from .constants import _A_IN, _DEFAULT_BASELINE_SUMMARY, OBJECTIVE_PROFILE_LAB_POC_V2
from .implicit import _resample_profile_result, _restore_feasible_implicit_solution
from .maingo_models import _import_maingopy, _retcode_name, _safe_solver_metric
from .models import BaselineSeed, CoarseProfileResult, HybridRunResult
from .profiles import WorkingFluidProfile, _normalize_objective_profile
from .reduced_implicit import _MAiNGOHybridReducedImplicitModelBase


_PACKAGE_DIR = Path(__file__).resolve().parent


def _resolve_maingo_settings_path(path: str | Path) -> Path:
    settings_path = Path(path)
    if settings_path.exists():
        return settings_path
    candidate = _PACKAGE_DIR / "settings" / settings_path.name
    if candidate.exists():
        return candidate
    return settings_path


def _handoff_bounds_from_best(best: CoarseProfileResult) -> dict[str, dict[str, float]]:
    return {
        "n_p_in": {
            "guess": float(best.inlet_design.n_p),
            "min": float(best.inlet_design.n_p * 0.95),
            "max": float(best.inlet_design.n_p * 1.05),
        },
        "T_e_in": {
            "guess": float(best.inlet_design.T_e),
            "min": float(best.inlet_design.T_e * 0.95),
            "max": float(best.inlet_design.T_e * 1.05),
        },
        "Z_in": {
            "guess": float(best.inlet_design.Z),
            "min": float(best.inlet_design.Z * 0.90),
            "max": float(best.inlet_design.Z * 1.10),
        },
        "I_0": {
            "guess": float(best.inlet_design.I_0),
            "min": float(best.inlet_design.I_0 * 0.90),
            "max": float(best.inlet_design.I_0 * 1.10),
        },
        "seed_fraction": {
            "guess": float(best.inlet_design.seed_fraction),
            "min": float(max(best.inlet_design.seed_fraction * 0.5, 1e-13)),
            "max": float(min(best.inlet_design.seed_fraction * 2.0, 5e-2)),
        },
    }


def _write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path


def run_hybrid_maingo_casadi(
    *,
    out_dir: str | Path,
    baseline_summary_path: str | Path = _DEFAULT_BASELINE_SUMMARY,
    coarse_n_intervals: int = 40,
    handoff_n_intervals: int = 80,
    maingo_settings_path: str = "",
    maingo_max_time: float | None = None,
    coarse_model: str = "reduced_implicit",
    reduced_implicit_newton_steps: int = 10,
    critical_mode: bool = False,
    critical_residual_tolerance: float = 1e-4,
    include_rk4_benchmark: bool = True,
    objective_profile: str = OBJECTIVE_PROFILE_LAB_POC_V2,
    working_fluid_profile: str | WorkingFluidProfile | None = None,
    search_window_json: str | Path | None = None,
    skip_casadi_handoff: bool = False,
    n_p_in_lower_factor: float = 1.0,
    n_p_in_upper_factor: float = 1.0,
    T_e_in_lower_factor: float = 1.0,
    T_e_in_upper_factor: float = 1.0,
    Z_in_lower_factor: float = 1.0,
    Z_in_upper_factor: float = 1.0,
    I_0_lower_factor: float = 1.0,
    I_0_upper_factor: float = 1.0,
    seed_fraction_lower_factor: float = 1.0,
    seed_fraction_upper_factor: float = 1.0,
) -> HybridRunResult:
    maingopy = _import_maingopy()
    objective_profile = _normalize_objective_profile(objective_profile)
    coarse_model_name = str(coarse_model).strip().replace("-", "_").lower()
    if coarse_model_name in {"newton_reduced_implicit", "reduced_newton_implicit"}:
        coarse_model_name = "reduced_implicit"
    if coarse_model_name != "reduced_implicit":
        raise ValueError(
            f"unsupported coarse_model={coarse_model!r}; the production MAiNGO path is reduced_implicit. "
            "RK4 is evaluated only as a post-hoc benchmark because its explicit RHS can cross singular "
            "denominators during MAiNGO relaxation."
        )
    baseline = BaselineSeed.from_summary(baseline_summary_path)
    if working_fluid_profile is not None:
        baseline = baseline.with_working_fluid_profile(working_fluid_profile)
    search_window_path = None
    if search_window_json:
        search_window_path = Path(search_window_json).resolve()
        baseline = baseline.with_search_window_overrides(
            json.loads(search_window_path.read_text(encoding="utf-8"))
        )
    baseline = baseline.with_inlet_bound_factors(
        n_p_in_lower=float(n_p_in_lower_factor),
        n_p_in_upper=float(n_p_in_upper_factor),
        T_e_in_lower=float(T_e_in_lower_factor),
        T_e_in_upper=float(T_e_in_upper_factor),
        Z_in_lower=float(Z_in_lower_factor),
        Z_in_upper=float(Z_in_upper_factor),
        I_0_lower=float(I_0_lower_factor),
        I_0_upper=float(I_0_upper_factor),
        seed_fraction_lower=float(seed_fraction_lower_factor),
        seed_fraction_upper=float(seed_fraction_upper_factor),
    )
    out_dir_path = Path(out_dir).resolve()
    out_dir_path.mkdir(parents=True, exist_ok=True)
    model_impl = _MAiNGOHybridReducedImplicitModelBase(
        baseline=baseline,
        n_intervals=int(coarse_n_intervals),
        maingopy_module=maingopy,
        objective_profile=objective_profile,
        newton_steps=int(reduced_implicit_newton_steps),
        critical_mode=bool(critical_mode),
        critical_residual_tolerance=float(critical_residual_tolerance),
    )

    class HybridMAiNGOModel(maingopy.MAiNGOmodel):
        def __init__(self, *, impl):
            maingopy.MAiNGOmodel.__init__(self)
            self._impl = impl

        def get_variables(self):
            return self._impl.get_variables()

        def get_initial_point(self):
            return self._impl.get_initial_point()

        def evaluate(self, vars):
            return self._impl.evaluate(vars)

    model = HybridMAiNGOModel(impl=model_impl)
    solver = maingopy.MAiNGO(model)
    if maingo_settings_path:
        solver.read_settings(str(_resolve_maingo_settings_path(maingo_settings_path)))
    if maingo_max_time is not None and float(maingo_max_time) > 0.0:
        solver.set_option("maxTime", float(maingo_max_time))
    solver.set_log_file_name(str(out_dir_path / "maingo.log"))
    status = solver.solve()
    fallback_reason = None
    used_fallback_initial_point = False
    feasibility_restoration = {"used": False, "reason": None, "alpha": 1.0}
    try:
        solution_point = list(solver.get_solution_point())
        model_impl.decode_solution_point(solution_point)
    except Exception as exc:
        solution_point = list(model_impl.get_initial_point())
        fallback_reason = (
            "MAiNGO did not return a feasible solution point on this run; "
            "continuing from the baseline-connected reduced coarse seed. "
            f"Original MAiNGO error: {exc}"
        )
        used_fallback_initial_point = True
    incumbent_solution = model_impl.decode_solution_point(solution_point)
    incumbent_result = model_impl.evaluate_solution(incumbent_solution)
    incumbent_decision_vector = (
        dict(incumbent_solution.decision_vector)
        if hasattr(incumbent_solution, "decision_vector")
        else dict(incumbent_solution)
    )
    best_solution, coarse_best, feasibility_restoration = _restore_feasible_implicit_solution(
        baseline=baseline,
        n_intervals=int(coarse_n_intervals),
        reference_variables=model_impl._reference_variables,
        reference_result=model_impl._reference_profile,
        residual_scales=model_impl._residual_scales,
        candidate_variables=incumbent_solution,
        candidate_result=incumbent_result,
        objective_profile=objective_profile,
    )
    coarse_best = model_impl.evaluate_solution(best_solution)
    handoff_decision_vector = dict(best_solution.decision_vector)
    incumbent_result.diagnostics["formulation"] = str(getattr(model_impl, "formulation", coarse_model_name))
    coarse_best.diagnostics["formulation"] = str(getattr(model_impl, "formulation", coarse_model_name))
    coarse_best.diagnostics["reduced_implicit_newton_steps"] = int(reduced_implicit_newton_steps)
    if not bool(coarse_best.diagnostics.get("acceptable", False)):
        raise RuntimeError(
            "Unable to recover a coarse feasible point from the MAiNGO incumbent: "
            f"candidate max_ineq_residual={incumbent_result.diagnostics.get('max_ineq_residual')!r}, "
            f"candidate max_eq_residual={incumbent_result.diagnostics.get('max_eq_residual')!r}."
        )

    handoff_best = _resample_profile_result(
        baseline=baseline,
        result=coarse_best,
        n_intervals=int(handoff_n_intervals),
        objective_profile=objective_profile,
    )
    handoff_bounds = _handoff_bounds_from_best(handoff_best)
    rk4_benchmark = None
    if bool(include_rk4_benchmark):
        try:
            rk4_result = CasadiCoarseEvaluator(
                baseline=baseline,
                n_intervals=int(coarse_n_intervals),
                objective_profile=objective_profile,
            ).evaluate(handoff_decision_vector)
            rk4_benchmark = {
                "ok": True,
                "coarse_model": "rk4_reduced_benchmark",
                "score_delta_vs_reduced_implicit": float(rk4_result.objective_score - coarse_best.objective_score),
                "diagnostics": rk4_result.diagnostics,
                "result": rk4_result.to_summary_dict(),
            }
        except Exception as exc:
            rk4_benchmark = {
                "ok": False,
                "coarse_model": "rk4_reduced_benchmark",
                "error": str(exc),
            }

    maingo_summary_payload = {
        "solver": "maingo",
        "formulation": str(getattr(model_impl, "formulation", coarse_model_name)),
        "coarse_model": coarse_model_name,
        "objective_profile": objective_profile,
        "reduced_implicit_newton_steps": int(reduced_implicit_newton_steps),
        "critical_mode": bool(critical_mode),
        "critical_residual_tolerance": float(critical_residual_tolerance),
        "working_fluid_profile": baseline.working_fluid.to_dict(),
        "skip_casadi_handoff": bool(skip_casadi_handoff),
        "search_window_json": None if search_window_path is None else str(search_window_path),
        "baseline_seed": baseline.to_dict(),
        "search_window_expansion": {
            "n_p_in_lower_factor": float(n_p_in_lower_factor),
            "n_p_in_upper_factor": float(n_p_in_upper_factor),
            "T_e_in_lower_factor": float(T_e_in_lower_factor),
            "T_e_in_upper_factor": float(T_e_in_upper_factor),
            "Z_in_lower_factor": float(Z_in_lower_factor),
            "Z_in_upper_factor": float(Z_in_upper_factor),
            "I_0_lower_factor": float(I_0_lower_factor),
            "I_0_upper_factor": float(I_0_upper_factor),
            "seed_fraction_lower_factor": float(seed_fraction_lower_factor),
            "seed_fraction_upper_factor": float(seed_fraction_upper_factor),
        },
        "status": {
            "retcode": _retcode_name(status),
            "objective_value": _safe_solver_metric(lambda: float(solver.get_objective_value())),
            "best_lower_bound": _safe_solver_metric(lambda: float(solver.get_final_LBD())),
            "final_abs_gap": _safe_solver_metric(lambda: float(solver.get_final_abs_gap())),
            "final_rel_gap": _safe_solver_metric(lambda: float(solver.get_final_rel_gap())),
            "iterations": _safe_solver_metric(lambda: int(solver.get_iterations())),
            "cpu_solution_time_s": _safe_solver_metric(lambda: float(solver.get_cpu_solution_time())),
            "wallclock_solution_time_s": _safe_solver_metric(lambda: float(solver.get_wallclock_solution_time())),
            "solution_point": incumbent_decision_vector,
            "variable_count": int(model_impl.total_variables),
            "used_fallback_initial_point": used_fallback_initial_point,
            "fallback_reason": fallback_reason,
            "used_feasibility_restoration": bool(feasibility_restoration["used"]),
            "feasibility_restoration_reason": feasibility_restoration["reason"],
            "feasibility_restoration_alpha": float(feasibility_restoration["alpha"]),
            "handoff_solution_point": handoff_decision_vector,
            "rk4_benchmark_ok": None if rk4_benchmark is None else bool(rk4_benchmark.get("ok", False)),
            "rk4_benchmark": rk4_benchmark,
        },
        "model_metadata": model_impl.summary_metadata(),
        "rk4_benchmark": rk4_benchmark,
        "incumbent_diagnostics": incumbent_result.diagnostics,
        "coarse_best": coarse_best.to_summary_dict(),
        "handoff_bounds": handoff_bounds,
    }
    maingo_summary_path = _write_json(out_dir_path / "maingo_summary.json", maingo_summary_payload)

    maingo_best_profile_path = out_dir_path / "maingo_best_profile.npz"
    np.savez(maingo_best_profile_path, **handoff_best.to_npz_payload())

    continuation_out_dir = out_dir_path / "continuation"
    if bool(skip_casadi_handoff):
        continuation_payload = {
            "ok": False,
            "skipped": True,
            "reason": "skip_casadi_handoff",
            "summary_path": None,
        }
    elif not np.isclose(float(baseline.area_scale_m2), float(_A_IN), rtol=1e-12, atol=1e-15):
        continuation_payload = {
            "ok": False,
            "skipped": True,
            "reason": "physical_area_scale_not_supported_by_v6_casadi_v2_handoff",
            "summary_path": None,
            "area_scale_m2": float(baseline.area_scale_m2),
            "handoff_convention": (
                "v6_maingo_casadi stores physical A and total I_0, while "
                "v6_casadi_v2 continuation fixes A_in=1 and names the inlet intensity J_x_in."
            ),
        }
    else:
        warm_profile = load_warm_profile_npz(
            maingo_best_profile_path,
            n_p_in_guess=float(handoff_bounds["n_p_in"]["guess"]),
            n_p_in_min=float(handoff_bounds["n_p_in"]["min"]),
            n_p_in_max=float(handoff_bounds["n_p_in"]["max"]),
            T_e_in_guess=float(handoff_bounds["T_e_in"]["guess"]),
            T_e_in_min=float(handoff_bounds["T_e_in"]["min"]),
            T_e_in_max=float(handoff_bounds["T_e_in"]["max"]),
            Z_in_guess=float(handoff_bounds["Z_in"]["guess"]),
            Z_in_min=float(handoff_bounds["Z_in"]["min"]),
            Z_in_max=float(handoff_bounds["Z_in"]["max"]),
            J_x_in_guess=float(handoff_bounds["I_0"]["guess"]),
            J_x_in_min=float(handoff_bounds["I_0"]["min"]),
            J_x_in_max=float(handoff_bounds["I_0"]["max"]),
            seed_fraction_guess=float(handoff_bounds["seed_fraction"]["guess"]),
            seed_fraction_min=float(handoff_bounds["seed_fraction"]["min"]),
            seed_fraction_max=float(handoff_bounds["seed_fraction"]["max"]),
            B=float(baseline.B),
        )
        continuation_payload = run_continuation(
            n_p_in_guess=float(handoff_bounds["n_p_in"]["guess"]),
            n_p_in_min=float(handoff_bounds["n_p_in"]["min"]),
            n_p_in_max=float(handoff_bounds["n_p_in"]["max"]),
            T_e_in_guess=float(handoff_bounds["T_e_in"]["guess"]),
            T_e_in_min=float(handoff_bounds["T_e_in"]["min"]),
            T_e_in_max=float(handoff_bounds["T_e_in"]["max"]),
            Z_in_guess=float(handoff_bounds["Z_in"]["guess"]),
            Z_in_min=float(handoff_bounds["Z_in"]["min"]),
            Z_in_max=float(handoff_bounds["Z_in"]["max"]),
            J_x_in_guess=float(handoff_bounds["I_0"]["guess"]),
            J_x_in_min=float(handoff_bounds["I_0"]["min"]),
            J_x_in_max=float(handoff_bounds["I_0"]["max"]),
            seed_fraction_guess=float(handoff_bounds["seed_fraction"]["guess"]),
            seed_fraction_min=float(handoff_bounds["seed_fraction"]["min"]),
            seed_fraction_max=float(handoff_bounds["seed_fraction"]["max"]),
            inlet_margin_mode="lower-bound",
            B=float(baseline.B),
            L=float(baseline.L),
            stage_schedule=[dict(item) for item in baseline.schedule],
            out_dir=continuation_out_dir,
            stop_on_unacceptable=True,
            warm_start_policy="regular",
            adaptive_bridge_count=int(baseline.adaptive_bridge_count),
            adaptive_bridge_max_count=int(baseline.adaptive_bridge_max_count),
            warm_profile=warm_profile,
        )

    hybrid = HybridRunResult(
        baseline_seed=baseline,
        maingo_status=maingo_summary_payload["status"],
        maingo_best=coarse_best,
        handoff_bounds=handoff_bounds,
        maingo_summary_path=maingo_summary_path,
        maingo_best_profile_path=maingo_best_profile_path,
        continuation_out_dir=continuation_out_dir,
        continuation_summary=continuation_payload,
        hybrid_summary_path=out_dir_path / "hybrid_summary.json",
    )
    hybrid_summary_path = _write_json(hybrid.hybrid_summary_path, hybrid.to_dict())
    return HybridRunResult(
        baseline_seed=hybrid.baseline_seed,
        maingo_status=hybrid.maingo_status,
        maingo_best=hybrid.maingo_best,
        handoff_bounds=hybrid.handoff_bounds,
        maingo_summary_path=hybrid.maingo_summary_path,
        maingo_best_profile_path=hybrid.maingo_best_profile_path,
        continuation_out_dir=hybrid.continuation_out_dir,
        continuation_summary=hybrid.continuation_summary,
        hybrid_summary_path=hybrid_summary_path,
    )
