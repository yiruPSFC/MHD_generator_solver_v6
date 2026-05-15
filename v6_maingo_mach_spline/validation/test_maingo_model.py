from __future__ import annotations

import math
from pathlib import Path
import unittest

from v6_maingo_casadi.models import BaselineSeed
from v6_maingo_casadi.workflow import _resolve_mach_reference_profile_path

from v6_maingo_mach_spline.maingo_model import (
    MachSplineRK4SoftModelBase,
    MachSplineReducedImplicitModelBase,
    MachSplineTrapezoidModelBase,
)


REPO = Path(__file__).resolve().parents[2]
CASE_41 = REPO / "v6_maingo_casadi/outputs/maingo_yamasaki2004_neighborhood/tp_midpoint_semantics_smoke_20260511_v3"
PROFILE_41 = CASE_41 / "maingo_best_profile.npz"
SUMMARY_41 = CASE_41 / "maingo_summary.json"
BASELINE_SUMMARY = (
    REPO
    / "v6_maingo_casadi/outputs/cases/yamasaki2004/seeds/yamasaki2004_hecs_disk_geometry_reference_seed_summary.json"
)


class _FakeMaingo:
    VT_CONTINUOUS = "continuous"
    exp = staticmethod(math.exp)
    log = staticmethod(math.log)
    sqrt = staticmethod(math.sqrt)
    fabs = staticmethod(abs)
    max = staticmethod(max)
    min = staticmethod(min)
    pos = staticmethod(lambda value: max(value, 0.0))
    neg = staticmethod(lambda value: min(value, 0.0))
    lb_func = staticmethod(lambda value, lower: max(value, lower))
    ub_func = staticmethod(lambda value, upper: min(value, upper))
    bounding_func = staticmethod(lambda value, lower, upper: min(max(value, lower), upper))

    class Bounds:
        def __init__(self, lower, upper):
            self.lower = lower
            self.upper = upper

    class OptimizationVariable:
        def __init__(self, bounds, var_type, name):
            self.bounds = bounds
            self.var_type = var_type
            self.name = name

    class EvaluationContainer:
        pass

    class OutputVariable:
        def __init__(self, name, value):
            self.name = name
            self.value = value

    class ModelFunction(list):
        def push_back(self, item):
            self.append(item)


def _baseline_from_case_summary() -> BaselineSeed:
    return BaselineSeed.from_summary(BASELINE_SUMMARY)


class MachSplineMAiNGOModelTest(unittest.TestCase):
    def test_model_exposes_mach_decision_variables(self) -> None:
        model = MachSplineReducedImplicitModelBase(
            baseline=_baseline_from_case_summary(),
            reference_profile_path=PROFILE_41,
            n_intervals=80,
            maingopy_module=_FakeMaingo,
            newton_steps=10,
        )
        variables = model.get_variables()
        names = [item.name for item in variables]
        self.assertEqual(len(variables), 8)
        self.assertEqual(names[-3:], ["m1", "m2", "m3"])
        self.assertNotIn("a1", names)
        self.assertNotIn("a2", names)
        self.assertNotIn("a3", names)
        self.assertEqual(len(model.get_initial_point()), 8)

    def test_model_evaluate_returns_objective_constraints_and_outputs(self) -> None:
        model = MachSplineReducedImplicitModelBase(
            baseline=_baseline_from_case_summary(),
            reference_profile_path=PROFILE_41,
            n_intervals=80,
            maingopy_module=_FakeMaingo,
            newton_steps=10,
        )
        result = model.evaluate(model.get_initial_point())
        self.assertTrue(math.isfinite(float(result.objective)))
        self.assertGreater(len(result.ineq), 0)
        output_names = {item.name for item in result.output}
        self.assertIn("mach_spline_max_scaled_residual", output_names)
        self.assertIn("derived_area_outlet", output_names)
        self.assertIn("outlet_mach", output_names)

    def test_model_decodes_and_resamples_standard_profile_result(self) -> None:
        model = MachSplineReducedImplicitModelBase(
            baseline=_baseline_from_case_summary(),
            reference_profile_path=PROFILE_41,
            n_intervals=80,
            maingopy_module=_FakeMaingo,
            newton_steps=10,
        )
        solution = model.decode_solution_point(model.get_initial_point())
        result = model.evaluate_solution(solution)
        self.assertEqual(result.diagnostics["formulation"], model.formulation)
        self.assertIn("m1", result.decision_vector)
        self.assertEqual(result.x.size, 81)
        self.assertEqual(result.A.size, 81)
        self.assertTrue(math.isfinite(float(result.objective_score)))

        handoff = model.resample_solution_result(result, n_intervals=40)
        self.assertEqual(handoff.x.size, 41)
        self.assertEqual(handoff.A.size, 41)
        self.assertIn("mach_spline_max_scaled_residual", handoff.diagnostics)

    def test_workflow_infers_mach_reference_from_initial_solution_dir(self) -> None:
        resolved = _resolve_mach_reference_profile_path(
            explicit_profile_path=None,
            initial_solution_json=SUMMARY_41,
            baseline=_baseline_from_case_summary(),
        )
        self.assertEqual(resolved, PROFILE_41.resolve())

    def test_rk4_soft_model_returns_candidate_diagnostics(self) -> None:
        model = MachSplineRK4SoftModelBase(
            baseline=_baseline_from_case_summary(),
            reference_profile_path=PROFILE_41,
            n_intervals=10,
            maingopy_module=_FakeMaingo,
            newton_steps=10,
        )
        variables = model.get_variables()
        self.assertEqual([item.name for item in variables][-3:], ["m1", "m2", "m3"])

        result = model.evaluate(model.get_initial_point())
        self.assertTrue(math.isfinite(float(result.objective)))
        self.assertGreater(len(result.ineq), 0)
        self.assertLessEqual(max(float(item) for item in result.ineq), 0.0)
        output_names = {item.name for item in result.output}
        self.assertIn("sigma_smoothness_penalty", output_names)
        self.assertIn("tp_shortfall_penalty", output_names)
        self.assertIn("mach_spline_min_signed_det", output_names)

        solution = model.decode_solution_point(model.get_initial_point())
        profile = model.evaluate_solution(solution)
        self.assertEqual(profile.diagnostics["formulation"], "mach_spline_rk4_soft")
        self.assertEqual(profile.diagnostics["det_branch_sign"], "positive")
        self.assertTrue(profile.diagnostics["det_branch_acceptable"])
        self.assertIn("freidberg_interval_defects", profile.diagnostics)
        self.assertIn("physical_acceptable", profile.diagnostics)
        self.assertTrue(math.isfinite(float(profile.objective_score)))

    def test_trapezoid_model_keeps_low_dimensional_parametric_profiles(self) -> None:
        model = MachSplineTrapezoidModelBase(
            baseline=_baseline_from_case_summary(),
            reference_profile_path=PROFILE_41,
            n_intervals=4,
            maingopy_module=_FakeMaingo,
            newton_steps=10,
        )
        variables = model.get_variables()
        self.assertEqual(len(variables), 14)
        self.assertEqual([item.name for item in variables[:8]], [
            "log_n_p_in",
            "T_e_in",
            "Z_in",
            "I_0",
            "log_seed_fraction",
            "m1",
            "m2",
            "m3",
        ])
        self.assertEqual([item.name for item in variables[-6:]], [
            "log_n_ratio_1",
            "log_n_ratio_2",
            "log_n_ratio_3",
            "log_Te_ratio_1",
            "log_Te_ratio_2",
            "log_Te_ratio_3",
        ])

        result = model.evaluate(model.get_initial_point())
        self.assertTrue(math.isfinite(float(result.objective)))
        self.assertGreater(len(result.ineq), 0)
        self.assertEqual(len(result.eq), 0)
        output_names = {item.name for item in result.output}
        self.assertIn("mach_spline_max_scaled_residual", output_names)
        self.assertIn("mach_spline_min_signed_det", output_names)

        solution = model.decode_solution_point(model.get_initial_point())
        profile = model.evaluate_solution(solution)
        self.assertEqual(profile.diagnostics["formulation"], "mach_spline_trapezoid_parametric")
        self.assertEqual(profile.diagnostics["coarse_model"], "mach_spline_trapezoid")
        self.assertEqual(profile.diagnostics["equality_count"], 0)
        self.assertEqual(profile.diagnostics["residual_constraint_count"], 10)
        self.assertTrue(math.isfinite(float(profile.objective_score)))

    def test_trapezoid_model_can_freeze_inlet_variables(self) -> None:
        model = MachSplineTrapezoidModelBase(
            baseline=_baseline_from_case_summary(),
            reference_profile_path=PROFILE_41,
            n_intervals=4,
            maingopy_module=_FakeMaingo,
            newton_steps=10,
            fixed_inlet=True,
        )
        variables = model.get_variables()
        self.assertEqual(len(variables), 9)
        self.assertEqual([item.name for item in variables[:3]], ["m1", "m2", "m3"])
        self.assertEqual([item.name for item in variables[-6:]], [
            "log_n_ratio_1",
            "log_n_ratio_2",
            "log_n_ratio_3",
            "log_Te_ratio_1",
            "log_Te_ratio_2",
            "log_Te_ratio_3",
        ])
        self.assertEqual(len(model.get_initial_point()), 9)
        self.assertTrue(model.summary_metadata()["fixed_inlet"])

        result = model.evaluate(model.get_initial_point())
        self.assertTrue(math.isfinite(float(result.objective)))
        self.assertEqual(len(result.eq), 0)

        solution = model.decode_solution_point(model.get_initial_point())
        self.assertIn("log_n_p_in", solution.decision_vector)
        self.assertIn("m1", solution.decision_vector)
        profile = model.evaluate_solution(solution)
        self.assertTrue(profile.diagnostics["fixed_inlet"])
        self.assertEqual(profile.diagnostics["residual_constraint_count"], 10)
        self.assertTrue(math.isfinite(float(profile.objective_score)))


if __name__ == "__main__":
    unittest.main()
