from __future__ import annotations

from dataclasses import replace
import unittest

import numpy as np

from v6_firedrake_reduced.design import load_case_config
from v6_firedrake_reduced.forward import FiredrakeUnavailableError, solve_forward
from v6_firedrake_reduced.reduced_functional import (
    build_reduced_functional,
    evaluate_reduced_functional,
    reduced_functional_gradient,
)


def _require_firedrake_for_test():
    try:
        import firedrake  # noqa: F401
        import firedrake.adjoint  # noqa: F401
    except ImportError as exc:
        raise unittest.SkipTest(f"Firedrake/pyadjoint unavailable: {exc}") from exc


class FiredrakeReducedSmokeTest(unittest.TestCase):
    def test_tiny_forward_solve_baseline(self):
        _require_firedrake_for_test()
        config = load_case_config(case="yamasaki2004", n_intervals=4)
        result = solve_forward(design=config.design, config=config)
        if not result.ok:
            self.fail(result.error)
        self.assertIsNotNone(result.profile)
        self.assertIsNotNone(result.metrics)
        self.assertTrue(result.metrics.finite_profile)
        self.assertEqual(result.diagnostics["residual_scaling"], "inlet")
        self.assertIn("strong_residual_diagnostics", result.diagnostics)

    def test_residual_scaling_modes_report_strong_diagnostics(self):
        _require_firedrake_for_test()
        base_config = load_case_config(case="yamasaki2004", n_intervals=4)
        for mode in ("inlet", "characteristic", "dimensional"):
            with self.subTest(mode=mode):
                config = replace(base_config, metadata={**base_config.metadata, "residual_scaling": mode})
                result = solve_forward(design=config.design, config=config)
                if not result.ok:
                    self.fail(result.error)
                self.assertEqual(result.diagnostics["residual_scaling"], mode)
                strong = result.diagnostics["strong_residual_diagnostics"]
                self.assertGreater(strong["energy_scale_min"], 0.0)
                self.assertGreater(strong["momentum_scale_min"], 0.0)

    def test_reduced_functional_evaluation_is_finite(self):
        _require_firedrake_for_test()
        config = load_case_config(case="yamasaki2004", n_intervals=4)
        try:
            bundle = build_reduced_functional(design=config.design, config=config)
        except FiredrakeUnavailableError as exc:
            raise unittest.SkipTest(str(exc)) from exc
        value = evaluate_reduced_functional(bundle, config.design.as_array())
        self.assertTrue(np.isfinite(float(value)))
        perturbed = config.design.as_array()
        perturbed[3] += 10.0
        perturbed_value = evaluate_reduced_functional(bundle, perturbed)
        self.assertGreater(abs(float(perturbed_value) - float(value)), 1e-6)

    def test_taped_objective_tracks_enthalpy_extraction_diagnostic(self):
        _require_firedrake_for_test()
        config = load_case_config(case="yamasaki2004", n_intervals=4)
        result = solve_forward(design=config.design, config=config, annotate_objective=True)
        if not result.ok:
            self.fail(result.error)
        self.assertEqual(result.diagnostics["taped_objective"], "enthalpy_extraction")
        self.assertIsNotNone(result.metrics)
        self.assertIsNotNone(result.fd_objective)
        metric_value = float(result.metrics.objective_score)
        taped_value = float(result.fd_objective)
        scale = max(1.0, abs(metric_value), abs(taped_value))
        self.assertLess(abs(metric_value - taped_value) / scale, 0.1)

    def test_taped_objective_includes_velikhov_penalty_when_enabled(self):
        _require_firedrake_for_test()
        base_config = load_case_config(case="yamasaki2004", n_intervals=4)
        config = replace(
            base_config,
            metadata={
                **base_config.metadata,
                "velikhov_mode": "penalty",
                "velikhov_floor": 1e9,
                "velikhov_penalty_scale": 1e6,
                "velikhov_penalty_weight": 1.0,
            },
        )
        result = solve_forward(design=config.design, config=config, annotate_objective=True)
        if not result.ok:
            self.fail(result.error)
        self.assertIsNotNone(result.metrics)
        self.assertIsNotNone(result.fd_objective)
        self.assertGreater(result.metrics.velikhov_penalty, 0.0)
        self.assertLess(float(result.fd_objective), result.metrics.raw_enthalpy_extraction_percent)

    def test_adjoint_gradient_matches_loose_finite_difference_direction(self):
        _require_firedrake_for_test()
        config = load_case_config(case="yamasaki2004", n_intervals=4)
        try:
            bundle = build_reduced_functional(design=config.design, config=config)
        except FiredrakeUnavailableError as exc:
            raise unittest.SkipTest(str(exc)) from exc
        x0 = config.design.as_array()
        direction = np.zeros_like(x0)
        direction[3] = 1.0
        eps = 1e-2
        value0 = evaluate_reduced_functional(bundle, x0)
        value1 = evaluate_reduced_functional(bundle, x0 + eps * direction)
        finite_difference = (value1 - value0) / eps
        gradient_arr = reduced_functional_gradient(bundle)
        adjoint_directional = float(np.dot(gradient_arr, direction))
        scale = max(1.0, abs(finite_difference), abs(adjoint_directional))
        self.assertLess(abs(finite_difference - adjoint_directional) / scale, 5e-2)


if __name__ == "__main__":
    unittest.main()
