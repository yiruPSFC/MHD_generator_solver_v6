from __future__ import annotations

import math
import unittest

from v6_maingo_casadi.core import _import_maingopy
from v6_maingo_casadi.implicit_reduced_factorable import (
    make_toy_fixed_newton_maingo_model,
    optional_choke_gate,
    rollout_toy_fixed_newton,
)
from v6_maingo_casadi.numerics import _ops_for_numeric


def _toy_rhs(x: float, y: float, *, forcing_center: float) -> float:
    return (float(x) - float(forcing_center)) / (float(y) - 1.0)


def _toy_rk4_rollout(*, y0: float, forcing_center: float, n_intervals: int, length: float = 1.0) -> list[float]:
    dx = float(length) / int(n_intervals)
    x = 0.0
    y = float(y0)
    values = [y]
    for _ in range(int(n_intervals)):
        k1 = _toy_rhs(x, y, forcing_center=forcing_center)
        k2 = _toy_rhs(x + 0.5 * dx, y + 0.5 * dx * k1, forcing_center=forcing_center)
        k3 = _toy_rhs(x + 0.5 * dx, y + 0.5 * dx * k2, forcing_center=forcing_center)
        k4 = _toy_rhs(x + dx, y + dx * k3, forcing_center=forcing_center)
        y = y + dx * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
        x += dx
        values.append(y)
    return values


class ImplicitReducedFactorableTests(unittest.TestCase):
    def test_optional_choke_gate_is_inactive_when_trace_stays_far_from_sonic(self):
        gate = optional_choke_gate(
            ops=_ops_for_numeric(),
            x_nodes=[0.0, 0.5, 1.0],
            det_nodes=[0.4, 0.5, 0.6],
            x_choke=-0.5,
        )

        self.assertEqual(float(gate.near_sonic_gate), 0.0)
        self.assertEqual(float(gate.sign_change_gate), 0.0)
        self.assertEqual(float(gate.choke_gate), 0.0)
        self.assertEqual(float(gate.gated_location_residual), 0.0)
        self.assertTrue(all(float(residual) == 0.0 for residual in gate.gated_segment_residuals))

    def test_optional_choke_gate_cannot_be_disabled_by_moving_candidate_outside(self):
        gate = optional_choke_gate(
            ops=_ops_for_numeric(),
            x_nodes=[0.0, 0.5, 1.0],
            det_nodes=[0.2, -0.1, -0.2],
            x_choke=-0.5,
        )

        self.assertEqual(float(gate.inside_gate), 0.0)
        self.assertGreater(float(gate.sign_change_gate), 0.0)
        self.assertGreater(float(gate.choke_gate), 0.0)
        self.assertGreater(float(gate.gated_location_residual), 0.0)
        self.assertTrue(any(abs(float(residual)) > 0.0 for residual in gate.gated_segment_residuals))

    def test_fixed_newton_rollout_keeps_low_dimensional_inputs_and_optional_choke_outputs(self):
        rollout = rollout_toy_fixed_newton(
            y0=0.5,
            n_intervals=4,
            newton_steps=10,
        )

        self.assertEqual(len(rollout.y_nodes), 5)
        self.assertEqual(len(rollout.step_residuals), 4)
        self.assertTrue(math.isfinite(float(rollout.final_y)))
        self.assertTrue(math.isfinite(float(rollout.max_abs_step_residual)))
        self.assertGreater(float(rollout.optional_choke.choke_gate), 0.0)
        self.assertEqual(float(rollout.optional_choke.gated_location_residual), 0.0)
        self.assertTrue(all(float(residual) == 0.0 for residual in rollout.optional_choke.gated_segment_residuals))

    def test_fixed_newton_rollout_adds_critical_residuals_only_with_candidate_location(self):
        rollout = rollout_toy_fixed_newton(
            y0=0.5,
            x_choke=-0.5,
            n_intervals=4,
            newton_steps=10,
        )

        self.assertGreater(float(rollout.optional_choke.choke_gate), 0.0)
        self.assertGreater(float(rollout.optional_choke.gated_location_residual), 0.0)

    def test_nonchoking_fixed_newton_rollout_matches_rk4(self):
        y0 = 1.8
        forcing_center = 0.5
        n_intervals = 40

        implicit_rollout = rollout_toy_fixed_newton(
            y0=y0,
            forcing_center=forcing_center,
            n_intervals=n_intervals,
            newton_steps=10,
        )
        rk4_nodes = _toy_rk4_rollout(
            y0=y0,
            forcing_center=forcing_center,
            n_intervals=n_intervals,
        )

        self.assertGreater(min(abs(float(value) - 1.0) for value in implicit_rollout.y_nodes), 0.5)
        self.assertEqual(float(implicit_rollout.optional_choke.choke_gate), 0.0)
        self.assertLess(float(implicit_rollout.max_abs_step_residual), 1e-10)
        self.assertAlmostEqual(float(implicit_rollout.final_y), float(rk4_nodes[-1]), delta=1e-7)

    def test_toy_maingo_model_exposes_default_reduced_factorable_api(self):
        maingopy = _import_maingopy()
        model = make_toy_fixed_newton_maingo_model(
            maingopy_module=maingopy,
            n_intervals=4,
            newton_steps=10,
        )

        variables = model.get_variables()
        result = model.evaluate(model.get_initial_point())

        self.assertEqual(len(variables), 1)
        self.assertEqual(len(model.get_initial_point()), 1)
        self.assertTrue(hasattr(result, "objective"))
        self.assertTrue(hasattr(result, "ineq"))
        self.assertEqual(len(result.output), 5)

    def test_toy_maingo_model_exposes_critical_location_only_when_enabled(self):
        maingopy = _import_maingopy()
        model = make_toy_fixed_newton_maingo_model(
            maingopy_module=maingopy,
            n_intervals=4,
            newton_steps=10,
            critical_mode=True,
        )

        variables = model.get_variables()
        result = model.evaluate(model.get_initial_point())

        self.assertEqual(len(variables), 2)
        self.assertEqual(len(model.get_initial_point()), 2)
        self.assertTrue(hasattr(result, "objective"))


if __name__ == "__main__":
    unittest.main()
