from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np

from v6_firedrake_reduced.analyze_kkt import _recover_nonnegative_multipliers
from v6_firedrake_reduced.constraints import evaluate_velikhov_node_constraints
from v6_firedrake_reduced.cases.freidberg_reference import load_reference_profile
from v6_firedrake_reduced.design import DesignVector, load_case_config
from v6_firedrake_reduced.freidberg_area_only import AREA_CONTROL_NAMES, reference_profile_metrics
from v6_firedrake_reduced.freidberg_branch_audit import audit_freidberg_branches
from v6_firedrake_reduced.geometry import LogAreaSplineControl
from v6_firedrake_reduced.objective import evaluate_profile_metrics
from v6_firedrake_reduced.run_firedrake_reduced import _design_from_json, _validate_optimizer_options
from v6_firedrake_reduced.transport import (
    ELECTRON_TRANSPORT_E_ARGON,
    ELECTRON_TRANSPORT_E_HE,
    LXCAT_E_HE_MEDIAN_4300K_M2,
    normalize_electron_transport,
    working_fluid_for_config,
)
from v6_firedrake_reduced.yamasaki_geometry_audit import build_geometry_audit


class FiredrakeReducedPureContractTest(unittest.TestCase):
    def test_log_area_spline_keeps_inlet_normalized_and_direct_knots(self):
        control = LogAreaSplineControl(a1=0.2, a2=0.5, a3=1.0)
        # 3.0 is a nonphysical sentinel area scale; A[0] should preserve the supplied dimensional scale.
        profile = control.evaluate_profile(length=2.0, n_intervals=60, area_scale=3.0)
        self.assertAlmostEqual(float(profile["A"][0]), 3.0)
        knot_indices = [0, 20, 40, 60]
        logA = np.asarray(profile["logA"], dtype=float)
        self.assertAlmostEqual(float(logA[knot_indices[0]]), 0.0, places=12)
        self.assertAlmostEqual(float(logA[knot_indices[1]]), 0.2, places=12)
        self.assertAlmostEqual(float(logA[knot_indices[2]]), 0.5, places=12)
        self.assertAlmostEqual(float(logA[knot_indices[3]]), 1.0, places=12)

    def test_design_vector_json_round_trip_and_bounds(self):
        config = load_case_config(case="yamasaki2004")
        payload = json.loads(json.dumps(config.design.to_dict()))
        recovered = DesignVector.from_dict(payload)
        np.testing.assert_allclose(recovered.as_array(), config.design.as_array())
        self.assertTrue(config.bounds.contains(recovered))
        self.assertEqual(float(config.bounds.lower.B_T), 1.0)
        self.assertEqual(float(config.bounds.upper.B_T), 20.0)
        legacy = DesignVector.from_array(config.design.as_array()[:-1])
        self.assertEqual(float(legacy.B_T), 3.0)
        shifted = DesignVector.from_array(config.bounds.upper.as_array() + 1.0)
        self.assertFalse(config.bounds.contains(shifted))
        violations = config.bounds.violations(shifted)
        self.assertEqual(len(violations), len(config.design.as_array()))
        self.assertEqual(violations[0]["name"], "log_n_p_in")

    def test_design_json_loader_rejects_outside_bounds_unless_explicitly_allowed(self):
        config = load_case_config(case="yamasaki2004")
        outside = DesignVector.from_array(config.bounds.upper.as_array() + 1.0)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "outside_design.json"
            path.write_text(json.dumps(outside.to_dict()), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "outside the active case bounds"):
                _design_from_json(path, config=config)
            replayed = _design_from_json(path, config=config, allow_out_of_bounds=True)
        self.assertFalse(config.bounds.contains(replayed))
        np.testing.assert_allclose(replayed.as_array(), outside.as_array())

    def test_yamasaki_geometry_audit_separates_endpoint_and_swirl_area(self):
        report, _ = build_geometry_audit(n_intervals=12, swirl_angle_deg=45.0)
        current = report["area_variants"]["current_endpoint_fitted_effective_area"]
        swirl = report["area_variants"]["cos_swirl_annular_area_streamline_length"]
        self.assertAlmostEqual(current["reported_endpoint_errors"]["throat_rel"], 0.0, places=12)
        self.assertAlmostEqual(current["reported_endpoint_errors"]["exit_rel"], 0.0, places=12)
        self.assertGreater(abs(swirl["reported_endpoint_errors"]["exit_rel"]), 0.1)
        self.assertGreater(report["lengths"]["streamline_over_radial_length"], 1.0)
        inferred = report["inferred_swirl_from_reported_cross_sections"]
        self.assertGreater(inferred["flow_angle_deg"]["throat"], 40.0)
        self.assertLess(inferred["flow_angle_deg"]["exit"], 10.0)
        self.assertGreater(inferred["streamline_length_m"], report["lengths"]["radial_length_m"])
        self.assertLess(inferred["streamline_length_m"], report["lengths"]["streamline_length_m"])

    def test_geometry_length_modes_are_ordered(self):
        radial = load_case_config(case="yamasaki2004", geometry_length_mode="radial")
        inferred = load_case_config(case="yamasaki2004", geometry_length_mode="inferred_swirl")
        self.assertLess(radial.length_m, inferred.length_m)
        self.assertEqual(inferred.metadata["geometry_length_mode"], "inferred_swirl")
        with self.assertRaisesRegex(ValueError, "unknown geometry_length_mode"):
            load_case_config(case="yamasaki2004", geometry_length_mode="constant45")

    def test_electron_transport_selection_is_metadata_scoped(self):
        config = load_case_config(case="yamasaki2004")
        self.assertEqual(normalize_electron_transport("e_he"), ELECTRON_TRANSPORT_E_HE)
        self.assertEqual(normalize_electron_transport("e-Ar"), ELECTRON_TRANSPORT_E_ARGON)
        base = working_fluid_for_config(config)
        self.assertAlmostEqual(float(base.sigma_ep), LXCAT_E_HE_MEDIAN_4300K_M2)
        self.assertIn("LXCat e-He", base.sigma_ep_note)
        legacy = working_fluid_for_config(
            replace(
                config,
                metadata={
                    **config.metadata,
                    "electron_transport": ELECTRON_TRANSPORT_E_ARGON,
                },
            )
        )
        self.assertEqual(legacy.key, base.key)
        self.assertIn("e-Argon", legacy.sigma_ep_note)
        self.assertNotEqual(float(legacy.sigma_ep), float(base.sigma_ep))
        with self.assertRaisesRegex(ValueError, "arbitrary sigma_ep metadata is no longer supported"):
            working_fluid_for_config(
                replace(
                    config,
                    metadata={
                        **config.metadata,
                        "sigma_ep_override_m2": 5.0e-20,
                    },
                )
            )

    def test_freidberg_reference_case_is_area_only_argon_benchmark(self):
        config = load_case_config(case="freidberg_reference", n_intervals=8)
        self.assertEqual(config.working_fluid_profile, "argon_potassium")
        self.assertEqual(config.metadata["electron_transport"], ELECTRON_TRANSPORT_E_ARGON)
        self.assertAlmostEqual(config.area_scale_m2, 0.447)
        self.assertAlmostEqual(config.length_m, 5.325180763867631)
        lower = config.bounds.lower.to_dict()
        upper = config.bounds.upper.to_dict()
        for name, value in config.design.to_dict().items():
            if name in AREA_CONTROL_NAMES:
                self.assertLess(lower[name], value)
                self.assertGreater(upper[name], value)
            else:
                self.assertEqual(lower[name], value)
                self.assertEqual(upper[name], value)

    def test_freidberg_reference_profile_metrics_are_finite(self):
        config = load_case_config(case="freidberg_reference", n_intervals=8)
        profile = load_reference_profile()
        self.assertIn("sigma_logA", profile)
        metrics = reference_profile_metrics(config)
        self.assertTrue(metrics["finite_profile"])
        self.assertAlmostEqual(metrics["inlet_T_p_K"], 429.0, places=4)
        self.assertAlmostEqual(metrics["mhd_output_power_W"] / 1.0e6, 112.230765690579, places=5)
        self.assertGreater(metrics["raw_enthalpy_extraction_percent"], 30.0)
        self.assertAlmostEqual(metrics["estimated_total_plant_power_MWe"], 200.0, places=4)

    def test_synthetic_profile_metrics_are_finite(self):
        config = load_case_config(case="yamasaki2004", n_intervals=8)
        area = config.design.area_control.evaluate_profile(
            length=config.length_m,
            n_intervals=config.n_intervals,
            area_scale=config.area_scale_m2,
        )
        x = np.asarray(area["x"], dtype=float)
        profile = {
            "x": x,
            "x_norm": np.asarray(area["x_norm"], dtype=float),
            "n_p": np.linspace(config.design.n_p_in, 0.9 * config.design.n_p_in, x.size),
            "T_e": np.linspace(config.design.T_e_in, 1.05 * config.design.T_e_in, x.size),
            "A": np.asarray(area["A"], dtype=float),
            "sigma_logA": np.asarray(area["sigma_logA"], dtype=float),
        }
        metrics = evaluate_profile_metrics(profile=profile, design=config.design, config=config)
        payload = metrics.to_dict()
        self.assertTrue(payload["finite_profile"])
        for value in payload.values():
            if isinstance(value, bool):
                continue
            self.assertTrue(np.isfinite(float(value)))

    def test_freidberg_branch_audit_returns_json_safe_summary(self):
        config = load_case_config(case="yamasaki2004", n_intervals=2)
        area = config.design.area_control.evaluate_profile(
            length=config.length_m,
            n_intervals=config.n_intervals,
            area_scale=config.area_scale_m2,
        )
        x = np.asarray(area["x"], dtype=float)
        profile = {
            "x": x,
            "x_norm": np.asarray(area["x_norm"], dtype=float),
            "n_p": np.linspace(config.design.n_p_in, 0.95 * config.design.n_p_in, x.size),
            "T_e": np.linspace(config.design.T_e_in, 1.02 * config.design.T_e_in, x.size),
            "A": np.asarray(area["A"], dtype=float),
            "sigma_logA": np.asarray(area["sigma_logA"], dtype=float),
        }
        audit = audit_freidberg_branches(
            profile=profile,
            design=config.design,
            config=config,
            branch_policy="continuity",
        )
        summary = audit["summary"]
        self.assertEqual(summary["n_points"], x.size)
        self.assertIsInstance(summary["chosen_failure_count"], int)
        self.assertIsInstance(summary["subsonic_success_count"], int)
        self.assertIsInstance(summary["supersonic_success_count"], int)
        self.assertIn("closest_to_sonic", summary)
        self.assertEqual(len(audit["rows"]), x.size)
        self.assertIn("subsonic", audit["rows"][0]["branches"])
        self.assertIn("supersonic", audit["rows"][0]["branches"])
        json.dumps(audit, allow_nan=False)

    def test_velikhov_penalty_lowers_objective_when_floor_is_violated(self):
        config = load_case_config(case="yamasaki2004", n_intervals=8)
        area = config.design.area_control.evaluate_profile(
            length=config.length_m,
            n_intervals=config.n_intervals,
            area_scale=config.area_scale_m2,
        )
        x = np.asarray(area["x"], dtype=float)
        profile = {
            "x": x,
            "x_norm": np.asarray(area["x_norm"], dtype=float),
            "n_p": np.linspace(config.design.n_p_in, 0.9 * config.design.n_p_in, x.size),
            "T_e": np.linspace(config.design.T_e_in, 1.05 * config.design.T_e_in, x.size),
            "A": np.asarray(area["A"], dtype=float),
            "sigma_logA": np.asarray(area["sigma_logA"], dtype=float),
        }
        baseline = evaluate_profile_metrics(profile=profile, design=config.design, config=config)
        penalized_config = replace(
            config,
            metadata={
                **config.metadata,
                "velikhov_mode": "penalty",
                "velikhov_floor": baseline.min_velikhov_margin + 1.0,
                "velikhov_penalty_scale": 1.0,
                "velikhov_penalty_weight": 2.0,
            },
        )
        penalized = evaluate_profile_metrics(profile=profile, design=config.design, config=penalized_config)
        self.assertFalse(penalized.velikhov_passes_floor)
        self.assertGreater(penalized.velikhov_penalty, 0.0)
        self.assertAlmostEqual(penalized.raw_enthalpy_extraction_percent, baseline.raw_enthalpy_extraction_percent)
        self.assertLess(penalized.objective_score, penalized.raw_enthalpy_extraction_percent)

    def test_thermal_window_penalty_lowers_objective_when_window_is_violated(self):
        config = load_case_config(case="yamasaki2004", n_intervals=8)
        area = config.design.area_control.evaluate_profile(
            length=config.length_m,
            n_intervals=config.n_intervals,
            area_scale=config.area_scale_m2,
        )
        x = np.asarray(area["x"], dtype=float)
        profile = {
            "x": x,
            "x_norm": np.asarray(area["x_norm"], dtype=float),
            "n_p": np.linspace(config.design.n_p_in, 0.9 * config.design.n_p_in, x.size),
            "T_e": np.linspace(config.design.T_e_in, 1.05 * config.design.T_e_in, x.size),
            "A": np.asarray(area["A"], dtype=float),
            "sigma_logA": np.asarray(area["sigma_logA"], dtype=float),
        }
        baseline = evaluate_profile_metrics(profile=profile, design=config.design, config=config)
        penalized_config = replace(
            config,
            metadata={
                **config.metadata,
                "thermal_window_mode": "penalty",
                "thermal_tp_in_max_K": baseline.inlet_T_p_K - 10.0,
                "thermal_tp_floor_K": baseline.min_T_p_K + 10.0,
                "thermal_tp_penalty_scale_K": 10.0,
                "thermal_tp_in_penalty_weight": 1.0,
                "thermal_tp_path_penalty_weight": 1.0,
            },
        )
        penalized = evaluate_profile_metrics(profile=profile, design=config.design, config=penalized_config)
        self.assertTrue(penalized.thermal_window_active)
        self.assertFalse(penalized.thermal_window_passes)
        self.assertGreater(penalized.thermal_window_penalty, 0.0)
        self.assertGreater(penalized.thermal_Tp_in_penalty_candidate, 0.0)
        self.assertGreater(penalized.thermal_Tp_low_penalty_candidate, 0.0)
        self.assertAlmostEqual(penalized.raw_enthalpy_extraction_percent, baseline.raw_enthalpy_extraction_percent)
        self.assertLess(penalized.objective_score, penalized.raw_enthalpy_extraction_percent)

    def test_velikhov_node_constraints_match_metric_minimum_and_sign(self):
        config = load_case_config(case="yamasaki2004", n_intervals=8)
        area = config.design.area_control.evaluate_profile(
            length=config.length_m,
            n_intervals=config.n_intervals,
            area_scale=config.area_scale_m2,
        )
        x = np.asarray(area["x"], dtype=float)
        profile = {
            "x": x,
            "x_norm": np.asarray(area["x_norm"], dtype=float),
            "n_p": np.linspace(config.design.n_p_in, 0.92 * config.design.n_p_in, x.size),
            "T_e": np.linspace(config.design.T_e_in, 1.04 * config.design.T_e_in, x.size),
            "A": np.asarray(area["A"], dtype=float),
            "sigma_logA": np.asarray(area["sigma_logA"], dtype=float),
        }
        metrics = evaluate_profile_metrics(profile=profile, design=config.design, config=config)
        constraints = evaluate_velikhov_node_constraints(
            profile=profile,
            design=config.design,
            config=config,
            floor=metrics.min_velikhov_margin + 1.0,
        )
        self.assertAlmostEqual(constraints.min_G_node, metrics.min_velikhov_margin)
        self.assertAlmostEqual(constraints.min_margin, -1.0)
        self.assertGreaterEqual(constraints.argmin_index, 0)
        self.assertEqual(constraints.to_dict()["sampling"], "nodes")

    def test_hard_velikhov_constraint_rejects_soft_penalty_mix(self):
        with self.assertRaisesRegex(ValueError, "cannot be combined"):
            _validate_optimizer_options(
                SimpleNamespace(
                    optimizer="constrained_slsqp",
                    velikhov_constraint_mode="hard",
                    velikhov_mode="penalty",
                )
            )
        with self.assertRaisesRegex(ValueError, "requires --optimizer constrained_slsqp"):
            _validate_optimizer_options(
                SimpleNamespace(
                    optimizer="projected_gradient",
                    velikhov_constraint_mode="hard",
                    velikhov_mode="diagnostic",
                )
            )

    def test_kkt_multiplier_recovery_balances_all_control_components(self):
        gradient = np.array([2.0, 3.0], dtype=float)
        columns = [
            np.array([2.0, 0.0], dtype=float),
            np.array([0.0, 3.0], dtype=float),
        ]
        recovered = _recover_nonnegative_multipliers(
            gradient_minimize=gradient,
            columns=columns,
        )
        np.testing.assert_allclose(recovered["support"], gradient, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(recovered["stationarity"], np.zeros_like(gradient), atol=1e-12)
        np.testing.assert_allclose(recovered["multipliers"], np.ones(2), rtol=1e-12, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
