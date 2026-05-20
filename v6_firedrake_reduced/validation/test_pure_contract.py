from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from v6_firedrake_reduced.design import DesignVector, load_case_config
from v6_firedrake_reduced.freidberg_branch_audit import audit_freidberg_branches
from v6_firedrake_reduced.geometry import LogAreaSplineControl
from v6_firedrake_reduced.objective import evaluate_profile_metrics
from v6_firedrake_reduced.run_firedrake_reduced import _design_from_json
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


if __name__ == "__main__":
    unittest.main()
