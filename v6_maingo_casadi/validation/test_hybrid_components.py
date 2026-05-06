from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from v6_casadi_v2.optimize_area_profile_casadi_v2 import _evaluate_inlet_design_numeric
from v6_casadi_v2.run_casadi_continuation_v2 import load_warm_profile_npz
from v6_maingo_casadi.core import (
    BaselineSeed,
    CasadiCoarseEvaluator,
    OBJECTIVE_PROFILE_ENTHALPY_EXTRACTION,
    SplineAreaDesign,
    WORKING_FLUID_PROFILE_HELIUM_CESIUM,
    _MAiNGOHybridReducedImplicitModelBase,
    _build_implicit_reference,
    _handoff_bounds_from_best,
    _import_maingopy,
    evaluate_inlet_design_numeric,
)


REPO_DIR = Path(__file__).resolve().parents[2]
BASELINE_SUMMARY = (
    REPO_DIR
    / "v6_casadi_v2"
    / "outputs"
    / "continuation"
    / "baseline_release_from_v6_candidate_022_sigma_0p5"
    / "continuation_summary.json"
)


class HybridComponentTests(unittest.TestCase):
    def test_baseline_seed_loads_current_summary(self):
        seed = BaselineSeed.from_summary(BASELINE_SUMMARY)
        self.assertTrue(seed.warm_profile_npz_path.exists())
        self.assertEqual(len(seed.schedule), 3)
        self.assertGreater(seed.n_p_in_nominal, 0.0)
        self.assertGreater(seed.T_e_in_nominal, 0.0)
        self.assertGreater(seed.I_0_nominal, 0.0)
        self.assertGreater(seed.seed_fraction_nominal, 0.0)

    def test_spline_profile_stays_positive_and_normalized(self):
        seed = BaselineSeed.from_summary(BASELINE_SUMMARY)
        profile = seed.area_design_nominal.evaluate_profile(length=seed.L, n_intervals=80)
        self.assertAlmostEqual(float(profile["A"][0]), 1.0, places=12)
        self.assertTrue(np.all(profile["A"] > 0.0))

        logA = np.asarray(profile["logA"], dtype=float)
        x = np.asarray(profile["x"], dtype=float)
        sigma = np.asarray(profile["sigma_logA"], dtype=float)
        finite_diff = np.gradient(logA, x)
        self.assertLess(float(np.max(np.abs(sigma[2:-2] - finite_diff[2:-2]))), 3e-2)

    def test_inlet_closure_matches_casadi_v2(self):
        seed = BaselineSeed.from_summary(BASELINE_SUMMARY)
        local = evaluate_inlet_design_numeric(
            n_p_in=seed.n_p_in_nominal,
            T_e_in=seed.T_e_in_nominal,
            Z_in=seed.Z_in_nominal,
            I_0=seed.I_0_nominal,
            seed_fraction=seed.seed_fraction_nominal,
            B=seed.B,
        )
        reference = _evaluate_inlet_design_numeric(
            n_p_in=seed.n_p_in_nominal,
            T_e_in=seed.T_e_in_nominal,
            Z_in=seed.Z_in_nominal,
            I_0=seed.I_0_nominal,
            seed_fraction=seed.seed_fraction_nominal,
            B=seed.B,
        )
        for name in ("T_p", "mach", "velikhov_margin", "dot_N", "I_0"):
            self.assertTrue(
                math.isclose(getattr(local, name), getattr(reference, name), rel_tol=1e-10, abs_tol=1e-8),
                msg=f"mismatch for {name}: {getattr(local, name)} vs {getattr(reference, name)}",
            )

    def test_working_fluid_switch_changes_inlet_closure(self):
        seed = BaselineSeed.from_summary(BASELINE_SUMMARY)
        he_seed = seed.with_working_fluid_profile(WORKING_FLUID_PROFILE_HELIUM_CESIUM)
        self.assertEqual(he_seed.working_fluid.working_gas, "He")
        self.assertEqual(he_seed.working_fluid.seed_species, "Cs")
        self.assertLess(
            float(he_seed.working_fluid.heavy_particle_mass_kg),
            float(seed.working_fluid.heavy_particle_mass_kg),
        )
        self.assertLess(
            float(he_seed.working_fluid.seed_ionization_energy_eV),
            float(seed.working_fluid.seed_ionization_energy_eV),
        )

        argon = evaluate_inlet_design_numeric(
            n_p_in=seed.n_p_in_nominal,
            T_e_in=seed.T_e_in_nominal,
            Z_in=seed.Z_in_nominal,
            I_0=seed.I_0_nominal,
            seed_fraction=seed.seed_fraction_nominal,
            B=seed.B,
            working_fluid_profile=seed.working_fluid,
        )
        helium = evaluate_inlet_design_numeric(
            n_p_in=he_seed.n_p_in_nominal,
            T_e_in=he_seed.T_e_in_nominal,
            Z_in=he_seed.Z_in_nominal,
            I_0=he_seed.I_0_nominal,
            seed_fraction=he_seed.seed_fraction_nominal,
            B=he_seed.B,
            working_fluid_profile=he_seed.working_fluid,
        )
        self.assertFalse(math.isclose(argon.T_p, helium.T_p, rel_tol=1e-6, abs_tol=1e-6))

    def test_spline_projection_respects_bounds(self):
        seed = BaselineSeed.from_summary(BASELINE_SUMMARY)
        with np.load(seed.warm_profile_npz_path) as warm_data:
            projected = SplineAreaDesign.project_from_profile(
                x=np.asarray(warm_data["x"], dtype=float),
                A=np.asarray(warm_data["A"], dtype=float),
            )
        lower = SplineAreaDesign.lower_bound()
        upper = SplineAreaDesign.upper_bound()
        for value in projected.as_array():
            self.assertGreaterEqual(float(value), lower)
            self.assertLessEqual(float(value), upper)

    def test_inlet_upper_bound_expansion_only_changes_requested_maxima(self):
        seed = BaselineSeed.from_summary(BASELINE_SUMMARY)
        expanded = seed.with_inlet_upper_bound_factors(n_p_in=1.5, Z_in=1.25)
        self.assertAlmostEqual(
            float(expanded.inlet_windows["n_p_in"]["max"]),
            1.5 * float(seed.inlet_windows["n_p_in"]["max"]),
        )
        self.assertAlmostEqual(
            float(expanded.inlet_windows["Z_in"]["max"]),
            1.25 * float(seed.inlet_windows["Z_in"]["max"]),
        )
        self.assertAlmostEqual(
            float(expanded.inlet_windows["T_e_in"]["max"]),
            float(seed.inlet_windows["T_e_in"]["max"]),
        )
        self.assertAlmostEqual(
            float(expanded.inlet_windows["I_0"]["max"]),
            float(seed.inlet_windows["I_0"]["max"]),
        )
        self.assertAlmostEqual(
            float(expanded.inlet_windows["seed_fraction"]["max"]),
            float(seed.inlet_windows["seed_fraction"]["max"]),
        )

    def test_inlet_bound_scaling_changes_requested_minima_and_maxima(self):
        seed = BaselineSeed.from_summary(BASELINE_SUMMARY)
        expanded = seed.with_inlet_bound_factors(
            n_p_in_lower=0.75,
            n_p_in_upper=1.5,
            seed_fraction_lower=0.25,
            seed_fraction_upper=2.0,
        )
        self.assertAlmostEqual(
            float(expanded.inlet_windows["n_p_in"]["min"]),
            0.75 * float(seed.inlet_windows["n_p_in"]["min"]),
        )
        self.assertAlmostEqual(
            float(expanded.inlet_windows["n_p_in"]["max"]),
            1.5 * float(seed.inlet_windows["n_p_in"]["max"]),
        )
        self.assertAlmostEqual(
            float(expanded.inlet_windows["seed_fraction"]["min"]),
            0.25 * float(seed.inlet_windows["seed_fraction"]["min"]),
        )
        self.assertAlmostEqual(
            float(expanded.inlet_windows["seed_fraction"]["max"]),
            2.0 * float(seed.inlet_windows["seed_fraction"]["max"]),
        )
        self.assertAlmostEqual(
            float(expanded.inlet_windows["T_e_in"]["min"]),
            float(seed.inlet_windows["T_e_in"]["min"]),
        )
        self.assertAlmostEqual(
            float(expanded.inlet_windows["Z_in"]["max"]),
            float(seed.inlet_windows["Z_in"]["max"]),
        )
        self.assertEqual(expanded.inlet_windows["n_p_in"]["guess"], seed.inlet_windows["n_p_in"]["guess"])

    def test_npz_payload_is_compatible_with_casadi_warm_loader(self):
        seed = BaselineSeed.from_summary(BASELINE_SUMMARY)
        runner = CasadiCoarseEvaluator(baseline=seed, n_intervals=80)
        result = runner.evaluate(
            {
                "log_n_p_in": math.log(seed.n_p_in_nominal),
                "T_e_in": seed.T_e_in_nominal,
                "Z_in": seed.Z_in_nominal,
                "I_0": seed.I_0_nominal,
                "log_seed_fraction": math.log(seed.seed_fraction_nominal),
                "a1": seed.area_design_nominal.a1,
                "a2": seed.area_design_nominal.a2,
                "a3": seed.area_design_nominal.a3,
            }
        )
        bounds = _handoff_bounds_from_best(result)
        with tempfile.TemporaryDirectory() as tmpdir:
            warm_path = Path(tmpdir) / "warm_profile.npz"
            np.savez(warm_path, **result.to_npz_payload())
            warm = load_warm_profile_npz(
                warm_path,
                n_p_in_guess=bounds["n_p_in"]["guess"],
                n_p_in_min=bounds["n_p_in"]["min"],
                n_p_in_max=bounds["n_p_in"]["max"],
                T_e_in_guess=bounds["T_e_in"]["guess"],
                T_e_in_min=bounds["T_e_in"]["min"],
                T_e_in_max=bounds["T_e_in"]["max"],
                Z_in_guess=bounds["Z_in"]["guess"],
                Z_in_min=bounds["Z_in"]["min"],
                Z_in_max=bounds["Z_in"]["max"],
                J_x_in_guess=bounds["I_0"]["guess"],
                J_x_in_min=bounds["I_0"]["min"],
                J_x_in_max=bounds["I_0"]["max"],
                seed_fraction_guess=bounds["seed_fraction"]["guess"],
                seed_fraction_min=bounds["seed_fraction"]["min"],
                seed_fraction_max=bounds["seed_fraction"]["max"],
                B=seed.B,
            )
        self.assertEqual(warm.x.size, 81)
        self.assertEqual(warm.sigma_logA.size, 80)

    def test_enthalpy_extraction_objective_reports_percent_score(self):
        seed = BaselineSeed.from_summary(BASELINE_SUMMARY)
        runner = CasadiCoarseEvaluator(
            baseline=seed,
            n_intervals=80,
            objective_profile=OBJECTIVE_PROFILE_ENTHALPY_EXTRACTION,
        )
        result = runner.evaluate(
            {
                "log_n_p_in": math.log(seed.n_p_in_nominal),
                "T_e_in": seed.T_e_in_nominal,
                "Z_in": seed.Z_in_nominal,
                "I_0": seed.I_0_nominal,
                "log_seed_fraction": math.log(seed.seed_fraction_nominal),
                "a1": seed.area_design_nominal.a1,
                "a2": seed.area_design_nominal.a2,
                "a3": seed.area_design_nominal.a3,
            }
        )
        terms = result.value_terms
        self.assertEqual(result.value_profile["profile_name"], "enthalpy_extraction_objective")
        self.assertTrue(np.isfinite(float(terms["inlet_enthalpy_flux_MW"])))
        self.assertGreater(float(terms["inlet_enthalpy_flux_MW"]), 0.0)
        self.assertTrue(np.isfinite(float(terms["outlet_enthalpy_extraction_ratio"])))
        self.assertGreater(float(terms["outlet_enthalpy_extraction_ratio"]), 0.0)
        self.assertAlmostEqual(
            float(terms["raw_design_score"]),
            float(terms["outlet_enthalpy_extraction_percent"]),
            delta=max(1e-10, 1e-8 * abs(float(terms["outlet_enthalpy_extraction_percent"]))),
        )
        self.assertAlmostEqual(
            float(result.objective_score),
            float(terms["raw_design_score"]) - float(terms["velikhov_margin_penalty"]),
            delta=max(1e-10, 1e-8 * abs(float(result.objective_score))),
        )

    def test_implicit_reference_seed_is_feasible(self):
        seed = BaselineSeed.from_summary(BASELINE_SUMMARY)
        _, result, _ = _build_implicit_reference(baseline=seed, n_intervals=40)
        self.assertTrue(result.diagnostics["acceptable"])
        self.assertLess(float(result.diagnostics["max_eq_residual"]), 1e-9)
        self.assertLess(float(result.diagnostics["max_ineq_residual"]), 0.0)

    def test_reduced_implicit_model_keeps_maingo_dimension_low(self):
        seed = BaselineSeed.from_summary(BASELINE_SUMMARY)
        model = _MAiNGOHybridReducedImplicitModelBase(
            baseline=seed,
            n_intervals=4,
            maingopy_module=_import_maingopy(),
            newton_steps=10,
        )
        self.assertEqual(model.total_variables, 8)
        self.assertEqual(len(model.get_variables()), 8)
        self.assertEqual(len(model.get_initial_point()), 8)
        solution = model.decode_solution_point(model.get_initial_point())
        result = model.evaluate_solution(solution)
        self.assertTrue(result.diagnostics["finite_profile"])
        self.assertLess(float(result.diagnostics["max_eq_residual"]), 1e-2)
        self.assertFalse(result.diagnostics["critical_mode"])

    def test_reduced_implicit_critical_mode_adds_one_sonic_variable(self):
        seed = BaselineSeed.from_summary(BASELINE_SUMMARY)
        model = _MAiNGOHybridReducedImplicitModelBase(
            baseline=seed,
            n_intervals=4,
            maingopy_module=_import_maingopy(),
            newton_steps=10,
            critical_mode=True,
        )
        self.assertEqual(model.total_variables, 9)
        self.assertEqual(len(model.get_variables()), 9)
        self.assertEqual(len(model.get_initial_point()), 9)
        self.assertGreaterEqual(float(model.get_initial_point()[-1]), 0.0)
        self.assertLessEqual(float(model.get_initial_point()[-1]), float(seed.L))
        solution = model.decode_solution_point(model.get_initial_point())
        result = model.evaluate_solution(solution)
        self.assertTrue(result.diagnostics["critical_mode"])
        self.assertIn("critical_x_sonic", result.diagnostics)
        self.assertTrue(np.isfinite(float(result.diagnostics["critical_x_sonic"])))
        self.assertTrue(np.isfinite(float(result.diagnostics["critical_max_abs_residual"])))
        self.assertGreaterEqual(float(result.diagnostics["critical_max_gate"]), 0.0)
        self.assertLessEqual(float(result.diagnostics["critical_max_gate"]), 1.0)
        evaluation = model.evaluate(model.get_initial_point())
        self.assertIsNotNone(evaluation.objective)


if __name__ == "__main__":
    unittest.main()
