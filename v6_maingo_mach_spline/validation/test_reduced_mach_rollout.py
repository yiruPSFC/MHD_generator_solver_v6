from __future__ import annotations

from pathlib import Path
import unittest

from v6_maingo_mach_spline.reduced_implicit import MACH_DECISION_NAMES, rollout_summary_from_profile


REPO = Path(__file__).resolve().parents[2]

CASE_41 = REPO / "v6_maingo_casadi/outputs/maingo_yamasaki2004_neighborhood/tp_midpoint_semantics_smoke_20260511_v3"
PROFILE_41 = CASE_41 / "maingo_best_profile.npz"
SUMMARY_41 = CASE_41 / "maingo_summary.json"

CASE_147 = (
    REPO
    / "v6_maingo_casadi/outputs/maingo_yamasaki2004_neighborhood/"
    "direct_spline_expanded_coarse10_dense80_bab_from_41175_mach0_newton8_short_20260511"
)
PROFILE_147 = CASE_147 / "maingo_handoff_profile.npz"
SUMMARY_147 = CASE_147 / "maingo_summary.json"


class ReducedMachRolloutTest(unittest.TestCase):
    def test_decision_vector_replaces_area_coefficients_with_mach_coefficients(self) -> None:
        self.assertEqual(MACH_DECISION_NAMES[-3:], ("m1", "m2", "m3"))
        self.assertNotIn("a1", MACH_DECISION_NAMES)
        self.assertNotIn("a2", MACH_DECISION_NAMES)
        self.assertNotIn("a3", MACH_DECISION_NAMES)

    def test_41_percent_reference_rollout_is_stable(self) -> None:
        report = rollout_summary_from_profile(profile_path=PROFILE_41, summary_path=SUMMARY_41, newton_steps=10)
        self.assertLess(report["max_abs_scaled_residual"], 1e-8)
        self.assertLess(report["max_rel_mach_error"], 0.03)
        self.assertLess(report["max_rel_A_error"], 0.10)
        self.assertLess(report["max_abs_T_e_error_K"], 250.0)

    def test_147_percent_reference_rollout_drifts_strongly(self) -> None:
        report = rollout_summary_from_profile(profile_path=PROFILE_147, summary_path=SUMMARY_147, newton_steps=10)
        self.assertLess(report["max_abs_scaled_residual"], 1e-8)
        self.assertGreater(report["max_abs_T_e_error_K"], 1000.0)
        self.assertGreater(report["max_rel_A_error"], 0.5)


if __name__ == "__main__":
    unittest.main()
