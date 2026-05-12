from __future__ import annotations

from pathlib import Path
import unittest

from v6_maingo_mach_spline.shadow import profile_mach_spline_summary


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


class MachSplineClosureTest(unittest.TestCase):
    def assert_exact_mach_recovers_profile(self, report: dict) -> None:
        exact = report["exact_mach_reconstruction"]
        self.assertLess(exact["max_rel_A_error"], 1e-8)
        self.assertLess(exact["max_abs_T_p_error_K"], 1e-3)
        self.assertLess(exact["max_rel_v_p_error"], 1e-8)
        self.assertLess(exact["max_abs_Z_error"], 1e-6)

    def test_exact_mach_closure_recovers_41_percent_profile(self) -> None:
        report = profile_mach_spline_summary(profile_path=PROFILE_41, summary_path=SUMMARY_41)
        self.assert_exact_mach_recovers_profile(report)
        fit = report["fitted_mach_spline_reconstruction"]
        self.assertLess(fit["max_rel_mach_error"], 0.03)
        self.assertLess(fit["max_rel_A_error"], 0.03)

    def test_exact_mach_closure_recovers_147_percent_profile(self) -> None:
        report = profile_mach_spline_summary(profile_path=PROFILE_147, summary_path=SUMMARY_147)
        self.assert_exact_mach_recovers_profile(report)
        self.assertGreater(abs(report["freidberg_interval_defects_original"]["terminal_H_defect_MW"]), 1.0)

    def test_mach_spline_preserves_41_percent_freidberg_scale(self) -> None:
        report = profile_mach_spline_summary(profile_path=PROFILE_41, summary_path=SUMMARY_41)
        original = report["freidberg_interval_defects_original"]
        fitted = report["freidberg_interval_defects_fitted_mach"]
        self.assertLess(abs(original["terminal_H_defect_MW"]), 0.1)
        self.assertLess(abs(fitted["terminal_H_defect_MW"]), 0.2)
        self.assertLess(fitted["max_abs_H_defect_MW"], 0.03)


if __name__ == "__main__":
    unittest.main()
