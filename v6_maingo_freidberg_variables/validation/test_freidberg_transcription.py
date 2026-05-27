from __future__ import annotations

from pathlib import Path
import unittest

from v6_maingo_freidberg_variables.algebra import profile_to_freidberg_arrays
from v6_maingo_freidberg_variables.models import FreidbergConfig
from v6_maingo_freidberg_variables.rollout import compare_rollout_to_reference, integrate_fixed_te
from v6_maingo_freidberg_variables.transcription import transcription_summary_from_profile


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


class FreidbergTranscriptionTest(unittest.TestCase):
    def test_41_percent_interval_defects_are_small(self) -> None:
        report = transcription_summary_from_profile(profile_path=PROFILE_41, summary_path=SUMMARY_41)
        summary = report["freidberg_interval_defects"]["summary"]
        self.assertLess(abs(summary["terminal_H_defect_MW"]), 0.1)
        self.assertLess(summary["max_abs_H_defect_MW"], 0.02)
        self.assertLess(abs(summary["terminal_L_defect"]), 0.005)
        self.assertLess(summary["max_abs_L_defect"], 5e-4)

    def test_147_percent_interval_defects_are_large(self) -> None:
        report = transcription_summary_from_profile(profile_path=PROFILE_147, summary_path=SUMMARY_147)
        summary = report["freidberg_interval_defects"]["summary"]
        self.assertGreater(abs(summary["terminal_H_defect_MW"]), 1.0)
        self.assertGreater(summary["max_abs_H_defect_MW"], 0.1)
        self.assertGreater(abs(summary["terminal_L_defect"]), 0.02)

    def test_short_fixed_te_rollout_stays_near_41_percent_profile(self) -> None:
        config = FreidbergConfig.from_summary_and_profile(SUMMARY_41, PROFILE_41)
        arrays = profile_to_freidberg_arrays(PROFILE_41, config)
        n = 8
        result = integrate_fixed_te(
            x=arrays["x"][:n],
            T_e=arrays["T_e"][:n],
            H_p0=float(arrays["H_p"][0]),
            L_p0=float(arrays["L_p"][0]),
            config=config,
            mach_hints=arrays["mach"][:n],
            method="heun",
            branch="subsonic",
            closure_tolerance_K=1e-2,
        )
        comparison = compare_rollout_to_reference(
            result,
            H_p_reference=arrays["H_p"][:n],
            L_p_reference=arrays["L_p"][:n],
            config=config,
        )
        self.assertTrue(result.succeeded)
        self.assertLess(comparison["max_abs_H_error_MW"], 0.05)
        self.assertLess(comparison["max_abs_L_error"], 0.001)


if __name__ == "__main__":
    unittest.main()
