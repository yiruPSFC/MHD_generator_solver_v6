from __future__ import annotations

from pathlib import Path
import unittest

from v6_maingo_freidberg_variables.audit import audit_profile
from v6_maingo_freidberg_variables.models import FreidbergConfig, PrimitivePoint
from v6_maingo_freidberg_variables.rhs import freidberg_rhs


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


class FreidbergAlgebraBridgeTest(unittest.TestCase):
    def assert_bridge_is_tight(self, report: dict) -> None:
        bridge = report["bridge"]
        self.assertLess(bridge["max_abs_closure_residual_K"], 1e-4)
        self.assertLess(bridge["max_abs_mach_error"], 1e-8)
        self.assertLess(bridge["max_abs_T_p_error_K"], 1e-3)
        self.assertLess(bridge["max_rel_A_error"], 1e-8)
        self.assertLess(bridge["max_rel_n_p_error"], 1e-8)
        self.assertLess(bridge["max_rel_v_p_error"], 1e-8)
        self.assertLess(bridge["max_abs_Z_error"], 1e-6)

    def test_round_trip_41_percent_profile(self) -> None:
        report = audit_profile(profile_path=PROFILE_41, summary_path=SUMMARY_41)
        self.assert_bridge_is_tight(report)
        self.assertLess(abs(report["freidberg_H"]["residual_MW"]), 0.1)

    def test_round_trip_147_percent_profile(self) -> None:
        report = audit_profile(profile_path=PROFILE_147, summary_path=SUMMARY_147)
        self.assert_bridge_is_tight(report)
        self.assertGreater(report["reported_style_primary_extraction_percent"], 100.0)
        self.assertGreater(abs(report["freidberg_H"]["residual_MW"]), 1.0)

    def test_rhs_finite_at_sonic(self) -> None:
        config = FreidbergConfig.from_summary_and_profile(SUMMARY_41, PROFILE_41)
        point = PrimitivePoint(
            x=0.0,
            n_p=3.0e24,
            T_e=7000.0,
            T_p=6500.0,
            A=config.inlet_area_m2,
            v_p=(5.0 * 1.380649e-23 * 6500.0 / (3.0 * config.working_fluid.heavy_particle_mass_kg)) ** 0.5,
            n_e=2.0e21,
            beta=70.0,
            eta=8.0e-5,
            Z=200.0,
            J_x=2.0e5,
            J_y=-5.0e5,
            E_x=-4.0e3,
            mach=1.0,
        )
        dHdx, dLdx = freidberg_rhs(point, config)
        self.assertTrue(abs(dHdx) < float("inf"))
        self.assertTrue(abs(dLdx) < float("inf"))

    def test_zero_current_rhs_is_constant(self) -> None:
        config = FreidbergConfig.from_summary_and_profile(SUMMARY_41, PROFILE_41)
        point = PrimitivePoint(
            x=0.0,
            n_p=3.0e24,
            T_e=7000.0,
            T_p=6500.0,
            A=config.inlet_area_m2,
            v_p=400.0,
            n_e=2.0e21,
            beta=70.0,
            eta=8.0e-5,
            Z=200.0,
            J_x=0.0,
            J_y=0.0,
            E_x=0.0,
            mach=0.1,
        )
        dHdx, dLdx = freidberg_rhs(point, config)
        self.assertEqual(dHdx, 0.0)
        self.assertEqual(dLdx, 0.0)


if __name__ == "__main__":
    unittest.main()
