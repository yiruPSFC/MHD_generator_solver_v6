from __future__ import annotations

import math
from pathlib import Path
import unittest

import numpy as np

from v6_maingo_casadi.numerics import _ops_for_numeric
from v6_maingo_casadi.physics import _inlet_design_generic
from v6_maingo_casadi.profiles import _DEFAULT_WORKING_FLUID_PROFILE
from v6_maingo_mach_spline.reduced_implicit import (
    MACH_DECISION_NAMES,
    _scaled_momentum_energy_residuals_mach,
    _scaled_momentum_energy_residuals_mach_log_jacobian,
    rollout_summary_from_profile,
)


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

    @unittest.skipUnless(PROFILE_41.exists() and SUMMARY_41.exists(), "requires stored MAiNGO output profile")
    def test_41_percent_reference_rollout_is_stable(self) -> None:
        report = rollout_summary_from_profile(profile_path=PROFILE_41, summary_path=SUMMARY_41, newton_steps=10)
        self.assertEqual(report["jacobian_mode"], "analytic")
        self.assertLess(report["max_abs_scaled_residual"], 1e-8)
        self.assertLess(report["max_rel_mach_error"], 0.03)
        self.assertLess(report["max_rel_A_error"], 0.10)
        self.assertLess(report["max_abs_T_e_error_K"], 250.0)

    @unittest.skipUnless(PROFILE_147.exists() and SUMMARY_147.exists(), "requires stored MAiNGO output profile")
    def test_147_percent_reference_rollout_drifts_strongly(self) -> None:
        report = rollout_summary_from_profile(profile_path=PROFILE_147, summary_path=SUMMARY_147, newton_steps=10)
        self.assertLess(report["max_abs_scaled_residual"], 1e-8)
        self.assertGreater(report["max_abs_T_e_error_K"], 1000.0)
        self.assertGreater(report["max_rel_A_error"], 0.5)

    def test_analytic_log_state_jacobian_matches_central_difference(self) -> None:
        ops = _ops_for_numeric()
        n_prev = 1.0e21
        T_e_prev = 5000.0
        A_prev = 0.2
        I_0 = 100.0
        seed_fraction = 1.0e-2
        B = 2.0
        inlet = _inlet_design_generic(
            ops=ops,
            n_p_in=n_prev,
            T_e_in=T_e_prev,
            Z_in=0.0,
            I_0=I_0,
            seed_fraction=seed_fraction,
            B=B,
            inlet_A=A_prev,
            working_fluid=_DEFAULT_WORKING_FLUID_PROFILE,
        )
        n_next = 0.995e21
        T_e_next = 5050.0
        log_n = math.log(n_next)
        log_Te = math.log(T_e_next)
        args = dict(
            n_prev=n_prev,
            T_e_prev=T_e_prev,
            A_prev=A_prev,
            mach_next=float(inlet["mach"]) * 1.03,
            dot_N=float(inlet["dot_N"]),
            I_0=I_0,
            seed_fraction=seed_fraction,
            B=B,
            dx=0.1,
            momentum_scale=1.0,
            energy_scale=1.0,
            working_fluid=_DEFAULT_WORKING_FLUID_PROFILE,
        )

        _, _, j11, j12, j21, j22 = _scaled_momentum_energy_residuals_mach_log_jacobian(
            ops=ops,
            log_n_next=log_n,
            log_Te_next=log_Te,
            **args,
        )

        def residuals(candidate_log_n: float, candidate_log_Te: float) -> np.ndarray:
            r_m, r_e, *_ = _scaled_momentum_energy_residuals_mach(
                ops=ops,
                n_next=math.exp(candidate_log_n),
                T_e_next=math.exp(candidate_log_Te),
                **args,
            )
            return np.asarray([r_m, r_e], dtype=float)

        h = 1e-5
        fd = np.column_stack(
            (
                (residuals(log_n + h, log_Te) - residuals(log_n - h, log_Te)) / (2.0 * h),
                (residuals(log_n, log_Te + h) - residuals(log_n, log_Te - h)) / (2.0 * h),
            )
        )
        analytic = np.asarray([[j11, j12], [j21, j22]], dtype=float)
        rel = np.abs(analytic - fd) / np.maximum(1.0, np.abs(fd))
        self.assertLess(float(np.max(rel)), 1e-5)


if __name__ == "__main__":
    unittest.main()
