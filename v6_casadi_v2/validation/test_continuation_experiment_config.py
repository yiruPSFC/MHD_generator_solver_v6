from __future__ import annotations

import unittest

import numpy as np

from v6_casadi_v2.run_casadi_continuation_v2 import (
    _canonicalize_inlet_windows,
    _inlet_problem_from_values,
    _stage_inlet_problem,
)
from v6_casadi_v2.runners.run_baseline_release_from_v6_sweep_v2 import _build_stage_schedule


class ContinuationExperimentConfigTest(unittest.TestCase):
    def _base_windows(self):
        return _inlet_problem_from_values(
            n_p_in_guess=10.0,
            n_p_in_min=8.0,
            n_p_in_max=12.0,
            T_e_in_guess=100.0,
            T_e_in_min=90.0,
            T_e_in_max=120.0,
            Z_in_guess=3.0,
            Z_in_min=2.0,
            Z_in_max=4.0,
            J_x_in_guess=5.0,
            J_x_in_min=4.0,
            J_x_in_max=6.0,
            seed_fraction_guess=1e-4,
            seed_fraction_min=1e-5,
            seed_fraction_max=1e-3,
        )

    def test_canonicalizes_legacy_inlet_keys(self):
        windows = _canonicalize_inlet_windows(
            {
                "np_in": {"guess": 10.0, "min": 8.0, "max": 12.0},
                "te_in": {"guess": 100.0, "min": 90.0, "max": 120.0},
                "z_in": {"guess": 3.0, "min": 2.0, "max": 4.0},
                "jx_in": {"guess": 5.0, "min": 4.0, "max": 6.0},
                "seed_fraction": {"guess": 1e-4, "min": 1e-5, "max": 1e-3},
            }
        )
        self.assertEqual(set(windows), {"n_p_in", "T_e_in", "Z_in", "I_0", "seed_fraction"})
        self.assertEqual(windows["I_0"]["guess"], 5.0)

    def test_stage_window_override_applies_to_one_stage(self):
        base = self._base_windows()
        stage = {
            "name": "stage",
            "inlet_windows": {
                "z_in": {"guess": 3.2, "min": 3.0, "max": 3.5},
            },
        }
        resolved = _stage_inlet_problem(stage, base)
        self.assertEqual(resolved["Z_in"]["guess"], 3.2)
        self.assertEqual(resolved["n_p_in"]["guess"], 10.0)

    def test_staged_release_pins_unreleased_variables(self):
        source_schedule = [
            {
                "name": "anchor",
                "n_intervals": 4,
                "transcription": "hermite-simpson",
                "A_max_ratio": 2.0,
                "max_abs_dlogA_dx": 0.1,
                "ipopt_max_iter": 10,
            },
            {
                "name": "release",
                "n_intervals": 4,
                "transcription": "hermite-simpson",
                "A_max_ratio": 2.0,
                "max_abs_dlogA_dx": 0.1,
                "objective_weight": 0.01,
                "ipopt_max_iter": 10,
            },
            {
                "name": "final",
                "n_intervals": 4,
                "transcription": "hermite-simpson",
                "A_max_ratio": 2.5,
                "max_abs_dlogA_dx": 0.2,
                "objective_weight": 0.1,
                "ipopt_max_iter": 10,
            },
        ]
        stages = _build_stage_schedule(
            source_schedule=source_schedule,
            selected_sigma=0.2,
            warm_A=np.array([1.0, 1.1, 1.2]),
            inlet_windows=self._base_windows(),
            schedule_mode="staged-release",
            pin_relative_radius=1e-6,
            anchor_sigma_extra=0.05,
            anchor_warm_profile_track_weight=1.0,
            anchor_warm_control_track_weight=1.0,
            anchor_ipopt_max_iter=10,
            release_warm_profile_track_weight=1.0,
            release_warm_control_track_weight=1.0,
            release_ipopt_max_iter=10,
            final_warm_profile_track_weight=1.0,
            final_warm_control_track_weight=1.0,
            final_ipopt_max_iter=10,
        )
        self.assertEqual([stage["release_group"] for stage in stages], [
            "baseline_anchor",
            "release_n_p_T_e",
            "release_Z",
            "release_I0_seed_fraction",
            "release_geometry_objective",
        ])
        release_np_te = stages[1]["inlet_windows"]
        self.assertEqual(release_np_te["n_p_in"]["min"], 8.0)
        self.assertLess(release_np_te["Z_in"]["max"] - release_np_te["Z_in"]["min"], 1e-4)


if __name__ == "__main__":
    unittest.main()
