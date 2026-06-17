from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from v6_active_boundary_reduced.core.policy import (
    AnchorState,
    PolicySettings,
    State,
    _choose_sonic_sigma,
    _primitive_sonic_compatibility,
    rollout_policy_from_anchor,
)
from v6_firedrake_reduced.design import load_case_config
from v6_firedrake_reduced.sonic_compatibility import solve_local_sonic_match


def _freidberg_sonic_state():
    config = load_case_config(case="freidberg_reference")
    match = solve_local_sonic_match(design=config.design, config=config)
    sonic = dict(match.get("sonic_point") or {})
    assert sonic, "solve_local_sonic_match did not return a sonic_point"
    state = State(
        log_n=float(np.log(max(float(sonic["n_p"]), 1.0e-300))),
        log_Te=float(np.log(max(float(sonic["T_e"]), 1.0))),
        logA=float(sonic["logA"]),
    )
    return config, state


def test_shared_primitive_sonic_compatibility_uses_left_null_condition() -> None:
    config, state = _freidberg_sonic_state()
    check = _primitive_sonic_compatibility(state, config=config)

    assert check["ok"]
    assert abs(float(check["mach"]) - 1.0) <= 1.0e-6
    assert np.isfinite(float(check["sigma_sonic"]))
    assert abs(float(check["compatibility_scaled_residual"])) <= 1.0e-10


def test_main_policy_uses_explicit_sonic_branch_near_choking() -> None:
    config, anchor_state = _freidberg_sonic_state()
    sonic = _primitive_sonic_compatibility(anchor_state, config=config)

    result = rollout_policy_from_anchor(
        config=config,
        anchor=AnchorState(
            state=anchor_state,
            sigma_logA=float(sonic["sigma_sonic"]),
            source="sonic_policy_test",
            source_index=-1,
        ),
        settings=PolicySettings(
            direction="reverse",
            objective="delta_drop",
            n_steps=1,
            sigma_min=-2.0,
            sigma_max=2.0,
            curvature_max=None,
            sonic_mode="on",
        ),
        dx=1.0e-4,
    )

    assert result["ok"]
    segment = dict(result["segments"][0])
    assert bool(segment["sonic_branch_used"])
    assert segment["solver_method"] == "sonic_left_null_explicit_A_prime"
    assert segment["support_type"] == "sonic_compatible_left_null"
    assert abs(float(segment["sigma"]) - float(sonic["sigma_sonic"])) <= 1.0e-12
    assert float(segment["delta_gain"]) < 0.0
    assert float(segment["constraint_margins"]["G"]) >= -1.0e-6
    assert segment["step_error_kind"] == "physical_residual"
    assert bool(segment["physical_residual_ok"])
    assert np.isfinite(float(segment["physical_residual_scaled"]))


def test_sonic_compatibility_choice_classifies_degenerate_cases() -> None:
    settings = PolicySettings(sonic_compatibility_tol=1.0e-6, active_tol=1.0e-9)
    root = _choose_sonic_sigma(
        sonic={"ellTf0": -2.0, "ellTf1": 1.0},
        area=1.0,
        lo=0.0,
        hi=4.0,
        sigma_reference=0.5,
        settings=settings,
    )
    assert root["ok"]
    assert root["status"] == "root_in_interval"
    assert abs(float(root["sigma"]) - 2.0) <= 1.0e-12

    flat = _choose_sonic_sigma(
        sonic={"ellTf0": 1.0e-12, "ellTf1": 0.0},
        area=1.0,
        lo=-1.0,
        hi=1.0,
        sigma_reference=0.25,
        settings=settings,
    )
    assert flat["ok"]
    assert flat["status"] == "flat_compatible"
    assert abs(float(flat["sigma"]) - 0.25) <= 1.0e-12

    flat_forcing = _choose_sonic_sigma(
        sonic={"ellTf0": 1.0, "ellTf1": 0.0},
        area=1.0,
        lo=-1.0,
        hi=1.0,
        sigma_reference=0.25,
        settings=settings,
    )
    assert not flat_forcing["ok"]
    assert flat_forcing["status"] == "unreachable_flat_forcing"

    outside = _choose_sonic_sigma(
        sonic={"ellTf0": -10.0, "ellTf1": 1.0},
        area=1.0,
        lo=0.0,
        hi=1.0,
        sigma_reference=0.5,
        settings=settings,
    )
    assert not outside["ok"]
    assert outside["status"] == "unreachable_interval"
    assert abs(float(outside["best_interval_sigma"]) - 1.0) <= 1.0e-12


if __name__ == "__main__":
    test_shared_primitive_sonic_compatibility_uses_left_null_condition()
    test_main_policy_uses_explicit_sonic_branch_near_choking()
    test_sonic_compatibility_choice_classifies_degenerate_cases()
    print("sonic policy smoke passed")
