from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from v6_firedrake_reduced.design import load_case_config

from v6_active_boundary_reduced.sonic_delta_profile import (
    SonicDeltaSettings,
    build_sonic_delta_profile,
    primitive_sonic_compatibility,
)


def test_sonic_delta_profile_crosses_mach_one_with_g_admissible_nodes() -> None:
    config = load_case_config(case="freidberg_reference")
    payload = build_sonic_delta_profile(
        config=config,
        settings=SonicDeltaSettings(
            dx=1.0e-4,
            n_steps_each_side=1,
            scan_points=15,
        ),
    )

    assert payload["ok"]
    mach = np.asarray(payload["profile_arrays"]["mach"], dtype=float)
    G = np.asarray(payload["profile_arrays"]["G"], dtype=float)
    assert float(np.nanmin(mach)) < 1.0
    assert float(np.nanmax(mach)) > 1.0
    assert float(np.nanmin(G)) >= -1.0e-7

    sonic = dict(payload["sonic_primitive_compatibility"])
    assert abs(float(sonic["mach"]) - 1.0) <= 1.0e-6
    assert abs(float(sonic["compatibility_scaled_residual"])) <= 1.0e-10


def test_primitive_sonic_compatibility_uses_primitive_left_null_condition() -> None:
    config = load_case_config(case="freidberg_reference")
    payload = build_sonic_delta_profile(
        config=config,
        settings=SonicDeltaSettings(
            dx=1.0e-4,
            n_steps_each_side=0,
            scan_points=5,
        ),
    )
    node = dict(payload["nodes"][0])
    state = payload["sonic_seed"]
    assert state["source"] == "v6_firedrake_reduced.solve_local_sonic_match"

    from v6_active_boundary_reduced.policy import State

    check = primitive_sonic_compatibility(
        State(log_n=float(node["log_n"]), log_Te=float(node["log_Te"]), logA=float(node["logA"])),
        config=config,
    )
    assert np.isfinite(float(check["sigma_sonic"]))
    assert abs(float(check["compatibility_scaled_residual"])) <= 1.0e-10


def test_main_policy_uses_explicit_sonic_branch_near_choking() -> None:
    config = load_case_config(case="freidberg_reference")
    payload = build_sonic_delta_profile(
        config=config,
        settings=SonicDeltaSettings(
            dx=1.0e-4,
            n_steps_each_side=0,
            scan_points=5,
        ),
    )
    node = dict(payload["nodes"][0])
    sonic = dict(payload["sonic_primitive_compatibility"])

    from v6_active_boundary_reduced.policy import AnchorState, PolicySettings, State, rollout_policy_from_anchor

    anchor_state = State(
        log_n=float(node["log_n"]),
        log_Te=float(node["log_Te"]),
        logA=float(node["logA"]),
    )
    result = rollout_policy_from_anchor(
        config=config,
        anchor=AnchorState(
            state=anchor_state,
            sigma_logA=float(sonic["sigma_sonic"]),
            source="sonic_delta_profile_test",
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
    from v6_active_boundary_reduced.policy import PolicySettings, _choose_sonic_sigma

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


def test_branch_agnostic_pedal_mode_does_not_force_mach_side() -> None:
    config = load_case_config(case="freidberg_reference")
    payload = build_sonic_delta_profile(
        config=config,
        settings=SonicDeltaSettings(
            dx=1.0e-5,
            n_steps_each_side=1,
            scan_points=15,
            objective="pedal",
            selection_mode="steepest",
            branch_mode="agnostic",
        ),
    )

    assert payload["ok"]
    summary = dict(payload["active_summary"])
    assert summary["branch_mode"] == "agnostic"
    assert int(summary["reverse_direction_violation_count"]) == 0
    assert int(summary["forward_direction_violation_count"]) == 0

    segments = list(payload["segments"])
    reverse_step = next(item for item in segments if float(item["x_next"]) < float(item["x_current"]))
    forward_step = next(item for item in segments if float(item["x_next"]) > float(item["x_current"]))
    assert float(reverse_step["delta_change"]) < 0.0
    assert float(forward_step["delta_change"]) > 0.0
    assert reverse_step["selected_mach_branch"] == "subsonic"
    assert forward_step["selected_mach_branch"] == "supersonic"
    assert not bool(reverse_step["branch_filter_enabled"])
    assert not bool(forward_step["branch_filter_enabled"])


if __name__ == "__main__":
    test_sonic_delta_profile_crosses_mach_one_with_g_admissible_nodes()
    test_primitive_sonic_compatibility_uses_primitive_left_null_condition()
    test_main_policy_uses_explicit_sonic_branch_near_choking()
    test_sonic_compatibility_choice_classifies_degenerate_cases()
    test_branch_agnostic_pedal_mode_does_not_force_mach_side()
    print("sonic delta profile smoke passed")
