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


if __name__ == "__main__":
    test_sonic_delta_profile_crosses_mach_one_with_g_admissible_nodes()
    test_primitive_sonic_compatibility_uses_primitive_left_null_condition()
    print("sonic delta profile smoke passed")
