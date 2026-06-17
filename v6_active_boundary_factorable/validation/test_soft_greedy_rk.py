from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from v6_active_boundary_factorable.soft_greedy_rk import (
    FactorableState,
    SoftGreedySettings,
    factorable_params_from_config,
    finite_numeric,
    initial_state_from_design,
    rollout_soft_greedy,
    soft_greedy_step,
    sonic_sigma_chart,
)
from v6_active_boundary_reduced.core.policy import State, _primitive_sonic_compatibility
from v6_firedrake_reduced.design import load_case_config
from v6_firedrake_reduced.sonic_compatibility import solve_local_sonic_match
from v6_maingo_casadi.numerics import _ops_for_numeric


def _sonic_state_and_reference():
    config = load_case_config(case="freidberg_reference")
    match = solve_local_sonic_match(design=config.design, config=config)
    sonic = dict(match.get("sonic_point") or {})
    assert sonic, "solve_local_sonic_match did not return a sonic_point"
    policy_state = State(
        log_n=float(np.log(max(float(sonic["n_p"]), 1.0e-300))),
        log_Te=float(np.log(max(float(sonic["T_e"]), 1.0))),
        logA=float(sonic["logA"]),
    )
    state = FactorableState(log_n=policy_state.log_n, log_Te=policy_state.log_Te, logA=policy_state.logA)
    reference = _primitive_sonic_compatibility(policy_state, config=config)
    return config, state, reference


def test_sonic_chart_exposes_unscaled_svd_left_null_mismatch() -> None:
    config, state, reference = _sonic_state_and_reference()
    params = factorable_params_from_config(config)
    chart = sonic_sigma_chart(
        state=state,
        params=params,
        settings=SoftGreedySettings(sonic_den_eps=1.0e-18),
    )

    sigma_chart = float(chart["sigma_sonic"])
    sigma_reference = float(reference["sigma_sonic"])
    assert np.isfinite(sigma_chart)
    # This mismatch is intentional: the current active-boundary reference uses
    # an unscaled SVD on a severely row-scaled primitive matrix.  The factorable
    # chart uses the algebraic row-null condition and exposes that scaling issue.
    assert abs(sigma_chart - sigma_reference) > 1.0e-2
    assert abs(float(chart["compat1_scaled"])) <= 1.0e-8
    assert abs(float(chart["compat2_scaled"])) <= 1.0e-8


def test_soft_selector_is_factorable_and_approaches_endpoint() -> None:
    config = load_case_config(case="freidberg_reference")
    params = factorable_params_from_config(config)
    state = initial_state_from_design(config.design)
    wide = SoftGreedySettings(
        dx=1.0e-3,
        selector_eps=1.0e-12,
        sigma_min=-0.25,
        sigma_max=0.75,
        use_mach_gate=False,
        det_gate_eps=1.0e-12,
    )
    segment = soft_greedy_step(state=state, sigma_prev=0.0, params=params, settings=wide)
    sigma = float(segment["sigma"])
    lo = float(segment["sigma_lo"])
    hi = float(segment["sigma_hi"])
    assert lo <= sigma <= hi
    preference = float(segment["preference"])
    if preference > 0.0:
        assert abs(sigma - hi) <= 1.0e-6
    else:
        assert abs(sigma - lo) <= 1.0e-6


def test_sonic_gate_keeps_rhs_finite_at_choking_seed() -> None:
    config, state, _ = _sonic_state_and_reference()
    params = factorable_params_from_config(config)
    segment = soft_greedy_step(
        state=state,
        sigma_prev=0.0,
        params=params,
        settings=SoftGreedySettings(
            dx=1.0e-5,
            sigma_min=-2.0,
            sigma_max=2.0,
            curvature_max=None,
            det_gate_eps=1.0e-1,
            mach_gate_eps=1.0e-2,
            rhs_det_eps=1.0e-5,
            sonic_den_eps=1.0e-18,
        ),
    )

    assert float(segment["sonic_gate"]) > 0.95
    for key in ("sigma", "dn_dx", "dTe_dx", "next_G", "step_momentum", "step_energy"):
        assert finite_numeric(segment[key]), key
    assert abs(float(segment["compat1_scaled"])) <= 1.0e-7
    assert abs(float(segment["compat2_scaled"])) <= 1.0e-7


def test_short_forward_rollout_reports_expected_diagnostics() -> None:
    config = load_case_config(case="freidberg_reference")
    params = factorable_params_from_config(config)
    payload = rollout_soft_greedy(
        state=initial_state_from_design(config.design),
        sigma_initial=0.0,
        params=params,
        settings=SoftGreedySettings(
            dx=1.0e-4,
            n_steps=4,
            sigma_min=-0.5,
            sigma_max=0.5,
            det_gate_eps=1.0e-3,
            mach_gate_eps=1.0e-3,
            rhs_det_eps=1.0e-4,
        ),
    )

    assert len(payload["segments"]) == 4
    required = {
        "sigma_greedy",
        "sigma_sonic",
        "sonic_gate",
        "det",
        "mach",
        "G",
        "next_G",
        "step_momentum",
        "step_energy",
        "step_momentum_scaled",
        "step_energy_scaled",
    }
    for segment in payload["segments"]:
        assert required.issubset(segment)
        for key in required:
            assert finite_numeric(segment[key]), key


if __name__ == "__main__":
    test_sonic_chart_exposes_unscaled_svd_left_null_mismatch()
    test_soft_selector_is_factorable_and_approaches_endpoint()
    test_sonic_gate_keeps_rhs_finite_at_choking_seed()
    test_short_forward_rollout_reports_expected_diagnostics()
    print("soft-greedy factorable prototype smoke passed")
