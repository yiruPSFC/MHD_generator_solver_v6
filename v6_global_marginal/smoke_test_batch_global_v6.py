from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np

REPO_DIR = Path(__file__).resolve().parents[1]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from v6_global_marginal.global_postprocess_v6 import compute_global_metrics
from v6_core.local_algebraic_closure import SIGMA_EP, local_closure_with_partials
from v6_global_marginal.local_algebraic_closure_global import local_closure_global_with_partials
from v6_batch.pde_solver_v6_batch import _prepare_inlet_constants
from v6_global_marginal.pde_solver_v6_batch_global import (
    ForwardPDESolverV6BatchGlobal,
    _evaluate_state_global,
    project_seed_fraction_to_marginal_inlet,
)


@dataclass
class SmokeCase:
    n_p_in: float
    Z_in: float
    T_p_in: float
    T_e_in: float
    seed_fraction: float
    A_in: float


CASES = [
    SmokeCase(2.9418007053766115e21, 56.32891972495766, 1000.0, 5727.0800768822055, 0.0004799999764588285, 1.0),
    SmokeCase(1.5242875089416995e21, 61.23685251715171, 1000.0, 4793.910217845207, 0.00022630567813314458, 1.0),
]


def main() -> int:
    failures: list[str] = []

    for i, case in enumerate(CASES):
        inlet_status, m0, Jx0 = _prepare_inlet_constants(
            case.n_p_in,
            case.Z_in,
            case.T_p_in,
            case.T_e_in,
            case.seed_fraction,
            0.02,
        )
        if inlet_status != 0:
            failures.append(f"closure case {i}: inlet preparation failed")
            continue

        old_vals = local_closure_with_partials(
            n_p=case.n_p_in,
            T_e=case.T_e_in,
            m_0=m0,
            J_x0=Jx0,
            seed_fraction=case.seed_fraction,
        )
        new_vals = local_closure_global_with_partials(
            n_p=case.n_p_in,
            T_e=case.T_e_in,
            A=case.A_in,
            dot_N=m0 * case.A_in,
            I_0=Jx0 * case.A_in,
            seed_fraction=case.seed_fraction,
        )

        labels = ["v_p", "n_e", "beta", "eta", "Z", "T_p", "dTp_dTe", "dTp_dnp"]
        for j, label in enumerate(labels):
            old = float(old_vals[j])
            new = float(new_vals[j])
            scale = max(abs(old), 1.0)
            err = abs(new - old) / scale
            if err > 5e-10:
                failures.append(f"closure case {i}: {label} mismatch rel={err:.3e}")

        seed_status, projected_seed = project_seed_fraction_to_marginal_inlet(
            n_p_in=case.n_p_in,
            T_p_in=case.T_p_in,
            T_e_in=case.T_e_in,
            B=0.02,
        )
        if seed_status != 0 or (not np.isfinite(projected_seed)) or projected_seed <= 0.0:
            failures.append(f"projection case {i}: invalid projected seed_fraction status={seed_status} value={projected_seed}")

    clipped_cases = [
        # delta_T nearly zero, should clip consistently and stay finite
        (1.0e23, 5.0e3, 2.0, 1.0e20, 1.0, 5.0e-2),
        # f_I nearly one, should clip consistently and stay finite
        (1.0e20, 1.0e5, 5.0, 1.0e17, 1.0e-2, 1.0e-4),
    ]
    for k, (n_p, T_e, A, dot_N, I_0, seed_fraction) in enumerate(clipped_cases):
        clipped = local_closure_global_with_partials(
            n_p=n_p,
            T_e=T_e,
            A=A,
            dot_N=dot_N,
            I_0=I_0,
            seed_fraction=seed_fraction,
        )
        for idx, label in ((13, "G"), (14, "G_np"), (15, "G_Te"), (16, "G_A")):
            if not np.isfinite(float(clipped[idx])):
                failures.append(f"clipped closure case {k} returned non-finite {label}")

    inlet_status, m0, Jx0 = _prepare_inlet_constants(
        CASES[0].n_p_in,
        CASES[0].Z_in,
        CASES[0].T_p_in,
        CASES[0].T_e_in,
        CASES[0].seed_fraction,
        0.02,
    )
    if inlet_status == 0:
        bad_tp = local_closure_global_with_partials(
            n_p=CASES[0].n_p_in,
            T_e=CASES[0].T_e_in,
            A=1.0e-6,
            dot_N=m0,
            I_0=Jx0,
            seed_fraction=CASES[0].seed_fraction,
        )
        if np.isfinite(float(bad_tp[13])) or np.isfinite(float(bad_tp[14])) or np.isfinite(float(bad_tp[15])) or np.isfinite(float(bad_tp[16])):
            failures.append("negative-Tp closure path should return non-finite G partials")

        invalid_state = _evaluate_state_global(
            CASES[0].n_p_in,
            CASES[0].T_e_in,
            -1.0,
            m0,
            Jx0,
            CASES[0].seed_fraction,
            0.02,
            SIGMA_EP,
        )
        if int(invalid_state[14]) == 0:
            failures.append("negative-area state should be rejected")

    solver = ForwardPDESolverV6BatchGlobal(B=0.02, length=1e-8)
    inlet = solver.evaluate_inlet_batch(
        n_p_in=np.array([c.n_p_in for c in CASES], dtype=float),
        Z_in=np.array([c.Z_in for c in CASES], dtype=float),
        T_p_in=np.array([c.T_p_in for c in CASES], dtype=float),
        T_e_in=np.array([c.T_e_in for c in CASES], dtype=float),
        A_in=np.array([c.A_in for c in CASES], dtype=float),
    )
    if not np.all(inlet.success):
        failures.append(f"inlet metrics failures: success={inlet.success.tolist()} event={inlet.event_code.tolist()}")
    if not np.all(np.isfinite(inlet.seed_fraction)) or not np.all(inlet.seed_fraction > 0.0):
        failures.append(f"inlet metrics returned invalid projected seed_fraction {inlet.seed_fraction.tolist()}")

    short = solver.solve_batch(
        n_p_in=np.array([CASES[0].n_p_in], dtype=float),
        Z_in=np.array([CASES[0].Z_in], dtype=float),
        T_p_in=np.array([CASES[0].T_p_in], dtype=float),
        T_e_in=np.array([CASES[0].T_e_in], dtype=float),
        A_in=np.array([CASES[0].A_in], dtype=float),
        dx=1e-8,
        store_profiles=True,
    )
    if int(short.valid_points[0]) < 1:
        failures.append("short global solve returned no valid points")
    if not np.isfinite(float(short.T_p[0, 0])):
        failures.append("short global solve produced non-finite inlet Tp")

    metrics = compute_global_metrics(
        x=np.array([0.0, 1.0], dtype=float),
        A=np.array([1.0, 1.0], dtype=float),
        v_p=np.array([2.0, 2.0], dtype=float),
        eta=np.array([0.5, 0.5], dtype=float),
        J_x=np.array([3.0, 3.0], dtype=float),
        J_y=np.array([-4.0, -4.0], dtype=float),
        E_x=np.array([-10.0, -10.0], dtype=float),
        B=2.0,
        velikhov_margin=np.array([0.0, 0.0], dtype=float),
        furnace_power_MW=363.0,
        steam_cycle_efficiency=0.35,
    )
    if not np.isfinite(metrics.mhd_output_power_MWe):
        failures.append("global postprocess returned non-finite power")

    if failures:
        print("SMOKE TEST FAILED")
        for item in failures:
            print(item)
        return 1

    print("SMOKE TEST PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
