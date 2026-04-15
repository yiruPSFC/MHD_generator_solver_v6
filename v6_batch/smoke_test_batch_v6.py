from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np

REPO_DIR = Path(__file__).resolve().parents[1]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from v6_core.non_batch.pde_solver_v6 import ForwardPDESolverV6
from v6_batch.pde_solver_v6_batch import ForwardPDESolverV6Batch, event_name_from_code


@dataclass
class SmokeCase:
    n_p_in: float
    Z_in: float
    T_p_in: float
    T_e_in: float
    seed_fraction: float


CASES = [
    SmokeCase(2.9418007053766115e21, 56.32891972495766, 1000.0, 5727.0800768822055, 0.0004799999764588285),
    SmokeCase(1.5242875089416995e21, 61.23685251715171, 1000.0, 4793.910217845207, 0.00022630567813314458),
    SmokeCase(6.221604028055478e23, 30.340058190509357, 1000.0, 5013.648876370421, 0.001472078329934333),
]


def _event_code_from_name(name: str | None) -> int:
    if name == "mach_0p99":
        return 1
    if name == "mach_1p01":
        return 2
    return 0


def main() -> int:
    single = ForwardPDESolverV6()
    batch = ForwardPDESolverV6Batch()

    dx = 1e-5
    batch_out = batch.solve_batch(
        n_p_in=np.array([c.n_p_in for c in CASES], dtype=float),
        Z_in=np.array([c.Z_in for c in CASES], dtype=float),
        T_p_in=np.array([c.T_p_in for c in CASES], dtype=float),
        T_e_in=np.array([c.T_e_in for c in CASES], dtype=float),
        seed_fraction=np.array([c.seed_fraction for c in CASES], dtype=float),
        dx=dx,
        store_profiles=False,
    )
    profile_out = batch.solve_batch(
        n_p_in=np.array([c.n_p_in for c in CASES[:2]], dtype=float),
        Z_in=np.array([c.Z_in for c in CASES[:2]], dtype=float),
        T_p_in=np.array([c.T_p_in for c in CASES[:2]], dtype=float),
        T_e_in=np.array([c.T_e_in for c in CASES[:2]], dtype=float),
        seed_fraction=np.array([c.seed_fraction for c in CASES[:2]], dtype=float),
        dx=5e-5,
        store_profiles=True,
    )

    failures = []
    print(f"batch step_size={batch_out.step_size:.6e} n_cases={len(CASES)}")
    for i, case in enumerate(CASES):
        out = single.solve(
            n_p_in=case.n_p_in,
            Z_in=case.Z_in,
            T_p_in=case.T_p_in,
            T_e_in=case.T_e_in,
            seed_fraction=case.seed_fraction,
        )
        event_batch = event_name_from_code(int(batch_out.event_code[i]))

        te_err = abs(float(batch_out.T_e[i]) - float(out.T_e[-1]))
        tp_err = abs(float(batch_out.T_p[i]) - float(out.T_p[-1]))
        np_rel = abs(float(batch_out.n_p[i]) - float(out.n_p[-1])) / max(abs(float(out.n_p[-1])), 1.0)
        event_expected = _event_code_from_name(out.event_name)
        event_match = int(batch_out.event_code[i]) == event_expected

        print(
            "case={} single_event={} batch_event={} te_err={:.6e} tp_err={:.6e} np_rel={:.6e}".format(
                i,
                out.event_name,
                event_batch,
                te_err,
                tp_err,
                np_rel,
            )
        )

        if not out.success:
            failures.append(f"case {i}: single solver failed")
        if not bool(batch_out.success[i]):
            failures.append(f"case {i}: batch solver failed")
        if te_err > 3.0:
            failures.append(f"case {i}: Te error too large ({te_err})")
        if tp_err > 3.0:
            failures.append(f"case {i}: Tp error too large ({tp_err})")
        if np_rel > 5e-4:
            failures.append(f"case {i}: n_p relative error too large ({np_rel})")
        if not event_match:
            failures.append(
                f"case {i}: event mismatch single={out.event_name} batch={event_batch}"
            )

    if profile_out.n_p.shape[0] != 2 or profile_out.n_p.shape[1] <= 1:
        failures.append(f"profile path returned invalid shape {profile_out.n_p.shape}")
    if not np.all(profile_out.valid_points > 1):
        failures.append(f"profile path returned invalid valid_points {profile_out.valid_points.tolist()}")

    if failures:
        print("SMOKE TEST FAILED")
        for item in failures:
            print(item)
        return 1

    print("SMOKE TEST PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
