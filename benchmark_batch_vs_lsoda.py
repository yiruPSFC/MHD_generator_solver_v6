from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass

import numpy as np
from scipy.stats.qmc import LatinHypercube

from local_algebraic_closure import B_FIELD
from pde_solver_v6 import ForwardPDESolverV6
from pde_solver_v6_batch import ForwardPDESolverV6Batch


LENGTH = 0.039


@dataclass
class Candidate:
    n_p_in: float
    Z_in: float
    T_p_in: float
    T_e_in: float
    seed_fraction: float


SEED_CASES = [
    Candidate(2.9418007053766115e21, 56.32891972495766, 1000.0, 5727.0800768822055, 0.0004799999764588285),
    Candidate(1.5242875089416995e21, 61.23685251715171, 1000.0, 4793.910217845207, 0.00022630567813314458),
    Candidate(6.221604028055478e23, 30.340058190509357, 1000.0, 5013.648876370421, 0.001472078329934333),
]


def _log_map(u: float, lo: float, hi: float) -> float:
    return float(10.0 ** (np.log10(lo) + u * (np.log10(hi) - np.log10(lo))))


def _lin_map(u: float, lo: float, hi: float) -> float:
    return float(lo + u * (hi - lo))


def _build_seed_jitter_candidates(n_cases: int, seed: int, tp_in: float) -> list[Candidate]:
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_cases):
        base = SEED_CASES[int(rng.integers(0, len(SEED_CASES)))]
        n_p_in = float(base.n_p_in * (10.0 ** rng.uniform(-0.35, 0.35)))
        Z_in = float(np.clip(base.Z_in + rng.normal(0.0, 6.0), 1.0, 120.0))
        T_e_in = float(np.clip(base.T_e_in + rng.normal(0.0, 250.0), tp_in + 50.0, 6000.0))
        seed_fraction = float(np.clip(base.seed_fraction * (10.0 ** rng.uniform(-0.35, 0.35)), 1e-4, 5e-2))
        out.append(
            Candidate(
                n_p_in=n_p_in,
                Z_in=Z_in,
                T_p_in=float(tp_in),
                T_e_in=T_e_in,
                seed_fraction=seed_fraction,
            )
        )
    return out


def _build_lhs_candidates(n_cases: int, seed: int, tp_in: float) -> list[Candidate]:
    bounds = {
        "n_p_in": (1e21, 1e24),
        "Z_in": (1.0, 120.0),
        "T_e_in": (1500.0, 6000.0),
        "seed_fraction": (1e-4, 5e-2),
    }
    lhs = LatinHypercube(d=4, seed=seed)
    U = lhs.random(n=n_cases)
    out = []
    for row in U:
        out.append(
            Candidate(
                n_p_in=_log_map(float(row[0]), *bounds["n_p_in"]),
                Z_in=_lin_map(float(row[1]), *bounds["Z_in"]),
                T_p_in=float(tp_in),
                T_e_in=_lin_map(float(row[2]), *bounds["T_e_in"]),
                seed_fraction=_log_map(float(row[3]), *bounds["seed_fraction"]),
            )
        )
    return out


def _event_code_from_name(name: str | None) -> int:
    if name == "mach_0p99":
        return 1
    if name == "mach_1p01":
        return 2
    return 0


def _build_candidates(sampler: str, n_cases: int, seed: int, tp_in: float) -> list[Candidate]:
    if sampler == "lhs":
        return _build_lhs_candidates(n_cases, seed, tp_in)
    return _build_seed_jitter_candidates(n_cases, seed, tp_in)


def _candidate_arrays(candidates: list[Candidate]):
    return (
        np.array([c.n_p_in for c in candidates], dtype=float),
        np.array([c.Z_in for c in candidates], dtype=float),
        np.array([c.T_p_in for c in candidates], dtype=float),
        np.array([c.T_e_in for c in candidates], dtype=float),
        np.array([c.seed_fraction for c in candidates], dtype=float),
    )


def _dx_from_n_steps(n_steps: int, length: float) -> float:
    if n_steps <= 0:
        raise ValueError("n_steps must be positive.")
    return float(length) / float(n_steps)


def _warm_up(batch: ForwardPDESolverV6Batch, single: ForwardPDESolverV6, candidates: list[Candidate], throughput_dx: float, accuracy_steps: list[int]) -> None:
    warm_n = min(8, len(candidates))
    if warm_n == 0:
        return
    arr_np, arr_z, arr_tp, arr_te, arr_seed = _candidate_arrays(candidates[:warm_n])
    batch.solve_batch(arr_np, arr_z, arr_tp, arr_te, arr_seed, dx=throughput_dx, store_profiles=False, parallel=True)
    batch.solve_batch(arr_np, arr_z, arr_tp, arr_te, arr_seed, dx=throughput_dx, store_profiles=False, parallel=False)
    for steps in accuracy_steps:
        batch.solve_batch(arr_np[:1], arr_z[:1], arr_tp[:1], arr_te[:1], arr_seed[:1], dx=_dx_from_n_steps(steps, batch.length), store_profiles=False, parallel=True)
    single.solve(
        n_p_in=candidates[0].n_p_in,
        Z_in=candidates[0].Z_in,
        T_p_in=candidates[0].T_p_in,
        T_e_in=candidates[0].T_e_in,
        seed_fraction=candidates[0].seed_fraction,
    )


def _run_throughput(
    batch: ForwardPDESolverV6Batch,
    candidates: list[Candidate],
    n_steps: int,
    sampler: str,
):
    arr_np, arr_z, arr_tp, arr_te, arr_seed = _candidate_arrays(candidates)
    dx = _dx_from_n_steps(n_steps, batch.length)

    t0 = time.perf_counter()
    serial_out = batch.solve_batch(
        arr_np,
        arr_z,
        arr_tp,
        arr_te,
        arr_seed,
        dx=dx,
        store_profiles=False,
        parallel=False,
    )
    serial_time = time.perf_counter() - t0

    t1 = time.perf_counter()
    parallel_out = batch.solve_batch(
        arr_np,
        arr_z,
        arr_tp,
        arr_te,
        arr_seed,
        dx=dx,
        store_profiles=False,
        parallel=True,
    )
    parallel_time = time.perf_counter() - t1

    comparable = int(np.count_nonzero(serial_out.success & parallel_out.success))
    te_err = np.abs(parallel_out.T_e - serial_out.T_e)
    tp_err = np.abs(parallel_out.T_p - serial_out.T_p)
    np_rel = np.abs(parallel_out.n_p - serial_out.n_p) / np.maximum(np.abs(serial_out.n_p), 1.0)
    event_mismatch = int(np.count_nonzero(parallel_out.event_code != serial_out.event_code))

    print(f"mode=throughput sampler={sampler} n_cases={len(candidates)} n_steps={n_steps} dx={dx:.6e}")
    print(f"serial_time_s={serial_time:.6f} serial_cases_per_s={len(candidates) / serial_time:.2f}")
    print(f"batch_time_s={parallel_time:.6f} batch_cases_per_s={len(candidates) / parallel_time:.2f}")
    print(f"speedup_vs_serial={serial_time / parallel_time if parallel_time > 0.0 else math.inf:.2f}x")
    print(f"serial_ok={int(np.count_nonzero(serial_out.success))} batch_ok={int(np.count_nonzero(parallel_out.success))}")
    print(f"serial_events={np.bincount(serial_out.event_code, minlength=5).tolist()}")
    print(f"batch_events={np.bincount(parallel_out.event_code, minlength=5).tolist()}")
    if comparable > 0:
        mask = serial_out.success & parallel_out.success
        print(f"comparable_cases={comparable}")
        print(f"max_te_diff={float(np.max(te_err[mask])):.6e} mean_te_diff={float(np.mean(te_err[mask])):.6e}")
        print(f"max_tp_diff={float(np.max(tp_err[mask])):.6e} mean_tp_diff={float(np.mean(tp_err[mask])):.6e}")
        print(f"max_np_rel_diff={float(np.max(np_rel[mask])):.6e} mean_np_rel_diff={float(np.mean(np_rel[mask])):.6e}")
        print(f"event_mismatch={event_mismatch}")
    else:
        print("comparable_cases=0")


def _run_accuracy(
    batch: ForwardPDESolverV6Batch,
    single: ForwardPDESolverV6,
    candidates: list[Candidate],
    accuracy_cases: int,
    accuracy_steps: list[int],
    lsoda_max_step: float | None,
    sampler: str,
):
    acc_candidates = candidates[: min(accuracy_cases, len(candidates))]
    if not acc_candidates:
        print("mode=accuracy accuracy_cases=0")
        return

    refs = []
    t_ref = time.perf_counter()
    for cand in acc_candidates:
        try:
            refs.append(
                single.solve(
                    n_p_in=cand.n_p_in,
                    Z_in=cand.Z_in,
                    T_p_in=cand.T_p_in,
                    T_e_in=cand.T_e_in,
                    seed_fraction=cand.seed_fraction,
                    max_step=lsoda_max_step,
                )
            )
        except Exception as exc:
            refs.append(exc)
    ref_time = time.perf_counter() - t_ref
    lsoda_ok = sum(1 for item in refs if not isinstance(item, Exception) and item.success)

    print(f"mode=accuracy sampler={sampler} accuracy_cases={len(acc_candidates)} lsoda_ok={lsoda_ok} lsoda_time_s={ref_time:.6f}")
    if lsoda_max_step is not None:
        print(f"lsoda_max_step={lsoda_max_step:.6e}")
    else:
        print("lsoda_max_step=None")

    arr_np, arr_z, arr_tp, arr_te, arr_seed = _candidate_arrays(acc_candidates)
    for n_steps in accuracy_steps:
        dx = _dx_from_n_steps(n_steps, batch.length)
        t0 = time.perf_counter()
        out = batch.solve_batch(
            arr_np,
            arr_z,
            arr_tp,
            arr_te,
            arr_seed,
            dx=dx,
            store_profiles=False,
            parallel=True,
        )
        batch_time = time.perf_counter() - t0

        te_err = []
        tp_err = []
        np_rel = []
        event_mismatch = 0
        comparable = 0
        for i, ref in enumerate(refs):
            if isinstance(ref, Exception):
                continue
            if not ref.success or (not bool(out.success[i])):
                continue
            comparable += 1
            te_err.append(abs(float(out.T_e[i]) - float(ref.T_e[-1])))
            tp_err.append(abs(float(out.T_p[i]) - float(ref.T_p[-1])))
            np_rel.append(abs(float(out.n_p[i]) - float(ref.n_p[-1])) / max(abs(float(ref.n_p[-1])), 1.0))
            if int(out.event_code[i]) != _event_code_from_name(ref.event_name):
                event_mismatch += 1

        print(f"rk4_n_steps={n_steps} dx={dx:.6e} batch_time_s={batch_time:.6f} batch_ok={int(np.count_nonzero(out.success))}")
        if comparable > 0:
            print(f"comparable_cases={comparable}")
            print(f"max_te_err={max(te_err):.6e} mean_te_err={float(np.mean(te_err)):.6e}")
            print(f"max_tp_err={max(tp_err):.6e} mean_tp_err={float(np.mean(tp_err)):.6e}")
            print(f"max_np_rel={max(np_rel):.6e} mean_np_rel={float(np.mean(np_rel)):.6e}")
            print(f"event_mismatch={event_mismatch}")
        else:
            print("comparable_cases=0")


def _parse_accuracy_steps(value: str) -> list[int]:
    out = []
    for item in value.split(","):
        token = item.strip()
        if not token:
            continue
        out.append(int(token))
    if not out:
        raise ValueError("accuracy_steps cannot be empty.")
    return out


def main() -> int:
    p = argparse.ArgumentParser(description="Benchmark batch fixed-step solver throughput and accuracy")
    p.add_argument("--mode", type=str, default="both", choices=["throughput", "accuracy", "both"])
    p.add_argument("--sampler", type=str, default="seed-jitter", choices=["seed-jitter", "lhs"])
    p.add_argument("--n-cases", type=int, default=10000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tp-in", type=float, default=1000.0)
    p.add_argument("--throughput-n-steps", type=int, default=200)
    p.add_argument("--accuracy-cases", type=int, default=128)
    p.add_argument("--accuracy-steps", type=str, default="200,800,2000")
    p.add_argument("--lsoda-max-step", type=float, default=None)
    parsed = p.parse_args()

    accuracy_steps = _parse_accuracy_steps(parsed.accuracy_steps)
    candidates = _build_candidates(parsed.sampler, int(parsed.n_cases), int(parsed.seed), float(parsed.tp_in))

    batch = ForwardPDESolverV6Batch(B=B_FIELD, length=LENGTH)
    single = ForwardPDESolverV6(B=B_FIELD, length=LENGTH)
    _warm_up(batch, single, candidates, _dx_from_n_steps(parsed.throughput_n_steps, LENGTH), accuracy_steps)

    if parsed.mode in ("throughput", "both"):
        _run_throughput(batch, candidates, int(parsed.throughput_n_steps), parsed.sampler)
    if parsed.mode in ("accuracy", "both"):
        _run_accuracy(batch, single, candidates, int(parsed.accuracy_cases), accuracy_steps, parsed.lsoda_max_step, parsed.sampler)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
