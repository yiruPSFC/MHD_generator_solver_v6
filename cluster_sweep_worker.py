#!/usr/bin/env python3
from __future__ import annotations

"""
Cluster sweep worker for V6 explicit solver.

- Candidate generation: scipy.stats.qmc.LatinHypercube
- Fast forward solve: ForwardPDESolverV6
- Outputs JSONL rows for downstream merge/analyze jobs.
"""

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy.stats.qmc import LatinHypercube

from local_algebraic_closure import B_FIELD, E_I, K_B
from pde_solver_v6 import ForwardPDESolverV6


@dataclass
class Candidate:
    n_p_in: float
    Z_in: float
    T_p_in: float
    T_e_in: float
    seed_fraction: float
    B: float


@dataclass
class EvalSummary:
    ok: bool
    feasible: bool
    reason: str
    objective: float
    outlet_te: float
    outlet_tp: float
    outlet_ratio: float
    min_velikhov_margin: float
    event: str | None
    params: Dict[str, float]


def _log_map(u: float, lo: float, hi: float) -> float:
    return float(10.0 ** (np.log10(lo) + u * (np.log10(hi) - np.log10(lo))))


def _lin_map(u: float, lo: float, hi: float) -> float:
    return float(lo + u * (hi - lo))


def _candidate_from_unit(u: np.ndarray, bounds: Dict[str, Tuple[float, float]], T_p_in: float, B: float) -> Candidate:
    return Candidate(
        n_p_in=_log_map(float(u[0]), *bounds["n_p_in"]),
        Z_in=_lin_map(float(u[1]), *bounds["Z_in"]),
        T_p_in=float(T_p_in),
        T_e_in=_lin_map(float(u[2]), *bounds["T_e_in"]),
        seed_fraction=_log_map(float(u[3]), *bounds["seed_fraction"]),
        B=float(B),
    )


def _velikhov_margin(
    beta: np.ndarray,
    T_e: np.ndarray,
    T_p: np.ndarray,
    n_e: np.ndarray,
    n_p: np.ndarray,
    seed_fraction: float,
) -> float:
    ns = seed_fraction * np.maximum(n_p, 1e-300)
    fI = np.clip(n_e / np.maximum(ns, 1e-300), 1e-12, 1.0 - 1e-12)

    delta = np.maximum(T_e / np.maximum(T_p, 1e-300) - 1.0, 1e-12)
    alpha = (K_B * T_e / (2.0 * E_I)) * (2.0 - fI) / np.maximum(1.0 - fI, 1e-12)

    rhs = 4.0 * alpha * (2.0 + 1.0 / delta) * (1.0 + alpha * (1.0 + 1.0 / delta))
    lhs = beta * beta
    margin = rhs - lhs
    return float(np.min(margin))


def evaluate_candidate(c: Candidate, L: float, velikhov_required: bool = True) -> EvalSummary:
    try:
        solver = ForwardPDESolverV6(B=c.B, length=L)
        out = solver.solve(
            n_p_in=c.n_p_in,
            Z_in=c.Z_in,
            T_p_in=c.T_p_in,
            T_e_in=c.T_e_in,
            seed_fraction=c.seed_fraction,
            rtol=1e-6,
            atol=1e-8,
        )

        if not out.success or len(out.x) == 0:
            return EvalSummary(
                ok=False,
                feasible=False,
                reason=f"ivp_failed:{out.message}",
                objective=1e30,
                outlet_te=np.nan,
                outlet_tp=np.nan,
                outlet_ratio=np.nan,
                min_velikhov_margin=np.nan,
                event=out.event_name,
                params=asdict(c),
            )

        te_out = float(out.T_e[-1])
        tp_out = float(out.T_p[-1])
        ratio_out = te_out / max(tp_out, 1e-300)

        margin = _velikhov_margin(
            beta=out.beta,
            T_e=out.T_e,
            T_p=out.T_p,
            n_e=out.n_e,
            n_p=out.n_p,
            seed_fraction=c.seed_fraction,
        )

        stable = (margin >= 0.0) if velikhov_required else True
        feasible = bool(stable)

        # Objective: maximize outlet Te/Tp -> minimize negative ratio
        obj = -ratio_out if feasible else 1e6 - ratio_out

        return EvalSummary(
            ok=True,
            feasible=feasible,
            reason="ok" if feasible else "velikhov_unstable",
            objective=float(obj),
            outlet_te=te_out,
            outlet_tp=tp_out,
            outlet_ratio=ratio_out,
            min_velikhov_margin=float(margin),
            event=out.event_name,
            params=asdict(c),
        )

    except Exception as exc:
        return EvalSummary(
            ok=False,
            feasible=False,
            reason=f"exception:{exc.__class__.__name__}:{exc}",
            objective=1e30,
            outlet_te=np.nan,
            outlet_tp=np.nan,
            outlet_ratio=np.nan,
            min_velikhov_margin=np.nan,
            event=None,
            params=asdict(c),
        )


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="V6 cluster sweep worker (LHS)")
    p.add_argument("--n-total", type=int, default=10000, help="total LHS points")
    p.add_argument("--seed", type=int, default=0, help="RNG seed")
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--shard-count", type=int, default=1)
    p.add_argument("--top-k", type=int, default=200, help="keep best K results per shard")
    p.add_argument("--out", type=str, default="sweep_results.jsonl")

    p.add_argument("--B", type=float, default=B_FIELD)
    p.add_argument("--L", type=float, default=0.039)
    p.add_argument("--tp-in", type=float, default=1000.0)

    p.add_argument("--np-min", type=float, default=1e21)
    p.add_argument("--np-max", type=float, default=1e24)
    p.add_argument("--z-min", type=float, default=1.0)
    p.add_argument("--z-max", type=float, default=120.0)
    p.add_argument("--te-min", type=float, default=1500.0)
    p.add_argument("--te-max", type=float, default=6000.0)
    p.add_argument("--seedf-min", type=float, default=1e-4)
    p.add_argument("--seedf-max", type=float, default=5e-2)

    p.add_argument("--no-velikhov-check", action="store_true")
    return p


def _shard_window(n_total: int, shard_index: int, shard_count: int) -> Tuple[int, int]:
    if shard_count <= 0:
        raise ValueError("shard_count must be >= 1")
    if shard_index < 0 or shard_index >= shard_count:
        raise ValueError("invalid shard_index")
    base = n_total // shard_count
    rem = n_total % shard_count
    start = shard_index * base + min(shard_index, rem)
    size = base + (1 if shard_index < rem else 0)
    return start, size


def main() -> int:
    args = _build_parser().parse_args()

    bounds = {
        "n_p_in": (float(args.np_min), float(args.np_max)),
        "Z_in": (float(args.z_min), float(args.z_max)),
        "T_e_in": (float(args.te_min), float(args.te_max)),
        "seed_fraction": (float(args.seedf_min), float(args.seedf_max)),
    }

    for k, (lo, hi) in bounds.items():
        if lo <= 0.0 and k in ("n_p_in", "seed_fraction"):
            raise ValueError(f"{k} must be positive")
        if hi <= lo:
            raise ValueError(f"{k} bounds invalid: {lo}, {hi}")

    start, count = _shard_window(int(args.n_total), int(args.shard_index), int(args.shard_count))

    lhs = LatinHypercube(d=4, seed=int(args.seed))
    U = lhs.random(n=int(args.n_total))
    U = U[start:start + count]

    rows: List[EvalSummary] = []
    for i in range(U.shape[0]):
        cand = _candidate_from_unit(U[i], bounds=bounds, T_p_in=float(args.tp_in), B=float(args.B))
        res = evaluate_candidate(cand, L=float(args.L), velikhov_required=(not args.no_velikhov_check))
        rows.append(res)

    rows_sorted = sorted(rows, key=lambda r: r.objective)
    k = max(1, int(args.top_k))
    best = rows_sorted[:k]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in best:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")

    feasible_n = sum(1 for r in rows if r.feasible)
    ok_n = sum(1 for r in rows if r.ok)
    print(
        json.dumps(
            {
                "n_total": int(args.n_total),
                "shard_index": int(args.shard_index),
                "shard_count": int(args.shard_count),
                "evaluated": int(len(rows)),
                "ok": int(ok_n),
                "feasible": int(feasible_n),
                "saved": int(len(best)),
                "output": str(out_path),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
