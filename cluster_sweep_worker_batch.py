#!/usr/bin/env python3
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import List

import numpy as np
from scipy.stats.qmc import LatinHypercube

from local_algebraic_closure import B_FIELD
from cluster_sweep_worker import (
    Candidate,
    EvalSummary,
    _build_parser,
    _candidate_from_unit,
    _shard_window,
    _velikhov_margin,
)
from pde_solver_v6_batch import ForwardPDESolverV6Batch, event_name_from_code


def _evaluate_chunk(
    solver: ForwardPDESolverV6Batch,
    chunk: List[Candidate],
    dx: float,
    velikhov_required: bool,
) -> List[EvalSummary]:
    out = solver.solve_batch(
        n_p_in=np.array([c.n_p_in for c in chunk], dtype=float),
        Z_in=np.array([c.Z_in for c in chunk], dtype=float),
        T_p_in=np.array([c.T_p_in for c in chunk], dtype=float),
        T_e_in=np.array([c.T_e_in for c in chunk], dtype=float),
        seed_fraction=np.array([c.seed_fraction for c in chunk], dtype=float),
        dx=dx,
        store_profiles=True,
    )

    rows: List[EvalSummary] = []
    for i, cand in enumerate(chunk):
        event_name = event_name_from_code(int(out.event_code[i]))
        if not bool(out.success[i]) or int(out.valid_points[i]) <= 0:
            rows.append(
                EvalSummary(
                    ok=False,
                    feasible=False,
                    reason=f"batch_failed:{event_name or 'unknown'}",
                    objective=1e30,
                    outlet_te=np.nan,
                    outlet_tp=np.nan,
                    outlet_ratio=np.nan,
                    min_velikhov_margin=np.nan,
                    event=event_name,
                    params=asdict(cand),
                )
            )
            continue

        idx_last = int(out.valid_points[i]) - 1
        te_out = float(out.T_e[i, idx_last])
        tp_out = float(out.T_p[i, idx_last])
        ratio_out = te_out / max(tp_out, 1e-300)

        margin = _velikhov_margin(
            beta=out.beta[i, : idx_last + 1],
            T_e=out.T_e[i, : idx_last + 1],
            T_p=out.T_p[i, : idx_last + 1],
            n_e=out.n_e[i, : idx_last + 1],
            n_p=out.n_p[i, : idx_last + 1],
            seed_fraction=cand.seed_fraction,
        )
        stable = (margin >= 0.0) if velikhov_required else True
        feasible = bool(stable)
        obj = -ratio_out if feasible else 1e6 - ratio_out

        rows.append(
            EvalSummary(
                ok=True,
                feasible=feasible,
                reason="ok" if feasible else "velikhov_unstable",
                objective=float(obj),
                outlet_te=te_out,
                outlet_tp=tp_out,
                outlet_ratio=ratio_out,
                min_velikhov_margin=float(margin),
                event=event_name,
                params=asdict(cand),
            )
        )

    return rows


def main() -> int:
    p = _build_parser()
    p.description = "V6 cluster sweep worker (batch fixed-step)"
    p.add_argument("--batch-size", type=int, default=64, help="number of candidates per batch solve")
    p.add_argument("--dx", type=float, default=2e-5, help="fixed spatial step for batch RK4")
    args = p.parse_args()

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

    solver = ForwardPDESolverV6Batch(B=float(args.B), length=float(args.L))
    batch_size = max(1, int(args.batch_size))
    rows: List[EvalSummary] = []

    for offset in range(0, U.shape[0], batch_size):
        block = U[offset : offset + batch_size]
        chunk = [
            _candidate_from_unit(block[i], bounds=bounds, T_p_in=float(args.tp_in), B=float(args.B))
            for i in range(block.shape[0])
        ]
        rows.extend(
            _evaluate_chunk(
                solver=solver,
                chunk=chunk,
                dx=float(args.dx),
                velikhov_required=(not args.no_velikhov_check),
            )
        )

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
                "batch_size": batch_size,
                "dx": float(args.dx),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
