#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

_THIS_DIR = Path(__file__).resolve().parent
_CASADI_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = Path(__file__).resolve().parents[2]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from v6_casadi.run_casadi_continuation_v6 import run_continuation


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run the Jeffrey-local continuation workflow used to validate the CasADi NLP near the reference case"
    )
    p.add_argument("--np-in", type=float, default=3.05e25)
    p.add_argument("--z-in", type=float, default=75.954994)
    p.add_argument("--tp-in", type=float, default=429.0)
    p.add_argument("--te-in", type=float, default=4420.0)
    p.add_argument("--A-in", type=float, default=0.447)
    p.add_argument("--B", type=float, default=10.2)
    p.add_argument("--L", type=float, default=5.4)
    p.add_argument("--seed-fraction", type=float, default=None)
    p.add_argument("--warm-start-dx", type=float, default=0.01)
    p.add_argument(
        "--out-dir",
        type=str,
        default=str(_CASADI_DIR / "outputs" / "continuation" / "reference_case"),
        help="directory for the Jeffrey-local continuation artifacts",
    )
    p.add_argument("--out-json", type=str, default="")
    return p


def main() -> int:
    args = _build_parser().parse_args()
    payload = run_continuation(
        n_p_in=float(args.np_in),
        Z_in=float(args.z_in),
        T_p_in=float(args.tp_in),
        T_e_in=float(args.te_in),
        A_in=float(args.A_in),
        B=float(args.B),
        L=float(args.L),
        seed_fraction=None if args.seed_fraction is None else float(args.seed_fraction),
        warm_start_dx=float(args.warm_start_dx),
        out_dir=args.out_dir,
        out_json=args.out_json,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
