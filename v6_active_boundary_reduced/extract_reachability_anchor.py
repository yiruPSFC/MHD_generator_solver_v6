from __future__ import annotations

import argparse
import json
from pathlib import Path

from v6_firedrake_reduced.design import load_case_config

from .reachability_common import (
    anchor_from_node_payload,
    anchor_payload,
    load_profile_anchor,
    write_json,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Extract one fixed-endpoint reachability anchor from the Freidberg "
            "reference profile, another profile NPZ, or a preparation summary."
        )
    )
    parser.add_argument("--case", default="freidberg_reference")
    parser.add_argument("--out-json", type=Path, default=None)
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--summary-json",
        type=Path,
        help="preparation_recovery_summary.json; defaults to extracting one node.",
    )
    source.add_argument(
        "--profile-npz",
        type=Path,
        default=None,
        help="Profile NPZ. Omit for the built-in Freidberg reference profile.",
    )
    parser.add_argument(
        "--profile-index",
        type=int,
        default=0,
        help="Node index for --profile-npz or the built-in Freidberg profile.",
    )
    parser.add_argument(
        "--node-index",
        type=int,
        default=-1,
        help="Node index when --summary-json is used; -1 is the recovered upstream endpoint.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_case_config(case=str(args.case))

    if args.summary_json is not None:
        summary_path = Path(args.summary_json)
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        nodes = list(summary.get("nodes", []))
        if not nodes:
            raise ValueError(f"{summary_path} does not contain nodes.")
        idx = int(args.node_index)
        node = dict(nodes[idx])
        anchor = anchor_from_node_payload(
            node,
            config=config,
            source=f"{summary_path}:nodes[{idx}]",
            source_index=int(node.get("k", idx)),
        )
    else:
        profile_path = None if args.profile_npz is None else Path(args.profile_npz)
        anchor = load_profile_anchor(
            profile_path,
            index=int(args.profile_index),
            config=config,
            source=str(profile_path or f"{config.case}:built_in_profile"),
        )

    payload = anchor_payload(anchor, config=config)
    if args.out_json is not None:
        write_json(Path(args.out_json), payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
