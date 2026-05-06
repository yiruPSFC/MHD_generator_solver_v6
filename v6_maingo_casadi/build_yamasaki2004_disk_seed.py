#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from v6_maingo_casadi.yamasaki2004_parameters import YAMASAKI2004, YAMASAKI2004_MODEL_SEED


REPO_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = (
    REPO_DIR
    / "v6_maingo_casadi"
    / "outputs"
    / "maingo_yamasaki2004_neighborhood"
    / "seed_hecs"
)


def _repo_relative(path: Path) -> str:
    return str(path.resolve().relative_to(REPO_DIR))


def build_seed(out_dir: Path, *, n_intervals: int = 80) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paper = YAMASAKI2004
    model_seed = YAMASAKI2004_MODEL_SEED
    geom = paper.geometry
    profile = geom.profile(n_intervals=int(n_intervals))

    warm_path = out_dir / "yamasaki2004_hecs_disk_geometry_reference_profile.npz"
    np.savez(warm_path, **profile)

    summary_path = out_dir / "yamasaki2004_hecs_disk_geometry_reference_seed_summary.json"
    reference = paper.to_reference_dict()
    reference.update(
        {
            "reference_volume_m3": float(profile["volume_m3"]),
            "notes": (
                "Area variables a1/a2/a3 are deviations around the paper geometry, "
                "not the geometry itself; this keeps the MAiNGO search dimension unchanged. "
                "Hall voltage is a diagnostic target computed from -integral(E_x dx), "
                "not an independent optimizer variable."
            ),
        }
    )
    payload = {
        "working_fluid_profile": paper.working_fluid_profile,
        "B": float(paper.magnetic_field_T),
        "L": float(geom.length_m),
        "area_scale_m2": float(geom.cross_section_throat_m2),
        "schedule": [model_seed.schedule_entry()],
        "adaptive_bridge_count": int(model_seed.adaptive_bridge_count),
        "adaptive_bridge_max_count": int(model_seed.adaptive_bridge_max_count),
        "source_alignment": {
            "warm_profile_npz": _repo_relative(warm_path),
            "area_scale_m2": float(geom.cross_section_throat_m2),
            "area_reference_mode": "multiplicative",
            "aligned_inlet_window": model_seed.aligned_inlet_window(paper),
            "aligned_area_window": model_seed.aligned_area_window(),
            "diagnostic_targets": {
                "hall_voltage_V": paper.hall_voltage_V.to_dict(),
                "hall_current_A": paper.hall_current_A.to_dict(),
                "reported_electric_power_MW": float(paper.reported_electric_power_MW),
            },
        },
        "reference": reference,
    }
    summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return warm_path, summary_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write Yamasaki 2004 disk-geometry He/Cs seed artifacts.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--n-intervals", type=int, default=80)
    args = parser.parse_args(argv)
    warm_path, summary_path = build_seed(args.out_dir, n_intervals=int(args.n_intervals))
    print(json.dumps({"warm_profile_npz": str(warm_path), "summary_json": str(summary_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
