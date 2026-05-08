#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_DIR = Path(__file__).resolve().parents[3]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from v6_maingo_casadi.cases.yamasaki2004.parameters import YAMASAKI2004, YAMASAKI2004_MODEL_SEED
from v6_maingo_casadi.geometry import SplineAreaDesign


DEFAULT_OUT_DIR = (
    REPO_DIR
    / "v6_maingo_casadi"
    / "outputs"
    / "cases"
    / "yamasaki2004"
    / "seeds"
)


def _repo_relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_DIR))
    except ValueError:
        return str(resolved)


def build_seed(out_dir: Path, *, n_intervals: int = 80) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paper = YAMASAKI2004
    model_seed = YAMASAKI2004_MODEL_SEED
    geom = paper.geometry
    profile = geom.profile(n_intervals=int(n_intervals))
    area_design = SplineAreaDesign.project_from_profile(
        x=np.asarray(profile["x"], dtype=float),
        A=np.asarray(profile["A"], dtype=float),
    )

    warm_path = out_dir / "yamasaki2004_hecs_disk_geometry_reference_profile.npz"
    np.savez(warm_path, **profile)

    summary_path = out_dir / "yamasaki2004_hecs_disk_geometry_reference_seed_summary.json"
    reference = paper.to_reference_dict()
    reference.update(
        {
            "reference_volume_m3": float(profile["volume_m3"]),
            "area_spline_nominal": area_design.to_dict(),
            "notes": (
                "Area variables a1/a2/a3 are direct log-area spline values at "
                "x/L = 1/3, 2/3, and 1, fitted to the paper geometry. "
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
            "aligned_inlet_window": model_seed.aligned_inlet_window(paper),
            "aligned_area_window": model_seed.aligned_area_window(area_design.as_array()),
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
