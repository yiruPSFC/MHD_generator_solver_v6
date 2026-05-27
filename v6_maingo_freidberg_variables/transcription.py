from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .algebra import primitive_to_freidberg
from .models import FreidbergConfig, PrimitivePoint
from .rhs import freidberg_rhs_arrays


@dataclass(frozen=True)
class FreidbergIntervalDefects:
    x_left: np.ndarray
    x_right: np.ndarray
    H_defect: np.ndarray
    L_defect: np.ndarray
    H_defect_MW: np.ndarray
    H_delta_MW: np.ndarray
    H_rhs_MW: np.ndarray
    L_delta: np.ndarray
    L_rhs: np.ndarray

    def summary(self) -> dict[str, float]:
        return {
            "n_intervals": int(self.H_defect.size),
            "max_abs_H_defect_MW": float(np.nanmax(np.abs(self.H_defect_MW))),
            "rms_H_defect_MW": _rms(self.H_defect_MW),
            "terminal_H_defect_MW": float(np.nansum(self.H_defect_MW)),
            "max_abs_L_defect": float(np.nanmax(np.abs(self.L_defect))),
            "rms_L_defect": _rms(self.L_defect),
            "terminal_L_defect": float(np.nansum(self.L_defect)),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary(),
            "intervals": [
                {
                    "x_left": float(self.x_left[i]),
                    "x_right": float(self.x_right[i]),
                    "H_delta_MW": float(self.H_delta_MW[i]),
                    "H_rhs_MW": float(self.H_rhs_MW[i]),
                    "H_defect_MW": float(self.H_defect_MW[i]),
                    "L_delta": float(self.L_delta[i]),
                    "L_rhs": float(self.L_rhs[i]),
                    "L_defect": float(self.L_defect[i]),
                }
                for i in range(self.H_defect.size)
            ],
        }


def _rms(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return float("nan")
    return float(math.sqrt(float(np.mean(finite * finite))))


def _load_points(profile_path: str | Path) -> list[PrimitivePoint]:
    with np.load(profile_path) as data:
        return [PrimitivePoint.from_npz(data, idx) for idx in range(np.asarray(data["x"], dtype=float).size)]


def interval_defects_from_points(points: list[PrimitivePoint], config: FreidbergConfig) -> FreidbergIntervalDefects:
    if len(points) < 2:
        raise ValueError("at least two points are required for interval defects")
    x = np.asarray([point.x for point in points], dtype=float)
    if not np.all(np.diff(x) > 0.0):
        raise ValueError("x grid must be strictly increasing")
    states = [primitive_to_freidberg(point, config) for point in points]
    H = np.asarray([state.H_p for state in states], dtype=float)
    L = np.asarray([state.L_p for state in states], dtype=float)
    rhs = freidberg_rhs_arrays(points, config)
    dx = np.diff(x)
    H_delta = np.diff(H)
    H_rhs = 0.5 * dx * (rhs["dHdx"][:-1] + rhs["dHdx"][1:])
    L_delta = np.diff(L)
    L_rhs = 0.5 * dx * (rhs["dLdx"][:-1] + rhs["dLdx"][1:])
    H_defect = H_delta - H_rhs
    L_defect = L_delta - L_rhs
    A0 = config.inlet_area_m2
    return FreidbergIntervalDefects(
        x_left=x[:-1],
        x_right=x[1:],
        H_defect=H_defect,
        L_defect=L_defect,
        H_defect_MW=H_defect * A0 / 1e6,
        H_delta_MW=H_delta * A0 / 1e6,
        H_rhs_MW=H_rhs * A0 / 1e6,
        L_delta=L_delta,
        L_rhs=L_rhs,
    )


def transcription_summary_from_profile(*, profile_path: str | Path, summary_path: str | Path) -> dict[str, Any]:
    profile_path = Path(profile_path).resolve()
    summary_path = Path(summary_path).resolve()
    config = FreidbergConfig.from_summary_and_profile(summary_path, profile_path)
    points = _load_points(profile_path)
    defects = interval_defects_from_points(points, config)
    return {
        "profile_path": str(profile_path),
        "summary_path": str(summary_path),
        "config": config.to_dict(),
        "freidberg_interval_defects": defects.to_dict(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate Freidberg direct-transcription interval defects for a stored primitive profile."
    )
    parser.add_argument("profile", type=Path)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    report = transcription_summary_from_profile(profile_path=args.profile, summary_path=args.summary)
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.out is None:
        print(text)
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")
        print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
