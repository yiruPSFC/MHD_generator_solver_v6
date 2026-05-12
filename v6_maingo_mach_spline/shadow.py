from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from v6_maingo_freidberg_variables.models import FreidbergConfig, PrimitivePoint
from v6_maingo_freidberg_variables.transcription import interval_defects_from_points

from .closure import primitive_arrays, reconstruct_points_from_mach
from .geometry import MachSplineDesign


def _max_rel(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(np.nanmax(np.abs(a - b) / np.maximum(np.abs(b), 1e-300)))


def _rms_rel(a: np.ndarray, b: np.ndarray) -> float:
    rel = (np.asarray(a, dtype=float) - np.asarray(b, dtype=float)) / np.maximum(np.abs(b), 1e-300)
    return float(np.sqrt(np.nanmean(rel * rel)))


def _max_abs(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.nanmax(np.abs(np.asarray(a, dtype=float) - np.asarray(b, dtype=float))))


def _points_from_npz(data: Any) -> list[PrimitivePoint]:
    return [PrimitivePoint.from_npz(data, idx) for idx in range(np.asarray(data["x"], dtype=float).size)]


def profile_mach_spline_summary(*, profile_path: str | Path, summary_path: str | Path) -> dict[str, Any]:
    profile_path = Path(profile_path).resolve()
    summary_path = Path(summary_path).resolve()
    config = FreidbergConfig.from_summary_and_profile(summary_path, profile_path)
    with np.load(profile_path) as data:
        x = np.asarray(data["x"], dtype=float)
        n_p = np.asarray(data["n_p"], dtype=float)
        T_e = np.asarray(data["T_e"], dtype=float)
        mach = np.asarray(data["mach"], dtype=float)
        A_ref = np.asarray(data["A"], dtype=float)
        T_p_ref = np.asarray(data["T_p"], dtype=float)
        v_ref = np.asarray(data["v_p"], dtype=float)
        Z_ref = np.asarray(data["Z"], dtype=float)
        original_points = _points_from_npz(data)

    exact_points = reconstruct_points_from_mach(x=x, n_p=n_p, T_e=T_e, mach=mach, config=config)
    exact = primitive_arrays(exact_points)
    design = MachSplineDesign.project_from_profile(x=x, mach=mach)
    x_norm = (x - float(x[0])) / max(float(x[-1] - x[0]), 1e-300)
    fitted_profile = design.evaluate_on_normalized_grid(x_norm, length=float(x[-1] - x[0]), mach_in=float(mach[0]))
    mach_fit = np.asarray(fitted_profile["mach"], dtype=float)
    fitted_points = reconstruct_points_from_mach(x=x, n_p=n_p, T_e=T_e, mach=mach_fit, config=config)
    fitted = primitive_arrays(fitted_points)

    original_defects = interval_defects_from_points(original_points, config).summary()
    fitted_defects = interval_defects_from_points(fitted_points, config).summary()

    return {
        "profile_path": str(profile_path),
        "summary_path": str(summary_path),
        "config": config.to_dict(),
        "mach_spline_design": design.to_dict(),
        "exact_mach_reconstruction": {
            "max_rel_A_error": _max_rel(exact["A"], A_ref),
            "max_abs_T_p_error_K": _max_abs(exact["T_p"], T_p_ref),
            "max_rel_v_p_error": _max_rel(exact["v_p"], v_ref),
            "max_abs_Z_error": _max_abs(exact["Z"], Z_ref),
        },
        "fitted_mach_spline_reconstruction": {
            "max_rel_mach_error": _max_rel(mach_fit, mach),
            "rms_rel_mach_error": _rms_rel(mach_fit, mach),
            "max_rel_A_error": _max_rel(fitted["A"], A_ref),
            "rms_rel_A_error": _rms_rel(fitted["A"], A_ref),
            "max_abs_T_p_error_K": _max_abs(fitted["T_p"], T_p_ref),
            "max_abs_Z_error": _max_abs(fitted["Z"], Z_ref),
        },
        "freidberg_interval_defects_original": original_defects,
        "freidberg_interval_defects_fitted_mach": fitted_defects,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Shadow-test Mach-spline closure against a stored primitive profile.")
    parser.add_argument("profile", type=Path)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    report = profile_mach_spline_summary(profile_path=args.profile, summary_path=args.summary)
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
