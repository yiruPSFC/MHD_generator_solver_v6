from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .cases.yamasaki2004 import YAMASAKI2004
from .design import load_case_config
from .geometry import LogAreaSplineControl


def _json_default(value: Any):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    raise TypeError(f"object of type {type(value).__name__} is not JSON serializable")


def _safe_rel_error(value: float, reference: float) -> float:
    return float((float(value) - float(reference)) / max(abs(float(reference)), 1e-300))


def _stats(values: np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=float).reshape(-1)
    return {
        "min": float(np.nanmin(arr)),
        "max": float(np.nanmax(arr)),
        "start": float(arr[0]),
        "end": float(arr[-1]),
    }


def _project_area_profile(
    *,
    name: str,
    x_norm: np.ndarray,
    area: np.ndarray,
    length_m: float,
    reported_throat_m2: float,
    reported_exit_m2: float,
    current_area_scale_m2: float,
    area_bounds: dict[str, dict[str, float]],
) -> dict[str, Any]:
    x = np.asarray(x_norm, dtype=float) * float(length_m)
    area = np.asarray(area, dtype=float)
    control = LogAreaSplineControl.project_from_profile(x=x, A=area)
    control_dict = control.to_dict()
    reconstructed = control.evaluate_profile(
        length=float(length_m),
        n_intervals=int(x_norm.size - 1),
        area_scale=float(area[0]),
    )
    log_area = np.log(np.maximum(area / max(float(area[0]), 1e-300), 1e-300))
    log_fit = np.asarray(reconstructed["logA"], dtype=float)
    rel_fit_error = np.abs(np.asarray(reconstructed["A"], dtype=float) - area) / np.maximum(np.abs(area), 1e-300)
    containment = {
        key: {
            "value": float(control_dict[key]),
            "min": float(area_bounds[key]["min"]),
            "max": float(area_bounds[key]["max"]),
            "inside_current_window": bool(
                float(area_bounds[key]["min"]) <= float(control_dict[key]) <= float(area_bounds[key]["max"])
            ),
        }
        for key in ("a1", "a2", "a3")
    }
    projection_factor_to_reported_annulus = None
    return {
        "name": name,
        "length_m": float(length_m),
        "area_scale_m2": float(area[0]),
        "area_ratio_exit_to_throat": float(area[-1] / max(float(area[0]), 1e-300)),
        "reported_endpoint_errors": {
            "throat_rel": _safe_rel_error(float(area[0]), float(reported_throat_m2)),
            "exit_rel": _safe_rel_error(float(area[-1]), float(reported_exit_m2)),
        },
        "requires_solver_area_scale_change": bool(
            not math.isclose(float(area[0]), float(current_area_scale_m2), rel_tol=1e-12, abs_tol=1e-15)
        ),
        "current_solver_area_scale_m2": float(current_area_scale_m2),
        "projected_log_area_control": control_dict,
        "current_area_window_containment": containment,
        "all_projected_a_inside_current_window": bool(
            all(item["inside_current_window"] for item in containment.values())
        ),
        "log_area_stats": _stats(log_area),
        "sigma_logA_per_m_stats": _stats(np.gradient(log_area, x)),
        "projection_fit": {
            "max_abs_log_error": float(np.nanmax(np.abs(log_fit - log_area))),
            "max_abs_relative_area_error": float(np.nanmax(rel_fit_error)),
            "note": "Projection uses existing direct cubic logA controls at x/L = 1/3, 2/3, 1.",
        },
        "projection_factor_to_reported_annulus": projection_factor_to_reported_annulus,
    }


def _write_profiles_csv(path: Path, rows: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows.keys())
    n = len(next(iter(rows.values())))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for idx in range(n):
            writer.writerow({key: float(np.asarray(rows[key], dtype=float)[idx]) for key in keys})


def _markdown_summary(report: dict[str, Any]) -> str:
    variants = report["area_variants"]
    current = variants["current_endpoint_fitted_effective_area"]
    annular = variants["annular_2pi_r_h_radial_length"]
    swirl = variants["cos_swirl_annular_area_streamline_length"]
    inferred = report["inferred_swirl_from_reported_cross_sections"]
    return (
        "# Yamasaki Geometry Audit\n\n"
        "This audit separates three issues that were previously easy to mix: "
        "total current versus current density, paper endpoint cross-section recovery, "
        "and optional/variable swirl streamwise length.\n\n"
        "## Main Findings\n\n"
        f"- Current code geometry exactly matches the reported throat/exit cross sections: "
        f"throat error `{current['reported_endpoint_errors']['throat_rel']:.6g}`, "
        f"exit error `{current['reported_endpoint_errors']['exit_rel']:.6g}`.\n"
        f"- Direct `2*pi*r*h` annular area does not match the reported throat area: "
        f"throat error `{annular['reported_endpoint_errors']['throat_rel']:.6g}`, "
        f"exit error `{annular['reported_endpoint_errors']['exit_rel']:.6g}`.\n"
        f"- Constant 45 degree `cos(theta)*2*pi*r*h` is close at the throat but far at exit: "
        f"throat error `{swirl['reported_endpoint_errors']['throat_rel']:.6g}`, "
        f"exit error `{swirl['reported_endpoint_errors']['exit_rel']:.6g}`.\n"
        f"- Existing `a1/a2/a3` window contains current endpoint-fitted geometry: "
        f"`{current['all_projected_a_inside_current_window']}`, but contains the constant-swirl "
        f"annular geometry: `{swirl['all_projected_a_inside_current_window']}`.\n"
        f"- With 45 degree swirl, radial length `{report['lengths']['radial_length_m']:.6g} m` "
        f"becomes streamwise length `{report['lengths']['streamline_length_m']:.6g} m`.\n"
        f"- If the paper cross sections are interpreted as `A = 2*pi*r*h*cos(theta)`, "
        f"the implied angle changes from `{inferred['flow_angle_deg']['throat']:.3g}` degrees "
        f"at the throat to `{inferred['flow_angle_deg']['exit']:.3g}` degrees at the exit; "
        f"the corresponding variable-swirl length is `{inferred['streamline_length_m']:.6g} m`.\n"
        "\n"
        "See `geometry_audit.json` for the full numeric report and `geometry_profiles.csv` "
        "for the sampled profiles.\n"
    )


def build_geometry_audit(*, n_intervals: int = 240, swirl_angle_deg: float = 45.0) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    paper = YAMASAKI2004
    config = load_case_config(case="yamasaki2004")
    geom = paper.geometry
    x_norm = np.linspace(0.0, 1.0, int(n_intervals) + 1, dtype=float)
    r = np.asarray(geom.radius(x_norm), dtype=float)
    h = np.asarray(geom.height(x_norm), dtype=float)
    radial_x = x_norm * float(geom.length_m)
    theta = math.radians(float(swirl_angle_deg))
    cos_theta = math.cos(theta)
    if cos_theta <= 0.0:
        raise ValueError("--swirl-angle-deg must be smaller than 90 degrees.")
    streamline_length = float(geom.length_m) / cos_theta
    streamline_x = x_norm * streamline_length

    reported_effective_area = np.asarray(geom.effective_area(x_norm), dtype=float)
    reported_linear_area = float(geom.cross_section_throat_m2) + x_norm * (
        float(geom.cross_section_exit_m2) - float(geom.cross_section_throat_m2)
    )
    annular_area = np.asarray(geom.annular_area(x_norm), dtype=float)
    cos_swirl_annular_area = cos_theta * annular_area
    area_bounds = config.bounds.to_dict()

    variants = {
        "current_endpoint_fitted_effective_area": _project_area_profile(
            name="current_endpoint_fitted_effective_area",
            x_norm=x_norm,
            area=reported_effective_area,
            length_m=float(geom.length_m),
            reported_throat_m2=float(geom.cross_section_throat_m2),
            reported_exit_m2=float(geom.cross_section_exit_m2),
            current_area_scale_m2=float(config.area_scale_m2),
            area_bounds=area_bounds,
        ),
        "reported_cross_section_linear_in_radius": _project_area_profile(
            name="reported_cross_section_linear_in_radius",
            x_norm=x_norm,
            area=reported_linear_area,
            length_m=float(geom.length_m),
            reported_throat_m2=float(geom.cross_section_throat_m2),
            reported_exit_m2=float(geom.cross_section_exit_m2),
            current_area_scale_m2=float(config.area_scale_m2),
            area_bounds=area_bounds,
        ),
        "annular_2pi_r_h_radial_length": _project_area_profile(
            name="annular_2pi_r_h_radial_length",
            x_norm=x_norm,
            area=annular_area,
            length_m=float(geom.length_m),
            reported_throat_m2=float(geom.cross_section_throat_m2),
            reported_exit_m2=float(geom.cross_section_exit_m2),
            current_area_scale_m2=float(config.area_scale_m2),
            area_bounds=area_bounds,
        ),
        "current_endpoint_area_streamline_length": _project_area_profile(
            name="current_endpoint_area_streamline_length",
            x_norm=x_norm,
            area=reported_effective_area,
            length_m=streamline_length,
            reported_throat_m2=float(geom.cross_section_throat_m2),
            reported_exit_m2=float(geom.cross_section_exit_m2),
            current_area_scale_m2=float(config.area_scale_m2),
            area_bounds=area_bounds,
        ),
        "cos_swirl_annular_area_streamline_length": _project_area_profile(
            name="cos_swirl_annular_area_streamline_length",
            x_norm=x_norm,
            area=cos_swirl_annular_area,
            length_m=streamline_length,
            reported_throat_m2=float(geom.cross_section_throat_m2),
            reported_exit_m2=float(geom.cross_section_exit_m2),
            current_area_scale_m2=float(config.area_scale_m2),
            area_bounds=area_bounds,
        ),
    }

    annular_projection_factor = reported_effective_area / np.maximum(annular_area, 1e-300)
    annular_projection_factor_clipped = np.clip(annular_projection_factor, 1e-12, 1.0)
    implied_angle = np.degrees(np.arccos(annular_projection_factor_clipped))
    implied_swirl_ratio = np.sqrt(np.maximum(1.0 / (annular_projection_factor_clipped**2) - 1.0, 0.0))
    inferred_streamline_length = float(np.trapezoid(1.0 / annular_projection_factor_clipped, radial_x))
    current_range = paper.hall_current_A
    current_values = {
        "min": float(current_range.minimum),
        "nominal": float(current_range.nominal),
        "max": float(current_range.maximum),
    }

    current_convention = {
        "model_contract": "I_0 is total Hall current in amperes; local current density is J_x(x)=I_0/A(x).",
        "paper_hall_current_A": current_values,
        "if_paper_current_is_total_current_density_A_m2": {
            name: {
                key: float(value * variant["area_scale_m2"])
                for key, value in current_values.items()
            }
            for name, variant in variants.items()
        },
        "correct_Jx_density_from_total_current_A_m2": {
            name: {
                key: float(value / max(float(variant["area_scale_m2"]), 1e-300))
                for key, value in current_values.items()
            }
            for name, variant in variants.items()
        },
        "normalized_area_trap": {
            "A_in_normalized_m2": 1.0,
            "Jx_if_I0_1000_and_Ain_1_A_m2": 1000.0,
            "Jx_if_I0_1000_and_reported_Ain_A_m2": float(1000.0 / float(geom.cross_section_throat_m2)),
            "density_underestimate_factor_if_Ain_1_used_as_physical": float(1.0 / float(geom.cross_section_throat_m2)),
        },
    }

    report = {
        "paper_source_values": {
            "r_throat_m": float(geom.r_throat_m),
            "r_exit_m": float(geom.r_exit_m),
            "height_throat_m": float(geom.height_throat_m),
            "height_exit_m": float(geom.height_exit_m),
            "reported_cross_section_throat_m2": float(geom.cross_section_throat_m2),
            "reported_cross_section_exit_m2": float(geom.cross_section_exit_m2),
            "reported_cross_section_area_ratio": float(geom.area_ratio),
            "magnetic_field_T": float(paper.magnetic_field_T),
        },
        "lengths": {
            "radial_length_m": float(geom.length_m),
            "swirl_angle_deg": float(swirl_angle_deg),
            "cos_swirl_angle": float(cos_theta),
            "streamline_length_m": float(streamline_length),
            "streamline_over_radial_length": float(1.0 / cos_theta),
            "constant_angle_note": (
                "For constant swirl angle, r is still linear in normalized streamwise coordinate; "
                "the area shape versus x/L is unchanged, but d(logA)/ds is scaled by cos(theta)."
            ),
        },
        "paper_text_interpretation": {
            "quoted_geometry_statement": (
                "The disk height varies linearly from 19.6 mm at a throat (r=85 mm) "
                "to 24.0 mm at a channel exit (r=276 mm). The throat cross section along "
                "the swirl-flow direction is 6.9e3 mm^2 and the exit cross section is "
                "41.2e3 mm^2; the swirl factor is 0 at the exit."
            ),
            "exit_area_decimal_note": (
                "The exit area must be 41.2e3 mm^2, not 412e3 mm^2, to match the stated "
                "area ratio of about 5.9."
            ),
        },
        "current_code_geometry_semantics": {
            "area_formula": (
                "effective_area = 2*pi*r*h multiplied by a linear correction factor chosen to match "
                "the reported throat and exit cross sections."
            ),
            "not_equivalent_to": [
                "raw 2*pi*r*h",
                "constant cos(theta)*2*pi*r*h with theta=45 deg",
            ],
            "projection_factor_reported_effective_over_annular": {
                "min": float(np.nanmin(annular_projection_factor)),
                "max": float(np.nanmax(annular_projection_factor)),
                "throat": float(annular_projection_factor[0]),
                "exit": float(annular_projection_factor[-1]),
            },
            "if_projection_factor_is_cos_angle_implied_deg": {
                "min": float(np.nanmin(implied_angle)),
                "max": float(np.nanmax(implied_angle)),
                "throat": float(implied_angle[0]),
                "exit": float(implied_angle[-1]),
            },
        },
        "inferred_swirl_from_reported_cross_sections": {
            "assumption": (
                "Interpret the reported swirl-flow-direction cross section as "
                "A_reported = 2*pi*r*h*cos(theta), where theta is the local angle "
                "between the streamline and radial direction."
            ),
            "projection_factor_cos_theta": {
                "min": float(np.nanmin(annular_projection_factor_clipped)),
                "max": float(np.nanmax(annular_projection_factor_clipped)),
                "throat": float(annular_projection_factor_clipped[0]),
                "exit": float(annular_projection_factor_clipped[-1]),
            },
            "flow_angle_deg": {
                "min": float(np.nanmin(implied_angle)),
                "max": float(np.nanmax(implied_angle)),
                "throat": float(implied_angle[0]),
                "exit": float(implied_angle[-1]),
            },
            "swirl_ratio_tan_theta": {
                "min": float(np.nanmin(implied_swirl_ratio)),
                "max": float(np.nanmax(implied_swirl_ratio)),
                "throat": float(implied_swirl_ratio[0]),
                "exit": float(implied_swirl_ratio[-1]),
            },
            "streamline_length_m": inferred_streamline_length,
            "streamline_over_radial_length": float(inferred_streamline_length / float(geom.length_m)),
            "interpretation": (
                "This is consistent with an approximately 45 degree throat swirl that decays "
                "toward nearly radial flow at the exit. It is not a constant 45 degree spiral."
            ),
        },
        "area_variants": variants,
        "current_convention": current_convention,
        "audit_interpretation": {
            "area_window_result": (
                "The current a-window is wide enough around the endpoint-fitted reported cross-section geometry. "
                "It is not a substitute for changing area_scale_m2 if the intended geometry uses raw annular "
                "or cos-swirl annular throat area."
            ),
            "swirl_result": (
                "A constant 45 degree swirl does not reproduce both reported cross-section endpoints. "
                "The paper text is more consistent with a large throat swirl angle that decays toward zero "
                "at the exit; the inferred variable-swirl length is reported separately from the constant "
                "sqrt(2) estimate."
            ),
            "current_result": (
                "Using paper Hall current directly as I_0 is consistent with the current code contract only if the paper "
                "quantity is total current. The geometry audit reports the derived J_x density so old normalized-current "
                "artifacts can be checked explicitly."
            ),
        },
    }

    profile_rows = {
        "x_norm": x_norm,
        "radial_x_m": radial_x,
        "streamline_x_m": streamline_x,
        "radius_m": r,
        "height_m": h,
        "reported_effective_area_m2": reported_effective_area,
        "reported_linear_area_m2": reported_linear_area,
        "annular_2pi_r_h_area_m2": annular_area,
        "cos_swirl_annular_area_m2": cos_swirl_annular_area,
        "reported_effective_over_annular": annular_projection_factor,
        "implied_projection_angle_deg": implied_angle,
        "implied_swirl_ratio_tan_theta": implied_swirl_ratio,
    }
    return report, profile_rows


def write_geometry_audit(*, out_dir: Path, n_intervals: int, swirl_angle_deg: float) -> dict[str, Any]:
    report, profile_rows = build_geometry_audit(n_intervals=n_intervals, swirl_angle_deg=swirl_angle_deg)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "geometry_audit.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    _write_profiles_csv(out_dir / "geometry_profiles.csv", profile_rows)
    (out_dir / "README.md").write_text(_markdown_summary(report), encoding="utf-8")
    return {
        "out_dir": str(out_dir),
        "geometry_audit_json": str(out_dir / "geometry_audit.json"),
        "geometry_profiles_csv": str(out_dir / "geometry_profiles.csv"),
        "readme": str(out_dir / "README.md"),
        "headline": {
            "current_endpoint_a_inside_window": report["area_variants"][
                "current_endpoint_fitted_effective_area"
            ]["all_projected_a_inside_current_window"],
            "cos_swirl_a_inside_window": report["area_variants"][
                "cos_swirl_annular_area_streamline_length"
            ]["all_projected_a_inside_current_window"],
            "current_code_implied_projection_angle_deg": report["current_code_geometry_semantics"][
                "if_projection_factor_is_cos_angle_implied_deg"
            ],
            "streamline_length_m": report["lengths"]["streamline_length_m"],
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit Yamasaki 2004 geometry and current/area conventions.")
    parser.add_argument("--n-intervals", type=int, default=240)
    parser.add_argument("--swirl-angle-deg", type=float, default=45.0)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("v6_firedrake_reduced/outputs/yamasaki_geometry_audit_20260519"),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = write_geometry_audit(
        out_dir=Path(args.out_dir),
        n_intervals=int(args.n_intervals),
        swirl_angle_deg=float(args.swirl_angle_deg),
    )
    print(json.dumps(payload, indent=2, sort_keys=True, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
