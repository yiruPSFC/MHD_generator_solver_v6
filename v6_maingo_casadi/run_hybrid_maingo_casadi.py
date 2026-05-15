#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from v6_maingo_casadi.core import (
    _DEFAULT_BASELINE_SUMMARY,
    _import_maingopy,
    OBJECTIVE_PROFILES,
    WORKING_FLUID_PROFILES,
    run_hybrid_maingo_casadi,
)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run the hybrid MAiNGO coarse search followed by the v6_casadi_v2 continuation handoff."
    )
    p.add_argument(
        "--baseline-summary",
        type=str,
        default=str(_DEFAULT_BASELINE_SUMMARY),
        help="baseline continuation summary used to seed inlet windows, schedule, and warm-profile projection",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default=str(Path(__file__).resolve().parent / "outputs" / "hybrid_default"),
        help=(
            "output directory for maingo_summary.json, maingo_coarse_profile.npz, "
            "maingo_handoff_profile.npz, hybrid_summary.json, and continuation artifacts"
        ),
    )
    p.add_argument(
        "--coarse-n-intervals",
        type=int,
        default=40,
        help="number of coarse intervals used inside the reduced MAiNGO rollout",
    )
    p.add_argument(
        "--coarse-model",
        type=str,
        default="reduced_implicit",
        choices=("reduced_implicit", "mach_spline", "mach_spline_rk4_soft", "mach_spline_trapezoid"),
        help=(
            "reduced MAiNGO coarse model: existing area-spline closure, "
            "Mach-spline-derived area closure, soft RK4 Mach-spline candidate generator, "
            "or low-dimensional Mach-spline parametric residual transcription"
        ),
    )
    p.add_argument(
        "--reduced-implicit-newton-steps",
        type=int,
        default=10,
        help="fixed Gauss-Newton iterations per interval for the reduced implicit coarse model",
    )
    p.add_argument(
        "--critical-mode",
        action="store_true",
        help="enable the optional sonic critical-point search variable and compatibility residual",
    )
    p.add_argument(
        "--critical-residual-tolerance",
        type=float,
        default=1e-4,
        help="max scaled residual allowed for the optional sonic critical-point compatibility check",
    )
    p.add_argument(
        "--skip-rk4-benchmark",
        action="store_true",
        help="skip the post-hoc RK4 reduced benchmark at the reduced-implicit handoff decision",
    )
    p.add_argument(
        "--mach-reference-profile",
        type=str,
        default="",
        help=(
            "optional NPZ profile used to project the nominal Mach spline for --coarse-model mach_spline; "
            "if omitted, the workflow tries to infer one from --initial-solution-json"
        ),
    )
    p.add_argument(
        "--mach-window-radius",
        type=float,
        default=1.0,
        help="half-width applied around the projected Mach-spline coefficients for --coarse-model mach_spline",
    )
    p.add_argument(
        "--mach-fixed-inlet",
        action="store_true",
        help=(
            "for --coarse-model mach_spline_trapezoid, freeze inlet variables at the "
            "reference/initial values and expose only Mach plus state spline controls"
        ),
    )
    p.add_argument(
        "--mach-rk4-det-branch",
        type=str,
        default="positive",
        choices=("positive", "negative", "none"),
        help="determinant sign branch enforced by --coarse-model mach_spline_rk4_soft",
    )
    p.add_argument(
        "--mach-rk4-det-floor",
        type=float,
        default=1e-3,
        help="minimum signed scaled determinant margin for --coarse-model mach_spline_rk4_soft",
    )
    p.add_argument(
        "--handoff-n-intervals",
        type=int,
        default=80,
        help="number of intervals used when exporting the MAiNGO best profile as a CasADi warm start",
    )
    p.add_argument(
        "--maingo-settings",
        type=str,
        default="",
        help="optional MAiNGO settings file passed through read_settings(...) before solve()",
    )
    p.add_argument(
        "--maingo-max-time",
        type=float,
        default=0.0,
        help="optional MAiNGO CPU time limit in seconds; <=0 leaves the default solver setting untouched",
    )
    p.add_argument(
        "--objective-profile",
        type=str,
        default="lab_poc_v2",
        choices=OBJECTIVE_PROFILES,
        help="coarse MAiNGO objective profile",
    )
    p.add_argument(
        "--working-fluid-profile",
        type=str,
        default=None,
        choices=WORKING_FLUID_PROFILES,
        help="working gas / seed profile used by the MAiNGO coarse closure; defaults to the baseline summary profile",
    )
    p.add_argument(
        "--search-window-json",
        type=str,
        default="",
        help=(
            "optional JSON file with absolute inlet_windows and/or area_design_windows "
            "guess/min/max overrides; applied after baseline loading and before bound factors"
        ),
    )
    p.add_argument(
        "--initial-solution-json",
        type=str,
        default="",
        help=(
            "optional JSON or maingo_summary.json containing status.solution_point; "
            "used as the MAiNGO initial point before solve()"
        ),
    )
    p.add_argument(
        "--skip-casadi-handoff",
        action="store_true",
        help="write the MAiNGO best profile and summary without running v6_casadi_v2 continuation",
    )
    p.add_argument(
        "--n-p-in-lower-factor",
        type=float,
        default=1.0,
        help="multiplicative factor applied to the baseline-aligned lower bound for n_p_in",
    )
    p.add_argument(
        "--n-p-in-upper-factor",
        type=float,
        default=1.0,
        help="multiplicative factor applied to the baseline-aligned upper bound for n_p_in",
    )
    p.add_argument(
        "--t-e-in-lower-factor",
        type=float,
        default=1.0,
        help="multiplicative factor applied to the baseline-aligned lower bound for T_e_in",
    )
    p.add_argument(
        "--t-e-in-upper-factor",
        type=float,
        default=1.0,
        help="multiplicative factor applied to the baseline-aligned upper bound for T_e_in",
    )
    p.add_argument(
        "--z-in-lower-factor",
        type=float,
        default=1.0,
        help="multiplicative factor applied to the baseline-aligned lower bound for Z_in",
    )
    p.add_argument(
        "--z-in-upper-factor",
        type=float,
        default=1.0,
        help="multiplicative factor applied to the baseline-aligned upper bound for Z_in",
    )
    p.add_argument(
        "--i0-lower-factor",
        type=float,
        default=1.0,
        help="multiplicative factor applied to the baseline-aligned lower bound for I_0",
    )
    p.add_argument(
        "--i0-upper-factor",
        type=float,
        default=1.0,
        help="multiplicative factor applied to the baseline-aligned upper bound for I_0",
    )
    p.add_argument(
        "--seed-fraction-lower-factor",
        type=float,
        default=1.0,
        help="multiplicative factor applied to the baseline-aligned lower bound for seed_fraction",
    )
    p.add_argument(
        "--seed-fraction-upper-factor",
        type=float,
        default=1.0,
        help="multiplicative factor applied to the baseline-aligned upper bound for seed_fraction",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    _import_maingopy()
    result = run_hybrid_maingo_casadi(
        out_dir=args.out_dir,
        baseline_summary_path=args.baseline_summary,
        coarse_n_intervals=int(args.coarse_n_intervals),
        handoff_n_intervals=int(args.handoff_n_intervals),
        maingo_settings_path=str(args.maingo_settings),
        maingo_max_time=None if float(args.maingo_max_time) <= 0.0 else float(args.maingo_max_time),
        coarse_model=str(args.coarse_model),
        reduced_implicit_newton_steps=int(args.reduced_implicit_newton_steps),
        critical_mode=bool(args.critical_mode),
        critical_residual_tolerance=float(args.critical_residual_tolerance),
        mach_reference_profile_path=(
            None if not str(args.mach_reference_profile).strip() else str(args.mach_reference_profile)
        ),
        mach_window_radius=float(args.mach_window_radius),
        mach_fixed_inlet=bool(args.mach_fixed_inlet),
        mach_rk4_det_branch=str(args.mach_rk4_det_branch),
        mach_rk4_det_floor=float(args.mach_rk4_det_floor),
        include_rk4_benchmark=not bool(args.skip_rk4_benchmark),
        objective_profile=str(args.objective_profile),
        working_fluid_profile=args.working_fluid_profile,
        search_window_json=None if not str(args.search_window_json).strip() else str(args.search_window_json),
        initial_solution_json=(
            None if not str(args.initial_solution_json).strip() else str(args.initial_solution_json)
        ),
        skip_casadi_handoff=bool(args.skip_casadi_handoff),
        n_p_in_lower_factor=float(args.n_p_in_lower_factor),
        n_p_in_upper_factor=float(args.n_p_in_upper_factor),
        T_e_in_lower_factor=float(args.t_e_in_lower_factor),
        T_e_in_upper_factor=float(args.t_e_in_upper_factor),
        Z_in_lower_factor=float(args.z_in_lower_factor),
        Z_in_upper_factor=float(args.z_in_upper_factor),
        I_0_lower_factor=float(args.i0_lower_factor),
        I_0_upper_factor=float(args.i0_upper_factor),
        seed_fraction_lower_factor=float(args.seed_fraction_lower_factor),
        seed_fraction_upper_factor=float(args.seed_fraction_upper_factor),
    )
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
