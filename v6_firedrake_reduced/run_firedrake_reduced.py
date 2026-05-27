from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from .constraints import evaluate_velikhov_node_constraints
from .design import CASE_NAMES, GEOMETRY_LENGTH_MODES, CaseConfig, DesignVector, load_case_config
from .forward import EQUATION_FORMS
from .forward import FiredrakeUnavailableError, solve_forward
from .freidberg_branch_audit import audit_freidberg_branches
from .reference_profile import (
    build_freidberg_reference_profile,
    build_implicit_reference_profile,
    build_reference_profile,
)
from .reduced_functional import (
    minimize_constrained_slsqp,
    minimize_constrained_slsqp_state_restarts,
    minimize_multistart,
)
from .transport import ELECTRON_TRANSPORT_MODELS, normalize_electron_transport


def _json_default(value: Any):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    raise TypeError(f"object of type {type(value).__name__} is not JSON serializable")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")


def _with_design(config: CaseConfig, design: DesignVector, *, metadata: dict[str, Any] | None = None) -> CaseConfig:
    return CaseConfig(
        case=config.case,
        objective_profile=config.objective_profile,
        length_m=config.length_m,
        area_scale_m2=config.area_scale_m2,
        B_T=float(design.B_T),
        working_fluid_profile=config.working_fluid_profile,
        n_intervals=config.n_intervals,
        design=design,
        bounds=config.bounds,
        metadata=dict(config.metadata if metadata is None else metadata),
    )


def _format_bound_violations(violations: list[dict[str, Any]]) -> str:
    return "; ".join(
        f"{item['name']}={float(item['value']):.12g} outside [{float(item['min']):.12g}, {float(item['max']):.12g}]"
        for item in violations
    )


def _design_from_json(path: Path, *, config: CaseConfig, allow_out_of_bounds: bool = False) -> DesignVector:
    design = DesignVector.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
    violations = config.bounds.violations(design)
    if violations and not bool(allow_out_of_bounds):
        raise ValueError(
            "--design-json is outside the active case bounds. "
            "Use --allow-out-of-bounds-design-json only when exact out-of-window replay is intended. "
            f"Violations: {_format_bound_violations(violations)}"
        )
    return design


def _profile_from_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return {name: np.asarray(data[name], dtype=float) for name in data.files}


def _append_failure(path: Path, *, design: DesignVector, error: str, context: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"design": design.to_dict(), "error": str(error), "context": dict(context)}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _write_snapshot(path: Path, config: CaseConfig) -> None:
    text = (
        "# Firedrake reduced run snapshot\n\n"
        f"- case: `{config.case}`\n"
        f"- objective: `{config.objective_profile}`\n"
        f"- n_intervals: `{config.n_intervals}`\n"
        f"- forward solver: `implicit_residual_firedrake_snes_newtonls`\n"
        f"- design variables: `{', '.join(config.design.to_dict().keys())}`\n"
    )
    path.write_text(text, encoding="utf-8")


def _velikhov_constraint_floor(config: CaseConfig) -> float:
    return float(config.metadata.get("velikhov_hard_floor", 0.0))


def _velikhov_node_constraints_payload(
    *,
    profile: dict[str, Any],
    design: DesignVector,
    config: CaseConfig,
) -> dict[str, Any]:
    try:
        return evaluate_velikhov_node_constraints(
            profile=profile,
            design=design,
            config=config,
            floor=_velikhov_constraint_floor(config),
        ).to_dict()
    except Exception as exc:
        return {
            "sampling": "nodes",
            "floor": _velikhov_constraint_floor(config),
            "error": f"{type(exc).__name__}: {exc}",
        }


def _validate_optimizer_options(args: argparse.Namespace) -> None:
    if str(args.velikhov_constraint_mode) == "hard" and str(args.velikhov_mode) == "penalty":
        raise ValueError(
            "--velikhov-constraint-mode hard cannot be combined with --velikhov-mode penalty. "
            "Use diagnostic mode so the constrained optimizer sees the raw enthalpy objective."
        )
    if str(args.optimizer) == "projected_gradient" and str(args.velikhov_constraint_mode) == "hard":
        raise ValueError("--velikhov-constraint-mode hard requires --optimizer constrained_slsqp.")


def _write_freidberg_branch_audit(
    *,
    out_dir: Path,
    filename: str,
    profile_kind: str,
    profile: dict[str, Any],
    config: CaseConfig,
    branch_policy: str,
    tolerance_K: float,
) -> dict[str, Any]:
    audit_path = out_dir / filename
    try:
        audit = audit_freidberg_branches(
            profile=profile,
            design=config.design,
            config=config,
            branch_policy=branch_policy,
            tolerance_K=float(tolerance_K),
        )
        payload = {
            **audit,
            "profile_kind": profile_kind,
            "case_config": config.to_dict(),
        }
        _write_json(audit_path, payload)
        return {
            "ok": bool(audit.get("summary", {}).get("ok", False)),
            "profile_kind": profile_kind,
            "json": str(audit_path),
            "summary": dict(audit.get("summary", {})),
        }
    except Exception as exc:
        payload = {
            "audit": "freidberg_branch_audit",
            "schema_version": 1,
            "profile_kind": profile_kind,
            "case_config": config.to_dict(),
            "error": f"{type(exc).__name__}: {exc}",
            "summary": {
                "ok": False,
                "branch_policy": str(branch_policy),
                "tolerance_K": float(tolerance_K),
            },
        }
        _write_json(audit_path, payload)
        return {
            "ok": False,
            "profile_kind": profile_kind,
            "json": str(audit_path),
            "error": payload["error"],
            "summary": dict(payload["summary"]),
        }


def _run_evaluate(
    config: CaseConfig,
    out_dir: Path,
    *,
    initial_profile: dict[str, Any] | None = None,
    initial_profile_context: dict[str, Any] | None = None,
    freidberg_branch_audit: bool = False,
    freidberg_branch_policy: str = "continuity",
    freidberg_branch_tolerance: float = 1e-3,
) -> dict[str, Any]:
    failure_log = out_dir / "failure_log.jsonl"
    result = solve_forward(design=config.design, config=config, initial_profile=initial_profile)
    if not result.ok:
        failed_profile_npz = None
        branch_audit_info = None
        velikhov_node_constraints = None
        if result.profile is not None:
            failed_profile_npz = out_dir / "failed_profile.npz"
            np.savez(failed_profile_npz, **result.profile)
            velikhov_node_constraints = _velikhov_node_constraints_payload(
                profile=result.profile,
                design=config.design,
                config=config,
            )
            if freidberg_branch_audit:
                branch_audit_info = _write_freidberg_branch_audit(
                    out_dir=out_dir,
                    filename="freidberg_branch_audit_failed.json",
                    profile_kind="failed_profile",
                    profile=result.profile,
                    config=config,
                    branch_policy=freidberg_branch_policy,
                    tolerance_K=float(freidberg_branch_tolerance),
                )
        _append_failure(
            failure_log,
            design=config.design,
            error=result.error or "unknown forward failure",
            context={
                "mode": "evaluate",
                "diagnostics": result.diagnostics,
                "initial_profile_context": dict(initial_profile_context or {}),
            },
        )
        payload = {
            "ok": False,
            "case_config": config.to_dict(),
            "diagnostics": result.diagnostics,
            "error": result.error,
            "failure_log": str(failure_log),
            "initial_profile_context": dict(initial_profile_context or {}),
            "failed_profile_npz": None if failed_profile_npz is None else str(failed_profile_npz),
            "failed_metrics": None if result.metrics is None else result.metrics.to_dict(),
            "velikhov_node_constraints": velikhov_node_constraints,
        }
        if branch_audit_info is not None:
            payload["freidberg_branch_audit"] = branch_audit_info
        _write_json(out_dir / "run_summary.json", payload)
        return payload

    assert result.profile is not None
    assert result.metrics is not None
    np.savez(out_dir / "profile.npz", **result.profile)
    _write_json(out_dir / "best_design.json", config.design.to_dict())
    branch_audit_info = None
    if freidberg_branch_audit:
        branch_audit_info = _write_freidberg_branch_audit(
            out_dir=out_dir,
            filename="freidberg_branch_audit.json",
            profile_kind="profile",
            profile=result.profile,
            config=config,
            branch_policy=freidberg_branch_policy,
            tolerance_K=float(freidberg_branch_tolerance),
        )
    payload = {
        "ok": True,
        "case_config": config.to_dict(),
        "metrics": result.metrics.to_dict(),
        "velikhov_node_constraints": _velikhov_node_constraints_payload(
            profile=result.profile,
            design=config.design,
            config=config,
        ),
        "diagnostics": result.diagnostics,
        "profile_npz": str(out_dir / "profile.npz"),
        "failure_log": str(failure_log),
        "initial_profile_context": dict(initial_profile_context or {}),
    }
    if branch_audit_info is not None:
        payload["freidberg_branch_audit"] = branch_audit_info
    _write_json(out_dir / "run_summary.json", payload)
    _write_snapshot(out_dir / "README_snapshot.md", config)
    if not failure_log.exists():
        failure_log.write_text("", encoding="utf-8")
    return payload


def _run_optimize(
    config: CaseConfig,
    out_dir: Path,
    *,
    optimizer: str,
    velikhov_constraint_mode: str,
    multistart: int,
    seed: int,
    max_iterations: int,
    initial_profile: dict[str, Any] | None = None,
    initial_profile_context: dict[str, Any] | None = None,
    slsqp_state_restarts: int = 1,
    freidberg_branch_audit: bool = False,
    freidberg_branch_policy: str = "continuity",
    freidberg_branch_tolerance: float = 1e-3,
) -> dict[str, Any]:
    failure_log = out_dir / "failure_log.jsonl"
    try:
        if str(optimizer) == "constrained_slsqp":
            if str(velikhov_constraint_mode) != "hard":
                raise ValueError("constrained_slsqp requires velikhov_constraint_mode='hard'.")
            if int(slsqp_state_restarts) > 1:
                opt = minimize_constrained_slsqp_state_restarts(
                    config=config,
                    multistart=int(multistart),
                    seed=int(seed),
                    max_iterations=int(max_iterations),
                    velikhov_hard_floor=_velikhov_constraint_floor(config),
                    initial_profile=initial_profile,
                    state_restarts=int(slsqp_state_restarts),
                )
            else:
                opt = minimize_constrained_slsqp(
                    config=config,
                    multistart=int(multistart),
                    seed=int(seed),
                    max_iterations=int(max_iterations),
                    velikhov_hard_floor=_velikhov_constraint_floor(config),
                    initial_profile=initial_profile,
                )
        else:
            opt = minimize_multistart(
                config=config,
                multistart=int(multistart),
                seed=int(seed),
                max_iterations=int(max_iterations),
                initial_profile=initial_profile,
            )
    except Exception as exc:
        _append_failure(failure_log, design=config.design, error=f"{type(exc).__name__}: {exc}", context={"mode": "optimize"})
        payload = {
            "ok": False,
            "case_config": config.to_dict(),
            "error": f"{type(exc).__name__}: {exc}",
            "failure_log": str(failure_log),
        }
        _write_json(out_dir / "run_summary.json", payload)
        return payload

    history_path = out_dir / "objective_history.csv"
    with history_path.open("w", encoding="utf-8", newline="") as handle:
        extra_fields = sorted(
            {
                key
                for row in opt["history"]
                for key in row.keys()
                if key not in {"start_index", "success", "message", "fun", "x"}
            }
        )
        writer = csv.DictWriter(handle, fieldnames=["start_index", "success", "message", "fun", "x", *extra_fields])
        writer.writeheader()
        for row in opt["history"]:
            writer.writerow({**row, "x": json.dumps(row["x"])})
    for failure in opt.get("trial_failures", []):
        _append_failure(
            failure_log,
            design=DesignVector.from_array(failure["x"]),
            error=failure["error"],
            context={
                "mode": "optimize",
                "start_index": failure["start_index"],
                "iteration": failure["iteration"],
                "trial": failure["trial"],
            },
        )
    if opt["best"] is None:
        payload = {
            "ok": False,
            "case_config": config.to_dict(),
            "error": "no finite optimizer start was available",
            "failure_log": str(failure_log),
            "optimizer": {
                "optimizer": str(optimizer),
                "velikhov_constraint_mode": str(velikhov_constraint_mode),
                "multistart": int(multistart),
                "seed": int(seed),
                "history_csv": str(history_path),
                "method": opt.get("method"),
                "trial_failures": opt.get("trial_failures", []),
            },
        }
        _write_json(out_dir / "run_summary.json", payload)
        return payload
    best_design = DesignVector.from_array(opt["best"]["x"])
    best_config = CaseConfig(
        case=config.case,
        objective_profile=config.objective_profile,
        length_m=config.length_m,
        area_scale_m2=config.area_scale_m2,
        B_T=float(best_design.B_T),
        working_fluid_profile=config.working_fluid_profile,
        n_intervals=config.n_intervals,
        design=best_design,
        bounds=config.bounds,
        metadata=config.metadata,
    )
    _write_json(out_dir / "best_design.json", best_design.to_dict())
    best_initial_profile = opt.get("best_profile", initial_profile)
    best_initial_profile_context = dict(initial_profile_context or {})
    if opt.get("best_profile") is not None:
        best_initial_profile_context = {
            **best_initial_profile_context,
            "state_restart_profile": "optimizer_refreshed_best_profile",
        }
    evaluate_payload = _run_evaluate(
        best_config,
        out_dir,
        initial_profile=best_initial_profile,
        initial_profile_context=best_initial_profile_context,
        freidberg_branch_audit=freidberg_branch_audit,
        freidberg_branch_policy=freidberg_branch_policy,
        freidberg_branch_tolerance=float(freidberg_branch_tolerance),
    )
    payload = {
        **evaluate_payload,
        "optimizer": {
            "optimizer": str(optimizer),
            "velikhov_constraint_mode": str(velikhov_constraint_mode),
            "multistart": int(multistart),
            "seed": int(seed),
            "history_csv": str(history_path),
            "method": opt.get("method"),
            "max_iterations": opt.get("max_iterations"),
            "constraint_sampling": opt.get("constraint_sampling"),
            "velikhov_hard_floor": opt.get("velikhov_hard_floor"),
            "slsqp_state_restarts": int(slsqp_state_restarts),
            "state_restarts_completed": opt.get("state_restarts_completed"),
            "state_restart_summaries": opt.get("state_restart_summaries"),
            "trial_failure_count": len(opt.get("trial_failures", [])),
            "best_raw": opt["best"],
        },
    }
    _write_json(out_dir / "run_summary.json", payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Firedrake/pyadjoint reduced-functional MHD prototype.")
    parser.add_argument("--case", default="yamasaki2004", choices=CASE_NAMES)
    parser.add_argument("--mode", default="evaluate", choices=("evaluate", "optimize"))
    parser.add_argument("--objective", default="enthalpy_extraction")
    parser.add_argument("--n-area-controls", type=int, default=3, help="v0 only supports 3 direct logA controls.")
    parser.add_argument("--n-intervals", type=int, default=None)
    parser.add_argument(
        "--geometry-length-mode",
        default="radial",
        choices=GEOMETRY_LENGTH_MODES,
        help="Effective streamwise length convention for the Yamasaki disk geometry.",
    )
    parser.add_argument("--multistart", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max-iterations", type=int, default=8)
    parser.add_argument(
        "--slsqp-state-restarts",
        type=int,
        default=1,
        help="For constrained_slsqp, rebuild the reduced functional this many times using the last converged profile as the next state initial guess.",
    )
    parser.add_argument(
        "--optimizer",
        default="projected_gradient",
        choices=("projected_gradient", "constrained_slsqp"),
        help="Outer optimizer for --mode optimize.",
    )
    parser.add_argument(
        "--velikhov-constraint-mode",
        default="none",
        choices=("none", "hard"),
        help="Treat node G as diagnostics only, or enforce G_node - hard_floor >= 0 in constrained_slsqp.",
    )
    parser.add_argument(
        "--velikhov-hard-floor",
        type=float,
        default=0.0,
        help="Hard node constraint floor used by --velikhov-constraint-mode hard.",
    )
    parser.add_argument("--design-json", type=Path, default=None, help="Optional design JSON to use as the initial/evaluate design.")
    parser.add_argument(
        "--allow-out-of-bounds-design-json",
        action="store_true",
        help="Replay --design-json exactly when it is outside the active case bounds.",
    )
    parser.add_argument(
        "--residual-scaling",
        default="inlet",
        choices=("inlet", "characteristic", "dimensional"),
        help="Weak residual row scaling. 'inlet' uses one cheap inlet-row scale per equation.",
    )
    parser.add_argument(
        "--equation-form",
        default="primitive",
        choices=EQUATION_FORMS,
        help="Forward weak residual form: legacy primitive momentum/energy, or Freidberg H/L balance equations.",
    )
    parser.add_argument(
        "--freidberg-branch-audit",
        action="store_true",
        help="Write a diagnostic H/L/T_e branch reconstruction audit for the final or failed profile.",
    )
    parser.add_argument(
        "--freidberg-branch-policy",
        default="continuity",
        choices=("continuity", "subsonic", "supersonic", "any"),
        help="Branch-selection policy used only by --freidberg-branch-audit.",
    )
    parser.add_argument(
        "--freidberg-branch-tolerance",
        type=float,
        default=1e-3,
        help="Absolute T_p closure residual tolerance in K for --freidberg-branch-audit.",
    )
    parser.add_argument(
        "--initial-profile-npz",
        type=Path,
        default=None,
        help="Optional profile.npz used to initialize delta_log(n_p), delta_log(T_e).",
    )
    parser.add_argument(
        "--reference-initial",
        default="none",
        choices=("none", "explicit", "implicit", "freidberg_hl"),
        help="Generate an initial profile before evaluate using the selected pure-Python reference marcher.",
    )
    parser.add_argument("--reference-residual-tol", type=float, default=1e-7)
    parser.add_argument("--reference-substeps-per-interval", type=int, default=10)
    parser.add_argument("--reference-max-log-step", type=float, default=0.25)
    parser.add_argument("--snes-max-it", type=int, default=None)
    parser.add_argument("--snes-dtol", type=float, default=None)
    parser.add_argument("--snes-linesearch-type", default=None)
    parser.add_argument(
        "--electron-transport",
        default=None,
        help=f"Electron-heavy collision model, one of {ELECTRON_TRANSPORT_MODELS}; defaults to the active case metadata.",
    )
    parser.add_argument(
        "--velikhov-mode",
        default="diagnostic",
        choices=("diagnostic", "penalty"),
        help="Use Velikhov margin only as a diagnostic, or subtract a soft path penalty from the objective.",
    )
    parser.add_argument("--velikhov-floor", type=float, default=5e-7)
    parser.add_argument("--velikhov-penalty-scale", type=float, default=1e-2)
    parser.add_argument("--velikhov-penalty-weight", type=float, default=25.0)
    parser.add_argument(
        "--thermal-window-mode",
        default="diagnostic",
        choices=("diagnostic", "penalty"),
        help="Use reconstructed T_p / T_e/T_p only as diagnostics, or subtract a soft thermal-window penalty.",
    )
    parser.add_argument(
        "--thermal-tp-in-max",
        type=float,
        default=2000.0,
        help="Soft inlet T_p ceiling [K] used by --thermal-window-mode penalty.",
    )
    parser.add_argument(
        "--thermal-tp-floor",
        type=float,
        default=300.0,
        help="Soft path floor for reconstructed T_p nodes [K].",
    )
    parser.add_argument(
        "--thermal-tp-path-max",
        type=float,
        default=None,
        help="Optional soft path ceiling for reconstructed T_p nodes [K].",
    )
    parser.add_argument(
        "--thermal-te-over-tp-min",
        type=float,
        default=None,
        help="Optional soft lower band edge for reconstructed T_e/T_p.",
    )
    parser.add_argument(
        "--thermal-te-over-tp-max",
        type=float,
        default=None,
        help="Optional soft upper band edge for reconstructed T_e/T_p.",
    )
    parser.add_argument("--thermal-tp-penalty-scale", type=float, default=100.0)
    parser.add_argument("--thermal-te-over-tp-penalty-scale", type=float, default=1.0)
    parser.add_argument("--thermal-tp-in-penalty-weight", type=float, default=1.0)
    parser.add_argument("--thermal-tp-path-penalty-weight", type=float, default=1.0)
    parser.add_argument("--thermal-te-over-tp-penalty-weight", type=float, default=1.0)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if int(args.n_area_controls) != 3:
        raise ValueError("v0 only supports --n-area-controls 3.")
    _validate_optimizer_options(args)
    config = load_case_config(
        case=args.case,
        objective_profile=args.objective,
        n_intervals=args.n_intervals,
        geometry_length_mode=str(args.geometry_length_mode),
    )
    electron_transport = normalize_electron_transport(
        args.electron_transport
        if args.electron_transport is not None
        else config.metadata.get("electron_transport", "e-He")
    )
    metadata = {
        **config.metadata,
        "electron_transport": electron_transport,
        "residual_scaling": str(args.residual_scaling),
        "equation_form": str(args.equation_form),
        "freidberg_branch_audit_requested": bool(args.freidberg_branch_audit),
        "freidberg_branch_policy": str(args.freidberg_branch_policy),
        "freidberg_branch_tolerance_K": float(args.freidberg_branch_tolerance),
        "velikhov_mode": str(args.velikhov_mode),
        "velikhov_floor": float(args.velikhov_floor),
        "velikhov_penalty_scale": float(args.velikhov_penalty_scale),
        "velikhov_penalty_weight": float(args.velikhov_penalty_weight),
        "thermal_window_mode": str(args.thermal_window_mode),
        "thermal_tp_in_max_K": None if args.thermal_tp_in_max is None else float(args.thermal_tp_in_max),
        "thermal_tp_floor_K": None if args.thermal_tp_floor is None else float(args.thermal_tp_floor),
        "thermal_tp_path_max_K": None if args.thermal_tp_path_max is None else float(args.thermal_tp_path_max),
        "thermal_te_over_tp_min": (
            None if args.thermal_te_over_tp_min is None else float(args.thermal_te_over_tp_min)
        ),
        "thermal_te_over_tp_max": (
            None if args.thermal_te_over_tp_max is None else float(args.thermal_te_over_tp_max)
        ),
        "thermal_tp_penalty_scale_K": float(args.thermal_tp_penalty_scale),
        "thermal_te_over_tp_penalty_scale": float(args.thermal_te_over_tp_penalty_scale),
        "thermal_tp_in_penalty_weight": float(args.thermal_tp_in_penalty_weight),
        "thermal_tp_path_penalty_weight": float(args.thermal_tp_path_penalty_weight),
        "thermal_te_over_tp_penalty_weight": float(args.thermal_te_over_tp_penalty_weight),
        "optimizer": str(args.optimizer),
        "velikhov_constraint_mode": str(args.velikhov_constraint_mode),
        "velikhov_hard_floor": float(args.velikhov_hard_floor),
        "velikhov_constraint_sampling": "nodes",
    }
    if args.snes_max_it is not None:
        metadata = {**metadata, "snes_max_it": int(args.snes_max_it)}
    if args.snes_dtol is not None:
        metadata = {**metadata, "snes_dtol": float(args.snes_dtol)}
    if args.snes_linesearch_type is not None:
        metadata = {**metadata, "snes_linesearch_type": str(args.snes_linesearch_type)}
    config = _with_design(config, config.design, metadata=metadata)
    if args.design_json is not None:
        design = _design_from_json(
            Path(args.design_json),
            config=config,
            allow_out_of_bounds=bool(args.allow_out_of_bounds_design_json),
        )
        metadata = {
            **metadata,
            "design_json": str(Path(args.design_json)),
            "design_json_bounds_policy": "allow" if bool(args.allow_out_of_bounds_design_json) else "error",
        }
        config = _with_design(config, design, metadata=metadata)
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    initial_profile = None
    initial_profile_context: dict[str, Any] = {}
    if args.initial_profile_npz is not None and str(args.reference_initial) != "none":
        raise ValueError("--initial-profile-npz and --reference-initial are mutually exclusive.")
    if args.initial_profile_npz is not None:
        initial_profile = _profile_from_npz(Path(args.initial_profile_npz))
        initial_profile_context = {
            "source": "npz",
            "path": str(Path(args.initial_profile_npz)),
        }
    elif str(args.reference_initial) != "none":
        reference_mode = str(args.reference_initial)
        if reference_mode == "explicit":
            reference = build_reference_profile(design=config.design, config=config)
        elif reference_mode == "implicit":
            reference = build_implicit_reference_profile(
                design=config.design,
                config=config,
                residual_tol=float(args.reference_residual_tol),
                initial_substeps_per_interval=int(args.reference_substeps_per_interval),
                max_log_step=float(args.reference_max_log_step),
            )
        else:
            reference = build_freidberg_reference_profile(
                design=config.design,
                config=config,
                residual_tol=float(args.reference_residual_tol),
                initial_substeps_per_interval=int(args.reference_substeps_per_interval),
                max_log_step=float(args.reference_max_log_step),
            )
        _write_json(
            out_dir / "reference_initial_summary.json",
            {
                "ok": bool(reference.ok),
                "mode": reference_mode,
                "diagnostics": reference.diagnostics,
                "error": reference.error,
            },
        )
        initial_profile_context = {
            "source": "reference_initial",
            "mode": reference_mode,
            "summary_json": str(out_dir / "reference_initial_summary.json"),
        }
        if not reference.ok or reference.profile is None:
            payload = {
                "ok": False,
                "case_config": config.to_dict(),
                "error": reference.error or "reference initial generation failed",
                "reference_initial": {
                    "mode": reference_mode,
                    "diagnostics": reference.diagnostics,
                    "summary_json": str(out_dir / "reference_initial_summary.json"),
                },
            }
            _write_json(out_dir / "run_summary.json", payload)
            print(json.dumps(payload, indent=2, sort_keys=True, default=_json_default))
            return 2
        initial_profile = reference.profile
        np.savez(out_dir / "reference_initial_profile.npz", **reference.profile)
        initial_profile_context["profile_npz"] = str(out_dir / "reference_initial_profile.npz")
    try:
        if args.mode == "evaluate":
            payload = _run_evaluate(
                config,
                out_dir,
                initial_profile=initial_profile,
                initial_profile_context=initial_profile_context,
                freidberg_branch_audit=bool(args.freidberg_branch_audit),
                freidberg_branch_policy=str(args.freidberg_branch_policy),
                freidberg_branch_tolerance=float(args.freidberg_branch_tolerance),
            )
        else:
            payload = _run_optimize(
                config,
                out_dir,
                optimizer=str(args.optimizer),
                velikhov_constraint_mode=str(args.velikhov_constraint_mode),
                multistart=int(args.multistart),
                seed=int(args.seed),
                max_iterations=int(args.max_iterations),
                initial_profile=initial_profile,
                initial_profile_context=initial_profile_context,
                slsqp_state_restarts=int(args.slsqp_state_restarts),
                freidberg_branch_audit=bool(args.freidberg_branch_audit),
                freidberg_branch_policy=str(args.freidberg_branch_policy),
                freidberg_branch_tolerance=float(args.freidberg_branch_tolerance),
            )
    except FiredrakeUnavailableError as exc:
        failure_log = out_dir / "failure_log.jsonl"
        _append_failure(failure_log, design=config.design, error=str(exc), context={"mode": args.mode})
        payload = {
            "ok": False,
            "case_config": config.to_dict(),
            "error": str(exc),
            "failure_log": str(failure_log),
        }
        _write_json(out_dir / "run_summary.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True, default=_json_default))
    return 0 if bool(payload.get("ok", False)) else 2


if __name__ == "__main__":
    raise SystemExit(main())
