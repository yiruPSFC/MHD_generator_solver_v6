from __future__ import annotations

import argparse
import json
import math
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from .design import DESIGN_VARIABLE_NAMES, CaseConfig, DesignVector
from .forward import ForwardResult, solve_forward
from .legacy_physics import dynamic_system_terms, inlet_design_generic, ops_for_numeric
from .transport import working_fluid_for_config


def _json_default(value: Any):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        out = float(value)
        return out if math.isfinite(out) else None
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    raise TypeError(f"object of type {type(value).__name__} is not JSON serializable")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")


def load_profile_npz(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(Path(path)) as data:
        return {name: np.asarray(data[name], dtype=float) for name in data.files}


def config_with_design(config: CaseConfig, design: DesignVector) -> CaseConfig:
    return replace(config, design=design, B_T=float(design.B_T))


def interpolate_design(start: DesignVector, target: DesignVector, alpha: float) -> DesignVector:
    a = float(alpha)
    values = (1.0 - a) * start.as_array() + a * target.as_array()
    return DesignVector.from_array(values)


def _closure_rows(
    *,
    profile: dict[str, Any],
    design: DesignVector,
    config: CaseConfig,
) -> list[dict[str, float | int]]:
    ops = ops_for_numeric()
    fluid = working_fluid_for_config(config)
    x = np.asarray(profile["x"], dtype=float).reshape(-1)
    if "x_norm" in profile:
        x_norm = np.asarray(profile["x_norm"], dtype=float).reshape(-1)
    else:
        span = float(x[-1] - x[0]) if x.size >= 2 else 1.0
        x_norm = (x - float(x[0])) / max(span, 1e-300)
    n_p = np.asarray(profile["n_p"], dtype=float).reshape(-1)
    T_e = np.asarray(profile["T_e"], dtype=float).reshape(-1)
    A = np.asarray(profile["A"], dtype=float).reshape(-1)
    sigma = np.asarray(profile["sigma_logA"], dtype=float).reshape(-1)
    inlet = inlet_design_generic(
        ops=ops,
        n_p_in=design.n_p_in,
        T_e_in=float(design.T_e_in),
        Z_in=float(design.Z_in),
        I_0=float(design.I_0),
        seed_fraction=design.seed_fraction,
        B=float(design.B_T),
        inlet_A=float(config.area_scale_m2),
        working_fluid=fluid,
    )
    rows: list[dict[str, float | int]] = []
    for idx, (x_val, xn_val, n_val, te_val, area_val, sigma_val) in enumerate(
        zip(x, x_norm, n_p, T_e, A, sigma, strict=True)
    ):
        closure, terms = dynamic_system_terms(
            ops=ops,
            n_p=float(n_val),
            T_e=float(te_val),
            A=float(area_val),
            sigma=float(sigma_val),
            dot_N=float(inlet["dot_N"]),
            I_0=float(design.I_0),
            seed_fraction=design.seed_fraction,
            B=float(design.B_T),
            working_fluid=fluid,
        )
        rows.append(
            {
                "i": int(idx),
                "x": float(x_val),
                "x_norm": float(xn_val),
                "n_p": float(n_val),
                "T_e": float(te_val),
                "A": float(area_val),
                "sigma_logA": float(sigma_val),
                "T_p": float(closure["T_p"]),
                "mach": float(closure["mach"]),
                "G": float(closure["G"]),
                "det": float(terms["det"]),
            }
        )
    return rows


def classify_forward_result(
    *,
    result: ForwardResult,
    design: DesignVector,
    config: CaseConfig,
    T_p_floor_K: float = 1.0,
    sonic_tol: float = 1e-2,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "ok": bool(result.ok),
        "error": result.error,
        "reason": "ok" if result.ok else "forward_nonlinear",
    }
    if result.metrics is not None:
        payload["metrics"] = {
            "raw_enthalpy_extraction_percent": float(result.metrics.raw_enthalpy_extraction_percent),
            "objective_score": float(result.metrics.objective_score),
            "min_T_p_K": float(result.metrics.min_T_p_K),
            "min_mach": float(result.metrics.min_mach),
            "outlet_mach": float(result.metrics.outlet_mach),
            "min_velikhov_margin": float(result.metrics.min_velikhov_margin),
            "max_Te_over_Tp": float(result.metrics.max_Te_over_Tp),
        }
    if result.profile is None:
        payload["reason"] = "no_profile" if not result.ok else payload["reason"]
        return payload

    try:
        rows = _closure_rows(profile=result.profile, design=design, config=config)
    except Exception as exc:
        payload["profile_classification_error"] = f"{type(exc).__name__}: {exc}"
        return payload

    state_bad = next(
        (
            row
            for row in rows
            if (
                not math.isfinite(float(row["n_p"]))
                or not math.isfinite(float(row["T_e"]))
                or float(row["n_p"]) <= 0.0
                or float(row["T_e"]) <= 0.0
            )
        ),
        None,
    )
    T_p_bad = next(
        (
            row
            for row in rows
            if (not math.isfinite(float(row["T_p"]))) or float(row["T_p"]) <= float(T_p_floor_K)
        ),
        None,
    )
    closest_mach = min(
        rows,
        key=lambda row: abs(float(row["mach"]) - 1.0) if math.isfinite(float(row["mach"])) else float("inf"),
    )
    min_abs_det = min(
        rows,
        key=lambda row: abs(float(row["det"])) if math.isfinite(float(row["det"])) else float("inf"),
    )
    payload.update(
        {
            "T_p_floor_K": float(T_p_floor_K),
            "first_state_domain_node": state_bad,
            "first_T_p_floor_node": T_p_bad,
            "closest_mach_node": closest_mach,
            "min_abs_det_node": min_abs_det,
        }
    )
    if state_bad is not None:
        payload["reason"] = "state_domain"
    elif T_p_bad is not None:
        payload["reason"] = "T_p_domain"
    elif abs(float(closest_mach["mach"]) - 1.0) <= float(sonic_tol):
        payload["reason"] = "sonic_near" if not result.ok else "ok_sonic_near"
    elif result.ok:
        payload["reason"] = "ok"
    return payload


def solve_forward_with_design_continuation(
    *,
    start_design: DesignVector,
    target_design: DesignVector,
    start_profile: dict[str, Any],
    config: CaseConfig,
    initial_step_fraction: float = 1.0,
    min_step_fraction: float = 1e-6,
    max_step_fraction: float = 1.0,
    growth: float = 1.5,
    max_attempts: int = 80,
    T_p_floor_K: float = 1.0,
    enforce_T_p_floor: bool = True,
) -> dict[str, Any]:
    current_alpha = 0.0
    current_design = start_design
    current_profile = start_profile
    step_fraction = min(max(float(initial_step_fraction), float(min_step_fraction)), float(max_step_fraction))
    rows: list[dict[str, Any]] = []
    last_trial_result: ForwardResult | None = None
    accepted_result: ForwardResult | None = None
    final_classification: dict[str, Any] | None = None

    for attempt in range(int(max_attempts)):
        if current_alpha >= 1.0 - 1e-14:
            break
        trial_alpha = min(1.0, current_alpha + step_fraction)
        trial_design = interpolate_design(start_design, target_design, trial_alpha)
        trial_config = config_with_design(config, trial_design)
        result = solve_forward(
            design=trial_design,
            config=trial_config,
            initial_profile=current_profile,
        )
        last_trial_result = result
        classification = classify_forward_result(
            result=result,
            design=trial_design,
            config=trial_config,
            T_p_floor_K=float(T_p_floor_K),
        )
        admissible = bool(result.ok and result.profile is not None and result.metrics is not None)
        if enforce_T_p_floor and result.metrics is not None:
            admissible = admissible and float(result.metrics.min_T_p_K) > float(T_p_floor_K)
        row = {
            "attempt": int(attempt),
            "alpha_start": float(current_alpha),
            "alpha_trial": float(trial_alpha),
            "step_fraction": float(step_fraction),
            "accepted": bool(admissible),
            "design": trial_design.to_dict(),
            "classification": classification,
        }
        rows.append(row)
        if admissible:
            current_alpha = trial_alpha
            current_design = trial_design
            current_profile = result.profile
            accepted_result = result
            final_classification = classification
            step_fraction = min(float(max_step_fraction), step_fraction * float(growth), 1.0 - current_alpha)
            if current_alpha >= 1.0 - 1e-14:
                break
            step_fraction = max(step_fraction, float(min_step_fraction))
            continue

        final_classification = classification
        step_fraction *= 0.5
        if step_fraction < float(min_step_fraction):
            break

    payload: dict[str, Any] = {
        "ok": bool(current_alpha >= 1.0 - 1e-14),
        "reached_alpha": float(current_alpha),
        "target_reached": bool(current_alpha >= 1.0 - 1e-14),
        "accepted_design": current_design.to_dict(),
        "target_design": target_design.to_dict(),
        "attempt_count": int(len(rows)),
        "rows": rows,
        "final_classification": final_classification,
    }
    if accepted_result is not None and accepted_result.metrics is not None:
        payload["accepted_metrics"] = accepted_result.metrics.to_dict()
        payload["final_metrics"] = accepted_result.metrics.to_dict()
    if last_trial_result is not None and last_trial_result.metrics is not None:
        payload["last_trial_metrics"] = last_trial_result.metrics.to_dict()
    if current_alpha < 1.0 - 1e-14:
        payload["error"] = "continuation stalled before target"
    payload["_accepted_profile"] = current_profile
    return payload


def solve_forward_with_persistent_solver_continuation(
    *,
    solver: Any,
    start_design: DesignVector,
    target_design: DesignVector,
    start_profile: dict[str, Any],
    config: CaseConfig,
    initial_step_fraction: float = 1.0,
    min_step_fraction: float = 1e-6,
    max_step_fraction: float = 1.0,
    growth: float = 1.5,
    max_attempts: int = 80,
    T_p_floor_K: float = 1.0,
    enforce_T_p_floor: bool = True,
) -> dict[str, Any]:
    current_alpha = 0.0
    current_design = start_design
    current_profile = start_profile
    step_fraction = min(max(float(initial_step_fraction), float(min_step_fraction)), float(max_step_fraction))
    rows: list[dict[str, Any]] = []
    last_trial_result: ForwardResult | None = None
    accepted_result: ForwardResult | None = None
    final_classification: dict[str, Any] | None = None

    for attempt in range(int(max_attempts)):
        if current_alpha >= 1.0 - 1e-14:
            break
        trial_alpha = min(1.0, current_alpha + step_fraction)
        trial_design = interpolate_design(start_design, target_design, trial_alpha)
        trial_config = config_with_design(config, trial_design)
        result = solver.solve(design=trial_design, initial_profile=current_profile)
        last_trial_result = result
        classification = classify_forward_result(
            result=result,
            design=trial_design,
            config=trial_config,
            T_p_floor_K=float(T_p_floor_K),
        )
        admissible = bool(result.ok and result.profile is not None and result.metrics is not None)
        if enforce_T_p_floor and result.metrics is not None:
            admissible = admissible and float(result.metrics.min_T_p_K) > float(T_p_floor_K)
        row = {
            "attempt": int(attempt),
            "alpha_start": float(current_alpha),
            "alpha_trial": float(trial_alpha),
            "step_fraction": float(step_fraction),
            "accepted": bool(admissible),
            "design": trial_design.to_dict(),
            "classification": classification,
            "solver_timing": dict(result.diagnostics.get("timing", {}) or {}),
        }
        rows.append(row)
        if admissible:
            current_alpha = trial_alpha
            current_design = trial_design
            current_profile = result.profile
            accepted_result = result
            final_classification = classification
            step_fraction = min(float(max_step_fraction), step_fraction * float(growth), 1.0 - current_alpha)
            if current_alpha >= 1.0 - 1e-14:
                break
            step_fraction = max(step_fraction, float(min_step_fraction))
            continue
        final_classification = classification
        step_fraction *= 0.5
        if step_fraction < float(min_step_fraction):
            break

    payload: dict[str, Any] = {
        "ok": bool(current_alpha >= 1.0 - 1e-14),
        "reached_alpha": float(current_alpha),
        "target_reached": bool(current_alpha >= 1.0 - 1e-14),
        "accepted_design": current_design.to_dict(),
        "target_design": target_design.to_dict(),
        "attempt_count": int(len(rows)),
        "rows": rows,
        "final_classification": final_classification,
    }
    if accepted_result is not None and accepted_result.metrics is not None:
        payload["accepted_metrics"] = accepted_result.metrics.to_dict()
        payload["final_metrics"] = accepted_result.metrics.to_dict()
    if last_trial_result is not None and last_trial_result.metrics is not None:
        payload["last_trial_metrics"] = last_trial_result.metrics.to_dict()
    if current_alpha < 1.0 - 1e-14:
        payload["error"] = "continuation stalled before target"
    payload["_accepted_profile"] = current_profile
    return payload


def solve_forward_with_persistent_design_continuation(
    *,
    start_design: DesignVector,
    target_design: DesignVector,
    start_profile: dict[str, Any],
    config: CaseConfig,
    initial_step_fraction: float = 1.0,
    min_step_fraction: float = 1e-6,
    max_step_fraction: float = 1.0,
    growth: float = 1.5,
    max_attempts: int = 80,
    T_p_floor_K: float = 1.0,
    enforce_T_p_floor: bool = True,
) -> dict[str, Any]:
    from .persistent_forward import PersistentForwardSolver

    solver = PersistentForwardSolver(config=config_with_design(config, start_design))
    return solve_forward_with_persistent_solver_continuation(
        solver=solver,
        start_design=start_design,
        target_design=target_design,
        start_profile=start_profile,
        config=config,
        initial_step_fraction=float(initial_step_fraction),
        min_step_fraction=float(min_step_fraction),
        max_step_fraction=float(max_step_fraction),
        growth=float(growth),
        max_attempts=int(max_attempts),
        T_p_floor_K=float(T_p_floor_K),
        enforce_T_p_floor=bool(enforce_T_p_floor),
    )


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _load_failure_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _select_failure_design(
    *,
    rows: list[dict[str, Any]],
    start_design: DesignVector,
    index: int | None,
    selection: str,
) -> DesignVector:
    if index is not None:
        return DesignVector.from_dict(rows[int(index)]["design"])
    if str(selection) == "first":
        return DesignVector.from_dict(rows[0]["design"])
    start = start_design.as_array()
    return DesignVector.from_dict(
        min(
            rows,
            key=lambda row: float(np.linalg.norm(DesignVector.from_dict(row["design"]).as_array() - start)),
        )["design"]
    )


def _resolve_path(base: Path, value: str | None, default_name: str) -> Path:
    if value:
        path = Path(value)
    else:
        path = base / default_name
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Replay a design-space forward solve with adaptive profile continuation.")
    parser.add_argument("--run-summary", type=Path, required=True)
    parser.add_argument("--start-profile-npz", type=Path, default=None)
    parser.add_argument("--start-design-json", type=Path, default=None)
    parser.add_argument("--target-design-json", type=Path, default=None)
    parser.add_argument("--failure-log-jsonl", type=Path, default=None)
    parser.add_argument("--failure-index", type=int, default=None)
    parser.add_argument("--failure-selection", default="nearest", choices=("nearest", "first"))
    parser.add_argument("--initial-step-fraction", type=float, default=1.0)
    parser.add_argument("--min-step-fraction", type=float, default=1e-6)
    parser.add_argument("--max-step-fraction", type=float, default=1.0)
    parser.add_argument("--max-attempts", type=int, default=80)
    parser.add_argument("--T-p-floor-K", type=float, default=1.0)
    parser.add_argument("--allow-low-T-p", action="store_true")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    from .analyze_kkt import _config_from_case_payload

    summary_path = Path(args.run_summary).resolve()
    summary = _load_json(summary_path)
    config = _config_from_case_payload(summary["case_config"])
    base_dir = summary_path.parent
    start_design = (
        DesignVector.from_dict(_load_json(args.start_design_json))
        if args.start_design_json is not None
        else DesignVector.from_dict(summary["case_config"]["design"])
    )
    start_profile_path = (
        Path(args.start_profile_npz).resolve()
        if args.start_profile_npz is not None
        else _resolve_path(base_dir, summary.get("profile_npz"), "profile.npz")
    )
    start_profile = load_profile_npz(start_profile_path)
    if args.target_design_json is not None:
        target_design = DesignVector.from_dict(_load_json(args.target_design_json))
        target_source = {"kind": "design_json", "path": str(Path(args.target_design_json).resolve())}
    else:
        failure_log_path = (
            Path(args.failure_log_jsonl).resolve()
            if args.failure_log_jsonl is not None
            else _resolve_path(base_dir, summary.get("failure_log"), "failure_log.jsonl")
        )
        failure_rows = _load_failure_rows(failure_log_path)
        target_design = _select_failure_design(
            rows=failure_rows,
            start_design=start_design,
            index=args.failure_index,
            selection=str(args.failure_selection),
        )
        target_source = {
            "kind": "failure_log",
            "path": str(failure_log_path),
            "selection": str(args.failure_selection),
            "index": args.failure_index,
        }

    payload = solve_forward_with_design_continuation(
        start_design=start_design,
        target_design=target_design,
        start_profile=start_profile,
        config=config_with_design(config, start_design),
        initial_step_fraction=float(args.initial_step_fraction),
        min_step_fraction=float(args.min_step_fraction),
        max_step_fraction=float(args.max_step_fraction),
        max_attempts=int(args.max_attempts),
        T_p_floor_K=float(args.T_p_floor_K),
        enforce_T_p_floor=not bool(args.allow_low_T_p),
    )
    accepted_profile = payload.pop("_accepted_profile", None)
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    if accepted_profile is not None:
        np.savez(out_dir / "accepted_profile.npz", **accepted_profile)
        payload["accepted_profile_npz"] = str(out_dir / "accepted_profile.npz")
    payload["start_profile_npz"] = str(start_profile_path)
    payload["start_design"] = start_design.to_dict()
    payload["target_source"] = target_source
    payload["case_config"] = config_with_design(config, start_design).to_dict()
    _write_json(out_dir / "continuation_summary.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True, default=_json_default))
    return 0 if payload.get("ok") else 2


if __name__ == "__main__":
    raise SystemExit(main())
