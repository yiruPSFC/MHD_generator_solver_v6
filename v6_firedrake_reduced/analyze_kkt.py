from __future__ import annotations

import argparse
import json
import math
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from .constraints import evaluate_velikhov_node_constraints
from .design import DESIGN_VARIABLE_NAMES, CaseConfig, DesignBounds, DesignVector
from .forward import solve_forward
from .reduced_functional import (
    _stop_annotating_context,
    build_reduced_functional,
    evaluate_reduced_functional,
    reduced_functional_gradient,
)


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
    if isinstance(value, np.bool_):
        return bool(value)
    raise TypeError(f"object of type {type(value).__name__} is not JSON serializable")


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _config_from_case_payload(payload: dict[str, Any]) -> CaseConfig:
    bounds_payload = dict(payload["bounds"])
    lower = DesignVector.from_dict({name: values["min"] for name, values in bounds_payload.items()})
    upper = DesignVector.from_dict({name: values["max"] for name, values in bounds_payload.items()})
    design = DesignVector.from_dict(dict(payload["design"]))
    return CaseConfig(
        case=str(payload["case"]),
        objective_profile=str(payload["objective_profile"]),
        length_m=float(payload["length_m"]),
        area_scale_m2=float(payload["area_scale_m2"]),
        B_T=float(design.B_T),
        working_fluid_profile=str(payload["working_fluid_profile"]),
        n_intervals=int(payload["n_intervals"]),
        design=design,
        bounds=DesignBounds(lower=lower, upper=upper),
        metadata=dict(payload.get("metadata", {}) or {}),
    )


def _resolve_profile_path(summary_path: Path, run_summary: dict[str, Any]) -> Path:
    candidate = run_summary.get("profile_npz")
    if not candidate:
        candidate = summary_path.with_name("profile.npz")
    path = Path(str(candidate))
    if not path.is_absolute():
        path = summary_path.parent / path
    if not path.exists():
        raise FileNotFoundError(f"profile npz not found: {path}")
    return path.resolve()


def _load_profile(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return {name: np.asarray(data[name], dtype=float) for name in data.files}


def _active_bounds(
    *,
    x: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    abs_tol: float,
    rel_tol: float,
) -> list[dict[str, Any]]:
    active = []
    for idx, name in enumerate(DESIGN_VARIABLE_NAMES):
        value = float(x[idx])
        lo = float(lower[idx])
        hi = float(upper[idx])
        width = max(abs(hi - lo), 1.0)
        tol = max(float(abs_tol), float(rel_tol) * width)
        lower_distance = value - lo
        upper_distance = hi - value
        if lower_distance <= tol:
            active.append(
                {
                    "type": "bound",
                    "variable": name,
                    "index": int(idx),
                    "side": "lower",
                    "value": value,
                    "bound": lo,
                    "distance": float(lower_distance),
                    "constraint_value": float(lower_distance),
                }
            )
        if upper_distance <= tol:
            active.append(
                {
                    "type": "bound",
                    "variable": name,
                    "index": int(idx),
                    "side": "upper",
                    "value": value,
                    "bound": hi,
                    "distance": float(upper_distance),
                    "constraint_value": float(upper_distance),
                }
            )
    return active


def _g_margins_for_design(
    *,
    values: np.ndarray,
    config: CaseConfig,
    floor: float,
    initial_profile: dict[str, np.ndarray] | None = None,
) -> np.ndarray:
    design = DesignVector.from_array(values)
    with _stop_annotating_context():
        result = solve_forward(
            design=design,
            config=replace(config, design=design, B_T=float(design.B_T)),
            initial_profile=initial_profile,
        )
    if not result.ok or result.profile is None:
        raise RuntimeError(result.error or "forward solve failed")
    summary = evaluate_velikhov_node_constraints(
        profile=result.profile,
        design=design,
        config=config,
        floor=float(floor),
    )
    return np.asarray(summary.margins, dtype=float)


def _finite_difference_active_g_jacobian(
    *,
    x0: np.ndarray,
    base_margins: np.ndarray,
    active_indices: list[int],
    config: CaseConfig,
    floor: float,
    rel_step: float,
    initial_profile: dict[str, np.ndarray] | None = None,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    lower = config.bounds.lower.as_array()
    upper = config.bounds.upper.as_array()
    jac = np.zeros((len(active_indices), len(DESIGN_VARIABLE_NAMES)), dtype=float)
    failures: list[dict[str, Any]] = []
    if not active_indices:
        return jac, failures
    active_base = np.asarray(base_margins, dtype=float)[active_indices]
    for var_idx, name in enumerate(DESIGN_VARIABLE_NAMES):
        step = float(rel_step) * max(abs(float(x0[var_idx])), 1.0)
        if step <= 0.0:
            step = float(rel_step)
        direction = None
        if float(x0[var_idx] + step) <= float(upper[var_idx]):
            trial = np.array(x0, dtype=float)
            trial[var_idx] += step
            direction = "forward"
        elif float(x0[var_idx] - step) >= float(lower[var_idx]):
            trial = np.array(x0, dtype=float)
            trial[var_idx] -= step
            direction = "backward"
        else:
            failures.append(
                {
                    "variable": name,
                    "error": "finite-difference step does not fit inside active bounds",
                }
            )
            continue
        try:
            trial_margins = _g_margins_for_design(
                values=trial,
                config=config,
                floor=floor,
                initial_profile=initial_profile,
            )[active_indices]
        except Exception as exc:
            failures.append(
                {
                    "variable": name,
                    "direction": direction,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            continue
        if direction == "forward":
            jac[:, var_idx] = (np.asarray(trial_margins, dtype=float) - active_base) / step
        else:
            jac[:, var_idx] = (active_base - np.asarray(trial_margins, dtype=float)) / step
    return jac, failures


def _recover_nonnegative_multipliers(
    *,
    gradient_minimize: np.ndarray,
    columns: list[np.ndarray],
) -> dict[str, Any]:
    grad = np.asarray(gradient_minimize, dtype=float).reshape(-1)
    if not columns:
        return {
            "multipliers": np.zeros(0, dtype=float),
            "support": np.zeros_like(grad),
            "stationarity": grad,
            "quality": {
                "status": "no_active_constraints",
                "stationarity_l2": float(np.linalg.norm(grad)),
                "stationarity_inf": float(np.max(np.abs(grad))) if grad.size else 0.0,
            },
        }
    try:
        from scipy.optimize import nnls  # type: ignore
    except ImportError as exc:
        raise RuntimeError("scipy.optimize.nnls is required for KKT multiplier recovery.") from exc

    matrix = np.column_stack(columns)
    column_norms = np.maximum(np.linalg.norm(matrix, axis=0), 1.0)
    scaled_matrix = matrix / column_norms.reshape(1, -1)
    solution_scaled, residual_norm = nnls(scaled_matrix, grad)
    multipliers = solution_scaled / column_norms
    support = matrix @ multipliers
    stationarity = grad - support
    return {
        "multipliers": multipliers,
        "support": support,
        "stationarity": stationarity,
        "quality": {
            "status": "nonnegative_least_squares_recovered_at_fixed_design",
            "stationarity_l2": float(np.linalg.norm(stationarity)),
            "stationarity_inf": float(np.max(np.abs(stationarity))) if stationarity.size else 0.0,
            "gradient_l2": float(np.linalg.norm(grad)),
            "gradient_inf": float(np.max(np.abs(grad))) if grad.size else 0.0,
            "nnls_residual_norm_column_scaled": float(residual_norm),
            "active_constraint_count": int(len(columns)),
        },
    }


def analyze_reduced_kkt(
    run_summary_path: str | Path,
    *,
    hard_floor: float | None = None,
    active_tol: float = 1e-6,
    bound_abs_tol: float = 1e-8,
    bound_rel_tol: float = 1e-6,
    fd_rel_step: float = 1e-5,
) -> dict[str, Any]:
    summary_path = Path(run_summary_path).resolve()
    run_summary = _load_json(summary_path)
    if not bool(run_summary.get("ok", False)):
        raise ValueError("KKT analysis requires a successful run_summary.json.")
    config = _config_from_case_payload(dict(run_summary["case_config"]))
    floor = float(config.metadata.get("velikhov_hard_floor", 0.0) if hard_floor is None else hard_floor)
    raw_metadata = {**config.metadata, "velikhov_mode": "diagnostic"}
    config = replace(config, metadata=raw_metadata)

    design_path = summary_path.with_name("best_design.json")
    design = DesignVector.from_dict(_load_json(design_path)) if design_path.exists() else config.design
    config = replace(config, design=design)
    x0 = design.as_array()
    lower = config.bounds.lower.as_array()
    upper = config.bounds.upper.as_array()
    profile_path = _resolve_profile_path(summary_path, run_summary)
    profile = _load_profile(profile_path)
    node_constraints = evaluate_velikhov_node_constraints(
        profile=profile,
        design=design,
        config=config,
        floor=floor,
        active_tolerance=float(active_tol),
    )
    base_margins = np.asarray(node_constraints.margins, dtype=float)
    active_g_indices = node_constraints.active_indices()

    bundle = build_reduced_functional(design=design, config=config, initial_profile=profile)
    objective_to_maximize = evaluate_reduced_functional(bundle, x0)
    grad_maximize = reduced_functional_gradient(bundle)
    grad_minimize = -np.asarray(grad_maximize, dtype=float)
    with _stop_annotating_context():
        direct_result = solve_forward(
            design=design,
            config=replace(config, design=design, B_T=float(design.B_T)),
            initial_profile=profile,
        )
    if not direct_result.ok or direct_result.metrics is None:
        direct_objective = float("nan")
        direct_error = direct_result.error or "forward solve failed"
    else:
        direct_objective = float(direct_result.metrics.objective_score)
        direct_error = None

    g_jac, fd_failures = _finite_difference_active_g_jacobian(
        x0=x0,
        base_margins=base_margins,
        active_indices=active_g_indices,
        config=config,
        floor=floor,
        rel_step=float(fd_rel_step),
        initial_profile=profile,
    )
    active_bounds = _active_bounds(
        x=x0,
        lower=lower,
        upper=upper,
        abs_tol=float(bound_abs_tol),
        rel_tol=float(bound_rel_tol),
    )

    columns: list[np.ndarray] = []
    labels: list[dict[str, Any]] = []
    path_column_count = 0
    for row_idx, node_idx in enumerate(active_g_indices):
        columns.append(np.asarray(g_jac[row_idx, :], dtype=float))
        labels.append(
            {
                "type": "path",
                "family": "velikhov_node_lower",
                "name": f"G_node_{node_idx}",
                "node_index": int(node_idx),
                "x": float(node_constraints.x[node_idx]),
                "G_node": float(node_constraints.G_node[node_idx]),
                "margin": float(node_constraints.margins[node_idx]),
                "sense": "G_node - floor >= 0",
            }
        )
        path_column_count += 1
    for bound in active_bounds:
        col = np.zeros_like(x0)
        if bound["side"] == "lower":
            col[int(bound["index"])] = 1.0
        else:
            col[int(bound["index"])] = -1.0
        columns.append(col)
        labels.append(
            {
                "type": "bound",
                "family": "design_box",
                "name": f"{bound['variable']}_{bound['side']}",
                **bound,
            }
        )

    recovered = _recover_nonnegative_multipliers(
        gradient_minimize=grad_minimize,
        columns=columns,
    )
    multipliers = np.asarray(recovered["multipliers"], dtype=float)
    path_support = (
        np.column_stack(columns[:path_column_count]) @ multipliers[:path_column_count]
        if path_column_count
        else np.zeros_like(x0)
    )
    bound_support = (
        np.column_stack(columns[path_column_count:]) @ multipliers[path_column_count:]
        if len(columns) > path_column_count
        else np.zeros_like(x0)
    )
    stationarity = np.asarray(recovered["stationarity"], dtype=float)

    active_entries = []
    for idx, label in enumerate(labels):
        entry = dict(label)
        entry["recovered_multiplier"] = float(multipliers[idx])
        active_entries.append(entry)

    controls = []
    for idx, name in enumerate(DESIGN_VARIABLE_NAMES):
        controls.append(
            {
                "name": name,
                "value": float(x0[idx]),
                "lower": float(lower[idx]),
                "upper": float(upper[idx]),
                "objective_gradient_minimize": float(grad_minimize[idx]),
                "path_constraint_support": float(path_support[idx]),
                "bound_constraint_support": float(bound_support[idx]),
                "total_constraint_support": float(path_support[idx] + bound_support[idx]),
                "stationarity_residual": float(stationarity[idx]),
                "box_fraction": float((x0[idx] - lower[idx]) / max(upper[idx] - lower[idx], 1e-300)),
            }
        )

    closest_nodes = sorted(
        [
            {
                "node_index": int(idx),
                "x": float(node_constraints.x[idx]),
                "G_node": float(node_constraints.G_node[idx]),
                "margin": float(node_constraints.margins[idx]),
            }
            for idx in range(base_margins.size)
        ],
        key=lambda item: float(item["margin"]),
    )[:25]

    return {
        "summary_path": str(summary_path),
        "analysis_mode": "reduced_space_node_constraint_kkt_diagnostic",
        "operator_semantics": (
            "Local reduced-space KKT diagnostic at a fixed design. "
            "Multipliers are recovered by nonnegative least squares; this is not a global certificate."
        ),
        "objective": {
            "objective_to_maximize": float(objective_to_maximize),
            "objective_to_minimize": float(-objective_to_maximize),
            "postprocess_objective_to_maximize": float(direct_objective),
            "taped_objective_minus_postprocess_objective": float(objective_to_maximize - direct_objective),
            "postprocess_forward_error": direct_error,
            "gradient_source": "pyadjoint reduced functional",
            "objective_semantics_note": (
                "objective_to_maximize is the taped Firedrake objective used for gradients; "
                "postprocess_objective_to_maximize is recomputed from the saved nodal profile."
            ),
            "velikhov_mode_for_gradient": "diagnostic",
        },
        "node_constraints": node_constraints.to_dict(),
        "active_path_constraints": [entry for entry in active_entries if entry["type"] == "path"],
        "active_bound_constraints": [entry for entry in active_entries if entry["type"] == "bound"],
        "closest_node_constraints": closest_nodes,
        "control_stationarity": controls,
        "recovered_multipliers": {
            "quality": recovered["quality"],
            "active_constraints": active_entries,
        },
        "finite_difference": {
            "rel_step": float(fd_rel_step),
            "active_g_jacobian_rows": int(g_jac.shape[0]),
            "active_g_jacobian_cols": int(g_jac.shape[1]) if g_jac.ndim == 2 else 0,
            "failures": fd_failures,
            "note": "Finite differences are used only for active/near-active G_node Jacobian rows.",
        },
        "inputs": {
            "profile_npz": str(profile_path),
            "best_design_json": str(design_path) if design_path.exists() else None,
            "hard_floor": floor,
            "active_tolerance": float(active_tol),
            "bound_abs_tolerance": float(bound_abs_tol),
            "bound_rel_tolerance": float(bound_rel_tol),
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze reduced-space KKT balance for a Firedrake reduced run.")
    parser.add_argument("summary", type=Path, help="Path to a successful run_summary.json.")
    parser.add_argument("--out", type=Path, default=None, help="Output JSON path; defaults to kkt_analysis.json next to summary.")
    parser.add_argument("--hard-floor", type=float, default=None, help="Override G_node hard floor. Defaults to run metadata or 0.")
    parser.add_argument("--active-tol", type=float, default=1e-6, help="Treat G_node margins <= this value as active.")
    parser.add_argument("--bound-abs-tol", type=float, default=1e-8, help="Absolute active-bound tolerance.")
    parser.add_argument("--bound-rel-tol", type=float, default=1e-6, help="Relative active-bound tolerance.")
    parser.add_argument("--fd-rel-step", type=float, default=1e-5, help="Relative finite-difference step for active G_node Jacobian rows.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = analyze_reduced_kkt(
        args.summary,
        hard_floor=args.hard_floor,
        active_tol=float(args.active_tol),
        bound_abs_tol=float(args.bound_abs_tol),
        bound_rel_tol=float(args.bound_rel_tol),
        fd_rel_step=float(args.fd_rel_step),
    )
    out = args.out or Path(args.summary).resolve().with_name("kkt_analysis.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
