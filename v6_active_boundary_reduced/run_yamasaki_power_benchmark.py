from __future__ import annotations

import argparse
import csv
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from v6_firedrake_reduced.cases.yamasaki2004 import YAMASAKI2004, YAMASAKI2004_MODEL_SEED
from v6_firedrake_reduced.design import GEOMETRY_LENGTH_MODES, load_case_config
from v6_firedrake_reduced.transport import ELECTRON_TRANSPORT_MODELS, working_fluid_for_config

from .numba_physics import inlet_design_numba
from .objective import evaluate_profile_metrics
from .policy import AnchorState, PolicySettings, State, _closure_metrics, _evaluate_sigma, rollout_policy_from_anchor
from .reachability_common import json_default, save_profile_npz, write_csv, write_json


def _paper_sigma_profile(*, n_intervals: int) -> dict[str, np.ndarray | float]:
    profile = YAMASAKI2004.geometry.profile(n_intervals=max(int(n_intervals), 2))
    x = np.asarray(profile["x"], dtype=float)
    area = np.asarray(profile["A"], dtype=float)
    sigma = np.gradient(np.log(np.maximum(area / max(float(area[0]), 1e-300), 1e-300)), x, edge_order=1)
    return {**profile, "sigma_logA": sigma}


def _config_with_transport(*, n_intervals: int, geometry_length_mode: str, electron_transport: str):
    config = load_case_config(
        case="yamasaki2004",
        n_intervals=int(n_intervals),
        geometry_length_mode=str(geometry_length_mode),
    )
    metadata = dict(config.metadata)
    metadata["electron_transport"] = str(electron_transport)
    return replace(config, metadata=metadata)


def _anchor_from_paper_inlet(config, *, n_intervals: int) -> AnchorState:
    geometry = _paper_sigma_profile(n_intervals=n_intervals)
    return AnchorState(
        state=State(
            log_n=float(config.design.log_n_p_in),
            log_Te=float(np.log(float(config.design.T_e_in))),
            logA=0.0,
        ),
        sigma_logA=float(np.asarray(geometry["sigma_logA"], dtype=float)[0]),
        x=0.0,
        source="yamasaki2004_paper_inlet_design_seed",
        source_index=0,
    )


def _inlet_payload(config, anchor: AnchorState) -> dict[str, float]:
    fluid = working_fluid_for_config(config)
    inlet = inlet_design_numba(
        float(config.design.n_p_in),
        float(config.design.T_e_in),
        float(config.design.Z_in),
        float(config.design.I_0),
        float(config.design.seed_fraction),
        float(config.design.B_T),
        float(config.area_scale_m2),
        float(fluid.heavy_particle_mass_kg),
        float(fluid.seed_ionization_energy_J),
        float(fluid.sigma_ep),
    )
    closure = _closure_metrics(anchor.state, config=config)
    return {
        "inlet_design_T_p_K": float(inlet[3]),
        "inlet_design_mach": float(inlet[9]),
        "inlet_design_G": float(inlet[10]),
        "inlet_design_v_m_s": float(inlet[7]),
        "inlet_design_dot_N_s": float(inlet[6]),
        "anchor_closure_T_p_K": float(closure["T_p"]),
        "anchor_closure_mach": float(closure["mach"]),
        "anchor_closure_G": float(closure["G"]),
        "anchor_power_density_W_per_m": float(closure["power_density_W_per_m"]),
    }


def _profile_metrics(payload: dict[str, Any], config) -> dict[str, float]:
    arrays = dict(payload.get("profile_arrays", {}) or {})
    profile = {
        name: np.asarray(arrays[name], dtype=float).reshape(-1)
        for name in ("x", "n_p", "T_e", "A", "sigma_logA")
    }
    order = np.argsort(profile["x"])
    profile = {name: values[order] for name, values in profile.items()}
    metrics = evaluate_profile_metrics(profile=profile, design=config.design, config=config).to_dict()
    return {name: float(value) for name, value in metrics.items() if isinstance(value, (int, float, np.floating))}


def _area_ratio(nodes: list[dict[str, Any]]) -> float:
    areas = [float(node["A"]) for node in nodes if np.isfinite(float(node["A"])) and float(node["A"]) > 0.0]
    if not areas:
        return float("nan")
    return float(max(areas) / min(areas))


def _node_payload(k: int, x: float, state: State, *, config, sigma: float) -> dict[str, float | int]:
    return {
        "k": int(k),
        "x": float(x),
        "n_p": float(state.n_p),
        "T_e": float(state.T_e),
        "A": float(state.area(config)),
        "logA": float(state.logA),
        "sigma_logA": float(sigma),
        **_closure_metrics(state, config=config),
    }


def _payload_arrays(nodes: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    return {
        "x": np.asarray([float(node["x"]) for node in nodes], dtype=float),
        "n_p": np.asarray([float(node["n_p"]) for node in nodes], dtype=float),
        "T_e": np.asarray([float(node["T_e"]) for node in nodes], dtype=float),
        "A": np.asarray([float(node["A"]) for node in nodes], dtype=float),
        "sigma_logA": np.asarray([float(node["sigma_logA"]) for node in nodes], dtype=float),
    }


def _simple_active_summary(nodes: list[dict[str, Any]], segments: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "n_steps_completed": int(len(segments)),
        "Delta_start": float(nodes[0]["Delta"]),
        "Delta_end": float(nodes[-1]["Delta"]),
        "G_min_excluding_anchor": float(min(float(node["G"]) for node in nodes[1:])) if len(nodes) > 1 else float("nan"),
        "Tp_min_K": float(min(float(node["T_p"]) for node in nodes)),
        "Te_max_K": float(max(float(node["T_e"]) for node in nodes)),
        "mach_max": float(max(float(node["mach"]) for node in nodes)),
        "sigma_min": float(min(float(node["sigma_logA"]) for node in nodes)),
        "sigma_max": float(max(float(node["sigma_logA"]) for node in nodes)),
        "max_abs_scaled_residual": (
            float(max(float(seg.get("max_abs_scaled_residual", 0.0)) for seg in segments))
            if segments
            else 0.0
        ),
    }


def _result_row(
    *,
    config,
    objective: str,
    payload: dict[str, Any],
    out_dir: Path,
) -> dict[str, Any]:
    metrics = _profile_metrics(payload, config)
    active = dict(payload.get("active_summary", {}) or {})
    nodes = list(payload.get("nodes", []))
    paper = YAMASAKI2004
    return {
        "electron_transport": str(config.metadata["electron_transport"]),
        "objective": str(objective),
        "ok": bool(payload.get("ok", False)),
        "n_steps_completed": int(active.get("n_steps_completed", 0)),
        "mhd_output_power_MW": float(metrics.get("mhd_output_power_W", float("nan")) / 1.0e6),
        "raw_enthalpy_extraction_percent": float(metrics.get("raw_enthalpy_extraction_percent", float("nan"))),
        "reported_enthalpy_extraction_percent": float(paper.reported_enthalpy_extraction_percent),
        "enthalpy_extraction_minus_reported_pctpt": float(
            metrics.get("raw_enthalpy_extraction_percent", float("nan"))
            - float(paper.reported_enthalpy_extraction_percent)
        ),
        "reported_electric_power_MW": float(paper.reported_electric_power_MW),
        "mhd_output_power_minus_reported_MW": float(
            metrics.get("mhd_output_power_W", float("nan")) / 1.0e6
            - float(paper.reported_electric_power_MW)
        ),
        "hall_voltage_V": float(metrics.get("hall_voltage_V", float("nan"))),
        "inlet_enthalpy_flux_MW": float(metrics.get("inlet_enthalpy_flux_W", float("nan")) / 1.0e6),
        "area_ratio": _area_ratio(nodes),
        "paper_area_ratio": float(paper.geometry.area_ratio),
        "area_ratio_minus_paper": float(_area_ratio(nodes) - float(paper.geometry.area_ratio)),
        "T_p_min_K": float(active.get("Tp_min_K", float("nan"))),
        "T_e_max_K": float(active.get("Te_max_K", float("nan"))),
        "mach_max": float(active.get("mach_max", float("nan"))),
        "G_min_excluding_anchor": float(active.get("G_min_excluding_anchor", float("nan"))),
        "Delta_start": float(active.get("Delta_start", float("nan"))),
        "Delta_end": float(active.get("Delta_end", float("nan"))),
        "sigma_min": float(active.get("sigma_min", float("nan"))),
        "sigma_max": float(active.get("sigma_max", float("nan"))),
        "max_abs_scaled_residual": float(active.get("max_abs_scaled_residual", float("nan"))),
        "out_dir": str(out_dir),
    }


def _run_one(
    *,
    config,
    anchor: AnchorState,
    objective: str,
    args: argparse.Namespace,
    out_dir: Path,
) -> dict[str, Any]:
    settings = PolicySettings(
        direction="forward",
        objective=str(objective),
        n_steps=int(args.n_steps),
        sigma_min=float(args.sigma_min),
        sigma_max=float(args.sigma_max),
        curvature_max=None if bool(args.no_curvature_bound) else float(args.curvature_max),
        g_floor=float(args.g_floor),
        tp_floor_K=float(args.tp_floor),
        scan_points=int(args.scan_points),
        refine_iterations=int(args.refine_iterations),
        active_tol=float(args.active_tol),
        sonic_mode=str(args.sonic_mode),
        sonic_mach_tol=float(args.sonic_mach_tol),
        sonic_det_abs_tol=float(args.sonic_det_abs_tol),
        sonic_compatibility_tol=float(args.sonic_compatibility_tol),
        sonic_residual_tol=float(args.sonic_residual_tol),
        rk4_substeps=int(args.rk4_substeps),
        rk4_error_tol=float(args.rk4_error_tol),
        rk4_rhs_mode=str(args.rk4_rhs_mode),
        rk4_stage_replay_tol=float(args.rk4_stage_replay_tol),
        rk4_stage_diagnostics=bool(args.rk4_stage_diagnostics),
        rk4_stage_gate=bool(args.rk4_stage_gate),
        rk4_stage_cond_max=float(args.rk4_stage_cond_max),
        rk4_stage_mach_max=float(args.rk4_stage_mach_max),
        rk4_stage_tp_floor_K=args.rk4_stage_tp_floor,
        rk4_stage_g_floor=args.rk4_stage_g_floor,
    )
    dx = float(config.length_m) / float(settings.n_steps)
    payload = rollout_policy_from_anchor(config=config, anchor=anchor, settings=settings, dx=dx)
    case_dir = out_dir / str(config.metadata["electron_transport"]).replace("-", "_") / str(objective)
    case_dir.mkdir(parents=True, exist_ok=True)
    write_json(case_dir / "policy_rollout_summary.json", payload)
    write_csv(case_dir / "nodes.csv", list(payload["nodes"]))
    write_csv(case_dir / "segments.csv", list(payload["segments"]))
    save_profile_npz(case_dir / "profile.npz", payload["profile_arrays"])

    return _result_row(config=config, objective=str(objective), payload=payload, out_dir=case_dir)


def _run_fixed_paper_geometry(
    *,
    config,
    anchor: AnchorState,
    args: argparse.Namespace,
    out_dir: Path,
) -> dict[str, Any]:
    geometry = _paper_sigma_profile(n_intervals=int(args.n_steps))
    x = np.asarray(geometry["x"], dtype=float)
    area = np.asarray(geometry["A"], dtype=float)
    logA = np.log(np.maximum(area / max(float(area[0]), 1.0e-300), 1.0e-300))
    dx = float(config.length_m) / float(args.n_steps)
    settings = PolicySettings(
        direction="forward",
        objective="power_next",
        n_steps=int(args.n_steps),
        sigma_min=-float("inf"),
        sigma_max=float("inf"),
        curvature_max=None,
        g_floor=float(args.g_floor),
        tp_floor_K=float(args.tp_floor),
        scan_points=3,
        refine_iterations=int(args.refine_iterations),
        active_tol=float(args.active_tol),
        sonic_mode=str(args.sonic_mode),
        sonic_mach_tol=float(args.sonic_mach_tol),
        sonic_det_abs_tol=float(args.sonic_det_abs_tol),
        sonic_compatibility_tol=float(args.sonic_compatibility_tol),
        sonic_residual_tol=float(args.sonic_residual_tol),
        rk4_substeps=int(args.rk4_substeps),
        rk4_error_tol=float(args.rk4_error_tol),
        rk4_rhs_mode=str(args.rk4_rhs_mode),
        rk4_stage_replay_tol=float(args.rk4_stage_replay_tol),
        rk4_stage_diagnostics=bool(args.rk4_stage_diagnostics),
        rk4_stage_gate=bool(args.rk4_stage_gate),
        rk4_stage_cond_max=float(args.rk4_stage_cond_max),
        rk4_stage_mach_max=float(args.rk4_stage_mach_max),
        rk4_stage_tp_floor_K=args.rk4_stage_tp_floor,
        rk4_stage_g_floor=args.rk4_stage_g_floor,
    )
    states = [anchor.state]
    nodes = [_node_payload(0, float(x[0]), states[0], config=config, sigma=float((logA[1] - logA[0]) / dx))]
    segments: list[dict[str, Any]] = []
    for k in range(int(args.n_steps)):
        sigma = float((float(logA[k + 1]) - float(logA[k])) / dx)
        chosen = _evaluate_sigma(
            current=states[-1],
            sigma=sigma,
            dx=dx,
            direction=1,
            config=config,
            settings=settings,
        )
        states.append(chosen["next_state"])
        node = _node_payload(k + 1, float(x[k + 1]), states[-1], config=config, sigma=sigma)
        nodes.append(node)
        segment = {key: value for key, value in chosen.items() if key != "next_state"}
        segment.update({"k": int(k), "support_type": "fixed_paper_geometry"})
        segments.append(segment)
        if not bool(chosen.get("feasible", False)):
            break

    payload = {
        "ok": bool(all(bool(seg.get("feasible", False)) for seg in segments)),
        "mode": "fixed_paper_geometry_forward_replay",
        "settings": settings.__dict__,
        "case_config": config.to_dict(),
        "nodes": nodes,
        "segments": segments,
        "active_summary": _simple_active_summary(nodes, segments),
        "profile_arrays": _payload_arrays(nodes),
    }
    case_dir = out_dir / str(config.metadata["electron_transport"]).replace("-", "_") / "fixed_paper_geometry"
    case_dir.mkdir(parents=True, exist_ok=True)
    write_json(case_dir / "policy_rollout_summary.json", payload)
    write_csv(case_dir / "nodes.csv", list(payload["nodes"]))
    write_csv(case_dir / "segments.csv", list(payload["segments"]))
    save_profile_npz(case_dir / "profile.npz", payload["profile_arrays"])
    return _result_row(config=config, objective="fixed_paper_geometry", payload=payload, out_dir=case_dir)


def _write_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a paper-aligned Yamasaki active-boundary power benchmark."
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--n-intervals", type=int, default=40)
    parser.add_argument("--n-steps", type=int, default=20)
    parser.add_argument("--geometry-length-mode", choices=GEOMETRY_LENGTH_MODES, default="radial")
    parser.add_argument(
        "--electron-transport",
        choices=(*ELECTRON_TRANSPORT_MODELS, "both"),
        default="both",
        help="Run current e-He parameters, legacy e-Argon replay parameters, or both.",
    )
    parser.add_argument(
        "--objective",
        action="append",
        choices=("delta_gain", "power_next"),
        help="Objective(s) to run. Defaults to both delta_gain and power_next.",
    )
    parser.add_argument("--no-fixed-paper-geometry", action="store_true")
    parser.add_argument("--allow-invalid-anchor", action="store_true")
    parser.add_argument("--sigma-min", type=float, default=-18.0)
    parser.add_argument("--sigma-max", type=float, default=18.0)
    parser.add_argument("--curvature-max", type=float, default=8.0)
    parser.add_argument("--no-curvature-bound", action="store_true")
    parser.add_argument("--g-floor", type=float, default=0.0)
    parser.add_argument("--tp-floor", type=float, default=1800.0)
    parser.add_argument("--scan-points", type=int, default=61)
    parser.add_argument("--refine-iterations", type=int, default=24)
    parser.add_argument("--active-tol", type=float, default=1.0e-6)
    parser.add_argument("--sonic-mode", choices=("auto", "off", "on"), default="auto")
    parser.add_argument("--sonic-mach-tol", type=float, default=1.0e-3)
    parser.add_argument("--sonic-det-abs-tol", type=float, default=1.0e-2)
    parser.add_argument("--sonic-compatibility-tol", type=float, default=1.0e-7)
    parser.add_argument("--sonic-residual-tol", type=float, default=1.0e-6)
    parser.add_argument("--rk4-substeps", type=int, default=4)
    parser.add_argument("--rk4-error-tol", type=float, default=1.0e-6)
    parser.add_argument("--rk4-rhs-mode", choices=("raw", "log", "nondim"), default="raw")
    parser.add_argument("--rk4-stage-replay-tol", type=float, default=float("inf"))
    parser.add_argument("--rk4-stage-diagnostics", action="store_true")
    parser.add_argument("--rk4-stage-gate", action="store_true")
    parser.add_argument("--rk4-stage-cond-max", type=float, default=float("inf"))
    parser.add_argument("--rk4-stage-mach-max", type=float, default=float("inf"))
    parser.add_argument("--rk4-stage-tp-floor", type=float, default=None)
    parser.add_argument("--rk4-stage-g-floor", type=float, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    transports = (
        list(ELECTRON_TRANSPORT_MODELS)
        if str(args.electron_transport) == "both"
        else [str(args.electron_transport)]
    )
    objectives = [str(item) for item in (args.objective or ["delta_gain", "power_next"])]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    paper_geometry = _paper_sigma_profile(n_intervals=int(args.n_intervals))
    paper = YAMASAKI2004
    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    configs = []
    for transport in transports:
        config = _config_with_transport(
            n_intervals=int(args.n_intervals),
            geometry_length_mode=str(args.geometry_length_mode),
            electron_transport=str(transport),
        )
        anchor = _anchor_from_paper_inlet(config, n_intervals=int(args.n_intervals))
        inlet = _inlet_payload(config, anchor)
        anchor_ok = bool(
            inlet["anchor_closure_T_p_K"] >= float(args.tp_floor) - float(args.active_tol)
            and inlet["anchor_closure_G"] >= float(args.g_floor) - float(args.active_tol)
        )
        configs.append(
            {
                "electron_transport": str(transport),
                "case_config": config.to_dict(),
                "paper_aligned_design": config.design.to_dict(),
                "inlet": inlet,
                "anchor_ok": anchor_ok,
            }
        )
        if not anchor_ok and not bool(args.allow_invalid_anchor):
            skipped.append(
                {
                    "electron_transport": str(transport),
                    "reason": "anchor violates G/Tp guard",
                    **inlet,
                }
            )
            continue
        if not bool(args.no_fixed_paper_geometry):
            rows.append(_run_fixed_paper_geometry(config=config, anchor=anchor, args=args, out_dir=out_dir))
        for objective in objectives:
            rows.append(_run_one(config=config, anchor=anchor, objective=objective, args=args, out_dir=out_dir))

    summary = {
        "paper_reference": paper.to_reference_dict(),
        "paper_geometry": {
            "length_m": float(paper.geometry.length_m),
            "area_ratio": float(paper.geometry.area_ratio),
            "sigma_min": float(np.min(np.asarray(paper_geometry["sigma_logA"], dtype=float))),
            "sigma_max": float(np.max(np.asarray(paper_geometry["sigma_logA"], dtype=float))),
            "sigma_inlet": float(np.asarray(paper_geometry["sigma_logA"], dtype=float)[0]),
            "sigma_exit": float(np.asarray(paper_geometry["sigma_logA"], dtype=float)[-1]),
        },
        "model_seed_note": YAMASAKI2004_MODEL_SEED.__doc__,
        "configs": configs,
        "skipped": skipped,
        "results": rows,
    }
    write_json(out_dir / "yamasaki_power_benchmark_summary.json", summary)
    _write_rows_csv(out_dir / "yamasaki_power_benchmark_results.csv", rows)
    print(json.dumps(summary, indent=2, sort_keys=True, default=json_default))
    return 0 if rows else 2


if __name__ == "__main__":
    raise SystemExit(main())
