from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from v6_firedrake_reduced.design import load_case_config
from v6_firedrake_reduced.geometry import LogAreaSplineControl

from ..core.local_affine import ForwardAffineCoefficients, compute_forward_affine_coefficients
from ..core.policy import (
    AnchorState,
    PolicySettings,
    State,
    _closure_metrics,
    _evaluate_sigma,
    _physics_params,
)
from .common import json_default, load_anchor_json, save_profile_npz, write_csv, write_json


def _finite_or_none(value: float) -> float | None:
    out = float(value)
    return out if np.isfinite(out) else None


def _max_with_source(current: float, source: str, candidate: float, candidate_source: str) -> tuple[float, str]:
    if float(candidate) > float(current):
        return float(candidate), str(candidate_source)
    return float(current), str(source)


def _min_with_source(current: float, source: str, candidate: float, candidate_source: str) -> tuple[float, str]:
    if float(candidate) < float(current):
        return float(candidate), str(candidate_source)
    return float(current), str(source)


def _build_forward_interval(
    *,
    current: State,
    A_current: float,
    sigma_prev: float | None,
    dx: float,
    settings: PolicySettings,
    logA_min: float,
    logA_max: float,
    q0_forward: float,
    q1_forward: float,
    q1_tol: float,
    g_margin_tol: float,
) -> dict[str, Any]:
    step = float(dx)
    lo = -float("inf")
    hi = float("inf")
    lower_source = "none"
    upper_source = "none"

    sigma_slope_lower = float(settings.sigma_min)
    sigma_slope_upper = float(settings.sigma_max)
    sigma_logA_lower = float((float(logA_min) - float(current.logA)) / step)
    sigma_logA_upper = float((float(logA_max) - float(current.logA)) / step)
    if (
        settings.curvature_max is None
        or not np.isfinite(float(settings.curvature_max))
        or sigma_prev is None
        or not np.isfinite(float(sigma_prev))
    ):
        sigma_curvature_lower = -float("inf")
        sigma_curvature_upper = float("inf")
    else:
        width = abs(float(settings.curvature_max))
        sigma_curvature_lower = float(sigma_prev) - width
        sigma_curvature_upper = float(sigma_prev) + width

    lo, lower_source = _max_with_source(lo, lower_source, sigma_slope_lower, "slope_min")
    hi, upper_source = _min_with_source(hi, upper_source, sigma_slope_upper, "slope_max")
    lo, lower_source = _max_with_source(lo, lower_source, sigma_logA_lower, "area_min")
    hi, upper_source = _min_with_source(hi, upper_source, sigma_logA_upper, "area_max")
    lo, lower_source = _max_with_source(lo, lower_source, sigma_curvature_lower, "curvature_min")
    hi, upper_source = _min_with_source(hi, upper_source, sigma_curvature_upper, "curvature_max")

    forward_G_bound_kind = "none"
    sigma_G_bound = float("nan")
    if float(q1_forward) > float(q1_tol):
        sigma_G_bound = float(-float(q0_forward) / (float(q1_forward) * float(A_current)))
        lo, lower_source = _max_with_source(lo, lower_source, sigma_G_bound, "G_lower")
        forward_G_bound_kind = "lower"
    elif float(q1_forward) < -float(q1_tol):
        sigma_G_bound = float(-float(q0_forward) / (float(q1_forward) * float(A_current)))
        hi, upper_source = _min_with_source(hi, upper_source, sigma_G_bound, "G_upper")
        forward_G_bound_kind = "upper"
    elif float(q0_forward) < -float(g_margin_tol):
        forward_G_bound_kind = "infeasible_flat"

    return {
        "ok": bool(lo <= hi and forward_G_bound_kind != "infeasible_flat"),
        "sigma_interval_lower": float(lo),
        "sigma_interval_upper": float(hi),
        "lower_source": lower_source,
        "upper_source": upper_source,
        "sigma_slope_lower": sigma_slope_lower,
        "sigma_slope_upper": sigma_slope_upper,
        "sigma_logA_lower": sigma_logA_lower,
        "sigma_logA_upper": sigma_logA_upper,
        "sigma_curvature_lower": sigma_curvature_lower,
        "sigma_curvature_upper": sigma_curvature_upper,
        "forward_G_bound_kind": forward_G_bound_kind,
        "sigma_G_bound": sigma_G_bound,
    }


def _support_from_source(source: str, *, p1_forward: float, q1_forward: float) -> str:
    if source in {"G_lower", "G_upper"}:
        product = float(p1_forward) * float(q1_forward)
        if product < 0.0:
            return "G_limited_forward"
        if product > 0.0:
            return "G_permissive_forward_selected_by_other_effect"
        return "G_flat_forward"
    if source in {"slope_min", "slope_max"}:
        return "geometry_limited"
    if source in {"area_min", "area_max"}:
        return "area_limited"
    if source in {"curvature_min", "curvature_max"}:
        return "curvature_limited"
    return "interior_or_regularized"


def _scan_best_feasible(
    *,
    current: State,
    lo: float,
    hi: float,
    dx: float,
    config,
    settings: PolicySettings,
) -> dict[str, Any] | None:
    scan = [
        _evaluate_sigma(
            current=current,
            sigma=float(sigma),
            dx=float(dx),
            direction=1,
            config=config,
            settings=settings,
        )
        for sigma in np.linspace(float(lo), float(hi), max(int(settings.scan_points), 3))
    ]
    feasible = [item for item in scan if bool(item.get("feasible", False))]
    if not feasible:
        return None
    return max(feasible, key=lambda item: float(item.get("objective_value", -np.inf)))


def _node_payload(k: int, x: float, state: State, *, config, sigma: float | None) -> dict[str, Any]:
    metrics = _closure_metrics(state, config=config)
    sigma_value = float("nan") if sigma is None else float(sigma)
    return {
        "k": int(k),
        "x": float(x),
        "n_p": float(state.n_p),
        "T_e": float(state.T_e),
        "A": float(state.area(config)),
        "logA": float(state.logA),
        "sigma_logA": sigma_value,
        **metrics,
    }


def rollout_forward_phi_greedy(
    *,
    config,
    source: AnchorState,
    length: float,
    n_steps: int,
    settings: PolicySettings,
    A_floor: float | None,
    A_ceil: float | None,
) -> dict[str, Any]:
    total_length = float(length)
    steps = int(n_steps)
    if total_length <= 0.0 or steps <= 0:
        raise ValueError("length and n_steps must be positive.")
    dx = total_length / float(steps)
    if A_floor is None:
        logA_min = LogAreaSplineControl.lower_bound()
    else:
        logA_min = math.log(max(float(A_floor) / max(float(config.area_scale_m2), 1e-300), 1e-300))
    if A_ceil is None:
        logA_max = LogAreaSplineControl.upper_bound()
    else:
        logA_max = math.log(max(float(A_ceil) / max(float(config.area_scale_m2), 1e-300), 1e-300))

    states = [source.state]
    sigma_prev: float | None = None
    nodes = [_node_payload(0, 0.0, source.state, config=config, sigma=source.sigma_logA)]
    segments: list[dict[str, Any]] = []
    params = _physics_params(config)

    for k in range(steps):
        current = states[-1]
        A_current = current.area(config)
        coeff = compute_forward_affine_coefficients(
            n_p=current.n_p,
            T_e=current.T_e,
            A=A_current,
            logA=current.logA,
            params=params,
        )
        p0_forward = dx * float(coeff.a0)
        p1_forward = dx * float(coeff.a1)
        q0_forward = float(coeff.G_current) - float(settings.g_floor) + dx * float(coeff.b0)
        q1_forward = dx * float(coeff.b1)
        interval = _build_forward_interval(
            current=current,
            A_current=A_current,
            sigma_prev=sigma_prev,
            dx=dx,
            settings=settings,
            logA_min=logA_min,
            logA_max=logA_max,
            q0_forward=q0_forward,
            q1_forward=q1_forward,
            q1_tol=1.0e-14,
            g_margin_tol=float(settings.active_tol),
        )
        if not bool(interval["ok"]):
            segments.append(
                {
                    "k": int(k),
                    "ok": False,
                    "support_type": "empty_forward_sigma_interval",
                    "sigma": float("nan"),
                    "error": "empty forward sigma interval",
                    **interval,
                }
            )
            break

        if p1_forward > 0.0:
            sigma_selected = float(interval["sigma_interval_upper"])
            endpoint_source = str(interval["upper_source"])
            objective_bound_kind = "upper"
        elif p1_forward < 0.0:
            sigma_selected = float(interval["sigma_interval_lower"])
            endpoint_source = str(interval["lower_source"])
            objective_bound_kind = "lower"
        else:
            regularizer = 0.0 if sigma_prev is None or not np.isfinite(float(sigma_prev)) else float(sigma_prev)
            sigma_selected = float(np.clip(regularizer, interval["sigma_interval_lower"], interval["sigma_interval_upper"]))
            endpoint_source = "regularizer"
            objective_bound_kind = "flat"

        chosen = _evaluate_sigma(
            current=current,
            sigma=sigma_selected,
            dx=dx,
            direction=1,
            config=config,
            settings=settings,
        )
        solver_method = "forward_phi_endpoint"
        if not bool(chosen.get("feasible", False)):
            fallback = _scan_best_feasible(
                current=current,
                lo=float(interval["sigma_interval_lower"]),
                hi=float(interval["sigma_interval_upper"]),
                dx=dx,
                config=config,
                settings=settings,
            )
            if fallback is not None:
                chosen = fallback
                sigma_selected = float(chosen["sigma"])
                endpoint_source = "scan_best_feasible"
                objective_bound_kind = "scan"
                solver_method = "forward_phi_scan_best_feasible"

        next_state = chosen["next_state"]
        support_type = _support_from_source(endpoint_source, p1_forward=p1_forward, q1_forward=q1_forward)
        finite_phi_gain = float(chosen.get("delta_gain", float("nan")))
        predicted_phi_gain = float(p0_forward + p1_forward * A_current * sigma_selected)
        segments.append(
            {
                "k": int(k),
                "ok": bool(chosen.get("ok", False) and chosen.get("feasible", False)),
                "support_type": support_type,
                "sigma": float(sigma_selected),
                "sigma_selected": float(sigma_selected),
                "endpoint_source": endpoint_source,
                "objective_bound_kind": objective_bound_kind,
                "solver_method": solver_method,
                "a0": float(coeff.a0),
                "a1": float(coeff.a1),
                "b0": float(coeff.b0),
                "b1": float(coeff.b1),
                "p0_forward": float(p0_forward),
                "p1_forward": float(p1_forward),
                "q0_forward": float(q0_forward),
                "q1_forward": float(q1_forward),
                "p1q1_forward": float(p1_forward * q1_forward),
                "Phi_current": float(coeff.phi_current),
                "Phi_prime_forward_predicted": float(coeff.a0 + coeff.a1 * A_current * sigma_selected),
                "Phi_gain_predicted": predicted_phi_gain,
                "Phi_gain_finite": finite_phi_gain,
                "G_current": float(coeff.G_current),
                "G_margin_current": float(coeff.G_current - float(settings.g_floor)),
                "G_margin_next": float(dict(chosen.get("constraint_margins", {})).get("G", float("nan"))),
                "T_p_current": float(coeff.T_p_current),
                "T_p_next": float(chosen.get("T_p", float("nan"))),
                "A_current": float(A_current),
                "A_prime_selected": float(A_current * sigma_selected),
                "det_D": float(coeff.det_D),
                "max_abs_scaled_residual": float(chosen.get("max_abs_scaled_residual", float("nan"))),
                **interval,
            }
        )
        states.append(next_state)
        sigma_prev = float(sigma_selected)
        nodes.append(_node_payload(k + 1, dx * float(k + 1), next_state, config=config, sigma=sigma_prev))
        if not bool(chosen.get("ok", False) and chosen.get("feasible", False)):
            break

    arrays = {
        "x": np.asarray([float(node["x"]) for node in nodes], dtype=float),
        "n_p": np.asarray([float(node["n_p"]) for node in nodes], dtype=float),
        "T_e": np.asarray([float(node["T_e"]) for node in nodes], dtype=float),
        "A": np.asarray([float(node["A"]) for node in nodes], dtype=float),
        "sigma_logA": np.asarray([float(node["sigma_logA"]) for node in nodes], dtype=float),
    }
    ok = len(segments) == steps and all(bool(seg["ok"]) for seg in segments)
    return {
        "ok": bool(ok),
        "mode": "forward_phi_greedy",
        "case_config": config.to_dict(),
        "source": {
            "x": float(source.x),
            "n_p": float(source.state.n_p),
            "T_e": float(source.state.T_e),
            "A": float(source.state.area(config)),
            "log_n": float(source.state.log_n),
            "log_Te": float(source.state.log_Te),
            "logA": float(source.state.logA),
            "sigma_logA": _finite_or_none(float(source.sigma_logA)) if source.sigma_logA is not None else None,
            "source": str(source.source),
            "source_index": int(source.source_index),
        },
        "settings": {
            **settings.__dict__,
            "length": total_length,
            "dx": dx,
            "A_floor": _finite_or_none(float(A_floor)) if A_floor is not None else None,
            "A_ceil": _finite_or_none(float(A_ceil)) if A_ceil is not None else None,
        },
        "summary": _summary(nodes=nodes, segments=segments),
        "nodes": nodes,
        "segments": segments,
        "profile_arrays": arrays,
    }


def _summary(*, nodes: list[dict[str, Any]], segments: list[dict[str, Any]]) -> dict[str, Any]:
    support_counts: dict[str, int] = {}
    for seg in segments:
        key = str(seg.get("support_type", "unknown"))
        support_counts[key] = support_counts.get(key, 0) + 1
    delta = np.asarray([float(node["Delta"]) for node in nodes], dtype=float)
    G = np.asarray([float(node["G"]) for node in nodes], dtype=float)
    return {
        "n_nodes": int(len(nodes)),
        "n_segments": int(len(segments)),
        "support_counts": support_counts,
        "Delta_start": float(delta[0]) if delta.size else None,
        "Delta_end": float(delta[-1]) if delta.size else None,
        "Delta_max": float(np.nanmax(delta)) if delta.size else None,
        "Delta_min": float(np.nanmin(delta)) if delta.size else None,
        "G_min": float(np.nanmin(G)) if G.size else None,
        "G_max": float(np.nanmax(G)) if G.size else None,
        "max_abs_scaled_residual": (
            float(max(float(seg.get("max_abs_scaled_residual", 0.0)) for seg in segments)) if segments else 0.0
        ),
    }


def _load_compare_nodes(path: Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            rows.append({str(k): float(v) for k, v in row.items() if v != ""})
    return rows


def _write_plot(out_dir: Path, payload: dict[str, Any], compare_nodes: list[dict[str, Any]] | None, compare_label: str) -> str | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None

    nodes = list(payload["nodes"])
    segments = list(payload["segments"])
    x = np.asarray([float(node["x"]) for node in nodes], dtype=float)
    Delta = np.asarray([float(node["Delta"]) for node in nodes], dtype=float)
    G = np.asarray([float(node["G"]) for node in nodes], dtype=float)
    Te = np.asarray([float(node["T_e"]) for node in nodes], dtype=float)
    Tp = np.asarray([float(node["T_p"]) for node in nodes], dtype=float)
    A = np.asarray([float(node["A"]) for node in nodes], dtype=float)
    xs = x[1:]
    sigma = np.asarray([float(seg["sigma_selected"]) for seg in segments], dtype=float)

    fig, axes = plt.subplots(4, 1, figsize=(10, 11), sharex=True, constrained_layout=True)
    fig.suptitle("Forward greedy policy maximizing local dPhi/dx", fontsize=13)
    axes[0].plot(x, Delta, color="tab:blue", label="forward greedy")
    axes[0].set_ylabel("Phi")
    axes[0].grid(True, alpha=0.25)
    axes[1].semilogy(x, np.maximum(G, 1e-12), color="tab:green", label="forward greedy")
    axes[1].set_ylabel("G clipped")
    axes[1].grid(True, alpha=0.25)
    axes[2].plot(xs, sigma, color="tab:orange", label="forward greedy")
    axes[2].set_ylabel("sigma")
    axes[2].grid(True, alpha=0.25)
    axes[3].plot(x, Te, color="tab:red", label="Te greedy")
    axes[3].plot(x, Tp, color="tab:brown", label="Tp greedy")
    axes[3].set_ylabel("T [K]")
    axes[3].set_xlabel("forward x [m]")
    axes[3].grid(True, alpha=0.25)
    ax3b = axes[3].twinx()
    ax3b.plot(x, A, color="tab:purple", alpha=0.65, label="A greedy")
    ax3b.set_ylabel("A [m^2]")

    if compare_nodes:
        xc = np.asarray([float(node["x"]) for node in compare_nodes], dtype=float)
        axes[0].plot(xc, [float(node["Delta"]) for node in compare_nodes], color="black", ls="--", label=compare_label)
        axes[1].semilogy(xc, np.maximum([float(node["G"]) for node in compare_nodes], 1e-12), color="black", ls="--")
        axes[3].plot(xc, [float(node["T_e"]) for node in compare_nodes], color="tab:red", ls="--", alpha=0.55)
        axes[3].plot(xc, [float(node["T_p"]) for node in compare_nodes], color="tab:brown", ls="--", alpha=0.55)
        if "sigma_logA" in compare_nodes[0]:
            axes[2].plot(xc, [float(node["sigma_logA"]) for node in compare_nodes], color="black", ls="--", label=compare_label)

    axes[0].legend(loc="best")
    axes[2].legend(loc="best")
    axes[3].legend(loc="best")
    path = out_dir / "forward_phi_greedy_comparison.png"
    fig.savefig(path, dpi=180)
    return str(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Forward local greedy rollout that maximizes d(Phi)/dx under one-step constraints."
    )
    parser.add_argument("--case", default="freidberg_reference")
    parser.add_argument("--source-anchor-json", type=Path, required=True)
    parser.add_argument("--length", type=float, required=True)
    parser.add_argument("--n-steps", type=int, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--sigma-min", type=float, default=-0.5)
    parser.add_argument("--sigma-max", type=float, default=0.5)
    parser.add_argument("--curvature-max", type=float, default=8.0)
    parser.add_argument("--no-curvature-bound", action="store_true")
    parser.add_argument("--g-floor", type=float, default=0.0)
    parser.add_argument("--tp-floor", type=float, default=300.0)
    parser.add_argument("--active-tol", type=float, default=1e-6)
    parser.add_argument("--scan-points", type=int, default=41)
    parser.add_argument("--A-floor", type=float, default=None)
    parser.add_argument("--A-ceil", type=float, default=None)
    parser.add_argument("--compare-nodes-csv", type=Path, default=None)
    parser.add_argument("--compare-label", default="comparison")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_case_config(case=str(args.case))
    source = load_anchor_json(Path(args.source_anchor_json), config=config)
    settings = PolicySettings(
        direction="forward",
        objective="delta_gain",
        n_steps=int(args.n_steps),
        sigma_min=float(args.sigma_min),
        sigma_max=float(args.sigma_max),
        curvature_max=None if bool(args.no_curvature_bound) else float(args.curvature_max),
        g_floor=float(args.g_floor),
        tp_floor_K=float(args.tp_floor),
        scan_points=int(args.scan_points),
        active_tol=float(args.active_tol),
    )
    payload = rollout_forward_phi_greedy(
        config=config,
        source=source,
        length=float(args.length),
        n_steps=int(args.n_steps),
        settings=settings,
        A_floor=None if args.A_floor is None else float(args.A_floor),
        A_ceil=None if args.A_ceil is None else float(args.A_ceil),
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "forward_phi_greedy_summary.json", payload)
    write_csv(out_dir / "nodes.csv", list(payload["nodes"]))
    write_csv(out_dir / "segments.csv", list(payload["segments"]))
    save_profile_npz(out_dir / "profile.npz", dict(payload["profile_arrays"]))

    compare_nodes = _load_compare_nodes(Path(args.compare_nodes_csv)) if args.compare_nodes_csv is not None else None
    plot_path = _write_plot(out_dir, payload, compare_nodes, str(args.compare_label))
    short = {
        "ok": bool(payload["ok"]),
        "out_dir": str(out_dir),
        "plot": plot_path,
        **dict(payload["summary"]),
    }
    if compare_nodes:
        compare_delta = np.asarray([float(node["Delta"]) for node in compare_nodes], dtype=float)
        short["compare_label"] = str(args.compare_label)
        short["compare_Delta_end"] = float(compare_delta[-1])
        short["compare_Delta_max"] = float(np.nanmax(compare_delta))
        short["greedy_minus_compare_Delta_end"] = float(payload["summary"]["Delta_end"] - compare_delta[-1])
        short["greedy_minus_compare_Delta_max"] = float(payload["summary"]["Delta_max"] - np.nanmax(compare_delta))
    print(json.dumps(short, indent=2, sort_keys=True, default=json_default))
    return 0 if bool(payload["ok"]) else 2


if __name__ == "__main__":
    raise SystemExit(main())
