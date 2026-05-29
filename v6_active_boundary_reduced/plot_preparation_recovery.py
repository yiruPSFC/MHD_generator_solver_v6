from __future__ import annotations

import argparse
import csv
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from v6_firedrake_reduced.design import CaseConfig, DesignBounds, DesignVector
from v6_firedrake_reduced.forward import _freidberg_balance_terms
from v6_firedrake_reduced.legacy_physics import closure_state, inlet_design_generic, ops_for_numeric
from v6_firedrake_reduced.transport import working_fluid_for_config


def _json_default(value: Any):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        out = float(value)
        return out if np.isfinite(out) else None
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, np.bool_):
        return bool(value)
    raise TypeError(f"object of type {type(value).__name__} is not JSON serializable")


def _config_from_payload(payload: dict[str, Any]) -> CaseConfig:
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


def _rows_from_summary(summary: dict[str, Any]) -> list[dict[str, float]]:
    config = _config_from_payload(dict(summary["case_config"]))
    settings = dict(summary.get("settings", {}) or {})
    g_floor = float(settings.get("g_floor", 0.0))
    ops = ops_for_numeric()
    fluid = working_fluid_for_config(config)
    design = config.design
    inlet = inlet_design_generic(
        ops=ops,
        n_p_in=design.n_p_in,
        T_e_in=design.T_e_in,
        Z_in=design.Z_in,
        I_0=design.I_0,
        seed_fraction=design.seed_fraction,
        B=float(design.B_T),
        inlet_A=float(config.area_scale_m2),
        working_fluid=fluid,
    )
    rows: list[dict[str, float]] = []
    for node in summary["nodes"]:
        closure = closure_state(
            ops=ops,
            n_p=float(node["n_p"]),
            T_e=float(node["T_e"]),
            A=float(node["A"]),
            dot_N=float(inlet["dot_N"]),
            I_0=float(design.I_0),
            seed_fraction=design.seed_fraction,
            B=float(design.B_T),
            working_fluid=fluid,
        )
        balances = _freidberg_balance_terms(
            ops=ops,
            closure=closure,
            A=float(node["A"]),
            B_T=float(design.B_T),
            area_scale_m2=float(config.area_scale_m2),
            heavy_particle_mass_kg=float(fluid.heavy_particle_mass_kg),
            length_m=float(config.length_m),
        )
        n_s = max(float(closure["n_s"]), 1e-300)
        f_I = float(closure["n_e"]) / n_s
        rows.append(
            {
                "k": int(node["k"]),
                "x": float(node["x"]),
                "n_p": float(node["n_p"]),
                "T_e": float(node["T_e"]),
                "T_p": float(closure["T_p"]),
                "delta_T": float(float(node["T_e"]) / max(float(closure["T_p"]), 1e-300) - 1.0),
                "mach": float(closure["mach"]),
                "A": float(node["A"]),
                "sigma_logA": float(node["sigma_logA"]),
                "G": float(closure["G"]),
                "G_margin": float(closure["G"] - g_floor),
                "f_I": f_I,
                "v_p": float(closure["v_p"]),
                "beta": float(closure["beta"]),
                "Z": float(closure["Z"]),
                "n_e": float(closure["n_e"]),
                "H_p": float(balances["H_p"]),
                "L_p": float(balances["L_p"]),
                "rhs_H": float(balances["rhs_H"]),
                "rhs_L": float(balances["rhs_L"]),
                "H_scale": float(balances["H_scale"]),
                "L_scale": float(balances["L_scale"]),
            }
        )
    _attach_hl_residuals(rows)
    return rows


def _attach_hl_residuals(rows: list[dict[str, float]]) -> None:
    x = np.asarray([row["x"] for row in rows], dtype=float)
    H = np.asarray([row["H_p"] for row in rows], dtype=float)
    L = np.asarray([row["L_p"] for row in rows], dtype=float)
    if len(rows) < 2:
        for row in rows:
            row["dH_dx"] = float("nan")
            row["dL_dx"] = float("nan")
            row["H_residual"] = float("nan")
            row["L_residual"] = float("nan")
            row["scaled_H_residual"] = float("nan")
            row["scaled_L_residual"] = float("nan")
        return
    dH_dx = np.gradient(H, x, edge_order=1)
    dL_dx = np.gradient(L, x, edge_order=1)
    for idx, row in enumerate(rows):
        h_res = float(dH_dx[idx] - float(row["rhs_H"]))
        l_res = float(dL_dx[idx] - float(row["rhs_L"]))
        row["dH_dx"] = float(dH_dx[idx])
        row["dL_dx"] = float(dL_dx[idx])
        row["H_residual"] = h_res
        row["L_residual"] = l_res
        row["scaled_H_residual"] = h_res / max(abs(float(row["H_scale"])), 1e-300)
        row["scaled_L_residual"] = l_res / max(abs(float(row["L_scale"])), 1e-300)


def _segments(summary: dict[str, Any]) -> list[dict[str, Any]]:
    return list(summary.get("segments", []))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _support_color(name: str) -> str:
    if name == "G_supported":
        return "#d62728"
    if "curvature" in name:
        return "#1f77b4"
    if "sigma" in name:
        return "#9467bd"
    if "Tp" in name:
        return "#ff7f0e"
    if "no_feasible" in name:
        return "#7f7f7f"
    return "#2ca02c"


def _plot_overview(rows: list[dict[str, float]], segments: list[dict[str, Any]], path: Path) -> None:
    x = np.asarray([row["x"] for row in rows], dtype=float)
    fig, axes = plt.subplots(4, 1, figsize=(10.5, 11.0), sharex=True)
    for ax in axes:
        _shade_segments(ax, rows, segments)
        ax.grid(True, alpha=0.25)
    axes[0].plot(x, [row["T_e"] for row in rows], label="Te [K]", color="#1f77b4")
    axes[0].plot(x, [row["T_p"] for row in rows], label="Tp [K]", color="#d62728")
    axes[0].set_ylabel("temperature [K]")
    axes[0].legend(loc="best")
    axes[1].plot(x, [row["delta_T"] for row in rows], color="#111111")
    axes[1].set_ylabel("Delta = Te/Tp - 1")
    axes[2].plot(x, [row["mach"] for row in rows], color="#9467bd")
    axes[2].axhline(1.0, color="#999999", linewidth=1.0, linestyle="--")
    axes[2].set_ylabel("Mach")
    axes[3].plot(x, [row["G_margin"] for row in rows], color="#2ca02c")
    axes[3].axhline(0.0, color="#d62728", linewidth=1.0, linestyle="--")
    axes[3].set_ylabel("G margin")
    axes[3].set_xlabel("x [m] (upstream is negative)")
    fig.suptitle("Preparation recovery overview")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_closure(rows: list[dict[str, float]], segments: list[dict[str, Any]], path: Path) -> None:
    x = np.asarray([row["x"] for row in rows], dtype=float)
    fig, axes = plt.subplots(4, 1, figsize=(10.5, 11.0), sharex=True)
    for ax in axes:
        _shade_segments(ax, rows, segments)
        ax.grid(True, alpha=0.25)
    axes[0].plot(x, [row["n_p"] for row in rows], color="#1f77b4")
    axes[0].set_ylabel("n_p [m^-3]")
    axes[1].plot(x, [row["v_p"] for row in rows], color="#d62728")
    axes[1].set_ylabel("v_p [m/s]")
    axes[2].plot(x, [row["f_I"] for row in rows], color="#2ca02c")
    axes[2].set_ylabel("f_I")
    axes[3].plot(x, [row["beta"] for row in rows], label="beta", color="#9467bd")
    axes[3].plot(x, [row["Z"] for row in rows], label="Z", color="#8c564b")
    axes[3].set_ylabel("beta / Z")
    axes[3].legend(loc="best")
    axes[3].set_xlabel("x [m] (upstream is negative)")
    fig.suptitle("Preparation recovery closure quantities")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_geometry(rows: list[dict[str, float]], segments: list[dict[str, Any]], path: Path) -> None:
    x = np.asarray([row["x"] for row in rows], dtype=float)
    fig, axes = plt.subplots(3, 1, figsize=(10.5, 8.5), sharex=True)
    for ax in axes:
        _shade_segments(ax, rows, segments)
        ax.grid(True, alpha=0.25)
    axes[0].plot(x, [row["A"] for row in rows], color="#1f77b4")
    axes[0].set_ylabel("A [m^2]")
    axes[1].plot(x, [row["sigma_logA"] for row in rows], color="#d62728")
    axes[1].set_ylabel("sigma=dlogA/dx")
    mid_x = []
    codes = []
    labels: list[str] = []
    for seg in segments:
        k = int(seg["k"])
        if k + 1 >= len(rows):
            continue
        name = str(seg.get("support_type", "unknown"))
        if name not in labels:
            labels.append(name)
        mid_x.append(0.5 * (rows[k]["x"] + rows[k + 1]["x"]))
        codes.append(labels.index(name))
    axes[2].scatter(mid_x, codes, c=[_support_color(labels[c]) for c in codes], s=24)
    axes[2].set_yticks(range(len(labels)))
    axes[2].set_yticklabels(labels)
    axes[2].set_ylabel("active boundary")
    axes[2].set_xlabel("x [m] (upstream is negative)")
    fig.suptitle("Preparation recovery geometry and active set")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_hl_residuals(rows: list[dict[str, float]], segments: list[dict[str, Any]], path: Path) -> None:
    x = np.asarray([row["x"] for row in rows], dtype=float)
    fig, axes = plt.subplots(4, 1, figsize=(10.5, 11.0), sharex=True)
    for ax in axes:
        _shade_segments(ax, rows, segments)
        ax.grid(True, alpha=0.25)
    axes[0].plot(x, [row["H_p"] for row in rows], label="H_p", color="#1f77b4")
    axes[0].set_ylabel("H_p")
    axes[1].plot(x, [row["L_p"] for row in rows], label="L_p", color="#d62728")
    axes[1].set_ylabel("L_p")
    axes[2].plot(x, [row["scaled_H_residual"] for row in rows], label="scaled H residual", color="#1f77b4")
    axes[2].axhline(0.0, color="#999999", linewidth=1.0, linestyle="--")
    axes[2].set_ylabel("(dH/dx-rhs_H)/scale")
    axes[3].plot(x, [row["scaled_L_residual"] for row in rows], label="scaled L residual", color="#d62728")
    axes[3].axhline(0.0, color="#999999", linewidth=1.0, linestyle="--")
    axes[3].set_ylabel("(dL/dx-rhs_L)/scale")
    axes[3].set_xlabel("x [m] (upstream is negative)")
    fig.suptitle("Freidberg H/L residual diagnostics")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _hl_summary(rows: list[dict[str, float]]) -> dict[str, float]:
    return {
        "max_abs_H_residual": float(max(abs(row["H_residual"]) for row in rows)),
        "max_abs_L_residual": float(max(abs(row["L_residual"]) for row in rows)),
        "max_abs_scaled_H_residual": float(max(abs(row["scaled_H_residual"]) for row in rows)),
        "max_abs_scaled_L_residual": float(max(abs(row["scaled_L_residual"]) for row in rows)),
        "H_min": float(min(row["H_p"] for row in rows)),
        "H_max": float(max(row["H_p"] for row in rows)),
        "L_min": float(min(row["L_p"] for row in rows)),
        "L_max": float(max(row["L_p"] for row in rows)),
    }


def _shade_segments(ax, rows: list[dict[str, float]], segments: list[dict[str, Any]]) -> None:
    for seg in segments:
        k = int(seg["k"])
        if k + 1 >= len(rows):
            continue
        x0 = float(rows[k]["x"])
        x1 = float(rows[k + 1]["x"])
        name = str(seg.get("support_type", "unknown"))
        ax.axvspan(min(x0, x1), max(x0, x1), color=_support_color(name), alpha=0.06, linewidth=0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot and tabulate a preparation recovery run.")
    parser.add_argument("summary", type=Path, help="Path to preparation_recovery_summary.json.")
    parser.add_argument("--out-dir", type=Path, default=None)
    return parser


def write_preparation_diagnostics(summary_path: str | Path, *, out_dir: str | Path | None = None) -> dict[str, Any]:
    summary_path = Path(summary_path).resolve()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    out_dir = Path(out_dir) if out_dir is not None else summary_path.parent / "diagnostic_plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = _rows_from_summary(summary)
    segments = _segments(summary)
    _write_csv(out_dir / "node_closure_diagnostics.csv", rows)
    (out_dir / "node_closure_diagnostics.json").write_text(
        json.dumps(rows, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    _plot_overview(rows, segments, out_dir / "overview.png")
    _plot_closure(rows, segments, out_dir / "closure_quantities.png")
    _plot_geometry(rows, segments, out_dir / "geometry_active_set.png")
    _plot_hl_residuals(rows, segments, out_dir / "hl_residuals.png")
    hl_summary = _hl_summary(rows)
    (out_dir / "hl_residual_summary.json").write_text(
        json.dumps(hl_summary, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    payload = {
        "out_dir": str(out_dir),
        "overview_png": str(out_dir / "overview.png"),
        "closure_quantities_png": str(out_dir / "closure_quantities.png"),
        "geometry_active_set_png": str(out_dir / "geometry_active_set.png"),
        "hl_residuals_png": str(out_dir / "hl_residuals.png"),
        "hl_residual_summary_json": str(out_dir / "hl_residual_summary.json"),
        "node_closure_csv": str(out_dir / "node_closure_diagnostics.csv"),
        "hl_summary": hl_summary,
        "node_count": len(rows),
        "segment_count": len(segments),
    }
    (out_dir / "diagnostic_manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    return payload


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = write_preparation_diagnostics(args.summary, out_dir=args.out_dir)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
