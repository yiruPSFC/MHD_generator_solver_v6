#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
REPO_DIR = _THIS_DIR.parents[0]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from v6_casadi.optimize_area_profile_casadi_v6 import WarmStartProfile, optimize_area_profile
from v6_global_marginal.reference_recovery.global_plotting_v6 import plot_global_results_v6


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run HS grid refinement from an existing stage_3 warm profile")
    p.add_argument("--np-in", type=float, default=3.05e25)
    p.add_argument("--z-in", type=float, default=75.954994)
    p.add_argument("--tp-in", type=float, default=429.0)
    p.add_argument("--te-in", type=float, default=4420.0)
    p.add_argument("--A-in", type=float, default=0.447)
    p.add_argument("--B", type=float, default=10.2)
    p.add_argument("--L", type=float, default=5.4)
    p.add_argument("--warm-start-dx", type=float, default=0.01)
    p.add_argument(
        "--out-dir",
        type=str,
        default=str(_THIS_DIR / "outputs" / "continuation" / "reference_case"),
    )
    p.add_argument(
        "--stage3-npz",
        type=str,
        default="",
        help="warm-start npz; defaults to <out-dir>/stage_3.npz",
    )
    p.add_argument(
        "--intervals",
        type=str,
        default="160,320",
        help="comma-separated interval counts for HS refinement",
    )
    return p


def _load_warm_profile(path: Path) -> WarmStartProfile:
    data = np.load(path)
    return WarmStartProfile(
        x=np.asarray(data["x"], dtype=float),
        n_p=np.asarray(data["n_p"], dtype=float),
        T_e=np.asarray(data["T_e"], dtype=float),
        A=np.asarray(data["A"], dtype=float),
        sigma_logA=np.asarray(data["sigma_logA"], dtype=float),
        source=str(path),
    )


def _profile_payload(result) -> dict[str, np.ndarray]:
    return {
        "x": result.x,
        "n_p": result.n_p,
        "T_e": result.T_e,
        "T_p": result.T_p,
        "A": result.A,
        "v_p": result.v_p,
        "n_e": result.n_e,
        "beta": result.beta,
        "eta": result.eta,
        "Z": result.Z,
        "J_x": result.J_x,
        "J_y": result.J_y,
        "E_x": result.E_x,
        "mach": result.mach,
        "velikhov_margin": result.velikhov_margin,
        "sigma_logA": result.sigma_logA,
    }


def _save_profile_bundle(*, out_dir: Path, stem: str, result, B: float) -> dict[str, object]:
    npz_path = out_dir / f"{stem}.npz"
    png_path = out_dir / f"{stem}.png"
    payload = _profile_payload(result)
    np.savez(npz_path, **payload)
    plot_stats = plot_global_results_v6(
        payload,
        png_path,
        B=B,
        seed_fraction=result.inlet.seed_fraction,
        title=(
            f"{stem}: {result.transcription}, status={result.return_status}, "
            f"acceptable={result.acceptable}, dTe={result.objective_delta_Te:.2f} K"
        ),
    )
    return {
        "npz_path": str(npz_path),
        "plot_path": str(png_path),
        "plot_stats": {k: float(v) for k, v in plot_stats.items()},
    }


def _make_overlay_plots(out_dir: Path, profiles: dict[int, dict[str, np.ndarray]]) -> None:
    colors = {80: "C0", 160: "C1", 320: "C2"}

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    series = [
        ("A (m^2)", "A"),
        ("T_e (K)", "T_e"),
        ("T_p (K)", "T_p"),
        ("Mach", "mach"),
        ("Velikhov margin G", "velikhov_margin"),
        ("load power density (MW/m^3)", None),
    ]
    for ax, (title, key) in zip(axes.ravel(), series):
        for n, d in profiles.items():
            x = d["x"]
            y = -(d["J_x"] * d["E_x"]) / 1e6 if key is None else d[key]
            ax.plot(x, y, label=f"HS {n}", lw=2, color=colors.get(n))
        ax.set_title(title)
        ax.set_xlabel("x (m)")
        ax.grid(True, alpha=0.3)
    axes[0, 0].legend(fontsize=9)
    fig.suptitle("Stage 3 refinement comparison: 80 vs 160 vs 320 intervals", fontsize=14)
    fig.savefig(out_dir / "stage3_refinement_overlay.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    zoom = (3.2, 4.3)
    zoom_series = [
        ("T_e (K)", "T_e"),
        ("T_p (K)", "T_p"),
        ("Mach", "mach"),
        ("Velikhov margin G", "velikhov_margin"),
    ]
    for ax, (title, key) in zip(axes.ravel(), zoom_series):
        for n, d in profiles.items():
            x = d["x"]
            mask = (x >= zoom[0]) & (x <= zoom[1])
            ax.plot(x[mask], d[key][mask], label=f"HS {n}", lw=2, color=colors.get(n))
        ax.set_title(title)
        ax.set_xlabel("x (m)")
        ax.grid(True, alpha=0.3)
    axes[0, 0].legend(fontsize=9)
    fig.suptitle("Kink-region zoom: x in [3.2, 4.3] m", fontsize=14)
    fig.savefig(out_dir / "stage3_refinement_zoom.png", dpi=180)
    plt.close(fig)


def _refinement_summary(profiles: dict[int, dict[str, np.ndarray]]) -> dict[str, dict[str, float | None]]:
    summary: dict[str, dict[str, float | None]] = {}
    for n, d in profiles.items():
        x = d["x"]
        Te = d["T_e"]
        Tp = d["T_p"]
        A = d["A"]
        mach = d["mach"]
        G = d["velikhov_margin"]
        dTe_dx = np.gradient(Te, x)
        after = x >= 3.0
        idx_G100 = np.where(after & (G > 100.0))[0]
        idx_Te4450 = np.where(after & (Te > 4450.0))[0]
        idx_dTedx500 = np.where(after & (dTe_dx > 500.0))[0]
        summary[str(n)] = {
            "Te_out": float(Te[-1]),
            "Tp_min": float(np.min(Tp)),
            "A_out": float(A[-1]),
            "Mach_min": float(np.min(mach)),
            "G_max": float(np.max(G)),
            "x_after3m_G_gt_1e2_m": None if idx_G100.size == 0 else float(x[idx_G100[0]]),
            "x_after3m_Te_gt_4450K_m": None if idx_Te4450.size == 0 else float(x[idx_Te4450[0]]),
            "x_after3m_dTe_dx_gt_500_m": None if idx_dTedx500.size == 0 else float(x[idx_dTedx500[0]]),
        }
    return summary


def main() -> int:
    args = _build_parser().parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stage3_path = Path(args.stage3_npz) if args.stage3_npz else out_dir / "stage_3.npz"
    warm = _load_warm_profile(stage3_path)

    interval_list = [int(part.strip()) for part in str(args.intervals).split(",") if part.strip()]
    results_out: dict[str, object] = {}
    profiles: dict[int, dict[str, np.ndarray]] = {80: dict(np.load(stage3_path))}

    for n in interval_list:
        result = optimize_area_profile(
            n_p_in=float(args.np_in),
            Z_in=float(args.z_in),
            T_p_in=float(args.tp_in),
            T_e_in=float(args.te_in),
            A_in=float(args.A_in),
            B=float(args.B),
            length=float(args.L),
            n_intervals=n,
            transcription="hermite-simpson",
            min_margin=0.0,
            mach_min=1.4,
            A_min_ratio=0.95,
            A_max_ratio=2.1,
            max_abs_dlogA_dx=0.20,
            smooth_weight=0.01,
            warm_profile_track_weight=50.0,
            warm_control_track_weight=10.0,
            objective_weight=0.02,
            warm_start="marginal",
            warm_start_dx=float(args.warm_start_dx),
            warm_profile=warm,
            ipopt_max_iter=1500,
        )
        bundle = _save_profile_bundle(out_dir=out_dir, stem=f"refine_{n}", result=result, B=float(args.B))
        results_out[str(n)] = {
            "success": bool(result.success),
            "acceptable": bool(result.acceptable),
            "return_status": result.return_status,
            "objective_delta_Te_K": float(result.objective_delta_Te),
            "diagnostics": result.diagnostics,
            "artifacts": bundle,
        }
        warm = WarmStartProfile(
            x=result.x,
            n_p=result.n_p,
            T_e=result.T_e,
            A=result.A,
            sigma_logA=result.sigma_logA,
            source=f"refine_{n}",
        )
        profiles[n] = _profile_payload(result)

    _make_overlay_plots(out_dir, profiles)
    summary = _refinement_summary(profiles)
    payload = {
        "stage3_npz": str(stage3_path),
        "intervals": interval_list,
        "results": results_out,
        "summary": summary,
        "overlay_plot": str(out_dir / "stage3_refinement_overlay.png"),
        "zoom_plot": str(out_dir / "stage3_refinement_zoom.png"),
    }
    (out_dir / "stage3_refinement_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
