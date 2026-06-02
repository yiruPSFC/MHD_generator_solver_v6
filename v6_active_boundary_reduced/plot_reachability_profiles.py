from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class ProfileCase:
    name: str
    source: str
    path: Path
    kind: str
    color: str
    linestyle: str


def _load_baseline_summary(path: Path) -> dict[str, np.ndarray]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    nodes = list(payload["nodes"])
    rows = list(reversed(nodes))
    x_original = np.asarray([float(row["x"]) for row in rows], dtype=float)
    s = x_original - float(x_original[0])
    return {
        "s": s,
        "s_norm": s / max(float(s[-1]), 1e-300),
        "n_p": np.asarray([float(row["n_p"]) for row in rows], dtype=float),
        "T_e": np.asarray([float(row["T_e"]) for row in rows], dtype=float),
        "T_p": np.asarray([float(row["T_p"]) for row in rows], dtype=float),
        "Delta": np.asarray([float(row["Delta"]) for row in rows], dtype=float),
        "A": np.asarray([float(row["A"]) for row in rows], dtype=float),
        "sigma_logA": np.asarray([float(row["sigma_logA"]) for row in rows], dtype=float),
        "G": np.asarray([float(row["G"]) for row in rows], dtype=float),
        "mach": np.asarray([float(row["mach"]) for row in rows], dtype=float),
    }


def _load_ipopt_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        x = np.asarray(data["x"], dtype=float)
        s = x - float(x[0])
        T_e = np.asarray(data["T_e"], dtype=float)
        T_p = np.asarray(data["T_p"], dtype=float)
        return {
            "s": s,
            "s_norm": s / max(float(s[-1]), 1e-300),
            "n_p": np.asarray(data["n_p"], dtype=float),
            "T_e": T_e,
            "T_p": T_p,
            "Delta": T_e / np.maximum(T_p, 1e-300) - 1.0,
            "A": np.asarray(data["A"], dtype=float),
            "sigma_logA": np.asarray(data["sigma_logA"], dtype=float),
            "G": np.asarray(data["velikhov_margin"], dtype=float),
            "mach": np.asarray(data["mach"], dtype=float),
        }


def _load_case(case: ProfileCase) -> dict[str, Any]:
    if case.kind == "baseline_summary":
        profile = _load_baseline_summary(case.path)
    elif case.kind == "ipopt_npz":
        profile = _load_ipopt_npz(case.path)
    else:
        raise ValueError(f"unknown case kind: {case.kind}")
    return {"case": case, "profile": profile}


def _sigma_node_s(profile: dict[str, np.ndarray]) -> np.ndarray:
    sigma = np.asarray(profile["sigma_logA"], dtype=float).reshape(-1)
    s = np.asarray(profile["s"], dtype=float).reshape(-1)
    if sigma.size == s.size:
        return s
    if sigma.size == s.size - 1:
        return 0.5 * (s[:-1] + s[1:])
    return np.linspace(float(s[0]), float(s[-1]), sigma.size)


def _sigma_node_s_norm(profile: dict[str, np.ndarray]) -> np.ndarray:
    sigma = np.asarray(profile["sigma_logA"], dtype=float).reshape(-1)
    s = np.asarray(profile["s_norm"], dtype=float).reshape(-1)
    if sigma.size == s.size:
        return s
    if sigma.size == s.size - 1:
        return 0.5 * (s[:-1] + s[1:])
    return np.linspace(float(s[0]), float(s[-1]), sigma.size)


def _plot_panel(ax, loaded: list[dict[str, Any]], key: str, ylabel: str, *, xkey: str = "s") -> None:
    for item in loaded:
        case: ProfileCase = item["case"]
        profile = item["profile"]
        if key == "sigma_logA":
            x = _sigma_node_s(profile) if xkey == "s" else _sigma_node_s_norm(profile)
        else:
            x = profile[xkey]
        ax.plot(
            x,
            profile[key],
            label=case.name,
            color=case.color,
            linestyle=case.linestyle,
            linewidth=1.8,
        )
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)


def _write_summary_csv(path: Path, loaded: list[dict[str, Any]]) -> None:
    rows: list[dict[str, Any]] = []
    for item in loaded:
        case: ProfileCase = item["case"]
        p = item["profile"]
        sigma = np.asarray(p["sigma_logA"], dtype=float)
        d_sigma = np.diff(sigma) if sigma.size > 1 else np.zeros(0)
        rows.append(
            {
                "case": case.name,
                "source": case.source,
                "length_m": float(np.asarray(p["s"])[-1]),
                "Delta_start": float(np.asarray(p["Delta"])[0]),
                "Delta_end": float(np.asarray(p["Delta"])[-1]),
                "A_min_m2": float(np.nanmin(p["A"])),
                "A_max_m2": float(np.nanmax(p["A"])),
                "sigma_min": float(np.nanmin(sigma)),
                "sigma_max": float(np.nanmax(sigma)),
                "max_abs_delta_sigma": float(np.nanmax(np.abs(d_sigma))) if d_sigma.size else 0.0,
                "T_p_min_K": float(np.nanmin(p["T_p"])),
                "G_min": float(np.nanmin(p["G"])),
                "mach_max": float(np.nanmax(p["mach"])),
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _default_cases(root: Path) -> list[ProfileCase]:
    return [
        ProfileCase(
            name="baseline marginal L=5.4m",
            source="reverse active-boundary baseline",
            path=root / "baseline_5p4/L_5p4/preparation_recovery_summary.json",
            kind="baseline_summary",
            color="black",
            linestyle="-",
        ),
        ProfileCase(
            name="IPOPT smooth L=5.4m",
            source="policy-euler, sigma_step_max=0.05",
            path=root / "ipopt_policy_euler_boundary_270_sigma_step/L_5p4/profile.npz",
            kind="ipopt_npz",
            color="#1f77b4",
            linestyle="--",
        ),
        ProfileCase(
            name="IPOPT no-step L=5.0m",
            source="policy-euler, no sigma-step bound",
            path=root / "ipopt_policy_euler_boundary_270_no_sigma_step/L_5/profile.npz",
            kind="ipopt_npz",
            color="#ff7f0e",
            linestyle="-.",
        ),
        ProfileCase(
            name="IPOPT no-step L=4.2m",
            source="policy-euler, no sigma-step bound",
            path=root / "ipopt_policy_euler_refine_270_no_sigma_step/L_4p2/profile.npz",
            kind="ipopt_npz",
            color="#2ca02c",
            linestyle=":",
        ),
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot Freidberg endpoint reachability profiles.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("outputs/freidberg_reachability_full_20260531_154455"),
    )
    parser.add_argument("--out-dir", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(args.root)
    out_dir = Path(args.out_dir) if args.out_dir is not None else root / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    cases = _default_cases(root)
    missing = [str(case.path) for case in cases if not case.path.exists()]
    if missing:
        raise FileNotFoundError("missing profile inputs:\n" + "\n".join(missing))
    loaded = [_load_case(case) for case in cases]
    _write_summary_csv(out_dir / "profile_plot_summary.csv", loaded)

    fig, axes = plt.subplots(4, 2, figsize=(13.5, 14.0), sharex=False)
    axes = axes.ravel()
    panels = [
        ("A", "A [m^2]"),
        ("sigma_logA", "sigma = dlogA/ds [1/m]"),
        ("Delta", "Te/Tp - 1"),
        ("T_e", "Te [K]"),
        ("T_p", "Tp [K]"),
        ("G", "Velikhov G"),
        ("mach", "Mach"),
        ("n_p", "n_p [m^-3]"),
    ]
    for ax, (key, ylabel) in zip(axes, panels, strict=True):
        _plot_panel(ax, loaded, key, ylabel)
        if key == "G":
            ax.axhline(0.0, color="0.3", linewidth=0.9)
            ax.set_yscale("symlog", linthresh=1e-3)
        if key == "T_p":
            ax.axhline(300.0, color="0.3", linewidth=0.9, linestyle=":")
        if key == "sigma_logA":
            ax.axhline(0.5, color="0.5", linewidth=0.8, linestyle=":")
            ax.axhline(-0.5, color="0.5", linewidth=0.8, linestyle=":")
    for ax in axes[-2:]:
        ax.set_xlabel("forward coordinate s [m]")
    axes[0].legend(loc="best", fontsize=9)
    fig.suptitle(
        "Freidberg endpoint reachability: forward from recovered source to Freidberg target",
        fontsize=14,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.975))
    fig.savefig(out_dir / "profile_comparison_forward.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(11.0, 7.0), sharex=True)
    for key, ylabel, ax in [
        ("A", "A [m^2]", axes[0]),
        ("sigma_logA", "sigma = dlogA/ds [1/m]", axes[1]),
    ]:
        _plot_panel(ax, loaded, key, ylabel)
        if key == "sigma_logA":
            ax.axhline(0.5, color="0.5", linewidth=0.8, linestyle=":")
            ax.axhline(-0.5, color="0.5", linewidth=0.8, linestyle=":")
    axes[1].set_xlabel("forward coordinate s [m]")
    axes[0].legend(loc="best", fontsize=9)
    fig.suptitle("Area and A-prime control comparison")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    fig.savefig(out_dir / "area_sigma_forward.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(11.0, 9.0), sharex=True)
    for key, ylabel, ax in [
        ("A", "A [m^2]", axes[0]),
        ("sigma_logA", "sigma = dlogA/ds [1/m]", axes[1]),
        ("Delta", "Te/Tp - 1", axes[2]),
    ]:
        _plot_panel(ax, loaded, key, ylabel, xkey="s_norm")
    axes[2].set_xlabel("normalized coordinate s/L")
    axes[0].legend(loc="best", fontsize=9)
    fig.suptitle("Shape comparison on normalized coordinate")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    fig.savefig(out_dir / "profile_comparison_normalized.png", dpi=180)
    plt.close(fig)

    print(json.dumps({"out_dir": str(out_dir), "cases": [case.name for case in cases]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
