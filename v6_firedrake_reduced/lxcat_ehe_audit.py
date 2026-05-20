from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPO_ROOT / "references/cross_sections/e_He_LXCat_all_elastic_momentum_transfer.txt"
DEFAULT_OUT_DIR = REPO_ROOT / "references/cross_sections"

K_B_EV_PER_K = 8.617333262145e-5
CURRENT_CODE_SIGMA_EP_M2 = 3.942573033087758e-21
CLASSIC_E_HE_LOW_M2 = 5.0e-20
CLASSIC_E_HE_HIGH_M2 = 6.6e-20
YAMASAKI_INFERRED_SEED_5E4_M2 = 1.86e-19


@dataclass(frozen=True)
class LxcatBlock:
    database: str
    collision_type: str
    species: str
    process: str
    columns: str
    comments: tuple[str, ...]
    energy_eV: np.ndarray
    sigma_m2: np.ndarray

    def sigma_at_energy_eV(self, energy_eV: float) -> float:
        x = np.asarray(self.energy_eV, dtype=float)
        y = np.asarray(self.sigma_m2, dtype=float)
        mask = (x > 0.0) & (y > 0.0)
        x = x[mask]
        y = y[mask]
        if energy_eV <= float(x[0]):
            return float(y[0])
        if energy_eV >= float(x[-1]):
            return float(y[-1])
        return float(np.exp(np.interp(np.log(float(energy_eV)), np.log(x), np.log(y))))

    def sigma_at_temperature_K(self, temperature_K: float) -> float:
        return self.sigma_at_energy_eV(K_B_EV_PER_K * float(temperature_K))

    def to_dict(self, *, temperatures_K: tuple[float, ...]) -> dict[str, Any]:
        return {
            "database": self.database,
            "collision_type": self.collision_type,
            "species": self.species,
            "process": self.process,
            "columns": self.columns,
            "comments": list(self.comments),
            "n_rows": int(self.energy_eV.size),
            "energy_min_eV": float(np.nanmin(self.energy_eV)),
            "energy_max_eV": float(np.nanmax(self.energy_eV)),
            "sigma_m2_by_T_K": {
                str(float(temperature)): self.sigma_at_temperature_K(float(temperature))
                for temperature in temperatures_K
            },
        }


def parse_lxcat_blocks(path: Path) -> list[LxcatBlock]:
    blocks: list[LxcatBlock] = []
    database = ""
    collision_type = ""
    species = ""
    process = ""
    columns = ""
    comments: list[str] = []
    current_rows: list[tuple[float, float]] | None = None

    for raw_line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if line.startswith("DATABASE:"):
            database = line.split(":", 1)[1].strip()
            continue
        if line in {"ELASTIC", "EFFECTIVE", "EXCITATION", "IONIZATION", "ATTACHMENT"}:
            collision_type = line
            species = ""
            process = ""
            columns = ""
            comments = []
            continue
        if line.startswith("SPECIES:"):
            species = line.split(":", 1)[1].strip()
            continue
        if line.startswith("PROCESS:"):
            process = line.split(":", 1)[1].strip()
            continue
        if line.startswith("COMMENT:"):
            comments.append(line.split(":", 1)[1].strip())
            continue
        if line.startswith("COLUMNS:"):
            columns = line.split(":", 1)[1].strip()
            continue
        if line.startswith("-----"):
            if current_rows is None:
                current_rows = []
            else:
                if current_rows:
                    rows = np.asarray(current_rows, dtype=float)
                    blocks.append(
                        LxcatBlock(
                            database=database,
                            collision_type=collision_type,
                            species=species,
                            process=process,
                            columns=columns,
                            comments=tuple(comments),
                            energy_eV=rows[:, 0],
                            sigma_m2=rows[:, 1],
                        )
                    )
                current_rows = None
            continue
        if current_rows is not None and line:
            parts = line.split()
            if len(parts) != 2:
                continue
            try:
                current_rows.append((float(parts[0]), float(parts[1])))
            except ValueError:
                continue
    return blocks


def build_report(path: Path, *, temperatures_K: tuple[float, ...]) -> dict[str, Any]:
    all_blocks = parse_lxcat_blocks(path)
    blocks = [
        block
        for block in all_blocks
        if block.collision_type == "ELASTIC"
        and block.species.lower().replace(" ", "") == "e/he"
        and "elastic" in block.process.lower()
    ]
    if not blocks:
        raise RuntimeError(f"no e / He elastic blocks found in {path}")

    summary = {}
    for temperature in temperatures_K:
        values = np.array([block.sigma_at_temperature_K(temperature) for block in blocks], dtype=float)
        summary[str(float(temperature))] = {
            "energy_eV": float(K_B_EV_PER_K * float(temperature)),
            "min_m2": float(np.nanmin(values)),
            "median_m2": float(np.nanmedian(values)),
            "max_m2": float(np.nanmax(values)),
            "relative_span_over_median": float((np.nanmax(values) - np.nanmin(values)) / np.nanmedian(values)),
            "current_code_sigma_ratio_to_median": float(CURRENT_CODE_SIGMA_EP_M2 / np.nanmedian(values)),
            "yamasaki_seed_5e-4_inferred_ratio_to_median": float(
                YAMASAKI_INFERRED_SEED_5E4_M2 / np.nanmedian(values)
            ),
        }

    return {
        "source_file": str(Path(path).resolve()),
        "selection": {
            "species": "e / He",
            "collision_type": "ELASTIC",
            "process_contains": "Elastic",
            "interpretation": "LXCat ELASTIC denotes elastic momentum-transfer cross section.",
        },
        "reference_values": {
            "current_code_sigma_ep_m2": CURRENT_CODE_SIGMA_EP_M2,
            "classic_e_he_momentum_transfer_range_m2": [CLASSIC_E_HE_LOW_M2, CLASSIC_E_HE_HIGH_M2],
            "yamasaki_inferred_seed_5e-4_m2": YAMASAKI_INFERRED_SEED_5E4_M2,
        },
        "blocks": [block.to_dict(temperatures_K=temperatures_K) for block in blocks],
        "summary_by_temperature_K": summary,
    }


def _write_readme(path: Path, report: dict[str, Any], *, temperatures_K: tuple[float, ...]) -> None:
    lines = [
        "# e-He LXCat Elastic Momentum-Transfer Audit",
        "",
        "This folder contains a local copy of the LXCat download for electron-helium elastic",
        "momentum-transfer cross sections and a reproducible audit summary.",
        "",
        "Use this data for the current `sigma_ep` closure. Do not use the Cs+ / He ion-neutral",
        "Q(01) data for electron momentum transport.",
        "",
        "Representative median values across the included LXCat databases:",
        "",
        "| T_e (K) | E = kBT (eV) | median sigma (m^2) | database spread |",
        "| ---: | ---: | ---: | ---: |",
    ]
    summary = report["summary_by_temperature_K"]
    for temperature in temperatures_K:
        item = summary[str(float(temperature))]
        lines.append(
            f"| {temperature:g} | {item['energy_eV']:.6g} | "
            f"{item['median_m2']:.6e} | {100.0 * item['relative_span_over_median']:.3g}% |"
        )
    lines.extend(
        [
            "",
            "At 4300 K, the median LXCat value is close to the classic e-He momentum-transfer",
            "range and about 16x larger than the current legacy `sigma_ep` value.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit LXCat e-He elastic momentum-transfer data.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--temperatures-K", type=float, nargs="+", default=[2250.0, 3000.0, 4300.0, 4900.0, 6200.0])
    args = parser.parse_args(argv)

    temperatures = tuple(float(value) for value in args.temperatures_K)
    report = build_report(Path(args.input), temperatures_K=temperatures)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "e_he_lxcat_elastic_audit.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_readme(out_dir / "README.md", report, temperatures_K=temperatures)
    print(json.dumps(report["summary_by_temperature_K"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
