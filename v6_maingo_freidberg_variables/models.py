from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from v6_core.local_algebraic_closure import SIGMA_EP


AMU_KG = 1.66053906660e-27


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def _finite_median(values: np.ndarray, *, name: str) -> float:
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        raise ValueError(f"cannot infer {name}: no finite values")
    return float(np.median(finite))


@dataclass(frozen=True)
class WorkingFluidProfile:
    key: str
    working_gas: str
    seed_species: str
    heavy_particle_mass_kg: float
    seed_ionization_energy_J: float
    sigma_ep: float = SIGMA_EP

    @classmethod
    def from_payload(cls, payload: dict[str, Any] | None) -> "WorkingFluidProfile":
        raw = dict(payload or {})
        mass_kg = raw.get("heavy_particle_mass_kg")
        if mass_kg is None and raw.get("heavy_particle_mass_amu") is not None:
            mass_kg = float(raw["heavy_particle_mass_amu"]) * AMU_KG
        ion_j = raw.get("seed_ionization_energy_J")
        if ion_j is None and raw.get("seed_ionization_energy_eV") is not None:
            ion_j = float(raw["seed_ionization_energy_eV"]) * 1.602176634e-19
        if mass_kg is None:
            raise ValueError("working fluid payload is missing heavy_particle_mass_kg")
        if ion_j is None:
            raise ValueError("working fluid payload is missing seed_ionization_energy_J")
        return cls(
            key=str(raw.get("key", "unknown")),
            working_gas=str(raw.get("working_gas", "")),
            seed_species=str(raw.get("seed_species", "")),
            heavy_particle_mass_kg=float(mass_kg),
            seed_ionization_energy_J=float(ion_j),
            sigma_ep=float(raw.get("sigma_ep", SIGMA_EP)),
        )

    @classmethod
    def from_summary(cls, summary: dict[str, Any]) -> "WorkingFluidProfile":
        baseline = dict(summary.get("baseline_seed", {}) or {})
        payload = summary.get("working_fluid_profile") or summary.get("working_fluid") or baseline.get("working_fluid")
        return cls.from_payload(payload)

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "working_gas": self.working_gas,
            "seed_species": self.seed_species,
            "heavy_particle_mass_kg": float(self.heavy_particle_mass_kg),
            "seed_ionization_energy_J": float(self.seed_ionization_energy_J),
            "sigma_ep": float(self.sigma_ep),
        }


@dataclass(frozen=True)
class FreidbergConfig:
    B_T: float
    inlet_area_m2: float
    dot_N: float
    I_0: float
    seed_fraction: float
    working_fluid: WorkingFluidProfile

    @classmethod
    def from_summary_and_profile(cls, summary_path: str | Path, profile_path: str | Path) -> "FreidbergConfig":
        summary_path = Path(summary_path)
        profile_path = Path(profile_path)
        summary = _load_json(summary_path)
        baseline = dict(summary.get("baseline_seed", {}) or {})
        fluid = WorkingFluidProfile.from_summary(summary)
        B = summary.get("B", baseline.get("B"))
        if B is None:
            raise ValueError(f"cannot infer B from {summary_path}")
        with np.load(profile_path) as data:
            A = np.asarray(data["A"], dtype=float)
            n_p = np.asarray(data["n_p"], dtype=float)
            v_p = np.asarray(data["v_p"], dtype=float)
            J_x = np.asarray(data["J_x"], dtype=float)
            if "seed_fraction" in data:
                seed_fraction = _finite_median(np.asarray(data["seed_fraction"], dtype=float), name="seed_fraction")
            else:
                seed_fraction = float(baseline.get("seed_fraction"))
            inlet_area = baseline.get("area_scale_m2", baseline.get("A_in"))
            if inlet_area is None or not math.isfinite(float(inlet_area)) or float(inlet_area) <= 0.0:
                inlet_area = float(A[0])
            dot_N = _finite_median(n_p * v_p * A, name="dot_N")
            I_0 = _finite_median(J_x * A, name="I_0")
        return cls(
            B_T=float(B),
            inlet_area_m2=float(inlet_area),
            dot_N=float(dot_N),
            I_0=float(I_0),
            seed_fraction=float(seed_fraction),
            working_fluid=fluid,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "B_T": float(self.B_T),
            "inlet_area_m2": float(self.inlet_area_m2),
            "dot_N": float(self.dot_N),
            "I_0": float(self.I_0),
            "seed_fraction": float(self.seed_fraction),
            "working_fluid": self.working_fluid.to_dict(),
        }


@dataclass(frozen=True)
class PrimitivePoint:
    x: float
    n_p: float
    T_e: float
    T_p: float
    A: float
    v_p: float
    n_e: float
    beta: float
    eta: float
    Z: float
    J_x: float
    J_y: float
    E_x: float
    mach: float
    velikhov_margin: float | None = None
    seed_fraction: float | None = None

    @classmethod
    def from_npz(cls, data: Any, idx: int) -> "PrimitivePoint":
        def scalar(name: str, default: float | None = None) -> float | None:
            if name not in data:
                return default
            return float(np.asarray(data[name], dtype=float)[idx])

        return cls(
            x=float(np.asarray(data["x"], dtype=float)[idx]),
            n_p=float(np.asarray(data["n_p"], dtype=float)[idx]),
            T_e=float(np.asarray(data["T_e"], dtype=float)[idx]),
            T_p=float(np.asarray(data["T_p"], dtype=float)[idx]),
            A=float(np.asarray(data["A"], dtype=float)[idx]),
            v_p=float(np.asarray(data["v_p"], dtype=float)[idx]),
            n_e=float(np.asarray(data["n_e"], dtype=float)[idx]),
            beta=float(np.asarray(data["beta"], dtype=float)[idx]),
            eta=float(np.asarray(data["eta"], dtype=float)[idx]),
            Z=float(np.asarray(data["Z"], dtype=float)[idx]),
            J_x=float(np.asarray(data["J_x"], dtype=float)[idx]),
            J_y=float(np.asarray(data["J_y"], dtype=float)[idx]),
            E_x=float(np.asarray(data["E_x"], dtype=float)[idx]),
            mach=float(np.asarray(data["mach"], dtype=float)[idx]),
            velikhov_margin=scalar("velikhov_margin"),
            seed_fraction=scalar("seed_fraction"),
        )

    def to_dict(self) -> dict[str, float | None]:
        return {
            "x": float(self.x),
            "n_p": float(self.n_p),
            "T_e": float(self.T_e),
            "T_p": float(self.T_p),
            "A": float(self.A),
            "v_p": float(self.v_p),
            "n_e": float(self.n_e),
            "beta": float(self.beta),
            "eta": float(self.eta),
            "Z": float(self.Z),
            "J_x": float(self.J_x),
            "J_y": float(self.J_y),
            "E_x": float(self.E_x),
            "mach": float(self.mach),
            "velikhov_margin": None if self.velikhov_margin is None else float(self.velikhov_margin),
            "seed_fraction": None if self.seed_fraction is None else float(self.seed_fraction),
        }


@dataclass(frozen=True)
class FreidbergState:
    H_p: float
    L_p: float
    T_e: float
    x: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return {
            "x": float(self.x),
            "H_p": float(self.H_p),
            "L_p": float(self.L_p),
            "T_e": float(self.T_e),
        }
