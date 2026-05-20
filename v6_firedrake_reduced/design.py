from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .cases.yamasaki2004 import YAMASAKI2004, YAMASAKI2004_MODEL_SEED
from .geometry import LogAreaSplineControl


DESIGN_VARIABLE_NAMES = (
    "log_n_p_in",
    "T_e_in",
    "Z_in",
    "I_0",
    "log_seed_fraction",
    "a1",
    "a2",
    "a3",
)

GEOMETRY_LENGTH_MODES = ("radial", "inferred_swirl")
OBJECTIVE_PROFILE_ENTHALPY_EXTRACTION = "enthalpy_extraction"


@dataclass(frozen=True)
class DesignVector:
    log_n_p_in: float
    T_e_in: float
    Z_in: float
    I_0: float
    log_seed_fraction: float
    a1: float
    a2: float
    a3: float

    @property
    def n_p_in(self) -> float:
        return float(np.exp(self.log_n_p_in))

    @property
    def seed_fraction(self) -> float:
        return float(np.exp(self.log_seed_fraction))

    @property
    def area_control(self) -> LogAreaSplineControl:
        return LogAreaSplineControl(a1=float(self.a1), a2=float(self.a2), a3=float(self.a3))

    def as_array(self) -> np.ndarray:
        return np.array([float(getattr(self, name)) for name in DESIGN_VARIABLE_NAMES], dtype=float)

    @classmethod
    def from_array(cls, values: np.ndarray | list[float] | tuple[float, ...]) -> "DesignVector":
        arr = np.asarray(values, dtype=float).reshape(len(DESIGN_VARIABLE_NAMES))
        return cls(**{name: float(arr[idx]) for idx, name in enumerate(DESIGN_VARIABLE_NAMES)})

    def to_dict(self) -> dict[str, float]:
        return {name: float(getattr(self, name)) for name in DESIGN_VARIABLE_NAMES}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DesignVector":
        return cls(**{name: float(payload[name]) for name in DESIGN_VARIABLE_NAMES})


@dataclass(frozen=True)
class DesignBounds:
    lower: DesignVector
    upper: DesignVector

    def violations(self, design: DesignVector, *, atol: float = 0.0) -> list[dict[str, float | str]]:
        values = design.as_array()
        lower = self.lower.as_array()
        upper = self.upper.as_array()
        rows: list[dict[str, float | str]] = []
        for name, value, lo, hi in zip(DESIGN_VARIABLE_NAMES, values, lower, upper, strict=True):
            if float(value) < float(lo) - float(atol) or float(value) > float(hi) + float(atol):
                rows.append(
                    {
                        "name": name,
                        "value": float(value),
                        "min": float(lo),
                        "max": float(hi),
                    }
                )
        return rows

    def contains(self, design: DesignVector, *, atol: float = 0.0) -> bool:
        values = design.as_array()
        return bool(
            np.all(values >= self.lower.as_array() - float(atol))
            and np.all(values <= self.upper.as_array() + float(atol))
        )

    def to_dict(self) -> dict[str, dict[str, float]]:
        return {
            name: {
                "min": float(getattr(self.lower, name)),
                "max": float(getattr(self.upper, name)),
            }
            for name in DESIGN_VARIABLE_NAMES
        }


@dataclass(frozen=True)
class CaseConfig:
    case: str
    objective_profile: str
    length_m: float
    area_scale_m2: float
    B_T: float
    working_fluid_profile: str
    n_intervals: int
    design: DesignVector
    bounds: DesignBounds
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "case": self.case,
            "objective_profile": self.objective_profile,
            "length_m": float(self.length_m),
            "area_scale_m2": float(self.area_scale_m2),
            "B_T": float(self.B_T),
            "working_fluid_profile": self.working_fluid_profile,
            "n_intervals": int(self.n_intervals),
            "design": self.design.to_dict(),
            "bounds": self.bounds.to_dict(),
            "metadata": dict(self.metadata),
        }


def _area_bounds(nominal_log_area: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    seed = YAMASAKI2004_MODEL_SEED
    window = seed.aligned_area_window(nominal_log_area=nominal_log_area)
    guess = np.array([window[name]["guess"] for name in ("a1", "a2", "a3")], dtype=float)
    lower = np.array([window[name]["min"] for name in ("a1", "a2", "a3")], dtype=float)
    upper = np.array([window[name]["max"] for name in ("a1", "a2", "a3")], dtype=float)
    return guess, lower, upper


def inferred_swirl_length_m(*, n_intervals: int = 4096) -> float:
    """Infer streamwise length from reported Yamasaki cross sections.

    This treats the reported swirl-flow-direction cross section as
    A_reported = 2*pi*r*h*cos(theta), so ds/dr = 1/cos(theta).
    """

    paper = YAMASAKI2004
    geom = paper.geometry
    x_norm = np.linspace(0.0, 1.0, int(n_intervals) + 1, dtype=float)
    radial_x = x_norm * float(geom.length_m)
    reported_area = np.asarray(geom.effective_area(x_norm), dtype=float)
    annular_area = np.asarray(geom.annular_area(x_norm), dtype=float)
    cos_theta = np.clip(reported_area / np.maximum(annular_area, 1e-300), 1e-12, 1.0)
    return float(np.trapezoid(1.0 / cos_theta, radial_x))


def geometry_length_m(mode: str) -> float:
    paper = YAMASAKI2004
    normalized = str(mode).strip().lower().replace("-", "_")
    if normalized == "radial":
        return float(paper.geometry.length_m)
    if normalized == "inferred_swirl":
        return inferred_swirl_length_m()
    raise ValueError(f"unknown geometry_length_mode={mode!r}; expected one of {GEOMETRY_LENGTH_MODES}")


def load_case_config(
    *,
    case: str = "yamasaki2004",
    objective_profile: str = OBJECTIVE_PROFILE_ENTHALPY_EXTRACTION,
    n_intervals: int | None = None,
    geometry_length_mode: str = "radial",
) -> CaseConfig:
    if str(case).lower() != "yamasaki2004":
        raise ValueError("v0 only supports case='yamasaki2004'.")

    paper = YAMASAKI2004
    seed = YAMASAKI2004_MODEL_SEED
    geometry = paper.geometry.profile(n_intervals=80)
    area_design = LogAreaSplineControl.project_from_profile(
        x=np.asarray(geometry["x"], dtype=float),
        A=np.asarray(geometry["A"], dtype=float),
    )
    area_guess, area_lower, area_upper = _area_bounds(area_design.as_array())
    inlet = seed.aligned_inlet_window(paper)

    design = DesignVector(
        log_n_p_in=float(np.log(inlet["np_in"]["guess"])),
        T_e_in=float(inlet["te_in"]["guess"]),
        Z_in=float(inlet["z_in"]["guess"]),
        I_0=float(inlet["I_0"]["guess"]),
        log_seed_fraction=float(np.log(inlet["seed_fraction"]["guess"])),
        a1=float(area_guess[0]),
        a2=float(area_guess[1]),
        a3=float(area_guess[2]),
    )
    lower = DesignVector(
        log_n_p_in=float(np.log(inlet["np_in"]["min"])),
        T_e_in=float(inlet["te_in"]["min"]),
        Z_in=float(inlet["z_in"]["min"]),
        I_0=float(inlet["I_0"]["min"]),
        log_seed_fraction=float(np.log(inlet["seed_fraction"]["min"])),
        a1=float(area_lower[0]),
        a2=float(area_lower[1]),
        a3=float(area_lower[2]),
    )
    upper = DesignVector(
        log_n_p_in=float(np.log(inlet["np_in"]["max"])),
        T_e_in=float(inlet["te_in"]["max"]),
        Z_in=float(inlet["z_in"]["max"]),
        I_0=float(inlet["I_0"]["max"]),
        log_seed_fraction=float(np.log(inlet["seed_fraction"]["max"])),
        a1=float(area_upper[0]),
        a2=float(area_upper[1]),
        a3=float(area_upper[2]),
    )
    length_mode = str(geometry_length_mode).strip().lower().replace("-", "_")
    length_m = geometry_length_m(length_mode)
    return CaseConfig(
        case="yamasaki2004",
        objective_profile=str(objective_profile),
        length_m=float(length_m),
        area_scale_m2=float(paper.geometry.cross_section_throat_m2),
        B_T=float(paper.magnetic_field_T),
        working_fluid_profile=str(paper.working_fluid_profile),
        n_intervals=int(seed.schedule_n_intervals if n_intervals is None else n_intervals),
        design=design,
        bounds=DesignBounds(lower=lower, upper=upper),
        metadata={
            "paper_doi": paper.doi,
            "reported_enthalpy_extraction_percent": float(paper.reported_enthalpy_extraction_percent),
            "reported_isentropic_efficiency_percent": float(paper.reported_isentropic_efficiency_percent),
            "geometry_length_mode": length_mode,
            "radial_length_m": float(paper.geometry.length_m),
            "effective_length_m": float(length_m),
        },
    )
