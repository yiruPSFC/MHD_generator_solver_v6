from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np


@dataclass(frozen=True)
class RangeValue:
    minimum: float
    maximum: float
    nominal: float

    def to_dict(self) -> dict[str, float]:
        return {
            "min": float(self.minimum),
            "max": float(self.maximum),
            "nominal": float(self.nominal),
        }

    def inlet_window_dict(self) -> dict[str, float]:
        return {
            "guess": float(self.nominal),
            "min": float(self.minimum),
            "max": float(self.maximum),
        }


@dataclass(frozen=True)
class Yamasaki2004DiskGeometry:
    """Disk-generator geometry from Murakami/Okuno/Yamasaki 2004 Fig. 2 text."""

    r_throat_m: float = 85.0e-3
    r_exit_m: float = 276.0e-3
    height_throat_m: float = 19.6e-3
    height_exit_m: float = 24.0e-3
    cross_section_throat_m2: float = 6.9e-3
    cross_section_exit_m2: float = 41.2e-3

    @property
    def length_m(self) -> float:
        return float(self.r_exit_m - self.r_throat_m)

    @property
    def area_ratio(self) -> float:
        return float(self.cross_section_exit_m2 / self.cross_section_throat_m2)

    def radius(self, x_norm: np.ndarray) -> np.ndarray:
        x_norm = np.asarray(x_norm, dtype=float)
        return self.r_throat_m + x_norm * self.length_m

    def height(self, x_norm: np.ndarray) -> np.ndarray:
        x_norm = np.asarray(x_norm, dtype=float)
        return self.height_throat_m + x_norm * (self.height_exit_m - self.height_throat_m)

    def annular_area(self, x_norm: np.ndarray) -> np.ndarray:
        return 2.0 * math.pi * self.radius(x_norm) * self.height(x_norm)

    def effective_area(self, x_norm: np.ndarray) -> np.ndarray:
        """Reported flow-direction cross-section, tied to the paper endpoints."""

        x_norm = np.asarray(x_norm, dtype=float)
        ann = self.annular_area(x_norm)
        throat_factor = self.cross_section_throat_m2 / float(self.annular_area(np.array([0.0]))[0])
        exit_factor = self.cross_section_exit_m2 / float(self.annular_area(np.array([1.0]))[0])
        factor = throat_factor + x_norm * (exit_factor - throat_factor)
        return ann * factor

    def profile(self, *, n_intervals: int = 80) -> dict[str, np.ndarray | float]:
        x_norm = np.linspace(0.0, 1.0, int(n_intervals) + 1, dtype=float)
        x = x_norm * self.length_m
        area = self.effective_area(x_norm)
        log_area = np.log(area / float(area[0]))
        sigma = np.gradient(log_area, x)
        return {
            "x": x,
            "x_norm": x_norm,
            "r_m": self.radius(x_norm),
            "height_m": self.height(x_norm),
            "A": area,
            "annular_area_m2": self.annular_area(x_norm),
            "length_m": self.length_m,
            "area_scale_m2": self.cross_section_throat_m2,
            "area_ratio": self.area_ratio,
            "volume_m3": float(np.trapezoid(area, x)),
        }


@dataclass(frozen=True)
class Yamasaki2004PaperParameters:
    title: str = (
        "Achievement of the Highest Performance of a CCMHD Generator: "
        "An Isentropic Efficiency of 63% and an Enthalpy Extraction Ratio of 31%"
    )
    doi: str = "10.1109/TPS.2004.835484"
    working_fluid_profile: str = "he_cs"
    magnetic_field_T: float = 3.0
    stagnation_temperature_K: float = 2250.0
    stagnation_pressure_Pa: RangeValue = field(default_factory=lambda: RangeValue(0.12e6, 0.16e6, 0.14e6))
    thermal_input_MW: RangeValue = field(default_factory=lambda: RangeValue(2.7, 4.3, 4.0))
    mass_flow_rate_kg_s: RangeValue = field(default_factory=lambda: RangeValue(0.27, 0.35, 0.35))
    seed_fraction: RangeValue = field(default_factory=lambda: RangeValue(1.0e-4, 8.0e-4, 5.0e-4))
    hall_voltage_V: RangeValue = field(default_factory=lambda: RangeValue(900.0, 1100.0, 1000.0))
    hall_current_A: RangeValue = field(default_factory=lambda: RangeValue(800.0, 1300.0, 1000.0))
    reported_enthalpy_extraction_percent: float = 30.8
    reported_isentropic_efficiency_percent: float = 63.0
    reported_electric_power_MW: float = 1.23
    reported_power_density_MW_m3: float = 297.0
    reported_generating_volume_m3: float = 0.004
    max_power_hall_voltage_V: float = 970.0
    max_power_hall_current_A: float = 1270.0
    load_resistance_ohm: RangeValue = field(default_factory=lambda: RangeValue(0.62, 0.77, 0.62))
    geometry: Yamasaki2004DiskGeometry = field(default_factory=Yamasaki2004DiskGeometry)

    def to_reference_dict(self) -> dict[str, object]:
        return {
            "title": self.title,
            "doi": self.doi,
            "working_fluid_profile": self.working_fluid_profile,
            "magnetic_field_T": float(self.magnetic_field_T),
            "stagnation_temperature_K": float(self.stagnation_temperature_K),
            "stagnation_pressure_Pa": self.stagnation_pressure_Pa.to_dict(),
            "thermal_input_MW": self.thermal_input_MW.to_dict(),
            "mass_flow_rate_kg_s": self.mass_flow_rate_kg_s.to_dict(),
            "seed_fraction": self.seed_fraction.to_dict(),
            "hall_voltage_V": self.hall_voltage_V.to_dict(),
            "hall_current_A": self.hall_current_A.to_dict(),
            "reported_enthalpy_extraction_percent": float(self.reported_enthalpy_extraction_percent),
            "reported_isentropic_efficiency_percent": float(self.reported_isentropic_efficiency_percent),
            "reported_electric_power_MW": float(self.reported_electric_power_MW),
            "reported_power_density_MW_m3": float(self.reported_power_density_MW_m3),
            "reported_generating_volume_m3": float(self.reported_generating_volume_m3),
            "max_power_hall_voltage_V": float(self.max_power_hall_voltage_V),
            "max_power_hall_current_A": float(self.max_power_hall_current_A),
            "load_resistance_ohm": self.load_resistance_ohm.to_dict(),
            "geometry_source": (
                "Fig. 2 text: disk height linear from 19.6 mm at r=85.0 mm "
                "to 24.0 mm at r=276 mm; throat/exit cross sections "
                "6.9e3/41.2e3 mm^2."
            ),
            "r_throat_m": float(self.geometry.r_throat_m),
            "r_exit_m": float(self.geometry.r_exit_m),
            "length_m": float(self.geometry.length_m),
            "height_throat_m": float(self.geometry.height_throat_m),
            "height_exit_m": float(self.geometry.height_exit_m),
            "cross_section_throat_m2": float(self.geometry.cross_section_throat_m2),
            "cross_section_exit_m2": float(self.geometry.cross_section_exit_m2),
            "cross_sectional_area_ratio": float(self.geometry.area_ratio),
        }


@dataclass(frozen=True)
class Yamasaki2004ModelSeed:
    """Solver initialization windows near the paper case.

    These are not additional paper measurements. They only provide an initial
    neighborhood for state variables that the paper does not report directly.
    """

    n_p_in_m3: RangeValue = field(default_factory=lambda: RangeValue(4.0e24, 5.5e24, 4.7587e24))
    electron_temperature_in_K: RangeValue = field(default_factory=lambda: RangeValue(4200.0, 6200.0, 4900.0))
    z_in: RangeValue = field(default_factory=lambda: RangeValue(40.0, 120.0, 80.0))
    area_log_window_offset: RangeValue = field(default_factory=lambda: RangeValue(-0.15, 0.15, 0.0))
    schedule_n_intervals: int = 40
    adaptive_bridge_count: int = 2
    adaptive_bridge_max_count: int = 8
    min_margin: float = 0.0
    mach_min: float = 0.9
    max_abs_dlogA_dx: float = 18.0
    tp_min_K: float = 1800.0

    def aligned_inlet_window(self, paper: Yamasaki2004PaperParameters) -> dict[str, dict[str, float]]:
        return {
            "np_in": self.n_p_in_m3.inlet_window_dict(),
            "te_in": self.electron_temperature_in_K.inlet_window_dict(),
            "z_in": self.z_in.inlet_window_dict(),
            "I_0": paper.hall_current_A.inlet_window_dict(),
            "seed_fraction": paper.seed_fraction.inlet_window_dict(),
        }

    def aligned_area_window(self, nominal_log_area: np.ndarray | None = None) -> dict[str, dict[str, float]]:
        """Return direct spline-coordinate bounds around the fitted paper geometry."""

        window = self.area_log_window_offset.inlet_window_dict()
        nominal = np.zeros(3, dtype=float) if nominal_log_area is None else np.asarray(nominal_log_area, dtype=float)
        nominal = nominal.reshape(3)
        return {
            "a1": {key: float(value) + float(nominal[0]) for key, value in window.items()},
            "a2": {key: float(value) + float(nominal[1]) for key, value in window.items()},
            "a3": {key: float(value) + float(nominal[2]) for key, value in window.items()},
        }

    def schedule_entry(self) -> dict[str, float | int | str]:
        return {
            "name": "maingo_yamasaki2004_disk_geometry_reference",
            "n_intervals": int(self.schedule_n_intervals),
            "transcription": "reduced_implicit_fixed_newton_backward_euler",
            "min_margin": float(self.min_margin),
            "mach_min": float(self.mach_min),
            "max_abs_dlogA_dx": float(self.max_abs_dlogA_dx),
            "tp_min": float(self.tp_min_K),
        }


YAMASAKI2004 = Yamasaki2004PaperParameters()
YAMASAKI2004_MODEL_SEED = Yamasaki2004ModelSeed()


def yamasaki2004_disk_geometry() -> Yamasaki2004DiskGeometry:
    return YAMASAKI2004.geometry
