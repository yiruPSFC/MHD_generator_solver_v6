from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from v6_maingo_casadi.constants import _EPS
from v6_maingo_casadi.geometry import SplineAreaDesign


@dataclass(frozen=True)
class MachSplineDesign:
    """Three-knot log-Mach ratio spline using the existing area-spline basis."""

    m1: float
    m2: float
    m3: float

    KNOTS = SplineAreaDesign.KNOTS
    _LOWER = math.log(0.1)
    _UPPER = math.log(10.0)

    @classmethod
    def lower_bound(cls) -> float:
        return float(cls._LOWER)

    @classmethod
    def upper_bound(cls) -> float:
        return float(cls._UPPER)

    @classmethod
    def from_sequence(cls, values: list[float] | tuple[float, float, float] | np.ndarray) -> "MachSplineDesign":
        arr = np.asarray(values, dtype=float).reshape(3)
        return cls(m1=float(arr[0]), m2=float(arr[1]), m3=float(arr[2]))

    @classmethod
    def clipped(cls, values: list[float] | tuple[float, float, float] | np.ndarray) -> "MachSplineDesign":
        arr = np.clip(np.asarray(values, dtype=float).reshape(3), cls._LOWER, cls._UPPER)
        return cls.from_sequence(arr)

    @classmethod
    def project_from_profile(cls, *, x: np.ndarray, mach: np.ndarray) -> "MachSplineDesign":
        x = np.asarray(x, dtype=float).reshape(-1)
        mach = np.asarray(mach, dtype=float).reshape(-1)
        if x.size != mach.size or x.size < 2:
            raise ValueError("profile x and mach must have the same length >= 2.")
        if np.any(mach <= 0.0):
            raise ValueError("Mach profile must be strictly positive.")
        x_norm = (x - float(x[0])) / max(float(x[-1] - x[0]), _EPS)
        log_ratio = np.log(np.maximum(mach / max(float(mach[0]), _EPS), _EPS))
        knot_values = np.interp(cls.KNOTS[1:], x_norm, log_ratio)
        return cls.clipped(knot_values)

    def as_array(self) -> np.ndarray:
        return np.array([self.m1, self.m2, self.m3], dtype=float)

    def to_dict(self) -> dict[str, float]:
        return {"m1": float(self.m1), "m2": float(self.m2), "m3": float(self.m3)}

    def evaluate_on_normalized_grid(
        self,
        x_norm: np.ndarray,
        *,
        length: float,
        mach_in: float,
    ) -> dict[str, np.ndarray]:
        x_norm = np.asarray(x_norm, dtype=float).reshape(-1)
        if float(mach_in) <= 0.0:
            raise ValueError("mach_in must be positive.")
        basis, slopes = SplineAreaDesign.basis_matrices(x_norm)
        params = self.as_array()
        log_mach_ratio = basis @ params
        dlogM_dx_norm = slopes @ params
        mach = float(mach_in) * np.exp(log_mach_ratio)
        return {
            "x_norm": x_norm,
            "x": x_norm * float(length),
            "log_mach_ratio": log_mach_ratio,
            "mach": mach,
            "dlogM_dx": dlogM_dx_norm / max(float(length), _EPS),
        }

    def evaluate_profile(self, *, length: float, n_intervals: int, mach_in: float) -> dict[str, np.ndarray]:
        x_norm = np.linspace(0.0, 1.0, int(n_intervals) + 1, dtype=float)
        return self.evaluate_on_normalized_grid(x_norm, length=float(length), mach_in=float(mach_in))
