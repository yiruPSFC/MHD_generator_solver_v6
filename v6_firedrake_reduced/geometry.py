from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np


@dataclass(frozen=True)
class LogAreaSplineControl:
    """Direct log-area spline control with existing v6 knot semantics."""

    a1: float
    a2: float
    a3: float

    KNOTS = np.array([0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0], dtype=float)
    _LOWER = math.log(0.25)
    _UPPER = math.log(8.0)

    @classmethod
    def from_sequence(cls, values: list[float] | tuple[float, float, float] | np.ndarray) -> "LogAreaSplineControl":
        arr = np.asarray(values, dtype=float).reshape(3)
        return cls(a1=float(arr[0]), a2=float(arr[1]), a3=float(arr[2]))

    @classmethod
    def clipped(cls, values: list[float] | tuple[float, float, float] | np.ndarray) -> "LogAreaSplineControl":
        arr = np.clip(np.asarray(values, dtype=float).reshape(3), cls._LOWER, cls._UPPER)
        return cls.from_sequence(arr)

    @classmethod
    def lower_bound(cls) -> float:
        return float(cls._LOWER)

    @classmethod
    def upper_bound(cls) -> float:
        return float(cls._UPPER)

    @staticmethod
    def _second_derivatives(x_nodes: np.ndarray, y_nodes: np.ndarray) -> np.ndarray:
        x_nodes = np.asarray(x_nodes, dtype=float).reshape(-1)
        y_nodes = np.asarray(y_nodes, dtype=float).reshape(-1)
        n_nodes = x_nodes.size
        if n_nodes < 3:
            return np.zeros(n_nodes, dtype=float)
        h = np.diff(x_nodes)
        rhs = np.zeros(n_nodes - 2, dtype=float)
        mat = np.zeros((n_nodes - 2, n_nodes - 2), dtype=float)
        for idx in range(1, n_nodes - 1):
            row = idx - 1
            if row > 0:
                mat[row, row - 1] = h[idx - 1]
            mat[row, row] = 2.0 * (h[idx - 1] + h[idx])
            if row < n_nodes - 3:
                mat[row, row + 1] = h[idx]
            rhs[row] = 6.0 * (
                (y_nodes[idx + 1] - y_nodes[idx]) / h[idx]
                - (y_nodes[idx] - y_nodes[idx - 1]) / h[idx - 1]
            )
        second = np.zeros(n_nodes, dtype=float)
        second[1:-1] = np.linalg.solve(mat, rhs)
        return second

    @classmethod
    def _evaluate_spline(
        cls,
        *,
        x_nodes: np.ndarray,
        y_nodes: np.ndarray,
        x_eval: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        x_nodes = np.asarray(x_nodes, dtype=float).reshape(-1)
        y_nodes = np.asarray(y_nodes, dtype=float).reshape(-1)
        x_eval = np.asarray(x_eval, dtype=float).reshape(-1)
        second = cls._second_derivatives(x_nodes, y_nodes)
        values = np.zeros_like(x_eval, dtype=float)
        derivs = np.zeros_like(x_eval, dtype=float)
        for j, xq in enumerate(x_eval):
            if xq <= x_nodes[0]:
                idx = 0
            elif xq >= x_nodes[-1]:
                idx = x_nodes.size - 2
            else:
                idx = int(np.searchsorted(x_nodes, xq, side="right") - 1)
                idx = min(max(idx, 0), x_nodes.size - 2)
            x0 = x_nodes[idx]
            x1 = x_nodes[idx + 1]
            h = x1 - x0
            dx0 = x1 - xq
            dx1 = xq - x0
            m0 = second[idx]
            m1 = second[idx + 1]
            y0 = y_nodes[idx]
            y1 = y_nodes[idx + 1]
            values[j] = (
                m0 * dx0**3 / (6.0 * h)
                + m1 * dx1**3 / (6.0 * h)
                + (y0 - m0 * h * h / 6.0) * dx0 / h
                + (y1 - m1 * h * h / 6.0) * dx1 / h
            )
            derivs[j] = (
                -m0 * dx0 * dx0 / (2.0 * h)
                + m1 * dx1 * dx1 / (2.0 * h)
                - (y0 - m0 * h * h / 6.0) / h
                + (y1 - m1 * h * h / 6.0) / h
            )
        return values, derivs

    @classmethod
    def basis_matrices(cls, x_norm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x_norm = np.asarray(x_norm, dtype=float).reshape(-1)
        basis = []
        slopes = []
        for y_nodes in (
            np.array([0.0, 1.0, 0.0, 0.0], dtype=float),
            np.array([0.0, 0.0, 1.0, 0.0], dtype=float),
            np.array([0.0, 0.0, 0.0, 1.0], dtype=float),
        ):
            vals, derivs = cls._evaluate_spline(x_nodes=cls.KNOTS, y_nodes=y_nodes, x_eval=x_norm)
            basis.append(vals)
            slopes.append(derivs)
        return np.vstack(basis).T, np.vstack(slopes).T

    @classmethod
    def project_from_profile(cls, *, x: np.ndarray, A: np.ndarray) -> "LogAreaSplineControl":
        x = np.asarray(x, dtype=float).reshape(-1)
        A = np.asarray(A, dtype=float).reshape(-1)
        if x.size != A.size or x.size < 2:
            raise ValueError("warm profile x and A must have the same length >= 2.")
        eps = 1e-30
        x_norm = (x - float(x[0])) / max(float(x[-1] - x[0]), eps)
        logA = np.log(np.maximum(A / max(float(A[0]), eps), eps))
        knot_values = np.interp(cls.KNOTS[1:], x_norm, logA)
        return cls.clipped(knot_values)

    def as_array(self) -> np.ndarray:
        return np.array([self.a1, self.a2, self.a3], dtype=float)

    def to_dict(self) -> dict[str, float]:
        return {"a1": float(self.a1), "a2": float(self.a2), "a3": float(self.a3)}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "LogAreaSplineControl":
        return cls(a1=float(payload["a1"]), a2=float(payload["a2"]), a3=float(payload["a3"]))

    def evaluate_profile(
        self,
        *,
        length: float,
        n_intervals: int,
        area_scale: float = 1.0,
    ) -> dict[str, np.ndarray]:
        x_norm = np.linspace(0.0, 1.0, int(n_intervals) + 1, dtype=float)
        basis, slopes = self.basis_matrices(x_norm)
        params = self.as_array()
        logA = basis @ params
        dlogA_dx_norm = slopes @ params
        eps = 1e-30
        return {
            "x_norm": x_norm,
            "x": x_norm * float(length),
            "logA": logA,
            "A": float(area_scale) * np.exp(logA),
            "sigma_logA": dlogA_dx_norm / max(float(length), eps),
        }
