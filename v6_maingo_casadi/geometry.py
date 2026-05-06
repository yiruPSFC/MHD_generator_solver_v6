from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .constants import _EPS

@dataclass(frozen=True)
class SplineAreaDesign:
    a1: float
    a2: float
    a3: float

    KNOTS = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=float)
    _LOWER = math.log(0.25)
    _UPPER = math.log(8.0)

    @classmethod
    def lower_bound(cls) -> float:
        return float(cls._LOWER)

    @classmethod
    def upper_bound(cls) -> float:
        return float(cls._UPPER)

    @classmethod
    def from_sequence(cls, values: list[float] | tuple[float, float, float] | np.ndarray) -> "SplineAreaDesign":
        arr = np.asarray(values, dtype=float).reshape(3)
        return cls(a1=float(arr[0]), a2=float(arr[1]), a3=float(arr[2]))

    @classmethod
    def clipped(cls, values: list[float] | tuple[float, float, float] | np.ndarray) -> "SplineAreaDesign":
        arr = np.clip(np.asarray(values, dtype=float).reshape(3), cls._LOWER, cls._UPPER)
        return cls.from_sequence(arr)

    @staticmethod
    def _second_derivatives(x_nodes: np.ndarray, y_nodes: np.ndarray) -> np.ndarray:
        x_nodes = np.asarray(x_nodes, dtype=float).reshape(-1)
        y_nodes = np.asarray(y_nodes, dtype=float).reshape(-1)
        n = x_nodes.size
        if n < 3:
            return np.zeros(n, dtype=float)
        h = np.diff(x_nodes)
        rhs = np.zeros(n - 2, dtype=float)
        mat = np.zeros((n - 2, n - 2), dtype=float)
        for i in range(1, n - 1):
            row = i - 1
            if row > 0:
                mat[row, row - 1] = h[i - 1]
            mat[row, row] = 2.0 * (h[i - 1] + h[i])
            if row < n - 3:
                mat[row, row + 1] = h[i]
            rhs[row] = 6.0 * (
                (y_nodes[i + 1] - y_nodes[i]) / h[i]
                - (y_nodes[i] - y_nodes[i - 1]) / h[i - 1]
            )
        m = np.zeros(n, dtype=float)
        m[1:-1] = np.linalg.solve(mat, rhs)
        return m

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
                i = 0
            elif xq >= x_nodes[-1]:
                i = x_nodes.size - 2
            else:
                i = int(np.searchsorted(x_nodes, xq, side="right") - 1)
                i = min(max(i, 0), x_nodes.size - 2)
            x0 = x_nodes[i]
            x1 = x_nodes[i + 1]
            h = x1 - x0
            dx0 = x1 - xq
            dx1 = xq - x0
            m0 = second[i]
            m1 = second[i + 1]
            y0 = y_nodes[i]
            y1 = y_nodes[i + 1]
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
            np.array([0.0, 1.0, 0.0, 0.0, 0.0], dtype=float),
            np.array([0.0, 0.0, 1.0, 0.0, 0.0], dtype=float),
            np.array([0.0, 0.0, 0.0, 1.0, 1.0], dtype=float),
        ):
            vals, derivs = cls._evaluate_spline(
                x_nodes=cls.KNOTS,
                y_nodes=y_nodes,
                x_eval=x_norm,
            )
            basis.append(vals)
            slopes.append(derivs)
        return np.vstack(basis).T, np.vstack(slopes).T

    @classmethod
    def project_from_profile(
        cls,
        *,
        x: np.ndarray,
        A: np.ndarray,
    ) -> "SplineAreaDesign":
        x = np.asarray(x, dtype=float).reshape(-1)
        A = np.asarray(A, dtype=float).reshape(-1)
        if x.size != A.size or x.size < 2:
            raise ValueError("warm profile x and A must have the same length >= 2.")
        x_norm = (x - float(x[0])) / max(float(x[-1] - x[0]), _EPS)
        logA = np.log(np.maximum(A / max(float(A[0]), _EPS), _EPS))
        basis, _ = cls.basis_matrices(x_norm)
        coeffs, *_ = np.linalg.lstsq(basis, logA, rcond=None)
        return cls.clipped(coeffs)

    def as_array(self) -> np.ndarray:
        return np.array([self.a1, self.a2, self.a3], dtype=float)

    def to_dict(self) -> dict[str, float]:
        return {
            "a1": float(self.a1),
            "a2": float(self.a2),
            "a3": float(self.a3),
        }

    def evaluate_on_normalized_grid(
        self,
        x_norm: np.ndarray,
        *,
        length: float,
        area_scale: float = 1.0,
    ) -> dict[str, np.ndarray]:
        x_norm = np.asarray(x_norm, dtype=float).reshape(-1)
        basis, slopes = self.basis_matrices(x_norm)
        params = self.as_array()
        logA = basis @ params
        dlogA_dx_norm = slopes @ params
        A = float(area_scale) * np.exp(logA)
        sigma = dlogA_dx_norm / max(float(length), _EPS)
        return {
            "x_norm": x_norm,
            "x": x_norm * float(length),
            "logA": logA,
            "A": A,
            "sigma_logA": sigma,
        }

    def evaluate_profile(
        self,
        *,
        length: float,
        n_intervals: int,
        area_scale: float = 1.0,
    ) -> dict[str, np.ndarray]:
        x_norm = np.linspace(0.0, 1.0, int(n_intervals) + 1, dtype=float)
        return self.evaluate_on_normalized_grid(x_norm, length=float(length), area_scale=float(area_scale))


def _sample_area_reference(
    x_norm: np.ndarray,
    *,
    area_reference_x_norm: np.ndarray | None = None,
    area_reference_factor: np.ndarray | None = None,
    area_reference_sigma_logA: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    x_norm = np.asarray(x_norm, dtype=float).reshape(-1)
    if area_reference_x_norm is None or area_reference_factor is None:
        return np.ones_like(x_norm, dtype=float), np.zeros_like(x_norm, dtype=float)
    ref_x = np.asarray(area_reference_x_norm, dtype=float).reshape(-1)
    ref_factor = np.asarray(area_reference_factor, dtype=float).reshape(-1)
    if ref_x.size != ref_factor.size or ref_x.size < 2:
        raise ValueError("area reference x/factor arrays must have the same length >= 2.")
    factor = np.interp(x_norm, ref_x, ref_factor)
    if area_reference_sigma_logA is None:
        ref_sigma = np.zeros_like(ref_x, dtype=float)
    else:
        ref_sigma = np.asarray(area_reference_sigma_logA, dtype=float).reshape(-1)
        if ref_sigma.size != ref_x.size:
            raise ValueError("area reference sigma array must match reference x length.")
    sigma = np.interp(x_norm, ref_x, ref_sigma)
    return factor, sigma


def _evaluate_area_design_samples(
    *,
    ops,
    area_design: SplineAreaDesign,
    length: float,
    x_norm: np.ndarray,
    area_scale: float = 1.0,
    area_reference_x_norm: np.ndarray | None = None,
    area_reference_factor: np.ndarray | None = None,
    area_reference_sigma_logA: np.ndarray | None = None,
):
    x_norm = np.asarray(x_norm, dtype=float).reshape(-1)
    basis_nodes, slopes_nodes = SplineAreaDesign.basis_matrices(x_norm)
    params = [area_design.a1, area_design.a2, area_design.a3]
    ref_factor, ref_sigma = _sample_area_reference(
        x_norm,
        area_reference_x_norm=area_reference_x_norm,
        area_reference_factor=area_reference_factor,
        area_reference_sigma_logA=area_reference_sigma_logA,
    )

    def _apply_basis(row):
        return row[0] * params[0] + row[1] * params[1] + row[2] * params[2]

    logA_nodes = []
    A_nodes = []
    sigma_nodes = []
    for idx in range(int(x_norm.size)):
        logA = _apply_basis(basis_nodes[idx, :])
        sigma = float(ref_sigma[idx]) + _apply_basis(slopes_nodes[idx, :]) / float(length)
        logA_nodes.append(logA)
        A_nodes.append(float(area_scale) * float(ref_factor[idx]) * ops.exp(logA))
        sigma_nodes.append(sigma)
    return {
        "x_norm": x_norm,
        "x": x_norm * float(length),
        "logA": logA_nodes,
        "A": A_nodes,
        "sigma_logA": sigma_nodes,
    }


def _evaluate_area_design_nodes(
    *,
    ops,
    area_design: SplineAreaDesign,
    length: float,
    n_intervals: int,
    area_scale: float = 1.0,
    area_reference_x_norm: np.ndarray | None = None,
    area_reference_factor: np.ndarray | None = None,
    area_reference_sigma_logA: np.ndarray | None = None,
):
    x_norm = np.linspace(0.0, 1.0, int(n_intervals) + 1, dtype=float)
    return _evaluate_area_design_samples(
        ops=ops,
        area_design=area_design,
        length=float(length),
        x_norm=x_norm,
        area_scale=float(area_scale),
        area_reference_x_norm=area_reference_x_norm,
        area_reference_factor=area_reference_factor,
        area_reference_sigma_logA=area_reference_sigma_logA,
    )
