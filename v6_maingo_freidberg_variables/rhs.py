from __future__ import annotations

import math

import numpy as np

from v6_core.local_algebraic_closure import K_B

from .algebra import primitive_to_freidberg
from .models import FreidbergConfig, PrimitivePoint


def freidberg_rhs(point: PrimitivePoint, config: FreidbergConfig) -> tuple[float, float]:
    state = primitive_to_freidberg(point, config)
    M2 = point.mach * point.mach
    J2 = point.J_x * point.J_x + point.J_y * point.J_y
    p_p = point.n_p * K_B * point.T_p
    dHdx = (point.A / config.inlet_area_m2) * (point.v_p * point.J_y * config.B_T + point.eta * J2)
    denom = (M2 + 3.0) * p_p * point.v_p
    if abs(denom) < 1e-300:
        denom = 1e-300 if denom >= 0.0 else -1e-300
    dLdx = (
        -(12.0 / 5.0)
        * state.L_p
        / denom
        * (point.v_p * point.J_y * config.B_T - ((5.0 * M2 + 3.0) / 12.0) * point.eta * J2)
    )
    return float(dHdx), float(dLdx)


def freidberg_rhs_arrays(points: list[PrimitivePoint], config: FreidbergConfig) -> dict[str, np.ndarray]:
    rhs = np.asarray([freidberg_rhs(point, config) for point in points], dtype=float)
    return {
        "dHdx": rhs[:, 0],
        "dLdx": rhs[:, 1],
    }
