from __future__ import annotations

from typing import Any, Callable

import numpy as np


def finite_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return out if np.isfinite(out) else float(default)


def profile_stat(
    nodes: list[dict[str, Any]],
    name: str,
    reducer: Callable[[list[float]], float],
    default: float = float("nan"),
) -> float:
    values = [finite_float(node.get(name)) for node in nodes]
    finite = [value for value in values if np.isfinite(value)]
    if not finite:
        return float(default)
    return float(reducer(finite))


def area_ratio_from_nodes(nodes: list[dict[str, Any]]) -> float:
    a_min = profile_stat(nodes, "A", min)
    a_max = profile_stat(nodes, "A", max)
    if not np.isfinite(a_min) or not np.isfinite(a_max) or a_min <= 0.0:
        return float("nan")
    return float(a_max / a_min)


def soft_square(shortfall: float, scale: float) -> float:
    denom = max(float(scale), 1e-300)
    return float(max(float(shortfall), 0.0) / denom) ** 2
