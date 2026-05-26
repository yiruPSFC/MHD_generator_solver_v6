from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .design import CaseConfig, DesignVector
from .legacy_physics import closure_state, inlet_design_generic, ops_for_numeric
from .transport import working_fluid_for_config


@dataclass(frozen=True)
class NodeConstraintSummary:
    floor: float
    active_tolerance: float
    violation_tolerance: float
    x: np.ndarray
    G_node: np.ndarray
    margins: np.ndarray

    @property
    def min_G_node(self) -> float:
        return float(np.nanmin(self.G_node))

    @property
    def min_margin(self) -> float:
        return float(np.nanmin(self.margins))

    @property
    def argmin_index(self) -> int:
        return int(np.nanargmin(self.margins))

    @property
    def argmin_x(self) -> float:
        return float(self.x[self.argmin_index])

    def active_indices(self) -> list[int]:
        return [
            int(idx)
            for idx, value in enumerate(np.asarray(self.margins, dtype=float))
            if float(value) <= float(self.active_tolerance)
        ]

    def violated_indices(self) -> list[int]:
        return [
            int(idx)
            for idx, value in enumerate(np.asarray(self.margins, dtype=float))
            if float(value) < -float(self.violation_tolerance)
        ]

    def to_dict(self) -> dict[str, Any]:
        rows = []
        active = set(self.active_indices())
        violated = set(self.violated_indices())
        for idx, (x_val, g_val, margin) in enumerate(
            zip(self.x, self.G_node, self.margins, strict=True)
        ):
            rows.append(
                {
                    "index": int(idx),
                    "x": float(x_val),
                    "G_node": float(g_val),
                    "margin": float(margin),
                    "active": int(idx) in active,
                    "violated": int(idx) in violated,
                }
            )
        return {
            "sampling": "nodes",
            "floor": float(self.floor),
            "active_tolerance": float(self.active_tolerance),
            "violation_tolerance": float(self.violation_tolerance),
            "min_G_node": self.min_G_node,
            "min_margin": self.min_margin,
            "argmin_index": self.argmin_index,
            "argmin_x": self.argmin_x,
            "active_count": int(len(active)),
            "violated_count": int(len(violated)),
            "G_node": [float(value) for value in np.asarray(self.G_node, dtype=float)],
            "margins": [float(value) for value in np.asarray(self.margins, dtype=float)],
            "rows": rows,
        }


def evaluate_velikhov_node_constraints(
    *,
    profile: dict[str, Any],
    design: DesignVector,
    config: CaseConfig,
    floor: float = 0.0,
    active_tolerance: float = 1e-6,
    violation_tolerance: float = 0.0,
) -> NodeConstraintSummary:
    """Evaluate the node-only reduced path constraint G_node - floor >= 0."""

    ops = ops_for_numeric()
    fluid = working_fluid_for_config(config)
    x = np.asarray(profile["x"], dtype=float).reshape(-1)
    n_p = np.asarray(profile["n_p"], dtype=float).reshape(-1)
    T_e = np.asarray(profile["T_e"], dtype=float).reshape(-1)
    A = np.asarray(profile["A"], dtype=float).reshape(-1)
    sigma = np.asarray(profile["sigma_logA"], dtype=float).reshape(-1)
    if not (x.size == n_p.size == T_e.size == A.size == sigma.size):
        raise ValueError("profile arrays x, n_p, T_e, A, and sigma_logA must have matching lengths.")

    inlet = inlet_design_generic(
        ops=ops,
        n_p_in=design.n_p_in,
        T_e_in=float(design.T_e_in),
        Z_in=float(design.Z_in),
        I_0=float(design.I_0),
        seed_fraction=design.seed_fraction,
        B=float(design.B_T),
        inlet_A=float(config.area_scale_m2),
        working_fluid=fluid,
    )
    values = []
    for n_val, te_val, area_val, sigma_val in zip(n_p, T_e, A, sigma, strict=True):
        closure = closure_state(
            ops=ops,
            n_p=float(n_val),
            T_e=float(te_val),
            A=float(area_val),
            dot_N=float(inlet["dot_N"]),
            I_0=float(design.I_0),
            seed_fraction=design.seed_fraction,
            B=float(design.B_T),
            working_fluid=fluid,
        )
        values.append(float(closure["G"]))
    g_node = np.asarray(values, dtype=float)
    return NodeConstraintSummary(
        floor=float(floor),
        active_tolerance=float(active_tolerance),
        violation_tolerance=float(violation_tolerance),
        x=x,
        G_node=g_node,
        margins=g_node - float(floor),
    )
