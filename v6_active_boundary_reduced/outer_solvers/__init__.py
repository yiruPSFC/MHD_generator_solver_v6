"""Outer optimization layer for the active-boundary reduced solver."""

from __future__ import annotations

from typing import Any

__all__ = [
    "LbfgsbOuterSolverConfig",
    "OuterRewardWeights",
    "PrescreenSettings",
    "prescreen_candidates",
    "run_outer_lbfgsb",
    "score_outer_result",
]


def __getattr__(name: str) -> Any:
    if name in {"LbfgsbOuterSolverConfig", "run_outer_lbfgsb"}:
        from .lbfgsb import LbfgsbOuterSolverConfig, run_outer_lbfgsb

        return {
            "LbfgsbOuterSolverConfig": LbfgsbOuterSolverConfig,
            "run_outer_lbfgsb": run_outer_lbfgsb,
        }[name]
    if name in {"PrescreenSettings", "prescreen_candidates"}:
        from .prescreen import PrescreenSettings, prescreen_candidates

        return {
            "PrescreenSettings": PrescreenSettings,
            "prescreen_candidates": prescreen_candidates,
        }[name]
    if name in {"OuterRewardWeights", "score_outer_result"}:
        from .reward import OuterRewardWeights, score_outer_result

        return {
            "OuterRewardWeights": OuterRewardWeights,
            "score_outer_result": score_outer_result,
        }[name]
    raise AttributeError(name)
