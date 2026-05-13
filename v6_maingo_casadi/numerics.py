from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

try:
    import casadi as ca
except ModuleNotFoundError:
    ca = None

from .constants import _EPS, _G_HARD_MARGIN, _G_PENALTY_SCALE, _G_PENALTY_WEIGHT

def _json_load(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _safe_float(value: Any) -> float:
    return float(np.asarray(value, dtype=float))


def _max_op(ops, a, b):
    return ops.max(a, b)


def _min_op(ops, a, b):
    return ops.min(a, b)


def _safe_pos(ops, value, floor):
    if getattr(ops, "lb_func", None) is not None:
        return ops.lb_func(_max_op(ops, value, floor), floor)
    return _max_op(ops, value, floor)


def _clip_range(ops, value, lower, upper):
    clipped = _min_op(ops, _max_op(ops, value, lower), upper)
    if getattr(ops, "bounding_func", None) is not None:
        return ops.bounding_func(clipped, lower, upper)
    return clipped


def _floored_pos(ops, value, floor):
    floored = _max_op(ops, value, floor)
    if getattr(ops, "lb_func", None) is not None:
        return ops.lb_func(floored, floor)
    return floored


def _reduce_min(ops, values):
    items = list(values)
    if not items:
        raise ValueError("cannot reduce empty sequence.")
    acc = items[0]
    for item in items[1:]:
        acc = _min_op(ops, acc, item)
    return acc


def _velikhov_margin_penalty(ops, min_g):
    shortfall = _max_op(ops, float(_G_HARD_MARGIN) - min_g, 0.0)
    scaled = shortfall / float(_G_PENALTY_SCALE)
    return float(_G_PENALTY_WEIGHT) * scaled * scaled


def _ops_for_numeric():
    return SimpleNamespace(
        exp=math.exp,
        log=math.log,
        sqrt=math.sqrt,
        fabs=abs,
        max=max,
        min=min,
        pos=None,
        neg=None,
        lb_func=None,
        ub_func=None,
        bounding_func=None,
    )


def _ops_for_casadi():
    if ca is None:
        raise ModuleNotFoundError("casadi is required for _ops_for_casadi().")
    return SimpleNamespace(
        exp=ca.exp,
        log=ca.log,
        sqrt=ca.sqrt,
        fabs=ca.fabs,
        max=ca.fmax,
        min=ca.fmin,
        pos=None,
        neg=None,
        lb_func=None,
        ub_func=None,
        bounding_func=None,
    )


def _ops_for_maingo(maingopy_module):
    return SimpleNamespace(
        exp=maingopy_module.exp,
        log=maingopy_module.log,
        sqrt=maingopy_module.sqrt,
        fabs=maingopy_module.fabs,
        max=maingopy_module.max,
        min=maingopy_module.min,
        pos=maingopy_module.pos,
        neg=maingopy_module.neg,
        lb_func=maingopy_module.lb_func,
        ub_func=maingopy_module.ub_func,
        bounding_func=maingopy_module.bounding_func,
    )


def _safe_signed_denom(ops, value, *, sign_hint: str):
    if sign_hint == "positive":
        if getattr(ops, "pos", None) is not None:
            return ops.pos(value)
        return _max_op(ops, value, _EPS)
    if sign_hint == "negative":
        if getattr(ops, "neg", None) is not None:
            return ops.neg(value)
        return _min_op(ops, value, -_EPS)
    return value
