"""Compatibility wrappers for legacy MHD physics functions.

Case parameters must live under ``v6_firedrake_reduced.cases``.  This module is
only a temporary bridge for closure equations, numerical helpers, and
working-fluid profile normalization that have not yet been rewritten locally.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
import types
from functools import lru_cache
from pathlib import Path
from typing import Any


_REPO_ROOT = Path(__file__).resolve().parents[1]
_LEGACY_DIR = _REPO_ROOT / "v6_maingo_casadi"
_LEGACY_PACKAGE = "_v6_firedrake_reduced_legacy_maingo"


def _ensure_casadi_importable() -> None:
    if "casadi" in sys.modules:
        return
    if importlib.util.find_spec("casadi") is not None:
        return
    stub = types.ModuleType("casadi")
    stub.__dict__["_v6_firedrake_reduced_stub"] = True
    sys.modules["casadi"] = stub


def _ensure_legacy_package() -> None:
    if _LEGACY_PACKAGE in sys.modules:
        return
    package = types.ModuleType(_LEGACY_PACKAGE)
    package.__path__ = [str(_LEGACY_DIR)]  # type: ignore[attr-defined]
    sys.modules[_LEGACY_PACKAGE] = package


def _import_legacy_module(name: str):
    _ensure_casadi_importable()
    _ensure_legacy_package()
    return importlib.import_module(f"{_LEGACY_PACKAGE}.{name}")


@lru_cache(maxsize=None)
def legacy_physics():
    return _import_legacy_module("physics")


@lru_cache(maxsize=None)
def legacy_numerics():
    return _import_legacy_module("numerics")


@lru_cache(maxsize=None)
def legacy_profiles():
    return _import_legacy_module("profiles")


def normalize_working_fluid_profile(profile: Any):
    return legacy_profiles()._normalize_working_fluid_profile(profile)


def ops_for_numeric():
    return legacy_numerics()._ops_for_numeric()


def inlet_design_generic(**kwargs: Any):
    return legacy_physics()._inlet_design_generic(**kwargs)


def closure_state(**kwargs: Any):
    return legacy_physics()._closure_state(**kwargs)


def dynamic_system_terms(**kwargs: Any):
    return legacy_physics()._dynamic_system_terms(**kwargs)
