#!/usr/bin/env python3
"""Compatibility entrypoint for building the Yamasaki 2004 seed artifacts."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parents[1]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from v6_maingo_casadi.cases.yamasaki2004.build_seed import DEFAULT_OUT_DIR, build_seed, main

__all__ = ["DEFAULT_OUT_DIR", "build_seed", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
