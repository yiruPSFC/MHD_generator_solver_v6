#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys

REPO_DIR = Path(__file__).resolve().parents[2]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from v6_casadi.runners.analyze_continuation_failure_v6 import main


if __name__ == "__main__":
    raise SystemExit(main())
