#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import sys
import tempfile
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parents[2]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from v6_maingo_casadi.run_hybrid_maingo_casadi import main


def run_missing_dependency_smoke() -> None:
    try:
        main(["--out-dir", str(Path(tempfile.mkdtemp()) / "hybrid_missing_maingo")])
    except ImportError as exc:
        message = str(exc)
        if "maingopy" not in message:
            raise AssertionError(f"unexpected import failure message: {message}") from exc
        print("PASS: missing maingopy fails fast with an explicit message.")
        return
    raise AssertionError("expected ImportError when maingopy is unavailable.")


def run_available_dependency_smoke() -> None:
    out_dir = Path(tempfile.mkdtemp()) / "hybrid_with_maingo"
    exit_code = main(
        [
            "--out-dir",
            str(out_dir),
            "--maingo-max-time",
            "5",
        ]
    )
    if int(exit_code) != 0:
        raise AssertionError(f"hybrid main() returned nonzero exit code {exit_code}")
    expected = [
        out_dir / "maingo_summary.json",
        out_dir / "maingo_best_profile.npz",
        out_dir / "hybrid_summary.json",
        out_dir / "continuation" / "continuation_summary.json",
    ]
    missing = [str(path) for path in expected if not path.exists()]
    if missing:
        raise AssertionError(f"expected hybrid artifacts were not written: {missing}")
    print("PASS: maingopy-backed hybrid smoke test wrote all expected artifacts.")


def main_smoke() -> int:
    if importlib.util.find_spec("maingopy") is None:
        run_missing_dependency_smoke()
        return 0
    run_available_dependency_smoke()
    return 0


if __name__ == "__main__":
    raise SystemExit(main_smoke())
