from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from v6_firedrake_reduced.cases.freidberg_reference import load_reference_profile
from v6_firedrake_reduced.design import CaseConfig

from ..core.policy import AnchorState, State, _closure_metrics, anchor_from_dict, anchor_from_profile


def json_default(value: Any):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        out = float(value)
        return out if np.isfinite(out) else None
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, np.bool_):
        return bool(value)
    raise TypeError(f"object of type {type(value).__name__} is not JSON serializable")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=json_default) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_profile(path: Path | None, *, case: str) -> dict[str, np.ndarray]:
    if path is None:
        if str(case).strip().lower().replace("-", "_") != "freidberg_reference":
            raise ValueError("--profile-npz is required unless case is freidberg_reference.")
        return load_reference_profile()
    with np.load(path) as data:
        return {name: np.asarray(data[name], dtype=float) for name in data.files}


def anchor_payload(anchor: AnchorState, *, config: CaseConfig) -> dict[str, Any]:
    metrics = _closure_metrics(anchor.state, config=config)
    return {
        "source": str(anchor.source),
        "source_index": int(anchor.source_index),
        "x": float(anchor.x),
        "n_p": float(anchor.state.n_p),
        "T_e": float(anchor.state.T_e),
        "A": float(anchor.state.area(config)),
        "log_n": float(anchor.state.log_n),
        "log_Te": float(anchor.state.log_Te),
        "logA": float(anchor.state.logA),
        "sigma_logA": None if anchor.sigma_logA is None else float(anchor.sigma_logA),
        **metrics,
    }


def anchor_from_node_payload(
    node: dict[str, Any],
    *,
    config: CaseConfig,
    source: str,
    source_index: int | None = None,
) -> AnchorState:
    payload = dict(node)
    payload["source"] = source
    payload["source_index"] = int(payload.get("k", -1) if source_index is None else source_index)
    return anchor_from_dict(payload, config=config)


def load_anchor_json(path: Path, *, config: CaseConfig) -> AnchorState:
    return anchor_from_dict(json.loads(path.read_text(encoding="utf-8")), config=config)


def load_profile_anchor(
    path: Path | None,
    *,
    index: int,
    config: CaseConfig,
    source: str | None = None,
) -> AnchorState:
    profile = load_profile(path, case=config.case)
    return anchor_from_profile(
        profile,
        index=int(index),
        config=config,
        source=str(source or path or f"{config.case}:built_in_profile"),
    )


def node_payload_to_state(node: dict[str, Any], *, config: CaseConfig) -> State:
    if "log_n" in node and "log_Te" in node and "logA" in node:
        return State(log_n=float(node["log_n"]), log_Te=float(node["log_Te"]), logA=float(node["logA"]))
    return State(
        log_n=float(np.log(max(float(node["n_p"]), 1e-300))),
        log_Te=float(np.log(max(float(node["T_e"]), 1.0))),
        logA=float(np.log(max(float(node["A"]) / max(float(config.area_scale_m2), 1e-300), 1e-300))),
    )


def profile_arrays_from_nodes(nodes: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    return {
        "x": np.asarray([float(node["x"]) for node in nodes], dtype=float),
        "n_p": np.asarray([float(node["n_p"]) for node in nodes], dtype=float),
        "T_e": np.asarray([float(node["T_e"]) for node in nodes], dtype=float),
        "A": np.asarray([float(node["A"]) for node in nodes], dtype=float),
        "sigma_logA": np.asarray([float(node["sigma_logA"]) for node in nodes], dtype=float),
    }


def save_profile_npz(path: Path, arrays: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **{key: np.asarray(value, dtype=float) for key, value in arrays.items()})
