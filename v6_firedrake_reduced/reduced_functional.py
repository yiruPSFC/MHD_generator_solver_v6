from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import sys
from typing import Any

import numpy as np

from .constraints import evaluate_velikhov_node_constraints
from .design import DESIGN_VARIABLE_NAMES, CaseConfig, DesignVector
from .forward import FiredrakeUnavailableError, solve_forward


@dataclass
class ReducedFunctionalBundle:
    reduced_functional: Any
    controls: list[Any]
    variable_names: tuple[str, ...]
    constant_factory: Any


def control_values(bundle: ReducedFunctionalBundle, values: np.ndarray | list[float] | tuple[float, ...]) -> list[Any]:
    arr = np.asarray(values, dtype=float).reshape(len(bundle.variable_names))
    return [bundle.constant_factory(float(value)) for value in arr]


def evaluate_reduced_functional(bundle: ReducedFunctionalBundle, values: np.ndarray | list[float] | tuple[float, ...]) -> float:
    return float(bundle.reduced_functional(control_values(bundle, values)))


def _scalar_from_control_like(value: Any) -> float:
    if hasattr(value, "dat"):
        return float(np.asarray(value.dat.data_ro, dtype=float).reshape(-1)[0])
    return float(value)


def reduced_functional_gradient(bundle: ReducedFunctionalBundle) -> np.ndarray:
    gradient = bundle.reduced_functional.derivative()
    return np.array([_scalar_from_control_like(item) for item in gradient], dtype=float)


def build_reduced_functional(*, design: DesignVector, config: CaseConfig) -> ReducedFunctionalBundle:
    original_argv = sys.argv[:]
    try:
        sys.argv = ["v6_firedrake_reduced"]
        import firedrake as fd  # type: ignore
        from firedrake.adjoint import Control, ReducedFunctional  # type: ignore
        from pyadjoint import Tape, continue_annotation, set_working_tape, stop_annotating  # type: ignore
    except ImportError as exc:
        raise FiredrakeUnavailableError(
            "firedrake.adjoint is unavailable. Install Firedrake/pyadjoint in .venv_firedrake first."
        ) from exc
    finally:
        sys.argv = original_argv

    set_working_tape(Tape())
    continue_annotation()
    result = solve_forward(design=design, config=config, annotate_objective=True)
    if not result.ok or result.fd_objective is None:
        raise RuntimeError(f"cannot build ReducedFunctional because forward solve failed: {result.error}")
    if result.fd_control_space is None:
        raise RuntimeError("cannot build ReducedFunctional without scalar Function controls.")
    controls = [Control(result.fd_controls[name]) for name in DESIGN_VARIABLE_NAMES]

    def control_factory(value: float):
        with stop_annotating():
            control = fd.Function(result.fd_control_space, name="control_value")
            control.assign(float(value), annotate=False)
            return control

    return ReducedFunctionalBundle(
        reduced_functional=ReducedFunctional(result.fd_objective, controls),
        controls=controls,
        variable_names=DESIGN_VARIABLE_NAMES,
        constant_factory=control_factory,
    )


def minimize_multistart(
    *,
    config: CaseConfig,
    multistart: int,
    seed: int = 1,
    max_iterations: int = 8,
) -> dict[str, Any]:
    """Small local optimizer wrapper for the first Firedrake smoke experiments."""

    lower = config.bounds.lower.as_array()
    upper = config.bounds.upper.as_array()
    width = np.maximum(upper - lower, 1e-30)

    def to_normalized(values: np.ndarray) -> np.ndarray:
        return (np.asarray(values, dtype=float) - lower) / width

    def from_normalized(values: np.ndarray) -> np.ndarray:
        return lower + np.asarray(values, dtype=float) * width

    rng = np.random.default_rng(int(seed))
    starts = [config.design.as_array()]
    for _ in range(max(int(multistart) - 1, 0)):
        starts.append(lower + rng.random(lower.shape) * width)

    history = []
    trial_failures = []
    best = None
    for idx, start in enumerate(starts):
        y = np.clip(to_normalized(np.asarray(start, dtype=float)), 0.0, 1.0)
        x = from_normalized(y)
        try:
            bundle = build_reduced_functional(design=DesignVector.from_array(x), config=config)
        except Exception as exc:
            trial_failures.append(
                {
                    "start_index": int(idx),
                    "iteration": 0,
                    "trial": 0,
                    "x": [float(v) for v in x],
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            history.append(
                {
                    "start_index": int(idx),
                    "success": False,
                    "message": f"initial reduced-functional build failed: {type(exc).__name__}: {exc}",
                    "fun": float("inf"),
                    "x": [float(v) for v in x],
                }
            )
            continue

        try:
            value = evaluate_reduced_functional(bundle, x)
        except Exception as exc:
            trial_failures.append(
                {
                    "start_index": int(idx),
                    "iteration": 0,
                    "trial": 0,
                    "x": [float(v) for v in x],
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            history.append(
                {
                    "start_index": int(idx),
                    "success": False,
                    "message": f"initial forward solve failed: {type(exc).__name__}: {exc}",
                    "fun": float("inf"),
                    "x": [float(v) for v in x],
                }
            )
            continue

        accepted_steps = 0
        message = "finite initial point"
        for iteration in range(int(max_iterations)):
            gradient_x = reduced_functional_gradient(bundle)
            gradient_y = gradient_x * width
            grad_norm = float(np.linalg.norm(gradient_y, ord=np.inf))
            if not np.isfinite(grad_norm) or grad_norm <= 1e-10:
                message = "projected gradient below tolerance"
                break

            step = 0.15 * gradient_y / grad_norm
            accepted = False
            alpha = 1.0
            for trial in range(10):
                y_trial = np.clip(y + alpha * step, 0.0, 1.0)
                if np.array_equal(y_trial, y):
                    alpha *= 0.5
                    continue
                x_trial = from_normalized(y_trial)
                try:
                    value_trial = evaluate_reduced_functional(bundle, x_trial)
                except Exception as exc:
                    trial_failures.append(
                        {
                            "start_index": int(idx),
                            "iteration": int(iteration),
                            "trial": int(trial),
                            "x": [float(v) for v in x_trial],
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
                    alpha *= 0.5
                    continue
                if float(value_trial) > float(value):
                    y = y_trial
                    x = x_trial
                    value = float(value_trial)
                    accepted_steps += 1
                    accepted = True
                    message = f"accepted {accepted_steps} projected-gradient steps"
                    break
                alpha *= 0.5
            if not accepted:
                message = f"line search stopped after {accepted_steps} accepted steps"
                break

        row = {
            "start_index": int(idx),
            "success": bool(np.isfinite(value)),
            "message": message,
            "fun": float(-value),
            "x": [float(v) for v in x],
        }
        history.append(row)
        if best is None or row["fun"] < best["fun"]:
            best = row
    return {
        "best": best,
        "history": history,
        "trial_failures": trial_failures,
        "method": "projected_gradient_backtracking",
        "max_iterations": int(max_iterations),
    }


def _stop_annotating_context():
    try:
        from pyadjoint import stop_annotating  # type: ignore
    except ImportError:
        return nullcontext()
    return stop_annotating()


def minimize_constrained_slsqp(
    *,
    config: CaseConfig,
    multistart: int,
    seed: int = 1,
    max_iterations: int = 8,
    velikhov_hard_floor: float = 0.0,
) -> dict[str, Any]:
    """Local SLSQP optimizer with node-only reduced constraints G_node - floor >= 0."""

    try:
        from scipy.optimize import minimize  # type: ignore
    except ImportError as exc:
        raise RuntimeError("scipy.optimize is required for constrained_slsqp.") from exc

    if str(config.metadata.get("velikhov_mode", "diagnostic")).lower() == "penalty":
        raise ValueError("constrained_slsqp requires velikhov_mode='diagnostic'; do not mix hard G constraints with the soft penalty.")

    lower = config.bounds.lower.as_array()
    upper = config.bounds.upper.as_array()
    width = np.maximum(upper - lower, 1e-30)
    constraint_count = int(config.n_intervals) + 1
    hard_floor = float(velikhov_hard_floor)

    def to_normalized(values: np.ndarray) -> np.ndarray:
        return (np.asarray(values, dtype=float) - lower) / width

    def from_normalized(values: np.ndarray) -> np.ndarray:
        return lower + np.asarray(values, dtype=float) * width

    rng = np.random.default_rng(int(seed))
    starts = [config.design.as_array()]
    for _ in range(max(int(multistart) - 1, 0)):
        starts.append(lower + rng.random(lower.shape) * width)

    history: list[dict[str, Any]] = []
    trial_failures: list[dict[str, Any]] = []
    best = None
    failure_keys: set[tuple[str, tuple[float, ...], str]] = set()

    def record_failure(*, source: str, start_index: int, values: np.ndarray, error: str) -> None:
        raw = np.asarray(values, dtype=float).reshape(len(DESIGN_VARIABLE_NAMES))
        key = (str(source), tuple(np.round(raw, 12)), str(error))
        if key in failure_keys:
            return
        failure_keys.add(key)
        trial_failures.append(
            {
                "start_index": int(start_index),
                "iteration": None,
                "trial": None,
                "source": str(source),
                "x": [float(v) for v in raw],
                "error": str(error),
            }
        )

    for idx, start in enumerate(starts):
        y0 = np.clip(to_normalized(np.asarray(start, dtype=float)), 0.0, 1.0)
        x0 = from_normalized(y0)
        try:
            bundle = build_reduced_functional(design=DesignVector.from_array(x0), config=config)
        except Exception as exc:
            record_failure(
                source="initial_reduced_functional",
                start_index=idx,
                values=x0,
                error=f"{type(exc).__name__}: {exc}",
            )
            history.append(
                {
                    "start_index": int(idx),
                    "success": False,
                    "message": f"initial reduced-functional build failed: {type(exc).__name__}: {exc}",
                    "fun": float("inf"),
                    "x": [float(v) for v in x0],
                    "min_constraint_margin": float("nan"),
                }
            )
            continue

        constraint_cache: dict[tuple[float, ...], np.ndarray] = {}

        def raw_from_y(y_values: np.ndarray) -> np.ndarray:
            return from_normalized(np.clip(np.asarray(y_values, dtype=float), 0.0, 1.0))

        def objective(y_values: np.ndarray) -> float:
            x = raw_from_y(y_values)
            try:
                return float(-evaluate_reduced_functional(bundle, x))
            except Exception as exc:
                record_failure(
                    source="objective",
                    start_index=idx,
                    values=x,
                    error=f"{type(exc).__name__}: {exc}",
                )
                return 1.0e100

        def objective_jac(y_values: np.ndarray) -> np.ndarray:
            x = raw_from_y(y_values)
            try:
                _ = evaluate_reduced_functional(bundle, x)
                return -reduced_functional_gradient(bundle) * width
            except Exception as exc:
                record_failure(
                    source="objective_jac",
                    start_index=idx,
                    values=x,
                    error=f"{type(exc).__name__}: {exc}",
                )
                return np.zeros_like(y0)

        def constraints(y_values: np.ndarray) -> np.ndarray:
            x = raw_from_y(y_values)
            key = tuple(np.asarray(x, dtype=float))
            cached = constraint_cache.get(key)
            if cached is not None:
                return cached.copy()
            design = DesignVector.from_array(x)
            try:
                with _stop_annotating_context():
                    result = solve_forward(design=design, config=config)
                if not result.ok or result.profile is None:
                    raise RuntimeError(result.error or "forward solve failed")
                summary = evaluate_velikhov_node_constraints(
                    profile=result.profile,
                    design=design,
                    config=config,
                    floor=hard_floor,
                )
                margins = np.asarray(summary.margins, dtype=float).reshape(constraint_count)
            except Exception as exc:
                record_failure(
                    source="constraint",
                    start_index=idx,
                    values=x,
                    error=f"{type(exc).__name__}: {exc}",
                )
                margins = np.full(constraint_count, -1.0e30, dtype=float)
            constraint_cache[key] = margins
            return margins.copy()

        result = minimize(
            objective,
            y0,
            jac=objective_jac,
            method="SLSQP",
            bounds=[(0.0, 1.0)] * y0.size,
            constraints=({"type": "ineq", "fun": constraints},),
            options={
                "maxiter": int(max_iterations),
                "ftol": 1e-7,
                "disp": False,
            },
        )
        x_best = raw_from_y(np.asarray(result.x, dtype=float))
        margins_best = constraints(np.asarray(result.x, dtype=float))
        row = {
            "start_index": int(idx),
            "success": bool(result.success and np.isfinite(float(result.fun))),
            "message": str(result.message),
            "fun": float(result.fun),
            "x": [float(v) for v in x_best],
            "nit": int(getattr(result, "nit", 0)),
            "min_constraint_margin": float(np.nanmin(margins_best)),
        }
        history.append(row)
        if np.isfinite(row["fun"]) and (best is None or row["fun"] < best["fun"]):
            best = row

    return {
        "best": best,
        "history": history,
        "trial_failures": trial_failures,
        "method": "constrained_slsqp_node_velikhov",
        "max_iterations": int(max_iterations),
        "velikhov_hard_floor": hard_floor,
        "constraint_sampling": "nodes",
    }
