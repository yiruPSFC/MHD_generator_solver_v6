from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, replace
import sys
import time
from typing import Any

import numpy as np

from .constraints import evaluate_velikhov_node_constraints
from .design import DESIGN_VARIABLE_NAMES, CaseConfig, DesignVector
from .design_continuation import (
    config_with_design,
    solve_forward_with_design_continuation,
    solve_forward_with_persistent_solver_continuation,
)
from .forward import FiredrakeUnavailableError, _initial_delta_arrays, solve_forward


@dataclass
class ReducedFunctionalBundle:
    reduced_functional: Any
    controls: list[Any]
    variable_names: tuple[str, ...]
    constant_factory: Any
    initial_profile_updater: Any | None = None
    _initial_profile_values: list[Any] | None = None


def _variable_indices(variable_names: tuple[str, ...]) -> list[int]:
    return [DESIGN_VARIABLE_NAMES.index(name) for name in variable_names]


def _active_variable_names_from_width(width: np.ndarray, *, tol: float = 1e-20) -> tuple[str, ...]:
    active = tuple(
        name
        for name, value in zip(DESIGN_VARIABLE_NAMES, np.asarray(width, dtype=float), strict=True)
        if float(value) > float(tol)
    )
    return active if active else DESIGN_VARIABLE_NAMES


def _values_for_bundle(
    bundle: ReducedFunctionalBundle,
    values: np.ndarray | list[float] | tuple[float, ...],
) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size == len(bundle.variable_names):
        return arr
    if arr.size == len(DESIGN_VARIABLE_NAMES):
        return arr[_variable_indices(bundle.variable_names)]
    raise ValueError(
        f"expected {len(bundle.variable_names)} values for controls {bundle.variable_names} "
        f"or {len(DESIGN_VARIABLE_NAMES)} full design values; got {arr.size}."
    )


def control_values(bundle: ReducedFunctionalBundle, values: np.ndarray | list[float] | tuple[float, ...]) -> list[Any]:
    arr = _values_for_bundle(bundle, values)
    return [bundle.constant_factory(float(value)) for value in arr]


def evaluate_reduced_functional(bundle: ReducedFunctionalBundle, values: np.ndarray | list[float] | tuple[float, ...]) -> float:
    return float(bundle.reduced_functional(control_values(bundle, values)))


def update_reduced_functional_initial_profile(
    bundle: ReducedFunctionalBundle,
    *,
    design: DesignVector,
    profile: dict[str, Any],
) -> None:
    if bundle.initial_profile_updater is None:
        raise RuntimeError("ReducedFunctionalBundle was not built with an initial-profile placeholder.")
    bundle.initial_profile_updater(design=design, profile=profile)


def _scalar_from_control_like(value: Any) -> float:
    if hasattr(value, "dat"):
        return float(np.asarray(value.dat.data_ro, dtype=float).reshape(-1)[0])
    return float(value)


def reduced_functional_gradient(bundle: ReducedFunctionalBundle) -> np.ndarray:
    gradient = bundle.reduced_functional.derivative()
    return np.array([_scalar_from_control_like(item) for item in gradient], dtype=float)


def reduced_functional_gradient_full(bundle: ReducedFunctionalBundle) -> np.ndarray:
    gradient = reduced_functional_gradient(bundle)
    if len(bundle.variable_names) == len(DESIGN_VARIABLE_NAMES):
        return gradient
    full = np.zeros(len(DESIGN_VARIABLE_NAMES), dtype=float)
    for name, value in zip(bundle.variable_names, gradient, strict=True):
        full[DESIGN_VARIABLE_NAMES.index(name)] = float(value)
    return full


def _projected_gradient_residual_minimize(
    *,
    gradient_y_minimize: np.ndarray,
    y: np.ndarray,
    bound_tol: float = 1e-8,
) -> tuple[float, list[float]]:
    residuals = []
    for grad, value in zip(np.asarray(gradient_y_minimize, dtype=float), np.asarray(y, dtype=float), strict=True):
        if float(value) <= float(bound_tol):
            residual = max(0.0, -float(grad))
        elif float(value) >= 1.0 - float(bound_tol):
            residual = max(0.0, float(grad))
        else:
            residual = abs(float(grad))
        residuals.append(float(residual))
    return (
        float(np.max(residuals)) if residuals else 0.0,
        residuals,
    )


def build_reduced_functional(
    *,
    design: DesignVector,
    config: CaseConfig,
    initial_profile: dict[str, Any] | None = None,
    enable_initial_profile_placeholder: bool = False,
    variable_names: tuple[str, ...] = DESIGN_VARIABLE_NAMES,
) -> ReducedFunctionalBundle:
    if enable_initial_profile_placeholder and initial_profile is None:
        raise ValueError("initial-profile Placeholder reuse requires an explicit initial_profile.")
    variable_names = tuple(variable_names)
    unknown = [name for name in variable_names if name not in DESIGN_VARIABLE_NAMES]
    if unknown:
        raise ValueError(f"unknown reduced-functional control names: {unknown!r}")
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
    placeholder_info: dict[str, Any] | None = {} if enable_initial_profile_placeholder else None
    result = solve_forward(
        design=design,
        config=config,
        initial_profile=initial_profile,
        annotate_objective=True,
        initial_profile_placeholder=placeholder_info,
    )
    if not result.ok or result.fd_objective is None:
        raise RuntimeError(f"cannot build ReducedFunctional because forward solve failed: {result.error}")
    if result.fd_control_space is None:
        raise RuntimeError("cannot build ReducedFunctional without scalar Function controls.")
    controls = [Control(result.fd_controls[name]) for name in variable_names]

    def control_factory(value: float):
        with stop_annotating():
            control = fd.Function(result.fd_control_space, name="control_value")
            control.assign(float(value), annotate=False)
            return control

    initial_profile_values: list[Any] | None = None
    initial_profile_updater = None
    if placeholder_info is not None:
        if not {"placeholder", "space", "x_norm"}.issubset(placeholder_info):
            raise RuntimeError("initial-profile placeholder requested, but solve_forward did not expose it.")
        placeholder = placeholder_info["placeholder"]
        state_space = placeholder_info["space"]
        x_norm = np.asarray(placeholder_info["x_norm"], dtype=float).copy()
        initial_profile_values = []

        def initial_profile_updater(*, design: DesignVector, profile: dict[str, Any]) -> None:
            delta_log_n_values, delta_log_Te_values = _initial_delta_arrays(
                initial_profile=profile,
                design=design,
                target_x_norm=x_norm,
            )
            with stop_annotating():
                value = fd.Function(state_space, name="placeholder_initial_profile_delta")
                value.subfunctions[0].dat.data[:] = delta_log_n_values
                value.subfunctions[1].dat.data[:] = delta_log_Te_values
            placeholder.set_value(value)
            initial_profile_values.append(value)

    return ReducedFunctionalBundle(
        reduced_functional=ReducedFunctional(result.fd_objective, controls),
        controls=controls,
        variable_names=variable_names,
        constant_factory=control_factory,
        initial_profile_updater=initial_profile_updater,
        _initial_profile_values=initial_profile_values,
    )


def minimize_multistart(
    *,
    config: CaseConfig,
    multistart: int,
    seed: int = 1,
    max_iterations: int = 8,
    initial_profile: dict[str, Any] | None = None,
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
            bundle = build_reduced_functional(
                design=DesignVector.from_array(x),
                config=config,
                initial_profile=initial_profile,
            )
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
    initial_profile: dict[str, Any] | None = None,
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
    active_variable_names = _active_variable_names_from_width(width)
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
            bundle = build_reduced_functional(
                design=DesignVector.from_array(x0),
                config=config,
                initial_profile=initial_profile,
                variable_names=active_variable_names,
            )
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
                return -reduced_functional_gradient_full(bundle) * width
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
                    result = solve_forward(design=design, config=config, initial_profile=initial_profile)
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
        y_best = np.clip(np.asarray(result.x, dtype=float), 0.0, 1.0)
        x_best = raw_from_y(y_best)
        objective_replay_at_best = float("nan")
        objective_gradient_inf = float("nan")
        projected_gradient_inf = float("nan")
        projected_gradient_by_variable: list[float] = [float("nan")] * len(DESIGN_VARIABLE_NAMES)
        try:
            objective_replay_at_best = evaluate_reduced_functional(bundle, x_best)
            grad_y_minimize = -reduced_functional_gradient_full(bundle) * width
            objective_gradient_inf = float(np.linalg.norm(grad_y_minimize, ord=np.inf))
            projected_gradient_inf, projected_gradient_by_variable = _projected_gradient_residual_minimize(
                gradient_y_minimize=grad_y_minimize,
                y=y_best,
            )
        except Exception as exc:
            record_failure(
                source="post_slsqp_stationarity",
                start_index=idx,
                values=x_best,
                error=f"{type(exc).__name__}: {exc}",
            )
        margins_best = constraints(np.asarray(result.x, dtype=float))
        row = {
            "start_index": int(idx),
            "success": bool(result.success and np.isfinite(float(result.fun))),
            "message": str(result.message),
            "fun": float(result.fun),
            "objective_replay_at_best": float(objective_replay_at_best),
            "objective_replay_minus_optimizer_fun": float(objective_replay_at_best + float(result.fun)),
            "objective_gradient_inf_normalized": float(objective_gradient_inf),
            "projected_gradient_inf_normalized": float(projected_gradient_inf),
            "projected_gradient_by_variable_normalized": [
                float(value) for value in projected_gradient_by_variable
            ],
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
        "active_control_names": list(active_variable_names),
    }


def minimize_constrained_slsqp_trial_continuation(
    *,
    config: CaseConfig,
    multistart: int,
    seed: int = 1,
    max_iterations: int = 8,
    velikhov_hard_floor: float = 0.0,
    initial_profile: dict[str, Any] | None = None,
    continuation_initial_step_fraction: float = 1.0,
    continuation_min_step_fraction: float = 1e-6,
    continuation_max_step_fraction: float = 1.0,
    continuation_max_attempts: int = 50,
    continuation_T_p_floor_K: float = 1.0,
    reuse_reduced_functional_tape: bool = True,
) -> dict[str, Any]:
    """SLSQP with each trial solved from the nearest accepted profile.

    This is intentionally more expensive than the taped replay path: each new
    trial first does an untaped continuation solve, then builds a local adjoint
    tape at the converged state for value/gradient evaluation.
    """

    try:
        from scipy.optimize import minimize  # type: ignore
    except ImportError as exc:
        raise RuntimeError("scipy.optimize is required for constrained_slsqp.") from exc

    if str(config.metadata.get("velikhov_mode", "diagnostic")).lower() == "penalty":
        raise ValueError("constrained_slsqp requires velikhov_mode='diagnostic'; do not mix hard G constraints with the soft penalty.")
    if initial_profile is None:
        raise ValueError("trial-continuation SLSQP requires an initial_profile for the first Newton warm start.")

    lower = config.bounds.lower.as_array()
    upper = config.bounds.upper.as_array()
    width = np.maximum(upper - lower, 1e-30)
    active_variable_names = _active_variable_names_from_width(width)
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
    best_profile = None
    failure_keys: set[tuple[str, tuple[float, ...], str]] = set()
    continuation_summaries: list[dict[str, Any]] = []
    continuation_backend = "persistent_firedrake"
    continuation_backend_fallbacks: list[dict[str, Any]] = []
    reduced_functional_mode = "reused_placeholder_tape" if reuse_reduced_functional_tape else "rebuilt_tape_per_trial"
    reduced_functional_build_count = 0
    reduced_functional_timing_sums = {
        "build_s": 0.0,
        "placeholder_update_s": 0.0,
        "value_s": 0.0,
        "value_replay_for_gradient_s": 0.0,
        "gradient_s": 0.0,
    }

    def record_failure(
        *,
        source: str,
        start_index: int,
        values: np.ndarray,
        error: str,
        classification: dict[str, Any] | None = None,
    ) -> None:
        raw = np.asarray(values, dtype=float).reshape(len(DESIGN_VARIABLE_NAMES))
        key = (str(source), tuple(np.round(raw, 12)), str(error))
        if key in failure_keys:
            return
        failure_keys.add(key)
        row: dict[str, Any] = {
            "start_index": int(start_index),
            "iteration": None,
            "trial": None,
            "source": str(source),
            "x": [float(v) for v in raw],
            "error": str(error),
        }
        if classification is not None:
            row["classification"] = classification
        trial_failures.append(row)

    for idx, start in enumerate(starts):
        y0 = np.clip(to_normalized(np.asarray(start, dtype=float)), 0.0, 1.0)
        x0 = from_normalized(y0)
        start_design = DesignVector.from_array(x0)
        start_config = config_with_design(config, start_design)
        persistent_solver = None
        start_result = None
        try:
            from .persistent_forward import PersistentForwardSolver

            with _stop_annotating_context():
                persistent_solver = PersistentForwardSolver(config=start_config)
                start_result = persistent_solver.solve(
                    design=start_design,
                    initial_profile=initial_profile,
                )
        except ValueError as exc:
            continuation_backend = "functional_solve_forward_fallback"
            continuation_backend_fallbacks.append(
                {
                    "start_index": int(idx),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            with _stop_annotating_context():
                start_result = solve_forward(
                    design=start_design,
                    config=start_config,
                    initial_profile=initial_profile,
                )
        if not start_result.ok or start_result.profile is None or start_result.metrics is None:
            record_failure(
                source="initial_forward",
                start_index=idx,
                values=x0,
                error=start_result.error or "forward solve failed",
            )
            history.append(
                {
                    "start_index": int(idx),
                    "success": False,
                    "message": f"initial continuation anchor failed: {start_result.error}",
                    "fun": float("inf"),
                    "x": [float(v) for v in x0],
                    "min_constraint_margin": float("nan"),
                }
            )
            continue

        last_success_design = start_design
        last_success_profile = start_result.profile
        eval_cache: dict[tuple[float, ...], dict[str, Any]] = {}
        bundle = None
        if reuse_reduced_functional_tape:
            try:
                t_rf0 = time.perf_counter()
                bundle = build_reduced_functional(
                    design=start_design,
                    config=start_config,
                    initial_profile=last_success_profile,
                    enable_initial_profile_placeholder=True,
                    variable_names=active_variable_names,
                )
                reduced_functional_timing_sums["build_s"] += time.perf_counter() - t_rf0
                reduced_functional_build_count += 1
            except Exception as exc:
                record_failure(
                    source="initial_reused_reduced_functional",
                    start_index=idx,
                    values=x0,
                    error=f"{type(exc).__name__}: {exc}",
                )
                history.append(
                    {
                        "start_index": int(idx),
                        "success": False,
                        "message": f"initial reused ReducedFunctional build failed: {type(exc).__name__}: {exc}",
                        "fun": float("inf"),
                        "x": [float(v) for v in x0],
                        "min_constraint_margin": float("nan"),
                    }
                )
                continue
        active_rf_key: tuple[float, ...] | None = None

        def raw_from_y(y_values: np.ndarray) -> np.ndarray:
            return from_normalized(np.clip(np.asarray(y_values, dtype=float), 0.0, 1.0))

        def cache_key(values: np.ndarray) -> tuple[float, ...]:
            return tuple(np.round(np.asarray(values, dtype=float), 14))

        def set_placeholder_profile(*, design: DesignVector, profile: dict[str, Any]) -> None:
            if bundle is None:
                raise RuntimeError("placeholder update requested without a reusable ReducedFunctional bundle.")
            t0 = time.perf_counter()
            update_reduced_functional_initial_profile(bundle, design=design, profile=profile)
            reduced_functional_timing_sums["placeholder_update_s"] += time.perf_counter() - t0

        def replay_reduced_value(entry: dict[str, Any], *, timing_key: str = "value_s") -> float:
            nonlocal active_rf_key, reduced_functional_build_count
            key = cache_key(entry["x"])
            if bundle is None:
                t_build = time.perf_counter()
                local_bundle = build_reduced_functional(
                    design=entry["design"],
                    config=config_with_design(config, entry["design"]),
                    initial_profile=entry["profile"],
                    variable_names=active_variable_names,
                )
                reduced_functional_timing_sums["build_s"] += time.perf_counter() - t_build
                reduced_functional_build_count += 1
                entry["bundle"] = local_bundle
                value_bundle = local_bundle
            else:
                set_placeholder_profile(design=entry["design"], profile=entry["profile"])
                value_bundle = bundle
            t0 = time.perf_counter()
            value = evaluate_reduced_functional(value_bundle, entry["x"])
            reduced_functional_timing_sums[timing_key] += time.perf_counter() - t0
            active_rf_key = key
            entry["value"] = float(value)
            return float(value)

        def ensure_reduced_gradient(entry: dict[str, Any], *, source: str) -> np.ndarray | None:
            nonlocal active_rf_key
            if not entry.get("ok"):
                return None
            if entry.get("gradient") is not None:
                return np.asarray(entry["gradient"], dtype=float)
            key = cache_key(entry["x"])
            try:
                if bundle is None:
                    if entry.get("bundle") is None:
                        replay_reduced_value(entry, timing_key="value_replay_for_gradient_s")
                    gradient_bundle = entry["bundle"]
                else:
                    if active_rf_key != key:
                        replay_reduced_value(entry, timing_key="value_replay_for_gradient_s")
                    gradient_bundle = bundle
                t0 = time.perf_counter()
                gradient = reduced_functional_gradient_full(gradient_bundle)
                reduced_functional_timing_sums["gradient_s"] += time.perf_counter() - t0
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                record_failure(
                    source=f"{source}_reduced_gradient",
                    start_index=idx,
                    values=np.asarray(entry["x"], dtype=float),
                    error=error,
                )
                entry["ok"] = False
                entry["error"] = error
                return None
            entry["gradient"] = np.asarray(gradient, dtype=float)
            return np.asarray(entry["gradient"], dtype=float)

        def evaluate_trial(y_values: np.ndarray, *, source: str) -> dict[str, Any]:
            nonlocal last_success_design, last_success_profile, active_rf_key
            x = raw_from_y(y_values)
            key = cache_key(x)
            cached = eval_cache.get(key)
            if cached is not None:
                return cached
            design = DesignVector.from_array(x)
            with _stop_annotating_context():
                if persistent_solver is None:
                    continuation = solve_forward_with_design_continuation(
                        start_design=last_success_design,
                        target_design=design,
                        start_profile=last_success_profile,
                        config=config_with_design(config, last_success_design),
                        initial_step_fraction=float(continuation_initial_step_fraction),
                        min_step_fraction=float(continuation_min_step_fraction),
                        max_step_fraction=float(continuation_max_step_fraction),
                        max_attempts=int(continuation_max_attempts),
                        T_p_floor_K=float(continuation_T_p_floor_K),
                        enforce_T_p_floor=True,
                    )
                else:
                    continuation = solve_forward_with_persistent_solver_continuation(
                        solver=persistent_solver,
                        start_design=last_success_design,
                        target_design=design,
                        start_profile=last_success_profile,
                        config=config_with_design(config, last_success_design),
                        initial_step_fraction=float(continuation_initial_step_fraction),
                        min_step_fraction=float(continuation_min_step_fraction),
                        max_step_fraction=float(continuation_max_step_fraction),
                        max_attempts=int(continuation_max_attempts),
                        T_p_floor_K=float(continuation_T_p_floor_K),
                        enforce_T_p_floor=True,
                    )
            row_timings = [dict(row.get("solver_timing", {}) or {}) for row in continuation.get("rows", [])]
            continuation_summaries.append(
                {
                    "source": str(source),
                    "start_index": int(idx),
                    "backend": "functional" if persistent_solver is None else "persistent_firedrake",
                    "ok": bool(continuation.get("ok", False)),
                    "reached_alpha": float(continuation.get("reached_alpha", 0.0)),
                    "attempt_count": int(continuation.get("attempt_count", 0)),
                    "target_design": design.to_dict(),
                    "final_classification": continuation.get("final_classification"),
                    "solver_timing_sums": {
                        name: float(sum(float(timing.get(name, 0.0)) for timing in row_timings))
                        for name in ("assign_s", "solve_s", "postprocess_s", "total_s")
                    },
                }
            )
            if not bool(continuation.get("ok", False)):
                classification = dict(continuation.get("final_classification", {}) or {})
                error = str(continuation.get("error") or classification.get("reason") or "continuation failed")
                record_failure(
                    source=f"{source}_continuation",
                    start_index=idx,
                    values=x,
                    error=error,
                    classification=classification,
                )
                entry = {
                    "ok": False,
                    "x": x,
                    "error": error,
                    "classification": classification,
                    "continuation": continuation,
                }
                eval_cache[key] = entry
                return entry

            profile = continuation.get("_accepted_profile")
            if profile is None:
                error = "continuation succeeded without accepted profile"
                record_failure(source=f"{source}_continuation_profile", start_index=idx, values=x, error=error)
                entry = {"ok": False, "x": x, "error": error, "continuation": continuation}
                eval_cache[key] = entry
                return entry
            try:
                entry_for_replay = {
                    "x": x,
                    "design": design,
                    "profile": profile,
                }
                value = replay_reduced_value(entry_for_replay)
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                record_failure(source=f"{source}_reduced_functional", start_index=idx, values=x, error=error)
                entry = {"ok": False, "x": x, "error": error, "continuation": continuation}
                eval_cache[key] = entry
                return entry

            last_success_design = design
            last_success_profile = profile
            entry = {
                "ok": True,
                "x": x,
                "design": design,
                "profile": profile,
                "value": float(value),
                "gradient": None,
                "margins": None,
                "continuation": continuation,
            }
            if "bundle" in entry_for_replay:
                entry["bundle"] = entry_for_replay["bundle"]
            eval_cache[key] = entry
            return entry

        def objective(y_values: np.ndarray) -> float:
            entry = evaluate_trial(y_values, source="objective")
            if not entry.get("ok"):
                return 1.0e100
            return float(-entry["value"])

        def objective_jac(y_values: np.ndarray) -> np.ndarray:
            entry = evaluate_trial(y_values, source="objective_jac")
            if not entry.get("ok"):
                return np.zeros_like(y0)
            gradient = ensure_reduced_gradient(entry, source="objective_jac")
            if gradient is None:
                return np.zeros_like(y0)
            return -np.asarray(gradient, dtype=float) * width

        def constraints(y_values: np.ndarray) -> np.ndarray:
            entry = evaluate_trial(y_values, source="constraint")
            if not entry.get("ok"):
                return np.full(constraint_count, -1.0e30, dtype=float)
            if entry.get("margins") is None:
                design = entry["design"]
                summary = evaluate_velikhov_node_constraints(
                    profile=entry["profile"],
                    design=design,
                    config=config_with_design(config, design),
                    floor=hard_floor,
                )
                entry["margins"] = np.asarray(summary.margins, dtype=float).reshape(constraint_count)
            return np.asarray(entry["margins"], dtype=float).copy()

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
        y_best = np.clip(np.asarray(result.x, dtype=float), 0.0, 1.0)
        x_best = raw_from_y(y_best)
        entry_best = evaluate_trial(y_best, source="post_slsqp")
        objective_replay_at_best = float(entry_best["value"]) if entry_best.get("ok") else float("nan")
        objective_gradient_inf = float("nan")
        projected_gradient_inf = float("nan")
        projected_gradient_by_variable: list[float] = [float("nan")] * len(DESIGN_VARIABLE_NAMES)
        if entry_best.get("ok"):
            gradient_best = ensure_reduced_gradient(entry_best, source="post_slsqp")
            grad_y_minimize = (
                -np.asarray(gradient_best, dtype=float) * width
                if gradient_best is not None
                else np.full_like(y0, float("nan"), dtype=float)
            )
            objective_gradient_inf = float(np.linalg.norm(grad_y_minimize, ord=np.inf))
            projected_gradient_inf, projected_gradient_by_variable = _projected_gradient_residual_minimize(
                gradient_y_minimize=grad_y_minimize,
                y=y_best,
            )
        margins_best = constraints(y_best)
        row = {
            "start_index": int(idx),
            "success": bool(result.success and np.isfinite(float(result.fun))),
            "message": str(result.message),
            "fun": float(result.fun),
            "objective_replay_at_best": float(objective_replay_at_best),
            "objective_replay_minus_optimizer_fun": float(objective_replay_at_best + float(result.fun)),
            "objective_gradient_inf_normalized": float(objective_gradient_inf),
            "projected_gradient_inf_normalized": float(projected_gradient_inf),
            "projected_gradient_by_variable_normalized": [
                float(value) for value in projected_gradient_by_variable
            ],
            "x": [float(v) for v in x_best],
            "nit": int(getattr(result, "nit", 0)),
            "min_constraint_margin": float(np.nanmin(margins_best)),
            "trial_continuation_eval_count": int(len(continuation_summaries)),
        }
        history.append(row)
        if np.isfinite(row["fun"]) and (best is None or row["fun"] < best["fun"]):
            best = row
            if entry_best.get("ok"):
                best_profile = entry_best["profile"]

    return {
        "best": best,
        "best_profile": best_profile,
        "history": history,
        "trial_failures": trial_failures,
        "method": "constrained_slsqp_node_velikhov_trial_continuation",
        "max_iterations": int(max_iterations),
        "velikhov_hard_floor": hard_floor,
        "constraint_sampling": "nodes",
        "trial_continuation": {
            "backend": continuation_backend,
            "backend_fallbacks": continuation_backend_fallbacks,
            "active_control_names": list(active_variable_names),
            "reduced_functional_mode": reduced_functional_mode,
            "reduced_functional_build_count": int(reduced_functional_build_count),
            "reduced_functional_timing_sums": {
                key: float(value) for key, value in reduced_functional_timing_sums.items()
            },
            "initial_step_fraction": float(continuation_initial_step_fraction),
            "min_step_fraction": float(continuation_min_step_fraction),
            "max_step_fraction": float(continuation_max_step_fraction),
            "max_attempts": int(continuation_max_attempts),
            "T_p_floor_K": float(continuation_T_p_floor_K),
            "summary_count": int(len(continuation_summaries)),
            "summaries": continuation_summaries,
        },
    }


def minimize_constrained_slsqp_state_restarts(
    *,
    config: CaseConfig,
    multistart: int,
    seed: int = 1,
    max_iterations: int = 8,
    velikhov_hard_floor: float = 0.0,
    initial_profile: dict[str, Any] | None = None,
    state_restarts: int = 1,
    improvement_tol: float = 1e-7,
) -> dict[str, Any]:
    """Run SLSQP in short windows, refreshing the nonlinear state initial guess between windows."""

    restart_limit = max(int(state_restarts), 1)
    current_config = config
    current_profile = initial_profile
    previous_score: float | None = None
    best = None
    best_profile = None
    history: list[dict[str, Any]] = []
    trial_failures: list[dict[str, Any]] = []
    restart_summaries: list[dict[str, Any]] = []

    for restart_index in range(restart_limit):
        opt = minimize_constrained_slsqp(
            config=current_config,
            multistart=int(multistart) if restart_index == 0 else 1,
            seed=int(seed) + restart_index,
            max_iterations=int(max_iterations),
            velikhov_hard_floor=float(velikhov_hard_floor),
            initial_profile=current_profile,
        )
        for row in opt.get("history", []):
            history.append({**row, "state_restart_index": int(restart_index)})
        for failure in opt.get("trial_failures", []):
            trial_failures.append({**failure, "state_restart_index": int(restart_index)})

        row = opt.get("best")
        if row is None:
            restart_summaries.append(
                {
                    "state_restart_index": int(restart_index),
                    "ok": False,
                    "error": "inner SLSQP produced no finite best point",
                    "trial_failure_count": int(len(opt.get("trial_failures", []))),
                }
            )
            break

        design = DesignVector.from_array(row["x"])
        eval_config = replace(current_config, design=design, B_T=float(design.B_T))
        with _stop_annotating_context():
            result = solve_forward(
                design=design,
                config=eval_config,
                initial_profile=current_profile,
            )
        if not result.ok or result.profile is None or result.metrics is None:
            trial_failures.append(
                {
                    "start_index": int(row.get("start_index", 0)),
                    "iteration": None,
                    "trial": None,
                    "source": "state_restart_final_evaluate",
                    "state_restart_index": int(restart_index),
                    "x": [float(v) for v in design.as_array()],
                    "error": result.error or "forward solve failed",
                }
            )
            restart_summaries.append(
                {
                    "state_restart_index": int(restart_index),
                    "ok": False,
                    "error": result.error or "forward solve failed",
                    "trial_failure_count": int(len(opt.get("trial_failures", []))),
                }
            )
            break

        score = float(result.metrics.objective_score)
        evaluated_row = {
            **row,
            "fun": -score,
            "optimizer_fun": float(row["fun"]),
            "evaluated_objective_score": score,
            "taped_objective_minus_postprocess_objective": float(-float(row["fun"]) - score),
            "state_restart_index": int(restart_index),
        }
        restart_summaries.append(
            {
                "state_restart_index": int(restart_index),
                "ok": True,
                "optimizer_fun": float(row["fun"]),
                "evaluated_objective_score": score,
                "taped_objective_minus_postprocess_objective": float(-float(row["fun"]) - score),
                "mhd_output_power_W": float(result.metrics.mhd_output_power_W),
                "raw_enthalpy_extraction_percent": float(result.metrics.raw_enthalpy_extraction_percent),
                "min_velikhov_margin": float(result.metrics.min_velikhov_margin),
                "trial_failure_count": int(len(opt.get("trial_failures", []))),
                "x": [float(v) for v in design.as_array()],
            }
        )

        if best is None or float(evaluated_row["fun"]) < float(best["fun"]):
            best = evaluated_row
            best_profile = result.profile

        if previous_score is not None and abs(score - previous_score) <= float(improvement_tol):
            current_profile = result.profile
            current_config = eval_config
            break
        previous_score = score
        current_profile = result.profile
        current_config = eval_config

    return {
        "best": best,
        "best_profile": best_profile,
        "history": history,
        "trial_failures": trial_failures,
        "method": "constrained_slsqp_node_velikhov_state_restarts",
        "inner_method": "constrained_slsqp_node_velikhov",
        "max_iterations": int(max_iterations),
        "velikhov_hard_floor": float(velikhov_hard_floor),
        "constraint_sampling": "nodes",
        "state_restarts_requested": restart_limit,
        "state_restarts_completed": int(len(restart_summaries)),
        "state_restart_improvement_tol": float(improvement_tol),
        "state_restart_summaries": restart_summaries,
    }
