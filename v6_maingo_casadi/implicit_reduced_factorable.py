from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .numerics import _ops_for_maingo, _ops_for_numeric


def _max_op(ops, a, b):
    return ops.max(a, b)


def _min_op(ops, a, b):
    return ops.min(a, b)


def _clip_unit(ops, value):
    return _min_op(ops, 1.0, _max_op(ops, 0.0, value))


def _safe_positive_denom(ops, value, floor: float):
    shifted = value + float(floor)
    if getattr(ops, "lb_func", None) is not None:
        return ops.lb_func(shifted, float(floor))
    return _max_op(ops, shifted, float(floor))


def _reduce_max(ops, values: list[Any]):
    if not values:
        raise ValueError("cannot reduce an empty list.")
    acc = values[0]
    for value in values[1:]:
        acc = _max_op(ops, acc, value)
    return acc


def _reduce_min(ops, values: list[Any]):
    if not values:
        raise ValueError("cannot reduce an empty list.")
    acc = values[0]
    for value in values[1:]:
        acc = _min_op(ops, acc, value)
    return acc


def _model_function(maingopy_module, items):
    if getattr(maingopy_module, "ModelFunction", None) is None:
        return list(items)
    model_function = maingopy_module.ModelFunction()
    for item in items:
        model_function.push_back(item)
    return model_function


@dataclass(frozen=True)
class OptionalChokeGate:
    inside_gate: Any
    near_sonic_gate: Any
    sign_change_gate: Any
    choke_gate: Any
    min_abs_det: Any
    max_crossing_score: Any
    gated_location_residual: Any
    gated_segment_residuals: list[Any]


@dataclass(frozen=True)
class ToyFixedNewtonRollout:
    x_nodes: list[float]
    y_nodes: list[Any]
    det_nodes: list[Any]
    step_residuals: list[Any]
    max_abs_step_residual: Any
    optional_choke: OptionalChokeGate

    @property
    def final_y(self):
        return self.y_nodes[-1]


def _toy_step_residual(*, y_left, y_right, x_left: float, x_right: float, forcing_center: float):
    """Residual form of (y - 1) dy/dx = x - forcing_center.

    The explicit ODE divides by y - 1 and is singular at the sonic-like point.
    This midpoint residual keeps the factorable equation finite at y = 1.
    """
    dx = float(x_right - x_left)
    x_mid = 0.5 * (float(x_left) + float(x_right))
    y_mid = 0.5 * (y_left + y_right)
    return (y_mid - 1.0) * (y_right - y_left) - dx * (x_mid - float(forcing_center))


def _toy_step_residual_dy_right(*, y_right):
    return y_right - 1.0


def fixed_newton_scalar(
    *,
    ops,
    initial,
    residual_fn,
    jacobian_fn,
    steps: int = 10,
    regularization: float = 1e-8,
):
    """Unroll a fixed number of damped Newton/Gauss-Newton updates.

    This is intentionally factorable: no convergence-dependent while-loop and
    no Python branch depending on optimization variables.
    """
    value = initial
    for _ in range(int(steps)):
        residual = residual_fn(value)
        jacobian = jacobian_fn(value)
        denom = _safe_positive_denom(ops, jacobian * jacobian, float(regularization))
        value = value - residual * jacobian / denom
    return value


def optional_choke_gate(
    *,
    ops,
    x_nodes: list[float],
    det_nodes: list[Any],
    x_choke=None,
    length: float = 1.0,
    det_activation_width: float = 5e-2,
    location_padding: float = 1e-2,
) -> OptionalChokeGate:
    """Build a factorable optional-choke activation gate.

    Choke activation is driven by the trajectory, not by the candidate choke
    location. This prevents the optimizer from moving x_choke outside the
    domain simply to disable critical-point residuals. If x_choke is None,
    only trajectory diagnostics are computed and no critical residual is
    imposed.
    """
    if len(x_nodes) != len(det_nodes):
        raise ValueError("x_nodes and det_nodes must have matching lengths.")
    if len(x_nodes) < 2:
        raise ValueError("at least two nodes are required.")

    abs_det_nodes = [ops.fabs(det) for det in det_nodes]
    min_abs_det = _reduce_min(ops, abs_det_nodes)
    near_sonic_gate = _clip_unit(ops, 1.0 - min_abs_det / float(det_activation_width))

    crossing_scores = []
    det_scale_sq = float(det_activation_width) * float(det_activation_width)
    for x_left, x_right, det_left, det_right in zip(
        x_nodes[:-1],
        x_nodes[1:],
        det_nodes[:-1],
        det_nodes[1:],
        strict=True,
    ):
        crossing_score = _clip_unit(ops, -(det_left * det_right) / det_scale_sq)
        crossing_scores.append(crossing_score)

    max_crossing_score = _reduce_max(ops, crossing_scores)
    sign_change_gate = _clip_unit(ops, max_crossing_score)
    choke_gate = _max_op(ops, near_sonic_gate, sign_change_gate)

    if x_choke is None:
        inside_gate = 0.0
        gated_location_residual = 0.0
        gated_segment_residuals = [0.0 for _ in crossing_scores]
    else:
        outside_left = _max_op(ops, -x_choke, 0.0)
        outside_right = _max_op(ops, x_choke - float(length), 0.0)
        outside_distance = _max_op(ops, outside_left, outside_right)
        inside_gate = _clip_unit(ops, 1.0 - outside_distance / float(location_padding))
        gated_location_residual = choke_gate * outside_distance
        gated_segment_residuals = []
        for x_left, x_right, det_left, det_right, crossing_score in zip(
            x_nodes[:-1],
            x_nodes[1:],
            det_nodes[:-1],
            det_nodes[1:],
            crossing_scores,
            strict=True,
        ):
            # Linear-interpolation consistency for the candidate critical location.
            segment_residual = det_left * (float(x_right) - x_choke) + det_right * (x_choke - float(x_left))
            gated_segment_residuals.append(crossing_score * segment_residual)
    return OptionalChokeGate(
        inside_gate=inside_gate,
        near_sonic_gate=near_sonic_gate,
        sign_change_gate=sign_change_gate,
        choke_gate=choke_gate,
        min_abs_det=min_abs_det,
        max_crossing_score=max_crossing_score,
        gated_location_residual=gated_location_residual,
        gated_segment_residuals=gated_segment_residuals,
    )


def rollout_toy_fixed_newton(
    *,
    ops=None,
    y0,
    forcing_center: float = 0.5,
    x_choke=None,
    n_intervals: int = 4,
    length: float = 1.0,
    newton_steps: int = 10,
    newton_regularization: float = 1e-8,
    predictor_slope: float = 0.0,
    det_activation_width: float = 5e-2,
) -> ToyFixedNewtonRollout:
    """Roll out a sonic-neighborhood toy problem with fixed Newton correctors."""
    ops = _ops_for_numeric() if ops is None else ops
    n_intervals = int(n_intervals)
    if n_intervals <= 0:
        raise ValueError("n_intervals must be positive.")
    length = float(length)
    dx = length / n_intervals
    x_nodes = [float(idx * dx) for idx in range(n_intervals + 1)]
    y_nodes: list[Any] = [y0]
    step_residuals: list[Any] = []

    for idx in range(n_intervals):
        x_left = x_nodes[idx]
        x_right = x_nodes[idx + 1]
        y_left = y_nodes[-1]
        initial = y_left + float(predictor_slope) * dx

        def residual_fn(y_right, *, _y_left=y_left, _x_left=x_left, _x_right=x_right):
            return _toy_step_residual(
                y_left=_y_left,
                y_right=y_right,
                x_left=_x_left,
                x_right=_x_right,
                forcing_center=float(forcing_center),
            )

        def jacobian_fn(y_right):
            return _toy_step_residual_dy_right(y_right=y_right)

        y_right = fixed_newton_scalar(
            ops=ops,
            initial=initial,
            residual_fn=residual_fn,
            jacobian_fn=jacobian_fn,
            steps=int(newton_steps),
            regularization=float(newton_regularization),
        )
        y_nodes.append(y_right)
        step_residuals.append(residual_fn(y_right))

    det_nodes = [node - 1.0 for node in y_nodes]
    abs_step_residuals = [ops.fabs(residual) for residual in step_residuals]
    max_abs_step_residual = _reduce_max(ops, abs_step_residuals)
    choke = optional_choke_gate(
        ops=ops,
        x_nodes=x_nodes,
        det_nodes=det_nodes,
        x_choke=x_choke,
        length=length,
        det_activation_width=float(det_activation_width),
    )
    return ToyFixedNewtonRollout(
        x_nodes=x_nodes,
        y_nodes=y_nodes,
        det_nodes=det_nodes,
        step_residuals=step_residuals,
        max_abs_step_residual=max_abs_step_residual,
        optional_choke=choke,
    )


class ToyFixedNewtonImplicitReducedModelBase:
    """Small MAiNGO-compatible prototype for factorable implicit reduced solve."""

    def __init__(
        self,
        *,
        maingopy_module,
        n_intervals: int = 4,
        newton_steps: int = 10,
        target_final_y: float = 1.5,
        forcing_center: float = 0.5,
        critical_mode: bool = False,
    ):
        self._maingopy = maingopy_module
        self._ops = _ops_for_numeric() if maingopy_module is None else _ops_for_maingo(maingopy_module)
        self._n_intervals = int(n_intervals)
        self._newton_steps = int(newton_steps)
        self._target_final_y = float(target_final_y)
        self._forcing_center = float(forcing_center)
        self._critical_mode = bool(critical_mode)

    def get_variables(self):
        variables = [
            self._maingopy.OptimizationVariable(
                self._maingopy.Bounds(0.2, 1.8),
                self._maingopy.VT_CONTINUOUS,
                "toy_y0",
            ),
        ]
        if self._critical_mode:
            variables.append(
                self._maingopy.OptimizationVariable(
                    self._maingopy.Bounds(-0.5, 1.5),
                    self._maingopy.VT_CONTINUOUS,
                    "toy_x_choke_optional",
                )
            )
        return variables

    def get_initial_point(self):
        initial = [0.5]
        if self._critical_mode:
            initial.append(0.5)
        return initial

    def evaluate(self, vars):
        y0 = vars[0]
        x_choke = vars[1] if self._critical_mode else None
        rollout = rollout_toy_fixed_newton(
            ops=self._ops,
            y0=y0,
            forcing_center=self._forcing_center,
            x_choke=x_choke,
            n_intervals=self._n_intervals,
            newton_steps=self._newton_steps,
        )
        critical_penalty = (
            rollout.optional_choke.gated_location_residual * rollout.optional_choke.gated_location_residual
        )
        for residual in rollout.optional_choke.gated_segment_residuals:
            critical_penalty = critical_penalty + residual * residual
        step_residual_penalty = rollout.max_abs_step_residual * rollout.max_abs_step_residual
        final_target_penalty = (rollout.final_y - self._target_final_y) * (rollout.final_y - self._target_final_y)

        result = self._maingopy.EvaluationContainer()
        result.objective = final_target_penalty + 10.0 * step_residual_penalty + critical_penalty
        result.ineq = _model_function(
            self._maingopy,
            [
                rollout.max_abs_step_residual - 1e-4,
            ],
        )
        result.output = [
            self._maingopy.OutputVariable("toy_final_y", rollout.final_y),
            self._maingopy.OutputVariable("toy_max_abs_step_residual", rollout.max_abs_step_residual),
            self._maingopy.OutputVariable("toy_choke_gate", rollout.optional_choke.choke_gate),
            self._maingopy.OutputVariable("toy_near_sonic_gate", rollout.optional_choke.near_sonic_gate),
            self._maingopy.OutputVariable("toy_sign_change_gate", rollout.optional_choke.sign_change_gate),
        ]
        return result


def make_toy_fixed_newton_maingo_model(*, maingopy_module, **kwargs):
    base = ToyFixedNewtonImplicitReducedModelBase(maingopy_module=maingopy_module, **kwargs)

    class ToyFixedNewtonMAiNGOModel(maingopy_module.MAiNGOmodel):
        def __init__(self):
            maingopy_module.MAiNGOmodel.__init__(self)

        def get_variables(self):
            return base.get_variables()

        def get_initial_point(self):
            return base.get_initial_point()

        def evaluate(self, vars):
            return base.evaluate(vars)

    return ToyFixedNewtonMAiNGOModel()
