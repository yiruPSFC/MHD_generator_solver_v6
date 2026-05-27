from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from .design import CaseConfig, DesignVector
from .forward import (
    ForwardResult,
    _basis_function,
    _equation_form,
    _freidberg_balance_terms,
    _inlet_freidberg_scale_values,
    _inlet_residual_scale_values,
    _initial_delta_arrays,
    _ops_for_firedrake,
    _require_firedrake,
    _residual_scaling_mode,
)
from .legacy_physics import dynamic_system_terms, inlet_design_generic
from .objective import evaluate_profile_metrics
from .transport import working_fluid_for_config


@dataclass
class PersistentSolveTiming:
    assign_s: float
    solve_s: float
    postprocess_s: float
    total_s: float

    def to_dict(self) -> dict[str, float]:
        return {
            "assign_s": float(self.assign_s),
            "solve_s": float(self.solve_s),
            "postprocess_s": float(self.postprocess_s),
            "total_s": float(self.total_s),
        }


class PersistentForwardSolver:
    """Reusable Firedrake nonlinear solver for untaped continuation benchmarks.

    The state, residual, boundary conditions, and PETSc SNES are built once.
    Design coefficients are assignable Firedrake Constants. This intentionally
    supports the current continuation use case rather than every solve_forward
    feature.
    """

    def __init__(self, *, config: CaseConfig):
        self.config = config
        self.fd = _require_firedrake()
        self.ops = _ops_for_firedrake(self.fd)
        self.fluid = working_fluid_for_config(config)
        self.equation_form = _equation_form(config)
        self.residual_scaling = _residual_scaling_mode(config)
        fixed_area_source = str(config.metadata.get("fixed_area_profile_source", "design_spline")).strip().lower()
        if fixed_area_source != "design_spline":
            raise ValueError("PersistentForwardSolver currently supports fixed_area_profile_source='design_spline' only.")
        if self.residual_scaling != "inlet":
            raise ValueError("PersistentForwardSolver currently supports residual_scaling='inlet' only.")

        fd = self.fd
        self.mesh = fd.IntervalMesh(int(config.n_intervals), float(config.length_m))
        self.measure = fd.dx(domain=self.mesh)
        self.V = fd.FunctionSpace(self.mesh, "CG", 1)
        self.W = self.V * self.V
        self.state = fd.Function(self.W, name="persistent_delta_log_state")
        delta_log_n, delta_log_Te = fd.split(self.state)
        test_n, test_Te = fd.TestFunctions(self.W)
        self.x_norm = np.linspace(0.0, 1.0, int(config.n_intervals) + 1, dtype=float)

        self.controls = {name: fd.Constant(float(value)) for name, value in config.design.to_dict().items()}
        n_p_in = self.ops.exp(self.controls["log_n_p_in"])
        seed_fraction = self.ops.exp(self.controls["log_seed_fraction"])
        inlet = inlet_design_generic(
            ops=self.ops,
            n_p_in=n_p_in,
            T_e_in=self.controls["T_e_in"],
            Z_in=self.controls["Z_in"],
            I_0=self.controls["I_0"],
            seed_fraction=seed_fraction,
            B=self.controls["B_T"],
            inlet_A=float(config.area_scale_m2),
            working_fluid=self.fluid,
        )

        basis, slopes = config.design.area_control.basis_matrices(self.x_norm)
        b1 = _basis_function(fd, self.V, basis[:, 0], name="persistent_area_basis_a1")
        b2 = _basis_function(fd, self.V, basis[:, 1], name="persistent_area_basis_a2")
        b3 = _basis_function(fd, self.V, basis[:, 2], name="persistent_area_basis_a3")
        s1 = _basis_function(fd, self.V, slopes[:, 0], name="persistent_area_slope_a1")
        s2 = _basis_function(fd, self.V, slopes[:, 1], name="persistent_area_slope_a2")
        s3 = _basis_function(fd, self.V, slopes[:, 2], name="persistent_area_slope_a3")

        log_Te_in = self.ops.log(self.ops.max(self.controls["T_e_in"], 1.0))
        log_n = self.controls["log_n_p_in"] + delta_log_n
        log_Te = log_Te_in + delta_log_Te
        logA = self.controls["a1"] * b1 + self.controls["a2"] * b2 + self.controls["a3"] * b3
        sigma = (self.controls["a1"] * s1 + self.controls["a2"] * s2 + self.controls["a3"] * s3) / float(
            config.length_m
        )
        A = float(config.area_scale_m2) * self.ops.exp(logA)
        n_p = self.ops.exp(log_n)
        T_e = self.ops.exp(log_Te)
        dn_dx = n_p * delta_log_n.dx(0)
        dTe_dx = T_e * delta_log_Te.dx(0)

        closure, terms = dynamic_system_terms(
            ops=self.ops,
            n_p=n_p,
            T_e=T_e,
            A=A,
            sigma=sigma,
            dot_N=inlet["dot_N"],
            I_0=self.controls["I_0"],
            seed_fraction=seed_fraction,
            B=self.controls["B_T"],
            working_fluid=self.fluid,
        )
        self.row1_scale = fd.Constant(1.0)
        self.row2_scale = fd.Constant(1.0)
        if self.equation_form == "freidberg_hl":
            balances = _freidberg_balance_terms(
                ops=self.ops,
                closure=closure,
                A=A,
                B_T=self.controls["B_T"],
                area_scale_m2=float(config.area_scale_m2),
                heavy_particle_mass_kg=float(self.fluid.heavy_particle_mass_kg),
            )
            row1 = balances["H_p"].dx(0) - balances["rhs_H"]
            row2 = balances["L_p"].dx(0) - balances["rhs_L"]
        else:
            row1 = terms["M11"] * dn_dx + terms["M12"] * dTe_dx - terms["rhs_m"]
            row2 = terms["E11"] * dn_dx + terms["E12"] * dTe_dx - terms["rhs_e"]
        residual = ((row1 / self.row1_scale) * test_n + (row2 / self.row2_scale) * test_Te) * self.measure

        self.bcs = [
            fd.DirichletBC(self.W.sub(0), fd.Constant(0.0), 1),
            fd.DirichletBC(self.W.sub(1), fd.Constant(0.0), 1),
        ]
        self.problem = fd.NonlinearVariationalProblem(residual, self.state, bcs=self.bcs)
        solver_parameters = {
            "snes_type": str(config.metadata.get("snes_type", "newtonls")),
            "snes_rtol": 1e-8,
            "snes_atol": 1e-9,
            "snes_max_it": int(config.metadata.get("snes_max_it", 50)),
            "ksp_type": "preonly",
            "pc_type": "lu",
        }
        if "snes_dtol" in config.metadata:
            solver_parameters["snes_dtol"] = float(config.metadata["snes_dtol"])
        if "snes_linesearch_type" in config.metadata:
            solver_parameters["snes_linesearch_type"] = str(config.metadata["snes_linesearch_type"])
        self.solver = fd.NonlinearVariationalSolver(self.problem, solver_parameters=solver_parameters)

    def _assign_design(self, design: DesignVector) -> None:
        for name, value in design.to_dict().items():
            self.controls[name].assign(float(value))
        if self.equation_form == "freidberg_hl":
            row1, row2 = _inlet_freidberg_scale_values(design=design, config=self.config)
        else:
            row1, row2 = _inlet_residual_scale_values(design=design, config=self.config)
        self.row1_scale.assign(float(row1))
        self.row2_scale.assign(float(row2))

    def assign_initial_profile(self, *, profile: dict[str, Any], design: DesignVector) -> None:
        delta_log_n_values, delta_log_Te_values = _initial_delta_arrays(
            initial_profile=profile,
            design=design,
            target_x_norm=self.x_norm,
        )
        self.state.subfunctions[0].dat.data[:] = delta_log_n_values
        self.state.subfunctions[1].dat.data[:] = delta_log_Te_values

    def reset_zero(self) -> None:
        self.state.subfunctions[0].interpolate(self.fd.Constant(0.0))
        self.state.subfunctions[1].interpolate(self.fd.Constant(0.0))

    def _profile_from_state(self, *, design: DesignVector) -> dict[str, np.ndarray]:
        delta_log_n_fn, delta_log_Te_fn = self.state.subfunctions
        area = design.area_control.evaluate_profile(
            length=float(self.config.length_m),
            n_intervals=int(self.config.n_intervals),
            area_scale=float(self.config.area_scale_m2),
        )
        return {
            "x": self.x_norm * float(self.config.length_m),
            "x_norm": self.x_norm,
            "n_p": np.exp(float(design.log_n_p_in) + np.asarray(delta_log_n_fn.dat.data_ro, dtype=float).copy()),
            "T_e": np.exp(
                float(np.log(max(float(design.T_e_in), 1.0)))
                + np.asarray(delta_log_Te_fn.dat.data_ro, dtype=float).copy()
            ),
            "A": np.asarray(area["A"], dtype=float),
            "sigma_logA": np.asarray(area["sigma_logA"], dtype=float),
        }

    def solve(
        self,
        *,
        design: DesignVector,
        initial_profile: dict[str, Any] | None = None,
        reset_zero: bool = False,
    ) -> ForwardResult:
        t0 = time.perf_counter()
        self._assign_design(design)
        initial_guess = "persistent_previous_state"
        if reset_zero:
            self.reset_zero()
            initial_guess = "zero_delta"
        if initial_profile is not None:
            self.assign_initial_profile(profile=initial_profile, design=design)
            initial_guess = "profile_interpolated_delta"
        t1 = time.perf_counter()
        try:
            self.solver.solve()
        except Exception as exc:
            t2 = time.perf_counter()
            profile = None
            metrics = None
            try:
                profile = self._profile_from_state(design=design)
                if (
                    np.all(np.isfinite(profile["n_p"]))
                    and np.all(np.isfinite(profile["T_e"]))
                    and np.all(profile["n_p"] > 0.0)
                    and np.all(profile["T_e"] > 0.0)
                ):
                    metrics = evaluate_profile_metrics(profile=profile, design=design, config=self.config)
            except Exception:
                profile = None
                metrics = None
            t3 = time.perf_counter()
            return ForwardResult(
                ok=False,
                design=design,
                config=self.config,
                profile=profile,
                metrics=metrics,
                diagnostics={
                    "solver": "persistent_firedrake_snes_newtonls",
                    "equation_form": self.equation_form,
                    "residual_scaling": self.residual_scaling,
                    "initial_guess": initial_guess,
                    "timing": PersistentSolveTiming(t1 - t0, t2 - t1, t3 - t2, t3 - t0).to_dict(),
                },
                error=f"{type(exc).__name__}: {exc}",
            )
        t2 = time.perf_counter()
        profile = self._profile_from_state(design=design)
        metrics = evaluate_profile_metrics(profile=profile, design=design, config=self.config)
        t3 = time.perf_counter()
        return ForwardResult(
            ok=True,
            design=design,
            config=self.config,
            profile=profile,
            metrics=metrics,
            diagnostics={
                "solver": "persistent_firedrake_snes_newtonls",
                "equation_form": self.equation_form,
                "residual_scaling": self.residual_scaling,
                "initial_guess": initial_guess,
                "timing": PersistentSolveTiming(t1 - t0, t2 - t1, t3 - t2, t3 - t0).to_dict(),
            },
        )
