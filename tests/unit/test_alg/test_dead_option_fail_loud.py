#!/usr/bin/env python3
"""Issue #1426: solver options that are stored but never applied must fail loud on a non-default
value instead of being silent no-ops. Defaults remain accepted (baseline-safe).

Covers the GFDM ``congestion_mode`` / WENO ``weno_m_parameter`` options (S0-23/24), plus the
solver-specific dead knobs ``FPGFDMSolver.boundary_indices`` / ``domain_bounds`` (S0-26) and
``FPSLJacobianSolver.characteristic_solver`` (S0-27). These last two are guarded on those specific
solvers only — the namesakes are live on other solvers / geometry APIs. (Network knobs S0-25 are
pinned in ``test_fp_network_solver`` / ``test_hjb_network_solver`` alongside their live siblings.)
"""

import warnings

import pytest

import numpy as np

from mfgarchon.alg.numerical.fp_solvers.fp_gfdm import FPGFDMSolver
from mfgarchon.alg.numerical.fp_solvers.fp_semi_lagrangian import FPSLJacobianSolver
from mfgarchon.alg.numerical.hjb_solvers import HJBGFDMSolver, HJBWENOSolver
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.core.mfg_components import MFGComponents
from mfgarchon.core.mfg_problem import MFGProblem
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import no_flux_bc


def _problem(nx=21):
    comp = MFGComponents(
        m_initial=lambda x: np.exp(-10 * (x - 0.5) ** 2),
        u_terminal=lambda x: 0.0,
        hamiltonian=SeparableHamiltonian(
            control_cost=QuadraticControlCost(control_cost=1.0),
            coupling=lambda m: m,
            coupling_dm=lambda m: 1.0,
        ),
    )
    domain = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[nx], boundary_conditions=no_flux_bc(dimension=1))
    return MFGProblem(geometry=domain, T=1.0, Nt=21, sigma=0.5, components=comp)


def _pts(problem):
    bounds = problem.geometry.get_bounds()
    (nx,) = problem.geometry.get_grid_shape()
    return np.linspace(bounds[0][0], bounds[1][0], nx).reshape(-1, 1)


class TestDeadOptionFailLoud:
    def test_gfdm_congestion_mode_multiplicative_raises(self):
        problem = _problem()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(NotImplementedError, match="congestion_mode"):
                HJBGFDMSolver(problem, _pts(problem), monotonicity_scheme="none", congestion_mode="multiplicative")

    def test_gfdm_congestion_mode_additive_ok(self):
        problem = _problem()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            HJBGFDMSolver(problem, _pts(problem), monotonicity_scheme="none", congestion_mode="additive")

    def test_weno_m_parameter_nondefault_raises(self):
        problem = _problem()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(NotImplementedError, match="weno_m_parameter"):
                HJBWENOSolver(problem, weno_m_parameter=2.0)

    def test_weno_m_parameter_default_ok(self):
        problem = _problem()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            HJBWENOSolver(problem, weno_m_parameter=1.0)

    # Issue #1426 S0-26: FPGFDMSolver.boundary_indices / domain_bounds stored, never read.

    def test_fp_gfdm_boundary_indices_raises(self):
        problem = _problem()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(NotImplementedError, match="boundary_indices"):
                FPGFDMSolver(problem, collocation_points=_pts(problem), boundary_indices={0, 1})

    def test_fp_gfdm_domain_bounds_raises(self):
        problem = _problem()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(NotImplementedError, match="domain_bounds"):
                FPGFDMSolver(problem, collocation_points=_pts(problem), domain_bounds=[(0.0, 1.0)])

    def test_fp_gfdm_boundary_defaults_ok(self):
        problem = _problem()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            FPGFDMSolver(problem, collocation_points=_pts(problem))

    # Issue #1426 S0-27: FPSLJacobianSolver.characteristic_solver stored, never read.

    def test_fp_sl_characteristic_solver_nondefault_raises(self):
        problem = _problem()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(NotImplementedError, match="characteristic_solver"):
                FPSLJacobianSolver(problem, characteristic_solver="rk4")

    def test_fp_sl_characteristic_solver_default_ok(self):
        problem = _problem()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            FPSLJacobianSolver(problem)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
