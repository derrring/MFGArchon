#!/usr/bin/env python3
"""Issue #1426: solver options that are stored but never applied must fail loud on a non-default
value instead of being silent no-ops. Defaults remain accepted (baseline-safe).

Covers the two cleanly-dead options (no live non-default setter anywhere): GFDM ``congestion_mode``
and WENO ``weno_m_parameter``. (The FPSL/FPGFDM ``characteristic_solver`` / ``boundary_indices``
namesakes are live on other solvers and are handled separately.)
"""

import warnings

import pytest

import numpy as np

from mfgarchon.alg.numerical.hjb_solvers import HJBGFDMSolver, HJBWenoSolver
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
                HJBWenoSolver(problem, weno_m_parameter=2.0)

    def test_weno_m_parameter_default_ok(self):
        problem = _problem()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            HJBWenoSolver(problem, weno_m_parameter=1.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
