"""HJBGFDMSolver(use_wind_dependent_bc=True) constructs with its boundary detected (#2604).

The wind-dependent branch of `__init__` logged `len(boundary_indices)`, the constructor's parameter. With the default
boundary detection that is None, so the documented option raised `TypeError: object of type 'NoneType' has no len()`
at 52219be2 unless `boundary_indices` was passed explicitly.
"""

from __future__ import annotations

import warnings

import numpy as np

from mfgarchon.alg.numerical.hjb_solvers import HJBGFDMSolver
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.core.mfg_components import MFGComponents
from mfgarchon.core.mfg_problem import MFGProblem
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import no_flux_bc


def test_the_wind_dependent_bc_constructs_with_detected_boundary_indices():
    grid = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[11], boundary_conditions=no_flux_bc(dimension=1))
    comps = MFGComponents(
        hamiltonian=SeparableHamiltonian(control_cost=QuadraticControlCost(control_cost=1.0)),
        m_initial=lambda x: 1.0,
        u_terminal=lambda x: 0.0,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        problem = MFGProblem(geometry=grid, components=comps, T=1.0, Nt=4, volatility=0.5)
        solver = HJBGFDMSolver(
            problem,
            collocation_points=np.linspace(0.0, 1.0, 11).reshape(-1, 1),
            monotonicity_scheme="none",
            use_wind_dependent_bc=True,
            use_ghost_nodes=True,
        )
    np.testing.assert_array_equal(np.sort(solver.boundary_indices), [0, 10])
