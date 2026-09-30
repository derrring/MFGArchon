"""The semi-Lagrangian pair keeps a reflection-symmetric problem symmetric (#1880).

The fixture is the capability matrix's smoke problem. It is invariant under ``x -> 1 - x``:
``m0 = exp(-10 (x - 1/2)^2)`` on ``[0, 1]``, ``u_T = 0``, no-flux at both walls, and a coupling with
no spatial dependence. So the exact solution is symmetric. That is the oracle: a law of the problem,
not another scheme's output.

At the fixture's time step a characteristic crosses about 10 cells per step. The HJB half cut such a
step into sub-steps; the FP half made one forward splat. The coupled map then amplified round-off
asymmetry about 6.6x per Picard sweep, and the density ended piled against one wall. Both halves now
cut a step by the one rule, ``cfl_substeps``.
"""

from __future__ import annotations

import numpy as np

from mfgarchon import Conditions, MFGProblem, Model
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import no_flux_bc
from mfgarchon.types import NumericalScheme


def _symmetric_problem():
    grid = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[21], boundary_conditions=no_flux_bc(dimension=1))
    bump = np.exp(-10 * (grid.coordinates[0] - 0.5) ** 2)
    mass = grid.integrate(bump)  # normalised to mass 1; the reflection symmetry is unaffected
    model = Model(
        hamiltonian=SeparableHamiltonian(
            control_cost=QuadraticControlCost(lambda_=1.0), coupling=lambda m: -m, coupling_dm=lambda m: -1.0
        ),
        volatility=0.0,
    )
    conditions = Conditions(
        m_initial=lambda x: np.exp(-10 * (np.asarray(x) - 0.5) ** 2) / mass, u_terminal=lambda x: 0.0, T=1.0
    )
    return MFGProblem(model=model, domain=grid, conditions=conditions, Nt=10)


def test_sl_linear_keeps_a_symmetric_problem_symmetric():
    result = _symmetric_problem().solve(
        scheme=NumericalScheme.SL_LINEAR, max_iterations=12, tolerance=1e-30, verbose=False
    )
    M = np.asarray(result.M)
    asymmetry = np.linalg.norm(M - M[:, ::-1]) / np.linalg.norm(M)
    # Round-off level. With the FP half unsub-stepped it measured 6.8e-9 after 12 sweeps (#1880).
    assert asymmetry < 1e-12, f"relative asymmetry {asymmetry:.3e} after 12 sweeps"
