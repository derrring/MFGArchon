"""The finite-difference HJB Jacobian linearises the residual at any gradient, steep ones included (#1884).

The per-point FD fallback of `compute_hjb_jacobian` used to clip every probed momentum to |p| <= 1e6. Where
the gradient exceeds that, both probes U +/- eps clip to the same value, so the Hamiltonian part of the
row comes out zero: the Jacobian of a different operator, exactly where the HJB is steepest. Nothing said
so. Measured on this file's state (|p| about 2e6), worst relative error over the interior rows, in all
four cases: 1.00 with the clip, i.e. the Hamiltonian block lost entirely; 3.9e-5 to 4.9e-5 without it,
the round-off of the fallback's own eps = 1e-7 at |U| about 2e6. The threshold sits between the two.

Oracle: a two-sided difference of `compute_hjb_residual`, computed without reference to the Jacobian code.
The Hamiltonian is quadratic, so the residual is quadratic in U and a central difference is exact up to
round-off; the tolerance is relative to the row's scale. Both are called with a NumPy backend, which is
what routes the Jacobian to the FD fallback (the analytic block, the default since #1884, is reached with
`backend=None` and is pinned by `test_hjb_jacobian_advection_1896.py`).

Interior rows only. On the periodic grid the end nodes are one node stored twice, so perturbing one copy
alone is not a direction the residual is defined along; under upwind its row 0 differs from the oracle by
the same amount with and without the clip, so it says nothing about this defect.
"""

from __future__ import annotations

import pytest

import numpy as np

from mfgarchon import Conditions, MFGProblem, Model
from mfgarchon.alg.numerical.hjb_solvers.base_hjb import compute_hjb_jacobian, compute_hjb_residual
from mfgarchon.backends import NumPyBackend
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import no_flux_bc, periodic_bc

NX = 21
X = np.linspace(0.0, 1.0, NX)
BOUNDS = np.array([[0.0, 1.0]])


@pytest.mark.parametrize("bc_factory", [no_flux_bc, periodic_bc], ids=["no_flux", "periodic"])
@pytest.mark.parametrize("upwind", [True, False], ids=["upwind", "central"])
def test_every_row_linearises_the_residual_at_a_gradient_beyond_the_old_clip(bc_factory, upwind):
    bc = bc_factory(dimension=1)
    problem = MFGProblem(
        model=Model(
            hamiltonian=SeparableHamiltonian(
                control_cost=QuadraticControlCost(control_cost=1.0), coupling=lambda m: m, coupling_dm=lambda m: 1.0
            ),
            volatility=0.3,
        ),
        domain=TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[NX], boundary_conditions=bc),
        conditions=Conditions(m_initial=lambda x: 1.0, u_terminal=lambda x: 0.0, T=1.0),
        Nt=10,
    )
    backend = NumPyBackend()
    m = np.ones(NX)
    # |p| about 2e6 at every node, twice the old clip limit; the sine keeps the upwind side varying.
    u = 2.0e6 * np.sin(np.pi * X) / np.pi + 1.0e5 * np.sin(6 * np.pi * X)
    u_next = np.zeros(NX)

    def residual(state):
        return np.asarray(
            compute_hjb_residual(state, u_next, m, problem, 5, backend, None, upwind, bc=bc, domain_bounds=BOUNDS),
            dtype=float,
        )

    jacobian = compute_hjb_jacobian(u, u, m, problem, 5, backend, None, upwind, bc=bc, domain_bounds=BOUNDS).toarray()
    eps = 1.0
    oracle = np.column_stack([(residual(u + eps * e) - residual(u - eps * e)) / (2 * eps) for e in np.eye(NX)])
    interior = slice(1, NX - 1)
    error = np.abs(jacobian[interior] - oracle[interior]) / np.abs(oracle[interior]).max(axis=1, keepdims=True)
    assert error.max() < 1e-3, (
        f"interior row {1 + int(np.unravel_index(error.argmax(), error.shape)[0])} of the FD Jacobian is not the "
        f"derivative of the residual (relative error {error.max():.2e}); a clipped momentum gives a zero "
        "Hamiltonian block (#1884)"
    )
