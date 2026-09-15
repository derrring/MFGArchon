"""The 1-D HJB Jacobian is the derivative of the residual it linearises, including at a periodic wrap (#1834).

`compute_hjb_jacobian`'s FD fallback -- the default `HJBFDMSolver` path, since its backend is not None -- took
row i's neighbours as ``(i - 1) % Nx`` and ``(i + 1) % Nx``. On the endpoint-inclusive periodic grid a
`TensorProductGrid` builds, node Nx-1 is node 0, so the residual's ghost for row 0 is ``U[Nx-2]`` and for row
Nx-1 is ``U[1]``. The `% Nx` columns were never those, and the Hamiltonian half of both wrap entries was
missing: ``J[0, 19] = -18`` against a residual derivative of ``-155.7`` on #1822's fixture at 6c0610d2, and
Newton stalled. Under ``bc=None``, the legacy exclusive-periodic residual, the band assembly dropped the wrap
entries too: 85% relative on a random state, on both schemes, at 1fcd15b4.

Oracle: the definition of a Jacobian. Each column of the central finite difference of `compute_hjb_residual`
must match the assembled column, for both assembly paths (the FD fallback and the analytic chain rule), both
schemes, and states whose wrap rows take both upwind branches. No-flux is the control that has no wrap.
"""

from __future__ import annotations

import warnings

import pytest

import numpy as np

import mfgarchon.alg.numerical.hjb_solvers.base_hjb as bh
from mfgarchon import Conditions, MFGProblem, Model
from mfgarchon.backends import create_backend
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import no_flux_bc, periodic_bc

_N = 21
_X = np.linspace(0.0, 1.0, _N)
_M = 1.0 + 0.5 * np.cos(2 * np.pi * _X + 1.1)


def _problem(bc):
    hamiltonian = SeparableHamiltonian(
        control_cost=QuadraticControlCost(control_cost=1.0), coupling=lambda m: m, coupling_dm=lambda m: 1.0
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return MFGProblem(
            model=Model(hamiltonian=hamiltonian, sigma=0.3),
            domain=TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[_N], boundary_conditions=bc),
            conditions=Conditions(m_initial=lambda z: 1.0, u_terminal=lambda z: 0.0, T=0.5),
            Nt=10,
        )


def _states(identify_endpoints: bool):
    """Four states: two phases, each with both signs, so each wrap row sees the forward and the backward branch.

    ``identify_endpoints`` makes ``U[-1] == U[0]``, which the grid-periodic convention requires. The legacy
    ``bc=None`` residual wraps between those two columns, so on such a state its wrap difference is zero and
    its wrap entries carry no Hamiltonian part to get wrong: that case gets states whose ends differ.
    """
    for phase in (0.7, 2.4):
        for sign in (1.0, -1.0):
            state = sign * (np.sin(2 * np.pi * _X + phase) + 0.3 * np.cos(6 * np.pi * _X + 0.5 + phase))
            if identify_endpoints:
                state[-1] = state[0]
            else:
                state = state + sign * 0.8 * _X
            yield state


@pytest.mark.parametrize("assembly", ["fd_fallback", "analytic"])
@pytest.mark.parametrize("upwind", [True, False], ids=["upwind", "central"])
@pytest.mark.parametrize("boundary", ["periodic", "periodic_bc_none", "no_flux"])
def test_the_jacobian_is_the_derivative_of_the_residual(boundary: str, upwind: bool, assembly: str):
    problem = _problem(no_flux_bc(dimension=1) if boundary == "no_flux" else periodic_bc(dimension=1))
    bc = None if boundary == "periodic_bc_none" else problem.geometry.get_boundary_conditions()
    bounds = problem.geometry.get_bounds()
    domain_bounds = np.array([[bounds[0][0], bounds[1][0]]])
    backend = create_backend("numpy") if assembly == "fd_fallback" else None
    worst = 0.0
    for state in _states(identify_endpoints=boundary == "periodic"):

        def residual(u, state=state):
            return np.asarray(
                bh.compute_hjb_residual(
                    u,
                    0.9 * state,
                    _M,
                    problem,
                    5,
                    None,
                    0.3,
                    upwind,
                    bc=bc,
                    domain_bounds=domain_bounds,
                    current_time=0.25,
                ),
                dtype=float,
            )

        derivative = np.column_stack([(residual(state + e) - residual(state - e)) / 2e-7 for e in 1e-7 * np.eye(_N)])
        assembled = bh.compute_hjb_jacobian(
            state, state, _M, problem, 5, backend, 0.3, upwind, bc=bc, domain_bounds=domain_bounds, current_time=0.25
        ).toarray()
        gap = np.abs(assembled - derivative)
        worst = max(worst, float(gap.max()) / float(np.abs(derivative).max()))
        i, j = np.unravel_index(gap.argmax(), gap.shape)
        assert worst < 1e-6, (
            f"{boundary}/{'upwind' if upwind else 'central'}/{assembly}: J[{i},{j}] = {assembled[i, j]:.4g} against "
            f"d(residual)/dU = {derivative[i, j]:.4g} (#1834)"
        )
