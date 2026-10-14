"""A failed inner solve does not return a value missing part of the cost (#1689 member 4).

The semi-Lagrangian n-D DPP step minimises dt * L(x, alpha, m, t) + U_next(x + alpha dt) per node with L-BFGS-B,
started at alpha = 0. For a non-smooth L that is the kink, and the line search often ends with ``success=False``.
The step then used U_next(x): the cost at alpha = 0 without its running cost. A constant potential V, which must
shift U by exactly V * (T - t), shifted it by less -- by up to 1.004 against 1.0 at V = 2 on this fixture. The step
now takes the minimiser's attained cost, the cost of the control it returns, whatever `success` says.

The network HJB's ODE path integrates backward with solve_ivp. When the integrator stopped short of t = 0, the
solver warned and filled every level it had not reached by extrapolating the trajectory, so a node potential singular
at t = 0.25 returned U(0) = 256 where the integral diverges. A failed integration is now refused, naming the last time level it
computed.
"""

from __future__ import annotations

import warnings

import pytest

import numpy as np
from scipy.sparse import csr_matrix

from mfgarchon import Conditions, MFGProblem, Model
from mfgarchon.alg.numerical.hjb_solvers import HJBSemiLagrangianSolver
from mfgarchon.core.hamiltonian import L1ControlCost, SeparableHamiltonian
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import no_flux_bc

N, NT, T = 11, 5, 0.5


def _solve(potential: float, monkeypatch=None, failures: list | None = None) -> np.ndarray:
    import mfgarchon.alg.numerical.hjb_solvers.hjb_semi_lagrangian as sl

    if monkeypatch is not None:
        real = sl.minimize

        def counting(fun, x0, *args, **kwargs):
            result = real(fun, x0, *args, **kwargs)
            failures.append(not result.success)
            return result

        monkeypatch.setattr(sl, "minimize", counting)
    grid = TensorProductGrid(
        bounds=[(0.0, 1.0), (0.0, 1.0)], Nx_points=[N, N], boundary_conditions=no_flux_bc(dimension=2)
    )
    hamiltonian = SeparableHamiltonian(
        control_cost=L1ControlCost(lambda_=1.0), potential=lambda t, x: potential * np.ones(np.asarray(x).shape[:-1])
    )
    problem = MFGProblem(
        model=Model(hamiltonian=hamiltonian, volatility=0.15),
        domain=grid,
        conditions=Conditions(u_terminal=lambda z: 0.0, m_initial=lambda z: 1.0, T=T),
        Nt=NT,
    )
    x, y = np.meshgrid(*grid.coordinates, indexing="ij")
    terminal = (x - 0.5) ** 2 + (y - 0.3) ** 2
    density = np.ones((NT + 1, N, N))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.asarray(HJBSemiLagrangianSolver(problem).solve_hjb_system(density, terminal, np.zeros_like(density)))


def test_a_constant_potential_shifts_the_value_by_v_times_t(monkeypatch):
    failures: list[bool] = []
    shifted = _solve(2.0, monkeypatch, failures)
    assert any(failures), "the fixture no longer reaches a failed minimisation; the pin would pass vacuously"
    monkeypatch.undo()
    shift = shifted[0] - _solve(0.0)[0]
    assert np.max(np.abs(shift - 2.0 * T)) < 1e-9, (shift.min(), shift.max())


def _network(singularity: float) -> np.ndarray:
    from mfgarchon.alg.numerical.network_solvers.hjb_network import NetworkHJBSolver
    from mfgarchon.extensions.topology import NetworkMFGComponents, NetworkMFGProblem
    from mfgarchon.geometry.graph.network_geometry import CustomNetwork

    n, nt = 6, 20
    ring = np.zeros((n, n))
    for k in range(n):
        ring[k, (k + 1) % n] = ring[(k + 1) % n, k] = 1.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PendingDeprecationWarning)  # the network problem's own components= form
        problem = NetworkMFGProblem(
            geometry=CustomNetwork(csr_matrix(ring)),
            T=0.5,
            Nt=nt,
            components=NetworkMFGComponents(node_potential_func=lambda t, i: 1.0 / (t - singularity) ** 2),
        )
    return NetworkHJBSolver(problem).solve_hjb_system(M_density=np.zeros((nt + 1, n)), U_terminal=np.zeros(n))


def test_a_failed_network_integration_is_refused_not_extrapolated():
    """U = the integral of V from t to T, which diverges as t falls to 0.25. The control, with the singularity outside
    [0, T], integrates to U(0) = 1 - 1/1.5 = 1/3."""
    from mfgarchon.utils.exceptions import NumericalInstabilityError

    with pytest.raises(
        NumericalInstabilityError,
        match=r"stopped before reaching t = 0; the last time level it computed is t = 0\.2\d*",
    ):
        _network(0.25)
    assert _network(-1.0)[0] == pytest.approx(np.full(6, 1.0 / 3.0), rel=1e-5)
