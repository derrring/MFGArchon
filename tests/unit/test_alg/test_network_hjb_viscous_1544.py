"""The network HJB has its viscous term: ``-u_t - D Lap_G(u) + H = 0`` with ``D = sigma^2/2`` (Issue #1544).

At 6404fdc5 a nonzero volatility was refused: the solvers had no viscous term, though `FPNetworkSolver` applies
``D Lap_G(m)``. ``Lap_G = -L``, ``L = deg - A`` the problem's graph Laplacian.

The oracle is a manufactured solution on a weighted 8-cycle, u*(t, i) = (1 + t^2) cos(2 pi i / 8 + 0.7) + t:
- the phase and the unequal edge weights leave no reflection of the cycle that maps u* to itself;
- the control is active at u*: 7 of 8 nodes have a nonzero rate;
- sigma = 0.6, so D = 0.18 differs from sigma and from sigma^2;
- the forcing goes through the node potential, V = -u*_t + D L u* + H_control(u*), with no density.

Measured at this PR's head:
- the ODE path matches u*(0) to 1.45e-11;
- policy iteration is backward Euler: 1.21e-2, 6.05e-3, 3.03e-3 at Nt = 20, 40, 80, an EOC of 1.00.

The viscous term dropped, D read as sigma, and its sign flipped each miss by 0.07 to 0.16 on both paths. The policy
error then stops falling with dt.
"""

from __future__ import annotations

import warnings

import pytest

import numpy as np
from scipy.sparse import csr_matrix

from mfgarchon.alg.numerical.network_solvers.hjb_network import NetworkHJBSolver, NetworkPolicyIterationHJBSolver
from mfgarchon.extensions.topology import NetworkMFGComponents, NetworkMFGProblem
from mfgarchon.geometry.graph.network_geometry import CustomNetwork

N, SIGMA, T = 8, 0.6, 0.5
D = SIGMA**2 / 2
A = np.zeros((N, N))
for _k in range(N):
    A[_k, (_k + 1) % N] = A[(_k + 1) % N, _k] = 1.0 + 0.5 * _k / N
L = np.diag(A.sum(axis=1)) - A
X = 2 * np.pi * np.arange(N) / N + 0.7


def _u_star(t):
    return (1 + t**2) * np.cos(X) + t


def _h_control(u):
    return np.array([0.5 * sum(A[i, j] * max(u[i] - u[j], 0.0) ** 2 for j in range(N) if A[i, j]) for i in range(N)])


def _forcing(t, i):
    return float((-(2 * t * np.cos(X) + 1.0) + D * L @ _u_star(t) + _h_control(_u_star(t)))[i])


def _error_at_t0(solver_class, nt, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        problem = NetworkMFGProblem(
            geometry=CustomNetwork(csr_matrix(A)),
            T=T,
            Nt=nt,
            components=NetworkMFGComponents(node_potential_func=_forcing),
        )
        U = solver_class(problem, **kwargs).solve_hjb_system(
            M_density=np.zeros((nt + 1, N)), U_terminal=_u_star(T), volatility=SIGMA
        )
    return float(np.abs(U[0] - _u_star(0.0)).max())


@pytest.mark.parametrize("path", ["ODE", "policy iteration"])
def test_the_viscous_term_reproduces_a_manufactured_solution(path):
    if path == "ODE":
        assert _error_at_t0(NetworkHJBSolver, 20, tolerance=1e-10) < 1e-8
        return
    errors = [_error_at_t0(NetworkPolicyIterationHJBSolver, nt) for nt in (20, 40, 80)]
    orders = np.log2(np.array(errors[:-1]) / np.array(errors[1:]))
    assert np.all(np.abs(orders - 1.0) < 0.05), (errors, orders)
    assert errors[-1] < 4e-3, errors


@pytest.mark.parametrize(
    ("volatility", "kind"), [(np.full(N, SIGMA), "field"), (lambda t, x: SIGMA, None)], ids=["array", "callable"]
)
def test_a_volatility_fp_network_refuses_is_refused_here_too(volatility, kind):
    """The pair accepts the same inputs: a scalar. FPNetworkSolver refuses an array or a callable volatility."""
    problem = NetworkMFGProblem(geometry=CustomNetwork(csr_matrix(A)), T=T, Nt=4)
    with pytest.raises(NotImplementedError, match="takes a scalar volatility"):
        NetworkHJBSolver(problem).solve_hjb_system(
            M_density=np.zeros((5, N)), U_terminal=np.zeros(N), volatility=volatility, volatility_kind=kind
        )
