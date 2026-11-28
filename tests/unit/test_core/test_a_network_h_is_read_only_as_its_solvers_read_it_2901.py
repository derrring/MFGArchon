"""#2901, stage 1c-iii: on a geometry that declares itself a network, construction reads no Hamiltonian on a grid probe.

The network solvers hand x a node, and p and m one entry per node: NetworkHJBSolver reads `__call__` with the node as an
integer, and policy iteration and the network FP read the rates with ``np.array([i])``. No network solver reads `dm` or
`dp`. Construction's grid probe hands x a point's coordinates and p the coordinate dimension, a form none of them hands,
so where the geometry's declared type is a network no check reads it, whatever the Hamiltonian (`_has_a_grid_probe`).
With a custom `hamiltonian_func`, `__call__` reads that function while the rates stay the built-in control's, so both
readers of the rates refuse it (#1545); those two refusals are what keeps the two from meeting in a solve.
"""

from __future__ import annotations

import pytest

import numpy as np

from mfgarchon import Conditions, MFGProblem, Model
from mfgarchon.alg.numerical.network_solvers.fp_network import FPNetworkSolver
from mfgarchon.alg.numerical.network_solvers.hjb_network import NetworkPolicyIterationHJBSolver
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.extensions.topology import NetworkMFGComponents, NetworkMFGProblem
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import no_flux_bc
from mfgarchon.geometry.graph.network_geometry import GridNetwork


def _network():
    net = GridNetwork(width=3, height=3)
    net.create_network()
    return net


def _grid():
    return TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[11], boundary_conditions=no_flux_bc(dimension=1))


def _own_density_dm(t, node, neighbors, p, m):
    return -m[node]


def _one_sided_h(t, node, neighbors, p, m):
    return 0.5 * sum(max(p[node] - p[j], 0.0) ** 2 for j in neighbors) - 0.5 * m[node] ** 2


@pytest.mark.parametrize(
    ("solver", "refusal"),
    [
        (FPNetworkSolver, r"FPNetworkSolver cannot honour a custom hamiltonian_func"),
        (NetworkPolicyIterationHJBSolver, r"NetworkPolicyIterationHJBSolver cannot honour a custom hamiltonian_func"),
    ],
)
def test_a_reader_of_the_built_in_rates_refuses_a_custom_h(solver, refusal):
    """The rates are the built-in control's whatever H is given, so the network FP and policy iteration refuse a custom
    `hamiltonian_func` by name (#1545): without the refusal, the density would move by one control while the HJB solved
    another."""
    problem = NetworkMFGProblem(
        geometry=_network(),
        T=0.5,
        Nt=5,
        components=NetworkMFGComponents(hamiltonian_func=_one_sided_h, hamiltonian_dm_func=_own_density_dm),
    )
    with pytest.raises(NotImplementedError, match=refusal):
        solver(problem)


@pytest.mark.parametrize(
    ("domain", "density", "read"), [(_network, 1.0 / 9, False), (_grid, 1.0, True)], ids=["network", "grid"]
)
def test_the_geometry_decides_whether_construction_reads_the_hamiltonian(domain, density, read):
    """A `SeparableHamiltonian`, which is no `NetworkHamiltonian`, with a soft wall, recording every read of its
    sources. On a network geometry construction refuses it by its type, reading none of them (#1472 stage 2a: a problem
    on a network takes a `NetworkHamiltonian`); on a grid it reads them, which shows the recorder records. A test of how
    its methods are bound would read it on the network."""
    reads: list[str] = []

    class Recording(SeparableHamiltonian):
        def __call__(self, t, x, p, m):
            reads.append("__call__")
            return super().__call__(t=t, x=x, p=p, m=m)

        def evaluate_H(self, state):
            reads.append("evaluate_H")
            return super().evaluate_H(state)

        def dp(self, t, x, p, m):
            reads.append("dp")
            return super().dp(t=t, x=x, p=p, m=m)

        def evaluate_dp(self, state):
            reads.append("evaluate_dp")
            return super().evaluate_dp(state)

        def dm(self, t, x, p, m):
            reads.append("dm")
            return super().dm(t=t, x=x, p=p, m=m)

        def optimal_control(self, t, x, p, m):
            reads.append("optimal_control")
            return super().optimal_control(t=t, x=x, p=p, m=m)

    def wall(x):
        return np.ones(np.shape(x)[1:]) if np.ndim(x) >= 2 else 1.0

    def build():
        return MFGProblem(
            model=Model(hamiltonian=Recording(QuadraticControlCost(control_cost=1.0)), volatility=0.2),
            domain=domain(),
            conditions=Conditions(m_initial=lambda x: density, u_terminal=lambda x: 0.0, T=0.5),
            Nt=5,
            state_penalty=wall,
        )

    if read:
        build()
    else:
        with pytest.raises(TypeError, match=r"takes a NetworkHamiltonian.*was given Recording"):
            build()
    assert bool(reads) is read
