"""#2901, stage 1c-iii: a network Hamiltonian is read only in the forms its solvers hand it.

NetworkHJBSolver reads `__call__` with the node as an integer, and p and m one entry per node. Policy iteration and the
network FP read the rates with ``np.array([i])``. No solver reads its `dm` or `dp`. With a custom `hamiltonian_func`,
`__call__` reads that function while the rates stay the built-in control's, so both readers of the rates refuse it
(#1545); those two refusals are what keeps the two from meeting in a solve. Construction's grid probe, which hands x a
point's coordinates and p the coordinate dimension, reads no network Hamiltonian (`_grid_probes_apply`).
"""

from __future__ import annotations

import pytest

import numpy as np

from mfgarchon.alg.numerical.network_solvers.fp_network import FPNetworkSolver
from mfgarchon.alg.numerical.network_solvers.hjb_network import NetworkPolicyIterationHJBSolver
from mfgarchon.extensions.topology import NetworkMFGComponents, NetworkMFGProblem
from mfgarchon.geometry.graph.network_geometry import GridNetwork


def _network():
    net = GridNetwork(width=3, height=3)
    net.create_network()
    return net


def _own_density_dm(t, node, neighbors, p, m):
    return -m[node]


def _custom_h_problem(hamiltonian_func, hamiltonian_dm_func=_own_density_dm):
    return NetworkMFGProblem(
        geometry=_network(),
        T=0.5,
        Nt=5,
        components=NetworkMFGComponents(hamiltonian_func=hamiltonian_func, hamiltonian_dm_func=hamiltonian_dm_func),
    )


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
    problem = _custom_h_problem(_one_sided_h)
    with pytest.raises(NotImplementedError, match=refusal):
        solver(problem)


def test_construction_reads_no_network_h_on_a_grid_probe():
    """A custom whole H and its custom dm, each recording the size of every p it is handed. Construction hands neither
    one of another size than the node count: the grid probe, whose p has the coordinate dimension (2 on this network),
    reads neither `__call__` (`validate_hamiltonian`) nor `dm` (`validate_hamiltonian_derivative`). The recorder is
    shown to work on the solvers' own routes, `NetworkMFGProblem.hamiltonian` and `hamiltonian_dm`. The network's `dp`
    is the library's own and calls neither, so it is not recorded here."""
    sizes: list[tuple[str, int]] = []

    def recording_h(t, node, neighbors, p, m):
        sizes.append(("H", int(np.size(p))))
        return _one_sided_h(t, node, neighbors, p, m) if np.size(p) == 9 else 0.0

    def recording_dm(t, node, neighbors, p, m):
        sizes.append(("dm", int(np.size(p))))
        return _own_density_dm(t, node, neighbors, p, m) if np.size(p) == 9 else 0.0

    problem = _custom_h_problem(recording_h, recording_dm)
    assert [(source, size) for source, size in sizes if size != 9] == []
    before = len(sizes)
    u = np.linspace(0.0, 1.0, 9)
    m = np.full(9, 1.0 / 9)
    neighbors = problem.get_node_neighbors(1)
    problem.hamiltonian(t=0.0, node=1, neighbors=neighbors, p=u, m=m)
    problem.hamiltonian_dm(t=0.0, node=1, neighbors=neighbors, p=u, m=m)
    assert sizes[before:] == [("H", 9), ("dm", 9)]
