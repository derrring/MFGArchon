"""Issue #1470 / #910: pin ``NetworkHamiltonian.__call__`` byte-identical to the live
``NetworkMFGProblem.hamiltonian`` method across the full dispatch surface.

The ``NetworkHamiltonian`` object (Issue #910) is built in every ``NetworkMFGProblem`` but
orphaned — ``NetworkMFGProblem.__init__`` overwrites ``self.components`` after constructing it,
so ``problem.hamiltonian_class`` is ``None`` and the object is never exercised. Being dead, it had
silently diverged from the method it wraps: it defaulted node congestion to ``0.0`` (vs the
method's ``0.5 * m[node]**2``) and read the dead ``congestion_func`` field (vs the live
``node_interaction_func``). The Issue #1470 reconciliation makes the object a faithful single
source; this test fails node-for-node against the pre-reconciliation object.

It pins the object in **isolation** (``H.__call__`` vs the method). Wiring the reconciled object
into the live solve paths as the single source is deferred (it flips the currently-NaN
``NetworkPolicyIterationHJBSolver`` to finite — a behavior change tracked separately in #1470).
"""

import pytest

import numpy as np

from mfgarchon.extensions.topology import (
    NetworkHamiltonian,
    NetworkMFGComponents,
    NetworkMFGProblem,
)
from mfgarchon.geometry.graph.network_geometry import GridNetwork

igraph = pytest.importorskip("igraph")


def _build_H(prob: NetworkMFGProblem) -> NetworkHamiltonian:
    """The object exactly as ``NetworkMFGProblem.__init__`` builds it, from the stored components."""
    return NetworkHamiltonian(
        network_data=prob.network_data,
        hamiltonian_func=prob.components.hamiltonian_func,
        hamiltonian_dm_func=prob.components.hamiltonian_dm_func,
        node_potential_func=prob.components.node_potential_func,
        node_interaction_func=prob.components.node_interaction_func,
    )


@pytest.mark.parametrize("case", ["default", "interaction", "custom"])
def test_network_hamiltonian_object_equals_method_node_by_node(case):
    """Byte-identity (``==``, not ``allclose``) of ``NetworkHamiltonian.__call__`` vs the
    ``NetworkMFGProblem.hamiltonian`` method, per node, over the three dispatch branches:
    default quadratic H, a custom ``node_interaction_func``, and a full custom ``hamiltonian_func``.
    """
    net = GridNetwork(width=4, height=3)
    net.create_network()
    if case == "default":
        comps = NetworkMFGComponents(node_potential_func=lambda n, t: 0.2 * n)
    elif case == "interaction":
        comps = NetworkMFGComponents(
            node_interaction_func=lambda n, m, t: 0.3 * m[n] + 0.1 * m[n] ** 3,
            node_potential_func=lambda n, t: 0.2 * n,
        )
    else:  # custom full H (dispatch through hamiltonian_func)
        comps = NetworkMFGComponents(
            hamiltonian_func=lambda n, nb, m, p, t: sum((p[j] - p[n]) ** 2 for j in nb) + 0.7 * m[n]
        )
    prob = NetworkMFGProblem(network_geometry=net, T=1.0, Nt=5, components=comps)
    H = _build_H(prob)

    rng = np.random.default_rng(7)
    N = prob.num_nodes
    m = rng.random(N)
    m /= m.sum()
    u = rng.random(N)
    t = 0.4

    for i in range(N):
        nbrs = prob.get_node_neighbors(i)
        method_val = prob.hamiltonian(i, nbrs, m, u, t)
        object_val = float(H(np.array([i]), m, u, t))
        assert method_val == object_val, f"node {i} ({case}): method={method_val!r} object={object_val!r}"
