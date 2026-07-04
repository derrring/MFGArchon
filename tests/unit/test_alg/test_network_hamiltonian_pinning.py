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

from mfgarchon.alg.numerical.network_solvers.hjb_network import (
    NetworkHJBSolver,
    NetworkPolicyIterationHJBSolver,
)
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
    prob = NetworkMFGProblem(geometry=net, T=1.0, Nt=5, components=comps)
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


def test_network_components_is_mfg_components_byte_identical():
    """Issue #1470 Stage 1: ``NetworkMFGComponents`` IS-A ``MFGComponents`` (type unification), and
    the change is byte-identical.

    ``isinstance`` holds; the solve path is unchanged because ``hamiltonian_class`` stays ``None``
    (the network solvers read the method / legacy rate paths, not the object). ``get_problem_info()``
    now works (previously ``AttributeError`` on the non-subclass components' missing ``description``),
    and a graph problem with no geometry node-BC resolves ``get_boundary_conditions()`` to ``None``
    (Issue #1471 moved node-BC ownership to the geometry).
    """
    from mfgarchon.core.mfg_components import MFGComponents

    net = GridNetwork(width=3, height=3)
    net.create_network()
    prob = NetworkMFGProblem(geometry=net, T=0.5, Nt=4)

    assert isinstance(prob.components, MFGComponents)
    # Issue #1474: the NetworkHamiltonian is now WIRED as the single-source Hamiltonian (previously
    # orphaned/None). isinstance and get_problem_info remain the Stage-1 wins.
    assert prob.hamiltonian_class is not None, "the single-source NetworkHamiltonian must be wired"
    assert isinstance(prob.get_problem_info(), dict), "get_problem_info must not raise (was AttributeError)"
    assert prob.get_boundary_conditions() is None, "no geometry node-BC -> resolves to None (Issue #1471)"

    # network-native construction is unaffected
    comps = NetworkMFGComponents(node_potential_func=lambda n, t: 0.1 * n)
    assert isinstance(comps, MFGComponents)


def test_network_hamiltonian_minimize_consistency():
    """Issue #1474: the NetworkHamiltonian value H, optimal control, and dp form ONE consistent
    finite-state MFG (controlled CTMC, sense=MINIMIZE), replacing the previous mismatch (full
    quadratic __call__ vs upwind-uphill optimal_control) that made RK45 and FP solve different HJBs.

    Invariants on a line graph with a strictly increasing value ``u = [0,1,2,3,4]``:
    - control is DOWNHILL: at node 2, ``alpha* > 0`` toward the lower neighbour 1, ``== 0`` toward the
      higher neighbour 3 (the old code had it backwards);
    - rates are non-negative (valid conservative generator);
    - the control part of H is one-sided and equals the envelope ``0.5 * sum(alpha*^2)`` (unit weights);
    - the method ``NetworkMFGProblem.hamiltonian`` (used by RK45) equals the object ``__call__``.
    """
    net = GridNetwork(width=5, height=1)
    net.create_network()
    prob = NetworkMFGProblem(geometry=net, T=0.5, Nt=20)
    H = prob.hamiltonian_class
    assert H is not None
    N = prob.num_nodes
    u = np.arange(N, dtype=float)
    m = np.ones(N) / N
    t = 0.1

    alpha2 = np.atleast_1d(H.optimal_control(np.array([2]), m, u, t))
    assert alpha2[1] > 0, f"control must flow to the lower neighbour (MINIMIZE); got {alpha2}"
    assert alpha2[3] == 0, f"control must not flow to the higher neighbour (MINIMIZE); got {alpha2}"

    for i in range(N):
        ai = np.atleast_1d(H.optimal_control(np.array([i]), m, u, t))
        assert (ai >= -1e-12).all(), f"rates must be >= 0 at node {i}: {ai}"

    coupling2 = 0.5 * m[2] ** 2  # default node congestion at node 2
    control2 = float(H(np.array([2]), m, u, t)) - coupling2
    envelope2 = 0.5 * float(np.sum(alpha2**2))
    assert abs(control2 - 0.5) < 1e-9, f"one-sided control at node2 should be 0.5, got {control2}"
    assert abs(control2 - envelope2) < 1e-9, f"H control != envelope 0.5*sum(alpha^2): {control2} vs {envelope2}"

    method2 = prob.hamiltonian(2, prob.get_node_neighbors(2), m, u, t)
    assert abs(method2 - float(H(np.array([2]), m, u, t))) < 1e-9, "RK45 method H must equal object H"


def test_network_policy_iteration_converges_to_rk45():
    """Issue #1474 (N15 decisive): policy iteration and the RK45 value ODE now solve the SAME
    finite-state MFG.

    Before the fix they converged to *different* continuous equations (the gap plateaued at ~0.708
    under dt-refinement, ratio ~1.0). After reconciling the Hamiltonian (value = control), fixing the
    base-solver integration sign + source separation, and rewriting policy evaluation to the full-rate
    M-matrix ``A = I/dt + L^pi``, the remaining difference is pure time discretization (backward-Euler
    vs RK45): it roughly halves with each dt refinement, i.e. first-order convergence to zero.
    """
    net = GridNetwork(width=5, height=1)
    net.create_network()
    g = np.array([0.0, 0.0, 0.0, 0.0, 10.0])
    errs = []
    for nt in (20, 40, 80):
        prob = NetworkMFGProblem(geometry=net, T=0.5, Nt=nt)
        n = prob.num_nodes
        m = np.ones((nt + 1, n)) / n
        u_rk = NetworkHJBSolver(prob, scheme="RK45").solve_hjb_system(M_density=m, U_terminal=g)
        u_pi = NetworkPolicyIterationHJBSolver(prob).solve_hjb_system(M_density=m, U_terminal=g)
        assert np.isfinite(u_pi).all(), "policy-iteration value must be finite"
        assert np.isfinite(u_rk).all(), "RK45 value must be finite"
        errs.append(float(np.max(np.abs(u_pi[0] - u_rk[0]))))
    assert errs[0] < 0.2, f"PI and RK45 must agree closely (same HJB), got {errs[0]:.3f}"
    assert errs[-1] < 0.55 * errs[0], f"gap must shrink under dt-refinement (N15 plateau closed): {errs}"


def test_network_hamiltonian_maximize_fails_loud():
    """Issue #1474 / #1476: the network finite-state MFG is implemented for ``sense=MINIMIZE`` only.
    ``sense=MAXIMIZE`` must fail loud rather than silently compute the MINIMIZE (downhill) math. Full
    MAXIMIZE (reward-to-go / uphill) support — the mirror — is tracked in #1476."""
    from mfgarchon.core.hamiltonian import OptimizationSense

    net = GridNetwork(width=3, height=1)
    net.create_network()
    prob = NetworkMFGProblem(geometry=net, T=0.5, Nt=4)
    with pytest.raises(NotImplementedError, match="MINIMIZE"):
        NetworkHamiltonian(network_data=prob.network_data, sense=OptimizationSense.MAXIMIZE)


def test_network_hamiltonian_method_equals_object():
    """Issue #1472: ``NetworkMFGProblem.hamiltonian()`` IS the single-source ``NetworkHamiltonian``
    object, not a second hand-synced copy. The RK45 base solver (which reads the method) and the FP /
    policy-iteration solvers (which read the object via ``optimal_control``) therefore see the
    identical Hamiltonian — this pins the single source so the #1474/N15 divergence cannot re-open.
    """
    m5 = np.ones(5) / 5
    p5 = np.array([0.0, 1.0, 0.5, 2.0, 1.5])
    for comps in (
        NetworkMFGComponents(),
        NetworkMFGComponents(
            node_potential_func=lambda n, t: 0.3 * n,
            node_interaction_func=lambda n, m, t: 2.0 * m[n],
        ),
        NetworkMFGComponents(
            hamiltonian_func=lambda node, nbrs, m, p, t: sum(max(p[node] - p[j], 0) ** 2 for j in nbrs) + 0.7
        ),
    ):
        net = GridNetwork(width=5, height=1)
        net.create_network()
        prob = NetworkMFGProblem(geometry=net, T=0.5, Nt=10, components=comps)
        obj = prob.hamiltonian_class
        for node in range(prob.num_nodes):
            nbrs = prob.get_node_neighbors(node)
            method_val = prob.hamiltonian(node, nbrs, m5, p5, 0.1)
            object_val = float(obj(node, m5, p5, 0.1))
            assert method_val == object_val, f"method != object at node {node} (single-source broken)"


def test_network_geometry_alias_equivalence_and_deprecation():
    """Issue #1472 (deprecation policy): the constructor param is ``geometry`` (aligned with
    ``MFGProblem``); ``network_geometry`` is a deprecated alias that redirects IDENTICALLY. Proves
    old == new construction and that the alias warns.
    """
    net_new = GridNetwork(width=4, height=1)
    net_new.create_network()
    net_old = GridNetwork(width=4, height=1)
    net_old.create_network()

    prob_new = NetworkMFGProblem(geometry=net_new, T=0.5, Nt=10)
    with pytest.warns(DeprecationWarning, match="network_geometry"):
        prob_old = NetworkMFGProblem(network_geometry=net_old, T=0.5, Nt=10)

    # byte-identical construction: same node count, same single-source Hamiltonian, same values
    assert prob_new.num_nodes == prob_old.num_nodes
    assert type(prob_new.hamiltonian_class) is type(prob_old.hamiltonian_class)
    m = np.ones(prob_new.num_nodes) / prob_new.num_nodes
    p = np.arange(prob_new.num_nodes, dtype=float)
    for node in range(prob_new.num_nodes):
        h_new = prob_new.hamiltonian(node, prob_new.get_node_neighbors(node), m, p, 0.1)
        h_old = prob_old.hamiltonian(node, prob_old.get_node_neighbors(node), m, p, 0.1)
        assert h_new == h_old, f"alias construction diverged at node {node}"


def test_network_geometry_both_or_neither_fail_loud():
    """Issue #1472: passing both geometry= and network_geometry=, or neither, fails loud."""
    net = GridNetwork(width=3, height=1)
    net.create_network()
    with pytest.raises(ValueError, match="do not pass both"):
        NetworkMFGProblem(geometry=net, network_geometry=net, T=0.5, Nt=4)
    with pytest.raises(ValueError, match="requires a graph geometry"):
        NetworkMFGProblem(T=0.5, Nt=4)


def test_source_term_single_source_and_multipop_slice():
    """Issue #1470 Strand A: ``NetworkHamiltonian.source_term`` (V + f_m) is the SINGLE source consumed
    by both ``__call__`` (control + source) and ``hjb_network._source_terms``. Single-population it is
    ``V + 0.5*m[node]^2``; for stacked multi-population ``m`` it reads the OWN slice
    (``_extract_own_density``), matching ``__call__`` — the fork the HJB used to carry when it
    re-derived ``density_coupling`` on the raw stacked ``m``.
    """
    net = GridNetwork(width=3, height=3)
    net.create_network()
    comps = NetworkMFGComponents(node_potential_func=lambda n, t: 0.2 * n)  # default congestion
    prob = NetworkMFGProblem(geometry=net, components=comps, T=0.5, Nt=5)
    H = _build_H(prob)
    N = prob.num_nodes
    m = np.linspace(0.1, 0.9, N)
    p = 0.1 * np.arange(N, dtype=float)

    for node in range(N):
        src = H.source_term(node, m, 0.0)
        assert src == pytest.approx(0.2 * node + 0.5 * m[node] ** 2)
        # Decomposition: control = __call__ - source is p-dependent but m-INDEPENDENT (all m-dependence
        # is in the source), so scaling m does not change (__call__ - source).
        assert H(node, m, p, 0.0) - src == pytest.approx(H(node, 2 * m, p, 0.0) - H.source_term(node, 2 * m, 0.0))

    # Multi-population: population 1's source reads its OWN slice (0.9), not the raw stacked m[node] (0.1).
    H1 = NetworkHamiltonian(network_data=prob.network_data, population_index=1)
    m_stacked = np.concatenate([np.full(N, 0.1), np.full(N, 0.9)])
    for node in range(N):
        assert H1.source_term(node, m_stacked, 0.0) == pytest.approx(0.5 * 0.9**2)  # own slice; V=0 (no potential)
