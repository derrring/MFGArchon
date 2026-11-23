"""A coupled route's cold start holds, from level 1 on, the FP's own value at a node the FP's step prescribes (#2991).

The class: the iterate's later levels disagreeing with the FP's rows at the nodes the FP prescribes. FP-FDM's implicit
Dirichlet rows, and FP-FEM's condensed dofs, hold a value p at every later level whatever they are handed: 0 at the
problem's Dirichlet face, which the FP reads as an exit (#2512), and g(t_n) on the FP's own DIRICHLET(g). Each sweep
blends the FP's row into the iterate with a weight theta < 1, so a cold start that put m_initial there kept
prod(1 - theta_j)(m_initial - p) after any number of sweeps, and its first sweep handed the HJB m_initial there.

The cold start now asks the FP (`BaseFPSolver.prescribed_density`), which answers with the read its own step makes. At
f9a41a31, with the walls planted at 7.0, the density the first sweep hands the HJB held 7.0 there on every route below,
and the density returned after one sweep at relaxation 0.5 held 3.5 = (1 - 0.5)(7 - 0) (3.3 against g(t_n) = 0.3 + t_n
at the first level), except RegimeSwitching's, whose first sweep takes the FP's row whole.
"""

from __future__ import annotations

import functools

import pytest

import numpy as np

from mfgarchon import Conditions, Model
from mfgarchon.alg.numerical.coupling import (
    BlockGaussSeidelIterator,
    BlockJacobiIterator,
    FictitiousPlayIterator,
    FixedPointIterator,
)
from mfgarchon.alg.numerical.coupling.graph_coupling import AdjacencyCoupling
from mfgarchon.alg.numerical.coupling.graph_mfg_solver import GraphMFGSolver
from mfgarchon.alg.numerical.coupling.multi_population_iterator import MultiPopulationIterator
from mfgarchon.alg.numerical.coupling.regime_switching_iterator import RegimeSwitchingIterator
from mfgarchon.alg.numerical.fem import FPFEMSolver, HJBFEMSolver
from mfgarchon.alg.numerical.fp_solvers.fp_fdm import FPFDMSolver
from mfgarchon.alg.numerical.hjb_solvers import HJBFDMSolver
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.core.mfg_problem import MFGProblem
from mfgarchon.core.multi_population import MultiPopulationProblem
from mfgarchon.core.regime_switching import RegimeSwitchingConfig
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import BCSegment, BCType, BoundaryConditions, dirichlet_bc
from mfgarchon.geometry.meshes.mesh_1d import Mesh1D

NT, PLANT = 5, 7.0


def _model():
    return Model(hamiltonian=SeparableHamiltonian(QuadraticControlCost(control_cost=1.0)), volatility=0.2)


def _coordinate(x):
    return np.asarray(x[0] if isinstance(x, (tuple, list)) else x, dtype=float)


def _m_initial(x):
    """1 + x/2 inside, PLANT at x = 0 and x = 1."""
    a = _coordinate(x)
    return np.where(np.isclose(a, 0.0) | np.isclose(a, 1.0), PLANT, 1.0 + 0.5 * a)


def _problem(bc, mesh=False):
    if mesh:
        domain = Mesh1D(bounds=(0.0, 1.0), num_elements=10)
        domain.generate_mesh()
        domain.boundary_conditions = bc
    else:
        domain = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[11], boundary_conditions=bc)
    return MFGProblem(
        model=_model(),
        domain=domain,
        conditions=Conditions(m_initial=_m_initial, u_terminal=lambda x: 0.0 * _coordinate(x), T=0.5),
        Nt=NT,
    )


# (FP arm, the problem's BC, the FP's own BC or None, the HJB class, the FP class, a mesh, p at level n)
ARMS = {
    "FP-FDM, the problem's Dirichlet face (an exit, p = 0)": (
        lambda: dirichlet_bc(value=0.0, dimension=1),
        None,
        HJBFDMSolver,
        FPFDMSolver,
        False,
        lambda n, dt: 0.0,
    ),
    "FP-FDM, its own DIRICHLET(0.3 + t)": (
        lambda: dirichlet_bc(value=lambda t: 0.3 + t, dimension=1),
        lambda: dirichlet_bc(value=lambda t: 0.3 + t, dimension=1),
        HJBFDMSolver,
        FPFDMSolver,
        False,
        lambda n, dt: 0.3 + n * dt,
    ),
    "FP-FEM, the problem's Dirichlet face (an exit, p = 0)": (
        lambda: dirichlet_bc(value=0.0, dimension=1),
        None,
        lambda q: HJBFEMSolver(q, order=1),
        lambda q, **kw: FPFEMSolver(q, order=1, **kw),
        True,
        lambda n, dt: 0.0,
    ),
}


def _solve(route, arm, iterations):
    """Run ``route`` on ``arm`` for ``iterations`` sweeps; return the (Nt + 1, 11) density of the first population."""
    problem_bc, own_bc, hjb_cls, fp_cls, mesh, _ = ARMS[arm]

    def problem():
        return _problem(problem_bc(), mesh=mesh)

    def fp(q):
        return fp_cls(q, boundary_conditions=own_bc()) if own_bc is not None else fp_cls(q)

    if route == "problem.solve":
        return np.asarray(problem().solve(max_iterations=iterations, verbose=False).M)
    if route in ("FixedPoint", "FictitiousPlay", "BlockGaussSeidel", "BlockJacobi"):
        q = problem()
        cls, kw = {
            "FixedPoint": (FixedPointIterator, {"relaxation": 0.5}),
            # Its learning rate is alpha_0 / (j + 1): the default alpha_0 = 1 takes the first FP row whole.
            "FictitiousPlay": (FictitiousPlayIterator, {"initial_learning_rate": 0.5}),
            "BlockGaussSeidel": (BlockGaussSeidelIterator, {"relaxation": 0.5}),
            "BlockJacobi": (BlockJacobiIterator, {"relaxation": 0.5}),
        }[route]
        return np.asarray(cls(q, hjb_cls(q), fp(q), **kw).solve(max_iterations=iterations, verbose=False).M)
    if route == "MultiPopulation":
        q = problem()
        multi = MultiPopulationProblem(populations=[q], population_names=["a"])
        result = MultiPopulationIterator(multi, [hjb_cls(q)], [fp(q)], relaxation=0.5)
        return np.asarray(result.solve(max_iterations=iterations, tolerance=1e-12).M[0])
    if route == "Graph":
        q = problem()
        coupling = AdjacencyCoupling(np.zeros((1, 1)), alpha=0.05, beta=0.02)
        solver = GraphMFGSolver(
            [q], coupling, [hjb_cls(q)], [fp(q)], max_iterations=iterations, tolerance=1e-12, damping=0.5
        )
        return np.asarray(solver.solve().densities[0])
    qs = [problem(), problem()]
    config = RegimeSwitchingConfig(transition_matrix=np.array([[-0.1, 0.1], [0.2, -0.2]]))
    solver = RegimeSwitchingIterator(
        qs,
        config,
        [hjb_cls(q) for q in qs],
        [fp(q) for q in qs],
        max_iterations=iterations,
        tolerance=1e-12,
        damping=0.5,
    )
    return np.asarray(solver.solve().densities[0])


ROUTES = ["FixedPoint", "FictitiousPlay", "BlockGaussSeidel", "BlockJacobi", "MultiPopulation", "Graph", "Regime"]
CASES = [
    *[(route, arm) for arm in list(ARMS)[:1] for route in [*ROUTES, "problem.solve"]],
    # RegimeSwitchingIterator refuses an FP's own BC; problem.solve has no FP-own-BC arm.
    *[(route, list(ARMS)[1]) for route in ROUTES if route != "Regime"],
    # The coupled FEM routes go through `initialize_cold_start`.
    *[(route, list(ARMS)[2]) for route in ("FixedPoint", "BlockGaussSeidel")],
]


@pytest.mark.parametrize(("route", "arm"), CASES, ids=[f"{route} | {arm}" for route, arm in CASES])
def test_the_cold_start_holds_the_fps_prescribed_value_at_its_later_levels(route, arm, monkeypatch):
    """At the walls, from level 1 on: the density the first sweep hands the HJB, and the density returned after one
    sweep, are the FP's prescribed value p(t_n), not the caller's 7.0.

    The first is read by a recorder on the HJB's `solve_hjb_system`, wrapped so the signature the iterators read
    (`hjb_solver_sig_params`) is the solver's own; its control is that the route returns the same density with the
    recorder as without it, and that the recorder saw a call."""
    *_, mesh, p_at = ARMS[arm]
    unrecorded = _solve(route, arm, 1)

    handed: list[np.ndarray] = []
    hjb_class = HJBFEMSolver if mesh else HJBFDMSolver
    original = hjb_class.solve_hjb_system

    @functools.wraps(original)
    def recording(self, *args, **kwargs):
        handed.append(np.array(args[0] if args else kwargs["M_density"], dtype=float, copy=True))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(hjb_class, "solve_hjb_system", recording)
    returned = _solve(route, arm, 1)
    assert handed, f"the recorder saw no HJB call on {route}"
    np.testing.assert_array_equal(returned, unrecorded, err_msg="the recorder changed what the route returns")

    dt = 0.5 / NT
    walls = [0, -1]
    p = np.array([p_at(n, dt) for n in range(1, NT + 1)])
    first = handed[0].reshape(NT + 1, -1)
    assert np.max(np.abs(first[1:, walls] - p[:, None])) <= 1e-12, (
        f"the first sweep handed the HJB {first[1:, walls].tolist()} at the walls, against p {p.tolist()}"
    )
    assert np.max(np.abs(returned.reshape(NT + 1, -1)[1:, walls] - p[:, None])) <= 1e-12, (
        f"one sweep returned {returned.reshape(NT + 1, -1)[1:, walls].tolist()} at the walls, against p {p.tolist()}"
    )


def test_a_2d_cold_start_holds_the_value_on_the_face_the_fp_prescribes():
    """On a 9 x 7 grid whose FP holds x_min at g(t) = 0.2 + 2t, one FixedPoint sweep returns g(t_n) on x_min from level 1
    on, and the no-flux faces keep the caller's 7.0 at level 1, where the HJB was handed them. The grid is not square and
    the face is one, so a cold start that wrote the FP's flat node indices in another order would miss it."""
    dims = (9, 7)
    bc = BoundaryConditions(
        dimension=2,
        segments=[BCSegment(name="exit", bc_type=BCType.DIRICHLET, value=lambda t: 0.2 + 2.0 * t, boundary="x_min")],
        default_bc=BCType.NO_FLUX,
    )

    def m_initial(x):
        a, b = (np.asarray(c, dtype=float) for c in x[:2])
        edge = np.isclose(a, 0.0) | np.isclose(a, 1.0) | np.isclose(b, 0.0) | np.isclose(b, 1.0)
        return np.where(edge, PLANT, 1.0 + 0.5 * a + 0.25 * b)

    q = MFGProblem(
        model=_model(),
        domain=TensorProductGrid(bounds=[(0.0, 1.0)] * 2, Nx_points=list(dims), boundary_conditions=bc),
        conditions=Conditions(m_initial=m_initial, u_terminal=lambda x: 0.0 * _coordinate(x), T=0.5),
        Nt=NT,
    )
    handed: list[np.ndarray] = []
    hjb = HJBFDMSolver(q)
    original = hjb.solve_hjb_system

    @functools.wraps(original)
    def recording(*args, **kwargs):
        handed.append(np.array(args[0], dtype=float, copy=True))
        return original(*args, **kwargs)

    hjb.solve_hjb_system = recording
    M = np.asarray(
        FixedPointIterator(q, hjb, FPFDMSolver(q, boundary_conditions=bc), relaxation=0.5)
        .solve(max_iterations=1, verbose=False)
        .M
    )
    g = np.array([0.2 + 2.0 * n * q.dt for n in range(1, NT + 1)])
    assert M.shape == (NT + 1, *dims)
    assert float(np.max(np.abs(M[1:, 0, :] - g[:, None]))) <= 1e-12, M[1:, 0, :].tolist()
    first = handed[0].reshape(NT + 1, *dims)
    assert float(np.max(np.abs(first[1:, 0, :] - g[:, None]))) <= 1e-12, first[1:, 0, :].tolist()
    assert np.all(first[1, -1, :] == PLANT)
    assert np.all(first[1, 1:, 0] == PLANT)
    assert np.all(first[1, 1:, -1] == PLANT)


def _regime(problems, hjb_solvers=None):
    config = RegimeSwitchingConfig(transition_matrix=np.array([[-0.1, 0.1], [0.2, -0.2]]))
    hjbs = hjb_solvers or [HJBFDMSolver(q) for q in problems]
    return RegimeSwitchingIterator(
        problems, config, hjbs, [FPFDMSolver(q) for q in problems], max_iterations=1, tolerance=1e-12, damping=0.5
    )


def test_regimes_first_fp_source_reads_the_cold_starts_prescribed_value():
    """RegimeSwitchingIterator builds its first sweep's FP-source density apart from its HJB input. A problem's
    `source_term_fp` is handed that density's slice at each level the FP step reads, t_1..T: at the walls, the FP's
    prescribed value 0, not the caller's 7.0."""
    handed: list[tuple[float, np.ndarray]] = []
    solving = [False]

    def source(t, x, m):
        if solving[0]:
            handed.append((float(t), np.array(m, dtype=float).reshape(-1)[[0, -1]]))
        return np.zeros(np.shape(m))

    def problem():
        return MFGProblem(
            model=_model(),
            domain=TensorProductGrid(
                bounds=[(0.0, 1.0)], Nx_points=[11], boundary_conditions=dirichlet_bc(value=0.0, dimension=1)
            ),
            conditions=Conditions(m_initial=_m_initial, u_terminal=lambda x: 0.0 * _coordinate(x), T=0.5),
            Nt=NT,
            source_term_fp=source,
        )

    solver = _regime([problem(), problem()])
    solving[0] = True
    solver.solve()
    later = [walls for t, walls in handed if t > 0.5 / NT / 2]
    assert later, "the FP source was handed no later level"
    assert max(float(np.max(np.abs(w))) for w in later) == 0.0, [w.tolist() for w in later]


def test_regimes_early_return_reports_the_cold_start_it_held(monkeypatch):
    """When the first sweep's HJB diverges, RegimeSwitchingIterator returns the densities it held, its cold start: at the
    walls from level 1 on, the FP's prescribed value 0. The HJB is replaced by one that returns NaN, wrapped so the
    signature the iterator reads is the solver's own."""
    original = HJBFDMSolver.solve_hjb_system

    @functools.wraps(original)
    def diverging(self, *args, **kwargs):
        U = np.asarray(original(self, *args, **kwargs), dtype=float)
        return np.full_like(U, np.nan)

    problems = [_problem(dirichlet_bc(value=0.0, dimension=1)) for _ in range(2)]
    solver = _regime(problems)
    monkeypatch.setattr(HJBFDMSolver, "solve_hjb_system", diverging)
    result = solver.solve()
    assert not result.converged
    densities = np.asarray(result.densities[0])
    assert densities.shape == (NT + 1, 11)
    assert float(np.max(np.abs(densities[1:, [0, -1]]))) == 0.0, densities[1:, [0, -1]].tolist()


def _held_by_the_step(fp, shape):
    """The oracle, which does not ask the protocol: the FP's direct call on two unrelated inputs. A (level, node) is held
    when the two outputs agree there to 1e-12; returns {level: (nodes, values)} for levels 1..NT."""
    outs = []
    for seed, shift in ((1, 0.0), (2, 5.0)):
        m0 = np.random.default_rng(seed).uniform(0.5, 2.0, size=shape) + shift
        U = np.random.default_rng(seed + 10).normal(size=(NT + 1, *shape)) * 0.1
        outs.append(np.asarray(fp.solve_fp_system(m0, U, show_progress=False), dtype=float).reshape(NT + 1, -1))
    a, b = outs
    held = {}
    for n in range(1, NT + 1):
        nodes = np.flatnonzero(np.abs(a[n] - b[n]) <= 1e-12 * np.maximum(1.0, np.abs(a[n])))
        held[n] = (nodes, a[n][nodes])
    return held


def _grid_fp(dims, bc, own):
    q = MFGProblem(
        model=_model(),
        domain=TensorProductGrid(bounds=[(0.0, 1.0)] * len(dims), Nx_points=list(dims), boundary_conditions=bc),
        conditions=Conditions(
            m_initial=lambda x: 1.0 + 0.0 * _coordinate(x), u_terminal=lambda x: 0.0 * _coordinate(x), T=0.5
        ),
        Nt=NT,
    )
    return (FPFDMSolver(q, boundary_conditions=bc) if own else FPFDMSolver(q)), dims


def _mesh_fp(bc, own):
    q = _problem(bc, mesh=True)
    return (FPFEMSolver(q, order=1, boundary_conditions=bc) if own else FPFEMSolver(q, order=1)), (11,)


LAYOUTS = {
    # A non-square grid with one Dirichlet face reads the flat node order; a transposed reading moves the nodes.
    "FP-FDM, 9 x 7, its own DIRICHLET(0.2 + 2t + y/10) on x_min, no-flux elsewhere": lambda: _grid_fp(
        (9, 7),
        BoundaryConditions(
            dimension=2,
            segments=[
                BCSegment(
                    name="exit",
                    bc_type=BCType.DIRICHLET,
                    value=lambda t, x: 0.2 + 2.0 * t + 0.1 * float(np.asarray(x).reshape(-1)[-1]),
                    boundary="x_min",
                )
            ],
            default_bc=BCType.NO_FLUX,
        ),
        own=True,
    ),
    # Two Dirichlet faces with different values meet at a corner, and each meets a no-flux face at another.
    "FP-FDM, 9 x 7, its own DIRICHLET 0.4 on x_min and 0.9 on y_min, no-flux elsewhere": lambda: _grid_fp(
        (9, 7),
        BoundaryConditions(
            dimension=2,
            segments=[
                BCSegment(name="a", bc_type=BCType.DIRICHLET, value=0.4, boundary="x_min"),
                BCSegment(name="b", bc_type=BCType.DIRICHLET, value=0.9, boundary="y_min"),
            ],
            default_bc=BCType.NO_FLUX,
        ),
        own=True,
    ),
    "FP-FDM, 1-D, the problem's Dirichlet x_min beside a NO_FLUX x_max": lambda: _grid_fp(
        (11,),
        BoundaryConditions(
            dimension=1,
            segments=[
                BCSegment(name="exit", bc_type=BCType.DIRICHLET, value=0.0, boundary="x_min"),
                BCSegment(name="wall", bc_type=BCType.NO_FLUX, boundary="x_max"),
            ],
        ),
        own=False,
    ),
    "FP-FEM, Mesh1D, its own DIRICHLET(0.3 + t + x/10)": lambda: _mesh_fp(
        dirichlet_bc(dimension=1, value=lambda t, x: 0.3 + t + 0.1 * float(np.asarray(x).reshape(-1)[0])), own=True
    ),
}


@pytest.mark.parametrize("layout", list(LAYOUTS))
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_the_fp_names_exactly_the_nodes_its_step_holds_and_their_values(layout):
    """`prescribed_density(n)` returns the step's own read: the nodes the direct call holds whatever it is handed, at
    every level, with the values it holds there. A second derivation of p could disagree on a corner, a face, the node
    order or the level's time; this fails then."""
    fp, shape = LAYOUTS[layout]()
    held = _held_by_the_step(fp, shape)
    assert any(len(held[n][0]) for n in held), "the oracle found no held node: the layout does not reach the protocol"
    for n in range(1, NT + 1):
        answer = fp.prescribed_density(n)
        assert answer is not None, f"level {n}: the step holds {held[n][0].tolist()}, the protocol names none"
        order = np.argsort(answer[0])
        np.testing.assert_array_equal(np.asarray(answer[0])[order], held[n][0], err_msg=f"level {n}: the nodes")
        np.testing.assert_allclose(np.asarray(answer[1])[order], held[n][1], rtol=0, atol=1e-14, err_msg=f"level {n}")


def test_fp_fdm_names_no_value_at_a_capped_exit():
    """A capped exit's density at a later level follows each step's active set (#2581), so FP-FDM prescribes nothing
    there and refuses to name a value. Every coupled route refuses a capped exit before its cold start."""
    capped = BoundaryConditions(
        dimension=1,
        segments=[
            BCSegment(name="x_min", bc_type=BCType.NO_FLUX, boundary="x_min"),
            BCSegment(name="exit", bc_type=BCType.DIRICHLET, value=0.0, boundary="x_max", flux_capacity=0.5),
        ],
    )
    fp = FPFDMSolver(_problem(dirichlet_bc(value=0.0, dimension=1)), boundary_conditions=capped)
    with pytest.raises(NotImplementedError, match=r"segment\(s\) \['exit'\] carry a flux_capacity"):
        fp.prescribed_density(1)


def test_the_network_fp_refuses_to_name_what_its_step_holds_at_a_node_bc():
    """FPNetworkSolver's step holds an ABSORBING node at 0 whatever it is handed (#1478), and the protocol refuses to name
    it, by #2991, rather than answer None: a real answer would need the step's SOURCE time, its CUSTOM rules' reading of
    the field and its order of conditions. Without a node BC the step holds nothing and the protocol answers None. No
    cold-start route reaches a network with node BCs today (#2568, #2923)."""
    from mfgarchon.alg.numerical.network_solvers.fp_network import FPNetworkSolver
    from mfgarchon.extensions.topology import NetworkMFGProblem
    from mfgarchon.geometry.boundary.applicator_graph import GraphBCConfig, GraphBCType, NodeBC
    from mfgarchon.geometry.graph.network_geometry import GridNetwork

    def held_nodes(fp, n_nodes):
        outs = []
        for seed, shift in ((1, 0.0), (2, 5.0)):
            m0 = np.random.default_rng(seed).uniform(0.5, 2.0, size=n_nodes) + shift
            outs.append(np.asarray(fp.solve_fp_system(M_initial=m0, potential_field=np.zeros((11, n_nodes)))))
        a, b = outs
        return sorted({int(j) for j in np.flatnonzero((np.abs(a[1:] - b[1:]) <= 1e-12).all(axis=0))})

    for node_bc, held in (
        (GraphBCConfig(node_bcs=[NodeBC(nodes=[0], bc_type=GraphBCType.ABSORBING, value=0.0)]), [0]),
        (None, []),
    ):
        net = GridNetwork(width=3, height=3, boundary_conditions=node_bc)
        net.create_network()
        fp = FPNetworkSolver(NetworkMFGProblem(geometry=net, T=0.5, Nt=10))
        assert held_nodes(fp, 9) == held
        if node_bc is None:
            assert fp.prescribed_density(1) is None
        else:
            with pytest.raises(NotImplementedError, match=r"a SOURCE value read at t_n for level n \+ 1.*\(#2991\)"):
                fp.prescribed_density(1)


def test_a_solver_whose_step_prescribes_nothing_names_nothing():
    """FP-FDM on a no-flux BC holds no node; the protocol answers None, so the cold start is m_initial everywhere."""
    from mfgarchon.geometry.boundary import no_flux_bc

    fp, shape = _grid_fp((11,), no_flux_bc(dimension=1), own=False)
    held = _held_by_the_step(fp, shape)
    assert not any(len(held[n][0]) for n in held)
    assert all(fp.prescribed_density(n) is None for n in range(1, NT + 1))
