"""A coupled route's cold start holds, from level 1 on, the FP's own value at a node the FP's step prescribes (#2991).

The class: the iterate's later levels disagreeing with the FP's rows at the nodes the FP prescribes. FP-FDM's implicit
Dirichlet rows, and FP-FEM's condensed dofs, hold a value p at every later level whatever they are handed: 0 at the
problem's Dirichlet face, which the FP reads as an exit (#2512), and g(t_n) on the FP's own DIRICHLET(g). A cold start
put m_initial there, so its first sweep handed the HJB m_initial there, and a route that blends the FP's row into the
iterate with a weight theta < 1 kept prod(1 - theta_j)(m_initial - p) there after any number of sweeps.

The cold start now asks the FP (`BaseFPSolver.prescribed_density`), which answers with the read its own step makes. At
f9a41a31, with the walls planted at 7.0, the density the first sweep hands the HJB held 7.0 there on every route below,
and the density returned after one sweep at relaxation 0.5 held 3.5 = (1 - 0.5)(7 - 0), and 3.7 at the first level
against g(t_1) = 0.4 (3.3 off it), except RegimeSwitching's, whose first sweep takes the FP's row whole.
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
from mfgarchon.config.damping import DampingPolicy
from mfgarchon.core.hamiltonian import CongestionHamiltonian, QuadraticControlCost, SeparableHamiltonian
from mfgarchon.core.mfg_problem import MFGProblem
from mfgarchon.core.multi_population import MultiPopulationProblem
from mfgarchon.core.regime_switching import RegimeSwitchingConfig
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import BCSegment, BCType, BoundaryConditions, dirichlet_bc, no_flux_bc
from mfgarchon.geometry.meshes.mesh_1d import Mesh1D

NT, PLANT = 5, 7.0

# The fixtures' initial densities are planted, not normalised; the oracle's random inputs and potentials can take FP-FEM's
# Galerkin step negative; the network fixture leaves FPNetworkSolver's diffusion at its default.
pytestmark = [
    pytest.mark.filterwarnings("ignore:initial density mass:UserWarning"),
    pytest.mark.filterwarnings("ignore:FPFEMSolver. .* time levels hold a negative value:UserWarning"),
    pytest.mark.filterwarnings("ignore:FPNetworkSolver. network diffusion default:UserWarning"),
]


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


# (FP arm, the problem's BC, the FP's own BC or None, the HJB class, the FP class, a mesh, p at level n). The problem's
# face carries 0.3, which the FP reads as an exit, p = 0: a protocol that read the raw value would name 0.3.
ARMS = {
    "FP-FDM, the problem's Dirichlet face (an exit, p = 0)": (
        lambda: dirichlet_bc(value=0.3, dimension=1),
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
        lambda: dirichlet_bc(value=0.3, dimension=1),
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
            "FixedPoint": (FixedPointIterator, {"damping": DampingPolicy.constant(0.5)}),
            # Its learning rate is alpha_0 / (j + 1): the default alpha_0 = 1 takes the first FP row whole.
            "FictitiousPlay": (FictitiousPlayIterator, {"damping": DampingPolicy.fictitious_play(0.5)}),
            "BlockGaussSeidel": (BlockGaussSeidelIterator, {"damping": DampingPolicy.constant(0.5)}),
            "BlockJacobi": (BlockJacobiIterator, {"damping": DampingPolicy.constant(0.5)}),
        }[route]
        return np.asarray(cls(q, hjb_cls(q), fp(q), **kw).solve(max_iterations=iterations, verbose=False).M)
    if route == "MultiPopulation":
        q = problem()
        multi = MultiPopulationProblem(populations=[q], population_names=["a"])
        result = MultiPopulationIterator(multi, [hjb_cls(q)], [fp(q)], damping=DampingPolicy.constant(0.5))
        return np.asarray(result.solve(max_iterations=iterations, tolerance=1e-12).M[0])
    if route == "Graph":
        q = problem()
        coupling = AdjacencyCoupling(np.zeros((1, 1)), alpha=0.05, beta=0.02)
        solver = GraphMFGSolver(
            [q],
            coupling,
            [hjb_cls(q)],
            [fp(q)],
            max_iterations=iterations,
            tolerance=1e-12,
            damping=DampingPolicy.constant(0.5),
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
        damping=DampingPolicy.constant(0.5),
    )
    return np.asarray(solver.solve().densities[0])


ROUTES = ["FixedPoint", "FictitiousPlay", "BlockGaussSeidel", "BlockJacobi", "MultiPopulation", "Graph", "Regime"]
CASES = [
    *[(route, arm) for arm in list(ARMS)[:1] for route in [*ROUTES, "problem.solve"]],
    # RegimeSwitchingIterator refuses an FP's own BC; problem.solve has no FP-own-BC arm.
    *[(route, list(ARMS)[1]) for route in ROUTES if route != "Regime"],
    # The coupled FEM routes start from the iterate's owner, `IterationState.start` (#2997).
    *[(route, list(ARMS)[2]) for route in ("FixedPoint", "BlockGaussSeidel")],
]


@pytest.mark.parametrize(("route", "arm"), CASES, ids=[f"{route} | {arm}" for route, arm in CASES])
def test_the_cold_start_holds_the_fps_prescribed_value_at_its_later_levels(route, arm, monkeypatch):
    """At the walls, from level 1 on: the density the first sweep hands the HJB, and the density returned after one
    sweep, are the FP's prescribed value p(t_n), not the caller's 7.0. At level 0 the HJB is handed the caller's 7.0.

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
    assert np.all(first[0, walls] == PLANT), f"level 0 is the caller's density (#2984), handed {first[0, walls]}"
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
        FixedPointIterator(q, hjb, FPFDMSolver(q, boundary_conditions=bc), damping=DampingPolicy.constant(0.5))
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
        problems,
        config,
        hjbs,
        [FPFDMSolver(q) for q in problems],
        max_iterations=1,
        tolerance=1e-12,
        damping=DampingPolicy.constant(0.5),
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


def test_each_regime_asks_its_own_fp_and_the_inflow_reads_the_cold_start():
    """Regime 0's FP holds its exit walls at 0; regime 1's no-flux FP holds no node. The first sweep hands regime 0's HJB
    0 at the walls from level 1 on and regime 1's the caller's 7.0, and the inflow into regime 1, Q[0, 1] m_0, reads
    regime 0's cold start there: 0, not the caller's 7.0. The inflow is read off the source regime 1's FP is handed,
    exp(q_1 t) Q[0, 1] m_0, at each level t_1..T."""
    qs = [_problem(dirichlet_bc(value=0.3, dimension=1)), _problem(no_flux_bc(dimension=1))]
    Q = np.array([[-0.5, 0.5], [0.2, -0.2]])
    hjbs, fps = [HJBFDMSolver(q) for q in qs], [FPFDMSolver(q) for q in qs]
    handed: dict[int, np.ndarray] = {}
    for k, hjb in enumerate(hjbs):
        original = hjb.solve_hjb_system

        @functools.wraps(original)
        def recording(*args, _k=k, _original=original, **kwargs):
            handed.setdefault(_k, np.array(args[0], dtype=float, copy=True))
            return _original(*args, **kwargs)

        hjb.solve_hjb_system = recording
    sources = []
    original_fp = fps[1].solve_fp_system

    @functools.wraps(original_fp)
    def recording_fp(*args, **kwargs):
        sources.append(kwargs.get("source_term"))
        return original_fp(*args, **kwargs)

    fps[1].solve_fp_system = recording_fp
    RegimeSwitchingIterator(
        qs,
        RegimeSwitchingConfig(transition_matrix=Q),
        hjbs,
        fps,
        max_iterations=1,
        tolerance=1e-12,
        damping=DampingPolicy.constant(0.5),
    ).solve()
    walls = [0, -1]
    assert np.all(handed[0][1:, walls] == 0.0), handed[0][1:, walls].tolist()
    assert np.all(handed[1][1:, walls] == PLANT), handed[1][1:, walls].tolist()
    assert sources[0] is not None, "regime 1's FP was handed no inflow"
    x = qs[1].geometry.get_spatial_grid()
    read = [
        np.asarray(sources[0](n * qs[1].dt, x), dtype=float)[walls] / (np.exp(0.2 * n * qs[1].dt) * Q[0, 1])
        for n in range(1, NT + 1)
    ]
    assert np.max(np.abs(read)) <= 1e-12, [r.tolist() for r in read]


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


def test_fp_fdm_names_the_uncapped_dirichlet_node_and_not_the_capped_exit():
    """A capped exit's node is held at 0 only while its outflow stays under the cap; each step's active set decides its
    row (#2581), so FP-FDM does not name it, and names the uncapped Dirichlet node beside it. The step's two-input oracle
    agrees: the capped node differs between the inputs at every level, and x_min holds 0.4."""
    capped = BoundaryConditions(
        dimension=1,
        segments=[
            BCSegment(name="own", bc_type=BCType.DIRICHLET, value=0.4, boundary="x_min"),
            BCSegment(name="exit", bc_type=BCType.DIRICHLET, value=0.0, boundary="x_max", flux_capacity=0.05),
        ],
    )
    fp = FPFDMSolver(_problem(dirichlet_bc(value=0.3, dimension=1)), boundary_conditions=capped)
    held = _held_by_the_step(fp, (11,))
    for n in range(1, NT + 1):
        assert held[n][0].tolist() == [0], f"level {n}: the step holds {held[n][0].tolist()}"
        nodes, values = fp.prescribed_density(n)
        assert nodes.tolist() == [0]
        np.testing.assert_allclose(values, [0.4], rtol=0, atol=1e-15)


def test_the_network_fp_refuses_to_name_what_its_step_holds_at_a_node_bc():
    """FPNetworkSolver's step holds an ABSORBING node at 0 whatever it is handed (#1478), and the protocol refuses to name
    it, by #2991, rather than answer None: a real answer would need the step's SOURCE time, its CUSTOM rules' reading of
    the field and its order of conditions. Without a node BC the step holds nothing and the protocol answers None.
    GraphMFGSolver reaches a network with node BCs, with NetworkPolicyIterationHJBSolver, and this refusal fires there;
    the other cold-start routes refuse one before their cold start (#2568, #2923)."""
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
    fp, shape = _grid_fp((11,), no_flux_bc(dimension=1), own=False)
    held = _held_by_the_step(fp, shape)
    assert not any(len(held[n][0]) for n in held)
    assert all(fp.prescribed_density(n) is None for n in range(1, NT + 1))


def test_a_cold_start_refuses_an_fp_whose_class_does_not_define_the_protocol():
    """The cold start reads ``prescribed_density`` off the FP's class (``__class__``, which ``isinstance`` reads) and
    refuses by name, under the user's ruling of 2026-11-24, an FP whose class does not define it, where main solved
    with ``m_initial`` at every later level. A class that defines it without subclassing BaseFPSolver is asked."""
    from unittest.mock import Mock

    from mfgarchon.alg.numerical.coupling.fixed_point_utils import cold_start_density_at_level

    q = _problem(dirichlet_bc(value=0.0, dimension=1))
    m0 = _m_initial(np.linspace(0.0, 1.0, 11))

    class _Bare:
        problem = q

        def solve_fp_system(self, M_initial, drift_field=None, **kw):
            return np.stack([M_initial] * (NT + 1))

    class _Defines(_Bare):
        def prescribed_density(self, level):
            return np.array([0]), np.array([0.25])

    refusal = (
        r"_Bare does not define `prescribed_density`.*define it on the class, or forward it there to the solver it wraps "
        r"\(an instance attribute or `__getattr__` is not read\).*subclassing `BaseFPSolver` gives None, which is right "
        r"only where its step holds no node \(#2991\)"
    )
    with pytest.raises(TypeError, match=refusal):
        cold_start_density_at_level(m0, _Bare(), 1)
    on_the_instance = _Bare()
    on_the_instance.prescribed_density = lambda level: None
    with pytest.raises(TypeError, match=refusal):
        cold_start_density_at_level(m0, on_the_instance, 1)
    with pytest.raises(TypeError, match=refusal):
        FixedPointIterator(q, HJBFDMSolver(q), _Bare(), damping=DampingPolicy.constant(0.5)).solve(
            max_iterations=1, verbose=False
        )

    class _Forwards:
        """Forwards every attribute it lacks, ``prescribed_density`` included, through ``__getattr__``."""

        def __init__(self, inner):
            object.__setattr__(self, "_inner", inner)

        def __getattr__(self, name):
            return getattr(self._inner, name)

    forwarder = _Forwards(FPFDMSolver(q))
    assert forwarder.prescribed_density(1) is not None  # the instance answers; the class does not define it
    with pytest.raises(TypeError, match=r"_Forwards does not define `prescribed_density`.*`__getattr__` is not read"):
        cold_start_density_at_level(m0, forwarder, 1)

    held = cold_start_density_at_level(m0, _Defines(), 1)
    assert held[0] == 0.25
    np.testing.assert_array_equal(held[1:], m0[1:])
    spec_mock = Mock(spec=FPFDMSolver)
    spec_mock.prescribed_density.return_value = None
    assert cold_start_density_at_level(m0, spec_mock, 1) is m0


def test_a_wrapper_over_a_capped_fp_forwards_the_protocol_and_solves():
    """A user's wrapper that hides a capped FP-FDM's ``boundary_conditions`` is not seen by the coupled route's cap
    refusal (user ruling 2026-11-01). Under the user's ruling of 2026-11-24, on a cold start it defines or forwards
    ``prescribed_density``: forwarded, FP-FDM names no capped node, so the cold start keeps the caller's density there
    and the solve runs; a wrapper that does neither is refused by name by the cold start. This checks the solve and the
    cold start, not the cap."""
    q = _problem(dirichlet_bc(value=0.0, dimension=1))
    capped = BoundaryConditions(
        dimension=1,
        segments=[
            BCSegment(name="x_min", bc_type=BCType.NO_FLUX, boundary="x_min"),
            BCSegment(name="exit", bc_type=BCType.DIRICHLET, value=0.0, boundary="x_max", flux_capacity=0.5),
        ],
    )

    class _Hides:
        def __init__(self, inner):
            object.__setattr__(self, "_inner", inner)

        def __getattr__(self, name):
            if name == "boundary_conditions":
                raise AttributeError(name)
            return getattr(self._inner, name)

    class _Forwards(_Hides):
        def prescribed_density(self, level):
            return self._inner.prescribed_density(level)

    with pytest.raises(TypeError, match=r"_Hides does not define `prescribed_density`.*forward it"):
        FixedPointIterator(
            q, HJBFDMSolver(q), _Hides(FPFDMSolver(q, boundary_conditions=capped)), damping=DampingPolicy.constant(0.5)
        ).solve(max_iterations=1, verbose=False)
    handed: list[np.ndarray] = []
    hjb = HJBFDMSolver(q)
    original = hjb.solve_hjb_system

    @functools.wraps(original)
    def recording(*args, **kwargs):
        handed.append(np.array(args[0], dtype=float, copy=True))
        return original(*args, **kwargs)

    hjb.solve_hjb_system = recording
    fp = _Forwards(FPFDMSolver(q, boundary_conditions=capped))
    assert all(fp.prescribed_density(n) is None for n in range(1, NT + 1))
    M = np.asarray(
        FixedPointIterator(q, hjb, fp, damping=DampingPolicy.constant(0.5)).solve(max_iterations=1, verbose=False).M
    )
    assert np.all(np.isfinite(M))
    assert np.all(handed[0][1:, -1] == PLANT), handed[0][1:, -1].tolist()


def test_the_cold_start_holds_the_prescribed_values_whatever_the_initial_densitys_memory_layout():
    """The protocol's indices are C-order flat positions in one level's array. An initial density handed in Fortran order,
    or as a transposed view, gets the same values at the same nodes as a C-order one: on the 9 x 7 grid whose x_min holds
    0.4 and y_min 0.9, through the level helper, which the iterate's owner (`IterationState.start`, #2997) reads at each
    later level, and `cold_start_iterate`."""
    from mfgarchon.alg.numerical.coupling.fixed_point_utils import cold_start_density_at_level, cold_start_iterate

    fp, shape = LAYOUTS["FP-FDM, 9 x 7, its own DIRICHLET 0.4 on x_min and 0.9 on y_min, no-flux elsewhere"]()
    for level in range(1, NT + 1):
        nodes, values = fp.prescribed_density(level)
        expected = np.full(shape, PLANT)
        expected.reshape(-1)[nodes] = values
        assert sorted(set(np.round(values, 12))) == [0.4, 0.9]
        for layout, m0 in (
            ("C order", np.full(shape, PLANT)),
            ("Fortran order", np.asfortranarray(np.full(shape, PLANT))),
            ("a transposed view", np.full(shape[::-1], PLANT).T),
        ):
            np.testing.assert_array_equal(cold_start_density_at_level(m0, fp, level), expected, err_msg=layout)
            np.testing.assert_array_equal(cold_start_iterate(m0, fp, NT)[level], expected, err_msg=layout)


def test_each_population_and_graph_node_asks_its_own_fp():
    """Two populations, and two graph nodes, on one grid: a's FP holds its exit walls at 0, b's no-flux FP holds no node.
    The first sweep hands a's HJB 0 at the walls from level 1 on and b's HJB the caller's 7.0, so a cold start that asked
    one member's FP for another's density would hand b 0."""
    qa, qb = _problem(dirichlet_bc(value=0.3, dimension=1)), _problem(no_flux_bc(dimension=1))
    walls = [0, -1]

    def recorded(hjb):
        seen = []
        original = hjb.solve_hjb_system

        @functools.wraps(original)
        def recording(*args, **kwargs):
            seen.append(np.array(args[0] if args else kwargs["M_density"], dtype=float, copy=True))
            return original(*args, **kwargs)

        hjb.solve_hjb_system = recording
        return hjb, seen

    (ha, seen_a), (hb, seen_b) = recorded(HJBFDMSolver(qa)), recorded(HJBFDMSolver(qb))
    multi = MultiPopulationProblem(populations=[qa, qb], population_names=["a", "b"])
    MultiPopulationIterator(
        multi, [ha, hb], [FPFDMSolver(qa), FPFDMSolver(qb)], damping=DampingPolicy.constant(0.5)
    ).solve(max_iterations=1, tolerance=1e-12)
    (ga, graph_a), (gb, graph_b) = recorded(HJBFDMSolver(qa)), recorded(HJBFDMSolver(qb))
    coupling = AdjacencyCoupling(np.zeros((2, 2)), alpha=0.05, beta=0.02)
    GraphMFGSolver(
        [qa, qb],
        coupling,
        [ga, gb],
        [FPFDMSolver(qa), FPFDMSolver(qb)],
        max_iterations=1,
        tolerance=1e-12,
        damping=DampingPolicy.constant(0.5),
    ).solve()
    for route, a, b in (("MultiPopulation", seen_a, seen_b), ("Graph", graph_a, graph_b)):
        assert a, f"{route}: the recorder saw no HJB call for a"
        assert b, f"{route}: the recorder saw no HJB call for b"
        first_a, first_b = (x.reshape(NT + 1, -1) for x in (a[0], b[0]))
        assert np.all(first_a[1:, walls] == 0.0), f"{route}: a was handed {first_a[1:, walls].tolist()}"
        assert np.all(first_b[1:, walls] == PLANT), f"{route}: b was handed {first_b[1:, walls].tolist()}"


@pytest.mark.parametrize("hamiltonian", ["quadratic", "congestion"])
def test_a_first_regime_sweep_with_the_regimes_decoupled_is_a_fixed_point_sweep(hamiltonian):
    """With Q = 0 the regimes decouple: no cross term and an integrating factor of 1, so one RegimeSwitchingIterator sweep
    at damping 1 is one FixedPoint sweep at relaxation 1 on each regime, on the problem's exit. CongestionHamiltonian's
    FP drift is on the velocity route, which reads the density trajectory the first sweep hands it (#2905): at f9a41a31
    the router was handed the single initial density there, and regime 0's first-sweep M was 9.232e-02 off FixedPoint's.
    The quadratic H, on the value route, agrees at both trees. At damping 0.5 the first sweep is damped in M and judged,
    as FixedPoint's is (#2997): a tolerance it meets stops both routes there, with one iterate."""

    def problem(centre):
        H = (
            SeparableHamiltonian(QuadraticControlCost(control_cost=1.0), coupling=lambda m: 0.5 * m)
            if hamiltonian == "quadratic"
            else CongestionHamiltonian(
                QuadraticControlCost(control_cost=1.0), congestion_factor=lambda m: 1.0 + 0.5 * m
            )
        )
        return MFGProblem(
            model=Model(hamiltonian=H, volatility=0.3),
            domain=TensorProductGrid(
                bounds=[(0.0, 1.0)], Nx_points=[21], boundary_conditions=dirichlet_bc(value=0.3, dimension=1)
            ),
            conditions=Conditions(
                m_initial=lambda x: 0.2 + np.exp(-40 * (_coordinate(x) - centre) ** 2),
                u_terminal=lambda x: 2.0 * (_coordinate(x) - 0.8) ** 2,
                T=0.5,
            ),
            Nt=10,
        )

    qs = [problem(c) for c in (0.3, 0.6)]
    regime = RegimeSwitchingIterator(
        qs,
        RegimeSwitchingConfig(transition_matrix=np.zeros((2, 2))),
        [HJBFDMSolver(q) for q in qs],
        [FPFDMSolver(q) for q in qs],
        max_iterations=1,
        tolerance=1e-12,
        damping=DampingPolicy.constant(1.0),
    ).solve()
    for k, q in enumerate(qs):
        fixed = FixedPointIterator(q, HJBFDMSolver(q), FPFDMSolver(q), damping=DampingPolicy.constant(1.0)).solve(
            max_iterations=1, verbose=False
        )
        np.testing.assert_allclose(np.asarray(regime.densities[k]), np.asarray(fixed.M), rtol=0, atol=1e-12)
        np.testing.assert_allclose(np.asarray(regime.values[k]), np.asarray(fixed.U), rtol=0, atol=1e-12)
    # The first sweep is damped and judged as FixedPoint's is (#2997 ruling 2, the approver's ruling of 2026-11-25): at
    # damping 0.5 and a tolerance the sweep meets, each regime stops at its first sweep with FixedPoint's iterate. At
    # 4cd1803f it took the FP's density whole and was not judged, so it did not stop.
    fresh = [problem(c) for c in (0.3, 0.6)]
    damped = RegimeSwitchingIterator(
        fresh,
        RegimeSwitchingConfig(transition_matrix=np.zeros((2, 2))),
        [HJBFDMSolver(q) for q in fresh],
        [FPFDMSolver(q) for q in fresh],
        max_iterations=1,
        tolerance=1e9,
        damping=DampingPolicy.constant(0.5),
    ).solve()
    assert damped.converged
    assert damped.iterations == 1
    for k, q in enumerate(fresh):
        fixed = FixedPointIterator(q, HJBFDMSolver(q), FPFDMSolver(q), damping=DampingPolicy.constant(0.5)).solve(
            max_iterations=1, tolerance=1e9, verbose=False
        )
        assert fixed.converged
        np.testing.assert_allclose(np.asarray(damped.densities[k]), np.asarray(fixed.M), rtol=0, atol=1e-12)
        np.testing.assert_allclose(np.asarray(damped.values[k]), np.asarray(fixed.U), rtol=0, atol=1e-12)
