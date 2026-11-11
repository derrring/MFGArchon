"""What `optimal_control` returns has p's shape, wherever the library reads it (#2875, stage 3, first slice).

`HamiltonianBase.optimal_control` and `ControlCostBase.optimal_control` document "same shape as p". Every library read
of what `optimal_control` or `evaluate_optimal_control` returns passes it through one owner, `checked_control`, which
refuses another shape and names the reader, the source and its method, the shapes of p and m handed, and the shape
that came back. Before it, each read decided alone (main bfe6de4d): FEM and meshless Galerkin took a wrong shape
silently, the network solvers read the first n entries of a longer return, FVM and the velocity route reshaped or
ravelled any return of p's size, the velocity route's nD read and `utils/manufactured.py` checked with messages of their
own, and `cone_policy` raised numpy's `IndexError`.

- **Each read refuses another shape** (a mutation-verified pin). Two rows per read: a return of p's size in another
  shape, and a return of another size. The message is pinned whole up to the shape that came back: the reader, the
  source and its method, p's shape and the m clause. A read that calls past the owner, or that reshapes a same-size
  return or truncates a longer one before handing it on, fails its own rows.
- **Each read takes the library's control** (a mutation-verified pin). Through the same fixture, so a refusal is the
  form's and not the fixture's. A read that hands the owner a p other than the one it handed the control fails here.
- **The 0-d admission** (a mutation-verified pin). *Not yet met (#2875): the form of p.* FVM's 1-D wrap face hands a 0-d
  p, and the library's own control answers (1,). The owner admits it and returns it 0-d; stage 3 deletes this.
- **The owner alone** (a mutation-verified pin): it refuses (1,) for a p that is not 0-d, (2,) for a 0-d p, and (1,) for
  a p of shape (1, 1).
"""

from __future__ import annotations

import logging
import re
import warnings

import pytest

import numpy as np

from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian, checked_control
from mfgarchon.core.mfg_components import MFGComponents
from mfgarchon.core.mfg_problem import MFGProblem
from mfgarchon.geometry import Mesh1D, TensorProductGrid
from mfgarchon.geometry.boundary import no_flux_bc, periodic_bc

T, NT, SIGMA = 0.5, 10, 0.3

#: The forms a test control can take on the library's own return ``a`` for ``p``.
FORMS = {
    "library": lambda a, p: a,
    "lead": lambda a, p: np.asarray(a)[None],  # the same size, a leading unit axis
    "double": lambda a, p: np.concatenate([np.ravel(a), np.ravel(a)]),  # another size
    "transposed": lambda a, p: np.asarray(a).T,
}


def _separable(form, *, only_0d=False, only_rows=None):
    """A SeparableHamiltonian whose control is the library's in another form. ``only_0d``: only where p is 0-d;
    ``only_rows``: only where p has that many rows, so a reader's other reads stay the library's."""

    class Formed(SeparableHamiltonian):
        def optimal_control(self, t, x, p, m):
            alpha = super().optimal_control(t=t, x=x, p=p, m=m)
            if (only_0d and np.ndim(p) != 0) or (only_rows is not None and np.shape(p)[0] != only_rows):
                return alpha
            return FORMS[form](alpha, p)

    return Formed(control_cost=QuadraticControlCost(control_cost=1.0))


def _quiet(fn):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        logging.disable(logging.CRITICAL)
        try:
            return fn()
        finally:
            logging.disable(logging.NOTSET)


def _components(h, dim=1, level=1.0):
    # A constant density of mass 1 on the problem's measure; the FP-only reads pass their own m0.
    return MFGComponents(
        m_initial=lambda x: np.full_like(np.asarray(x[0], dtype=float), level),
        u_terminal=lambda x: 0.0 * np.asarray(x[0], dtype=float),
        hamiltonian=h,
    )


def _grid_problem(h, n, dim=1, bc=None):
    bc = no_flux_bc(dimension=dim) if bc is None else bc
    grid = TensorProductGrid(bounds=[(0.0, 1.0)] * dim, Nx_points=[n] * dim, boundary_conditions=bc)
    return MFGProblem(geometry=grid, T=T, Nt=NT, volatility=SIGMA, components=_components(h, dim))


def _u(x):
    return np.stack([0.5 * (x - 0.3) ** 2 * (1 + 0.1 * k) for k in range(NT + 1)])


def _m0(x):
    return np.exp(-20 * (x - 0.5) ** 2)


def _fem(h):
    from mfgarchon.alg.numerical.fem.fp_fem_solver import FPFEMSolver

    mesh = Mesh1D(bounds=(0.0, 1.0), num_elements=14)
    mesh.generate_mesh()
    mesh.boundary_conditions = no_flux_bc(dimension=1)
    # A 1-D mesh is measured on uniform cells, one per node: 15 of width 1/14, so mass 1 is the level 14/15.
    problem = MFGProblem(geometry=mesh, T=T, Nt=NT, volatility=SIGMA, components=_components(h, level=14 / 15))
    solver = FPFEMSolver(problem, order=1)
    x = solver._disc.dof_coordinates[:, 0]
    return solver.solve_fp_system(_m0(x), potential_field=_u(x))


def _meshless(h):
    from mfgarchon.alg.numerical.meshless_galerkin.fp_solver import MeshlessGalerkinFPSolver

    x = np.linspace(0.0, 1.0, 15)
    solver = MeshlessGalerkinFPSolver(_grid_problem(h, 15), collocation_points=x[:, None], delta=3.5 / 14)
    return solver.solve_fp_system(_m0(x), potential_field=_u(x))


def _particle(h):
    from mfgarchon.alg.numerical.fp_solvers import FPParticleSolver

    x = np.linspace(0.0, 1.0, 15)
    solver = FPParticleSolver(_grid_problem(h, 15), num_particles=500, seed=2)
    return solver.solve_fp_system(_m0(x), potential_field=_u(x), show_progress=False)


def _fvm(h, bc=None):
    from mfgarchon.alg.numerical.fp_solvers.fp_fvm import FPFVMSolver

    problem = _grid_problem(h, 15, bc=bc)
    x = problem.geometry.coordinates[0]
    return FPFVMSolver(problem).solve_fp_system(_m0(x), potential_field=_u(x), show_progress=False)


def _fvm_periodic(h):
    return _fvm(h, bc=periodic_bc(dimension=1))


def _network_problem(form):
    pytest.importorskip("igraph")
    from mfgarchon.extensions.topology import NetworkMFGProblem
    from mfgarchon.geometry.graph.network_geometry import GridNetwork

    net = GridNetwork(width=3, height=3)
    net.create_network()
    problem = NetworkMFGProblem(geometry=net, T=T, Nt=NT)
    inner = problem.hamiltonian_class

    class Formed:
        def optimal_control(self, x, m, p, t):
            return FORMS[form](inner.optimal_control(x=x, m=m, p=p, t=t), p)

        def __getattr__(self, name):
            return getattr(inner, name)

    problem.components._hamiltonian_class = Formed()  # the backing field of MFGProblem.hamiltonian_class
    return problem


def _network_fp(form):
    from mfgarchon.alg.numerical.network_solvers.fp_network import FPNetworkSolver

    problem = _network_problem(form)
    n = problem.num_nodes
    U = np.stack([np.linspace(0.0, 2.0, n) * (1 + 0.1 * k) for k in range(NT + 1)])
    return FPNetworkSolver(problem, scheme="explicit").solve_fp_system(np.full(n, 1.0 / n), U, show_progress=False)


def _hjb_network(form):
    from mfgarchon.alg.numerical.network_solvers.hjb_network import NetworkPolicyIterationHJBSolver

    problem = _network_problem(form)
    n = problem.num_nodes
    return NetworkPolicyIterationHJBSolver(problem).solve_hjb_system(
        M_density=np.full((NT + 1, n), 1.0 / n), U_terminal=np.linspace(0.0, 1.0, n)
    )


def _velocity(h, dim, n):
    from mfgarchon.alg.numerical.coupling.fixed_point_utils import compute_fp_velocity_field

    problem = MFGProblem(
        geometry=TensorProductGrid(
            bounds=[(0.0, 1.0)] * dim, Nx_points=[n] * dim, boundary_conditions=no_flux_bc(dimension=dim)
        ),
        T=0.5,
        Nt=4,
        volatility=SIGMA,
        components=_components(h, dim),
    )
    pts = np.meshgrid(*[np.linspace(0, 1, n)] * dim, indexing="ij")
    U = np.stack([0.5 * sum((c - 0.3) ** 2 for c in pts) * (1 + 0.1 * k) for k in range(5)])
    M = np.stack([1.0 + 0.5 * pts[0] * (1 + 0.1 * k) for k in range(5)])
    return compute_fp_velocity_field(problem, U, M, h)


def _pair():
    from mfgarchon.utils.manufactured import ManufacturedPair

    a1, a2, c = (lambda t: 1.0 + 0.1 * t), (lambda t: 0.4 * np.cos(t)), np.pi
    return ManufacturedPair(
        u=lambda t, x: a1(t) * (np.cos(c * x[0]) + 0.5 * np.cos(c * x[1])),
        u_t=lambda t, x: 0.1 * (np.cos(c * x[0]) + 0.5 * np.cos(c * x[1])),
        grad_u=lambda t, x: np.stack([-a1(t) * c * np.sin(c * x[0]), -0.5 * a1(t) * c * np.sin(c * x[1])], axis=-1),
        hess_u=lambda t, x: np.zeros((*np.shape(x[0]), 2, 2)),
        m=lambda t, x: 1.0 + a2(t) * np.cos(c * x[0]) * np.cos(c * x[1]),
        m_t=lambda t, x: -0.4 * np.sin(t) * np.cos(c * x[0]) * np.cos(c * x[1]),
        grad_m=lambda t, x: np.zeros((*np.shape(x[0]), 2)),
        hess_m=lambda t, x: np.zeros((*np.shape(x[0]), 2, 2)),
        name="shape_2875",
    )


def _manufactured_source(form):
    from mfgarchon.utils.manufactured import fp_source

    points = np.stack(np.meshgrid(np.linspace(0, 1, 4), np.linspace(0, 1, 4), indexing="ij"), axis=-1).reshape(-1, 2)
    # Formed only at the 16 evaluation points: the source's build-time probe (4 rows) stays the library's.
    return fp_source(_pair(), _separable(form, only_rows=len(points)), SIGMA)(0.7, points)


def _cone_policy(form):
    from mfgarchon.alg.numerical.gfdm_components.positive_qp_fan import cone_policy

    class Formed(QuadraticControlCost):
        def optimal_control(self, p):
            return FORMS[form](super().optimal_control(p), p)

    # One sector at node 0, spanned by its two neighbours along the axes.
    sectors = {
        "node": np.array([0]),
        "j0": np.array([1]),
        "j1": np.array([2]),
        "z0": np.array([[1.0, 0.0]]),
        "z1": np.array([[0.0, 1.0]]),
        "zinv": np.eye(2)[None],
    }
    return cone_policy(np.array([0.0, 1.0, 2.0]), sectors, Formed(control_cost=1.0))


def _manufactured_coefficient(form):
    from mfgarchon.utils.manufactured import _drift_coefficient

    return _drift_coefficient(_separable(form), 2)


#: read -> (the reader the refusal names, the method it names, how it is run with a control of a form, the m clause it
#: names, and two (form, shape of p handed, shape that came back): one of p's size, one of another size). At bfe6de4d each
#: read took each of these forms silently (wrong, or right by accident), refused it with a message of its own, or raised
#: a numpy error; the PR's revert run lists which.
READS = {
    "FEM": (
        "FPFEMSolver",
        "optimal_control",
        lambda f: _fem(_separable(f)),
        "m None",
        [("transposed", (1, 14, 2), (2, 14, 1)), ("double", (1, 14, 2), (56,))],
    ),
    "meshless Galerkin": (
        "MeshlessGalerkinFPSolver",
        "optimal_control",
        lambda f: _meshless(_separable(f)),
        "m None",
        [("transposed", (15, 1), (1, 15)), ("double", (15, 1), (30,))],
    ),
    "particle": (
        "FPParticleSolver",
        "optimal_control",
        lambda f: _particle(_separable(f)),
        "m None",
        [("lead", (500,), (1, 500)), ("double", (500,), (1000,))],
    ),
    "FVM face": (
        "FPFVMSolver",
        "optimal_control",
        lambda f: _fvm(_separable(f)),
        "m of shape (14,)",
        [("lead", (14,), (1, 14)), ("double", (14,), (28,))],
    ),
    "FVM wrap face": (
        "FPFVMSolver",
        "optimal_control",
        lambda f: _fvm_periodic(_separable(f, only_0d=True)),
        "m of shape ()",
        [("lead", (), (1, 1)), ("double", (), (2,))],
    ),
    "network FP": (
        "FPNetworkSolver",
        "optimal_control",
        _network_fp,
        "m of shape (9,)",
        [("lead", (9,), (1, 9)), ("double", (9,), (18,))],
    ),
    "velocity route 1-D": (
        "compute_fp_velocity_field",
        "evaluate_optimal_control",
        lambda f: _velocity(_separable(f), 1, 15),
        "m of shape (14,)",
        [("transposed", (14, 1), (1, 14)), ("double", (14, 1), (28,))],
    ),
    "velocity route nD": (
        "compute_fp_velocity_field",
        "evaluate_optimal_control",
        lambda f: _velocity(_separable(f), 2, 7),
        "m of shape (42,)",
        [("lead", (42, 2), (1, 42, 2)), ("double", (42, 2), (168,))],
    ),
    "network HJB": (
        "NetworkPolicyIterationHJBSolver",
        "optimal_control",
        _hjb_network,
        "m of shape (9,)",
        [("lead", (9,), (1, 9)), ("double", (9,), (18,))],
    ),
    "manufactured coefficient": (
        "manufactured._drift_coefficient",
        "optimal_control",
        _manufactured_coefficient,
        "m of shape (4,)",
        [("lead", (4, 2), (1, 4, 2)), ("double", (4, 2), (16,))],
    ),
    "manufactured source": (
        "manufactured.fp_source",
        "optimal_control",
        _manufactured_source,
        "m of shape (16,)",
        [("lead", (16, 2), (1, 16, 2)), ("double", (16, 2), (64,))],
    ),
    "cone_policy": (
        "cone_policy",
        "optimal_control",
        _cone_policy,
        "m None",
        [("transposed", (1, 2), (2, 1)), ("double", (1, 2), (4,))],
    ),
}
REFUSALS = [(read, form) for read, row in READS.items() for form, _, _ in row[4]]


@pytest.mark.parametrize(("read", "form"), REFUSALS)
def test_each_read_refuses_a_control_of_another_shape(read, form):
    reader, method, run, m_clause, cases = READS[read]
    p_shape, got_shape = next((p, got) for f, p, got in cases if f == form)
    message = (
        rf"^{re.escape(reader)}: Formed\.{method} was handed p of shape {re.escape(str(p_shape))} and "
        rf"{re.escape(m_clause)}, and returned shape {re.escape(str(got_shape))}\."
    )
    with pytest.raises(ValueError, match=message):
        _quiet(lambda: run(form))


@pytest.mark.parametrize("read", list(READS))
def test_each_read_takes_the_librarys_control(read):
    """The control every reader must accept, through the same fixture: the refusal above is the form's, not the
    fixture's."""
    _, _, run, _, _ = READS[read]
    result = _quiet(lambda: run("library"))
    assert result is not None


def test_a_0d_p_may_come_back_as_one_value():
    """*Not yet met (#2875):* FVM's 1-D wrap face hands a 0-d p, and the library's control answers (1,). Admitted, and
    returned 0-d, until stage 3 sets the form of p."""
    library = SeparableHamiltonian(control_cost=QuadraticControlCost(control_cost=1.0))
    assert library.optimal_control(t=0.0, x=(0.5,), p=np.float64(2.0), m=1.0).shape == (1,)
    got = checked_control(np.array([-2.0]), np.float64(2.0), reader="here", source=library)
    assert got.shape == ()
    assert got == -2.0
    assert np.all(np.isfinite(np.asarray(_quiet(lambda: _fvm_periodic(library)))))


@pytest.mark.parametrize(
    ("returned", "p"),
    [(np.array([-2.0]), np.zeros(3)), (np.zeros(2), np.float64(2.0)), (np.array([-2.0]), np.zeros((1, 1)))],
    ids=["(1,) for p (3,)", "(2,) for a 0-d p", "(1,) for p (1, 1)"],
)
def test_the_owner_refuses_what_the_admission_does_not_cover(returned, p):
    message = rf"^here: object\.optimal_control was handed p of shape {re.escape(str(np.shape(p)))} and m None, "
    with pytest.raises(ValueError, match=message):
        checked_control(returned, p, reader="here", source=object())
