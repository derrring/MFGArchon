"""The FP velocity is built at the faces FP-FDM's divergence-upwind stencil reads it at (#2904).

FP-FDM's stencil reads ``interface_velocity[d][i]`` as the velocity at the face ``(i + 1/2)`` along ``d``, the node
across it being ``i + 1``, wrapped through `periodic_axis_span` on an axis that wraps. `compute_fp_velocity_field` built
another field: on a periodic 1-D grid its last face read the endpoint the grid repeats instead of node 0, and in nD it
returned node values of centred differences. FP-FDM's velocity route then carried another drift than its
value-function route, and nothing raised: at 8f7c0db8 the first test's cells are 2.093 apart in M on the 1-D periodic
grid, 7.790 on the 2-D periodic one and 14.81 on the 3-D periodic one.

Three oracles, each for what the other two cannot see:

- The quadratic control cost, whose drift both routes carry: the velocity route equals the value-function route in M,
  to round-off, on each wall kind, dimension and volatility kind below. A frozen U that is not symmetric, has a normal
  derivative at every wall, and takes other values at the two ends of each axis.
- The library controls the router hands the builder, L1's sign, the bounded clip and congestion's ``-p / c(m)``, read
  only the ``d``-th component of ``p`` at their face. On a U of total degree 2 the forward difference across a face is
  the exact gradient at the face, so the velocity is the control at the exact gradient and the face's mean density;
  across the wrap, ``p`` is the difference of U from the last node before the repeated one to node 0. The solver is
  given its own BC, of the other wrap than its problem's, so the velocity must be built on the solver's walls.
- A control that couples the components of ``p`` reads the transverse ones too, which the builder takes as the mean
  of the two nodes' centred differences: for a user-written isotropic control, the velocity converges at second order
  to the control at the exact gradient at the face. No library control couples them.
"""

from __future__ import annotations

import inspect
import warnings

import pytest

import numpy as np

from mfgarchon import Conditions, MFGProblem, Model
from mfgarchon.alg.numerical.coupling.fixed_point_utils import compute_fp_velocity_field, resolve_fp_drift_kwargs
from mfgarchon.alg.numerical.fp_solvers import FPFDMSolver
from mfgarchon.alg.numerical.fp_solvers.base_fp import BaseFPSolver
from mfgarchon.alg.numerical.hjb_solvers import HJBFDMSolver
from mfgarchon.core.hamiltonian import (
    BoundedControlCost,
    CongestionHamiltonian,
    HamiltonianBase,
    L1ControlCost,
    QuadraticControlCost,
    SeparableHamiltonian,
)
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import (
    BCSegment,
    BCType,
    BoundaryConditions,
    dirichlet_bc,
    neumann_bc,
    no_flux_bc,
    periodic_bc,
    robin_bc,
)
from mfgarchon.geometry.boundary.types import BoundaryFace, PeriodicGridConvention

L1_LAMBDA = 0.53
MAX_CONTROL = 0.8
CONGESTION = 2.0


def _walls(kind, dimension):
    if kind == "no-flux":
        return no_flux_bc(dimension=dimension)
    if kind == "periodic":
        return periodic_bc(dimension=dimension)
    if kind == "Dirichlet":
        return dirichlet_bc(0.0, dimension=dimension)
    # mixed: no-flux on the first axis, Dirichlet on the others
    segments = []
    for axis in range(dimension):
        for side in ("min", "max"):
            face = BoundaryFace(axis, side).to_string()
            if axis == 0:
                segments.append(BCSegment(name=face, bc_type=BCType.NO_FLUX, boundary=face))
            else:
                segments.append(BCSegment(name=face, bc_type=BCType.DIRICHLET, boundary=face, value=0.0))
    return BoundaryConditions(dimension=dimension, segments=segments)


def _problem(hamiltonian, dimension, n, walls, volatility=0.3, volatility_kind=None, Nt=6):
    grid = TensorProductGrid(bounds=[(0.0, 1.0)] * dimension, Nx_points=[n] * dimension, boundary_conditions=walls)
    kind = {"volatility_kind": volatility_kind} if volatility_kind else {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return MFGProblem(
            model=Model(hamiltonian=hamiltonian, volatility=volatility, **kind),
            domain=grid,
            conditions=Conditions(m_initial=lambda x: 1.0, u_terminal=lambda x: 0.0, T=0.5),
            Nt=Nt,
        )


@pytest.mark.parametrize(
    ("dimension", "walls", "n", "field"),
    [
        (1, "no-flux", 21, False),
        (1, "Dirichlet", 21, False),
        (1, "periodic", 21, False),
        (1, "periodic", 21, True),
        (2, "no-flux", 9, False),
        (2, "Dirichlet", 9, False),
        (2, "periodic", 9, False),
        (2, "mixed", 9, False),
        (2, "no-flux", 9, True),
        (3, "no-flux", 5, False),
        (3, "periodic", 5, False),
    ],
)
def test_the_velocity_route_carries_the_value_route_drift(dimension, walls, n, field):
    hamiltonian = SeparableHamiltonian(QuadraticControlCost(control_cost=1.0))
    x = np.meshgrid(*[np.linspace(0.0, 1.0, n)] * dimension, indexing="ij")
    volatility = 0.2 + 0.1 * sum(x) if field else 0.3
    problem = _problem(hamiltonian, dimension, n, _walls(walls, dimension), volatility, "field" if field else None)
    u = 0.3 * np.sin(2 * np.pi * x[0]) + 0.2 * x[0]
    for k in range(1, dimension):
        u = u + 0.3 * (k + 1) * x[k] + 0.1 * x[0] * x[k]
    U = np.stack([u] * 7)
    m0 = 1.0 + 0.5 * np.cos(2 * np.pi * x[0])
    fp = FPFDMSolver(problem)
    by_value = np.asarray(fp.solve_fp_system(m0, potential_field=U, show_progress=False))
    velocity = compute_fp_velocity_field(
        problem, U, np.stack([m0] * 7), hamiltonian, boundary_conditions=fp.boundary_conditions
    )
    by_velocity = np.asarray(fp.solve_fp_system(m0, drift_field=velocity, show_progress=False))
    np.testing.assert_allclose(by_velocity, by_value, rtol=0.0, atol=1e-12)


A = (1.5, 0.7, 0.9)
B = (-0.9, 0.3, -0.5)
C = {(0, 1): 0.6, (0, 2): 0.4, (1, 2): -0.3}
STEPS = np.arange(5)


def _degree_two(x):
    """U of total degree 2, scaled by (1 + 0.1 n) at step n, and its gradient."""
    dimension = len(x)
    u = sum(A[k] * x[k] ** 2 + B[k] * x[k] for k in range(dimension))
    u = u + sum(c * x[k] * x[q] for (k, q), c in C.items() if q < dimension)
    gradient = [
        2 * A[k] * x[k]
        + B[k]
        + sum(c * x[q if j == k else j] for (j, q), c in C.items() if q < dimension and k in (j, q))
        for k in range(dimension)
    ]
    scale = (1.0 + 0.1 * STEPS).reshape(-1, *[1] * dimension)
    return scale * u, [scale * g for g in gradient]


def _density(x):
    return 0.2 + x[0] ** 2 + 0.3 * x[-1] + 0.1 * x[0] * x[-1] + 0.05 * STEPS.reshape(-1, *[1] * len(x))


CONTROLS = {
    "L1": (
        lambda: SeparableHamiltonian(L1ControlCost(lambda_=L1_LAMBDA)),
        lambda p, m: np.where(np.abs(p) > L1_LAMBDA, -np.sign(p), 0.0),
    ),
    "bounded": (
        lambda: SeparableHamiltonian(BoundedControlCost(control_cost=1.0, max_control=MAX_CONTROL)),
        lambda p, m: np.clip(-p, -MAX_CONTROL, MAX_CONTROL),
    ),
    "congestion": (
        lambda: CongestionHamiltonian(
            control_cost=QuadraticControlCost(control_cost=1.0),
            congestion_factor=lambda m: 1.0 + CONGESTION * np.asarray(m),
        ),
        lambda p, m: -p / (1.0 + CONGESTION * m),
    ),
}


@pytest.mark.parametrize("control", sorted(CONTROLS))
@pytest.mark.parametrize(
    ("own", "problems"), [("periodic", "no-flux"), ("no-flux", "periodic"), ("Dirichlet", "periodic")]
)
@pytest.mark.parametrize("dimension", [1, 2, 3])
def test_each_library_control_is_read_at_the_exact_gradient_at_its_face(dimension, own, problems, control):
    make, closed_form = CONTROLS[control]
    hamiltonian = make()
    n = 11 if dimension == 1 else 9
    h = 1.0 / (n - 1)
    problem = _problem(hamiltonian, dimension, n, _walls(problems, dimension), Nt=len(STEPS) - 1)
    fp = FPFDMSolver(problem, boundary_conditions=_walls(own, dimension))
    x = np.meshgrid(*[np.linspace(0.0, 1.0, n)] * dimension, indexing="ij")
    U, _ = _degree_two(x)
    M = _density(x)
    kwargs, _ = resolve_fp_drift_kwargs(
        problem,
        set(inspect.signature(fp.solve_fp_system).parameters),
        None,
        U,
        M,
        h_class=hamiltonian,
        drift_convention=fp._drift_convention,
        fp_solver=fp,
    )
    velocity = kwargs["drift_field"]

    wraps = own == "periodic"
    expected = np.zeros((len(STEPS), dimension, *[n] * dimension))
    for d in range(dimension):
        at_face = [x[k] + 0.5 * h * (k == d) for k in range(dimension)]
        across = [x[k] + h * (k == d) for k in range(dimension)]
        _, gradient = _degree_two(at_face)
        p = gradient[d]
        last = (slice(None), *[n - 1 if k == d else slice(None) for k in range(dimension)])
        if wraps:
            # The face across the wrap, at the last node before the one the grid repeats, reaches node 0.
            wrap = (slice(None), *[n - 2 if k == d else slice(None) for k in range(dimension)])
            across = [np.where(x[d] > 1.0 - 1.5 * h, 0.0, across[k]) if k == d else across[k] for k in range(dimension)]
            p[wrap] = ((_degree_two(across)[0] - U) / h)[wrap]
        if control == "L1":
            # The fixture keeps every face off L1's switch, where round-off would decide the sign.
            assert np.abs(np.abs(p) - L1_LAMBDA).min() > 1e-6
        component = closed_form(p, 0.5 * (M + _density(across)))
        if not wraps:
            component[last] = 0.0
        expected[:, d] = component

    if dimension == 1:
        np.testing.assert_allclose(velocity, expected[:, 0, : n - 1], rtol=0.0, atol=1e-12)
    else:
        # On an axis that wraps, the repeated node is not one the stencil reads a face at.
        nodes = slice(0, n - 1) if wraps else slice(None)
        compared = (slice(None), slice(None), *[nodes] * dimension)
        np.testing.assert_allclose(velocity[compared], expected[compared], rtol=0.0, atol=1e-12)


SMOOTHING = 0.25
TAU = 2 * np.pi


class _Isotropic(HamiltonianBase):
    """H = sqrt(|p|^2 + 0.25), whose control -p / sqrt(|p|^2 + 0.25) couples the components of p."""

    def __call__(self, t, x, p, m):
        q = np.asarray(p, dtype=float)
        return np.sqrt(np.sum(q * q, axis=-1) + SMOOTHING)

    def dp(self, t, x, p, m):
        q = np.asarray(p, dtype=float)
        return q / np.sqrt(np.sum(q * q, axis=-1, keepdims=True) + SMOOTHING)


def _smooth(x, periodic):
    """A U with a normal derivative at every wall and across the wrap, not symmetric, and its gradient."""
    if periodic:
        w = TAU * sum((1, 2, 1)[k] * x[k] for k in range(len(x)))
        u = np.sin(TAU * x[0]) * np.cos(TAU * x[1]) + 0.3 * np.sin(w)
        g = [TAU * np.cos(TAU * x[0]) * np.cos(TAU * x[1]), -TAU * np.sin(TAU * x[0]) * np.sin(TAU * x[1])]
        g = [g[k] + 0.3 * TAU * (1, 2, 1)[k] * np.cos(w) for k in range(2)]
        if len(x) == 3:
            u = u + 0.2 * np.sin(TAU * x[0]) * np.cos(TAU * x[2])
            g[0] = g[0] + 0.2 * TAU * np.cos(TAU * x[0]) * np.cos(TAU * x[2])
            g.append(0.3 * TAU * np.cos(w) - 0.2 * TAU * np.sin(TAU * x[0]) * np.sin(TAU * x[2]))
    else:
        u = np.sin(1.3 * x[0] + 0.4) * np.cos(0.9 * x[1]) + 0.3 * x[0] * x[1]
        g = [
            1.3 * np.cos(1.3 * x[0] + 0.4) * np.cos(0.9 * x[1]) + 0.3 * x[1],
            -0.9 * np.sin(1.3 * x[0] + 0.4) * np.sin(0.9 * x[1]) + 0.3 * x[0],
        ]
        if len(x) == 3:
            u = u + 0.2 * x[2] * (1 + x[0]) + 0.1 * np.sin(1.1 * x[2] + 0.2 * x[1])
            g[0] = g[0] + 0.2 * x[2]
            g[1] = g[1] + 0.02 * np.cos(1.1 * x[2] + 0.2 * x[1])
            g.append(0.2 * (1 + x[0]) + 0.11 * np.cos(1.1 * x[2] + 0.2 * x[1]))
    return u, g


def _coupled_control_error(dimension, walls, n):
    periodic = walls == "periodic"
    problem = _problem(_Isotropic(), dimension, n, _walls(walls, dimension), Nt=1)
    x = np.meshgrid(*[np.linspace(0.0, 1.0, n)] * dimension, indexing="ij")
    u, _ = _smooth(x, periodic)
    U = np.stack([u, u])
    velocity = compute_fp_velocity_field(problem, U, np.ones_like(U), _Isotropic())[0]
    h = 1.0 / (n - 1)
    error = 0.0
    for d in range(dimension):
        _, g = _smooth([x[k] + 0.5 * h * (k == d) for k in range(dimension)], periodic)
        exact = -g[d] / np.sqrt(sum(c * c for c in g) + SMOOTHING)
        faces = tuple(slice(0, n - 1) if (periodic or k == d) else slice(None) for k in range(dimension))
        error = max(error, float(np.abs(velocity[d][faces] - exact[faces]).max()))
    return error


@pytest.mark.parametrize(
    ("dimension", "walls", "coarse"), [(2, "no-flux", 33), (2, "periodic", 65), (3, "no-flux", 17), (3, "periodic", 17)]
)
def test_a_control_that_couples_the_components_converges_at_second_order(dimension, walls, coarse):
    errors = [_coupled_control_error(dimension, walls, n) for n in (coarse, 2 * coarse - 1)]
    order = np.log2(errors[0] / errors[1])
    assert order > 1.6, (
        f"order {order:.2f} from errors {errors[0]:.3e}, {errors[1]:.3e} at n = {coarse}, {2 * coarse - 1}"
    )


def _coupled_control_error_on(sizes, bounds):
    """`_coupled_control_error` on no-flux walls of a box whose node counts and lengths differ per axis."""
    dimension = len(sizes)
    problem = _grid_problem(_Isotropic(), sizes, bounds, _walls("no-flux", dimension), Nt=1)
    x = np.meshgrid(*[np.linspace(lo, hi, size) for (lo, hi), size in zip(bounds, sizes, strict=True)], indexing="ij")
    u, _ = _smooth(x, False)
    velocity = compute_fp_velocity_field(problem, np.stack([u, u]), np.ones((2, *u.shape)), _Isotropic())[0]
    h = [(hi - lo) / (size - 1) for (lo, hi), size in zip(bounds, sizes, strict=True)]
    error = 0.0
    for d in range(dimension):
        _, g = _smooth([x[k] + 0.5 * h[k] * (k == d) for k in range(dimension)], False)
        exact = -g[d] / np.sqrt(sum(c * c for c in g) + SMOOTHING)
        faces = tuple(slice(0, sizes[k] - 1) if k == d else slice(None) for k in range(dimension))
        error = max(error, float(np.abs(velocity[d][faces] - exact[faces]).max()))
    return error


@pytest.mark.parametrize(
    ("sizes", "bounds"),
    [((17, 9), [(0.0, 1.0), (0.0, 2.5)]), ((9, 7, 13), [(0.0, 1.5), (0.0, 0.8), (0.0, 2.0)])],
    ids=["2-D", "3-D"],
)
def test_on_a_box_whose_axes_differ_the_coupled_control_converges_at_second_order(sizes, bounds):
    """The coupled control on a box whose node counts, lengths and spacings differ per axis, where a transverse
    difference taken with another axis's spacing cannot pass: the observed order between it and the box refined once,
    to 2n - 1 nodes along each axis, stays above 1.6."""
    finer = tuple(2 * size - 1 for size in sizes)
    errors = [_coupled_control_error_on(sizes, bounds), _coupled_control_error_on(finer, bounds)]
    order = np.log2(errors[0] / errors[1])
    assert order > 1.6, f"order {order:.2f} from errors {errors[0]:.3e}, {errors[1]:.3e} at {sizes}, {finer}"


def _moved_one_cell(u, axis):
    """u on an endpoint-inclusive periodic axis moved by one cell: the distinct nodes rolled, the repeated one
    restored from node 0."""
    moved = np.roll(np.take(u, np.arange(u.shape[axis] - 1), axis=axis), 1, axis=axis)
    return np.concatenate([moved, np.take(moved, [0], axis=axis)], axis=axis)


@pytest.mark.parametrize("dimension", [2, 3])
def test_on_a_periodic_grid_the_velocity_moves_with_u(dimension):
    """A periodic grid is a torus, so U moved by one cell along an axis moves the velocity by one cell, for a control
    that couples the components of p too: each transverse component is a centred difference across the wrap as well
    as inside, where a one-sided difference at the ends would tie the field to where the grid's ends sit."""
    n = 9
    problem = _problem(_Isotropic(), dimension, n, _walls("periodic", dimension), Nt=1)
    x = np.meshgrid(*[np.linspace(0.0, 1.0, n)] * dimension, indexing="ij")
    u, _ = _smooth(x, True)

    def velocity(v):
        return compute_fp_velocity_field(problem, np.stack([v, v]), np.ones((2, *v.shape)), _Isotropic())

    distinct = (slice(None), slice(None), *[slice(0, n - 1)] * dimension)
    unmoved = velocity(u)[distinct]
    for axis in range(dimension):
        moved = velocity(_moved_one_cell(u, axis))[distinct]
        np.testing.assert_allclose(
            moved, np.roll(unmoved, 1, axis=2 + axis), rtol=0.0, atol=1e-13, err_msg=f"axis {axis}"
        )


def test_at_one_population_the_stack_reads_the_faces_m_does_across_the_wrap():
    """At one population the stack's one row is m, averaged at the faces by the same function: on a periodic 1-D grid
    a congestion factor reading ``cross_density[0]`` gives the velocity a factor reading ``m`` gives, at the face
    across the wrap too, for a density whose repeated endpoint differs from node 0."""
    n = 11
    x = [np.linspace(0.0, 1.0, n)]
    by_m = CongestionHamiltonian(
        QuadraticControlCost(control_cost=1.0), congestion_factor=lambda m: 1.0 + CONGESTION * np.asarray(m)
    )
    by_rows = CongestionHamiltonian(
        QuadraticControlCost(control_cost=1.0),
        congestion_factor=lambda m, cross_density: 1.0 + CONGESTION * np.asarray(cross_density)[0],
    )
    problem = _problem(by_m, 1, n, _walls("periodic", 1), Nt=len(STEPS) - 1)
    U, _ = _degree_two(x)
    M = _density(x)
    assert np.abs(M[:, -1] - M[:, 0]).min() > 1.0
    np.testing.assert_array_equal(
        compute_fp_velocity_field(problem, U, M, by_rows, cross_density=M.copy()),
        compute_fp_velocity_field(problem, U, M, by_m),
    )


def _census_fields(x):
    """The census test's frozen U and initial density on the nodes ``x``."""
    u = 0.3 * np.sin(2 * np.pi * x[0]) + 0.2 * x[0]
    for k in range(1, len(x)):
        u = u + 0.3 * (k + 1) * x[k] + 0.1 * x[0] * x[k]
    return np.stack([u] * 7), 1.0 + 0.5 * np.cos(2 * np.pi * x[0])


def _routes_on(problem, hamiltonian, x, fp, handed=None):
    """FP-FDM's value route and its velocity route, the velocity built on the BC ``handed`` to the builder."""
    U, m0 = _census_fields(x)
    by_value = np.asarray(fp.solve_fp_system(m0, potential_field=U, show_progress=False))
    velocity = compute_fp_velocity_field(problem, U, np.stack([m0] * len(U)), hamiltonian, boundary_conditions=handed)
    return np.asarray(fp.solve_fp_system(m0, drift_field=velocity, show_progress=False)), by_value


@pytest.mark.parametrize("convention", ["unstated", "stated_inclusive"])
@pytest.mark.parametrize("dimension", [1, 2])
def test_a_periodic_bc_handed_to_the_builder_is_read_with_the_grids_layout(dimension, convention):
    """The grid's coordinates own the periodic layout: an unstated convention takes the grid's, and a stated one that
    agrees is accepted. Handed the same BC, the velocity route equals FP-FDM's value route."""
    n = 21 if dimension == 1 else 9
    hamiltonian = SeparableHamiltonian(QuadraticControlCost(control_cost=1.0))
    problem = _problem(hamiltonian, dimension, n, _walls("periodic", dimension))
    handed = (
        periodic_bc(dimension=dimension)
        if convention == "unstated"
        else periodic_bc(dimension=dimension, convention=PeriodicGridConvention.ENDPOINT_INCLUSIVE)
    )
    x = np.meshgrid(*[np.linspace(0.0, 1.0, n)] * dimension, indexing="ij")
    by_velocity, by_value = _routes_on(
        problem, hamiltonian, x, FPFDMSolver(problem, boundary_conditions=handed), handed
    )
    np.testing.assert_allclose(by_velocity, by_value, rtol=0.0, atol=1e-12)


CONTRADICTS_THE_GRID = r"This grid's coordinates are ENDPOINT_INCLUSIVE on axis 0 .* the periodic boundary condition declares ENDPOINT_EXCLUSIVE"


@pytest.mark.parametrize("dimension", [1, 2])
def test_a_convention_that_contradicts_the_grid_is_refused_with_the_grids_message(dimension):
    """A periodic BC whose stated convention contradicts the grid's coordinates is refused with the grid's message,
    through the check the grid binds every BC with: the grid's coordinates own the layout. FP-FDM handed such a BC
    still solves on a torus one cell too long, which is #2910's."""
    n = 21 if dimension == 1 else 9
    hamiltonian = SeparableHamiltonian(QuadraticControlCost(control_cost=1.0))
    problem = _problem(hamiltonian, dimension, n, _walls("periodic", dimension))
    x = np.meshgrid(*[np.linspace(0.0, 1.0, n)] * dimension, indexing="ij")
    U, m0 = _census_fields(x)
    contradicting = periodic_bc(dimension=dimension, convention=PeriodicGridConvention.ENDPOINT_EXCLUSIVE)
    with pytest.raises(ValueError, match=CONTRADICTS_THE_GRID):
        compute_fp_velocity_field(problem, U, np.stack([m0] * len(U)), hamiltonian, boundary_conditions=contradicting)


@pytest.mark.parametrize("grid_walls", ["periodic", "no-flux"])
@pytest.mark.parametrize(
    "hamiltonian",
    [
        lambda: SeparableHamiltonian(L1ControlCost(lambda_=0.3)),
        lambda: CongestionHamiltonian(
            control_cost=QuadraticControlCost(control_cost=1.0), congestion_factor=lambda m: 1.0 + np.asarray(m)
        ),
    ],
    ids=["L1", "congestion"],
)
def test_fp_fvm_handed_a_contradicting_convention_is_refused(hamiltonian, grid_walls):
    """FP-FVM handed a stated ENDPOINT_EXCLUSIVE on an endpoint-inclusive grid keeps it. The velocity the router builds
    for it is refused with the grid's message, as 8f7c0db8 refused the solve, rather than handed to FVM as faces it
    would read as node values."""
    from mfgarchon.alg.numerical.fp_solvers import FPFVMSolver

    h = hamiltonian()
    problem = _problem(h, 1, 21, _walls(grid_walls, 1))
    contradicting = periodic_bc(dimension=1, convention=PeriodicGridConvention.ENDPOINT_EXCLUSIVE)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(ValueError, match=CONTRADICTS_THE_GRID):
            problem.solve(
                max_iterations=2,
                tolerance=1e-12,
                verbose=False,
                hjb_solver=HJBFDMSolver(problem),
                fp_solver=FPFVMSolver(problem, boundary_conditions=contradicting),
            )


class _KeepsTheBCItWasHanded(BaseFPSolver):
    """A user FP solver: the BC it was handed kept as it was handed, and FP-FDM's solve behind it."""

    def __init__(self, problem, boundary_conditions):
        super().__init__(problem)
        self.boundary_conditions = boundary_conditions
        self._inner = FPFDMSolver(problem, boundary_conditions=boundary_conditions)

    def solve_fp_system(
        self, M_initial, drift_field=None, potential_field=None, volatility=None, source_term=None, show_progress=False
    ):
        return self._inner.solve_fp_system(
            M_initial,
            drift_field=drift_field,
            potential_field=potential_field,
            volatility=volatility,
            source_term=source_term,
            show_progress=show_progress,
        )


@pytest.mark.parametrize("dimension", [1, 2])
def test_a_user_solver_keeping_a_periodic_bc_solves_as_fp_fdm_handed_it(dimension):
    """Through the router, a user solver that keeps an unstated periodic BC as it was handed solves as FP-FDM handed
    the same BC, which completes it: the builder completes it the same way. The terminal condition is not periodic,
    so U's repeated endpoint differs from node 0 and the face across the wrap reads which one it takes."""
    n = 21 if dimension == 1 else 9
    hamiltonian = CongestionHamiltonian(
        control_cost=QuadraticControlCost(control_cost=1.0), congestion_factor=lambda m: 1.0 + np.asarray(m)
    )
    grid = TensorProductGrid(
        bounds=[(0.0, 1.0)] * dimension, Nx_points=[n] * dimension, boundary_conditions=_walls("periodic", dimension)
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        problem = MFGProblem(
            model=Model(hamiltonian=hamiltonian, volatility=0.3),
            domain=grid,
            conditions=Conditions(
                m_initial=lambda x: 1.0 + 0.5 * np.cos(2 * np.pi * x[0]),
                u_terminal=lambda x: 2.0 * (x[0] - 0.8) ** 2 + 0.5 * x[-1],
                T=0.5,
            ),
            Nt=6,
        )
    handed = periodic_bc(dimension=dimension)

    def solved(fp):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = problem.solve(
                max_iterations=3, tolerance=1e-12, verbose=False, hjb_solver=HJBFDMSolver(problem), fp_solver=fp
            )
        return np.asarray(result.M)

    np.testing.assert_array_equal(
        solved(_KeepsTheBCItWasHanded(problem, handed)), solved(FPFDMSolver(problem, boundary_conditions=handed))
    )


def _grid_problem(hamiltonian, sizes, bounds, walls, Nt=6):
    grid = TensorProductGrid(bounds=bounds, Nx_points=list(sizes), boundary_conditions=walls)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return MFGProblem(
            model=Model(hamiltonian=hamiltonian, volatility=0.3),
            domain=grid,
            conditions=Conditions(m_initial=lambda x: 1.0, u_terminal=lambda x: 0.0, T=0.5),
            Nt=Nt,
        )


@pytest.mark.parametrize("sizes", [(9, 2), (2, 9)], ids=["9x2", "2x9"])
def test_an_axis_of_two_nodes_carries_the_value_route_drift(sizes):
    """An axis of two nodes has no second-order one-sided difference; its centred difference is first order there,
    and the quadratic control cost's velocity route still equals FP-FDM's value route."""
    hamiltonian = SeparableHamiltonian(QuadraticControlCost(control_cost=1.0))
    bounds = [(0.0, 1.0), (0.0, 0.6)]
    problem = _grid_problem(hamiltonian, sizes, bounds, _walls("no-flux", 2))
    x = np.meshgrid(*[np.linspace(lo, hi, size) for (lo, hi), size in zip(bounds, sizes, strict=True)], indexing="ij")
    by_velocity, by_value = _routes_on(problem, hamiltonian, x, FPFDMSolver(problem))
    np.testing.assert_allclose(by_velocity, by_value, rtol=0.0, atol=1e-12)


class _InsideOnly(_Isotropic):
    """The isotropic H, whose control refuses a point outside the unit square, as a map read off the grid does."""

    def dp(self, t, x, p, m):
        coordinates = np.atleast_2d(np.asarray(x, dtype=float).T).T if not isinstance(x, tuple) else x
        if any(np.any((np.asarray(c) < -1e-12) | (np.asarray(c) > 1.0 + 1e-12)) for c in coordinates):
            raise ValueError("_InsideOnly: a point outside the domain")
        return super().dp(t, x, p, m)


@pytest.mark.parametrize("walls", ["no-flux", "periodic"])
def test_the_control_is_asked_only_inside_the_domain(walls):
    """Each axis's batch holds the faces that exist, so the control is never asked at the phantom face past the last
    node: a solve with a control that refuses a point outside the domain runs, on no-flux and periodic walls."""
    problem = _problem(_InsideOnly(), 2, 9, _walls(walls, 2), Nt=4)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = problem.solve(max_iterations=2, tolerance=1e-12, verbose=False)
    assert np.isfinite(np.asarray(result.M)).all()


def test_a_node_a_wrapping_axis_repeats_carries_its_twins_face():
    """On a wrapping axis the repeated node's face is its twin's face, node 0's: the entry is that face's velocity, bit
    for bit, not a value of its own."""
    n = 9
    problem = _problem(_Isotropic(), 2, n, _walls("periodic", 2), Nt=1)
    x = np.meshgrid(*[np.linspace(0.0, 1.0, n)] * 2, indexing="ij")
    u, _ = _smooth(x, True)
    velocity = compute_fp_velocity_field(problem, np.stack([u, u]), np.ones((2, n, n)), _Isotropic())
    for d in range(2):
        np.testing.assert_array_equal(
            np.take(velocity[:, d], [n - 1], axis=1 + d), np.take(velocity[:, d], [0], axis=1 + d), err_msg=f"axis {d}"
        )


def test_a_bc_wrapping_one_axis_only_is_refused_as_fp_fdm_refuses_it():
    """FP-FDM wraps every axis or none and refuses a BC periodic on some faces only (#2495); handed such a BC, the
    builder refuses it through the same owner rather than build as if no axis wrapped."""
    segments = []
    for axis in range(2):
        for side in ("min", "max"):
            face = BoundaryFace(axis, side).to_string()
            kind = BCType.PERIODIC if axis == 0 else BCType.NO_FLUX
            segments.append(BCSegment(name=face, bc_type=kind, boundary=face))
    channel = BoundaryConditions(dimension=2, segments=segments)
    problem = _problem(_Isotropic(), 2, 9, _walls("no-flux", 2), Nt=1)
    x = np.meshgrid(*[np.linspace(0.0, 1.0, 9)] * 2, indexing="ij")
    u, _ = _smooth(x, False)
    with pytest.raises(NotImplementedError, match="periodic on some faces"):
        FPFDMSolver(problem, boundary_conditions=channel)
    with pytest.raises(
        NotImplementedError, match=r"^compute_fp_velocity_field, which builds for FP-FDM: .*periodic on some faces"
    ):
        compute_fp_velocity_field(
            problem, np.stack([u, u]), np.ones((2, 9, 9)), _Isotropic(), boundary_conditions=channel
        )


@pytest.mark.parametrize(
    ("walls", "said"),
    [
        (neumann_bc(value=0.5, dimension=1), "NEUMANN value that is not zero"),
        (robin_bc(value=0.5, dimension=1), "ROBIN"),
    ],
    ids=["neumann-g-not-zero", "robin"],
)
def test_handed_no_bc_the_builder_reads_the_problems_walls_as_the_fp_does(walls, said):
    """Handed no BC, the builder reads the problem's shared BC as an FP solver reads it (`fp_view_of_shared_bc`): a
    Neumann value that is not zero, or a Robin condition, is the HJB's datum and says nothing of the density's walls,
    so the builder refuses it, as FP-FDM on the same problem does."""
    hamiltonian = CongestionHamiltonian(
        control_cost=QuadraticControlCost(control_cost=1.0),
        congestion_factor=lambda m: 1.0 + CONGESTION * np.asarray(m),
    )
    problem = _problem(hamiltonian, 1, 11, walls, Nt=len(STEPS) - 1)
    x = [np.linspace(0.0, 1.0, 11)]
    U, _ = _degree_two(x)
    for consumer, call in (
        ("compute_fp_velocity_field", lambda: compute_fp_velocity_field(problem, U, _density(x), hamiltonian)),
        ("FPFDMSolver", lambda: FPFDMSolver(problem)),
    ):
        with pytest.raises(
            NotImplementedError, match=rf"^{consumer}: the problem's shared boundary condition has a {said}"
        ):
            call()
