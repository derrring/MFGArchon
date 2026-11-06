"""The FP velocity is built at the faces FP-FDM's divergence-upwind stencil reads it at (#2904).

FP-FDM's stencil reads ``interface_velocity[d][i]`` as the velocity at the face ``(i + 1/2)`` along ``d``, the node
across it being ``i + 1``, wrapped through `periodic_axis_span` on an axis that wraps. `compute_fp_velocity_field` built
another field: on a periodic 1-D grid its last face read the endpoint the grid repeats instead of node 0, and in nD it
returned node values of centred differences. FP-FDM's velocity route then carried another drift than its
value-function route, 9.3 apart in M on a 2-D periodic grid at f59bb154, and nothing raised.

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
from mfgarchon.core.hamiltonian import (
    BoundedControlCost,
    CongestionHamiltonian,
    HamiltonianBase,
    L1ControlCost,
    QuadraticControlCost,
    SeparableHamiltonian,
)
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import BCSegment, BCType, BoundaryConditions, dirichlet_bc, no_flux_bc, periodic_bc
from mfgarchon.geometry.boundary.types import BoundaryFace

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
    problem = _problem(
        hamiltonian, dimension, n, _walls(walls, dimension), volatility, "field" if field else None
    )
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
    u = u + sum(c * x[k] * x[l] for (k, l), c in C.items() if l < dimension)
    gradient = [
        2 * A[k] * x[k] + B[k] + sum(c * x[l if j == k else j] for (j, l), c in C.items() if l < dimension and k in (j, l))
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
@pytest.mark.parametrize(("own", "problems"), [("periodic", "no-flux"), ("no-flux", "periodic"), ("Dirichlet", "periodic")])
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
    assert order > 1.6, f"order {order:.2f} from errors {errors[0]:.3e}, {errors[1]:.3e} at n = {coarse}, {2 * coarse - 1}"
