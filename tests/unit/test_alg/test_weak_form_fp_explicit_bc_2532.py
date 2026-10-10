"""FP-FEM and meshless Galerkin take a BC of their own, and the shared one is still read as before (#2532).

`WeakFormFPSolver` takes ``boundary_conditions=``. A BC handed to it is the FP's own and is used as given, a
NEUMANN refused (`BaseFPSolver._fp_own_bc`); without one the solver reads the problem's shared BC through
`fp_view_of_shared_bc`, where a DIRICHLET is an absorbing exit whose value is the HJB's (#2512, row B3).

- **FP-FEM** gains a prescribed density: an explicit DIRICHLET(g) holds m = g at the wall, and its value
  enters the solve through the condensation's lift, not only the post-solve write.
- **Meshless Galerkin** gains no wall (#2581, A-4: no oracle checks a meshless density). Its explicit route
  solves NO_FLUX and an absorbing DIRICHLET(0) as the shared route does, and refuses a DIRICHLET with a
  value. The route exists so that a shared NEUMANN(g) or ROBIN can be run by giving the FP its own no-flux BC.
"""

from __future__ import annotations

from decimal import Decimal
from fractions import Fraction

import pytest

import numpy as np

from mfgarchon import Conditions, MFGProblem, Model
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import BCSegment, BCType, BoundaryConditions, no_flux_bc
from mfgarchon.geometry.boundary.providers import ConstantProvider

SIGMA, T, NT = 0.4, 0.5, 10
EXIT_VALUE = 0.7


def _model() -> Model:
    return Model(
        hamiltonian=SeparableHamiltonian(control_cost=QuadraticControlCost(control_cost=1.0)), volatility=SIGMA
    )


def _problem(domain, bc: BoundaryConditions, horizon: float = T) -> MFGProblem:
    """``domain(bc)`` builds the domain carrying ``bc`` as the shared BC."""
    from mfgarchon.geometry.meshes.mesh_1d import Mesh1D

    geometry = domain(bc)
    # Unit mass on the domain's own measure: Mesh1D's uniform-cell measure gives a constant c the mass c (n+1)/n.
    c = 16 / 17 if isinstance(geometry, Mesh1D) else 1.0
    return MFGProblem(
        model=_model(),
        domain=geometry,
        conditions=Conditions(
            m_initial=lambda x: c + 0.0 * np.asarray(x, dtype=float)[..., 0], u_terminal=lambda x: 0.0, T=horizon
        ),
        Nt=NT,
    )


def _mesh(dim: int):
    """A builder of the unit interval (16 cells) or square (8 x 8), carrying the BC it is given."""

    def build(bc: BoundaryConditions):
        if dim == 1:
            from mfgarchon.geometry.meshes.mesh_1d import Mesh1D

            geometry = Mesh1D(bounds=(0.0, 1.0), num_elements=16)
            geometry.generate_mesh()
        else:
            import skfem

            from mfgarchon.alg.numerical.fem.mesh_adapter import skfem_to_meshdata
            from mfgarchon.geometry.meshes.mesh_2d import Mesh2D

            geometry = Mesh2D(domain_type="rectangle", bounds=(0.0, 1.0, 0.0, 1.0))
            xs = np.linspace(0.0, 1.0, 9)
            geometry.mesh_data = skfem_to_meshdata(skfem.MeshTri.init_tensor(xs, xs))
        geometry.boundary_conditions = bc
        return geometry

    return build


def _grid(bc: BoundaryConditions):
    return TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[21], boundary_conditions=bc)


def _exit(dim: int, value: float) -> BoundaryConditions:
    """DIRICHLET(value) on x_max, NO_FLUX on every other face."""
    walls = ["x_min"] if dim == 1 else ["x_min", "y_min", "y_max"]
    return BoundaryConditions(
        dimension=dim,
        segments=[
            BCSegment(name="exit", bc_type=BCType.DIRICHLET, value=value, boundary="x_max"),
            *(BCSegment(name=face, bc_type=BCType.NO_FLUX, value=0.0, boundary=face) for face in walls),
        ],
    )


def _fem(problem: MFGProblem, bc: BoundaryConditions | None = None):
    from mfgarchon.alg.numerical.fem.fp_fem_solver import FPFEMSolver

    return FPFEMSolver(problem, order=1, boundary_conditions=bc)


def _meshless(problem: MFGProblem, bc: BoundaryConditions | None = None):
    from mfgarchon.alg.numerical.meshless_galerkin.fp_solver import MeshlessGalerkinFPSolver

    return MeshlessGalerkinFPSolver(
        problem, np.linspace(0.0, 1.0, 21).reshape(-1, 1), delta=0.35, boundary_conditions=bc
    )


def _drift_into_x_max(solver, m0: float) -> np.ndarray:
    x = solver._disc.dof_coordinates[:, 0]
    return np.asarray(
        solver.solve_fp_system(
            M_initial=np.full(x.size, m0), potential_field=np.broadcast_to(-x, (NT + 1, x.size)).copy()
        )
    )


@pytest.mark.parametrize("dim", [1, 2])
def test_fem_holds_an_explicit_density_and_reads_a_shared_dirichlet_as_an_exit(dim):
    """One problem, its shared BC DIRICHLET(0.7) on x_max: the HJB's exit cost.

    Read from the problem, the FP absorbs: from m = 0 with no drift nothing ever enters, so the density stays
    exactly 0. Handed the same BC as its own, the FP holds m = 0.7 at the wall, and with no drift and a no-flux
    wall opposite the exact steady state is m = 0.7 everywhere, which a long horizon (T = 100, dt = 10)
    reaches: measured max|m(T) - 0.7| = 1.6e-5 in 1-D and 2-D. A route that wrote 0.7 after a solve lifted at
    0 (the write-only rival of `test_fp_fem_bc_oracles_2512.py`, review 1) misses that state by 0.699 in 1-D
    and 0.698 in 2-D, its density falling to 6e-4 and 1.9e-3.
    """
    problem = _problem(_mesh(dim), _exit(dim, EXIT_VALUE), horizon=100.0)
    shared, own = _fem(problem), _fem(problem, _exit(dim, EXIT_VALUE))
    x = own._disc.dof_coordinates[:, 0]
    at_exit = np.isclose(x, 1.0)

    M_shared = np.asarray(shared.solve_fp_system(M_initial=np.zeros(x.size), potential_field=None))
    M_own = np.asarray(own.solve_fp_system(M_initial=np.zeros(x.size), potential_field=None))

    np.testing.assert_array_equal(M_shared, 0.0)
    np.testing.assert_array_equal(M_own[1:, at_exit], EXIT_VALUE)
    assert np.abs(M_own[-1] - EXIT_VALUE).max() < 1e-3


@pytest.mark.parametrize(
    ("build", "domain", "dim"),
    [(_fem, _mesh(1), 1), (_fem, _mesh(2), 2), (_meshless, _grid, 1)],
    ids=["fem-1d", "fem-2d", "meshless-1d"],
)
def test_an_explicit_no_flux_solves_as_the_shared_one(build, domain, dim):
    """The route changes where the BC comes from, not what a no-flux wall is, under a drift into x_max."""
    problem = _problem(domain, no_flux_bc(dimension=dim))
    np.testing.assert_array_equal(
        _drift_into_x_max(build(problem), 1.0), _drift_into_x_max(build(problem, no_flux_bc(dimension=dim)), 1.0)
    )


def test_meshless_reads_an_explicit_absorbing_wall_as_the_shared_exit():
    """Meshless gains no wall: its own DIRICHLET(0) is the exit it already had from a shared DIRICHLET(0.7)."""
    problem = _problem(_grid, _exit(1, EXIT_VALUE))
    np.testing.assert_array_equal(
        _drift_into_x_max(_meshless(problem), 1.0), _drift_into_x_max(_meshless(problem, _exit(1, 0.0)), 1.0)
    )


@pytest.mark.parametrize("value", [EXIT_VALUE, lambda x: EXIT_VALUE], ids=["number", "callable"])
def test_meshless_refuses_an_explicit_density(value):
    """A DIRICHLET with a value not provably zero, handed to meshless, would be m = g: refused (#2581, A-4)."""
    problem = _problem(_grid, no_flux_bc(dimension=1))
    with pytest.raises(NotImplementedError, match="would be a prescribed density m = g"):
        _meshless(problem, _exit(1, value))


# ------------------------------------------------------------------------- the FEM adapter's Dirichlet value
# The explicit route hands FP-FEM a Dirichlet value the shared view used to zero first, and the FEM adapter read
# any value but a plain int or float as 0: np.float32(0.7) held m = 0 (#2593, review 1). HJB-FEM reads the same
# adapter from the shared BC, so it held u = 0 there on main. A value is now a callable g(x), a finite real number
# or a finite 0-d real array, and anything else is refused at construction.
_READABLE = [0.7, np.float32(0.7), np.int64(1), np.array(0.7), Fraction(7, 10), lambda x: 0.7]
_READABLE_IDS = ["float", "float32", "int64", "0-d", "Fraction", "callable"]
_UNREADABLE = [ConstantProvider(0.7), "0.7", np.array([0.7]), Decimal("0.7"), float("nan"), np.inf, np.array(0.7 + 0j)]
_UNREADABLE_IDS = ["provider", "str", "1-d", "Decimal", "nan", "inf", "complex"]


def _as_number(value) -> float:
    return float(value(np.zeros(1))) if callable(value) else float(value)


@pytest.mark.parametrize("value", _READABLE, ids=_READABLE_IDS)
def test_fem_holds_an_explicit_density_of_every_readable_value(value):
    """The steady state of the density test, m = g everywhere, for each value class."""
    own = _fem(_problem(_mesh(1), _exit(1, EXIT_VALUE), horizon=100.0), _exit(1, value))
    x = own._disc.dof_coordinates[:, 0]
    M = np.asarray(own.solve_fp_system(M_initial=np.zeros(x.size), potential_field=None))
    assert np.abs(M[-1] - _as_number(value)).max() < 1e-3


@pytest.mark.parametrize("value", _READABLE, ids=_READABLE_IDS)
def test_hjb_fem_holds_a_shared_dirichlet_of_every_readable_value(value):
    """The HJB's u = g on the exit at every level the solve writes, for each value class."""
    from mfgarchon.alg.numerical.fem.hjb_fem_solver import HJBFEMSolver

    solver = HJBFEMSolver(_problem(_mesh(1), _exit(1, value)))
    x = solver._disc.dof_coordinates[:, 0]
    U = np.asarray(solver.solve_hjb_system(M_density=np.ones((NT + 1, x.size)), U_terminal=np.zeros(x.size)))
    np.testing.assert_allclose(U[:-1, np.isclose(x, 1.0)], _as_number(value), rtol=1e-6)


@pytest.mark.parametrize("value", _UNREADABLE, ids=_UNREADABLE_IDS)
@pytest.mark.parametrize("route", ["fp-explicit", "hjb-shared"])
def test_fem_refuses_a_dirichlet_value_it_cannot_read(route, value):
    from mfgarchon.alg.numerical.fem.hjb_fem_solver import HJBFEMSolver

    if route == "fp-explicit":
        problem, own = _problem(_mesh(1), _exit(1, EXIT_VALUE)), _exit(1, value)
        build = lambda: _fem(problem, own)  # noqa: E731
    else:
        problem = _problem(_mesh(1), _exit(1, value))
        build = lambda: HJBFEMSolver(problem)  # noqa: E731
    with pytest.raises(NotImplementedError, match=r"Dirichlet segment 'exit' has a (value of type|non-finite)"):
        build()


@pytest.mark.parametrize("route", ["fp-explicit", "hjb-shared"])
def test_fem_refuses_a_dirichlet_default_no_segment_names(route):
    """The FEM adapter imposes conditions only from segments, so a DIRICHLET default_bc behind a segment set that
    leaves a face unnamed got the natural condition there. That BC is refused; one whose segments name every face,
    and dirichlet_bc(g), whose segment names the whole boundary, are not."""
    from mfgarchon.alg.numerical.fem.hjb_fem_solver import HJBFEMSolver
    from mfgarchon.geometry.boundary import dirichlet_bc

    def build(bc):
        if route == "fp-explicit":
            return _fem(_problem(_mesh(1), _exit(1, EXIT_VALUE)), bc)
        return HJBFEMSolver(_problem(_mesh(1), bc))

    wall = BCSegment(name="wall", bc_type=BCType.NO_FLUX, value=0.0, boundary="x_min")
    exit_ = BCSegment(name="exit", bc_type=BCType.DIRICHLET, value=EXIT_VALUE, boundary="x_max")
    with pytest.raises(NotImplementedError, match=r"default_bc=DIRICHLET governs 1 boundary facet"):
        build(BoundaryConditions(dimension=1, segments=[wall], default_bc=BCType.DIRICHLET, default_value=EXIT_VALUE))
    build(BoundaryConditions(dimension=1, segments=[wall, exit_], default_bc=BCType.DIRICHLET, default_value=0.7))
    build(dirichlet_bc(value=EXIT_VALUE, dimension=1))
