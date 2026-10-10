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

import pytest

import numpy as np

from mfgarchon import Conditions, MFGProblem, Model
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import BCSegment, BCType, BoundaryConditions, no_flux_bc

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
    0 (the write-only rival of `test_fp_fem_bc_oracles_2512.py`, review 1) stays near 0 in the far interior:
    0.699 in 1-D and 0.698 in 2-D.
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
