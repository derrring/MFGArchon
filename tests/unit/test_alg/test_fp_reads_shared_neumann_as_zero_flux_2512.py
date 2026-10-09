"""An FP solver reads a shared NEUMANN as zero flux and refuses one of its own (#2512, row B3).

A shared BC carries one datum per face, and its value is the HJB's: for NEUMANN(g), the boundary cost per
unit of local time, du/dn = g. The FP sees the physical pairing, zero total flux J.n = 0, whatever g is,
because the agents are reflected (NEUMANN(0) ruled 2026-10-08; any g, the audit session's reading of
2026-10-09). `BaseFPSolver._fp_view_of_shared` owns that reading. A NEUMANN handed to an FP solver
explicitly would mean dm/dn = g, which no FP solver implements, so it is refused (`_fp_own_bc`).

The routing pins solve under a drift into the x_max wall, where J.n = 0 and dm/dn = 0 differ. With g = 0
the two spellings already solved alike, so g = 0.7 is the case that needs the owner: before it, the FP
refused a nonzero shared Neumann value.
"""

from __future__ import annotations

import pytest

import numpy as np

from mfgarchon import Conditions, MFGProblem, Model
from mfgarchon.alg.numerical.fp_solvers.fp_fdm import FPFDMSolver
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import (
    BCSegment,
    BCType,
    BoundaryConditions,
    neumann_bc,
    no_flux_bc,
)

SIGMA, T, NT = 0.4, 0.5, 10
OWN_REFUSAL = r"a NEUMANN boundary condition passed to an FP solver means dm/dn = g"


def _model() -> Model:
    return Model(
        hamiltonian=SeparableHamiltonian(control_cost=QuadraticControlCost(control_cost=1.0)), volatility=SIGMA
    )


def _grid_problem(bc: BoundaryConditions, dim: int) -> MFGProblem:
    n = 21 if dim == 1 else 11
    grid = TensorProductGrid(bounds=[(0.0, 1.0)] * dim, Nx_points=[n] * dim, boundary_conditions=bc)
    return MFGProblem(
        model=_model(),
        domain=grid,
        conditions=Conditions(
            m_initial=lambda x: 1.0 + 0.0 * np.asarray(x, dtype=float)[..., 0], u_terminal=lambda x: 0.0, T=T
        ),
        Nt=NT,
    )


def _fdm_solve(bc: BoundaryConditions, dim: int) -> np.ndarray:
    problem = _grid_problem(bc, dim)
    solver = FPFDMSolver(problem)
    shape = tuple(problem.geometry.get_grid_shape())
    x = np.meshgrid(*[np.linspace(0.0, 1.0, s) for s in shape], indexing="ij")[0]
    # U = -x, so the drift -grad U points into the x_max wall.
    return np.asarray(
        solver.solve_fp_system(M_initial=np.ones(shape), potential_field=np.broadcast_to(-x, (NT + 1, *shape)))
    )


def _mesh(dim: int):
    if dim == 1:
        from mfgarchon.geometry.meshes.mesh_1d import Mesh1D

        geometry = Mesh1D(bounds=(0.0, 1.0), num_elements=16)
        geometry.generate_mesh()
        return geometry
    import skfem

    from mfgarchon.alg.numerical.fem.mesh_adapter import skfem_to_meshdata
    from mfgarchon.geometry.meshes.mesh_2d import Mesh2D

    geometry = Mesh2D(domain_type="rectangle", bounds=(0.0, 1.0, 0.0, 1.0))
    xs = np.linspace(0.0, 1.0, 9)
    geometry.mesh_data = skfem_to_meshdata(skfem.MeshTri.init_tensor(xs, xs))
    return geometry


def _fem_solve(bc: BoundaryConditions, dim: int) -> np.ndarray:
    from mfgarchon.alg.numerical.fem.fp_fem_solver import FPFEMSolver

    geometry = _mesh(dim)
    geometry.boundary_conditions = bc
    # Unit mass on the mesh's own measure: Mesh1D's uniform-cell measure gives a constant c the mass c (n+1)/n.
    c = 16 / 17 if dim == 1 else 1.0
    problem = MFGProblem(
        model=_model(),
        domain=geometry,
        conditions=Conditions(m_initial=lambda p: c + 0.0 * np.asarray(p)[..., 0], u_terminal=lambda p: 0.0, T=T),
        Nt=NT,
    )
    solver = FPFEMSolver(problem, order=1)
    x = solver._disc.dof_coordinates[:, 0]
    return np.asarray(
        solver.solve_fp_system(M_initial=np.ones_like(x), potential_field=np.broadcast_to(-x, (NT + 1, x.size)))
    )


@pytest.mark.parametrize("g", [0.0, 0.7])
@pytest.mark.parametrize(
    ("family", "dim"),
    [("fdm", 1), ("fdm", 2), ("fem", 1), ("fem", 2)],
    ids=["fdm-1d", "fdm-2d", "fem-1d", "fem-2d"],
)
def test_a_shared_neumann_solves_as_zero_flux(family, dim, g):
    solve = _fdm_solve if family == "fdm" else _fem_solve
    np.testing.assert_array_equal(solve(neumann_bc(value=g, dimension=dim), dim), solve(no_flux_bc(dimension=dim), dim))


@pytest.mark.parametrize("dim", [1, 2])
def test_fdm_refuses_a_neumann_of_its_own(dim):
    """FP-FDM takes an explicit BC, and a NEUMANN in it is refused, g = 0 included. FP-FEM takes none (#2532)."""
    with pytest.raises(NotImplementedError, match=OWN_REFUSAL):
        FPFDMSolver(_grid_problem(no_flux_bc(dimension=dim), dim), boundary_conditions=neumann_bc(dimension=dim))
    FPFDMSolver(_grid_problem(no_flux_bc(dimension=dim), dim), boundary_conditions=no_flux_bc(dimension=dim))


def _families_with_an_explicit_route():
    from mfgarchon.alg.numerical.fp_solvers.fp_fvm import FPFVMSolver
    from mfgarchon.alg.numerical.fp_solvers.fp_gfdm import FPGFDMSolver
    from mfgarchon.alg.numerical.fp_solvers.fp_particle import FPParticleSolver
    from mfgarchon.alg.numerical.fp_solvers.fp_semi_lagrangian_adjoint import FPSLSolver

    points = np.linspace(0.0, 1.0, 21).reshape(-1, 1)
    return {
        "FVM": lambda p, bc: FPFVMSolver(p, boundary_conditions=bc),
        "GFDM": lambda p, bc: FPGFDMSolver(p, collocation_points=points, boundary_conditions=bc),
        "SL": lambda p, bc: FPSLSolver(p, boundary_conditions=bc),
        "Particle": lambda p, bc: FPParticleSolver(p, num_particles=200, seed=0, boundary_conditions=bc),
    }


@pytest.mark.parametrize("family", ["FVM", "GFDM", "SL", "Particle"])
def test_every_explicit_route_refuses_a_neumann(family):
    build = _families_with_an_explicit_route()[family]
    problem = _grid_problem(no_flux_bc(dimension=1), 1)
    with pytest.raises(NotImplementedError, match=OWN_REFUSAL):
        build(problem, neumann_bc(dimension=1))
    build(problem, no_flux_bc(dimension=1))


def test_the_refusal_names_a_neumann_fall_through():
    """A BC whose segments cover every face can still carry default_bc=NEUMANN, which the deprecated
    mixed_bc sets; the refusal says so and how to avoid it. It refuses even where no face reaches the
    default, until #2512's row B1 resolves face coverage."""
    covered = BoundaryConditions(
        dimension=1,
        segments=[
            BCSegment(name="left", bc_type=BCType.DIRICHLET, value=0.0, boundary="x_min"),
            BCSegment(name="right", bc_type=BCType.DIRICHLET, value=0.0, boundary="x_max"),
        ],
        default_bc=BCType.NEUMANN,
    )
    with pytest.raises(NotImplementedError, match=r"default_bc=NEUMANN.*pass default_bc=BCType.NO_FLUX"):
        FPFDMSolver(_grid_problem(no_flux_bc(dimension=1), 1), boundary_conditions=covered)


@pytest.mark.parametrize("family", ["FDM", "FEM", "FVM", "GFDM", "SL", "Particle", "Meshless"])
def test_every_family_constructs_on_a_shared_nonzero_neumann(family):
    """Each FP family reads a shared NEUMANN(0.7) through the owner: it constructs, and holds no NEUMANN.

    Before #2512's row B3 every family refused it, as an inhomogeneous Neumann value it could not honour.
    """
    if family == "FEM":
        from mfgarchon.alg.numerical.fem.fp_fem_solver import FPFEMSolver

        geometry = _mesh(2)
        geometry.boundary_conditions = neumann_bc(value=0.7, dimension=2)
        problem = MFGProblem(
            model=_model(),
            domain=geometry,
            conditions=Conditions(m_initial=lambda p: 1.0 + 0.0 * np.asarray(p)[..., 0], u_terminal=lambda p: 0.0, T=T),
            Nt=NT,
        )
        held = FPFEMSolver(problem, order=1)._bc
    elif family == "Meshless":
        from mfgarchon.alg.numerical.meshless_galerkin.fp_solver import MeshlessGalerkinFPSolver

        problem = _grid_problem(neumann_bc(value=0.7, dimension=1), 1)
        held = MeshlessGalerkinFPSolver(problem, np.linspace(0.0, 1.0, 11).reshape(-1, 1), delta=0.35)._bc
    elif family == "FDM":
        held = FPFDMSolver(_grid_problem(neumann_bc(value=0.7, dimension=1), 1)).boundary_conditions
    else:
        solver = _families_with_an_explicit_route()[family](_grid_problem(neumann_bc(value=0.7, dimension=1), 1), None)
        held = solver.boundary_conditions if family in ("FVM", "Particle") else solver.get_boundary_conditions()
    assert held.default_bc != BCType.NEUMANN
    assert all(seg.bc_type != BCType.NEUMANN for seg in held.segments)
