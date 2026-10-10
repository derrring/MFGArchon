"""An FP solver reads a shared NEUMANN(0) as zero flux, refuses a shared NEUMANN(g != 0), and refuses one of its own (#2512, row B3).

A shared BC carries one datum per face, and its value is the HJB's: for NEUMANN(g), the boundary cost per
unit of local time, du/dn = g. `BaseFPSolver._fp_view_of_shared` owns the FP's reading of it:
- NEUMANN(0) is the reflecting pairing, zero total flux J.n = 0 (ruled 2026-10-08);
- NEUMANN(g != 0) is refused at the FP (user ruling 2026-10-09). Its g says nothing about the agents'
  mass at the wall, and a mass flux through the wall is not a Neumann condition, so the two equations'
  BCs are specified separately: the shared BC keeps NEUMANN(g) for the HJB, and the FP solver is given
  its own no-flux BC. Every FP solver that reads a segment BC takes one (FP-FEM and meshless Galerkin since #2532).

A NEUMANN handed to an FP solver explicitly would mean dm/dn = g, which no FP solver implements, so it is
refused, g = 0 included (`_fp_own_bc`).

The routing pins solve under a drift into the x_max wall, where J.n = 0 and dm/dn = 0 differ.
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
SHARED_REFUSAL = r"the problem's shared boundary condition has a NEUMANN value that is not zero"


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


def _mesh_problem(bc: BoundaryConditions, dim: int) -> MFGProblem:
    geometry = _mesh(dim)
    geometry.boundary_conditions = bc
    # Unit mass on the mesh's own measure: Mesh1D's uniform-cell measure gives a constant c the mass c (n+1)/n.
    c = 16 / 17 if dim == 1 else 1.0
    return MFGProblem(
        model=_model(),
        domain=geometry,
        conditions=Conditions(m_initial=lambda p: c + 0.0 * np.asarray(p)[..., 0], u_terminal=lambda p: 0.0, T=T),
        Nt=NT,
    )


def _fem_solve(bc: BoundaryConditions, dim: int) -> np.ndarray:
    from mfgarchon.alg.numerical.fem.fp_fem_solver import FPFEMSolver

    solver = FPFEMSolver(_mesh_problem(bc, dim), order=1)
    x = solver._disc.dof_coordinates[:, 0]
    return np.asarray(
        solver.solve_fp_system(M_initial=np.ones_like(x), potential_field=np.broadcast_to(-x, (NT + 1, x.size)))
    )


_CELLS = pytest.mark.parametrize(
    ("family", "dim"),
    [("fdm", 1), ("fdm", 2), ("fem", 1), ("fem", 2)],
    ids=["fdm-1d", "fdm-2d", "fem-1d", "fem-2d"],
)


@_CELLS
def test_a_shared_neumann_zero_solves_as_zero_flux(family, dim):
    solve = _fdm_solve if family == "fdm" else _fem_solve
    np.testing.assert_array_equal(solve(neumann_bc(dimension=dim), dim), solve(no_flux_bc(dimension=dim), dim))


@_CELLS
def test_a_shared_neumann_value_is_refused_naming_both_readings(family, dim):
    """g = 0.7: the HJB's boundary cost, which the FP cannot read a wall condition from."""
    solve = _fdm_solve if family == "fdm" else _fem_solve
    with pytest.raises(NotImplementedError, match=SHARED_REFUSAL) as excinfo:
        solve(neumann_bc(value=0.7, dimension=dim), dim)
    message = str(excinfo.value)
    assert "du/dn = g" in message
    assert "a mass flux through the wall is not a Neumann condition" in message
    solver = "FPFDMSolver" if family == "fdm" else "FPFEMSolver"
    assert f"pass {solver} boundary_conditions=no_flux_bc(dimension=...)" in message


@pytest.mark.parametrize("dim", [1, 2])
def test_fdm_refuses_a_neumann_of_its_own(dim):
    """FP-FDM takes an explicit BC, and a NEUMANN in it is refused, g = 0 included."""
    with pytest.raises(NotImplementedError, match=OWN_REFUSAL):
        FPFDMSolver(_grid_problem(no_flux_bc(dimension=dim), dim), boundary_conditions=neumann_bc(dimension=dim))
    FPFDMSolver(_grid_problem(no_flux_bc(dimension=dim), dim), boundary_conditions=no_flux_bc(dimension=dim))


@pytest.mark.parametrize("dim", [1, 2])
def test_fem_refuses_a_neumann_of_its_own(dim):
    """FP-FEM takes an explicit BC since #2532, and a NEUMANN in it is refused, g = 0 included."""
    from mfgarchon.alg.numerical.fem.fp_fem_solver import FPFEMSolver

    problem = _mesh_problem(no_flux_bc(dimension=dim), dim)
    with pytest.raises(NotImplementedError, match=OWN_REFUSAL):
        FPFEMSolver(problem, order=1, boundary_conditions=neumann_bc(dimension=dim))
    FPFEMSolver(problem, order=1, boundary_conditions=no_flux_bc(dimension=dim))


def _families_with_an_explicit_route():
    """The grid families' builders; FP-FEM needs a mesh and is built where it is used."""
    from mfgarchon.alg.numerical.fp_solvers.fp_fvm import FPFVMSolver
    from mfgarchon.alg.numerical.fp_solvers.fp_particle import FPParticleSolver
    from mfgarchon.alg.numerical.fp_solvers.fp_semi_lagrangian_adjoint import FPSLSolver
    from mfgarchon.alg.numerical.meshless_galerkin.fp_solver import MeshlessGalerkinFPSolver

    cloud = np.linspace(0.0, 1.0, 21).reshape(-1, 1)
    return {
        "FVM": lambda p, bc: FPFVMSolver(p, boundary_conditions=bc),
        "SL": lambda p, bc: FPSLSolver(p, boundary_conditions=bc),
        "Particle": lambda p, bc: FPParticleSolver(p, num_particles=200, seed=0, boundary_conditions=bc),
        "Meshless": lambda p, bc: MeshlessGalerkinFPSolver(p, cloud, delta=0.35, boundary_conditions=bc),
    }


@pytest.mark.parametrize("family", ["FVM", "SL", "Particle", "Meshless"])
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


def _build_on_shared(family: str, g: float):
    """``family``'s FP solver on a problem whose shared BC is NEUMANN(g), and the BC it holds."""
    if family == "FEM":
        from mfgarchon.alg.numerical.fem.fp_fem_solver import FPFEMSolver

        return FPFEMSolver(_mesh_problem(neumann_bc(value=g, dimension=2), 2), order=1)._bc
    if family == "Meshless":
        from mfgarchon.alg.numerical.meshless_galerkin.fp_solver import MeshlessGalerkinFPSolver

        problem = _grid_problem(neumann_bc(value=g, dimension=1), 1)
        return MeshlessGalerkinFPSolver(problem, np.linspace(0.0, 1.0, 11).reshape(-1, 1), delta=0.35)._bc
    if family == "FDM":
        return FPFDMSolver(_grid_problem(neumann_bc(value=g, dimension=1), 1)).boundary_conditions
    solver = _families_with_an_explicit_route()[family](_grid_problem(neumann_bc(value=g, dimension=1), 1), None)
    return solver.boundary_conditions if family in ("FVM", "Particle") else solver.get_boundary_conditions()


#: FP-GFDM is withdrawn (#2583) and constructs on no problem; its refusal is pinned in its own file.
_ALL_FAMILIES = ["FDM", "FEM", "FVM", "SL", "Particle", "Meshless"]


@pytest.mark.parametrize("family", _ALL_FAMILIES)
def test_every_family_reads_a_shared_neumann_zero_as_no_flux(family):
    held = _build_on_shared(family, 0.0)
    assert held.default_bc != BCType.NEUMANN
    assert all(seg.bc_type != BCType.NEUMANN for seg in held.segments)


@pytest.mark.parametrize("family", _ALL_FAMILIES)
def test_every_family_refuses_a_shared_neumann_value(family):
    """Every family is advised to give the FP its own no-flux BC; each takes one (#2532 for FEM and meshless).
    Through problem.solve both solvers are passed: given fp_solver= alone, problem.solve raises "Expert Mode requires
    BOTH hjb_solver and fp_solver" (#2593, review 2)."""
    with pytest.raises(NotImplementedError, match=SHARED_REFUSAL) as excinfo:
        _build_on_shared(family, 0.7)
    message = str(excinfo.value)
    assert "boundary_conditions=no_flux_bc(dimension=...) for reflected agents" in message
    assert "build both solvers yourself and pass them as hjb_solver= and fp_solver=" in message


@pytest.mark.parametrize("family", ["FDM", "FVM", "SL", "Particle", "Meshless", "FEM"])
def test_the_refusals_advice_runs(family):
    """Followed: the shared BC keeps NEUMANN(0.7) for the HJB, and the FP is given its own no-flux BC.

    The FP solves under the drift into the x_max wall, and the wall holds: the density piles up against it
    and the mass stays. Completion alone passed a wall-less solve, FP-GFDM's, whose density stayed uniform
    (#2583). An HJB solver on the same problem still reads g = 0.7. FP-FEM runs on a 16-element mesh of the
    same interval, beside HJB-FEM.
    """
    if family == "FEM":
        from mfgarchon.alg.numerical.fem.fp_fem_solver import FPFEMSolver
        from mfgarchon.alg.numerical.fem.hjb_fem_solver import HJBFEMSolver

        problem = _mesh_problem(neumann_bc(value=0.7, dimension=1), 1)
        solver = FPFEMSolver(problem, order=1, boundary_conditions=no_flux_bc(dimension=1))
        x = solver._disc.dof_coordinates[:, 0]
        hjb = HJBFEMSolver(problem)
    else:
        from mfgarchon.alg.numerical.hjb_solvers import HJBFDMSolver

        problem = _grid_problem(neumann_bc(value=0.7, dimension=1), 1)
        builders = {"FDM": lambda p, bc: FPFDMSolver(p, boundary_conditions=bc), **_families_with_an_explicit_route()}
        solver = builders[family](problem, no_flux_bc(dimension=1))
        x = np.linspace(0.0, 1.0, 21)
        hjb = HJBFDMSolver(problem)
    M = np.asarray(
        solver.solve_fp_system(M_initial=np.ones(x.size), potential_field=np.broadcast_to(-x, (NT + 1, x.size)))
    )
    assert np.shape(M) == (NT + 1, x.size)
    assert np.all(np.isfinite(M))
    right, middle, left = (int(np.argmin(np.abs(x - at))) for at in (1.0, 0.5, 0.0))
    # Measured: m(T) at x_max over the midpoint is 11.4 (FDM), 19.2 (FVM), 8.9 (SL), 7.7 (Particle), 20.7
    # (Meshless) and 14.6 (FEM); a wall-less solve gives 1.0.
    assert M[-1, right] > 3.0 * M[-1, middle] > 3.0 * M[-1, left]
    order = np.argsort(x)
    assert abs(float(np.trapezoid(M[-1, order], x[order])) - 1.0) < 0.05
    hjb_bc = hjb.get_boundary_conditions()
    assert [(seg.bc_type, seg.value) for seg in hjb_bc.segments] == [(BCType.NEUMANN, 0.7)]


@pytest.mark.parametrize(
    ("shared", "what"),
    [
        (lambda: neumann_bc(value=lambda t: 0.7, dimension=1), "<callable>"),
        (
            lambda: BoundaryConditions(
                dimension=1,
                segments=[
                    BCSegment(name="left", bc_type=BCType.DIRICHLET, value=0.0, boundary="x_min"),
                    BCSegment(name="right", bc_type=BCType.DIRICHLET, value=0.0, boundary="x_max"),
                ],
                default_bc=BCType.NEUMANN,
                default_value=0.7,
            ),
            "default_bc=NEUMANN, the fall-through",
        ),
    ],
    ids=["callable", "fall-through"],
)
def test_a_value_not_provably_zero_is_refused(shared, what):
    """A callable is not assumed zero, and the default_bc fall-through is a value too (#1686)."""
    with pytest.raises(NotImplementedError, match=SHARED_REFUSAL) as excinfo:
        FPFDMSolver(_grid_problem(shared(), 1))
    assert what in str(excinfo.value)
