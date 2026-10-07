"""A shared DIRICHLET(g) is an exit, so FP-FDM and FP-FEM absorb there (#2525, #2512 convention row 5).

The HJB reads the shared BC as the exit cost u = g; the FP solvers that accept DIRICHLET read it as an
absorbing wall, m = 0, through `bc_utils.fp_view_of_shared_bc`. A BC passed to FP-FDM explicitly is its
own, and there DIRICHLET(g) is a prescribed density m = g. Both solvers read the shared value literally:
on `e8633fe1`, `problem.solve(scheme="fdm_upwind")` with the exit on x_max pinned it at M = 0.7 and the
mass rose 1.0000 -> 3.3247 (the coupled test's fixture, converged); FP-FEM held its Dirichlet wall at
0.7000.

The fixtures put the exit on ONE wall, x_max, with no-flux on x_min. A symmetric fixture (both walls
Dirichlet, sin(pi x)) cannot tell a reader that translates one wall from one that translates both.
"""

import pytest

import numpy as np

from mfgarchon import Conditions, MFGProblem, Model
from mfgarchon.alg.numerical.fp_solvers.fp_fdm import FPFDMSolver
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import BCSegment, BCType, BoundaryConditions

G = 0.7
SIGMA = 0.4
T = 0.5
D = SIGMA**2 / 2


def _exit_on_x_max(dimension: int, g: float = G) -> BoundaryConditions:
    segments = [
        BCSegment(name="wall", bc_type=BCType.NO_FLUX, boundary="x_min"),
        BCSegment(name="exit", bc_type=BCType.DIRICHLET, boundary="x_max", value=g),
    ]
    if dimension == 2:
        segments += [
            BCSegment(name="bottom", bc_type=BCType.NO_FLUX, boundary="y_min"),
            BCSegment(name="top", bc_type=BCType.NO_FLUX, boundary="y_max"),
        ]
    return BoundaryConditions(dimension=dimension, segments=segments)


def _mode(x):
    """cos(pi x / 2): zero flux at x = 0, zero at x = 1. Under pure diffusion it decays as
    exp(-D (pi/2)^2 t) exactly, so the absorbing reading has a closed-form answer."""
    return np.cos(np.pi * np.asarray(x, dtype=float) / 2)


def _hamiltonian():
    return SeparableHamiltonian(control_cost=QuadraticControlCost(control_cost=1.0))


def _problem_1d(n: int, nt: int, m_initial) -> MFGProblem:
    x = np.linspace(0.0, 1.0, n)
    scale = 1.0 / np.trapezoid(m_initial(x), x)  # unit mass on the grid measure
    grid = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[n], boundary_conditions=_exit_on_x_max(1))
    return MFGProblem(
        model=Model(hamiltonian=_hamiltonian(), volatility=SIGMA),
        domain=grid,
        conditions=Conditions(
            m_initial=lambda x: scale * m_initial(x),
            u_terminal=lambda x: 0.0 * np.asarray(x, dtype=float),
            T=T,
        ),
        Nt=nt,
    )


@pytest.mark.parametrize(("n", "nt", "tol"), [(21, 10, 1e-3), (41, 40, 3e-4)])
def test_fp_fdm_absorbs_a_shared_exit_and_matches_the_decaying_mode(n, nt, tol):
    """Measured 4.8e-04 / 1.2e-04 / 3.0e-05 at 21 / 41 / 81 points, second order. Reading the exit
    literally leaves the wall at g = 0.7, an error of 0.7."""
    x = np.linspace(0.0, 1.0, n)
    m0 = _mode(x)
    M = np.asarray(FPFDMSolver(_problem_1d(n, nt, _mode)).solve_fp_system(m0, drift_field=np.zeros((nt + 1, n))))
    assert np.all(M[1:, -1] == 0.0), "the exit wall must hold m = 0 at every step"
    np.testing.assert_allclose(M[-1], m0 * np.exp(-D * (np.pi / 2) ** 2 * T), atol=tol)


def test_fp_fdm_keeps_an_explicit_dirichlet_as_a_prescribed_density():
    """The other half of the convention: the same BC handed to the FP solver is its own, m = g."""
    n, nt = 21, 10
    x = np.linspace(0.0, 1.0, n)
    solver = FPFDMSolver(_problem_1d(n, nt, _mode), boundary_conditions=_exit_on_x_max(1))
    M = np.asarray(solver.solve_fp_system(_mode(x), drift_field=np.zeros((nt + 1, n))))
    np.testing.assert_allclose(M[1:, -1], G)


def test_a_coupled_exit_cost_holds_u_at_g_and_absorbs_m():
    """Through `problem.solve`, the route #2525 was measured on. The bump sits nearer the exit, so the
    drift and the diffusion both reach it; read literally (main, e8633fe1), the mass rose 1.0000 -> 3.3247 here."""
    problem = _problem_1d(41, 20, lambda x: np.exp(-20 * (np.asarray(x, dtype=float) - 0.6) ** 2))
    result = problem.solve(scheme="fdm_upwind", max_iterations=40, tolerance=1e-6)
    U, M = np.asarray(result.U), np.asarray(result.M)
    np.testing.assert_allclose(U[:-1, -1], G, atol=1e-6)  # the last row is the terminal data
    # Picard damps each iterate toward the initial guess, which is nonzero at the exit, so the exit
    # reaches 0 only to the Picard tolerance (4.9e-08 measured); read literally it was 0.7.
    assert np.abs(M[1:, -1]).max() < 1e-6
    mass = np.array([problem.geometry.integrate(m) for m in M])
    assert np.all(np.diff(mass) <= 0.0), f"an absorbing wall cannot add mass: {np.diff(mass).max():.2e}"
    assert mass[-1] < mass[0] - 1e-3


def _fem_problem(g: float):
    import skfem

    from mfgarchon.alg.numerical.fem.mesh_adapter import skfem_to_meshdata
    from mfgarchon.geometry.meshes.mesh_2d import Mesh2D

    mesh = skfem.MeshTri.init_sqsymmetric().refined(3)
    scale = 1.0 / np.mean(_mode(mesh.p[0]))  # unit mass on the point-average measure a mesh is checked with
    geometry = Mesh2D(domain_type="rectangle", bounds=(0.0, 1.0, 0.0, 1.0))
    geometry.mesh_data = skfem_to_meshdata(mesh)
    geometry.boundary_conditions = _exit_on_x_max(2, g)
    return MFGProblem(
        model=Model(hamiltonian=_hamiltonian(), volatility=SIGMA),
        domain=geometry,
        conditions=Conditions(m_initial=lambda p: scale * _mode(p[0]), u_terminal=lambda p: 0.0, T=T),
        Nt=20,
    )


def _fem_solve(g: float):
    from mfgarchon.alg.numerical.fem.fp_fem_solver import FPFEMSolver

    solver = FPFEMSolver(_fem_problem(g), order=1)
    x = solver._disc.dof_coordinates[:, 0]
    M = np.asarray(solver.solve_fp_system(M_initial=_mode(x), potential_field=np.zeros((21, x.size))))
    return x, M


def test_fp_fem_absorbs_a_shared_exit_and_matches_the_decaying_mode():
    """Measured 4.4e-03 / 1.5e-03 / 6.3e-04 at refined(2) / (3) / (4), Nt = 20. Condensed literally, the
    exit dofs held 0.7000."""
    x, M = _fem_solve(G)
    exit_dofs = np.isclose(x, 1.0)
    assert np.all(M[1:, exit_dofs] == 0.0), "the exit wall must hold m = 0 at every step"
    np.testing.assert_allclose(M[-1], _mode(x) * np.exp(-D * (np.pi / 2) ** 2 * T), atol=3e-3)


def test_fp_fem_drops_the_shared_value_by_design():
    """The value of a shared Dirichlet belongs to the HJB: g = 0.7 solves bit-identically to g = 0."""
    np.testing.assert_array_equal(_fem_solve(G)[1], _fem_solve(0.0)[1])


def _refusal_text(family: str) -> str:
    """The FP Neumann refusal as `family` prints it, for a shared neumann_bc(0.7)."""
    from mfgarchon.geometry.boundary import neumann_bc

    if family == "FEM":
        from mfgarchon.alg.numerical.fem.fp_fem_solver import FPFEMSolver

        problem = _fem_problem(G)
        problem.geometry.boundary_conditions = neumann_bc(dimension=2, value=0.7)
        build = lambda: FPFEMSolver(problem, order=1)  # noqa: E731
    else:
        grid = TensorProductGrid(
            bounds=[(0.0, 1.0)], Nx_points=[21], boundary_conditions=neumann_bc(dimension=1, value=0.7)
        )
        problem = MFGProblem(
            model=Model(hamiltonian=_hamiltonian(), volatility=SIGMA),
            domain=grid,
            conditions=Conditions(
                m_initial=lambda x: 1.0 + 0.0 * np.asarray(x, dtype=float),
                u_terminal=lambda x: 0.0 * np.asarray(x, dtype=float),
                T=T,
            ),
            Nt=10,
        )
        if family == "FDM":
            build = lambda: FPFDMSolver(problem)  # noqa: E731
        elif family == "FVM":
            from mfgarchon.alg.numerical.fp_solvers.fp_fvm import FPFVMSolver

            build = lambda: FPFVMSolver(problem)  # noqa: E731
        elif family == "GFDM":
            from mfgarchon.alg.numerical.fp_solvers.fp_gfdm import FPGFDMSolver

            build = lambda: FPGFDMSolver(problem, collocation_points=np.linspace(0.0, 1.0, 21).reshape(-1, 1))  # noqa: E731
        elif family == "SL":
            from mfgarchon.alg.numerical.fp_solvers.fp_semi_lagrangian_adjoint import FPSLSolver

            build = lambda: FPSLSolver(problem)  # noqa: E731
        else:
            from mfgarchon.alg.numerical.fp_solvers.fp_particle import FPParticleSolver

            build = lambda: FPParticleSolver(problem, num_particles=200, seed=0)  # noqa: E731
    with pytest.raises(NotImplementedError) as exc:
        build()
    return str(exc.value)


@pytest.mark.parametrize("family", ["FDM", "FEM", "FVM", "GFDM", "SL", "Particle"])
def test_the_neumann_refusal_advice_is_true_for_every_solver_that_prints_it(family):
    """The FP Neumann refusal used to advise "a Dirichlet segment if you meant a prescribed density"; on a
    shared BC that is now an exit, so following it silently solved a different problem. Its replacement
    must hold for all six FP families that print it: FVM, GFDM and SL refuse DIRICHLET at all, FEM takes
    no explicit BC, and the particle FP refuses a nonzero one -- only FPFDMSolver imposes m = g."""
    text = _refusal_text(family)
    assert "imposed only by FPFDMSolver" in text
    assert "a Dirichlet segment if you meant" not in text
    assert "the FP absorbs" not in text
