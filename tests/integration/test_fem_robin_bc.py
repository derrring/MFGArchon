"""Integration tests for FEM Robin boundary conditions (Issue #1237).

Robin ``alpha*u + beta*du/dn = g`` is implemented as an OPERATOR AUGMENTATION at the
weak-form level: integrating the diffusion operator ``-D*Delta u`` by parts and substituting
``du/dn = (g - alpha*u)/beta`` adds

    A_robin   = D*(alpha/beta) * int_dOmega phi_i phi_j   (boundary mass -> operator)
    rhs_robin = D*(1/beta)     * int_dOmega g phi_i        (boundary load -> RHS)

to ``M/dt + D*K``. Both scale with ``D`` exactly like the stiffness, and the boundary mass is
symmetric (so the FP term is the adjoint of the HJB term). The term coexists with Dirichlet
condensation (Robin dofs stay free).

The non-negotiable gate is the **1D rod convergence test**: a closed-form Robin solution solved
on refined meshes must converge at the P1 rate (~O(h^2)). A wrong sign, D-scaling, or alpha/beta
factor will NOT converge. Periodic FEM BC stays fail-loud (deferred per the issue).
"""

from __future__ import annotations

import pytest

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.core.mfg_components import MFGComponents
from mfgarchon.core.mfg_problem import MFGProblem
from mfgarchon.geometry.boundary.conditions import BCSegment, BCType, BoundaryConditions
from mfgarchon.geometry.meshes.mesh_1d import Mesh1D

# scikit-fem is needed only for the FEM solver classes (imported lazily inside tests); the
# module-level mfgarchon imports above are skfem-independent. importorskip skips the whole file
# (and thus the lazy FEM imports) when scikit-fem is unavailable.
skfem = pytest.importorskip("skfem", reason="scikit-fem required for FEM tests")

# Manufactured 1D problem on [0, 1]:  u*(x) = sin(2x),  -D u'' = 4 D sin(2x).
# Outward normal is -1 at x=0 and +1 at x=1, so du/dn = -u'(0) at the left end, +u'(1) at the right.
_ALPHA, _BETA = 1.0, 1.0


def _sigma_D(sigma: float = 1.0) -> tuple[float, float]:
    return sigma, sigma**2 / 2.0


def _u_star(x: np.ndarray) -> np.ndarray:
    return np.sin(2.0 * x)


def _source(x: np.ndarray, D: float) -> np.ndarray:
    return 4.0 * D * np.sin(2.0 * x)  # -D u''


def _robin_g_ends(D: float) -> tuple[float, float]:
    g_left = _ALPHA * np.sin(0.0) + _BETA * (-2.0 * np.cos(0.0))  # alpha*u(0) + beta*(-u'(0))
    g_right = _ALPHA * np.sin(2.0) + _BETA * (2.0 * np.cos(2.0))  # alpha*u(1) + beta*(+u'(1))
    return float(g_left), float(g_right)


def _robin_problem_1d(num_elements: int, sigma: float, g_left: float, g_right: float, hamiltonian=None):
    """A 1D line-mesh MFG problem with Robin BC on both ends."""
    geom = Mesh1D(bounds=(0.0, 1.0), num_elements=num_elements)
    geom.generate_mesh()
    geom.boundary_conditions = BoundaryConditions(
        dimension=1,
        segments=[
            BCSegment(name="L", bc_type=BCType.ROBIN, alpha=_ALPHA, beta=_BETA, value=g_left, boundary="x_min"),
            BCSegment(name="R", bc_type=BCType.ROBIN, alpha=_ALPHA, beta=_BETA, value=g_right, boundary="x_max"),
        ],
    )
    if hamiltonian is None:
        hamiltonian = SeparableHamiltonian(control_cost=QuadraticControlCost(lambda_=1.0), coupling=lambda m: 0.0)
    components = MFGComponents(m_initial=lambda x: 1.0, u_terminal=lambda x: 0.0, hamiltonian=hamiltonian)
    return MFGProblem(geometry=geom, T=0.1, Nt=2, sigma=sigma, components=components, coupling_coefficient=0.0)


def _l2(err: np.ndarray, M: sparse.spmatrix) -> float:
    return float(np.sqrt(err @ (M @ err)))


def _slopes(errs: np.ndarray, hs: np.ndarray) -> np.ndarray:
    return np.log(errs[:-1] / errs[1:]) / np.log(hs[:-1] / hs[1:])


@pytest.mark.integration
class TestRobinConvergence:
    """The correctness gate: a closed-form Robin solution must converge at the P1 rate."""

    @pytest.mark.parametrize("solver_name", ["hjb", "fp"])
    def test_robin_steady_convergence_1d(self, solver_name):
        """1D rod with Robin on both ends, manufactured u*=sin(2x): the L2 error must decrease at
        ~O(h^2). Driven through the solver's real ``_robin_operator_terms`` hook (the assembled
        boundary mass + load), combined with the solver's own ``_K``/``_M``. A wrong sign /
        D-scaling / (alpha/beta) factor will not converge -- this is the physics gate.

        Both HJB and FP are checked: their Robin terms are identical (the boundary mass is
        symmetric, so the FP term is the adjoint of the HJB term)."""
        from mfgarchon.alg.numerical.fem.fp_fem_solver import FPFEMSolver
        from mfgarchon.alg.numerical.fem.hjb_fem_solver import HJBFEMSolver

        sigma, D = _sigma_D()
        g_left, g_right = _robin_g_ends(D)
        solver_cls = HJBFEMSolver if solver_name == "hjb" else FPFEMSolver

        errs, hs = [], []
        for ne in [8, 16, 32, 64]:
            problem = _robin_problem_1d(ne, sigma, g_left, g_right)
            solver = solver_cls(problem, order=1)
            A_robin, rhs_robin = solver._robin_operator_terms(D)
            assert A_robin is not None, "Robin hook returned no-op operator for a Robin problem"
            assert rhs_robin is not None, "Robin hook returned no-op load for a Robin problem"
            x = solver._disc.dof_coordinates[:, 0]
            A = (D * solver._K + A_robin).tocsc()
            b = solver._M @ _source(x, D) + rhs_robin
            u_h = spsolve(A, b)
            errs.append(_l2(u_h - _u_star(x), solver._M))
            hs.append(1.0 / ne)

        errs, hs = np.array(errs), np.array(hs)
        slopes = _slopes(errs, hs)
        # Monotone decrease, P1 rate on the finest pair, tight final tolerance.
        assert np.all(np.diff(errs) < 0.0), f"L2 error not monotone decreasing: {errs}"
        assert slopes[-1] > 1.9, f"{solver_name} Robin L2 slope {slopes[-1]:.3f} below P1 rate (~2); term is wrong"
        assert errs[-1] < 5e-4, f"{solver_name} Robin finest-mesh L2 error {errs[-1]:.2e} too large"

    def test_2d_wall_robin_convergence(self):
        """The FacetBasis path over a 2D wall (an edge, not a 1D point): u*(x,y)=sin(2x) (y-flat),
        Robin on x_min/x_max, natural Neumann on the y-walls. Must converge at ~O(h^2). Guards the
        boundary-mass assembly over real facets (not just the 1D point-facet special case)."""
        from mfgarchon.alg.numerical.fem.hjb_fem_solver import HJBFEMSolver
        from mfgarchon.alg.numerical.fem.mesh_adapter import skfem_to_meshdata
        from mfgarchon.geometry.meshes.mesh_2d import Mesh2D

        sigma, D = _sigma_D()
        g_left = _ALPHA * np.sin(0.0) + _BETA * (-2.0 * np.cos(0.0))
        g_right = _ALPHA * np.sin(2.0) + _BETA * (2.0 * np.cos(2.0))

        errs, hs = [], []
        for r in [1, 2, 3, 4]:
            mesh = skfem.MeshTri.init_sqsymmetric().refined(r)
            geom = Mesh2D(domain_type="rectangle", bounds=(0.0, 1.0, 0.0, 1.0))
            geom.mesh_data = skfem_to_meshdata(mesh)
            geom.boundary_conditions = BoundaryConditions(
                dimension=2,
                segments=[
                    BCSegment(name="L", bc_type=BCType.ROBIN, alpha=_ALPHA, beta=_BETA, value=g_left, boundary="x_min"),
                    BCSegment(
                        name="R", bc_type=BCType.ROBIN, alpha=_ALPHA, beta=_BETA, value=g_right, boundary="x_max"
                    ),
                ],
            )
            components = MFGComponents(
                m_initial=lambda x: 1.0,
                u_terminal=lambda x: 0.0,
                hamiltonian=SeparableHamiltonian(
                    control_cost=QuadraticControlCost(lambda_=1.0), coupling=lambda m: 0.0
                ),
            )
            problem = MFGProblem(
                geometry=geom, T=0.1, Nt=2, sigma=sigma, components=components, coupling_coefficient=0.0
            )
            solver = HJBFEMSolver(problem, order=1)
            X = solver._disc.dof_coordinates
            A_robin, rhs_robin = solver._robin_operator_terms(D)
            u_h = spsolve((D * solver._K + A_robin).tocsc(), solver._M @ (4.0 * D * np.sin(2.0 * X[:, 0])) + rhs_robin)
            errs.append(_l2(u_h - np.sin(2.0 * X[:, 0]), solver._M))
            hs.append(2.0**-r)

        errs, hs = np.array(errs), np.array(hs)
        slopes = _slopes(errs, hs)
        assert np.all(np.diff(errs) < 0.0), f"2D Robin L2 error not monotone: {errs}"
        assert slopes[-1] > 1.85, f"2D wall Robin L2 slope {slopes[-1]:.3f} below P1 rate (~2)"


@pytest.mark.integration
class TestRobinSolveLoopWiring:
    """The Robin terms must actually be folded into the solve loops (operator + each-timestep RHS),
    not just exposed by the hook. Verified by fixed-point reproduction: if the loop omits A_robin
    or rhs_robin, the discrete steady solution is no longer a fixed point and the output drifts."""

    def test_hjb_linear_loop_fixed_point(self):
        """``solve_hjb_system`` (linear/Picard path) folds A_robin into the operator and rhs_robin
        into each RHS. The Hamiltonian potential V(x)=4D sin(2x) is the source (evaluated at the
        zero previous-iterate gradient), so the discrete steady solution u_h* is an exact fixed
        point; the solver must reproduce it to machine precision."""
        from mfgarchon.alg.numerical.fem.hjb_fem_solver import HJBFEMSolver

        sigma, D = _sigma_D()
        g_left, g_right = _robin_g_ends(D)
        ne = 48
        ham = SeparableHamiltonian(
            control_cost=QuadraticControlCost(lambda_=1.0),
            potential=lambda x, t: 4.0 * D * np.sin(2.0 * x[:, 0]),
            coupling=lambda m: 0.0,
        )
        problem = _robin_problem_1d(ne, sigma, g_left, g_right, hamiltonian=ham)
        solver = HJBFEMSolver(problem, order=1)
        N = solver.n_dof
        x = solver._disc.dof_coordinates[:, 0]
        A_robin, rhs_robin = solver._robin_operator_terms(D)
        u_steady = spsolve((D * solver._K + A_robin).tocsc(), solver._M @ _source(x, D) + rhs_robin)

        U = solver.solve_hjb_system(
            M_density=np.ones((problem.Nt + 1, N)) / N,
            U_terminal=u_steady.copy(),
            U_coupling_prev=np.zeros((problem.Nt + 1, N)),
        )
        assert np.all(np.isfinite(U)), "solve_hjb_system produced non-finite output on a Robin problem"
        assert np.abs(U[0] - u_steady).max() < 1e-9, "Robin terms not folded into the HJB linear loop"

    def test_hjb_newton_loop_fixed_point(self):
        """``solve_hjb_system(use_newton=True)`` folds A_robin into the Jacobian and the residual
        (and rhs_robin into the residual). Source-free linear manufactured solution (control cost
        made negligible, so H~0), whose discrete steady solution is a fixed point of the Newton
        timestep -- reproduced to machine precision."""
        from mfgarchon.alg.numerical.fem.hjb_fem_solver import HJBFEMSolver

        sigma, D = _sigma_D()
        ne = 32
        u_lin = lambda x: 0.5 + 0.3 * x  # noqa: E731 - linear, -D u''=0, P1-exact
        g_left = _ALPHA * u_lin(0.0) + _BETA * (-0.3)
        g_right = _ALPHA * u_lin(1.0) + _BETA * (0.3)
        ham = SeparableHamiltonian(control_cost=QuadraticControlCost(lambda_=1e12), coupling=lambda m: 0.0)
        problem = _robin_problem_1d(ne, sigma, g_left, g_right, hamiltonian=ham)
        solver = HJBFEMSolver(problem, order=1)
        N = solver.n_dof
        x = solver._disc.dof_coordinates[:, 0]
        A_robin, rhs_robin = solver._robin_operator_terms(D)
        u_steady = spsolve((D * solver._K + A_robin).tocsc(), rhs_robin)  # no source
        assert np.abs(u_steady - u_lin(x)).max() < 1e-10, "linear Robin solution not P1-exact"

        U = solver.solve_hjb_system(
            M_density=np.ones((problem.Nt + 1, N)) / N,
            U_terminal=u_steady.copy(),
            U_coupling_prev=np.zeros((problem.Nt + 1, N)),
            use_newton=True,
        )
        assert np.all(np.isfinite(U))
        assert np.abs(U[0] - u_steady).max() < 1e-8, "Robin terms not folded into the HJB Newton loop"

    def test_fp_forward_loop_fixed_point(self):
        """``solve_fp_system`` (pure diffusion, no advection) folds A_robin + rhs_robin. A linear
        m*=0.5+0.3x (positive, P1-exact, -D m''=0) is the discrete steady solution; starting from
        it, every step must reproduce it. Also confirms the FP Robin term is the symmetric adjoint
        of the HJB one (same boundary mass)."""
        from mfgarchon.alg.numerical.fem.fp_fem_solver import FPFEMSolver

        sigma, D = _sigma_D()
        ne = 40
        m_lin = lambda x: 0.5 + 0.3 * x  # noqa: E731
        g_left = _ALPHA * m_lin(0.0) + _BETA * (-0.3)
        g_right = _ALPHA * m_lin(1.0) + _BETA * (0.3)
        problem = _robin_problem_1d(ne, sigma, g_left, g_right)
        solver = FPFEMSolver(problem, order=1)
        x = solver._disc.dof_coordinates[:, 0]
        A_robin, rhs_robin = solver._robin_operator_terms(D)
        m_steady = spsolve((D * solver._K + A_robin).tocsc(), rhs_robin)
        assert np.abs(m_steady - m_lin(x)).max() < 1e-10

        M = solver.solve_fp_system(m_steady.copy(), potential_field=None)
        assert np.all(np.isfinite(M))
        assert np.abs(M[-1] - m_steady).max() < 1e-9, "Robin terms not folded into the FP forward loop"

    def test_fp_adjoint_mode_fixed_point(self):
        """``solve_fp_step_adjoint_mode`` folds A_robin + rhs_robin. With a zero externally supplied
        advection, the linear discrete steady m* is reproduced (Robin term carried through the
        adjoint timestep)."""
        from mfgarchon.alg.numerical.fem.fp_fem_solver import FPFEMSolver

        sigma, D = _sigma_D()
        ne = 40
        m_lin = lambda x: 0.5 + 0.3 * x  # noqa: E731
        g_left = _ALPHA * m_lin(0.0) + _BETA * (-0.3)
        g_right = _ALPHA * m_lin(1.0) + _BETA * (0.3)
        problem = _robin_problem_1d(ne, sigma, g_left, g_right)
        solver = FPFEMSolver(problem, order=1)
        N = solver.n_dof
        A_robin, rhs_robin = solver._robin_operator_terms(D)
        m_steady = spsolve((D * solver._K + A_robin).tocsc(), rhs_robin)

        m_next = solver.solve_fp_step_adjoint_mode(
            m_steady.copy().reshape(-1, 1), sparse.csr_matrix((N, N)), sigma=sigma
        )
        assert np.abs(m_next.ravel() - m_steady).max() < 1e-9, "Robin terms not folded into the FP adjoint mode"

    def test_pure_robin_no_dirichlet_solves(self):
        """A pure-Robin problem (no Dirichlet segments) must solve: the Dirichlet dof set is empty
        so all dofs are free and the augmented operator is solved on the full system (no raise)."""
        from mfgarchon.alg.numerical.fem.hjb_fem_solver import HJBFEMSolver

        sigma, D = _sigma_D()
        g_left, g_right = _robin_g_ends(D)
        problem = _robin_problem_1d(24, sigma, g_left, g_right)
        solver = HJBFEMSolver(problem, order=1)
        assert not solver._is_pure_neumann()  # Robin routes to the (empty-Dirichlet) condense branch
        d_dofs, _ = solver._dirichlet_dofs_and_values()
        assert len(d_dofs) == 0, "Robin segments must be excluded from the Dirichlet dof set"
        N = solver.n_dof
        U = solver.solve_hjb_system(M_density=np.ones((problem.Nt + 1, N)) / N, U_terminal=np.zeros(N))
        assert np.all(np.isfinite(U))


@pytest.mark.integration
class TestRobinFailLoud:
    """Scope boundaries: beta=0 / callable g are deferred (fail loud), Periodic stays deferred."""

    def test_robin_beta_zero_fails_loud(self):
        from mfgarchon.alg.numerical.fem.assembly import create_basis
        from mfgarchon.alg.numerical.fem.bc_adapter import assemble_robin_terms

        mesh = skfem.MeshLine(np.linspace(0, 1, 5)).with_boundaries({"x_min": lambda x: np.isclose(x[0], 0.0)})
        basis = create_basis(mesh, order=1)
        bc = BoundaryConditions(
            dimension=1,
            segments=[BCSegment(name="L", bc_type=BCType.ROBIN, alpha=1.0, beta=0.0, value=1.0, boundary="x_min")],
        )
        with pytest.raises(NotImplementedError, match="beta=0"):
            assemble_robin_terms(basis, bc, D=0.5)

    def test_robin_callable_value_fails_loud(self):
        from mfgarchon.alg.numerical.fem.assembly import create_basis
        from mfgarchon.alg.numerical.fem.bc_adapter import assemble_robin_terms

        mesh = skfem.MeshLine(np.linspace(0, 1, 5)).with_boundaries({"x_min": lambda x: np.isclose(x[0], 0.0)})
        basis = create_basis(mesh, order=1)
        bc = BoundaryConditions(
            dimension=1,
            segments=[
                BCSegment(name="L", bc_type=BCType.ROBIN, alpha=1.0, beta=1.0, value=lambda x: x[0], boundary="x_min")
            ],
        )
        with pytest.raises(NotImplementedError, match="non-constant"):
            assemble_robin_terms(basis, bc, D=0.5)

    def test_periodic_still_fails_loud(self):
        """Periodic FEM BC remains deferred (Issue #1237): the condensation adapter must still
        raise. This is unchanged behavior -- the Robin work does not touch Periodic."""
        from mfgarchon.alg.numerical.fem.assembly import assemble_stiffness, create_basis
        from mfgarchon.alg.numerical.fem.bc_adapter import apply_bc_to_fem_system

        mesh = skfem.MeshTri.init_sqsymmetric().refined(1)
        basis = create_basis(mesh, order=1)
        A = assemble_stiffness(basis)
        bc = BoundaryConditions(
            dimension=2, segments=[BCSegment(name="p", bc_type=BCType.PERIODIC, alpha=1.0, beta=1.0)]
        )
        with pytest.raises(NotImplementedError, match="Periodic"):
            apply_bc_to_fem_system(A, np.zeros(basis.N), basis, bc)


@pytest.mark.integration
class TestRobinHookIsNoOpWithoutRobin:
    """The new hook must be a strict no-op when no Robin segment is present (so the existing
    natural/Dirichlet FEM paths and the meshless Nitsche path are byte-unchanged)."""

    def test_fem_hook_noop_for_neumann(self):
        from mfgarchon.alg.numerical.fem.fp_fem_solver import FPFEMSolver
        from mfgarchon.alg.numerical.fem.hjb_fem_solver import HJBFEMSolver
        from mfgarchon.geometry.boundary import no_flux_bc

        geom = Mesh1D(bounds=(0.0, 1.0), num_elements=8)
        geom.generate_mesh()
        geom.boundary_conditions = no_flux_bc(dimension=1)
        components = MFGComponents(
            m_initial=lambda x: 1.0,
            u_terminal=lambda x: 0.0,
            hamiltonian=SeparableHamiltonian(control_cost=QuadraticControlCost(lambda_=1.0), coupling=lambda m: 0.0),
        )
        problem = MFGProblem(geometry=geom, T=0.1, Nt=2, sigma=1.0, components=components, coupling_coefficient=0.0)
        for solver in (HJBFEMSolver(problem), FPFEMSolver(problem)):
            assert solver._robin_operator_terms(0.5) == (None, None)

    def test_meshless_does_not_override_robin_hook(self):
        """The meshless-Galerkin solvers must inherit the base no-op Robin hook (their Nitsche path
        is untouched), so the #1145 EOC is byte-unperturbed."""
        from mfgarchon.alg.numerical.meshless_galerkin.fp_solver import MeshlessGalerkinFPSolver
        from mfgarchon.alg.numerical.meshless_galerkin.hjb_solver import MeshlessGalerkinHJBSolver
        from mfgarchon.alg.numerical.weak_form_fp_solver import WeakFormFPSolver
        from mfgarchon.alg.numerical.weak_form_hjb_solver import WeakFormHJBSolver

        assert MeshlessGalerkinHJBSolver._robin_operator_terms is WeakFormHJBSolver._robin_operator_terms
        assert MeshlessGalerkinFPSolver._robin_operator_terms is WeakFormFPSolver._robin_operator_terms


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
