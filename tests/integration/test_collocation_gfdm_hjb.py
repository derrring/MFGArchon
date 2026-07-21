import pytest

import numpy as np

from mfgarchon.alg.numerical.hjb_solvers.hjb_gfdm import HJBGFDMSolver as GFDMHJBSolver
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.core.mfg_components import MFGComponents
from mfgarchon.core.mfg_problem import MFGProblem
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import no_flux_bc


class MockGeometry:
    """Mock geometry for testing."""

    def __init__(self, n_points: int = 10):
        self._n_points = n_points

    def get_grid_shape(self) -> tuple[int, ...]:
        """Return grid shape (1D with n_points)."""
        return (self._n_points,)

    def get_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Return domain bounds."""
        return np.array([0.0]), np.array([1.0])


class MockMFGProblem:
    """Mock MFG problem for testing."""

    def __init__(self):
        # Nx is number of intervals, so Nx+1 is number of grid points
        # With 10 collocation points, we need Nx=9 (so Nx+1=10 grid points)
        self.Nx = 9
        self.Nt = 5
        self.Dx = 0.1
        self.Dt = 0.2
        self.sigma = 0.1
        self.geometry = MockGeometry(n_points=10)

    def H(self, x_idx, m_at_x, p_values, t_idx):
        """Simple quadratic Hamiltonian."""
        if "forward" in p_values:
            p = p_values["forward"]
        elif "x" in p_values:
            p = p_values["x"]
        else:
            p = 0.0
        return 0.5 * p**2

    def get_hjb_hamiltonian_jacobian_contrib(self, U_prev, t_idx):
        """Return None to use numerical Jacobian."""
        return None

    def get_hjb_residual_m_coupling_term(self, M_density, U_derivatives, i, t_idx):
        """No m-coupling term for this test."""
        return None


class TestGFDMHJBSolver:
    """Test suite for the Collocation GFDM HJB solver."""

    def setup_method(self):
        """Set up test fixtures."""
        self.problem = MockMFGProblem()

        # Create simple 1D collocation points
        self.collocation_points = np.linspace(0, 1, 10).reshape(-1, 1)

        # Create solver
        self.solver = GFDMHJBSolver(
            problem=self.problem,
            collocation_points=self.collocation_points,
            delta=0.3,
            taylor_order=2,
            max_newton_iterations=5,
            newton_tolerance=1e-4,
        )

    def test_initialization(self):
        """Test solver initialization."""
        assert self.solver.hjb_method_name == "GFDM"
        assert self.solver.n_points == 10
        assert self.solver.dimension == 1
        assert self.solver.delta == 0.3
        assert self.solver.taylor_order == 2
        assert len(self.solver.neighborhoods) == 10
        assert len(self.solver.taylor_matrices) == 10

    def test_neighborhood_structure(self):
        """Test neighborhood structure building."""
        # Check that each point has neighbors
        for i in range(self.solver.n_points):
            assert i in self.solver.neighborhoods
            neighborhood = self.solver.neighborhoods[i]
            assert "indices" in neighborhood
            assert "points" in neighborhood
            assert "distances" in neighborhood
            assert "size" in neighborhood
            assert neighborhood["size"] > 0
            # Point should be in its own neighborhood
            assert i in neighborhood["indices"]

    def test_multi_index_generation(self):
        """Test multi-index set generation.

        The multi-indices are now pre-computed during initialization and stored
        as solver.multi_indices. For 1D with taylor_order=2, we expect indices
        for first and second derivatives.
        """
        # 1D solver with taylor_order=2
        multi_indices = self.solver.multi_indices

        # Should have multi-indices for derivatives up to order 2
        # For 1D: (1,) for first derivative, (2,) for second derivative
        assert len(multi_indices) > 0

        # Verify we have first and second derivative indices
        # The exact format depends on implementation
        assert any(sum(idx) == 1 for idx in multi_indices), "Should have first derivative"
        assert any(sum(idx) == 2 for idx in multi_indices), "Should have second derivative"

    def test_taylor_matrix_construction(self):
        """Test Taylor expansion matrix construction."""
        # Check that matrices were built for most points
        non_none_matrices = sum(1 for i in range(self.solver.n_points) if self.solver.taylor_matrices[i] is not None)
        assert non_none_matrices > 0

        # Check matrix structure for a point with sufficient neighbors
        for i in range(self.solver.n_points):
            if self.solver.taylor_matrices[i] is not None:
                taylor_data = self.solver.taylor_matrices[i]
                assert "A" in taylor_data
                assert "W" in taylor_data
                # New API: SVD components stored instead of AtW
                assert "S" in taylor_data or "AtW" in taylor_data  # Support both old and new API

                A = taylor_data["A"]
                W = taylor_data["W"]
                assert A.shape[1] == len(self.solver.multi_indices)
                assert W.shape[0] == W.shape[1]  # Square diagonal matrix
                break

    def test_derivative_approximation(self):
        """Test derivative approximation.

        Uses the public approximate_derivatives() method to verify the GFDM
        derivative approximation is working correctly for a simple polynomial.
        """
        # Create test function u(x) = x^2
        u_values = (self.collocation_points[:, 0] ** 2).flatten()

        # Test derivative approximation at a middle point
        point_idx = 5
        if self.solver.taylor_matrices[point_idx] is not None:
            derivatives = self.solver.approximate_derivatives(u_values, point_idx)

            # For u(x) = x^2, derivative should be approximately 2x
            x_val = self.collocation_points[point_idx, 0]
            expected_first_deriv = 2 * x_val

            # Check first derivative (key format may be (1,) for 1D)
            if (1,) in derivatives:
                actual_first_deriv = derivatives[(1,)]
                # Allow some numerical error (GFDM is approximate)
                assert abs(actual_first_deriv - expected_first_deriv) < 1.0, (
                    f"First derivative error too large: got {actual_first_deriv}, expected {expected_first_deriv}"
                )
        else:
            pytest.skip(f"Taylor matrix not available for point {point_idx}")

    def test_boundary_conditions_dirichlet(self):
        """Test Dirichlet boundary conditions."""
        # Create solver with boundary conditions
        boundary_indices = np.array([0, 9])  # First and last points
        boundary_conditions = {"type": "dirichlet", "value": 1.0}

        solver_with_bc = GFDMHJBSolver(
            problem=self.problem,
            collocation_points=self.collocation_points,
            delta=0.3,
            boundary_indices=boundary_indices,
            boundary_conditions=boundary_conditions,
        )

        # Verify boundary indices are stored
        assert solver_with_bc.boundary_indices is not None
        assert len(solver_with_bc.boundary_indices) == 2

        # Test applying boundary conditions to solution
        u_test = np.zeros(10)
        u_modified = solver_with_bc._apply_boundary_conditions_to_solution(u_test, 0)

        # Boundary values should be set to 1.0
        assert u_modified[0] == 1.0
        assert u_modified[9] == 1.0

    def test_solve_hjb_system_shape(self):
        """solve_hjb_system on a REAL problem returns U of shape (Nt, Nx).

        Issue #1567: the previous version drove ``solve_hjb_system`` through the hand-rolled
        ``MockMFGProblem`` and wrapped it in ``try/except -> pytest.skip``. The mock lacks the
        attributes a full solve needs (``T``, ``dimension``, ...), so the call always raised and
        the skip swallowed it -- the integration-tier solve-shape contract was never verified.
        Rebuilt on a real ``MFGProblem`` (the mock is kept for the unit-surface tests in this
        file that only exercise neighborhoods / Taylor stencils). No ``try/except -> skip``: a
        solver raise is a real failure now."""
        n = 21
        domain = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[n], boundary_conditions=no_flux_bc(dimension=1))
        components = MFGComponents(
            m_initial=lambda x: np.exp(-10 * (np.asarray(x) - 0.5) ** 2),
            u_terminal=lambda x: np.asarray(x) * 0.0,
            hamiltonian=SeparableHamiltonian(control_cost=QuadraticControlCost(control_cost=1.0)),
        )
        problem = MFGProblem(geometry=domain, T=1.0, Nt=5, sigma=0.3, components=components)
        collocation_points = np.linspace(0.0, 1.0, n).reshape(-1, 1)
        solver = GFDMHJBSolver(problem, collocation_points, delta=0.3, monotonicity_scheme="none")

        Nt, Nx = problem.Nt, n
        M_density = np.ones((Nt, Nx)) * 0.1
        U_final = np.zeros(Nx)
        U_prev = np.zeros((Nt, Nx))

        # Solve (may not converge perfectly, but must return the correct shape).
        U_solution = solver.solve_hjb_system(M_density, U_final, U_prev)
        assert U_solution.shape == (Nt, Nx)

    def test_weight_functions(self):
        """Test different weight functions for GFDM weighting (integration test)."""
        distances = np.array([0.0, 0.1, 0.2, 0.3])

        # Test default weight function (wendland) via component
        weights = self.solver._neighborhood_builder.compute_weights(distances)
        assert len(weights) == len(distances)
        # Distance 0 should give max weight
        assert weights[0] >= weights[-1], "Weight should decrease with distance"
        assert np.all(weights >= 0), "Weights should be non-negative"

        # Test uniform weights
        solver_uniform = GFDMHJBSolver(
            problem=self.problem,
            collocation_points=self.collocation_points,
            delta=0.3,
            weight_function="uniform",
        )
        weights_uniform = solver_uniform._neighborhood_builder.compute_weights(distances)
        assert np.all(weights_uniform == 1.0), "Uniform weights should all be 1.0"

    def test_grid_collocation_mapping(self):
        """Test mapping between grid and collocation points (integration test)."""
        # Test when sizes match
        u_grid = np.ones(10)
        # Access component through solver (post-refactoring)
        u_collocation = self.solver._mapper.map_grid_to_collocation(u_grid)
        assert len(u_collocation) == 10

        u_grid_back = self.solver._mapper.map_collocation_to_grid(u_collocation)
        assert len(u_grid_back) == 10

        # Verify round-trip preserves values
        np.testing.assert_array_almost_equal(u_grid, u_grid_back)


if __name__ == "__main__":
    pytest.main([__file__])
