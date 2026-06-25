#!/usr/bin/env python3
"""
Unit tests for FPNetworkSolver.

Tests the Fokker-Planck solver for Mean Field Games on network/graph structures,
including density evolution, mass conservation, and various time discretization schemes.
"""

import pytest

import numpy as np

from mfgarchon.alg.numerical.network_solvers.fp_network import FPNetworkSolver
from mfgarchon.extensions.topology import NetworkMFGProblem
from mfgarchon.geometry.graph.network_geometry import GridNetwork

# Skip all tests if igraph is not available (network backend dependency)
igraph = pytest.importorskip("igraph")


class TestFPNetworkSolverInitialization:
    """Test FPNetworkSolver initialization and configuration."""

    def test_basic_initialization(self):
        """Test basic solver initialization with default parameters."""
        network = GridNetwork(width=3, height=3)
        network.create_network()

        problem = NetworkMFGProblem(
            network_geometry=network,
            T=1.0,
            Nt=10,
        )

        solver = FPNetworkSolver(problem)

        assert solver.fp_method_name == "NetworkFP_explicit"
        assert solver.scheme == "explicit"
        assert solver.diffusion_coefficient == 0.1
        assert solver.cfl_factor == 0.5
        assert solver.max_iterations == 1000
        assert solver.tolerance == 1e-6
        assert solver.enforce_mass_conservation is True

    def test_explicit_scheme_initialization(self):
        """Test initialization with explicit scheme."""
        network = GridNetwork(width=3, height=3)
        network.create_network()

        problem = NetworkMFGProblem(
            network_geometry=network,
            T=1.0,
            Nt=10,
        )

        solver = FPNetworkSolver(problem, scheme="explicit")

        assert solver.scheme == "explicit"
        assert solver.fp_method_name == "NetworkFP_explicit"

    def test_implicit_scheme_initialization(self):
        """Test initialization with implicit scheme."""
        network = GridNetwork(width=3, height=3)
        network.create_network()

        problem = NetworkMFGProblem(
            network_geometry=network,
            T=1.0,
            Nt=10,
        )

        solver = FPNetworkSolver(problem, scheme="implicit")

        assert solver.scheme == "implicit"
        assert solver.fp_method_name == "NetworkFP_implicit"

    def test_upwind_scheme_initialization(self):
        """Test initialization with upwind scheme."""
        network = GridNetwork(width=3, height=3)
        network.create_network()

        problem = NetworkMFGProblem(
            network_geometry=network,
            T=1.0,
            Nt=10,
        )

        solver = FPNetworkSolver(problem, scheme="upwind")

        assert solver.scheme == "upwind"
        assert solver.fp_method_name == "NetworkFP_upwind"

    def test_custom_diffusion_coefficient(self):
        """Test initialization with custom diffusion coefficient."""
        network = GridNetwork(width=3, height=3)
        network.create_network()

        problem = NetworkMFGProblem(
            network_geometry=network,
            T=1.0,
            Nt=10,
        )

        solver = FPNetworkSolver(problem, diffusion_coefficient=0.2)

        assert solver.diffusion_coefficient == 0.2

    def test_custom_cfl_factor(self):
        """Test initialization with custom CFL factor."""
        network = GridNetwork(width=3, height=3)
        network.create_network()

        problem = NetworkMFGProblem(
            network_geometry=network,
            T=1.0,
            Nt=10,
        )

        solver = FPNetworkSolver(problem, cfl_factor=0.3)

        assert solver.cfl_factor == 0.3

    def test_dead_iteration_parameters_fail_loud(self):
        """Issue #1426 (S0-25): max_iterations / tolerance are dead — the implicit step is a direct
        sparse solve (spsolve), not iterative — so a non-default value fails loud; defaults are a
        no-op and construct fine."""
        network = GridNetwork(width=3, height=3)
        network.create_network()

        problem = NetworkMFGProblem(
            network_geometry=network,
            T=1.0,
            Nt=10,
        )

        with pytest.raises(NotImplementedError, match="max_iterations"):
            FPNetworkSolver(problem, max_iterations=500)
        with pytest.raises(NotImplementedError, match="tolerance"):
            FPNetworkSolver(problem, tolerance=1e-8)

        # Defaults construct fine (the no-op knobs are unchanged).
        solver = FPNetworkSolver(problem)
        assert solver.max_iterations == 1000
        assert solver.tolerance == 1e-6

    def test_mass_conservation_flag(self):
        """Test mass conservation enforcement flag."""
        network = GridNetwork(width=3, height=3)
        network.create_network()

        problem = NetworkMFGProblem(
            network_geometry=network,
            T=1.0,
            Nt=10,
        )

        solver_with = FPNetworkSolver(problem, enforce_mass_conservation=True)
        solver_without = FPNetworkSolver(problem, enforce_mass_conservation=False)

        assert solver_with.enforce_mass_conservation is True
        assert solver_without.enforce_mass_conservation is False

    def test_network_properties_extracted(self):
        """Test that network properties are properly extracted."""
        network = GridNetwork(width=4, height=4)
        network.create_network()

        problem = NetworkMFGProblem(
            network_geometry=network,
            T=1.0,
            Nt=10,
        )

        solver = FPNetworkSolver(problem)

        assert solver.num_nodes == 16
        assert solver.adjacency_matrix is not None
        assert solver.laplacian_matrix is not None

    def test_time_discretization(self):
        """Test that time discretization is properly computed."""
        network = GridNetwork(width=3, height=3)
        network.create_network()

        problem = NetworkMFGProblem(
            network_geometry=network,
            T=2.0,
            Nt=20,
        )

        solver = FPNetworkSolver(problem)

        assert np.isclose(solver.dt, 0.1)
        assert len(solver.times) == 21
        assert np.isclose(solver.times[0], 0.0)
        assert np.isclose(solver.times[-1], 2.0)

    def test_divergence_operators_initialized(self):
        """Test that divergence operators are initialized."""
        network = GridNetwork(width=3, height=3)
        network.create_network()

        problem = NetworkMFGProblem(
            network_geometry=network,
            T=1.0,
            Nt=10,
        )

        solver = FPNetworkSolver(problem)

        assert hasattr(solver, "divergence_ops")
        assert len(solver.divergence_ops) == solver.num_nodes


class TestFPNetworkSolverSolveFPSystem:
    """Test the main solve_fp_system method."""

    def test_solve_fp_system_shape(self):
        """Test that solve_fp_system returns correct shape."""
        network = GridNetwork(width=3, height=3)
        network.create_network()

        problem = NetworkMFGProblem(
            network_geometry=network,
            T=0.5,
            Nt=10,
        )

        solver = FPNetworkSolver(problem, scheme="implicit")

        Nt = problem.Nt + 1
        num_nodes = problem.num_nodes

        # Create inputs
        m_initial = np.ones(num_nodes) / num_nodes
        U_solution = np.zeros((Nt, num_nodes))

        # Solve
        M_solution = solver.solve_fp_system(m_initial, U_solution)

        assert M_solution.shape == (Nt, num_nodes)
        assert np.all(np.isfinite(M_solution))

    def test_solve_fp_system_initial_condition(self):
        """Test that initial condition is preserved."""
        network = GridNetwork(width=3, height=3)
        network.create_network()

        problem = NetworkMFGProblem(
            network_geometry=network,
            T=0.5,
            Nt=10,
        )

        solver = FPNetworkSolver(problem, scheme="implicit")

        Nt = problem.Nt + 1
        num_nodes = problem.num_nodes

        # Create inputs with specific initial condition
        m_initial = np.random.rand(num_nodes)
        m_initial = m_initial / np.sum(m_initial)
        U_solution = np.zeros((Nt, num_nodes))

        # Solve
        M_solution = solver.solve_fp_system(m_initial, U_solution)

        # Initial time step should match initial condition (with normalization)
        assert np.allclose(M_solution[0, :], m_initial, rtol=0.1)

    def test_solve_with_explicit_scheme(self):
        """Test solving with explicit scheme."""
        network = GridNetwork(width=3, height=3)
        network.create_network()

        problem = NetworkMFGProblem(
            network_geometry=network,
            T=0.1,  # Short time for stability
            Nt=20,
        )

        solver = FPNetworkSolver(problem, scheme="explicit", cfl_factor=0.3)

        Nt = problem.Nt + 1
        num_nodes = problem.num_nodes

        m_initial = np.ones(num_nodes) / num_nodes
        U_solution = np.zeros((Nt, num_nodes))

        M_solution = solver.solve_fp_system(m_initial, U_solution)

        assert np.all(np.isfinite(M_solution))

    def test_solve_with_implicit_scheme(self):
        """Test solving with implicit scheme."""
        network = GridNetwork(width=3, height=3)
        network.create_network()

        problem = NetworkMFGProblem(
            network_geometry=network,
            T=0.5,
            Nt=10,
        )

        solver = FPNetworkSolver(problem, scheme="implicit")

        Nt = problem.Nt + 1
        num_nodes = problem.num_nodes

        m_initial = np.ones(num_nodes) / num_nodes
        U_solution = np.zeros((Nt, num_nodes))

        M_solution = solver.solve_fp_system(m_initial, U_solution)

        assert np.all(np.isfinite(M_solution))

    def test_solve_with_upwind_scheme(self):
        """Test solving with upwind scheme."""
        network = GridNetwork(width=3, height=3)
        network.create_network()

        problem = NetworkMFGProblem(
            network_geometry=network,
            T=0.5,
            Nt=10,
        )

        solver = FPNetworkSolver(problem, scheme="upwind")

        Nt = problem.Nt + 1
        num_nodes = problem.num_nodes

        m_initial = np.ones(num_nodes) / num_nodes
        U_solution = np.zeros((Nt, num_nodes))

        M_solution = solver.solve_fp_system(m_initial, U_solution)

        assert np.all(np.isfinite(M_solution))

    def test_solve_with_non_zero_drift(self):
        """Test solving with non-zero drift field."""
        network = GridNetwork(width=3, height=3)
        network.create_network()

        problem = NetworkMFGProblem(
            network_geometry=network,
            T=0.5,
            Nt=10,
        )

        solver = FPNetworkSolver(problem, scheme="implicit")

        Nt = problem.Nt + 1
        num_nodes = problem.num_nodes

        # Create non-zero drift
        m_initial = np.ones(num_nodes) / num_nodes
        U_solution = np.random.rand(Nt, num_nodes)

        M_solution = solver.solve_fp_system(m_initial, U_solution)

        assert np.all(np.isfinite(M_solution))

    def test_invalid_scheme_raises_error(self):
        """Test that invalid scheme raises error."""
        network = GridNetwork(width=3, height=3)
        network.create_network()

        problem = NetworkMFGProblem(
            network_geometry=network,
            T=0.5,
            Nt=10,
        )

        solver = FPNetworkSolver(problem, scheme="explicit")
        solver.scheme = "invalid_scheme"  # Manually set invalid scheme

        Nt = problem.Nt + 1
        num_nodes = problem.num_nodes

        m_initial = np.ones(num_nodes) / num_nodes
        U_solution = np.zeros((Nt, num_nodes))

        with pytest.raises(ValueError, match="Unknown scheme"):
            solver.solve_fp_system(m_initial, U_solution)


class TestFPNetworkSolverNumericalProperties:
    """Test numerical properties of network FP solutions."""

    def test_solution_finiteness(self):
        """Test that solution remains finite throughout."""
        network = GridNetwork(width=4, height=4)
        network.create_network()

        problem = NetworkMFGProblem(
            network_geometry=network,
            T=0.5,
            Nt=15,
        )

        solver = FPNetworkSolver(problem, scheme="implicit")

        Nt = problem.Nt + 1
        num_nodes = problem.num_nodes

        m_initial = np.random.rand(num_nodes)
        m_initial = m_initial / np.sum(m_initial)
        U_solution = np.zeros((Nt, num_nodes))

        M_solution = solver.solve_fp_system(m_initial, U_solution)

        # All values should be finite
        assert np.all(np.isfinite(M_solution))

    def test_forward_time_propagation(self):
        """Test that solution propagates forward in time."""
        network = GridNetwork(width=3, height=3)
        network.create_network()

        problem = NetworkMFGProblem(
            network_geometry=network,
            T=0.5,
            Nt=10,
        )

        solver = FPNetworkSolver(problem, scheme="implicit")

        Nt = problem.Nt + 1
        num_nodes = problem.num_nodes

        # Concentrated initial condition
        m_initial = np.zeros(num_nodes)
        m_initial[num_nodes // 2] = 1.0

        U_solution = np.zeros((Nt, num_nodes))

        M_solution = solver.solve_fp_system(m_initial, U_solution)

        # Solution should spread from initial concentration
        # Final distribution should be more diffuse
        initial_concentration = np.max(M_solution[0, :])
        final_concentration = np.max(M_solution[-1, :])
        assert final_concentration < initial_concentration

    def test_mass_conservation(self):
        """Test that total mass is conserved."""
        network = GridNetwork(width=4, height=4)
        network.create_network()

        problem = NetworkMFGProblem(
            network_geometry=network,
            T=0.5,
            Nt=15,
        )

        solver = FPNetworkSolver(problem, scheme="implicit", enforce_mass_conservation=True)

        Nt = problem.Nt + 1
        num_nodes = problem.num_nodes

        m_initial = np.random.rand(num_nodes)
        m_initial = m_initial / np.sum(m_initial)
        U_solution = np.zeros((Nt, num_nodes))

        M_solution = solver.solve_fp_system(m_initial, U_solution)

        # Check mass conservation across time
        initial_mass = np.sum(M_solution[0, :])
        for t in range(Nt):
            current_mass = np.sum(M_solution[t, :])
            # Allow some numerical error
            assert np.isclose(current_mass, initial_mass, rtol=0.1)

    def test_non_negativity(self):
        """Test that density remains non-negative."""
        network = GridNetwork(width=3, height=3)
        network.create_network()

        problem = NetworkMFGProblem(
            network_geometry=network,
            T=0.5,
            Nt=10,
        )

        solver = FPNetworkSolver(problem, scheme="implicit")

        Nt = problem.Nt + 1
        num_nodes = problem.num_nodes

        m_initial = np.ones(num_nodes) / num_nodes
        U_solution = np.zeros((Nt, num_nodes))

        M_solution = solver.solve_fp_system(m_initial, U_solution)

        # Density should be non-negative (with small tolerance for numerical errors)
        assert np.all(M_solution >= -1e-10)


class TestFPNetworkSolverDifferentNetworks:
    """Test solver with different network geometries."""

    def test_small_grid_network(self):
        """Test solver on small grid network."""
        network = GridNetwork(width=3, height=3)
        network.create_network()

        problem = NetworkMFGProblem(
            network_geometry=network,
            T=0.5,
            Nt=10,
        )

        solver = FPNetworkSolver(problem, scheme="implicit")

        Nt = problem.Nt + 1
        num_nodes = problem.num_nodes

        m_initial = np.ones(num_nodes) / num_nodes
        U_solution = np.zeros((Nt, num_nodes))

        M_solution = solver.solve_fp_system(m_initial, U_solution)

        assert M_solution.shape == (Nt, num_nodes)
        assert np.all(np.isfinite(M_solution))

    def test_rectangular_grid_network(self):
        """Test solver on non-square grid."""
        network = GridNetwork(width=4, height=3)
        network.create_network()

        problem = NetworkMFGProblem(
            network_geometry=network,
            T=0.5,
            Nt=10,
        )

        solver = FPNetworkSolver(problem, scheme="implicit")

        Nt = problem.Nt + 1
        num_nodes = problem.num_nodes

        m_initial = np.ones(num_nodes) / num_nodes
        U_solution = np.zeros((Nt, num_nodes))

        M_solution = solver.solve_fp_system(m_initial, U_solution)

        assert num_nodes == 12
        assert M_solution.shape == (Nt, 12)

    def test_periodic_grid_network(self):
        """Test solver on periodic grid."""
        network = GridNetwork(width=4, height=4, periodic=True)
        network.create_network()

        problem = NetworkMFGProblem(
            network_geometry=network,
            T=0.5,
            Nt=10,
        )

        solver = FPNetworkSolver(problem, scheme="implicit")

        Nt = problem.Nt + 1
        num_nodes = problem.num_nodes

        m_initial = np.ones(num_nodes) / num_nodes
        U_solution = np.zeros((Nt, num_nodes))

        M_solution = solver.solve_fp_system(m_initial, U_solution)

        assert np.all(np.isfinite(M_solution))


class TestFPNetworkSolverIntegration:
    """Integration tests with actual FP problems."""

    def test_solver_not_abstract(self):
        """Test that FPNetworkSolver can be instantiated."""
        import inspect

        network = GridNetwork(width=3, height=3)
        network.create_network()

        problem = NetworkMFGProblem(
            network_geometry=network,
            T=0.5,
            Nt=10,
        )

        # Should not raise TypeError about abstract methods
        solver = FPNetworkSolver(problem)
        assert isinstance(solver, FPNetworkSolver)

        # Should not have abstract methods
        assert not inspect.isabstract(FPNetworkSolver)

    def test_solver_with_different_parameters(self):
        """Test solver with various parameter configurations."""
        network = GridNetwork(width=3, height=3)
        network.create_network()

        configs = [
            {"scheme": "explicit", "cfl_factor": 0.4},
            {"scheme": "implicit"},
            {"scheme": "upwind", "diffusion_coefficient": 0.15},
        ]

        for config in configs:
            problem = NetworkMFGProblem(
                network_geometry=network,
                T=0.2,
                Nt=10,
            )

            solver = FPNetworkSolver(problem, **config)

            Nt = problem.Nt + 1
            num_nodes = problem.num_nodes

            m_initial = np.ones(num_nodes) / num_nodes
            U_solution = np.zeros((Nt, num_nodes))

            M_solution = solver.solve_fp_system(m_initial, U_solution)

            assert np.all(np.isfinite(M_solution))


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
