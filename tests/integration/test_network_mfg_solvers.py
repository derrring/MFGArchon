#!/usr/bin/env python3
"""
Integration tests for network MFG solvers.

Tests complete MFG problem solving on network/graph structures,
including various network geometries, solver schemes, and coupling methods.
"""

import pytest

import numpy as np

from mfgarchon.alg.numerical.coupling.fixed_point_iterator import FixedPointIterator
from mfgarchon.alg.numerical.coupling.network_mfg_solver import (
    create_network_mfg_solver,
    create_simple_network_solver,
)
from mfgarchon.extensions.topology import NetworkMFGComponents, NetworkMFGProblem
from mfgarchon.geometry.graph.network_geometry import GridNetwork

# Skip all tests if igraph is not available (network backend dependency)
igraph = pytest.importorskip("igraph")


class TestNetworkMFGSolverCreation:
    """Test network MFG solver factory functions."""

    def test_create_network_solver_explicit(self):
        """Test network solver with explicit schemes."""
        network = GridNetwork(width=4, height=4)
        network.create_network()

        problem = NetworkMFGProblem(
            geometry=network,
            T=1.0,
            Nt=20,
        )

        solver = create_network_mfg_solver(
            problem,
            hjb_solver_type="explicit",
            fp_solver_type="explicit",
        )

        assert solver is not None

        # `is not None` makes this test byte-identical to its implicit sibling apart from the
        # string, i.e. the one parameter that separates them goes unobserved. The factory
        # forwards hjb_solver_type/fp_solver_type as scheme= to the two network solvers, so
        # that is where the dispatch becomes visible.
        assert isinstance(solver, FixedPointIterator)
        assert solver.hjb_solver.scheme == "explicit"
        assert solver.fp_solver.scheme == "explicit"

    def test_create_network_solver_implicit(self):
        """Test network solver with implicit schemes."""
        network = GridNetwork(width=4, height=4)
        network.create_network()

        problem = NetworkMFGProblem(
            geometry=network,
            T=1.0,
            Nt=20,
        )

        solver = create_network_mfg_solver(
            problem,
            hjb_solver_type="implicit",
            fp_solver_type="implicit",
        )

        assert solver is not None

        # The implicit half of the dispatch. Without these three lines this test and
        # test_create_network_solver_explicit differ only in a string neither of them reads,
        # which is what makes them look like duplicates when they are actually two branches.
        assert isinstance(solver, FixedPointIterator)
        assert solver.hjb_solver.scheme == "implicit"
        assert solver.fp_solver.scheme == "implicit"

    def test_create_solver_with_custom_damping(self):
        """Test solver creation with custom damping factor."""
        network = GridNetwork(width=3, height=3)
        network.create_network()

        problem = NetworkMFGProblem(
            geometry=network,
            T=1.0,
            Nt=10,
        )

        solver = create_network_mfg_solver(
            problem,
            damping_factor=0.7,
        )

        assert solver.relaxation == 0.7


class TestNetworkMFGProblemSetup:
    """Test network MFG problem configuration."""

    def test_grid_network_problem(self):
        """Test MFG problem on grid network."""
        network = GridNetwork(width=5, height=5)
        network.create_network()

        problem = NetworkMFGProblem(
            geometry=network,
            T=1.0,
            Nt=20,
        )

        assert problem.is_network_problem is True
        assert problem.num_nodes == 25
        assert problem.T == 1.0
        assert problem.Nt == 20

    def test_small_grid_network_problem(self):
        """Test MFG problem on small grid."""
        network = GridNetwork(width=3, height=3)
        network.create_network()

        problem = NetworkMFGProblem(
            geometry=network,
            T=0.5,
            Nt=10,
        )

        assert problem.num_nodes == 9
        assert problem.network_geometry is network

    def test_network_problem_with_components(self):
        """Test network problem with custom components (real, functional fields — the dead
        diffusion_coefficient/drift_coefficient knobs were removed in the #1470 Strand A purge)."""
        network = GridNetwork(width=4, height=4)
        network.create_network()

        components = NetworkMFGComponents(
            node_potential_func=lambda n, t: 0.2 * n,
            node_interaction_func=lambda n, m, t: 0.3 * m[n] ** 2,
        )

        problem = NetworkMFGProblem(
            geometry=network,
            T=1.0,
            Nt=10,
            components=components,
        )

        # The functional components are wired into the single-source Hamiltonian object and reachable
        # through the delegating problem methods (Issue #1470 Strand A).
        assert problem.components.node_potential_func is components.node_potential_func
        assert problem.node_potential(3, 0.0) == pytest.approx(0.6)
        m = np.ones(problem.num_nodes) / problem.num_nodes
        assert problem.density_coupling(2, m, 0.0) == pytest.approx(0.3 * m[2] ** 2)


class TestNetworkSolutionProperties:
    """Test mathematical properties of network MFG solutions."""

    @pytest.mark.skip(reason="Architecture gap: NetworkGraph incompatible with GFDM solver (requires CartesianGrid)")
    def test_mass_conservation(self):
        """Test that total mass is approximately conserved."""
        network = GridNetwork(width=4, height=4)
        network.create_network()

        problem = NetworkMFGProblem(
            geometry=network,
            T=0.5,
            Nt=15,
        )

        solver = create_simple_network_solver(problem, scheme="implicit")

        result = solver.solve(max_iterations=10, tolerance=1e-4)
        _U, M = result[:2]

        # Check mass conservation across time
        initial_mass = np.sum(M[0, :])
        for t in range(problem.Nt + 1):
            current_mass = np.sum(M[t, :])
            # Allow some numerical error
            assert np.isclose(current_mass, initial_mass, rtol=0.2)

    @pytest.mark.skip(reason="Architecture gap: NetworkGraph incompatible with GFDM solver (requires CartesianGrid)")
    def test_solution_evolution(self):
        """Test that solution evolves over time."""
        network = GridNetwork(width=4, height=4)
        network.create_network()

        problem = NetworkMFGProblem(
            geometry=network,
            T=1.0,
            Nt=20,
        )

        solver = create_simple_network_solver(problem, scheme="implicit")

        result = solver.solve(max_iterations=10, tolerance=1e-4)
        U, _M = result[:2]

        # Value function should evolve backward in time
        # (Note: density M may remain constant for symmetric problems with uniform initial conditions)
        assert not np.allclose(U[0, :], U[-1, :])


class TestNetworkGeometryVariations:
    """Test different network geometries."""

    @pytest.mark.skip(reason="Architecture gap: NetworkGraph incompatible with GFDM solver (requires CartesianGrid)")
    def test_rectangular_grid_network(self):
        """Test MFG on non-square grid."""
        network = GridNetwork(width=6, height=3)
        network.create_network()

        problem = NetworkMFGProblem(
            geometry=network,
            T=0.5,
            Nt=10,
        )

        solver = create_simple_network_solver(problem, scheme="implicit")

        result = solver.solve(max_iterations=8, tolerance=1e-4)

        assert result is not None
        assert problem.num_nodes == 18


class TestSolverRobustness:
    """Test solver robustness to various configurations."""

    @pytest.mark.skip(reason="Architecture gap: NetworkGraph incompatible with GFDM solver (requires CartesianGrid)")
    def test_different_network_sizes(self):
        """Test solver with different network sizes."""
        for size in [3, 4, 5]:
            network = GridNetwork(width=size, height=size)
            network.create_network()

            problem = NetworkMFGProblem(
                geometry=network,
                T=0.5,
                Nt=10,
            )

            solver = create_simple_network_solver(problem, scheme="implicit")

            result = solver.solve(max_iterations=8, tolerance=1e-4)

            assert result is not None
            assert problem.num_nodes == size * size


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
