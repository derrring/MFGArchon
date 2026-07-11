#!/usr/bin/env python3
"""
Unit tests for NetworkHJBSolver.

Tests the HJB solver for Mean Field Games on network/graph structures,
including various time discretization schemes and network operators.
"""

import pytest

import numpy as np

from mfgarchon.alg.numerical.network_solvers.hjb_network import (
    NetworkHJBSolver,
    NetworkPolicyIterationHJBSolver,
)
from mfgarchon.extensions.topology import NetworkMFGProblem
from mfgarchon.geometry.graph.network_geometry import GridNetwork

# Skip all tests if igraph is not available (network backend dependency)
igraph = pytest.importorskip("igraph")


class TestNetworkHJBSolverInitialization:
    """Test NetworkHJBSolver initialization and configuration."""

    def test_basic_initialization(self):
        """Test basic solver initialization with default parameters."""
        network = GridNetwork(width=3, height=3)
        network.create_network()

        problem = NetworkMFGProblem(
            geometry=network,
            T=1.0,
            Nt=10,
        )

        solver = NetworkHJBSolver(problem)

        assert solver.hjb_method_name == "NetworkHJB_RK45"
        assert solver.scheme == "RK45"
        assert solver.tolerance == 1e-6

    def test_bdf_scheme_initialization(self):
        """Test initialization with BDF (stiff) scheme."""
        network = GridNetwork(width=3, height=3)
        network.create_network()

        problem = NetworkMFGProblem(
            geometry=network,
            T=1.0,
            Nt=10,
        )

        solver = NetworkHJBSolver(problem, scheme="BDF")

        assert solver.scheme == "BDF"
        assert solver.hjb_method_name == "NetworkHJB_BDF"

    def test_custom_tolerance(self):
        """Test initialization with custom tolerance."""
        network = GridNetwork(width=3, height=3)
        network.create_network()

        problem = NetworkMFGProblem(
            geometry=network,
            T=1.0,
            Nt=10,
        )

        solver = NetworkHJBSolver(problem, tolerance=1e-8)
        assert solver.tolerance == 1e-8

    def test_policy_iteration_policy_tolerance_fail_loud(self):
        """Issue #1426 (S0-25): NetworkPolicyIterationHJBSolver.policy_tolerance is dead — policy
        iteration converges on policy stability (`_policies_equal`), not a value tolerance — so a
        non-default value fails loud; the default constructs fine and max_policy_iterations (live)
        is freely settable."""
        network = GridNetwork(width=3, height=3)
        network.create_network()
        problem = NetworkMFGProblem(geometry=network, T=1.0, Nt=10)

        with pytest.raises(NotImplementedError, match="policy_tolerance"):
            NetworkPolicyIterationHJBSolver(problem, policy_tolerance=1e-8)

        solver = NetworkPolicyIterationHJBSolver(problem, max_policy_iterations=20)
        assert solver.policy_tolerance == 1e-6
        assert solver.max_policy_iterations == 20

    def test_network_properties_extracted(self):
        """Test that network properties are properly extracted."""
        network = GridNetwork(width=4, height=4)
        network.create_network()

        problem = NetworkMFGProblem(
            geometry=network,
            T=1.0,
            Nt=10,
        )

        solver = NetworkHJBSolver(problem)

        assert solver.num_nodes == 16
        assert solver.adjacency_matrix is not None
        assert solver.laplacian_matrix is not None

    def test_time_discretization(self):
        """Test that time discretization is properly computed."""
        network = GridNetwork(width=3, height=3)
        network.create_network()

        problem = NetworkMFGProblem(
            geometry=network,
            T=2.0,
            Nt=20,
        )

        solver = NetworkHJBSolver(problem)

        assert np.isclose(solver.dt, 0.1)
        assert len(solver.times) == 21
        assert np.isclose(solver.times[0], 0.0)
        assert np.isclose(solver.times[-1], 2.0)

    def test_gradient_operators_initialized(self):
        """Test that gradient operators are initialized."""
        network = GridNetwork(width=3, height=3)
        network.create_network()

        problem = NetworkMFGProblem(
            geometry=network,
            T=1.0,
            Nt=10,
        )

        solver = NetworkHJBSolver(problem)

        assert hasattr(solver, "gradient_ops")
        assert len(solver.gradient_ops) == solver.num_nodes


class TestNetworkHJBSolverSolveHJBSystem:
    """Test the main solve_hjb_system method."""

    def test_solve_hjb_system_shape(self):
        """Test that solve_hjb_system returns correct shape."""
        network = GridNetwork(width=3, height=3)
        network.create_network()

        problem = NetworkMFGProblem(
            geometry=network,
            T=0.5,
            Nt=10,
        )

        solver = NetworkHJBSolver(problem, scheme="BDF")

        Nt = problem.Nt + 1
        num_nodes = problem.num_nodes

        # Create inputs
        M_density = np.ones((Nt, num_nodes))
        U_final = np.zeros(num_nodes)

        # Solve
        U_solution = solver.solve_hjb_system(M_density, U_final)

        assert U_solution.shape == (Nt, num_nodes)
        assert np.all(np.isfinite(U_solution))

    def test_nonzero_volatility_rejected(self):
        """Issue #1544: the network HJB has no viscous term, so a nonzero volatility_field must be
        rejected (not silently ignored -- ignoring it solves a non-adjoint, self-consistent WRONG
        equilibrium against the diffusive network FP). volatility_field=None / 0 solves normally."""
        network = GridNetwork(width=3, height=3)
        network.create_network()
        problem = NetworkMFGProblem(geometry=network, T=0.5, Nt=10)
        Nt, num_nodes = problem.Nt + 1, problem.num_nodes
        M_density, U_final = np.ones((Nt, num_nodes)), np.zeros(num_nodes)

        for cls in (NetworkHJBSolver, NetworkPolicyIterationHJBSolver):
            solver = cls(problem, scheme="BDF") if cls is NetworkHJBSolver else cls(problem)
            with pytest.raises(NotImplementedError, match="1544"):
                solver.solve_hjb_system(M_density, U_final, volatility_field=0.3)
            # None and 0.0 are the deterministic-control (D=0) case -> no raise.
            assert solver.solve_hjb_system(M_density, U_final, volatility_field=None).shape == (Nt, num_nodes)
            assert solver.solve_hjb_system(M_density, U_final, volatility_field=0.0).shape == (Nt, num_nodes)

    def test_solve_hjb_system_final_condition(self):
        """Test that final condition is preserved."""
        network = GridNetwork(width=3, height=3)
        network.create_network()

        problem = NetworkMFGProblem(
            geometry=network,
            T=0.5,
            Nt=10,
        )

        solver = NetworkHJBSolver(problem, scheme="BDF")

        Nt = problem.Nt + 1
        num_nodes = problem.num_nodes

        # Create inputs with specific final condition
        M_density = np.ones((Nt, num_nodes))
        U_final = np.random.rand(num_nodes)

        # Solve
        U_solution = solver.solve_hjb_system(M_density, U_final)

        # Final time step should match final condition
        assert np.allclose(U_solution[-1, :], U_final)

    def test_solve_with_explicit_scheme(self):
        """Test solving with explicit scheme."""
        network = GridNetwork(width=3, height=3)
        network.create_network()

        problem = NetworkMFGProblem(
            geometry=network,
            T=0.1,  # Short time for stability
            Nt=20,
        )

        solver = NetworkHJBSolver(problem, scheme="RK45")

        Nt = problem.Nt + 1
        num_nodes = problem.num_nodes

        M_density = np.ones((Nt, num_nodes))
        U_final = np.zeros(num_nodes)

        U_solution = solver.solve_hjb_system(M_density, U_final)

        assert np.all(np.isfinite(U_solution))

    def test_solve_with_implicit_scheme(self):
        """Test solving with implicit scheme."""
        network = GridNetwork(width=3, height=3)
        network.create_network()

        problem = NetworkMFGProblem(
            geometry=network,
            T=0.5,
            Nt=10,
        )

        solver = NetworkHJBSolver(problem, scheme="BDF")

        Nt = problem.Nt + 1
        num_nodes = problem.num_nodes

        M_density = np.ones((Nt, num_nodes))
        U_final = np.zeros(num_nodes)

        U_solution = solver.solve_hjb_system(M_density, U_final)

        assert np.all(np.isfinite(U_solution))

    def test_solve_with_semi_implicit_scheme(self):
        """Test solving with semi-implicit scheme."""
        network = GridNetwork(width=3, height=3)
        network.create_network()

        problem = NetworkMFGProblem(
            geometry=network,
            T=0.5,
            Nt=10,
        )

        solver = NetworkHJBSolver(problem, scheme="Radau")

        Nt = problem.Nt + 1
        num_nodes = problem.num_nodes

        M_density = np.ones((Nt, num_nodes))
        U_final = np.zeros(num_nodes)

        U_solution = solver.solve_hjb_system(M_density, U_final)

        assert np.all(np.isfinite(U_solution))

    def test_solve_with_varying_density(self):
        """Test solving with non-uniform density."""
        network = GridNetwork(width=3, height=3)
        network.create_network()

        problem = NetworkMFGProblem(
            geometry=network,
            T=0.5,
            Nt=10,
        )

        solver = NetworkHJBSolver(problem, scheme="BDF")

        Nt = problem.Nt + 1
        num_nodes = problem.num_nodes

        # Create non-uniform density
        M_density = np.random.rand(Nt, num_nodes)
        M_density = M_density / np.sum(M_density, axis=1, keepdims=True)

        U_final = np.zeros(num_nodes)

        U_solution = solver.solve_hjb_system(M_density, U_final)

        assert np.all(np.isfinite(U_solution))

    def test_invalid_scheme_raises_error(self):
        """Test that invalid scheme raises error."""
        network = GridNetwork(width=3, height=3)
        network.create_network()

        problem = NetworkMFGProblem(
            geometry=network,
            T=0.5,
            Nt=10,
        )

        solver = NetworkHJBSolver(problem, scheme="RK45")
        solver.scheme = "invalid_scheme"  # Manually set invalid scheme

        Nt = problem.Nt + 1
        num_nodes = problem.num_nodes

        M_density = np.ones((Nt, num_nodes))
        U_final = np.zeros(num_nodes)

        with pytest.raises((ValueError, RuntimeError)):
            solver.solve_hjb_system(M_density, U_final)


class TestNetworkHJBSolverNumericalProperties:
    """Test numerical properties of network HJB solutions."""

    def test_solution_finiteness(self):
        """Test that solution remains finite throughout."""
        network = GridNetwork(width=4, height=4)
        network.create_network()

        problem = NetworkMFGProblem(
            geometry=network,
            T=0.5,
            Nt=15,
        )

        solver = NetworkHJBSolver(problem, scheme="BDF")

        Nt = problem.Nt + 1
        num_nodes = problem.num_nodes

        M_density = np.ones((Nt, num_nodes)) * 0.5
        U_final = np.random.rand(num_nodes)

        U_solution = solver.solve_hjb_system(M_density, U_final)

        # All values should be finite
        assert np.all(np.isfinite(U_solution))

    def test_backward_time_propagation(self):
        """Test that solution propagates backward in time."""
        network = GridNetwork(width=3, height=3)
        network.create_network()

        problem = NetworkMFGProblem(
            geometry=network,
            T=0.5,
            Nt=10,
        )

        solver = NetworkHJBSolver(problem, scheme="BDF")

        Nt = problem.Nt + 1
        num_nodes = problem.num_nodes

        M_density = np.ones((Nt, num_nodes))
        U_final = np.ones(num_nodes)  # Non-zero final condition

        U_solution = solver.solve_hjb_system(M_density, U_final)

        # Solution should propagate backward (not remain zero at t=0)
        assert not np.allclose(U_solution[0, :], 0.0)


class TestNetworkHJBSolverDifferentNetworks:
    """Test solver with different network geometries."""

    def test_small_grid_network(self):
        """Test solver on small grid network."""
        network = GridNetwork(width=3, height=3)
        network.create_network()

        problem = NetworkMFGProblem(
            geometry=network,
            T=0.5,
            Nt=10,
        )

        solver = NetworkHJBSolver(problem, scheme="BDF")

        Nt = problem.Nt + 1
        num_nodes = problem.num_nodes

        M_density = np.ones((Nt, num_nodes))
        U_final = np.zeros(num_nodes)

        U_solution = solver.solve_hjb_system(M_density, U_final)

        assert U_solution.shape == (Nt, num_nodes)
        assert np.all(np.isfinite(U_solution))

    def test_rectangular_grid_network(self):
        """Test solver on non-square grid."""
        network = GridNetwork(width=4, height=3)
        network.create_network()

        problem = NetworkMFGProblem(
            geometry=network,
            T=0.5,
            Nt=10,
        )

        solver = NetworkHJBSolver(problem, scheme="BDF")

        Nt = problem.Nt + 1
        num_nodes = problem.num_nodes

        M_density = np.ones((Nt, num_nodes))
        U_final = np.zeros(num_nodes)

        U_solution = solver.solve_hjb_system(M_density, U_final)

        assert num_nodes == 12
        assert U_solution.shape == (Nt, 12)

    def test_periodic_grid_network(self):
        """Test solver on periodic grid."""
        network = GridNetwork(width=4, height=4, periodic=True)
        network.create_network()

        problem = NetworkMFGProblem(
            geometry=network,
            T=0.5,
            Nt=10,
        )

        solver = NetworkHJBSolver(problem, scheme="BDF")

        Nt = problem.Nt + 1
        num_nodes = problem.num_nodes

        M_density = np.ones((Nt, num_nodes))
        U_final = np.zeros(num_nodes)

        U_solution = solver.solve_hjb_system(M_density, U_final)

        assert np.all(np.isfinite(U_solution))


class TestNetworkHJBSolverIntegration:
    """Integration tests with actual MFG problems."""

    def test_solver_not_abstract(self):
        """Test that NetworkHJBSolver can be instantiated."""
        import inspect

        network = GridNetwork(width=3, height=3)
        network.create_network()

        problem = NetworkMFGProblem(
            geometry=network,
            T=0.5,
            Nt=10,
        )

        # Should not raise TypeError about abstract methods
        solver = NetworkHJBSolver(problem)
        assert isinstance(solver, NetworkHJBSolver)

        # Should not have abstract methods
        assert not inspect.isabstract(NetworkHJBSolver)

    def test_solver_with_different_parameters(self):
        """Test solver with various parameter configurations."""
        network = GridNetwork(width=3, height=3)
        network.create_network()

        configs = [
            {"scheme": "RK45"},
            {"scheme": "BDF", "tolerance": 1e-7},
            {"scheme": "Radau"},
        ]

        for config in configs:
            problem = NetworkMFGProblem(
                geometry=network,
                T=0.2,
                Nt=10,
            )

            solver = NetworkHJBSolver(problem, **config)

            Nt = problem.Nt + 1
            num_nodes = problem.num_nodes

            M_density = np.ones((Nt, num_nodes))
            U_final = np.zeros(num_nodes)

            U_solution = solver.solve_hjb_system(M_density, U_final)

            assert np.all(np.isfinite(U_solution))


class TestNetworkHJBIssue1468NodeBCGate:
    """Issue #1468 / #1471 (BC-capability, #1456 network family).

    Node boundary conditions live on the graph geometry (``GraphGeometry``). The base
    ``NetworkHJBSolver`` integrates the backward HJB ODE (``solve_ivp``) with terminal data only and
    never applies node-BC, so it must fail loud. ``NetworkPolicyIterationHJBSolver`` applies
    ``apply_boundary_conditions`` (the geometry-owned ``GraphApplicator`` DIRICHLET pin) at every
    backward step, so it honors node-BC and is exempt from the gate.
    """

    def _network(self, value=None):
        from mfgarchon.geometry.boundary.applicator_graph import GraphBCConfig, GraphBCType, NodeBC

        bc = None
        if value is not None:
            bc = GraphBCConfig(node_bcs=[NodeBC(nodes=[0], bc_type=GraphBCType.DIRICHLET, value=value)])
        network = GridNetwork(width=3, height=3, boundary_conditions=bc)
        network.create_network()
        return network

    def test_base_solver_node_bc_fail_loud(self):
        problem = NetworkMFGProblem(geometry=self._network(value=0.0), T=0.5, Nt=10)
        with pytest.raises(NotImplementedError, match="node boundary conditions"):
            NetworkHJBSolver(problem)

    def test_base_solver_no_node_bc_constructs(self):
        problem = NetworkMFGProblem(geometry=self._network(), T=0.5, Nt=10)
        solver = NetworkHJBSolver(problem)  # no raise
        assert solver._honors_node_bc is False

    def test_policy_iteration_exempt_and_honors_node_bc(self):
        """Policy iteration constructs with a geometry node-BC (gate-exempt) and actually pins the
        node value at every backward step via the geometry-owned ``GraphApplicator``."""
        problem = NetworkMFGProblem(geometry=self._network(value=lambda node, t: 7.0), T=0.5, Nt=10)
        solver = NetworkPolicyIterationHJBSolver(problem)  # no raise
        assert solver._honors_node_bc is True

        n_time_points = problem.Nt + 1
        num_nodes = problem.num_nodes
        U = solver.solve_hjb_system(np.ones((n_time_points, num_nodes)), np.zeros(num_nodes))
        # apply_boundary_conditions pins node 0 = 7.0 at every backward step (indices 0..Nt-1);
        # the terminal step (index Nt) keeps U_terminal (= 0), which the BC pass does not touch.
        assert np.allclose(U[:-1, 0], 7.0), f"boundary node not pinned: U[:, 0] = {U[:, 0]}"
        assert U[-1, 0] == 0.0, "terminal condition must be preserved (BC not applied at t=T)"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
