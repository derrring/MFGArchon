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

        # The density coupling must actually reach U.  With the uniform M every other test in
        # this file feeds, the per-time spatial spread is exactly 0.0; a non-uniform M is what
        # makes the coupling observable, and this is the only test that supplies one.  Measured
        # spread over 30 unseeded draws: min 2.4e-3, median 4.0e-3 -- so 1e-4 sits 24x below the
        # minimum observed, while a solver ignoring M_density would give exactly 0.0.
        spatial_spread = np.max(U_solution.max(axis=1) - U_solution.min(axis=1))
        assert spatial_spread > 1e-4, "non-uniform density did not reach U; the coupling is not wired"

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

        # Cross-integrator agreement is a genuine external oracle here: RK45, BDF and Radau are
        # independent scipy integrators of the same backward ODE, so agreement is evidence about
        # the ODE right-hand side rather than about any one integrator.  U_final is unseeded, so
        # the threshold is set from the tail: over 150 random draws the worst RK45-BDF gap was
        # 9.8e-6 and the worst RK45-Radau gap 6.5e-6, against a solution spread of ~0.9.  1e-3
        # is 100x above the observed worst case and ~1000x below a real disagreement.
        U_rk = NetworkHJBSolver(problem, scheme="RK45").solve_hjb_system(M_density, U_final)
        U_radau = NetworkHJBSolver(problem, scheme="Radau").solve_hjb_system(M_density, U_final)
        assert np.max(np.abs(U_rk - U_solution)) < 1e-3
        assert np.max(np.abs(U_rk - U_radau)) < 1e-3

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
        """Test solver on periodic grid: the wrap edges must change the answer.

        A uniform terminal condition makes this test vacuous -- with U_final = 0 and M = 1 the
        value function is spatially constant on both the torus (all degrees 4) and the open grid
        (degrees 2, 3, 4), spread exactly 0.0 in both, so the wrap edges never enter the answer.
        A spike terminal makes the graph topology observable.
        """
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
        U_final[0] = 1.0

        U_solution = solver.solve_hjb_system(M_density, U_final)

        assert np.all(np.isfinite(U_solution))

        # The same terminal spike on the open 4x4 grid must give a different value function,
        # since node 0 is a degree-2 corner there and a degree-4 interior node on the torus.
        # Measured separation 0.167 (and exactly 0.0 under the uniform terminal above).
        open_network = GridNetwork(width=4, height=4, periodic=False)
        open_network.create_network()
        open_problem = NetworkMFGProblem(geometry=open_network, T=0.5, Nt=10)
        U_open = NetworkHJBSolver(open_problem, scheme="BDF").solve_hjb_system(M_density, U_final)

        assert np.max(np.abs(U_solution - U_open)) > 0.1, "periodic wrap edges did not affect the solution"


class TestNetworkHJBSolverIntegration:
    """Integration tests with actual MFG problems."""

    def test_solver_with_different_parameters(self):
        """The three integrators must agree on the same backward ODE.

        Under the zero terminal condition this file usually feeds, U is spatially constant
        (spread exactly 0.0) and the graph is irrelevant, so the three schemes agree trivially.
        A spike terminal makes the answer depend on the network.
        """
        network = GridNetwork(width=3, height=3)
        network.create_network()

        configs = [
            {"scheme": "RK45"},
            {"scheme": "BDF", "tolerance": 1e-7},
            {"scheme": "Radau"},
        ]

        solutions = {}
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
            U_final[0] = 1.0

            U_solution = solver.solve_hjb_system(M_density, U_final)

            assert np.all(np.isfinite(U_solution))
            solutions[config["scheme"]] = U_solution

        # The spike is what makes this non-trivial: measured spatial spread 1.0, versus exactly
        # 0.0 under the previous zero terminal.  Measured integrator gaps 2.0e-6 for both pairs,
        # so 1e-4 is a 50x margin and still ~1e4 below a genuine dispatch error.
        assert np.max(solutions["RK45"].max(axis=1) - solutions["RK45"].min(axis=1)) > 0.1
        np.testing.assert_allclose(solutions["RK45"], solutions["BDF"], atol=1e-4)
        np.testing.assert_allclose(solutions["RK45"], solutions["Radau"], atol=1e-4)


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
