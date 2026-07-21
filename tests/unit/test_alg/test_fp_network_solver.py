#!/usr/bin/env python3
"""
Unit tests for FPNetworkSolver.

Tests the Fokker-Planck solver for Mean Field Games on network/graph structures,
including density evolution, mass conservation, and various time discretization schemes.
"""

import warnings

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
            geometry=network,
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
            geometry=network,
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
            geometry=network,
            T=1.0,
            Nt=10,
        )

        solver = FPNetworkSolver(problem, scheme="implicit")

        assert solver.scheme == "implicit"
        assert solver.fp_method_name == "NetworkFP_implicit"

    def test_upwind_scheme_fails_loud(self):
        """Issue #1541: scheme='upwind' was a physics-free identity map; it now fails loud at
        construction rather than silently returning an unevolved density."""
        network = GridNetwork(width=3, height=3)
        network.create_network()

        problem = NetworkMFGProblem(
            geometry=network,
            T=1.0,
            Nt=10,
        )

        with pytest.raises(NotImplementedError, match="1541"):
            FPNetworkSolver(problem, scheme="upwind")

    def test_custom_diffusion_coefficient(self):
        """Test initialization with custom diffusion coefficient."""
        network = GridNetwork(width=3, height=3)
        network.create_network()

        problem = NetworkMFGProblem(
            geometry=network,
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
            geometry=network,
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
            geometry=network,
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
            geometry=network,
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
            geometry=network,
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
            geometry=network,
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
            geometry=network,
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
            geometry=network,
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

    def test_volatility_field_is_sigma_not_diffusion(self):
        """Issue #1429 (S0-15): volatility_field is the SDE volatility sigma (D = sigma^2/2), the
        base_fp contract shared with FDM/FVM/GFDM — not the diffusion D directly. So a solve with
        volatility_field=sigma equals a solve with diffusion_coefficient=0.5*sigma**2 (which the
        pre-fix `D = volatility_field` fork would have failed)."""
        network = GridNetwork(width=3, height=3)
        network.create_network()
        problem = NetworkMFGProblem(geometry=network, T=0.5, Nt=10)
        num_nodes = problem.num_nodes
        # NON-uniform initial density: a uniform m is a diffusion fixed point (Laplacian of a
        # constant is 0), so D would be unobservable. A ramp makes the diffusion (hence D) matter.
        m0 = np.arange(1, num_nodes + 1, dtype=float)
        m0 /= m0.sum()
        U = np.zeros((problem.Nt + 1, num_nodes))
        sigma = 0.4

        m_via_sigma = FPNetworkSolver(problem, scheme="explicit").solve_fp_system(m0, U, volatility_field=sigma)
        m_via_diffusion = FPNetworkSolver(
            problem, scheme="explicit", diffusion_coefficient=0.5 * sigma**2
        ).solve_fp_system(m0, U)

        np.testing.assert_allclose(
            m_via_sigma,
            m_via_diffusion,
            rtol=1e-12,
            atol=1e-12,
            err_msg="volatility_field=sigma must yield D=sigma^2/2 (== diffusion_coefficient=0.5*sigma^2)",
        )

    def test_solve_fp_system_initial_condition(self):
        """Test that initial condition is preserved."""
        network = GridNetwork(width=3, height=3)
        network.create_network()

        problem = NetworkMFGProblem(
            geometry=network,
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
            geometry=network,
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
            geometry=network,
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

    def test_solve_with_non_zero_drift(self):
        """Test solving with non-zero drift field."""
        network = GridNetwork(width=3, height=3)
        network.create_network()

        problem = NetworkMFGProblem(
            geometry=network,
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
        """Issue #1541: an unsupported scheme is rejected at construction (fail loud), not silently
        run or discovered at solve time."""
        network = GridNetwork(width=3, height=3)
        network.create_network()

        problem = NetworkMFGProblem(
            geometry=network,
            T=0.5,
            Nt=10,
        )

        with pytest.raises(NotImplementedError, match="1541"):
            FPNetworkSolver(problem, scheme="invalid_scheme")


class TestFPNetworkSolverNumericalProperties:
    """Test numerical properties of network FP solutions."""

    def test_solution_finiteness(self):
        """Test that solution remains finite throughout."""
        network = GridNetwork(width=4, height=4)
        network.create_network()

        problem = NetworkMFGProblem(
            geometry=network,
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
            geometry=network,
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
            geometry=network,
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
            geometry=network,
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
            geometry=network,
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
            geometry=network,
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
            geometry=network,
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
            geometry=network,
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
            {"scheme": "explicit", "diffusion_coefficient": 0.15},  # was "upwind" (removed, Issue #1541)
        ]

        for config in configs:
            problem = NetworkMFGProblem(
                geometry=network,
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


class TestFPNetworkSolverAbsorbingNodeBC:
    """Issue #1478 (Stage 2b): FPNetworkSolver honors an ABSORBING (exit) node — its mass leaves
    (``m -> 0``) and the mass renorm is gated off so the absorption is not hidden.
    """

    def _network(self, absorbing_node=None):
        from mfgarchon.geometry.boundary.applicator_graph import GraphBCConfig, GraphBCType, NodeBC

        bc = None
        if absorbing_node is not None:
            bc = GraphBCConfig(node_bcs=[NodeBC(nodes=[absorbing_node], bc_type=GraphBCType.ABSORBING, value=0.0)])
        network = GridNetwork(width=3, height=3, boundary_conditions=bc)
        network.create_network()
        return network

    def test_absorbing_fp_mass_decreases(self):
        problem = NetworkMFGProblem(geometry=self._network(absorbing_node=0), T=0.5, Nt=20)
        solver = FPNetworkSolver(problem, scheme="explicit")  # honors it — no raise
        n = problem.num_nodes
        m0 = np.ones(n) / n
        u = np.zeros((21, n))  # zero value -> pure diffusion toward the absorbing node
        m = solver.solve_fp_system(M_initial=m0, potential_field=u)
        assert m[-1].sum() < m[0].sum() - 1e-3, "absorption: total mass must strictly decrease"
        assert np.allclose(m[1:, 0], 0.0), "the absorbing node's density is zeroed each step"

    def test_no_node_bc_constructs_and_conserves(self):
        """No node BC -> construction unchanged and mass is conserved (renorm active)."""
        problem = NetworkMFGProblem(geometry=self._network(), T=0.5, Nt=10)
        solver = FPNetworkSolver(problem)  # no raise
        assert solver.num_nodes == 9
        assert solver._mass_changing_bc is False


class TestFPNetworkDiffusionDefaultWarning1532:
    """Issue #1532: relying on the D=0.1 diffusion fallback (no explicit ``diffusion_coefficient`` and
    no ``volatility_field``) must WARN — otherwise the physical diffusion silently decouples from the
    problem (NetworkMFGProblem carries no sigma to source from). Byte-value 0.1 is retained for
    backward compatibility; only the silence is fixed."""

    @staticmethod
    def _setup():
        net = GridNetwork(width=3, height=3)
        net.create_network()
        problem = NetworkMFGProblem(geometry=net, T=0.5, Nt=10)
        n = problem.num_nodes
        return problem, np.ones(n) / n, np.zeros((problem.Nt, n))

    def test_defaulted_diffusion_warns_on_solve(self):
        problem, m0, U = self._setup()
        solver = FPNetworkSolver(problem)  # diffusion defaulted -> fallback 0.1
        assert solver.diffusion_coefficient == 0.1  # value unchanged (backward compatible)
        with pytest.warns(UserWarning, match="1532"):
            solver.solve_fp_system(m0, U)

    def test_explicit_diffusion_does_not_warn(self):
        problem, m0, U = self._setup()
        solver = FPNetworkSolver(problem, diffusion_coefficient=0.05)
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            solver.solve_fp_system(m0, U)
        assert not any("1532" in str(w.message) for w in rec)

    def test_volatility_field_suppresses_warning(self):
        problem, m0, U = self._setup()
        solver = FPNetworkSolver(problem)  # defaulted, but sigma given at solve time
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            solver.solve_fp_system(m0, U, volatility_field=0.3)
        assert not any("1532" in str(w.message) for w in rec)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
