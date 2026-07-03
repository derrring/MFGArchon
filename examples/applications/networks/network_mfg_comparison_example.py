#!/usr/bin/env python3
"""
Advanced Network MFG Comparison Example.

This example demonstrates advanced features of Network MFG including:
- Comparison of different network topologies (grid, random, scale-free)
- Different solver types (explicit, implicit, flow-based)
- Performance analysis and benchmarking
- Advanced visualization and flow analysis

Mathematical Setup:
- Multiple network types with same number of nodes
- Identical MFG problem formulation across networks
- Comparative analysis of solution properties
- Flow pattern analysis for different topologies
"""

import time
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from mfgarchon.alg.numerical.coupling.network_mfg_solver import create_network_mfg_solver

# MFGarchon imports
from mfgarchon.extensions.topology import NetworkMFGComponents, NetworkMFGProblem
from mfgarchon.geometry.graph.network_geometry import (
    GridNetwork,
    RandomNetwork,
    ScaleFreeNetwork,
    compute_network_statistics,
)

# Optional: use Plotly for interactive plots if available
try:
    import plotly.express as px  # noqa: F401
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


class NetworkMFGBenchmark:
    """
    Comprehensive benchmarking suite for Network MFG problems.

    Compares different network topologies and solver configurations
    on standardized MFG problems.
    """

    def __init__(self, num_nodes: int = 25, T: float = 1.0, Nt: int = 50):
        """
        Initialize benchmark suite.

        Args:
            num_nodes: Number of nodes in each network
            T: Terminal time
            Nt: Number of time steps
        """
        self.num_nodes = num_nodes
        self.T = T
        self.Nt = Nt

        # Benchmark results storage
        self.results = {}
        self.network_data = {}
        self.solver_performance = {}

    def create_benchmark_networks(self) -> dict[str, Any]:
        """
        Create different network topologies for comparison.

        Returns:
            Dictionary of network geometries
        """
        print("Creating benchmark networks...")

        networks = {}

        # 1. Grid network (5x5)
        grid_size = int(np.sqrt(self.num_nodes))
        networks["grid"] = GridNetwork(grid_size, grid_size, periodic=False)
        networks["grid"].create_network()

        # 2. Random network (Erdős–Rényi)
        networks["random"] = RandomNetwork(self.num_nodes, connection_prob=0.15)
        networks["random"].create_network(seed=42)

        # 3. Scale-free network (Barabási–Albert)
        networks["scale_free"] = ScaleFreeNetwork(self.num_nodes, num_edges_per_node=3)
        networks["scale_free"].create_network(seed=42)

        # Store network statistics
        for name, network in networks.items():
            stats = compute_network_statistics(network.network_data)
            self.network_data[name] = {"geometry": network, "statistics": stats}
            print(f"  {name.capitalize()}: {stats['num_edges']} edges, avg degree {stats['average_degree']:.2f}")

        print()
        return networks

    def create_standard_mfg_problem(self, network_geometry) -> NetworkMFGProblem:
        """
        Create standardized MFG problem for fair comparison.

        Args:
            network_geometry: Network geometry object

        Returns:
            Network MFG problem instance
        """

        # Define standardized problem components
        def terminal_value_func(node: int) -> float:
            """Target certain nodes with high rewards."""
            # Target nodes with high degree (hubs)
            degree = network_geometry.network_data.get_node_degree(node)
            return -2.0 * degree  # Negative for minimization

        def initial_density_func(node: int) -> float:
            """Start with uniform distribution."""
            return 1.0 / self.num_nodes

        def node_potential_func(node: int, t: float) -> float:
            """Small staying cost."""
            return 0.05

        def node_interaction_func(node: int, m: np.ndarray, t: float) -> float:
            """Quadratic congestion."""
            return 1.5 * m[node] ** 2

        # Create problem
        components = NetworkMFGComponents(
            terminal_node_value_func=terminal_value_func,
            initial_node_density_func=initial_density_func,
            node_potential_func=node_potential_func,
            node_interaction_func=node_interaction_func,
            diffusion_coefficient=0.1,
            drift_coefficient=1.0,
        )

        problem = NetworkMFGProblem(
            geometry=network_geometry,
            T=self.T,
            Nt=self.Nt,
            components=components,
            problem_name=f"Benchmark_{network_geometry.network_type.value}",
        )

        return problem

    def run_solver_comparison(self, problem: NetworkMFGProblem, network_name: str) -> dict[str, Any]:
        """
        Compare different solver types on the same problem.

        Args:
            problem: Network MFG problem
            network_name: Name of network topology

        Returns:
            Dictionary with solver comparison results
        """
        solver_configs = {
            "explicit": {
                "solver_type": "fixed_point",
                "hjb_solver_type": "explicit",
                "fp_solver_type": "explicit",
                "damping_factor": 0.5,
                "hjb_kwargs": {"cfl_factor": 0.3},
                "fp_kwargs": {"cfl_factor": 0.3},
            },
            "implicit": {
                "solver_type": "fixed_point",
                "hjb_solver_type": "implicit",
                "fp_solver_type": "implicit",
                "damping_factor": 0.7,
                "hjb_kwargs": {"max_iterations": 50},
                "fp_kwargs": {"max_iterations": 50},
            },
            "flow": {
                "solver_type": "flow",
                "hjb_solver_type": "explicit",
                "fp_solver_type": "flow",
                "damping_factor": 0.6,
                "hjb_kwargs": {"cfl_factor": 0.3},
                "fp_kwargs": {"enforce_mass_conservation": True},
            },
        }

        comparison_results = {}

        for solver_name, config in solver_configs.items():
            print(f"    Testing {solver_name} solver...")

            try:
                # Create solver
                solver = create_network_mfg_solver(problem, **config)

                # Solve with timing
                start_time = time.time()
                U, M, convergence_info = solver.solve(max_iterations=25, tolerance=1e-4, verbose=False)
                solve_time = time.time() - start_time

                # Analyze solution quality
                solution_quality = self._analyze_solution_quality(U, M, problem)

                # Store results
                comparison_results[solver_name] = {
                    "convergence_info": convergence_info,
                    "solve_time": solve_time,
                    "solution_quality": solution_quality,
                    "U": U,
                    "M": M,
                    "converged": convergence_info["converged"],
                    "final_error": convergence_info["final_error"],
                    "iterations": convergence_info["iterations"],
                }

                print(
                    f"      Converged: {convergence_info['converged']}, "
                    f"Time: {solve_time:.2f}s, "
                    f"Error: {convergence_info['final_error']:.2e}"
                )

            except Exception as e:
                print(f"      Failed: {e}")
                comparison_results[solver_name] = {"error": str(e), "converged": False, "solve_time": float("inf")}

        return comparison_results

    def _analyze_solution_quality(self, U: np.ndarray, M: np.ndarray, problem: NetworkMFGProblem) -> dict[str, float]:
        """Analyze quality metrics of the solution."""
        # Mass conservation
        mass_over_time = np.sum(M, axis=1)
        mass_conservation = np.std(mass_over_time)

        # Solution smoothness (variation across network)
        final_density_var = np.var(M[-1, :])
        final_value_var = np.var(U[-1, :])

        # Convergence to equilibrium (temporal stability)
        late_density_var = np.var(M[-5:, :], axis=0).mean()

        return {
            "mass_conservation": mass_conservation,
            "final_density_variance": final_density_var,
            "final_value_variance": final_value_var,
            "temporal_stability": late_density_var,
            "total_final_cost": self._compute_total_cost(U[-1, :], M[-1, :], problem),
        }

    def _compute_total_cost(self, u: np.ndarray, m: np.ndarray, problem: NetworkMFGProblem) -> float:
        """Compute total cost of final configuration."""
        total_cost = 0.0

        for i in range(problem.num_nodes):
            # Terminal cost
            total_cost += m[i] * u[i]

            # Congestion cost
            total_cost += m[i] * problem.density_coupling(i, m, problem.T)

        return total_cost

    def run_full_benchmark(self) -> dict[str, Any]:
        """
        Run complete benchmark across all networks and solvers.

        Returns:
            Comprehensive benchmark results
        """
        print("=" * 80)
        print("NETWORK MFG BENCHMARK SUITE")
        print("=" * 80)
        print(f"Configuration: {self.num_nodes} nodes, T={self.T}, Nt={self.Nt}")
        print()

        # Create networks
        networks = self.create_benchmark_networks()

        # Run benchmarks
        all_results = {}

        for network_name, network_geometry in networks.items():
            print(f"Benchmarking {network_name} network...")

            # Create problem
            problem = self.create_standard_mfg_problem(network_geometry)

            # Run solver comparison
            solver_results = self.run_solver_comparison(problem, network_name)

            all_results[network_name] = {
                "problem": problem,
                "solver_results": solver_results,
                "network_stats": self.network_data[network_name]["statistics"],
            }

            print()

        self.results = all_results
        return all_results

    def create_benchmark_report(self, save_path: str | None = None) -> Any:
        """
        Create comprehensive benchmark report with visualizations.

        Args:
            save_path: Path to save report (optional)

        Returns:
            Report visualization object
        """
        if not self.results:
            raise ValueError("No benchmark results available. Run benchmark first.")

        if PLOTLY_AVAILABLE:
            return self._create_plotly_report(save_path)
        else:
            return self._create_matplotlib_report(save_path)

    def _create_plotly_report(self, save_path: str | None = None) -> go.Figure:
        """Create interactive benchmark report using Plotly."""
        # Prepare data for visualization
        network_names = list(self.results.keys())
        solver_names = ["explicit", "implicit", "flow"]

        # Extract performance metrics
        solve_times = {}
        convergence_rates = {}
        solution_qualities = {}

        for network in network_names:
            solve_times[network] = []
            convergence_rates[network] = []
            solution_qualities[network] = []

            for solver in solver_names:
                result = self.results[network]["solver_results"].get(solver, {})

                solve_times[network].append(result.get("solve_time", float("inf")))
                convergence_rates[network].append(1 if result.get("converged", False) else 0)

                quality = result.get("solution_quality", {})
                solution_qualities[network].append(quality.get("mass_conservation", 1.0))

        # Create subplots
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=["Network Properties", "Solver Performance", "Convergence Rates", "Solution Quality"],
            specs=[[{"type": "bar"}, {"type": "bar"}], [{"type": "bar"}, {"type": "scatter"}]],
        )

        # Network properties comparison
        densities = [self.results[name]["network_stats"]["density"] for name in network_names]
        [self.results[name]["network_stats"]["average_degree"] for name in network_names]

        fig.add_trace(
            go.Bar(x=network_names, y=densities, name="Network Density", marker_color="lightblue"), row=1, col=1
        )

        # Solver performance (solve times)
        for i, solver in enumerate(solver_names):
            times = [solve_times[network][i] for network in network_names]
            fig.add_trace(
                go.Bar(x=network_names, y=times, name=f"{solver.capitalize()} Solver", offsetgroup=i), row=1, col=2
            )

        # Convergence rates
        for i, solver in enumerate(solver_names):
            rates = [convergence_rates[network][i] for network in network_names]
            fig.add_trace(
                go.Bar(x=network_names, y=rates, name=f"{solver.capitalize()} Convergence", offsetgroup=i), row=2, col=1
            )

        # Solution quality scatter
        for network in network_names:
            qualities = solution_qualities[network]
            times = solve_times[network]
            fig.add_trace(
                go.Scatter(x=times, y=qualities, mode="markers", name=f"{network.capitalize()}", marker={"size": 10}),
                row=2,
                col=2,
            )

        fig.update_layout(title="Network MFG Benchmark Report", height=800, showlegend=True)

        # Update axes labels
        fig.update_xaxes(title_text="Network Type", row=1, col=1)
        fig.update_yaxes(title_text="Density", row=1, col=1)
        fig.update_xaxes(title_text="Network Type", row=1, col=2)
        fig.update_yaxes(title_text="Solve Time (s)", row=1, col=2)
        fig.update_xaxes(title_text="Network Type", row=2, col=1)
        fig.update_yaxes(title_text="Convergence Rate", row=2, col=1)
        fig.update_xaxes(title_text="Solve Time (s)", row=2, col=2)
        fig.update_yaxes(title_text="Mass Conservation", row=2, col=2)

        if save_path:
            fig.write_html(save_path)

        return fig

    def _create_matplotlib_report(self, save_path: str | None = None) -> plt.Figure:
        """Create benchmark report using matplotlib."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle("Network MFG Benchmark Report", fontsize=16)

        network_names = list(self.results.keys())
        solver_names = ["explicit", "implicit", "flow"]
        colors = ["blue", "orange", "green"]

        # Network properties
        ax = axes[0, 0]
        densities = [self.results[name]["network_stats"]["density"] for name in network_names]
        ax.bar(network_names, densities, color="lightblue", alpha=0.7)
        ax.set_ylabel("Network Density")
        ax.set_title("Network Properties")
        ax.tick_params(axis="x", rotation=45)

        # Solver performance
        ax = axes[0, 1]
        x_pos = np.arange(len(network_names))
        width = 0.25

        for i, solver in enumerate(solver_names):
            times = []
            for network in network_names:
                result = self.results[network]["solver_results"].get(solver, {})
                times.append(result.get("solve_time", float("inf")))

            ax.bar(x_pos + i * width, times, width, label=solver.capitalize(), color=colors[i], alpha=0.7)

        ax.set_xlabel("Network Type")
        ax.set_ylabel("Solve Time (s)")
        ax.set_title("Solver Performance")
        ax.set_xticks(x_pos + width)
        ax.set_xticklabels(network_names)
        ax.legend()
        ax.set_yscale("log")

        # Convergence rates
        ax = axes[1, 0]
        for i, solver in enumerate(solver_names):
            rates = []
            for network in network_names:
                result = self.results[network]["solver_results"].get(solver, {})
                rates.append(1 if result.get("converged", False) else 0)

            ax.bar(x_pos + i * width, rates, width, label=solver.capitalize(), color=colors[i], alpha=0.7)

        ax.set_xlabel("Network Type")
        ax.set_ylabel("Convergence Rate")
        ax.set_title("Convergence Success")
        ax.set_xticks(x_pos + width)
        ax.set_xticklabels(network_names)
        ax.legend()

        # Solution quality
        ax = axes[1, 1]
        for network in network_names:
            times, qualities = [], []
            for solver in solver_names:
                result = self.results[network]["solver_results"].get(solver, {})
                if result.get("converged", False):
                    times.append(result.get("solve_time", float("inf")))
                    quality = result.get("solution_quality", {})
                    qualities.append(quality.get("mass_conservation", 1.0))

            if times and qualities:
                ax.scatter(times, qualities, label=network.capitalize(), s=50, alpha=0.7)

        ax.set_xlabel("Solve Time (s)")
        ax.set_ylabel("Mass Conservation Error")
        ax.set_title("Quality vs Performance")
        ax.legend()
        ax.set_xscale("log")
        ax.set_yscale("log")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig


def main():
    """Main function for advanced network MFG comparison."""
    print("=" * 80)
    print("ADVANCED NETWORK MFG COMPARISON")
    print("=" * 80)
    print("This example compares different network topologies and solver types")
    print("for Mean Field Games on networks.")
    print()

    try:
        # Create benchmark suite
        benchmark = NetworkMFGBenchmark(num_nodes=25, T=1.0, Nt=40)

        # Run full benchmark
        results = benchmark.run_full_benchmark()

        # Create and display report
        print("Creating benchmark report...")
        report_fig = benchmark.create_benchmark_report()

        if hasattr(report_fig, "show"):
            report_fig.show()  # Plotly
        else:
            plt.show()  # Matplotlib

        # Print summary
        print("=" * 80)
        print("BENCHMARK SUMMARY")
        print("=" * 80)

        for network_name, result in results.items():
            print(f"\n{network_name.upper()} NETWORK:")
            stats = result["network_stats"]
            print(f"  Edges: {stats['num_edges']}, Density: {stats['density']:.3f}")
            print(f"  Average degree: {stats['average_degree']:.2f}")

            print("  Solver Performance:")
            for solver_name, solver_result in result["solver_results"].items():
                if "error" not in solver_result:
                    converged = solver_result["converged"]
                    time_taken = solver_result["solve_time"]
                    error = solver_result["final_error"]
                    print(f"    {solver_name:10}: {'✓' if converged else '✗'} ({time_taken:.2f}s, error: {error:.2e})")
                else:
                    print(f"    {solver_name:10}: Failed - {solver_result['error']}")

        print("\n" + "=" * 80)
        print("Key Insights:")
        print("- Different network topologies require different solution approaches")
        print("- Scale-free networks often show hub concentration effects")
        print("- Flow-based solvers provide better mass conservation")
        print("- Explicit schemes are faster but may need smaller time steps")
        print("=" * 80)

    except Exception as e:
        print(f"Error in advanced network MFG example: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
