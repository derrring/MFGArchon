"""
Visualization utilities for MFGarchon.

Provides convergence and solver diagnostics plotting (matplotlib).
For solution visualization, use matplotlib/plotly/pyvista directly.

For **data, not figures** — rendering-free, plot-ready numpy/pandas artifacts (convergence history,
space-time slices, mass series, per-node graph trajectories) — use ``extract.*`` (Issue #1362), which
imports no renderer.
"""

from .convergence_plots import (
    plot_convergence_rate,
    plot_convergence_summary,
    plot_distribution_evolution,
    plot_error_history,
    plot_from_monitor,
    plot_mass_history,
    plot_multi_error_history,
    plot_wasserstein_history,
)
from .extract import (
    extract_convergence_history,
    extract_density_slices,
    extract_graph_trajectories,
    extract_mass_history,
    extract_value_slices,
)
from .vtk_export import export_mesh_solution_vtk, export_time_series_vtk

__all__ = [
    "export_mesh_solution_vtk",
    "export_time_series_vtk",
    "extract_convergence_history",
    "extract_density_slices",
    "extract_graph_trajectories",
    "extract_mass_history",
    "extract_value_slices",
    "plot_convergence_rate",
    "plot_convergence_summary",
    "plot_distribution_evolution",
    "plot_error_history",
    "plot_from_monitor",
    "plot_mass_history",
    "plot_multi_error_history",
    "plot_wasserstein_history",
]
