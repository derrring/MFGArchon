#!/usr/bin/env python3
"""
Jupyter Notebook Reporting System for MFGarchon
============================================

Comprehensive reporting system that generates interactive Jupyter notebooks
with Plotly visualizations, LaTeX mathematical expressions, and professional
research documentation for Mean Field Games analysis.

Features:
- Automatic notebook generation with templated structure
- Plotly integration for interactive visualizations
- LaTeX mathematical notation support
- Professional research report formatting
- HTML export capability for sharing
- Modular report sections
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

try:
    import nbformat as nbf
    from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

    NOTEBOOK_AVAILABLE = True
except ImportError:
    NOTEBOOK_AVAILABLE = False

try:
    import plotly.io as pio

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

from mfgarchon.utils.mfg_logging import get_logger

logger = get_logger(__name__)


class NotebookReportError(Exception):
    """Exception for notebook reporting errors."""


class MFGNotebookReporter:
    """
    Comprehensive notebook reporting system for Mean Field Games research.

    Generates interactive Jupyter notebooks with Plotly visualizations,
    LaTeX mathematical expressions, and professional documentation.
    """

    def __init__(
        self,
        output_dir: str = "reports",
        enable_latex: bool = True,
        plotly_renderer: str = "notebook",
        template_style: str = "research",
    ):
        """
        Initialize the notebook reporting system.

        Args:
            output_dir: Directory for generated reports
            enable_latex: Enable LaTeX mathematical notation
            plotly_renderer: Plotly renderer ('notebook', 'browser', 'json')
            template_style: Report template ('research', 'presentation', 'minimal')
        """
        if not NOTEBOOK_AVAILABLE:
            raise NotebookReportError(
                "Jupyter notebook support not available. Install with: pip install nbformat jupyter"
            )

        if not PLOTLY_AVAILABLE:
            raise NotebookReportError(
                "Plotly not available for interactive visualizations. Install with: pip install plotly"
            )

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.enable_latex = enable_latex
        self.plotly_renderer = plotly_renderer
        self.template_style = template_style
        self.logger = get_logger(__name__)

        # Configure Plotly for notebook rendering
        pio.renderers.default = plotly_renderer

        self.logger.info("MFG Notebook Reporter initialized")
        self.logger.info(f"Output directory: {self.output_dir}")
        self.logger.info(f"Plotly renderer: {plotly_renderer}")

    def create_research_report(
        self,
        title: str,
        solver_results: dict[str, Any],
        problem_config: dict[str, Any],
        analysis_metadata: dict[str, Any] | None = None,
        custom_sections: list[dict[str, Any]] | None = None,
    ) -> str:
        """
        Create comprehensive research report notebook.

        Args:
            title: Report title
            solver_results: Dictionary containing U, M, convergence info, etc.
            problem_config: MFG problem configuration
            analysis_metadata: Additional metadata for research context
            custom_sections: List of custom sections to add

        Returns:
            Path to generated notebook file
        """
        self.logger.info(f"Creating research report: {title}")

        # Create new notebook
        nb = new_notebook()

        # Add title and metadata
        self._add_title_section(nb, title, analysis_metadata)

        # Add problem configuration section
        self._add_problem_configuration_section(nb, problem_config)

        # Add mathematical framework section
        self._add_mathematical_framework_section(nb)

        # Add solver results and analysis
        self._add_solver_results_section(nb, solver_results)

        # Add interactive visualizations
        self._add_visualization_section(nb, solver_results)

        # Add convergence analysis
        if "convergence_info" in solver_results:
            self._add_convergence_analysis_section(nb, solver_results["convergence_info"])

        # Add mass conservation analysis
        if "M" in solver_results:
            self._add_mass_conservation_section(nb, solver_results)

        # Add custom sections if provided
        if custom_sections:
            for section in custom_sections:
                self._add_custom_section(nb, section)

        # Add conclusions and export information
        self._add_conclusions_section(nb)
        self._add_export_section(nb)

        # Save notebook
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_title = "".join(c for c in title if c.isalnum() or c in (" ", "-", "_")).rstrip()
        safe_title = safe_title.replace(" ", "_")
        filename = f"{safe_title}_{timestamp}.ipynb"
        notebook_path = self.output_dir / filename

        with open(notebook_path, "w") as f:
            nbf.write(nb, f)

        self.logger.info(f"Research report saved: {notebook_path}")
        return str(notebook_path)

    def _add_title_section(self, nb: nbf.NotebookNode, title: str, metadata: dict | None = None):
        """Add title and metadata section."""
        date_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        title_markdown = f"""# {title}

**Generated:** {date_str}
**Platform:** MFGarchon Computational Framework
**Report Type:** Interactive Research Analysis

---

## Research Context

This interactive notebook presents a comprehensive analysis of Mean Field Games (MFG) solutions using the MFGarchon computational framework. The report includes mathematical formulations, numerical results, interactive visualizations, and detailed convergence analysis.
"""

        if metadata:
            title_markdown += "\n### Analysis Metadata\n\n"
            for key, value in metadata.items():
                title_markdown += f"- **{key}**: {value}\n"

        title_markdown += "\n---\n"

        nb.cells.append(new_markdown_cell(title_markdown))

    def _add_problem_configuration_section(self, nb: nbf.NotebookNode, config: dict[str, Any]):
        """Add problem configuration section."""
        config_markdown = """## Problem Configuration

The Mean Field Game system is defined with the following parameters:

"""

        # Format configuration nicely
        for key, value in config.items():
            if isinstance(value, int | float | str | bool):
                if key in ["volatility", "sigma", "T", "coupling_coefficient"]:
                    # Add mathematical context for key parameters. A user config may still spell the
                    # volatility `sigma`; either way it is the SDE volatility, not a diffusion (#2378).
                    if key in ("volatility", "sigma"):
                        config_markdown += f"- **SDE volatility** $\\sigma = {value}$\n"
                    elif key == "T":
                        config_markdown += f"- **Time horizon** $T = {value}$\n"
                    elif key == "coupling_coefficient":
                        config_markdown += f"- **Coupling strength** $\\alpha = {value}$\n"
                    else:
                        config_markdown += f"- **{key}**: {value}\n"
                else:
                    config_markdown += f"- **{key}**: {value}\n"

        nb.cells.append(new_markdown_cell(config_markdown))

        # Add code cell to display configuration programmatically
        config_code = f"""# Problem Configuration Details
import pprint

config = {config}
print("MFG Problem Configuration:")
print("=" * 40)
pprint.pprint(config, width=60, indent=2)
"""
        nb.cells.append(new_code_cell(config_code))

    def _add_mathematical_framework_section(self, nb: nbf.NotebookNode):
        """Add mathematical framework section with LaTeX equations."""
        math_markdown = """## Mathematical Framework

### Mean Field Game System

The Mean Field Game system consists of two coupled partial differential equations:

#### Hamilton-Jacobi-Bellman Equation
The value function $u(t,x)$ satisfies:

$$-\\frac{\\partial u}{\\partial t} + H\\left(t, x, \\nabla u, m\\right) = 0$$

with Hamiltonian:
$$H(t, x, p, m) = \\frac{1}{2}|p|^2 + f(t, x, u, m)$$

#### Fokker-Planck Equation
The population density $m(t,x)$ evolves according to:

$$\\frac{\\partial m}{\\partial t} - \\nabla \\cdot \\left(m \\nabla H_p\\right) - \\frac{\\sigma^2}{2}\\Delta m = 0$$

where $H_p = \\frac{\\partial H}{\\partial p}$ is the optimal control strategy.

### Numerical Method

The system is solved using a **particle-collocation approach**:
- **HJB Equation**: Generalized Finite Difference Method (GFDM) with QP constraints
- **FP Equation**: Particle method with kernel density estimation
- **Coupling**: Fixed-point iteration with adaptive convergence criteria

### Key Properties

1. **Mass Conservation**: $\\int_{\\Omega} m(t,x) dx = 1$ for all $t \\in [0,T]$
2. **Monotonicity**: Value function maintains monotonicity through QP constraints
3. **Nash Equilibrium**: Solution represents Nash equilibrium of the MFG system

---
"""
        nb.cells.append(new_markdown_cell(math_markdown))

    def _add_solver_results_section(self, nb: nbf.NotebookNode, results: dict[str, Any]):
        """Add solver results section with code to display results."""
        results_markdown = """## Solver Results

### Solution Overview

The numerical solution provides the value function $u(t,x)$ and population density $m(t,x)$ over the time-space domain.
"""
        nb.cells.append(new_markdown_cell(results_markdown))

        # Add code to display basic results information
        results_code = """# Import required libraries
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

# Display solution dimensions and basic statistics
print("Solution Summary:")
print("=" * 50)
"""

        if "U" in results:
            results_code += f"""
print(f"Value function U shape: {results["U"].shape if hasattr(results["U"], "shape") else "scalar"}")
if hasattr(results['U'], 'shape'):
    print(f"U range: [{np.min(results["U"]):.6f}, {np.max(results["U"]):.6f}]")
    print(f"U mean: {np.mean(results["U"]):.6f}")
"""

        if "M" in results:
            results_code += f"""
print(f"Density M shape: {results["M"].shape if hasattr(results["M"], "shape") else "scalar"}")
if hasattr(results['M'], 'shape'):
    print(f"M range: [{{np.min(results['M']):.6f}}, {{np.max(results['M']):.6f}}]")
    print(f"M mean: {{np.mean(results['M']):.6f}}")

    # Check mass conservation
    if len(results['M'].shape) == 2:
        from numpy import trapezoid
        total_mass = trapezoid(results['M'][-1, :], dx=0.05)  # Approximate dx
        mass_error = abs(total_mass - 1.0)
        print(f"Final total mass: {{total_mass:.8f}}")
        print(f"Mass conservation error: {{mass_error:.2e}}")
"""

        # Store results in notebook variables
        for key, value in results.items():
            if isinstance(value, np.ndarray):
                results_code += f"\n{key} = np.{value!r}"
            elif isinstance(value, dict | list):
                results_code += f"\n{key} = {value!r}"

        nb.cells.append(new_code_cell(results_code))

    def _add_visualization_section(self, nb: nbf.NotebookNode, results: dict[str, Any]):
        """Add interactive visualization section."""
        viz_markdown = """## Interactive Visualizations

### Solution Evolution

The following interactive plots show the evolution of the value function $u(t,x)$ and population density $m(t,x)$ over time.
"""
        nb.cells.append(new_markdown_cell(viz_markdown))

        # Create comprehensive visualization code
        viz_code = """# Create interactive 3D surface plots
fig = make_subplots(
    rows=2, cols=2,
    specs=[[{'type': 'surface'}, {'type': 'surface'}],
           [{'type': 'xy'}, {'type': 'xy'}]],
    subplot_titles=[
        'Value Function u(t,x)',
        'Population Density m(t,x)',
        'Final Profiles',
        'Time Evolution at x=0.5'
    ],
    vertical_spacing=0.08
)

# Assume standard grid setup (adjust based on actual data)
if 'U' in locals() and 'M' in locals():
    Nt, Nx = U.shape
    x_grid = np.linspace(0, 1, Nx)
    t_grid = np.linspace(0, 1, Nt)  # Adjust T based on actual problem
    X_mesh, T_mesh = np.meshgrid(x_grid, t_grid)

    # 3D Surface: Value Function
    fig.add_trace(
        go.Surface(
            x=T_mesh, y=X_mesh, z=U,
            name='u(t,x)',
            colorscale='viridis',
            showscale=True
        ),
        row=1, col=1
    )

    # 3D Surface: Density
    fig.add_trace(
        go.Surface(
            x=T_mesh, y=X_mesh, z=M,
            name='m(t,x)',
            colorscale='plasma',
            showscale=True
        ),
        row=1, col=2
    )

    # Final profiles comparison
    fig.add_trace(
        go.Scatter(
            x=x_grid, y=U[-1, :],
            mode='lines+markers',
            name='Final u(T,x)',
            line=dict(color='blue', width=3)
        ),
        row=2, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=x_grid, y=M[-1, :],
            mode='lines+markers',
            name='Final m(T,x)',
            line=dict(color='red', width=3),
            yaxis='y2'
        ),
        row=2, col=1
    )

    # Time evolution at center point
    center_idx = Nx // 2
    fig.add_trace(
        go.Scatter(
            x=t_grid, y=U[:, center_idx],
            mode='lines+markers',
            name='u(t,0.5)',
            line=dict(color='green', width=3)
        ),
        row=2, col=2
    )

    fig.add_trace(
        go.Scatter(
            x=t_grid, y=M[:, center_idx],
            mode='lines+markers',
            name='m(t,0.5)',
            line=dict(color='orange', width=3),
            yaxis='y4'
        ),
        row=2, col=2
    )

# Update layout for professional appearance
fig.update_layout(
    title=dict(
        text="MFG Solution: Interactive Analysis",
        x=0.5,
        font=dict(size=20)
    ),
    height=800,
    showlegend=True,
    font=dict(size=12)
)

# Update 3D scene properties
fig.update_scenes(
    xaxis_title="Time t",
    yaxis_title="Space x",
    zaxis_title="Value",
    camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
)

# Update 2D plot properties
fig.update_xaxes(title_text="Space x", row=2, col=1)
fig.update_yaxes(title_text="u(T,x)", row=2, col=1)
fig.update_xaxes(title_text="Time t", row=2, col=2)
fig.update_yaxes(title_text="Value", row=2, col=2)

# Show the interactive plot
fig.show()
"""
        nb.cells.append(new_code_cell(viz_code))

        # Add code for additional specialized plots
        specialized_viz_code = """# Additional Specialized Visualizations

# 1. Convergence-focused analysis
if 'convergence_info' in locals():
    fig_conv = go.Figure()

    if 'error_history' in convergence_info:
        fig_conv.add_trace(
            go.Scatter(
                y=convergence_info['error_history'],
                mode='lines+markers',
                name='L2 Error',
                line=dict(color='blue', width=2)
            )
        )

    fig_conv.update_layout(
        title="Convergence History",
        xaxis_title="Iteration",
        yaxis_title="L2 Error",
        yaxis_type="log",
        height=400
    )

    fig_conv.show()

# 2. Mass conservation tracking
if 'M' in locals():
    mass_history = []
    for t_idx in range(M.shape[0]):
        mass = trapezoid(M[t_idx, :], x=x_grid)
        mass_history.append(mass)

    fig_mass = go.Figure()
    fig_mass.add_trace(
        go.Scatter(
            x=t_grid,
            y=mass_history,
            mode='lines+markers',
            name='Total Mass',
            line=dict(color='red', width=2)
        )
    )

    # Add ideal mass line
    fig_mass.add_hline(
        y=1.0,
        line_dash="dash",
        line_color="black",
        annotation_text="Ideal Mass = 1.0"
    )

    fig_mass.update_layout(
        title="Mass Conservation Over Time",
        xaxis_title="Time t",
        yaxis_title="Total Mass ∫m(t,x)dx",
        height=400
    )

    fig_mass.show()
"""
        nb.cells.append(new_code_cell(specialized_viz_code))

    def _add_convergence_analysis_section(self, nb: nbf.NotebookNode, convergence_info: dict[str, Any]):
        """Add detailed convergence analysis section."""
        conv_markdown = """## Convergence Analysis

### Numerical Convergence

Analysis of the iterative solver convergence, including error evolution and convergence criteria satisfaction.
"""
        nb.cells.append(new_markdown_cell(conv_markdown))

        conv_code = f"""# Convergence Analysis
convergence_info = {convergence_info}

print("Convergence Summary:")
print("=" * 40)
for key, value in convergence_info.items():
    if isinstance(value, (int, float, bool, str)):
        print(f"{{key}}: {{value}}")
    elif isinstance(value, list) and len(value) < 10:
        print(f"{{key}}: {{value}}")
    elif isinstance(value, list):
        print(f"{{key}}: [list with {{len(value)}} elements]")

# Analyze convergence rate if error history available
if 'error_history' in convergence_info and len(convergence_info['error_history']) > 2:
    errors = np.array(convergence_info['error_history'])
    # Compute convergence rate
    ratios = errors[1:] / errors[:-1]
    avg_ratio = np.mean(ratios[ratios > 0])

    print(f"\\nConvergence Rate Analysis:")
    print(f"Average error reduction ratio: {{avg_ratio:.4f}}")
    print(f"Estimated convergence rate: {{-np.log(avg_ratio):.4f}}")
"""
        nb.cells.append(new_code_cell(conv_code))

    def _add_mass_conservation_section(self, nb: nbf.NotebookNode, results: dict[str, Any]):
        """Add mass conservation analysis section."""
        mass_markdown = """## Mass Conservation Analysis

### Conservation Properties

Analysis of mass conservation throughout the evolution, which is crucial for the physical validity of the Mean Field Game solution.

The total mass should satisfy: $\\int_{\\Omega} m(t,x) dx = 1$ for all $t \\in [0,T]$
"""
        nb.cells.append(new_markdown_cell(mass_markdown))

        mass_code = """# Mass Conservation Analysis
if 'M' in locals() and hasattr(M, 'shape') and len(M.shape) == 2:
    # Compute mass at each time step
    mass_evolution = []
    mass_errors = []

    for t_idx in range(M.shape[0]):
        mass = trapezoid(M[t_idx, :], x=x_grid) if 'x_grid' in locals() else np.sum(M[t_idx, :]) * (1.0 / M.shape[1])
        error = abs(mass - 1.0)
        mass_evolution.append(mass)
        mass_errors.append(error)

    print("Mass Conservation Summary:")
    print("=" * 45)
    print(f"Initial mass: {mass_evolution[0]:.8f}")
    print(f"Final mass: {mass_evolution[-1]:.8f}")
    print(f"Maximum mass error: {max(mass_errors):.2e}")
    print(f"Average mass error: {np.mean(mass_errors):.2e}")
    print(f"Mass drift: {abs(mass_evolution[-1] - mass_evolution[0]):.2e}")

    # Conservation quality assessment
    max_error = max(mass_errors)
    if max_error < 1e-6:
        print("\\nExcellent mass conservation achieved")
    elif max_error < 1e-3:
        print("\\nGood mass conservation achieved")
    elif max_error < 1e-2:
        print("\\nAcceptable mass conservation")
    else:
        print("\\nWARNING: Mass conservation could be improved")

    # Store results for potential further analysis
    mass_conservation_data = {
        'mass_evolution': mass_evolution,
        'mass_errors': mass_errors,
        'max_error': max_error,
        'mean_error': np.mean(mass_errors)
    }
else:
    print("Mass conservation analysis requires 2D density array M")
"""
        nb.cells.append(new_code_cell(mass_code))

    def _add_custom_section(self, nb: nbf.NotebookNode, section: dict[str, Any]):
        """Add custom section to notebook."""
        if "title" in section:
            title_markdown = f"## {section['title']}\n\n"
            if "description" in section:
                title_markdown += f"{section['description']}\n\n"
            nb.cells.append(new_markdown_cell(title_markdown))

        if "code" in section:
            nb.cells.append(new_code_cell(section["code"]))

        if "markdown" in section:
            nb.cells.append(new_markdown_cell(section["markdown"]))

    def _add_conclusions_section(self, nb: nbf.NotebookNode):
        """Add conclusions and summary section."""
        conclusions_markdown = """## Conclusions and Analysis Summary

### Key Findings

1. **Solution Quality**: The numerical method successfully computed both the value function $u(t,x)$ and population density $m(t,x)$
2. **Mass Conservation**: Analysis shows the level of mass conservation achieved by the particle method
3. **Convergence**: The iterative solver achieved convergence within specified tolerances
4. **Physical Validity**: The solution satisfies the fundamental properties of Mean Field Game systems

### Computational Performance

- **Method**: Particle-collocation with GFDM for HJB and particle method for FP
- **Stability**: QP constraints ensure monotonicity preservation
- **Efficiency**: Optimized implementation with intelligent constraint activation

### Research Applications

This solution can be used for:
- **Economic Modeling**: Large population strategic interactions
- **Crowd Dynamics**: Pedestrian flow optimization
- **Financial Mathematics**: Optimal trading strategies in large markets
- **Engineering Control**: Multi-agent system coordination

---

*This report was generated using the MFGarchon computational framework with interactive Jupyter notebook capabilities.*
"""
        nb.cells.append(new_markdown_cell(conclusions_markdown))

    def _add_export_section(self, nb: nbf.NotebookNode):
        """Add export and sharing information section."""
        export_markdown = """## Export and Sharing

### Available Export Options

This interactive notebook can be exported in multiple formats for sharing and presentation:

"""
        nb.cells.append(new_markdown_cell(export_markdown))

        export_code = """# Export Options
print("Export and Sharing Options:")
print("=" * 40)
print("1. HTML Export (with interactive plots):")
print("   jupyter nbconvert --to html --execute notebook_name.ipynb")
print()
print("2. PDF Export (static plots):")
print("   jupyter nbconvert --to pdf notebook_name.ipynb")
print()
print("3. Slides (reveal.js presentation):")
print("   jupyter nbconvert --to slides --post serve notebook_name.ipynb")
print()
print("4. Python Script:")
print("   jupyter nbconvert --to python notebook_name.ipynb")
print()
print("5. LaTeX Export:")
print("   jupyter nbconvert --to latex notebook_name.ipynb")

# Instructions for saving plots
print("\\nSaving Interactive Plots:")
print("- Plotly plots can be saved as HTML: fig.write_html('plot.html')")
print("- Static export: fig.write_image('plot.png') # Requires kaleido")
print("- Vector format: fig.write_image('plot.pdf')")
"""
        nb.cells.append(new_code_cell(export_code))

    def export_to_html(self, notebook_path: str | Path, output_name: str | None = None) -> str:
        """
        Export notebook to HTML with embedded interactive plots.

        Args:
            notebook_path: Path to the notebook file
            output_name: Optional custom output name

        Returns:
            Path to generated HTML file
        """
        try:
            import subprocess

            notebook_path = Path(notebook_path)
            if not notebook_path.exists():
                raise NotebookReportError(f"Notebook not found: {notebook_path}")

            html_path = self.output_dir / f"{output_name}.html" if output_name else notebook_path.with_suffix(".html")

            # Execute and convert to HTML
            cmd = [
                "jupyter",
                "nbconvert",
                "--to",
                "html",
                "--execute",
                "--allow-errors",
                "--output",
                str(html_path),
                str(notebook_path),
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0:
                self.logger.info(f"HTML export successful: {html_path}")
                return str(html_path)
            else:
                raise NotebookReportError(f"HTML export failed: {result.stderr}")

        except ImportError as e:
            raise NotebookReportError("Jupyter nbconvert not available. Install with: pip install jupyter") from e
        except Exception as e:
            raise NotebookReportError(f"HTML export error: {e}") from e

    def create_comparative_report(
        self,
        results_dict: dict[str, dict[str, Any]],
        title: str = "Comparative MFG Analysis",
    ) -> str:
        """
        Create comparative analysis notebook for multiple solver results.

        Args:
            results_dict: Dictionary of {method_name: results}
            title: Report title

        Returns:
            Path to generated comparative notebook
        """
        self.logger.info(f"Creating comparative report: {title}")

        nb = new_notebook()

        # Add title
        comp_title_markdown = f"""# {title}

**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**Type:** Comparative Analysis Report
**Methods Compared:** {", ".join(results_dict.keys())}

---

## Overview

This notebook provides a comprehensive comparison of different numerical methods applied to the same Mean Field Game problem. Each method's performance, accuracy, and computational characteristics are analyzed and compared.

### Methods Under Comparison

"""

        for i, method_name in enumerate(results_dict.keys(), 1):
            comp_title_markdown += f"{i}. **{method_name}**\n"

        comp_title_markdown += "\n---\n"
        nb.cells.append(new_markdown_cell(comp_title_markdown))

        # Add comparative visualization code
        comp_viz_code = f"""# Comparative Analysis Setup
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Results data
results_dict = {results_dict}

print("Comparative Analysis Summary:")
print("=" * 50)

method_colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
methods = list(results_dict.keys())

# Create comparative plots
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=[
        'Final Value Functions u(T,x)',
        'Final Densities m(T,x)',
        'Convergence Comparison',
        'Mass Conservation Comparison'
    ]
)

for i, (method, results) in enumerate(results_dict.items()):
    color = method_colors[i % len(method_colors)]

    if 'U' in results and hasattr(results['U'], 'shape'):
        # Assume standard grid
        Nx = results['U'].shape[1] if len(results['U'].shape) > 1 else len(results['U'])
        x_grid = np.linspace(0, 1, Nx)

        # Final value function
        final_U = results['U'][-1, :] if len(results['U'].shape) > 1 else results['U']
        fig.add_trace(
            go.Scatter(x=x_grid, y=final_U, name=f'{{method}} u(T,x)',
                      line=dict(color=color, width=2)),
            row=1, col=1
        )

        # Final density
        if 'M' in results:
            final_M = results['M'][-1, :] if len(results['M'].shape) > 1 else results['M']
            fig.add_trace(
                go.Scatter(x=x_grid, y=final_M, name=f'{{method}} m(T,x)',
                          line=dict(color=color, width=2, dash='dash')),
                row=1, col=2
            )

    # Convergence comparison
    if 'convergence_info' in results and 'error_history' in results['convergence_info']:
        errors = results['convergence_info']['error_history']
        fig.add_trace(
            go.Scatter(y=errors, name=f'{{method}} convergence',
                      line=dict(color=color, width=2)),
            row=2, col=1
        )

fig.update_layout(height=800, title="Method Comparison")
fig.update_yaxes(type="log", row=2, col=1)
fig.show()

# Summary statistics
print("\\nMethod Comparison Summary:")
for method, results in results_dict.items():
    print(f"\\n{{method}}:")
    if 'convergence_info' in results:
        conv_info = results['convergence_info']
        if 'converged' in conv_info:
            print(f"  Converged: {{conv_info['converged']}}")
        if 'iterations' in conv_info:
            print(f"  Iterations: {{conv_info['iterations']}}")
        if 'final_error' in conv_info:
            print(f"  Final error: {{conv_info['final_error']:.2e}}")
"""

        nb.cells.append(new_code_cell(comp_viz_code))

        # Add conclusions
        comp_conclusions = """## Comparative Conclusions

### Performance Summary

The comparative analysis reveals the relative strengths and characteristics of each numerical method:

- **Accuracy**: Comparison of final solution quality
- **Convergence**: Rate and reliability of iterative convergence
- **Conservation**: Mass conservation properties
- **Computational Cost**: Relative computational requirements

### Method Recommendations

Based on the comparative analysis, recommendations for different use cases:

1. **High Accuracy Requirements**: Method with lowest final errors
2. **Fast Computation**: Method with fastest convergence
3. **Robust Performance**: Method with most reliable convergence
4. **Mass Conservation**: Method with best conservation properties

---
"""
        nb.cells.append(new_markdown_cell(comp_conclusions))

        # Save comparative notebook
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"comparative_analysis_{timestamp}.ipynb"
        notebook_path = self.output_dir / filename

        with open(notebook_path, "w") as f:
            nbf.write(nb, f)

        self.logger.info(f"Comparative report saved: {notebook_path}")
        return str(notebook_path)


# Convenience functions for easy usage
def create_mfg_research_report(
    title: str,
    solver_results: dict[str, Any],
    problem_config: dict[str, Any],
    output_dir: str = "reports",
    export_html: bool = True,
) -> dict[str, str]:
    """
    Convenience function to create a complete MFG research report.

    Args:
        title: Report title
        solver_results: Solver results dictionary
        problem_config: Problem configuration
        output_dir: Output directory for reports
        export_html: Whether to also export HTML version

    Returns:
        Dictionary with 'notebook' and optionally 'html' paths
    """
    reporter = MFGNotebookReporter(output_dir=output_dir)

    notebook_path = reporter.create_research_report(
        title=title, solver_results=solver_results, problem_config=problem_config
    )

    result_paths = {"notebook": notebook_path}

    if export_html:
        try:
            html_path = reporter.export_to_html(notebook_path)
            result_paths["html"] = html_path
        except Exception as e:
            print(f"HTML export failed: {e}")

    return result_paths


def create_comparative_analysis(
    results_dict: dict[str, dict[str, Any]],
    title: str = "MFG Method Comparison",
    output_dir: str = "reports",
    export_html: bool = True,
) -> dict[str, str]:
    """
    Convenience function to create comparative analysis report.

    Args:
        results_dict: Dictionary of {method_name: results}
        title: Report title
        output_dir: Output directory
        export_html: Whether to export HTML

    Returns:
        Dictionary with file paths
    """
    reporter = MFGNotebookReporter(output_dir=output_dir)

    notebook_path = reporter.create_comparative_report(results_dict=results_dict, title=title)

    result_paths = {"notebook": notebook_path}

    if export_html:
        try:
            html_path = reporter.export_to_html(notebook_path)
            result_paths["html"] = html_path
        except Exception as e:
            print(f"HTML export failed: {e}")

    return result_paths
