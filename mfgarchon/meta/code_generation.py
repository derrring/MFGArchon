"""
Automatic code generation for MFG solvers and discretizations.

This module provides meta-programming capabilities for generating optimized
solver implementations from high-level mathematical specifications.

Features:
- Automatic finite difference scheme generation
- Solver class generation from templates
- Optimization-specific code specialization
- Backend-agnostic code generation
"""

from __future__ import annotations

import ast
import textwrap
from dataclasses import dataclass


@dataclass
class DiscretizationScheme:
    """Represents a discretization scheme for PDE operators."""

    operator: str  # "grad", "div", "laplacian", etc.
    order: int  # Approximation order
    stencil: list[int]  # Stencil points
    coefficients: list[float]  # Stencil coefficients
    boundary_treatment: str = "periodic"


class CodeGenerator:
    """
    Base class for generating optimized numerical code.

    Provides utilities for AST manipulation, code templating,
    and optimization-specific transformations.
    """

    def __init__(self, backend: str = "numpy"):
        self.backend = backend
        self.templates: dict[str, str] = {}
        self.optimizations: list[str] = []

    def add_template(self, name: str, template: str) -> None:
        """Add code template."""
        self.templates[name] = template

    def generate_from_template(self, template_name: str, **kwargs) -> str:
        """Generate code from template with substitutions."""
        if template_name not in self.templates:
            raise ValueError(f"Template '{template_name}' not found")

        template = self.templates[template_name]
        return template.format(**kwargs)

    def optimize_code(self, code: str) -> str:
        """Apply optimization transformations to generated code."""
        # Parse code to AST
        tree = ast.parse(code)

        # Apply optimizations
        for opt in self.optimizations:
            tree = self._apply_optimization(tree, opt)  # type: ignore[assignment]

        # Convert back to code
        return ast.unparse(tree)

    def _apply_optimization(self, tree: ast.AST, optimization: str) -> ast.AST:
        """Apply specific optimization to AST."""
        if optimization == "loop_unrolling":
            return self._unroll_loops(tree)
        elif optimization == "constant_folding":
            return self._fold_constants(tree)
        elif optimization == "vectorization":
            return self._vectorize_operations(tree)
        else:
            return tree

    def _unroll_loops(self, tree: ast.AST) -> ast.AST:
        """Unroll small loops for performance."""

        # Simplified loop unrolling implementation
        class LoopUnroller(ast.NodeTransformer):
            def visit_For(self, node):
                # Check if loop can be unrolled (constant bounds, small size)
                if self._can_unroll(node):
                    return self._unroll_for_loop(node)
                return node

            def _can_unroll(self, node):
                # Simplified check - in practice would be more sophisticated
                return False

            def _unroll_for_loop(self, node):
                # Simplified unrolling - would expand loop body
                return node

        return LoopUnroller().visit(tree)

    def _fold_constants(self, tree: ast.AST) -> ast.AST:
        """Fold constant expressions."""
        # Would implement constant folding optimization
        return tree

    def _vectorize_operations(self, tree: ast.AST) -> ast.AST:
        """Convert scalar operations to vectorized ones."""
        # Would implement vectorization transformations
        return tree


class MFGSolverGenerator(CodeGenerator):
    """
    Specialized code generator for MFG solvers.

    Generates complete solver classes from mathematical specifications
    and discretization schemes.
    """

    def __init__(self, backend: str = "numpy"):
        super().__init__(backend)
        self._setup_mfg_templates()

    def _setup_mfg_templates(self):
        """Initialize MFG-specific code templates."""

        # HJB solver template
        self.add_template(
            "hjb_solver",
            '''
class Generated{solver_name}(BaseHJBSolver):
    """Auto-generated HJB solver with {discretization} discretization."""

    def __init__(self, problem, config=None):
        super().__init__(problem, config)
        self.stencil = {stencil}
        self.coefficients = {coefficients}

    def _compute_spatial_derivatives(self, u, x_idx):
        """Compute spatial derivatives using {discretization} scheme."""
        {derivative_code}

    def _apply_boundary_conditions(self, u):
        """Apply {boundary_type} boundary conditions."""
        {boundary_code}

    def _newton_iteration_step(self, u_current, m_current, time_idx):
        """Single Newton iteration step."""
        {newton_code}
''',
        )

        # Fokker-Planck solver template
        self.add_template(
            "fp_solver",
            '''
class Generated{solver_name}(BaseFPSolver):
    """Auto-generated Fokker-Planck solver."""

    def _compute_divergence(self, flux, x_idx):
        """Compute divergence using {discretization} scheme."""
        {divergence_code}

    def _compute_flux(self, m, optimal_control):
        """Compute flux m * v - sigma^2/2 * grad(m)."""
        {flux_code}
''',
        )

        # Full MFG solver template
        self.add_template(
            "mfg_solver",
            '''
class Generated{solver_name}(BaseMFGSolver):
    """Auto-generated MFG solver combining HJB and FP components."""

    def __init__(self, problem, config=None):
        super().__init__(problem, config)
        self.hjb_solver = Generated{hjb_name}(problem, config.hjb)
        self.fp_solver = Generated{fp_name}(problem, config.fp)

    def solve(self, initial_conditions=None):
        """Solve MFG system using generated solvers."""
        {solve_code}
''',
        )

    def generate_hjb_solver(
        self,
        name: str,
        discretization: DiscretizationScheme,
        hamiltonian_code: str,
        boundary_type: str = "periodic",
    ) -> str:
        """Generate HJB solver class."""

        # Generate derivative computation code
        derivative_code = self._generate_derivative_code(discretization)

        # Generate boundary condition code
        boundary_code = self._generate_boundary_code(boundary_type)

        # Generate Newton iteration code
        newton_code = self._generate_newton_code(hamiltonian_code)

        return self.generate_from_template(
            "hjb_solver",
            solver_name=name,
            discretization=discretization.operator,
            stencil=discretization.stencil,
            coefficients=discretization.coefficients,
            derivative_code=derivative_code,
            boundary_type=boundary_type,
            boundary_code=boundary_code,
            newton_code=newton_code,
        )

    def generate_fp_solver(self, name: str, discretization: DiscretizationScheme) -> str:
        """Generate Fokker-Planck solver class."""

        divergence_code = self._generate_divergence_code(discretization)
        flux_code = self._generate_flux_code()

        return self.generate_from_template(
            "fp_solver",
            solver_name=name,
            discretization=discretization.operator,
            divergence_code=divergence_code,
            flux_code=flux_code,
        )

    def generate_complete_solver(
        self,
        name: str,
        hjb_scheme: DiscretizationScheme,
        fp_scheme: DiscretizationScheme,
        hamiltonian_code: str,
    ) -> str:
        """Generate complete MFG solver."""

        hjb_name = f"{name}HJB"
        fp_name = f"{name}FP"

        # Generate component solvers
        hjb_code = self.generate_hjb_solver(hjb_name, hjb_scheme, hamiltonian_code)
        fp_code = self.generate_fp_solver(fp_name, fp_scheme)

        # Generate main solver
        solve_code = self._generate_mfg_solve_code()

        main_solver = self.generate_from_template(
            "mfg_solver",
            solver_name=name,
            hjb_name=hjb_name,
            fp_name=fp_name,
            solve_code=solve_code,
        )

        return "\n\n".join([hjb_code, fp_code, main_solver])

    def _generate_derivative_code(self, scheme: DiscretizationScheme) -> str:
        """Generate code for spatial derivative computation."""
        if scheme.operator == "gradient":
            return self._generate_gradient_code(scheme)
        elif scheme.operator == "laplacian":
            return self._generate_laplacian_code(scheme)
        else:
            raise ValueError(f"Unsupported operator: {scheme.operator}")

    def _generate_gradient_code(self, scheme: DiscretizationScheme) -> str:
        """Generate gradient computation code."""
        if scheme.order == 2:
            return textwrap.dedent(
                """
            dx = self.problem.geometry.get_grid_spacing()[0]
            if x_idx == 0:
                # Forward difference at left boundary
                grad = (-3*u[0] + 4*u[1] - u[2]) / (2*dx)
            elif x_idx == len(u)-1:
                # Backward difference at right boundary
                grad = (u[-3] - 4*u[-2] + 3*u[-1]) / (2*dx)
            else:
                # Central difference in interior
                grad = (u[x_idx+1] - u[x_idx-1]) / (2*dx)
            return grad
            """
            ).strip()
        elif scheme.order == 4:
            return textwrap.dedent(
                """
            dx = self.problem.geometry.get_grid_spacing()[0]
            # Fourth-order central difference
            if x_idx < 2 or x_idx >= len(u)-2:
                # Fall back to second-order near boundaries
                if x_idx <= 1:
                    grad = (-3*u[x_idx] + 4*u[x_idx+1] - u[x_idx+2]) / (2*dx)
                else:
                    grad = (u[x_idx-2] - 4*u[x_idx-1] + 3*u[x_idx]) / (2*dx)
            else:
                grad = (-u[x_idx+2] + 8*u[x_idx+1] - 8*u[x_idx-1] + u[x_idx-2]) / (12*dx)
            return grad
            """
            ).strip()
        else:
            raise ValueError(f"Unsupported gradient order: {scheme.order}")

    def _generate_laplacian_code(self, scheme: DiscretizationScheme) -> str:
        """Generate Laplacian computation code."""
        return textwrap.dedent(
            """
        dx = self.problem.geometry.get_grid_spacing()[0]
        if x_idx == 0 or x_idx == len(u)-1:
            # Boundary - use one-sided stencil or Neumann condition
            laplacian = 0.0
        else:
            # Standard three-point stencil
            laplacian = (u[x_idx-1] - 2*u[x_idx] + u[x_idx+1]) / (dx**2)
        return laplacian
        """
        ).strip()

    def _generate_boundary_code(self, boundary_type: str) -> str:
        """Generate boundary condition code."""
        if boundary_type == "periodic":
            return "# Periodic boundary conditions handled in stencil"
        elif boundary_type == "dirichlet":
            return textwrap.dedent(
                """
            # Dirichlet boundary conditions
            u[0] = self.problem.boundary_left
            u[-1] = self.problem.boundary_right
            """
            ).strip()
        elif boundary_type == "neumann":
            return textwrap.dedent(
                """
            # Neumann boundary conditions (zero gradient)
            u[0] = u[1]
            u[-1] = u[-2]
            """
            ).strip()
        else:
            return f"# {boundary_type} boundary conditions not implemented"

    def _generate_newton_code(self, hamiltonian_code: str) -> str:
        """Generate Newton iteration code."""
        return textwrap.dedent(
            f"""
        # Compute Hamiltonian and its derivatives
        grad_u = self._compute_spatial_derivatives(u_current, x_idx)
        H = {hamiltonian_code}
        H_p = self._compute_hamiltonian_derivative_p(grad_u, m_current[x_idx])
        H_pp = self._compute_hamiltonian_derivative_pp(grad_u, m_current[x_idx])

        # Newton update
        residual = H + self.problem.running_cost(x_idx, m_current[x_idx])
        if abs(H_pp) > 1e-12:
            update = -residual / H_pp
        else:
            update = 0.0

        return u_current[x_idx] + self.config.damping_factor * update
        """
        ).strip()

    def _generate_divergence_code(self, scheme: DiscretizationScheme) -> str:
        """Generate divergence computation code."""
        return textwrap.dedent(
            """
        dx = self.problem.geometry.get_grid_spacing()[0]
        if x_idx == 0:
            # Forward difference
            div = (flux[1] - flux[0]) / dx
        elif x_idx == len(flux)-1:
            # Backward difference
            div = (flux[-1] - flux[-2]) / dx
        else:
            # Central difference
            div = (flux[x_idx+1] - flux[x_idx-1]) / (2*dx)
        return div
        """
        ).strip()

    def _generate_flux_code(self) -> str:
        """Generate flux computation code."""
        return textwrap.dedent(
            """
        sigma = float(self.problem.volatility)
        dx = self.problem.geometry.get_grid_spacing()[0]

        # Compute density gradient
        if x_idx == 0:
            m_grad = (m[1] - m[0]) / dx
        elif x_idx == len(m)-1:
            m_grad = (m[-1] - m[-2]) / dx
        else:
            m_grad = (m[x_idx+1] - m[x_idx-1]) / (2*dx)

        # Flux = m * v - sigma^2/2 * grad(m)
        flux = m[x_idx] * optimal_control[x_idx] - 0.5 * sigma**2 * m_grad
        return flux
        """
        ).strip()

    def _generate_mfg_solve_code(self) -> str:
        """Generate main MFG solve loop code."""
        return textwrap.dedent(
            """
        if initial_conditions is None:
            u, m = self._create_initial_conditions()
        else:
            u, m = initial_conditions

        for iteration in range(self.config.max_iterations):
            # HJB step
            u_new = self.hjb_solver.solve_backward(u, m)

            # Compute optimal control
            optimal_control = self.hjb_solver.compute_optimal_control(u_new, m)

            # Fokker-Planck step
            m_new = self.fp_solver.solve_forward(m, optimal_control)

            # Check convergence
            u_error = np.linalg.norm(u_new - u)
            m_error = np.linalg.norm(m_new - m)

            if u_error < self.config.tolerance and m_error < self.config.tolerance:
                break

            u, m = u_new, m_new

        return u, m
        """
        ).strip()


def generate_solver_class(
    solver_name: str,
    mathematical_system,
    discretization_schemes: dict[str, DiscretizationScheme],
    backend: str = "numpy",
) -> str:
    """
    Generate complete solver class from mathematical specification.

    Args:
        solver_name: Name for generated solver class
        mathematical_system: CompiledMFGSystem instance
        discretization_schemes: Dict mapping operators to schemes
        backend: Computational backend ("numpy", "jax", "numba")

    Returns:
        Generated solver class code as string
    """
    generator = MFGSolverGenerator(backend)

    # Extract Hamiltonian code
    if "hamiltonian" in mathematical_system.expressions:
        hamiltonian_expr = mathematical_system.expressions["hamiltonian"]
        hamiltonian_code = hamiltonian_expr.expression
    else:
        hamiltonian_code = "0.5 * p**2"  # Default quadratic

    # Get discretization schemes
    hjb_scheme = discretization_schemes.get("gradient", DiscretizationScheme("gradient", 2, [-1, 0, 1], [-0.5, 0, 0.5]))
    fp_scheme = discretization_schemes.get(
        "divergence", DiscretizationScheme("divergence", 2, [-1, 0, 1], [-0.5, 0, 0.5])
    )

    return generator.generate_complete_solver(solver_name, hjb_scheme, fp_scheme, hamiltonian_code)


def generate_discretization(operator: str, order: int, domain_type: str = "interval") -> DiscretizationScheme:
    """
    Generate discretization scheme for given operator and order.

    Args:
        operator: gradient, "laplacian", "divergence", etc.
        order: Approximation order (2, 4, 6, etc.)
        domain_type: interval, "periodic", "circle", etc.

    Returns:
        DiscretizationScheme with appropriate stencil and coefficients
    """
    if operator == "gradient":
        if order == 2:
            return DiscretizationScheme("gradient", 2, [-1, 1], [-0.5, 0.5])
        elif order == 4:
            return DiscretizationScheme("gradient", 4, [-2, -1, 1, 2], [1 / 12, -8 / 12, 8 / 12, -1 / 12])
        else:
            raise ValueError(f"Unsupported gradient order: {order}")

    elif operator == "laplacian":
        if order == 2:
            return DiscretizationScheme("laplacian", 2, [-1, 0, 1], [1, -2, 1])
        elif order == 4:
            return DiscretizationScheme(
                "laplacian",
                4,
                [-2, -1, 0, 1, 2],
                [-1 / 12, 16 / 12, -30 / 12, 16 / 12, -1 / 12],
            )
        else:
            raise ValueError(f"Unsupported Laplacian order: {order}")

    else:
        raise ValueError(f"Unsupported operator: {operator}")


# Example usage and testing functions


def test_code_generation():
    """Test the code generation framework."""
    from .mathematical_dsl import quadratic_mfg_system

    # Create mathematical system
    system = quadratic_mfg_system(control_cost=0.5, state_cost=1.0)

    # Define discretization schemes
    schemes = {
        "gradient": generate_discretization("gradient", 2),
        "laplacian": generate_discretization("laplacian", 2),
        "divergence": generate_discretization("gradient", 2),  # Divergence uses gradient
    }

    # Generate solver
    solver_code = generate_solver_class("TestQuadraticSolver", system, schemes)

    print("Generated Solver Code:")
    print("=" * 50)
    print(solver_code)

    return solver_code


if __name__ == "__main__":
    test_code_generation()
