"""
Domain-Specific Language for Mathematical MFG Formulations.

This module provides a meta-programming framework for expressing mathematical
MFG systems as code that can be automatically compiled into efficient solvers.

Mathematical expressions are represented as abstract syntax trees that can be
manipulated, optimized, and compiled to different backends (NumPy, JAX, Numba).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass
class MathematicalExpression:
    """
    Represents a mathematical expression in abstract form.

    Can be compiled to different computational backends and automatically
    differentiated for gradient-based optimization.
    """

    expression: str
    variables: list[str] = field(default_factory=list)
    parameters: dict[str, Any] = field(default_factory=dict)
    backend: str = "numpy"
    _compiled_func: Callable | None = field(default=None, init=False, repr=False)

    def compile(self, backend: str | None = None) -> Callable:
        """Compile expression to specified backend."""
        if backend is None:
            backend = self.backend

        if self._compiled_func is not None and backend == self.backend:
            return self._compiled_func

        if backend == "numpy":
            self._compiled_func = self._compile_numpy()
        elif backend == "jax":
            self._compiled_func = self._compile_jax()
        elif backend == "numba":
            self._compiled_func = self._compile_numba()
        else:
            raise ValueError(f"Unsupported backend: {backend}")

        self.backend = backend
        return self._compiled_func

    def _compile_numpy(self) -> Callable:
        """Compile to NumPy function."""
        # Create safe namespace for evaluation
        safe_dict = {
            "np": np,
            "sin": np.sin,
            "cos": np.cos,
            "exp": np.exp,
            "log": np.log,
            "sqrt": np.sqrt,
            "abs": np.abs,
            **self.parameters,
        }

        # Build function string
        args_str = ", ".join(self.variables)
        func_str = f"lambda {args_str}: {self.expression}"

        try:
            return eval(func_str, {"__builtins__": {}}, safe_dict)
        except Exception as e:
            raise ValueError(f"Failed to compile expression '{self.expression}': {e}") from e

    def _compile_jax(self) -> Callable:
        """Compile to JAX function with automatic differentiation."""
        try:
            import jax
            import jax.numpy as jnp

            safe_dict = {
                "jnp": jnp,
                "sin": jnp.sin,
                "cos": jnp.cos,
                "exp": jnp.exp,
                "log": jnp.log,
                "sqrt": jnp.sqrt,
                "abs": jnp.abs,
                **self.parameters,
            }

            args_str = ", ".join(self.variables)
            func_str = f"lambda {args_str}: {self.expression.replace('np.', 'jnp.')}"

            func = eval(func_str, {"__builtins__": {}}, safe_dict)
            return jax.jit(func)

        except ImportError:
            raise ValueError("JAX not available for compilation") from None

    def _compile_numba(self) -> Callable:
        """Compile to Numba JIT function."""
        try:
            import numba

            # For Numba, we need to create actual function, not lambda
            func_code = f"""
def compiled_func({", ".join(self.variables)}):
    return {self.expression}
"""

            # Execute in controlled namespace
            namespace = {"numba": numba, "np": np}
            exec(func_code, namespace)

            return numba.jit(namespace["compiled_func"])

        except ImportError:
            raise ValueError("Numba not available for compilation") from None

    def differentiate(self, var: str) -> MathematicalExpression:
        """Symbolic differentiation (simplified implementation)."""
        # This is a placeholder for symbolic differentiation
        # In practice, would use SymPy or similar for full symbolic math
        diff_expr = f"derivative_of({self.expression})_wrt_{var}"
        return MathematicalExpression(diff_expr, self.variables, self.parameters, self.backend)


class MFGSystemBuilder:
    """
    Builder for constructing MFG systems from mathematical expressions.

    Allows fluent API for building complex MFG formulations:

    Example:
        system = (MFGSystemBuilder()
                 .hamiltonian("0.5 * p**2 + V(x, m)")
                 .running_cost("0.5 * (x - x_target)**2")
                 .terminal_cost("0.5 * (x - x_final)**2")
                 .build())
    """

    def __init__(self) -> None:
        self.expressions: dict[str, MathematicalExpression] = {}
        self.constraints: list[MathematicalExpression] = []
        self.parameters: dict[str, Any] = {}
        self.domain_info: dict[str, Any] = {}

    def hamiltonian(self, expr: str, variables: list[str] | None = None) -> MFGSystemBuilder:
        """Define Hamiltonian function H(t,x,p,m)."""
        if variables is None:
            variables = ["t", "x", "p", "m"]

        self.expressions["hamiltonian"] = MathematicalExpression(expr, variables, self.parameters)
        return self

    def lagrangian(self, expr: str, variables: list[str] | None = None) -> MFGSystemBuilder:
        """Define Lagrangian function L(t,x,v,m)."""
        if variables is None:
            variables = ["t", "x", "v", "m"]

        self.expressions["lagrangian"] = MathematicalExpression(expr, variables, self.parameters)
        return self

    def running_cost(self, expr: str, variables: list[str] | None = None) -> MFGSystemBuilder:
        """Define running cost function f(t,x,m)."""
        if variables is None:
            variables = ["t", "x", "m"]

        self.expressions["running_cost"] = MathematicalExpression(expr, variables, self.parameters)
        return self

    def terminal_cost(self, expr: str, variables: list[str] | None = None) -> MFGSystemBuilder:
        """Define terminal cost function g(x,m)."""
        if variables is None:
            variables = ["x", "m"]

        self.expressions["terminal_cost"] = MathematicalExpression(expr, variables, self.parameters)
        return self

    def constraint(self, expr: str, variables: list[str] | None = None) -> MFGSystemBuilder:
        """Add constraint to the MFG system."""
        if variables is None:
            variables = ["t", "x", "m"]

        self.constraints.append(MathematicalExpression(expr, variables, self.parameters))
        return self

    def parameter(self, name: str, value: Any) -> MFGSystemBuilder:
        """Set parameter value."""
        self.parameters[name] = value
        return self

    def domain(
        self,
        xmin: float,
        xmax: float,
        tmax: float,
        Nx: int | None = None,
        Nt: int | None = None,
    ) -> MFGSystemBuilder:
        """
        Define computational domain.

        Args:
            xmin: Minimum spatial coordinate
            xmax: Maximum spatial coordinate
            tmax: Final time
            Nx: Number of spatial grid points (preferred, default: 100)
            Nt: Number of temporal grid points (preferred, default: 50)

        Returns:
            Self for method chaining
        """
        # Set defaults if not provided
        if Nx is None:
            Nx = 100
        if Nt is None:
            Nt = 50

        self.domain_info = {
            "xmin": xmin,
            "xmax": xmax,
            "tmax": tmax,
            "Nx": Nx,  # Use uppercase in storage
            "Nt": Nt,  # Use uppercase in storage
        }
        return self

    def build(self) -> CompiledMFGSystem:
        """Build and compile the MFG system."""
        return CompiledMFGSystem(self.expressions, self.constraints, self.parameters, self.domain_info)


class HamiltonianBuilder(MFGSystemBuilder):
    """Specialized builder for Hamiltonian MFG systems."""

    def quadratic_control_cost(self, coeff: float = 0.5) -> HamiltonianBuilder:
        """Add quadratic control cost: coeff * p^2."""
        self.hamiltonian(f"{coeff} * p**2")
        return self

    def potential(self, expr: str) -> HamiltonianBuilder:
        """Add potential function V(x,m)."""
        current_h = self.expressions.get("hamiltonian")
        if current_h:
            new_expr = f"{current_h.expression} + ({expr})"
        else:
            new_expr = expr

        self.hamiltonian(new_expr)
        return self

    def interaction_potential(self, kernel: str = "x * m") -> HamiltonianBuilder:
        """Add mean-field interaction potential."""
        self.potential(f"interaction_strength * ({kernel})")
        return self


class LagrangianBuilder(MFGSystemBuilder):
    """Specialized builder for Lagrangian MFG systems."""

    def kinetic_energy(self, coeff: float = 0.5) -> LagrangianBuilder:
        """Add kinetic energy: coeff * v^2."""
        self.lagrangian(f"{coeff} * v**2")
        return self

    def congestion_cost(self, expr: str = "m") -> LagrangianBuilder:
        """Add congestion cost function."""
        current_l = self.expressions.get("lagrangian")
        if current_l:
            new_expr = f"{current_l.expression} + congestion_coeff * ({expr})"
        else:
            new_expr = f"congestion_coeff * ({expr})"

        self.lagrangian(new_expr)
        return self


@dataclass
class CompiledMFGSystem:
    """
    Compiled MFG system ready for numerical solution.

    Contains all mathematical expressions compiled to efficient functions
    and metadata for solver generation.
    """

    expressions: dict[str, MathematicalExpression]
    constraints: list[MathematicalExpression]
    parameters: dict[str, Any]
    domain_info: dict[str, Any]

    def compile_all(self, backend: str = "numpy") -> None:
        """Compile all expressions to specified backend."""
        for expr in self.expressions.values():
            expr.compile(backend)

        for constraint in self.constraints:
            constraint.compile(backend)

    def get_compiled_function(self, name: str) -> Callable:
        """Get compiled function by name."""
        if name not in self.expressions:
            raise ValueError(f"Expression '{name}' not found")

        expr = self.expressions[name]
        if expr._compiled_func is None:
            expr.compile()

        if expr._compiled_func is None:
            raise RuntimeError(f"Failed to compile expression '{name}'")

        return expr._compiled_func

    def to_mfg_problem(self):
        """Convert to standard MFGProblem instance."""
        from mfgarchon.core.mfg_problem import MFGProblem

        # Extract domain parameters
        domain = self.domain_info

        problem = MFGProblem(
            xmin=domain["xmin"],
            xmax=domain["xmax"],
            T=domain["tmax"],
            Nx=domain["nx"],
            Nt=domain["nt"],
        )

        # Attach compiled functions using setattr for dynamic attributes
        if "hamiltonian" in self.expressions:
            problem.hamiltonian = self.get_compiled_function("hamiltonian")

        if "running_cost" in self.expressions:
            problem.running_cost = self.get_compiled_function("running_cost")

        if "terminal_cost" in self.expressions:
            problem.terminal_cost = self.get_compiled_function("terminal_cost")

        return problem


# Convenience functions for common MFG systems


def quadratic_mfg_system(
    control_cost: float = 0.5,
    state_cost: float = 0.5,
    interaction_strength: float = 1.0,
    target: float = 0.0,
) -> CompiledMFGSystem:
    """Create standard quadratic MFG system."""
    return (
        HamiltonianBuilder()
        .quadratic_control_cost(control_cost)
        .potential(f"{state_cost} * (x - {target})**2")
        .parameter("interaction_strength", interaction_strength)
        .build()
    )


def congestion_mfg_system(
    kinetic_cost: float = 0.5, congestion_cost: float = 1.0, target_cost: float = 0.5
) -> CompiledMFGSystem:
    """Create congestion-based MFG system."""
    return (
        LagrangianBuilder()
        .kinetic_energy(kinetic_cost)
        .congestion_cost("m**2")
        .running_cost(f"{target_cost} * (x - x_target)**2")
        .parameter("congestion_coeff", congestion_cost)
        .build()
    )


def network_mfg_system(network_structure: dict[str, Any], transition_costs: dict[str, str]) -> CompiledMFGSystem:
    """Create network-based MFG system."""
    builder = MFGSystemBuilder()

    # Build Hamiltonian from network structure
    hamiltonian_terms = []
    for edge, cost_expr in transition_costs.items():
        hamiltonian_terms.append(f"({cost_expr}) * I_{edge}(x)")

    full_hamiltonian = " + ".join(hamiltonian_terms)

    return builder.hamiltonian(full_hamiltonian).parameter("network", network_structure).build()
