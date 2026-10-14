"""
Meta-programming framework for MFGarchon.

**Experimental, unverified: no test covers its output** (#2648). Nothing in the library uses this
subpackage, and it carries no documentation beyond its docstrings. Its code generator's solver source
does not compile today -- ``generate_solver_class`` emits an HJB class whose methods are mis-indented, so
no generated solver has run. Treat everything here as a sketch, not as a solver.

This module provides advanced meta-programming capabilities for automatically
generating, optimizing, and composing MFG solvers and mathematical formulations.

Key Features:
- Automatic solver generation from mathematical specifications
- Runtime code optimization and specialization
- Mathematical expression compilation
- Dynamic type system for numerical methods
"""

from .code_generation import CodeGenerator, MFGSolverGenerator, generate_discretization, generate_solver_class
from .mathematical_dsl import HamiltonianBuilder, LagrangianBuilder, MathematicalExpression, MFGSystemBuilder
from .optimization_meta import JITSolverFactory, OptimizationCompiler, create_optimized_solver
from .type_system import DynamicSolver, MFGType, SolverMetaclass, TypedMFGProblem

__all__ = [
    "CodeGenerator",
    "DynamicSolver",
    "HamiltonianBuilder",
    "JITSolverFactory",
    "LagrangianBuilder",
    "MFGSolverGenerator",
    "MFGSystemBuilder",
    "MFGType",
    "MathematicalExpression",
    "OptimizationCompiler",
    "SolverMetaclass",
    "TypedMFGProblem",
    "create_optimized_solver",
    "generate_discretization",
    "generate_solver_class",
]
