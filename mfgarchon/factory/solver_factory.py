#!/usr/bin/env python3
"""
MFG Solver Factory

Provides factory for creating MFG solvers with default configuration.

Config Type Support
-------------------
The factory accepts both legacy MFGSolverConfig and the modern SolverConfig:

- SolverConfig (recommended): Clean hierarchy from mfgarchon.config.core
- MFGSolverConfig (deprecated): Legacy config with backward compatibility

Both configs share compatible structure for the fields used by solvers
(picard.max_iterations, picard.tolerance, etc.), enabling gradual migration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from mfgarchon.alg.numerical.coupling import FixedPointIterator
from mfgarchon.config import MFGSolverConfig
from mfgarchon.utils.deprecation import deprecated
from mfgarchon.utils.mfg_logging.logger import get_logger

logger = get_logger(__name__)

if TYPE_CHECKING:
    from mfgarchon.alg.numerical.fp_solvers.base_fp import BaseFPSolver
    from mfgarchon.alg.numerical.hjb_solvers.base_hjb import BaseHJBSolver
    from mfgarchon.core.mfg_problem import MFGProblem

SolverType = Literal["fixed_point"]


@dataclass
class SolverFactoryConfig:
    """Configuration for solver factory behavior."""

    solver_type: SolverType = "fixed_point"
    custom_config: MFGSolverConfig | None = None
    solver_kwargs: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.solver_kwargs is None:
            self.solver_kwargs = {}


class SolverFactory:
    """Factory for creating MFG solvers with default configuration."""

    @staticmethod
    def create_solver(
        problem: MFGProblem,
        solver_type: SolverType = "fixed_point",
        hjb_solver: BaseHJBSolver | None = None,
        fp_solver: BaseFPSolver | None = None,
        config: MFGSolverConfig | None = None,
        **kwargs: Any,
    ) -> FixedPointIterator:
        """
        Create an MFG solver.

        Args:
            problem: MFG problem to solve
            solver_type: Type of solver to create
            hjb_solver: HJB solver instance
            fp_solver: FP solver instance
            config: Solver configuration (uses defaults if None)
            **kwargs: Additional solver-specific parameters

        Returns:
            Configured solver instance
        """
        # Get configuration
        if config is None:
            config = MFGSolverConfig()

        # Update config with any kwargs
        config = SolverFactory._update_config_with_kwargs(config, **kwargs)

        # Validate problem first
        if problem is None:
            raise ValueError(
                "Problem cannot be None. Please provide a valid MFGProblem instance.\n"
                "Example: problem = MFGProblem(Nx=50, Nt=100, T=1.0)"
            )

        # Validate solver type with helpful suggestions
        valid_types = ["fixed_point"]
        if solver_type not in valid_types:
            suggestions = "\n".join([f"  • {t}" for t in valid_types])
            raise ValueError(
                f"Unknown solver type: '{solver_type}'\n\n"
                f"Valid solver types are:\n{suggestions}\n\n"
                f"Note: 'particle_collocation' has been removed from core package.\n"
                f"Example: create_fast_solver(problem, solver_type='fixed_point')"
            )

        # Create base solver based on type
        if solver_type == "fixed_point":
            base_solver = SolverFactory._create_fixed_point_solver(problem, config, hjb_solver, fp_solver, **kwargs)
        else:
            raise ValueError(
                f"Unsupported solver type: {solver_type}. "
                "Note: 'particle_collocation' has been removed from core package. "
                f"Available types: ['fixed_point']"
            )

        return base_solver

    @staticmethod
    def _update_config_with_kwargs(config: MFGSolverConfig, **kwargs: Any) -> MFGSolverConfig:
        """Update configuration with keyword arguments."""
        # Create a copy to avoid modifying original
        import copy

        updated_config = copy.deepcopy(config)

        # Common config updates
        if "max_picard_iterations" in kwargs:
            updated_config.picard.max_iterations = kwargs["max_picard_iterations"]
        if "picard_tolerance" in kwargs:
            updated_config.picard.tolerance = kwargs["picard_tolerance"]
        if "max_newton_iterations" in kwargs:
            updated_config.hjb.newton.max_iterations = kwargs["max_newton_iterations"]
        if "newton_tolerance" in kwargs:
            updated_config.hjb.newton.tolerance = kwargs["newton_tolerance"]
        # Note: return_structured and warm_start were deprecated and removed from MFGSolverConfig

        # Particle-specific updates
        if "num_particles" in kwargs:
            updated_config.fp.particle.num_particles = kwargs["num_particles"]
        if "delta" in kwargs:
            # Issue #1284 / 2026-06-11 survey: guard the gfdm sub-config.
            # When hjb.method == 'fdm' (the default), gfdm is None and
            # accessing .delta raises AttributeError.  Only set delta when
            # the GFDM sub-config is present.
            if updated_config.hjb.gfdm is not None:
                updated_config.hjb.gfdm.delta = kwargs["delta"]
            else:
                logger.warning(
                    "Ignoring 'delta' kwarg: hjb.gfdm sub-config is None "
                    "(hjb.method=%r is not 'gfdm'). Set hjb.method='gfdm' first.",
                    updated_config.hjb.method,
                )

        return updated_config

    @staticmethod
    def _create_fixed_point_solver(
        problem: MFGProblem,
        config: MFGSolverConfig,
        hjb_solver: BaseHJBSolver | None,
        fp_solver: BaseFPSolver | None,
        **kwargs: Any,
    ) -> FixedPointIterator:
        """Create a fixed point iterator solver."""
        if hjb_solver is None or fp_solver is None:
            raise ValueError("Fixed point solver requires both hjb_solver and fp_solver")

        # Filter out config-related kwargs that shouldn't be passed to constructor
        config_keys = {
            "max_picard_iterations",
            "picard_tolerance",
            "max_newton_iterations",
            "newton_tolerance",
            # return_structured and warm_start were deprecated and removed
            "num_particles",
            "delta",
        }
        constructor_kwargs = {k: v for k, v in kwargs.items() if k not in config_keys}

        return FixedPointIterator(
            problem=problem,
            hjb_solver=hjb_solver,
            fp_solver=fp_solver,
            config=config,
            **constructor_kwargs,
        )

    # Note: _create_particle_collocation_solver, _create_monitored_particle_solver,
    # and _create_adaptive_particle_solver methods have been removed.
    # Particle-collocation methods have been removed from core package.


# Convenience function


@deprecated(
    since="v0.17.0",
    replacement=(
        "Use the new three-mode solving API instead (Issue #580):\n"
        "  - Safe Mode: problem.solve(scheme=NumericalScheme.FDM_UPWIND)\n"
        "  - Expert Mode: problem.solve(hjb_solver=hjb, fp_solver=fp)\n"
        "  - Auto Mode: problem.solve()\n"
        "See examples/basic/three_mode_api_demo.py for details."
    ),
)
def create_solver(
    problem: MFGProblem,
    hjb_solver: BaseHJBSolver | None = None,
    fp_solver: BaseFPSolver | None = None,
    config: MFGSolverConfig | None = None,
    **kwargs: Any,
) -> FixedPointIterator:
    """
    Create an MFG solver.

    .. deprecated:: 0.17.0
        Use the new three-mode solving API instead (Issue #580):

        - Safe Mode: ``problem.solve(scheme=NumericalScheme.FDM_UPWIND)``
        - Expert Mode: ``problem.solve(hjb_solver=hjb, fp_solver=fp)``
        - Auto Mode: ``problem.solve()``

        The new API ensures adjoint duality between HJB and FP solvers.
        See examples/basic/three_mode_api_demo.py for details.

    Args:
        problem: MFG problem to solve
        hjb_solver: HJB solver instance
        fp_solver: FP solver instance
        config: Solver configuration (uses defaults if None)
        **kwargs: Additional parameters

    Returns:
        Configured solver instance

    Example:
        >>> from mfgarchon import MFGProblem
        >>> from mfgarchon.types import NumericalScheme
        >>> problem = MFGProblem(Nx=50, Nt=20, T=1.0)
        >>> result = problem.solve(scheme=NumericalScheme.FDM_UPWIND)  # Preferred

    Note:
        Prefer problem.solve() which handles solver creation and duality validation.
    """
    return SolverFactory.create_solver(
        problem=problem,
        hjb_solver=hjb_solver,
        fp_solver=fp_solver,
        config=config,
        **kwargs,
    )


# =============================================================================
# REMOVED FUNCTIONS - Use problem.solve() or create_solver() instead
# =============================================================================


def _removed_function_error(name: str) -> None:
    """Raise informative error for removed functions."""
    raise NotImplementedError(f"{name}() has been removed. Use create_solver() or problem.solve() instead.")


def create_basic_solver(*args: Any, **kwargs: Any) -> Any:
    """Removed: Use problem.solve()."""
    _removed_function_error("create_basic_solver")


def create_standard_solver(*args: Any, **kwargs: Any) -> Any:
    """Removed: Use problem.solve()."""
    _removed_function_error("create_standard_solver")


def create_fast_solver(*args: Any, **kwargs: Any) -> Any:
    """Removed: Use problem.solve()."""
    _removed_function_error("create_fast_solver")


def create_accurate_solver(*args: Any, **kwargs: Any) -> Any:
    """Removed: Use problem.solve()."""
    _removed_function_error("create_accurate_solver")


def create_research_solver(*args: Any, **kwargs: Any) -> Any:
    """Removed: Use problem.solve()."""
    _removed_function_error("create_research_solver")


def create_semi_lagrangian_solver(*args: Any, **kwargs: Any) -> Any:
    """Removed: Use problem.solve()."""
    _removed_function_error("create_semi_lagrangian_solver")


def create_amr_solver(*args: Any, **kwargs: Any) -> Any:
    """Removed: Use problem.solve()."""
    _removed_function_error("create_amr_solver")
