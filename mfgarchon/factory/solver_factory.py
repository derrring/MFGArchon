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

from typing import TYPE_CHECKING, Any, Literal

from mfgarchon.alg.numerical.coupling import FixedPointIterator
from mfgarchon.config import MFGSolverConfig
from mfgarchon.utils.mfg_logging.logger import get_logger

logger = get_logger(__name__)

if TYPE_CHECKING:
    from mfgarchon.alg.numerical.fp_solvers.base_fp import BaseFPSolver
    from mfgarchon.alg.numerical.hjb_solvers.base_hjb import BaseHJBSolver
    from mfgarchon.core.mfg_problem import MFGProblem

SolverType = Literal["fixed_point"]


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
                "Example: problem = MFGProblem("
                "geometry=TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[51]), Nt=100, T=1.0)"
            )

        # Validate solver type with helpful suggestions
        valid_types = ["fixed_point"]
        if solver_type not in valid_types:
            suggestions = "\n".join([f"  • {t}" for t in valid_types])
            raise ValueError(f"Unknown solver type: '{solver_type}'\n\nValid solver types are:\n{suggestions}")

        # Create base solver based on type
        if solver_type == "fixed_point":
            base_solver = SolverFactory._create_fixed_point_solver(problem, config, hjb_solver, fp_solver, **kwargs)
        else:
            raise ValueError(f"Unsupported solver type: {solver_type}. Available types: ['fixed_point']")

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
            # Issue #1317: guard the particle sub-config.
            # When fp.method != 'particle' (e.g. 'fdm'/'fem'/'network'),
            # particle is None and accessing .num_particles raises
            # AttributeError.  Only set num_particles when the particle
            # sub-config is present.
            if updated_config.fp.particle is not None:
                updated_config.fp.particle.num_particles = kwargs["num_particles"]
            else:
                logger.warning(
                    "Ignoring 'num_particles' kwarg: fp.particle sub-config is None "
                    "(fp.method=%r is not 'particle'). Set fp.method='particle' first.",
                    updated_config.fp.method,
                )
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
