#!/usr/bin/env python3
"""
MFG-specific convergence monitors and utilities.

This module provides convergence monitoring specifically designed for
Mean Field Game problems with coupled HJB-FP systems:

- _ErrorHistoryTracker: Internal helper for tracking error statistics
- DistributionConvergenceMonitor: Multi-criteria monitor with Wasserstein/KL/moments
  (renamed from AdvancedConvergenceMonitor)
- SolverTypeDetector: Detect if solver uses particle methods
  (renamed from ParticleMethodDetector)
- ConvergenceWrapper: Universal wrapper for adaptive convergence
  (renamed from AdaptiveConvergenceWrapper)

These utilities are designed for MFG problems where:
- U (value function) and M (distribution) need separate convergence tracking
- Distribution metrics (Wasserstein, KL) are relevant
- Particle vs grid-based methods require different convergence criteria
"""

from __future__ import annotations

import inspect
import warnings
from collections import deque
from functools import wraps
from typing import TYPE_CHECKING, Any

import numpy as np

from mfgarchon.utils.mfg_logging import get_logger

from .convergence_metrics import DistributionComparator

if TYPE_CHECKING:
    from mfgarchon.alg.base_mfg_solver import MFGSolver  # type: ignore[import-not-found]

logger = get_logger(__name__)
# =============================================================================
# INTERNAL ERROR HISTORY TRACKER
# =============================================================================


class _ErrorHistoryTracker:
    """
    Internal helper to track error history and compute statistics.

    This replaces OscillationDetector with a simpler, statistics-only interface.
    No threshold-based decisions - just collects data and reports statistics.
    """

    def __init__(self, history_length: int = 10):
        """
        Initialize error history tracker.

        Args:
            history_length: Number of recent values to track
        """
        self.history_length = history_length
        self.error_history: deque[float] = deque(maxlen=history_length)

    def add_sample(self, error: float):
        """Add new error sample to history."""
        self.error_history.append(error)

    def get_statistics(self) -> dict[str, float | int]:
        """
        Get statistics of tracked errors.

        Returns:
            Dictionary with mean, std, min, max, samples
        """
        if len(self.error_history) == 0:
            return {"samples": 0}

        errors = np.array(self.error_history)
        return {
            "mean": float(np.mean(errors)),
            "std": float(np.std(errors)),
            "min": float(np.min(errors)),
            "max": float(np.max(errors)),
            "samples": len(self.error_history),
        }

    def is_below_threshold(self, mean_threshold: float, std_threshold: float) -> tuple[bool, dict[str, Any]]:
        """
        Check if errors are below thresholds.

        Args:
            mean_threshold: Maximum acceptable mean error
            std_threshold: Maximum acceptable standard deviation

        Returns:
            (passed, diagnostics)
        """
        if len(self.error_history) < self.history_length:
            return False, {
                "status": "insufficient_history",
                "samples": len(self.error_history),
                "required": self.history_length,
            }

        stats = self.get_statistics()
        mean_ok = stats["mean"] < mean_threshold
        std_ok = stats["std"] < std_threshold

        return bool(mean_ok and std_ok), {
            **stats,
            "mean_ok": mean_ok,
            "std_ok": std_ok,
        }


# =============================================================================
# DISTRIBUTION CONVERGENCE MONITOR (renamed from AdvancedConvergenceMonitor)
# =============================================================================


class DistributionConvergenceMonitor:
    """
    Comprehensive convergence monitoring for MFG systems.

    Implements robust convergence criteria combining:
    - Wasserstein distance for distribution (M) convergence
    - Error stabilization for value function (U) convergence
    - Multi-criteria validation

    This monitor is specifically designed for MFG problems where both
    U and M need to converge, and distribution metrics are meaningful.
    """

    def __init__(
        self,
        wasserstein_tol: float = 1e-4,
        kl_divergence_tol: float = 1e-3,
        moment_tol: float = 1e-5,
        u_magnitude_tol: float = 1e-3,
        u_stability_tol: float = 1e-4,
        history_length: int = 10,
    ):
        """
        Initialize distribution convergence monitor.

        Args:
            wasserstein_tol: Tolerance for Wasserstein distance
            kl_divergence_tol: Tolerance for KL divergence
            moment_tol: Tolerance for statistical moment changes
            u_magnitude_tol: Tolerance for mean L2 error of value function
            u_stability_tol: Tolerance for L2 error standard deviation
            history_length: Number of iterations to track for stability
        """
        self.wasserstein_tol = wasserstein_tol
        self.kl_divergence_tol = kl_divergence_tol
        self.moment_tol = moment_tol
        self.u_magnitude_tol = u_magnitude_tol
        self.u_stability_tol = u_stability_tol

        # Initialize components
        self.comparator = DistributionComparator()
        self._error_tracker = _ErrorHistoryTracker(history_length)

        # History tracking
        self.iteration_count = 0
        self.convergence_history: list[dict[str, Any]] = []
        self.previous_m: np.ndarray | None = None
        self.previous_m_moments: dict[str, float] | None = None
        self.x_grid: np.ndarray | None = None

    def update(
        self,
        u_current: np.ndarray,
        u_previous: np.ndarray,
        m_current: np.ndarray,
        x_grid: np.ndarray,
    ) -> dict[str, Any]:
        """
        Update convergence monitoring with current iteration data.

        Args:
            u_current, u_previous: Current and previous value functions
            m_current: Current distribution
            x_grid: Spatial grid points

        Returns:
            Dictionary with convergence diagnostics
        """
        self.iteration_count += 1
        self.x_grid = x_grid

        # Compute L2 error for value function
        u_l2_error = float(np.linalg.norm(u_current - u_previous))
        self._error_tracker.add_sample(u_l2_error)

        # Initialize diagnostics
        diagnostics: dict[str, Any] = {
            "iteration": self.iteration_count,
            "u_l2_error": u_l2_error,
            "converged": False,
            "convergence_criteria": {},
        }

        # Distribution convergence analysis
        if self.previous_m is not None:
            # Wasserstein distance
            try:
                wasserstein_dist = self.comparator.wasserstein_1d(m_current, self.previous_m, x_grid)
                diagnostics["wasserstein_distance"] = wasserstein_dist
                diagnostics["convergence_criteria"]["wasserstein"] = wasserstein_dist < self.wasserstein_tol
            except Exception as e:
                warnings.warn(f"Wasserstein computation failed: {e}")
                diagnostics["convergence_criteria"]["wasserstein"] = False

            # KL divergence
            try:
                kl_div = self.comparator.kl_divergence(m_current, self.previous_m)
                diagnostics["kl_divergence"] = kl_div
                diagnostics["convergence_criteria"]["kl_divergence"] = kl_div < self.kl_divergence_tol
            except Exception as e:
                warnings.warn(f"KL divergence computation failed: {e}")
                diagnostics["convergence_criteria"]["kl_divergence"] = False

            # Statistical moments
            current_moments = self.comparator.statistical_moments(m_current, x_grid)
            if self.previous_m_moments is not None:
                mean_diff = abs(current_moments["mean"] - self.previous_m_moments["mean"])
                var_diff = abs(current_moments["variance"] - self.previous_m_moments["variance"])

                diagnostics["moment_differences"] = {
                    "mean_diff": mean_diff,
                    "var_diff": var_diff,
                }
                diagnostics["convergence_criteria"]["moments"] = (
                    mean_diff < self.moment_tol and var_diff < self.moment_tol
                )
            else:
                diagnostics["convergence_criteria"]["moments"] = False

            self.previous_m_moments = current_moments
        else:
            # First iteration - no comparison possible
            diagnostics["convergence_criteria"].update({"wasserstein": False, "kl_divergence": False, "moments": False})

        # Value function stabilization analysis
        u_stabilized, u_diagnostics = self._error_tracker.is_below_threshold(self.u_magnitude_tol, self.u_stability_tol)

        diagnostics["u_stabilization"] = u_diagnostics
        diagnostics["convergence_criteria"]["u_stabilized"] = u_stabilized

        # Overall convergence assessment
        criteria = diagnostics["convergence_criteria"]
        if self.iteration_count >= 3:  # Need some history for meaningful assessment
            diagnostics["converged"] = criteria.get("wasserstein", False) and criteria.get("u_stabilized", False)

        # Store in history
        self.convergence_history.append(diagnostics)
        self.previous_m = m_current.copy()

        return diagnostics

    def get_convergence_summary(self) -> dict[str, Any]:
        """
        Get summary of convergence behavior over all iterations.

        Returns:
            Summary dictionary with convergence statistics
        """
        if not self.convergence_history:
            return {"status": "no_data"}

        # Extract time series
        u_errors = [d["u_l2_error"] for d in self.convergence_history]
        wasserstein_dists = [d.get("wasserstein_distance", np.nan) for d in self.convergence_history]

        # Convergence detection
        converged_iterations = [i for i, d in enumerate(self.convergence_history) if d["converged"]]

        summary = {
            "total_iterations": len(self.convergence_history),
            "converged": len(converged_iterations) > 0,
            "convergence_iteration": (converged_iterations[0] if converged_iterations else None),
            "final_u_error": u_errors[-1],
            "final_wasserstein": (wasserstein_dists[-1] if not np.isnan(wasserstein_dists[-1]) else None),
            "u_error_trend": {
                "min": float(np.min(u_errors)),
                "max": float(np.max(u_errors)),
                "final_mean": float(np.mean(u_errors[-5:]) if len(u_errors) >= 5 else np.mean(u_errors)),
                "final_std": float(np.std(u_errors[-5:]) if len(u_errors) >= 5 else np.std(u_errors)),
            },
        }

        return summary

    def get_plot_data(self) -> dict[str, list[float]]:
        """
        Get data formatted for plotting.

        Returns:
            Dictionary with 'iterations', 'u_errors', 'wasserstein_distances'
        """
        if not self.convergence_history:
            return {"iterations": [], "u_errors": [], "wasserstein_distances": []}

        return {
            "iterations": [d["iteration"] for d in self.convergence_history],
            "u_errors": [d["u_l2_error"] for d in self.convergence_history],
            "wasserstein_distances": [d.get("wasserstein_distance", np.nan) for d in self.convergence_history],
        }


# =============================================================================
# SOLVER TYPE DETECTOR (renamed from ParticleMethodDetector)
# =============================================================================


class SolverTypeDetector:
    """
    Utility class to detect solver characteristics.

    Primarily used to detect if a solver uses particle-based methods,
    which require different convergence criteria than grid-based methods.
    """

    @staticmethod
    def detect_particle_methods(solver: MFGSolver) -> tuple[bool, dict[str, Any]]:
        """
        Detect if solver uses particle-based methods.

        Args:
            solver: MFG solver instance

        Returns:
            (has_particles, detection_info)
        """
        detection_info: dict[str, Any] = {
            "particle_components": [],
            "detection_methods": [],
            "confidence": 0.0,
        }

        # Check 1: Look for particle-related attributes
        particle_attributes = [
            "num_particles",
            "particles",
            "particle_solver",
            "fp_solver",
            "particle_fp",
            "kde_bandwidth",
        ]

        found_attributes = []
        for attr in particle_attributes:
            if getattr(solver, attr, None) is not None:
                found_attributes.append(attr)
                detection_info["particle_components"].append(f"attribute:{attr}")

        if found_attributes:
            detection_info["detection_methods"].append("attribute_scan")
            detection_info["confidence"] += 0.3

        # Check 2: Inspect solver components for particle classes
        particle_class_names = [
            "ParticleFP",
            "FPParticle",
            "ParticleCollocation",
            "ParticleFPSolver",
            "FPParticleSolver",
        ]

        # Check solver's fp_solver if it exists
        fp_solver = getattr(solver, "fp_solver", None)
        if fp_solver is not None:
            fp_class_name = fp_solver.__class__.__name__
            if any(particle_name in fp_class_name for particle_name in particle_class_names):
                detection_info["particle_components"].append(f"fp_solver:{fp_class_name}")
                detection_info["detection_methods"].append("component_class_inspection")
                detection_info["confidence"] += 0.5

        # Check solver's own class name
        solver_class_name = solver.__class__.__name__
        if any(particle_name in solver_class_name for particle_name in particle_class_names):
            detection_info["particle_components"].append(f"solver:{solver_class_name}")
            detection_info["detection_methods"].append("solver_class_inspection")
            detection_info["confidence"] += 0.4

        # Check 3: Look for particle-specific methods
        particle_methods = [
            "update_particles",
            "particle_step",
            "kde_estimation",
            "particle_density",
            "resample_particles",
        ]

        found_methods = []
        for method in particle_methods:
            method_obj = getattr(solver, method, None)
            if method_obj is not None and callable(method_obj):
                found_methods.append(method)
                detection_info["particle_components"].append(f"method:{method}")

        if found_methods:
            detection_info["detection_methods"].append("method_inspection")
            detection_info["confidence"] += 0.2

        # Check 4: Analyze solve method signature for particle-related parameters
        if getattr(solver, "solve", None) is not None:
            try:
                sig = inspect.signature(solver.solve)
                particle_params = [
                    "num_particles",
                    "kde_bandwidth",
                    "normalize_kde_output",
                ]
                found_params = [p for p in particle_params if p in sig.parameters]
                if found_params:
                    detection_info["particle_components"].extend([f"param:{p}" for p in found_params])
                    detection_info["detection_methods"].append("parameter_inspection")
                    detection_info["confidence"] += 0.1
            except Exception as e:
                logger.warning(f"Failed to inspect solve method signature: {e}")
                # Ignore signature inspection errors

        # Final decision based on confidence
        has_particles = detection_info["confidence"] > 0.3

        # Boost confidence if multiple detection methods agree
        if len(detection_info["detection_methods"]) >= 2:
            detection_info["confidence"] = min(1.0, detection_info["confidence"] * 1.2)
            has_particles = True

        return has_particles, detection_info


# =============================================================================
# CONVERGENCE WRAPPER (renamed from AdaptiveConvergenceWrapper)
# =============================================================================


class ConvergenceWrapper:
    """
    Wrapper that automatically applies appropriate convergence criteria
    based on whether the solver uses particle methods.

    Usage:
        @adaptive_convergence()
        class MySolver(BaseMFGSolver):
            ...

    Or as a wrapper:
        solver = MySolver(...)
        adaptive_solver = ConvergenceWrapper(solver)
    """

    def __init__(
        self,
        solver: MFGSolver | None = None,
        classical_tol: float = 1e-3,
        force_particle_mode: bool | None = None,
        verbose: bool = True,
        **advanced_convergence_kwargs,
    ):
        """
        Initialize convergence wrapper.

        Args:
            solver: MFG solver to wrap (if used as wrapper)
            classical_tol: L2 error tolerance for classical methods
            force_particle_mode: Force particle/classical mode (overrides detection)
            verbose: Print convergence mode information
            **advanced_convergence_kwargs: Parameters for distribution convergence monitor
        """
        self._wrapped_solver = solver
        self.classical_tol = classical_tol
        self.force_particle_mode = force_particle_mode
        self.verbose = verbose
        self.advanced_convergence_kwargs = advanced_convergence_kwargs

        # State variables
        self._particle_mode: bool | None = None
        self._detection_info: dict[str, Any] | None = None
        self._convergence_monitor: DistributionConvergenceMonitor | None = None
        self._original_solve = None

        if solver is not None:
            self._wrap_solver(solver)

    def __call__(self, solver_class):
        """Use as decorator on solver classes."""
        original_init = solver_class.__init__
        wrapper = self

        @wraps(original_init)
        def wrapped_init(self_solver, *args, **kwargs):
            # Call original constructor
            original_init(self_solver, *args, **kwargs)
            # Apply convergence wrapper
            wrapper._wrap_solver(self_solver)

        solver_class.__init__ = wrapped_init
        return solver_class

    def _wrap_solver(self, solver: MFGSolver):
        """Apply convergence wrapping to a solver instance."""
        self._wrapped_solver = solver

        # Detect particle methods
        if self.force_particle_mode is not None:
            self._particle_mode = self.force_particle_mode
            self._detection_info = {"forced": True, "mode": self.force_particle_mode}
        else:
            self._particle_mode, self._detection_info = SolverTypeDetector.detect_particle_methods(solver)

        # Set up convergence monitoring
        if self._particle_mode:
            self._convergence_monitor = create_distribution_monitor(**self.advanced_convergence_kwargs)

        # Wrap the solve method
        solve_method = getattr(solver, "solve", None)
        if solve_method is not None and callable(solve_method):
            self._original_solve = solver.solve
            solver.solve = self._adaptive_solve

        # Store reference to wrapper in solver for debugging
        solver._convergence_wrapper = self  # type: ignore[attr-defined]

        if self.verbose:
            self._print_convergence_mode()

    def _print_convergence_mode(self):
        """Print information about the detected convergence mode."""
        print("ADAPTIVE CONVERGENCE DETECTION")
        print("-" * 50)

        if self._particle_mode:
            print("PARTICLE METHODS DETECTED")
            print("   -> Using DISTRIBUTION convergence criteria")
            print("   -> Wasserstein distance + error stabilization")

            # Show detection details
            if self._detection_info and "particle_components" in self._detection_info:
                particle_components = self._detection_info["particle_components"]
                if isinstance(particle_components, list) and particle_components:
                    components = particle_components[:3]  # Show first 3
                    print(f"   -> Evidence: {', '.join(components)}")

            confidence = self._detection_info.get("confidence", 0) if self._detection_info else 0
            print(f"   -> Confidence: {confidence:.1%}")

            # Show monitor settings
            monitor = self._convergence_monitor
            if monitor:
                print(f"   -> Wasserstein tolerance: {monitor.wasserstein_tol}")
                print(f"   -> U magnitude tolerance: {monitor.u_magnitude_tol}")
                print(f"   -> Stability tolerance: {monitor.u_stability_tol}")
        else:
            print("GRID-BASED METHODS DETECTED")
            print("   -> Using CLASSICAL L2 error convergence")
            print(f"   -> L2 tolerance: {self.classical_tol}")

        print()

    def _adaptive_solve(self, *args, **kwargs):
        """Adaptive solve method that uses appropriate convergence criteria."""
        if not self._particle_mode:
            return self._classical_solve(*args, **kwargs)
        else:
            return self._particle_aware_solve(*args, **kwargs)

    def _classical_solve(self, *args, **kwargs):
        """Classical solve with L2 error convergence."""
        if self._original_solve is not None:
            return self._original_solve(*args, **kwargs)
        else:
            raise RuntimeError("No original solve method available")

    def _particle_aware_solve(
        self, Niter: int = 20, l2errBound: float | None = None, verbose: bool | None = None, **kwargs
    ):
        """Particle-aware solve with distribution convergence criteria."""
        if verbose is None:
            verbose = self.verbose

        if l2errBound is None:
            l2errBound = self.classical_tol

        if verbose:
            print("SOLVING WITH DISTRIBUTION CONVERGENCE CRITERIA")
            print("-" * 50)

        solver = self._wrapped_solver
        problem = getattr(solver, "problem", None)

        if problem is None:
            warnings.warn("Cannot access problem from solver - falling back to classical convergence")
            return self._classical_solve(Niter, l2errBound, verbose, **kwargs)

        # Initialize convergence monitoring
        self._convergence_monitor = create_distribution_monitor(**self.advanced_convergence_kwargs)

        # Extract spatial grid for convergence analysis
        bounds = problem.geometry.get_bounds()
        grid_shape = problem.geometry.get_grid_shape()
        x_grid = np.linspace(bounds[0][0], bounds[1][0], grid_shape[0])

        try:
            if self._original_solve is not None:
                results = self._original_solve(Niter, l2errBound, verbose=False, **kwargs)
            else:
                raise RuntimeError("No original solve method available")

            # Analyze results with distribution criteria
            if len(results) >= 2:
                U, M = results[0], results[1]
                info = results[2] if len(results) > 2 else {}

                advanced_info = self._analyze_solution_convergence(U, M, x_grid)

                if isinstance(info, dict):
                    info.update(advanced_info)
                else:
                    info = advanced_info

                return U, M, info
            else:
                return results

        except Exception as e:
            warnings.warn(f"Distribution convergence analysis failed: {e}. Falling back to classical.")
            return self._classical_solve(Niter, l2errBound, verbose, **kwargs)

    def _analyze_solution_convergence(self, U: np.ndarray, M: np.ndarray, x_grid: np.ndarray) -> dict[str, Any]:
        """Post-hoc analysis of solution convergence using distribution criteria."""
        if U.ndim >= 2 and M.ndim >= 2:
            final_m = M[-1, :]

            # Compute distribution properties
            comparator = DistributionComparator()
            moments = comparator.statistical_moments(final_m, x_grid)

            advanced_info: dict[str, Any] = {
                "convergence_mode": "distribution_aware",
                "particle_detection": self._detection_info,
                "final_distribution_moments": moments,
                "distribution_convergence_available": True,
            }

            # Add mass tracking
            if M.shape[0] > 1:
                dx = (x_grid[-1] - x_grid[0]) / (len(x_grid) - 1)
                initial_mass = np.sum(M[0, :]) * dx
                final_mass = np.sum(M[-1, :]) * dx
                mass_change = abs(final_mass - initial_mass) / initial_mass * 100

                advanced_info["mass_change_percent"] = mass_change

            return advanced_info

        return {
            "convergence_mode": "distribution_aware",
            "particle_detection": self._detection_info,
            "distribution_convergence_available": True,
        }

    def get_convergence_mode(self) -> str:
        """Get current convergence mode."""
        return "particle_aware" if self._particle_mode else "classical"

    def get_detection_info(self) -> dict[str, Any]:
        """Get solver type detection information."""
        return self._detection_info or {}

    def __getattr__(self, name):
        """Delegate attribute access to wrapped solver."""
        if self._wrapped_solver is not None:
            return getattr(self._wrapped_solver, name)
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================


def create_distribution_monitor(**kwargs) -> DistributionConvergenceMonitor:
    """
    Create distribution convergence monitor with default settings.

    Args:
        **kwargs: Override default parameters

    Returns:
        Configured DistributionConvergenceMonitor
    """
    defaults = {
        "wasserstein_tol": 1e-4,
        "kl_divergence_tol": 1e-3,
        "moment_tol": 1e-5,
        "u_magnitude_tol": 1e-3,
        "u_stability_tol": 1e-4,
        "history_length": 10,
    }
    defaults.update(kwargs)

    return DistributionConvergenceMonitor(**defaults)  # type: ignore[arg-type]


def adaptive_convergence(
    classical_tol: float = 1e-3,
    force_particle_mode: bool | None = None,
    verbose: bool = True,
    **advanced_kwargs,
):
    """
    Decorator factory for adaptive convergence.

    Args:
        classical_tol: L2 error tolerance for classical methods
        force_particle_mode: Force particle/classical mode (overrides detection)
        verbose: Print convergence mode information
        **advanced_kwargs: Parameters for distribution convergence monitor

    Usage:
        @adaptive_convergence(classical_tol=1e-3, wasserstein_tol=1e-4)
        class MySolver(BaseMFGSolver):
            ...
    """

    def decorator(solver_class):
        return ConvergenceWrapper(
            classical_tol=classical_tol,
            force_particle_mode=force_particle_mode,
            verbose=verbose,
            **advanced_kwargs,
        )(solver_class)

    return decorator


def wrap_solver_with_adaptive_convergence(solver: MFGSolver, **kwargs) -> MFGSolver:
    """
    Wrap an existing solver instance with adaptive convergence.

    Args:
        solver: Solver instance to wrap
        **kwargs: Parameters for ConvergenceWrapper

    Returns:
        Wrapped solver with adaptive convergence

    Usage:
        solver = ParticleCollocationSolver(...)
        adaptive_solver = wrap_solver_with_adaptive_convergence(solver)
        U, M, info = adaptive_solver.solve(...)
    """
    ConvergenceWrapper(solver, **kwargs)
    return solver  # The solver is modified in-place by the wrapper


def test_particle_detection(solver: MFGSolver) -> dict[str, Any]:
    """
    Test solver type detection without wrapping.

    Args:
        solver: Solver to test

    Returns:
        Detection results dictionary
    """
    has_particles, detection_info = SolverTypeDetector.detect_particle_methods(solver)

    return {
        "has_particles": has_particles,
        "detection_info": detection_info,
        "recommended_convergence": "distribution" if has_particles else "classical",
    }
