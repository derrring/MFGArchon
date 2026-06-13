"""
Base solver classes for the new algorithm structure.

This module defines the foundational classes that all algorithm paradigms inherit from,
ensuring consistent interfaces while allowing paradigm-specific customization.

BC Integration (Issue #527):
    All solvers access boundary conditions via self.problem.geometry.boundary_conditions.
    Each paradigm has helper methods for paradigm-appropriate BC handling:
    - Numerical: apply_boundary_conditions() -> field operations
    - Neural: sample_boundary_points(), compute_boundary_loss() -> training
    - RL: _configure_environment_boundaries() -> environment setup
    - Optimization: get_domain_constraints() -> constraint generation
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from mfgarchon.config import BaseConfig
    from mfgarchon.core import MFGProblem
    from mfgarchon.geometry.boundary import BoundaryConditions


class SchemeFamily(Enum):
    """
    Numerical scheme families for duality validation.

    This enum is used internally by the solver pairing system to validate
    HJB-FP duality relationships without relying on fragile class name matching.

    Solvers annotate themselves with a _scheme_family class attribute to enable
    refactoring-safe duality checking (Issue #543 validator pattern).

    Usage (solver implementation):
        >>> class HJBFDMSolver(BaseHJBSolver):
        ...     _scheme_family = SchemeFamily.FDM

    Usage (duality validation):
        >>> hjb_family = getattr(hjb_solver, '_scheme_family', SchemeFamily.GENERIC)
        >>> fp_family = getattr(fp_solver, '_scheme_family', SchemeFamily.GENERIC)
        >>> if hjb_family == fp_family and hjb_family != SchemeFamily.GENERIC:
        ...     # Same family → likely dual

    Scheme Families
    ---------------

    **FDM** (Finite Difference Methods):
        - Structured grid discretization
        - Discrete adjoint: div and grad are matrix transposes
        - Examples: Upwind, centered differences, WENO
        - Duality: Type A (exact discrete transpose)

    **SL** (Semi-Lagrangian):
        - Characteristic-based discretization
        - Discrete adjoint: Forward splatting (scatter) is transpose of backward interpolation (gather)
        - Examples: Linear interpolation, cubic interpolation
        - Duality: Type A (exact discrete transpose)

    **FVM** (Finite Volume Methods):
        - Conservation form discretization
        - Discrete adjoint: Numerical flux must be consistent
        - Examples: Godunov, Lax-Friedrichs, Roe
        - Duality: Type A (exact discrete transpose)
        - Status: Future extension

    **GFDM** (Meshfree/Generalized Finite Differences):
        - Point cloud discretization with weighted least squares
        - Continuous adjoint: Asymmetric neighborhoods prevent discrete transpose
        - Examples: RBF-FD, GFDM, particle methods
        - Duality: Type B (continuous only, L_FP = L_HJB^T + O(h))

    **FEM** (Finite Element Methods):
        - Variational discretization on unstructured meshes
        - Discrete adjoint: Mass and stiffness matrices are symmetric
        - Examples: P1 (linear), P2 (quadratic) Lagrange elements
        - Duality: Type A (exact discrete transpose for symmetric bilinear forms)
        - Backend: scikit-fem assembly (Issue #773)

    **PINN** (Physics-Informed Neural Networks):
        - Neural network discretization
        - Shared architecture for dual solvers
        - Duality: Type B (continuous only)
        - Status: Future extension

    **GENERIC** (Unknown/Custom):
        - Default for solvers without explicit family annotation
        - Used as fallback when _scheme_family attribute missing
        - Duality: Unknown (validation skipped)

    See Also
    --------
    - NumericalScheme: User-facing enum for Safe Mode API
    - check_solver_duality(): Validation function using this enum
    - Issue #706: Adjoint operators mathematical foundation

    References
    ----------
    - Issue #580: Adjoint-aware solver pairing
    - Issue #543: Validator pattern (try/except instead of hasattr)
    """

    FDM = "fdm"  # Finite Difference Methods
    SL = "semi_lagrangian"  # Semi-Lagrangian
    FVM = "fvm"  # Finite Volume (future)
    FEM = "fem"  # Finite Element Methods (Issue #773)
    GFDM = "gfdm"  # Meshfree GFDM (strong-form, Type B)
    MESHLESS_GALERKIN = "meshless_galerkin"  # Meshfree Galerkin / MLS (weak-form, Type A; Issue #1131)
    PINN = "pinn"  # Physics-Informed Neural Network (future)
    GENERIC = "generic"  # Unknown/custom solvers

    def __str__(self) -> str:
        """Human-readable string representation."""
        return self.value


class BaseMFGSolver(ABC):
    """
    Abstract base class for all MFG solvers across paradigms.

    This class defines the common interface that all solvers must implement,
    regardless of their mathematical approach (numerical, neural, RL, optimization).
    """

    def __init__(self, problem: MFGProblem, config: BaseConfig) -> None:
        """
        Initialize the solver with a problem and configuration.

        Args:
            problem: The MFG problem to solve
            config: Solver-specific configuration
        """
        self.problem = problem
        self.config = config
        self._is_solved = False
        self._solution: Any | None = None

    @abstractmethod
    def solve(self) -> Any:  # Concrete solvers should override with specific return type
        """
        Solve the MFG problem.

        Returns:
            SolverResult object containing u(t,x), m(t,x) and metadata.
            Note: For backward compatibility, SolverResult supports tuple unpacking.
        """

    @abstractmethod
    def validate_solution(self) -> dict[str, float]:
        """
        Validate the computed solution.

        Returns:
            Dictionary of validation metrics (Nash gap, mass conservation, etc.)
        """

    @property
    def is_solved(self) -> bool:
        """Check if the solver has computed a solution."""
        return self._is_solved

    @property
    def solution(self) -> Any:
        """Get the computed solution."""
        if not self._is_solved:
            raise RuntimeError("Solver has not been run. Call solve() first.")
        return self._solution

    def get_boundary_conditions(self) -> BoundaryConditions | None:
        """
        Access boundary conditions from problem geometry.

        Returns:
            BoundaryConditions object or None if not available.

        Note:
            This is the single source of truth for BC information.
            All paradigms should access BCs through this method.
            Subclasses may override or cache boundary conditions as attributes.

        Resolution Order:
            1. Instance attribute `_boundary_conditions` (cached BCs)
            2. `geometry.boundary_conditions` (attribute access)
            3. `geometry.get_boundary_conditions()` (method accessor)
            4. `problem.boundary_conditions` (direct on problem)
            5. `problem.get_boundary_conditions()` (method on problem)
        """
        # Priority 1: Check for instance attribute (solvers may cache BCs)
        try:
            if self._boundary_conditions is not None:
                return self._boundary_conditions
        except AttributeError:
            pass

        # Priority 2: geometry.boundary_conditions (direct attribute)
        try:
            bc = self.problem.geometry.boundary_conditions
            if bc is not None:
                return bc
        except AttributeError:
            pass

        # Priority 3: geometry.get_boundary_conditions() (method accessor)
        try:
            bc = self.problem.geometry.get_boundary_conditions()
            if bc is not None:
                return bc
        except AttributeError:
            pass

        # Priority 4: problem.boundary_conditions (direct attribute)
        try:
            bc = self.problem.boundary_conditions
            if bc is not None:
                return bc
        except AttributeError:
            pass

        # Priority 5: problem.get_boundary_conditions() (method accessor)
        try:
            bc = self.problem.get_boundary_conditions()
            if bc is not None:
                return bc
        except AttributeError:
            pass

        # No BC found
        return None


class BaseNumericalSolver(BaseMFGSolver):
    """Base class for numerical methods (FDM, FEM, spectral, etc.)."""

    # Subclasses should override with their discretization type
    discretization_type: str = "FDM"

    def __init__(self, problem: MFGProblem, config: BaseConfig) -> None:
        super().__init__(problem, config)
        self.convergence_history: list[float] = []

    def _detect_dimension(self) -> int:
        """
        Detect spatial dimension from geometry (unified interface).

        Issue #633: Centralized dimension detection for all numerical solvers.

        Returns
        -------
        int
            Problem dimension (1, 2, 3, ...)

        Raises
        ------
        ValueError
            If dimension cannot be determined from problem.
        """
        # Primary: Use geometry.dimension (standard for all modern problems)
        try:
            return self.problem.geometry.dimension
        except AttributeError:
            pass

        # Fallback: problem.dimension attribute
        try:
            return self.problem.dimension
        except AttributeError:
            pass

        raise ValueError(
            "Cannot determine problem dimension. "
            "Ensure problem has 'geometry' with 'dimension' attribute or 'dimension' property."
        )

    @abstractmethod
    def discretize(self) -> None:
        """Set up the spatial and temporal discretization."""

    def apply_boundary_conditions(
        self,
        field: NDArray[np.floating],
        time: float = 0.0,
    ) -> NDArray[np.floating]:
        """
        Apply boundary conditions to a field using unified dispatch.

        This method provides a standard way for numerical solvers to apply BCs.
        It delegates to the geometry/boundary/dispatch.py infrastructure.

        Args:
            field: Field array to apply BCs to (interior values)
            time: Time for time-dependent BCs (default: 0.0)

        Returns:
            Field with BCs applied (may be padded for FDM methods)

        Note:
            Solvers can override this method for custom BC handling,
            but using the unified dispatch is recommended.
        """
        bc = self.get_boundary_conditions()
        if bc is None:
            return field  # No BCs to apply

        from mfgarchon.geometry.boundary.dispatch import apply_bc

        return apply_bc(
            geometry=self.problem.geometry,
            field=field,
            boundary_conditions=bc,
            time=time,
            discretization=self.discretization_type,
        )

    def get_convergence_info(self) -> dict[str, Any]:
        """Get convergence information."""
        return {
            "iterations": len(self.convergence_history),
            "final_error": self.convergence_history[-1] if self.convergence_history else None,
            "convergence_rate": self._estimate_convergence_rate(),
        }

    def _estimate_convergence_rate(self) -> float | None:
        """Estimate convergence rate from history."""
        if len(self.convergence_history) < 3:
            return None

        errors = self.convergence_history[-3:]
        if errors[-1] == 0 or errors[-2] == 0:
            return None

        return abs(errors[-1] / errors[-2])


class BaseOptimizationSolver(BaseMFGSolver):
    """Base class for optimization methods (variational, optimal transport, etc.)."""

    def __init__(self, problem: MFGProblem, config: BaseConfig) -> None:
        super().__init__(problem, config)
        self.objective_history: list[float] = []

    @abstractmethod
    def compute_objective(self, variables: Any) -> float:
        """Compute the objective function value."""

    @abstractmethod
    def compute_gradient(self, variables: Any) -> Any:
        """Compute the gradient of the objective function."""

    def get_domain_constraints(self) -> list[dict[str, Any]]:
        """
        Generate optimization constraints from boundary conditions.

        Translates BoundaryConditions into constraint dictionaries that can
        be used by optimization solvers (scipy.optimize, cvxpy, etc.).

        Returns:
            List of constraint dictionaries with keys:
            - type: "eq" (equality), "ineq" (inequality), "grad" (gradient)
            - bc_type: Original BC type (Dirichlet, Neumann, etc.)
            - region: Boundary region identifier
            - value: Target value or function

        Example usage with scipy:
            constraints = solver.get_domain_constraints()
            for c in constraints:
                if c["type"] == "eq":
                    # Add equality constraint u(boundary) = value
                    scipy_constraints.append({"type": "eq", "fun": ...})
        """
        bc = self.get_boundary_conditions()
        if bc is None:
            return []

        from mfgarchon.geometry.boundary import BCType

        constraints = []

        # Handle segments if available
        segments = getattr(bc, "segments", [])
        for segment in segments:
            bc_type = segment.bc_type
            constraint = {
                "region": getattr(segment, "region", "boundary"),
                "value": getattr(segment, "value", 0.0),
                "bc_type": bc_type,
            }

            if bc_type == BCType.DIRICHLET:
                constraint["type"] = "eq"  # u(x_b) = g
            elif bc_type in (BCType.NEUMANN, BCType.NO_FLUX):
                constraint["type"] = "grad"  # du/dn = g
            elif bc_type == BCType.ROBIN:
                constraint["type"] = "mixed"  # alpha*u + beta*du/dn = g
            else:
                constraint["type"] = "other"

            constraints.append(constraint)

        # If no segments, create constraint from default BC.
        # Issue #1100: a unified BoundaryConditions with default_bc unset (None)
        # is fully unspecified here -> fail loud rather than silently apply a default.
        if not constraints:
            from mfgarchon.geometry.boundary import BoundaryConditions

            if isinstance(bc, BoundaryConditions):
                bc_type = bc._resolve_default_bc("BaseOptimizationSolver.get_domain_constraints")
                constraint = {
                    "region": "all",
                    "value": getattr(bc, "value", 0.0),
                    "bc_type": bc_type,
                    "type": "eq" if bc_type == BCType.DIRICHLET else "grad",
                }
                constraints.append(constraint)
            # else: legacy/non-unified BC object carries no default constraint

        return constraints


class BaseNeuralSolver(BaseMFGSolver):
    """Base class for neural network methods (PINN, neural operators, etc.)."""

    def __init__(self, problem: MFGProblem, config: BaseConfig) -> None:
        super().__init__(problem, config)
        self.training_history: dict[str, list[float]] = {}

    @abstractmethod
    def build_networks(self) -> None:
        """Build the neural network architectures."""

    @abstractmethod
    def compute_loss(self, *args: Any, **kwargs: Any) -> dict[str, float]:
        """Compute the total loss and its components."""

    @abstractmethod
    def train_step(self) -> dict[str, float]:
        """Perform one training step."""

    def sample_boundary_points(self, n_points: int) -> NDArray[np.floating]:
        """
        Sample points on domain boundary for BC loss computation.

        Args:
            n_points: Number of boundary points to sample

        Returns:
            Array of shape (n_points, dimension) with boundary coordinates

        Note:
            Uses geometry.sample_boundary_points() if available,
            otherwise falls back to uniform sampling on boundary faces.
        """
        geometry = self.problem.geometry

        # Try geometry method first
        try:
            return geometry.sample_boundary_points(n_points)
        except AttributeError:
            pass

        # Fallback: sample on domain boundary
        bounds = geometry.get_bounds()
        if bounds is None:
            raise ValueError("Cannot sample boundary: geometry has no bounds")

        min_coords, max_coords = bounds
        dim = geometry.dimension

        # Sample uniformly on boundary faces
        points_per_face = n_points // (2 * dim)
        boundary_points = []

        for d in range(dim):
            for is_max in [False, True]:
                face_points = np.random.uniform(min_coords, max_coords, size=(points_per_face, dim))
                # Fix one coordinate to boundary
                face_points[:, d] = max_coords[d] if is_max else min_coords[d]
                boundary_points.append(face_points)

        return np.vstack(boundary_points)

    def get_boundary_target_values(
        self,
        boundary_points: NDArray[np.floating],
        time: float = 0.0,
    ) -> NDArray[np.floating] | None:
        """
        Get target BC values at given boundary points.

        Args:
            boundary_points: Points on domain boundary
            time: Time for time-dependent BCs

        Returns:
            Target values at boundary points, or None if BC has no explicit values
        """
        bc = self.get_boundary_conditions()
        if bc is None:
            return None

        # Try to get values from BoundaryConditions
        try:
            return bc.get_value_at_points(boundary_points, time)
        except (AttributeError, NotImplementedError):
            pass

        # Try scalar value
        try:
            value = bc.value
            if callable(value):
                return np.array([value(p, time) for p in boundary_points])
            return np.full(len(boundary_points), value)
        except AttributeError:
            return None


class BaseRLSolver(BaseMFGSolver):
    """Base class for reinforcement learning methods."""

    def __init__(self, problem: MFGProblem, config: BaseConfig) -> None:
        super().__init__(problem, config)
        self.training_metrics: dict[str, list[float]] = {}
        self.population_size: int | None = getattr(config, "population_size", None)

    @abstractmethod
    def create_environment(self) -> Any:
        """Create the MFG environment for RL agents."""

    @abstractmethod
    def create_agents(self) -> Any:
        """Create the RL agents."""

    @abstractmethod
    def train_agents(self) -> dict[str, float]:
        """Train the RL agents to reach Nash equilibrium."""

    def evaluate_nash_gap(self) -> float:
        """Evaluate the Nash equilibrium gap."""
        # Default implementation - should be overridden
        return 0.0

    def scale_to_mean_field(self) -> Any:
        """Convert finite-population solution to mean field limit."""
        if self.population_size is None or self.population_size == float("inf"):
            return self.solution
        # Default implementation - should be overridden by specific solvers
        return self.solution

    def get_environment_boundary_config(self) -> dict[str, Any]:
        """
        Get environment boundary configuration from problem BCs.

        Returns a config dict that environments can use to handle boundaries:
        - bounds: Domain bounds (low, high)
        - boundary_mode: How to handle boundary crossings
            - "wrap": Periodic (wrap around)
            - "reflect": Elastic bounce (reflecting)
            - "clip": Clip to bounds (default)
            - "absorb": Episode ends at boundary

        Returns:
            Dictionary with boundary configuration for environment setup
        """
        config: dict[str, Any] = {}

        # Get domain bounds
        bounds = self.problem.geometry.get_bounds()
        if bounds is not None:
            min_coords, max_coords = bounds
            config["bounds"] = {
                "low": min_coords,
                "high": max_coords,
            }

        # Map BC type to environment behavior
        bc = self.get_boundary_conditions()
        if bc is not None:
            from mfgarchon.geometry.boundary import BCType, BoundaryConditions

            # Issue #1100: a unified BoundaryConditions with default_bc unset (None)
            # would previously map to "wrap" (silent PERIODIC). Fail loud instead.
            if isinstance(bc, BoundaryConditions):
                bc_type = bc._resolve_default_bc("BaseRLSolver.get_environment_boundary_config")
            else:
                bc_type = getattr(bc, "default_bc", None)

            if bc_type == BCType.PERIODIC:
                config["boundary_mode"] = "wrap"
            elif bc_type in (BCType.REFLECTING, BCType.NO_FLUX, BCType.NEUMANN):
                config["boundary_mode"] = "reflect"
            elif bc_type == BCType.DIRICHLET:
                config["boundary_mode"] = "absorb"
            else:
                config["boundary_mode"] = "clip"
        else:
            config["boundary_mode"] = "clip"

        return config
