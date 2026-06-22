from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

    import numpy as np

    from mfgarchon.core.mfg_problem import MFGProblem
    from mfgarchon.types.solver_types import SolverReturnTuple
else:
    pass


class BaseCouplingIterator(ABC):
    """
    Abstract base class for iterative coupling solvers (Picard, block, fictitious play).

    Provides the interface for MFG solvers that iterate between HJB and FP
    sub-solvers to solve the coupled system. Distinguished from
    ``alg.base_solver.BaseMFGSolver`` which is the cross-paradigm base.

    Subclasses should call ``_init_solver_signatures(hjb_solver, fp_solver)``
    in ``__init__`` to enable ``_build_hjb_kwargs`` / ``_build_fp_kwargs``.
    """

    def __init__(self, problem: MFGProblem) -> None:
        """
        Initialize the MFG solver with a problem definition.

        Args:
            problem: The MFG problem to solve
        """
        self.problem = problem
        self.warm_start_data: dict[str, Any] | None = None
        self._solution_computed: bool = False
        self._hjb_sig_params: set[str] | None = None
        self._fp_sig_params: set[str] | None = None
        self._hjb_solver_name: str = "<hjb solver>"
        self._fp_solver_name: str = "<fp solver>"

    def _init_solver_signatures(self, hjb_solver: Any, fp_solver: Any) -> None:
        """Cache solver method signatures for conditional parameter passing.

        Call this in subclass ``__init__`` after storing solver references.
        Replaces per-iterator ``_cache_solver_signatures`` methods.
        """
        self._hjb_solver_name = type(hjb_solver).__name__
        self._fp_solver_name = type(fp_solver).__name__
        try:
            sig = inspect.signature(hjb_solver.solve_hjb_system)
            self._hjb_sig_params = set(sig.parameters.keys())
        except (AttributeError, ValueError):
            self._hjb_sig_params = None
        try:
            sig = inspect.signature(fp_solver.solve_fp_system)
            self._fp_sig_params = set(sig.parameters.keys())
        except (AttributeError, ValueError):
            self._fp_sig_params = None

    def _build_hjb_kwargs(
        self,
        *,
        volatility_field: float | np.ndarray | Any | None = None,
        source_term: Callable | None = None,
    ) -> dict[str, Any]:
        """Build kwargs for solve_hjb_system, respecting solver capabilities.

        Progress is handled automatically via context routing (Issue #934) —
        solver's ``create_progress_bar`` detects the parent ``HierarchicalProgress``.
        """
        kwargs: dict[str, Any] = {}
        params = self._hjb_sig_params
        if params is None:
            return kwargs
        if "volatility_field" in params and volatility_field is not None:
            kwargs["volatility_field"] = volatility_field
        if source_term is not None:
            if "source_term" not in params:
                raise NotImplementedError(
                    f"{self._hjb_solver_name}.solve_hjb_system does not accept 'source_term', but the "
                    f"problem defines a source / nonlocal / obstacle term (composed into a non-None HJB "
                    f"source). Silently dropping it would solve the wrong problem (Issue #1424). Use an "
                    f"FDM HJB solver, or remove source_term_hjb / nonlocal_operator / obstacle from the "
                    f"problem."
                )
            kwargs["source_term"] = source_term
        return kwargs

    def _build_fp_kwargs(
        self,
        *,
        drift_field: np.ndarray | Callable | Any | None = None,
        volatility_field: float | np.ndarray | Any | None = None,
        source_term: Callable | None = None,
    ) -> dict[str, Any]:
        """Build kwargs for solve_fp_system, respecting solver capabilities.

        Progress is handled automatically via context routing (Issue #934).
        """
        kwargs: dict[str, Any] = {}
        params = self._fp_sig_params
        if params is None:
            return kwargs
        if "drift_field" in params and drift_field is not None:
            kwargs["drift_field"] = drift_field
        if "volatility_field" in params and volatility_field is not None:
            kwargs["volatility_field"] = volatility_field
        if source_term is not None:
            if "source_term" not in params:
                raise NotImplementedError(
                    f"{self._fp_solver_name}.solve_fp_system does not accept 'source_term', but the "
                    f"problem defines an FP source term (composed into a non-None FP source). Silently "
                    f"dropping it would solve the wrong problem (Issue #1424). Use an FDM FP solver, or "
                    f"remove source_term_fp from the problem."
                )
            kwargs["source_term"] = source_term
        return kwargs

    @abstractmethod
    def solve(self, max_iterations: int, tolerance: float = 1e-5, **kwargs: Any) -> SolverReturnTuple:
        """
        Solve the coupled MFG system.

        Args:
            max_iterations: Maximum number of iterations
            tolerance: Convergence tolerance
            **kwargs: Additional solver-specific parameters

        Returns:
            Tuple of (U, M, convergence_info) where:
            - U: Hamilton-Jacobi-Bellman solution array
            - M: Fokker-Planck density array
            - convergence_info: Dictionary with convergence details
        """

    @abstractmethod
    def get_results(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Get the computed solution arrays.

        Returns:
            Tuple of (U, M) solution arrays
        """

    def set_warm_start_data(
        self,
        previous_solution: tuple[np.ndarray, np.ndarray],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Set warm start data from a previous solution.

        Args:
            previous_solution: Tuple of (U, M) arrays from previous solve
            metadata: Optional metadata about the previous solution
        """
        U_prev, M_prev = previous_solution

        # Validate dimensions with enhanced error messages
        from mfgarchon.utils.exceptions import validate_array_dimensions

        # Get expected shape from geometry
        spatial_shape = tuple(self.problem.geometry.get_grid_shape())
        expected_shape = (self.problem.Nt + 1, *spatial_shape)

        try:
            validate_array_dimensions(
                U_prev,
                expected_shape=expected_shape,
                array_name="warm_start_U",
            )
            validate_array_dimensions(
                M_prev,
                expected_shape=expected_shape,
                array_name="warm_start_M",
            )
        except Exception as e:
            raise ValueError(f"Invalid warm start data dimensions: {e}") from e

        self.warm_start_data = {
            "U_prev": U_prev.copy(),
            "M_prev": M_prev.copy(),
            "metadata": metadata or {},
        }

    def get_warm_start_data(self) -> tuple[np.ndarray, np.ndarray] | None:
        """
        Get warm start data if available.

        Returns:
            Tuple of (U, M) arrays if warm start data exists, None otherwise
        """
        if self.warm_start_data is None:
            return None
        return self.warm_start_data["U_prev"], self.warm_start_data["M_prev"]

    def clear_warm_start_data(self) -> None:
        """Clear any stored warm start data."""
        self.warm_start_data = None

    @property
    def has_warm_start_data(self) -> bool:
        """Check if warm start data is available."""
        return self.warm_start_data is not None

    @property
    def is_solved(self) -> bool:
        """Check if the solver has computed a solution."""
        return self._solution_computed


if __name__ == "__main__":
    """Quick smoke test for development."""
    print("Testing BaseCouplingIterator...")

    # Test base class availability
    assert BaseCouplingIterator is not None
    print("  BaseCouplingIterator class available")

    # Note: BaseCouplingIterator is abstract and requires implementation
    # See FixedPointMFGSolver for concrete implementation

    print("Smoke tests passed!")
