from __future__ import annotations

import inspect
import numbers
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

    import numpy as np

    from mfgarchon.alg.numerical.fp_solvers.base_fp import DriftConvention
    from mfgarchon.core.mfg_problem import MFGProblem
    from mfgarchon.types.solver_types import SolverReturnTuple
else:
    pass


def assert_bc_providers_resolvable(problem: MFGProblem, iterator_name: str) -> None:
    """Fail loud if ``problem`` carries a dynamic BC provider this coupling loop cannot resolve.

    RFC #1574 / Issue #1563: only ``FixedPointIterator`` resolves a ``BCValueProvider`` (e.g.
    ``AdjointConsistentProvider``) stored in a ``BCSegment.value`` -- it calls
    ``problem.using_resolved_bc(state)`` each Picard step (fixed_point_iterator.py). The other
    coupling loops do not, so a provider would otherwise reach the solver unresolved: a deep GFDM
    row-builder ``ValueError``, or a silent miss on a non-Robin provider segment. Raise up front
    (naming the loop) instead of surfacing the failure deep in the solver, or not at all.
    """
    # A problem without grid BC resolution (a network problem, or a lightweight test stub) cannot
    # carry a grid BCValueProvider, so there is nothing to guard -- skip rather than AttributeError.
    _get_bc = getattr(problem, "get_boundary_conditions", None)
    if not callable(_get_bc):
        return
    if _get_bc().has_providers():
        raise NotImplementedError(
            f"{iterator_name} does not resolve dynamic BC providers (a BCValueProvider stored in a "
            f"BCSegment.value, e.g. AdjointConsistentProvider). Only FixedPointIterator resolves them "
            f"per iteration (Issue #625/#1563). Use FixedPointIterator for a provider-based "
            f"(adjoint-consistent) boundary condition, or a statically-valued BC with this loop."
        )


def assert_paired_solver_sigma(hjb_solver: Any, fp_solver: Any, context: str) -> None:
    """Fail loud if a coupled HJB / FP solver pair was built from problems with different ``sigma``.

    Issue #1603 / #1081 / RFC #1574 (C14): each paired solver reads sigma from its OWN embedded
    problem (``hjb_solver.problem.sigma``, ``fp_solver.problem.sigma``). A coupled HJB-FP pair is an
    adjoint pair and must share the volatility; if the two problems' sigma disagree, HJB and FP
    diffuse at different rates with no warning and the fixed point is neither problem's MFG.

    Extracted from ``FixedPointIterator`` (Issue #1603) to a single owner so EVERY coupling loop --
    FixedPoint, Block, FictitiousPlay, Newton, and the regime / multi-population / graph lists (which
    had no guard) -- shares one check. For a list-based iterator, call once per sub-problem pair
    (naming the sub-problem in ``context``). Compares only real scalars, so Mock test doubles whose
    auto-resolved ``.sigma`` is not a number do not trip the guard (#1489).
    """
    hjb_sigma = getattr(getattr(hjb_solver, "problem", None), "sigma", None)
    fp_sigma = getattr(getattr(fp_solver, "problem", None), "sigma", None)
    if (
        isinstance(hjb_sigma, (int, float))
        and isinstance(fp_sigma, (int, float))
        and abs(float(hjb_sigma) - float(fp_sigma)) > 1e-12
    ):
        raise ValueError(
            f"{context}: paired HJB / FP solvers were built from problems with different sigma "
            f"(HJB={hjb_sigma}, FP={fp_sigma}); a coupled MFG pair is an adjoint pair and must share "
            f"the volatility -- the Picard fixed point would correspond to neither problem. Build both "
            f"solvers from the same MFGProblem, or use create_paired_solvers. Issue #1603."
        )


def matches_problem_sigma(problem: Any, volatility_field: Any) -> bool:
    """Whether this field is indistinguishable from the problem's own sigma.

    Scalars only. An array or callable cannot be shown equivalent to sigma without evaluating it,
    and guessing there is what #1316 was about, so those are never called indistinguishable.
    ``numbers.Real`` rather than ``(int, float)``: ``np.float32(0.3)`` is neither, and rejecting it
    would refuse a solve that is byte-identical to one it accepts.
    """
    sigma = getattr(problem, "sigma", None)
    if not isinstance(volatility_field, numbers.Real) or not isinstance(sigma, numbers.Real):
        return False
    return abs(float(volatility_field) - float(sigma)) <= 1e-12


def resolve_volatility_kwarg(
    params: Any, volatility_field: Any, problem: Any, solver_name: str, method: str, side: str
) -> dict[str, Any]:
    """The ``volatility_field`` kwarg for ``method``, or ``{}`` when there is nothing to forward.

    One owner for all four coupling call sites -- both Picard sides (:class:`BaseCouplingIterator`)
    and both Newton sides (:class:`~.mfg_residual.MFGResidual`). Issue #1783 was filed against two
    of them, and a review of the first fix found the same silent drop still live on the other two.
    The four sites each restated the same three-way decision, and the restatement is what let them
    disagree; a fifth site restating it is the next instance, so the decision lives here only.

    Three outcomes, in this order:

    - **The solver names the parameter: forward it, unconditionally.** Including when the field is
      indistinguishable from ``problem.sigma`` -- it is still the caller's explicit value, and
      ``problem.volatility_field`` is not always ``problem.sigma``. Construct with an array sigma
      and the field is the array while ``problem.sigma`` is its mean, so a solver that falls back
      through ``get_diffusion_coefficient_field(None)`` would pick up the array. Declining to
      forward an "equivalent" scalar hands the solver a different field than the one asked for.
      The first fix of #1783 put the forward inside the hazard branch and did exactly that.
    - **The solver does not name it, and the field is a hazard: raise.** Signature introspection
      answers "does this callable name the parameter", not "can this solver consume it", and a
      ``**kwargs`` override makes those two diverge. Measured on the meshless pair, whose HJB
      wrapper delegates through ``(*args, use_newton=None, **kwargs)``: with ``problem.sigma = 0.3``
      and a field of mean 0.7, the HJB side ran at D = 0.045 while the paired FP side -- which does
      name the parameter -- ran at D = 0.245. A 5.4x mismatch, no warning, and a converged density
      for a problem nobody posed. Treating ``VAR_KEYWORD`` as accept-anything was the other
      candidate; it assumes a solver consumes what it swallows, the assumption that produced #1316.
    - **The solver does not name it, and the field is indistinguishable: drop it silently**, because
      dropping it changes nothing. ``MFGProblem.volatility_field`` defaults to ``problem.sigma``, so
      the coupling loop hands a non-None value on EVERY ordinary solve; refusing those is a refusal
      to run at all, which the full suite caught two tests deep in the meshless recipe when the
      first version of this fix keyed on "is not None".
    """
    if volatility_field is None:
        return {}
    if "volatility_field" in params:
        return {"volatility_field": volatility_field}
    if matches_problem_sigma(problem, volatility_field):
        return {}
    other = "FP" if side == "HJB" else "HJB"
    raise NotImplementedError(
        f"{solver_name}.{method} does not accept 'volatility_field', but a volatility_field was "
        f"supplied. Its signature is ({', '.join(sorted(params))}). Dropping it would leave the "
        f"{side} side on problem.sigma while the {other} side uses the field, so the two equations "
        f"would be solved with different diffusion and the result would be neither problem "
        f"(Issue #1783). Either declare volatility_field on the solver's {method}, or remove it "
        f"from the solve. A solver taking **kwargs does not count as accepting it -- the parameter "
        f"must be named."
    )


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
        # Issue #1489 (S1): the FP solver's declared drift-input convention, cached beside its
        # signature so resolve_fp_drift_kwargs can route by convention rather than param presence.
        self._fp_drift_convention: DriftConvention | None = None
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
        # Issue #1489 (S1): cache the FP solver's drift-input convention for resolve_fp_drift_kwargs.
        self._fp_drift_convention = getattr(fp_solver, "_drift_convention", None)

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
        kwargs.update(
            resolve_volatility_kwarg(
                params,
                volatility_field,
                self.problem,
                self._hjb_solver_name,
                "solve_hjb_system",
                "HJB",
            )
        )
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
        kwargs.update(
            resolve_volatility_kwarg(
                params,
                volatility_field,
                self.problem,
                self._fp_solver_name,
                "solve_fp_system",
                "FP",
            )
        )
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
