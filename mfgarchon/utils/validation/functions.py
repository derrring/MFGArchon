"""
Validation for custom MFG functions (Hamiltonian, drift, running_cost).

This module validates user-provided mathematical functions to catch
errors before they propagate into solver iterations.

Issue #686: Custom function validation (Hamiltonian, drift, running_cost)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from mfgarchon.utils.exceptions import MFGSolverError
from mfgarchon.utils.validation.protocol import ValidationResult

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from mfgarchon.geometry.protocol import GeometryProtocol

# --- Finite-difference consistency checking (Issue #1642, capability C1) -----
#
# The consistency check is allowed to *gate* (set is_valid False, so the caller's
# `raise ValidationError(result)` fires), but only on an exhibited witness: a
# concrete (x, m, p) where the derivative identity measurably fails by a margin
# no finite-difference artifact can produce. Anything weaker stays a warning,
# because a numerical check that raises on a false positive is worse than one
# that warns.

_FD_STEP = 1e-6

# Warning threshold: relative discrepancy above this is reported, not gated.
_WARN_RTOL = 1e-4

# Witness threshold: a wrong derivative (sign flip, missing term, wrong factor)
# is O(1) relative. 1e-2 sits two orders above the warning tier and ~8 orders
# above the measured relative noise of this stencil on the shipped Hamiltonians.
_WITNESS_RTOL = 1e-2

# Central differencing has roundoff error ~ eps_mach * |H| / step. A witness must
# clear that bound by this factor before it can be called a defect rather than noise.
_WITNESS_NOISE_SAFETY = 1e4

# One-sided slopes of a smooth H agree to O(step); at a kink they disagree by O(1).
# Above this relative disagreement the derivative does not exist at the probe, so
# no witness can be exhibited there (L1 and bounded control costs both have kinks).
_SMOOTHNESS_RTOL = 1e-3

#: What a Hamiltonian, a method or a door raises when it cannot be evaluated at a probe: a wrong shape or call, a kink
#: its own minimiser refuses (#2644's ConvergenceError is a RuntimeError), a value out of its domain. A comparison that
#: meets one is warned about and never gates (#2901); anything else is a bug and propagates.
_EVALUATION_ERRORS: tuple[type[BaseException], ...] = (
    ArithmeticError,
    AttributeError,
    IndexError,
    KeyError,
    NotImplementedError,
    RuntimeError,
    TypeError,
    ValueError,
    MFGSolverError,
)


def _raw_callable_advice(fn: Any) -> str:
    """What a raw callable `_call_h` could not call is told, which keeps what a partial froze. A family object is told
    how it was called, and not a cause: its TypeError may come from its methods, a body it holds, its own signature or
    the binder's refusal of a frozen slot."""
    import functools

    from mfgarchon.core.hamiltonian import MFGOperatorBase
    from mfgarchon.types.callable_protocols import HAMILTONIAN_SLOTS, keeping_what_it_froze

    target = fn.func if isinstance(fn, functools.partial) else fn
    if isinstance(getattr(target, "__self__", target), MFGOperatorBase):
        return (
            "It is called by keyword, with t, x, p and m at one point, x a tuple of numbers and the density a number."
        )

    named = f"Name each parameter of {getattr(fn, '__qualname__', fn)!r} after what it reads, from {', '.join(HAMILTONIAN_SLOTS.order)}"
    return f"{keeping_what_it_froze(fn, named)}."


def _call_h(fn: Any, x: Any, m: Any, p: Any, t: float = 0.0) -> Any:
    """Evaluate a Hamiltonian, or one of its derivatives, the way its kind expects.

    A family object (a ``HamiltonianBase``/``LagrangianBase`` instance, or a bound method of
    one) takes ``(t, x, p, m)`` since #2378 phase 5 and is called by keyword. A raw user callable
    goes through its binding (``HAMILTONIAN_SLOTS``), like every user callable the library accepts.
    """
    import functools

    from mfgarchon.core.hamiltonian import MFGOperatorBase
    from mfgarchon.types.callable_protocols import HAMILTONIAN_SLOTS, _remembered, point_tuple, refuse_overridden

    x = point_tuple(x)
    target = fn.func if isinstance(fn, functools.partial) else fn  # partial(H.dm) is still a family method
    if isinstance(getattr(target, "__self__", target), MFGOperatorBase):
        refuse_overridden(fn, HAMILTONIAN_SLOTS.order, role="Hamiltonian callable")
        return fn(t=t, x=x, p=p, m=m)
    return _remembered(fn, HAMILTONIAN_SLOTS, "Hamiltonian callable")(t=t, x=x, p=p, m=m)


def _get_sample_inputs(
    geometry: GeometryProtocol,
    location: str,
) -> tuple[np.ndarray, np.ndarray, float, int, ValidationResult | None]:
    """Extract a sample point (x, p, m) from geometry for validation.

    Returns:
        (x_sample, p_sample, m_sample, dimension, error_result)
        If error_result is not None, the caller should return it immediately.
    """
    try:
        grid = geometry.get_spatial_grid()
        if isinstance(grid, np.ndarray) and grid.ndim == 2:
            # (N, d) array from TensorProductGrid, ImplicitDomain, etc.
            mid = grid.shape[0] // 2
            x_sample = grid[mid]  # shape (d,)
            dimension = grid.shape[1]
        elif isinstance(grid, np.ndarray) and grid.ndim == 1:
            # 1D flat grid
            x_sample = grid[len(grid) // 2]
            dimension = 1
        elif isinstance(grid, (list, tuple)):
            # Legacy: tuple of 1D arrays (meshgrid-style)
            x_sample = np.array([g[len(g) // 2] for g in grid])
            dimension = len(grid)
        else:
            x_sample = np.atleast_1d(grid)[len(np.atleast_1d(grid)) // 2]
            dimension = 1
    except Exception as e:
        result = ValidationResult()
        result.add_error(
            f"Could not get sample point from geometry: {e}",
            location=location,
        )
        return np.array([]), np.array([]), 1.0, 1, result

    x_sample = np.atleast_1d(x_sample).astype(float)
    p_sample = np.zeros(dimension, dtype=float)
    m_sample = 1.0
    return x_sample, p_sample, m_sample, dimension, None


def validate_custom_functions(
    hamiltonian: Any | None,
    dH_dm: Callable | None,
    dH_dp: Callable | None,
    geometry: GeometryProtocol,
    *,
    check_consistency: bool = True,
    optimal_control: Callable | None = None,
) -> ValidationResult:
    """
    Validate all custom Hamiltonian-related functions.

    Supports HamiltonianBase instances (preferred) and raw callables.

    Args:
        hamiltonian: HamiltonianBase instance (called as H(t, x, p, m) by keyword) or a raw
            callable H(t, x, p, m), bound by name or position (#2375 ruling 8)
        dH_dm: Derivative dH/dm: a family bound method (called by keyword) or a raw callable(t, x, p, m)
        dH_dp: Derivative dH/dp: a family bound method (called by keyword) or a raw callable(t, x, p, m)
        geometry: Geometry for sample point generation
        check_consistency: If True (default), verify dH_dm/dH_dp are the
            derivatives of H by finite differences. Costs O(dimension) extra
            scalar H evaluations; see validate_hamiltonian_consistency for when
            that check gates versus warns.
        optimal_control: The Hamiltonian's optimal control, a family bound method or a raw callable(t, x, p, m).
            With check_consistency, it is checked to be -dH/dp where H is smooth (#2901).

    Returns:
        ValidationResult with any issues found

    Issue #686: Custom function validation
    Issue #1642 (C1): consistency findings propagate to is_valid
    """
    results: list[ValidationResult] = []

    if hamiltonian is not None:
        results.append(validate_hamiltonian(hamiltonian, geometry))

    if dH_dm is not None:
        results.append(validate_hamiltonian_derivative(dH_dm, geometry, "dH_dm"))

    if dH_dp is not None:
        results.append(validate_hamiltonian_derivative(dH_dp, geometry, "dH_dp"))

    # Check consistency if requested and all functions provided
    if check_consistency and hamiltonian is not None and dH_dm is not None:
        results.append(
            validate_hamiltonian_consistency(hamiltonian, dH_dm, geometry, dH_dp=dH_dp, optimal_control=optimal_control)
        )

    # ValidationResult.combine is the one implementation of this merge: issues
    # concatenated, contexts merged, is_valid False if any part is False.
    return ValidationResult.combine(results)


def validate_hamiltonian(
    hamiltonian: Any,
    geometry: GeometryProtocol,
) -> ValidationResult:
    """
    Validate a Hamiltonian: a HamiltonianBase instance, or a raw callable of (t, x, p, m), which declares p and what else it reads.

    Supports HamiltonianBase instances (called as H(t, x, p, m) by keyword)
    and raw callables (tried with same signature).

    Checks:
    - Callable with correct signature
    - Returns float or array
    - No NaN/Inf in output

    Args:
        hamiltonian: HamiltonianBase instance or callable
        geometry: Geometry for sample point

    Returns:
        ValidationResult
    """
    result = ValidationResult()

    x_sample, p_sample, m_sample, _dim, err = _get_sample_inputs(geometry, "hamiltonian")
    if err is not None:
        return err

    # Evaluate: a HamiltonianBase by keyword; a raw callable through its binding, (t, x, p, m)
    try:
        value = _call_h(hamiltonian, x_sample, m_sample, p_sample)
    except TypeError as e:
        result.add_error(
            f"Hamiltonian has wrong signature: {e}",
            location="hamiltonian",
            suggestion=f"A Hamiltonian takes (t, x, p, m) (#2375 ruling 8). {_raw_callable_advice(hamiltonian)}",
        )
        return result
    except Exception as e:
        result.add_warning(
            f"Hamiltonian raised exception at sample point: {e}",
            location="hamiltonian",
        )
        return result

    # Check return type
    if not isinstance(value, (int, float, np.integer, np.floating, np.ndarray)):
        result.add_error(
            f"Hamiltonian must return float or ndarray, got {type(value).__name__}",
            location="hamiltonian",
        )
        return result

    # Check for NaN/Inf
    if np.isscalar(value):
        if not np.isfinite(value):
            result.add_error(
                "Hamiltonian returned NaN or Inf at sample point",
                location="hamiltonian",
            )
    elif isinstance(value, np.ndarray) and not np.all(np.isfinite(value)):
        result.add_error(
            "Hamiltonian returned array with NaN or Inf values",
            location="hamiltonian",
        )

    return result


def validate_hamiltonian_derivative(
    derivative_func: Callable,
    geometry: GeometryProtocol,
    name: str,
) -> ValidationResult:
    """
    Validate a Hamiltonian derivative function (dH_dm or dH_dp).

    The derivative should have signature f(t, x, p, m) matching
    HamiltonianBase.dm() / HamiltonianBase.dp().

    Args:
        derivative_func: Derivative function (bound method or callable)
        geometry: Geometry for sample point
        name: Name for error messages ("dH_dm" or "dH_dp")

    Returns:
        ValidationResult
    """
    result = ValidationResult()

    x_sample, p_sample, m_sample, _dim, err = _get_sample_inputs(geometry, name)
    if err is not None:
        return err

    # Evaluate: a family bound method by keyword; a raw callable through its binding, (t, x, p, m)
    try:
        value = _call_h(derivative_func, x_sample, m_sample, p_sample)
    except TypeError as e:
        result.add_error(
            f"{name} has wrong signature: {e}",
            location=name,
            suggestion=f"{name} takes (t, x, p, m) (#2375 ruling 8). {_raw_callable_advice(derivative_func)}",
        )
        return result
    except Exception as e:
        result.add_warning(
            f"{name} raised exception at sample point: {e}",
            location=name,
        )
        return result

    # Check return type
    if not isinstance(value, (int, float, np.integer, np.floating, np.ndarray)):
        result.add_error(
            f"{name} must return float or ndarray, got {type(value).__name__}",
            location=name,
        )

    return result


@dataclass(frozen=True)
class _FDComparison:
    """One finite-difference vs analytical comparison at one probe point."""

    m: float
    p: np.ndarray
    numerical: float
    analytical: float
    diff: float
    noise_bound: float
    smooth: bool
    finite: bool
    warn_rtol: float

    @property
    def is_inconsistent(self) -> bool:
        """Warning grade: relative discrepancy above the reporting tolerance.

        Also required to clear the differencing roundoff bound. Without that, a
        correct Hamiltonian of large magnitude (a big potential term) reports a
        discrepancy the stencil cannot resolve at this step size.
        """
        if not self.finite:
            return True
        if self.diff <= self.noise_bound:
            return False
        return self.diff / max(abs(self.analytical), 1e-10) > self.warn_rtol

    @property
    def is_witness(self) -> bool:
        """Error grade: a discrepancy no finite-difference artifact can produce.

        All three conditions are required. Smoothness rules out a kink (where the
        derivative simply does not exist); the noise bound rules out roundoff; the
        relative margin rules out a merely imprecise analytical derivative.
        """
        if not (self.finite and self.smooth):
            return False
        return self.diff > _WITNESS_NOISE_SAFETY * self.noise_bound and self.diff > _WITNESS_RTOL * max(
            abs(self.numerical), abs(self.analytical)
        )


def _compare_derivative(
    h_minus: float,
    h_center: float,
    h_plus: float,
    step: float,
    analytical: float,
    m: float,
    p: np.ndarray,
    warn_rtol: float,
) -> _FDComparison:
    """Compare a central difference of H against a claimed analytical derivative."""
    numerical = (h_plus - h_minus) / (2.0 * step)
    forward = (h_plus - h_center) / step
    backward = (h_center - h_minus) / step

    finite = bool(np.isfinite([h_minus, h_center, h_plus, analytical]).all())
    smooth = finite and abs(forward - backward) <= _SMOOTHNESS_RTOL * max(abs(forward), abs(backward), 1.0)
    noise_bound = float(np.finfo(float).eps) * max(abs(h_minus), abs(h_center), abs(h_plus), 1.0) / step

    return _FDComparison(
        m=m,
        p=p.copy(),
        numerical=numerical,
        analytical=analytical,
        diff=abs(numerical - analytical) if finite else float("inf"),
        noise_bound=noise_bound,
        smooth=smooth,
        finite=finite,
        warn_rtol=warn_rtol,
    )


def _consistency_probes(dimension: int) -> Iterator[tuple[float, np.ndarray]]:
    """Probe points (m, p) for the derivative-consistency search.

    Two densities because a wrong dH_dm can still be right at a single m (a user
    writing a constant where the derivative is affine). Three momenta because
    dH_dp of any Hamiltonian even in p vanishes at p=0, and CongestionHamiltonian's
    dH_dm is proportional to |p|^2 and therefore also vanishes there — probing only
    p=0 makes both checks structurally blind.

    The nonzero magnitudes are deliberately not round numbers. L1ControlCost has a
    kink at |p| = control_cost and BoundedControlCost at |p| = control_cost *
    max_control, and users pick round values for those; a probe landing exactly on
    a kink produces a true-but-useless warning. The smoothness guard in
    _FDComparison.is_witness is the backstop that keeps such a probe from gating.
    """
    for m in (0.5, 1.0):
        for p_magnitude in (0.0, 0.4157, 1.3691):
            yield m, np.full(dimension, p_magnitude, dtype=float)


def _report(
    result: ValidationResult,
    location: str,
    quantity: str,
    x_sample: np.ndarray,
    witness: _FDComparison | None,
    inconsistent: _FDComparison | None,
) -> None:
    """Emit at most one issue for `location`, at the highest grade observed."""
    worst = witness if witness is not None else inconsistent
    if worst is None:
        return

    point = f"x={np.array2string(x_sample, precision=4)}, m={worst.m:.4g}, p={np.array2string(worst.p, precision=4)}"

    if not worst.finite:
        result.add_warning(
            f"{quantity} consistency could not be evaluated at {point}: "
            f"non-finite values (analytical={worst.analytical}, finite-difference={worst.numerical})",
            location=location,
            suggestion=f"Ensure H and {quantity} are finite on the problem's admissible (m, p) range",
        )
        return

    detail = (
        f"at {point}: analytical={worst.analytical:.8g}, "
        f"finite-difference={worst.numerical:.8g} (step={_FD_STEP:g}), "
        f"|difference|={worst.diff:.6g}"
    )
    result.context[f"{location}_numerical"] = worst.numerical
    result.context[f"{location}_analytical"] = worst.analytical
    result.context[f"{location}_witness_m"] = worst.m
    result.context[f"{location}_witness_p"] = worst.p.tolist()

    if witness is not None:
        result.add_error(
            f"{quantity} is not the derivative of H {detail}. This exceeds "
            f"{_WITNESS_RTOL:g} relative and {_WITNESS_NOISE_SAFETY:g}x the "
            f"finite-difference roundoff bound {worst.noise_bound:.3g}, and H is "
            f"smooth at this point, so the discrepancy is not a differencing artifact.",
            location=location,
            suggestion=f"Correct {quantity} so it equals the derivative of H, or correct H",
        )
    elif not worst.smooth:
        result.add_warning(
            f"{quantity} may be inconsistent with H {detail}. H is not "
            f"differentiable at this probe (its one-sided slopes disagree), so the "
            f"central difference does not approximate a derivative here.",
            location=location,
            suggestion=f"Ignore if {quantity} is a subgradient at a kink of H; otherwise verify it",
        )
    else:
        result.add_warning(
            f"{quantity} may be inconsistent with H {detail}. Relative discrepancy "
            f"exceeds {worst.warn_rtol:g} but is not large enough to rule out a "
            f"finite-difference artifact.",
            location=location,
            suggestion=f"Verify {quantity} is the correct derivative of H",
        )


def _report_control(
    result: ValidationResult,
    i: int,
    x_sample: np.ndarray,
    witness: tuple[_FDComparison, float] | None,
    inconsistent: tuple[_FDComparison, float] | None,
) -> None:
    """Emit at most one issue for the optimal control's component i, at the highest grade observed (#2901).

    The comparison's analytical value is -optimal_control[i], against the same central difference of H as dp[i]'s. The
    message names the pair that disagrees: the control against dp and H when dp agrees with H at that probe, and the
    control against H otherwise, where dp's own issue names dp."""
    worst = witness if witness is not None else inconsistent
    if worst is None:
        return
    comparison, dp_i = worst
    location = f"optimal_control[{i}]"
    point = f"x={np.array2string(x_sample, precision=4)}, m={comparison.m:.4g}, p={np.array2string(comparison.p, precision=4)}"
    if not comparison.finite:
        result.add_warning(
            f"optimal_control[{i}] could not be compared with dH/dp at {point}: non-finite values "
            f"(-optimal_control={comparison.analytical}, finite-difference={comparison.numerical})",
            location=location,
        )
        return
    dp_agrees = bool(np.isfinite(dp_i)) and abs(dp_i - comparison.numerical) <= max(
        _WITNESS_RTOL * max(abs(dp_i), abs(comparison.numerical)), _WITNESS_NOISE_SAFETY * comparison.noise_bound
    )
    pair = "optimal_control disagrees with dp and with H" if dp_agrees else "optimal_control disagrees with H"
    detail = (
        f"at {point}: -optimal_control[{i}]={comparison.analytical:.8g}, dp[{i}]={dp_i:.8g}, and the central "
        f"difference of H is {comparison.numerical:.8g} (step={_FD_STEP:g})"
    )
    result.context[f"{location}_witness_m"] = comparison.m
    result.context[f"{location}_witness_p"] = comparison.p.tolist()
    if witness is not None:
        result.add_error(
            f"{pair} {detail}. Where H is smooth, as it is here, its optimal control is -dH/dp. The solve reads H and dp "
            f"in its HJB residual and Jacobian and the optimal control in its Fokker-Planck drift, so it would read "
            f"two Hamiltonians (#2901).",
            location=location,
            suggestion="Make optimal_control return -dp, or override neither: HamiltonianBase derives optimal_control "
            "from dp, and dp from H",
        )
    elif not comparison.smooth:
        result.add_warning(
            f"optimal_control may disagree with H {detail}. H is not differentiable at this probe (its one-sided "
            f"slopes disagree), and a non-smooth cost's control is one selection of -dH/dp there.",
            location=location,
        )
    else:
        result.add_warning(
            f"optimal_control may disagree with H {detail}. The relative gap exceeds {comparison.warn_rtol:g} but is "
            f"too small to rule out a finite-difference artifact.",
            location=location,
        )


#: The doors the library's readers call, each with the method it stands for (#2901).
_DOORS = (
    ("evaluate_H", "__call__"),
    ("evaluate_dp", "dp"),
    ("evaluate_optimal_control", "optimal_control"),
    ("dH_dm", "dm"),
)


def _compare_doors_with_methods(
    hamiltonian: Any, x_sample: np.ndarray, dimension: int, tolerance: float, result: ValidationResult
) -> None:
    """Each door of a ``HamiltonianBase`` against its method, at the consistency probes and one population (#2901).

    The library's solvers read the doors, and a door a subclass overrides is a source of its own: one that returns
    another value than its method makes the solve read two Hamiltonians. A Hamiltonian that does not read a batch
    (``reads_a_batch`` False) is handed one point, the others a batch of one, and the method is called at exactly the
    door's inputs, so a point-only Hamiltonian is left to the batch check. A gap is an error above the witness margin
    and a warning above ``tolerance``, both relative to the larger of the two values and above roundoff; a door that
    cannot be evaluated is warned about, never gating."""
    from mfgarchon.core.hamiltonian import HamiltonianBase, HEvalState, _state_x

    if not isinstance(hamiltonian, HamiltonianBase):
        return
    worst: dict[str, tuple[int, float, float, np.ndarray, np.ndarray, np.ndarray]] = {}
    failures: dict[str, str] = {}
    for m, p in _consistency_probes(dimension):
        if hamiltonian.reads_a_batch:
            state = HEvalState(x=x_sample[np.newaxis], p=p[np.newaxis], m=np.array([m]), t=0.0)
        else:
            state = HEvalState(x=x_sample, p=p, m=m, t=0.0)
        for door_name, method_name in _DOORS:
            if door_name in failures:
                continue
            try:
                door = np.ravel(np.asarray(getattr(hamiltonian, door_name)(state), dtype=float))
                method = hamiltonian if method_name == "__call__" else getattr(hamiltonian, method_name)
                # The method at exactly the door's inputs, so the comparison is the door's alone: how the Hamiltonian
                # reads a batch against a point is `refuse_a_hamiltonian_whose_reads_disagree`'s fact.
                value = np.ravel(np.asarray(method(t=state.t, x=_state_x(state), p=state.p, m=state.m), dtype=float))
            except _EVALUATION_ERRORS as e:  # reported as a warning, never gating
                failures[door_name] = f"{type(e).__name__}: {e}"
                continue
            if door.shape != value.shape:
                grade, gap, scale = 2, float("inf"), 0.0
            else:
                gap = float(np.max(np.abs(door - value), initial=0.0))
                scale = float(max(np.max(np.abs(door), initial=0.0), np.max(np.abs(value), initial=0.0)))
                above_noise = gap > _WITNESS_NOISE_SAFETY * float(np.finfo(float).eps) * max(scale, 1.0)
                grade = (
                    2
                    if above_noise and gap > _WITNESS_RTOL * scale
                    else 1
                    if above_noise and gap > tolerance * scale
                    else 0
                )
            if grade and grade > worst.get(door_name, (0,))[0]:
                worst[door_name] = (grade, m, gap, p.copy(), door, value)
    method_of = dict(_DOORS)
    for door_name, (grade, m, _gap, p, door, value) in worst.items():
        method_name = method_of[door_name]
        detail = (
            f"at x={np.array2string(x_sample, precision=4)}, m={m:.4g}, p={np.array2string(p, precision=4)}: "
            f"{door_name} returns {np.array2string(door, precision=8)} and {method_name} returns "
            f"{np.array2string(value, precision=8)}"
        )
        if grade == 2:
            result.add_error(
                f"{door_name} and {method_name} disagree {detail}. The library's solvers read {door_name}, so a solve "
                f"would read a Hamiltonian other than the one {method_name} describes (#2901).",
                location=door_name,
                suggestion=f"Make {door_name} return what {method_name} returns at one point, or override neither",
            )
        else:
            result.add_warning(f"{door_name} may disagree with {method_name} {detail}.", location=door_name)
    for door_name, failure in failures.items():
        result.add_warning(f"Could not compare {door_name} with {method_of[door_name]}: {failure}", location=door_name)


def _compare_network_rates_with_dp(hamiltonian: Any, result: ValidationResult) -> None:
    """A ``NetworkHamiltonian``'s two sources against each other, on its own probes (#1474, #2901).

    Its optimal control is the transition rates alpha_ij out of node i, and its dp is dH/du over the nodes: dp[j] is
    -alpha_ij for each other node j, and dp[i] is the sum of the rates. The grid probes do not apply, since a node plays
    x and p is u at every node. Checked at every node, at three costate vectors whose entries are distinct, so no edge
    sits on its kink u_i = u_j, with the uniform density."""
    try:
        n = int(hamiltonian.num_nodes)
    except _EVALUATION_ERRORS as e:  # reported as a warning, never gating
        result.add_warning(
            f"Could not compare the network's rates with dp: {type(e).__name__}: {e}", location="optimal_control"
        )
        return
    k = np.arange(n, dtype=float)
    costates = (np.linspace(0.0, 2.0, n), np.sqrt(k + 1.0), np.sin(1.3 * k + 0.2))
    density = np.full(n, 1.0 / n)
    worst: tuple[float, int, np.ndarray, np.ndarray, np.ndarray] | None = None
    try:
        for u in costates:
            for i in range(n):
                rates = np.ravel(
                    np.asarray(hamiltonian.optimal_control(t=0.0, x=np.array([i]), p=u, m=density), dtype=float)
                )
                grad = np.ravel(np.asarray(hamiltonian.dp(t=0.0, x=np.array([i]), p=u, m=density), dtype=float))
                expected = -rates.copy()
                expected[i] = float(np.sum(rates) - rates[i])
                gap = float(np.max(np.abs(grad - expected), initial=0.0))
                scale = float(max(np.max(np.abs(grad), initial=0.0), np.max(np.abs(expected), initial=0.0)))
                above_noise = gap > _WITNESS_NOISE_SAFETY * float(np.finfo(float).eps) * max(scale, 1.0)
                if above_noise and gap > _WITNESS_RTOL * scale and (worst is None or gap > worst[0]):
                    worst = (gap, i, u.copy(), grad, expected)
    except _EVALUATION_ERRORS as e:  # reported as a warning, never gating
        result.add_warning(
            f"Could not compare the network's rates with dp: {type(e).__name__}: {e}", location="optimal_control"
        )
        return
    if worst is not None:
        gap, i, u, grad, expected = worst
        result.add_error(
            f"optimal_control and dp disagree on the network at node {i}, u={np.array2string(u, precision=4)}, uniform "
            f"density: dp returns {np.array2string(grad, precision=6)}, while the rates optimal_control returns give "
            f"{np.array2string(expected, precision=6)} (dp[j] = -alpha_ij for each other node j, dp[{i}] their sum). The "
            f"policy-iteration HJB reads the rates and dp's readers read dp, so a solve would read two Hamiltonians "
            f"(#2901).",
            location="optimal_control",
            suggestion="Make dp the gradient of the Hamiltonian whose rates optimal_control returns, or override neither",
        )


def validate_hamiltonian_consistency(
    hamiltonian: Any,
    dH_dm: Callable,
    geometry: GeometryProtocol,
    tolerance: float = _WARN_RTOL,
    dH_dp: Callable | None = None,
    optimal_control: Callable | None = None,
) -> ValidationResult:
    """
    Check that dH_dm and dH_dp are the derivatives of H, and that the optimal control is -dH/dp, by finite differences.

    Numerical checks, over a small grid of probe points (m, p):
        dH_dm_numerical = (H(t, x, p, m+eps) - H(t, x, p, m-eps)) / (2*eps)
        dH_dp_numerical[i] = (H(t, x, p+eps*e_i, m) - H(t, x, p-eps*e_i, m)) / (2*eps)

    Severity (Issue #1642, capability C1). This validator can invalidate its
    result, so a caller doing `if not result.is_valid: raise ValidationError(...)`
    will fire. It does so only on an **exhibited witness** — a probe point where
    H is smooth, all values are finite, and the discrepancy clears both a relative
    margin and the differencing roundoff bound by a wide factor. Every weaker
    signal (small discrepancy, kink at the probe, non-finite value, failure to
    evaluate) is a warning, because a false positive here would refuse a correct
    problem at construction time.

    Args:
        hamiltonian: HamiltonianBase instance (called as H(t, x, p, m) by keyword) or a raw
            callable H(t, x, p, m), bound by name or position (#2375 ruling 8)
        dH_dm: Claimed derivative dH/dm: a family bound method (called by keyword) or a raw callable(t, x, p, m)
        geometry: Geometry for sample point
        tolerance: Relative tolerance for the warning tier
        dH_dp: Claimed gradient dH/dp, the same kinds as dH_dm. Optional.
        optimal_control: Claimed optimal control, the same kinds. Optional. Checked to be -dH/dp at the same probes,
            with the same witness rule, so a non-smooth cost is compared only where H is smooth at the probe (#2901).

    A ``HamiltonianBase`` also has each door (``evaluate_H``, ``evaluate_dp``, ``evaluate_optimal_control``,
    ``dH_dm``) compared with its method at the probes, at one population (#2901). A ``NetworkHamiltonian`` is checked
    on its own probes instead, with its own identity between the rates it returns and ``dp`` (#2901).

    Returns:
        ValidationResult; is_valid is False when a witness was exhibited.
    """
    from mfgarchon.extensions.topology import NetworkHamiltonian

    result = ValidationResult()
    if isinstance(hamiltonian, NetworkHamiltonian):
        _compare_network_rates_with_dp(hamiltonian, result)
        return result

    x_sample, _p_sample, _m_sample, dimension, err = _get_sample_inputs(geometry, "hamiltonian")
    if err is not None:
        return err

    step = _FD_STEP
    dm_witness: _FDComparison | None = None
    dm_inconsistent: _FDComparison | None = None
    dp_witness: list[_FDComparison | None] = [None] * dimension
    dp_inconsistent: list[_FDComparison | None] = [None] * dimension
    # Each with the dp the Hamiltonian returned at that probe, for the message.
    control_witness: list[tuple[_FDComparison, float] | None] = [None] * dimension
    control_inconsistent: list[tuple[_FDComparison, float] | None] = [None] * dimension
    control_failure: str | None = None

    try:
        for m, p in _consistency_probes(dimension):
            h_center = float(_call_h(hamiltonian, x_sample, m, p))

            # --- dH_dm ---
            h_m_plus = float(_call_h(hamiltonian, x_sample, m + step, p))
            h_m_minus = float(_call_h(hamiltonian, x_sample, m - step, p))
            comparison = _compare_derivative(
                h_m_minus,
                h_center,
                h_m_plus,
                step,
                float(_call_h(dH_dm, x_sample, m, p)),
                m,
                p,
                tolerance,
            )
            if comparison.is_witness and dm_witness is None:
                dm_witness = comparison
            if comparison.is_inconsistent and dm_inconsistent is None:
                dm_inconsistent = comparison

            # --- dH_dp and the optimal control, per component ---
            if dH_dp is None and optimal_control is None:
                continue
            dp_analytical = None if dH_dp is None else np.atleast_1d(_call_h(dH_dp, x_sample, m, p)).astype(float)
            control = None
            if optimal_control is not None and control_failure is None:
                # Its own guard: a control that cannot be evaluated is warned about and leaves the dp check standing.
                try:
                    control = np.atleast_1d(_call_h(optimal_control, x_sample, m, p)).astype(float)
                except _EVALUATION_ERRORS as e:  # reported as a warning, never gating
                    control_failure = f"{type(e).__name__}: {e}"
            for i in range(dimension):
                p_plus = p.copy()
                p_minus = p.copy()
                p_plus[i] += step
                p_minus[i] -= step
                h_minus = float(_call_h(hamiltonian, x_sample, m, p_minus))
                h_plus = float(_call_h(hamiltonian, x_sample, m, p_plus))
                if dp_analytical is not None:
                    comparison = _compare_derivative(
                        h_minus, h_center, h_plus, step, float(dp_analytical[i]), m, p, tolerance
                    )
                    if comparison.is_witness and dp_witness[i] is None:
                        dp_witness[i] = comparison
                    if comparison.is_inconsistent and dp_inconsistent[i] is None:
                        dp_inconsistent[i] = comparison
                if control is not None:
                    dp_i = float("nan") if dp_analytical is None else float(dp_analytical[i])
                    comparison = _compare_derivative(
                        h_minus, h_center, h_plus, step, -float(control[i]), m, p, tolerance
                    )
                    if comparison.is_witness and control_witness[i] is None:
                        control_witness[i] = (comparison, dp_i)
                    if comparison.is_inconsistent and control_inconsistent[i] is None:
                        control_inconsistent[i] = (comparison, dp_i)

    except Exception as e:
        # Evaluation failed, so no witness can be exhibited: warn, never gate.
        result.add_warning(
            f"Could not verify Hamiltonian consistency: {e}",
            location="hamiltonian",
        )
        return result

    _report(result, "dH_dm", "dH_dm", x_sample, dm_witness, dm_inconsistent)
    for i in range(dimension):
        _report(result, f"dH_dp[{i}]", f"dH_dp[{i}]", x_sample, dp_witness[i], dp_inconsistent[i])
    if control_failure is not None:
        result.add_warning(
            f"Could not compare the optimal control with dH/dp: {control_failure}", location="optimal_control"
        )
    for i in range(dimension):
        _report_control(result, i, x_sample, control_witness[i], control_inconsistent[i])
    _compare_doors_with_methods(hamiltonian, x_sample, dimension, tolerance, result)

    return result
