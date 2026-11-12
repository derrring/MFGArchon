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
    from collections.abc import Callable, Iterator, Sequence

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


def _is_family_callable(fn: Any) -> bool:
    """Whether ``fn`` is a family object, a ``HamiltonianBase``/``LagrangianBase`` instance or a bound method of one or
    of its class (a classmethod), or a ``functools.partial`` over one: called by keyword, as the library's doors call
    it, not through a raw callable's binding (#2901)."""
    import functools

    from mfgarchon.core.hamiltonian import MFGOperatorBase

    target = fn.func if isinstance(fn, functools.partial) else fn
    owner = getattr(target, "__self__", target)
    return isinstance(owner, MFGOperatorBase) or (isinstance(owner, type) and issubclass(owner, MFGOperatorBase))


def _raw_callable_advice(fn: Any) -> str:
    """What a raw callable `_call_h` could not call is told, which keeps what a partial froze. A family object is told
    how it was called, and not a cause: its TypeError may come from its methods, a body it holds, its own signature or
    the binder's refusal of a frozen slot."""
    from mfgarchon.types.callable_protocols import HAMILTONIAN_SLOTS, keeping_what_it_froze

    if _is_family_callable(fn):
        return (
            "It is called by keyword, with t, x, p and m at one point, x a tuple of numbers and the density a number."
        )

    named = f"Name each parameter of {getattr(fn, '__qualname__', fn)!r} after what it reads, from {', '.join(HAMILTONIAN_SLOTS.order)}"
    return f"{keeping_what_it_froze(fn, named)}."


def _call_h(fn: Any, x: Any, m: Any, p: Any, t: float = 0.0) -> Any:
    """Evaluate a Hamiltonian, or one of its derivatives, the way its kind expects.

    A family object (`_is_family_callable`: a ``HamiltonianBase``/``LagrangianBase`` instance, or a
    bound method of one or of its class) takes ``(t, x, p, m)`` since #2378 phase 5 and is called by
    keyword. A raw user callable goes through its binding (``HAMILTONIAN_SLOTS``), like every user
    callable the library accepts.
    """
    from mfgarchon.types.callable_protocols import HAMILTONIAN_SLOTS, _remembered, point_tuple, refuse_overridden

    x = point_tuple(x)
    if _is_family_callable(fn):  # partial(H.dm) is still a family method
        refuse_overridden(fn, HAMILTONIAN_SLOTS.order, role="Hamiltonian callable")
        return fn(t=t, x=x, p=p, m=m)
    return _remembered(fn, HAMILTONIAN_SLOTS, "Hamiltonian callable")(t=t, x=x, p=p, m=m)


class SourceShapeError(Exception):
    """A source returned a value of another size than the one read: it breaks its contract wherever it does so, so a
    check refuses it at any probe (#2901). Not one of ``_EVALUATION_ERRORS``, which skip a probe outside a source's
    domain, and not a ``ValueError``, so no guard that skips a probe takes it."""


def _read_values(value: Any, *, size: int, name: str, points: int = 1) -> np.ndarray:
    """A source's value as ``points * size`` floats in one row: the reader's one conversion (#2901).

    A value of one entry stands for every point when one number is read per point, as a constant does on a batch. Any
    other size raises ``SourceShapeError``, naming the source, the shape it returned and the count read."""
    values = np.asarray(value, dtype=float).reshape(-1)
    if values.size == points * size:
        return values
    if values.size == 1 and size == 1:
        return np.full(points, values[0])
    read = ("1 value is" if size == 1 else f"{size} values are") if points == 1 else f"{points} points of {size} are"
    raise SourceShapeError(f"{name} returned shape {np.shape(value)} where {read} read")


def _read_source(
    source: Any, *, t: float, x: Any, p: Any, m: Any, size: int, name: str, form: str = "point", points: int = 1
) -> np.ndarray:
    """Read one of a Hamiltonian's sources the way its readers do, then convert it through `_read_values`: the one
    reader of every check that compares a Hamiltonian's sources (#2901).

    ``form`` is how the source is called:

    - ``"point"``: a method or a callable at one point of a grid, bound by its declared names (`_call_h`);
    - ``"node"``: a ``NetworkHamiltonian``'s method by keyword, ``x`` as its network readers hand it;
    - ``"door"``: a door, handed ``HEvalState(x, p, m, t)``;
    - ``"potential"``: the potential a ``SeparableHamiltonian`` ``source`` holds, at ``x`` and ``t``.

    An evaluation error propagates to the caller's guard, and so does a ``SourceShapeError``."""
    if form == "point":
        value = _call_h(source, x, m, p, t)
    elif form == "node":
        value = source(t=t, x=x, p=p, m=m)
    elif form == "door":
        from mfgarchon.core.hamiltonian import HEvalState

        value = source(HEvalState(x=x, p=p, m=m, t=t))
    elif form == "potential":
        from mfgarchon.core.hamiltonian import _bound_potential
        from mfgarchon.types.callable_protocols import point_tuple

        value = _bound_potential(source)(x=point_tuple(x), t=t)
    else:
        raise ValueError(f"_read_source: unknown form {form!r}")
    return _read_values(value, size=size, name=name, points=points)


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
    times: Sequence[float] = (0.0,),
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
        times: The consistency probes' times; a problem hands 0, T/2 and T (#2901).

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
            validate_hamiltonian_consistency(
                hamiltonian, dH_dm, geometry, dH_dp=dH_dp, optimal_control=optimal_control, times=times
            )
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
    t: float = 0.0
    x: np.ndarray | None = None

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
    t: float = 0.0,
    x: np.ndarray | None = None,
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
        t=float(t),
        x=None if x is None else np.array(x, dtype=float),
    )


def _grid_nodes(geometry: GeometryProtocol, dimension: int) -> np.ndarray | None:
    """The geometry's nodes as an (N, d) array, or None when its grid is not an array of nodes."""
    grid = geometry.get_spatial_grid()
    if isinstance(grid, np.ndarray) and grid.ndim == 2 and grid.shape[1] == dimension:
        return np.asarray(grid, dtype=float)
    if isinstance(grid, np.ndarray) and grid.ndim == 1 and dimension == 1:
        return np.asarray(grid, dtype=float).reshape(-1, 1)
    return None


def _probe_points(geometry: GeometryProtocol, middle: np.ndarray) -> list[np.ndarray]:
    """The x the consistency probes read: the node nearest the centre of the nodes' bounding box (the middle node when
    every axis has an odd number of nodes), then along each axis the two end nodes nearest it in the other coordinates,
    so at most 2d + 1 nodes, all inside the domain (#2901). A geometry whose grid is not an array of nodes is probed at
    the sample point alone."""
    nodes = _grid_nodes(geometry, middle.size)
    if nodes is None:
        return [middle]
    centre = 0.5 * (nodes.min(axis=0) + nodes.max(axis=0))
    middle = nodes[int(np.argmin(np.sum((nodes - centre) ** 2, axis=1)))]
    points = [middle]
    for k in range(nodes.shape[1]):
        for end in (nodes[:, k].min(), nodes[:, k].max()):
            on_end = nodes[nodes[:, k] == end]
            nearest = on_end[int(np.argmin(np.sum((on_end - middle) ** 2, axis=1)))]
            if not any(np.array_equal(nearest, q) for q in points):
                points.append(nearest)
    return points


def _consistency_probes(dimension: int) -> Iterator[tuple[float, np.ndarray]]:
    """Probe points (m, p) for the derivative-consistency search.

    Two densities because a wrong dH_dm can still be right at a single m (a user
    writing a constant where the derivative is affine). Three momenta because
    dH_dp of any Hamiltonian even in p vanishes at p=0, and CongestionHamiltonian's
    dH_dm is proportional to |p|^2 and therefore also vanishes there — probing only
    p=0 makes both checks structurally blind.

    Component k of each momentum is a scale times (-1)**k (1 + 0.3 k), and the two nonzero scales have opposite signs,
    so every component is probed at both signs and at a magnitude of its own: a control that is right only for p >= 0,
    or only where two components are equal, is seen (#2901).

    The nonzero magnitudes are deliberately not round numbers. L1ControlCost has a
    kink at |p| = control_cost and BoundedControlCost at |p| = control_cost *
    max_control, and users pick round values for those; a probe landing exactly on
    a kink produces a true-but-useless warning. In d = 1 to 3 every nonzero probe's |p| is at least 4.6% from 0.5, 1,
    2, 3, 4 and 5. The smoothness guard in _FDComparison.is_witness is the backstop that keeps such a probe from gating.
    """
    pattern = np.array([(-1.0) ** k * (1.0 + 0.3 * k) for k in range(dimension)])
    for m in (0.5, 1.0):
        for p_scale in (0.0, 0.4157, -1.3691):
            yield m, p_scale * pattern


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

    where = x_sample if worst.x is None else worst.x
    point = (
        f"t={worst.t:.4g}, x={np.array2string(where, precision=4)}, m={worst.m:.4g}, "
        f"p={np.array2string(worst.p, precision=4)}"
    )

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
    result.context[f"{location}_witness_t"] = worst.t
    result.context[f"{location}_witness_x"] = where.tolist()

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
    where = x_sample if comparison.x is None else comparison.x
    point = (
        f"t={comparison.t:.4g}, x={np.array2string(where, precision=4)}, m={comparison.m:.4g}, "
        f"p={np.array2string(comparison.p, precision=4)}"
    )
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
    result.context[f"{location}_witness_t"] = comparison.t
    result.context[f"{location}_witness_x"] = where.tolist()
    if witness is not None:
        result.add_error(
            f"{pair} {detail}. Where H is smooth, as it is here, its optimal control is -dH/dp. The solve reads H and dp "
            f"in its HJB residual and Jacobian and the optimal control in its Fokker-Planck drift, so it would read "
            f"two Hamiltonians (#2901).",
            location=location,
            suggestion="Make optimal_control return -dp. A HamiltonianBase subclass that overrides only __call__ gets "
            "dp and optimal_control derived from it; a SeparableHamiltonian's dp and optimal_control are its control "
            "cost's, so a subclass that redefines __call__ must redefine both",
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


def _probe_text(t: float, x: np.ndarray, m: float, p: np.ndarray) -> str:
    """A probe (t, x, m, p) as the reports write it."""
    return f"t={t:.4g}, x={np.array2string(np.asarray(x), precision=4)}, m={m:.4g}, p={np.array2string(p, precision=4)}"


def _refuse_shapes(result: ValidationResult, shapes: dict[str, str]) -> None:
    """Each source that returned a value of another size than the one read, refused at the first probe it did so: its
    contract is broken there, whatever the other probes give (#2901)."""
    for name, where in shapes.items():
        result.add_error(
            f"{where}. A source that returns another number of values than its readers read breaks its contract, so a "
            f"solve would read another value in its place or fail (#2901).",
            location=name,
            suggestion=f"Make {name} return one value per point, and a vector of the dimension for dp and the control",
        )


def _compare_doors_with_methods(
    hamiltonian: Any,
    probes_x: Sequence[tuple[float, Sequence[np.ndarray]]],
    dimension: int,
    tolerance: float,
    result: ValidationResult,
) -> None:
    """Each door of a ``HamiltonianBase`` against its method, at the consistency probes and one population (#2901).

    The library's solvers read the doors, and a door a subclass overrides is a source of its own: one that returns
    another value than its method makes the solve read two Hamiltonians. ``probes_x`` gives each time with its points.
    Both are read through `_read_source`: the method at the point, as D2 reads it, and the door as the solvers hand it,
    one point to a Hamiltonian that does not read a batch (``reads_a_batch`` False) and a batch of one to the others. A
    gap is an error above the witness margin and a warning above ``tolerance``, both relative to the larger of the two
    values and above roundoff. Each read has its own guard: a probe at which the method cannot be evaluated is skipped
    for that door, one at which the door alone cannot is warned about and the door still compared at the others, and a
    door that raises at every probe at which its method could be evaluated is an error, whatever ``reads_a_batch`` is.
    A method or door that returns another number of values than is read is an error at any probe (#2901)."""
    from mfgarchon.core.hamiltonian import HamiltonianBase

    if not isinstance(hamiltonian, HamiltonianBase):
        return
    worst: dict[str, tuple[int, float, np.ndarray, float, np.ndarray, np.ndarray, np.ndarray]] = {}
    failures: dict[str, str] = {}
    shapes: dict[str, str] = {}
    # Per door: the probes at which its method could be evaluated, and those at which the door then raised.
    method_ok = dict.fromkeys(dict(_DOORS), 0)
    door_failed = dict.fromkeys(dict(_DOORS), 0)
    sizes = {"evaluate_H": 1, "evaluate_dp": dimension, "evaluate_optimal_control": dimension, "dH_dm": 1}
    probes = [(t, x, m, p) for t, points in probes_x for x in points for m, p in _consistency_probes(dimension)]
    for t, x, m, p in probes:
        t = float(t)
        batch = hamiltonian.reads_a_batch
        door_at = {"x": x[np.newaxis], "p": p[np.newaxis], "m": np.array([m])} if batch else {"x": x, "p": p, "m": m}
        for door_name, method_name in _DOORS:
            method = hamiltonian if method_name == "__call__" else getattr(hamiltonian, method_name)
            size = sizes[door_name]
            try:
                value = _read_source(method, t=t, x=x, p=p, m=m, size=size, name=method_name)
            except SourceShapeError as e:
                shapes.setdefault(method_name, f"{e}, at {_probe_text(t, x, m, p)}")
                continue
            except _EVALUATION_ERRORS:  # D2 reports the method; this door is not compared here
                continue
            method_ok[door_name] += 1
            try:
                door = _read_source(
                    getattr(hamiltonian, door_name), t=t, size=size, name=door_name, form="door", **door_at
                )
            except SourceShapeError as e:
                shapes.setdefault(door_name, f"{e}, at {_probe_text(t, x, m, p)}")
                continue
            except _EVALUATION_ERRORS as e:  # warned about; the other probes still compare
                door_failed[door_name] += 1
                failures.setdefault(door_name, f"at {_probe_text(t, x, m, p)}: {type(e).__name__}: {e}")
                continue
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
                worst[door_name] = (grade, t, x, m, p.copy(), door, value)
    _refuse_shapes(result, shapes)
    method_of = dict(_DOORS)
    for door_name, (grade, t, x, m, p, door, value) in worst.items():
        method_name = method_of[door_name]
        detail = (
            f"at t={t:.4g}, x={np.array2string(x, precision=4)}, m={m:.4g}, p={np.array2string(p, precision=4)}: "
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
        if door_failed[door_name] == method_ok[door_name]:
            result.add_error(
                f"{door_name} could not be evaluated at any probe at which {method_of[door_name]} could "
                f"({method_ok[door_name]} of them), first {failure}. The library's solvers read {door_name}, so a "
                f"solve would fail there (#2901).",
                location=door_name,
                suggestion=f"Make {door_name} evaluate where {method_of[door_name]} does, or do not override it",
            )
        else:
            result.add_warning(
                f"Could not compare {door_name} with {method_of[door_name]} {failure} ({door_failed[door_name]} of "
                f"{method_ok[door_name]} probes); the probes that could be evaluated were compared",
                location=door_name,
            )


def _refuse_unread(result: ValidationResult, failed: dict[str, int], attempted: int, failures: dict[str, str]) -> None:
    """A network read that raised at every node, time and costate it was read at is an error; one that raised at some,
    or a read beside a probe that raised, is a warning, the others having been compared (#2901)."""
    for name in failures:
        count = failed.get(name, 0)
        if count == attempted:
            result.add_error(
                f"The network's {name} could not be evaluated at any of the {attempted} probes it was read at, first "
                f"{failures[name]}. Its readers read it, so a solve would fail there (#2901).",
                location=_NETWORK_LOCATION[name],
                suggestion=f"Make {name} evaluate at every node",
            )
        else:
            result.add_warning(
                f"Could not evaluate the network's {name} {failures[name]} ({count} of {attempted} probes); the probes "
                "that could be evaluated were compared",
                location=_NETWORK_LOCATION[name],
            )


#: Where a network read's issue is reported.
_NETWORK_LOCATION = {"optimal_control": "optimal_control", "dp": "dH_dp", "__call__": "hamiltonian", "dm": "dH_dm"}


def _compare_network_rates_with_dp(
    hamiltonian: Any, result: ValidationResult, times: Sequence[float], n: int, nodes: Sequence[int]
) -> None:
    """A ``NetworkHamiltonian``'s two sources against each other, on its own probes (#1474, #2901).

    Its optimal control is the transition rates alpha_ij out of node i, and its dp is dH/du over the nodes: dp[j] is
    -alpha_ij for each other node j, and dp[i] is the sum of the rates. The grid probes do not apply, since a node plays
    x and p is u at every node. Checked at ``nodes`` (`_network_probe_nodes`), each as x, at three costate vectors whose
    entries are distinct, so no edge sits on its kink u_i = u_j, with the uniform density, at each of ``times``. A
    disagreement at a node not probed is not seen. Both are read through `_read_source` with ``x = np.array([i])``, as
    policy iteration and the network FP hand ``optimal_control``, each in its own guard: a read that fails at some
    probes is warned about and the others compared, one that fails at every probe is an error, and one that returns
    another number of values than the n nodes is an error at any probe (#2901)."""
    k = np.arange(n, dtype=float)
    costates = (np.linspace(0.0, 2.0, n), np.sqrt(k + 1.0), np.sin(1.3 * k + 0.2))
    density = np.full(n, 1.0 / n)
    worst: tuple[float, int, np.ndarray, np.ndarray, np.ndarray, float] | None = None
    failures: dict[str, str] = {}
    failed = {"optimal_control": 0, "dp": 0}
    shapes: dict[str, str] = {}
    attempted = 0
    for t in (float(s) for s in times):
        for u in costates:
            for i in nodes:
                attempted += 1
                node = np.array([i])
                read: dict[str, np.ndarray] = {}
                for name in ("optimal_control", "dp"):
                    try:
                        read[name] = _read_source(
                            getattr(hamiltonian, name), t=t, x=node, p=u, m=density, size=n, name=name, form="node"
                        )
                    except SourceShapeError as e:
                        shapes.setdefault(_NETWORK_LOCATION[name], f"{e}, at t={t:.4g}, node {i}")
                    except _EVALUATION_ERRORS as e:  # this read is skipped here; the other probes still compare
                        failed[name] += 1
                        failures.setdefault(name, f"at t={t:.4g}, node {i}: {type(e).__name__}: {e}")
                if len(read) < 2:
                    continue
                rates, grad = read["optimal_control"], read["dp"]
                expected = -rates.copy()
                expected[i] = np.sum(rates) - rates[i]
                gap = float(np.max(np.abs(grad - expected), initial=0.0))
                scale = float(max(np.max(np.abs(grad), initial=0.0), np.max(np.abs(expected), initial=0.0)))
                above_noise = gap > _WITNESS_NOISE_SAFETY * float(np.finfo(float).eps) * max(scale, 1.0)
                if above_noise and gap > _WITNESS_RTOL * scale and (worst is None or gap > worst[0]):
                    worst = (gap, i, u.copy(), grad, expected, t)
    _refuse_shapes(result, shapes)
    _refuse_unread(result, failed, attempted, failures)
    if worst is not None:
        gap, i, u, grad, expected, t = worst
        result.add_error(
            f"optimal_control and dp disagree on the network at t={t:.4g}, node {i}, u={np.array2string(u, precision=4)}, uniform "
            f"density: dp returns {np.array2string(grad, precision=6)}, while the rates optimal_control returns give "
            f"{np.array2string(expected, precision=6)} (dp[j] = -alpha_ij for each other node j, dp[{i}] their sum). The "
            f"policy-iteration HJB reads the rates and dp's readers read dp, so a solve would read two Hamiltonians "
            f"(#2901).",
            location="optimal_control",
            suggestion="Make dp the gradient of the Hamiltonian whose rates optimal_control returns, or override neither",
        )


def _network_probe_nodes(hamiltonian: Any, n: int) -> list[int]:
    """The nodes a network's rates and derivatives are compared at (#2901): both ends of the node order, a middle node,
    and the node with the most neighbours (the first, on a tie), counted by `get_neighbors` and not by edge weight. At
    most four nodes, whatever the network's size, as the grid is probed at 2d + 1 nodes and not at every node."""
    nodes = [0, n - 1, n // 2]
    data = getattr(hamiltonian, "network_data", None)
    if data is not None:
        neighbours = [len(data.get_neighbors(i)) for i in range(n)]
        nodes.append(int(np.argmax(neighbours)))
    return list(dict.fromkeys(nodes))


def _compare_network_derivatives_with_h(
    hamiltonian: Any, result: ValidationResult, times: Sequence[float], tolerance: float, n: int, nodes: Sequence[int]
) -> None:
    """A ``NetworkHamiltonian``'s ``dp`` and ``dm`` against central differences of its ``__call__`` (#2901).

    NetworkHJBSolver reads ``__call__``, policy iteration and the network FP read the rates, and ``dp``'s readers read
    ``dp``; the rates are held to ``dp`` by :func:`_compare_network_rates_with_dp`, and this holds ``dp`` and ``dm`` to
    ``__call__``. Compared at ``nodes`` (`_network_probe_nodes`), each over its own neighbourhood (the node and its
    neighbours, the costate entries its H reads), at the rates check's three costates and at each of ``times``, with
    the uniform density; ``dm`` against the difference in the node's own density. A disagreement at a node not probed
    is not seen. The witness rule is the grid's (`_compare_derivative`). Each source is read through `_read_source`
    as its readers hand it: ``__call__`` and ``dm`` the node as an integer, as ``NetworkMFGProblem.hamiltonian`` and
    ``hamiltonian_dm`` do, and ``dp`` ``np.array([i])``, as the rates check reads it. Each read has its own guard: a
    node, a neighbour or a ``dm`` read that cannot be evaluated is warned about and the others are still compared, a
    source that cannot be evaluated at any probe is an error, and one that returns another number of values than is
    read is an error at any probe (#2901).

    A custom ``hamiltonian_func`` replaces ``__call__``, and with a custom ``hamiltonian_dm_func`` also ``dm``; ``dp``
    and the rates stay the built-in control by design (#1545, open, is where routing a custom network H through them
    lives), and every reader of those refuses such a Hamiltonian through its ``has_custom_hamiltonian``, so only
    NetworkHJBSolver reads it. Its derivatives are therefore not compared here."""
    if getattr(hamiltonian, "has_custom_hamiltonian", False):
        return
    try:
        data = hamiltonian.network_data
    except _EVALUATION_ERRORS as e:  # reported as a warning, never gating
        result.add_warning(
            f"Could not compare the network's dp and dm with H: {type(e).__name__}: {e}", location="dH_dp"
        )
        return
    k = np.arange(n, dtype=float)
    costates = (np.linspace(0.0, 2.0, n), np.sqrt(k + 1.0), np.sin(1.3 * k + 0.2))
    density = np.full(n, 1.0 / n)
    step = _FD_STEP
    witness: dict[str, tuple[_FDComparison, int, int]] = {}
    failures: dict[str, str] = {}
    failed = {"__call__": 0, "dp": 0, "dm": 0}
    shapes: dict[str, str] = {}
    attempted = 0
    unreadable_data = ""

    def fail(name: str, t: float, i: int, what: str, e: BaseException) -> None:
        if isinstance(e, SourceShapeError):
            shapes.setdefault(_NETWORK_LOCATION[name], f"{e}, at t={t:.4g}, node {i} ({what})")
        else:
            failures.setdefault(name, f"at t={t:.4g}, node {i} ({what}): {type(e).__name__}: {e}")

    for t in (float(s) for s in times):
        for u in costates:
            for i in nodes:
                attempted += 1
                node = np.array([i])
                try:
                    h_center = _read_source(
                        hamiltonian, t=t, x=i, p=u, m=density, size=1, name="__call__", form="node"
                    )[0]
                except (SourceShapeError, *_EVALUATION_ERRORS) as e:  # this node is skipped, reported
                    failed["__call__"] += not isinstance(e, SourceShapeError)
                    fail("__call__", t, i, "H", e)
                    continue
                try:
                    neighbours = [i, *(int(q) for q in data.get_neighbors(i))]
                except _EVALUATION_ERRORS as e:  # the network's own data, not a source: warned about
                    unreadable_data = unreadable_data or f"at t={t:.4g}, node {i}: {type(e).__name__}: {e}"
                    continue
                try:
                    grad = _read_source(hamiltonian.dp, t=t, x=node, p=u, m=density, size=n, name="dp", form="node")
                except (SourceShapeError, *_EVALUATION_ERRORS) as e:
                    failed["dp"] += not isinstance(e, SourceShapeError)
                    fail("dp", t, i, "dp", e)
                    grad = None
                for j in neighbours if grad is not None else ():
                    e_j = np.zeros(n)
                    e_j[j] = step
                    try:
                        h_minus = _read_source(
                            hamiltonian, t=t, x=i, p=u - e_j, m=density, size=1, name="__call__", form="node"
                        )[0]
                        h_plus = _read_source(
                            hamiltonian, t=t, x=i, p=u + e_j, m=density, size=1, name="__call__", form="node"
                        )[0]
                    except (SourceShapeError, *_EVALUATION_ERRORS) as e:  # this neighbour is skipped
                        fail("__call__", t, i, f"H beside u[{j}]", e)
                        continue
                    comparison = _compare_derivative(
                        h_minus, h_center, h_plus, step, grad[j], density[i], u, tolerance, t, node.astype(float)
                    )
                    if comparison.is_witness and "dp" not in witness:
                        witness["dp"] = (comparison, i, j)
                e_i = np.zeros(n)
                e_i[i] = step
                try:
                    dm = _read_source(hamiltonian.dm, t=t, x=i, p=u, m=density, size=1, name="dm", form="node")[0]
                except (SourceShapeError, *_EVALUATION_ERRORS) as e:
                    failed["dm"] += not isinstance(e, SourceShapeError)
                    fail("dm", t, i, "dm", e)
                    continue
                try:
                    h_minus = _read_source(
                        hamiltonian, t=t, x=i, p=u, m=density - e_i, size=1, name="__call__", form="node"
                    )[0]
                    h_plus = _read_source(
                        hamiltonian, t=t, x=i, p=u, m=density + e_i, size=1, name="__call__", form="node"
                    )[0]
                except (SourceShapeError, *_EVALUATION_ERRORS) as e:
                    fail("__call__", t, i, "H beside m", e)
                    continue
                comparison = _compare_derivative(
                    h_minus, h_center, h_plus, step, dm, density[i], u, tolerance, t, node.astype(float)
                )
                if comparison.is_witness and "dm" not in witness:
                    witness["dm"] = (comparison, i, i)
    _refuse_shapes(result, shapes)
    _refuse_unread(result, failed, attempted, failures)
    if unreadable_data:
        result.add_warning(f"Could not read a probed node's neighbours {unreadable_data}", location="dH_dp")
    for name, (comparison, i, j) in witness.items():
        quantity = f"dp[{j}]" if name == "dp" else "dm"
        result.add_error(
            f"{quantity} is not the derivative of the network's H at t={comparison.t:.4g}, node {i}, "
            f"u={np.array2string(comparison.p, precision=4)}, uniform density: analytical={comparison.analytical:.6g}, "
            f"finite-difference={comparison.numerical:.6g}. NetworkHJBSolver reads __call__ and the other readers read "
            f"{'dp and the rates' if name == 'dp' else 'dm'}, so a solve would read two Hamiltonians (#2901).",
            location="dH_dp" if name == "dp" else "dH_dm",
            suggestion="Make dp and dm the derivatives of __call__, or override none of them",
        )


def validate_composed_wall(
    walled: Any,
    unwalled: Any,
    geometry: GeometryProtocol,
    times: Sequence[float] = (0.0,),
    tolerance: float = _WARN_RTOL,
) -> ValidationResult:
    """The soft wall composed into a ``SeparableHamiltonian``'s potential, read back through the composed H (#2002,
    #2901).

    ``MFGProblem`` composes ``state_penalty`` into the Hamiltonian's ``_potential``, and the solvers read the composed
    H. A subclass whose ``__call__`` does not read ``_potential`` would drop the wall from every solve, silently. At
    D2's probes (t, x, m, p), with x also the grid node where the wall is largest at that t, the walled H minus the
    unwalled one must equal minus the change of the potential, read as ``SeparableHamiltonian`` reads it
    (``H = cost - V - f``); a gap above the witness margin is an error. So a ``__call__`` that drops the wall is seen
    wherever the wall is nonzero on the grid, an interior obstacle between D2's x probes included; one that reads the
    potential at some x and not others is seen only at those x.

    D2 ran before the composition and saw the unwalled H alone, so the composed H's doors are compared with its methods
    here, at the same points and times (`_compare_doors_with_methods`): a door that does not read the potential would
    make the solvers read the problem without its wall. Every read goes through `_read_source`; a read that cannot be
    evaluated is warned about, and one that returns another number of values than is read is an error (#2901)."""
    result = ValidationResult()
    x_sample, _p_sample, _m_sample, dimension, err = _get_sample_inputs(geometry, "hamiltonian")
    if err is not None:
        return err

    def potential(h: Any, t: float, x: np.ndarray) -> float:
        if getattr(h, "_potential", None) is None:
            return 0.0
        return _read_source(h, t=t, x=x, p=None, m=None, size=1, name="potential", form="potential")[0]

    worst: tuple[float, float, np.ndarray, float, np.ndarray, float, float] | None = None
    failure: str | None = None
    shapes: dict[str, str] = {}
    probes_x: list[tuple[float, list[np.ndarray]]] = []
    nodes = _grid_nodes(geometry, dimension)
    for t in (float(s) for s in times):
        points = _probe_points(geometry, x_sample)
        if nodes is not None:
            try:
                wall = [abs(potential(walled, t, x) - potential(unwalled, t, x)) for x in nodes]
            except SourceShapeError as e:
                shapes.setdefault("state_penalty", f"{e}, at t={t:.4g}, reading the potential at the grid's nodes")
            except _EVALUATION_ERRORS as e:  # reported as a warning, never gating
                if failure is None:
                    failure = f"at t={t:.4g}, reading the potential at the grid's nodes: {type(e).__name__}: {e}"
            else:
                largest = nodes[int(np.argmax(wall))]
                if not any(np.array_equal(largest, q) for q in points):
                    points.append(largest)
        probes_x.append((t, points))
        for x in points:
            for m, p in _consistency_probes(dimension):
                try:
                    expected = -(potential(walled, t, x) - potential(unwalled, t, x))
                    h_walled = _read_source(walled, t=t, x=x, p=p, m=m, size=1, name="__call__")[0]
                    h_unwalled = _read_source(unwalled, t=t, x=x, p=p, m=m, size=1, name="__call__")[0]
                except SourceShapeError as e:
                    shapes.setdefault("state_penalty", f"{e}, at {_probe_text(t, x, m, p)}")
                    continue
                except _EVALUATION_ERRORS as e:  # reported as a warning, never gating
                    if failure is None:
                        failure = f"at {_probe_text(t, x, m, p)}: {type(e).__name__}: {e}"
                    continue
                moved = h_walled - h_unwalled
                gap = abs(moved - expected)
                scale = max(abs(moved), abs(expected))
                above_noise = gap > _WITNESS_NOISE_SAFETY * float(np.finfo(float).eps) * max(scale, 1.0)
                if above_noise and gap > _WITNESS_RTOL * scale and (worst is None or gap > worst[0]):
                    worst = (gap, t, x, m, p.copy(), moved, expected)
    _refuse_shapes(result, shapes)
    if failure is not None:
        result.add_warning(f"Could not compare the composed wall {failure}", location="state_penalty")
    if worst is not None:
        _gap, t, x, m, p, moved, expected = worst
        result.add_error(
            f"the soft wall composed into {type(walled).__name__}'s potential does not reach its __call__ at "
            f"{_probe_text(t, x, m, p)}: __call__ moved by {moved:.6g}, and the wall is {expected:.6g} there. Its "
            f"__call__ does not read the potential the wall was composed into, so every solve would drop the wall (#2002, "
            f"#2901).",
            location="state_penalty",
            suggestion="Read the potential through super().__call__, or fold the penalty into your own Hamiltonian directly",
        )
    _compare_doors_with_methods(walled, probes_x, dimension, tolerance, result)
    return result


def validate_hamiltonian_consistency(
    hamiltonian: Any,
    dH_dm: Callable,
    geometry: GeometryProtocol,
    tolerance: float = _WARN_RTOL,
    dH_dp: Callable | None = None,
    optimal_control: Callable | None = None,
    times: Sequence[float] = (0.0,),
) -> ValidationResult:
    """
    Check that dH_dm and dH_dp are the derivatives of H, and that the optimal control is -dH/dp, by finite differences.

    The probes are every (t, x, m, p) of (#2901):

    - t in ``times`` (a problem hands 0, T/2 and T);
    - x the node nearest the centre of the nodes' box and, along each axis, its two end nodes nearest it, at most 2d + 1
      nodes (`_probe_points`);
    - m in 0.5, 1.0;
    - p with component k a scale times (-1)**k (1 + 0.3 k), the scale in 0, 0.4157, -1.3691, so every component is
      probed at both signs and at a magnitude of its own (`_consistency_probes`).

    A disagreement that falls between the probes is not seen: the check proves a disagreement where it finds one, and an
    agreement only at the probes. Every source is read through `_read_source`, the checks' one reader, inside its
    comparison's guard (#2901). A probe at which H cannot be evaluated is skipped, and one at which a method
    (``dH_dm``, ``dH_dp``, ``optimal_control``) or H beside the probe cannot is skipped for that comparison alone, each
    warned about once, naming the first such probe. H that cannot be evaluated at any probe is an error, and so is a
    method that raises at every probe at which H could be evaluated, since whatever reads it would fail or read another
    source. A source that returns another number of values than is read (one for H and ``dH_dm``, d for ``dH_dp`` and the
    control) is an error at the first probe it does so, naming that probe and the shape: it breaks its contract there.
    An error outside ``_EVALUATION_ERRORS`` and ``SourceShapeError`` propagates.

    This check owns one axis, the methods agreeing at a point. The other, a door's batch read agreeing with its point
    read, is ``refuse_a_hamiltonian_whose_reads_disagree``'s, which a problem runs beside this; and the cost standing for
    H only as the library ships it is ``_why_not_as_it_ships``'s.

    Numerical checks, at each probe:
        dH_dm_numerical = (H(t, x, p, m+eps) - H(t, x, p, m-eps)) / (2*eps)
        dH_dp_numerical[i] = (H(t, x, p+eps*e_i, m) - H(t, x, p-eps*e_i, m)) / (2*eps)

    Severity (Issue #1642, capability C1). This validator can invalidate its
    result, so a caller doing `if not result.is_valid: raise ValidationError(...)`
    will fire. It does so only on an **exhibited witness** — a probe point where
    H is smooth, all values are finite, and the discrepancy clears both a relative
    margin and the differencing roundoff bound by a wide factor. Every weaker
    signal (small discrepancy, kink at the probe, non-finite value, failure to
    evaluate somewhere) is a warning, because a false positive here would refuse a correct
    problem at construction time. The exceptions are a source that cannot be evaluated at any
    probe, and one that returns another number of values than is read (#2901).

    Args:
        hamiltonian: HamiltonianBase instance (called as H(t, x, p, m) by keyword) or a raw
            callable H(t, x, p, m), bound by name or position (#2375 ruling 8)
        dH_dm: Claimed derivative dH/dm: a family bound method (called by keyword) or a raw callable(t, x, p, m)
        geometry: Geometry for sample point
        tolerance: Relative tolerance for the warning tier
        dH_dp: Claimed gradient dH/dp, the same kinds as dH_dm. Optional.
        optimal_control: Claimed optimal control, the same kinds. Optional. Checked to be -dH/dp at the same probes,
            with the same witness rule, so a non-smooth cost is compared only where H is smooth at the probe (#2901).
        times: The probes' times. A problem hands 0, T/2 and T (#2901).

    A ``HamiltonianBase`` also has each door (``evaluate_H``, ``evaluate_dp``, ``evaluate_optimal_control``,
    ``dH_dm``) compared with its method at the same probes, at one point, at one population: the method read as this check
    reads it, and the door handed one point by a point-only Hamiltonian and a batch of one by the others; a door that
    raises at every probe at which its method could be evaluated is an error (#2901). A ``NetworkHamiltonian`` is checked on its own probes instead, at `_network_probe_nodes`
    computed once: the rates it returns against ``dp`` by the network's identity, and ``dp`` and ``dm`` against central
    differences of its ``__call__``, ``dp`` over each node's neighbourhood, at each of ``times``, three costates and the
    uniform density. A disagreement at a node not probed is not seen (#2901).

    Returns:
        ValidationResult; is_valid is False when a witness was exhibited, when H, a method or a door cannot be evaluated
        at any probe, and when a source returns another number of values than is read.
    """
    from mfgarchon.extensions.topology import NetworkHamiltonian

    result = ValidationResult()
    if isinstance(hamiltonian, NetworkHamiltonian):
        try:
            n = int(hamiltonian.num_nodes)
            nodes = _network_probe_nodes(hamiltonian, n)
        except _EVALUATION_ERRORS as e:  # reported as a warning, never gating
            result.add_warning(
                f"Could not read the network's nodes, so its sources were not compared: {type(e).__name__}: {e}",
                location="hamiltonian",
            )
            return result
        _compare_network_rates_with_dp(hamiltonian, result, times, n, nodes)
        _compare_network_derivatives_with_h(hamiltonian, result, times, tolerance, n, nodes)
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
    # Each comparison has its own guard inside a probe, so one that cannot be evaluated leaves the others standing, and
    # every read goes through `_read_source` inside that guard (#2901). `failures` keeps the first evaluation error of
    # each source, for its warning; `failed` counts the probes at which a source's own read raised, against `probes`
    # for H and against `evaluated`, the probes at which H could be evaluated, for the methods. `shapes` keeps the first
    # probe at which a source returned another number of values than is read, which is refused at any probe.
    failures: dict[str, str] = {}
    failed = {"hamiltonian": 0, "dH_dm": 0, "dH_dp": 0, "optimal_control": 0}
    shapes: dict[str, str] = {}
    evaluated = 0

    def fail(name: str, t: float, x: np.ndarray, m: float, p: np.ndarray, e: BaseException) -> None:
        if isinstance(e, SourceShapeError):
            shapes.setdefault(name, f"{e}, at {_probe_text(t, x, m, p)}")
        else:
            failures.setdefault(name, f"at {_probe_text(t, x, m, p)}: {type(e).__name__}: {e}")

    points = _probe_points(geometry, x_sample)
    probes = [(t, x, m, p) for t in times for x in points for m, p in _consistency_probes(dimension)]
    for t, x, m, p in probes:
        try:
            h_center = _read_source(hamiltonian, t=t, x=x, p=p, m=m, size=1, name="hamiltonian")[0]
        except (SourceShapeError, *_EVALUATION_ERRORS) as e:  # H cannot be read here: the probe is skipped, reported
            failed["hamiltonian"] += not isinstance(e, SourceShapeError)
            fail("hamiltonian", t, x, m, p, e)
            continue
        evaluated += 1

        # --- dH_dm: its read, then H's two reads beside the probe ---
        try:
            dm_value = _read_source(dH_dm, t=t, x=x, p=p, m=m, size=1, name="dH_dm")[0]
        except (SourceShapeError, *_EVALUATION_ERRORS) as e:
            failed["dH_dm"] += not isinstance(e, SourceShapeError)
            fail("dH_dm", t, x, m, p, e)
        else:
            try:
                h_m_plus = _read_source(hamiltonian, t=t, x=x, p=p, m=m + step, size=1, name="hamiltonian")[0]
                h_m_minus = _read_source(hamiltonian, t=t, x=x, p=p, m=m - step, size=1, name="hamiltonian")[0]
            except (SourceShapeError, *_EVALUATION_ERRORS) as e:
                fail("hamiltonian", t, x, m, p, e)
            else:
                comparison = _compare_derivative(h_m_minus, h_center, h_m_plus, step, dm_value, m, p, tolerance, t, x)
                if comparison.is_witness and dm_witness is None:
                    dm_witness = comparison
                if comparison.is_inconsistent and dm_inconsistent is None:
                    dm_inconsistent = comparison

        # --- dH_dp and the optimal control: each read, then H's two reads per component ---
        if dH_dp is None and optimal_control is None:
            continue
        dp_analytical = None
        if dH_dp is not None:
            try:
                dp_analytical = _read_source(dH_dp, t=t, x=x, p=p, m=m, size=dimension, name="dH_dp")
            except (SourceShapeError, *_EVALUATION_ERRORS) as e:
                failed["dH_dp"] += not isinstance(e, SourceShapeError)
                fail("dH_dp", t, x, m, p, e)
        control = None
        if optimal_control is not None:
            try:
                control = _read_source(optimal_control, t=t, x=x, p=p, m=m, size=dimension, name="optimal_control")
            except (SourceShapeError, *_EVALUATION_ERRORS) as e:
                failed["optimal_control"] += not isinstance(e, SourceShapeError)
                fail("optimal_control", t, x, m, p, e)
        if dp_analytical is None and control is None:
            continue
        for i in range(dimension):
            p_plus = p.copy()
            p_minus = p.copy()
            p_plus[i] += step
            p_minus[i] -= step
            try:
                h_minus = _read_source(hamiltonian, t=t, x=x, p=p_minus, m=m, size=1, name="hamiltonian")[0]
                h_plus = _read_source(hamiltonian, t=t, x=x, p=p_plus, m=m, size=1, name="hamiltonian")[0]
            except (SourceShapeError, *_EVALUATION_ERRORS) as e:
                fail("hamiltonian", t, x, m, p, e)
                continue
            if dp_analytical is not None:
                comparison = _compare_derivative(
                    h_minus, h_center, h_plus, step, dp_analytical[i], m, p, tolerance, t, x
                )
                if comparison.is_witness and dp_witness[i] is None:
                    dp_witness[i] = comparison
                if comparison.is_inconsistent and dp_inconsistent[i] is None:
                    dp_inconsistent[i] = comparison
            if control is not None:
                dp_i = float("nan") if dp_analytical is None else dp_analytical[i]
                comparison = _compare_derivative(h_minus, h_center, h_plus, step, -control[i], m, p, tolerance, t, x)
                if comparison.is_witness and control_witness[i] is None:
                    control_witness[i] = (comparison, dp_i)
                if comparison.is_inconsistent and control_inconsistent[i] is None:
                    control_inconsistent[i] = (comparison, dp_i)

    _refuse_shapes(result, shapes)
    if "hamiltonian" in failures:
        if failed["hamiltonian"] == len(probes):
            # An H that cannot be evaluated at any probe is compared nowhere: whatever reads it fails (#2901).
            result.add_error(
                f"The Hamiltonian could not be evaluated at any of the {len(probes)} probes, first "
                f"{failures['hamiltonian']}, so none of its sources was compared. The library's solvers read it, so a "
                "solve would fail there (#2901).",
                location="hamiltonian",
                suggestion="Make the Hamiltonian evaluate at one point, x a tuple of numbers, p of shape (d,), m a number",
            )
        else:
            result.add_warning(
                f"Could not evaluate the Hamiltonian {failures['hamiltonian']} ({failed['hamiltonian']} of "
                f"{len(probes)} probes, and reads beside a probe); the probes that could be evaluated were compared",
                location="hamiltonian",
            )
    for name in ("dH_dm", "dH_dp", "optimal_control"):
        if name not in failures:
            continue
        if failed[name] == evaluated:
            # A method that raises wherever H can be evaluated is not a gap between probes: whatever reads it fails, and
            # a point-only Hamiltonian's doors are not read at construction by the batch check (#2901).
            result.add_error(
                f"{name} could not be evaluated at any probe at which H could ({evaluated} of them), first "
                f"{failures[name]}. The library's solvers read {name}, so a solve would fail there or read another "
                f"source in its place (#2901).",
                location=name,
                suggestion=f"Make {name} evaluate wherever H does",
            )
        else:
            result.add_warning(
                f"Could not evaluate {name} {failures[name]} ({failed[name]} of {evaluated} probes); the probes that "
                "could be evaluated were compared",
                location=name,
            )

    _report(result, "dH_dm", "dH_dm", x_sample, dm_witness, dm_inconsistent)
    for i in range(dimension):
        _report(result, f"dH_dp[{i}]", f"dH_dp[{i}]", x_sample, dp_witness[i], dp_inconsistent[i])
    for i in range(dimension):
        _report_control(result, i, x_sample, control_witness[i], control_inconsistent[i])
    _compare_doors_with_methods(hamiltonian, [(t, points) for t in times], dimension, tolerance, result)

    return result
