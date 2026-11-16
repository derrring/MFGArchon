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
# concrete (t, x, m, p) where a derivative identity, the control's against -dH/dp among
# them (#2901), measurably fails by a margin no finite-difference artifact can produce;
# where a source returns what the owner of its shape, `checked_control` or
# `checked_value`, refuses (#2875); or where H, dH_dp or the control cannot be read at
# any probe. Anything weaker stays a warning,
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


def _is_family_callable(fn: Any) -> bool:
    """Whether `_call_h` calls ``fn`` by keyword as a family method: a ``HamiltonianBase``/``LagrangianBase`` instance, a
    method bound to one, or a classmethod bound to a family class, which the solvers call by keyword as they do any
    method (#2901); a ``functools.partial`` of any of these is one too. The one predicate `_call_h` and
    `_raw_callable_advice` read."""
    import functools

    from mfgarchon.core.hamiltonian import MFGOperatorBase

    target = fn.func if isinstance(fn, functools.partial) else fn  # partial(H.dm) is still a family method
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

    A family object (`_is_family_callable`: a ``HamiltonianBase``/``LagrangianBase`` instance, a method bound to one,
    or a classmethod bound to a family class) takes ``(t, x, p, m)`` since #2378 phase 5 and is called by keyword. A raw
    user callable goes through its binding (``HAMILTONIAN_SLOTS``), like every user callable the library accepts.
    """
    from mfgarchon.types.callable_protocols import HAMILTONIAN_SLOTS, _remembered, point_tuple, refuse_overridden

    x = point_tuple(x)
    if _is_family_callable(fn):
        refuse_overridden(fn, HAMILTONIAN_SLOTS.order, role="Hamiltonian callable")
        return fn(t=t, x=x, p=p, m=m)
    return _remembered(fn, HAMILTONIAN_SLOTS, "Hamiltonian callable")(t=t, x=x, p=p, m=m)


#: What a Hamiltonian or one of its methods raises when it cannot be evaluated at a probe: a wrong call, a kink its own
#: minimiser refuses (#2644's ConvergenceError is a RuntimeError), a value out of its domain. `_read_source` marks one
#: raised by the source's own call as `SourceEvaluationError`, and a comparison that meets that skips the probe and
#: warns, never gating on it alone (#2901); anything else is a bug and propagates.
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


class SourceShapeError(Exception):
    """A source returned what the owner of its return's shape refuses (``checked_control``, ``checked_value``). It
    breaks its contract wherever it does so, so a check refuses it at any probe (#2901). It is neither one of
    ``_EVALUATION_ERRORS`` nor a ``ValueError``, so no guard that skips a probe takes it."""


class SourceEvaluationError(Exception):
    """A source could not be evaluated where `_read_source` read it: its own call raised one of ``_EVALUATION_ERRORS``,
    or what it returned holds no number. The cause is chained, and the message is its type and text. The checks' guards
    take this and `SourceShapeError` and nothing else, so an error in the reader's own code propagates as the bug it is
    and is never read as the source failing (#2901)."""

    def __init__(self, cause: BaseException) -> None:
        super().__init__(f"{type(cause).__name__}: {str(cause).rstrip('.')}")


#: The owner of each method's and door's return shape (#2875, #2901): p's shape for dp, the control and their doors, m's
#: for the value, dm and ``evaluate_H``.
_SHAPE_OWNER = {
    "dp": "control",
    "optimal_control": "control",
    "evaluate_dp": "control",
    "evaluate_optimal_control": "control",
    "__call__": "value",
    "dm": "value",
    "evaluate_H": "value",
}


def _read_source(
    source: Any,
    *,
    t: float,
    x: Any,
    p: Any,
    m: Any,
    method: str,
    form: str = "point",
    reader: str = "validate_hamiltonian_consistency",
) -> np.ndarray:
    """Read one of a Hamiltonian's sources, as the checks' one reader (#2901).

    ``form`` is how the source is called:

    - ``"point"``: as `_call_h` calls it at one point, a family method by keyword, a raw callable through its binding;
    - ``"door"``: a door, handed ``HEvalState(t, x, p, m)``, at one point (p of shape (d,), m a number) or on a batch
      (x and p of shape (N, d), m of shape (N,)), as the solvers hand it;
    - ``"potential"``: ``source._potential``, the potential a ``SeparableHamiltonian`` holds, bound and read as its
      ``__call__`` reads it, at one point (``x`` of shape (d,)) or at the rows of ``x`` of shape (N, d), handed as a
      coordinate tuple; ``p`` and ``m`` are not read.

    What a method or door returns goes through the owner of its return shape: ``checked_control`` for ``dp``,
    ``optimal_control`` and their doors, ``checked_value`` for ``__call__``, ``dm`` and ``evaluate_H``. A potential is one
    value per point read, or one value for all of them, as ``SeparableHamiltonian`` broadcasts it. A return the owner
    refuses raises `SourceShapeError`, naming ``reader``.

    The reader tells three things apart (#2901). One of ``_EVALUATION_ERRORS`` raised by the source's call, which is the
    source and its binding (the binder of a raw callable or a potential, and ``refuse_overridden``'s refusal of a family
    method's parameters), or by converting what it returned to floats, raises `SourceEvaluationError`: the source could
    not be evaluated there. numpy alone reads ``None`` in an object array as NaN, so such a return is converted entry by
    entry, and one that holds no number is a probe its source could not be evaluated at, never a NaN value. A shape the
    owner refuses raises `SourceShapeError`. Anything else propagates as it is, and so does any error in the reader's own
    code before or after that call, the conversion of ``x`` to the form handed included."""
    from mfgarchon.core.hamiltonian import HEvalState, _bound_potential, checked_control, checked_value
    from mfgarchon.types.callable_protocols import coordinate_tuple, point_tuple

    rows = form == "potential" and np.ndim(x) == 2
    if form == "potential":
        at = coordinate_tuple(np.asarray(x, dtype=float), np.shape(x)[1]) if rows else point_tuple(x)
        call = lambda: _bound_potential(source)(x=at, t=t)  # noqa: E731
    elif form == "door":
        state = HEvalState(t=t, x=x, p=p, m=m)
        call = lambda: source(state)  # noqa: E731
    elif form == "point":
        at = point_tuple(x)
        call = lambda: _call_h(source, at, m, p, t)  # noqa: E731
    else:
        raise ValueError(f"_read_source: unknown form {form!r}")
    try:
        value = call()
    except _EVALUATION_ERRORS as e:
        raise SourceEvaluationError(e) from e
    if form != "potential":
        owner_of = getattr(source, "__self__", source)
        try:
            if _SHAPE_OWNER[method] == "control":
                value = checked_control(value, p, reader=reader, source=owner_of, m=m, method=method)
            else:
                value = checked_value(value, m, reader=reader, source=owner_of, p=p, method=method)
        except ValueError as refusal:
            raise SourceShapeError(str(refusal)) from refusal
    # The reader's one conversion, for every form: an object array entry by entry, so None raises rather than reading
    # as NaN.
    try:
        array = np.asarray(value)
        floats = (
            np.array([float(v) for v in array.reshape(-1).tolist()]).reshape(array.shape)
            if array.dtype == object
            else np.asarray(array, dtype=float)
        )
    except _EVALUATION_ERRORS as e:
        raise SourceEvaluationError(e) from e
    if form != "potential":
        return floats
    wanted = np.shape(x)[0] if rows else 1
    if floats.size not in (1, wanted):
        raise SourceShapeError(
            f"{type(source).__name__}'s potential returned shape {floats.shape} for {wanted} points, where one value per "
            "point, or one for all of them, is read"
        )
    return np.broadcast_to(floats.reshape(-1), (wanted,)).copy() if rows else floats.reshape(())


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
        optimal_control: The optimal control, the same kinds as dH_dp. Optional; checked to be -dH/dp (#2901).
        times: The consistency probes' times. A problem hands 0, T/2 and T (#2901).

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


def _grid_probes_apply(source: Any) -> bool:
    """Whether the construction checks' grid probes apply to a Hamiltonian source (#2901, stage 1c-iii).

    They do not for a ``NetworkHamiltonian``, or a method bound to one. Its x is a node and its p the costate at every
    node, so a grid probe hands it a form no network solver hands: x a point's coordinates, which it casts to a node,
    p of the coordinate dimension and m one number. Nor is a network Hamiltonian compared some other way, and it needs
    no comparison. With the built-in control, ``__call__``, the rates ``optimal_control`` returns and ``source_term``
    come from one formula. With a custom ``hamiltonian_func``, ``__call__`` reads that function, and both readers of
    the rates, ``FPNetworkSolver`` and ``NetworkPolicyIterationHJBSolver``, refuse it (#1545), so the two never meet
    in a solve. No solver reads its ``dm`` or ``dp`` (the stage's census, at d1b3879f).

    Every check that would read a network Hamiltonian on a grid probe reads this, and returns before reading. A network
    method that is no longer bound, such as a ``functools.partial`` of ``h.dm`` handed to
    ``validate_hamiltonian_derivative`` directly, is not recognised and is read on the grid probe; construction hands
    the bound method.
    """
    import inspect

    from mfgarchon.extensions.topology import NetworkHamiltonian

    owner = source.__self__ if inspect.ismethod(source) else source
    return not isinstance(owner, NetworkHamiltonian)


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

    A ``NetworkHamiltonian`` is not read here: the grid probe hands it a form no network solver hands
    (`_grid_probes_apply`).

    Args:
        hamiltonian: HamiltonianBase instance or callable
        geometry: Geometry for sample point

    Returns:
        ValidationResult
    """
    result = ValidationResult()
    if not _grid_probes_apply(hamiltonian):
        return result

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

    A method bound to a ``NetworkHamiltonian`` is not read here: the grid probe hands it a form no network solver
    hands (`_grid_probes_apply`).

    Args:
        derivative_func: Derivative function (bound method or callable)
        geometry: Geometry for sample point
        name: Name for error messages ("dH_dm" or "dH_dp")

    Returns:
        ValidationResult
    """
    result = ValidationResult()
    if not _grid_probes_apply(derivative_func):
        return result

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
        wanted = "an ndarray of p's shape" if name == "dH_dp" else "float or ndarray"
        result.add_error(
            f"{name} must return {wanted}, got {type(value).__name__}",
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
    then the nodes `spread_samples` picks, so at most 2d + 4 nodes in d <= 2 and 2d + 5 in 3-D, all inside the domain
    (#2875, #2901). A geometry whose grid is not an array of nodes is probed at the sample point alone."""
    nodes = _grid_nodes(geometry, middle.size)
    if nodes is None:
        return [middle]
    return _probe_nodes(nodes)


def _probe_nodes(nodes: np.ndarray) -> list[np.ndarray]:
    """`_probe_points` of an (N, d) array of nodes: the one owner of which nodes the checks probe (#2901). After the
    centre and the end nodes come the nodes `spread_samples` picks, 3, or 4 in 3-D: on a uniform grid the centre and
    the ends are where a term such as sin(2 pi x) takes one value, so a read frozen at one node passes there (#2875)."""
    from mfgarchon.types.callable_protocols import _sample_count, spread_samples

    centre = 0.5 * (nodes.min(axis=0) + nodes.max(axis=0))
    middle = nodes[int(np.argmin(np.sum((nodes - centre) ** 2, axis=1)))]
    points = [middle]
    for k in range(nodes.shape[1]):
        for end in (nodes[:, k].min(), nodes[:, k].max()):
            on_end = nodes[nodes[:, k] == end]
            nearest = on_end[int(np.argmin(np.sum((on_end - middle) ** 2, axis=1)))]
            if not any(np.array_equal(nearest, q) for q in points):
                points.append(nearest)
    for i in spread_samples(nodes.shape[0], _sample_count(nodes.shape[1])):
        if not any(np.array_equal(nodes[i], q) for q in points):
            points.append(nodes[i])
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
    a kink produces a true-but-useless warning. In d = 1 to 3 every nonzero probe's |p| is at least 4.57% from 0.5, 1,
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
    point = _probe_text(worst.t, where, worst.m, worst.p)

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


def _probe_text(t: float, x: np.ndarray, m: float, p: np.ndarray) -> str:
    """A probe (t, x, m, p) as the reports write it."""
    return f"t={t:.4g}, x={np.array2string(np.asarray(x), precision=4)}, m={m:.4g}, p={np.array2string(p, precision=4)}"


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
    point = _probe_text(comparison.t, where, comparison.m, comparison.p)
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
    read_dp = f"dp[{i}]={dp_i:.8g}, " if np.isfinite(dp_i) else ""  # no dp read here: none handed, or none finite
    detail = (
        f"at {point}: -optimal_control[{i}]={comparison.analytical:.8g}, {read_dp}and the central "
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
    - x the node nearest the centre of the nodes' box and, along each axis, its two end nodes nearest it, then the 3
      nodes ``spread_samples`` picks (4 in 3-D), at most 2d + 4 nodes, 2d + 5 in 3-D (`_probe_points`);
    - m in 0.5, 1.0;
    - p with component k a scale times (-1)**k (1 + 0.3 k), the scale in 0, 0.4157, -1.3691, so every component is
      probed at both signs and at a magnitude of its own (`_consistency_probes`).

    A disagreement that falls between the probes is not seen: the check proves a disagreement where it finds one, and an
    agreement only at the probes. Every source is read through `_read_source`, the checks' one reader, inside its
    comparison's guard (#2901). What a source returns goes through the owner of its shape: ``checked_value`` for H and
    ``dH_dm``, one value at a point, and ``checked_control`` for ``dH_dp`` and the control, p's shape (#2875). A probe at
    which H cannot be evaluated is skipped, and one at which a method or H beside the probe cannot is skipped for that
    comparison alone, each warned about once, naming the first such probe. H that cannot be evaluated at any probe is an
    error, and so is ``dH_dp`` or the control raising at every probe at which H could be evaluated, since whatever reads
    it would fail or read another source; ``dH_dm``, which no grid solver reads, is warned about there. A return the owner refuses is an error at the first probe it comes back, naming that
    probe: it breaks the source's contract there. Any other error, the reader's own included, propagates.

    Numerical checks, at each probe:
        dH_dm_numerical = (H(t, x, p, m+eps) - H(t, x, p, m-eps)) / (2*eps)
        dH_dp_numerical[i] = (H(t, x, p+eps*e_i, m) - H(t, x, p-eps*e_i, m)) / (2*eps)
        and -optimal_control[i] against dH_dp_numerical[i]

    Severity (Issue #1642, capability C1). This validator can invalidate its
    result, so a caller doing `if not result.is_valid: raise ValidationError(...)`
    will fire. It does so only on an **exhibited witness** — a probe point where
    H is smooth, all values are finite, and the discrepancy clears both a relative
    margin and the differencing roundoff bound by a wide factor. Every weaker
    signal (small discrepancy, kink at the probe, non-finite value, failure to
    evaluate somewhere) is a warning, because a false positive here would refuse a correct
    problem at construction time. The exceptions are a source that cannot be evaluated at any
    probe, and a return the owner of its shape refuses (#2875, #2901).

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

    A ``NetworkHamiltonian`` is not probed here, and its sources are compared nowhere (`_grid_probes_apply` says why).

    Returns:
        ValidationResult; is_valid is False when a witness was exhibited, when H or a method cannot be evaluated at any
        probe, and when a source returns what the owner of its shape refuses.
    """
    result = ValidationResult()
    if not _grid_probes_apply(hamiltonian):
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
    # probe at which a source returned what the owner of its shape refuses, which is refused at any probe.
    failures: dict[str, str] = {}
    failed = {"hamiltonian": 0, "dH_dm": 0, "dH_dp": 0, "optimal_control": 0}
    shapes: dict[str, str] = {}
    evaluated = 0

    def fail(name: str, t: float, x: np.ndarray, m: float, p: np.ndarray, e: BaseException) -> None:
        if isinstance(e, SourceShapeError):
            shapes.setdefault(name, f"{e} At {_probe_text(t, x, m, p)}.")
        else:
            failures.setdefault(name, f"at {_probe_text(t, x, m, p)}: {e}")

    points = _probe_points(geometry, x_sample)
    probes = [(float(t), x, m, p) for t in times for x in points for m, p in _consistency_probes(dimension)]
    for t, x, m, p in probes:
        try:
            h_center = float(_read_source(hamiltonian, t=t, x=x, p=p, m=m, method="__call__"))
        except (SourceShapeError, SourceEvaluationError) as e:  # H cannot be read here: the probe is skipped, reported
            failed["hamiltonian"] += not isinstance(e, SourceShapeError)
            fail("hamiltonian", t, x, m, p, e)
            continue
        evaluated += 1

        # --- dH_dm: its read, then H's two reads beside the probe ---
        try:
            dm_value = float(_read_source(dH_dm, t=t, x=x, p=p, m=m, method="dm"))
        except (SourceShapeError, SourceEvaluationError) as e:
            failed["dH_dm"] += not isinstance(e, SourceShapeError)
            fail("dH_dm", t, x, m, p, e)
        else:
            try:
                h_m_plus = float(_read_source(hamiltonian, t=t, x=x, p=p, m=m + step, method="__call__"))
                h_m_minus = float(_read_source(hamiltonian, t=t, x=x, p=p, m=m - step, method="__call__"))
            except (SourceShapeError, SourceEvaluationError) as e:
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
                dp_analytical = _read_source(dH_dp, t=t, x=x, p=p, m=m, method="dp")
            except (SourceShapeError, SourceEvaluationError) as e:
                failed["dH_dp"] += not isinstance(e, SourceShapeError)
                fail("dH_dp", t, x, m, p, e)
        control = None
        if optimal_control is not None:
            try:
                control = _read_source(optimal_control, t=t, x=x, p=p, m=m, method="optimal_control")
            except (SourceShapeError, SourceEvaluationError) as e:
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
                h_minus = float(_read_source(hamiltonian, t=t, x=x, p=p_minus, m=m, method="__call__"))
                h_plus = float(_read_source(hamiltonian, t=t, x=x, p=p_plus, m=m, method="__call__"))
            except (SourceShapeError, SourceEvaluationError) as e:
                fail("hamiltonian", t, x, m, p, e)
                continue
            if dp_analytical is not None:
                comparison = _compare_derivative(
                    h_minus, h_center, h_plus, step, float(dp_analytical[i]), m, p, tolerance, t, x
                )
                if comparison.is_witness and dp_witness[i] is None:
                    dp_witness[i] = comparison
                if comparison.is_inconsistent and dp_inconsistent[i] is None:
                    dp_inconsistent[i] = comparison
            if control is not None:
                dp_i = float("nan") if dp_analytical is None else float(dp_analytical[i])
                comparison = _compare_derivative(
                    h_minus, h_center, h_plus, step, -float(control[i]), m, p, tolerance, t, x
                )
                if comparison.is_witness and control_witness[i] is None:
                    control_witness[i] = (comparison, dp_i)
                if comparison.is_inconsistent and control_inconsistent[i] is None:
                    control_inconsistent[i] = (comparison, dp_i)

    for name, where in shapes.items():
        result.add_error(where, location=name)
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
        # dH_dm is read by no grid solver (#2901's census, at 45fe3298), so one that raises everywhere is warned about.
        if failed[name] == evaluated and name != "dH_dm":
            # A method that raises wherever H can be evaluated is not a gap between probes: whatever reads it fails.
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

    return result


def _potential_at(h: Any, t: float, x: np.ndarray, reader: str) -> np.ndarray:
    """The potential ``h`` holds, read as its ``__call__`` reads it: at one point, shape (), or at the rows of an (N, d)
    ``x``, shape (N,); 0 where it holds none."""
    if getattr(h, "_potential", None) is None:
        return np.zeros(np.shape(x)[0]) if np.ndim(x) == 2 else np.zeros(())
    return _read_source(h, t=t, x=x, p=None, m=None, method="potential", form="potential", reader=reader)


def _largest_wall_node(walled: Any, unwalled: Any, nodes: np.ndarray, t: float, reader: str) -> np.ndarray:
    """The node of ``nodes`` at which the soft wall composed into ``walled`` moves the potential most at ``t``: where a
    read that drops the wall is furthest off (#2901). Each potential is read once, on all the nodes together. An error
    reading a potential propagates to the caller's guard."""
    wall = np.abs(_potential_at(walled, t, nodes, reader) - _potential_at(unwalled, t, nodes, reader))
    return nodes[int(np.argmax(wall))]


def validate_composed_wall(
    walled: Any, unwalled: Any, geometry: GeometryProtocol, times: Sequence[float] = (0.0,)
) -> ValidationResult:
    """The soft wall composed into a ``SeparableHamiltonian``'s potential, read back through the composed ``__call__``
    (#2002, #2901).

    ``MFGProblem`` composes ``state_penalty`` into the Hamiltonian's potential, and the solvers read the composed H. A
    ``__call__`` that does not read the potential drops the wall from every solve, silently. At each of ``times``, at the
    consistency probes (t, x, m, p), x also the node where the wall is largest at that t, the walled ``__call__`` minus
    the unwalled one must equal minus the change of the potential, read as ``SeparableHamiltonian`` reads it
    (H = cost - V - f). A gap above the witness margin is an error. So a ``__call__`` that drops the wall is seen
    wherever the wall is nonzero on the grid, an interior obstacle between the probes included; one that reads the
    potential at some x and not others is seen only at those x.

    The composed H's doors are compared with its methods at the same nodes and times by
    `refuse_a_hamiltonian_whose_reads_disagree`. Every read goes through `_read_source`: a read that cannot be evaluated
    is warned about, once, and a return the owner of its shape refuses is an error."""
    result = ValidationResult()
    x_sample, _p_sample, _m_sample, dimension, err = _get_sample_inputs(geometry, "state_penalty")
    if err is not None:
        return err
    reader = "validate_composed_wall"
    nodes = _grid_nodes(geometry, dimension)
    worst: tuple[float, float, np.ndarray, float, np.ndarray, float, float] | None = None
    failure: str | None = None
    shapes: dict[str, str] = {}
    for t in (float(s) for s in times):
        points = _probe_points(geometry, x_sample)
        if nodes is not None:
            try:
                largest = _largest_wall_node(walled, unwalled, nodes, t, reader)
            except SourceShapeError as e:
                shapes.setdefault("state_penalty", f"{e}, at t={t:.4g}, reading the potential at the grid's nodes")
            except SourceEvaluationError as e:
                if failure is None:
                    failure = f"at t={t:.4g}, reading the potential at the grid's nodes: {e}"
            else:
                if not any(np.array_equal(largest, q) for q in points):
                    points.append(largest)
        for x in points:
            for m, p in _consistency_probes(dimension):
                try:
                    expected = -float(_potential_at(walled, t, x, reader) - _potential_at(unwalled, t, x, reader))
                    h_walled = float(_read_source(walled, t=t, x=x, p=p, m=m, method="__call__", reader=reader))
                    h_unwalled = float(_read_source(unwalled, t=t, x=x, p=p, m=m, method="__call__", reader=reader))
                except SourceShapeError as e:
                    shapes.setdefault("state_penalty", f"{e} At {_probe_text(t, x, m, p)}.")
                    continue
                except SourceEvaluationError as e:
                    if failure is None:
                        failure = f"at {_probe_text(t, x, m, p)}: {e}"
                    continue
                moved = h_walled - h_unwalled
                gap = abs(moved - expected)
                scale = max(abs(moved), abs(expected))
                above_noise = gap > _WITNESS_NOISE_SAFETY * float(np.finfo(float).eps) * max(scale, 1.0)
                if above_noise and gap > _WITNESS_RTOL * scale and (worst is None or gap > worst[0]):
                    worst = (gap, t, x, m, p.copy(), moved, expected)
    for name, where in shapes.items():
        result.add_error(where, location=name)
    if failure is not None:
        result.add_warning(f"Could not compare the composed soft wall {failure}", location="state_penalty")
    if worst is not None:
        _gap, t, x, m, p, moved, expected = worst
        result.add_error(
            f"The soft wall composed into {type(walled).__name__}'s potential does not reach its __call__ at "
            f"{_probe_text(t, x, m, p)}: __call__ moved by {moved:.6g}, and the wall moves H by {expected:.6g} there. "
            f"Its __call__ does not read the potential the wall was composed into, so every solve would drop the wall "
            f"(#2002, #2901).",
            location="state_penalty",
            suggestion="Read the potential through super().__call__, or fold the penalty into your own Hamiltonian",
        )
    return result
