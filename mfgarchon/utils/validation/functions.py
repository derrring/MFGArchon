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


def _floats_entry_by_entry(value: Any) -> np.ndarray:
    """What a source returned, as floats: an object array entry by entry, since numpy alone reads ``None`` there as NaN.
    A return that holds no number raises (``TypeError``) rather than reading as a NaN value (#2901, #2974)."""
    array = np.asarray(value)
    if array.dtype == object:
        return np.array([float(v) for v in array.reshape(-1).tolist()]).reshape(array.shape)
    return np.asarray(array, dtype=float)


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
    # The reader's one conversion, for every form.
    try:
        floats = _floats_entry_by_entry(value)
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


def _has_a_grid_probe(geometry: GeometryProtocol) -> bool:
    """Whether a construction check's grid probe applies on ``geometry`` (#2901, stage 1c-iii).

    The grid probe (`_get_sample_inputs`) hands a Hamiltonian's sources x as one point's coordinates, p of the
    coordinate dimension and m one number. A check reads a source in the forms its readers hand it, and the geometry's
    declared type, ``geometry_type``, which ``MFGProblem`` reads to choose its construction, decides which solvers
    those are. On a geometry that declares ``GeometryType.NETWORK`` they are NetworkHJBSolver, policy iteration and the
    network FP, which accept only a network problem and hand x a node, and p and m one entry per node. So no grid probe
    applies there, whatever the source is or how it is bound. Each check that takes a geometry reads this first and
    returns before reading, the door check's caller in ``MFGProblem`` reads it before handing points, and
    `_get_sample_inputs` refuses to build a grid probe where it is false. The network HJB solvers accept a problem by
    its node data rather than its declared type, so a network geometry that declares another type is probed, and
    refused there (#2972).

    Every other geometry type keeps it. The solvers the stage's census saw reading a problem on a grid, an implicit
    domain, a mesh or a maze (``GeometryType.MAZE``) hand x a point's coordinates or a batch of them; on a maze that is
    the particle FP, which reads the control at its particles' coordinates. On a point cloud it saw none read H.

    Nothing compares a network Hamiltonian's sources. With the built-in control, ``__call__``, the rates
    ``optimal_control`` returns and ``source_term`` come from one formula. With a custom ``hamiltonian_func``, both
    readers of the rates, ``FPNetworkSolver`` and ``NetworkPolicyIterationHJBSolver``, refuse it (#1545). A Hamiltonian
    handed to ``MFGProblem`` on a network through a model or its components, a ``NetworkHamiltonian`` subclass among
    them, makes a problem no network solver accepts: the two HJB solvers refuse it by name, and ``FPNetworkSolver``
    raises ``AttributeError`` (#2970). No network solver reads ``dm`` or ``dp``.
    """
    from mfgarchon.geometry import GeometryType

    return geometry.geometry_type is not GeometryType.NETWORK


def _draws_its_nodes(geometry: GeometryProtocol) -> bool:
    """Whether ``geometry`` draws its points afresh at each ``get_spatial_grid()``, by its declared type: an implicit
    domain samples a point cloud from its SDF at each call (#2778). So its problem's arrays and a check's reads stand
    at different points, and no read pairs a node with its entry in ``m_initial`` (#2974)."""
    from mfgarchon.geometry import GeometryType

    return geometry.geometry_type is GeometryType.IMPLICIT


def _get_sample_inputs(
    geometry: GeometryProtocol,
    location: str,
) -> tuple[np.ndarray, np.ndarray, float, int, ValidationResult | None]:
    """Extract a sample point (x, p, m) from geometry for validation.

    A geometry with no grid probe (`_has_a_grid_probe`) gets none: a caller that asks for one has not read the owner.

    Returns:
        (x_sample, p_sample, m_sample, dimension, error_result)
        If error_result is not None, the caller should return it immediately.
    """
    if not _has_a_grid_probe(geometry):
        raise AssertionError(
            f"{location}: a grid probe was asked of a {type(geometry).__name__}, which has none; read "
            f"_has_a_grid_probe before probing (#2901)"
        )
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
    m_initial: np.ndarray | None = None,
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
        m_initial: The problem's initial density, one value per node, which `validate_hamiltonian` reads at its sample
            node (#2974). A problem hands it; without it the sample's m is 1.

    Returns:
        ValidationResult with any issues found

    Issue #686: Custom function validation
    Issue #1642 (C1): consistency findings propagate to is_valid
    """
    results: list[ValidationResult] = []

    if hamiltonian is not None:
        results.append(validate_hamiltonian(hamiltonian, geometry, m_initial=m_initial))

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


def _interior_probe_nodes(nodes: np.ndarray) -> list[np.ndarray]:
    """`_probe_nodes` off the box's boundary, the centre first: the nodes `_refuse_a_non_finite_source` reads an in-domain
    callable at, and the first the one `validate_hamiltonian` reads a Hamiltonian's own ``__call__`` at (#2974). A grid
    with an axis of 2 nodes has none."""
    lower, upper = nodes.min(axis=0), nodes.max(axis=0)
    return [q for q in _probe_nodes(nodes) if not bool(np.any((q == lower) | (q == upper)))]


def _refuse_a_non_finite_node_source(
    problem: Any, operators: list[Any], hjb_times: list[float], count: int, held: Any, read: Any, first_read: Any
) -> None:
    """`_refuse_a_non_finite_source` on a network (#2974). The network HJB solvers read a held source in the node form,
    an integer node: ``NetworkHJBSolver`` at the stages of its ODE and policy iteration at t_0 ... t_{Nt-1}, so the one
    grid time both read is t_0. A source that also takes p or m is handed the arrays the caller gives the solver, not
    the problem's, and none is held (``NetworkHamiltonian.NOT_HELD``). A held source is read at every node, a node a
    ``NodeBC`` governs among them: policy iteration hands it there too, and a non-finite value there stops its
    convergence. A ``NodeBC``'s callable value is read at its nodes at the HJB side's times, which policy iteration
    hands it, through `_read_constant` as the graph applicator reads it."""
    from mfgarchon.core.hamiltonian import held_sources
    from mfgarchon.geometry.boundary.types import _read_constant
    from mfgarchon.types.callable_protocols import bound_attribute, spread_samples

    def sampled(nodes: list[int]) -> list[int]:
        if len(nodes) <= 5:
            return nodes
        return [nodes[0], nodes[-1], *(nodes[int(i)] for i in spread_samples(len(nodes), 3))]

    try:
        conditions = problem.get_boundary_conditions()
    except _EVALUATION_ERRORS:
        conditions = None
    callables: list[tuple[Any, list[int]]] = []
    for condition in tuple(getattr(conditions, "node_bcs", ()) or ()):
        try:
            nodes = [int(n) for n in condition.get_nodes(count)]
        except _EVALUATION_ERRORS:
            continue
        # A CUSTOM condition takes its rule apart and a value of 0 (`NodeBC`), so a callable value is a value's.
        if callable(condition.value):
            callables.append((condition, nodes))
    nodes_read = sampled(list(range(count)))
    for operator in operators:
        for parameter, (attr, slots) in held_sources(operator).items():
            if "node" not in slots.order or not first_read(getattr(operator, attr, None), slots):
                continue
            bound = bound_attribute(operator, attr, slots, role=parameter)
            name = f"{type(operator).__name__}'s {parameter}"
            for node in nodes_read:
                neighbors = operator.network_data.get_neighbors(node)
                read(
                    name,
                    lambda b=bound, q=slots, n=node, k=neighbors: held(b, q, t=0.0, node=n, neighbors=k),
                    f"at node {node}, t=0",
                )
    for condition, nodes in callables:
        for node in sampled(sorted(set(nodes))):
            for t in hjb_times:
                try:
                    returned = condition._reader(t=t, node=node)
                except _EVALUATION_ERRORS:
                    continue
                _read_constant(
                    returned,
                    kind=f"NodeBC {condition.name!r}: value (what its callable returned at node {node}, t={t:g})",
                    where="the graph applicator",
                    returned=True,
                    advice="return one real number",
                )


def _refuse_a_non_finite_source(problem: Any, *, after_binding: bool = False) -> None:
    """Refuse a caller-supplied value or callable that is non-finite where the solvers will read it, where it enters, by
    name (#2974). ``MFGProblem`` calls it twice. First, after it has built ``m_initial`` and ``u_terminal`` and before
    the consistency check and the soft wall's composition, for everything but the coupling costs and what takes v.
    Then, with ``after_binding``, right after it binds the coupling costs, where it always has: for the costs and for
    what takes v (`_refuse_a_non_finite_read_of_the_bound_terms`). Nothing between the two reads a source the second
    reads.

    It reads only states that every `NumericalScheme` that solves the problem hands each callable, as #2974's
    recording census over the enum found them and its R1 test pins them, and each through the function the solve reads
    it with, handed what the iterator's cold start hands: m from ``get_m_init``, v from ``terminal_value`` at it.
    Expert-mode solvers, which ``problem.solve`` is handed rather than builds, and the iterator's warm start
    (``set_warm_start_data``), which ``problem.solve`` does not offer, are outside that census.

    - ``m_initial``, ``u_terminal``, a scalar or array volatility, ``T`` and ``state_penalty_scale`` whole;
    - times: nodes of the problem's time grid only, t_n = n dt. The interior nodes t_1, t_{Nt // 2} and t_{Nt - 1},
      which every scheme reads; t_0 as well for the potential and the Hamiltonian's other time-dependent sources, a
      callable volatility and a boundary callable; and T alone for the terminal coupling. ``source_term_hjb`` and
      ``coupling_cost`` are read at the interior nodes only, since the semi-Lagrangian schemes read them at
      t_1 ... T, and so is ``source_term_fp``, since the finite-volume ones read it at t_0 ... t_{Nt-1}. At Nt = 1
      there is no interior node: the endpoint reads alone remain;
    - space, on a geometry with a grid probe (`_has_a_grid_probe`) whose nodes the arrays hold and that does not draw
      them afresh (`_draws_its_nodes`): the probe nodes off the box's boundary (`_interior_probe_nodes`). On a grid that
      is every node a solve reads but a Dirichlet node, whose value does not reach the solution; on a mesh whose
      boundary is not the box, some of them lie on the mesh's boundary, which the solve reads too. There it reads a
      callable volatility (``CoefficientField.read_callable``, the read ``evaluate_at`` makes), ``source_term_hjb``,
      ``source_term_fp`` (``_call_problem_source``), ``state_penalty`` (``wall_values``) and every source a Hamiltonian
      or Lagrangian holds (`held_sources`), each callable once, named by the operator the caller gave: a potential as
      the Hamiltonians' evaluation reads it (``_read_of_the_potential``), a density term on a batch
      (``_read_of_the_density``). A held source declared with any other slots is the check's own error, and raises. A
      boundary segment's callable value is read at the boundary probe nodes it governs (``segment_for_point``);
    - m and v: the first Picard iteration hands the HJB the initial density at every t_n, and v the terminal value at
      it, ``terminal_value(get_m_init())``, which is ``u_terminal`` without a terminal coupling; the FP is handed the
      initial density, with v only at T. So a callable that takes them is handed those values at its nodes, and the
      FP's source, if it takes v, is not read: no finite-volume scheme reads it at T. Where a terminal coupling is
      declared and the problem binds none (off a grid), v is the HJB solver's binding's, and what takes v is not read.
      ``coupling_cost`` is read through ``_coupling_part`` at the initial density and the interior nodes,
      ``terminal_coupling`` through ``terminal_value``, at T, and ``nonlocal_operator`` applied to that v;
    - on a network (`_refuse_a_non_finite_node_source`), a held source in the node form at t_0, the one grid time both
      network HJB solvers read, at sampled nodes, and a ``NodeBC``'s callable value at its nodes at the
      HJB side's times.

    A scalar its own object refuses where it is built is not read again: a control cost's ``lambda_``, a bounded
    control's ``max_control``, ``QuadraticMFGHamiltonian``'s ``coupling_coefficient``, a radial kernel's parameters
    and a boundary segment's constants in the fields its type reads (``FIELDS_READ``). A volatility override handed
    to a solver is refused by ``resolve_volatility_override``.

    A Hamiltonian's own ``__call__``, ``dp`` and ``optimal_control`` take p, which construction does not know, and are
    not read here. ``validate_hamiltonian`` reads ``__call__`` at one state, inside these reads, and the consistency
    check compares the three.

    **What it does not see.** A source finite at these reads and non-finite elsewhere: at a boundary node, which the
    solve reads under no-flux; at another time, m or v; or at a state the solve reaches. The point-of-use checks see it
    (#2975, #2976). Nor, at Nt = 1, the sources and the coupling cost; on a grid with an axis of 2 nodes, which has no
    probe node off the boundary, what is read there (a callable volatility, the held sources, the two source terms
    and the wall); or on a network, a source that takes p or m, which the solvers are handed by their caller. A read
    that raises one of ``_EVALUATION_ERRORS`` is not this fact and is left to the readers at use.

    A return that holds no number (None) is judged as the read made here converts it. A held potential or density term
    is read as the Hamiltonians' evaluation reads it, and the wall and the costs through the functions the solve reads
    them with; each converts as numpy does, so a None is NaN, and is refused by name, whichever reader the solve then
    uses. Other readers of a potential or a density term raise on a None (#2901), among them ``SeparableLagrangian``'s
    call, which a semi-Lagrangian solve with a non-smooth control cost reaches. The source terms, a callable volatility
    and a network node's source are read as they return, before their readers' checks, and those readers raise on a
    None at use, so it is left to them (``_floats_entry_by_entry``).
    """
    from mfgarchon.core.hamiltonian import (
        HamiltonianBase,
        LagrangianBase,
        _own_rows,
        _read_of_the_density,
        _read_of_the_potential,
        held_sources,
    )
    from mfgarchon.core.mfg_problem import _call_problem_source, wall_values
    from mfgarchon.geometry.boundary import BoundaryConditions
    from mfgarchon.geometry.boundary.providers import is_provider
    from mfgarchon.geometry.boundary.types import _read_constant
    from mfgarchon.types.callable_protocols import (
        DENSITY_SLOTS,
        POTENTIAL_SLOTS,
        REGION_SLOTS,
        SOURCE_TERM_SLOTS,
        _remembered,
    )
    from mfgarchon.utils.validation import ValidationError

    if after_binding:
        # The second phase reads the coupling costs and what takes v, and nothing else: with none of them, it has
        # nothing to read, and a large grid's nodes are not built a second time for it.
        source = problem.source_term_hjb
        takes_v = callable(source) and "v" in _remembered(source, SOURCE_TERM_SLOTS, "source_term_hjb").passes
        if not (
            takes_v
            or problem.nonlocal_operator is not None
            or problem._coupling_energy is not None
            or problem._terminal_energy is not None
        ):
            return

    result = ValidationResult()

    def refuse(name: str, value: Any, where: str = "") -> None:
        values = np.asarray(value, dtype=float)
        if not np.all(np.isfinite(values)):
            bad = int(np.sum(~np.isfinite(values)))
            at = f" {where}" if where else ""
            result.add_error(
                f"{name} is non-finite where the solvers read it: {bad} of {values.size} values{at} are NaN or "
                "infinite (#2974).",
                location=name,
            )

    def read(name: str, call: Any, where: str) -> None:
        # ``call`` is the solve's own read of the source, so what it returns is what the solver reads. A return that
        # holds no number reaches here only from a read that converts nothing, whose reader at use raises on it, and is
        # left to that reader, as one that cannot be evaluated is.
        try:
            value = _floats_entry_by_entry(call())
        except _EVALUATION_ERRORS:
            return
        refuse(name, value, where)

    def held(bound: Any, slots: Any, **state: Any) -> Any:
        # A node's held source is handed its own slots only, as the network solvers hand them.
        return bound(**{slot: value for slot, value in state.items() if slot in slots.order})

    def stop_if_refused() -> None:
        if not result.is_valid:
            raise ValidationError(result)

    dt, steps = float(problem.dt), int(problem.Nt)
    interior_steps = sorted({1, steps // 2, steps - 1} - {0, steps})
    interior = [n * dt for n in interior_steps]
    hjb_steps = [0, *interior_steps]
    hjb_times = [n * dt for n in hjb_steps]
    # m is the initial density the iterator's cold start reads (`get_m_init`).
    m_all = np.asarray(problem.get_m_init(), dtype=float).reshape(-1)
    volatility = problem.volatility
    if not after_binding:
        refuse("T", float(problem.T))
        refuse("m_initial", problem.m_initial)
        refuse("u_terminal", problem.u_terminal)
        refuse("state_penalty_scale", problem.state_penalty_scale)
        if volatility is not None and not callable(volatility):
            refuse("volatility", volatility)
        # The arrays and T are the state every read below is made at.
        stop_if_refused()

    components = problem.components
    # The operators that hold sources, the caller's own first, then the one derived from it and any it holds (a dual's
    # Lagrangian). A derived operator holds the caller's callables, so each callable is read once, and named by the
    # operator the caller gave.
    operators: list[Any] = []
    for candidate in (
        getattr(components, "hamiltonian", None),
        getattr(components, "lagrangian", None),
        getattr(components, "_hamiltonian_class", None),
        getattr(components, "_lagrangian_class", None),
    ):
        if isinstance(candidate, (HamiltonianBase, LagrangianBase)) and all(candidate is not o for o in operators):
            operators.append(candidate)
    for operator in list(operators):
        for inner in vars(operator).values():
            if isinstance(inner, (HamiltonianBase, LagrangianBase)) and all(inner is not o for o in operators):
                operators.append(inner)
    read_already: set[tuple[int, tuple[str, ...]]] = set()

    def first_read(source: Any, slots: Any) -> bool:
        key = (id(source), slots.order)
        if source is None or key in read_already:
            return False
        read_already.add(key)
        return True

    if problem.geometry is not None and not _has_a_grid_probe(problem.geometry):
        if not after_binding:
            _refuse_a_non_finite_node_source(problem, operators, hjb_times, m_all.size, held, read, first_read)
            stop_if_refused()
        return

    # The callables are read at nodes whose m and v the arrays hold, so a geometry whose nodes are not the arrays'
    # entries, or that draws them afresh at each read, has its arrays and scalars read, above, and nothing else.
    geometry = problem.geometry
    grid = None if geometry is None or _draws_its_nodes(geometry) else _grid_nodes(geometry, geometry.dimension)
    if grid is None or grid.shape[0] != m_all.size:
        stop_if_refused()
        return
    # In space, the probe nodes off the box's boundary (`_interior_probe_nodes`). On a grid that is the box's interior,
    # every node a solve reads but a Dirichlet node, which is handed its callable and imposes its value; on a mesh whose
    # boundary is not the box, some of them lie on the mesh's boundary, which the solve reads too. The first is the
    # centre, the node `validate_hamiltonian` reads a Hamiltonian's own __call__ at.
    points = _interior_probe_nodes(grid)
    rows = _node_rows(grid, points, geometry) if points else []
    probe = np.asarray(points, dtype=float).reshape(len(points), grid.shape[1])
    x = tuple(probe[:, k] for k in range(grid.shape[1]))
    m = m_all[rows]
    if after_binding:
        _refuse_a_non_finite_read_of_the_bound_terms(problem, grid, probe, rows, m_all, interior, read, stop_if_refused)
        return

    if points:
        # In the first Picard iteration the HJB is handed m_initial at every t_n, and the FP m_initial. The source terms
        # are read through `_call_problem_source`, the one place the solve invokes them, at the probe rows; one that
        # takes v is read after the coupling costs are bound (`_refuse_a_non_finite_read_of_the_bound_terms`).
        if callable(volatility):
            field = problem.get_diffusion_coefficient_field(field_name="volatility", dimension=grid.shape[1])
            for n in hjb_steps:
                read(
                    "volatility",
                    lambda k=n: field.read_callable(k, probe, m, problem.dt),
                    f"at the interior probe nodes, t={n * dt:g}",
                )
        for operator in operators:
            for parameter, (attr, slots) in held_sources(operator).items():
                # A node's source, held by a NetworkHamiltonian on a geometry that is not a network, is not read at x:
                # that pairing is #2972's.
                if "node" in slots.order or not first_read(getattr(operator, attr, None), slots):
                    continue
                name = f"{type(operator).__name__}'s {parameter}"
                # Each through the solve's own read of its kind, so what it returns is judged as the solver reads it.
                if slots == POTENTIAL_SLOTS:
                    for t in hjb_times:
                        read(
                            name,
                            lambda o=operator, a=attr, s=t: _read_of_the_potential(o, a, x, s),
                            f"at the interior probe nodes, t={t:g}",
                        )
                elif slots == DENSITY_SLOTS:
                    read(
                        name,
                        lambda o=operator, a=attr: _read_of_the_density(o, a, m, _own_rows(m), m.size),
                        "at the interior probe nodes",
                    )
                else:
                    raise TypeError(
                        f"{type(operator).__name__}.HELD_SOURCES declares {parameter!r} with the slots {slots.order}, "
                        "which no reader of a held source on a grid reads: the construction check reads a potential "
                        "(POTENTIAL_SLOTS) and a density term (DENSITY_SLOTS), each as the solve reads it. Read the new "
                        "kind there as its solver does, or the source is not checked (#2974)."
                    )
        for field_name in ("source_term_hjb", "source_term_fp"):
            source = getattr(problem, field_name, None)
            # A source that takes v: the FP's is handed u_terminal as v at T alone, which no finite-volume scheme
            # reads, so it is not read; the HJB's is read after the binding, with the v the solve hands.
            if not callable(source) or "v" in _remembered(source, SOURCE_TERM_SLOTS, field_name).passes:
                continue
            for t in interior:
                read(
                    field_name,
                    lambda f=field_name, s=t: _call_problem_source(problem, f, t=s, x=probe, m=m, v=None),
                    f"at the interior probe nodes, t={t:g}",
                )
        if callable(problem.state_penalty):
            wall = _remembered(problem.state_penalty, REGION_SLOTS, "state_penalty")
            read("state_penalty", lambda: wall_values(wall, x), "at the interior probe nodes")
    stop_if_refused()

    # A boundary segment's callable value, at each probe node on the box's boundary, read with the segment that governs
    # that node, as the solvers read it: the BC is the problem's one (`get_boundary_conditions`, #2530), and the
    # segment the one `segment_for_point` names. A BC or a placement that its owner refuses is left to its readers.
    try:
        conditions = problem.get_boundary_conditions()
    except _EVALUATION_ERRORS:
        return
    if not isinstance(conditions, BoundaryConditions):
        return
    lower, upper = grid.min(axis=0), grid.max(axis=0)
    for q in _probe_nodes(grid):
        if not bool(np.any((q == lower) | (q == upper))):
            continue
        try:
            segment = conditions.segment_for_point(q, geometry=problem.geometry)
        except _EVALUATION_ERRORS:
            continue
        if not callable(segment.value) or is_provider(segment.value):
            continue
        binding = segment.value_binding()
        for t in hjb_times:
            try:
                returned = binding(t=t, x=tuple(float(c) for c in q))
            except _EVALUATION_ERRORS:
                continue
            _read_constant(
                returned,
                kind=f"BCSegment {segment.name!r} (what its callable returned at t={t:g})",
                where="a solver that reads it",
                returned=True,
            )


def _refuse_a_non_finite_read_of_the_bound_terms(
    problem: Any,
    grid: np.ndarray,
    probe: np.ndarray,
    rows: list[int],
    m_all: np.ndarray,
    interior: list[float],
    read: Any,
    stop_if_refused: Any,
) -> None:
    """`_refuse_a_non_finite_source`'s reads that need the coupling costs bound, made right after `MFGProblem` binds
    them, where it always has (#2974). Each through the function the solve reads it with:

    - ``coupling_cost`` through `_coupling_part`, with the problem's own binding, at ``m_initial`` and the interior
      nodes;
    - ``terminal_coupling`` through ``problem.terminal_value(m_initial)``, the iterator's cold start, at T;
    - what takes v -- ``source_term_hjb`` taking v, through `_call_problem_source`, and ``nonlocal_operator`` -- at the
      v the first iteration hands: ``terminal_value(m_initial)``, which every row of the cold start holds. Where a
      terminal coupling is declared and the problem binds none (off a grid), that v is the HJB solver's binding's, which
      construction does not have: ``terminal_value`` refuses there, and what takes v is not read.
    """
    from mfgarchon.core.mfg_problem import _call_problem_source, _coupling_part, problem_coupling_binding
    from mfgarchon.types.callable_protocols import SOURCE_TERM_SLOTS, _remembered

    binding = problem_coupling_binding(problem)
    if binding is not None:
        for t in interior:
            read(
                "coupling_cost",
                lambda s=t: _coupling_part(binding.energy, binding.nodes, m_all, grid, s),
                f"at m_initial, t={t:g}",
            )
    if problem._terminal_energy is not None:
        read("terminal_coupling", lambda: problem.terminal_value(m_all), f"at m_initial, t={float(problem.T):g}")
    stop_if_refused()
    try:
        v_all = np.asarray(problem.terminal_value(m_all), dtype=float).reshape(-1)
    except _EVALUATION_ERRORS:
        return
    source = problem.source_term_hjb
    if rows and callable(source) and "v" in _remembered(source, SOURCE_TERM_SLOTS, "source_term_hjb").passes:
        m, v = m_all[rows], v_all[rows]
        for t in interior:
            read(
                "source_term_hjb",
                lambda s=t: _call_problem_source(problem, "source_term_hjb", t=s, x=probe, m=m, v=v),
                f"at the interior probe nodes, t={t:g}",
            )
    if problem.nonlocal_operator is not None:
        read("nonlocal_operator", lambda: problem.nonlocal_operator @ v_all, "applied to the terminal value")
    stop_if_refused()


def validate_hamiltonian(
    hamiltonian: Any,
    geometry: GeometryProtocol,
    *,
    m_initial: np.ndarray | None = None,
) -> ValidationResult:
    """
    Validate a Hamiltonian: a HamiltonianBase instance, or a raw callable of (t, x, p, m), which declares p and what else it reads.

    Supports HamiltonianBase instances (called as H(t, x, p, m) by keyword)
    and raw callables (tried with same signature).

    Checks:
    - Callable with correct signature
    - Returns float or array
    - No NaN/Inf in output: the one check at construction of a Hamiltonian's own ``__call__`` being finite, at one
      state: t 0, the centre probe node off the boundary (`_interior_probe_nodes`; the sample node on a grid with no
      interior, or on an implicit domain), p 0, and m ``m_initial``'s value there (1 without it). A source the
      Hamiltonian holds is read at that state first, by `_refuse_a_non_finite_source`, and refused there by name
      (#2974).

    On a network geometry nothing is read here: no grid probe applies there (`_has_a_grid_probe`).

    Args:
        hamiltonian: HamiltonianBase instance or callable
        geometry: Geometry for sample point
        m_initial: The initial density, one value per node, read at the sample node (#2974)

    Returns:
        ValidationResult
    """
    result = ValidationResult()
    if not _has_a_grid_probe(geometry):
        return result

    x_sample, p_sample, m_sample, dimension, err = _get_sample_inputs(geometry, "hamiltonian")
    if err is not None:
        return err
    # The node is the owner's centre probe node, where the grid has an interior: there every solve reads the
    # Hamiltonian, and the owner reads every source it holds at this state first, by name (#2974).
    nodes = None if _draws_its_nodes(geometry) else _grid_nodes(geometry, dimension)
    if nodes is not None:
        centre = _interior_probe_nodes(nodes)
        x_sample = centre[0] if centre else x_sample
        if m_initial is not None and nodes.shape[0] == np.size(m_initial):
            m_sample = float(np.ravel(m_initial)[_node_rows(nodes, [x_sample], geometry)[0]])

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

    On a network geometry nothing is read here: no grid probe applies there (`_has_a_grid_probe`).

    Args:
        derivative_func: Derivative function (bound method or callable)
        geometry: Geometry for sample point
        name: Name for error messages ("dH_dm" or "dH_dp")

    Returns:
        ValidationResult
    """
    result = ValidationResult()
    if not _has_a_grid_probe(geometry):
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


def _node_rows(nodes: np.ndarray, points: list[np.ndarray], geometry: GeometryProtocol) -> list[int]:
    """The row of ``nodes`` each of ``points`` is, for reading an array that holds one entry per node (#2974). A point
    that is no node means the geometry returned other nodes on another read, which `_draws_its_nodes` says of none
    but an implicit domain."""
    rows = []
    for point in points:
        at = np.flatnonzero(np.all(nodes == point, axis=1))
        if at.size == 0:
            raise AssertionError(
                f"{type(geometry).__name__}.get_spatial_grid() returned other nodes on another read, and "
                f"{np.asarray(point).tolist()} is not among them; `_draws_its_nodes` should say so (#2974)"
            )
        rows.append(int(at[0]))
    return rows


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

    On a network geometry nothing is probed here, and a network Hamiltonian's sources are compared nowhere
    (`_has_a_grid_probe` says why).

    Returns:
        ValidationResult; is_valid is False when a witness was exhibited, when H or a method cannot be evaluated at any
        probe, and when a source returns what the owner of its shape refuses.
    """
    result = ValidationResult()
    if not _has_a_grid_probe(geometry):
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
    is warned about, once, and a return the owner of its shape refuses is an error.

    On a network geometry nothing is read here: no grid probe applies there (`_has_a_grid_probe`)."""
    result = ValidationResult()
    if not _has_a_grid_probe(geometry):
        return result
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
