"""What a coupling cost binds against: a point measure, points with one quadrature weight each (#1642 stage 1b-2).

A coupling cost is a specification of F[m] free of any discretization (`coupling_costs`). Binding it means choosing
the points F's flat derivative is read at and the weights its integrals are taken with. That pair is a
`PointMeasure`, a small protocol, so a grid, a finite-element space and a collocation cloud each supply their own
and every cost binds through one route:

- `GridPointMeasure`: a problem's grid nodes in the grid's C order, with the problem's measure (its trapezoid
  weights);
- ``LumpedMassPointMeasure`` (``alg.numerical.fem``): HJB-FEM P1's dofs with its consistent-mass row sums, which
  ``HJBFEMSolver`` binds the cost on (stage 1b-2);
- a collocation cloud's points with their clipped Voronoi cells (#2761, item 7).

`bind_on_measure` is the one place a measure is put to use. It checks the measure's contract and refuses a cost
that cannot bind on a point measure. A cost binds through `bind_on`, the protocol `BindsOnPointMeasure`.
"""

from __future__ import annotations

import numbers
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from .energy_functionals import EnergyFunctional

# The weights must sum to the domain's measure within this relative tolerance. A grid's trapezoid weights and a P1
# space's consistent-mass row sums meet it up to round-off.
WEIGHT_SUM_RTOL = 1e-9


@runtime_checkable
class PointMeasure(Protocol):
    """Points with one quadrature weight each, on a domain of known measure.

    - ``points``: shape ``(n, d)``, in the problem's frame, in the order the density is given in. That order is the
      implementer's responsibility: no cheap check can tell a permuted measure from its own.
    - ``weights``: shape ``(n,)``, positive, summing to ``domain_measure`` (checked where the measure is put to use,
      `bind_on_measure`). Positive, not merely non-negative: every library energy refuses a zero weight
      (``validate_weights``), so the contract states that rule once.
    - ``domain_measure``: |Omega|, the measure of the geometry as built, obstacles excluded. It is read from the
      geometry and never derived from the weights: a measure whose ``domain_measure`` is its own weights' sum passes
      the check whatever its weights are, so the check would hold nothing. The grid's, `GridPointMeasure`, is the box
      its nodes span, the domain the problem's grid measure integrates over, until #2769 moves that measure onto the
      grid's box.

    The points are read as points of Euclidean space: `InteractionCost`'s direct quadrature takes their differences
    ``x - y``, so on a domain with a periodic axis the interaction is the implementer's responsibility, since a measure
    carries no topology. On the grid's measure `InteractionCost` takes the grid path, which refuses a periodic axis by
    name (#2725); HJB-FEM refuses a PERIODIC condition at construction (#1456).
    """

    @property
    def points(self) -> NDArray: ...

    @property
    def weights(self) -> NDArray: ...

    @property
    def domain_measure(self) -> float: ...


@runtime_checkable
class BindsOnPointMeasure(Protocol):
    """A coupling cost that binds on a point measure: ``bind_on(measure)`` returns the `EnergyFunctional` of F on
    those points and weights. The library's costs implement it; a user's cost may."""

    def bind_on(self, measure: PointMeasure) -> EnergyFunctional: ...


def checked_point_measure(measure: PointMeasure, owner: str) -> tuple[NDArray, NDArray]:
    """``measure``'s points, shape ``(n, d)``, and weights, shape ``(n,)``, in float64, once its contract holds;
    refused by name otherwise. The contract: real points of shape ``(n, d)`` with ``d >= 1``, one finite positive real
    weight per point in an array of shape ``(n,)``, and a domain measure that is one positive finite real number, which
    the weights sum to within `WEIGHT_SUM_RTOL`. The shapes are checked as given, not reshaped, because the measure
    itself is what a cost's ``bind_on`` receives; the values are checked in float64 whatever their precision."""
    if not isinstance(measure, PointMeasure):
        raise TypeError(
            f"{owner}: {type(measure).__name__} is not a PointMeasure, an object with points, weights and "
            "domain_measure (#1642)."
        )
    points = np.asarray(measure.points)
    weights = np.asarray(measure.weights)
    total = _one_real(measure.domain_measure)
    problems = []
    if points.ndim != 2 or points.shape[1] < 1 or not _real(points):
        problems.append(f"points of shape {points.shape} and dtype {points.dtype}, not real numbers of shape (n, d)")
    elif weights.shape != (points.shape[0],) or not _real(weights):
        problems.append(
            f"weights of shape {weights.shape} and dtype {weights.dtype} for {points.shape[0]} points, not real "
            f"numbers of shape ({points.shape[0]},)"
        )
    else:
        points, weights = points.astype(np.float64), weights.astype(np.float64)
        if not np.all(np.isfinite(points)) or not np.all(np.isfinite(weights)):
            problems.append("non-finite points or weights")
        elif np.any(weights <= 0):
            problems.append(f"a weight that is not positive, {float(weights.min())!r}")
        elif total is None or not np.isfinite(total) or total <= 0:
            problems.append(f"a domain measure of {measure.domain_measure!r}, not one positive finite real number")
        elif not np.isclose(weights.sum(), total, rtol=WEIGHT_SUM_RTOL, atol=0.0):
            problems.append(f"weights summing to {float(weights.sum())!r}, not to the domain's measure {total!r}")
    if problems:
        raise ValueError(
            f"{owner}: the point measure {type(measure).__name__} is not a quadrature on its domain: {problems[0]}. "
            "A point measure's weights are positive and sum to the domain's measure (#1642)."
        )
    return points, weights


def _real(array: NDArray) -> bool:
    """Whether ``array`` holds real numbers, integer or floating, and not booleans, complex numbers, time deltas or
    objects."""
    if np.issubdtype(array.dtype, np.timedelta64):
        return False
    return bool(np.issubdtype(array.dtype, np.integer) or np.issubdtype(array.dtype, np.floating))


def _one_real(value: Any) -> float | None:
    """``value`` as a float when it is one real number -- a Python or numpy real, a ``Fraction``, a 0-d real array --
    and None otherwise: a boolean, a complex number, a time delta, an array of another shape, None, or a real beyond
    float64's range."""
    if isinstance(value, (bool, np.bool_, np.timedelta64)):
        return None
    try:
        if isinstance(value, numbers.Real):
            return float(value)
        if isinstance(value, np.ndarray) and value.shape == () and _real(value):
            return float(value)
    except OverflowError:
        return None
    return None


def bind_on_measure(cost: Any, measure: PointMeasure) -> EnergyFunctional:
    """``cost.bind_on(measure)``: the one place a point measure is put to use. The measure's contract is checked
    here, a cost that cannot bind on a point measure is refused by its name, and the energy returned is checked to be
    an `EnergyFunctional` on the measure's points."""
    from .energy_functionals import missing_energy_functional_members

    points, _ = checked_point_measure(measure, f"{type(cost).__name__}.bind_on")
    if not isinstance(cost, BindsOnPointMeasure):
        raise NotImplementedError(
            f"{type(cost).__name__} has no bind_on(measure), so it binds only on a grid, through bind(problem). Off "
            "a grid, a coupling cost binds on the solver's own points and weights, a PointMeasure; give it "
            "bind_on (BindsOnPointMeasure, in mfgarchon.operators.interaction.point_measure) (#1642)."
        )
    energy = cost.bind_on(measure)
    missing = missing_energy_functional_members(energy)
    if missing:
        raise TypeError(
            f"{type(cost).__name__}.bind_on(measure) returned {type(energy).__name__}, which is not an "
            f"EnergyFunctional: it lacks {', '.join(missing)} (#1642)."
        )
    if np.shape(energy.weights) != (len(points),):
        raise ValueError(
            f"{type(cost).__name__}.bind_on(measure) returned an energy on {np.shape(energy.weights)} weights for a "
            f"measure of {len(points)} points (#1642)."
        )
    return energy


class GridPointMeasure:
    """A problem's grid as a point measure: the rows of ``geometry.get_spatial_grid()``, which are the grid's own
    C-order ravel and the order ``MFGProblem`` evaluates ``m_initial`` in, with the problem's measure,
    ``problem.spatial_measure().weights``.

    Its ``domain_measure`` is the volume of the box the grid's nodes span, the product over the axes of the extent of
    ``geometry.coordinates``, read from the geometry independently of the weights. That is what the problem's measure
    integrates over: trapezoid weights over the nodes. It is the grid's box wherever the nodes lie on the bounds, which
    the default grid's do. Where they do not, the problem's measure integrates over another region than the box: less
    of it for cell-centred nodes or a periodic axis whose endpoint is excluded, more for nodes beyond the bounds. Every
    integral it takes then misses or adds the region between the two boxes: O(h) where the nodes stop short of the
    bounds, and the whole overhang where they lie beyond them. That is the measure's defect, not this adapter's (#2769),
    and its fix returns ``domain_measure`` to the box in the same change.

    It holds its problem and nothing else. `InteractionCost` reads the grid and its measure from the problem for the
    convolution's grid path, and the problem's topology to refuse a periodic axis (#2725), so this measure must be
    exactly the problem's: ``points``, ``weights`` and ``domain_measure`` are derived from the problem at each read,
    the arrays as read-only copies, ``problem`` cannot be reassigned, and the class refuses subclassing. A deep copy or
    a pickle carries a copy of the problem and a shallow copy shares it; either way it stays its problem's measure.
    Another measure on the grid's points implements `PointMeasure` itself. Outside the contract: assigning
    ``__class__``, setting ``flags.writeable`` on a returned array, replacing a private attribute, a mixin whose
    ``__init_subclass__`` skips ``super()``, and patching the class's properties (``mock.patch.object``), which the grid
    path does not read. A ``weakref.proxy`` of the measure is not the measure's type, so it takes the direct path,
    where a periodic axis is not refused (#1642's condition on a measure that carries no topology).
    """

    def __init_subclass__(cls, **kwargs: Any) -> None:
        raise TypeError(
            f"{cls.__name__} subclasses GridPointMeasure, the problem's own grid measure: InteractionCost reads the "
            "grid and its measure from the problem, so other points or weights would be accepted and ignored there. "
            "A different measure on the grid's points implements PointMeasure (points, weights, domain_measure) "
            "itself, and binds by the direct quadrature on them (#1642)."
        )

    def __init__(self, problem: Any) -> None:
        geometry = problem.geometry
        if getattr(geometry, "coordinates", None) is None:
            raise TypeError(
                f"GridPointMeasure: the problem's geometry is a {type(geometry).__name__}, which has no grid "
                "coordinates; a grid's point measure is a TensorProductGrid's (#1642)."
            )
        self._problem = problem

    @property
    def problem(self) -> Any:
        return self._problem

    @property
    def points(self) -> NDArray:
        points = np.asarray(self._problem.geometry.get_spatial_grid(), dtype=float)
        return _read_only(points.reshape(points.shape[0], -1))

    @property
    def weights(self) -> NDArray:
        return _read_only(np.asarray(self._problem.spatial_measure().weights, dtype=float).ravel())

    @property
    def domain_measure(self) -> float:
        coordinates = self._problem.geometry.coordinates
        return float(np.prod([np.ptp(np.asarray(axis, dtype=float)) for axis in coordinates]))


def _read_only(array: NDArray) -> NDArray:
    """A copy of ``array`` that refuses writes."""
    out = np.array(array, dtype=float)
    out.flags.writeable = False
    return out
