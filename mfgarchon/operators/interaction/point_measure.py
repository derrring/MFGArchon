"""What a coupling cost binds against: a point measure, points with one quadrature weight each (#1642 stage 1b-2).

A coupling cost is a specification of F[m] free of any discretization (`coupling_costs`). Binding it means choosing
the points F's flat derivative is read at and the weights its integrals are taken with. That pair is a
`PointMeasure`, a small protocol, so a grid, a finite-element space and a collocation cloud each supply their own
and every cost binds through one route:

- `GridPointMeasure`: a problem's grid nodes in the grid's C order, with the problem's measure (its trapezoid
  weights);
- a finite-element space's dofs with its consistent-mass row sums (stage 1b-2);
- a collocation cloud's points with their clipped Voronoi cells (#2761, item 7).

`bind_on_measure` is the one place a measure is put to use. It checks the measure's contract and refuses a cost
that cannot bind on a point measure. A cost binds through `bind_on`, the protocol `BindsOnPointMeasure`.
"""

from __future__ import annotations

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
    """``measure``'s points, shape ``(n, d)``, and weights, shape ``(n,)``, once its contract holds; refused by name
    otherwise. The contract: points of shape ``(n, d)``, one finite positive weight per point in an array of shape
    ``(n,)``, and a domain measure that is one positive finite number, which the weights sum to within
    `WEIGHT_SUM_RTOL`. The shapes are checked as given, not reshaped, because the measure itself is what a cost's
    ``bind_on`` receives."""
    if not isinstance(measure, PointMeasure):
        raise TypeError(
            f"{owner}: {type(measure).__name__} is not a PointMeasure, an object with points, weights and "
            "domain_measure (#1642)."
        )
    points = np.asarray(measure.points)
    weights = np.asarray(measure.weights)
    total = np.asarray(measure.domain_measure)
    problems = []
    if points.ndim != 2 or not _real(points):
        problems.append(f"points of shape {points.shape} and dtype {points.dtype}, not real numbers of shape (n, d)")
    elif weights.shape != (points.shape[0],) or not _real(weights):
        problems.append(
            f"weights of shape {weights.shape} and dtype {weights.dtype} for {points.shape[0]} points, not real "
            f"numbers of shape ({points.shape[0]},)"
        )
    elif not np.all(np.isfinite(points)) or not np.all(np.isfinite(weights)):
        problems.append("non-finite points or weights")
    elif np.any(weights <= 0):
        problems.append(f"a weight that is not positive, {float(weights.min())!r}")
    elif total.shape != () or not _real(total) or not np.isfinite(total) or total <= 0:
        problems.append(f"a domain measure of {measure.domain_measure!r}, not one positive finite number")
    elif not np.isclose(weights.sum(), total, rtol=WEIGHT_SUM_RTOL, atol=0.0):
        problems.append(f"weights summing to {float(weights.sum())!r}, not to the domain's measure {float(total)!r}")
    if problems:
        raise ValueError(
            f"{owner}: the point measure {type(measure).__name__} is not a quadrature on its domain: {problems[0]}. "
            "A point measure's weights are positive and sum to the domain's measure (#1642)."
        )
    return points.astype(float), weights.astype(float)


def _real(array: NDArray) -> bool:
    """Whether ``array`` holds real numbers, integer or floating, and not booleans, complex numbers or objects."""
    return bool(np.issubdtype(array.dtype, np.integer) or np.issubdtype(array.dtype, np.floating))


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
    integrates over: trapezoid weights over the nodes. It is the grid's box wherever the nodes reach the bounds, which
    the default grid's do. Where they do not, as for cell-centred nodes or a periodic axis whose endpoint is excluded,
    the problem's measure integrates over less than the box, every integral it takes is off by O(h), and that is the
    measure's defect, not this adapter's (#2769). Its fix returns ``domain_measure`` to the box in the same change.

    It is the problem's grid measure and nothing else: `InteractionCost` reads the grid from ``problem`` for the
    convolution's FFT path, and the problem's topology to refuse a periodic axis (#2725). So it refuses subclassing,
    and its points and weights are read-only copies: a measure that changed either would be accepted and ignored on
    that path. Another measure on the grid's points implements `PointMeasure` itself.
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
        coordinates = getattr(geometry, "coordinates", None)
        if coordinates is None:
            raise TypeError(
                f"GridPointMeasure: the problem's geometry is a {type(geometry).__name__}, which has no grid "
                "coordinates; a grid's point measure is a TensorProductGrid's (#1642)."
            )
        self.problem = problem
        points = np.asarray(geometry.get_spatial_grid(), dtype=float)
        self._points = _read_only(points.reshape(points.shape[0], -1))
        self._weights = _read_only(np.asarray(problem.spatial_measure().weights, dtype=float).ravel())
        self._domain_measure = float(np.prod([np.ptp(np.asarray(axis, dtype=float)) for axis in coordinates]))

    @property
    def points(self) -> NDArray:
        return self._points

    @property
    def weights(self) -> NDArray:
        return self._weights

    @property
    def domain_measure(self) -> float:
        return self._domain_measure


def _read_only(array: NDArray) -> NDArray:
    """A copy of ``array`` that refuses writes."""
    out = np.array(array, dtype=float)
    out.flags.writeable = False
    return out
