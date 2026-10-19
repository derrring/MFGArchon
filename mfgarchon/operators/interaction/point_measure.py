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
    - ``weights``: shape ``(n,)``, non-negative, summing to ``domain_measure`` (checked where the measure is put to
      use, `bind_on_measure`).
    - ``domain_measure``: |Omega|, the measure of the geometry as built, obstacles excluded. It is read from the
      geometry and never derived from the weights: a measure whose ``domain_measure`` is its own weights' sum passes
      the check whatever its weights are, so the check would hold nothing.
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
    otherwise. The contract: one finite weight per point, none negative, summing to ``domain_measure`` within
    `WEIGHT_SUM_RTOL`."""
    if not isinstance(measure, PointMeasure):
        raise TypeError(
            f"{owner}: {type(measure).__name__} is not a PointMeasure, an object with points, weights and "
            "domain_measure (#1642)."
        )
    points = np.asarray(measure.points, dtype=float)
    if points.ndim == 1:
        points = points[:, None]
    weights = np.asarray(measure.weights, dtype=float).ravel()
    total = float(measure.domain_measure)
    problems = []
    if points.ndim != 2 or weights.shape != (points.shape[0],):
        problems.append(f"{weights.size} weights for points of shape {np.shape(measure.points)}")
    elif not np.all(np.isfinite(points)) or not np.all(np.isfinite(weights)):
        problems.append("non-finite points or weights")
    elif np.any(weights < 0.0):
        problems.append(f"a negative weight, {weights.min()!r}")
    elif not np.isfinite(total) or total <= 0.0:
        problems.append(f"a domain measure of {total!r}")
    elif not np.isclose(weights.sum(), total, rtol=WEIGHT_SUM_RTOL, atol=0.0):
        problems.append(f"weights summing to {weights.sum()!r}, not to the domain's measure {total!r}")
    if problems:
        raise ValueError(
            f"{owner}: the point measure {type(measure).__name__} is not a quadrature on its domain: {problems[0]}. "
            "A point measure's weights are non-negative and sum to the domain's measure (#1642)."
        )
    return points, weights


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
    ``problem.spatial_measure().weights``, on the grid's box. Its ``domain_measure`` is the box's volume from
    ``geometry.get_bounds()``, read independently of the weights.

    It keeps ``problem``: `InteractionCost` reads the grid from it for the convolution's FFT path, and the
    problem's topology to refuse a periodic axis (#2725).
    """

    def __init__(self, problem: Any) -> None:
        self.problem = problem
        geometry = problem.geometry
        points = np.asarray(geometry.get_spatial_grid(), dtype=float)
        self._points = points.reshape(points.shape[0], -1)
        self._weights = np.asarray(problem.spatial_measure().weights, dtype=float).ravel()
        lower, upper = (np.asarray(b, dtype=float) for b in geometry.get_bounds())
        self._domain_measure = float(np.prod(upper - lower))

    @property
    def points(self) -> NDArray:
        return self._points

    @property
    def weights(self) -> NDArray:
        return self._weights

    @property
    def domain_measure(self) -> float:
        return self._domain_measure
