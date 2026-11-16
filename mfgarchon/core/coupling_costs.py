"""Coupling costs: the values of ``Model.coupling_cost``, free of any discretization (#1642 stage 1).

A coupling cost is a specification of the interaction cost F[m]. ``bind(problem)`` returns the discrete
`EnergyFunctional` of F on that problem's nodes and measure, which is what the user's ruling 1 of 2026-10-14 asks: the
model holds a specification, and the library binds it to a discretization. The problem binds it at construction, and the
HJB carries its flat derivative ``delta F / delta m`` as a cost-signed source (`MFGProblem.coupling_energy`).

- `InteractionCost(kernel)` binds to `QuadraticInteractionEnergy` on the problem's convolution,
  ``F[m] = (1/2) integral integral K(x - y) m(x) m(y) dx dy``;
- `PotentialCost(V)` binds to `PotentialEnergy`, ``F[m] = integral V(x) m(x) dx``;
- their sum, `CombinedCost`, binds to `CombinedEnergy`; ``a + b`` builds it.

The protocol they satisfy is `mfgarchon.core.model.CouplingCost`.

Node order: every bound array is in the order of ``problem.geometry.get_spatial_grid()``'s rows, which is the grid's own
C-order ravel, the order ``MFGProblem`` evaluates ``m_initial`` in. The HJB source's coupling part flattens a density
slice the same way (`source_composition`).

Each bound energy also evaluates its flat derivative at points that are not the nodes, `PointwiseFlatDerivative`: V(x)
for a potential, the bound quadrature applied at x for an interaction. That is what a solver evaluating its source at
its own points, such as HJB-GFDM's collocation points, reads (#1642 stage 1b-1).

Every library cost binds on a point measure, `bind_on(measure)` (`mfgarchon.operators.interaction.point_measure`):
points with one quadrature weight each. For `InteractionCost` and `PotentialCost`, ``bind(problem)`` is ``bind_on`` on
the problem's grid, `GridPointMeasure`, through `bind_on_measure`, the one place a measure is put to use.
`CombinedCost.bind` binds each part through its own ``bind(problem)``, so a user's part with ``bind`` alone still binds
on a grid. A solver that integrates on its own points, such as a finite-element space, binds the same cost on its own
measure (stage 1b-2).

The model holds the specification, and this module also holds what reads a problem to bind it: `GridPointMeasure`,
`bound_energy_functional`, and the grid branch of `InteractionCost.bind_on`, which builds the convolution from the
problem's grid, measure and topology. The operators they bind to read no problem (#2938 stage 2d).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np

from mfgarchon.operators.interaction.convolution import ConvolutionCouplingOperator
from mfgarchon.operators.interaction.energy_functionals import (
    CombinedEnergy,
    PotentialEnergy,
    QuadraticInteractionEnergy,
    missing_energy_functional_members,
)
from mfgarchon.operators.interaction.point_measure import bind_on_measure

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from numpy.typing import NDArray

    from mfgarchon.core.mfg_problem import MFGProblem
    from mfgarchon.core.model import CouplingCost
    from mfgarchon.operators.interaction.energy_functionals import EnergyFunctional
    from mfgarchon.operators.interaction.kernels import RadialKernel
    from mfgarchon.operators.interaction.point_measure import PointMeasure


class _SummableCost:
    """``a + b`` for coupling costs: their `CombinedCost`, with either side any `CouplingCost`; and ``0 + a``, so
    ``sum`` of costs whose first is the library's is their sum. ``sum`` starts from ``0 + costs[0]``, which a user's
    class without ``__radd__`` cannot answer; ``CombinedCost([...])`` takes the costs in any order."""

    def __add__(self, other: object) -> CombinedCost:
        if not _is_coupling_cost(other):
            return NotImplemented
        return CombinedCost([cast("CouplingCost", self), cast("CouplingCost", other)])

    def __radd__(self, other: object) -> _SummableCost:
        if isinstance(other, int) and other == 0:
            return self
        if not _is_coupling_cost(other):
            return NotImplemented
        return CombinedCost([cast("CouplingCost", other), cast("CouplingCost", self)])


def _is_coupling_cost(obj: object) -> bool:
    from mfgarchon.core.model import CouplingCost

    return isinstance(obj, CouplingCost) and not isinstance(obj, type)


def _parts(cost: CouplingCost) -> list[CouplingCost]:
    return list(cost.costs) if isinstance(cost, CombinedCost) else [cost]


class InteractionCost(_SummableCost):
    """The interaction cost ``F[m] = (1/2) integral integral K(x - y) m(x) m(y) dx dy`` of a translation-invariant
    kernel K, whose flat derivative is ``(K * m)(x)``.

    Binds to `QuadraticInteractionEnergy` on a point measure, ``sum_j K(|x - y_j|) w_j m_j`` over its points and
    weights. On a problem's grid (`GridPointMeasure`) the convolution takes the grid path: the same quadrature, built
    from three things the problem owns. Its grid, a `TensorProductGrid`; its measure, :meth:`MFGProblem.spatial_measure`,
    whose ``weights`` become the operator's; and its topology, :meth:`MFGProblem.get_boundary_conditions` read face by
    face (`periodic_faces`). A uniform grid gets the operator's regular mode, where the FFT path is available; a graded
    grid gets its point mode with the same weights, which is direct only. The measure's points go with it, so the energy
    evaluates ``(K * m)(x)`` at points that are not nodes.

    Refused by name on a problem's grid:

    - a periodic axis. The convolution is Euclidean and does not wrap, so near one wall it would miss the mass just
      beyond the other (#2725);
    - a `CartesianGrid` that is not a `TensorProductGrid`. A geometry that is not a `CartesianGrid` at all is refused
      before, by `GridPointMeasure`.

    Args:
        kernel: The interaction kernel K, a `RadialKernel` (``GaussianKernel``, ``WendlandKernel``, ...).
        use_fft: The convolution's FFT path, as `ConvolutionCouplingOperator` takes it; on a problem's grid, the FFT
            path needs a uniform grid.
    """

    def __init__(self, kernel: RadialKernel, *, use_fft: bool | None = None):
        self.kernel = kernel
        self.use_fft = use_fft

    def bind(self, problem: MFGProblem) -> EnergyFunctional:
        return bind_on_measure(self, GridPointMeasure(problem))

    def bind_on(self, measure: PointMeasure) -> EnergyFunctional:
        if type(measure) is GridPointMeasure:
            from mfgarchon.geometry.boundary.conditions import periodic_faces
            from mfgarchon.geometry.grids.tensor_grid import TensorProductGrid

            problem = measure.problem
            geometry = problem.geometry
            if not isinstance(geometry, TensorProductGrid):
                raise TypeError(
                    f"InteractionCost: the problem's geometry is a {type(geometry).__name__}, not a "
                    "TensorProductGrid, whose shape, spacing and coordinates the grid path reads. To bind on these "
                    "points, give them with their quadrature weights as a PointMeasure (points, weights, "
                    "domain_measure) and bind there: bind_on_measure(cost, measure), from "
                    "mfgarchon.operators.interaction, builds the operator on those points with those weights (#1642)."
                )
            shape = tuple(geometry.get_grid_shape())
            periodic = periodic_faces(problem.get_boundary_conditions(), len(shape), geometry)
            wrapped = sorted({face // 2 for face, is_periodic in enumerate(periodic) if is_periodic})
            if wrapped:
                raise ValueError(
                    f"InteractionCost: axis {wrapped} of the problem's grid is periodic, and this convolution is "
                    "Euclidean: it does not wrap, so near one wall it would miss the mass just beyond the other. A "
                    "periodic interaction is #2725; pose the problem on a non-periodic grid (#1642)."
                )
            weights = problem.spatial_measure().weights
            if geometry.is_uniform:
                conv = ConvolutionCouplingOperator(
                    self.kernel,
                    grid_shape=shape,
                    spacings=geometry.get_grid_spacing(),
                    cell_volume=weights,
                    use_fft=self.use_fft,
                )
            else:
                mesh = np.meshgrid(*geometry.coordinates[: len(shape)], indexing="ij")
                points = np.stack([axis.ravel() for axis in mesh], axis=-1)
                conv = ConvolutionCouplingOperator(
                    self.kernel,
                    points=points,
                    cell_volume=np.asarray(weights, dtype=float).ravel(),
                    use_fft=self.use_fft,
                )
        else:
            if self.use_fft:
                raise ValueError(
                    f"InteractionCost(use_fft=True) binds on {type(measure).__name__}, a point measure that is not a "
                    "problem's grid: the FFT path needs a uniform grid, and on other points the interaction is the "
                    "direct quadrature on their weights. Leave use_fft unset (#1642)."
                )
            conv = ConvolutionCouplingOperator(
                self.kernel,
                points=np.asarray(measure.points).astype(np.float64),
                cell_volume=np.asarray(measure.weights).astype(np.float64),
                use_fft=self.use_fft,
            )
        return cast("EnergyFunctional", QuadraticInteractionEnergy(conv, nodes=measure.points))

    def __repr__(self) -> str:
        return f"InteractionCost({self.kernel!r}, use_fft={self.use_fft!r})"


class PotentialCost(_SummableCost):
    """The potential cost ``F[m] = integral V(x) m(x) dx``, whose flat derivative is ``V(x)``.

    Binds to `PotentialEnergy` on a point measure: V sampled at its points, its weights, and V itself, which it
    evaluates at points that are not nodes. On a problem's grid those are its nodes and its measure,
    ``problem.spatial_measure().weights`` (`GridPointMeasure`).

    Args:
        V: The potential, cost-signed (positive repels), called once per node as ``m_initial`` is: ``V(x)`` with
            ``x`` a coordinate tuple, returning one number per point. A ``V(t, x)`` is refused:
            this F is autonomous, and a time-dependent potential goes in the Hamiltonian, ``potential(t, x)``.
    """

    def __init__(self, V: Callable):
        self.V = V

    def bind(self, problem: MFGProblem) -> EnergyFunctional:
        return bind_on_measure(self, GridPointMeasure(problem))

    def bind_on(self, measure: PointMeasure) -> EnergyFunctional:
        nodes = np.asarray(measure.points)
        nodes = nodes.reshape(nodes.shape[0], -1)
        energy = PotentialEnergy.sampled_from(
            self.V, nodes, np.asarray(measure.weights, dtype=float).ravel(), owner="PotentialCost"
        )
        return cast("EnergyFunctional", energy)

    def __repr__(self) -> str:
        return f"PotentialCost({self.V!r})"


class CombinedCost(_SummableCost):
    """The sum of coupling costs, ``F = F_1 + F_2 + ...``. Binds to `CombinedEnergy` of each part's binding, on one
    measure. ``InteractionCost(K) + PotentialCost(V)`` builds it.

    Args:
        costs: The parts, each a `CouplingCost` -- the library's or the user's; a `CombinedCost` among them is
            flattened.
    """

    def __init__(self, costs: Sequence[CouplingCost]):
        flat = [part for cost in costs for part in _parts(cost)]
        if not flat:
            raise ValueError("CombinedCost: no parts. A coupling cost of zero is Model(coupling_cost=None).")
        not_costs = [
            f"the class {part.__name__}, not an instance of it" if isinstance(part, type) else type(part).__name__
            for part in flat
            if not _is_coupling_cost(part)
        ]
        if not_costs:
            raise TypeError(
                f"CombinedCost: {', '.join(not_costs)} is not a CouplingCost, an object with bind(problem) returning "
                "an EnergyFunctional (#1642)."
            )
        self.costs = flat

    def bind(self, problem: MFGProblem) -> EnergyFunctional:
        return cast("EnergyFunctional", CombinedEnergy([bound_energy_functional(cost, problem) for cost in self.costs]))

    def bind_on(self, measure: PointMeasure) -> EnergyFunctional:
        """Each part's ``bind_on`` through `bind_on_measure`, so a part that cannot bind on a point measure, a user's
        cost with ``bind`` alone, is refused by its name."""
        return cast("EnergyFunctional", CombinedEnergy([bind_on_measure(cost, measure) for cost in self.costs]))

    def __repr__(self) -> str:
        return f"CombinedCost({self.costs!r})"


def bound_energy_functional(cost: Any, problem: Any) -> EnergyFunctional:
    """``cost.bind(problem)``, refused by the cost's name where what it returns is not an `EnergyFunctional` (#1642):
    the one check for every binding, a part of a sum included, so a near-miss is refused at construction rather than
    at the first evaluation."""
    energy = cost.bind(problem)
    missing = missing_energy_functional_members(energy)
    if missing:
        raise TypeError(
            f"{type(cost).__name__}.bind(problem) returned {type(energy).__name__}, which is not an "
            f"EnergyFunctional: it lacks {', '.join(missing)} (#1642)."
        )
    return cast("EnergyFunctional", energy)


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
        from mfgarchon.geometry.base import CartesianGrid

        geometry = problem.geometry
        if not isinstance(geometry, CartesianGrid):
            raise TypeError(
                f"GridPointMeasure: the problem's geometry is a {type(geometry).__name__}, which is not a "
                "CartesianGrid, the type that declares grid coordinates and their order; a grid's point measure is a "
                "TensorProductGrid's (#1642, #2490)."
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
