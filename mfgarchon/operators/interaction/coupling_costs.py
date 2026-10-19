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

Every library cost binds on a point measure, `bind_on(measure)` (`point_measure`): points with one quadrature weight
each. For `InteractionCost` and `PotentialCost`, ``bind(problem)`` is ``bind_on`` on the problem's grid,
`GridPointMeasure`, through `bind_on_measure`, the one place a measure is put to use. `CombinedCost.bind` binds each part
through its own ``bind(problem)``, so a user's part with ``bind`` alone still binds on a grid. A solver that integrates
on its own points, such as a finite-element space, binds the same cost on its own measure (stage 1b-2).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np

from mfgarchon.operators.interaction.convolution import ConvolutionCouplingOperator
from mfgarchon.operators.interaction.energy_functionals import (
    CombinedEnergy,
    PotentialEnergy,
    QuadraticInteractionEnergy,
)
from mfgarchon.operators.interaction.point_measure import GridPointMeasure, bind_on_measure

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

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
    weights. On a problem's grid (`GridPointMeasure`) the convolution is ``ConvolutionCouplingOperator.from_problem``:
    the same quadrature, by FFT on a uniform grid, with the problem's topology, so a periodic axis is refused there by
    name (#2725). The measure's points go with it, so the energy evaluates ``(K * m)(x)`` at points that are not nodes.

    Args:
        kernel: The interaction kernel K, a `RadialKernel` (``GaussianKernel``, ``WendlandKernel``, ...).
        use_fft: The convolution's FFT path, as ``ConvolutionCouplingOperator.from_problem`` takes it.
    """

    def __init__(self, kernel: RadialKernel, *, use_fft: bool | None = None):
        self.kernel = kernel
        self.use_fft = use_fft

    def bind(self, problem: MFGProblem) -> EnergyFunctional:
        return bind_on_measure(self, GridPointMeasure(problem))

    def bind_on(self, measure: PointMeasure) -> EnergyFunctional:
        if isinstance(measure, GridPointMeasure):
            conv = ConvolutionCouplingOperator.from_problem(self.kernel, measure.problem, use_fft=self.use_fft)
        else:
            if self.use_fft:
                raise ValueError(
                    f"InteractionCost(use_fft=True) binds on {type(measure).__name__}, a point measure that is not a "
                    "problem's grid: the FFT path needs a uniform grid, and on other points the interaction is the "
                    "direct quadrature on their weights. Leave use_fft unset (#1642)."
                )
            conv = ConvolutionCouplingOperator(
                self.kernel, points=measure.points, cell_volume=measure.weights, use_fft=self.use_fft
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
            ``x`` a float in 1-D and an array of shape ``(d,)`` in n-D, returning one number. A ``V(t, x)`` is refused:
            this F is autonomous, and a time-dependent potential goes in the Hamiltonian, ``potential(t, x)``.
    """

    def __init__(self, V: Callable):
        self.V = V

    def bind(self, problem: MFGProblem) -> EnergyFunctional:
        return bind_on_measure(self, GridPointMeasure(problem))

    def bind_on(self, measure: PointMeasure) -> EnergyFunctional:
        from mfgarchon.utils.callable_adapter import CallableSignature, adapt_ic_callable

        nodes = np.asarray(measure.points, dtype=float)
        nodes = nodes.reshape(nodes.shape[0], -1)
        d = nodes.shape[1]
        in_the_hamiltonian = (
            "A time-dependent potential goes in the Hamiltonian, SeparableHamiltonian(potential=lambda t, x: ...). "
            "HJB-FDM and HJB-SL read it at each time; HJB-WENO reads a Hamiltonian at t = 0 (#2429), so with it the "
            "potential is V(0, x) for the whole horizon. There x is a point or an array of points, of shape (d,) or "
            "(N, d), so a V written for one point reads x[..., 0] in 1-D (#1642)."
        )
        try:
            signature, V = adapt_ic_callable(
                self.V, dimension=d, sample_point=float(nodes[0, 0]) if d == 1 else nodes[0], role="V"
            )
        except TypeError as unread:
            raise TypeError(
                f"PotentialCost(V): V takes the point x alone -- a float in 1-D, an array of shape (d,) in n-D -- and "
                f"the library could not read this V as V(x). {in_the_hamiltonian}"
            ) from unread
        if signature is CallableSignature.SPATIOTEMPORAL_TX:
            raise TypeError(
                "PotentialCost(V): V takes (t, x), and this F[m] = integral V m is autonomous -- binding V once would "
                f"read it at t = 0 for the whole horizon. {in_the_hamiltonian}"
            )

        def at(points: np.ndarray) -> np.ndarray:
            return np.array([V(float(x[0])) if d == 1 else V(x) for x in points], dtype=float)

        energy = PotentialEnergy(at(nodes), np.asarray(measure.weights, dtype=float).ravel(), V=at)
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
        from mfgarchon.operators.interaction.energy_functionals import bound_energy_functional

        return cast("EnergyFunctional", CombinedEnergy([bound_energy_functional(cost, problem) for cost in self.costs]))

    def bind_on(self, measure: PointMeasure) -> EnergyFunctional:
        """Each part's ``bind_on`` through `bind_on_measure`, so a part that cannot bind on a point measure, a user's
        cost with ``bind`` alone, is refused by its name."""
        return cast("EnergyFunctional", CombinedEnergy([bind_on_measure(cost, measure) for cost in self.costs]))

    def __repr__(self) -> str:
        return f"CombinedCost({self.costs!r})"
