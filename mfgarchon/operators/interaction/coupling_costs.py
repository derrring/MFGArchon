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
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from mfgarchon.operators.interaction.convolution import ConvolutionCouplingOperator
from mfgarchon.operators.interaction.energy_functionals import (
    CombinedEnergy,
    PotentialEnergy,
    QuadraticInteractionEnergy,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from mfgarchon.core.mfg_problem import MFGProblem
    from mfgarchon.operators.interaction.energy_functionals import EnergyFunctional
    from mfgarchon.operators.interaction.kernels import RadialKernel


class _SummableCost:
    """``a + b`` for coupling costs: their `CombinedCost`."""

    def __add__(self, other: object) -> CombinedCost:
        if not isinstance(other, _SummableCost):
            return NotImplemented
        return CombinedCost([*_parts(self), *_parts(other)])


def _parts(cost: _SummableCost) -> list[_SummableCost]:
    return list(cost.costs) if isinstance(cost, CombinedCost) else [cost]


class InteractionCost(_SummableCost):
    """The interaction cost ``F[m] = (1/2) integral integral K(x - y) m(x) m(y) dx dy`` of a translation-invariant
    kernel K, whose flat derivative is ``(K * m)(x)``.

    Binds to `QuadraticInteractionEnergy` on ``ConvolutionCouplingOperator.from_problem(kernel, problem)``: the
    problem's grid, its measure, and its topology, so a periodic axis is refused there by name (#2725).

    Args:
        kernel: The interaction kernel K, a `RadialKernel` (``GaussianKernel``, ``WendlandKernel``, ...).
        use_fft: The convolution's FFT path, as ``ConvolutionCouplingOperator.from_problem`` takes it.
    """

    def __init__(self, kernel: RadialKernel, *, use_fft: bool | None = None):
        self.kernel = kernel
        self.use_fft = use_fft

    def bind(self, problem: MFGProblem) -> EnergyFunctional:
        return QuadraticInteractionEnergy(
            ConvolutionCouplingOperator.from_problem(self.kernel, problem, use_fft=self.use_fft)
        )

    def __repr__(self) -> str:
        return f"InteractionCost({self.kernel!r})"


class PotentialCost(_SummableCost):
    """The potential cost ``F[m] = integral V(x) m(x) dx``, whose flat derivative is ``V(x)``.

    Binds to `PotentialEnergy` with V sampled at the problem's nodes and the problem's measure,
    ``problem.spatial_measure().weights``.

    Args:
        V: The potential, cost-signed (positive repels), called once per node as ``m_initial`` is: ``V(x)`` with
            ``x`` a float in 1-D and an array of shape ``(d,)`` in n-D, returning one number.
    """

    def __init__(self, V: Callable):
        self.V = V

    def bind(self, problem: MFGProblem) -> EnergyFunctional:
        from mfgarchon.utils.callable_adapter import adapt_ic_callable

        nodes = np.asarray(problem.geometry.get_spatial_grid(), dtype=float)
        nodes = nodes.reshape(nodes.shape[0], -1)
        d = nodes.shape[1]
        _, V = adapt_ic_callable(self.V, dimension=d, sample_point=float(nodes[0, 0]) if d == 1 else nodes[0], role="V")
        values = np.array([V(float(x[0])) if d == 1 else V(x) for x in nodes], dtype=float)
        return PotentialEnergy(values, np.asarray(problem.spatial_measure().weights, dtype=float).ravel())

    def __repr__(self) -> str:
        return f"PotentialCost({self.V!r})"


class CombinedCost(_SummableCost):
    """The sum of coupling costs, ``F = F_1 + F_2 + ...``. Binds to `CombinedEnergy` of each part's binding, on one
    measure. ``InteractionCost(K) + PotentialCost(V)`` builds it.

    Args:
        costs: The parts, each an `InteractionCost`, a `PotentialCost` or a `CombinedCost` (flattened).
    """

    def __init__(self, costs: Sequence[_SummableCost]):
        flat = [part for cost in costs for part in _parts(cost)]
        if not flat:
            raise ValueError("CombinedCost: no parts. A coupling cost of zero is Model(coupling_cost=None).")
        self.costs = flat

    def bind(self, problem: MFGProblem) -> EnergyFunctional:
        return CombinedEnergy([cost.bind(problem) for cost in self.costs])

    def __repr__(self) -> str:
        return f"CombinedCost({self.costs!r})"
