"""
Energy functionals with analytic Lions derivatives for measure-dependent MFG.

An :class:`EnergyFunctional` packages a coupling energy ``F[m]`` together with
its first variation (Lions derivative) ``delta F / delta m``, computed
*analytically* rather than by finite differences. Feeding such an object to
:func:`mfgarchon.alg.numerical.coupling.lions_correction.create_lions_source`
selects the exact-derivative path and skips the FD approximation.

Discretization convention
--------------------------
Densities are discrete vectors ``m`` on a point set with quadrature weights
(cell volumes). Following the established ``lions_correction`` convention, each
functional's :meth:`lions_derivative` returns the *pointwise* source term
``delta F / delta m(x_k)`` that enters the HJB right-hand side, and equals the
gradient of :meth:`energy` with respect to the unweighted entry ``m_k`` (the
perturbation ``m -> m + epsilon delta_k`` used by
:class:`~mfgarchon.utils.functional_calculus.FiniteDifferenceFunctionalDerivative`).
Concretely:

- Interaction ``F[m] = (1/2) integral integral K(x-y) m(x) m(y) dx dy`` has
  ``delta F / delta m = K * m``, carrying one quadrature factor from the ``dy``
  integral (already inside the convolution operator).
- Potential ``F[m] = integral V(x) m(x) dx`` has ``delta F / delta m = V(x)``,
  a pointwise cost with no quadrature factor.

Issue #1023: ``operators/interaction/`` subpackage (Phase 2 Lions bridge).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

    from .convolution import ConvolutionCouplingOperator


@runtime_checkable
class EnergyFunctional(Protocol):
    """Protocol for coupling energies with analytic Lions derivatives.

    Implementations provide the scalar energy ``F[m]`` and its first variation
    ``delta F / delta m[m](x)`` evaluated on the grid.
    """

    def energy(self, m: NDArray) -> float:
        """Evaluate the coupling energy ``F[m]`` (scalar)."""
        ...

    def lions_derivative(self, m: NDArray) -> NDArray:
        """Evaluate the Lions derivative ``delta F / delta m[m](x)``, shape ``(N,)``."""
        ...


class QuadraticInteractionEnergy:
    """Quadratic interaction energy ``F[m] = (1/2) <m, K * m>``.

    Wraps a :class:`~mfgarchon.operators.interaction.convolution.ConvolutionCouplingOperator`
    ``F_op`` so that

        energy(m)           = (1/2) * m . (F_op @ m)
        lions_derivative(m) = F_op @ m  = (K * m)(x).

    The convolution operator carries the ``dy`` quadrature weight, so the
    derivative is the discrete ``delta F / delta m``. The analytic derivative
    matches the unweighted finite-difference gradient of :meth:`energy` when the
    operator matrix is symmetric (uniform cell volume), which holds on regular
    grids and for a scalar ``cell_volume``.

    Parameters
    ----------
    convolution_operator : ConvolutionCouplingOperator
        The interaction convolution ``F_op[m] = integral K(x-y) m(y) dy``.
    """

    def __init__(self, convolution_operator: ConvolutionCouplingOperator):
        self._conv = convolution_operator

    def energy(self, m: NDArray) -> float:
        m = np.asarray(m, dtype=float).ravel()
        return 0.5 * float(np.dot(m, self._conv @ m))

    def lions_derivative(self, m: NDArray) -> NDArray:
        m = np.asarray(m, dtype=float).ravel()
        return np.asarray(self._conv @ m, dtype=float)

    def __repr__(self) -> str:
        return f"QuadraticInteractionEnergy({self._conv!r})"


class PotentialEnergy:
    """Linear potential energy ``F[m] = integral V(x) m(x) dx``.

    With a fixed potential field ``V`` sampled on the grid,

        energy(m)           = V . m
        lions_derivative(m) = V   (independent of m).

    ``V`` is the pointwise cost an agent pays for occupying ``x`` (cost-signed:
    positive ``V`` repels). The discrete ``energy`` is the generating functional
    whose unweighted gradient is the pointwise source ``V(x)``; multiply by the
    cell volume to recover the physical integral ``integral V m dx``.

    Parameters
    ----------
    potential : NDArray
        Potential field ``V`` sampled on the grid, shape ``(N,)``.
    """

    def __init__(self, potential: NDArray):
        self._V = np.asarray(potential, dtype=float).ravel()

    def energy(self, m: NDArray) -> float:
        m = np.asarray(m, dtype=float).ravel()
        return float(np.dot(self._V, m))

    def lions_derivative(self, m: NDArray) -> NDArray:
        return self._V.copy()

    def __repr__(self) -> str:
        return f"PotentialEnergy(V shape={self._V.shape})"


class CombinedEnergy:
    """Sum of energy functionals ``F[m] = sum_k F_k[m]``.

    Energy and Lions derivative are additive:

        energy(m)           = sum_k F_k.energy(m)
        lions_derivative(m) = sum_k F_k.lions_derivative(m).

    Used to combine a repulsive interaction with a central attractive potential
    (towel-on-the-beach): ``CombinedEnergy([interaction, potential])``.

    Parameters
    ----------
    components : Sequence[EnergyFunctional]
        Energy functionals to sum. Must be non-empty and act on the same grid.
    """

    def __init__(self, components: Sequence[EnergyFunctional]):
        comps = list(components)
        if not comps:
            raise ValueError("CombinedEnergy requires at least one component")
        self._components = comps

    def energy(self, m: NDArray) -> float:
        return float(sum(c.energy(m) for c in self._components))

    def lions_derivative(self, m: NDArray) -> NDArray:
        m = np.asarray(m, dtype=float).ravel()
        total = np.zeros_like(m)
        for c in self._components:
            total = total + np.asarray(c.lions_derivative(m), dtype=float).ravel()
        return total

    def __repr__(self) -> str:
        return f"CombinedEnergy({self._components!r})"
