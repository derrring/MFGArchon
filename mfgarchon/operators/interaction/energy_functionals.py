"""
Energy functionals with analytic flat derivatives for measure-dependent MFG.

An :class:`EnergyFunctional` packages a coupling energy ``F[m]`` together with
its first variation ``delta F / delta m``, computed *analytically* rather than
by finite differences. Feeding such an object to
:func:`mfgarchon.alg.numerical.coupling.lions_correction.create_lions_source`
selects the exact-derivative path and skips the FD approximation.

Discretization convention
--------------------------
A discrete measure is the pair ``(m, weights)``: densities ``m`` sampled on a
point set, and ``weights`` the **quadrature weights of the measure
representation** -- cell volumes on a grid, particle masses for an empirical
measure. On a uniform grid ``weights`` is ``prod(spacings)`` repeated; on a
scattered GFDM cloud it is the per-point cell volume array. Integrals are
discretized as ``integral g(x) m(x) dx = sum_k w_k g_k m_k``.

Grid cell volumes sum to ``|Omega|``; particle masses are normalized to sum to
1 (compare ``utils/functional_calculus.py`` and ``core/measure.py``, which
normalize the two differently under the same word). Both are valid inputs here
-- the functional never renormalizes ``weights``, it only integrates against
them, so the two ontologies are handled by the same code path.

Three related but distinct objects, and the factor that separates them:

- ``energy(m)`` is the **physical** scalar ``F[m]``, mesh-independent: it
  converges under refinement rather than scaling as ``1/h``.
- ``flat_derivative(m)`` is the **flat** (linear-functional) derivative
  ``delta F / delta m``, a scalar field of shape ``(N,)``. This is the object
  that enters the HJB as a source term. It is *not* the Lions / L- /
  Wasserstein derivative, which is ``grad_x delta F / delta m``, a vector field
  of shape ``(N, d)``; no class here returns that.
- The **unweighted entry gradient** ``d F / d m_k`` returned by
  :class:`~mfgarchon.utils.functional_calculus.FiniteDifferenceFunctionalDerivative`
  (which perturbs ``m_k += epsilon``, an unweighted Dirac) equals
  ``w_k * (delta F / delta m)_k``.

That last relation is the FD bridge, and it lives in exactly one place:
:func:`flat_derivative_from_energy_gradient`. Anything comparing an analytic
``flat_derivative`` against an FD gradient of ``energy`` must go through it.

Concretely, for the two shipped forms:

- Interaction ``F[m] = (1/2) integral integral K(x-y) m(x) m(y) dx dy`` has
  ``delta F / delta m = K * m`` and ``delta^2 F / delta m^2 = K``.
- Potential ``F[m] = integral V(x) m(x) dx`` has ``delta F / delta m = V(x)``
  and ``delta^2 F / delta m^2 = 0``.

Issue #1023: ``operators/interaction/`` subpackage (Phase 2 Lions bridge).
Issue #1642: contract freeze (D-1..D-5) and the ``energy()`` quadrature repair.
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
    """Protocol for coupling energies with analytic flat derivatives.

    Frozen contract (Issue #1642 §6). Because this Protocol is
    ``runtime_checkable``, ``isinstance`` checks member **presence only** --
    an arity mismatch would surface as a ``TypeError`` inside a Picard
    iteration rather than at construction. The signatures below are therefore
    the contract; implementers must match them.

    All time arguments are **keyword-only**. The source pipeline already
    carries the true per-slice ``t`` (``source_composition.compose_hjb_source``
    calls ``source_term_hjb(x, m, v, t)``), so non-autonomous running couplings
    ``F_t[m]`` are expressible without a later signature break. The functionals
    shipped in this module are autonomous and ignore ``t``.

    Attributes
    ----------
    weights : NDArray
        Quadrature weights of the measure representation, shape ``(N,)``: cell
        volumes on a grid, particle masses for an empirical measure. Strictly
        positive.

    Notes
    -----
    ``second_variation`` is part of the required member set: a class that does
    not define it is **not** an ``EnergyFunctional`` under ``isinstance``, even
    though this Protocol supplies a default body. Implementers that want the
    ``None`` default may subclass ``EnergyFunctional`` explicitly; structural
    implementers must define all three methods and ``weights``.

    These are **single-population** functionals. A stacked multi-population
    density of length ``K*N`` is refused loudly (see
    :func:`as_single_population`) rather than silently contracted against
    itself.
    """

    weights: NDArray

    def energy(self, m: NDArray, *, t: float = 0.0) -> float:
        """Evaluate the physical coupling energy ``F[m]`` (scalar)."""
        ...

    def flat_derivative(self, m: NDArray, *, t: float = 0.0) -> NDArray:
        """Evaluate the flat derivative ``delta F / delta m``, shape ``(N,)``."""
        ...

    def second_variation(self, m: NDArray, *, t: float = 0.0) -> NDArray | None:
        """Evaluate ``delta^2 F / delta m^2``, shape ``(N, N)``, or ``None``.

        The returned kernel is the symmetric bilinear form in the *measure*
        pairing: ``d^2/deps^2 F[m + eps nu] = sum_ij w_i nu_i S_ij w_j nu_j``.
        The Jacobian of :meth:`flat_derivative` with respect to the unweighted
        entries ``m_l`` is ``S * weights[None, :]`` (not symmetric when the
        weights are non-uniform).

        ``None`` means "not available for this functional", not "zero".
        """
        return None


def validate_weights(weights: NDArray, owner: str) -> NDArray:
    """Return ``weights`` as a validated 1-D float array.

    Quadrature weights must be finite and strictly positive: the FD bridge
    divides by them (:func:`flat_derivative_from_energy_gradient`) and a zero
    or negative cell volume is a broken discretization, not a degenerate case
    to absorb.
    """
    w = np.asarray(weights, dtype=float).ravel()
    if w.size == 0:
        raise ValueError(f"{owner}: weights must be non-empty")
    if not np.all(np.isfinite(w)):
        raise ValueError(f"{owner}: weights must be finite, got non-finite entries")
    if not np.all(w > 0.0):
        raise ValueError(f"{owner}: quadrature weights must be strictly positive, got min {w.min()!r}")
    return w


def as_single_population(m: NDArray, weights: NDArray, owner: str) -> NDArray:
    """Return ``m`` as a flat single-population density, or refuse loudly.

    Issue #1642 D-5. ``EnergyFunctional`` is single-population by decision. A
    stacked ``(K*N,)`` multi-population density is a caller error, and the
    refusal has to be explicit: without it, some functionals contract a stacked
    array against a broadcastable operand and return a plausible number.
    """
    arr = np.asarray(m, dtype=float).ravel()
    if arr.shape != weights.shape:
        raise ValueError(
            f"{owner} is a single-population functional: expected a density of shape "
            f"({weights.size},) matching its quadrature weights, got shape {arr.shape}. "
            "A stacked multi-population density (K*N,) is not supported here -- split it "
            "per population and combine the per-population energies explicitly "
            "(see MultiPopulationMFGProblem; Issue #1642 D-5)."
        )
    return arr


def flat_derivative_from_energy_gradient(gradient: NDArray, weights: NDArray) -> NDArray:
    """Convert an unweighted entry gradient ``d F / d m_k`` to ``delta F / delta m``.

    The single owner of the quadrature factor between the two derivative
    conventions (Issue #1642 A2). :class:`FiniteDifferenceFunctionalDerivative`
    perturbs ``m_k += epsilon`` -- an *unweighted* Dirac -- so its output is
    ``d F / d m_k = w_k * (delta F / delta m)_k``. Dividing by ``w_k`` here, and
    only here, keeps the two conventions from forking.

    Parameters
    ----------
    gradient : NDArray
        Entry gradient ``d F / d m_k``, shape ``(N,)``.
    weights : NDArray
        Quadrature weights, shape ``(N,)``, strictly positive.

    Returns
    -------
    NDArray
        The flat derivative ``delta F / delta m``, shape ``(N,)``.
    """
    w = validate_weights(weights, "flat_derivative_from_energy_gradient")
    g = np.asarray(gradient, dtype=float).ravel()
    if g.shape != w.shape:
        raise ValueError(f"gradient shape {g.shape} does not match weights shape {w.shape}")
    return g / w


class QuadraticInteractionEnergy:
    """Quadratic interaction energy ``F[m] = (1/2) integral integral K(x-y) m(x) m(y) dx dy``.

    Wraps a :class:`~mfgarchon.operators.interaction.convolution.ConvolutionCouplingOperator`
    ``F_op`` (which applies ``F_op @ m = W @ (m * w)``, carrying the ``dy``
    quadrature weight) so that

        energy(m)            = (1/2) * sum_k w_k m_k (F_op @ m)_k
        flat_derivative(m)   = F_op @ m  = (K * m)(x)
        second_variation(m)  = W         (the raw kernel matrix, symmetric)

    The outer ``dx`` weight in ``energy`` is what makes the value physical: it
    converges under refinement instead of scaling as ``1/h``.

    Parameters
    ----------
    convolution_operator : ConvolutionCouplingOperator
        The interaction convolution ``F_op[m] = integral K(x-y) m(y) dy``. Its
        ``weights`` become this functional's quadrature weights.
    """

    def __init__(self, convolution_operator: ConvolutionCouplingOperator):
        self._conv = convolution_operator
        self._w = validate_weights(convolution_operator.weights, "QuadraticInteractionEnergy")

    @property
    def weights(self) -> NDArray:
        """Quadrature weights of the measure representation, shape ``(N,)``."""
        return self._w

    def energy(self, m: NDArray, *, t: float = 0.0) -> float:
        """Physical interaction energy ``F[m]`` (autonomous: ``t`` is ignored)."""
        arr = as_single_population(m, self._w, "QuadraticInteractionEnergy")
        return 0.5 * float(np.dot(self._w * arr, self._conv @ arr))

    def flat_derivative(self, m: NDArray, *, t: float = 0.0) -> NDArray:
        """``delta F / delta m = (K * m)(x)`` (autonomous: ``t`` is ignored)."""
        arr = as_single_population(m, self._w, "QuadraticInteractionEnergy")
        return np.asarray(self._conv @ arr, dtype=float)

    def second_variation(self, m: NDArray, *, t: float = 0.0) -> NDArray:
        """``delta^2 F / delta m^2 = K(x - y)``, the raw kernel matrix, shape ``(N, N)``.

        Independent of ``m`` (the energy is quadratic). Allocates a dense
        ``(N, N)`` array, including on the FFT path where the operator itself
        never forms one.
        """
        as_single_population(m, self._w, "QuadraticInteractionEnergy")
        return self._conv.kernel_matrix()

    def __repr__(self) -> str:
        return f"QuadraticInteractionEnergy({self._conv!r})"


class PotentialEnergy:
    """Linear potential energy ``F[m] = integral V(x) m(x) dx``.

    With a fixed potential field ``V`` sampled on the point set,

        energy(m)            = sum_k w_k V_k m_k
        flat_derivative(m)   = V   (independent of m)
        second_variation(m)  = 0   (shape (N, N))

    ``V`` is the pointwise cost an agent pays for occupying ``x`` (cost-signed:
    positive ``V`` repels).

    Parameters
    ----------
    potential : NDArray
        Potential field ``V`` sampled on the point set, shape ``(N,)``.
    weights : NDArray
        Quadrature weights of the measure representation, shape ``(N,)`` (or a
        scalar cell volume, broadcast to ``(N,)``). Required -- a default would
        silently substitute a mesh-independent number for a mesh-dependent one.
    """

    def __init__(self, potential: NDArray, weights: NDArray | float):
        self._V = np.asarray(potential, dtype=float).ravel()
        w = np.asarray(weights, dtype=float)
        if w.ndim == 0:
            w = np.full(self._V.shape, float(w))
        self._w = validate_weights(w, "PotentialEnergy")
        if self._w.shape != self._V.shape:
            raise ValueError(
                f"PotentialEnergy: weights shape {self._w.shape} does not match potential shape {self._V.shape}"
            )

    @property
    def weights(self) -> NDArray:
        """Quadrature weights of the measure representation, shape ``(N,)``."""
        return self._w

    def energy(self, m: NDArray, *, t: float = 0.0) -> float:
        """Physical potential energy ``F[m]`` (autonomous: ``t`` is ignored)."""
        arr = as_single_population(m, self._w, "PotentialEnergy")
        return float(np.dot(self._w * self._V, arr))

    def flat_derivative(self, m: NDArray, *, t: float = 0.0) -> NDArray:
        """``delta F / delta m = V(x)`` (autonomous: ``t`` is ignored)."""
        as_single_population(m, self._w, "PotentialEnergy")
        return self._V.copy()

    def second_variation(self, m: NDArray, *, t: float = 0.0) -> NDArray:
        """``delta^2 F / delta m^2 = 0``, shape ``(N, N)``. Identically zero (F is linear)."""
        as_single_population(m, self._w, "PotentialEnergy")
        n = self._V.size
        return np.zeros((n, n), dtype=float)

    def __repr__(self) -> str:
        return f"PotentialEnergy(V shape={self._V.shape})"


class CombinedEnergy:
    """Sum of energy functionals ``F[m] = sum_k F_k[m]``.

    Energy, flat derivative and second variation are all additive:

        energy(m)            = sum_k F_k.energy(m)
        flat_derivative(m)   = sum_k F_k.flat_derivative(m)
        second_variation(m)  = sum_k F_k.second_variation(m)

    Used to combine a repulsive interaction with a central attractive potential
    (towel-on-the-beach): ``CombinedEnergy([interaction, potential])``.

    All components must carry the same quadrature weights: summing energies
    discretized against different measures is meaningless, and the mismatch
    would otherwise show up as a wrong number rather than an error.

    Parameters
    ----------
    components : Sequence[EnergyFunctional]
        Energy functionals to sum. Must be non-empty and share the same
        ``weights``.
    """

    _WEIGHT_RTOL = 1e-12

    def __init__(self, components: Sequence[EnergyFunctional]):
        comps = list(components)
        if not comps:
            raise ValueError("CombinedEnergy requires at least one component")
        w0 = validate_weights(comps[0].weights, "CombinedEnergy")
        for i, c in enumerate(comps[1:], start=1):
            wi = np.asarray(c.weights, dtype=float).ravel()
            if wi.shape != w0.shape or not np.allclose(wi, w0, rtol=self._WEIGHT_RTOL, atol=0.0):
                raise ValueError(
                    f"CombinedEnergy components must share the same quadrature weights: "
                    f"component 0 ({comps[0]!r}) has weights of shape {w0.shape}, "
                    f"component {i} ({c!r}) has weights of shape {wi.shape} that differ. "
                    "Summing energies discretized against different measures is not defined."
                )
        self._components = comps
        self._w = w0

    @property
    def weights(self) -> NDArray:
        """Quadrature weights shared by all components, shape ``(N,)``."""
        return self._w

    def energy(self, m: NDArray, *, t: float = 0.0) -> float:
        """Sum of component energies (``t`` forwarded to each component)."""
        return float(sum(c.energy(m, t=t) for c in self._components))

    def flat_derivative(self, m: NDArray, *, t: float = 0.0) -> NDArray:
        """Sum of component flat derivatives (``t`` forwarded to each component)."""
        arr = as_single_population(m, self._w, "CombinedEnergy")
        total = np.zeros_like(arr)
        for c in self._components:
            total = total + np.asarray(c.flat_derivative(arr, t=t), dtype=float).ravel()
        return total

    def second_variation(self, m: NDArray, *, t: float = 0.0) -> NDArray | None:
        """Sum of component second variations, or ``None`` if any component lacks one.

        ``None`` propagates: a partial sum would be a wrong operator, not a
        degraded one.
        """
        arr = as_single_population(m, self._w, "CombinedEnergy")
        total = np.zeros((arr.size, arr.size), dtype=float)
        for c in self._components:
            part = c.second_variation(arr, t=t)
            if part is None:
                return None
            total = total + np.asarray(part, dtype=float)
        return total

    def __repr__(self) -> str:
        return f"CombinedEnergy({self._components!r})"
