"""
Lions derivative correction bridge for MFG source term injection.

Connects the functional calculus infrastructure (FunctionalDerivative)
to the HJB source term pipeline (MFGProblem.source_term_hjb).

For a coupling energy F[m], the Lions correction adds the first variation
delta F / delta m[m](x) as a source term to the HJB equation:

    -du/dt + H(x, Du, m) - sigma^2/2 Du + delta F / delta m[m](x) = 0

This is the "measure-dependent" coupling that goes beyond local f(m(x))
coupling (which is already handled by the Hamiltonian).

Issue #956: Part of Layer 2 (Measure-Dependent MFG).

Mathematical background:
    For nonlocal coupling F[m] = (1/2) int int W(x,y) m(x) m(y) dx dy,
    the first variation is:
        delta F / delta m[m](x) = int W(x,y) m(y) dy = (W * m)(x)

    For local coupling F[m] = int f(m(x)) dx:
        delta F / delta m[m](x) = f'(m(x))

    The FunctionalDerivative infrastructure computes this via finite
    differences or particle approximation, making this bridge agnostic
    to the specific coupling form.

Usage:
    >>> from mfgarchon.utils.functional_calculus import FiniteDifferenceFunctionalDerivative
    >>> from mfgarchon.alg.numerical.coupling.lions_correction import create_lions_source
    >>>
    >>> # Define nonlocal energy functional
    >>> W = ...  # interaction kernel matrix (Nx, Nx)
    >>> dx = 1.0 / Nx
    >>> def energy(m):
    ...     return 0.5 * np.sum(m * (W @ m)) * dx
    >>>
    >>> fd = FiniteDifferenceFunctionalDerivative(epsilon=1e-4)
    >>> source_hjb = create_lions_source(energy, fd)
    >>>
    >>> problem = MFGProblem(..., source_term_hjb=source_hjb)
    >>> result = problem.solve()
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

    from mfgarchon.operators.interaction.energy_functionals import EnergyFunctional
    from mfgarchon.utils.functional_calculus import FunctionalDerivative, FunctionalOnMeasures


def _spatial_density(m: NDArray) -> NDArray:
    """Return the flat spatial density, rejecting (Nt+1, Nx) trajectories.

    The composed source pipeline (:mod:`source_composition`) time-slices the
    density to the current backward time step before calling the source, so a
    per-time source always receives a 1-D spatial array. A 2-D ``(Nt+1, Nx)``
    trajectory reaching here is a caller error: silently collapsing it to the
    terminal slice ``m[-1]`` reintroduces the Issue #1285 wrong-slice bug.
    """
    if m.ndim == 2:
        raise ValueError(
            f"Lions source expects a 1-D spatial density (Nx,), got a 2-D array "
            f"with shape {m.shape}. Pass the time-t density slice, not the full "
            "(Nt+1, Nx) trajectory (see source_composition.compose_hjb_source; "
            "Issue #1285)."
        )
    return m.ravel()


def create_lions_source(
    energy_functional: FunctionalOnMeasures | EnergyFunctional,
    functional_derivative: FunctionalDerivative | None = None,
) -> Callable[[NDArray, NDArray, NDArray, float], NDArray]:
    """Create a source_term_hjb from a measure-dependent energy functional.

    Bridges the functional calculus infrastructure to the MFGProblem
    source_term_hjb interface by computing delta F / delta m[m](x)
    at each Picard iteration.

    Two paths, selected from the first argument:

    1. **Analytic** (Issue #1023, Phase 2): if ``energy_functional`` is an
       :class:`~mfgarchon.operators.interaction.energy_functionals.EnergyFunctional`
       (provides ``.flat_derivative``), its exact derivative is used and the FD
       path is skipped. ``functional_derivative`` is then ignored. The
       per-slice ``t`` is forwarded to ``flat_derivative``.
    2. **Finite difference** (original path): if ``energy_functional`` is a plain
       ``F[m] -> float`` callable, ``functional_derivative`` is required and
       ``delta F / delta m`` is approximated by finite differences.

    The two paths agree for an ``EnergyFunctional`` whose analytic
    ``flat_derivative`` matches the FD gradient of its ``.energy`` **divided by
    the quadrature weights** -- the FD engine perturbs an unweighted Dirac
    ``m_k += epsilon``, so its output is ``w_k * (delta F / delta m)_k``. The
    conversion has one owner,
    :func:`~mfgarchon.operators.interaction.energy_functionals.flat_derivative_from_energy_gradient`
    (Issue #1642 A2).

    Args:
        energy_functional: Either an ``EnergyFunctional`` (analytic path) or a
            plain ``F[m] -> float`` callable depending on the full measure
            (FD path). Density arrays have shape (Nx,).
        functional_derivative: FunctionalDerivative instance for the FD path
            (FiniteDifferenceFunctionalDerivative or
            ParticleApproximationFunctionalDerivative). Required only when
            ``energy_functional`` is a plain callable; ignored for an
            ``EnergyFunctional``.

    Returns:
        source_term_hjb(x, m, v, t) -> NDArray compatible with
        MFGProblem.source_term_hjb field. The returned array has shape
        matching x (one value per spatial grid point).

    Example:
        >>> import numpy as np
        >>> from mfgarchon.utils.functional_calculus import FiniteDifferenceFunctionalDerivative
        >>>
        >>> # Quadratic interaction: F[m] = (1/2) int m(x)^2 dx
        >>> dx = 0.02
        >>> def energy(m):
        ...     return 0.5 * np.sum(m**2) * dx
        >>>
        >>> fd = FiniteDifferenceFunctionalDerivative(epsilon=1e-4)
        >>> source = create_lions_source(energy, fd)
        >>>
        >>> # source(x, m, v, t) returns delta F / delta m = m(x) * dx
        >>> m = np.ones(50) / 50
        >>> result = source(np.linspace(0, 1, 50), m, np.zeros(50), 0.0)
    """
    # Analytic path: EnergyFunctional carries its own exact Lions derivative.
    from mfgarchon.operators.interaction.energy_functionals import EnergyFunctional

    if isinstance(energy_functional, EnergyFunctional):
        analytic = energy_functional

        def source_term_hjb_analytic(
            x: NDArray,
            m: NDArray,
            v: NDArray,
            t: float,
        ) -> NDArray:
            """Evaluate the analytic Lions correction delta F / delta m[m](x)."""
            m_flat = _spatial_density(m)
            return np.asarray(analytic.flat_derivative(m_flat, t=t)).ravel()

        return source_term_hjb_analytic

    if functional_derivative is None:
        raise ValueError(
            "functional_derivative is required when energy_functional is a plain "
            "callable (FD path); pass an EnergyFunctional to use the analytic path"
        )

    def source_term_hjb(
        x: NDArray,
        m: NDArray,
        v: NDArray,
        t: float,
    ) -> NDArray:
        """Evaluate Lions correction delta F / delta m[m](x).

        Args:
            x: Spatial grid points, shape (Nx,) or (Nx, d)
            m: Current spatial density, shape (Nx,). Must be a single time-t
                slice; the source pipeline time-slices before calling (a 2-D
                (Nt+1, Nx) trajectory is rejected — see Issue #1285).
            v: Current value function (unused — correction depends on m only)
            t: Current time (unused for time-independent F[m])

        Returns:
            Source term values, shape (Nx,)
        """
        m_flat = _spatial_density(m)
        Nx = len(m_flat)

        # Compute delta F / delta m at each grid point
        # y_points = all grid indices (perturb at each point)
        y_indices = np.arange(Nx)

        deriv = functional_derivative.compute(
            energy_functional,
            m_flat,
            x_points=x,
            y_points=y_indices,
        )

        return np.asarray(deriv).ravel()

    return source_term_hjb


def create_nonlocal_source(
    interaction_kernel: NDArray,
    grid_spacing: float,
) -> Callable[[NDArray, NDArray, NDArray, float], NDArray]:
    """Create source_term_hjb for nonlocal interaction coupling.

    Optimized path for the common case F[m] = (1/2) int int W(x,y) m(x) m(y) dx dy,
    where delta F / delta m[m](x) = int W(x,y) m(y) dy = (W @ m) * dx.

    This avoids the overhead of finite-difference functional derivatives
    by computing the convolution directly.

    Args:
        interaction_kernel: W(x_i, x_j) matrix, shape (N, N) where N is the
            total number of spatial grid points (N = Nx for 1D, Nx*Ny for 2D, etc.).
            Symmetric for undirected interaction. Operates on flattened density arrays.
        grid_spacing: Spatial grid spacing dx for quadrature (1D cell volume).
            For nD, pass the cell volume dx*dy*... instead.

    Returns:
        source_term_hjb(x, m, v, t) -> NDArray.

    Example:
        >>> # Gaussian interaction kernel
        >>> x = np.linspace(0, 1, 50)
        >>> W = np.exp(-((x[:, None] - x[None, :]) ** 2) / (2 * 0.1**2))
        >>> source = create_nonlocal_source(W, dx=x[1] - x[0])
    """
    W = interaction_kernel
    dx = grid_spacing

    def source_term_hjb(
        x: NDArray,
        m: NDArray,
        v: NDArray,
        t: float,
    ) -> NDArray:
        """Evaluate (W * m)(x) = int W(x,y) m(y) dy."""
        m_spatial = _spatial_density(m)
        return (W @ m_spatial) * dx

    return source_term_hjb
