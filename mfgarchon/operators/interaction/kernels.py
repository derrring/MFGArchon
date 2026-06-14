"""
Radial interaction kernels for spatial agent-agent coupling.

A radial kernel ``K(r)`` defines a translation-invariant interaction
``K(x - y) = K(|x - y|)`` that drives the non-local game coupling

    F[m](x) = integral K(x - y) m(y) dy,

used by :class:`~mfgarchon.operators.interaction.convolution.ConvolutionCouplingOperator`
and by the quadratic interaction energy
``F[m] = (1/2) integral integral K(x - y) m(x) m(y) dx dy`` whose Lions
derivative is ``delta F / delta m = K * m``.

Sign convention (cost-signed, gotcha G-002)
-------------------------------------------
The interaction enters the HJB through ``source_term_hjb = +delta F / delta m``,
which is *cost-signed* (a positive source is a cost that repels). Therefore:

- ``amplitude > 0`` -> **repulsive** congestion. ``delta F / delta m = K * m`` is
  large where the crowd is, so crowded regions are expensive and agents spread
  out. This is the towel-on-the-beach repulsion.
- ``amplitude < 0`` -> **attractive** aggregation. Crowded regions are rewarded,
  so agents cluster (Carrillo-style aggregation).

Each kernel exposes ``is_repulsive`` reflecting this convention.

Kernel zoo
----------
======================  =======================================================
Kernel                  Radial profile ``K(r)``
======================  =======================================================
``GaussianKernel``      ``A * exp(-r^2 / (2 ell^2))`` (smooth, global support)
``TentKernel``          ``A * max(0, 1 - r / ell)`` (compact, C^0)
``WendlandKernel``      ``A * (1 - r/ell)_+^4 (1 + 4 r/ell)`` (compact, C^2,
                        positive-definite for d <= 3)
``DipoleKernel``        short-range repulsion minus long-range attraction
                        (difference of Gaussians, "Mexican hat"); mixed sign
``PowerLawKernel``      ``A * (r^2 + eps^2)^(-p/2)`` (softened Riesz / Coulomb)
======================  =======================================================

Issue #1023: ``operators/interaction/`` subpackage.

References:
    Carrillo, Craig, Yao (2018), "Aggregation-Diffusion Equations".
    Wendland (1995), "Piecewise polynomial, positive definite and compactly
        supported radial functions of minimal degree".
    Burger, Di Francesco, Pietschmann, Schloeder (2013), "Nonlinear models for
        crowd dynamics".
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


class RadialKernel(ABC):
    """Abstract base for radial interaction kernels ``K(r)``.

    Subclasses implement :meth:`profile`, the radial profile evaluated on an
    array of non-negative distances ``r``. The base class provides the callable
    interface ``kernel(r)`` and the dense pairwise matrix builder
    :meth:`matrix`. All radial kernels are translation-invariant, so they are
    FFT-eligible on regular grids.
    """

    @abstractmethod
    def profile(self, r: NDArray) -> NDArray:
        """Evaluate the radial profile ``K(r)`` at distances ``r >= 0``."""
        ...

    def __call__(self, r: NDArray) -> NDArray:
        """Evaluate ``K(r)`` (alias for :meth:`profile`)."""
        return self.profile(np.asarray(r, dtype=float))

    def matrix(self, points: NDArray) -> NDArray:
        """Build the dense interaction matrix ``W_ij = K(|x_i - x_j|)``.

        Args:
            points: Point set, shape ``(N,)`` (1D) or ``(N, d)`` (nD).

        Returns:
            Symmetric matrix ``W``, shape ``(N, N)``, with
            ``W_ij = K(|x_i - x_j|)``.
        """
        pts = np.asarray(points, dtype=float)
        if pts.ndim == 1:
            pts = pts[:, None]
        diff = pts[:, None, :] - pts[None, :, :]
        r = np.sqrt(np.einsum("ijk,ijk->ij", diff, diff))
        return self.profile(r)

    @property
    def is_translational(self) -> bool:
        """Radial kernels are translation-invariant (FFT-eligible)."""
        return True

    @property
    @abstractmethod
    def is_repulsive(self) -> bool:
        """Whether the kernel is repulsive under the cost-signed convention."""
        ...


class GaussianKernel(RadialKernel):
    """Gaussian interaction ``K(r) = A * exp(-r^2 / (2 ell^2))``.

    Smooth (C-infinity) with global support; truncate via the grid extent for
    FFT use. The characteristic interaction length is ``ell``. Repulsive for
    ``amplitude > 0`` (towel-on-the-beach), attractive for ``amplitude < 0``.

    Parameters
    ----------
    amplitude : float
        Peak value ``A = K(0)``. Sign sets repulsive (``> 0``) vs attractive.
    length_scale : float
        Interaction length ``ell > 0``.
    """

    def __init__(self, amplitude: float = 1.0, length_scale: float = 0.1):
        if length_scale <= 0:
            raise ValueError(f"length_scale must be positive, got {length_scale}")
        self.amplitude = amplitude
        self.length_scale = length_scale

    def profile(self, r: NDArray) -> NDArray:
        return self.amplitude * np.exp(-(r**2) / (2.0 * self.length_scale**2))

    @property
    def is_repulsive(self) -> bool:
        return self.amplitude > 0

    def __repr__(self) -> str:
        return f"GaussianKernel(amplitude={self.amplitude}, length_scale={self.length_scale})"


class TentKernel(RadialKernel):
    """Triangular (tent) interaction ``K(r) = A * max(0, 1 - r / ell)``.

    Compact support ``[0, ell]``, continuous but not differentiable at ``r=0``
    and ``r=ell`` (C^0). Cheap and strictly local. Repulsive for
    ``amplitude > 0``.

    Parameters
    ----------
    amplitude : float
        Peak value ``A = K(0)``.
    length_scale : float
        Support radius ``ell > 0``; ``K(r) = 0`` for ``r >= ell``.
    """

    def __init__(self, amplitude: float = 1.0, length_scale: float = 0.1):
        if length_scale <= 0:
            raise ValueError(f"length_scale must be positive, got {length_scale}")
        self.amplitude = amplitude
        self.length_scale = length_scale

    def profile(self, r: NDArray) -> NDArray:
        return self.amplitude * np.maximum(0.0, 1.0 - r / self.length_scale)

    @property
    def is_repulsive(self) -> bool:
        return self.amplitude > 0

    def __repr__(self) -> str:
        return f"TentKernel(amplitude={self.amplitude}, length_scale={self.length_scale})"


class WendlandKernel(RadialKernel):
    """Wendland C^2 interaction ``K(r) = A * (1 - r/ell)_+^4 (1 + 4 r/ell)``.

    Compactly supported on ``[0, ell]``, twice continuously differentiable, and
    positive-definite for spatial dimension ``d <= 3`` (Wendland phi_{3,1}).
    Combines the locality of the tent kernel with the smoothness of the
    Gaussian, which makes the induced convolution well-behaved. Repulsive for
    ``amplitude > 0``.

    Parameters
    ----------
    amplitude : float
        Peak value ``A = K(0)``.
    length_scale : float
        Support radius ``ell > 0``; ``K(r) = 0`` for ``r >= ell``.
    """

    def __init__(self, amplitude: float = 1.0, length_scale: float = 0.1):
        if length_scale <= 0:
            raise ValueError(f"length_scale must be positive, got {length_scale}")
        self.amplitude = amplitude
        self.length_scale = length_scale

    def profile(self, r: NDArray) -> NDArray:
        q = r / self.length_scale
        base = np.maximum(0.0, 1.0 - q)
        return self.amplitude * base**4 * (1.0 + 4.0 * q)

    @property
    def is_repulsive(self) -> bool:
        return self.amplitude > 0

    def __repr__(self) -> str:
        return f"WendlandKernel(amplitude={self.amplitude}, length_scale={self.length_scale})"


class DipoleKernel(RadialKernel):
    """Mixed-sign interaction: short-range repulsion minus long-range attraction.

    Difference of Gaussians ("Mexican hat"):

        K(r) = A_rep * exp(-r^2 / (2 ell_rep^2)) - A_att * exp(-r^2 / (2 ell_att^2))

    With ``ell_rep < ell_att`` and positive amplitudes, the kernel is positive
    (repulsive) at short range and negative (attractive) at intermediate range,
    the canonical swarming interaction producing finite-size aggregates. Sign is
    mixed; :attr:`is_repulsive` reports the short-range (near-zero) behaviour.

    Parameters
    ----------
    rep_amplitude : float
        Short-range repulsion strength ``A_rep >= 0``.
    rep_scale : float
        Short-range length ``ell_rep > 0``.
    att_amplitude : float
        Long-range attraction strength ``A_att >= 0``.
    att_scale : float
        Long-range length ``ell_att > 0`` (typically ``> rep_scale``).
    """

    def __init__(
        self,
        rep_amplitude: float = 1.0,
        rep_scale: float = 0.05,
        att_amplitude: float = 0.5,
        att_scale: float = 0.2,
    ):
        if rep_scale <= 0 or att_scale <= 0:
            raise ValueError("rep_scale and att_scale must be positive")
        self.rep_amplitude = rep_amplitude
        self.rep_scale = rep_scale
        self.att_amplitude = att_amplitude
        self.att_scale = att_scale

    def profile(self, r: NDArray) -> NDArray:
        rep = self.rep_amplitude * np.exp(-(r**2) / (2.0 * self.rep_scale**2))
        att = self.att_amplitude * np.exp(-(r**2) / (2.0 * self.att_scale**2))
        return rep - att

    @property
    def is_repulsive(self) -> bool:
        """Short-range (near-zero distance) sign: ``K(0) = A_rep - A_att``."""
        return (self.rep_amplitude - self.att_amplitude) > 0

    def __repr__(self) -> str:
        return (
            f"DipoleKernel(rep_amplitude={self.rep_amplitude}, rep_scale={self.rep_scale}, "
            f"att_amplitude={self.att_amplitude}, att_scale={self.att_scale})"
        )


class PowerLawKernel(RadialKernel):
    """Softened power-law (Riesz / Coulomb-type) interaction.

        K(r) = A * (r^2 + eps^2)^(-p/2)

    The softening ``eps > 0`` regularizes the singularity at ``r = 0`` so the
    kernel is bounded and differentiable. ``exponent = 1`` gives a 3D-Coulomb
    profile; larger ``exponent`` decays faster. Repulsive for ``amplitude > 0``.

    Parameters
    ----------
    amplitude : float
        Strength ``A``.
    exponent : float
        Decay exponent ``p > 0`` (the profile decays like ``r^{-p}``).
    softening : float
        Regularization ``eps > 0`` capping the near-zero value at
        ``A * eps^{-p}``.
    """

    def __init__(self, amplitude: float = 1.0, exponent: float = 1.0, softening: float = 0.05):
        if exponent <= 0:
            raise ValueError(f"exponent must be positive, got {exponent}")
        if softening <= 0:
            raise ValueError(f"softening must be positive, got {softening}")
        self.amplitude = amplitude
        self.exponent = exponent
        self.softening = softening

    def profile(self, r: NDArray) -> NDArray:
        return self.amplitude * (r**2 + self.softening**2) ** (-self.exponent / 2.0)

    @property
    def is_repulsive(self) -> bool:
        return self.amplitude > 0

    def __repr__(self) -> str:
        return f"PowerLawKernel(amplitude={self.amplitude}, exponent={self.exponent}, softening={self.softening})"
