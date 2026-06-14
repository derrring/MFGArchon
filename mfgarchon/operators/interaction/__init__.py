"""
Spatial interaction operators for agent-agent game coupling.

This subpackage provides the *game-coupling* non-local form on Euclidean
domains, distinct from the integro-differential (Levy / graphon) operators in
``operators/integro_diff/`` which act on the single-agent PDE structure. The
coupling is a spatial convolution

    F[m](x) = integral K(x - y) m(y) dy,

the first variation of the interaction energy
``F[m] = (1/2) integral integral K(x - y) m(x) m(y) dx dy`` with Lions
derivative ``delta F / delta m = K * m``. With a repulsive kernel and a central
attractive potential it produces the towel-on-the-beach ring/bimodal
equilibrium (central depletion + annular density) that local ``f(m)`` cannot.

Contents:
    - kernels: radial kernel zoo (Gaussian, tent, Wendland C^2, dipole,
      power-law) with the cost-signed repulsive/attractive convention.
    - convolution: ConvolutionCouplingOperator (LinearOperator) with FFT path
      (regular grid) and direct-quadrature path (irregular cloud).
    - energy_functionals: EnergyFunctional protocol + QuadraticInteractionEnergy,
      PotentialEnergy, CombinedEnergy with analytic Lions derivatives.

The analytic Lions derivatives plug into
``alg.numerical.coupling.lions_correction.create_lions_source`` (Phase 2), which
recognizes an ``EnergyFunctional`` and skips the finite-difference path.

Issue #1023. ``aggregation.py`` (Carrillo ``div(m grad(K * m))``) is deferred to
Phase 1b.
"""

from .convolution import ConvolutionCouplingOperator
from .energy_functionals import (
    CombinedEnergy,
    EnergyFunctional,
    PotentialEnergy,
    QuadraticInteractionEnergy,
)
from .kernels import (
    DipoleKernel,
    GaussianKernel,
    PowerLawKernel,
    RadialKernel,
    TentKernel,
    WendlandKernel,
)

__all__ = [
    # Kernels
    "RadialKernel",
    "GaussianKernel",
    "TentKernel",
    "WendlandKernel",
    "DipoleKernel",
    "PowerLawKernel",
    # Convolution operator
    "ConvolutionCouplingOperator",
    # Energy functionals
    "EnergyFunctional",
    "QuadraticInteractionEnergy",
    "PotentialEnergy",
    "CombinedEnergy",
]
