"""
Integro-differential operators for non-local PDE structure.

This module provides operators for jump-diffusion and Lévy-driven MFG,
complementing the local differential operators in ``operators/differential/``.
"Integro-differential" is the standard literature term for Lévy-type non-local
operators (Jakobsen-Karlsen, Barles-Imbert); it is unambiguous and distinct from
the *game-coupling* non-local operators in ``operators/interaction/``.

- LevyIntegroDiffOperator: Non-local jump operator J[v] for HJB/FP
- LevyMeasure protocol + concrete implementations (Gaussian, compound Poisson)
- Graphon coupling/kernels: non-local integration on the graphon parameter space

Issue #923: Part of Layer 1 (Generalized PDE & Institutional MFG Plan).
Issue #1024: Renamed from ``operators/nonlocal_ops`` to ``operators/integro_diff``.
"""

from .levy_integro_diff import LevyIntegroDiffOperator
from .levy_measures import CompoundPoissonJumps, GaussianJumps, LevyMeasure

__all__ = [
    "LevyIntegroDiffOperator",
    "LevyMeasure",
    "GaussianJumps",
    "CompoundPoissonJumps",
]
