"""Finite Element Exterior Calculus (FEEC) — mixed structure-preserving discretization scaffold.

Infrastructure foundation for structure-preserving / symplectic MFG. See ``discretization.py`` for the
scope (structure-preserving building blocks are provided; the coupled saddle-point solve, positivity,
the nonlinear-Hamiltonian coupling, and symplectic time-stepping are research steps, fail-loud).
"""

from __future__ import annotations

from mfgarchon.alg.numerical.feec.discretization import (
    MixedWeakFormDiscretization,
    RaviartThomasDiscretization,
)

__all__ = ["MixedWeakFormDiscretization", "RaviartThomasDiscretization"]
