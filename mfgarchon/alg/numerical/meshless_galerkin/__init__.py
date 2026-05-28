"""
Meshless Galerkin (Moving Least Squares) weak-form backend.

Provides ``MeshlessGalerkinDiscretization``, a meshfree implementation of the
``WeakFormDiscretization`` protocol: weak-form operators are assembled by local
quadrature against MLS shape functions on a scattered point cloud, with no mesh.

The MLS shape-function derivatives have two interchangeable backends:
- ``"numpy"`` (default): analytic derivatives, core dependencies only.
- ``"jax"`` (optional): autodiff; requires jax. No silent fallback -- an
  explicit error is raised if jax is requested but unavailable.

Issue #1131 Phase 2.
"""

from mfgarchon.alg.numerical.meshless_galerkin.discretization import (
    MeshlessGalerkinDiscretization,
)
from mfgarchon.alg.numerical.meshless_galerkin.quadrature import tensor_gauss

__all__ = [
    "MeshlessGalerkinDiscretization",
    "tensor_gauss",
]
