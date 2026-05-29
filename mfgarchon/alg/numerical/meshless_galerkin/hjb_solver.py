"""
Meshless Galerkin (MLS) HJB solver.

Thin subclass over the backend-agnostic ``WeakFormHJBSolver``: builds a
``MeshlessGalerkinDiscretization`` from a scattered collocation cloud + support
radius ``delta`` (mirroring the ``HJBGFDMSolver(problem, collocation_points,
delta, ...)`` constructor), with interior tensor-Gauss quadrature. Time stepping,
Picard, and Newton are inherited.

Boundary conditions: Neumann / no-flux (the weak form's natural BC, reflecting-wall
MFG) and Dirichlet ``u = g`` imposed weakly by symmetric Nitsche (#1138) -- the MLS
basis is non-interpolatory, so nodal condensation is invalid; the Nitsche terms are
added to the diffusion block via ``_weak_bc_terms`` (see ``nitsche.py``). Robin and
other BC types are not implemented.

Issue #1131 Phase 2; Nitsche Dirichlet #1138.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mfgarchon.alg.base_solver import SchemeFamily
from mfgarchon.alg.numerical.meshless_galerkin.discretization import discretization_from_cloud
from mfgarchon.alg.numerical.weak_form_hjb_solver import WeakFormHJBSolver

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from mfgarchon.core.mfg_problem import MFGProblem


class MeshlessGalerkinHJBSolver(WeakFormHJBSolver):
    """HJB on a scattered point cloud via Galerkin MLS (Type-A discrete duality)."""

    _scheme_family = SchemeFamily.MESHLESS_GALERKIN

    def __init__(
        self,
        problem: MFGProblem,
        collocation_points: NDArray,
        delta: float = 0.1,
        degree: int = 2,
        n_gauss: int = 4,
        backend: str = "numpy",
        nitsche_penalty: float = 20.0,
    ) -> None:
        disc = discretization_from_cloud(collocation_points, delta, degree, n_gauss, backend)
        super().__init__(problem, disc)
        self.hjb_method_name = "MeshlessGalerkin"
        self._n_gauss = n_gauss
        self._nitsche_penalty = nitsche_penalty
        self._nitsche_cache: tuple | None = None
        self._nitsche_cache_D: float | None = None

    def _is_pure_neumann(self) -> bool:
        from mfgarchon.alg.numerical.fem.bc_adapter import is_pure_neumann

        return is_pure_neumann(self._bc)

    def _weak_bc_terms(self, D: float):
        """Symmetric Nitsche Dirichlet terms ``u = g`` for the HJB diffusion block.

        Returns ``(N_nitsche, rhs_data)`` to add to ``M/dt + D*K`` and the RHS, or
        ``(None, None)`` if no Dirichlet segments are present (then the natural
        Neumann/no-flux path is used). Cached: the block depends only on ``D``, which
        is constant across a solve."""
        if self._nitsche_cache is not None and self._nitsche_cache_D == D:
            return self._nitsche_cache
        from mfgarchon.alg.numerical.meshless_galerkin.nitsche import assemble_nitsche_terms

        terms = assemble_nitsche_terms(self._disc, self._bc, D, self._nitsche_penalty, self._n_gauss, include_data=True)
        self._nitsche_cache = terms
        self._nitsche_cache_D = D
        return terms

    def _dirichlet_dofs_and_values(self):
        raise NotImplementedError(
            "MeshlessGalerkinHJBSolver imposes Dirichlet BC weakly via Nitsche (_weak_bc_terms), not "
            "nodal condensation -- its MLS basis is non-interpolatory. This condensation hook is "
            "unreachable for Dirichlet/Neumann; reaching it means an unsupported BC type (e.g. Robin)."
        )

    def _apply_bc_to_system(self, matrix, rhs):
        raise NotImplementedError(
            "MeshlessGalerkinHJBSolver imposes Dirichlet BC weakly via Nitsche (_weak_bc_terms), not "
            "nodal condensation. Reaching this hook means an unsupported BC type (e.g. Robin)."
        )
