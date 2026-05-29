"""
Meshless Galerkin (MLS) HJB solver.

Thin subclass over the backend-agnostic ``WeakFormHJBSolver``: builds a
``MeshlessGalerkinDiscretization`` from a scattered collocation cloud + support
radius ``delta`` (mirroring the ``HJBGFDMSolver(problem, collocation_points,
delta, ...)`` constructor), with interior tensor-Gauss quadrature. Time stepping,
Picard, and Newton are inherited.

Boundary conditions: Neumann / no-flux only for now (the weak form's natural BC,
and the reflecting-wall MFG setting). Dirichlet via Nitsche is deferred (#1131).

Issue #1131 Phase 2.
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
    ) -> None:
        disc = discretization_from_cloud(collocation_points, delta, degree, n_gauss, backend)
        super().__init__(problem, disc)
        self.hjb_method_name = "MeshlessGalerkin"

    def _is_pure_neumann(self) -> bool:
        from mfgarchon.alg.numerical.fem.bc_adapter import is_pure_neumann

        return is_pure_neumann(self._bc)

    def _dirichlet_dofs_and_values(self):
        raise NotImplementedError(
            "MeshlessGalerkinHJBSolver supports Neumann/no-flux BC only; Dirichlet (Nitsche) deferred (#1131)."
        )

    def _apply_bc_to_system(self, matrix, rhs):
        raise NotImplementedError(
            "MeshlessGalerkinHJBSolver supports Neumann/no-flux BC only; Dirichlet (Nitsche) deferred (#1131)."
        )
