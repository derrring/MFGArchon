"""
Finite-element (mesh + Lagrange) backend for the weak-form discretization
protocol. Assembly is implemented with scikit-fem, but that is an internal
detail; the backend is named for the method, not the library.

``FEMDiscretization`` adapts the ``fem.assembly`` free functions to the
``WeakFormDiscretization`` protocol so weak-form solvers can be written against
the protocol rather than against ``skfem.Basis`` directly. It is a thin adapter:
each method delegates to the corresponding ``assemble_*`` function, so behavior
is identical to calling them directly.

Issue #1131 Phase 1.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .assembly import (
    assemble_advection,
    assemble_gradient_projection,
    assemble_mass,
    assemble_stiffness,
)

if TYPE_CHECKING:
    import skfem

    from numpy.typing import NDArray
    from scipy import sparse


class FEMDiscretization:
    """Mesh + Lagrange finite-element implementation of ``WeakFormDiscretization``.

    Assembly backend: scikit-fem (internal detail).
    """

    def __init__(self, basis: skfem.Basis) -> None:
        self._basis = basis

    @property
    def n_dof(self) -> int:
        return int(self._basis.N)

    @property
    def dim(self) -> int:
        return int(self._basis.mesh.dim())

    @property
    def dof_coordinates(self) -> NDArray:
        return self._basis.doflocs.T

    def stiffness(self) -> sparse.csr_matrix:
        return assemble_stiffness(self._basis)

    def mass(self) -> sparse.csr_matrix:
        return assemble_mass(self._basis)

    def advection(self, velocity: NDArray) -> sparse.csr_matrix:
        return assemble_advection(self._basis, velocity)

    def gradient_projection(self) -> list[sparse.csr_matrix]:
        return assemble_gradient_projection(self._basis)


if __name__ == "__main__":
    """Equivalence smoke test: the adapter must reproduce the free functions."""
    import skfem

    import numpy as np
    from scipy.sparse import linalg as sla

    from mfgarchon.alg.numerical.weak_form_discretization import WeakFormDiscretization

    from .assembly import create_basis

    mesh = skfem.MeshTri.init_sqsymmetric()
    basis = create_basis(mesh, order=1)
    disc: WeakFormDiscretization = FEMDiscretization(basis)

    assert isinstance(disc, WeakFormDiscretization), "protocol conformance failed"
    assert disc.n_dof == basis.N
    assert disc.dim == basis.mesh.dim()

    # Each operator must equal the free-function result exactly.
    assert sla.norm(disc.stiffness() - assemble_stiffness(basis)) == 0.0
    assert sla.norm(disc.mass() - assemble_mass(basis)) == 0.0
    for Rd, Rd_free in zip(disc.gradient_projection(), assemble_gradient_projection(basis), strict=True):
        assert sla.norm(Rd - Rd_free) == 0.0
    v = np.ones((mesh.dim(), basis.N))
    assert sla.norm(disc.advection(v) - assemble_advection(basis, v)) == 0.0

    print("FEMDiscretization equivalence smoke test passed.")
