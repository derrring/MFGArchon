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

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from .assembly import (
    _import_skfem,
    assemble_advection,
    assemble_gradient_projection,
    assemble_mass,
    assemble_stiffness,
)

if TYPE_CHECKING:
    import skfem

    from numpy.typing import NDArray
    from scipy import sparse


@dataclass(frozen=True)
class LumpedMassPointMeasure:
    """A finite-element space's dofs as a point measure (`PointMeasure`, #1642 stage 1b-2): each dof with its consistent
    mass matrix's row sum, ``integral phi_i``, as its weight, on the mesh's measure, the sum of its element volumes.

    For P1 every row sum is positive and the row sums sum to the mesh's measure, by the partition of unity; the
    quadrature they make is the lumped mass. The arrays are held as read-only float64 copies.
    """

    points: NDArray
    weights: NDArray
    domain_measure: float

    def __post_init__(self) -> None:
        for name in ("points", "weights"):
            array = np.array(getattr(self, name), dtype=np.float64)
            array.flags.writeable = False
            object.__setattr__(self, name, array)


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

    @property
    def lumps_time_mass(self) -> bool:
        """A P1 simplex element lumps the mass in the time derivative, ``diag(M 1)``; every other element keeps the
        consistent mass (#2845).

        The lumped implicit pure-diffusion step is an M-matrix step, so it takes no density negative, wherever every
        off-diagonal of the stiffness is <= 0: on P1 in 1-D always; in 2-D when each interior edge's two opposite angles
        sum to at most pi and each boundary edge's opposite angle is at most pi/2; in 3-D a Delaunay mesh does not give
        it. P2 keeps the consistent mass, since its vertex functions integrate to about zero or below (the row sums
        ``LumpedMassPointMeasure`` reads are positive only for P1); so do the tensor-product elements, Q1 included.
        """
        skfem = _import_skfem()
        return isinstance(self._basis.elem, (skfem.ElementLineP1, skfem.ElementTriP1, skfem.ElementTetP1))

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
