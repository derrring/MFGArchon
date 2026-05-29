"""
Backend-agnostic weak-form discretization protocol.

A weak-form HJB/FP solver depends only on the assembled sparse operators, not on
how they are built. Finite elements (mesh + Lagrange, via scikit-fem) and
meshfree Galerkin (point cloud + Moving Least Squares) produce the same operators
from the same bilinear forms; only the shape-function evaluation differs.
Decoupling the solver from the assembly backend is the purpose of this protocol.

Implementations:
- ``FEMDiscretization`` (``fem/discretization.py``) -- mesh + Lagrange.
- ``MeshlessGalerkinDiscretization`` -- point cloud + MLS (Phase 2).

Issue #1131.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from scipy import sparse


@runtime_checkable
class WeakFormDiscretization(Protocol):
    """Assembled weak-form operators a solver needs, independent of backend.

    All operators are returned as ``scipy`` sparse matrices of shape
    ``(n_dof, n_dof)`` (the gradient projection as one such matrix per spatial
    dimension), matching the coupling layer's expectations
    (``FixedPointIterator``, adjoint operators).
    """

    @property
    def n_dof(self) -> int:
        """Number of degrees of freedom (rows/cols of every operator)."""
        ...

    @property
    def dim(self) -> int:
        """Spatial dimension."""
        ...

    @property
    def dof_coordinates(self) -> NDArray:
        """Coordinates of the degrees of freedom, shape ``(n_dof, dim)``.

        The point at which nodal fields (value function, density, drift) live;
        a basis-agnostic solver evaluates the Hamiltonian here. For finite
        elements these are the Lagrange dof locations; for meshless Galerkin,
        the collocation cloud.
        """
        ...

    def stiffness(self) -> sparse.csr_matrix:
        r"""Stiffness matrix $K_{ij} = \int_\Omega \nabla\phi_i \cdot \nabla\phi_j \, dx$.

        Symmetric. Discretizes $-\Delta$; scale by the diffusion coefficient.
        """
        ...

    def mass(self) -> sparse.csr_matrix:
        r"""Mass matrix $M_{ij} = \int_\Omega \phi_i \phi_j \, dx$. Symmetric PD."""
        ...

    def advection(self, velocity: NDArray) -> sparse.csr_matrix:
        r"""Advection matrix $C_{ij} = \int_\Omega (v \cdot \nabla\phi_j) \phi_i \, dx$.

        Not symmetric in general. ``velocity`` is the field at the degrees of
        freedom, shape ``(dim, n_dof)`` (or ``(n_dof,)`` in 1D).
        """
        ...

    def gradient_projection(self) -> list[sparse.csr_matrix]:
        r"""Gradient-projection operators $[R_0, \dots, R_{d-1}]$.

        $R_d$ maps nodal values to the weak-form $d$-th partial derivative,
        used to build the Hamiltonian Jacobian in Newton iteration.
        """
        ...
