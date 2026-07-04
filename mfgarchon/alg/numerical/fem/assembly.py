"""
FEM assembly using scikit-fem backend.

Provides thin wrappers around skfem.assembly.asm() for the bilinear forms
needed in MFG systems: Laplacian (diffusion), mass, and advection.

All functions return scipy sparse matrices compatible with MFGarchon's
coupling layer (FixedPointIterator, BlockIterator, adjoint operators).

Issue #773 Phase 1: Core assembly integration
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy import sparse

if TYPE_CHECKING:
    import skfem

    from numpy.typing import NDArray


def _import_skfem():
    try:
        import skfem
    except ImportError:
        raise ImportError("scikit-fem is required for FEM assembly. Install with: pip install scikit-fem") from None
    return skfem


def create_basis(mesh: skfem.Mesh, order: int = 1) -> skfem.Basis:
    """
    Create a finite element basis on the given mesh.

    Args:
        mesh: scikit-fem Mesh object.
        order: Polynomial order (1 = P1/linear, 2 = P2/quadratic).

    Returns:
        scikit-fem Basis object for assembly.
    """
    skfem = _import_skfem()

    # Map mesh type + order to element type
    element_map = {
        (skfem.MeshTri, 1): skfem.ElementTriP1,
        (skfem.MeshTri, 2): skfem.ElementTriP2,
        (skfem.MeshTri1, 1): skfem.ElementTriP1,
        (skfem.MeshTri1, 2): skfem.ElementTriP2,
        (skfem.MeshTet, 1): skfem.ElementTetP1,
        (skfem.MeshTet, 2): skfem.ElementTetP2,
        (skfem.MeshTet1, 1): skfem.ElementTetP1,
        (skfem.MeshTet1, 2): skfem.ElementTetP2,
        (skfem.MeshQuad, 1): skfem.ElementQuad1,
        (skfem.MeshQuad, 2): skfem.ElementQuad2,
        (skfem.MeshQuad1, 1): skfem.ElementQuad1,
        (skfem.MeshQuad1, 2): skfem.ElementQuad2,
        (skfem.MeshHex, 1): skfem.ElementHex1,  # Issue #470: 3D tensor-product family
        (skfem.MeshHex, 2): skfem.ElementHex2,
        (skfem.MeshLine, 1): skfem.ElementLineP1,
        (skfem.MeshLine, 2): skfem.ElementLineP2,
        (skfem.MeshLine1, 1): skfem.ElementLineP1,
        (skfem.MeshLine1, 2): skfem.ElementLineP2,
    }

    element_cls = None
    for (mesh_cls, ord_), elem_cls in element_map.items():
        if isinstance(mesh, mesh_cls) and ord_ == order:
            element_cls = elem_cls
            break

    if element_cls is None:
        raise ValueError(
            f"No element for mesh type {type(mesh).__name__} with order {order}. "
            f"Supported: P1 (order=1) and P2 (order=2) for Tri/Tet/Line/Quad/Hex."
        )

    basis = skfem.Basis(mesh, element_cls())

    # Issue #1489 (F7): a degenerate element (zero-area / collinear / coincident nodes) has a ~zero
    # measure, so assembly (skfem.asm) fills its rows with NaN/Inf entries and only emits a numpy
    # RuntimeWarning -- never an exception -- corrupting the solve silently. Fail loud at basis
    # creation. skfem integrates with |detDF|, so an inverted-but-nonzero element still contributes its
    # true area; guard only the near-zero (genuinely degenerate) case, relative to the largest element.
    measures = np.asarray(basis.dx).sum(axis=1)  # per-element measure = integral of 1 over each element
    scale = float(measures.max()) if measures.size else 0.0
    bad = np.nonzero(measures < 1e-12 * max(scale, 1e-300))[0]
    if bad.size:
        raise ValueError(
            f"{bad.size} degenerate mesh element(s) with near-zero measure: e.g. element {int(bad[0])} "
            f"has measure {measures[bad[0]]:.3e} vs a max of {scale:.3e}. Zero-area / collinear / "
            f"coincident-node elements produce NaN stiffness and mass entries silently under assembly "
            f"(Issue #1489). Remove the degenerate / sliver elements from the mesh."
        )
    return basis


def assemble_stiffness(basis: skfem.Basis) -> sparse.csr_matrix:
    """
    Assemble the stiffness matrix K where K[i,j] = integral(grad(phi_i) . grad(phi_j)).

    This is the FEM discretization of the Laplacian: -div(grad(u)).
    For the diffusion term (sigma^2/2) * Laplacian(u), multiply by the coefficient.

    Args:
        basis: scikit-fem Basis object.

    Returns:
        Stiffness matrix K as scipy CSR matrix, shape (N_dof, N_dof).
    """
    skfem = _import_skfem()
    from skfem.models import laplace

    return skfem.asm(laplace, basis)


def assemble_mass(basis: skfem.Basis) -> sparse.csr_matrix:
    """
    Assemble the mass matrix M where M[i,j] = integral(phi_i * phi_j).

    Used for:
    - Time discretization: (M/dt + K) u^{n+1} = M/dt * u^n
    - L2 projection
    - Mass conservation verification

    Args:
        basis: scikit-fem Basis object.

    Returns:
        Mass matrix M as scipy CSR matrix, shape (N_dof, N_dof).
    """
    skfem = _import_skfem()
    from skfem.models import mass

    return skfem.asm(mass, basis)


def assemble_advection(
    basis: skfem.Basis,
    velocity: NDArray,
) -> sparse.csr_matrix:
    """
    Assemble the advection matrix A where A[i,j] = integral(v . grad(phi_j) * phi_i).

    This discretizes the advection term v . grad(u) in the HJB equation,
    or div(v * m) in the FP equation (via integration by parts).

    Args:
        basis: scikit-fem Basis object.
        velocity: Velocity field at DOF nodes, shape (dim, N_dof) or (N_dof,) for 1D.

    Returns:
        Advection matrix A as scipy CSR matrix, shape (N_dof, N_dof).
    """
    skfem = _import_skfem()
    from skfem import BilinearForm

    velocity = np.asarray(velocity, dtype=np.float64)
    if velocity.ndim == 1:
        velocity = velocity.reshape(1, -1)

    # Interpolate velocity to quadrature points
    v_at_dofs = velocity  # (dim, N_dof)

    @BilinearForm
    def advection_form(u, v, w):
        # w.x are quadrature point coordinates
        # u.grad is (dim, ...) gradient of trial function
        # v.value is test function value
        # Interpolate velocity field to integration points
        dim = u.grad.shape[0]
        result = 0.0
        for d in range(dim):
            v_d = basis.interpolate(v_at_dofs[d])
            result += v_d.value * u.grad[d]
        return result * v.value

    return skfem.asm(advection_form, basis)


def assemble_gradient_projection(basis: skfem.Basis) -> list[sparse.csr_matrix]:
    """
    Assemble gradient projection matrices R_d where R_d[i,j] = integral(dphi_j/dx_d * phi_i).

    These matrices map nodal values to their weak-form gradient representation.
    The nodal gradient is recovered via mass-lumped projection: grad_d(u) = diag(M_lumped)^{-1} @ R_d @ u.

    Used for Newton iteration in HJBFEMSolver: the Jacobian of the Hamiltonian term
    requires the gradient operator as a matrix (not just a function evaluation).

    Args:
        basis: scikit-fem Basis object.

    Returns:
        List of gradient projection matrices [R_0, R_1, ...], one per spatial dimension.
        Each R_d is a scipy CSR matrix of shape (N_dof, N_dof).
    """
    skfem = _import_skfem()
    from skfem import BilinearForm

    dim = basis.mesh.dim()
    R_list = []

    for d in range(dim):

        @BilinearForm
        def grad_proj_form(u, v, w, dim_idx=d):
            return u.grad[dim_idx] * v.value

        R_d = skfem.asm(grad_proj_form, basis)
        R_list.append(R_d)

    return R_list


def apply_dirichlet_bc(
    matrix: sparse.csr_matrix,
    rhs: NDArray,
    boundary_dofs: NDArray,
    values: NDArray | float = 0.0,
) -> tuple[sparse.csr_matrix, NDArray]:
    """
    Apply Dirichlet boundary conditions using scikit-fem's condense approach.

    Modifies the system (matrix, rhs) to enforce u = values at boundary DOFs.

    Args:
        matrix: System matrix (N_dof, N_dof).
        rhs: Right-hand side vector (N_dof,).
        boundary_dofs: Indices of boundary DOFs.
        values: Dirichlet values (scalar or array matching boundary_dofs).

    Returns:
        Tuple of (modified_matrix, modified_rhs).
    """
    if isinstance(values, (int, float)):
        values = np.full(len(boundary_dofs), float(values))

    # Use skfem.condense for clean BC application
    interior = np.setdiff1d(np.arange(matrix.shape[0]), boundary_dofs)

    K_int = matrix[np.ix_(interior, interior)]
    f_int = rhs[interior] - matrix[np.ix_(interior, boundary_dofs)] @ values

    return K_int, f_int


if __name__ == "__main__":
    """Smoke test for FEM assembly."""
    import skfem

    print("Testing FEM assembly...")

    # Create mesh and basis
    mesh = skfem.MeshTri.init_sqsymmetric()
    basis = create_basis(mesh, order=1)
    print(f"Mesh: {mesh.t.shape[1]} elements, {mesh.p.shape[1]} nodes")
    print(f"DOFs: {basis.N}")

    # Assemble matrices
    K = assemble_stiffness(basis)
    M = assemble_mass(basis)
    print(f"Stiffness K: {K.shape}, nnz={K.nnz}")
    print(f"Mass M: {M.shape}, nnz={M.nnz}")

    # Verify symmetry
    assert sparse.linalg.norm(K - K.T) < 1e-12, "K not symmetric"
    assert sparse.linalg.norm(M - M.T) < 1e-12, "M not symmetric"
    print("Symmetry: OK")

    # Verify mass matrix row sums (should be total area)
    total_mass = M.sum()
    print(f"Mass total: {total_mass:.6f} (expected: 1.0 for unit square)")

    # Test Dirichlet BC
    boundary = basis.mesh.boundary_nodes()
    rhs = np.ones(basis.N)
    K_int, f_int = apply_dirichlet_bc(K, rhs, boundary, values=0.0)
    print(f"After BC: system size {K_int.shape[0]} (from {K.shape[0]})")

    print("All assembly tests passed.")
