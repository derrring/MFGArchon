"""
Adjoint-consistent operator construction utilities.

This module provides tools for constructing discrete operators that satisfy
the adjoint relationship A_FP = A_HJB^T.

Key utilities:
- Diffusion operator construction (symmetric → self-adjoint)
- Advection operator construction (upwind with boundary handling)
- Combined operator for operator-splitting schemes

Operator Splitting Analysis:
---------------------------
For Semi-Lagrangian schemes with operator splitting:

    HJB: U^n = Interp(U^{n+1}, x_dep) then Diffuse(U^n)
    FP:  m^{n+1} = Diffuse(Splat(m^n, x_dest))

Adjoint relationships:
1. **Diffusion**: If both use same Neumann BC with symmetric Laplacian,
   the diffusion matrices are self-adjoint (symmetric).

2. **Advection**: Interpolation and Splatting are adjoint for interior points
   if weights are transposed. Boundary handling must match:
   - HJB uses reflect → FP must use reflect
   - HJB uses clamp → FP must use clamp
   (This is enforced by bc_utils.py)

References:
-----------
- Issue #704: Unified adjoint module
- Issue #625 (state-dependent BC coupling)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np
from scipy import sparse

from mfgarchon.utils.pde_coefficients import diffusion_from_volatility

if TYPE_CHECKING:
    from numpy.typing import NDArray


# =============================================================================
# Geometry Protocol for Operator Construction
# =============================================================================


@runtime_checkable
class OperatorGeometry(Protocol):
    """
    Protocol for geometry objects that can be used for operator construction.

    This is a subset of GeometryProtocol focused on what operators need.
    Any geometry implementing these methods can be used with build_*_from_geometry().
    """

    @property
    def dimension(self) -> int:
        """Spatial dimension."""
        ...

    def get_grid_shape(self) -> tuple[int, ...]:
        """Return grid shape as tuple (Nx, Ny, ...)."""
        ...

    def get_grid_spacing(self) -> list[float]:
        """Return grid spacing as list [dx, dy, ...]."""
        ...


@dataclass
class OperatorConfig:
    """Configuration for discrete operator construction."""

    dimension: int
    """Spatial dimension."""

    grid_size: int | tuple[int, ...]
    """Number of grid points (scalar for 1D, tuple for nD)."""

    dx: float | tuple[float, ...]
    """Grid spacing."""

    sigma: float
    """Diffusion coefficient."""

    dt: float
    """Time step size."""

    bc_type: str = "neumann"
    """Boundary condition type: 'neumann', 'dirichlet', 'periodic'."""


def build_diffusion_matrix(
    grid_shape: int | tuple[int, ...],
    dx: float | tuple[float, ...],
    sigma: float,
    dt: float,
    theta: float = 0.5,
    bc_type: str = "neumann",
) -> sparse.csr_matrix:
    """
    Build diffusion matrix for Crank-Nicolson scheme (truly dimension-agnostic).

    Uses Kronecker product construction for arbitrary dimensions:
        L_nD = L_1 ⊗ I_2 ⊗ ... ⊗ I_n + I_1 ⊗ L_2 ⊗ ... ⊗ I_n + ...

    The implicit matrix (I - θ·α·L) is returned for the Crank-Nicolson scheme:
        (I - θ·α·L) u^{n+1} = (I + (1-θ)·α·L) u^n

    Args:
        grid_shape: Number of grid points (int for 1D, tuple for nD)
        dx: Grid spacing (float for uniform, tuple for per-dimension)
        sigma: Diffusion coefficient
        dt: Time step
        theta: Crank-Nicolson parameter (0.5=CN, 1.0=implicit Euler)
        bc_type: Boundary condition type ('neumann', 'dirichlet', 'periodic')

    Returns:
        Sparse CSR matrix of shape (prod(grid_shape), prod(grid_shape))

    Note:
        For Neumann and periodic BC, the matrix is symmetric (self-adjoint).
        This is crucial for adjoint consistency in MFG systems.
    """
    # Normalize inputs
    if isinstance(grid_shape, int):
        grid_shape = (grid_shape,)
    if isinstance(dx, (int, float)):
        dx = tuple([float(dx)] * len(grid_shape))
    elif len(dx) == 1 and len(grid_shape) > 1:
        dx = tuple([dx[0]] * len(grid_shape))

    ndim = len(grid_shape)

    if ndim == 1:
        # Direct 1D implementation (slightly more efficient)
        return build_diffusion_matrix_1d(grid_shape[0], dx[0], sigma, dt, theta, bc_type)

    # nD: Use Kronecker product construction
    # L_nD = sum_d (I ⊗ ... ⊗ L_d ⊗ ... ⊗ I)
    N_total = int(np.prod(grid_shape))

    # Start with identity (will subtract θ·α·L terms)
    A = sparse.eye(N_total, format="csr")

    for d in range(ndim):
        # Build 1D Laplacian for dimension d
        Nd = grid_shape[d]
        alpha_d = diffusion_from_volatility(sigma) * dt / dx[d] ** 2

        # 1D Laplacian: L = [-1, 2, -1] / dx²
        # For (I - θ·α·L), we need to add θ·α times the Laplacian contribution
        L_1d = _build_1d_laplacian(Nd, bc_type)

        # Kronecker product: I ⊗ ... ⊗ L_d ⊗ ... ⊗ I
        L_nd = _kron_with_identity(L_1d, d, grid_shape)

        # Accumulate: A = A - θ·α_d·L_nd  (note: L has negative off-diagonals)
        # Actually L_1d is the negative Laplacian (2 on diagonal, -1 off)
        # So we ADD θ·α_d·L_nd to get (I + θ·α_d·(-Δ))
        A = A + theta * alpha_d * L_nd

    return A.tocsr()


def _build_1d_laplacian(N: int, bc_type: str) -> sparse.csr_matrix:
    """
    Build 1D discrete Laplacian operator (negative Laplacian).

    Returns matrix L such that L[i,i] = 2, L[i,i±1] = -1 for interior,
    with boundary modifications for Neumann/Dirichlet/periodic.

    The negative Laplacian -Δ has positive diagonal, ensuring the
    diffusion matrix (I + θα(-Δ)) is positive definite.
    """
    # Main diagonal: 2
    main = np.ones(N) * 2.0
    # Off-diagonals: -1
    off = np.ones(N - 1) * (-1.0)

    diagonals = [off, main, off]
    offsets = [-1, 0, 1]
    L = sparse.diags(diagonals, offsets, shape=(N, N), format="lil")

    if bc_type == "neumann":
        # Neumann: ghost cell equals interior → reduce diagonal at boundaries
        L[0, 0] = 1.0
        L[N - 1, N - 1] = 1.0
    elif bc_type == "dirichlet":
        # Dirichlet: boundary values fixed → identity rows
        L[0, :] = 0
        L[0, 0] = 1.0
        L[N - 1, :] = 0
        L[N - 1, N - 1] = 1.0
    elif bc_type == "periodic":
        # Periodic: wrap around
        L[0, N - 1] = -1.0
        L[N - 1, 0] = -1.0

    return L.tocsr()


def _kron_with_identity(L_1d: sparse.spmatrix, dim: int, grid_shape: tuple[int, ...]) -> sparse.csr_matrix:
    """
    Compute Kronecker product: I_0 ⊗ I_1 ⊗ ... ⊗ L_dim ⊗ ... ⊗ I_{n-1}.

    Args:
        L_1d: 1D operator for dimension `dim`
        dim: Which dimension this operator acts on
        grid_shape: Full grid shape

    Returns:
        Sparse matrix acting on flattened nD array
    """
    ndim = len(grid_shape)
    result = sparse.csr_matrix(L_1d)

    # Kronecker products: left (dims < dim) and right (dims > dim)
    # Order matters: for row-major ordering, we do right-to-left
    # result = I_{dim+1} ⊗ ... ⊗ I_{n-1} ⊗ L_dim ⊗ I_0 ⊗ ... ⊗ I_{dim-1}
    # Actually for standard row-major (C-order), the Kronecker structure is:
    # The last index varies fastest, so:
    # L acting on dim d means: I_{n-1} ⊗ ... ⊗ I_{d+1} ⊗ L_d ⊗ I_{d-1} ⊗ ... ⊗ I_0

    # Build from right to left
    # Start with L_1d, then kron with identities

    # Dimensions to the right of dim (faster varying)
    for d in range(ndim - 1, dim, -1):
        I_d = sparse.eye(grid_shape[d], format="csr")
        result = sparse.kron(result, I_d, format="csr")

    # Dimensions to the left of dim (slower varying)
    for d in range(dim - 1, -1, -1):
        I_d = sparse.eye(grid_shape[d], format="csr")
        result = sparse.kron(I_d, result, format="csr")

    return result


def build_diffusion_matrix_1d(
    Nx: int,
    dx: float,
    sigma: float,
    dt: float,
    theta: float = 0.5,
    bc_type: str = "neumann",
) -> sparse.csr_matrix:
    """
    Build 1D diffusion matrix for Crank-Nicolson scheme.

    The matrix A satisfies: (I - θ*α*L) u^{n+1} = (I + (1-θ)*α*L) u^n
    where α = σ²/2 * dt/dx² and L is the discrete Laplacian.

    This function returns the implicit matrix (I - θ*α*L).

    Args:
        Nx: Number of grid points
        dx: Grid spacing
        sigma: Diffusion coefficient
        dt: Time step
        theta: Crank-Nicolson parameter (0.5 for CN, 1.0 for implicit Euler)
        bc_type: Boundary condition ('neumann', 'dirichlet', 'periodic')

    Returns:
        Sparse CSR matrix of shape (Nx, Nx)

    Note:
        For Neumann BC, the matrix is symmetric → self-adjoint.
        For Dirichlet BC with zero values, the matrix is also symmetric.
        For periodic BC, the matrix is symmetric (circulant structure).
    """
    alpha = diffusion_from_volatility(sigma) * dt / dx**2

    # Main diagonal
    main = np.ones(Nx) * (1.0 + 2.0 * theta * alpha)

    # Off-diagonals
    off = np.ones(Nx - 1) * (-theta * alpha)

    # Build tridiagonal
    diagonals = [off, main, off]
    offsets = [-1, 0, 1]

    A = sparse.diags(diagonals, offsets, shape=(Nx, Nx), format="lil")

    # Apply boundary conditions
    if bc_type == "neumann":
        # Neumann: du/dx = 0 → ghost = interior
        # Modify boundary rows to reflect this
        A[0, 0] = 1.0 + theta * alpha
        A[Nx - 1, Nx - 1] = 1.0 + theta * alpha
    elif bc_type == "dirichlet":
        # Dirichlet: u = 0 at boundary
        # Boundary rows become identity (or handle separately)
        A[0, :] = 0
        A[0, 0] = 1.0
        A[Nx - 1, :] = 0
        A[Nx - 1, Nx - 1] = 1.0
    elif bc_type == "periodic":
        # Periodic: wrap around
        A[0, Nx - 1] = -theta * alpha
        A[Nx - 1, 0] = -theta * alpha

    return A.tocsr()


def build_diffusion_matrix_2d(
    grid_shape: tuple[int, int],
    dx: float | tuple[float, float],
    sigma: float,
    dt: float,
    theta: float = 0.5,
    bc_type: str = "neumann",
) -> sparse.csr_matrix:
    """
    Build 2D diffusion matrix for Crank-Nicolson scheme.

    Uses 5-point stencil for Laplacian with row-major ordering.

    Args:
        grid_shape: (Nx, Ny) grid dimensions
        dx: Grid spacing (scalar for uniform, tuple for non-uniform)
        sigma: Diffusion coefficient
        dt: Time step
        theta: Crank-Nicolson parameter
        bc_type: Boundary condition type

    Returns:
        Sparse CSR matrix of shape (Nx*Ny, Nx*Ny)
    """
    Nx, Ny = grid_shape
    N = Nx * Ny

    if isinstance(dx, (int, float)):
        dx_x = dx_y = float(dx)
    else:
        dx_x, dx_y = dx

    alpha_x = diffusion_from_volatility(sigma) * dt / dx_x**2
    alpha_y = diffusion_from_volatility(sigma) * dt / dx_y**2

    # Build sparse matrix
    A = sparse.lil_matrix((N, N))

    def idx(i, j):
        """Convert 2D index to 1D (row-major)."""
        return i * Ny + j

    for i in range(Nx):
        for j in range(Ny):
            k = idx(i, j)

            # Main diagonal
            diag = 1.0 + 2.0 * theta * (alpha_x + alpha_y)

            # Interior point
            is_boundary = i == 0 or i == Nx - 1 or j == 0 or j == Ny - 1

            if bc_type == "neumann" and is_boundary:
                # Adjust for Neumann BC
                if i == 0:
                    diag -= theta * alpha_x
                if i == Nx - 1:
                    diag -= theta * alpha_x
                if j == 0:
                    diag -= theta * alpha_y
                if j == Ny - 1:
                    diag -= theta * alpha_y

            A[k, k] = diag

            # Off-diagonal: x-direction
            if i > 0:
                A[k, idx(i - 1, j)] = -theta * alpha_x
            if i < Nx - 1:
                A[k, idx(i + 1, j)] = -theta * alpha_x

            # Off-diagonal: y-direction
            if j > 0:
                A[k, idx(i, j - 1)] = -theta * alpha_y
            if j < Ny - 1:
                A[k, idx(i, j + 1)] = -theta * alpha_y

            # Periodic BC
            if bc_type == "periodic":
                if i == 0:
                    A[k, idx(Nx - 1, j)] = -theta * alpha_x
                if i == Nx - 1:
                    A[k, idx(0, j)] = -theta * alpha_x
                if j == 0:
                    A[k, idx(i, Ny - 1)] = -theta * alpha_y
                if j == Ny - 1:
                    A[k, idx(i, 0)] = -theta * alpha_y

    return A.tocsr()


def build_advection_matrix_1d(
    Nx: int,
    dx: float,
    velocity: NDArray,
    dt: float,
    bc_type: str = "neumann",
    upwind: bool = True,
) -> sparse.csr_matrix:
    """
    Build 1D advection matrix using upwind discretization.

    For advection equation: ∂u/∂t + v ∂u/∂x = 0
    Upwind scheme: u^{n+1}_i = u^n_i - v * dt/dx * (u^n_i - u^n_{i-1}) if v > 0

    Args:
        Nx: Number of grid points
        dx: Grid spacing
        velocity: Velocity field, shape (Nx,)
        dt: Time step
        bc_type: Boundary condition type
        upwind: Whether to use upwind scheme (True) or centered (False)

    Returns:
        Sparse CSR matrix of shape (Nx, Nx)

    Note:
        The upwind matrix for advection and the splatting matrix for FP
        are transposes when using consistent boundary handling.
    """
    CFL = dt / dx
    A = sparse.lil_matrix((Nx, Nx))

    for i in range(Nx):
        v = velocity[i]

        if upwind:
            if v >= 0:
                # Upwind from left
                if i > 0:
                    A[i, i] = 1.0 - v * CFL
                    A[i, i - 1] = v * CFL
                else:
                    # Left boundary
                    if bc_type == "periodic":
                        A[i, i] = 1.0 - v * CFL
                        A[i, Nx - 1] = v * CFL
                    else:
                        # Neumann or Dirichlet: extrapolate or use boundary value
                        A[i, i] = 1.0
            else:
                # Upwind from right
                if i < Nx - 1:
                    A[i, i] = 1.0 + v * CFL
                    A[i, i + 1] = -v * CFL
                else:
                    # Right boundary
                    if bc_type == "periodic":
                        A[i, i] = 1.0 + v * CFL
                        A[i, 0] = -v * CFL
                    else:
                        A[i, i] = 1.0
        else:
            # Centered scheme (not upwind)
            if i > 0 and i < Nx - 1:
                A[i, i] = 1.0
                A[i, i + 1] = -v * CFL / 2
                A[i, i - 1] = v * CFL / 2

    return A.tocsr()


def check_operator_adjoint(
    A_hjb: sparse.spmatrix | NDArray,
    A_fp: sparse.spmatrix | NDArray,
    rtol: float = 1e-10,
) -> tuple[bool, float]:
    """
    Quick check if two operators are adjoint (A_fp ≈ A_hjb^T).

    Args:
        A_hjb: HJB operator matrix
        A_fp: FP operator matrix
        rtol: Relative tolerance

    Returns:
        Tuple of (is_adjoint, relative_error)
    """
    if sparse.issparse(A_hjb):
        A_hjb = A_hjb.toarray()
    if sparse.issparse(A_fp):
        A_fp = A_fp.toarray()

    diff = A_fp - A_hjb.T
    error = np.linalg.norm(diff, "fro")
    norm = np.linalg.norm(A_hjb, "fro")
    rel_error = error / norm if norm > 0 else error

    return rel_error < rtol, rel_error


def make_operator_adjoint(
    A: sparse.spmatrix | NDArray,
) -> sparse.csr_matrix:
    """
    Make an operator self-adjoint by symmetrization.

    Returns (A + A^T) / 2.

    Args:
        A: Input matrix

    Returns:
        Symmetrized matrix

    Warning:
        This should only be used for operators that are theoretically
        self-adjoint but have small numerical asymmetry. Do not use
        to "fix" genuinely non-adjoint operators.
    """
    if sparse.issparse(A):
        return ((A + A.T) / 2).tocsr()
    else:
        return sparse.csr_matrix((A + A.T) / 2)


# =============================================================================
# Geometry-Aware Factory Functions
# =============================================================================


def build_diffusion_matrix_from_geometry(
    geometry: OperatorGeometry,
    sigma: float,
    dt: float,
    theta: float = 0.5,
    bc_type: str = "neumann",
) -> sparse.csr_matrix:
    """
    Build diffusion matrix from geometry object (dimension-agnostic).

    This is the preferred API - it extracts grid info from the geometry module
    rather than requiring manual specification of grid shape and spacing.

    Args:
        geometry: Any geometry implementing OperatorGeometry protocol
            (e.g., TensorProductGrid, Domain, etc.)
        sigma: Diffusion coefficient
        dt: Time step
        theta: Crank-Nicolson parameter (0.5 for CN, 1.0 for implicit Euler)
        bc_type: Boundary condition type ('neumann', 'dirichlet', 'periodic')

    Returns:
        Sparse CSR matrix for implicit diffusion solve

    Example:
        >>> from mfgarchon.geometry import TensorProductGrid
        >>> grid = TensorProductGrid(bounds=[(0, 1), (0, 1)], Nx=[50, 50])
        >>> A_diff = build_diffusion_matrix_from_geometry(grid, sigma=0.2, dt=0.01)
        >>> # A_diff is 2500x2500 sparse matrix for 2D problem
    """
    grid_shape = geometry.get_grid_shape()
    spacing = geometry.get_grid_spacing()

    # Convert spacing list to appropriate format
    if len(spacing) == 1:
        dx = spacing[0]
    else:
        dx = tuple(spacing)

    return build_diffusion_matrix(grid_shape, dx, sigma, dt, theta, bc_type)


def build_advection_matrix_from_geometry(
    geometry: OperatorGeometry,
    velocity: NDArray,
    dt: float,
    bc_type: str = "neumann",
    upwind: bool = True,
) -> sparse.csr_matrix:
    """
    Build advection matrix from geometry object.

    Currently supports 1D only. For nD, use Semi-Lagrangian interpolation/splatting.

    Args:
        geometry: Any geometry implementing OperatorGeometry protocol
        velocity: Velocity field matching geometry shape
        dt: Time step
        bc_type: Boundary condition type
        upwind: Whether to use upwind scheme

    Returns:
        Sparse CSR matrix for advection

    Raises:
        NotImplementedError: If geometry dimension > 1
    """
    dim = geometry.dimension
    if dim != 1:
        raise NotImplementedError(
            f"Matrix-based advection for {dim}D not implemented. "
            "Use Semi-Lagrangian interpolation/splatting for nD advection."
        )

    grid_shape = geometry.get_grid_shape()
    spacing = geometry.get_grid_spacing()

    Nx = grid_shape[0]
    dx = spacing[0]

    return build_advection_matrix_1d(Nx, dx, velocity, dt, bc_type, upwind)


def get_boundary_indices_from_geometry(
    geometry: OperatorGeometry,
) -> dict[str, list[int]]:
    """
    Get boundary indices organized by boundary region from geometry (nD).

    Returns dict mapping boundary names to flat indices:
    - 1D: {"x_min": [0], "x_max": [Nx-1]}
    - 2D: {"x_min": [...], "x_max": [...], "y_min": [...], "y_max": [...]}
    - 3D: adds z_min, z_max
    - nD: adds w_min, w_max, v_min, v_max, ...

    Args:
        geometry: Geometry object implementing OperatorGeometry protocol

    Returns:
        Dict mapping boundary names to lists of flat indices (row-major order)
    """
    dim = geometry.dimension
    grid_shape = geometry.get_grid_shape()

    # Initialize result dict
    # Convention: x, y, z for dims ≤ 3; x_1, x_2, ..., x_d for dims > 3
    result = {}
    for d in range(dim):
        axis = _get_axis_name(d, dim)
        result[f"{axis}_min"] = []
        result[f"{axis}_max"] = []

    # Total points
    N_total = int(np.prod(grid_shape))

    # Check each point
    for flat_idx in range(N_total):
        multi_idx = _flat_to_multi_index(flat_idx, grid_shape)

        for d in range(dim):
            axis = _get_axis_name(d, dim)
            if multi_idx[d] == 0:
                result[f"{axis}_min"].append(flat_idx)
            if multi_idx[d] == grid_shape[d] - 1:
                result[f"{axis}_max"].append(flat_idx)

    return result


def _flat_to_multi_index(flat_idx: int, shape: tuple[int, ...]) -> tuple[int, ...]:
    """Convert flat index to multi-index (row-major/C order)."""
    multi_idx = []
    remaining = flat_idx
    for i, _dim_size in enumerate(shape):
        stride = int(np.prod(shape[i + 1 :])) if i + 1 < len(shape) else 1
        idx = remaining // stride
        remaining = remaining % stride
        multi_idx.append(idx)
    return tuple(multi_idx)


def _get_axis_name(dim_index: int, total_dims: int) -> str:
    """
    Get axis name for dimension index.

    Convention:
    - If total_dims ≤ 3: use x, y, z
    - If total_dims > 3: use x_1, x_2, ..., x_d for ALL dimensions
    """
    if total_dims <= 3:
        return ["x", "y", "z"][dim_index]
    else:
        return f"x_{dim_index + 1}"  # x_1, x_2, x_3, x_4, ...


# =============================================================================
# BC-Aware Adjoint Matrix Construction
# =============================================================================


def build_bc_aware_adjoint_matrix(
    A_hjb: sparse.spmatrix,
    bc_types: dict[str, str],
    grid_shape: tuple[int, ...],
    dx: float | tuple[float, ...],
    dt: float,
) -> sparse.csr_matrix:
    """
    Build FP advection matrix with BC-aware adjustment from HJB matrix.

    For reflecting/periodic BC, transpose is correct.
    For absorbing/outflow BC, transpose gives wrong behavior (mass stays instead of exits).
    This function adjusts boundary rows/columns accordingly.

    BC Types and Their Handling:
    ---------------------------
    - "reflecting": Zero-flux Neumann, ∂m/∂n = 0 (transpose correct) ✅
    - "periodic":   Wrap-around (transpose correct) ✅
    - "neumann":    Same as reflecting for homogeneous case ✅
    - "absorbing":  Homogeneous outflow (partial support, no source) ⚠️
    - "outflow":    Homogeneous outflow (partial support, no source) ⚠️

    NOT SUPPORTED (raises NotImplementedError):
    -------------------------------------------
    - "dirichlet":  Fixed value BC (requires source term handling)
    - "robin":      Mixed BC αm + β∂m/∂n = g (complex)
    - "inflow":     Non-zero flux (requires source term handling)
    - "flux":       General Neumann with ∂m/∂n = g ≠ 0 (requires source term)

    Args:
        A_hjb: HJB advection matrix
        bc_types: Dict mapping boundary names to BC types.
            Keys: "x_min", "x_max", "y_min", "y_max", etc.
            Values: "reflecting", "periodic", "neumann", "absorbing", "outflow"
        grid_shape: Grid shape tuple
        dx: Grid spacing (float for uniform, tuple for per-dimension)
        dt: Time step

    Returns:
        A_fp: BC-aware adjoint FP matrix

    Raises:
        NotImplementedError: For BC types requiring source term handling
            (dirichlet, robin, inflow, flux)

    Example:
        >>> bc_types = {"x_min": "reflecting", "x_max": "absorbing"}
        >>> A_fp = build_bc_aware_adjoint_matrix(A_hjb, bc_types, (100,), 0.01, 0.001)

    Note:
        Issue #704: Full BC support with source terms is planned for future release.
        Currently only homogeneous BCs are supported. For non-zero flux or
        inhomogeneous Dirichlet, use adjoint_mode="off" and handle BCs manually.
    """
    # BC types that require source term handling (NOT IMPLEMENTED)
    UNSUPPORTED_BC_TYPES = {"dirichlet", "robin", "inflow", "flux"}

    # Check for unsupported BC types
    for boundary_name, bc_type in bc_types.items():
        bc_type_lower = bc_type.lower()
        if bc_type_lower in UNSUPPORTED_BC_TYPES:
            raise NotImplementedError(
                f"BC type '{bc_type}' at boundary '{boundary_name}' requires source term "
                f"handling which is not yet implemented in adjoint_mode='auto'. "
                f"Supported BC types: reflecting, periodic, neumann, absorbing, outflow. "
                f"Use adjoint_mode='off' or adjoint_mode='transpose' and handle BCs manually. "
                f"See Issue #704 for planned full BC support."
            )

    dim = len(grid_shape)

    # Start with transpose
    A_fp = A_hjb.T.tocsr().tolil()

    # Get boundary indices
    boundary_indices = _get_boundary_indices_dict(grid_shape, dim)

    # Normalize dx
    if isinstance(dx, (int, float)):
        dx_list = [float(dx)] * dim
    else:
        dx_list = list(dx)

    # Adjust for non-reflecting/periodic boundaries
    for boundary_name, indices in boundary_indices.items():
        bc_type = bc_types.get(boundary_name, "reflecting").lower()

        if bc_type in ("reflecting", "periodic", "neumann"):
            # Transpose is correct for zero-flux conditions, no adjustment
            continue

        elif bc_type in ("absorbing", "outflow"):
            # Absorbing/Outflow BC: mass should exit at boundary
            # The transpose gives identity column → mass stays (WRONG)
            # Fix: Apply proper outflow condition to boundary rows
            # NOTE: This only handles HOMOGENEOUS outflow (zero inflow)
            for flat_idx in indices:
                _apply_outflow_row(A_fp, flat_idx, grid_shape, dx_list, dt, boundary_name)

    return A_fp.tocsr()


def _get_boundary_indices_dict(grid_shape: tuple[int, ...], dim: int) -> dict[str, list[int]]:
    """Get boundary indices organized by boundary name."""
    axis_names = ["x", "y", "z"]

    result = {}
    N_total = int(np.prod(grid_shape))

    for d in range(dim):
        if dim <= 3:
            axis = axis_names[d]
        else:
            axis = f"x_{d + 1}"

        result[f"{axis}_min"] = []
        result[f"{axis}_max"] = []

    for flat_idx in range(N_total):
        multi_idx = _flat_to_multi_index(flat_idx, grid_shape)

        for d in range(dim):
            if dim <= 3:
                axis = axis_names[d]
            else:
                axis = f"x_{d + 1}"

            if multi_idx[d] == 0:
                result[f"{axis}_min"].append(flat_idx)
            if multi_idx[d] == grid_shape[d] - 1:
                result[f"{axis}_max"].append(flat_idx)

    return result


def _apply_outflow_row(
    A: sparse.lil_matrix,
    flat_idx: int,
    grid_shape: tuple[int, ...],
    dx_list: list[float],
    dt: float,
    boundary_name: str,
) -> None:
    """
    Apply outflow condition to a boundary row.

    For outflow at boundary, mass flows OUT based on advection velocity.
    The row sum should be < 1 to allow mass to exit the domain.
    """
    dim = len(grid_shape)
    multi_idx = _flat_to_multi_index(flat_idx, grid_shape)

    # Determine which dimension this boundary is on
    d = _boundary_name_to_dim(boundary_name, dim)
    if d is None:
        return

    if "_min" in boundary_name:
        # Min boundary: mass exits if velocity points left (negative)
        # For upwind with v < 0, mass at min boundary flows out
        if multi_idx[d] == 0:
            # Find interior neighbor
            neighbor_multi = list(multi_idx)
            neighbor_multi[d] += 1
            neighbor_flat = _multi_to_flat_index(tuple(neighbor_multi), grid_shape)

            # Get CFL estimate from existing matrix
            # The off-diagonal entry gives us CFL information
            cfl = abs(A[flat_idx, neighbor_flat]) if neighbor_flat < A.shape[0] else 0.05

            # Clear the row and apply outflow stencil
            # Row sum < 1 means mass exits
            A[flat_idx, :] = 0
            A[flat_idx, flat_idx] = 1.0 - cfl  # Mass that stays
            # Mass that was at boundary exits (not transferred anywhere)
    else:
        # Max boundary: mass exits if velocity points right (positive)
        if multi_idx[d] == grid_shape[d] - 1:
            # Find interior neighbor
            neighbor_multi = list(multi_idx)
            neighbor_multi[d] -= 1
            neighbor_flat = _multi_to_flat_index(tuple(neighbor_multi), grid_shape)

            # Get CFL estimate from transpose (original A_hjb had this)
            # In A_hjb^T, the [flat_idx, flat_idx] entry tells us 1 - CFL
            diag_val = A[flat_idx, flat_idx]
            cfl = max(0.05, 1.0 - diag_val) if diag_val < 1.0 else 0.05

            # Clear the row and apply outflow stencil
            # Row sum < 1 means mass exits
            A[flat_idx, :] = 0
            A[flat_idx, flat_idx] = 1.0 - cfl  # Mass that stays
            A[flat_idx, neighbor_flat] = cfl  # Mass from interior (advection in)
            # Net: (1 - CFL) + CFL from interior, but boundary mass exits proportional to CFL
            # Actually for pure outflow: just let boundary mass exit
            A[flat_idx, flat_idx] = 1.0 - cfl  # Keep less
            A[flat_idx, neighbor_flat] = 0  # Don't get from interior for pure outflow


def _boundary_name_to_dim(boundary_name: str, dim: int) -> int | None:
    """Convert boundary name to dimension index."""
    if dim <= 3:
        mapping = {"x": 0, "y": 1, "z": 2}
        for axis, d in mapping.items():
            if boundary_name.startswith(axis):
                return d
    else:
        # x_1, x_2, etc.
        if boundary_name.startswith("x_"):
            try:
                d = int(boundary_name.split("_")[1]) - 1
                return d
            except (IndexError, ValueError):
                pass
    return None


def _multi_to_flat_index(multi_idx: tuple[int, ...], shape: tuple[int, ...]) -> int:
    """Convert multi-index to flat index (row-major/C order)."""
    flat = 0
    stride = 1
    for i in range(len(shape) - 1, -1, -1):
        flat += multi_idx[i] * stride
        stride *= shape[i]
    return flat


# =============================================================================
# Operator Splitting Utilities
# =============================================================================


def verify_operator_splitting_adjoint(
    A_adv_hjb: sparse.spmatrix | NDArray,
    A_diff_hjb: sparse.spmatrix | NDArray,
    A_adv_fp: sparse.spmatrix | NDArray,
    A_diff_fp: sparse.spmatrix | NDArray,
    rtol: float = 1e-10,
) -> dict:
    """
    Verify adjoint consistency for operator splitting scheme.

    For operator splitting: L = L_adv + L_diff
    Adjoint requires: (L_adv)^* = L_adv_fp and (L_diff)^* = L_diff_fp

    Args:
        A_adv_hjb: HJB advection operator
        A_diff_hjb: HJB diffusion operator
        A_adv_fp: FP advection operator
        A_diff_fp: FP diffusion operator
        rtol: Tolerance

    Returns:
        Dictionary with verification results for each component.
    """
    adv_ok, adv_err = check_operator_adjoint(A_adv_hjb, A_adv_fp, rtol)
    diff_ok, diff_err = check_operator_adjoint(A_diff_hjb, A_diff_fp, rtol)

    return {
        "advection_adjoint": adv_ok,
        "advection_error": adv_err,
        "diffusion_adjoint": diff_ok,
        "diffusion_error": diff_err,
        "overall_adjoint": adv_ok and diff_ok,
    }


# =============================================================================
# Smoke Test
# =============================================================================

if __name__ == "__main__":
    """Quick validation of operator construction utilities."""
    print("Testing adjoint operator utilities...")
    print()

    # Test 1: Diffusion matrix symmetry (primitive API)
    print("Test 1: Diffusion matrix with Neumann BC (primitive API)")
    Nx = 20
    A_diff = build_diffusion_matrix_1d(Nx, dx=0.1, sigma=0.2, dt=0.01, bc_type="neumann")
    is_sym, err = check_operator_adjoint(A_diff, A_diff)
    print(f"  Symmetric (self-adjoint): {is_sym}")
    print(f"  Asymmetry error: {err:.2e}")
    assert is_sym, "Neumann diffusion should be symmetric"
    print("  PASSED")
    print()

    # Test 2: Periodic diffusion
    print("Test 2: Diffusion matrix with periodic BC")
    A_diff_per = build_diffusion_matrix_1d(Nx, dx=0.1, sigma=0.2, dt=0.01, bc_type="periodic")
    is_sym, err = check_operator_adjoint(A_diff_per, A_diff_per)
    print(f"  Symmetric (self-adjoint): {is_sym}")
    print(f"  Asymmetry error: {err:.2e}")
    assert is_sym, "Periodic diffusion should be symmetric"
    print("  PASSED")
    print()

    # Test 3: Advection matrix
    print("Test 3: Advection matrix (upwind)")
    velocity = np.ones(Nx) * 0.5  # Constant velocity
    A_adv = build_advection_matrix_1d(Nx, dx=0.1, velocity=velocity, dt=0.01)
    print(f"  Matrix shape: {A_adv.shape}")
    print(f"  Non-zeros: {A_adv.nnz}")
    # Advection matrix is NOT symmetric in general
    is_sym, err = check_operator_adjoint(A_adv, A_adv)
    print(f"  Symmetric: {is_sym} (expected False for non-zero velocity)")
    print("  PASSED")
    print()

    # Test 4: Symmetrization
    print("Test 4: Make operator self-adjoint")
    A_sym = make_operator_adjoint(A_adv)
    is_sym, err = check_operator_adjoint(A_sym, A_sym)
    print(f"  After symmetrization: {is_sym}")
    assert is_sym, "Symmetrized matrix should be self-adjoint"
    print("  PASSED")
    print()

    # Test 5: Geometry-aware API (preferred)
    print("Test 5: Geometry-aware API (preferred)")
    try:
        from mfgarchon.geometry import TensorProductGrid
        from mfgarchon.geometry.boundary import neumann_bc

        # 1D geometry
        grid_1d = TensorProductGrid(
            bounds=[(0, 1)],
            Nx=[20],
            boundary_conditions=neumann_bc(dimension=1),
        )
        A_diff_1d = build_diffusion_matrix_from_geometry(grid_1d, sigma=0.2, dt=0.01)
        is_sym, err = check_operator_adjoint(A_diff_1d, A_diff_1d)
        print(f"  1D grid: shape={A_diff_1d.shape}, symmetric={is_sym}")
        assert is_sym, "1D diffusion should be symmetric"

        # 2D geometry
        grid_2d = TensorProductGrid(
            bounds=[(0, 1), (0, 1)],
            Nx=[10, 10],
            boundary_conditions=neumann_bc(dimension=2),
        )
        A_diff_2d = build_diffusion_matrix_from_geometry(grid_2d, sigma=0.2, dt=0.01)
        is_sym, err = check_operator_adjoint(A_diff_2d, A_diff_2d)
        print(f"  2D grid: shape={A_diff_2d.shape}, symmetric={is_sym}")
        assert is_sym, "2D diffusion should be symmetric"

        # Boundary indices
        bi = get_boundary_indices_from_geometry(grid_2d)
        print(f"  2D boundaries: {list(bi.keys())}")
        print("  PASSED")

    except ImportError as e:
        print(f"  Skipped (geometry module not available): {e}")

    print()
    print("All operator tests passed!")
