"""
Internal: Tensor Calculus Operators for Regular Grids.

Function-based differential operators (gradient, laplacian, divergence, etc.)
for structured grids. Used internally by geometry/operators/wrappers.py.

Public API: Use ``mfgarchon.operators`` (LinearOperator classes) instead.

References:
-----------
- docs/development/operator_architecture.md
- LeVeque (2007): Finite Difference Methods for ODEs and PDEs
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Literal

import numpy as np

# =============================================================================
# Numba JIT Support
# =============================================================================

try:
    from numba import njit

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

    def njit(*args, **kwargs):
        """Dummy decorator when Numba not available."""

        def decorator(func):
            return func

        if len(args) == 1 and callable(args[0]):
            return args[0]
        return decorator


USE_NUMBA = os.environ.get("MFG_USE_NUMBA", "auto")
if USE_NUMBA == "auto":
    USE_NUMBA = NUMBA_AVAILABLE
elif USE_NUMBA.lower() in ("true", "1", "yes"):
    USE_NUMBA = True
else:
    USE_NUMBA = False

# =============================================================================
# Note: This module is internal infrastructure. Not re-exported from
# utils.numerical.__init__. Public API: mfgarchon.operators.
# =============================================================================

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from mfgarchon.backends.backend_manager import ArrayBackend
    from mfgarchon.geometry.boundary.conditions import BoundaryConditions


# =============================================================================
# Type Aliases
# =============================================================================

GradientScheme = Literal["central", "upwind", "one_sided"]
AdvectionScheme = Literal["gradient", "divergence"]
AdvectionMethod = Literal["centered", "upwind"]


def _roll(xp: type, u: NDArray, shift: int, axis: int) -> NDArray:
    """Backend-aware periodic shift.

    ``numpy``/``cupy`` use the ``axis=`` keyword; ``torch.roll`` uses ``dims=``.
    Detecting the module by name keeps the numpy/cupy path byte-identical
    while routing torch tensors to the correct keyword (Issue #1194).
    """
    if xp.__name__ == "torch":
        return xp.roll(u, shift, dims=axis)
    return xp.roll(u, shift, axis=axis)


# =============================================================================
# First-Order Operators: Gradient
# =============================================================================


def gradient(
    u: NDArray,
    spacings: list[float] | tuple[float, ...],
    scheme: GradientScheme = "central",
    bc: BoundaryConditions | None = None,
    backend: ArrayBackend | None = None,
    time: float = 0.0,
) -> list[NDArray]:
    """
    Compute spatial gradient ∇u on regular grid.

    Parameters
    ----------
    u : NDArray
        Scalar field, shape (N0, N1, ..., Nd-1)
    spacings : list[float]
        Grid spacing per dimension [h0, h1, ..., hd-1]
    scheme : {"central", "upwind", "one_sided"}
        Difference scheme:
        - "central": Second-order central differences (default)
        - "upwind": Godunov upwind (monotone, first-order)
        - "one_sided": Forward at left, backward at right
    bc : BoundaryConditions, optional
        Boundary conditions. If None, uses periodic (np.roll).
    backend : ArrayBackend, optional
        GPU backend. Uses numpy if None.
    time : float
        Current time for time-dependent BCs.

    Returns
    -------
    list[NDArray]
        Gradient components [∂u/∂x0, ∂u/∂x1, ...], each with shape of u.

    Examples
    --------
    >>> u = np.sin(np.linspace(0, 2*np.pi, 100))
    >>> du_dx = gradient(u, spacings=[0.1])[0]

    >>> # 2D with BC
    >>> from mfgarchon.geometry.boundary import neumann_bc
    >>> u_2d = np.random.rand(32, 32)
    >>> grad = gradient(u_2d, [0.1, 0.1], bc=neumann_bc(dimension=2))
    """
    xp = backend.array_module if backend is not None else np
    dimension = u.ndim

    # Apply ghost cells if BC provided
    if bc is not None:
        u_work = _apply_ghost_cells_nd(u, bc, time)
    else:
        u_work = u

    gradients = []
    for d in range(dimension):
        h = spacings[d]
        if h < 1e-14:
            gradients.append(xp.zeros_like(u))
            continue

        # Compute gradient with selected scheme
        if scheme == "central":
            grad_d = _gradient_central(u_work, d, h, xp)
        elif scheme == "upwind":
            grad_d = _gradient_upwind(u_work, d, h, xp)
        elif scheme == "one_sided":
            grad_d = _gradient_central(u_work, d, h, xp)
            grad_d = _fix_boundaries_one_sided(grad_d, u_work, d, h, xp)
        else:
            raise ValueError(f"Unknown gradient scheme: {scheme}")

        # Extract interior if ghost cells were added
        if bc is not None:
            grad_d = _extract_interior(grad_d, dimension)

        gradients.append(grad_d)

    return gradients


def gradient_simple(
    u: NDArray,
    spacings: list[float] | tuple[float, ...],
    backend: ArrayBackend | None = None,
) -> list[NDArray]:
    """
    Simple central difference gradient without BC handling.

    Fast path for cases where BC is handled separately
    (e.g., particle methods).

    Parameters
    ----------
    u : NDArray
        Scalar field on grid.
    spacings : list[float]
        Grid spacing per dimension.
    backend : ArrayBackend, optional
        GPU backend.

    Returns
    -------
    list[NDArray]
        Gradient components per dimension.
    """
    xp = backend.array_module if backend is not None else np
    is_torch_backend = backend is not None and backend.__class__.__name__ == "TorchBackend"

    gradients = []
    for d in range(u.ndim):
        h = spacings[d]
        if h > 1e-14:
            # PyTorch uses 'dims', NumPy uses 'axis'
            if is_torch_backend:
                grad_d = (xp.roll(u, -1, dims=d) - xp.roll(u, 1, dims=d)) / (2 * h)
            else:
                grad_d = (xp.roll(u, -1, axis=d) - xp.roll(u, 1, axis=d)) / (2 * h)
        else:
            grad_d = xp.zeros_like(u)
        gradients.append(grad_d)

    return gradients


# =============================================================================
# First-Order Operators: Divergence
# =============================================================================


def divergence(
    F: list[NDArray] | tuple[NDArray, ...],
    spacings: list[float] | tuple[float, ...],
    bc: BoundaryConditions | None = None,
    backend: ArrayBackend | None = None,
    time: float = 0.0,
) -> NDArray:
    """
    Compute divergence ∇·F of a vector field.

    Parameters
    ----------
    F : list[NDArray]
        Vector field components [F0, F1, ...], each with shape (N0, N1, ...).
    spacings : list[float]
        Grid spacing per dimension.
    bc : BoundaryConditions, optional
        Boundary conditions.
    backend : ArrayBackend, optional
        GPU backend.
    time : float
        Current time for time-dependent BCs.

    Returns
    -------
    NDArray
        Divergence scalar field, ∇·F = ∂F0/∂x0 + ∂F1/∂x1 + ...

    Examples
    --------
    >>> Fx = np.ones((32, 32))  # Constant x-velocity
    >>> Fy = np.zeros((32, 32))  # Zero y-velocity
    >>> div_F = divergence([Fx, Fy], spacings=[0.1, 0.1])
    """
    xp = backend.array_module if backend is not None else np
    dimension = len(F)

    div = xp.zeros_like(F[0])

    for d in range(dimension):
        h = spacings[d]
        if h < 1e-14:
            continue

        F_d = F[d]

        # Apply ghost cells if BC provided
        if bc is not None:
            F_d_work = _apply_ghost_cells_nd(F_d, bc, time)
        else:
            F_d_work = F_d

        # Central difference for divergence: ∂F_d/∂x_d
        dF_d = (_roll(xp, F_d_work, -1, d) - _roll(xp, F_d_work, 1, d)) / (2 * h)

        # Extract interior if ghost cells were added
        if bc is not None:
            dF_d = _extract_interior(dF_d, dimension)

        div += dF_d

    return div


# =============================================================================
# Second-Order Operators: Laplacian
# =============================================================================


def laplacian(
    u: NDArray,
    spacings: list[float] | tuple[float, ...],
    bc: BoundaryConditions | None = None,
    backend: ArrayBackend | None = None,
    time: float = 0.0,
) -> NDArray:
    """
    Compute Laplacian Δu = ∇·∇u on regular grid.

    Uses standard 3-point stencil per dimension:
        ∂²u/∂x² ≈ (u[i+1] - 2u[i] + u[i-1]) / h²

    Parameters
    ----------
    u : NDArray
        Scalar field on grid.
    spacings : list[float]
        Grid spacing per dimension.
    bc : BoundaryConditions, optional
        Boundary conditions.
    backend : ArrayBackend, optional
        GPU backend.
    time : float
        Current time for time-dependent BCs.

    Returns
    -------
    NDArray
        Laplacian with same shape as u.

    Examples
    --------
    >>> # u = x² + y², Δu = 4
    >>> u = X**2 + Y**2
    >>> lap = laplacian(u, spacings=[dx, dy])
    """
    xp = backend.array_module if backend is not None else np

    # Apply ghost cells if BC provided
    if bc is not None:
        u_work = _apply_ghost_cells_nd(u, bc, time)
    else:
        u_work = u

    lap = xp.zeros_like(u_work)

    for d in range(u.ndim):
        h = spacings[d]
        if h > 1e-14:
            lap += (_roll(xp, u_work, -1, d) - 2 * u_work + _roll(xp, u_work, 1, d)) / (h * h)

    # Extract interior if ghost cells were added
    if bc is not None:
        lap = _extract_interior(lap, u.ndim)

    return lap


# =============================================================================
# Second-Order Operators: Hessian
# =============================================================================


def hessian(
    u: NDArray,
    spacings: list[float] | tuple[float, ...],
    bc: BoundaryConditions | None = None,
    backend: ArrayBackend | None = None,
    time: float = 0.0,
) -> NDArray:
    """
    Compute Hessian tensor ∇²u = [∂²u/∂xᵢ∂xⱼ].

    Parameters
    ----------
    u : NDArray
        Scalar field, shape (N0, N1, ..., Nd-1).
    spacings : list[float]
        Grid spacing per dimension.
    bc : BoundaryConditions, optional
        Boundary conditions.
    backend : ArrayBackend, optional
        GPU backend.
    time : float
        Current time for time-dependent BCs.

    Returns
    -------
    NDArray
        Hessian tensor, shape (*u.shape, d, d).

    Examples
    --------
    >>> u = X**2 + Y**2  # Shape (Nx, Ny)
    >>> H = hessian(u, [dx, dy])  # Shape (Nx, Ny, 2, 2)
    >>> # H[..., 0, 0] = ∂²u/∂x² = 2
    >>> # H[..., 1, 1] = ∂²u/∂y² = 2
    >>> # H[..., 0, 1] = ∂²u/∂x∂y = 0
    """
    xp = backend.array_module if backend is not None else np
    d = u.ndim

    # Apply ghost cells if BC provided
    if bc is not None:
        u_work = _apply_ghost_cells_nd(u, bc, time)
    else:
        u_work = u

    # Initialize Hessian tensor
    hess_shape = (*u.shape, d, d)
    H = xp.zeros(hess_shape, dtype=u.dtype)

    for i in range(d):
        hi = spacings[i]
        if hi < 1e-14:
            continue

        for j in range(d):
            hj = spacings[j]
            if hj < 1e-14:
                continue

            if i == j:
                # Diagonal: ∂²u/∂xᵢ²
                d2u = (_roll(xp, u_work, -1, i) - 2 * u_work + _roll(xp, u_work, 1, i)) / (hi * hi)
            else:
                # Off-diagonal: ∂²u/∂xᵢ∂xⱼ (mixed derivative)
                # Use central differences in both directions
                d2u = (
                    _roll(xp, _roll(xp, u_work, -1, i), -1, j)
                    - _roll(xp, _roll(xp, u_work, -1, i), 1, j)
                    - _roll(xp, _roll(xp, u_work, 1, i), -1, j)
                    + _roll(xp, _roll(xp, u_work, 1, i), 1, j)
                ) / (4 * hi * hj)

            # Extract interior if ghost cells were added
            if bc is not None:
                d2u = _extract_interior(d2u, d)

            H[..., i, j] = d2u

    return H


# =============================================================================
# Coefficient Operators: Diffusion (Unified)
# =============================================================================


def diffusion(
    u: NDArray,
    coeff: float | NDArray,
    spacings: list[float] | tuple[float, ...],
    bc: BoundaryConditions | None = None,
    backend: ArrayBackend | None = None,
    domain_bounds: NDArray | None = None,
    time: float = 0.0,
) -> NDArray:
    """
    Compute diffusion ∇·(Σ∇u) with automatic dispatch based on coefficient type.

    This is the unified diffusion operator that handles both isotropic and
    anisotropic cases. The coefficient type determines the computation:

        - scalar σ      → isotropic:  σ²Δu = ∇·(σ²I∇u)
        - (d,d) matrix  → anisotropic: ∇·(Σ∇u) with constant tensor
        - (*shape,d,d)  → spatially varying anisotropic diffusion

    Parameters
    ----------
    u : NDArray
        Scalar field, shape (N0, N1, ..., Nd-1).
    coeff : float | NDArray
        Diffusion coefficient:
        - scalar σ: Treated as σ² for isotropic diffusion (Σ = σ²I)
        - (d, d) array: Constant diffusion tensor Σ
        - (*u.shape, d, d) array: Spatially varying tensor Σ(x)
    spacings : list[float]
        Grid spacing per dimension [h0, h1, ..., hd-1].
    bc : BoundaryConditions, optional
        Boundary conditions.
    backend : ArrayBackend, optional
        GPU backend. Uses numpy if None.
    domain_bounds : NDArray, optional
        Domain bounds for mixed BCs.
    time : float
        Current time for time-dependent BCs.

    Returns
    -------
    NDArray
        Diffusion term ∇·(Σ∇u), same shape as u.

    Notes
    -----
    For isotropic diffusion with scalar σ, the fast path uses the Laplacian:
        σ²Δu = σ²(∂²u/∂x² + ∂²u/∂y² + ...)

    For anisotropic diffusion with tensor Σ, the full operator is computed:
        ∇·(Σ∇u) = Σᵢ ∂/∂xᵢ (Σⱼ Σᵢⱼ ∂u/∂xⱼ)

    Uses Numba JIT for 2D anisotropic case when available.

    Examples
    --------
    >>> # Isotropic diffusion: σ = 0.1
    >>> diff = diffusion(u, 0.1, [dx, dy])
    >>>
    >>> # Anisotropic: stronger in x than y
    >>> Sigma = np.array([[0.2, 0.0], [0.0, 0.05]])
    >>> diff = diffusion(u, Sigma, [dx, dy], bc=bc)
    >>>
    >>> # Spatially varying
    >>> Sigma_field = np.zeros((*u.shape, 2, 2))
    >>> Sigma_field[..., 0, 0] = sigma_x  # σ_xx(x,y)
    >>> Sigma_field[..., 1, 1] = sigma_y  # σ_yy(x,y)
    >>> diff = diffusion(u, Sigma_field, [dx, dy])
    """
    d = u.ndim

    # Dispatch based on coefficient type
    if np.isscalar(coeff):
        # Isotropic: σ²Δu (fast path)
        lap = laplacian(u, spacings, bc=bc, backend=backend, time=time)
        return float(coeff) ** 2 * lap

    coeff = np.asarray(coeff)

    if coeff.ndim == 0:
        # 0-d array (scalar wrapped in array)
        lap = laplacian(u, spacings, bc=bc, backend=backend, time=time)
        return float(coeff) ** 2 * lap

    if coeff.ndim == 1 and len(coeff) == d:
        # Diagonal tensor: [σ_0², σ_1², ...] → Σ = diag(σ²)
        # Convert to full tensor for uniform handling
        Sigma = np.diag(coeff)
        return _diffusion_tensor(u, Sigma, spacings, bc, domain_bounds, time)

    if coeff.ndim == 2 and coeff.shape == (d, d):
        # Constant tensor Σ
        return _diffusion_tensor(u, coeff, spacings, bc, domain_bounds, time)

    if coeff.shape == (*u.shape, d, d):
        # Spatially varying tensor Σ(x)
        return _diffusion_tensor(u, coeff, spacings, bc, domain_bounds, time)

    raise ValueError(
        f"Invalid coefficient shape {coeff.shape} for {d}D field. "
        f"Expected scalar, ({d},), ({d},{d}), or {(*u.shape, d, d)}."
    )


def _diffusion_tensor(
    u: NDArray,
    Sigma: NDArray,
    spacings: list[float] | tuple[float, ...],
    bc: BoundaryConditions | None,
    domain_bounds: NDArray | None,
    time: float,
) -> NDArray:
    """Internal: dispatch tensor diffusion by dimension."""
    d = u.ndim

    if d == 1:
        return _tensor_diffusion_1d(u, Sigma, spacings[0], bc)
    elif d == 2:
        return _tensor_diffusion_2d(u, Sigma, spacings[0], spacings[1], bc, domain_bounds, time)
    else:
        return _tensor_diffusion_nd(u, Sigma, tuple(spacings), bc)


def tensor_diffusion(
    u: NDArray,
    Sigma: NDArray,
    spacings: list[float] | tuple[float, ...],
    bc: BoundaryConditions | None = None,
    domain_bounds: NDArray | None = None,
    time: float = 0.0,
) -> NDArray:
    """
    Compute anisotropic diffusion ∇·(Σ∇u).

    .. deprecated:: 0.18.0
        Use ``diffusion(u, Sigma, ...)`` instead. This function is kept
        for backward compatibility and will be removed in v1.0.

    See Also
    --------
    diffusion : Unified diffusion operator (recommended).
    """
    import warnings

    warnings.warn(
        "tensor_diffusion is deprecated, use diffusion(u, Sigma, ...) instead. Will be removed in v1.0.0.",
        DeprecationWarning,
        stacklevel=2,
    )
    return diffusion(u, Sigma, spacings, bc=bc, domain_bounds=domain_bounds, time=time)


# =============================================================================
# Transport Operators: Advection
# =============================================================================


def advection(
    m: NDArray,
    v: list[NDArray] | tuple[NDArray, ...],
    spacings: list[float] | tuple[float, ...],
    form: AdvectionScheme = "gradient",
    method: AdvectionMethod = "upwind",
    bc: BoundaryConditions | None = None,
    backend: ArrayBackend | None = None,
    time: float = 0.0,
) -> NDArray:
    """
    Compute advection term for transport equation.

    Two mathematical forms:
        - "gradient": v·∇m (non-conservative, used in HJB)
        - "divergence": ∇·(vm) (conservative, used in FP)

    Parameters
    ----------
    m : NDArray
        Scalar field being advected.
    v : list[NDArray]
        Velocity components [v0, v1, ...].
    spacings : list[float]
        Grid spacing per dimension.
    form : {"gradient", "divergence"}
        Mathematical form of advection term.
    method : {"centered", "upwind"}
        Spatial discretization scheme.
    bc : BoundaryConditions, optional
        Boundary conditions.
    backend : ArrayBackend, optional
        GPU backend.
    time : float
        Current time for time-dependent BCs.

    Returns
    -------
    NDArray
        Advection term, same shape as m.

    Notes
    -----
    For incompressible flow (∇·v = 0), both forms are equivalent:
        v·∇m = ∇·(vm) - m(∇·v) = ∇·(vm)

    Examples
    --------
    >>> # FP advection: dm/dt + ∇·(vm) = ...
    >>> adv = advection(m, [vx, vy], [dx, dy], form="divergence")
    """
    xp = backend.array_module if backend is not None else np
    dimension = len(v)

    if form == "gradient":
        # v·∇m
        if method == "upwind":
            grad_m = gradient(m, spacings, scheme="upwind", bc=bc, backend=backend, time=time)
        else:
            grad_m = gradient(m, spacings, scheme="central", bc=bc, backend=backend, time=time)

        result = xp.zeros_like(m)
        for d in range(dimension):
            result += v[d] * grad_m[d]
        return result

    elif form == "divergence":
        # ∇·(vm)
        flux = [v[d] * m for d in range(dimension)]

        if method == "upwind":
            # Upwind flux: use donor cell
            return _divergence_upwind(flux, v, spacings, bc, backend, time)
        else:
            return divergence(flux, spacings, bc=bc, backend=backend, time=time)

    else:
        raise ValueError(f"Unknown advection form: {form}")


# =============================================================================
# Private Helper Functions
# =============================================================================


def _gradient_central(u: NDArray, axis: int, h: float, xp: type) -> NDArray:
    """Central difference: (u[i+1] - u[i-1]) / (2h)."""
    return (_roll(xp, u, -1, axis) - _roll(xp, u, 1, axis)) / (2 * h)


def _gradient_forward(u: NDArray, axis: int, h: float, xp: type) -> NDArray:
    """Forward difference: (u[i+1] - u[i]) / h."""
    return (_roll(xp, u, -1, axis) - u) / h


def _gradient_backward(u: NDArray, axis: int, h: float, xp: type) -> NDArray:
    """Backward difference: (u[i] - u[i-1]) / h."""
    return (u - _roll(xp, u, 1, axis)) / h


def _gradient_upwind(u: NDArray, axis: int, h: float, xp: type) -> NDArray:
    """Godunov upwind: select based on sign of central gradient."""
    grad_forward = _gradient_forward(u, axis, h, xp)
    grad_backward = _gradient_backward(u, axis, h, xp)
    grad_central = (grad_forward + grad_backward) / 2.0
    return xp.where(grad_central >= 0, grad_backward, grad_forward)


def _fix_boundaries_one_sided(grad: NDArray, u: NDArray, axis: int, h: float, xp: type) -> NDArray:
    """Replace boundary values with one-sided differences."""
    ndim = u.ndim

    # Left boundary: forward difference
    left_slice = [slice(None)] * ndim
    left_slice[axis] = 0
    next_slice = [slice(None)] * ndim
    next_slice[axis] = 1
    grad[tuple(left_slice)] = (u[tuple(next_slice)] - u[tuple(left_slice)]) / h

    # Right boundary: backward difference
    right_slice = [slice(None)] * ndim
    right_slice[axis] = -1
    prev_slice = [slice(None)] * ndim
    prev_slice[axis] = -2
    grad[tuple(right_slice)] = (u[tuple(right_slice)] - u[tuple(prev_slice)]) / h

    return grad


def _apply_ghost_cells_nd(u: NDArray, bc: BoundaryConditions, time: float) -> NDArray:
    """Apply ghost cells using boundary conditions."""
    from mfgarchon.geometry.boundary import pad_array_with_ghosts

    return pad_array_with_ghosts(u, bc, ghost_depth=1, time=time)


def _extract_interior(u_padded: NDArray, dimension: int) -> NDArray:
    """Extract interior from padded array (remove ghost cells)."""
    slices = [slice(1, -1)] * dimension
    return u_padded[tuple(slices)]


def _divergence_upwind(
    flux: list[NDArray],
    v: list[NDArray],
    spacings: list[float] | tuple[float, ...],
    bc: BoundaryConditions | None,
    backend: ArrayBackend | None,
    time: float,
) -> NDArray:
    """Upwind divergence for advection."""
    xp = backend.array_module if backend is not None else np
    dimension = len(flux)

    div = xp.zeros_like(flux[0])

    for d in range(dimension):
        h = spacings[d]
        if h < 1e-14:
            continue

        F_d = flux[d]
        v_d = v[d]

        # Apply ghost cells if BC provided
        if bc is not None:
            F_d = _apply_ghost_cells_nd(F_d, bc, time)
            v_d_work = _apply_ghost_cells_nd(v_d, bc, time)
        else:
            v_d_work = v_d

        # Upwind flux at faces
        F_forward = _roll(xp, F_d, -1, d)
        F_backward = F_d

        # Face velocity (average)
        v_face = 0.5 * (v_d_work + _roll(xp, v_d_work, -1, d))

        # Select upwind flux
        F_face_right = xp.where(v_face >= 0, F_backward, F_forward)

        F_backward_left = _roll(xp, F_d, 1, d)
        v_face_left = 0.5 * (_roll(xp, v_d_work, 1, d) + v_d_work)
        F_face_left = xp.where(v_face_left >= 0, F_backward_left, F_d)

        # Divergence
        dF_d = (F_face_right - F_face_left) / h

        # Extract interior if ghost cells were added
        if bc is not None:
            dF_d = _extract_interior(dF_d, dimension)

        div += dF_d

    return div


# =============================================================================
# Tensor Diffusion: Numba JIT Kernels
# =============================================================================


@njit(cache=True)
def _compute_full_tensor_kernel_2d(
    m_padded: np.ndarray,
    Sigma: np.ndarray,
    dx: float,
    dy: float,
) -> np.ndarray:
    """JIT-compiled kernel for 2D full tensor diffusion."""
    Ny, Nx = Sigma.shape[0], Sigma.shape[1]
    result = np.zeros((Ny, Nx))

    for i in range(Ny):
        for j in range(Nx):
            s11 = Sigma[i, j, 0, 0]
            s12 = Sigma[i, j, 0, 1]
            s21 = Sigma[i, j, 1, 0]
            s22 = Sigma[i, j, 1, 1]

            # Face-averaged tensor components
            if j < Nx - 1:
                s11_xp = 0.5 * (s11 + Sigma[i, j + 1, 0, 0])
                s12_xp = 0.5 * (s12 + Sigma[i, j + 1, 0, 1])
            else:
                s11_xp, s12_xp = s11, s12

            if j > 0:
                s11_xm = 0.5 * (s11 + Sigma[i, j - 1, 0, 0])
                s12_xm = 0.5 * (s12 + Sigma[i, j - 1, 0, 1])
            else:
                s11_xm, s12_xm = s11, s12

            if i < Ny - 1:
                s21_yp = 0.5 * (s21 + Sigma[i + 1, j, 1, 0])
                s22_yp = 0.5 * (s22 + Sigma[i + 1, j, 1, 1])
            else:
                s21_yp, s22_yp = s21, s22

            if i > 0:
                s21_ym = 0.5 * (s21 + Sigma[i - 1, j, 1, 0])
                s22_ym = 0.5 * (s22 + Sigma[i - 1, j, 1, 1])
            else:
                s21_ym, s22_ym = s21, s22

            # Padded indices
            ip, jp = i + 1, j + 1

            # Gradients at faces
            dm_dx_xp = (m_padded[ip, jp + 1] - m_padded[ip, jp]) / dx
            dm_dy_xp = (
                0.25
                * (
                    (m_padded[ip + 1, jp + 1] - m_padded[ip - 1, jp + 1])
                    + (m_padded[ip + 1, jp] - m_padded[ip - 1, jp])
                )
                / dy
            )

            dm_dx_xm = (m_padded[ip, jp] - m_padded[ip, jp - 1]) / dx
            dm_dy_xm = (
                0.25
                * (
                    (m_padded[ip + 1, jp] - m_padded[ip - 1, jp])
                    + (m_padded[ip + 1, jp - 1] - m_padded[ip - 1, jp - 1])
                )
                / dy
            )

            dm_dy_yp = (m_padded[ip + 1, jp] - m_padded[ip, jp]) / dy
            dm_dx_yp = (
                0.25
                * (
                    (m_padded[ip + 1, jp + 1] - m_padded[ip + 1, jp - 1])
                    + (m_padded[ip, jp + 1] - m_padded[ip, jp - 1])
                )
                / dx
            )

            dm_dy_ym = (m_padded[ip, jp] - m_padded[ip - 1, jp]) / dy
            dm_dx_ym = (
                0.25
                * (
                    (m_padded[ip, jp + 1] - m_padded[ip, jp - 1])
                    + (m_padded[ip - 1, jp + 1] - m_padded[ip - 1, jp - 1])
                )
                / dx
            )

            # Fluxes
            Fx_xp = s11_xp * dm_dx_xp + s12_xp * dm_dy_xp
            Fx_xm = s11_xm * dm_dx_xm + s12_xm * dm_dy_xm
            Fy_yp = s21_yp * dm_dx_yp + s22_yp * dm_dy_yp
            Fy_ym = s21_ym * dm_dx_ym + s22_ym * dm_dy_ym

            # Divergence
            result[i, j] = (Fx_xp - Fx_xm) / dx + (Fy_yp - Fy_ym) / dy

    return result


@njit(cache=True)
def _compute_diagonal_kernel_2d(
    m_padded: np.ndarray,
    sigma_x: np.ndarray,
    sigma_y: np.ndarray,
    dx: float,
    dy: float,
) -> np.ndarray:
    """JIT-compiled kernel for 2D diagonal tensor diffusion."""
    Ny, Nx = sigma_x.shape
    result = np.zeros((Ny, Nx))

    for i in range(Ny):
        for j in range(Nx):
            ip, jp = i + 1, j + 1

            # x-direction
            if j < Nx - 1:
                sigma_x_xp = 0.5 * (sigma_x[i, j] + sigma_x[i, j + 1])
            else:
                sigma_x_xp = sigma_x[i, j]
            if j > 0:
                sigma_x_xm = 0.5 * (sigma_x[i, j] + sigma_x[i, j - 1])
            else:
                sigma_x_xm = sigma_x[i, j]

            dm_dx_xp = (m_padded[ip, jp + 1] - m_padded[ip, jp]) / dx
            dm_dx_xm = (m_padded[ip, jp] - m_padded[ip, jp - 1]) / dx
            div_x = (sigma_x_xp * dm_dx_xp - sigma_x_xm * dm_dx_xm) / dx

            # y-direction
            if i < Ny - 1:
                sigma_y_yp = 0.5 * (sigma_y[i, j] + sigma_y[i + 1, j])
            else:
                sigma_y_yp = sigma_y[i, j]
            if i > 0:
                sigma_y_ym = 0.5 * (sigma_y[i, j] + sigma_y[i - 1, j])
            else:
                sigma_y_ym = sigma_y[i, j]

            dm_dy_yp = (m_padded[ip + 1, jp] - m_padded[ip, jp]) / dy
            dm_dy_ym = (m_padded[ip, jp] - m_padded[ip - 1, jp]) / dy
            div_y = (sigma_y_yp * dm_dy_yp - sigma_y_ym * dm_dy_ym) / dy

            result[i, j] = div_x + div_y

    return result


# =============================================================================
# Tensor Diffusion: Dimension-specific implementations
# =============================================================================


def _tensor_diffusion_1d(
    u: NDArray,
    Sigma: NDArray | float,
    dx: float,
    bc: BoundaryConditions | None,
) -> NDArray:
    """1D tensor diffusion (reduces to scalar)."""
    if isinstance(Sigma, np.ndarray):
        if Sigma.ndim == 2:
            sigma_sq = Sigma[0, 0]
        elif Sigma.ndim == 1:
            sigma_sq = Sigma
        else:
            sigma_sq = Sigma[:, 0, 0]
    else:
        sigma_sq = Sigma

    # Apply BC using flux-conservative ghost cells
    # For tensor diffusion with no-flux BC, use mode='edge' for flux conservation.
    # Rationale: Same as 2D case - divergence-form diffusion needs FLUX-based ghost cells.
    if bc is not None:
        from mfgarchon.geometry.boundary import BCType

        # Check if this is a uniform no-flux/Neumann BC
        is_noflux = False
        try:
            bc_type_str = bc.type.lower()
            is_noflux = bc_type_str in ["no_flux", "neumann"]
        except AttributeError:
            # Unified BC without .type attribute - check segments
            if bc.is_uniform and len(bc.segments) > 0:
                seg = bc.segments[0]
                is_noflux = seg.bc_type in [BCType.NO_FLUX, BCType.NEUMANN]

        if is_noflux:
            # Use mode='edge' for flux conservation (zero flux at boundary)
            u_padded = np.pad(u, 1, mode="edge")
        else:
            # Use unified interface for other BC types
            u_padded = _apply_ghost_cells_nd(u, bc, 0.0)
    else:
        u_padded = np.pad(u, 1, mode="wrap")

    Nx = len(u)

    if np.isscalar(sigma_sq) or (isinstance(sigma_sq, np.ndarray) and sigma_sq.ndim == 0):
        lap = (u_padded[2:] - 2 * u_padded[1:-1] + u_padded[:-2]) / dx**2
        return float(sigma_sq) * lap
    else:
        dm_dx = (u_padded[1:] - u_padded[:-1]) / dx
        sigma_face = np.zeros(Nx + 1)
        sigma_face[1:-1] = 0.5 * (sigma_sq[1:] + sigma_sq[:-1])
        sigma_face[0] = sigma_sq[0]
        sigma_face[-1] = sigma_sq[-1]
        flux = sigma_face * dm_dx
        return (flux[1:] - flux[:-1]) / dx


def _tensor_diffusion_2d(
    u: NDArray,
    Sigma: NDArray,
    dx: float,
    dy: float,
    bc: BoundaryConditions | None,
    domain_bounds: NDArray | None,
    time: float,
) -> NDArray:
    """2D tensor diffusion with optional Numba JIT."""
    from mfgarchon.geometry.boundary import pad_array_with_ghosts

    Ny, Nx = u.shape

    # Expand constant tensor
    if Sigma.ndim == 2:
        Sigma_full = np.tile(Sigma, (Ny, Nx, 1, 1))
    else:
        Sigma_full = Sigma

    # Apply BC using unified interface (Issue #577)
    # For tensor diffusion with no-flux BC, use mode='edge' for flux conservation.
    # Rationale: pad_array_with_ghosts uses O(h²) reflection optimized for VALUE-based
    # operations (HJB, semi-Lagrangian). But divergence-form diffusion needs FLUX-based
    # ghost cells: zero flux at boundary requires ghost=boundary (mode='edge'), not
    # ghost=next_interior (reflection). See Issue #542 vs flux conservation.
    if bc is not None:
        from mfgarchon.geometry.boundary.conditions import BoundaryConditions

        # Check if this is a uniform no-flux/Neumann BC
        is_noflux = False
        if isinstance(bc, BoundaryConditions):
            # Mixed BC - use unified interface (may not be uniform no-flux)
            u_padded = pad_array_with_ghosts(u, bc, ghost_depth=1, time=time)
        else:
            # Legacy or unified uniform BC - check type
            try:
                bc_type_str = bc.type.lower()
                is_noflux = bc_type_str in ["no_flux", "neumann"]
            except AttributeError:
                # Unified BC without .type attribute - check segments
                if bc.is_uniform and len(bc.segments) > 0:
                    from mfgarchon.geometry.boundary import BCType

                    seg = bc.segments[0]
                    is_noflux = seg.bc_type in [BCType.NO_FLUX, BCType.NEUMANN]

            if is_noflux:
                # Use mode='edge' for flux conservation (zero flux at boundary)
                u_padded = np.pad(u, 1, mode="edge")
            else:
                # Use unified interface for other BC types
                u_padded = pad_array_with_ghosts(u, bc, ghost_depth=1, time=time)
    else:
        # Default to periodic if no BC specified
        u_padded = np.pad(u, 1, mode="wrap")

    # Use JIT kernel if available
    if USE_NUMBA and NUMBA_AVAILABLE:
        return _compute_full_tensor_kernel_2d(u_padded, Sigma_full, dx, dy)

    # Pure NumPy fallback (simplified)
    dm_dx_x = (u_padded[1:-1, 1:] - u_padded[1:-1, :-1]) / dx
    dm_dy_y = (u_padded[1:, 1:-1] - u_padded[:-1, 1:-1]) / dy
    dm_dx_y = 0.5 * ((u_padded[1:, 2:] - u_padded[1:, :-2]) + (u_padded[:-1, 2:] - u_padded[:-1, :-2])) / (2 * dx)
    dm_dy_x = 0.5 * ((u_padded[2:, 1:] - u_padded[:-2, 1:]) + (u_padded[2:, :-1] - u_padded[:-2, :-1])) / (2 * dy)

    # Face-averaged tensors
    Sigma_x_faces = np.zeros((Ny, Nx + 1, 2, 2))
    Sigma_x_faces[:, 1:-1, :, :] = 0.5 * (Sigma_full[:, 1:, :, :] + Sigma_full[:, :-1, :, :])
    Sigma_x_faces[:, 0, :, :] = Sigma_full[:, 0, :, :]
    Sigma_x_faces[:, -1, :, :] = Sigma_full[:, -1, :, :]

    Sigma_y_faces = np.zeros((Ny + 1, Nx, 2, 2))
    Sigma_y_faces[1:-1, :, :, :] = 0.5 * (Sigma_full[1:, :, :, :] + Sigma_full[:-1, :, :, :])
    Sigma_y_faces[0, :, :, :] = Sigma_full[0, :, :, :]
    Sigma_y_faces[-1, :, :, :] = Sigma_full[-1, :, :, :]

    Fx = Sigma_x_faces[:, :, 0, 0] * dm_dx_x + Sigma_x_faces[:, :, 0, 1] * dm_dy_x
    Fy = Sigma_y_faces[:, :, 1, 0] * dm_dx_y + Sigma_y_faces[:, :, 1, 1] * dm_dy_y

    return (Fx[:, 1:] - Fx[:, :-1]) / dx + (Fy[1:, :] - Fy[:-1, :]) / dy


def _tensor_diffusion_nd(
    u: NDArray,
    Sigma: NDArray,
    spacings: tuple[float, ...],
    bc: BoundaryConditions | None,
) -> NDArray:
    """General nD tensor diffusion."""
    d = u.ndim
    shape = u.shape

    # Expand constant tensor
    if Sigma.ndim == 2:
        Sigma_full = np.broadcast_to(Sigma, (*shape, d, d)).copy()
    else:
        Sigma_full = Sigma

    # Apply BC
    if bc is not None:
        bc_type = bc.type.lower()
        if bc_type == "periodic":
            u_padded = np.pad(u, 1, mode="wrap")
        elif bc_type in ["no_flux", "neumann"]:
            u_padded = np.pad(u, 1, mode="edge")
        else:
            u_padded = np.pad(u, 1, mode="constant", constant_values=0.0)
    else:
        u_padded = np.pad(u, 1, mode="wrap")

    result = np.zeros(shape, dtype=u.dtype)

    for i in range(d):
        flux_shape = list(shape)
        flux_shape[i] += 1
        F_i = np.zeros(flux_shape, dtype=u.dtype)

        for j in range(d):
            if i == j:
                slice_plus = [slice(1, -1)] * d
                slice_minus = [slice(1, -1)] * d
                slice_plus[i] = slice(1, None)
                slice_minus[i] = slice(None, -1)
                dm_dxj = (u_padded[tuple(slice_plus)] - u_padded[tuple(slice_minus)]) / spacings[j]
            else:
                slice_j_plus = [slice(1, -1)] * d
                slice_j_minus = [slice(1, -1)] * d
                slice_j_plus[j] = slice(2, None)
                slice_j_minus[j] = slice(None, -2)
                slice_j_plus[i] = slice(None)
                slice_j_minus[i] = slice(None)

                dm_dxj_ext = (u_padded[tuple(slice_j_plus)] - u_padded[tuple(slice_j_minus)]) / (2 * spacings[j])

                slice_k = [slice(None)] * d
                slice_k1 = [slice(None)] * d
                slice_k[i] = slice(None, -1)
                slice_k1[i] = slice(1, None)
                dm_dxj = 0.5 * (dm_dxj_ext[tuple(slice_k)] + dm_dxj_ext[tuple(slice_k1)])

            # Average Sigma_ij to faces
            Sigma_ij = Sigma_full[..., i, j]
            face_shape = list(shape)
            face_shape[i] += 1
            Sigma_ij_faces = np.zeros(face_shape, dtype=Sigma_full.dtype)

            slice_interior = [slice(None)] * d
            slice_interior[i] = slice(1, -1)
            slice_left = [slice(None)] * d
            slice_left[i] = slice(None, -1)
            slice_right = [slice(None)] * d
            slice_right[i] = slice(1, None)

            Sigma_ij_faces[tuple(slice_interior)] = 0.5 * (Sigma_ij[tuple(slice_left)] + Sigma_ij[tuple(slice_right)])

            slice_first = [slice(None)] * d
            slice_first[i] = 0
            slice_last = [slice(None)] * d
            slice_last[i] = -1

            Sigma_ij_faces[tuple(slice_first)] = Sigma_ij[tuple(slice_first)]
            Sigma_ij_faces[tuple(slice_last)] = Sigma_ij[tuple(slice_last)]

            F_i += Sigma_ij_faces * dm_dxj

        slice_plus_i = [slice(None)] * d
        slice_minus_i = [slice(None)] * d
        slice_plus_i[i] = slice(1, None)
        slice_minus_i[i] = slice(None, -1)

        result += (F_i[tuple(slice_plus_i)] - F_i[tuple(slice_minus_i)]) / spacings[i]

    return result


# =============================================================================
# Smoke Tests
# =============================================================================

if __name__ == "__main__":
    print("Testing tensor_calculus module...")

    # Test 1D gradient
    print("\n[1D Gradient] Testing...")
    x = np.linspace(0, 2 * np.pi, 100)
    u_1d = np.sin(x)
    dx = x[1] - x[0]
    grad_1d = gradient_simple(u_1d, [dx])
    expected = np.cos(x)
    error = np.max(np.abs(grad_1d[0][5:-5] - expected[5:-5]))
    print(f"  Error (interior): {error:.2e}")
    assert error < 0.01

    # Test 2D gradient and laplacian
    print("\n[2D Gradient/Laplacian] Testing...")
    nx, ny = 32, 32
    dx, dy = 0.1, 0.1
    x = np.linspace(0, (nx - 1) * dx, nx)
    y = np.linspace(0, (ny - 1) * dy, ny)
    X, Y = np.meshgrid(x, y, indexing="ij")
    u_2d = X**2 + Y**2

    grads = gradient_simple(u_2d, [dx, dy])
    error_gx = np.max(np.abs(grads[0][5:-5, 5:-5] - 2 * X[5:-5, 5:-5]))
    error_gy = np.max(np.abs(grads[1][5:-5, 5:-5] - 2 * Y[5:-5, 5:-5]))
    print(f"  Gradient error: x={error_gx:.2e}, y={error_gy:.2e}")
    assert error_gx < 1e-10, f"Gradient x error too large: {error_gx}"
    assert error_gy < 1e-10, f"Gradient y error too large: {error_gy}"

    lap = laplacian(u_2d, [dx, dy])
    error_lap = np.max(np.abs(lap[5:-5, 5:-5] - 4.0))
    print(f"  Laplacian error: {error_lap:.2e}")
    assert error_lap < 1e-10

    # Test divergence
    print("\n[Divergence] Testing...")
    Fx = X  # F = (x, y) -> div F = 2
    Fy = Y
    div_F = divergence([Fx, Fy], [dx, dy])
    error_div = np.max(np.abs(div_F[5:-5, 5:-5] - 2.0))
    print(f"  Divergence error: {error_div:.2e}")
    assert error_div < 1e-10

    # Test hessian
    print("\n[Hessian] Testing...")
    H = hessian(u_2d, [dx, dy])
    print(f"  Hessian shape: {H.shape}")
    error_hxx = np.max(np.abs(H[5:-5, 5:-5, 0, 0] - 2.0))
    error_hyy = np.max(np.abs(H[5:-5, 5:-5, 1, 1] - 2.0))
    error_hxy = np.max(np.abs(H[5:-5, 5:-5, 0, 1]))
    print(f"  Hessian errors: H_xx={error_hxx:.2e}, H_yy={error_hyy:.2e}, H_xy={error_hxy:.2e}")
    assert error_hxx < 1e-10, f"Hessian H_xx error too large: {error_hxx}"
    assert error_hyy < 1e-10, f"Hessian H_yy error too large: {error_hyy}"
    assert error_hxy < 1e-10, f"Hessian H_xy error too large: {error_hxy}"

    # Test diffusion (unified operator)
    print("\n[Diffusion] Testing unified operator...")
    from mfgarchon.geometry.boundary.conditions import periodic_bc

    bc = periodic_bc(dimension=2)
    m = np.exp(-((X - 1.0) ** 2 + (Y - 0.75) ** 2) / 0.1)

    # Isotropic: scalar coefficient
    diff_iso = diffusion(m, 0.1, [dx, dy], bc=bc)
    print(f"  Isotropic (σ=0.1): shape={diff_iso.shape}, range=[{diff_iso.min():.3e}, {diff_iso.max():.3e}]")
    assert diff_iso.shape == m.shape
    assert not np.any(np.isnan(diff_iso))

    # Anisotropic: (d,d) matrix coefficient
    Sigma = np.array([[0.1, 0.0], [0.0, 0.05]])
    diff_aniso = diffusion(m, Sigma, [dx, dy], bc=bc)
    print(f"  Anisotropic (Σ): shape={diff_aniso.shape}, range=[{diff_aniso.min():.3e}, {diff_aniso.max():.3e}]")
    assert diff_aniso.shape == m.shape
    assert not np.any(np.isnan(diff_aniso))

    # Diagonal shorthand: [σ_x², σ_y²]
    diff_diag = diffusion(m, np.array([0.1, 0.05]), [dx, dy], bc=bc)
    print(f"  Diagonal [σx², σy²]: shape={diff_diag.shape}")
    assert diff_diag.shape == m.shape
    # Should match anisotropic with diagonal Σ
    error_diag = np.max(np.abs(diff_diag - diff_aniso))
    print(f"  Diagonal vs Anisotropic error: {error_diag:.2e}")
    assert error_diag < 1e-10, "Diagonal and anisotropic should match"

    # Test advection
    print("\n[Advection] Testing...")
    vx = np.ones_like(m)
    vy = np.zeros_like(m)
    adv = advection(m, [vx, vy], [dx, dy], form="gradient", method="upwind")
    print(f"  Advection shape: {adv.shape}")
    assert adv.shape == m.shape

    print("\nAll tests passed!")
