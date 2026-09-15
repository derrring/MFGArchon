"""
Finite Difference Stencils for MFGarchon.

This module provides low-level finite difference stencil implementations.
Stencils are the building blocks for differential operators - they define
the coefficient weights for approximating derivatives.

Conceptual Hierarchy:
    Stencils (this module)
        ↓ (fixed coefficients)
    Reconstruction (operators/reconstruction/)
        ↓ (adaptive weighting)
    Differential Operators (operators/differential/)
        ↓ (LinearOperator interface)
    Solvers (alg/numerical/)

Stencil Types:
    - CENTRAL: 2nd-order accurate, symmetric, no directional bias
    - FORWARD: 1st-order accurate, uses u[i+1] - u[i]
    - BACKWARD: 1st-order accurate, uses u[i] - u[i-1]
    - UPWIND: the HJB upwind momentum of a numerical Hamiltonian (``upwind_momentum``, #2313)
    - ONE_SIDED: 2nd-order one-sided boundary handling (3-point stencils at edges)

Mathematical Background:
    Central:   ∂u/∂x ≈ (u[i+1] - u[i-1]) / (2h)     Error: O(h²)
    Forward:   ∂u/∂x ≈ (u[i+1] - u[i]) / h          Error: O(h)
    Backward:  ∂u/∂x ≈ (u[i] - u[i-1]) / h          Error: O(h)

Usage:
    >>> from mfgarchon.operators.stencils import gradient_central, gradient_upwind
    >>> du_dx = gradient_central(u, axis=0, h=0.1)

Note:
    These functions operate on arrays directly using np.roll for periodic wrapping.
    For boundary-aware operators, use the LinearOperator classes in operators/differential/.

Created: 2026-01-24 (Operator module reorganization)
Extracted from: mfgarchon/utils/numerical/tensor_calculus.py
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from mfgarchon.geometry.boundary import BoundaryConditions


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
# First-Order Derivative Stencils
# =============================================================================


def gradient_central(u: NDArray, axis: int, h: float, xp: type = np) -> NDArray:
    """
    Central difference approximation for first derivative.

    Formula: ∂u/∂x ≈ (u[i+1] - u[i-1]) / (2h)

    Properties:
        - 2nd-order accurate: O(h²)
        - Symmetric stencil: [-1, 0, 1] / (2h)
        - No directional bias (good for elliptic problems)
        - May oscillate near discontinuities

    Args:
        u: Input array
        axis: Axis along which to differentiate
        h: Grid spacing
        xp: Array module (numpy or cupy for GPU)

    Returns:
        Approximation of ∂u/∂x with same shape as u

    Note:
        Uses periodic wrapping via np.roll. For non-periodic boundaries,
        use fix_boundaries_one_sided() or the LinearOperator classes.
    """
    return (_roll(xp, u, -1, axis) - _roll(xp, u, 1, axis)) / (2 * h)


def gradient_forward(u: NDArray, axis: int, h: float, xp: type = np) -> NDArray:
    """
    Forward difference approximation for first derivative.

    Formula: ∂u/∂x ≈ (u[i+1] - u[i]) / h

    Properties:
        - 1st-order accurate: O(h)
        - Stencil: [-1, 1] / h
        - Biased in positive direction
        - Stable for advection with negative velocity (v < 0)

    Args:
        u: Input array
        axis: Axis along which to differentiate
        h: Grid spacing
        xp: Array module (numpy or cupy for GPU)

    Returns:
        Approximation of ∂u/∂x with same shape as u
    """
    return (_roll(xp, u, -1, axis) - u) / h


def gradient_backward(u: NDArray, axis: int, h: float, xp: type = np) -> NDArray:
    """
    Backward difference approximation for first derivative.

    Formula: ∂u/∂x ≈ (u[i] - u[i-1]) / h

    Properties:
        - 1st-order accurate: O(h)
        - Stencil: [1, -1] / h (shifted)
        - Biased in negative direction
        - Stable for advection with positive velocity (v > 0)

    Args:
        u: Input array
        axis: Axis along which to differentiate
        h: Grid spacing
        xp: Array module (numpy or cupy for GPU)

    Returns:
        Approximation of ∂u/∂x with same shape as u
    """
    return (u - _roll(xp, u, 1, axis)) / h


NumericalHamiltonian = Literal["engquist_osher", "rouy_tourin"]
"""The monotone numerical Hamiltonians the FDM HJB offers (#2313). Neither is called "Godunov": ACD's Example 1 uses the
word for the Engquist-Osher form, and the Hamilton-Jacobi literature uses it for the Rouy-Tourin one."""

DEFAULT_NUMERICAL_HAMILTONIAN: NumericalHamiltonian = "engquist_osher"
"""User ruling on #2313 (2026-09-15): Engquist-Osher is the default, Rouy-Tourin a selectable preset."""


def upwind_momentum(
    backward: NDArray, forward: NDArray, numerical_hamiltonian: NumericalHamiltonian, xp: Any = np
) -> NDArray:
    """The signed per-axis momentum ``p`` at which a monotone numerical Hamiltonian evaluates ``H`` (#2313).

    With backward difference ``a``, forward difference ``b``, ``a+ = max(a, 0)`` and ``b- = min(b, 0)``, both presets
    are ``H`` evaluated at a magnitude built from ``(a+, b-)``, with the sign of the larger branch::

        rouy_tourin     p = a+ if a+ >= -b- else b-                       (infinity-norm)
        engquist_osher  p = +-sqrt(a+^2 + b-^2), + when a+ >= -b-         (2-norm; ACD's Example 1)

    They agree except where both branches are nonzero, which is a discrete local maximum along the axis. For ``H`` even
    in each momentum component and nondecreasing in its magnitude, the class the FDM HJB accepts (#2311), both are
    consistent and monotone, and ``H(p)`` does not depend on the sign. ``engquist_osher`` is C^1 in ``(a, b)`` for such
    an ``H`` built from ``p^2`` or ``|p_d|``, and its linearisation is the transpose of the FDM FP ``divergence_upwind``
    operator at maxima as well as elsewhere (#2313: 2.8e-14 against 37.9 for ``rouy_tourin`` on a 2-D fixture).

    Args:
        backward: ``(u_i - u_{i-1}) / h`` along one axis.
        forward: ``(u_{i+1} - u_i) / h`` along the same axis.
        numerical_hamiltonian: ``"engquist_osher"`` or ``"rouy_tourin"``.
        xp: Array module (numpy, cupy or torch).

    Returns:
        The momentum, same shape as the inputs.
    """
    backward_part = backward * (backward > 0)
    forward_part = forward * (forward < 0)
    backward_branch = backward_part >= -forward_part
    if numerical_hamiltonian == "rouy_tourin":
        return xp.where(backward_branch, backward_part, forward_part)
    if numerical_hamiltonian == "engquist_osher":
        magnitude = xp.sqrt(backward_part * backward_part + forward_part * forward_part)
        return xp.where(backward_branch, magnitude, -magnitude)
    raise ValueError(
        f"numerical_hamiltonian must be 'engquist_osher' or 'rouy_tourin', got {numerical_hamiltonian!r} (#2313)."
    )


def upwind_momentum_derivatives(
    backward: NDArray, forward: NDArray, numerical_hamiltonian: NumericalHamiltonian, xp: Any = np
) -> tuple[NDArray, NDArray]:
    """``(dp/da, dp/db)`` for `upwind_momentum`, so ``dH/da = H_p(p) dp/da`` and likewise for ``b`` (#2313).

    ``rouy_tourin``: 1 on the branch taken where that difference is strictly signed, else 0. ``engquist_osher``:
    ``s a+ / |p|`` and ``s b- / |p|`` with ``s`` the sign of ``p``, and 0 where ``p = 0``. These are the exact
    derivatives of the momentum the residual evaluates, so a Jacobian built from them is the residual's own.
    """
    backward_part = backward * (backward > 0)
    forward_part = forward * (forward < 0)
    backward_branch = backward_part >= -forward_part
    zero = xp.zeros_like(backward_part)
    one = xp.ones_like(backward_part)
    if numerical_hamiltonian == "rouy_tourin":
        d_backward = xp.where(backward_branch & (backward > 0), one, zero)
        d_forward = xp.where(~backward_branch & (forward < 0), one, zero)
        return d_backward, d_forward
    if numerical_hamiltonian == "engquist_osher":
        magnitude = xp.sqrt(backward_part * backward_part + forward_part * forward_part)
        safe = xp.where(magnitude > 0, magnitude, one)
        sign = xp.where(backward_branch, one, -one)
        d_backward = xp.where(magnitude > 0, sign * backward_part / safe, zero)
        d_forward = xp.where(magnitude > 0, sign * forward_part / safe, zero)
        return d_backward, d_forward
    raise ValueError(
        f"numerical_hamiltonian must be 'engquist_osher' or 'rouy_tourin', got {numerical_hamiltonian!r} (#2313)."
    )


def gradient_upwind(u: NDArray, axis: int, h: float, xp: type = np) -> NDArray:
    """
    Rouy-Tourin upwind momentum for a Hamilton-Jacobi Hamiltonian: `upwind_momentum` with ``"rouy_tourin"``.

    With backward difference ``a`` and forward difference ``b``::

        a+ = max(a, 0),  b- = min(b, 0),  p = a+ if a+ >= -b- else b-

    Applied per axis, ``H(p)`` is then the min/max numerical Hamiltonian -- per axis the minimum of
    ``H`` over ``[a, b]`` when ``a <= b`` and the maximum over ``[b, a]`` when ``a > b``, nested
    across axes -- exactly when, at fixed ``(x, m, t)``, ``H`` is even in each momentum component
    and nondecreasing in its magnitude. Every control cost ``SeparableHamiltonian`` and
    ``CongestionHamiltonian`` ship satisfies that (#2308). Neither half is enough alone: the double
    well ``(p^2 - 1)^2`` is even and fails, and so does an ``H`` minimised at ``p = 0`` over an
    asymmetric control bound (#2311).

    Where it departs from selecting on ``sign((a + b) / 2)`` (#2308): only at a discrete local
    minimum, ``a < 0 < b``. There the minimum of ``H`` over ``[a, b]`` is ``H(0)``, and this returns
    ``p = 0``; the sign rule returned the one-sided difference of smaller magnitude, a
    non-monotone numerical Hamiltonian that stayed silent because Newton still converges to its
    root. Everywhere else the two agree.

    Scope: this is a statement about the HJB momentum. Outside the condition above -- a
    ``DualHamiltonian`` whose Lagrangian is minimised away from ``0`` or bounded asymmetrically, or
    a ``CongestionHamiltonian`` with ``c(m) < 0`` -- it is not that min/max form, and neither was the rule
    it replaced; the two are wrong on different pairs there, so neither is the better one (#2311).
    Transport (``v . grad m``) upwinds by the sign of the velocity, not of any gradient, so this is
    not the upwind rule for advection either (#2309), nor for reinitialisation, which upwinds by the
    sign of the initial level set (#2310).

    Properties:
        - 1st-order accurate: O(h)
        - Returns ``0`` at a discrete local minimum, a one-sided difference elsewhere

    Args:
        u: Input array
        axis: Axis along which to differentiate
        h: Grid spacing
        xp: Array module (numpy, cupy or torch)

    Returns:
        Upwind momentum with the same shape as u
    """
    return upwind_momentum(gradient_backward(u, axis, h, xp), gradient_forward(u, axis, h, xp), "rouy_tourin", xp)


def gradient_upwind_by_velocity(u: NDArray, v: NDArray, axis: int, h: float, xp: Any = np) -> NDArray:
    """
    First derivative for the transport term ``v . grad u``, upwinded by the sign of the velocity.

    The backward difference where ``v >= 0``, the forward difference where ``v < 0``. The explicit
    update ``u - dt * v * D u`` is then monotone for ``|v| dt / h <= 1``: every coefficient is a
    nonnegative weight of ``u`` at the node and its upwind neighbour. Selecting on the sign of a
    gradient instead, as ``gradient_upwind`` does for the HJB momentum, is non-monotone for transport
    and blows up at constant velocity (#2309).

    Args:
        u: Transported field
        v: Velocity component along ``axis``, same shape as ``u``
        axis: Axis along which to differentiate
        h: Grid spacing
        xp: Array module (numpy, cupy or torch)

    Returns:
        Upwinded derivative with the same shape as ``u``
    """
    return xp.where(v >= 0, gradient_backward(u, axis, h, xp), gradient_forward(u, axis, h, xp))


def divergence_upwind_by_velocity(m: NDArray, v: NDArray, axis: int, h: float, xp: Any = np) -> NDArray:
    """
    Upwind divergence ``d(v m)/dx`` along ``axis``: each face carries its own velocity times the upwind density.

    With ``v_{i+1/2} = (v_i + v_{i+1}) / 2`` the face flux is
    ``F_{i+1/2} = max(v_{i+1/2}, 0) m_i + min(v_{i+1/2}, 0) m_{i+1}``, and the result is
    ``(F_{i+1/2} - F_{i-1/2}) / h``. The explicit update ``m - dt * D F`` has nonnegative coefficients
    for ``max|v| dt / h <= 1`` whatever the sign pattern of ``v``: node ``i`` empties at the rate
    ``max(v_{i+1/2}, 0) - min(v_{i-1/2}, 0)``, which is at most ``max|v|`` because both terms are nonzero
    only where it equals ``(v_{i+1} - v_{i-1}) / 2``. Summed over axes the condition is
    ``dt * sum_d max|v_d| / h_d <= 1``. ``F`` is first-order consistent with ``v m`` across a sign change
    of ``v``, because the donor switches where the face velocity is itself ``O(h)``.

    Two rules that look equivalent are not. Selecting a whole node flux ``v_i m_i`` by the sign of the
    face velocity sends it downwind where ``v_i < 0 <= v_{i+1/2}``: a coefficient of ``-CFL`` at a
    velocity step. Splitting each node flux by the sign of that node's velocity is monotone but, where
    ``v`` changes sign, its face flux is off by ``O(h)`` on one side and ``-O(h)`` on the other, so the
    divergence is off by ``O(1)`` at the two neighbouring nodes at every resolution (#2309).

    Periodic through ``np.roll``, so pad with ghost cells first for any other boundary.
    `AdvectionOperator`'s non-conservative divergence and `tensor_calculus.advection` both call it.

    Args:
        m: Transported density
        v: Velocity component along ``axis``, same shape as ``m``
        axis: Axis along which to differentiate
        h: Grid spacing
        xp: Array module (numpy, cupy or torch)

    Returns:
        Upwinded divergence contribution with the same shape as ``m``
    """
    v_face = 0.5 * (v + _roll(xp, v, -1, axis))
    zero = xp.zeros_like(v_face)
    face_flux = xp.where(v_face > 0, v_face, zero) * m + xp.where(v_face < 0, v_face, zero) * _roll(xp, m, -1, axis)
    return (face_flux - _roll(xp, face_flux, 1, axis)) / h


# =============================================================================
# Boundary Handling
# =============================================================================


def fix_boundaries_one_sided(grad: NDArray, u: NDArray, axis: int, h: float, xp: type = np) -> NDArray:
    """
    Replace boundary values with second-order one-sided differences.

    Central differences wrap around at boundaries (via np.roll), which is
    incorrect for non-periodic BCs. This function fixes the boundary values
    using 3-point one-sided stencils that match the O(h²) accuracy of the
    central interior, so a central-interior + one-sided-boundary gradient is
    O(h²) throughout rather than degrading to O(h) at the edges (Issue #1084).

    Boundary corrections (require ≥ 3 points along ``axis``):
        Left  (i=0):  forward  (-3u[0] + 4u[1] - u[2]) / (2h)     Error: O(h²)
        Right (i=-1): backward (3u[-1] - 4u[-2] + u[-3]) / (2h)   Error: O(h²)

    When the axis has only 2 points the 3-point stencil is unavailable, so the
    (exact-best) first-order one-sided difference O(h) is used for that axis.

    Args:
        grad: Gradient array computed with central differences
        u: Original input array
        axis: Axis along which gradient was computed
        h: Grid spacing
        xp: Array module (numpy or cupy for GPU)

    Returns:
        Gradient array with corrected boundary values (modified in-place)

    Example:
        >>> grad = gradient_central(u, axis=0, h=dx)
        >>> grad = fix_boundaries_one_sided(grad, u, axis=0, h=dx)
    """
    ndim = u.ndim
    n = u.shape[axis]

    def _at(idx: int) -> tuple:
        s = [slice(None)] * ndim
        s[axis] = idx
        return tuple(s)

    if n >= 3:
        # Left boundary: second-order forward, (-3u[0] + 4u[1] - u[2]) / (2h)
        grad[_at(0)] = (-3.0 * u[_at(0)] + 4.0 * u[_at(1)] - u[_at(2)]) / (2.0 * h)
        # Right boundary: second-order backward, (3u[-1] - 4u[-2] + u[-3]) / (2h)
        grad[_at(-1)] = (3.0 * u[_at(-1)] - 4.0 * u[_at(-2)] + u[_at(-3)]) / (2.0 * h)
    else:
        # Only 2 points along axis: the 3-point stencil cannot be formed;
        # first-order one-sided is the exact-best available approximation.
        grad[_at(0)] = (u[_at(1)] - u[_at(0)]) / h
        grad[_at(-1)] = (u[_at(-1)] - u[_at(-2)]) / h

    return grad


# =============================================================================
# Second-Order Derivative Stencils
# =============================================================================


def laplacian_stencil_1d(u: NDArray, h: float, xp: type = np) -> NDArray:
    """
    Standard 3-point Laplacian stencil in 1D.

    Formula: ∂²u/∂x² ≈ (u[i+1] - 2u[i] + u[i-1]) / h²

    Properties:
        - 2nd-order accurate: O(h²)
        - Stencil: [1, -2, 1] / h²
        - Symmetric, negative semi-definite

    Args:
        u: Input 1D array
        h: Grid spacing
        xp: Array module

    Returns:
        Approximation of ∂²u/∂x²
    """
    return (xp.roll(u, -1) - 2 * u + xp.roll(u, 1)) / (h * h)


def laplacian_stencil_nd(u: NDArray, spacings: list[float] | tuple[float, ...], xp: type = np) -> NDArray:
    """
    Standard Laplacian stencil in n dimensions.

    Formula: Δu = Σ_d ∂²u/∂x_d²

    Uses the 3-point stencil in each dimension and sums.

    Args:
        u: Input n-dimensional array
        spacings: Grid spacing per dimension [h₀, h₁, ..., hd₋₁]
        xp: Array module

    Returns:
        Approximation of Δu = ∇²u
    """
    result = xp.zeros_like(u)
    for axis, h in enumerate(spacings):
        result += (_roll(xp, u, -1, axis) - 2 * u + _roll(xp, u, 1, axis)) / (h * h)
    return result


def weighted_laplacian_stencil_nd(
    u: NDArray,
    spacings: list[float] | tuple[float, ...],
    axis_weights: NDArray,
    xp: type = np,
) -> NDArray:
    """
    Weighted Laplacian stencil in n dimensions.

    Formula: Σ_d w_d * ∂²u/∂x_d²

    For diagonal diffusion tensor Σ = diag(σ₀², σ₁², ...),
    computes the anisotropic diffusion: Σ_d σ_d² ∂²u/∂x_d².

    Args:
        u: Input n-dimensional array
        spacings: Grid spacing per dimension [h₀, h₁, ..., hd₋₁]
        axis_weights: Per-axis weights, shape (ndim,)
        xp: Array module

    Returns:
        Weighted Laplacian approximation
    """
    result = xp.zeros_like(u)
    for axis, (h, w) in enumerate(zip(spacings, axis_weights, strict=True)):
        result += w * (_roll(xp, u, -1, axis) - 2 * u + _roll(xp, u, 1, axis)) / (h * h)
    return result


# =============================================================================
# Stencil Coefficients (for matrix assembly)
# =============================================================================


def get_gradient_stencil_coefficients(scheme: str, h: float) -> tuple[list[int], list[float]]:
    """
    Get stencil offsets and coefficients for gradient approximation.

    Useful for building sparse matrices where you need explicit coefficients.

    Args:
        scheme: One of "central", "forward", "backward"
        h: Grid spacing

    Returns:
        Tuple of (offsets, coefficients) where:
            offsets: List of index offsets from center [e.g., [-1, 1]]
            coefficients: List of weights [e.g., [-0.5/h, 0.5/h]]

    Example:
        >>> offsets, coeffs = get_gradient_stencil_coefficients("central", h=0.1)
        >>> # offsets = [-1, 1], coeffs = [-5.0, 5.0]
    """
    if scheme == "central":
        return [-1, 1], [-1.0 / (2 * h), 1.0 / (2 * h)]
    elif scheme == "forward":
        return [0, 1], [-1.0 / h, 1.0 / h]
    elif scheme == "backward":
        return [-1, 0], [-1.0 / h, 1.0 / h]
    else:
        raise ValueError(f"Unknown scheme: {scheme}. Use 'central', 'forward', 'backward'")


def get_laplacian_stencil_coefficients(h: float) -> tuple[list[int], list[float]]:
    """
    Get stencil offsets and coefficients for 1D Laplacian.

    Args:
        h: Grid spacing

    Returns:
        Tuple of (offsets, coefficients)
        offsets = [-1, 0, 1], coeffs = [1/h², -2/h², 1/h²]
    """
    h2 = h * h
    return [-1, 0, 1], [1.0 / h2, -2.0 / h2, 1.0 / h2]


def gradient_nd(
    u: NDArray,
    spacings: list[float] | tuple[float, ...],
    xp: type = np,
) -> list[NDArray]:
    """
    Compute gradient in all dimensions using central differences.

    This is a simple wrapper that applies gradient_central to each dimension.
    No boundary condition handling - uses periodic wrapping via np.roll.

    Replaces tensor_calculus.gradient_simple for particle methods where
    BC is handled separately.

    Args:
        u: Input n-dimensional array
        spacings: Grid spacing per dimension [h₀, h₁, ..., hd₋₁]
        xp: Array module (numpy or cupy for GPU)

    Returns:
        List of gradient arrays, one per dimension: [∂u/∂x₀, ∂u/∂x₁, ...]

    Example:
        >>> u = np.sin(np.linspace(0, 2*np.pi, 100))
        >>> grad = gradient_nd(u, spacings=[0.1])
        >>> # grad[0] contains ∂u/∂x

    Note:
        Issue #625: Added to replace tensor_calculus.gradient_simple
    """
    gradients = []
    for axis, h in enumerate(spacings):
        if h > 1e-14:
            gradients.append(gradient_central(u, axis=axis, h=h, xp=xp))
        else:
            gradients.append(xp.zeros_like(u))
    return gradients


def laplacian_with_bc(
    u: NDArray,
    spacings: list[float] | tuple[float, ...],
    bc: BoundaryConditions | None = None,
    time: float = 0.0,
) -> NDArray:
    """
    Compute Laplacian with boundary condition handling.

    This function combines ghost cell padding from geometry/boundary
    with the laplacian_stencil_nd computation.

    Replaces tensor_calculus.laplacian for cases needing BC-aware Laplacian.

    Args:
        u: Input n-dimensional array
        spacings: Grid spacing per dimension [h₀, h₁, ..., hd₋₁]
        bc: Boundary conditions. If None, uses periodic BC.
        time: Current time for time-dependent BCs

    Returns:
        Laplacian of u with same shape as input

    Example:
        >>> from mfgarchon.geometry.boundary import neumann_bc
        >>> u = np.sin(np.linspace(0, np.pi, 50))
        >>> lap_u = laplacian_with_bc(u, spacings=[0.1], bc=neumann_bc(1))

    Note:
        Issue #625: Added to replace tensor_calculus.laplacian
    """
    from mfgarchon.geometry.boundary import pad_array_with_ghosts

    # Apply ghost cells if BC provided
    if bc is not None:
        # spacing threaded: without it the buffer falls back to dx = 1.0 and an inhomogeneous
        # Neumann value is applied as g/h rather than g (#1904). `spacings` is already a
        # parameter of this function -- it was simply not passed on.
        u_work = pad_array_with_ghosts(u, bc, ghost_depth=1, time=time, spacing=spacings)
    else:
        u_work = u

    # Compute laplacian
    lap = laplacian_stencil_nd(u_work, spacings)

    # Extract interior if ghost cells were added
    if bc is not None:
        slices = [slice(1, -1)] * u.ndim
        lap = lap[tuple(slices)]

    return lap


def weighted_laplacian_with_bc(
    u: NDArray,
    spacings: list[float] | tuple[float, ...],
    axis_weights: NDArray,
    bc: BoundaryConditions | None = None,
    time: float = 0.0,
) -> NDArray:
    """
    Compute weighted Laplacian with boundary condition handling.

    For diagonal diffusion tensor Sigma = diag(sigma_0^2, sigma_1^2, ...),
    computes: sum_d sigma_d^2 * d^2u/dx_d^2

    This is the anisotropic analogue of laplacian_with_bc().

    Args:
        u: Input n-dimensional array
        spacings: Grid spacing per dimension [h_0, h_1, ..., h_{d-1}]
        axis_weights: Per-axis weights (e.g., diagonal of diffusion tensor), shape (ndim,)
        bc: Boundary conditions. If None, uses periodic BC.
        time: Current time for time-dependent BCs

    Returns:
        Weighted Laplacian of u with same shape as input
    """
    from mfgarchon.geometry.boundary import pad_array_with_ghosts

    # Apply ghost cells if BC provided
    if bc is not None:
        # spacing threaded: without it the buffer falls back to dx = 1.0 and an inhomogeneous
        # Neumann value is applied as g/h rather than g (#1904). `spacings` is already a
        # parameter of this function -- it was simply not passed on.
        u_work = pad_array_with_ghosts(u, bc, ghost_depth=1, time=time, spacing=spacings)
    else:
        u_work = u

    # Compute weighted laplacian
    lap = weighted_laplacian_stencil_nd(u_work, spacings, axis_weights)

    # Extract interior if ghost cells were added
    if bc is not None:
        slices = [slice(1, -1)] * u.ndim
        lap = lap[tuple(slices)]

    return lap


__all__ = [
    # First-order derivatives
    "gradient_central",
    "gradient_forward",
    "gradient_backward",
    "gradient_upwind",
    "upwind_momentum",
    "upwind_momentum_derivatives",
    "NumericalHamiltonian",
    "DEFAULT_NUMERICAL_HAMILTONIAN",
    "gradient_upwind_by_velocity",
    "divergence_upwind_by_velocity",
    "gradient_nd",
    # Boundary handling
    "fix_boundaries_one_sided",
    # Second-order derivatives
    "laplacian_stencil_1d",
    "laplacian_stencil_nd",
    "laplacian_with_bc",
    "weighted_laplacian_stencil_nd",
    "weighted_laplacian_with_bc",
    # Coefficients for matrix assembly
    "get_gradient_stencil_coefficients",
    "get_laplacian_stencil_coefficients",
]
