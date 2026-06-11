"""Time-stepping routines for multi-dimensional Fokker-Planck FDM solver.

This module contains the forward time evolution logic for solving the FP equation
using finite differences on multi-dimensional grids.

Module structure per issue #388:
    fp_fdm_time_stepping.py - When - time integration (forward Euler, implicit schemes)

Functions:
    solve_fp_nd_full_system: Main time evolution loop for nD FP equation
    solve_timestep_tensor_explicit: Explicit timestep with tensor diffusion
    solve_timestep_full_nd: Implicit timestep solver for scalar diffusion

Mathematical Background:
    FP Equation: dm/dt + div(alpha * m) = (sigma^2/2) * Laplacian(m)

    The implicit scheme solves:
        (I/dt + A + D) m^{n+1} = m^n / dt

    where A is advection, D is diffusion operators.

Issue #597 Milestone 3 - Hybrid Operator Strategy:
    This module uses a **Defect Correction** approach for robustness:

    - **Explicit solvers**: Use AdvectionOperator (tensor_calculus.advection)
      for high-accuracy Godunov fluxes. See solve_timestep_tensor_explicit().

    - **Implicit solvers**: Use manual sparse matrix construction from
      fp_fdm_alg_*.py modules. These build velocity-based linear upwind
      matrices that form the well-conditioned Jacobian for Newton iteration.

    **Why this hybrid approach?**

    The AdvectionOperator uses state-dependent Godunov upwinding (nonlinear
    flux limiting). While this produces sharp, accurate solutions, it cannot
    be linearized into a sparse matrix via unit-vector probing.

    For implicit solvers, we need the **linearized velocity-based Jacobian**:
        J(m) ≈ ∂(∇·(vm))/∂m

    This is exactly what the manual sparse construction provides. It's the
    correct linear approximation for Newton/Picard iteration.

    **This is standard CFD practice**: Use a stable linear operator for the
    LHS (Jacobian) while evaluating the nonlinear residual on the RHS. Known
    as "Defect Correction" or "Picard Linearization."

Note:
    Functions are exported without leading underscore but should be imported
    with underscore alias in fp_fdm.py for internal use:

        from .fp_fdm_time_stepping import (
            solve_fp_nd_full_system as _solve_fp_nd_full_system,
        )
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import scipy.sparse as sparse

from mfgarchon.geometry.boundary import no_flux_bc
from mfgarchon.geometry.boundary.applicator_base import (
    LinearConstraint,
)
from mfgarchon.geometry.boundary.types import BoundaryFace
from mfgarchon.utils.pde_coefficients import CoefficientField, _DriftDispatcher, diffusion_from_volatility

if TYPE_CHECKING:
    from collections.abc import Callable

    from mfgarchon.geometry import BoundaryConditions

# Import from responsibility-based modules per issue #388
from mfgarchon.utils.mfg_logging import get_logger

from .fp_fdm_advection import compute_advection_from_drift_nd, compute_advection_term_nd
from .fp_fdm_alg_divergence_centered import (
    add_boundary_no_flux_entries_divergence_centered,
    add_interior_entries_divergence_centered,
)
from .fp_fdm_alg_divergence_upwind import add_interior_entries_divergence_upwind
from .fp_fdm_alg_gradient_centered import (
    add_boundary_no_flux_entries_gradient_centered,
    add_interior_entries_gradient_centered,
)
from .fp_fdm_alg_gradient_upwind import (
    add_boundary_no_flux_entries_gradient_upwind,
    add_interior_entries_gradient_upwind,
)
from .fp_fdm_bc import add_boundary_no_flux_entries_divergence_upwind
from .fp_fdm_operators import is_boundary_point

logger = get_logger(__name__)

# =============================================================================
# Scheme dispatch tables (Issue #859: replace if/elif chains)
# =============================================================================

_BOUNDARY_HANDLERS: dict[str, Any] = {
    "gradient_centered": add_boundary_no_flux_entries_gradient_centered,
    "gradient_upwind": add_boundary_no_flux_entries_gradient_upwind,
    "divergence_centered": add_boundary_no_flux_entries_divergence_centered,
    "divergence_upwind": add_boundary_no_flux_entries_divergence_upwind,
}

_INTERIOR_HANDLERS: dict[str, Any] = {
    "gradient_centered": add_interior_entries_gradient_centered,
    "gradient_upwind": add_interior_entries_gradient_upwind,
    "divergence_centered": add_interior_entries_divergence_centered,
    "divergence_upwind": add_interior_entries_divergence_upwind,
}


# =============================================================================
# Dirichlet BC support for nD (Issue #859)
# =============================================================================


def _add_boundary_dirichlet_entries(
    row_indices: list[int],
    col_indices: list[int],
    data_values: list[float],
    flat_idx: int,
    dt: float,
    bc_value: float,
) -> None:
    """Add sparse matrix entries for Dirichlet BC at a boundary point.

    Sets the row to enforce m(boundary) = bc_value by making the diagonal
    entry 1/dt and placing bc_value/dt in the RHS (handled by the caller).
    """
    row_indices.append(flat_idx)
    col_indices.append(flat_idx)
    data_values.append(1.0 / dt)


def _is_dirichlet_at_point(
    boundary_conditions: Any,
    multi_idx: tuple[int, ...],
    shape: tuple[int, ...],
) -> bool:
    """Check if boundary conditions specify Dirichlet at this point.

    Issue #1258 fix (2026-06-10 audit): prior code returned True for ANY Dirichlet
    segment in a mixed BC without checking which wall the point is on, causing every
    boundary point to be treated as Dirichlet regardless of its actual wall type.
    Fix: determine the wall(s) the grid point lies on from multi_idx/shape, then
    query get_bc_type_at_boundary per wall using the BCSegment.boundary matching API.
    """
    try:
        if boundary_conditions.is_uniform:
            return boundary_conditions.type == "dirichlet"
        # Mixed BC: determine which wall(s) this grid point is on, then query per wall.
        _AXIS_NAMES = {0: "x", 1: "y", 2: "z", 3: "w"}
        ndim = len(shape)
        for d in range(ndim):
            name = _AXIS_NAMES.get(d, f"axis{d}")
            if multi_idx[d] == 0:
                if boundary_conditions.get_bc_type_at_boundary(f"{name}_min").value == "dirichlet":
                    return True
            if multi_idx[d] == shape[d] - 1:
                if boundary_conditions.get_bc_type_at_boundary(f"{name}_max").value == "dirichlet":
                    return True
    except (AttributeError, ValueError):
        pass
    return False


def _get_dirichlet_value_at_point(
    boundary_conditions: Any,
    multi_idx: tuple[int, ...],
    shape: tuple[int, ...],
) -> float:
    """Get Dirichlet BC value at a boundary point.

    Issue #1258 fix (2026-06-10 audit): for mixed BC, resolve the value from the
    specific Dirichlet wall the grid point is on via get_bc_value_at_boundary, not
    from the first Dirichlet segment regardless of wall (which would give wrong values
    when multiple Dirichlet walls exist with different values).
    """
    try:
        if boundary_conditions.is_uniform:
            seg = boundary_conditions.segments[0]
            return float(seg.value) if not callable(seg.value) else 0.0
        # Mixed BC: find the Dirichlet wall this grid point is on, return its value.
        _AXIS_NAMES = {0: "x", 1: "y", 2: "z", 3: "w"}
        ndim = len(shape)
        for d in range(ndim):
            name = _AXIS_NAMES.get(d, f"axis{d}")
            if multi_idx[d] == 0:
                wall = f"{name}_min"
                if boundary_conditions.get_bc_type_at_boundary(wall).value == "dirichlet":
                    return float(boundary_conditions.get_bc_value_at_boundary(wall))
            if multi_idx[d] == shape[d] - 1:
                wall = f"{name}_max"
                if boundary_conditions.get_bc_type_at_boundary(wall).value == "dirichlet":
                    return float(boundary_conditions.get_bc_value_at_boundary(wall))
    except (AttributeError, ValueError, IndexError):
        pass
    return 0.0


def _get_bc_type(boundary_conditions: Any) -> str | None:
    """
    Get BC type string from boundary conditions object.

    Supports both:
    - Unified BoundaryConditions (conditions.py) with .type property
    - Legacy BoundaryConditions1DFDM (fdm_bc_1d.py) with .type attribute

    Args:
        boundary_conditions: Any BC object

    Returns:
        BC type string (e.g., "periodic", "dirichlet", "no_flux") or None
    """
    if boundary_conditions is None:
        return None

    # Issue #543 Phase 2: Replace hasattr with try/except
    # Unified BC: has is_uniform property and type property
    try:
        # Try accessing type - may raise ValueError for mixed BC
        return boundary_conditions.type
    except ValueError:
        # Mixed BC: type property raises ValueError when segments have different types.
        # If all segments share the same type, return it (e.g., all Dirichlet).
        try:
            types = {seg.bc_type.value for seg in boundary_conditions.segments}
            if len(types) == 1:
                return types.pop()
        except (AttributeError, TypeError):
            pass
        return None
    except AttributeError:
        # Legacy BC: type is a direct attribute
        return getattr(boundary_conditions, "type", None)


def _get_bc_value(boundary_conditions: Any, face: BoundaryFace) -> float:
    """
    Get BC value at a specific boundary.

    Supports both:
    - Unified BoundaryConditions with get_bc_value_at_boundary()
    - Legacy BoundaryConditions1DFDM with left_value/right_value

    Issue #946: Uses BoundaryFace instead of string identifiers for internal dispatch.

    Args:
        boundary_conditions: Any BC object
        face: Boundary face identifier

    Returns:
        BC value at the specified boundary
    """
    # Try unified BC first (modern API) — still takes string
    try:
        return boundary_conditions.get_bc_value_at_boundary(face.to_string())
    except AttributeError:
        # Fall back to legacy BC: use left_value/right_value
        if face.side == "min":
            return getattr(boundary_conditions, "left_value", 0.0)
        elif face.side == "max":
            return getattr(boundary_conditions, "right_value", 0.0)

    return 0.0


if TYPE_CHECKING:
    from collections.abc import Callable


def _build_diffusion_matrix_with_bc(
    shape: tuple[int, ...],
    spacing: tuple[float, ...],
    D: float,
    dt: float,
    ndim: int,
    bc_constraint_min: LinearConstraint | None = None,
    bc_constraint_max: LinearConstraint | None = None,
) -> tuple[sparse.csr_matrix, np.ndarray]:
    """
    Build implicit diffusion matrix using LinearConstraint pattern.

    .. deprecated:: 0.17.0
        This function is deprecated as part of Issue #597 Milestone 2B.
        Use `LaplacianOperator.as_scipy_sparse()` with `BoundaryConditions` instead.

        Migration example::

            # OLD: Manual matrix assembly with LinearConstraint
            bc_constraint = LinearConstraint(weights={0: 1.0}, bias=0.0)  # Neumann
            A_diffusion, b_bc = _build_diffusion_matrix_with_bc(
                shape=shape, spacing=spacing, D=D, dt=dt, ndim=ndim,
                bc_constraint_min=bc_constraint, bc_constraint_max=bc_constraint
            )
            b_rhs = M_current.ravel() / dt + b_bc
            M_star = sparse.linalg.spsolve(A_diffusion, b_rhs).reshape(shape)

            # NEW: Operator-based assembly with BoundaryConditions
            from mfgarchon.operators.differential.laplacian import LaplacianOperator
            from mfgarchon.geometry.boundary import no_flux_bc

            bc = no_flux_bc(dimension=ndim)
            L_op = LaplacianOperator(spacings=list(spacing), field_shape=shape, bc=bc)
            L_matrix = L_op.as_scipy_sparse()

            I = sparse.eye(np.prod(shape))
            A_diffusion = I / dt - D * L_matrix
            b_rhs = M_current.ravel() / dt
            M_star = sparse.linalg.spsolve(A_diffusion, b_rhs).reshape(shape)

    Implements the matrix assembly protocol from docs/development/matrix_assembly_bc_protocol.md.

    The assembled system is: A @ m^{k+1} = b where:
    - A contains (1/dt + diffusion terms)
    - b is modified by BC bias terms

    Phase 1 (Topology): Periodic BCs would use index wrapping (not handled here)
    Phase 2 (Physics): Bounded BCs use coefficient folding via LinearConstraint

    Parameters
    ----------
    shape : tuple[int, ...]
        Grid shape (N1, N2, ..., Nd)
    spacing : tuple[float, ...]
        Grid spacing (dx, dy, ...)
    D : float
        Diffusion coefficient (D = σ²/2)
    dt : float
        Time step
    ndim : int
        Spatial dimension
    bc_constraint_min : LinearConstraint, optional
        BC constraint for min boundary (default: Neumann du/dn=0)
    bc_constraint_max : LinearConstraint, optional
        BC constraint for max boundary (default: Neumann du/dn=0)

    Returns
    -------
    A : sparse.csr_matrix
        Sparse matrix of shape (N_total, N_total)
    b_bc : np.ndarray
        RHS modification from BC bias terms

    Notes
    -----
    The protocol's "Coefficient Folding" works as follows:
    When stencil at row i accesses ghost column j:
    1. Get LinearConstraint: u_ghost = sum(w_k * u[inner+k]) + bias
    2. Fold weights: A[i, inner+k] += stencil_weight * w_k
    3. Fold bias: b[i] -= stencil_weight * bias
    """
    N_total = int(np.prod(shape))

    # Default BCs: Neumann (du/dn = 0) via ZeroGradientCalculator
    if bc_constraint_min is None:
        bc_constraint_min = LinearConstraint(weights={0: 1.0}, bias=0.0)
    if bc_constraint_max is None:
        bc_constraint_max = LinearConstraint(weights={0: 1.0}, bias=0.0)

    # Triplet lists for COO format (protocol Section 3.2 performance note)
    row_indices = []
    col_indices = []
    data_values = []

    # RHS modification from BC bias terms
    b_bc = np.zeros(N_total)

    for flat_idx in range(N_total):
        multi_idx = np.unravel_index(flat_idx, shape)

        # Diagonal entry: 1/dt + sum of diffusion diagonal contributions
        diag_val = 1.0 / dt

        for d in range(ndim):
            dx = spacing[d]
            stencil_weight = D / (dx**2)  # Weight for off-diagonal diffusion terms

            # Add diffusion diagonal contribution (from interior formula)
            diag_val += 2.0 * stencil_weight

            # Process neighbors in dimension d
            for offset, side in [(-1, "min"), (1, "max")]:
                neighbor_idx = list(multi_idx)
                neighbor_idx[d] += offset
                neighbor_col = neighbor_idx[d]

                # ============================================
                # PHASE 2: PHYSICS FOLDING (bounded BCs)
                # ============================================
                if 0 <= neighbor_col < shape[d]:
                    # Interior point - direct assignment
                    neighbor_flat = np.ravel_multi_index(tuple(neighbor_idx), shape)
                    row_indices.append(flat_idx)
                    col_indices.append(neighbor_flat)
                    data_values.append(-stencil_weight)
                else:
                    # Boundary point - apply coefficient folding
                    constraint = bc_constraint_min if side == "min" else bc_constraint_max

                    # Fold weights into matrix
                    for rel_offset, fold_weight in constraint.weights.items():
                        # Map relative offset to global index
                        # For min boundary: rel_offset 0 -> multi_idx[d]=0
                        # For max boundary: rel_offset 0 -> multi_idx[d]=shape[d]-1
                        inner_idx = list(multi_idx)
                        if side == "min":
                            inner_idx[d] = rel_offset
                        else:
                            inner_idx[d] = shape[d] - 1 - rel_offset

                        # Ensure index is valid
                        if 0 <= inner_idx[d] < shape[d]:
                            inner_flat = np.ravel_multi_index(tuple(inner_idx), shape)
                            # Core formula: A[i, inner] += stencil_weight * fold_weight
                            # Note: stencil_weight is negative for diffusion (-D/dx²)
                            row_indices.append(flat_idx)
                            col_indices.append(inner_flat)
                            data_values.append(-stencil_weight * fold_weight)

                    # Fold bias into RHS (note the sign!)
                    # Original: -stencil_weight * (sum_of_weighted_terms + bias)
                    # Moving bias to RHS: b[i] -= (-stencil_weight) * bias = b[i] += stencil_weight * bias
                    b_bc[flat_idx] += stencil_weight * constraint.bias

        # Add diagonal entry
        row_indices.append(flat_idx)
        col_indices.append(flat_idx)
        data_values.append(diag_val)

    # Build sparse matrix (COO -> CSR for efficient solving)
    A = sparse.coo_matrix((data_values, (row_indices, col_indices)), shape=(N_total, N_total)).tocsr()

    return A, b_bc


def solve_timestep_explicit_with_drift(
    M_current: np.ndarray,
    drift: np.ndarray,
    dt: float,
    sigma: float | np.ndarray,
    spacing: tuple[float, ...],
    ndim: int,
    boundary_conditions: BoundaryConditions | None = None,
    source_term: np.ndarray | None = None,
    cached_laplacian: Any | None = None,  # Issue #927: pre-computed Laplacian sparse matrix
) -> np.ndarray:
    """
    Solve one timestep using semi-implicit scheme with direct drift.

    Uses operator splitting (Lie splitting):
    1. Implicit diffusion: (I - dt*σ²/2*Δ) m* = m^k
    2. Explicit advection: m^{k+1} = m* - dt * div(α * m*)

    This is unconditionally stable for diffusion while allowing direct drift.

    **Issue #597 Milestone 2B**: Refactored to use LaplacianOperator with
    trait-based geometry operators instead of manual matrix assembly.

    Parameters
    ----------
    M_current : np.ndarray
        Current density field
    drift : np.ndarray
        Drift field. For 1D: shape (N,). For nD: shape (ndim, N1, N2, ...)
    dt : float
        Time step
    sigma : float or np.ndarray
        Diffusion coefficient (scalar or spatially varying)
    spacing : tuple[float, ...]
        Grid spacing (dx, dy, ...)
    ndim : int
        Spatial dimension
    boundary_conditions : BoundaryConditions, optional
        Boundary conditions for the domain (default: no-flux on all boundaries)

    Returns
    -------
    np.ndarray
        Updated density at next timestep

    Notes
    -----
    Uses LaplacianOperator.as_scipy_sparse() for diffusion matrix assembly.
    This ensures correct one-sided stencils at Neumann boundaries, matching
    the coefficient folding behavior validated in Issue #597 Milestone 2.

    For full FP no-flux BCs with advection, use no_flux_bc() which properly
    handles mass conservation at boundaries.
    """
    from mfgarchon.operators.differential.laplacian import LaplacianOperator

    shape = M_current.shape
    N_total = int(np.prod(shape))

    # Set default boundary conditions
    if boundary_conditions is None:
        from mfgarchon.geometry.boundary import no_flux_bc

        boundary_conditions = no_flux_bc(dimension=ndim)

    sigma_arr = np.asarray(sigma)
    varying_sigma = sigma_arr.ndim > 0 and float(np.ptp(sigma_arr)) > 1e-12

    # Step 1: implicit diffusion (I/dt - L_D) m* = m^k/dt, mass-conservative no-flux stencil.
    if varying_sigma:
        # Issue #1183: per-point variable-coefficient diffusion. Bake the field D(x)=sigma(x)^2/2
        # into a conservative finite-volume Laplacian (face-averaged D_{i+1/2}, 1ᵀL=0 preserved),
        # so a non-uniform sigma is honored per point instead of collapsed to its mean.
        d_field = diffusion_from_volatility(sigma_arr, kind="field")
        L_matrix = LaplacianOperator(
            spacings=list(spacing),
            field_shape=shape,
            bc=boundary_conditions,
            mass_conservative=True,
            coefficient_field=d_field,
        ).as_scipy_sparse()
        A_diffusion = sparse.eye(N_total) / dt - L_matrix
    else:
        # Scalar (or uniform-array) sigma: unit mass-conservative Laplacian scaled by the scalar D.
        scalar_sigma = float(sigma_arr.reshape(-1)[0]) if sigma_arr.ndim > 0 else float(sigma)
        D = diffusion_from_volatility(scalar_sigma)
        # Issue #927: reuse the cached unit Laplacian (constant sigma across time steps) if given.
        if cached_laplacian is not None:
            L_matrix = cached_laplacian
        else:
            L_matrix = LaplacianOperator(
                spacings=list(spacing), field_shape=shape, bc=boundary_conditions, mass_conservative=True
            ).as_scipy_sparse()
        A_diffusion = sparse.eye(N_total) / dt - D * L_matrix

    # RHS: m^k / dt
    # Note: No b_bc term needed - BCs are incorporated into L_matrix
    b_rhs = M_current.ravel() / dt

    # Solve implicit diffusion
    M_star = sparse.linalg.spsolve(A_diffusion, b_rhs).reshape(shape)

    # Step 2: Explicit advection using direct drift.
    # Issue #1181: pass boundary_conditions so a no-flux domain does not default to
    # periodic (AdvectionOperator bc=None == periodic), which silently wraps wall mass.
    # Issue #1184: conservative FV divergence so mass driven against a no-flux wall by
    # strong drift does not leak (the explicit upwind advection was non-conservative there).
    advection_term = compute_advection_from_drift_nd(
        M_star, drift, spacing, ndim, bc=boundary_conditions, mass_conservative=True
    )
    M_next = M_star - dt * advection_term

    # Step 3: Add source term (MMS verification)
    if source_term is not None:
        M_next += dt * source_term.reshape(shape)

    return M_next


def solve_fp_nd_full_system(
    m_initial_condition: np.ndarray,
    U_solution_for_drift: np.ndarray | None,
    problem: Any,
    boundary_conditions: BoundaryConditions | None = None,
    show_progress: bool | None = None,
    backend: Any | None = None,
    diffusion_field: float | np.ndarray | Any | None = None,
    tensor_diffusion_field: np.ndarray | Callable | None = None,
    advection_scheme: str = "divergence_upwind",
    # Callable drift support (Phase 2 - Issue #487)
    drift_field: Callable | None = None,
    # Progress callback for HierarchicalProgress (Issue #640)
    progress_callback: Callable[[int], None] | None = None,
    # MMS verification support
    source_term: Callable | None = None,
    # Issue #919: Direct velocity field input
    velocity_field: np.ndarray | None = None,
) -> np.ndarray:
    """
    Solve multi-dimensional FP equation using full-dimensional sparse linear system.

    Evolves density forward in time from t=0 to t=T by directly assembling
    and solving the full multi-dimensional system at each timestep.

    This approach avoids the catastrophic operator splitting errors that occur
    with advection-dominated problems (high Peclet number).

    Parameters
    ----------
    m_initial_condition : np.ndarray
        Initial density at t=0. Shape: (N1, N2, ..., Nd)
    U_solution_for_drift : np.ndarray | None
        Value function over time-space grid. Shape: (Nt+1, N1, N2, ..., Nd)
        Used to compute drift velocity v = -coupling_coefficient * grad(U)
        Can be None if drift_field or velocity_field is provided.
    velocity_field : np.ndarray | None
        Precomputed velocity field α*(t, x). Shape: (Nt+1, *spatial_shape)
        for 1D, or (Nt+1, ndim, *spatial_shape) for nD.
        When provided, used directly without U→α conversion. (Issue #919)
    problem : MFGProblem
        The MFG problem definition with geometry and parameters
    boundary_conditions : BoundaryConditions | None
        Boundary condition specification (default: no-flux)
    show_progress : bool
        Whether to display progress bar
    backend : Any | None
        Array backend (currently unused, NumPy only)
    diffusion_field : float | np.ndarray | Callable | None
        Optional diffusion override (Phase 2.4):
        - None: Use problem.sigma
        - float: Constant diffusion
        - ndarray: Spatially/temporally varying diffusion
        - Callable: State-dependent diffusion D(t, x, m) -> float | ndarray
    tensor_diffusion_field : np.ndarray | Callable | None
        Tensor diffusion coefficient (Phase 3.0)
    advection_scheme : str
        Advection term discretization scheme. Options:
        - "gradient_centered": Gradient form + central differences
          (NOT conservative, oscillates for Peclet > 2)
        - "gradient_upwind": Gradient form + upwind differences
          (row-sum consistent but NOT mass-conservative at no-flux walls -- leaks
          mass even for pure diffusion, Issue #1075; stable, O(dx))
        - "divergence_centered": Divergence form + centered fluxes
          (conservative via telescoping, oscillates for Peclet > 2)
        - "divergence_upwind": Divergence form + upwind fluxes
          (conservative via telescoping, stable, O(dx))
        Legacy aliases: "centered"->"gradient_centered",
        "upwind"->"gradient_upwind", "flux"->"divergence_upwind"
    drift_field : Callable | None
        Optional callable drift field (Phase 2 - Issue #487):
        - None: Use U_solution_for_drift to compute drift
        - Callable: State-dependent drift α(t, x, m) -> ndarray
          Signature: (t: float, x: list[ndarray], m: ndarray) -> ndarray
          Returns drift vector field, shape (ndim, N1, N2, ..., Nd)
          where ndim is spatial dimension.

    Returns
    -------
    np.ndarray
        Density evolution over time. Shape: (Nt+1, N1, N2, ..., Nd)

    Notes
    -----
    - No splitting error: Direct discretization preserves operator coupling
    - Mass conservation: Proper flux balance at all boundaries
    - Enforces non-negativity: m >= 0 everywhere
    - Forward time evolution: k=0 -> Nt-1
    - Complexity: O(N^d) unknowns per timestep

    Mathematical Formulation
    ------------------------
    FP Equation: dm/dt + div(m * v) = (sigma^2/2) * Laplacian(m)

    Direct Discretization:
        (I/dt + A + D) m^{n+1} = m^n / dt

    where:
    - I: identity matrix (size N_total x N_total)
    - A: advection operator (full multi-D upwind discretization)
    - D: diffusion operator (full multi-D Laplacian)
    """
    # Get problem dimensions
    ndim = problem.geometry.dimension
    shape = tuple(problem.geometry.get_grid_shape())
    dt = problem.dt
    coupling_coefficient = getattr(problem, "coupling_coefficient", 1.0)

    # Get grid spacing and geometry
    spacing = problem.geometry.get_grid_spacing()
    grid = problem.geometry  # Geometry IS the grid

    # Determine if using callable drift (Phase 2 - Issue #487)
    use_callable_drift = drift_field is not None and callable(drift_field)

    # Issue #919 Phase 2: velocity_field → pass to implicit solver directly
    # via interface_velocity. No virtual-U conversion needed.
    _velocity_array = velocity_field  # Store for per-timestep slicing
    if velocity_field is not None:
        Nt = velocity_field.shape[0]
        # Create a zero-U dispatcher (U is unused when interface_velocity is set)
        drift = _DriftDispatcher(drift_field=None, Nt=Nt, spatial_shape=shape, dimension=ndim)
    elif U_solution_for_drift is not None:
        # Determine timestep count
        # Use U_solution shape for timestep count (allows flexible input sizes)
        Nt = U_solution_for_drift.shape[0]
        # Issue #641: Create unified _DriftDispatcher for cleaner time loop
        drift = _DriftDispatcher(
            drift_field=drift_field if use_callable_drift else U_solution_for_drift,
            Nt=Nt,
            spatial_shape=shape,
            dimension=ndim,
        )
    elif use_callable_drift:
        # When using callable drift, get Nt from problem
        Nt = problem.Nt + 1
        drift = _DriftDispatcher(
            drift_field=drift_field,
            Nt=Nt,
            spatial_shape=shape,
            dimension=ndim,
        )
    else:
        raise ValueError("Either velocity_field, U_solution_for_drift, or callable drift_field must be provided")

    # Handle tensor diffusion (Phase 3.0) vs scalar diffusion
    use_tensor_diffusion = tensor_diffusion_field is not None

    if use_tensor_diffusion:
        # Tensor diffusion path
        tensor_base = tensor_diffusion_field
        sigma_base = None  # Not used for tensor diffusion
    else:
        # Scalar diffusion path (Phase 2.4)
        if diffusion_field is None:
            sigma_base = problem.sigma
        elif isinstance(diffusion_field, (int, float)):
            sigma_base = float(diffusion_field)
        elif callable(diffusion_field):
            # Callable: will be evaluated per timestep
            sigma_base = diffusion_field
        elif isinstance(diffusion_field, np.ndarray):
            # Array: spatially or spatiotemporally varying
            sigma_base = diffusion_field
        else:
            sigma_base = problem.sigma
        tensor_base = None

    # Validate input shapes (spatial dimensions must match)
    assert m_initial_condition.shape == shape, (
        f"Initial condition shape {m_initial_condition.shape} doesn't match problem shape {shape}"
    )
    # Only validate spatial dimensions of U_solution (timestep count is flexible)
    if U_solution_for_drift is not None and Nt > 0:
        U_spatial_shape = U_solution_for_drift.shape[1:]
        assert U_spatial_shape == shape, (
            f"Value function spatial shape {U_spatial_shape} doesn't match problem shape {shape}"
        )

    # Edge case: zero timesteps - return empty array
    if Nt == 0:
        return np.zeros((0, *shape), dtype=np.float64)

    # Allocate solution array
    M_solution = np.zeros((Nt, *shape), dtype=np.float64)
    M_solution[0] = m_initial_condition.copy()

    # Ensure non-negativity of initial condition
    M_solution[0] = np.maximum(M_solution[0], 0)

    # Enforce Dirichlet BC on initial condition (for 1D problems)
    # Uses helper functions for unified/legacy BC compatibility
    bc_type = _get_bc_type(boundary_conditions)
    if bc_type == "dirichlet" and ndim == 1:
        M_solution[0, 0] = _get_bc_value(boundary_conditions, BoundaryFace(0, "min"))
        M_solution[0, -1] = _get_bc_value(boundary_conditions, BoundaryFace(0, "max"))

    # Edge cases
    if Nt <= 1:
        return M_solution

    # Set default boundary conditions
    if boundary_conditions is None:
        boundary_conditions = no_flux_bc(dimension=ndim)

    # Progress bar for forward timesteps
    # n_time_points - 1 steps to go from t=0 to t=T
    # Issue #640: When progress_callback is provided (from HierarchicalProgress),
    # suppress internal bar to avoid duplicate progress display
    use_external_progress = progress_callback is not None
    from mfgarchon.utils.progress import create_progress_bar, should_show_progress

    timestep_range = create_progress_bar(
        range(Nt - 1),
        verbose=should_show_progress(show_progress) and not use_external_progress,
        desc=f"FP {ndim}D (full system)",
    )

    # Pre-compute spatial grid for source term evaluation (if needed)
    if source_term is not None:
        x_grid = grid.get_spatial_grid()  # (N, d) ndarray

    # Issue #927: Pre-compute Laplacian matrix for constant-coefficient diffusion.
    # For constant sigma, the Laplacian matrix is the same at every time step.
    # Caching it avoids Nt redundant sparse matrix assemblies.
    _cached_laplacian = None
    _is_constant_sigma = not callable(getattr(problem, "volatility_field", None)) and not isinstance(
        getattr(problem, "volatility_field", None), np.ndarray
    )
    # Only cache for uniform BCs (LaplacianOperator doesn't support mixed/Dirichlet)
    _bc_is_uniform = getattr(boundary_conditions, "is_uniform", False) if boundary_conditions is not None else False
    if _is_constant_sigma and _bc_is_uniform:
        try:
            from mfgarchon.operators.differential.laplacian import LaplacianOperator

            # mass_conservative=True: cached implicit-FP diffusion matrix must conserve mass
            # at no-flux walls (column-conservative stencil; Issue #1184).
            _L_op = LaplacianOperator(
                spacings=list(spacing), field_shape=shape, bc=boundary_conditions, mass_conservative=True
            )
            _cached_laplacian = _L_op.as_scipy_sparse()
        except (ValueError, AttributeError):
            _cached_laplacian = None  # Fallback: rebuild each step

    # Time evolution loop (forward in time)
    for k in timestep_range:
        M_current = M_solution[k]

        # Evaluate source term at current timestep (implicit: evaluate at t_{k+1})
        if source_term is not None:
            t_next = (k + 1) * dt
            source_values = source_term(t_next, x_grid).ravel()
        else:
            source_values = None

        # Determine drift source using _DriftDispatcher evaluator (Issue #641)
        if drift.is_callable():
            # Callable drift: evaluate velocity directly
            # Drift callable signature: (t, x_coords, m) -> drift_array
            # For 1D: drift_array shape is (N,)
            # For nD: drift_array shape is (ndim, N1, N2, ...) for vector drift
            drift_values = drift.evaluate_velocity_at(k, grid.coordinates, M_current, dt)
            U_current = None  # Not needed when drift is provided directly
        else:
            # Array/MFG drift: get U slice for gradient computation in solver
            drift_values = None
            U_current = drift.get_U_at(k)

        # Issue #919: velocity_field → face-centered interface velocity
        vel_at_k = None
        if _velocity_array is not None:
            vel_idx = min(k, _velocity_array.shape[0] - 1)
            vel_slice = _velocity_array[vel_idx]
            # Wrap in list for per-dimension indexing: interface_velocity[d][...]
            if vel_slice.ndim == 1:
                # 1D face-centered: (Nx-1,) → [array(Nx-1,)]
                vel_at_k = [vel_slice]
            elif vel_slice.ndim == len(shape):
                # nD node-centered: (*shape) → [(shape,)] per dim (fallback)
                vel_at_k = [vel_slice] * ndim
            else:
                # nD: (ndim, *face_shape) → list of per-dim arrays
                vel_at_k = [vel_slice[d] for d in range(ndim)]

        if use_tensor_diffusion:
            # Tensor diffusion path - explicit timestepping
            M_next = solve_timestep_tensor_explicit(
                M_current,
                U_current,
                problem,
                dt,
                tensor_base,
                coupling_coefficient,
                spacing,
                grid,
                ndim,
                shape,
                boundary_conditions,
                k,
                drift=drift_values,
                source_term=source_values,
            )
        elif drift.is_callable():
            # Callable drift with scalar diffusion - use explicit Forward Euler
            # This avoids the mathematically incorrect synthetic U approach
            diffusion = CoefficientField(sigma_base, problem.sigma, "volatility_field", dimension=ndim)
            sigma_at_k = diffusion.evaluate_at(timestep_idx=k, grid=grid.coordinates, density=M_current, dt=dt)

            M_next = solve_timestep_explicit_with_drift(
                M_current,
                drift_values,
                dt,
                sigma_at_k,
                spacing,
                ndim,
                boundary_conditions,
                source_term=source_values,
                cached_laplacian=_cached_laplacian,  # Issue #927
            )
        else:
            # MFG-coupled mode: scalar diffusion + U/velocity-based drift - implicit solver
            diffusion = CoefficientField(sigma_base, problem.sigma, "volatility_field", dimension=ndim)
            sigma_at_k = diffusion.evaluate_at(timestep_idx=k, grid=grid.coordinates, density=M_current, dt=dt)

            M_next = solve_timestep_full_nd(
                M_current,
                U_current,
                problem,
                dt,
                sigma_at_k,
                coupling_coefficient,
                spacing,
                grid,
                ndim,
                shape,
                boundary_conditions,
                advection_scheme=advection_scheme,
                source_term=source_values,
                interface_velocity=vel_at_k,
            )

        M_solution[k + 1] = M_next

        # Issue #881: NaN/Inf detection after each timestep
        if not np.all(np.isfinite(M_next)):
            nan_count = np.sum(~np.isfinite(M_next))
            raise ValueError(
                f"FP solver produced NaN/Inf at timestep {k + 1}/{Nt - 1} "
                f"({nan_count} non-finite values). "
                "Check CFL condition: dt * sigma^2 / dx^2 should be < 0.5 for explicit schemes."
            )

        # Issue #880: Enforce non-negativity with diagnostic warning
        min_val = np.min(M_next)
        if min_val < 0:
            if min_val < -1e-10:
                logger.warning(
                    "FP solver: negative density clipped at timestep %d (min=%.2e). "
                    "This may indicate discretization issues.",
                    k + 1,
                    min_val,
                )
            M_solution[k + 1] = np.maximum(M_solution[k + 1], 0)

        # Enforce Dirichlet BC (using unified/legacy compatible helper)
        if bc_type == "dirichlet" and ndim == 1:
            M_solution[k + 1, 0] = _get_bc_value(boundary_conditions, BoundaryFace(0, "min"))
            M_solution[k + 1, -1] = _get_bc_value(boundary_conditions, BoundaryFace(0, "max"))

        # Issue #640: Report progress to hierarchical progress bar
        if progress_callback is not None:
            progress_callback(1)

    return M_solution


def solve_timestep_tensor_explicit(
    M_current: np.ndarray,
    U_current: np.ndarray | None,
    problem: Any,
    dt: float,
    tensor_field: np.ndarray | Callable,
    coupling_coefficient: float,
    spacing: tuple[float, ...],
    grid: Any,
    ndim: int,
    shape: tuple[int, ...],
    boundary_conditions: Any,
    timestep_idx: int,
    drift: np.ndarray | None = None,
    source_term: np.ndarray | None = None,
) -> np.ndarray:
    """
    Solve one timestep with tensor diffusion using explicit Forward Euler.

    Implements: m^{k+1} = m^k + dt * (div(Sigma * grad(m)) - div(alpha * m))

    Parameters
    ----------
    M_current : np.ndarray
        Current density
    U_current : np.ndarray or None
        Current value function (for MFG coupling). None if drift is provided.
    problem : Any
        MFG problem
    dt : float
        Time step
    tensor_field : np.ndarray or callable
        Tensor diffusion coefficient Sigma
    coupling_coefficient : float
        Drift coupling coefficient
    spacing : tuple
        Grid spacing (dx, dy, ...)
    grid : Any
        Geometry/grid object
    ndim : int
        Spatial dimension
    shape : tuple
        Grid shape
    boundary_conditions : BoundaryConditions
        Boundary condition specification
    timestep_idx : int
        Current timestep index
    drift : np.ndarray or None
        Direct drift field (for standalone FP). If provided, U_current is ignored.

    Returns
    -------
    np.ndarray
        Updated density at next timestep
    """
    from mfgarchon.operators.differential.diffusion import apply_diffusion

    # Evaluate tensor at current state
    if callable(tensor_field):
        # Callable tensor: Sigma(t, x, m) -> (d, d) array
        t = timestep_idx * dt
        Sigma = np.zeros((*shape, ndim, ndim))
        for idx in np.ndindex(shape):
            x_coords = np.array([grid.coordinates[d][idx[d]] for d in range(ndim)])
            m_at_point = M_current[idx]
            Sigma[idx] = tensor_field(t, x_coords, m_at_point)

        # Validate PSD
        coeff = CoefficientField(tensor_field, None, "tensor_diffusion_field", dimension=ndim)
        coeff.validate_tensor_psd(Sigma)
    elif isinstance(tensor_field, np.ndarray):
        # Array tensor: constant or spatially varying
        if tensor_field.ndim == 2:
            # Constant tensor (d, d)
            Sigma = tensor_field
        elif tensor_field.ndim == ndim + 2:
            # Spatially varying (*shape, d, d)
            Sigma = tensor_field
        else:
            raise ValueError(
                f"tensor_field array must have shape (d, d) or (*shape, d, d), "
                f"got shape {tensor_field.shape} for ndim={ndim}"
            )

        # Validate PSD
        coeff = CoefficientField(tensor_field, None, "tensor_diffusion_field", dimension=ndim)
        coeff.validate_tensor_psd(Sigma)
    else:
        raise TypeError(f"tensor_field must be np.ndarray or callable, got {type(tensor_field)}")

    # Compute tensor diffusion term: div(Sigma * grad(m))
    # Issue #625: Migrated from tensor_calculus to operators/differential
    diffusion_term = apply_diffusion(M_current, Sigma, list(spacing), bc=boundary_conditions)

    # Compute advection term: div(alpha * m)
    # Use upwind scheme from fp_fdm_advection module
    if drift is not None:
        # Direct drift provided (standalone FP mode).
        # Issue #1181: pass boundary_conditions (the U-derived branch below already does)
        # so a no-flux domain does not default to periodic and wrap wall mass.
        # Issue #1184: conservative FV divergence (zero advective flux through no-flux walls).
        advection_term = compute_advection_from_drift_nd(
            M_current, drift, spacing, ndim, bc=boundary_conditions, mass_conservative=True
        )
    else:
        # MFG coupled mode: derive drift from U
        advection_term = compute_advection_term_nd(
            M_current, U_current, coupling_coefficient, spacing, ndim, boundary_conditions
        )

    # Explicit Forward Euler update
    M_next = M_current + dt * (diffusion_term - advection_term)

    # Add source term (MMS verification)
    if source_term is not None:
        M_next += dt * source_term.reshape(shape)

    return M_next


def solve_timestep_full_nd(
    M_current: np.ndarray,
    U_current: np.ndarray,
    problem: Any,
    dt: float,
    sigma: float,
    coupling_coefficient: float,
    spacing: tuple[float, ...],
    grid: Any,
    ndim: int,
    shape: tuple[int, ...],
    boundary_conditions: Any,
    advection_scheme: str = "divergence_upwind",
    source_term: np.ndarray | None = None,
    interface_velocity: np.ndarray | None = None,
) -> np.ndarray:
    """
    Solve one timestep of the full nD FP equation.

    Assembles sparse matrix A and RHS b, then solves A*m_{k+1} = b.

    Parameters
    ----------
    M_current : np.ndarray
        Current density field. Shape: (N1, N2, ..., Nd)
    U_current : np.ndarray
        Current value function. Shape: (N1, N2, ..., Nd)
    problem : MFGProblem
        Problem definition
    dt : float
        Time step
    sigma : float
        Diffusion coefficient
    coupling_coefficient : float
        Coupling coefficient for drift term
    spacing : tuple[float, ...]
        Grid spacing in each dimension
    grid : TensorProductGrid
        Grid object
    ndim : int
        Spatial dimension
    shape : tuple[int, ...]
        Grid shape (N1, N2, ..., Nd)
    boundary_conditions : Any
        Boundary condition specification
    advection_scheme : str
        Advection term discretization scheme. Options:
        - "gradient_centered": Gradient form + central differences
          (NOT conservative, oscillates for Peclet > 2)
        - "gradient_upwind": Gradient form + upwind differences
          (row-sum consistent but NOT mass-conservative at no-flux walls -- leaks
          mass even for pure diffusion, Issue #1075; stable, O(dx))
        - "divergence_centered": Divergence form + centered fluxes
          (conservative via telescoping, oscillates for Peclet > 2)
        - "divergence_upwind": Divergence form + upwind fluxes
          (conservative via telescoping, stable, O(dx)) [DEFAULT]
        Legacy aliases: "centered"->"gradient_centered",
        "upwind"->"gradient_upwind", "flux"->"divergence_upwind"

    Returns
    -------
    np.ndarray
        Next density field. Shape: (N1, N2, ..., Nd)
    """
    # Map legacy scheme names to new names
    scheme_aliases = {
        "centered": "gradient_centered",
        "upwind": "gradient_upwind",
        "flux": "divergence_upwind",
    }
    advection_scheme = scheme_aliases.get(advection_scheme, advection_scheme)

    # Validate scheme name
    valid_schemes = {"gradient_centered", "gradient_upwind", "divergence_centered", "divergence_upwind"}
    if advection_scheme not in valid_schemes:
        raise ValueError(f"Unknown advection_scheme '{advection_scheme}'. Valid options: {sorted(valid_schemes)}")
    # Total number of unknowns
    N_total = int(np.prod(shape))

    # Flatten current state and value function
    m_flat = M_current.ravel()  # Row-major (C-order)
    u_flat = U_current.ravel()

    # Pre-allocate lists for COO format sparse matrix
    row_indices: list[int] = []
    col_indices: list[int] = []
    data_values: list[float] = []

    # Issue #1250 2026-06-10 audit: pre-validate BC type before the per-point loop.
    # Uniform 'robin' has no correct implicit stencil here; silently routing it to the
    # interior handler produces an absorbing wall (ghost-zero, +D/dx² row-sum).  Fail
    # loud once rather than silently corrupting every boundary row.
    _bc_is_uniform = getattr(boundary_conditions, "is_uniform", None)
    if _bc_is_uniform:
        _bc_type_str = boundary_conditions.type  # safe: is_uniform True → .type won't raise
        if _bc_type_str == "robin":
            raise NotImplementedError(
                "Implicit FDM FP assembly does not support uniform 'robin' boundary conditions; "
                "the boundary stencil for Robin BC is not implemented on this path. "
                "Use 'no_flux' or 'neumann' for zero-flux (reflecting) walls. (Issue #1250)"
            )

    # Build matrix by iterating over all grid points
    for flat_idx in range(N_total):
        # Convert flat index to multi-index (i, j, k, ...)
        multi_idx = grid.get_multi_index(flat_idx)

        # Extract local diffusion coefficient (scalar or from spatially varying array)
        if isinstance(sigma, np.ndarray):
            # For spatially varying diffusion, extract value at this grid point
            sigma_local = float(sigma[multi_idx])
        else:
            sigma_local = sigma

        # Check if this is a boundary point
        is_boundary = is_boundary_point(multi_idx, shape, ndim)

        # Determine BC type (Issue #859: explicit legacy handling, fail on unknown)
        # Issue #1250 2026-06-10 audit: uniform 'neumann' is homogeneous-Neumann, identical
        # to 'no_flux' for the FP equation; the operator paths already treat them identically
        # (laplacian.py:307, advection.py:316).  Route to the same no-flux boundary handler.
        try:
            is_no_flux = boundary_conditions.is_uniform and boundary_conditions.type in (
                "no_flux",
                "neumann",
            )
            is_uniform = boundary_conditions.is_uniform
        except AttributeError:
            # Legacy BoundaryConditions1D from fdm_bc_1d: has .type string but no .is_uniform
            legacy_type = getattr(boundary_conditions, "type", None)
            if legacy_type is None:
                raise TypeError(
                    f"Unsupported boundary_conditions type: {type(boundary_conditions).__name__}. "
                    "Expected BoundaryConditions from mfgarchon.geometry.boundary."
                ) from None
            # Legacy BC: treat as no-flux for backward compatibility
            # (legacy periodic BC relies on no-flux boundary assembly + interior wrapping)
            is_no_flux = True
            is_uniform = False

        # Dispatch to scheme-specific assembly function (Issue #859)
        if is_boundary and _is_dirichlet_at_point(boundary_conditions, multi_idx, shape):
            # Dirichlet BC: row becomes identity (m = prescribed value)
            bc_value = _get_dirichlet_value_at_point(boundary_conditions, multi_idx, shape)
            _add_boundary_dirichlet_entries(row_indices, col_indices, data_values, flat_idx, dt, bc_value)
        elif (is_no_flux or not is_uniform) and is_boundary:
            # No-flux or mixed BC at boundary: scheme-specific assembly
            _BOUNDARY_HANDLERS[advection_scheme](
                row_indices,
                col_indices,
                data_values,
                flat_idx,
                multi_idx,
                shape,
                ndim,
                dt,
                sigma_local,
                coupling_coefficient,
                spacing,
                u_flat,
                grid,
                interface_velocity=interface_velocity,
            )
        else:
            # Interior point or periodic boundary
            _INTERIOR_HANDLERS[advection_scheme](
                row_indices,
                col_indices,
                data_values,
                flat_idx,
                multi_idx,
                shape,
                ndim,
                dt,
                sigma_local,
                coupling_coefficient,
                spacing,
                u_flat,
                grid,
                boundary_conditions,
                interface_velocity=interface_velocity,
            )

    # Assemble sparse matrix
    A_matrix = sparse.coo_matrix((data_values, (row_indices, col_indices)), shape=(N_total, N_total)).tocsr()

    # Right-hand side
    b_rhs = m_flat / dt

    # Enforce Dirichlet RHS at boundary points (Issue #859, extended for mixed BC #1258).
    # Issue #1258 fix (2026-06-10 audit): _get_bc_type returns None for mixed BC, so the
    # prior check `== "dirichlet"` never fired, leaving b_rhs[boundary] = m_current/dt and
    # solving m_next[boundary] = m_current[boundary] (frozen at IC) instead of the prescribed
    # Dirichlet value.  Now detect mixed BC with any Dirichlet segment and apply per point.
    bc_type_str = _get_bc_type(boundary_conditions)
    _has_dirichlet = bc_type_str == "dirichlet" or (
        bc_type_str is None
        and any(seg.bc_type.value == "dirichlet" for seg in getattr(boundary_conditions, "segments", []))
    )
    if _has_dirichlet:
        for idx in range(N_total):
            multi_idx = np.unravel_index(idx, shape)
            if is_boundary_point(multi_idx, shape, ndim) and _is_dirichlet_at_point(
                boundary_conditions, multi_idx, shape
            ):
                bc_value = _get_dirichlet_value_at_point(boundary_conditions, multi_idx, shape)
                b_rhs[idx] = bc_value / dt

    # Add source term to RHS (MMS verification)
    if source_term is not None:
        b_rhs = b_rhs + source_term

    # Solve linear system
    m_next_flat = sparse.linalg.spsolve(A_matrix, b_rhs)

    # Reshape back to multi-dimensional array
    m_next = m_next_flat.reshape(shape)

    return m_next
