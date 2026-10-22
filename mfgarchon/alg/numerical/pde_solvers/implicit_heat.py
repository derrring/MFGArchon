"""
Implicit solver for the heat equation using theta method.

Solves: ∂T/∂t = α ∂²T/∂x² (1D) or ∂T/∂t = α ∇²T (nD)

Uses theta-method time discretization:
    (I - θ·α·dt·L)·T^{n+1} = (I + (1-θ)·α·dt·L)·T^n + dt·f^{n+θ}

Where:
    - θ = 0.5: Crank-Nicolson (2nd-order accurate, unconditionally stable)
    - θ = 1.0: Backward Euler (1st-order accurate, more stable)
    - θ = 0.0: Forward Euler (explicit, CFL-limited)

Key advantage: Allows CFL >> 1, reducing timesteps by 10-100× vs explicit schemes.

Created: 2026-01-18 (Issue #605 Phase 1.1)
"""

from collections.abc import Callable

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.base import CartesianGrid
from mfgarchon.geometry.boundary import BoundaryConditions


class ImplicitHeatSolver:
    """
    Implicit solver for heat equation using theta-method.

    Achieves unconditional stability (CFL >> 1) via implicit time stepping,
    reducing timesteps by 10-100× compared to explicit schemes.

    Examples
    --------
    >>> from mfgarchon.geometry import TensorProductGrid
    >>> from mfgarchon.geometry.boundary import neumann_bc
    >>> grid = TensorProductGrid(bounds=[(0, 1)], Nx=[100])
    >>> bc = neumann_bc(dimension=1)
    >>> solver = ImplicitHeatSolver(grid, alpha=0.01, bc=bc, theta=0.5)
    >>>
    >>> # Initial condition
    >>> T0 = np.exp(-50 * (grid.coordinates[0] - 0.5)**2)
    >>>
    >>> # Solve with large timestep (CFL = 10)
    >>> dt = 0.1 * 10 * grid.spacing[0]**2 / solver.alpha  # CFL = 10
    >>> T = solver.solve_step(T0, dt)  # Unconditionally stable!
    """

    def __init__(
        self,
        grid: TensorProductGrid,
        alpha: float,
        bc: BoundaryConditions,
        theta: float = 0.5,
        forcing: Callable | None = None,
    ):
        """
        Initialize implicit heat solver.

        Parameters
        ----------
        grid : TensorProductGrid
            Spatial grid for discretization.
        alpha : float
            Thermal diffusivity (m²/s). For heat equation: α = k/(ρ·c_p).
        bc : BoundaryConditions
            Boundary conditions for heat equation.
        theta : float, default=0.5
            Theta-method parameter:
                - 0.5: Crank-Nicolson (2nd-order, unconditionally stable)
                - 1.0: Backward Euler (1st-order, more stable for rough data)
                - 0.0: Forward Euler (explicit, not recommended)
        forcing : callable, optional
            Source term f(x, t). If None, assumes f = 0.

        Notes
        -----
        The solver precomputes and factors the system matrix, so initialization
        has O(N³) cost for direct solver, O(N) for iterative. Subsequent timesteps
        are O(N) for direct, O(N·k_iter) for iterative.

        For large 3D problems (N > 10⁶), consider using iterative solver instead
        of direct LU factorization.
        """
        # The forcing is evaluated on the grid's node axes, which only CartesianGrid declares (#2490).
        if not isinstance(grid, CartesianGrid):
            raise TypeError(
                f"ImplicitHeatSolver solves on a CartesianGrid, the type that declares node coordinates and their "
                f"order; got a {type(grid).__name__} (#2490)."
            )
        self.grid = grid
        self.alpha = alpha
        self.bc = bc
        self.theta = theta
        self.forcing = forcing

        # Validate theta
        if not 0.0 <= theta <= 1.0:
            raise ValueError(f"theta must be in [0, 1], got {theta}")

        # Get Laplacian operator from geometry (Issue #595)
        self.laplacian_op = grid.get_laplacian_operator(order=2, bc=bc)

        # Convert LinearOperator to sparse matrix for direct solver
        # (scipy.sparse.linalg.splu requires sparse matrix, not LinearOperator)
        if hasattr(self.laplacian_op, "get_matrix"):
            # Use explicit matrix if available
            self.laplacian_matrix = self.laplacian_op.get_matrix()
        else:
            # Fallback: convert LinearOperator to matrix via matvec
            N = grid.num_spatial_points
            I_dense = np.eye(N)
            L_dense = np.array([self.laplacian_op @ col for col in I_dense.T]).T
            self.laplacian_matrix = sp.csr_matrix(L_dense)

        # Precompute system matrices
        self._build_system_matrices()

    def _build_system_matrices(self):
        """
        Build and factor the system matrices for theta-method.

        Constructs:
            A_lhs = I - θ·α·dt·L   (implicit term)
            A_rhs = I + (1-θ)·α·dt·L   (explicit term)

        Note: We don't know dt yet, so we store the pattern and will
        scale by dt in solve_step().
        """
        N = self.grid.num_spatial_points
        identity_matrix = sp.eye(N, format="csr")

        # Store base matrices (without dt scaling)
        # A_lhs = I - θ·α·dt·L => I + θ·α·(-dt·L)
        # A_rhs = I + (1-θ)·α·dt·L

        # We'll construct: A_lhs_base = I, A_rhs_base = I
        # and L_scaled = α·L, then apply dt·theta in solve_step()

        self.identity_matrix = identity_matrix
        self.L = self.laplacian_matrix  # Sparse matrix with BC incorporated

        # For efficiency: precompute α·L
        self.alpha_L = self.alpha * self.L

        # We'll build the actual system matrix in solve_step() based on dt
        self._solver = None  # Will be created on first solve_step call
        self._last_dt = None  # Track if dt changes (requires refactorization)

    def solve_step(self, T_prev: np.ndarray, dt: float) -> np.ndarray:
        """
        Solve one timestep: T^{n+1} = solve((I - θ·α·dt·L), rhs).

        Parameters
        ----------
        T_prev : np.ndarray
            Temperature field at previous timestep, shape (Nx,) or (Nx, Ny, ...).
        dt : float
            Timestep size. Can violate CFL condition (implicit scheme is stable).

        Returns
        -------
        T_next : np.ndarray
            Temperature field at next timestep, same shape as T_prev.

        Notes
        -----
        For Crank-Nicolson (θ=0.5), the scheme is unconditionally stable for
        any dt > 0, allowing CFL factors >> 1.

        For Backward Euler (θ=1.0), the scheme is even more stable and suitable
        for stiff problems.
        """
        # Flatten if multidimensional
        original_shape = T_prev.shape
        T_flat = T_prev.ravel()

        # Build system matrix if dt changed or first call
        if self._solver is None or self._last_dt != dt:
            self._build_solver_for_dt(dt)
            self._last_dt = dt

        # Right-hand side: (I + (1-θ)·α·dt·L)·T^n
        rhs = self.identity_matrix @ T_flat + (1.0 - self.theta) * dt * (self.alpha_L @ T_flat)

        # Add forcing term if present: dt·f^{n+θ}
        if self.forcing is not None:
            # For simplicity, evaluate forcing at T_prev (explicit)
            # More sophisticated: evaluate at θ·T^{n+1} + (1-θ)·T^n (requires iteration)
            f_values = self.forcing(self.grid.coordinates, 0.0)  # t=0 placeholder
            rhs += dt * f_values.ravel()

        # Solve: (I - θ·α·dt·L)·T^{n+1} = rhs
        T_next_flat = self._solver(rhs)

        # Reshape back to original shape
        T_next = T_next_flat.reshape(original_shape)

        return T_next

    def _build_solver_for_dt(self, dt: float):
        """
        Build and factor the linear system for given timestep dt.

        Constructs: A = I - θ·α·dt·L
        Factorizes using sparse LU decomposition.
        """
        # Left-hand side: A = I - θ·α·dt·L
        A_lhs = self.identity_matrix - self.theta * dt * self.alpha_L

        # Factor the system (LU decomposition)
        # For large systems, could use iterative solver (spla.gmres, spla.cg)
        try:
            lu = spla.splu(A_lhs.tocsc())  # LU decomposition
            self._solver = lambda rhs: lu.solve(rhs)
        except RuntimeError as e:
            # Fallback to iterative solver if direct solver fails
            print(f"Warning: Direct solver failed ({e}), using iterative GMRES")
            self._solver = lambda rhs: spla.gmres(A_lhs, rhs, atol=1e-8)[0]

    def solve_multiple_steps(self, T_initial: np.ndarray, dt: float, num_steps: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Solve multiple timesteps with fixed dt.

        Parameters
        ----------
        T_initial : np.ndarray
            Initial temperature field.
        dt : float
            Timestep size (fixed for all steps).
        num_steps : int
            Number of timesteps to solve.

        Returns
        -------
        T_history : np.ndarray
            Temperature field history, shape (num_steps+1, *T_initial.shape).
        times : np.ndarray
            Time points, shape (num_steps+1,).

        Examples
        --------
        >>> solver = ImplicitHeatSolver(grid, alpha=0.01, bc=neumann_bc(1))
        >>> T0 = initial_condition(grid)
        >>> T_history, times = solver.solve_multiple_steps(T0, dt=0.01, num_steps=100)
        >>> T_final = T_history[-1]  # Final temperature
        """
        # Allocate storage
        T_history = np.zeros((num_steps + 1, *T_initial.shape))
        T_history[0] = T_initial

        times = np.arange(num_steps + 1) * dt

        # Time stepping
        T_current = T_initial.copy()
        for n in range(num_steps):
            T_current = self.solve_step(T_current, dt)
            T_history[n + 1] = T_current

        return T_history, times

    def get_cfl_number(self, dt: float) -> float:
        """
        Compute CFL number for given timestep.

        CFL = α·dt / (dx_min)²

        For explicit schemes, stability requires CFL < 0.5.
        For implicit schemes (theta > 0), there's no CFL restriction.

        Parameters
        ----------
        dt : float
            Timestep size.

        Returns
        -------
        cfl : float
            CFL number.
        """
        dx_min = min(self.grid.spacing)
        cfl = self.alpha * dt / dx_min**2
        return cfl

    def __repr__(self):
        """String representation."""
        scheme_name = {0.5: "Crank-Nicolson", 1.0: "Backward Euler", 0.0: "Forward Euler"}.get(
            self.theta, f"Theta-method (θ={self.theta})"
        )

        return (
            f"ImplicitHeatSolver(\n"
            f"  scheme={scheme_name},\n"
            f"  alpha={self.alpha},\n"
            f"  grid=({self.grid.dimension}D, {self.grid.num_spatial_points} points),\n"
            f"  bc={self.bc.uniform_segment.bc_type if self.bc.is_uniform else 'mixed'}\n"
            f")"
        )


if __name__ == "__main__":
    """Smoke test for ImplicitHeatSolver."""
    print("Testing ImplicitHeatSolver...")

    from mfgarchon.geometry import TensorProductGrid
    from mfgarchon.geometry.boundary import neumann_bc

    # 1D heat equation test
    grid = TensorProductGrid(bounds=[(0, 1)], Nx=[100])
    bc = neumann_bc(dimension=1)
    solver = ImplicitHeatSolver(grid, alpha=0.01, bc=bc, theta=0.5)

    print(f"\n{solver}")

    # Initial condition: Gaussian pulse
    x = grid.coordinates[0]
    T0 = np.exp(-50 * (x - 0.5) ** 2)

    # Solve with large timestep (CFL = 10)
    dx = grid.spacing[0]
    dt_explicit = 0.2 * dx**2 / solver.alpha  # CFL = 0.2 (explicit stability limit)
    dt_implicit = 10 * dt_explicit  # CFL = 2.0 (would be unstable for explicit)

    print(f"\nExplicit stable dt: {dt_explicit:.6f} (CFL = 0.2)")
    print(f"Using implicit dt: {dt_implicit:.6f} (CFL = {solver.get_cfl_number(dt_implicit):.1f})")

    # Single step
    T1 = solver.solve_step(T0, dt_implicit)

    print("\n✓ Single step completed")
    print(f"  Initial: max={T0.max():.4f}, min={T0.min():.4f}")
    print(f"  After dt: max={T1.max():.4f}, min={T1.min():.4f}")
    print(f"  Energy change: {(np.sum(T1) - np.sum(T0)) / np.sum(T0) * 100:.2f}%")

    # Multiple steps
    T_history, times = solver.solve_multiple_steps(T0, dt_implicit, num_steps=10)

    print(f"\n✓ Multiple steps completed ({len(times)} timesteps)")
    print(f"  Final time: {times[-1]:.4f}")
    print(f"  Final max: {T_history[-1].max():.4f}")

    # Check stability (should remain bounded)
    assert np.all(np.isfinite(T_history)), "Solution contains NaN/Inf!"
    assert T_history[-1].max() <= 1.5 * T0.max(), "Solution exploded (unstable)!"

    print("\n✓ All smoke tests passed!")
