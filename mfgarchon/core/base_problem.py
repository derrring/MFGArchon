#!/usr/bin/env python3
"""
Base problem infrastructure for dimension-agnostic MFG problems.

This module defines the core protocol that ALL MFG problems must implement,
regardless of dimension (1D, 2D, 3D, nD) or domain type (Cartesian grid,
network, manifold, implicit domain, etc.).

Mathematical Notation:
    - m(t,x): Density function
    - u(t,x): Value function
    - ∂u/∂x: Spatial gradient
    - H(x, m, p, t): Hamiltonian
    - g(x): Terminal cost
    - f(x, m, t): Running cost

Part of: Issue #245 - Incremental Evolution toward Unified nD Architecture
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray


@runtime_checkable
class MFGProblemProtocol(Protocol):
    """
    Minimal protocol that ALL MFG problems must satisfy.

    This protocol is intentionally geometry-agnostic, working with:
    - Cartesian grids (TensorProductGrid, Mesh1D, Mesh2D, Mesh3D)
    - Networks (NetworkMFGProblem)
    - Unstructured meshes (AMR geometries)
    - Implicit domains (level set, SDF)
    - Custom geometries

    Grid-specific properties (grid_shape, grid_spacing, etc.) are NOT
    required here. See CartesianGridMFGProtocol for grid-specific interface.

    Universal Properties:
        dimension: int | str
            Spatial dimension (int for grids/meshes, "network" for graphs)
        T: float
            Final time
        Nt: int
            Number of time steps
        tSpace: NDArray
            Time points array [t₀, t₁, ..., t_Nt]
        sigma: float | Callable
            Diffusion coefficient σ

    MFG Components:
        All problems must provide:
        - hamiltonian(x, m, p, t): H(x, m, p, t)
        - terminal_cost(x): g(x)
        - initial_density(x): m₀(x)
        - running_cost(x, m, t): f(x, m, t)

    Examples:
        >>> # Geometry-agnostic solver
        >>> def solve_hjb(problem: MFGProblemProtocol) -> NDArray:
        ...     # Works for grids, networks, meshes, etc.
        ...     T = problem.T
        ...     tSpace = problem.tSpace
        ...     sigma = problem.sigma
        ...     # ... solver code ...
        ...     return u

        >>> # Runtime validation
        >>> from mfgarchon.geometry import TensorProductGrid
        >>> grid = TensorProductGrid(bounds=[(0, 1)], Nx_points=[101])
        >>> problem = MFGProblem(geometry=grid, T=1, Nt=50, sigma=0.1)
        >>> assert isinstance(problem, MFGProblemProtocol)  # Should pass!
    """

    # ====================
    # Spatial (minimal)
    # ====================

    dimension: int | str  # int for grids/meshes, "network" for graphs

    # ====================
    # Temporal (universal)
    # ====================

    T: float  # Final time
    Nt: int  # Number of time steps
    tSpace: NDArray  # Time points [t₀, t₁, ..., t_Nt]

    # ====================
    # Physical (universal)
    # ====================

    sigma: float | Callable  # Diffusion coefficient

    # ====================
    # MFG Components
    # ====================

    def hamiltonian(self, x, m, p, t) -> float:
        """
        Hamiltonian H(x, m, p, t).

        Args:
            x: Spatial position
                - 1D: float
                - nD: tuple/array of length d
            m: Density value m(t,x) at this position
            p: Momentum/co-state ∂u/∂x
                - 1D: float
                - nD: tuple/array of length d
            t: Time

        Returns:
            Hamiltonian value H(x, m, p, t)

        Example:
            >>> # Quadratic Hamiltonian with congestion
            >>> def hamiltonian(self, x, m, p, t):
            ...     p_arr = np.array(p) if hasattr(p, '__iter__') else p
            ...     return 0.5 * np.sum(p_arr**2) + 0.1 * m
        """
        ...

    def terminal_cost(self, x) -> float:
        """
        Terminal cost g(x).

        Args:
            x: Spatial position

        Returns:
            Terminal cost value g(x)

        Example:
            >>> # Quadratic terminal cost
            >>> def terminal_cost(self, x):
            ...     x_arr = np.array(x) if hasattr(x, '__iter__') else np.array([x])
            ...     return 0.5 * np.sum((x_arr - 0.5)**2)
        """
        ...

    def initial_density(self, x) -> float:
        """
        Initial density m₀(x).

        Args:
            x: Spatial position

        Returns:
            Initial density value m₀(x) ≥ 0

        Example:
            >>> # Gaussian initial density
            >>> def initial_density(self, x):
            ...     x_arr = np.array(x) if hasattr(x, '__iter__') else np.array([x])
            ...     return np.exp(-10 * np.sum((x_arr - 0.5)**2))
        """
        ...

    def running_cost(self, x, m, t) -> float:
        """
        Running cost f(x, m, t).

        Args:
            x: Spatial position
            m: Density value m(t,x)
            t: Time

        Returns:
            Running cost value f(x, m, t)

        Example:
            >>> # Congestion cost
            >>> def running_cost(self, x, m, t):
            ...     return 0.1 * m  # Penalize high density
        """
        ...


@runtime_checkable
class CartesianGridMFGProtocol(MFGProblemProtocol, Protocol):
    """
    Extended protocol for Cartesian grid-based MFG problems.

    Adds grid-specific properties required by finite difference, WENO,
    and other structured grid solvers.

    Only applies to problems with GeometryType.CARTESIAN_GRID:
    - TensorProductGrid (all dimensions)

    Does NOT apply to:
    - Networks (NetworkMFGProblem)
    - Unstructured meshes (AMR geometries)
    - Implicit domains

    Additional Grid Properties:
        dimension: int
            Must be integer (not "network")
        spatial_bounds: list[tuple[float, float]]
            [(x₀_min, x₀_max), (x₁_min, x₁_max), ...]
        spatial_discretization: list[int]
            [N₀, N₁, ...] grid points per dimension
        xSpace: NDArray | list[NDArray]
            Coordinate arrays

    Grid-Specific Computed Properties:
        grid_shape: tuple[int, ...]
            Shape of grid (N₀, N₁, ...)
        grid_spacing: list[float]
            Spacing [Δx₀, Δx₁, ...] per dimension

    Examples:
        >>> # Grid-specific solver (FDM)
        >>> def solve_hjb_fdm(problem: CartesianGridMFGProtocol) -> NDArray:
        ...     dx = problem.grid_spacing  # Can safely assume regular grid
        ...     shape = problem.grid_shape
        ...     # ... FDM implementation ...
        ...     return u

        >>> # Runtime check
        >>> if isinstance(problem, CartesianGridMFGProtocol):
        ...     return solve_hjb_fdm(problem)  # Use FDM
        ... else:
        ...     return solve_hjb_particle(problem)  # Use particles
    """

    # ====================
    # Spatial (grid-specific)
    # ====================

    dimension: int  # Must be int, not "network"
    spatial_bounds: list[tuple[float, float]]  # [(x₀_min, x₀_max), ...]
    spatial_discretization: list[int]  # [N₀, N₁, ...]
    xSpace: NDArray | list[NDArray]  # Coordinate arrays

    # ====================
    # Grid Properties (computed)
    # ====================

    @property
    def grid_shape(self) -> tuple[int, ...]:
        """
        Shape of spatial grid (N₀, N₁, ...).

        Returns tuple for dimension-agnostic grid operations.

        Example:
            >>> problem.grid_shape  # (50, 50) for 2D grid
        """
        ...

    @property
    def grid_spacing(self) -> list[float]:
        """
        Grid spacing [Δx₀, Δx₁, ...] for each dimension.

        Computed from bounds and discretization:
        Δx_i = (xmax_i - xmin_i) / N_i

        Example:
            >>> problem.grid_spacing  # [0.02, 0.02] for 2D grid
        """
        ...
