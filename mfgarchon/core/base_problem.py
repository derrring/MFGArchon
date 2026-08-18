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

        Note:
            On the grid and mesh interface ``m`` is the density AT ``x``, a scalar -- not the
            measure. (``NetworkMFGProblem.hamiltonian`` takes a different, 5-argument form and
            does receive the full nodal array.) A coupling needing the whole density, such as a
            convolution ``(k * m)(x)``, cannot be written here; route it through
            ``source_term_hjb(x, m_t, v_t, t)``, which receives the full spatial array at the
            time slice. Two caveats worth knowing before you do:

            - Only the FDM HJB solvers accept a ``source_term``. Setting ``source_term_hjb``
              with any of the others raises rather than dropping it silently.
            - The source is bound to the PREVIOUS Picard iterate and the FP drift is derived
              from the Hamiltonian alone, never consulting it. A coupling with no ``p`` in it --
              the usual nonlocal case ``F(x, m)`` -- is fine there and is the canonical
              Lasry-Lions system. A ``p``-DEPENDENT coupling routed through the source is not:
              it changes the effective Hamiltonian's minimiser while the FP equation keeps
              advecting with the original ``-D_p H``, silently breaking the HJB/FP adjoint pair.

        Example:
            >>> # Separable Hamiltonian with a LOCAL coupling: H_0(p) plus a term in m alone.
            >>> def hamiltonian(self, x, m, p, t):
            ...     p_arr = np.array(p) if hasattr(p, '__iter__') else p
            ...     return 0.5 * np.sum(p_arr**2) - 0.1 * m

            This is NOT congestion, which this example previously claimed to be. Congestion means
            moving is costlier where the crowd is dense, so the density scales the MOMENTUM term:
            ``H(x, p, m) = (m**alpha / gamma) * (|p| / m**alpha)**gamma``, for ``1 <= gamma < 2``
            and ``0 < alpha < 1`` (Gomes, Pimentel & Voskanyan, *Regularity Theory for Mean-Field
            Game Systems*, p. 116). The quadratic specialisation ``0.5*|p|**2 / m**alpha`` is the
            shape this library implements; note it sits at ``gamma = 2``, outside the hypotheses of
            the section cited. Above, ``m`` appears in a term with no ``p`` in it at all.

            For congestion see :class:`~mfgarchon.core.hamiltonian.CongestionHamiltonian`, which is
            the FAMILY containing that form -- ``H = |p|**2 / (2*lam*c(m)) + V + f(m)`` with a
            user-supplied ``c(m)``. Pass ``congestion_factor=lambda m: m**alpha`` to obtain the
            form above, and ``congestion_factor_dm`` if you want ``dH/dm`` analytic rather than by
            finite difference.

            The sign is ``-0.1 * m`` rather than ``+0.1 * m`` so the example is Lasry--Lions
            monotone under the convention pairing ``H`` with ``-u_t + H = 0``. With ``+``, the
            coupling is crowd-ATTRACTING and lands in the known non-uniqueness regime -- fine for a
            manufactured verification instance, where the computed solution is compared against an
            imposed exact one, but a poor default for an example a reader copies.
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
