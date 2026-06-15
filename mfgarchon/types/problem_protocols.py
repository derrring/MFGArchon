"""
Problem interface protocols for MFG solvers.

This module defines explicit protocol interfaces that different MFG solvers expect.
Previously these interfaces were implicit, leading to confusing errors when problems
were used with incompatible solvers.

Protocols defined:
- CollocationProblem: For meshfree collocation solvers (HJB-GFDM)
- GridProblem: For grid-based finite difference solvers
- DirectAccessProblem: For solvers that access attributes directly

Usage:
    from mfgarchon.types.problem_protocols import CollocationProblem

    # Runtime validation
    if not isinstance(problem, CollocationProblem):
        raise TypeError("Solver requires CollocationProblem interface")

    # Type hints
    def my_solver(problem: CollocationProblem) -> SolverResult:
        ...
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from numpy.typing import NDArray


@runtime_checkable
class CollocationProblem(Protocol):
    """
    Problem interface for meshfree collocation solvers.

    This interface is used by solvers that operate on arbitrary point clouds
    without grid structure, such as the Generalized Finite Difference Method (GFDM).

    Required attributes:
        d (int): Spatial dimension (1, 2, 3, ...)

    Required methods:
        H(x, p, m): Hamiltonian function
        sigma(x): Diffusion coefficient function

    Key characteristics:
        - Operates on scattered collocation points (no grid structure)
        - Position-dependent diffusion supported via callable sigma(x)
        - Continuous position coordinates (not indices)
        - Momentum as gradient vector (not upwind dictionary)

    Example:
        >>> class MyCollocationProblem:
        ...     def __init__(self):
        ...         self.d = 2  # 2D problem
        ...
        ...     def H(self, x, p, m):
        ...         return 0.5 * np.sum(p**2) + m
        ...
        ...     def sigma(self, x):
        ...         return 0.1  # Constant diffusion

    See Also:
        - HJBGFDMSolver: Uses this interface
        - GridProblem: Alternative interface for grid-based solvers
    """

    d: int  # Spatial dimension

    def H(self, x: NDArray, p: NDArray, m: float) -> float:
        """
        Hamiltonian function H(x, p, m).

        Args:
            x: Position (d-dimensional array, e.g., [2.5, 3.7] for 2D)
            p: Momentum/gradient (d-dimensional array, e.g., [1.0, -0.5])
            m: Density at position x (scalar)

        Returns:
            Hamiltonian value H(x, p, m)

        Example:
            >>> H_val = problem.H(
            ...     x=np.array([2.5, 3.7]),
            ...     p=np.array([1.0, -0.5]),
            ...     m=0.01
            ... )
        """
        ...

    def sigma(self, x: NDArray) -> float:
        """
        Diffusion coefficient at position x.

        This must be a callable function to support position-dependent diffusion.
        For constant diffusion, simply return a constant value.

        Args:
            x: Position (d-dimensional array)

        Returns:
            Diffusion coefficient σ(x) > 0

        Example:
            >>> sigma_val = problem.sigma(np.array([2.5, 3.7]))
            >>> # For constant diffusion: return 0.1
            >>> # For position-dependent: return 0.1 + 0.01 * np.linalg.norm(x)
        """
        ...


@runtime_checkable
class GridProblem(Protocol):
    """
    Problem interface for grid-based finite difference solvers.

    This interface is used by solvers that operate on structured spatial grids,
    such as standard finite difference methods and semi-Lagrangian schemes.

    Required attributes:
        xmin, xmax, Nx, Dx: Spatial grid parameters
        T, Nt, Dt: Time discretization parameters
        xSpace, tSpace: Grid point arrays
        sigma: Diffusion coefficient (float attribute, not callable!)
        coupling_coefficient: Control cost coefficient

    Required methods:
        H(x_idx, m_at_x, p_values, t_idx): Hamiltonian function

    Key characteristics:
        - Operates on structured grids (1D, potentially 2D/3D)
        - Constant diffusion coefficient (attribute, not function)
        - Index-based access (x_idx instead of continuous position)
        - Upwind momentum representation (dict with "forward"/"backward")

    Example:
        >>> class MyGridProblem:
        ...     def __init__(self):
        ...         self.xmin, self.xmax, self.Nx = 0.0, 1.0, 50
        ...         self.dx = (self.xmax - self.xmin) / self.Nx
        ...         self.xSpace = np.linspace(self.xmin, self.xmax, self.Nx + 1)
        ...         self.sigma = 0.1  # Float attribute
        ...         self.coupling_coefficient = 0.5
        ...
        ...     def H(self, x_idx, m_at_x, p_values, t_idx):
        ...         p = p_values['forward']
        ...         return 0.5 * self.coupling_coefficient * p**2

    See Also:
        - BaseHJBSolver: Uses this interface
        - HJBSemiLagrangian: Uses this interface
        - CollocationProblem: Alternative interface for meshfree methods
    """

    # Temporal structure
    T: float
    Nt: int
    dt: float  # Lowercase (official naming convention)
    tSpace: NDArray

    # Physical parameters
    sigma: float  # Constant volatility (attribute, NOT callable!)
    coupling_coefficient: float  # Control cost coefficient

    def H(
        self,
        x_idx: int,
        m_at_x: float,
        derivs: dict[tuple, float] | None = None,
        p_values: dict[str, float] | None = None,
        t_idx: int | None = None,
    ) -> float:
        """
        Hamiltonian function H(x_idx, m, p, t).

        Supports both tuple notation (derivs) and legacy string-key (p_values) formats.

        Args:
            x_idx: Grid index (0 to Nx)
            m_at_x: Density at grid point x_idx
            derivs: Derivatives in tuple notation (NEW, preferred):
                   - 1D: {(0,): u, (1,): du/dx}
                   - 2D: {(0,0): u, (1,0): du/dx, (0,1): du/dy}
            p_values: Momentum dictionary (LEGACY, deprecated):
                     - "forward": Forward finite difference
                     - "backward": Backward finite difference
            t_idx: Time index (optional, for time-dependent Hamiltonians)

        Returns:
            Hamiltonian value H(x_idx, m, p, t)

        Example (new style with derivs):
            >>> derivs = {(0,): 1.0, (1,): 0.5}
            >>> H_val = problem.H(
            ...     x_idx=25,
            ...     m_at_x=0.01,
            ...     derivs=derivs,
            ...     t_idx=10
            ... )

        Example (legacy style with p_values):
            >>> p_values = {"forward": 1.5, "backward": 1.3}
            >>> H_val = problem.H(
            ...     x_idx=25,
            ...     m_at_x=0.01,
            ...     p_values=p_values,  # Deprecated
            ...     t_idx=10
            ... )

        Note:
            - Provide EITHER derivs OR p_values, not both
            - p_values is deprecated and will be removed in v1.0.0
            - Upwind schemes: p_values["forward"] for positive direction,
              p_values["backward"] for negative direction
        """
        ...


@runtime_checkable
class DirectAccessProblem(Protocol):
    """
    Problem interface for solvers that access attributes directly.

    This interface is used by solvers that compute Hamiltonians internally
    and only need access to problem parameters as attributes.

    Required attributes:
        sigma: Diffusion coefficient (float)
        coupling_coefficient: Control cost coefficient (float)
        Dx, Dt: Spatial and temporal discretization
        xSpace: Spatial grid points

    No H() method required - solvers compute Hamiltonian themselves.

    Key characteristics:
        - Simplified interface (no method calls)
        - Direct attribute access only
        - Used by FP solvers, WENO, variational methods

    Example:
        >>> class MyDirectAccessProblem:
        ...     def __init__(self):
        ...         self.sigma = 0.1
        ...         self.coupling_coefficient = 0.5
        ...         self.dx = 0.02
        ...         self.dt = 0.01
        ...         self.xSpace = np.linspace(0, 1, 51)

    See Also:
        - FPParticleSolver: Uses this interface
        - FPFDMSolver: Uses this interface
        - HJBWENOSolver: Partially uses this interface
    """

    sigma: float
    coupling_coefficient: float
    dt: float  # Lowercase (official naming convention)


# Type aliases for convenience
MFGProblemType = CollocationProblem | GridProblem | DirectAccessProblem
