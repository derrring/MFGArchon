"""
Unified problem factory supporting all MFG problem types.

This module provides factory functions for creating MFG problems with dual-output support:
- New unified MFGProblem (default, recommended)
- Old specialized classes (backward compatibility)

Supported Problem Types
-----------------------
- Standard HJB-FP (Hamiltonian-based)
- Network/Graph MFG (discrete domains)
- Variational/Lagrangian (optimization-based)
- Stochastic MFG (common noise)
- High-Dimensional MFG (n-D spatial domains)

IMPORTANT: Function Signature Requirements
-------------------------------------------
Hamiltonian and related functions must use the MFGProblem signature format:

def hamiltonian_func(x_idx, x_position, m_at_x, derivs, t_idx, current_time, problem):
    '''
    Args:
        x_idx: Grid index
        x_position: Physical position
        m_at_x: Density at position
        derivs: Dictionary with tuple keys:
            - (0,): u(x,t) function value
            - (1,): ∂u/∂x first derivative
        t_idx: Time index
        current_time: Physical time
        problem: MFGProblem reference
    '''
    du_dx = derivs.get((1,), 0.0)
    return 0.5 * du_dx**2 + m_at_x

See examples/basic/custom_hamiltonian_derivs_demo.py for complete examples.

Usage
-----
>>> # For simple examples, use MFGProblem
>>> from mfgarchon import MFGProblem
>>> problem = MFGProblem()
>>>
>>> # For custom problems, see custom_hamiltonian_derivs_demo.py
>>> # Factory functions are provided but require the complex signature above
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Literal, overload

from mfgarchon.core.mfg_problem import MFGComponents, MFGProblem
from mfgarchon.utils.mfg_logging.logger import get_logger

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

    from mfgarchon.geometry import BoundaryConditions
    from mfgarchon.geometry.domain import Domain

logger = get_logger(__name__)

# =============================================================================
# Classic LQ-MFG Default Conditions (Issue #670)
# =============================================================================
# These functions provide the classic Linear-Quadratic MFG setup that was
# previously built-in. Users must now explicitly import and use them.
# See examples/tutorials/01_hello_mfg.py for usage.


def lq_mfg_terminal_cost(Lx: float = 1.0):
    """Classic LQ-MFG terminal cost: g(x) = 5*(cos(2πx/L) + 0.4*sin(4πx/L)).

    Args:
        Lx: Domain length (default 1.0)

    Returns:
        Callable that computes terminal cost at position x

    Example:
        >>> problem = MFGProblem(
        ...     geometry=grid,
        ...     u_terminal=lq_mfg_terminal_cost(Lx=1.0),
        ...     m_initial=lq_mfg_initial_density(),
        ... )
    """
    import numpy as np

    def u_terminal(x: float) -> float:
        return 5 * (np.cos(x * 2 * np.pi / Lx) + 0.4 * np.sin(x * 4 * np.pi / Lx))

    return u_terminal


def lq_mfg_initial_density():
    """Classic LQ-MFG initial density: bimodal Gaussian at x=0.2 and x=0.8.

    Returns:
        Callable that computes initial density at position x

    Note:
        MFGProblem will automatically normalize the density to integrate to 1.

    Example:
        >>> problem = MFGProblem(
        ...     geometry=grid,
        ...     u_terminal=lq_mfg_terminal_cost(),
        ...     m_initial=lq_mfg_initial_density(),
        ... )
    """
    import numpy as np

    def m_initial(x: float) -> float:
        return 2 * np.exp(-200 * (x - 0.2) ** 2) + np.exp(-200 * (x - 0.8) ** 2)

    return m_initial


# =============================================================================
# Problem Type Constants
# =============================================================================

ProblemType = Literal["standard", "network", "variational", "stochastic", "highdim"]

# =============================================================================
# Main Factory Function
# =============================================================================


@overload
def create_mfg_problem(
    problem_type: ProblemType,
    components: MFGComponents,
    *,
    geometry: Domain,
    time_horizon: float = 1.0,
    num_timesteps: int = 100,
    use_unified: Literal[True] = True,
    **kwargs: Any,
) -> MFGProblem: ...


@overload
def create_mfg_problem(
    problem_type: Literal["network"],
    components: MFGComponents,
    *,
    geometry: Domain,
    time_horizon: float = 1.0,
    num_timesteps: int = 100,
    use_unified: Literal[False],
    **kwargs: Any,
) -> Any: ...  # NetworkMFGProblem


@overload
def create_mfg_problem(
    problem_type: Literal["variational"],
    components: MFGComponents,
    *,
    geometry: Domain,
    time_horizon: float = 1.0,
    num_timesteps: int = 100,
    use_unified: Literal[False],
    **kwargs: Any,
) -> Any: ...  # VariationalMFGProblem


@overload
def create_mfg_problem(
    problem_type: Literal["stochastic"],
    components: MFGComponents,
    *,
    geometry: Domain,
    time_horizon: float = 1.0,
    num_timesteps: int = 100,
    use_unified: Literal[False],
    **kwargs: Any,
) -> Any: ...  # StochasticMFGProblem


def create_mfg_problem(
    problem_type: ProblemType,
    components: MFGComponents,
    *,
    geometry: Domain,
    time_horizon: float = 1.0,
    num_timesteps: int = 100,
    use_unified: bool = True,
    **kwargs: Any,
) -> MFGProblem | Any:
    """
    Create MFG problem of specified type.

    This factory function provides a unified interface for creating any MFG problem
    type with support for both the new unified API and legacy specialized classes.

    Parameters
    ----------
    problem_type : {"standard", "network", "variational", "stochastic", "highdim"}
        Type of MFG problem to create
    components : MFGComponents
        Problem components (Hamiltonian, costs, densities, etc.)
    geometry : Domain
        Spatial domain
    time_horizon : float, default=1.0
        Final time T
    num_timesteps : int, default=100
        Number of time discretization points
    use_unified : bool, default=True
        If True, return unified MFGProblem (recommended)
        If False, return specialized class (backward compatibility)
    **kwargs : dict
        Additional problem-specific parameters

    Returns
    -------
    MFGProblem or specialized problem class
        If use_unified=True: MFGProblem instance
        If use_unified=False: Specialized class (NetworkMFGProblem, etc.)

    Examples
    --------
    >>> # Standard MFG problem (new unified API)
    >>> from mfgarchon.core import MFGComponents
    >>> from mfgarchon.factory import create_mfg_problem
    >>>
    >>> components = MFGComponents(
    ...     hamiltonian_func=H,
    ...     hamiltonian_dm_func=dH_dm,
    ...     terminal_cost_func=g,
    ...     m_initial=rho_0
    ... )
    >>> problem = create_mfg_problem("standard", components, geometry=domain)
    >>>
    >>> # Network MFG (new unified API)
    >>> components = MFGComponents(
    ...     geometry=graph,
    ...     node_interaction_func=node_cost,
    ...     edge_cost_func=edge_cost
    ... )
    >>> problem = create_mfg_problem("network", components, geometry=domain)
    >>>
    >>> # Backward compatibility
    >>> problem = create_mfg_problem("network", components, geometry=domain, use_unified=False)
    """
    # Set problem type in components
    if components.problem_type == "mfg":
        components.problem_type = problem_type

    if use_unified:
        # New unified API - create MFGProblem directly.
        # MFGProblem's constructor takes T / Nt; passing time_horizon /
        # num_timesteps would fall into **kwargs and be silently dropped
        # (T, Nt then default to 1.0 / 51). Map the factory's names.
        return MFGProblem(
            geometry=geometry,
            T=time_horizon,
            Nt=num_timesteps,
            components=components,
            **kwargs,
        )
    else:
        # Legacy API - create specialized classes
        warnings.warn(
            f"Specialized {problem_type} problem classes are deprecated. "
            f"Use MFGProblem with use_unified=True (default).",
            DeprecationWarning,
            stacklevel=2,
        )

        if problem_type == "network":
            from mfgarchon.extensions.topology import NetworkMFGProblem

            return NetworkMFGProblem(
                geometry=geometry,
                T=time_horizon,
                Nt=num_timesteps,
                components=components,
                **kwargs,
            )
        elif problem_type == "variational":
            from mfgarchon.solvers.variational import VariationalMFGProblem

            return VariationalMFGProblem(
                geometry=geometry,
                T=time_horizon,
                Nt=num_timesteps,
                components=components,
                **kwargs,
            )
        elif problem_type == "stochastic":
            from mfgarchon.core.stochastic.stochastic_problem import StochasticMFGProblem

            return StochasticMFGProblem(
                geometry=geometry,
                T=time_horizon,
                Nt=num_timesteps,
                components=components,
                **kwargs,
            )
        elif problem_type == "highdim":
            # GridBasedMFGProblem was removed in v0.14.0
            # Use MFGProblem with spatial_bounds and spatial_discretization instead
            return MFGProblem(
                geometry=geometry,
                T=time_horizon,
                Nt=num_timesteps,
                components=components,
                **kwargs,
            )
        else:
            # Standard MFG
            return MFGProblem(
                geometry=geometry,
                T=time_horizon,
                Nt=num_timesteps,
                components=components,
                **kwargs,
            )


# =============================================================================
# Convenience Factory Functions
# =============================================================================


def create_standard_problem(
    hamiltonian: Any,  # HamiltonianBase
    terminal_cost: Callable,
    initial_density: Callable,
    geometry: Domain,
    *,
    potential: Callable | None = None,
    boundary_conditions: BoundaryConditions | None = None,
    time_horizon: float = 1.0,
    num_timesteps: int = 100,
    use_unified: bool = True,
    **kwargs: Any,
) -> MFGProblem | Any:
    """
    Create standard HJB-FP MFG problem.

    Issue #673: Class-based Hamiltonian API.

    Parameters
    ----------
    hamiltonian : HamiltonianBase
        Class-based Hamiltonian (SeparableHamiltonian, QuadraticMFGHamiltonian, etc.)
    terminal_cost : Callable
        Terminal cost g(x)
    initial_density : Callable
        Initial density m_0(x)
    geometry : Domain
        Spatial domain
    potential : Callable, optional
        Additional potential V(x) if not in Hamiltonian
    boundary_conditions : BoundaryConditions, optional
        Boundary conditions
    time_horizon : float, default=1.0
        Final time T
    num_timesteps : int, default=100
        Number of time steps
    use_unified : bool, default=True
        Use unified MFGProblem (True) or specialized class (False)

    Returns
    -------
    MFGProblem
        Problem instance

    Examples
    --------
    >>> from mfgarchon.factory import create_standard_problem
    >>> from mfgarchon.core.hamiltonian import SeparableHamiltonian, QuadraticControlCost
    >>>
    >>> H = SeparableHamiltonian(
    ...     control_cost=QuadraticControlCost(control_cost=1.0),
    ...     coupling=lambda m: m,
    ...     coupling_dm=lambda m: 1.0,
    ... )
    >>> problem = create_standard_problem(
    ...     hamiltonian=H,
    ...     terminal_cost=lambda x: x**2,
    ...     initial_density=lambda x: np.exp(-x**2),
    ...     geometry=domain
    ... )
    """
    components = MFGComponents(
        hamiltonian=hamiltonian,
        u_terminal=terminal_cost,
        m_initial=initial_density,
        potential_func=potential,
        boundary_conditions=boundary_conditions,
        problem_type="standard",
    )

    return create_mfg_problem(
        "standard",
        components,
        geometry=geometry,
        time_horizon=time_horizon,
        num_timesteps=num_timesteps,
        use_unified=use_unified,
        **kwargs,
    )


def create_network_problem(
    network_geometry: Any,
    node_interaction: Callable,
    edge_cost: Callable,
    initial_density: Callable | NDArray,
    geometry: Domain,
    *,
    edge_interaction: Callable | None = None,
    terminal_cost: Callable | None = None,
    time_horizon: float = 1.0,
    num_timesteps: int = 100,
    use_unified: bool = True,
    **kwargs: Any,
) -> MFGProblem | Any:
    """
    Create network/graph MFG problem.

    Issue #673: Network problems require specialized NetworkMFGProblem class.

    For network MFG problems, use the specialized class directly:

    >>> from mfgarchon.extensions.topology import NetworkMFGProblem
    >>> problem = NetworkMFGProblem(
    ...     geometry=graph,
    ...     node_interaction=lambda node, m: m[node]**2,
    ...     edge_cost=lambda edge: 1.0,
    ...     initial_density=initial_m,
    ...     geometry=network_domain
    ... )

    Parameters
    ----------
    network_geometry : NetworkGeometry or graph object
        Network/graph structure
    node_interaction : Callable
        Node interaction cost function
    edge_cost : Callable
        Edge traversal cost function
    initial_density : Callable or array
        Initial density on nodes
    geometry : Domain
        Domain (typically NetworkDomain)

    Raises
    ------
    NotImplementedError
        Network problems require specialized extension module.
    """
    raise NotImplementedError(
        "Network MFG problems require the specialized NetworkMFGProblem class.\n\n"
        "Use the extension module directly:\n"
        "  from mfgarchon.extensions.topology import NetworkMFGProblem\n\n"
        "  problem = NetworkMFGProblem(\n"
        "      geometry=graph,\n"
        "      node_interaction=lambda node, m: m[node]**2,\n"
        "      edge_cost=lambda edge: 1.0,\n"
        "      initial_density=initial_m,\n"
        "      geometry=network_domain\n"
        "  )"
    )


def create_variational_problem(
    lagrangian: Any,  # LagrangianBase
    terminal_cost: Callable,
    initial_density: Callable,
    geometry: Domain,
    *,
    boundary_conditions: BoundaryConditions | None = None,
    time_horizon: float = 1.0,
    num_timesteps: int = 100,
    use_unified: bool = True,
    **kwargs: Any,
) -> MFGProblem | Any:
    """
    Create variational/Lagrangian MFG problem.

    Issue #673: Uses class-based LagrangianBase API.

    The Lagrangian is auto-converted to Hamiltonian via Legendre transform.

    Parameters
    ----------
    lagrangian : LagrangianBase
        Class-based Lagrangian L(x, alpha, m, t).
        Must be a LagrangianBase subclass with legendre_transform() method.
    terminal_cost : Callable
        Terminal cost Ψ(x)
    initial_density : Callable
        Initial density ρ₀(x)
    geometry : Domain
        Spatial domain
    boundary_conditions : BoundaryConditions, optional
        Boundary conditions
    time_horizon : float, default=1.0
        Final time T
    num_timesteps : int, default=100
        Number of time steps
    use_unified : bool, default=True
        Use unified MFGProblem (recommended)

    Returns
    -------
    MFGProblem
        Problem instance

    Examples
    --------
    >>> from mfgarchon.factory import create_variational_problem
    >>> from mfgarchon.core.hamiltonian import LagrangianBase
    >>>
    >>> class QuadraticLagrangian(LagrangianBase):
    ...     def __call__(self, x, alpha, m, t=0.0):
    ...         return 0.5 * np.sum(alpha**2) + m
    ...
    ...     def dalpha(self, x, alpha, m, t=0.0):
    ...         return alpha
    ...
    ...     def dm(self, x, alpha, m, t=0.0):
    ...         return 1.0
    >>>
    >>> problem = create_variational_problem(
    ...     lagrangian=QuadraticLagrangian(),
    ...     terminal_cost=lambda x: x**2,
    ...     initial_density=lambda x: np.exp(-x**2),
    ...     geometry=domain
    ... )
    """
    from mfgarchon.core.hamiltonian import LagrangianBase

    if not isinstance(lagrangian, LagrangianBase):
        raise TypeError(
            f"lagrangian must be a LagrangianBase instance, got {type(lagrangian).__name__}.\n\n"
            "Create a LagrangianBase subclass:\n"
            "  class MyLagrangian(LagrangianBase):\n"
            "      def __call__(self, x, alpha, m, t=0.0):\n"
            "          return 0.5 * np.sum(alpha**2) + m\n\n"
            "      def dalpha(self, x, alpha, m, t=0.0):\n"
            "          return alpha\n\n"
            "      def dm(self, x, alpha, m, t=0.0):\n"
            "          return 1.0"
        )

    # MFGComponents auto-converts Lagrangian to Hamiltonian
    components = MFGComponents(
        lagrangian=lagrangian,
        u_terminal=terminal_cost,
        m_initial=initial_density,
        boundary_conditions=boundary_conditions,
        problem_type="variational",
    )

    return create_mfg_problem(
        "variational",
        components,
        geometry=geometry,
        time_horizon=time_horizon,
        num_timesteps=num_timesteps,
        use_unified=use_unified,
        **kwargs,
    )


def create_stochastic_problem(
    hamiltonian: Any,  # HamiltonianBase
    terminal_cost: Callable,
    initial_density: Callable,
    geometry: Domain,
    *,
    noise_intensity: float = 0.0,
    common_noise: Callable | None = None,
    idiosyncratic_noise: Callable | None = None,
    correlation_matrix: NDArray | None = None,
    potential: Callable | None = None,
    time_horizon: float = 1.0,
    num_timesteps: int = 100,
    use_unified: bool = True,
    **kwargs: Any,
) -> MFGProblem | Any:
    """
    Create stochastic MFG problem with common noise.

    Issue #673: Class-based Hamiltonian API.

    Parameters
    ----------
    hamiltonian : HamiltonianBase
        Class-based Hamiltonian
    terminal_cost : Callable
        Terminal cost g(x, ξ)
    initial_density : Callable
        Initial density m_0(x)
    geometry : Domain
        Spatial domain
    noise_intensity : float, default=0.0
        Noise intensity σ
    common_noise : Callable, optional
        Common noise function ξ(t)
    idiosyncratic_noise : Callable, optional
        Idiosyncratic noise function
    correlation_matrix : ndarray, optional
        Noise correlation matrix
    potential : Callable, optional
        Additional potential V(x)
    time_horizon : float, default=1.0
        Final time T
    num_timesteps : int, default=100
        Number of time steps
    use_unified : bool, default=True
        Use unified MFGProblem (True) or StochasticMFGProblem (False)

    Returns
    -------
    MFGProblem or StochasticMFGProblem
        Problem instance
    """
    # Store stochastic parameters in the parameters dict
    stochastic_params = {
        "noise_intensity": noise_intensity,
        "common_noise_func": common_noise,
        "idiosyncratic_noise_func": idiosyncratic_noise,
        "correlation_matrix": correlation_matrix,
    }

    components = MFGComponents(
        hamiltonian=hamiltonian,
        u_terminal=terminal_cost,
        m_initial=initial_density,
        potential_func=potential,
        parameters=stochastic_params,
        problem_type="stochastic",
    )

    return create_mfg_problem(
        "stochastic",
        components,
        geometry=geometry,
        time_horizon=time_horizon,
        num_timesteps=num_timesteps,
        use_unified=use_unified,
        **kwargs,
    )


def create_highdim_problem(
    hamiltonian: Any,  # HamiltonianBase
    terminal_cost: Callable,
    initial_density: Callable,
    geometry: Domain,
    *,
    dimension: int,
    potential: Callable | None = None,
    boundary_conditions: BoundaryConditions | None = None,
    time_horizon: float = 1.0,
    num_timesteps: int = 100,
    use_unified: bool = True,
    **kwargs: Any,
) -> MFGProblem | Any:
    """
    Create high-dimensional MFG problem (d > 3).

    Issue #673: Class-based Hamiltonian API.

    Parameters
    ----------
    hamiltonian : HamiltonianBase
        Class-based Hamiltonian
    terminal_cost : Callable
        Terminal cost g(x)
    initial_density : Callable
        Initial density m_0(x)
    geometry : Domain
        High-dimensional domain
    dimension : int
        Spatial dimension d
    potential : Callable, optional
        Additional potential V(x)
    boundary_conditions : BoundaryConditions, optional
        Boundary conditions
    time_horizon : float, default=1.0
        Final time T
    num_timesteps : int, default=100
        Number of time steps
    use_unified : bool, default=True
        Use unified MFGProblem (recommended)

    Returns
    -------
    MFGProblem
        Problem instance
    """
    components = MFGComponents(
        hamiltonian=hamiltonian,
        u_terminal=terminal_cost,
        m_initial=initial_density,
        potential_func=potential,
        boundary_conditions=boundary_conditions,
        problem_type="highdim",
    )

    return create_mfg_problem(
        "highdim",
        components,
        geometry=geometry,
        time_horizon=time_horizon,
        num_timesteps=num_timesteps,
        use_unified=use_unified,
        dimension=dimension,
        **kwargs,
    )


# =============================================================================
# Legacy Compatibility Functions
# =============================================================================


def create_lq_problem(
    *,
    geometry: Domain,
    terminal_cost: Callable,
    initial_density: Callable,
    running_cost_control: float = 1.0,
    running_cost_congestion: float = 1.0,
    time_horizon: float = 1.0,
    num_timesteps: int = 100,
    use_unified: bool = True,
    **kwargs: Any,
) -> MFGProblem | Any:
    """
    Create Linear-Quadratic (LQ) MFG problem.

    Standard LQ-MFG with quadratic control cost and linear congestion.
    Uses Hamiltonian: H(x, p, m) = (α/2)|p|² + βm

    Parameters
    ----------
    geometry : Domain
        Spatial domain
    terminal_cost : Callable
        Terminal cost g(x)
    initial_density : Callable
        Initial density ρ₀(x)
    running_cost_control : float, default=1.0
        Control cost coefficient α
    running_cost_congestion : float, default=1.0
        Congestion cost coefficient β
    time_horizon : float, default=1.0
        Final time T
    num_timesteps : int, default=100
        Number of time steps
    use_unified : bool, default=True
        Use unified MFGProblem (True) or LQMFGProblem (False)

    Returns
    -------
    MFGProblem or LQMFGProblem
        Problem instance

    Examples
    --------
    >>> from mfgarchon.factory import create_lq_problem
    >>>
    >>> problem = create_lq_problem(
    ...     geometry=domain,
    ...     terminal_cost=lambda x: x**2,
    ...     initial_density=lambda x: np.exp(-x**2),
    ...     running_cost_control=0.5,
    ...     running_cost_congestion=1.0
    ... )
    """
    # Issue #673: Use class-based Hamiltonian
    from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian

    # LQ Hamiltonian: H = (α/2)|p|² + βm
    hamiltonian = SeparableHamiltonian(
        control_cost=QuadraticControlCost(control_cost=1.0 / running_cost_control),
        coupling=lambda m: running_cost_congestion * m,
        coupling_dm=lambda m: running_cost_congestion,
    )

    return create_standard_problem(
        hamiltonian=hamiltonian,
        terminal_cost=terminal_cost,
        initial_density=initial_density,
        geometry=geometry,
        time_horizon=time_horizon,
        num_timesteps=num_timesteps,
        use_unified=use_unified,
        **kwargs,
    )


def create_crowd_problem(
    *,
    geometry: Domain,
    target_location: NDArray | Callable,
    initial_density: Callable,
    running_cost_control: float = 1.0,
    congestion_sensitivity: float = 1.0,
    time_horizon: float = 1.0,
    num_timesteps: int = 100,
    use_unified: bool = True,
    **kwargs: Any,
) -> MFGProblem | Any:
    """
    Create crowd dynamics MFG problem.

    Models crowd motion toward target with congestion effects.
    Uses potential V(x) = |x - x_target|²/2

    Parameters
    ----------
    geometry : Domain
        Spatial domain
    target_location : array or Callable
        Target location x_target or function
    initial_density : Callable
        Initial crowd density ρ₀(x)
    running_cost_control : float, default=1.0
        Control cost coefficient
    congestion_sensitivity : float, default=1.0
        Congestion sensitivity
    time_horizon : float, default=1.0
        Final time T
    num_timesteps : int, default=100
        Number of time steps
    use_unified : bool, default=True
        Use unified MFGProblem (True) or CrowdDynamicsProblem (False)

    Returns
    -------
    MFGProblem or CrowdDynamicsProblem
        Problem instance

    Examples
    --------
    >>> from mfgarchon.factory import create_crowd_problem
    >>> import numpy as np
    >>>
    >>> problem = create_crowd_problem(
    ...     geometry=domain,
    ...     target_location=np.array([1.0, 1.0]),
    ...     initial_density=lambda x: np.exp(-np.linalg.norm(x)**2)
    ... )
    """
    import numpy as np

    from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian

    # Create potential function
    if callable(target_location):
        target_func = target_location
    else:
        target = np.asarray(target_location)
        target_func = lambda x, t=0.0: 0.5 * np.sum((np.atleast_1d(x) - target) ** 2)  # noqa: E731

    def terminal_cost(x: Any) -> float:
        """Terminal cost: distance to target"""
        return target_func(x)

    # Issue #673: Use class-based Hamiltonian
    # Crowd Hamiltonian: H = (α/2)|p|² + V(x) + βm
    hamiltonian = SeparableHamiltonian(
        control_cost=QuadraticControlCost(control_cost=1.0 / running_cost_control),
        potential=target_func,
        coupling=lambda m: congestion_sensitivity * m,
        coupling_dm=lambda m: congestion_sensitivity,
    )

    return create_standard_problem(
        hamiltonian=hamiltonian,
        terminal_cost=terminal_cost,
        initial_density=initial_density,
        geometry=geometry,
        time_horizon=time_horizon,
        num_timesteps=num_timesteps,
        use_unified=use_unified,
        **kwargs,
    )
