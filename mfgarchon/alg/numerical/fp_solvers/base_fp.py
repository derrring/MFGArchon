from __future__ import annotations

from abc import abstractmethod
from enum import Enum
from typing import TYPE_CHECKING

from mfgarchon.alg.base_solver import BaseNumericalSolver, SchemeFamily


class DriftConvention(Enum):
    """What an FP solver's drift input MEANS (Issue #1043).

    The FP equation only ever sees the advective velocity ``α`` in ``div(α m)``. Most solvers
    take that velocity directly; a few take the value function ``U`` and recover
    ``α = -coupling·∇U`` internally (e.g. weak-form/FEM/meshless, which differentiate ``U`` on
    their own quadrature/MLS basis rather than on a possibly-coarse FP grid). This trait makes
    that distinction explicit and machine-readable so a coupler can dispatch the right object to
    each solver instead of guessing from the parameter name.
    """

    VELOCITY = "velocity"  # input is α* = -∂_p H (consumed as flux = m·α)
    VALUE_FUNCTION = "value_function"  # input is U; the solver computes α = -coupling·∇U itself


if TYPE_CHECKING:
    from collections.abc import Callable

    import numpy as np

    from mfgarchon.config import BaseConfig
    from mfgarchon.core.mfg_problem import MFGProblem


class BaseFPSolver(BaseNumericalSolver):
    """
    Abstract Base Class for Fokker-Planck (FP) equation solvers.

    The FP equation describes density evolution:
        ∂m/∂t + ∇·(α m) = ∇·(D ∇m)

    where:
        m(t,x): probability density
        α(t,x,m): drift field (from various sources)
        D(t,x,m): diffusion tensor (isotropic, anisotropic, or state-dependent)

    Drift Sources (controlled by drift_field parameter):
        - Zero: α = 0 (pure diffusion)
        - Optimal control (MFG): α = -∇U (most common)
        - Prescribed field: α = v(t,x) (wind, currents)
        - Custom/state-dependent: α = f(t,x,m)

    Diffusion Types (controlled by volatility_field parameter):
        - Constant isotropic: D = σ²/2 (scalar, same in all directions)
        - Anisotropic: D = diag(σ₁², σ₂², ...) (different per direction)
        - Spatially varying: D(t,x) (depends on location)
        - State-dependent: D(t,x,m) (nonlinear, depends on density)
        - Zero: D = 0 (pure advection, requires WENO/SL)

    This base class provides a general, powerful interface supporting all drift and
    diffusion types while maintaining backward compatibility with MFG-centric usage.

    Design Philosophy:
        Make simple cases simple (MFG with constant diffusion), make complex cases
        possible (anisotropic, state-dependent, spatially varying). The solver
        handles drift and diffusion computation based on provided parameters.

    Note for Implementers:
        When implementing concrete solvers:
        - Handle D=0 (pure advection) carefully - FDM may be unstable, use WENO/SL
        - Support anisotropic diffusion tensors for realistic applications
        - Enable state-dependent coefficients for nonlinear PDEs
    """

    # Scheme family trait for duality validation (Issue #580)
    _scheme_family = SchemeFamily.GENERIC

    # Drift-input convention trait (Issue #1043). Default VELOCITY: the canonical contract is
    # drift_field = advective velocity α*. Solvers that take the value function U instead set
    # this to VALUE_FUNCTION. Metadata only today; a future coupler dispatch reads it (the
    # signature-sniffing it would replace carries paper-number risk, deferred).
    _drift_convention: DriftConvention = DriftConvention.VELOCITY

    def __init__(self, problem: MFGProblem, config: BaseConfig | None = None) -> None:
        """
        Initializes the FP solver with the MFG problem definition.

        Args:
            problem: An instance of an MFGProblem (or its subclass)
                    containing all problem-specific parameters and functions.
            config: Optional solver configuration. If None, a minimal config is created.
                    Most FP solvers don't require config (backward compatible).

        Note:
            Inherits from BaseNumericalSolver to access unified BC infrastructure.
            See BaseMFGSolver.get_boundary_conditions() for BC resolution hierarchy.
        """
        # Maintain backward compatibility - if no config provided, create a minimal one
        if config is None:
            config = type("MinimalFPConfig", (), {})()

        super().__init__(problem, config)
        self.fp_method_name: str = "BaseFP"  # Concrete solvers should override this
        self.backend = None  # Backend for array operations (NumPy, PyTorch, JAX)

        # Validate solver compatibility if problem supports it (Phase 3.1.5)
        self._validate_problem_compatibility()

    def _validate_problem_compatibility(self) -> None:
        """
        Validate that this solver is compatible with the problem.

        This method checks if the problem has solver compatibility detection
        (Phase 3.1 unified interface) and validates compatibility if available.
        For older problems without this feature, validation is skipped.
        """
        # Issue #543 Phase 2: Replace hasattr with try/except
        try:
            # Get solver type identifier from subclass
            solver_type = self._get_solver_type_id()
            if solver_type is not None:
                self.problem.validate_solver_type(solver_type)
        except AttributeError:
            # Backward compatibility: problem doesn't have validate_solver_type
            return
        except ValueError as e:
            # Re-raise validation errors with solver class information
            raise ValueError(f"Cannot use {self.__class__.__name__} with this problem.\n\n{e!s}") from e

    def _get_solver_type_id(self) -> str | None:
        """
        Get solver type identifier for compatibility checking.

        Subclasses should override this to return their type identifier.
        Returns None if solver type cannot be determined (skips validation).
        """
        # Map class names to solver type IDs
        class_name = self.__class__.__name__
        type_mapping = {
            "FPFDMSolver": "fdm",
            "FPParticleSolver": "particle",
            "FPNetworkSolver": "network_solver",
            "FPGFDMSolver": "gfdm",
        }
        return type_mapping.get(class_name)

    def discretize(self) -> None:
        """
        Set up spatial and temporal discretization.

        For FP solvers, discretization is typically handled in subclass __init__
        or solve_fp_system methods. This default implementation does nothing.
        Subclasses may override for explicit discretization setup.
        """

    def solve(self) -> np.ndarray:
        """
        Solve the FP problem (delegates to solve_fp_system).

        This method satisfies the BaseMFGSolver interface. For FP solvers,
        the primary interface is solve_fp_system() which accepts explicit
        drift and diffusion fields.

        Returns:
            Density evolution M(t,x) using default parameters.
        """
        # Get initial condition from problem
        m0 = self.problem.get_initial_density()
        return self.solve_fp_system(m0)

    def validate_solution(self) -> dict[str, float]:
        """
        Validate the computed solution.

        Returns:
            Dictionary with validation metrics (mass conservation, etc.)
        """
        return {"mass_conservation_error": 0.0}

    @abstractmethod
    def solve_fp_system(
        self,
        m_initial_condition: np.ndarray,
        drift_field: np.ndarray | Callable | None = None,
        volatility_field: float | np.ndarray | Callable | None = None,
        show_progress: bool | None = None,
        progress_callback: Callable[[int], None] | None = None,  # Issue #640
        # MMS verification support
        source_term: Callable[[float, np.ndarray], np.ndarray] | None = None,
    ) -> np.ndarray:
        """
        Solves the full Fokker-Planck (FP) system forward in time.

        This method computes density evolution M(t,x) from t=0 to t=T under
        the specified drift and diffusion.

        General FP Equation:
            ∂m/∂t + ∇·(α m) = ∇·(D ∇m)

            where:
            - α(t,x,m): drift field (controlled by drift_field)
            - D(t,x,m): diffusion tensor (controlled by volatility_field)

        Equation Types Supported:
            1. Advection-diffusion (D>0, α≠0): Standard MFG, transport with diffusion
            2. Pure diffusion (D>0, α=0): Heat equation
            3. Pure advection (D=0, α≠0): Transport equation (WENO/SL recommended)
            4. Anisotropic diffusion (D is tensor): Different diffusion in each direction
            5. State-dependent (D or α depend on m): Nonlinear PDEs

        Drift Specification:
            drift_field can be:
            - None: Zero drift α = 0 (pure diffusion)
            - np.ndarray: Precomputed drift field α(t,x)
              Shape: (Nt, Nx) for 1D scalar, (Nt, Nx, d) for d-dim vector
            - Callable: Function α(t, x, m) -> drift

        Volatility Specification (Issue #717 unified API):
            volatility_field can be:
            - None: Use problem.sigma (backward compatible)
            - float: Constant isotropic volatility σ → D = σ²/2
            - (d,) array: Diagonal volatility [σ₀, σ₁, ...] → D = diag(σᵢ²)/2
            - (d, d) array: Full volatility matrix Σ → D = ΣΣᵀ/2
            - (*shape, d, d) array: Spatially varying Σ(x) → D(x) = Σ(x)Σ(x)ᵀ/2
            - Callable: Function σ(t, x, m) or Σ(t, x, m) -> volatility

            Note: volatility_field is the SDE noise coefficient σ or Σ. Internally
            converted to diffusion D = σ²/2 (scalar) or D = ΣΣᵀ/2 (matrix).

            DEPRECATED: tensor_diffusion_field and volatility_matrix (on FPFDMSolver)
            are deprecated. Use volatility_field with appropriate shape.

        Args:
            m_initial_condition: Initial density M(0,x) at t=0
                Shape: (Nx,) for 1D, (Nx, Ny) for 2D, etc.

            drift_field: Drift field specification (optional):
                - None: Zero drift
                - np.ndarray: Precomputed drift α(t,x)
                - Callable: Function α(t, x, m) -> drift
                Default: None

            volatility_field: Volatility specification (optional, Issue #717):
                - None: Use problem.sigma
                - float: Constant isotropic volatility σ
                - np.ndarray: Spatially varying volatility
                - Callable: Function σ(t, x, m) -> volatility
                Default: None

            source_term: Source term for MMS verification (optional):
                - None: No source term (default, standard PDE)
                - Callable: Function S(t, x_grid) -> np.ndarray
                  Evaluated analytically at each timestep.
                  Used for Method of Manufactured Solutions verification.
                Default: None

            show_progress: Display progress bar for timesteps
                Default: True

        Returns:
            Density evolution M(t,x) over time
            Shape: (Nt, Nx) or (Nt, Nx, Ny) etc.

        Examples:
            # Pure diffusion (heat equation, D>0, α=0)
            >>> M = solver.solve_fp_system(m0)

            # MFG optimal control (isotropic diffusion)
            >>> drift = -problem.compute_gradient(U_hjb) / problem.control_cost
            >>> M = solver.solve_fp_system(m0, drift_field=drift)

            # Anisotropic diffusion
            >>> D = np.diag([0.1, 0.5])  # Different in x,y directions
            >>> M = solver.solve_fp_system(m0, drift_field=drift, volatility_field=D)

            # State-dependent diffusion
            >>> D_func = lambda t, x, m: 0.1 * (1 + m)  # Increases with density
            >>> M = solver.solve_fp_system(m0, drift_field=drift, volatility_field=D_func)

            # Pure advection (D=0, α≠0)
            >>> M = solver.solve_fp_system(m0, drift_field=drift, volatility_field=0.0)

            # Spatially varying diffusion
            >>> D_field = create_spatially_varying_diffusion(...)  # (Nt, Nx)
            >>> M = solver.solve_fp_system(m0, drift_field=drift, volatility_field=D_field)

        Note:
            For MFG problems:
            - drift = -∇U / λ (user computes externally)
            - diffusion = problem.sigma (default) or custom
            This gives full control over both drift and diffusion.

        Helper Functions (Phase 2):
            For convenience and clarity, drift helper functions are available:

            >>> from mfgarchon.utils import (
            ...     zero_drift,
            ...     optimal_control_drift,
            ...     prescribed_drift,
            ...     density_dependent_drift,
            ...     composite_drift,
            ... )

            These helpers return callables with clear intent. Callable drift support
            is planned for Phase 2. Current solvers (v0.13) support None and np.ndarray.

            See mfgarchon.utils.drift_helpers for details.
        """


if __name__ == "__main__":
    """Quick smoke test for development."""
    print("Testing BaseFPSolver...")

    # Test base class availability
    assert BaseFPSolver is not None
    print("  BaseFPSolver class available")

    # Test that BaseFPSolver is abstract (cannot be instantiated)
    from mfgarchon import MFGProblem
    from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
    from mfgarchon.core.mfg_problem import MFGComponents
    from mfgarchon.geometry import TensorProductGrid
    from mfgarchon.geometry.boundary import neumann_bc

    # Minimal components for testing
    H = SeparableHamiltonian(
        control_cost=QuadraticControlCost(control_cost=1.0),
        coupling=lambda m: 0.0,
        coupling_dm=lambda m: 0.0,
    )
    components = MFGComponents(
        hamiltonian=H,
        u_terminal=lambda x: 0.0,
        m_initial=lambda x: 1.0,
    )

    geometry = TensorProductGrid(
        bounds=[(0.0, 1.0)],
        Nx_points=[21],
        boundary_conditions=neumann_bc(dimension=1),
    )
    problem = MFGProblem(geometry=geometry, T=1.0, Nt=10, sigma=0.1, components=components)

    try:
        # This should fail because BaseFPSolver is abstract
        base_solver = BaseFPSolver(problem)
        raise AssertionError("Should have raised TypeError for abstract class")
    except TypeError:
        # Expected - abstract class cannot be instantiated
        print("  BaseFPSolver correctly abstract")

    print("Smoke tests passed!")
