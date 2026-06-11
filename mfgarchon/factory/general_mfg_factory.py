"""
Factory for creating general MFG problems from configurations and function specifications.

Issue #673: Updated to class-based Hamiltonian API only.

This module provides high-level interfaces for creating any MFG problem
through configuration files and programmatic function definitions.
"""

from __future__ import annotations

import importlib.util
import threading
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import numpy as np

from mfgarchon.config.omegaconf_manager import OmegaConfManager
from mfgarchon.core.mfg_problem import MFGComponents, MFGProblem
from mfgarchon.geometry import BoundaryConditions
from mfgarchon.geometry.boundary.conditions import uniform_bc as _bc_from_type
from mfgarchon.utils.mfg_logging.logger import get_logger

if TYPE_CHECKING:
    from collections.abc import Callable

    from mfgarchon.core.hamiltonian import HamiltonianBase

logger = get_logger(__name__)


class GeneralMFGFactory:
    """
    Factory for creating general MFG problems from various input formats.

    Issue #673: Updated to class-based Hamiltonian API.

    This factory can create any MFG problem from:
    - HamiltonianBase instances (recommended)
    - Configuration files with Hamiltonian class references
    - Inline Hamiltonian definitions in YAML

    Example:
        >>> from mfgarchon.factory import GeneralMFGFactory
        >>> from mfgarchon.core.hamiltonian import SeparableHamiltonian, QuadraticControlCost
        >>>
        >>> factory = GeneralMFGFactory()
        >>> H = SeparableHamiltonian(
        ...     control_cost=QuadraticControlCost(control_cost=1.0),
        ...     coupling=lambda m: m,
        ...     coupling_dm=lambda m: 1.0,
        ... )
        >>> problem = factory.create_from_hamiltonian(
        ...     hamiltonian=H,
        ...     domain_config={"xmin": 0.0, "xmax": 1.0, "Nx": 51},
        ...     time_config={"T": 1.0, "Nt": 51},
        ...     m_initial=lambda x: np.exp(-10 * (x - 0.3)**2),
        ...     u_terminal=lambda x: (x - 0.8)**2,
        ... )
    """

    def __init__(self) -> None:
        """Initialize general MFG factory."""
        self.hamiltonian_registry: dict[str, HamiltonianBase] = {}
        self.function_registry: dict[str, Callable[..., Any]] = {}

    def register_hamiltonian(self, name: str, hamiltonian: HamiltonianBase) -> None:
        """Register a named Hamiltonian for use in configurations."""
        self.hamiltonian_registry[name] = hamiltonian
        logger.info(f"Registered Hamiltonian: {name}")

    def register_functions(self, functions: dict[str, Callable[..., Any]]) -> None:
        """Register named functions (for initial/final conditions) in configurations."""
        self.function_registry.update(functions)
        logger.info(f"Registered {len(functions)} functions: {list(functions.keys())}")

    def create_from_hamiltonian(
        self,
        hamiltonian: HamiltonianBase,
        domain_config: dict[str, Any],
        time_config: dict[str, Any],
        solver_config: dict[str, Any] | None = None,
        *,
        m_initial: Callable | None = None,
        u_terminal: Callable | None = None,
        potential_func: Callable | None = None,
        boundary_conditions: BoundaryConditions | None = None,
        description: str = "MFG Problem",
        parameters: dict[str, Any] | None = None,
    ) -> MFGProblem:
        """
        Create MFG problem from a class-based Hamiltonian.

        Args:
            hamiltonian: HamiltonianBase instance (SeparableHamiltonian, etc.)
            domain_config: Domain specification {"xmin": ..., "xmax": ..., "Nx": ...}
            time_config: Time specification {"T": ..., "Nt": ...}
            solver_config: Solver parameters {"sigma": ..., "coupling_coefficient": ...}
            m_initial: Initial density function m_0(x)
            u_terminal: Terminal value function u_T(x)
            potential_func: Additional potential V(x, t) if not in Hamiltonian
            boundary_conditions: Boundary conditions
            description: Problem description
            parameters: Additional problem parameters

        Returns:
            MFGProblem instance
        """
        # Build MFGComponents with class-based Hamiltonian
        components = MFGComponents(
            hamiltonian=hamiltonian,
            m_initial=m_initial,
            u_terminal=u_terminal,
            potential_func=potential_func,
            boundary_conditions=boundary_conditions,
            description=description,
            parameters=parameters or {},
        )

        # Build MFGProblem
        problem_kwargs = {**domain_config, **time_config}
        if solver_config:
            problem_kwargs.update(solver_config)

        problem_kwargs["components"] = components

        return MFGProblem(**problem_kwargs)

    def create_from_config_file(self, config_path: str) -> MFGProblem:
        """
        Create MFG problem from YAML configuration file.

        Issue #673: Updated to class-based Hamiltonian API.

        Expected format:
        ```yaml
        problem:
          description: "My custom MFG problem"
          type: custom

        hamiltonian:
          type: separable  # or "quadratic", "custom"
          control_cost: 1.0
          coupling_coefficient: 1.0

        functions:
          potential: "lambda x: 0.5 * x**2"
          m_initial: "lambda x: np.exp(-10*(x-0.5)**2)"
          u_terminal: "lambda x: (x - 0.8)**2"

        domain:
          xmin: 0.0
          xmax: 1.0
          Nx: 101

        time:
          T: 1.0
          Nt: 101

        solver:
          sigma: 0.5
          coupling_coefficient: 1.0

        parameters:
          alpha: 2.0
          beta: 0.5
        ```
        """
        config_manager = OmegaConfManager()
        config = config_manager.load_config(config_path)

        return self.create_from_config_dict(dict(config))

    def create_from_config_dict(self, config: dict[str, Any]) -> MFGProblem:
        """Create MFG problem from configuration dictionary."""

        # Load Hamiltonian from config
        hamiltonian_config = config.get("hamiltonian", {})
        hamiltonian = self._load_hamiltonian(hamiltonian_config)

        if hamiltonian is None:
            raise ValueError(
                "Hamiltonian specification is required.\n\n"
                "Example config:\n"
                "  hamiltonian:\n"
                "    type: separable\n"
                "    control_cost: 1.0\n"
                "    coupling_coefficient: 1.0"
            )

        # Load optional functions (initial/final conditions)
        functions_config = config.get("functions", {})

        m_initial = self._load_function(functions_config.get("m_initial"))
        u_terminal = self._load_function(functions_config.get("u_terminal"))
        potential_func = self._load_function(functions_config.get("potential"))

        # Also support legacy aliases
        if m_initial is None:
            m_initial = self._load_function(functions_config.get("initial_density"))
        if u_terminal is None:
            # Try legacy alias u_final
            u_terminal = self._load_function(functions_config.get("u_final"))
        if u_terminal is None:
            u_terminal = self._load_function(functions_config.get("final_value"))

        # Extract configurations
        domain_config = config.get("domain", {"xmin": 0.0, "xmax": 1.0, "Nx": 51})
        time_config = config.get("time", {"T": 1.0, "Nt": 51})

        # Issue #1284 / 2026-06-11 survey: fail loud when 'solver' section is absent.
        # The previous code silently injected sigma=1.0, masking misconfigured callers.
        if "solver" not in config:
            raise ValueError(
                "Configuration is missing the required 'solver' section.\n"
                "A silent sigma=1.0 default is NOT injected — provide it explicitly.\n"
                "Example:\n"
                "  solver:\n"
                "    sigma: 0.5\n"
                "    coupling_coefficient: 1.0"
            )
        solver_config = config["solver"]

        # Extract other components
        boundary_conditions = None
        if "boundary_conditions" in config:
            bc_config = config["boundary_conditions"]
            if isinstance(bc_config, BoundaryConditions):
                # Already a proper BoundaryConditions object — pass through.
                boundary_conditions = bc_config
            elif isinstance(bc_config, Mapping):
                # BoundaryConditions 'type' is a @property, not a constructor param.
                # Use _bc_from_type (uniform_bc) to build from a type string.
                # Issue #1284 / 2026-06-11 survey.
                bc_type_str = str(bc_config.get("type", "periodic"))
                boundary_conditions = _bc_from_type(bc_type_str)
            else:
                boundary_conditions = bc_config

        parameters = config.get("parameters", {})
        description = "MFG Problem"
        if "problem" in config and "description" in config["problem"]:
            description = config["problem"]["description"]

        return self.create_from_hamiltonian(
            hamiltonian=hamiltonian,
            domain_config=domain_config,
            time_config=time_config,
            solver_config=solver_config,
            m_initial=m_initial,
            u_terminal=u_terminal,
            potential_func=potential_func,
            boundary_conditions=boundary_conditions,
            description=description,
            parameters=parameters,
        )

    def _load_hamiltonian(self, hamiltonian_config: dict[str, Any] | str) -> HamiltonianBase | None:
        """
        Load Hamiltonian from configuration.

        Supports:
        - Registry name: "my_registered_hamiltonian"
        - Type-based config: {"type": "separable", "control_cost": 1.0, ...}
        """
        from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian

        if hamiltonian_config is None:
            return None

        # Check if it's a registry name
        if isinstance(hamiltonian_config, str):
            if hamiltonian_config in self.hamiltonian_registry:
                return self.hamiltonian_registry[hamiltonian_config]
            logger.warning(f"Hamiltonian '{hamiltonian_config}' not found in registry")
            return None

        # Type-based config
        h_type = hamiltonian_config.get("type", "separable")

        if h_type == "separable":
            control_cost = hamiltonian_config.get("control_cost", 1.0)
            coupling_coeff = hamiltonian_config.get("coupling_coefficient", 1.0)

            return SeparableHamiltonian(
                control_cost=QuadraticControlCost(control_cost=control_cost),
                coupling=lambda m, c=coupling_coeff: c * m,
                coupling_dm=lambda m, c=coupling_coeff: c,
            )

        elif h_type == "quadratic":
            from mfgarchon.core.hamiltonian import QuadraticMFGHamiltonian

            control_cost = hamiltonian_config.get("control_cost", 1.0)
            coupling_coeff = hamiltonian_config.get("coupling_coefficient", 1.0)

            return QuadraticMFGHamiltonian(
                control_cost=control_cost,
                coupling_coefficient=coupling_coeff,
            )

        else:
            logger.warning(f"Unknown Hamiltonian type: {h_type}")
            return None

    def _load_function(self, func_spec: str | None) -> Callable | None:
        """
        Load function from various specifications.

        Supports:
        - Module path: "my_module.my_function"
        - Lambda expressions: "lambda x: x**2"
        - Registry names: registered_function_name
        """
        if func_spec is None:
            return None

        func_spec = func_spec.strip()

        # Check if it's in the registry
        if func_spec in self.function_registry:
            return self.function_registry[func_spec]

        # Check if it's a lambda expression
        if func_spec.startswith("lambda"):
            try:
                # Create safe namespace for lambda evaluation
                safe_namespace = {
                    "np": np,
                    "numpy": np,
                    "exp": np.exp,
                    "sin": np.sin,
                    "cos": np.cos,
                    "sqrt": np.sqrt,
                    "abs": abs,
                    "max": max,
                    "min": min,
                    "pi": np.pi,
                    "e": np.e,
                }
                return eval(func_spec, safe_namespace)
            except Exception as e:
                logger.error(f"Failed to evaluate lambda function '{func_spec}': {e}")
                return None

        # Check if it's a module path
        if "." in func_spec:
            try:
                module_path, function_name = func_spec.rsplit(".", 1)
                module = importlib.import_module(module_path)
                return getattr(module, function_name)
            except Exception as e:
                logger.error(f"Failed to import function '{func_spec}': {e}")
                return None

        logger.warning(f"Could not resolve function specification: '{func_spec}'")
        return None

    def create_template_config(self, filename: str) -> None:
        """Create a template configuration file.

        Issue #673: Uses class-based Hamiltonian API.
        """
        template = {
            "problem": {"description": "My custom MFG problem", "type": "custom"},
            "hamiltonian": {
                "type": "separable",
                "control_cost": 1.0,
                "coupling_coefficient": 1.0,
            },
            "functions": {
                "potential": "lambda x: 0.5 * x * (1 - x)",
                "m_initial": "lambda x: np.exp(-10 * (x - 0.3)**2)",
                "u_terminal": "lambda x: (x - 0.8)**2",
            },
            "domain": {"xmin": 0.0, "xmax": 1.0, "Nx": 51},
            "time": {"T": 1.0, "Nt": 51},
            "solver": {"sigma": 0.5, "coupling_coefficient": 1.0},
            "boundary_conditions": {"type": "periodic"},
            "parameters": {"alpha": 1.0, "beta": 0.5},
        }

        config_manager = OmegaConfManager()
        config_manager.save_config(template, filename)  # type: ignore[arg-type]
        logger.info(f"Created template configuration: {filename}")

    def validate_config(self, config: dict[str, Any]) -> dict[str, Any]:
        """Validate configuration for general MFG problem."""
        validation: dict[str, Any] = {"valid": True, "errors": [], "warnings": []}

        # Check required sections
        required_sections = ["hamiltonian", "domain", "time"]
        for section in required_sections:
            if section not in config:
                validation["valid"] = False
                validation["errors"].append(f"Missing required section: {section}")

        # Check Hamiltonian config
        if "hamiltonian" in config:
            h_config = config["hamiltonian"]
            if isinstance(h_config, dict):
                if "type" not in h_config:
                    validation["warnings"].append("Hamiltonian type not specified, defaulting to 'separable'")

        # Check domain parameters
        if "domain" in config:
            domain = config["domain"]
            required_domain = ["xmin", "xmax", "Nx"]
            for param in required_domain:
                if param not in domain:
                    validation["valid"] = False
                    validation["errors"].append(f"Missing domain parameter: {param}")

        # Check time parameters
        if "time" in config:
            time_config = config["time"]
            required_time = ["T", "Nt"]
            for param in required_time:
                if param not in time_config:
                    validation["valid"] = False
                    validation["errors"].append(f"Missing time parameter: {param}")

        return validation


# Global factory instance (thread-safe lazy singleton, Issue #759)
_global_general_factory: GeneralMFGFactory | None = None
_general_factory_lock = threading.Lock()


def get_general_factory() -> GeneralMFGFactory:
    """Get global general MFG factory instance (thread-safe)."""
    global _global_general_factory
    if _global_general_factory is None:
        with _general_factory_lock:
            # Double-check locking pattern
            if _global_general_factory is None:
                _global_general_factory = GeneralMFGFactory()
    return _global_general_factory


def create_general_mfg_problem(
    hamiltonian: HamiltonianBase,
    **kwargs: Any,
) -> MFGProblem:
    """
    Convenience function to create general MFG problem.

    Issue #673: Updated to class-based Hamiltonian API.

    Args:
        hamiltonian: HamiltonianBase instance
        **kwargs: Additional arguments including:
            - xmin, xmax, Nx: Domain parameters
            - T, Nt: Time parameters
            - sigma, coupling_coefficient: Solver parameters
            - m_initial, u_terminal: Initial/terminal conditions
            - potential_func: Additional potential
            - boundary_conditions: Boundary conditions

    Returns:
        MFGProblem instance

    Example:
        >>> from mfgarchon.factory import create_general_mfg_problem
        >>> from mfgarchon.core.hamiltonian import SeparableHamiltonian, QuadraticControlCost
        >>>
        >>> H = SeparableHamiltonian(
        ...     control_cost=QuadraticControlCost(control_cost=1.0),
        ...     coupling=lambda m: m,
        ...     coupling_dm=lambda m: 1.0,
        ... )
        >>> problem = create_general_mfg_problem(
        ...     hamiltonian=H,
        ...     xmin=0.0, xmax=1.0, Nx=51,
        ...     T=1.0, Nt=51,
        ...     m_initial=lambda x: np.exp(-10 * (x - 0.3)**2),
        ...     u_terminal=lambda x: (x - 0.8)**2,
        ... )
    """
    factory = get_general_factory()

    # Extract configurations
    domain_config = {
        "xmin": kwargs.pop("xmin", 0.0),
        "xmax": kwargs.pop("xmax", 1.0),
        "Nx": kwargs.pop("Nx", 51),
    }

    time_config = {"T": kwargs.pop("T", 1.0), "Nt": kwargs.pop("Nt", 51)}

    solver_config = {
        "sigma": kwargs.pop("sigma", 1.0),
        "coupling_coefficient": kwargs.pop("coupling_coefficient", 0.5),
    }

    # Extract optional components
    m_initial = kwargs.pop("m_initial", None)
    # Support both u_terminal and legacy u_final
    u_terminal = kwargs.pop("u_terminal", None)
    if u_terminal is None:
        u_terminal = kwargs.pop("u_final", None)
    potential_func = kwargs.pop("potential_func", None)
    boundary_conditions = kwargs.pop("boundary_conditions", None)
    description = kwargs.pop("description", "MFG Problem")
    parameters = kwargs.pop("parameters", None)

    return factory.create_from_hamiltonian(
        hamiltonian=hamiltonian,
        domain_config=domain_config,
        time_config=time_config,
        solver_config=solver_config,
        m_initial=m_initial,
        u_terminal=u_terminal,
        potential_func=potential_func,
        boundary_conditions=boundary_conditions,
        description=description,
        parameters=parameters,
    )
