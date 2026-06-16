"""
OmegaConf-based configuration management for MFGarchon.

This module provides YAML load / compose / CLI-override / parameter-sweep
functionality on plain DictConfig objects. It is the YAML transport layer
for the config system; validation lives in Pydantic models (see
`mfgarchon.config.core`, `.mfg_methods`, `.array_validation`), crossed into
via `mfgarchon.config.bridge.bridge_to_pydantic`.

History note: prior versions (before v0.19.4) maintained an OmegaConf
dataclass-schema hierarchy (`structured_schemas.MFGSchema` etc.) alongside
the canonical Pydantic hierarchy. That parallel schema layer was removed
in v0.19.4 per the North Star design — Pydantic is the single schema
authority; OmegaConf handles YAML transport only.

Features:
- YAML configuration files with interpolation (`${var}`)
- Configuration composition and inheritance via OmegaConf.merge
- CLI overrides (OmegaConf.from_cli, .from_dotlist)
- Parameter sweeps via `create_parameter_sweep`
- Bridging to Pydantic validation via `mfgarchon.config.bridge.bridge_to_pydantic`
  (the single canonical OmegaConf->Pydantic crossing; this manager is transport only)
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from mfgarchon.utils.mfg_logging import get_logger

# Simplified OmegaConf integration - minimize typing conflicts
if TYPE_CHECKING:
    # For static typing, use minimal type definitions

    DictConfig = dict[str, Any]

    class OmegaConf:
        @staticmethod
        def structured(obj: Any) -> Any: ...
        @staticmethod
        def load(file: Any) -> Any: ...
        @staticmethod
        def save(config: Any, file: Any) -> None: ...
        @staticmethod
        def merge(*configs: Any) -> Any: ...
        @staticmethod
        def create(obj: Any) -> Any: ...
        @staticmethod
        def resolve(config: Any) -> None: ...

    ConfigAttributeError = Exception
    UnsupportedInterpolationType = Exception
    OMEGACONF_AVAILABLE = True
else:
    # Runtime imports and fallbacks
    try:
        from omegaconf import DictConfig, OmegaConf
        from omegaconf.errors import ConfigAttributeError

        try:
            from omegaconf.errors import UnsupportedInterpolationType
        except ImportError:
            UnsupportedInterpolationType = Exception

        OMEGACONF_AVAILABLE = True
    except ImportError:
        OMEGACONF_AVAILABLE = False
        DictConfig = dict

        class OmegaConf:
            @staticmethod
            def structured(obj):
                return {}

            @staticmethod
            def load(file):
                return {}

            @staticmethod
            def save(config, file):
                pass

            @staticmethod
            def merge(*configs):
                return {}

            @staticmethod
            def create(obj):
                return {}

            @staticmethod
            def resolve(config):
                pass

        ConfigAttributeError = Exception
        UnsupportedInterpolationType = Exception


logger = get_logger(__name__)

if TYPE_CHECKING:
    from omegaconf import DictConfig as OmegaConfig
else:
    OmegaConfig = Any


class OmegaConfManager:
    """
    Configuration manager using OmegaConf for YAML-based configuration.

    Features:
    - YAML configuration files with interpolation
    - Configuration composition and inheritance
    - Integration with existing Pydantic configs
    - Parameter sweeps and experiment management
    - Type validation and schema enforcement
    """

    def __init__(self, config_dir: str | Path | None = None):
        """
        Initialize OmegaConf configuration manager.

        Args:
            config_dir: Directory containing configuration files
        """
        # Re-check availability at runtime
        try:
            from omegaconf import DictConfig as _DictConfig
            from omegaconf import OmegaConf as _OmegaConf

            self._OmegaConf = _OmegaConf
            self._DictConfig = _DictConfig
            self._omegaconf_available = True
        except ImportError as err:
            raise ImportError("OmegaConf is not available. Install with: pip install omegaconf") from err

        self.config_dir = Path(config_dir) if config_dir else Path(__file__).parent / "configs"
        self.config_dir.mkdir(exist_ok=True)
        self._schema_cache: dict[str, OmegaConfig] = {}

        # Initialize default configurations
        self._create_default_configs()

    def _create_default_configs(self) -> None:
        """Create default configuration files if they don't exist."""

        # Base MFG problem configuration
        base_mfg_config = {
            "problem": {
                "name": "base_mfg_problem",
                "T": 1.0,
                "Nx": 50,
                "Nt": 30,
                "domain": {"x_min": 0.0, "x_max": 1.0},
                "initial_condition": {
                    "type": "gaussian",
                    "parameters": {"center": 0.5, "width": 0.1},
                },
                "boundary_conditions": {
                    "m": {"type": "no_flux"},
                    "u": {"type": "neumann", "left_value": 0.0, "right_value": 0.0},
                },
            }
        }

        # Solver configuration
        solver_config = {
            "solver": {
                "type": "fixed_point",
                "max_iterations": 100,
                "tolerance": 1e-6,
                "damping": 0.5,
                "backend": "numpy",
                "hjb": {
                    "method": "gfdm",
                    "boundary_handling": "penalty",
                    "penalty_weight": 1000.0,
                    "newton": {
                        "max_iterations": 20,
                        "tolerance": 1e-8,
                        "line_search": True,
                    },
                },
                "fp": {"method": "fdm", "upwind_scheme": "central"},
            }
        }

        # Towel on Beach problem configuration
        beach_config = {
            "problem": {
                "name": "towel_on_beach",
                "type": "spatial_competition",
                "T": 2.0,
                "Nx": 80,
                "Nt": 40,
                "domain": {"x_min": 0.0, "x_max": 1.0},
                "parameters": {
                    "stall_position": 0.6,
                    "crowd_aversion": "${lambda}",  # Interpolation placeholder
                    "noise_level": 0.1,
                },
                "initial_condition": {
                    "type": "${init_type}",  # Can be overridden
                    "parameters": {
                        "gaussian": {"center": "${init_center:0.2}", "width": 0.05},
                        "uniform": {},
                        "bimodal": {"centers": [0.3, 0.7], "widths": [0.03, 0.03]},
                    },
                },
            }
        }

        # Experiment configuration
        experiment_config = {
            "experiment": {
                "name": "parameter_sweep",
                "description": "Parameter sweep experiment",
                "output_dir": "results/${experiment.name}",
                "logging": {
                    "level": "INFO",
                    "file": "${experiment.output_dir}/experiment.log",
                },
                "visualization": {
                    "enabled": True,
                    "save_plots": True,
                    "plot_dir": "${experiment.output_dir}/plots",
                    "formats": ["png", "html"],
                    "dpi": 300,
                },
                "sweeps": {
                    "lambda": [0.8, 1.2, 1.5, 2.0, 2.5, 3.0, 3.5],
                    "init_type": ["gaussian", "uniform", "bimodal"],
                },
            }
        }

        # Write default configs
        configs: dict[str, dict[str, Any]] = {
            "base_mfg.yaml": base_mfg_config,
            "solver.yaml": solver_config,
            "beach_problem.yaml": beach_config,
            "experiment.yaml": experiment_config,
        }

        for filename, config in configs.items():
            config_path = self.config_dir / filename
            if not config_path.exists():
                self._OmegaConf.save(self._OmegaConf.create(config), config_path)
                logger.info(f"Created default config: {config_path}")

    def load_config(self, config_path: str | Path, **overrides: Any) -> OmegaConfig:
        """
        Load configuration from YAML file with optional overrides.

        Args:
            config_path: Path to configuration file
            **overrides: Configuration overrides

        Returns:
            OmegaConf DictConfig object
        """
        config_path = Path(config_path)
        if not config_path.is_absolute():
            config_path = self.config_dir / config_path

        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        # Load base configuration
        config = self._OmegaConf.load(config_path)

        # Apply overrides
        if overrides:
            override_config = self._OmegaConf.create(overrides)
            config = self._OmegaConf.merge(config, override_config)

        # Resolve interpolations (modifies config in place)
        try:
            self._OmegaConf.resolve(config)
        except (UnsupportedInterpolationType, Exception) as e:
            logger.warning(f"Could not resolve all interpolations: {e}")

        # Cast to DictConfig for type consistency (our configs should be dictionaries)
        from typing import cast

        return cast("OmegaConfig", config)

    def compose_config(self, *config_paths: str | Path, **overrides: Any) -> OmegaConfig:
        """
        Compose configuration from multiple YAML files.

        Args:
            *config_paths: Paths to configuration files (in merge order)
            **overrides: Final configuration overrides

        Returns:
            Composed OmegaConf DictConfig object
        """
        configs = []
        for path in config_paths:
            config = self.load_config(path)
            configs.append(config)

        # Merge configurations
        composed_config = self._OmegaConf.merge(*configs)

        # Apply final overrides
        if overrides:
            override_config = self._OmegaConf.create(overrides)
            composed_config = self._OmegaConf.merge(composed_config, override_config)

        # Cast to DictConfig for type consistency
        from typing import cast

        return cast("OmegaConfig", composed_config)

    # NOTE (Issue #1392): the vestigial `create_pydantic_config` / `_map_omega_to_pydantic`
    # pair was removed. They were a pre-North-Star second OmegaConf->Pydantic bridge that
    # bypassed the canonical, single crossing `mfgarchon.config.bridge.bridge_to_pydantic`
    # and silently returned a default config (flat keys vs the nested MFGSolverConfig). Use
    # `bridge_to_pydantic(omega_cfg, MFGSolverConfig)` for validation; this manager is
    # transport only (load/merge/save/CLI).

    def save_config(self, config: OmegaConfig, output_path: str | Path) -> None:
        """
        Save configuration to YAML file.

        Args:
            config: OmegaConf configuration
            output_path: Output file path
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        self._OmegaConf.save(config, output_path)
        logger.info(f"Configuration saved to: {output_path}")

    def create_parameter_sweep(self, base_config: OmegaConfig, sweep_params: dict[str, list[Any]]) -> list[OmegaConfig]:
        """
        Create parameter sweep configurations.

        Args:
            base_config: Base configuration
            sweep_params: Dictionary of parameter names to value lists

        Returns:
            List of configurations for parameter sweep
        """
        import itertools

        # Get all combinations
        param_names = list(sweep_params.keys())
        param_values = list(sweep_params.values())
        combinations = list(itertools.product(*param_values))

        sweep_configs = []
        for combo in combinations:
            # Create configuration for this combination
            config = self._OmegaConf.create(base_config)

            # Apply parameter values using OmegaConf dot notation
            for param_name, param_value in zip(param_names, combo, strict=False):
                # Use OmegaConf.select for navigation and direct assignment
                # First ensure the nested structure exists
                keys = param_name.split(".")
                for i in range(len(keys) - 1):
                    partial_key = ".".join(keys[: i + 1])
                    if self._OmegaConf.select(config, partial_key) is None:
                        # Create the nested key by setting an empty dict
                        parent_key = ".".join(keys[:i]) if i > 0 else None
                        if parent_key:
                            parent = self._OmegaConf.select(config, parent_key)
                            setattr(parent, keys[i], {})
                        else:
                            setattr(config, keys[i], {})

                # Now set the final value
                if len(keys) == 1:
                    setattr(config, keys[0], param_value)
                else:
                    parent_key = ".".join(keys[:-1])
                    parent = self._OmegaConf.select(config, parent_key)
                    setattr(parent, keys[-1], param_value)

            # Add metadata
            if "experiment" not in config:
                config.experiment = {}
            config.experiment["current_params"] = dict(zip(param_names, combo, strict=False))

            sweep_configs.append(config)

        return sweep_configs

    def validate_config(self, config: OmegaConfig) -> bool:
        """
        Validate configuration.

        Args:
            config: Configuration to validate

        Returns:
            True if valid, False otherwise
        """
        try:
            # Basic validation - check required fields
            required_fields = ["problem", "solver"]
            for field in required_fields:
                if field not in config:
                    logger.error(f"Missing required field: {field}")
                    return False

            # Type validation
            if not isinstance(config.problem.T, int | float):
                logger.error("problem.T must be numeric")
                return False

            if not isinstance(config.problem.Nx, int):
                logger.error("problem.Nx must be integer")
                return False

            # Add more validation as needed
            return True

        except (AttributeError, ConfigAttributeError) as e:
            logger.error(f"Configuration validation error: {e}")
            return False

    def get_config_template(self, template_name: str) -> OmegaConfig:
        """
        Get configuration template by name.

        Args:
            template_name: Name of configuration template

        Returns:
            Configuration template
        """
        templates = {
            "beach_problem": "beach_problem.yaml",
            "base_mfg": "base_mfg.yaml",
            "solver": "solver.yaml",
            "experiment": "experiment.yaml",
        }

        if template_name not in templates:
            raise ValueError(f"Unknown template: {template_name}")

        return self.load_config(templates[template_name])


def create_omega_manager(config_dir: str | Path | None = None) -> OmegaConfManager:
    """
    Create OmegaConf configuration manager.

    Args:
        config_dir: Directory containing configuration files

    Returns:
        OmegaConfManager instance
    """
    return OmegaConfManager(config_dir)


# Convenience functions
def load_beach_config(**overrides: Any) -> OmegaConfig:
    """Load Towel on Beach configuration with overrides."""
    manager = create_omega_manager()
    return manager.load_config("beach_problem.yaml", **overrides)


def load_experiment_config(**overrides: Any) -> OmegaConfig:
    """Load experiment configuration with overrides."""
    manager = create_omega_manager()
    return manager.load_config("experiment.yaml", **overrides)


def create_parameter_sweep_configs(lambda_values: list[float], init_types: list[str]) -> list[OmegaConfig]:
    """Create parameter sweep configurations for beach problem."""
    manager = create_omega_manager()
    base_config = manager.load_config("beach_problem.yaml")

    sweep_params = {
        "problem.parameters.crowd_aversion": lambda_values,
        "problem.initial_condition.type": init_types,
    }

    return manager.create_parameter_sweep(base_config, sweep_params)  # type: ignore
