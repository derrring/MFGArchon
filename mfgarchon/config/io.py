"""
YAML I/O for solver configurations.

This module provides functions to load and save solver configurations from/to
YAML files with schema validation.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import yaml

if TYPE_CHECKING:
    from .core import SolverConfig


def load_solver_config(path: str | Path) -> SolverConfig:
    """
    Load solver configuration from YAML file.

    Parameters
    ----------
    path : str | Path
        Path to YAML configuration file

    Returns
    -------
    SolverConfig
        Validated solver configuration

    Raises
    ------
    FileNotFoundError
        If configuration file doesn't exist
    ValidationError
        If configuration is invalid
    yaml.YAMLError
        If YAML syntax is invalid

    Examples
    --------
    >>> config = load_solver_config("experiments/baseline.yaml")
    >>> result = solve_mfg(problem, config=config)

    YAML Format
    -----------
    The schema is FLAT: the top-level keys are the ``MFGSolverConfig`` fields
    (``hjb``, ``fp``, ``picard``, ``backend``, ``logging``). There is no
    ``solver:`` wrapper::

        hjb:
          method: fdm
          accuracy_order: 2
        fp:
          method: particle
          particle:
            num_particles: 5000
        picard:
          max_iterations: 50
          tolerance: 1.0e-6
        backend:
          type: numpy

    A ``solver:``-wrapped or OmegaConf-vocabulary YAML (e.g. the generated
    ``mfgarchon/config/configs/*.yaml``) is a different format; load those via
    :mod:`mfgarchon.config.omegaconf_manager`, not this function.
    """
    from .core import SolverConfig

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {path}\nPlease create a YAML configuration file or use programmatic config."
        )

    try:
        with open(path) as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Invalid YAML syntax in {path}: {e}") from e

    if data is None:
        data = {}

    # Fail fast on unknown top-level keys. The schema is flat, so a ``solver:``-wrapped
    # or OmegaConf-vocabulary YAML maps to nothing -- a plain ``model_validate`` would
    # silently drop the unknown keys and return all-default config (wrong-config-silently-
    # ignored). Surface it instead of defaulting (kernel fail-fast).
    if isinstance(data, dict):
        valid_keys = set(SolverConfig.model_fields)
        unknown = sorted(set(data) - valid_keys)
        if unknown:
            raise ValueError(
                f"Unknown top-level key(s) {unknown} in {path}.\n"
                f"The solver-config schema is flat; valid top-level keys are {sorted(valid_keys)}.\n"
                f"A 'solver:'-wrapped or OmegaConf-style YAML (e.g. config/configs/*.yaml) is a "
                f"different format -- load those via mfgarchon.config.omegaconf_manager."
            )

    # Validate and construct
    try:
        return SolverConfig.model_validate(data)
    except Exception as e:
        raise ValueError(
            f"Invalid configuration in {path}:\n{e}\n\nSee docs/user_guide/configuration.md for YAML format examples."
        ) from e


def save_solver_config(config: SolverConfig, path: str | Path) -> None:
    """
    Save solver configuration to YAML file.

    Parameters
    ----------
    config : SolverConfig
        Configuration to save
    path : str | Path
        Output file path

    Examples
    --------
    >>> config = SolverConfig(...)
    >>> save_solver_config(config, "experiments/my_config.yaml")

    >>> # Or use method
    >>> config.to_yaml("experiments/my_config.yaml")
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Dump config, excluding None values and using JSON-serializable format
    config_dict = config.model_dump(exclude_none=True, mode="json")

    with open(path, "w") as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False, indent=2, allow_unicode=True)


def validate_yaml_config(path: str | Path) -> tuple[bool, str]:
    """
    Validate YAML configuration without loading.

    Parameters
    ----------
    path : str | Path
        Path to YAML configuration file

    Returns
    -------
    tuple[bool, str]
        (is_valid, message) - True if valid, False with error message otherwise

    Examples
    --------
    >>> is_valid, msg = validate_yaml_config("config.yaml")
    >>> if not is_valid:
    ...     print(f"Config invalid: {msg}")
    """
    try:
        load_solver_config(path)
        return True, "Configuration is valid"
    except FileNotFoundError as e:
        return False, str(e)
    except yaml.YAMLError as e:
        return False, f"YAML syntax error: {e}"
    except ValueError as e:
        return False, f"Validation error: {e}"
    except Exception as e:
        return False, f"Unexpected error: {e}"
