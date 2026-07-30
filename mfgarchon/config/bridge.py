"""
Bridge utilities for Pydantic-OmegaConf interoperability.

This module provides generic adapters between OmegaConf DictConfigs and Pydantic
models, enabling seamless conversion between the two configuration systems.

Architecture Overview:
- **Pydantic** (`*Config`): Runtime validation, API safety, type strictness
- **OmegaConf** (`*Schema`): YAML management, CLI overrides, parameter sweeps

The bridge functions allow:
1. Converting OmegaConf configs to validated Pydantic models
2. Saving effective (resolved) configs for reproducibility

See `docs/development/PYDANTIC_OMEGACONF_COOPERATION.md` for the full guide.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

if TYPE_CHECKING:
    from omegaconf import DictConfig


def bridge_to_pydantic[T: BaseModel](
    omega_cfg: DictConfig,
    pydantic_cls: type[T],
    *,
    strict: bool = True,
) -> T:
    """
    Convert an OmegaConf DictConfig to a validated Pydantic model.

    This generic adapter handles the common pattern of loading experiment
    configuration from YAML (via OmegaConf) and validating it with Pydantic.

    Parameters
    ----------
    omega_cfg : DictConfig
        OmegaConf configuration, typically loaded from YAML.
    pydantic_cls : type[T]
        Target Pydantic model class.
    strict : bool, optional
        If True, use strict validation (no type coercion). Default True.

    Returns
    -------
    T
        Validated Pydantic model instance.

    Raises
    ------
    ValidationError
        If the config fails Pydantic validation.

    Examples
    --------
    >>> from omegaconf import OmegaConf
    >>> from mfgarchon.config import MFGSolverConfig
    >>> from mfgarchon.config.bridge import bridge_to_pydantic
    >>>
    >>> # Load from YAML
    >>> omega_cfg = OmegaConf.load("experiment.yaml")
    >>>
    >>> # Convert to validated Pydantic model
    >>> config = bridge_to_pydantic(omega_cfg, MFGSolverConfig)
    >>> print(config.picard.tolerance)  # Type-safe access; tolerance lives under picard
    """
    from omegaconf import OmegaConf

    # Resolve interpolations and missing values
    OmegaConf.resolve(omega_cfg)

    # Convert to plain Python dict
    container: dict[str, Any] = OmegaConf.to_container(omega_cfg, resolve=True)  # type: ignore[assignment]

    # Issue #1766: config models now forbid unknown fields, so a typo in Python is an error
    # rather than a silently dropped keyword. A YAML file is not the same boundary. Top-level
    # interpolation anchors are a legitimate OmegaConf idiom -- `base_tol: 1e-6` with
    # `picard.tolerance: ${base_tol}` -- and they are scaffolding, not configuration, so they
    # have no field to land in. Drop them here, but SAY which: a silent drop at a transport
    # boundary is how a genuine top-level typo would disappear.
    #
    # Only the top level is filtered. Nested keys still reach their own model, so
    # `picard.toleranse` is still a hard error.
    # A deprecated alias is by definition not in `model_fields`, so filtering on fields alone
    # dropped it before the `mode="before"` validator could translate it -- the value silently
    # reverted to its default and the warning below called a documented alias a typo. Union in
    # the model's own alias map so the two agree; the map has one owner on the model.
    accepted = set(pydantic_cls.model_fields) | set(getattr(pydantic_cls, "LEGACY_FIELD_ALIASES", {}))
    scaffolding = sorted(set(container) - accepted)
    if scaffolding:
        container = {k: v for k, v in container.items() if k not in scaffolding}
        warnings.warn(
            f"bridge_to_pydantic: dropped {len(scaffolding)} top-level key(s) that "
            f"{pydantic_cls.__name__} has no field for: {', '.join(scaffolding)}. Interpolation "
            f"anchors are expected here and are already resolved; anything else is a typo that "
            f"will not take effect.",
            UserWarning,
            stacklevel=2,
        )

    # Validate with Pydantic
    if strict:
        return pydantic_cls.model_validate(container, strict=True)
    return pydantic_cls.model_validate(container)


def save_effective_config(
    config: BaseModel,
    output_dir: str | Path,
    *,
    filename: str = "resolved_config.json",
    include_defaults: bool = True,
) -> Path:
    """
    Save the effective (resolved) Pydantic config to a JSON file.

    This function saves the complete configuration with all defaults filled in,
    enabling full reproducibility of experiment runs.

    Parameters
    ----------
    config : BaseModel
        Pydantic configuration model (e.g., MFGSolverConfig).
    output_dir : str | Path
        Directory to save the config file.
    filename : str, optional
        Output filename. Default "resolved_config.json".
    include_defaults : bool, optional
        If True, include fields with default values. Default True.

    Returns
    -------
    Path
        Path to the saved config file.

    Examples
    --------
    >>> from mfgarchon.config import MFGSolverConfig, PicardConfig
    >>> from mfgarchon.config.bridge import save_effective_config
    >>>
    >>> config = MFGSolverConfig(picard=PicardConfig(tolerance=1e-8, max_iterations=200))
    >>> path = save_effective_config(config, "results/experiment_001")
    >>> print(f"Config saved to {path}")

    Notes
    -----
    The output JSON includes all fields, including those with default values,
    ensuring the exact configuration can be reconstructed later.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    config_path = output_path / filename

    # Export to JSON-serializable dict
    if include_defaults:
        config_dict = config.model_dump(mode="json")
    else:
        config_dict = config.model_dump(mode="json", exclude_defaults=True)

    with open(config_path, "w") as f:
        json.dump(config_dict, f, indent=2)

    return config_path


def load_effective_config[T: BaseModel](
    config_path: str | Path,
    pydantic_cls: type[T],
) -> T:
    """
    Load a previously saved effective config from JSON.

    Parameters
    ----------
    config_path : str | Path
        Path to the JSON config file.
    pydantic_cls : type[T]
        Target Pydantic model class.

    Returns
    -------
    T
        Validated Pydantic model instance.

    Examples
    --------
    >>> from mfgarchon.config import MFGSolverConfig
    >>> from mfgarchon.config.bridge import load_effective_config
    >>>
    >>> config = load_effective_config(
    ...     "results/experiment_001/resolved_config.json",
    ...     MFGSolverConfig
    ... )
    """
    with open(config_path) as f:
        config_dict = json.load(f)

    return pydantic_cls.model_validate(config_dict)


__all__ = [
    "bridge_to_pydantic",
    "save_effective_config",
    "load_effective_config",
]
