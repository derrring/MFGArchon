"""
Core solver configuration classes.

This module provides the unified solver configuration system for MFGarchon.
Configurations specify HOW to solve problems (algorithmic choices), not WHAT
problems to solve (mathematical definitions - those are MFGProblem instances).

Key Principle
-------------
- MFGProblem (Python code): Mathematical definition (g, H, ρ₀, geometry)
- SolverConfig (YAML/Python): Algorithmic choices (method, tolerance, backend)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

if TYPE_CHECKING:
    from pathlib import Path


class BaseConfig(BaseModel):
    """Base for every configuration model in this package.

    Issue #1766. This was `BaseConfig = BaseModel`, a bare alias exported in `__all__` that
    nothing inherited and that carried no policy -- a name promising a config base and
    delivering Pydantic's.

    It exists now to own one thing: **unknown fields are an error**. Pydantic ignores extras by
    default, so `PicardConfig(anderson_acceleration=True)` constructed cleanly, dropped the
    field, and left `anderson_memory` at 0 -- Anderson off while the caller had just asked for
    it, with nothing raised. The API v1.0 design note taught exactly that call. A misspelled or
    obsolete field in any config was accepted and discarded.

    This is the fail-fast rule the package already applies to dead solver knobs (#1426 raises on
    `FPNetworkSolver(max_iterations=...)` rather than ignoring it); the config layer was the one
    place still accepting them silently.

    Deprecated aliases are unaffected: they are translated by `model_validator(mode="before")`
    hooks that `pop` the legacy key before validation runs, so `extra="forbid"` never sees it.
    Verified against `PicardConfig(damping_factor=0.7)`, which still warns and still sets
    `relaxation=0.7`.
    """

    model_config = ConfigDict(extra="forbid")


class LoggingConfig(BaseConfig):
    """
    Configuration for logging and progress reporting.

    Attributes
    ----------
    level : Literal["DEBUG", "INFO", "WARNING", "ERROR"]
        Logging level (default: INFO)
    progress_bar : bool
        Show progress bar during solving (default: True)
    save_intermediate : bool
        Save intermediate results during iteration (default: False)
    output_dir : str | None
        Directory for saving intermediate results (default: None)
    """

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    progress_bar: bool = True
    save_intermediate: bool = False
    output_dir: str | None = None

    @model_validator(mode="after")
    def validate_output_dir(self) -> LoggingConfig:
        """Validate that output_dir is provided if save_intermediate is True."""
        if self.save_intermediate and self.output_dir is None:
            raise ValueError("output_dir must be provided when save_intermediate is True")
        return self


class BackendConfig(BaseConfig):
    """
    Configuration for computational backend.

    Attributes
    ----------
    type : Literal["numpy", "jax", "pytorch"]
        Backend type (default: numpy)
    device : Literal["cpu", "gpu", "auto"]
        Compute device (default: cpu)
    precision : Literal["float32", "float64"]
        Floating point precision (default: float64)
    """

    type: Literal["numpy", "jax", "pytorch"] = "numpy"
    device: Literal["cpu", "gpu", "auto"] = "cpu"
    precision: Literal["float32", "float64"] = "float64"

    @model_validator(mode="after")
    def validate_device(self) -> BackendConfig:
        """Validate device compatibility with backend."""
        if self.type == "numpy" and self.device == "gpu":
            raise ValueError("NumPy backend does not support GPU device. Use JAX or PyTorch.")
        return self


class PicardConfig(BaseConfig):
    """
    Configuration for Picard (fixed-point) iteration with under-relaxation.

    Attributes
    ----------
    max_iterations : int
        Maximum number of iterations (default: 100)
    tolerance : float
        Convergence tolerance (default: 1e-6)
    relaxation : float
        Relaxation (under-relaxation) factor omega in (0, 1] for the update:
        u^{n+1} = omega * u_new + (1 - omega) * u^n.
        - 1.0: No relaxation (faster but may diverge)
        - 0.5: Moderate relaxation (balanced, default)
        - <0.3: Heavy relaxation (slower but more stable)
    relaxation_M : float | None
        Separate relaxation factor for M (None = use `relaxation` for both).
        Issue #719: Per-variable relaxation support.
    relaxation_schedule : str
        Iteration-based relaxation schedule for U: "constant", "harmonic",
        "sqrt", or "exponential". Issue #719 Phase 2.
    relaxation_schedule_M : str | None
        Separate schedule for M (None = follow U schedule).
    adaptive_relaxation : bool
        Enable error-reactive adaptive relaxation (Issue #583).
    anderson_memory : int
        Anderson acceleration memory depth (0 = disabled, default: 0)
    verbose : bool
        Print iteration progress (default: True)

    Legacy field names
    ------------------
    The fields were renamed from `damping_*` to `relaxation_*` in v0.19.1 for
    naming abstraction (`relaxation` extends cleanly to over-relaxation ω>1 if
    the range constraint is loosened in future work). Legacy names are still
    accepted with a `DeprecationWarning` via a `mode="before"` validator.

    Mapping: damping_factor -> relaxation, damping_factor_M -> relaxation_M,
    damping_schedule -> relaxation_schedule, damping_schedule_M ->
    relaxation_schedule_M, adaptive_damping -> adaptive_relaxation.
    """

    max_iterations: int = Field(default=100, ge=1)
    tolerance: float = Field(default=1e-6, gt=0)
    relaxation: float = Field(default=0.5, gt=0, le=1.0)
    relaxation_M: float | None = Field(default=None, gt=0, le=1.0)
    relaxation_schedule: Literal["constant", "harmonic", "sqrt", "exponential"] = "constant"
    relaxation_schedule_M: Literal["constant", "harmonic", "sqrt", "exponential"] | None = None
    adaptive_relaxation: bool = False
    anderson_memory: int = Field(default=0, ge=0)
    verbose: bool = True

    #: Legacy field names accepted with a DeprecationWarning (v0.19.1), mapped to the canonical
    #: name. Read by `_translate_legacy_damping_names` below AND by `config.bridge`, which must
    #: not filter these out before the validator can translate them (Issue #1766 follow-up): the
    #: bridge filters on `model_fields`, and an alias is by definition not a field. One owner.
    LEGACY_FIELD_ALIASES: ClassVar[dict[str, str]] = {
        "damping_factor": "relaxation",
        "damping_factor_M": "relaxation_M",
        "damping_schedule": "relaxation_schedule",
        "damping_schedule_M": "relaxation_schedule_M",
        "adaptive_damping": "adaptive_relaxation",
    }

    @model_validator(mode="before")
    @classmethod
    def _translate_legacy_damping_names(cls, values: Any) -> Any:
        """Accept legacy `damping_*` field names with DeprecationWarning (v0.19.1).

        Translates old names to canonical `relaxation_*` before Pydantic
        validation. Users hitting this path are on the deprecated API surface
        that will be removed per the standard 3-version deprecation window.
        """
        import warnings

        if not isinstance(values, dict):
            return values
        data = dict(values)
        for legacy, canonical in cls.LEGACY_FIELD_ALIASES.items():
            if legacy in data:
                if canonical in data:
                    raise ValueError(
                        f"PicardConfig: received both legacy '{legacy}' and canonical "
                        f"'{canonical}'. Pass only the canonical name."
                    )
                warnings.warn(
                    f"PicardConfig field '{legacy}' is deprecated since v0.19.1. Use '{canonical}' instead.",
                    DeprecationWarning,
                    stacklevel=3,
                )
                data[canonical] = data.pop(legacy)
        return data

    @model_validator(mode="after")
    def validate_anderson(self) -> PicardConfig:
        """Validate Anderson acceleration parameters."""
        if self.anderson_memory < 0:
            raise ValueError("anderson_memory must be non-negative")
        if self.anderson_memory > self.max_iterations:
            raise ValueError("anderson_memory cannot exceed max_iterations")
        return self


class MFGSolverConfig(BaseConfig):
    """
    Unified MFG solver configuration.

    This class specifies HOW to solve an MFG problem (algorithmic choices),
    not WHAT problem to solve (mathematical definition).

    The problem definition (terminal cost g, Hamiltonian H, initial density ρ₀)
    is specified via MFGProblem instances.

    Attributes
    ----------
    hjb : HJBConfig
        HJB solver configuration
    fp : FPConfig
        Fokker-Planck solver configuration
    picard : PicardConfig
        Picard iteration configuration
    backend : BackendConfig
        Computational backend configuration
    logging : LoggingConfig
        Logging configuration

    Examples
    --------
    >>> from mfgarchon.config import (
    ...     FPConfig, HJBConfig, MFGSolverConfig, ParticleConfig, PicardConfig,
    ... )

    >>> # From YAML file
    >>> config = MFGSolverConfig.from_yaml("config.yaml")

    >>> # Programmatically
    >>> config = MFGSolverConfig(
    ...     hjb=HJBConfig(method="fdm", accuracy_order=2),
    ...     fp=FPConfig(method="particle", particle=ParticleConfig(num_particles=5000)),
    ...     picard=PicardConfig(max_iterations=50, tolerance=1e-6)
    ... )

    """

    hjb: HJBConfig = Field(default_factory=lambda: HJBConfig())
    fp: FPConfig = Field(default_factory=lambda: FPConfig())
    picard: PicardConfig = Field(default_factory=PicardConfig)
    backend: BackendConfig = Field(default_factory=BackendConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    def to_yaml(self, path: str | Path) -> None:
        """
        Save configuration to YAML file.

        Parameters
        ----------
        path : str | Path
            Output file path

        Examples
        --------
        >>> config = MFGSolverConfig(...)
        >>> config.to_yaml("experiments/baseline.yaml")
        """
        from .io import save_solver_config

        save_solver_config(self, path)

    @classmethod
    def from_yaml(cls, path: str | Path) -> MFGSolverConfig:
        """
        Load configuration from YAML file.

        Parameters
        ----------
        path : str | Path
            Path to YAML configuration file

        Returns
        -------
        MFGSolverConfig
            Validated solver configuration

        Examples
        --------
        >>> config = MFGSolverConfig.from_yaml("experiments/baseline.yaml")
        """
        from .io import load_solver_config

        return load_solver_config(path)

    def model_dump_yaml(self) -> dict:
        """
        Dump configuration as dictionary suitable for YAML serialization.

        Returns
        -------
        dict
            Configuration as nested dictionary
        """
        return self.model_dump(exclude_none=True, mode="json")


# Backward compatibility alias
SolverConfig = MFGSolverConfig

# Forward references will be resolved after HJBConfig and FPConfig are imported
from .mfg_methods import FPConfig, HJBConfig  # noqa: E402

# Update forward references
MFGSolverConfig.model_rebuild()
