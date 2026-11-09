"""Data handling utilities for MFG experiments.

This module provides:
- Data recording with pluggable backends (NPZ, HDF5, future: Parquet, DB)
- Data validation utilities
- Polars integration for large-scale analysis (optional)

Usage:
    from mfgarchon.utils.data import create_recorder, load_experiment

    # Record experiment with HDF5 backend (incremental saves)
    with create_recorder('results/exp.h5', config) as rec:
        rec.save_grid(X=X, Y=Y)
        for k in range(max_iter):
            rec.save_iteration(k, U=U, M=M, error_U=err)
        rec.save_final(U, M, converged=True)

    # Load experiment data
    data = load_experiment('results/exp.h5')
"""

from __future__ import annotations

# Data recorder with pluggable backends
from .recorder import (
    BaseRecorder,
    DataRecorderProtocol,
    HDF5Recorder,
    NPZRecorder,
    create_recorder,
    get_convergence_history,
    inspect_experiment,
    load_experiment,
)

# Re-export validation utilities
from .validation import (
    safe_solution_return,
    validate_convergence_parameters,
    validate_mfg_solution,
    validate_solution_array,
)

__all__ = [
    # Recorder protocol and base
    "DataRecorderProtocol",
    "BaseRecorder",
    # Backend implementations
    "NPZRecorder",
    "HDF5Recorder",
    # Factory and utilities
    "create_recorder",
    "load_experiment",
    "get_convergence_history",
    "inspect_experiment",
    # Validation utilities
    "safe_solution_return",
    "validate_convergence_parameters",
    "validate_mfg_solution",
    "validate_solution_array",
]

# Polars integration: importable without polars; POLARS_AVAILABLE, owned there, says whether polars is installed
from .polars_integration import (  # noqa: F401
    POLARS_AVAILABLE,
    MFGDataFrame,
    benchmark_polars_vs_pandas,
    create_data_exporter,
    create_mfg_dataframe,
    create_parameter_sweep_analyzer,
    create_time_series_analyzer,
)

__all__.extend(
    [
        "MFGDataFrame",
        "benchmark_polars_vs_pandas",
        "create_data_exporter",
        "create_mfg_dataframe",
        "create_parameter_sweep_analyzer",
        "create_time_series_analyzer",
    ]
)
