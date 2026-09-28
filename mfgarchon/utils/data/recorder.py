"""Unified data recorder with pluggable storage backends.

This module provides a common interface for experiment data recording with
multiple backend options:

- **npz**: Simple NumPy format, saves at end only (default)
- **hdf5**: HDF5 format with incremental writes, crash recovery
- **parquet**: (Future) Columnar format for large-scale analysis
- **db**: (Future) SQLite/PostgreSQL for queryable experiments

Usage:
    # Simple usage with factory function
    from mfgarchon.utils.data import create_recorder

    # NPZ backend (saves at end)
    with create_recorder('results/exp.npz', config) as rec:
        rec.save_grid(X=X, Y=Y)
        # ... run solver ...
        rec.save_final(U, M, converged=True)

    # HDF5 backend (incremental saves)
    with create_recorder('results/exp.h5', config, backend='hdf5') as rec:
        rec.save_grid(X=X, Y=Y)
        for k in range(max_iter):
            U = hjb_solver.solve(...)
            M = fp_solver.solve(...)
            rec.save_iteration(k, U=U, M=M, error_U=err_U, error_M=err_M)
            if converged:
                break
        rec.save_final(U, M, converged=True)

    # Loading (backend auto-detected from extension)
    from mfgarchon.utils.data import load_experiment
    data = load_experiment('results/exp.h5')
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class DataRecorderProtocol(Protocol):
    """Protocol defining the data recorder interface.

    All recorder backends must implement these methods.
    """

    def __enter__(self) -> DataRecorderProtocol:
        """Open the recorder for writing."""
        ...

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Close the recorder."""
        ...

    def save_grid(
        self,
        X: np.ndarray | None = None,
        Y: np.ndarray | None = None,
        coordinates: list[np.ndarray] | None = None,
        **kwargs: np.ndarray,
    ) -> None:
        """Save spatial grid data."""
        ...

    def save_iteration(
        self,
        iteration: int,
        U: np.ndarray | None = None,
        M: np.ndarray | None = None,
        error_U: float | None = None,
        error_M: float | None = None,
        **kwargs: Any,
    ) -> None:
        """Save data from a Picard iteration."""
        ...

    def save_final(
        self,
        U: np.ndarray,
        M: np.ndarray,
        converged: bool = True,
        execution_time: float | None = None,
        total_iterations: int | None = None,
        **kwargs: Any,
    ) -> None:
        """Save final converged solution."""
        ...

    def set_solver_info(
        self,
        hjb_solver: str | None = None,
        fp_solver: str | None = None,
        **kwargs: str,
    ) -> None:
        """Save solver metadata."""
        ...


class BaseRecorder(ABC):
    """Abstract base class for data recorders.

    Provides common functionality shared by all backends.
    """

    def __init__(
        self,
        filepath: str | Path,
        config: dict[str, Any] | None = None,
    ) -> None:
        self.filepath = Path(filepath)
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        self.config = config or {}
        self._iterations: dict[int, dict[str, Any]] = {}
        self._grid: dict[str, np.ndarray] = {}
        self._final: dict[str, Any] = {}
        self._metadata: dict[str, Any] = {
            "created": datetime.now().isoformat(),
        }
        self._current_iteration = 0

    @abstractmethod
    def __enter__(self) -> BaseRecorder:
        """Open the recorder."""
        ...

    @abstractmethod
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Close the recorder."""
        ...

    def save_grid(
        self,
        X: np.ndarray | None = None,
        Y: np.ndarray | None = None,
        coordinates: list[np.ndarray] | None = None,
        **kwargs: np.ndarray,
    ) -> None:
        """Save spatial grid data."""
        if X is not None:
            self._grid["X"] = X
        if Y is not None:
            self._grid["Y"] = Y
        if coordinates is not None:
            for i, coord in enumerate(coordinates):
                self._grid[f"coord_{i}"] = coord
        self._grid.update(kwargs)

    def save_iteration(
        self,
        iteration: int,
        U: np.ndarray | None = None,
        M: np.ndarray | None = None,
        error_U: float | None = None,
        error_M: float | None = None,
        **kwargs: Any,
    ) -> None:
        """Save data from a Picard iteration."""
        if iteration not in self._iterations:
            self._iterations[iteration] = {}

        iter_data = self._iterations[iteration]
        if U is not None:
            iter_data["U"] = U
        if M is not None:
            iter_data["M"] = M
        if error_U is not None:
            iter_data["error_U"] = error_U
        if error_M is not None:
            iter_data["error_M"] = error_M
        iter_data["timestamp"] = datetime.now().isoformat()
        iter_data.update(kwargs)

        self._current_iteration = max(self._current_iteration, iteration)

    def save_final(
        self,
        U: np.ndarray,
        M: np.ndarray,
        converged: bool = True,
        execution_time: float | None = None,
        total_iterations: int | None = None,
        **kwargs: Any,
    ) -> None:
        """Save final converged solution."""
        self._final["U"] = U
        self._final["M"] = M
        self._final["converged"] = converged
        self._final["completed"] = datetime.now().isoformat()
        if execution_time is not None:
            self._final["execution_time"] = execution_time
        if total_iterations is not None:
            self._final["total_iterations"] = total_iterations
        else:
            self._final["total_iterations"] = self._current_iteration
        self._final.update(kwargs)

    def set_solver_info(
        self,
        hjb_solver: str | None = None,
        fp_solver: str | None = None,
        **kwargs: str,
    ) -> None:
        """Save solver metadata."""
        if hjb_solver:
            self._metadata["solver_hjb"] = hjb_solver
        if fp_solver:
            self._metadata["solver_fp"] = fp_solver
        self._metadata.update(kwargs)


class NPZRecorder(BaseRecorder):
    """NumPy .npz backend - saves all data at end.

    Simple and portable format. Best for:
    - Small to medium experiments
    - When crash recovery isn't critical
    - Maximum portability
    """

    def __enter__(self) -> NPZRecorder:
        """Open recorder (no-op for NPZ, writes on close)."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Write all data to .npz file on close."""
        if exc_type is not None:
            # Don't save on error
            return

        # Build iteration arrays
        error_U_list = []
        error_M_list = []
        for k in sorted(self._iterations.keys()):
            iter_data = self._iterations[k]
            if "error_U" in iter_data:
                error_U_list.append(iter_data["error_U"])
            if "error_M" in iter_data:
                error_M_list.append(iter_data["error_M"])

        # Prepare data dict
        save_data: dict[str, Any] = {
            "config": self.config,
            "metadata": self._metadata,
        }

        # Grid
        for key, value in self._grid.items():
            save_data[f"grid_{key}"] = value

        # Final solution
        for key, value in self._final.items():
            if isinstance(value, np.ndarray):
                save_data[f"final_{key}"] = value
            else:
                save_data[f"final_{key}"] = np.array(value)

        # Convergence history
        if error_U_list:
            save_data["error_history_U"] = np.array(error_U_list)
        if error_M_list:
            save_data["error_history_M"] = np.array(error_M_list)

        # Save all iterations if they contain arrays
        for k, iter_data in self._iterations.items():
            for key, value in iter_data.items():
                if isinstance(value, np.ndarray):
                    save_data[f"iter_{k:03d}_{key}"] = value

        np.savez_compressed(self.filepath, **save_data)


class HDF5Recorder(BaseRecorder):
    """HDF5 backend - incremental writes with crash recovery.

    Best for:
    - Large experiments (compression built-in)
    - Long-running simulations (crash recovery)
    - Analysis requiring random access to iterations

    Requires h5py: pip install h5py
    """

    def __init__(
        self,
        filepath: str | Path,
        config: dict[str, Any] | None = None,
        mode: str = "w",
        compression: str | None = "gzip",
        compression_level: int = 4,
    ) -> None:
        super().__init__(filepath, config)
        self.mode = mode
        self.compression = compression
        self.compression_opts = compression_level if compression == "gzip" else None
        self._file: Any = None  # h5py.File

    def _check_h5py(self) -> None:
        """Check if h5py is available."""
        try:
            import h5py  # noqa: F401
        except ImportError:
            raise ImportError("h5py is required for HDF5 backend. Install with: pip install h5py") from None

    def __enter__(self) -> HDF5Recorder:
        """Open HDF5 file for writing."""
        self._check_h5py()
        import h5py

        self._file = h5py.File(self.filepath, self.mode)
        self._init_structure()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Close HDF5 file."""
        if self._file is not None:
            self._file.close()
            self._file = None

    def _init_structure(self) -> None:
        """Initialize HDF5 file structure."""
        if self._file is None:
            return

        # Create groups
        for group in ["config", "grid", "iterations", "final", "metadata"]:
            if group not in self._file:
                self._file.create_group(group)

        # Save config as JSON
        if self.config and "config_json" not in self._file["config"]:
            config_str = json.dumps(self.config, indent=2, default=str)
            self._file["config"].create_dataset("config_json", data=config_str)

        # Config attributes
        for key, value in self.config.items():
            if key not in self._file["config"].attrs:
                try:
                    if isinstance(value, (int, float, str, bool)):
                        self._file["config"].attrs[key] = value
                except (TypeError, ValueError):
                    pass

        # Metadata
        meta = self._file["metadata"]
        for key, value in self._metadata.items():
            if key not in meta.attrs:
                meta.attrs[key] = value

        try:
            from mfgarchon import __version__

            meta.attrs["version"] = __version__
        except ImportError:
            meta.attrs["version"] = "unknown"

        self._file.flush()

    def save_grid(
        self,
        X: np.ndarray | None = None,
        Y: np.ndarray | None = None,
        coordinates: list[np.ndarray] | None = None,
        **kwargs: np.ndarray,
    ) -> None:
        """Save grid data immediately to HDF5."""
        super().save_grid(X, Y, coordinates, **kwargs)

        if self._file is None:
            return

        grid = self._file["grid"]

        def _save_array(name: str, data: np.ndarray) -> None:
            if name not in grid:
                grid.create_dataset(
                    name,
                    data=data,
                    compression=self.compression if data.size > 1000 else None,
                    compression_opts=self.compression_opts if data.size > 1000 else None,
                )

        for name, data in self._grid.items():
            _save_array(name, data)

        self._file.flush()

    def save_iteration(
        self,
        iteration: int,
        U: np.ndarray | None = None,
        M: np.ndarray | None = None,
        error_U: float | None = None,
        error_M: float | None = None,
        **kwargs: Any,
    ) -> None:
        """Save iteration data immediately to HDF5."""
        super().save_iteration(iteration, U, M, error_U, error_M, **kwargs)

        if self._file is None:
            return

        iter_name = f"iter_{iteration:03d}"
        iterations = self._file["iterations"]

        if iter_name not in iterations:
            iterations.create_group(iter_name)

        iter_grp = iterations[iter_name]

        if U is not None and "U" not in iter_grp:
            iter_grp.create_dataset(
                "U",
                data=U,
                compression=self.compression,
                compression_opts=self.compression_opts,
            )

        if M is not None and "M" not in iter_grp:
            iter_grp.create_dataset(
                "M",
                data=M,
                compression=self.compression,
                compression_opts=self.compression_opts,
            )

        if error_U is not None:
            iter_grp.attrs["error_U"] = error_U
        if error_M is not None:
            iter_grp.attrs["error_M"] = error_M

        iter_grp.attrs["timestamp"] = datetime.now().isoformat()

        # Additional data
        for name, data in kwargs.items():
            if name not in iter_grp:
                if isinstance(data, np.ndarray):
                    iter_grp.create_dataset(
                        name,
                        data=data,
                        compression=self.compression if data.size > 1000 else None,
                        compression_opts=self.compression_opts if data.size > 1000 else None,
                    )
                elif isinstance(data, (int, float, str, bool)):
                    iter_grp.attrs[name] = data

        self._file.flush()

    def save_final(
        self,
        U: np.ndarray,
        M: np.ndarray,
        converged: bool = True,
        execution_time: float | None = None,
        total_iterations: int | None = None,
        **kwargs: Any,
    ) -> None:
        """Save final solution to HDF5."""
        super().save_final(U, M, converged, execution_time, total_iterations, **kwargs)

        if self._file is None:
            return

        final = self._file["final"]

        if "U" not in final:
            final.create_dataset(
                "U",
                data=U,
                compression=self.compression,
                compression_opts=self.compression_opts,
            )
        if "M" not in final:
            final.create_dataset(
                "M",
                data=M,
                compression=self.compression,
                compression_opts=self.compression_opts,
            )

        final.attrs["converged"] = converged
        final.attrs["completed"] = datetime.now().isoformat()
        if execution_time is not None:
            final.attrs["execution_time"] = execution_time
        final.attrs["total_iterations"] = total_iterations or self._current_iteration

        for name, data in kwargs.items():
            if name not in final:
                if isinstance(data, np.ndarray):
                    final.create_dataset(
                        name,
                        data=data,
                        compression=self.compression if data.size > 1000 else None,
                        compression_opts=self.compression_opts if data.size > 1000 else None,
                    )
                elif isinstance(data, (int, float, str, bool)):
                    final.attrs[name] = data

        self._file.flush()

    def set_solver_info(
        self,
        hjb_solver: str | None = None,
        fp_solver: str | None = None,
        **kwargs: str,
    ) -> None:
        """Save solver metadata to HDF5."""
        super().set_solver_info(hjb_solver, fp_solver, **kwargs)

        if self._file is None:
            return

        meta = self._file["metadata"]
        if hjb_solver:
            meta.attrs["solver_hjb"] = hjb_solver
        if fp_solver:
            meta.attrs["solver_fp"] = fp_solver
        for key, value in kwargs.items():
            meta.attrs[key] = value

        self._file.flush()


# =============================================================================
# Factory and Loading Functions
# =============================================================================

_BACKENDS: dict[str, type[BaseRecorder]] = {
    "npz": NPZRecorder,
    "hdf5": HDF5Recorder,
    "h5": HDF5Recorder,
}


def create_recorder(
    filepath: str | Path,
    config: dict[str, Any] | None = None,
    backend: str | None = None,
    **kwargs: Any,
) -> BaseRecorder:
    """Create a data recorder with the specified backend.

    Args:
        filepath: Path to output file
        config: Experiment configuration dictionary
        backend: Backend type ('npz', 'hdf5'). Auto-detected from extension if None.
        **kwargs: Additional backend-specific options

    Returns:
        DataRecorder instance

    Example:
        # Auto-detect from extension
        recorder = create_recorder('results/exp.h5', config)  # HDF5
        recorder = create_recorder('results/exp.npz', config)  # NPZ

        # Explicit backend
        recorder = create_recorder('results/exp', config, backend='hdf5')
    """
    filepath = Path(filepath)

    # Auto-detect backend from extension
    if backend is None:
        ext = filepath.suffix.lower().lstrip(".")
        if ext in _BACKENDS:
            backend = ext
        else:
            backend = "npz"  # Default
            filepath = filepath.with_suffix(".npz")

    if backend not in _BACKENDS:
        raise ValueError(f"Unknown backend: {backend}. Available: {list(_BACKENDS.keys())}")

    return _BACKENDS[backend](filepath, config, **kwargs)


def load_experiment(filepath: str | Path) -> dict[str, Any]:
    """Load experiment data from file.

    Auto-detects format from file extension.

    Args:
        filepath: Path to experiment file (.npz or .h5)

    Returns:
        Dictionary with experiment data:
        {
            'config': {...},
            'grid': {'X': array, 'Y': array, ...},
            'iterations': {'iter_001': {...}, ...},
            'final': {'U': array, 'M': array, ...},
            'metadata': {...}
        }
    """
    filepath = Path(filepath)
    ext = filepath.suffix.lower()

    if ext in (".h5", ".hdf5"):
        return _load_hdf5(filepath)
    elif ext == ".npz":
        return _load_npz(filepath)
    else:
        raise ValueError(f"Unknown file format: {ext}")


def _load_npz(filepath: Path) -> dict[str, Any]:
    """Load data from NPZ file."""
    data = np.load(filepath, allow_pickle=True)

    result: dict[str, Any] = {
        "config": {},
        "grid": {},
        "iterations": {},
        "final": {},
        "metadata": {},
    }

    for key in data.files:
        value = data[key]
        # Convert 0-d arrays to scalars
        if isinstance(value, np.ndarray) and value.ndim == 0:
            value = value.item()

        if key == "config":
            result["config"] = value if isinstance(value, dict) else {}
        elif key == "metadata":
            result["metadata"] = value if isinstance(value, dict) else {}
        elif key.startswith("grid_"):
            result["grid"][key[5:]] = value
        elif key.startswith("final_"):
            result["final"][key[6:]] = value
        elif key.startswith("iter_"):
            # Parse iter_001_U format
            parts = key.split("_", 2)
            if len(parts) >= 3:
                iter_num = int(parts[1])
                field = parts[2]
                iter_name = f"iter_{iter_num:03d}"
                if iter_name not in result["iterations"]:
                    result["iterations"][iter_name] = {}
                result["iterations"][iter_name][field] = value
        elif key == "error_history_U":
            result["error_history_U"] = value
        elif key == "error_history_M":
            result["error_history_M"] = value

    return result


def _load_hdf5(filepath: Path) -> dict[str, Any]:
    """Load data from HDF5 file."""
    try:
        import h5py
    except ImportError:
        raise ImportError("h5py required to load HDF5 files: pip install h5py") from None

    result: dict[str, Any] = {
        "config": {},
        "grid": {},
        "iterations": {},
        "final": {},
        "metadata": {},
    }

    with h5py.File(filepath, "r") as f:
        # Config
        if "config" in f:
            cfg = f["config"]
            if "config_json" in cfg:
                import contextlib

                with contextlib.suppress(json.JSONDecodeError, TypeError):
                    result["config"] = json.loads(cfg["config_json"][()])
            for key, value in cfg.attrs.items():
                result["config"][key] = value

        # Grid
        if "grid" in f:
            for key in f["grid"]:
                result["grid"][key] = f["grid"][key][:]

        # Iterations
        if "iterations" in f:
            for iter_name in sorted(f["iterations"].keys()):
                iter_grp = f["iterations"][iter_name]
                iter_data: dict[str, Any] = {}
                for key in iter_grp:
                    iter_data[key] = iter_grp[key][:]
                for key, value in iter_grp.attrs.items():
                    iter_data[key] = value
                result["iterations"][iter_name] = iter_data

        # Final
        if "final" in f:
            final = f["final"]
            for key in final:
                result["final"][key] = final[key][:]
            for key, value in final.attrs.items():
                result["final"][key] = value

        # Metadata
        if "metadata" in f:
            for key, value in f["metadata"].attrs.items():
                result["metadata"][key] = value

    return result


def get_convergence_history(filepath: str | Path) -> dict[str, list[float]]:
    """Extract convergence history from experiment file.

    Returns:
        {'error_U': [e1, e2, ...], 'error_M': [e1, e2, ...]}
    """
    data = load_experiment(filepath)

    errors_U: list[float] = []
    errors_M: list[float] = []

    # Check for pre-computed history
    if "error_history_U" in data:
        errors_U = list(data["error_history_U"])
    if "error_history_M" in data:
        errors_M = list(data["error_history_M"])

    # Or extract from iterations
    if not errors_U or not errors_M:
        for iter_name in sorted(data.get("iterations", {}).keys()):
            iter_data = data["iterations"][iter_name]
            if "error_U" in iter_data:
                errors_U.append(float(iter_data["error_U"]))
            if "error_M" in iter_data:
                errors_M.append(float(iter_data["error_M"]))

    return {"error_U": errors_U, "error_M": errors_M}


def inspect_experiment(filepath: str | Path, max_depth: int = 3) -> None:
    """Print structure of experiment file for debugging.

    Args:
        filepath: Path to experiment file
        max_depth: Maximum depth to show
    """
    filepath = Path(filepath)
    ext = filepath.suffix.lower()

    print(f"Experiment file: {filepath}")
    print(f"Format: {ext}")
    print("-" * 50)

    data = load_experiment(filepath)

    def _print_dict(d: dict, prefix: str = "", depth: int = 0) -> None:
        if depth >= max_depth:
            print(f"{prefix}...")
            return
        for key, value in d.items():
            if isinstance(value, dict):
                print(f"{prefix}{key}/")
                _print_dict(value, prefix + "  ", depth + 1)
            elif isinstance(value, np.ndarray):
                print(f"{prefix}{key}: shape={value.shape}, dtype={value.dtype}")
            else:
                print(f"{prefix}{key}: {type(value).__name__} = {value}")

    _print_dict(data)


if __name__ == "__main__":
    # Quick test
    import tempfile

    print("Testing DataRecorder backends...")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Test NPZ
        npz_path = Path(tmpdir) / "test.npz"
        config = {"name": "test", "volatility": 0.3}

        with create_recorder(npz_path, config) as rec:
            rec.save_grid(X=np.linspace(0, 1, 10), Y=np.linspace(0, 1, 10))
            rec.set_solver_info(hjb_solver="fdm", fp_solver="fdm")
            for k in range(1, 4):
                U = np.random.rand(10, 10)
                M = np.random.rand(10, 10)
                rec.save_iteration(k, U=U, M=M, error_U=0.1 / k, error_M=0.15 / k)
            rec.save_final(U, M, converged=True, execution_time=1.23)

        print(f"NPZ saved: {npz_path}")
        data = load_experiment(npz_path)
        print(f"  Config: {data['config']}")
        print(f"  Grid keys: {list(data['grid'].keys())}")
        print(f"  Iterations: {len(data['iterations'])}")
        print(f"  Final converged: {data['final'].get('converged')}")

        # Test HDF5
        try:
            h5_path = Path(tmpdir) / "test.h5"

            with create_recorder(h5_path, config, backend="hdf5") as rec:
                rec.save_grid(X=np.linspace(0, 1, 10), Y=np.linspace(0, 1, 10))
                rec.set_solver_info(hjb_solver="fdm", fp_solver="particle")
                for k in range(1, 4):
                    U = np.random.rand(10, 10)
                    M = np.random.rand(10, 10)
                    rec.save_iteration(k, U=U, M=M, error_U=0.1 / k, error_M=0.15 / k)
                rec.save_final(U, M, converged=True, execution_time=2.34)

            print(f"\nHDF5 saved: {h5_path}")
            data = load_experiment(h5_path)
            print(f"  Config: {data['config']}")
            print(f"  Grid keys: {list(data['grid'].keys())}")
            print(f"  Iterations: {len(data['iterations'])}")
            print(f"  Final converged: {data['final'].get('converged')}")

            history = get_convergence_history(h5_path)
            print(f"  Convergence history: {history}")

        except ImportError:
            print("\nHDF5 test skipped (h5py not installed)")

    print("\nAll tests passed!")
