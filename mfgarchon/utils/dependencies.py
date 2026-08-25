"""
Optional dependency checking with helpful error messages.

Provides utilities for checking optional dependencies and displaying
clear, actionable error messages when dependencies are missing.
"""

import functools
import importlib.metadata
from typing import Literal

DependencyGroup = Literal["core", "neural", "reinforcement", "numerical", "performance", "visualization", "gpu", "all"]

# Mapping of packages to installation information
DEPENDENCY_MAP = {
    # Core dependencies (always required)
    "numpy": {
        "install_group": "core",
        "install_cmd": "pip install mfgarchon",
        "alternative": "pip install numpy",
        "description": "Array operations and numerical computing",
        "required": True,
    },
    "scipy": {
        "install_group": "core",
        "install_cmd": "pip install mfgarchon",
        "alternative": "pip install scipy",
        "description": "Scientific computing and optimization",
        "required": True,
    },
    "matplotlib": {
        "install_group": "core",
        "install_cmd": "pip install mfgarchon",
        "alternative": "pip install matplotlib",
        "description": "Plotting and visualization",
        "required": True,
    },
    # Neural networks
    "torch": {
        "install_group": "neural",
        "install_cmd": "pip install mfgarchon[neural]",
        "alternative": "pip install torch",
        "description": "Deep learning backends, RL algorithms, GPU acceleration",
        "required": False,
    },
    "jax": {
        "install_group": "performance",
        "install_cmd": "pip install mfgarchon[performance]",
        "alternative": "pip install jax jaxlib",
        "description": "JAX backend, autodiff, GPU kernels",
        "required": False,
    },
    # Reinforcement learning
    "gymnasium": {
        "install_group": "reinforcement",
        "install_cmd": "pip install mfgarchon[reinforcement]",
        "alternative": "pip install gymnasium",
        "description": "RL environments, MFG games",
        "required": False,
    },
    # Graph/network
    "igraph": {
        "install_group": "numerical",
        "install_cmd": "pip install mfgarchon[numerical]",
        "alternative": "pip install igraph",
        "description": "Network MFG, graph algorithms",
        "required": False,
    },
    "networkx": {
        "install_group": "numerical",
        "install_cmd": "pip install mfgarchon[numerical]",
        "alternative": "pip install networkx",
        "description": "Network visualization, graph analysis",
        "required": False,
    },
    # Visualization
    "plotly": {
        "install_group": "visualization",
        "install_cmd": "pip install mfgarchon[visualization]",
        "alternative": "pip install plotly",
        "description": "Interactive visualizations, 3D plots",
        "required": False,
    },
    "bokeh": {
        "install_group": "visualization",
        "install_cmd": "pip install mfgarchon[visualization]",
        "alternative": "pip install bokeh",
        "description": "Interactive plots, dashboards",
        "required": False,
    },
    # Performance
    "polars": {
        "install_group": "performance",
        "install_cmd": "pip install mfgarchon[performance]",
        "alternative": "pip install polars",
        "description": "Fast data analysis, parameter sweeps",
        "required": False,
    },
    "numba": {
        "install_group": "performance",
        "install_cmd": "pip install mfgarchon[performance]",
        "alternative": "pip install numba",
        "description": "JIT compilation for numerical code",
        "required": False,
    },
    # GPU
    "cupy": {
        "install_group": "gpu",
        "install_cmd": "pip install mfgarchon[gpu]",
        "alternative": "pip install cupy-cuda11x",
        "description": "GPU arrays and operations",
        "required": False,
    },
    # Progress/workflow
    "rich": {
        "install_group": "core",
        "install_cmd": "pip install mfgarchon",
        "alternative": "pip install rich",
        "description": "Progress bars and console output",
        "required": True,
    },
}


def is_available(package: str) -> bool:
    """
    Check if a package is available without raising an error.

    Args:
        package: Package name to check

    Returns:
        True if package can be imported, False otherwise

    Example:
        >>> if is_available('torch'):
        ...     import torch
        ...     # Use PyTorch features
    """
    try:
        __import__(package)
        return True
    except ImportError:
        return False


def check_dependency(
    package: str,
    purpose: str | None = None,
    install_command: str | None = None,
    raise_on_missing: bool = True,
) -> bool:
    """
    Check if an optional dependency is available.

    Args:
        package: Package name to check (e.g., 'torch', 'jax')
        purpose: Optional purpose description for error message
        install_command: Optional custom install command (overrides default)
        raise_on_missing: If True, raise ImportError when missing. If False, return bool

    Returns:
        True if package is available, False if missing (when raise_on_missing=False)

    Raises:
        ImportError: If package is missing and raise_on_missing=True

    Example:
        >>> # Check with error on missing
        >>> check_dependency('torch', purpose='neural network solvers')
        True  # If torch is installed

        >>> # Check without raising (return bool)
        >>> has_torch = check_dependency('torch', raise_on_missing=False)
        >>> if has_torch:
        ...     import torch

        >>> # Custom install command
        >>> check_dependency('torch', install_command='conda install pytorch')
    """
    if is_available(package):
        return True

    if not raise_on_missing:
        return False

    # Build helpful error message
    if package in DEPENDENCY_MAP:
        info = DEPENDENCY_MAP[package]
        purpose_msg = f" for {purpose}" if purpose else f" ({info['description']})"

        install_cmd = install_command or info["install_cmd"]
        alternative_cmd = info["alternative"]

        msg = f"{package} required{purpose_msg}.\n\nInstall options:\n  1. {install_cmd}\n  2. {alternative_cmd}"
    else:
        # Unknown package, provide basic message
        purpose_msg = f" for {purpose}" if purpose else ""
        install_cmd = install_command or f"pip install {package}"
        msg = f"{package} required{purpose_msg} but not installed.\nInstall with: {install_cmd}"

    raise ImportError(msg)


def require_dependencies(*packages: str, purpose: str | None = None, feature: str | None = None):
    """
    Decorator to require dependencies for a function or class.

    Args:
        *packages: Package names required
        purpose: Optional purpose description for error messages
        feature: Deprecated alias for purpose (for backward compatibility)

    Returns:
        Decorator function

    Example:
        >>> @require_dependencies('torch', purpose='Mean Field DDPG')
        ... def create_ddpg_agent(env):
        ...     import torch
        ...     # Implementation uses PyTorch

        >>> @require_dependencies('jax', 'jaxlib')
        ... class JAXBackend:
        ...     # Implementation uses JAX
    """
    # Handle backward compatibility: feature is an alias for purpose
    actual_purpose = purpose or feature

    def decorator(func_or_class):
        @functools.wraps(func_or_class)
        def wrapper(*args, **kwargs):
            # Check all required dependencies
            for pkg in packages:
                check_dependency(pkg, purpose=actual_purpose)
            return func_or_class(*args, **kwargs)

        return wrapper

    return decorator


def get_available_features() -> dict[str, bool]:
    """
    Get availability status of all optional features.

    Returns:
        Dictionary mapping feature names to availability status

    Example:
        >>> features = get_available_features()
        >>> if features['pytorch']:
        ...     # Use PyTorch features
        >>> if features['jax']:
        ...     # Use JAX features
    """
    return {
        "pytorch": is_available("torch"),
        "jax": is_available("jax"),
        "gymnasium": is_available("gymnasium"),
        "igraph": is_available("igraph"),
        "networkx": is_available("networkx"),
        "plotly": is_available("plotly"),
        "bokeh": is_available("bokeh"),
        "polars": is_available("polars"),
        "numba": is_available("numba"),
        "cupy": is_available("cupy"),
    }


def get_package_version(package: str) -> str | None:
    """
    Get installed package version.

    Args:
        package: Package name

    Returns:
        Version string if installed, None otherwise
    """
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def show_optional_features() -> None:
    """
    Display status of all optional features with version information.

    Prints a formatted table showing which dependencies are installed,
    their versions, and installation instructions for missing packages.

    Example:
        >>> import mfgarchon
        >>> mfgarchon.show_optional_features()
        MFGarchon Optional Features Status
        ================================================================================

        Core (always available):
          ✓ numpy 1.24.3 - Array operations and numerical computing
          ✓ scipy 1.10.1 - Scientific computing and optimization
          ✓ matplotlib 3.7.1 - Plotting and visualization

        Neural Methods:
          ✓ torch 2.0.1 - Deep learning backends, RL algorithms, GPU acceleration
          ✗ jax - JAX backend, autodiff, GPU kernels (pip install mfgarchon[performance])

        Visualization:
          ✓ plotly 5.14.1 - Interactive visualizations, 3D plots
          ✗ bokeh - Interactive plots, dashboards (pip install mfgarchon[visualization])

        Performance:
          ✓ numba 0.57.0 - JIT compilation for numerical code
          ✗ cupy - GPU arrays and operations (pip install mfgarchon[gpu])
        ...
    """

    print("MFGarchon Optional Features Status")
    print("=" * 80)
    print()

    # Group packages by category
    categories = {
        "Core (always available)": [],
        "Neural Methods": [],
        "Reinforcement Learning": [],
        "Graph/Network": [],
        "Visualization": [],
        "Performance": [],
        "GPU Acceleration": [],
        "Workflow": [],
    }

    for pkg_name, pkg_info in DEPENDENCY_MAP.items():
        group = pkg_info["install_group"]

        # Map install_group to display category
        if group == "core":
            if pkg_name in ["rich"]:
                category = "Workflow"
            else:
                category = "Core (always available)"
        elif group == "neural":
            category = "Neural Methods"
        elif group == "reinforcement":
            category = "Reinforcement Learning"
        elif group == "numerical":
            category = "Graph/Network"
        elif group == "visualization":
            category = "Visualization"
        elif group == "performance":
            if pkg_name == "jax":
                category = "Neural Methods"
            else:
                category = "Performance"
        elif group == "gpu":
            category = "GPU Acceleration"
        else:
            continue

        version = get_package_version(pkg_name)
        available = version is not None
        description = pkg_info["description"]

        if available:
            status_line = f"  ✓ {pkg_name} {version} - {description}"
        else:
            install_hint = pkg_info["install_cmd"]
            status_line = f"  ✗ {pkg_name} - {description} ({install_hint})"

        categories[category].append(status_line)

    # Print each category
    for category, items in categories.items():
        if items:
            print(f"{category}:")
            for item in sorted(items):
                print(item)
            print()

    print("=" * 80)
    print("Installation Options:")
    print("  All features:     pip install mfgarchon[all]")
    print("  Neural networks:  pip install mfgarchon[neural]")
    print("  Reinforcement:    pip install mfgarchon[reinforcement]")
    print("  Visualization:    pip install mfgarchon[visualization]")
    print("  Performance:      pip install mfgarchon[performance]")
    print("  GPU support:      pip install mfgarchon[gpu]")
    print("=" * 80)


# Pre-check common dependencies for module-level flags
TORCH_AVAILABLE = is_available("torch")
JAX_AVAILABLE = is_available("jax")
GYMNASIUM_AVAILABLE = is_available("gymnasium")
IGRAPH_AVAILABLE = is_available("igraph")
NETWORKX_AVAILABLE = is_available("networkx")
PLOTLY_AVAILABLE = is_available("plotly")
BOKEH_AVAILABLE = is_available("bokeh")
POLARS_AVAILABLE = is_available("polars")
NUMBA_AVAILABLE = is_available("numba")
CUPY_AVAILABLE = is_available("cupy")
