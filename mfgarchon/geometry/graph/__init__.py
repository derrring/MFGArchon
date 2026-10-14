"""
Graph-based geometries for MFG problems.

This module provides graph and network geometry support for MFG problems,
including network topologies and maze-based domains.
"""

# Network geometry
# Maze generation
from .maze_cellular_automata import CellularAutomataConfig, CellularAutomataGenerator
from .maze_generator import MazeAlgorithm, MazeGeometry
from .maze_hybrid import HybridMazeGenerator
from .maze_recursive_division import RecursiveDivisionConfig, RecursiveDivisionGenerator
from .maze_voronoi import VoronoiMazeGenerator
from .network_backend import (
    NetworkBackendType,
    OperationType,
    get_backend_manager,
    set_preferred_backend,
)
from .network_geometry import (
    CustomNetwork,
    GridNetwork,
    NetworkData,
    NetworkGeometry,
    NetworkType,
    RandomNetwork,
    ScaleFreeNetwork,
    compute_network_statistics,
    create_network,
)

# Backward compatibility alias
BaseNetworkGeometry = NetworkGeometry

# Add maze_ prefixes for clarity
maze_Geometry = MazeGeometry
maze_Algorithm = MazeAlgorithm
maze_CellularAutomataConfig = CellularAutomataConfig
maze_CellularAutomataGenerator = CellularAutomataGenerator
maze_HybridGenerator = HybridMazeGenerator
maze_RecursiveDivisionConfig = RecursiveDivisionConfig
maze_RecursiveDivisionGenerator = RecursiveDivisionGenerator
maze_VoronoiGenerator = VoronoiMazeGenerator

__all__ = [
    # Network geometry
    "NetworkType",
    "NetworkData",
    "NetworkGeometry",
    "BaseNetworkGeometry",  # Backward compatibility alias
    "CustomNetwork",
    "GridNetwork",
    "RandomNetwork",
    "ScaleFreeNetwork",
    "compute_network_statistics",
    "create_network",
    # Network backend
    "NetworkBackendType",
    "OperationType",
    "get_backend_manager",
    "set_preferred_backend",
    # Maze geometry (primary name)
    "MazeGeometry",
    "MazeAlgorithm",
    "CellularAutomataConfig",
    "CellularAutomataGenerator",
    "HybridMazeGenerator",
    "RecursiveDivisionConfig",
    "RecursiveDivisionGenerator",
    "VoronoiMazeGenerator",
    # Maze with maze_ prefix
    "maze_Geometry",
    "maze_Algorithm",
    "maze_CellularAutomataConfig",
    "maze_CellularAutomataGenerator",
    "maze_HybridGenerator",
    "maze_RecursiveDivisionConfig",
    "maze_RecursiveDivisionGenerator",
    "maze_VoronoiGenerator",
]
