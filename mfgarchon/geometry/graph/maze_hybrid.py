"""
Hybrid Maze Generation for Realistic MFG Environments

Combines multiple maze generation algorithms to create heterogeneous environments
that better model real-world spatial structures.

Real-world buildings combine multiple spatial patterns:
- Office buildings: Structured rooms (Recursive Division) + service corridors
- Museums: Irregular galleries (Voronoi) + garden courtyards (Cellular Automata)
- Shopping malls: Store spaces + connecting hallways
- Hospitals: Large wards + narrow corridors

This enables novel MFG research:
- Zone-specific behavior analysis
- Heterogeneous Nash equilibria
- Multi-scale planning problems
- Realistic evacuation modeling

Mathematical Foundation:
Hybrid mazes partition the spatial domain Omega into disjoint regions R_i,
where each region employs a different maze generation algorithm A_i.
Global connectivity is ensured by adding connecting passages at region boundaries.

Author: MFGarchon Team
Date: October 2025
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


class HybridStrategy(Enum):
    """Strategy for combining maze algorithms."""

    SPATIAL_SPLIT = "spatial_split"  # Divide grid into regions
    HIERARCHICAL = "hierarchical"  # Zones within zones
    CHECKERBOARD = "checkerboard"  # Alternating pattern
    RADIAL = "radial"  # Center vs periphery
    BLENDING = "blending"  # Smooth interpolation


@dataclass
class AlgorithmSpec:
    """
    Specification for one algorithm in hybrid maze.

    Attributes:
        algorithm: Algorithm type to use
        config: Algorithm-specific configuration parameters
        region: Region identifier for hierarchical strategies (optional)
    """

    algorithm: Literal["perfect", "recursive_division", "cellular_automata", "voronoi"]
    config: dict = field(default_factory=dict)
    region: str | None = None


@dataclass
class HybridMazeConfig:
    """
    Configuration for hybrid maze generation.

    Attributes:
        rows: Number of rows in maze grid
        cols: Number of columns in maze grid
        strategy: Combination strategy to use
        algorithms: List of algorithms to combine
        blend_ratio: Ratio for spatial split (0.0-1.0), default 0.5
        split_axis: Axis for spatial split
        num_zones: Number of zones for hierarchical strategy
        radial_center: Center point for radial strategy
        seed: Random seed for reproducibility
        ensure_connectivity: Add inter-zone connections if needed
    """

    rows: int
    cols: int
    strategy: HybridStrategy
    algorithms: list[AlgorithmSpec]

    # Strategy-specific parameters
    blend_ratio: float = 0.5
    split_axis: Literal["horizontal", "vertical", "both"] = "vertical"
    num_zones: int = 4
    radial_center: tuple[int, int] | None = None

    seed: int | None = None
    ensure_connectivity: bool = True

    def __post_init__(self):
        """Validate configuration parameters."""
        if self.rows <= 0 or self.cols <= 0:
            raise ValueError("Rows and cols must be positive")

        if not 0.0 <= self.blend_ratio <= 1.0:
            raise ValueError("blend_ratio must be in [0.0, 1.0]")

        if len(self.algorithms) == 0:
            raise ValueError("Must specify at least one algorithm")

        if self.strategy == HybridStrategy.SPATIAL_SPLIT and len(self.algorithms) < 2:
            raise ValueError("SPATIAL_SPLIT requires at least 2 algorithms")

        if self.num_zones < 2:
            raise ValueError("num_zones must be at least 2")


class HybridMazeGenerator:
    """
    Generate hybrid mazes combining multiple algorithms.

    Enables realistic, heterogeneous MFG environments that better represent
    real-world spatial structures.

    Example:
        Museum with galleries (Voronoi) + gardens (Cellular Automata)

        >>> config = HybridMazeConfig(
        ...     rows=80, cols=100,
        ...     strategy=HybridStrategy.SPATIAL_SPLIT,
        ...     algorithms=[
        ...         AlgorithmSpec("voronoi", {"num_points": 12}),
        ...         AlgorithmSpec("cellular_automata", {"initial_wall_prob": 0.42})
        ...     ],
        ...     blend_ratio=0.6,
        ...     seed=42
        ... )
        >>> generator = HybridMazeGenerator(config)
        >>> maze = generator.generate()
        >>> zone_map = generator.zone_map  # Which algorithm per cell
    """

    def __init__(self, config: HybridMazeConfig):
        """
        Initialize hybrid maze generator.

        Args:
            config: Hybrid maze configuration
        """
        self.config = config
        self.maze: NDArray[np.int_] | None = None
        self.zone_map: NDArray[np.int_] | None = None
        self.rng = np.random.default_rng(config.seed)

    def generate(self, seed: int | None = None) -> NDArray[np.int_]:
        """
        Generate hybrid maze.

        Args:
            seed: Optional override for random seed

        Returns:
            Maze array where 1 = wall, 0 = passage
        """
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        if self.config.strategy == HybridStrategy.SPATIAL_SPLIT:
            maze = self._spatial_split()
        elif self.config.strategy == HybridStrategy.HIERARCHICAL:
            maze = self._hierarchical()
        elif self.config.strategy == HybridStrategy.CHECKERBOARD:
            maze = self._checkerboard()
        elif self.config.strategy == HybridStrategy.RADIAL:
            maze = self._radial()
        elif self.config.strategy == HybridStrategy.BLENDING:
            maze = self._blending()
        else:
            raise ValueError(f"Unknown strategy: {self.config.strategy}")

        self.maze = maze

        if self.config.ensure_connectivity:
            self._ensure_global_connectivity()

        return self.maze

    def _spatial_split(self) -> NDArray[np.int_]:
        """
        Divide grid spatially and apply different algorithms.

        For vertical split with blend_ratio=0.6:
        - Left 60% uses algorithms[0]
        - Right 40% uses algorithms[1]

        Returns:
            Hybrid maze array
        """
        maze = np.ones((self.config.rows, self.config.cols), dtype=np.int_)
        self.zone_map = np.zeros((self.config.rows, self.config.cols), dtype=np.int_)

        if self.config.split_axis == "vertical":
            split_col = int(self.config.cols * self.config.blend_ratio)

            # Left region
            left_maze = self._generate_region(self.config.algorithms[0], 0, self.config.rows, 0, split_col)
            maze[:, :split_col] = left_maze
            self.zone_map[:, :split_col] = 0

            # Right region
            right_maze = self._generate_region(
                self.config.algorithms[1], 0, self.config.rows, split_col, self.config.cols
            )
            maze[:, split_col:] = right_maze
            self.zone_map[:, split_col:] = 1

        elif self.config.split_axis == "horizontal":
            split_row = int(self.config.rows * self.config.blend_ratio)

            # Top region
            top_maze = self._generate_region(self.config.algorithms[0], 0, split_row, 0, self.config.cols)
            maze[:split_row, :] = top_maze
            self.zone_map[:split_row, :] = 0

            # Bottom region
            bottom_maze = self._generate_region(
                self.config.algorithms[1], split_row, self.config.rows, 0, self.config.cols
            )
            maze[split_row:, :] = bottom_maze
            self.zone_map[split_row:, :] = 1

        elif self.config.split_axis == "both":
            # Quadrant split
            split_row = int(self.config.rows * 0.5)
            split_col = int(self.config.cols * 0.5)

            # Top-left
            tl_maze = self._generate_region(self.config.algorithms[0], 0, split_row, 0, split_col)
            maze[:split_row, :split_col] = tl_maze
            self.zone_map[:split_row, :split_col] = 0

            # Top-right
            tr_maze = self._generate_region(
                self.config.algorithms[1] if len(self.config.algorithms) > 1 else self.config.algorithms[0],
                0,
                split_row,
                split_col,
                self.config.cols,
            )
            maze[:split_row, split_col:] = tr_maze
            self.zone_map[:split_row, split_col:] = 1

            # Bottom-left
            bl_maze = self._generate_region(
                self.config.algorithms[2] if len(self.config.algorithms) > 2 else self.config.algorithms[0],
                split_row,
                self.config.rows,
                0,
                split_col,
            )
            maze[split_row:, :split_col] = bl_maze
            self.zone_map[split_row:, :split_col] = 2

            # Bottom-right
            br_maze = self._generate_region(
                self.config.algorithms[3] if len(self.config.algorithms) > 3 else self.config.algorithms[0],
                split_row,
                self.config.rows,
                split_col,
                self.config.cols,
            )
            maze[split_row:, split_col:] = br_maze
            self.zone_map[split_row:, split_col:] = 3

        return maze

    def _hierarchical(self) -> NDArray[np.int_]:
        """
        Use one algorithm for zones, others within zones.

        Implementation plan:
        1. Generate zone structure (e.g., Voronoi with large rooms)
        2. For each zone, apply specified sub-algorithm
        3. Connect zones at boundaries

        Returns:
            Hybrid maze array
        """
        raise NotImplementedError("HIERARCHICAL strategy not yet implemented")

    def _checkerboard(self) -> NDArray[np.int_]:
        """
        Alternating algorithm selection in grid pattern.

        Returns:
            Hybrid maze array
        """
        raise NotImplementedError("CHECKERBOARD strategy not yet implemented")

    def _radial(self) -> NDArray[np.int_]:
        """
        Different algorithms based on distance from center.

        Returns:
            Hybrid maze array
        """
        raise NotImplementedError("RADIAL strategy not yet implemented")

    def _blending(self) -> NDArray[np.int_]:
        """
        Smooth interpolation between two maze types using gradient masks.

        Returns:
            Hybrid maze array
        """
        raise NotImplementedError("BLENDING strategy not yet implemented")

    def _generate_region(
        self,
        spec: AlgorithmSpec,
        row_start: int,
        row_end: int,
        col_start: int,
        col_end: int,
    ) -> NDArray[np.int_]:
        """
        Generate maze for specific region using specified algorithm.

        Args:
            spec: Algorithm specification
            row_start: Starting row index
            row_end: Ending row index (exclusive)
            col_start: Starting column index
            col_end: Ending column index (exclusive)

        Returns:
            Maze array for region
        """
        region_rows = row_end - row_start
        region_cols = col_end - col_start

        # Type annotations for generator variables (different types per branch)
        from typing import Any

        config: Any
        generator: Any

        if spec.algorithm == "recursive_division":
            from mfgarchon.geometry.graph.maze_recursive_division import (
                RecursiveDivisionConfig,
                RecursiveDivisionGenerator,
            )

            config = RecursiveDivisionConfig(rows=region_rows, cols=region_cols, **spec.config)
            generator = RecursiveDivisionGenerator(config)
            return generator.generate(seed=int(self.rng.integers(0, 2**31)))

        elif spec.algorithm == "cellular_automata":
            from mfgarchon.geometry.graph.maze_cellular_automata import (
                CellularAutomataConfig,
                CellularAutomataGenerator,
            )

            config = CellularAutomataConfig(rows=region_rows, cols=region_cols, **spec.config)
            generator = CellularAutomataGenerator(config)
            return generator.generate(seed=int(self.rng.integers(0, 2**31)))

        elif spec.algorithm == "voronoi":
            from mfgarchon.geometry.graph.maze_voronoi import (
                VoronoiMazeConfig,
                VoronoiMazeGenerator,
            )

            config = VoronoiMazeConfig(rows=region_rows, cols=region_cols, **spec.config)
            generator = VoronoiMazeGenerator(config)
            return generator.generate(seed=int(self.rng.integers(0, 2**31)))

        elif spec.algorithm == "perfect":
            from mfgarchon.geometry.graph.maze_generator import (
                MazeGeometry,
            )

            # Perfect maze algorithms work on cell-based grids
            # Convert to wall-based representation
            grid_rows = region_rows // 2
            grid_cols = region_cols // 2

            generator = MazeGeometry(grid_rows, grid_cols)
            # generate() returns self to satisfy GeometryProtocol; the cell lattice is
            # the Grid it wraps. Reading get_cell off the return value was an
            # AttributeError -- the fourth broken branch of four (#1712).
            grid = generator.generate(seed=int(self.rng.integers(0, 2**31))).grid

            # Convert to wall-based array
            maze = np.ones((region_rows, region_cols), dtype=np.int_)
            for i in range(grid_rows):
                for j in range(grid_cols):
                    cell = grid.get_cell(i, j)
                    if cell is None:
                        continue
                    # Cell passage
                    maze[2 * i + 1, 2 * j + 1] = 0
                    # North passage
                    if cell.north:
                        maze[2 * i, 2 * j + 1] = 0
                    # South passage
                    if cell.south and i < grid_rows - 1:
                        maze[2 * i + 2, 2 * j + 1] = 0
                    # East passage
                    if cell.east and j < grid_cols - 1:
                        maze[2 * i + 1, 2 * j + 2] = 0
                    # West passage
                    if cell.west:
                        maze[2 * i + 1, 2 * j] = 0

            return maze

        else:
            raise ValueError(f"Unknown algorithm: {spec.algorithm}")

    def _ensure_global_connectivity(self):
        """
        Add connections between zones to ensure global connectivity.

        Uses flood fill to detect disconnected regions and adds minimal
        connecting passages at zone boundaries.
        """
        if self.maze is None or self.zone_map is None:
            return

        # Find disconnected regions using flood fill
        visited = np.zeros_like(self.maze, dtype=bool)
        regions = []

        for i in range(self.config.rows):
            for j in range(self.config.cols):
                if self.maze[i, j] == 0 and not visited[i, j]:
                    # Start new region
                    region = self._flood_fill(i, j, visited)
                    regions.append(region)

        # If only one region, already connected
        if len(regions) <= 1:
            return

        # Connect regions by adding doors at zone boundaries
        for region_idx in range(1, len(regions)):
            self._connect_regions(regions[0], regions[region_idx])

    def _flood_fill(self, start_row: int, start_col: int, visited: NDArray[np.bool_]) -> list[tuple[int, int]]:
        """
        Flood fill to find connected region.

        Args:
            start_row: Starting row
            start_col: Starting column
            visited: Visited cells tracker

        Returns:
            List of (row, col) coordinates in region
        """
        if self.maze is None:
            return []

        region = []
        stack = [(start_row, start_col)]

        while stack:
            row, col = stack.pop()

            if row < 0 or row >= self.config.rows or col < 0 or col >= self.config.cols:
                continue

            if visited[row, col] or self.maze[row, col] == 1:
                continue

            visited[row, col] = True
            region.append((row, col))

            # Add neighbors
            stack.extend([(row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)])

        return region

    def _connect_regions(self, region1: list[tuple[int, int]], region2: list[tuple[int, int]]) -> None:
        """
        Connect two regions by adding a door at the closest boundary.

        Args:
            region1: First region coordinates
            region2: Second region coordinates
        """
        if self.maze is None:
            return

        # Find closest pair of cells across regions
        min_dist = float("inf")
        best_pair = None

        for r1, c1 in region1:
            for r2, c2 in region2:
                dist = abs(r1 - r2) + abs(c1 - c2)
                if dist < min_dist:
                    min_dist = dist
                    best_pair = ((r1, c1), (r2, c2))

        if best_pair is None:
            return

        (r1, c1), (r2, c2) = best_pair

        # Create straight path between the two points
        if r1 == r2:
            # Horizontal connection
            start_col = min(c1, c2)
            end_col = max(c1, c2)
            self.maze[r1, start_col : end_col + 1] = 0
        elif c1 == c2:
            # Vertical connection
            start_row = min(r1, r2)
            end_row = max(r1, r2)
            self.maze[start_row : end_row + 1, c1] = 0
        else:
            # L-shaped connection
            self.maze[r1, min(c1, c2) : max(c1, c2) + 1] = 0
            self.maze[min(r1, r2) : max(r1, r2) + 1, c2] = 0


def create_museum_hybrid(rows: int = 80, cols: int = 100, seed: int | None = None) -> HybridMazeConfig:
    """
    Create museum configuration: Voronoi galleries + CA gardens.

    Args:
        rows: Number of rows
        cols: Number of columns
        seed: Random seed

    Returns:
        Museum hybrid maze configuration
    """
    return HybridMazeConfig(
        rows=rows,
        cols=cols,
        strategy=HybridStrategy.SPATIAL_SPLIT,
        algorithms=[
            AlgorithmSpec("voronoi", {"num_points": 12, "relaxation_iterations": 2}),
            AlgorithmSpec("cellular_automata", {"initial_wall_prob": 0.42, "num_iterations": 6}),
        ],
        blend_ratio=0.6,
        split_axis="vertical",
        seed=seed,
    )


def create_office_hybrid(rows: int = 80, cols: int = 100, seed: int | None = None) -> HybridMazeConfig:
    """
    Create office configuration: Recursive Division rooms + Perfect Maze corridors.

    Args:
        rows: Number of rows
        cols: Number of columns
        seed: Random seed

    Returns:
        Office hybrid maze configuration
    """
    return HybridMazeConfig(
        rows=rows,
        cols=cols,
        strategy=HybridStrategy.SPATIAL_SPLIT,
        algorithms=[
            AlgorithmSpec("recursive_division", {"min_room_width": 5, "min_room_height": 5}),
            AlgorithmSpec("perfect", {}),
        ],
        blend_ratio=0.7,
        split_axis="horizontal",
        seed=seed,
    )


def create_campus_hybrid(rows: int = 120, cols: int = 120, seed: int | None = None) -> HybridMazeConfig:
    """
    Create campus configuration: Four quadrants with different structures.

    Args:
        rows: Number of rows
        cols: Number of columns
        seed: Random seed

    Returns:
        Campus hybrid maze configuration
    """
    return HybridMazeConfig(
        rows=rows,
        cols=cols,
        strategy=HybridStrategy.SPATIAL_SPLIT,
        algorithms=[
            AlgorithmSpec("recursive_division", {"min_room_width": 6}),  # NW: Offices
            AlgorithmSpec("voronoi", {"num_points": 10}),  # NE: Labs
            AlgorithmSpec("perfect", {}),  # SW: Corridors
            AlgorithmSpec("cellular_automata", {"initial_wall_prob": 0.40}),  # SE: Gardens
        ],
        split_axis="both",
        seed=seed,
    )
