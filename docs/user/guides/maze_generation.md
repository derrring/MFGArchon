# [FROZEN 2026-07-30] Maze Generation Guide for MFG-RL Research

> ⛔ **FROZEN PROTOTYPE.** `mfgarchon.alg.reinforcement` is a design prototype, not under
> development — see `CLAUDE.md` § "FROZEN: alg/neural/ and alg/reinforcement/". Every status
> line in this document predates that decision. Treat everything here as illustrative of
> intent rather than as a supported contract: the behaviour is not pinned, and it may change
> or be removed without a deprecation cycle.

**Status**: ⛔ Frozen prototype — see the banner at the top of this guide
**Date**: 2025-10-01
**Scope**: Comprehensive guide to maze generation algorithms and hybrid methods for MFG research

## Overview

MFGArchon provides a complete suite of maze generation algorithms optimized for Mean Field Games reinforcement learning research. This guide covers all implemented algorithms, hybrid composition methods, and recommendations for different MFG scenarios.

## Implemented Algorithms

### 1. Perfect Mazes: Recursive Backtracking & Wilson's Algorithm

**Module**: `mfgarchon.geometry.graph.maze_generator` (`MazeGeometry`)

#### Mathematical Foundation
Perfect mazes are minimal spanning trees on grid graphs with two critical properties:
1. **Fully Connected**: Path exists between any two cells
2. **No Loops**: Exactly one unique path between any pair of cells

#### Recursive Backtracking (DFS)
```python
from mfgarchon.alg.reinforcement.environments import (
    PerfectMazeGenerator,
    MazeAlgorithm,
)

generator = PerfectMazeGenerator(
    rows=21,
    cols=21,
    algorithm=MazeAlgorithm.RECURSIVE_BACKTRACKING,
)
maze = generator.generate(seed=42)
```

**Properties**:
- Long, winding corridors
- High exploration difficulty
- River-like flow patterns

**MFG Applications**:
- Path-finding challenges
- Single-route congestion studies
- Baseline for algorithm testing

#### Wilson's Algorithm
```python
generator = PerfectMazeGenerator(
    rows=21,
    cols=21,
    algorithm=MazeAlgorithm.WILSONS,
)
maze = generator.generate(seed=42)
```

**Properties**:
- Unbiased sampling over all possible perfect mazes
- Structural diversity
- No algorithmic bias

**MFG Applications**:
- Research requiring fair maze distribution
- Benchmark testing across diverse structures
- Statistical analysis of maze properties

---

### 2. Recursive Division: Room-Based Environments

**Module**: `mfgarchon.geometry.graph.maze_recursive_division` (`RecursiveDivisionGenerator`)

#### Algorithm
Starts with empty space and recursively adds walls with doors, creating structured building-like layouts.

```python
from mfgarchon.alg.reinforcement.environments import (
    RecursiveDivisionGenerator,
    RecursiveDivisionConfig,
)

config = RecursiveDivisionConfig(
    rows=40,
    cols=60,
    min_room_width=5,
    min_room_height=5,
    door_width=2,          # Width of passages
    num_doors_per_wall=1,  # Connections per wall
    split_bias=0.5,        # 0.5 = no bias, <0.5 = favor horizontal
)

generator = RecursiveDivisionGenerator(config)
maze = generator.generate(seed=42)
```

**Properties**:
- Variable-width corridors and rooms
- Controllable room dimensions
- Adjustable door widths (bottlenecks)
- Building-like structured layouts

**MFG Applications**:
- **Crowd dynamics** in plazas, halls, auditoriums
- **Building evacuation** scenarios
- **Bottleneck studies** with controllable choke points
- **Multi-room navigation** problems

**Best for**: Structured MFG scenarios with distinct rooms and controlled connectivity.

---

### 3. Cellular Automata: Organic Environments

**Module**: `mfgarchon.geometry.graph.maze_cellular_automata` (`CellularAutomataGenerator`)

#### Algorithm
Random initialization followed by iterative smoothing rules, producing cave-like structures.

```python
from mfgarchon.alg.reinforcement.environments import (
    CellularAutomataGenerator,
    CellularAutomataConfig,
)

config = CellularAutomataConfig(
    rows=50,
    cols=50,
    initial_wall_prob=0.45,  # Starting wall density
    num_iterations=5,         # Smoothing iterations
    birth_limit=4,            # Neighbors to birth wall
    death_limit=3,            # Neighbors to keep wall
    use_moore_neighborhood=True,  # 8-connected vs 4-connected
)

generator = CellularAutomataGenerator(config)
maze = generator.generate(seed=42)
```

**Properties**:
- Organic, cave-like appearance
- Variable-width passages emerge naturally
- Unpredictable layouts
- No grid bias

**MFG Applications**:
- **Parks and natural spaces** simulation
- **Irregular urban environments**
- **Algorithm robustness testing** (non-structured)
- **Natural terrain navigation**

**Best for**: Testing MFG algorithms in organic, unstructured environments.

---

### 4. Braided Mazes: Loop Addition

**Module**: `mfgarchon.geometry.graph.maze_recursive_division` (`add_loops`)

#### Algorithm
Post-processing function that converts perfect mazes into braided mazes by removing walls to create loops.

```python
from mfgarchon.alg.reinforcement.environments import (
    PerfectMazeGenerator,
    add_loops,
)

# Generate base perfect maze
perfect_maze = PerfectMazeGenerator(rows=21, cols=21).generate()

# Add loops for route diversity
braided_maze = add_loops(
    perfect_maze,
    loop_density=0.15,  # Fraction of walls to remove (0.0-1.0)
    seed=42,
)
```

**Properties**:
- Multiple paths between points
- Route diversity and choice
- Controllable loop density
- Preserves connectivity

**MFG Applications**:
- **Route selection games** with multiple Nash equilibria
- **Congestion avoidance** studies
- **Strategic path planning** problems
- **Multi-path navigation**

**Best for**: MFG scenarios requiring route choice and congestion dynamics.

---

## Hybrid Maze Generation Methods

Combining algorithms creates sophisticated environments for advanced MFG research.

### Hybrid 1: Braided Perfect Maze ✅ READY TO USE

**Components**: Perfect Maze + Loop Addition

```python
from mfgarchon.alg.reinforcement.environments import (
    PerfectMazeGenerator,
    MazeAlgorithm,
    add_loops,
)

# Generate base maze
perfect = PerfectMazeGenerator(
    rows=31,
    cols=31,
    algorithm=MazeAlgorithm.RECURSIVE_BACKTRACKING,
).generate(seed=42)

# Add loops for route diversity
braided = add_loops(perfect, loop_density=0.20, seed=42)
```

**Result**: Complex maze with shortcuts and multiple paths

**MFG Use Cases**:
- Congestion avoidance games
- Nash equilibria in route selection
- Strategic path planning with density feedback

---

### Hybrid 2: Building with Labyrinth Rooms

**Components**: Recursive Division + Perfect Maze

**Concept**: Use Recursive Division for macro-level floor plan, then fill specific rooms with dense perfect mazes.

```python
from mfgarchon.alg.reinforcement.environments import (
    RecursiveDivisionGenerator,
    RecursiveDivisionConfig,
    PerfectMazeGenerator,
)

# 1. Create building structure
config = RecursiveDivisionConfig(
    rows=41, cols=61, min_room_width=8, min_room_height=8
)
building = RecursiveDivisionGenerator(config).generate(seed=42)

# 2. Fill specific rooms with mazes (manual region selection)
room_regions = [
    (5, 10, 5, 15),   # (row_start, row_end, col_start, col_end)
    (15, 25, 20, 35),
]

for r_start, r_end, c_start, c_end in room_regions:
    room_height = r_end - r_start
    room_width = c_end - c_start

    # Generate mini-maze for this room
    mini_maze = PerfectMazeGenerator(
        rows=room_height,
        cols=room_width
    ).generate(seed=42)

    # Overlay into building
    building[r_start:r_end, c_start:c_end] = mini_maze
```

**Result**: Structured layout with both open halls and complex maze-filled rooms

**MFG Use Cases**:
- Mixed-density areas (open plazas + crowded markets)
- Hierarchical navigation (macro-level room selection + micro-level pathfinding)
- Complex building interiors

---

### Hybrid 3: City Park

**Components**: Recursive Division + Cellular Automata

**Concept**: Combine structured buildings with organic natural spaces.

```python
from mfgarchon.alg.reinforcement.environments import (
    RecursiveDivisionGenerator,
    RecursiveDivisionConfig,
    CellularAutomataGenerator,
    CellularAutomataConfig,
)

# 1. Create city structure
city_config = RecursiveDivisionConfig(rows=50, cols=50, min_room_width=6)
city = RecursiveDivisionGenerator(city_config).generate(seed=42)

# 2. Replace specific regions with organic parks
park_config = CellularAutomataConfig(
    rows=15, cols=15, initial_wall_prob=0.40, num_iterations=4
)
park = CellularAutomataGenerator(park_config).generate(seed=42)

# Overlay park into city (example: center region)
city[17:32, 17:32] = park
```

**Result**: Map with rectilinear buildings and organic park areas

**MFG Use Cases**:
- Urban environment simulation
- Behavioral changes across environment types
- Mixed structured/unstructured navigation

---

### Hybrid 4: Post-Processed Perfect Maze (Wall Erosion)

**Components**: Perfect Maze + Wall Erosion

**Concept**: Generate unit-width maze, then erode walls to create wider passages and rooms.

```python
from mfgarchon.alg.reinforcement.environments import PerfectMazeGenerator
import numpy as np

def erode_maze_walls(maze: np.ndarray, erosion_passes: int = 2) -> np.ndarray:
    """
    Erode walls that have 3+ open neighbors, creating wider passages.

    Args:
        maze: Input maze (1 = wall, 0 = open)
        erosion_passes: Number of erosion iterations

    Returns:
        Maze with eroded walls
    """
    eroded = maze.copy()
    rows, cols = maze.shape

    for _ in range(erosion_passes):
        to_remove = []
        for r in range(1, rows - 1):
            for c in range(1, cols - 1):
                if eroded[r, c] == 1:  # Is wall
                    # Count open neighbors
                    neighbors = [
                        eroded[r-1, c], eroded[r+1, c],
                        eroded[r, c-1], eroded[r, c+1]
                    ]
                    if sum(n == 0 for n in neighbors) >= 3:
                        to_remove.append((r, c))

        for r, c in to_remove:
            eroded[r, c] = 0

    return eroded

# Generate and erode
perfect = PerfectMazeGenerator(rows=31, cols=31).generate(seed=42)
eroded = erode_maze_walls(perfect, erosion_passes=2)
```

**Result**: Perfect maze with opened passages and emergent small rooms

**MFG Use Cases**:
- Complex paths with natural open spaces
- Emergent congestion points
- Testing algorithm adaptation to varying passage widths

---

## Algorithm Selection Guide for MFG Scenarios

### Scenario 1: Building Evacuation
**Recommended**: Recursive Division
**Reason**: Structured rooms, controllable bottlenecks, realistic building layout

### Scenario 2: Route Choice Games
**Recommended**: Braided Perfect Maze (Hybrid 1)
**Reason**: Multiple paths, strategic route selection, congestion avoidance

### Scenario 3: Crowd Dynamics in Open Spaces
**Recommended**: Recursive Division with large rooms
**Reason**: Variable-width corridors, plazas, controllable capacity

### Scenario 4: Algorithm Robustness Testing
**Recommended**: Cellular Automata
**Reason**: Organic, unpredictable, no grid bias

### Scenario 5: Urban Navigation
**Recommended**: City Park (Hybrid 3)
**Reason**: Mixed structured/organic, realistic urban complexity

### Scenario 6: Complex Hierarchical Navigation
**Recommended**: Building with Labyrinths (Hybrid 2)
**Reason**: Macro/micro navigation levels, mixed density areas

---

## Performance Characteristics

| Algorithm | Generation Speed | Memory Usage | Configurability | Realism |
|:----------|:----------------|:-------------|:----------------|:--------|
| Recursive Backtracking | Fast | Low | Low | Low |
| Wilson's Algorithm | Moderate | Low | Low | Low |
| Recursive Division | Fast | Low | High | High |
| Cellular Automata | Fast | Low | High | High |
| Braided Mazes | Fast | Low | Moderate | Moderate |
| Hybrid Methods | Moderate | Moderate | Very High | Very High |

---

## Future Enhancements: Voronoi Diagrams

### Motivation
Voronoi-based mazes offer unique advantages for advanced MFG research:
- **Organic non-rectilinear layouts** (realistic public spaces)
- **Complex multi-angle junctions** → richer congestion dynamics
- **No grid bias** → more robust agent training
- **Natural irregular spaces** (parks, plazas, irregular buildings)

### Implementation Challenges
1. **Non-grid representation**: State space becomes Voronoi cell IDs or continuous (x,y)
2. **Complex action spaces**: Cannot use simple NORTH/SOUTH/EAST/WEST
3. **Generation complexity**: Requires scipy.spatial.Voronoi + graph algorithms
4. **Integration effort**: Existing MFGMazeEnvironment assumes grid structure

### Recommendation
- **Defer to Phase 5**: After mastering grid-based MFG experiments
- **High research value**: Excellent for robustness and generalizability studies
- **Current sufficiency**: Hybrid methods provide adequate complexity for initial research

---

## References

1. Jamis Buck, "Mazes for Programmers" (2015)
2. Stephen Wolfram, "A New Kind of Science" (Cellular Automata)
3. Classic dungeon generation techniques (Recursive Division)
4. Voronoi/Delaunay theory for computational geometry

---

## API Summary

```python
# All imports
from mfgarchon.alg.reinforcement.environments import (
    # Perfect mazes
    PerfectMazeGenerator,
    MazeAlgorithm,

    # Room-based mazes
    RecursiveDivisionGenerator,
    RecursiveDivisionConfig,

    # Organic mazes
    CellularAutomataGenerator,
    CellularAutomataConfig,

    # Braiding utility
    add_loops,

    # Environment integration
    MFGMazeEnvironment,
    MFGMazeConfig,
)
```

---

**Last Updated**: 2025-10-01
**Status**: ⛔ Frozen prototype — see the banner at the top of this guide
**Related Issues**: #57 (Advanced Maze Generation), #54 (RL Paradigm Development)
