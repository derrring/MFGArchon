"""Every branch of HybridMazeGenerator's algorithm dispatch must be reachable (#1712).

Three of the four branches imported modules that do not exist:

    from mfgarchon.geometry.graph.recursive_division import ...   ModuleNotFoundError
    from mfgarchon.geometry.graph.cellular_automata import ...    ModuleNotFoundError
    from mfgarchon.geometry.graph.voronoi_maze import ...         ModuleNotFoundError

The real modules carry a `maze_` prefix; a rename added it and `maze_hybrid` was not
updated. The imports are **function-local**, so nothing failed at import time, no
collection error surfaced, and `from mfgarchon.geometry.graph import HybridMazeGenerator`
kept working. The failure appeared only when a caller selected one of those three
algorithms.

This file existed nowhere before: `HybridMazeGenerator` had no test at all, which is how
three of four branches stayed broken across a rename. Each test below calls `generate()`
for real rather than asserting the import resolves -- a reachable branch that produces a
degenerate maze is still broken, and `pytest.importorskip`-style checks would not notice.
"""

import pytest

import numpy as np

from mfgarchon.geometry.graph.maze_hybrid import (
    AlgorithmSpec,
    HybridMazeConfig,
    HybridMazeGenerator,
    HybridStrategy,
)

ALGORITHMS = ["perfect", "recursive_division", "cellular_automata", "voronoi"]


def _generator(algorithm: str, rows: int = 48, cols: int = 48) -> HybridMazeGenerator:
    """48x48, not 32x32: SPATIAL_SPLIT halves each axis and maze_voronoi refuses a
    region under 20 on either side. maze_hybrid does not check its sub-generators'
    minimums, so a small hybrid maze fails inside the sub-generator with a message
    naming dimensions the caller never supplied. Noted on #1712, not fixed here."""
    return HybridMazeGenerator(
        HybridMazeConfig(
            rows=rows,
            cols=cols,
            strategy=HybridStrategy.SPATIAL_SPLIT,
            algorithms=[AlgorithmSpec(algorithm=algorithm), AlgorithmSpec(algorithm=algorithm)],
            seed=20260728,
        )
    )


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_every_declared_algorithm_generates(algorithm):
    """The regression: three of these raised ModuleNotFoundError at call time."""
    maze = _generator(algorithm).generate()
    assert maze.shape == (48, 48)
    assert set(np.unique(maze)) <= {0, 1}, f"maze is not a wall/passage array: {np.unique(maze)}"


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_every_algorithm_produces_both_walls_and_passages(algorithm):
    """A branch that returns all-wall or all-passage is reachable and still useless.

    Import-resolution alone would pass for such a branch, which is why these call
    `generate()` rather than checking that the module imports.
    """
    maze = _generator(algorithm).generate()
    assert maze.min() == 0, f"{algorithm}: no passages"
    assert maze.max() == 1, f"{algorithm}: no walls"


@pytest.mark.parametrize(
    "algorithm",
    [
        pytest.param(
            "perfect",
            marks=pytest.mark.xfail(
                strict=True,
                reason=(
                    "Issue #1742: MazeGeometry seeds the GLOBAL random module while "
                    "maze_hybrid seeds a private np.random.Generator, so this branch's "
                    "output depends on process-wide state that HybridMazeGenerator(seed=) "
                    "cannot control. Verified: pre-seeding `random` makes it reproducible. "
                    "Strict, so fixing #1742 XPASSes and fails the build until the marker "
                    "goes."
                ),
            ),
        ),
        "recursive_division",
        "cellular_automata",
        "voronoi",
    ],
)
def test_generation_is_reproducible_under_a_fixed_seed(algorithm):
    a = _generator(algorithm).generate(seed=7)
    b = _generator(algorithm).generate(seed=7)
    np.testing.assert_array_equal(a, b)


def test_the_parametrisation_covers_every_declared_algorithm():
    """`AlgorithmSpec.algorithm` is a Literal; this notices when a fifth is added.

    Without it, a new algorithm arrives with the same shape of hole these tests exist
    to close -- a dispatch branch nothing calls.
    """
    from typing import get_args, get_type_hints

    declared = set(get_args(get_type_hints(AlgorithmSpec)["algorithm"]))
    assert declared == set(ALGORITHMS), (
        f"AlgorithmSpec declares {sorted(declared)}, this file covers {sorted(ALGORITHMS)}"
    )


def test_the_prefixed_modules_are_the_only_ones_that_exist():
    """Pins the rename, so reverting an import path fails here rather than at call time."""
    import importlib

    for name in ("maze_recursive_division", "maze_cellular_automata", "maze_voronoi"):
        importlib.import_module(f"mfgarchon.geometry.graph.{name}")
    for name in ("recursive_division", "cellular_automata", "voronoi_maze"):
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module(f"mfgarchon.geometry.graph.{name}")
