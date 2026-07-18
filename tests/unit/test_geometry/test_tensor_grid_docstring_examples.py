"""Every ``TensorProductGrid(...)`` docstring example must actually construct.

Since #674 made `boundary_conditions` mandatory and the positional constructor was
removed, 36 of the 37 constructor examples in `tensor_grid.py` were dead: 7 raised
`TypeError` (positional form) and 29 raised `ValueError` (missing BC). The four in the
class-level docstring -- the first thing a user reads -- were all broken.

Nothing executed them, so the rot was invisible. This test executes them.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

import numpy as np

from mfgarchon.geometry.boundary import (
    BCType,
    BoundaryConditions,
    dirichlet_bc,
    neumann_bc,
    no_flux_bc,
    periodic_bc,
)
from mfgarchon.geometry.grids.tensor_grid import TensorProductGrid

_SOURCE = Path(TensorProductGrid.__module__.replace(".", "/") + ".py")
_MODULE_FILE = Path(__file__).resolve().parents[3] / _SOURCE


def _doctest_blocks() -> list[tuple[int, str]]:
    """Contiguous ``>>>`` / ``...`` runs, so a block's own setup lines are in scope."""
    lines = _MODULE_FILE.read_text(encoding="utf-8").split("\n")
    blocks: list[tuple[int, str]] = []
    current: list[str] = []
    start = 0
    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        if stripped.startswith(">>>"):
            if not current:
                start = i
            current.append(stripped[3:].lstrip())
        elif stripped.startswith("...") and current:
            current.append(stripped[3:].lstrip())
        elif current:
            blocks.append((start, "\n".join(current)))
            current = []
    if current:
        blocks.append((start, "\n".join(current)))
    return blocks


CONSTRUCTOR_BLOCKS = [(ln, code) for ln, code in _doctest_blocks() if "TensorProductGrid(" in code]


def test_the_examples_were_actually_found():
    """Guards the premise: a broken extractor would make this file pass vacuously."""
    assert len(CONSTRUCTOR_BLOCKS) >= 15, f"only found {len(CONSTRUCTOR_BLOCKS)} constructor examples"


@pytest.mark.parametrize(("lineno", "code"), CONSTRUCTOR_BLOCKS, ids=lambda v: f"L{v}" if isinstance(v, int) else "")
def test_constructor_docstring_example_runs(lineno, code):
    namespace = {
        "np": np,
        "TensorProductGrid": TensorProductGrid,
        "no_flux_bc": no_flux_bc,
        "dirichlet_bc": dirichlet_bc,
        "neumann_bc": neumann_bc,
        "periodic_bc": periodic_bc,
        "BoundaryConditions": BoundaryConditions,
        "BCType": BCType,
    }
    # Drop bare expression lines whose value is only shown as a comment ("# True").
    body = "\n".join(line for line in code.split("\n") if line.strip())
    try:
        exec(compile(body, f"{_SOURCE}:{lineno}", "exec"), namespace)
    except Exception as exc:  # the whole point is to report which example broke
        pytest.fail(f"{_SOURCE}:{lineno} docstring example failed: {type(exc).__name__}: {exc}\n---\n{body}")


def test_no_positional_constructor_examples_remain():
    """The positional signature was removed; an example using it raises TypeError."""
    offenders = [
        ln
        for ln, code in CONSTRUCTOR_BLOCKS
        if re.search(r"TensorProductGrid\(\s*\d", code) or re.search(r"TensorProductGrid\(\s*\[", code)
    ]
    assert not offenders, f"positional TensorProductGrid( examples at lines {offenders}"
