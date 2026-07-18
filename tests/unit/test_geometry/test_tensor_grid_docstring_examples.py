"""Every ``TensorProductGrid(...)`` docstring example must actually construct.

Since #674 made `boundary_conditions` mandatory and the positional constructor was
removed, 36 of the 37 constructor examples in `tensor_grid.py` were dead: 7 raised
`TypeError` (positional form) and 29 raised `ValueError` (missing BC). The four in the
class-level docstring -- the first thing a user reads -- were all broken.

Nothing executed them, so the rot was invisible. This test executes them.
"""

from __future__ import annotations

import ast
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
    body = "\n".join(line for line in code.split("\n") if line.strip())
    try:
        exec(compile(body, f"{_SOURCE}:{lineno}", "exec"), namespace)
    except Exception as exc:  # the whole point is to report which example broke
        pytest.fail(f"{_SOURCE}:{lineno} docstring example failed: {type(exc).__name__}: {exc}\n---\n{body}")


def _annotated_expressions(code: str) -> list[tuple[str, str]]:
    """``>>> expr  # <literal>`` pairs, where the comment states the expected value.

    Only comments that parse as a Python literal are checked; prose comments
    ("# Get coordinate matrices") carry no assertion.
    """
    pairs = []
    for line in code.split("\n"):
        if "#" not in line or line.lstrip().startswith(("#", "...")):
            continue
        expr, _, comment = line.partition("#")
        expr, comment = expr.strip(), comment.strip()
        if not expr or "=" in expr.replace("==", "") or expr.endswith((",", "(")):
            continue
        # "2 (inferred from len(bounds))" -> "2"; "True" -> "True"
        literal = re.split(r"\(|--|\s-\s|\bif\b|,\s*but\b", comment)[0].strip().rstrip(".")
        try:
            ast.literal_eval(literal)
        except (ValueError, SyntaxError):
            continue
        pairs.append((expr, literal))
    return pairs


ANNOTATED = [(ln, expr, expected) for ln, code in CONSTRUCTOR_BLOCKS for expr, expected in _annotated_expressions(code)]


def test_value_assertions_were_actually_collected():
    """Guards the premise of the truthfulness test, which is otherwise silently skippable.

    If the literal parser stops matching -- the shape a future regex edit takes --
    `ANNOTATED` empties, the parametrized test degrades to a skip, and the file goes
    green having asserted nothing. Same blind spot `test_the_examples_were_actually_found`
    exists to close, one level down.
    """
    assert len(ANNOTATED) >= 9, f"only {len(ANNOTATED)} value assertions collected; the literal parser regressed"


@pytest.mark.parametrize(("lineno", "expr", "expected"), ANNOTATED, ids=lambda v: str(v)[:30])
def test_annotated_example_values_are_truthful(lineno, expr, expected):
    """An example that runs can still teach something false.

    The repair of these docstrings initially added `no_flux_bc` to three examples
    whose prose said "periodic", so they executed cleanly while demonstrating the
    opposite of what they claimed. Running is not enough; where a comment states a
    value, that value must be what the expression returns.
    """
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
    block = dict(CONSTRUCTOR_BLOCKS)[lineno]
    setup = "\n".join(line for line in block.split("\n") if line.strip() and not line.strip().startswith(expr))
    exec(compile(setup, f"{_SOURCE}:{lineno}", "exec"), namespace)

    actual = eval(compile(expr, f"{_SOURCE}:{lineno}", "eval"), namespace)
    want = ast.literal_eval(expected)
    message = f"{_SOURCE}:{lineno}: `{expr}` -> {actual!r}, but the comment claims {want!r}"

    # bool before int, in both directions: `isinstance(True, int)` is True, so a numeric
    # comparison accepts 1 for a `# True` comment, and True for a `# 1` comment.
    if isinstance(want, bool) or isinstance(actual, bool):
        assert actual is want, message
    elif isinstance(want, (int, float)) or (isinstance(want, list) and want and isinstance(want[0], (int, float))):
        # Strings must not float-coerce: `np.asarray('2').astype(float)` would pass for `# 2`.
        assert not isinstance(actual, str), message
        assert np.all(np.isclose(np.asarray(actual, dtype=float), np.asarray(want, dtype=float), rtol=1e-12)), message
    else:
        assert actual == want, message


def test_no_positional_constructor_examples_remain():
    """The positional signature was removed; an example using it raises TypeError."""
    offenders = [
        ln
        for ln, code in CONSTRUCTOR_BLOCKS
        if re.search(r"TensorProductGrid\(\s*\d", code) or re.search(r"TensorProductGrid\(\s*\[", code)
    ]
    assert not offenders, f"positional TensorProductGrid( examples at lines {offenders}"
