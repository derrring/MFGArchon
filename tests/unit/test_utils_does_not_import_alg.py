"""`mfgarchon.utils` is below `mfgarchon.alg` and must not import upward.

The inversion is what makes the import graph unsplittable. Every heavy package — torch, jax,
numba, cvxpy — enters through one line, `utils/__init__.py`'s eager `from .adjoint_validation
import (...)`, which reaches `alg.base_solver` for a single enum and drags the whole `alg` tree
in behind it. Measured on 9f84c22c: `import mfgarchon` is 4.89s, of which 2.56s is those four
packages and 2.14s is the library importing itself.

That line cannot simply be made lazy. Removing it, or replacing it with a PEP 562 `__getattr__`,
both fail with

    ImportError: cannot import name 'clip_nonnegative_or_raise' from partially initialized
    module 'mfgarchon.utils.numerical' (most likely due to a circular import)

because `utils/numerical/__init__.py` imports `alg`, which reaches `fp_network`, which needs a
name from `utils.numerical` while it is still stopped at that line. The eager import is
load-bearing: it completes `utils.numerical` by another route first and hides the cycle. #1930.

This test ratchets the count DOWN only. It is not a claim that two edges are acceptable — they
are the two that need a design decision, and each removal should lower the number here.
"""

from __future__ import annotations

import ast
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
UTILS = REPO / "mfgarchon" / "utils"

# The two that remain, each with what it would take to remove it:
#   utils/numerical/__init__.py     re-exports `gfdm_strategies`, which lives in alg
#   utils/adjoint_validation.py     imports one enum, `SchemeFamily`, from alg.base_solver
ALLOWED_MODULE_LEVEL = 2


def _alg_imports() -> list[tuple[str, str, int, str]]:
    """(kind, file, line, target) for every import of `mfgarchon.alg` under utils/.

    Parsed, not grepped: `grep -c mfgarchon.alg` over this tree returns 19, and 14 of those are
    comments, docstrings and strings. The distinction that matters is module-level versus
    deferred, and only an AST can make it.
    """
    out = []
    for path in sorted(UTILS.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text())
        except (SyntaxError, UnicodeDecodeError):
            continue
        top_level = {id(node) for node in tree.body}
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Import, ast.ImportFrom)):
                continue
            target = getattr(node, "module", None) or (node.names[0].name if node.names else "")
            if not (target == "mfgarchon.alg" or target.startswith("mfgarchon.alg.")):
                continue
            kind = "module-level" if id(node) in top_level else "deferred"
            out.append((kind, str(path.relative_to(REPO)), node.lineno, target))
    return out


def test_the_parser_finds_the_edges_that_are_there():
    """Positive control. Every assertion below is an upper bound, and a bound is also satisfied
    by a parser that finds nothing — a moved directory, a changed package name, an `rglob` that
    matches no file."""
    edges = _alg_imports()
    assert edges, "no utils -> alg import found at all; the scanner is not reading the tree"
    files = list(UTILS.rglob("*.py"))
    assert len(files) > 50, f"utils/ has {len(files)} python files; the path is wrong"


def test_module_level_imports_of_alg_do_not_grow():
    """A ratchet, downward. Two deprecation shims were deleted here — both overdue on their own
    stated schedule, `v0.21.0` against a current `0.22.0.dev0` — taking two edges with them.

    The failure mode this guards is not a new feature importing `alg`; it is a compatibility
    shim being added back to keep an old import path alive, which is exactly what the two
    deleted files were.
    """
    module_level = [e for e in _alg_imports() if e[0] == "module-level"]
    assert len(module_level) <= ALLOWED_MODULE_LEVEL, (
        f"utils imports alg at module level in {len(module_level)} places, was "
        f"{ALLOWED_MODULE_LEVEL}. Each one pulls the whole alg tree into `import mfgarchon`.\n"
        + "\n".join(f"  {f}:{ln} -> {t}" for _k, f, ln, t in module_level)
        + "\nIf an edge is genuinely needed, defer it into the function that uses it, or move the "
        "shared name to a layer both may depend on. If you are lowering the count, lower "
        "ALLOWED_MODULE_LEVEL in the same commit."
    )


def test_the_deleted_shims_stay_deleted():
    """Named explicitly, because the ratchet above counts and this says which."""
    for name in ("anderson_acceleration.py", "gfdm_strategies.py"):
        shim = UTILS / "numerical" / name
        assert not shim.exists(), (
            f"{shim.relative_to(REPO)} is back. It was a re-export of the canonical module in "
            f"`alg`, overdue for removal since v0.21.0. Import from `mfgarchon.alg` directly."
        )
