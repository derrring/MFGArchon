"""What #2512's structural guards share: walking one scope of an AST, and planting rivals in a copy of `alg/`.

A structural guard scans `mfgarchon/alg/` for a rival to an owner, and proves its rules can fire by planting
a rival in a copy of the tree and scanning that (audit session ruling, 2026-10-09, #2512 comment
6073690080). The C1 guard (`test_c1_change_has_one_owner_2512.py`) and row B3's
(`test_fp_reads_the_shared_bc_through_its_owner_2512.py`) both import from here.
"""

from __future__ import annotations

import ast
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

REPO = Path(__file__).resolve().parents[1]
ALG = Path("mfgarchon") / "alg"
#: Every node that opens a scope of its own.
FUNCTIONS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)


def name_of(node: ast.AST) -> str | None:
    """The name a ``Name`` or ``Attribute`` ends in, else None."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def own_nodes(scope: ast.AST) -> Iterator[ast.AST]:
    """The nodes of ``scope`` itself: a nested function or lambda is its own scope and is not entered."""
    stack = list(ast.iter_child_nodes(scope))
    while stack:
        node = stack.pop()
        yield node
        if not isinstance(node, FUNCTIONS):
            stack.extend(ast.iter_child_nodes(node))


def copy_alg(tmp_path_factory, label: str) -> Path:
    """A copy of `mfgarchon/alg` under a fresh temporary root, to plant rivals in."""
    root = tmp_path_factory.mktemp(label)
    shutil.copytree(REPO / ALG, root / ALG)
    return root


def planted_scan(
    scan: Callable[[Path], tuple[list[str], list[str]]],
    root: Path,
    target: Path,
    planted: str,
    anchor: str | None = None,
) -> list[str]:
    """Plant ``planted`` in ``target`` under ``root``, before ``anchor`` or at the end, return ``scan``'s
    sites for ``root``, and restore ``target`` (or remove it, if the plant created it)."""
    original = target.read_text() if target.exists() else None
    try:
        if anchor is None:
            target.write_text((original + "\n\n" if original else "") + planted)
        else:
            assert original is not None, anchor
            assert original.count(anchor) == 1, anchor
            target.write_text(original.replace(anchor, planted + anchor))
        return scan(root)[1]
    finally:
        if original is None:
            target.unlink(missing_ok=True)
        else:
            target.write_text(original)
