"""Every `mfgarchon.*` module path named in an import must exist.

Runtime cannot catch this class: all seven sites found on 2026-08-17 sat inside `if TYPE_CHECKING:`
blocks, which Python never evaluates, so the imports were dead for months while every test passed.
mypy does catch it -- but `scripts/local_ci.sh` does not run mypy, and turning it on wholesale means
1318 errors of which this class is 3. This asserts only the unambiguous part.
"""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2] / "mfgarchon"


def _imported_internal_modules() -> dict[str, list[str]]:
    """Module path -> the files naming it. Walks the whole AST, so TYPE_CHECKING is included."""
    found: dict[str, list[str]] = {}
    for py in sorted(_ROOT.rglob("*.py")):
        tree = ast.parse(py.read_text(), filename=str(py))
        for node in ast.walk(tree):
            mods = []
            if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                mods = [node.module]
            elif isinstance(node, ast.Import):
                mods = [a.name for a in node.names]
            for m in mods:
                if m == "mfgarchon" or m.startswith("mfgarchon."):
                    found.setdefault(m, []).append(str(py.relative_to(_ROOT.parent)))
    return found


def test_every_internal_import_path_exists():
    imported = _imported_internal_modules()
    assert len(imported) > 100, f"the walk found only {len(imported)} internal imports; it broke"

    missing = {mod: sorted(set(files)) for mod, files in imported.items() if importlib.util.find_spec(mod) is None}
    assert not missing, "import paths that do not exist:\n" + "\n".join(
        f"  {m}  <- {', '.join(f)}" for m, f in sorted(missing.items())
    )
