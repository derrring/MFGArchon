"""No `try` in the library catches `ImportError` around an import from the library itself (#2088).

Such a guard cannot tell "an optional dependency is absent" from "this module names something that does not exist":
both give the same silent fallback. #2088's flag `VISUALIZATION_AVAILABLE` was `False` on every install because its
block imported three names that never existed. Every other internal import in the library imports with every optional
dependency unimportable, so the guards around those never fired either.

What the census sees: a `try` whose handlers name `ImportError` or `ModuleNotFoundError` (alone or in a tuple), with,
anywhere in its body, an `import mfgarchon...` or `from mfgarchon... import`, a relative `from . import`, or a call to
`import_module(...)` or `__import__(...)` whose first argument is a string or f-string starting with `mfgarchon` or `.`.
Each form has a planted positive control below.

What it does not see: a guard written as a bare `except`, `except Exception` or `except BaseException`; one written as
`contextlib.suppress(ImportError)`; a module loaded by path (`importlib.util.spec_from_file_location`) or from source
text (`exec`, `compile`); a module name built at run time from anything but a literal prefix; and a guard around a call
into a helper that imports. A census reports only what its predicate reaches, and #2088's own census first missed a
guarded `import_module` for exactly that reason.

An allowed guard is one that catches an absent third-party dependency and nothing else, with its reason. An allowance
that covers no guard fails, so a guard that is removed takes its allowance with it.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]

#: (file, import target as written) -> why the guard stays.
ALLOWED = {
    (
        "mfgarchon/utils/__init__.py",
        ".memory_management",
    ): "imports psutil at module top; catches only psutil's ModuleNotFoundError, since AGENTS.md calls psutil optional (#2930)",
    (
        "mfgarchon/utils/__init__.py",
        ".performance.monitoring",
    ): "imports psutil at module top; catches only psutil's ModuleNotFoundError, since AGENTS.md calls psutil optional (#2930)",
}

_IMPORT_ERRORS = {"ImportError", "ModuleNotFoundError"}


def _catches_import_error(handler: ast.ExceptHandler) -> bool:
    names = handler.type.elts if isinstance(handler.type, ast.Tuple) else [handler.type]
    return any(isinstance(n, ast.Name) and n.id in _IMPORT_ERRORS for n in names)


def _literal_prefix(node: ast.expr) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr) and node.values and isinstance(node.values[0], ast.Constant):
        return str(node.values[0].value)
    return None


def _internal_targets(stmt: ast.stmt):
    for node in ast.walk(stmt):
        if isinstance(node, ast.ImportFrom):
            if node.level > 0 or (node.module or "").split(".")[0] == "mfgarchon":
                yield "." * node.level + (node.module or "")
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] == "mfgarchon":
                    yield alias.name
        elif isinstance(node, ast.Call) and node.args:
            func = node.func
            name = func.id if isinstance(func, ast.Name) else func.attr if isinstance(func, ast.Attribute) else None
            prefix = _literal_prefix(node.args[0]) if name in ("import_module", "__import__") else None
            if prefix is not None and (prefix.startswith(".") or prefix.split(".")[0] == "mfgarchon"):
                yield prefix


def guarded_internal_imports(source: str) -> list[tuple[int, str]]:
    """(line of the `try`, import target as written) for each internal import under an ImportError guard."""
    hits = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Try) and any(_catches_import_error(h) for h in node.handlers):
            hits.extend((node.lineno, target) for stmt in node.body for target in _internal_targets(stmt))
    return hits


def census(root: Path) -> list[tuple[str, int, str]]:
    return [
        (path.relative_to(root).as_posix(), line, target)
        for path in sorted((root / "mfgarchon").rglob("*.py"))
        for line, target in guarded_internal_imports(path.read_text())
    ]


def test_no_guard_hides_an_internal_import():
    found = census(REPO)
    unexplained = [hit for hit in found if (hit[0], hit[2]) not in ALLOWED]
    assert unexplained == [], (
        "An `except ImportError` guards an import from the library itself. If a name it imports does not exist, the "
        "guard turns that into a silent fallback (#2088). Let the import fail, or narrow the handler to the absent "
        f"third-party dependency and add it to ALLOWED with its reason:\n{unexplained}"
    )
    stale = sorted(set(ALLOWED) - {(hit[0], hit[2]) for hit in found})
    assert stale == [], f"An allowance covers no guard any more; delete it: {stale}"


_PLANTED = {
    "Import": "import mfgarchon.core\n",
    "ImportFrom": "from .sibling import name_that_does_not_exist\n",
    "import_module": "from importlib import import_module\nmodule = import_module(f'.{NAME}', __name__)\n",
    "__import__": "module = __import__('mfgarchon.sibling')\n",
}


@pytest.mark.parametrize("form", sorted(_PLANTED))
def test_the_census_sees_each_import_form_it_claims(tmp_path, form):
    body = "".join("    " + line + "\n" for line in _PLANTED[form].splitlines())
    planted = tmp_path / "mfgarchon" / "planted.py"
    planted.parent.mkdir()
    planted.write_text(f"NAME = 'sibling'\ntry:\n{body}except ImportError:\n    FLAG = False\n")
    assert len(census(tmp_path)) == 1, census(tmp_path)


def test_the_census_does_not_fire_on_a_third_party_guard_or_an_unguarded_import(tmp_path):
    planted = tmp_path / "mfgarchon" / "planted.py"
    planted.parent.mkdir()
    planted.write_text(
        "from .sibling import name\n"
        "try:\n    import torch\nexcept ImportError:\n    torch = None\n"
        "try:\n    from .sibling import other\nexcept KeyError:\n    pass\n"
    )
    assert census(tmp_path) == []
