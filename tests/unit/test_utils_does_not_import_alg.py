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
#
# Pinned by EQUALITY, not by `<=`. That was the first form, and review named the reason it is
# wrong: a bound cannot tell "an edge was removed" from "the scanner stopped seeing edges", and
# this file had two live ways to stop seeing them (a module-level `try/except`, and a relative
# import -- both now fixed, both silently under a `<=`). It is also the house convention, for a
# recorded reason: `scripts/check_single_source.py:47` records a change "which removes nothing"
# dropping a count 6 -> 0 and printing SHRANK.
EXPECTED_MODULE_LEVEL = 2


def _absolute_target(node: ast.Import | ast.ImportFrom, path: Path) -> str:
    """The absolute module name an import names, resolving `from ...alg import x`.

    Relative imports were invisible to the first version: it read `node.module`, which for
    `from ...alg.base_solver import SchemeFamily` is `alg.base_solver` with `level == 3`, and
    the `mfgarchon.` prefix test then failed. Review demonstrated a working relative import of
    `alg` that the ratchet scored as absent. Not exploitable by accident today -- all 54
    relative imports under utils/ are `level == 1` -- but one refactor away.
    """
    if isinstance(node, ast.Import):
        return node.names[0].name if node.names else ""
    if not node.level:
        return node.module or ""
    package = path.relative_to(REPO).with_suffix("").parts
    if path.name == "__init__.py":
        package = package[:-1]
    base = package[: len(package) - (node.level - 1)] if node.level > 1 else package
    return ".".join([*base, node.module]) if node.module else ".".join(base)


def _runs_at_import(node: ast.AST, tree: ast.Module) -> bool:
    """Whether this import executes when the module is read.

    `id(node) in {id(n) for n in tree.body}` was the first test, and it recognised only imports
    that are DIRECT children of the module body. A module-level `try: import x / except
    ImportError:` executes at import and costs the full price, and was scored `deferred` --
    while 18 files under utils/ already use exactly that idiom. Review mutated one in and the
    ratchet stayed green.

    So: walk down from the module body through every statement that still executes at import
    (`try`, `if`, `with`, `for`, `else` branches), and stop at `def`, `class`, and
    `if TYPE_CHECKING:` -- those genuinely do not run, and TYPE_CHECKING costs nothing at all.
    """
    stack: list[tuple[ast.AST, bool]] = [(n, True) for n in tree.body]
    while stack:
        current, _ = stack.pop()
        if current is node:
            return True
        if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue  # a body that runs on call, not on import
        if isinstance(current, ast.If) and _is_type_checking(current.test):
            continue  # never true at runtime
        for child in ast.iter_child_nodes(current):
            if isinstance(child, ast.stmt):
                stack.append((child, True))
    return False


def _is_type_checking(test: ast.expr) -> bool:
    return (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING") or (
        isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"
    )


def _alg_imports() -> list[tuple[str, str, int, str]]:
    """(kind, file, line, target) for every import of `mfgarchon.alg` under utils/.

    Parsed, not grepped: `grep -c mfgarchon.alg` over this tree returns 19, and 14 of those are
    comments, docstrings and strings. The distinction that matters is "runs at import" versus
    "does not", and only an AST can make it -- see the two helpers above for the two ways the
    first version got that distinction wrong.
    """
    out = []
    for path in sorted(UTILS.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text())
        except (SyntaxError, UnicodeDecodeError) as exc:  # pragma: no cover - would be a real bug
            raise AssertionError(
                f"{path.relative_to(REPO)} could not be parsed, so it is silently absent from "
                f"every count below: {exc}"
            ) from exc
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Import, ast.ImportFrom)):
                continue
            target = _absolute_target(node, path)
            if not (target == "mfgarchon.alg" or target.startswith("mfgarchon.alg.")):
                continue
            kind = "module-level" if _runs_at_import(node, tree) else "deferred"
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
    assert len(module_level) == EXPECTED_MODULE_LEVEL, (
        f"utils imports alg at module level in {len(module_level)} places, expected "
        f"{EXPECTED_MODULE_LEVEL}. Each one pulls the whole alg tree into `import mfgarchon`.\n"
        + "\n".join(f"  {f}:{ln} -> {t}" for _k, f, ln, t in module_level)
        + "\nIf you ADDED one: defer it into the function that uses it, or move the shared name "
        "to a layer both may depend on. If you REMOVED one: lower EXPECTED_MODULE_LEVEL in the "
        "same commit -- and check the scanner still sees what remains, because a drop with no "
        "deletion behind it means this file stopped measuring, not that the graph improved."
    )


def test_the_deleted_shims_stay_deleted():
    """Named explicitly, because the ratchet above counts and this says which."""
    for name in ("anderson_acceleration.py", "gfdm_strategies.py"):
        shim = UTILS / "numerical" / name
        assert not shim.exists(), (
            f"{shim.relative_to(REPO)} is back. It was a re-export of the canonical module in "
            f"`alg`, overdue for removal since v0.21.0. Import from `mfgarchon.alg` directly."
        )


def test_a_lazy_shim_cannot_restore_a_deleted_import_path():
    """The guard above counts imports; a PEP 562 `__getattr__` is not one.

    Review demonstrated the hole: appending a module-level `__getattr__` to
    `utils/numerical/__init__.py` that resolves `gfdm_strategies` and `anderson_acceleration` to
    their canonical `alg` modules restores BOTH deprecated import paths -- DeprecationWarning and
    all -- while adding no file and no module-level import. Every assertion above stayed green.

    That is not a hypothetical construction here: #1930's own step 4 proposes making
    `utils/__init__.py` a PEP 562 lazy re-export hub. This asserts the deleted paths in
    particular stay dead, whatever mechanism is used, so that work cannot restore them by
    accident on its way past.
    """
    import importlib

    parent = importlib.import_module("mfgarchon.utils.numerical")
    for name in ("gfdm_strategies", "anderson_acceleration"):
        full = f"mfgarchon.utils.numerical.{name}"
        # BOTH forms, because they take different routes. `import_module` does NOT consult a
        # module-level `__getattr__` -- verified: a package with `__getattr__` returning a module
        # answers `getattr(pkg, "sub")` and still raises ModuleNotFoundError for
        # `import_module("pkg.sub")`. A test that checked only the first would have passed over
        # exactly the lazy shim it exists to catch, which is how the first version of this test
        # was written.
        try:
            importlib.import_module(full)
        except ModuleNotFoundError:
            pass
        else:
            raise AssertionError(f"{full} imports again as a module.")
        resolved = getattr(parent, name, None)
        assert resolved is None, (
            f"`mfgarchon.utils.numerical.{name}` resolves again, as an attribute rather than a "
            f"file -- a lazy re-export restores the deprecated path while adding no module-level "
            f"import, so neither the count nor the file check above can see it. It was deleted as "
            f"a shim overdue since v0.21.0; import from `mfgarchon.alg` directly."
        )

    # Positive control: the canonical paths MUST import, or the loop above passes because the
    # package is broken rather than because the shims are gone.
    for path in (
        "mfgarchon.alg.numerical.gfdm_components.gfdm_strategies",
        "mfgarchon.alg.numerical.coupling.anderson_acceleration",
    ):
        importlib.import_module(path)
