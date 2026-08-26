"""`mfgarchon.utils` is below `mfgarchon.alg` and must not import upward.

The inversion is what creates the cycle that blocks decoupling, and that is why this file exists.

~~Every heavy package enters through one line~~ and ~~`import mfgarchon` is 4.89s~~ [RETRACTED
2026-08-14, see #1930] — both were refuted by measurement after this file was written. It is not
a single entry point: with that line cut, torch arrives through `utils/__init__.py`'s
`from .geometry import` → `utils/geometry.py`, instead of its `from .adjoint_validation import`
→ `adjoint_validation.py`. `utils/__init__.py` is an
18-import, 106-name re-export hub, so cutting edges one at a time is a treadmill — all three
planned cuts applied together moved the total 4.31s → 4.62s, which is to say not at all.

~~torch arrives via `utils/data/polars_integration.py`~~ [CORRECTED 2026-08-14] — that named
the wrong witness. The probe watched `find_spec("torch")`, and polars PROBES for torch for its
`to_torch()` interop without importing it; `import polars` alone leaves `torch` out of
`sys.modules`. Re-measured by intercepting the module's actual execution: both routes converge
on the same leaf, `nonlinear_solvers.py` → `utils/acceleration/__init__.py` →
`torch_utils.py`. That convergence is the more useful fact and neither the original analysis
nor its first correction had it. The absolute figure is a property of
one process anyway (4.89s here, 6.37s cold and 4.04s warm elsewhere); what reproduces is the
decomposition, roughly half third-party and half the library importing itself.

A retracted rationale living on in a permanent test file is worse than none, which is why it is
struck here rather than quietly rewritten.

That line cannot simply be made lazy. Removing it, or replacing it with a PEP 562 `__getattr__`,
both fail with

    ImportError: cannot import name 'clip_nonnegative_or_raise' from partially initialized
    module 'mfgarchon.utils.numerical' (most likely due to a circular import)

because `utils/numerical/__init__.py` imports `alg`, which reaches `fp_network`, which needs a
name from `utils.numerical` while it is still stopped at that line. The eager import is
load-bearing: it completes `utils.numerical` by another route first and hides the cycle. #1930.

The count is pinned by EQUALITY, not bounded. ~~ratchets the count DOWN only~~ [CORRECTED
2026-08-14] — that described the `<=` form which the same commit replaced with `==`. A bound
cannot tell "an edge was removed" from "the scanner stopped seeing edges", and this file had two
live ways for the second to happen. Two edges is not a claim that two are acceptable: they are
the two that need a design decision, and each removal lowers `EXPECTED_MODULE_LEVEL` with it.
"""

from __future__ import annotations

import ast
import importlib.util
import os
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
UTILS = REPO / "mfgarchon" / "utils"

# The one that remains:
#   utils/adjoint_validation.py     imports one enum, `SchemeFamily`, from alg.base_solver
#
# ~~utils/numerical/__init__.py re-exports `gfdm_strategies`~~ removed 2026-08-14 (#1930 step 3).
# It was the cycle's back-edge and had zero consumers. Removing it is what makes deferring
# `utils/__init__.py`'s eager import possible at all -- measured: with the re-export present that
# deferral raises a circular ImportError, with it gone it succeeds.
#
# Pinned by EQUALITY, not by `<=`. That was the first form, and review named the reason it is
# wrong: a bound cannot tell "an edge was removed" from "the scanner stopped seeing edges", and
# this file had two live ways to stop seeing them (a module-level `try/except`, and a relative
# import -- both now fixed, both silently under a `<=`). It is also the house convention, for a
# recorded reason: `scripts/check_single_source.py` records a change "which removes nothing"
# dropping a count 6 -> 0 and printing SHRANK.
EXPECTED_MODULE_LEVEL = 1


def _absolute_target(node: ast.Import | ast.ImportFrom, path: Path) -> str:
    """The absolute module name an import names, resolving `from ...alg import x`.

    Delegated to `importlib.util.resolve_name`, which is the function Python itself uses. Two
    hand-derived versions of this arithmetic were wrong before it: the first ignored relative
    imports entirely, and the second stripped the last path component only for `__init__.py` --
    but a plain module's package is also `parts[:-1]`, so every non-package file resolved one
    level too deep and 10 of 16 ground-truth cases mismatched. The single mutation used to
    validate that version happened to land in the one file shape where it was right.

    Direction of the old bug, for the record: it produced false negatives only. Every computed
    prefix began `mfgarchon.utils`, which can never match `mfgarchon.alg`.
    """
    if isinstance(node, ast.Import):
        return node.names[0].name if node.names else ""
    if not node.level:
        return node.module or ""
    parts = path.relative_to(REPO).with_suffix("").parts
    package = ".".join(parts[:-1])  # unconditional: a module's package is its parent either way
    try:
        return importlib.util.resolve_name("." * node.level + (node.module or ""), package)
    except (ImportError, ValueError):
        # Beyond the top-level package. Python raises here too, so the file could not import;
        # returning "" keeps this scanner from inventing a name for something that cannot exist.
        return ""


# Statement types whose bodies execute when the module is read. `ast.iter_child_nodes` plus an
# `isinstance(child, ast.stmt)` filter was the first form and it silently dropped two of these:
# `ast.ExceptHandler` and `ast.match_case` are NOT `ast.stmt` subclasses, so an import in an
# `except ImportError:` handler -- the idiomatic fallback half of `try: new / except: old`, i.e.
# the same shape family as the defect this guard exists to catch -- was scored deferred, as was
# a `match`/`case` body. Neither exclusion was chosen by anyone; both fell out of the filter.
_CONTAINERS_THAT_RUN = (ast.stmt, ast.ExceptHandler, ast.match_case)


def _runs_at_import(node: ast.AST, tree: ast.Module) -> bool:
    """Whether this import executes when the module is read.

    Stops at exactly two things:

    - `def` / `async def` -- a body that runs on call.
    - `if TYPE_CHECKING:` -- never true at runtime. Its `else:` branch DOES run, and is walked;
      an earlier version `continue`d past the whole `If` node and swept the `orelse` up with it.

    A `class` body is NOT skipped: it executes when the module is read. The earlier version
    skipped it with the comment "a body that runs on call, not on import", which is true of
    `FunctionDef` and false of `ClassDef`.

    `if False:` (and any literal-falsy test) is treated as dead, its `orelse` walked. That was
    documented as a known over-count first; review argued the stronger case for closing it --
    not that the false red is costly, but that leaving the row in the table below makes the
    eventual FIX fail a test whose message tells the contributor their fix is wrong.

    Two limits remain, and they are inherent rather than deferred:

    - **A nested `def` called at module level.** `def _boot(): import ...` followed by `_boot()`
      executes at import, and deciding that needs call-graph analysis. It is also the shape the
      failure message's own advice ("defer it into the function that uses it") can be misread
      into, so it is named here.
    - **`importlib.import_module(...)` / `__import__(...)`** are not `ast.Import` nodes and are
      invisible to this SCANNER. They are not invisible to the file: review injected a
      module-level `import_module("mfgarchon.alg...gfdm_strategies")` into
      `utils/numerical/__init__.py` and `test_the_hub_can_be_deferred_without_a_circular_import`
      caught it (`1 failed, 23 passed`) while the AST ratchet stayed green -- because such an edge
      recreates the cycle, which that test measures directly. What remains uncovered is a dynamic
      edge that does NOT recreate a cycle.
    """
    stack: list[ast.AST] = list(tree.body)
    while stack:
        current = stack.pop()
        if current is node:
            return True
        if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if isinstance(current, ast.If) and _is_type_checking(current.test):
            stack.extend(current.orelse)  # the else branch runs; the body does not
            continue
        if isinstance(current, ast.If) and isinstance(current.test, ast.Constant) and not current.test.value:
            stack.extend(current.orelse)  # `if False:` -- the body is dead, the else runs
            continue
        stack.extend(c for c in ast.iter_child_nodes(current) if isinstance(c, _CONTAINERS_THAT_RUN))
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
                f"{path.relative_to(REPO)} could not be parsed, so it is silently absent from every count below: {exc}"
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
        f"utils imports alg at module level in {len(module_level)} place(s), expected "
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


# Every statement shape an import can sit in, and whether it executes when the module is read.
# A table, because this scanner has been wrong twice and both times the validating mutation
# happened to pick a shape the code got right: first a module-level `try/except` scored as
# deferred, then a relative import resolved one level too deep everywhere except `__init__.py`.
# One mutation cannot establish a classifier; a table can.
#
# Thirteen rows, but NOT thirteen independent guarantees: `try body`, `try finally`, `with`,
# `module-level for`, `if not TYPE_CHECKING` and `version guard` all rest on the single
# `_CONTAINERS_THAT_RUN` line, and no mutation reddens one without the other five. Their value
# is the record of which shapes were considered. The rows carrying independent discrimination
# are `except handler`, `match case`, `class body`, `else of TYPE_CHECKING`, `if TYPE_CHECKING`
# and `function body`. Stated because a thirteen-row table reads as thirteen checks.
_IMPORT = "from mfgarchon.alg.base_solver import SchemeFamily"
RUNS_AT_IMPORT = [
    ("try body", f"try:\n    {_IMPORT}\nexcept ImportError:\n    pass\n", True),
    # The fallback half of `try: new / except ImportError: old` -- the same shape family as the
    # defect this guard exists to catch, and invisible to the first two versions of the walk.
    ("except handler", f"try:\n    pass\nexcept ImportError:\n    {_IMPORT}\n", True),
    ("try finally", f"try:\n    pass\nfinally:\n    {_IMPORT}\n", True),
    ("with", f"import contextlib\nwith contextlib.suppress(ImportError):\n    {_IMPORT}\n", True),
    ("module-level for", f"for _ in (1,):\n    {_IMPORT}\n", True),
    ("if not TYPE_CHECKING", f"from typing import TYPE_CHECKING\nif not TYPE_CHECKING:\n    {_IMPORT}\n", True),
    ("version guard", f"import sys\nif sys.version_info >= (3, 12):\n    {_IMPORT}\n", True),
    ("match case", f"match 1:\n    case 1:\n        {_IMPORT}\n", True),
    (
        "else of TYPE_CHECKING",
        f"from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    pass\nelse:\n    {_IMPORT}\n",
        True,
    ),
    # A class body executes when the module is read. An earlier version skipped it with the
    # comment "a body that runs on call", which is true of `def` and false of `class`.
    ("class body", f"class C:\n    {_IMPORT}\n", True),
    ("if TYPE_CHECKING", f"from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    {_IMPORT}\n", False),
    ("function body", f"def f():\n    {_IMPORT}\n", False),
    # Statically dead: closed rather than documented, see `_runs_at_import`.
    ("if False", f"if False:\n    {_IMPORT}\n", False),
    # ... but its `else` still runs.
    ("else of if False", f"if False:\n    pass\nelse:\n    {_IMPORT}\n", True),
]


@pytest.mark.parametrize(("label", "source", "runs"), RUNS_AT_IMPORT)
def test_the_walk_classifies_every_statement_shape(label, source, runs):
    tree = ast.parse(source)
    imports = [
        n
        for n in ast.walk(tree)
        if isinstance(n, (ast.Import, ast.ImportFrom))
        and (getattr(n, "module", None) or "").startswith("mfgarchon.alg")
    ]
    assert imports, f"the fixture for {label!r} contains no alg import; it would prove nothing"
    assert any(_runs_at_import(n, tree) for n in imports) is runs, (
        f"{label}: classified as {'running' if not runs else 'deferred'} at import, which is wrong. "
        f"An import that runs but is scored deferred is invisible to the ratchet; one scored "
        f"running when it does not is a red nobody can act on."
    )


# (level, file, expected absolute target) -- checked against `importlib.util.resolve_name`, which
# is what Python itself uses. The hand-derived arithmetic this replaced was right for `__init__.py`
# and wrong for every plain module, at every level.
RELATIVE_CASES = [
    ("mfgarchon/utils/numerical/__init__.py", 3, "alg.base_solver", "mfgarchon.alg.base_solver"),
    ("mfgarchon/utils/adjoint_validation.py", 2, "alg.base_solver", "mfgarchon.alg.base_solver"),
    ("mfgarchon/utils/convergence/convergence_monitors.py", 3, "alg.base_solver", "mfgarchon.alg.base_solver"),
    ("mfgarchon/utils/numerical/particle/sampling.py", 4, "alg.base_solver", "mfgarchon.alg.base_solver"),
    ("mfgarchon/utils/numerical/__init__.py", 1, "particle", "mfgarchon.utils.numerical.particle"),
]


@pytest.mark.parametrize(("rel", "level", "module", "expected"), RELATIVE_CASES)
def test_relative_imports_resolve_to_the_same_name_python_would_use(rel, level, module, expected):
    path = REPO / rel
    assert path.exists(), f"{rel} moved; this case is testing a file that is not there"
    node = ast.ImportFrom(module=module, names=[ast.alias(name="X")], level=level)
    assert _absolute_target(node, path) == expected


def test_the_hub_can_be_deferred_without_a_circular_import():
    """The cycle is broken, asserted as the thing that was blocked rather than as its cause.

    `utils/__init__.py`'s eager `from .adjoint_validation import (...)` is what makes every heavy
    package arrive on `import mfgarchon`, and #1930 step 5 is to defer it. Before the
    `gfdm_strategies` re-export was removed from `utils/numerical/__init__.py`, that deferral
    failed:

        ImportError: cannot import name 'clip_nonnegative_or_raise' from partially initialized
        module 'mfgarchon.utils.numerical' (most likely due to a circular import)

    because `utils.numerical` imported `alg`, which reached `fp_network.py`, which needed a name
    from `utils.numerical` while it was still executing that import. The eager hub import hid the
    cycle by completing `utils.numerical` through another route first -- so the cycle was only
    observable by attempting the very change it blocked.

    This performs the attempt in a subprocess against a patched copy of the tree, so the property
    is checked rather than remembered. If it ever fails again, an upward import has returned.
    """
    import shutil
    import subprocess
    import sys
    import tempfile

    hub = REPO / "mfgarchon" / "utils" / "__init__.py"
    source = hub.read_text()
    match = re.search(
        r"^# Adjoint duality validation \(Issue #580\)\nfrom \.adjoint_validation import \(([^)]*)\)\n", source, re.M
    )
    assert match, "the eager hub import moved; this test is pinning something that is no longer there"
    names = tuple(n.strip().rstrip(",") for n in match.group(1).split() if n.strip().rstrip(","))
    assert names, "parsed no re-exported names; the deferral below would prove nothing"

    lazy = (
        f"_LAZY = {names!r}\n\n\n"
        "def __getattr__(name):\n"
        "    if name in _LAZY:\n"
        "        from . import adjoint_validation as _m\n\n"
        "        return getattr(_m, name)\n"
        "    raise AttributeError(name)\n"
    )

    with tempfile.TemporaryDirectory() as tmp:
        tree = Path(tmp) / "tree"
        shutil.copytree(REPO / "mfgarchon", tree / "mfgarchon", ignore=shutil.ignore_patterns("__pycache__"))
        (tree / "mfgarchon" / "utils" / "__init__.py").write_text(source.replace(match.group(0), lazy, 1))
        # `PYTHONPATH` and an explicit `sys.path` insert, not cwd: `local_ci.sh` runs pytest
        # under `PYTHONSAFEPATH=1`, which drops cwd from `sys.path` -- and the subprocess then
        # resolved `mfgarchon` from the EDITABLE INSTALL, which imports fine, so this test passed
        # over a tree it never read. Measured: back-edge restored + PYTHONSAFEPATH=1 -> 1 passed,
        # same mutation without it -> 1 failed. Found by review.
        env = {**os.environ}
        # Prepend, do not overwrite: the previous form dropped the parent's PYTHONPATH, which
        # nothing here needs but a future runner might.
        env["PYTHONPATH"] = os.pathsep.join(x for x in (str(tree), os.environ.get("PYTHONPATH", "")) if x)
        # `pop`, not `= ""`. The empty string does clear the flag (CPython treats empty env vars
        # as unset), but it stays visible to anything checking presence rather than truth, and
        # "0" and " " both ENABLE it -- a value worth not having to remember.
        env.pop("PYTHONSAFEPATH", None)
        proc = subprocess.run(
            [
                sys.executable,
                "-c",
                f"import sys; sys.path.insert(0, {str(tree)!r})\nimport mfgarchon; print('TREE:' + mfgarchon.__file__)",
            ],
            cwd=tree,
            capture_output=True,
            text=True,
            timeout=300,
            env=env,
        )

    assert proc.returncode == 0, (
        "deferring `utils/__init__.py`'s eager import of `adjoint_validation` fails, which means "
        "an upward `utils -> alg` import has reintroduced the cycle:\n" + proc.stderr[-2000:]
    )
    # The token identifies the tree, not just success. "OK" would be printed just as happily by
    # the installed package, which is the whole failure mode above.
    reported = [line[len("TREE:") :] for line in proc.stdout.splitlines() if line.startswith("TREE:")]
    assert reported, f"the probe printed no tree token:\n{proc.stdout[-600:]}"
    # `resolve().is_relative_to`, not `startswith`. On macOS TMPDIR is a symlink, so the logical
    # spelling (`/var/folders/...`) and the physical one `os.getcwd()` returns
    # (`/private/var/folders/...`) differ, and a string prefix compares False on the correct
    # tree. It passed only because `sys.path.insert` fed the import machinery the same string
    # this line compares against -- a guard whose correctness depended on a co-located
    # mechanism, which is the hazard this whole assertion exists to remove. Review measured it:
    # with the insert taken away the CLEAN tree reddens too, and a check red in both states is
    # stuck rather than discriminating.
    assert Path(reported[0]).resolve().is_relative_to(tree.resolve()), (
        f"the subprocess imported {reported[0]}, not the patched copy at {tree}. The result says "
        f"nothing about the patch."
    )
