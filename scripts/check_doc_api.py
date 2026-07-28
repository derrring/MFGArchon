#!/usr/bin/env python3
"""Ratchet: no document may teach an API the package does not have (Issue #1759).

Docs are the one artefact nothing checks. `local_ci.sh` runs `pytest tests/`, and no test
imports a doc example, so a rename silently leaves every tutorial that used the old name
teaching a `NameError`. The first sweep found 259 such claims across 110 files.

## Four checks

- `bad_imports`      `from mfgarchon.X import Y` where the package provides no such Y.
- `unknown_calls`    a call to a name the document neither defines nor gets from us.
- `bad_kwargs`       a parameter passed that the callable's signature does not accept.
- `drifted_sketches` a doc-defined class/function whose parameters contradict the real one.

## Why pure AST, and no imports

Importing the package makes the result depend on which optional dependencies happen to be
installed -- a doc referencing a torch-only symbol would be "unknown" on a machine without
torch and known on one with it, so the baseline would drift with the environment rather than
with the docs. Importing also has side effects (`mfgarchon.workflow` creates a workspace
directory on import). Everything here is read off the source tree.

The cost is that dynamically-created names are invisible. They are rare here, and a false
positive is visible in the report rather than silent.

## Scope decisions, each of which changed the count

- **Per file, not per block.** Tutorials define a helper in one block and use it in the next;
  treating each block as a closed scope reported 32 names that the document does define.
- **A name DEFINED in the doc shadows ours and is skipped; a name IMPORTED from mfgarchon is
  ours and stays checked.** Conflating them silenced every `bad_kwargs` finding -- an import
  is not a definition.
- **`**kwargs` in the real signature disables the parameter check** for that callable rather
  than guessing which extras are legitimate.
- `CHANGELOG.md` and `archive/` are exempt: an entry describing a v0.16 API is correct as
  written, and rewriting it would falsify the record.

## Known blind spot

A document that both **sketches** `class Foo` and calls the real `Foo` has the name shadowed,
so its calls are skipped. `NetworkHJBSolver(cfl_factor=...)` (#1757) is a live instance. The
counts are a lower bound, and `--self-test` asserts the checks that do work still fire.
"""

from __future__ import annotations

import argparse
import ast
import builtins
import json
import re
import sys
import warnings
from pathlib import Path

EXEMPT_DIRS = {"archive", ".github", ".git", "node_modules", ".venv", "build", "dist"}
EXEMPT_FILES = {Path("CHANGELOG.md")}
PLACEHOLDER = re.compile(
    r"^(Your|My|Custom|Example|Sample|Foo|Bar)|^(my|your|custom|example|sample|foo|bar|expensive|some)_"
)
FENCE = re.compile(r"^\s*```+\s*(\w*)\s*$")
CATEGORIES = ("bad_imports", "unknown_calls", "bad_kwargs", "drifted_sketches")
BUILTIN = set(dir(builtins)) | {"self", "cls", "np", "plt", "pd", "torch", "jax", "jnp"}


class PackageIndex:
    """What `mfgarchon` provides, read from source rather than imported."""

    def __init__(self, package_root: Path):
        self.provides: dict[str, set[str]] = {}
        self.params: dict[str, set[str] | None] = {}
        for path in sorted(package_root.rglob("*.py")):
            if set(path.relative_to(package_root.parent).parts) & EXEMPT_DIRS:
                continue
            try:
                tree = ast.parse(path.read_text(errors="replace"))
            except SyntaxError:
                continue
            rel = path.relative_to(package_root.parent).with_suffix("")
            parts = list(rel.parts)
            if parts[-1] == "__init__":
                parts.pop()
            module = ".".join(parts) if parts else package_root.name
            names = self.provides.setdefault(module, set())
            # Top-level definitions only: a class defined inside a function is not an export.
            for node in tree.body:
                if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                    names.add(node.name)
                    self._record_params(node)
                elif isinstance(node, ast.Assign):
                    for t in node.targets:
                        if isinstance(t, ast.Name):
                            names.add(t.id)
            # Imports and __all__ entries anywhere in the file, not just at top level. Optional
            # dependencies are re-exported from inside `try:` blocks and `if TYPE_CHECKING:`
            # guards -- `backends/__init__.py` hides 7 of its 11 imports that way and
            # `reinforcement/environments/__init__.py` hides 22 of 54. Reading only the top
            # level made 29% of this check's findings false positives, measured on a random
            # sample of 14; reading the whole tree brought that to 0 of 20 on the same method.
            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    names.update(a.asname or a.name.split(".")[0] for a in node.names if a.name != "*")
                elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                    continue
            names.update(self._all_entries(tree))
        self.known: set[str] = {n for v in self.provides.values() for n in v}

    @staticmethod
    def _all_entries(tree: ast.Module) -> set[str]:
        """Names added to `__all__` anywhere, including `__all__.extend([...])`."""
        out: set[str] = set()
        for node in ast.walk(tree):
            targets: list[ast.expr] = []
            if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "__all__" for t in node.targets
            ):
                targets = [node.value]
            elif (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in ("extend", "append")
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "__all__"
            ):
                targets = list(node.args)
            for t in targets:
                if isinstance(t, (ast.List, ast.Tuple, ast.Set)):
                    out.update(e.value for e in t.elts if isinstance(e, ast.Constant) and isinstance(e.value, str))
                elif isinstance(t, ast.Constant) and isinstance(t.value, str):
                    out.add(t.value)
        return out

    def _record_params(self, node: ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        target: ast.FunctionDef | ast.AsyncFunctionDef | None
        if isinstance(node, ast.ClassDef):
            target = next(
                (
                    b
                    for b in node.body
                    if isinstance(b, (ast.FunctionDef, ast.AsyncFunctionDef)) and b.name == "__init__"
                ),
                None,
            )
        else:
            target = node
        if target is None:
            return
        # First definition wins: a name defined in two modules is ambiguous, and guessing
        # which one a document meant would produce findings that depend on walk order.
        if node.name in self.params:
            self.params[node.name] = None
            return
        if target.args.kwarg is not None:
            self.params[node.name] = None
            return
        self.params[node.name] = {
            a.arg for a in [*target.args.args, *target.args.kwonlyargs, *target.args.posonlyargs]
        } - {"self", "cls"}

    def module_provides(self, module: str, name: str) -> bool:
        return name in self.provides.get(module, set())

    def module_known(self, module: str) -> bool:
        return module in self.provides


def python_blocks(path: Path):
    """Fenced blocks whose language is python, py, or unset."""
    lang, buf, start, inside = None, [], 0, False
    for i, line in enumerate(path.read_text(errors="replace").splitlines(), 1):
        m = FENCE.match(line)
        if m:
            if inside:
                if lang in ("python", "py", ""):
                    yield start, "\n".join(buf)
                inside, buf = False, []
            else:
                inside, lang, start = True, m.group(1), i
            continue
        if inside:
            buf.append(line)


def scan_document(path: Path, index: PackageIndex) -> dict[str, list[str]]:
    found: dict[str, list[str]] = {c: [] for c in CATEGORIES}
    text = path.read_text(errors="replace")
    if "mfgarchon" not in text:
        return found

    parsed = []
    for lineno, src in python_blocks(path):
        try:
            with warnings.catch_warnings():
                # Doc blocks are untrusted text; a stray backslash is theirs, not ours, and
                # a SyntaxWarning on stderr would make this ratchet look like it is failing.
                warnings.simplefilter("ignore", SyntaxWarning)
                parsed.append((lineno, ast.parse(src)))
        except SyntaxError:
            continue  # pseudo-code fragment: not a checkable claim

    defined: set[str] = set()
    foreign: set[str] = set()
    for lineno, tree in parsed:
        for n in ast.walk(tree):
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                defined.add(n.name)
            elif isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
                defined.add(n.id)
            elif isinstance(n, ast.arg):
                defined.add(n.arg)
            elif isinstance(n, ast.ImportFrom):
                module = n.module or ""
                if module.startswith("mfgarchon"):
                    for a in n.names:
                        if a.name == "*":
                            continue
                        if not index.module_known(module) or not index.module_provides(module, a.name):
                            found["bad_imports"].append(f"{path}:{lineno + n.lineno}: from {module} import {a.name}")
                else:
                    foreign.update(a.asname or a.name.split(".")[0] for a in n.names)
            elif isinstance(n, ast.Import):
                foreign.update(a.asname or a.name.split(".")[0] for a in n.names if not a.name.startswith("mfgarchon"))

    for lineno, tree in parsed:
        for n in ast.walk(tree):
            if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in index.known:
                real = index.params.get(n.name)
                if real is not None:
                    sketched = _sketched_params(n)
                    extra = sketched - real
                    if extra:
                        found["drifted_sketches"].append(
                            f"{path}:{lineno + n.lineno}: {n.name} sketched with {sorted(extra)}"
                        )
            if not (isinstance(n, ast.Call) and isinstance(n.func, ast.Name)):
                continue
            name = n.func.id
            if name in defined or name in foreign or name in BUILTIN or PLACEHOLDER.match(name):
                continue
            if name not in index.known:
                found["unknown_calls"].append(f"{path}:{lineno + n.lineno}: {name}(...)")
                continue
            allowed = index.params.get(name)
            if allowed is None:
                continue
            for kw in n.keywords:
                if kw.arg and kw.arg not in allowed:
                    found["bad_kwargs"].append(f"{path}:{lineno + n.lineno}: {name}({kw.arg}=...)")
    return found


def _sketched_params(node: ast.ClassDef | ast.FunctionDef) -> set[str]:
    target = node
    if isinstance(node, ast.ClassDef):
        init = next((b for b in node.body if isinstance(b, ast.FunctionDef) and b.name == "__init__"), None)
        if init is None:
            return set()
        target = init
    return {a.arg for a in [*target.args.args, *target.args.kwonlyargs]} - {"self", "cls"}


def scan(repo: Path) -> dict[str, list[str]]:
    index = PackageIndex(repo / "mfgarchon")
    results: dict[str, list[str]] = {c: [] for c in CATEGORIES}
    for path in sorted(repo.rglob("*.md")):
        rel = path.relative_to(repo)
        if set(rel.parts) & EXEMPT_DIRS or rel in EXEMPT_FILES:
            continue
        for cat, hits in scan_document(path, index).items():
            results[cat] += [h.replace(str(repo) + "/", "") for h in hits]
    return results


def self_test(repo: Path) -> int:
    """Construct a document that must be caught, and require every check to fire.

    A ratchet whose checks have gone inert reports zero and reads like success. This is the
    positive control: it is not enough that the count is stable, the checks must still work.
    """
    import tempfile

    doc = "\n".join(
        [
            "```python",
            "from mfgarchon.geometry import ThisClassDoesNotExistAnywhere",
            "from mfgarchon.geometry import TensorProductGrid",
            "",
            "result = a_function_that_does_not_exist(1, 2)",
            "grid = TensorProductGrid(bounds=[(0, 1)], this_parameter_is_invented=True)",
            "",
            "",
            "# Sketched name is deliberately DIFFERENT from the one called above: sketching and",
            "# calling the same name shadows it, which is this module's documented blind spot.",
            "# The first version of this control used one name for both and the self-test caught",
            "# it, which is the behaviour the self-test exists for.",
            "class GridNetwork:",
            "    def __init__(self, an_invented_parameter):",
            "        pass",
            "```",
        ]
    )
    index = PackageIndex(repo / "mfgarchon")
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "control.md"
        p.write_text(doc)
        found = scan_document(p, index)

    failures = []
    for cat in CATEGORIES:
        if not found[cat]:
            failures.append(f"  {cat}: did not fire on a document built to trigger it")
    if failures:
        print("SELF-TEST FAILED — the ratchet cannot see what it claims to check:")
        print("\n".join(failures))
        return 1
    print(f"self-test OK — all {len(CATEGORIES)} checks fire on the control document")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--path", default=".", help="Repository root")
    parser.add_argument("--limit", type=int, default=15, help="Lines printed per category")
    parser.add_argument("--all", action="store_true", help="Print every finding")
    parser.add_argument("--self-test", action="store_true", help="Verify each check still fires")
    parser.add_argument("--write-baseline", metavar="FILE", help="Write current counts and exit")
    parser.add_argument(
        "--check-baseline",
        metavar="FILE",
        help=(
            "Fail when a count rises. Also fails when one DROPS, so a fix cannot land without "
            "recording it -- regenerate with --write-baseline."
        ),
    )
    args = parser.parse_args()
    repo = Path(args.path).resolve()

    if args.self_test:
        sys.exit(self_test(repo))

    results = scan(repo)
    counts = {c: len(results[c]) for c in CATEGORIES}

    if args.write_baseline:
        Path(args.write_baseline).write_text(json.dumps(counts, indent=2) + "\n")
        print(f"baseline written to {args.write_baseline}: {counts}")
        sys.exit(0)

    if args.check_baseline:
        if self_test(repo) != 0:
            sys.exit(1)
        baseline = json.loads(Path(args.check_baseline).read_text())
        rose = {c: (baseline.get(c, 0), counts[c]) for c in CATEGORIES if counts[c] > baseline.get(c, 0)}
        fell = {c: (baseline.get(c, 0), counts[c]) for c in CATEGORIES if counts[c] < baseline.get(c, 0)}
        if rose:
            print("Docs teach MORE API the package does not have than the baseline records:")
            for c, (was, now) in rose.items():
                print(f"  {c}: {was} -> {now}")
                for line in results[c][: args.limit]:
                    print(f"      {line}")
            sys.exit(1)
        if fell:
            print("Doc-API findings DECREASED — tighten the baseline so the gain is recorded:")
            for c, (was, now) in fell.items():
                print(f"  {c}: {was} -> {now}")
            print("  python scripts/check_doc_api.py --path . --write-baseline scripts/doc_api_baseline.json")
            sys.exit(1)
        print(f"doc-API findings unchanged vs baseline: {counts}")
        sys.exit(0)

    total = sum(counts.values())
    print(f"{total} claims the package does not support\n")
    for cat in CATEGORIES:
        hits = results[cat]
        print(f"=== {cat}: {len(hits)}")
        shown = hits if args.all else hits[: args.limit]
        for line in shown:
            print(f"  {line}")
        if len(hits) > len(shown):
            print(f"  ... and {len(hits) - len(shown)} more (--all)")
        print()
    sys.exit(1 if total else 0)


if __name__ == "__main__":
    main()
