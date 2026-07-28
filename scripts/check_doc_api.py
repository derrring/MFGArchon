#!/usr/bin/env python3
"""Ratchet: no document may teach an API the package does not have (Issue #1759).

Docs are the one artefact nothing checks. `local_ci.sh` runs `pytest tests/`, and no test
imports a doc example, so a rename silently leaves every tutorial that used the old name
teaching a `NameError`. Run it with no arguments to see the current count and where the claims are.

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

## Known blind spots

The counts are a lower bound. `--stats` reports how many names each exemption currently
covers, computed from the tree rather than stated here -- a number written into this docstring
would be true when typed and silently false after the next rename.

Structural, and permanent unless the checks are extended:

- **Attribute-chain calls.** `mfgarchon.geometry.Domain2D(...)` is skipped; only bare
  `ast.Name` callables are checked. Bare `Domain2D(...)` is caught.
- **A doc that sketches `class Foo` and also calls the real `Foo`.** The name is shadowed
  file-wide, so its calls and kwargs are skipped. `NetworkHJBSolver(cfl_factor=...)` (#1757)
  is a live instance.
- **Positional arity.** Only keyword names are checked, never argument counts.
- **`import mfgarchon.does_not_exist`.** Only `from X import Y` is resolved.
- **Classes with no own `__init__`.** Parameters are inherited or dataclass-generated, so no
  signature is recorded and the kwarg check is skipped.
- **Names defined in two modules.** Ambiguous, so deliberately exempted rather than guessed.
- **Third-party names leak into the known set** via `import` statements, so `KDTree()` in a
  doc reads as provided. This weakens `unknown_calls` only; `bad_imports` uses the tighter
  per-module set.
- **One syntax error discards the whole block.**

Latent rather than live: fence tags other than `python`/`py`/empty are not read, and `~~~`
fences are not read. Neither form appears in this repo today.

`--self-test` guards against a check going *silent*, which is not the same as guarding its
coverage: a narrowing that still fires once on the control document passes. Closing that needs
a control with several distinct shapes per category (#1761).
"""

from __future__ import annotations

import argparse
import ast
import builtins
import json
import re
import subprocess
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
        # Modules whose surface a star import makes undeterminable from source.
        self.unresolvable: set[str] = set()
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
            # Definitions can sit under a top-level `if TORCH_AVAILABLE:` or `try:`, which is
            # how this package guards optional dependencies -- reading only `tree.body` made
            # every symbol in such a module read as absent. Descend one level into those.
            self._collect_definitions(tree.body, names)
            # Imports and __all__ entries anywhere in the file, not just at top level. Optional
            # dependencies are re-exported from inside `try:` blocks and `if TYPE_CHECKING:`
            # guards -- reading only the top level made this check substantially
            # false-positive; `--stats` reports the exemptions this leaves in force.
            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    names.update(a.asname or a.name.split(".")[0] for a in node.names if a.name != "*")
                    if any(a.name == "*" for a in node.names):
                        # A star import means this module's surface cannot be determined from
                        # its own source. Asserting a symbol is ABSENT from it would be a
                        # false positive, so the module is marked unresolvable instead.
                        self.unresolvable.add(module)
            names.update(self._all_entries(tree))
        self.known: set[str] = {n for v in self.provides.values() for n in v}

    def _collect_definitions(self, body: list[ast.stmt], names: set[str], depth: int = 0) -> None:
        """Top-level definitions, plus one level inside `if` / `try` guards."""
        for node in body:
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                names.add(node.name)
                self._record_params(node)
            elif isinstance(node, ast.Assign):
                for t in node.targets:
                    if isinstance(t, ast.Name):
                        names.add(t.id)
            elif isinstance(node, (ast.AnnAssign, ast.TypeAlias)):
                target = getattr(node, "target", None) or getattr(node, "name", None)
                if isinstance(target, ast.Name):
                    names.add(target.id)
            elif depth == 0 and isinstance(node, (ast.If, ast.Try)):
                for branch in (
                    getattr(node, "body", []),
                    getattr(node, "orelse", []),
                    getattr(node, "finalbody", []),
                ):
                    self._collect_definitions(branch, names, depth + 1)
                for handler in getattr(node, "handlers", []):
                    self._collect_definitions(handler.body, names, depth + 1)

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
        # A name defined in two modules is ambiguous: guessing which one a document meant
        # would produce findings that depend on walk order, so BOTH are discarded and the
        # kwarg check is skipped for that name. `--stats` counts how many that currently is.
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
                        submodule = f"{module}.{a.name}"
                        if module in index.unresolvable:
                            continue  # star import: this module's surface is undeterminable
                        if not index.module_known(module) or not (
                            index.module_provides(module, a.name) or index.module_known(submodule)
                        ):
                            found["bad_imports"].append(f"{path}:{lineno + n.lineno}: from {module} import {a.name}")
                else:
                    foreign.update(a.asname or a.name.split(".")[0] for a in n.names)
            elif isinstance(n, ast.Import):
                foreign.update(a.asname or a.name.split(".")[0] for a in n.names if not a.name.startswith("mfgarchon"))

    for lineno, tree in parsed:
        # Top level only. `ast.walk` picked up methods nested in doc-defined classes and
        # compared them against module-level functions of the same bare name -- a namespace
        # collision, not a drifted sketch. `UniversalLogger.log_convergence_analysis` was
        # flagged against `utils/mfg_logging/logger.py`'s free function of that name, which is
        # an unrelated object. The index records top-level definitions, so the comparison has
        # to be against top-level ones.
        for n in tree.body:
            # Names shorter than three characters are mathematical notation in this corpus,
            # not API claims: a doc defining `def f(x, y, m)` for a running cost collided with
            # a bare `f` the package happens to bind. Same namespace-collision class as
            # comparing a doc method against a top-level function of the same name.
            if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and len(n.name) > 2 and n.name in index.known:
                real = index.params.get(n.name)
                if real is not None:
                    sketched = _sketched_params(n)
                    extra = sketched - real
                    if extra:
                        found["drifted_sketches"].append(
                            f"{path}:{lineno + n.lineno}: {n.name} sketched with {sorted(extra)}"
                        )

    for lineno, tree in parsed:
        for n in ast.walk(tree):
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


def tracked_markdown(repo: Path) -> list[Path]:
    """`git ls-files`, not `rglob`.

    The baseline is committed, so the measurement has to be over files git knows about.
    Reading the filesystem instead makes the count depend on whatever is lying around: a
    worktree under `.claude/worktrees/` (which `.gitignore` does not cover, and which the
    review tooling creates) took `bad_imports` from 102 to 188, and a single untracked scratch
    note took it to 103. Either turns `local_ci.sh` red on an otherwise clean checkout.

    `ls-files` reports the **index**, not `HEAD`, and the index can name a path the working
    tree no longer has -- `rm foo.md` without `git rm` leaves the entry. Reading those blindly
    crashed the whole gate with a `FileNotFoundError`, and since `local_ci.sh` is wired into
    pre-push, that is every push after a plain `rm`. Missing entries are skipped and named:
    a deleted doc that carried findings still turns the ratchet red through the bidirectional
    check, so skipping cannot hide a regression.
    """
    out = subprocess.run(
        ["git", "-C", str(repo), "ls-files", "-z", "--", "*.md"],
        capture_output=True,
        text=True,
    )
    if out.returncode != 0:
        raise SystemExit(
            f"cannot list tracked markdown in {repo}: git exited {out.returncode}.\n"
            f"  {out.stderr.strip()}\n"
            f"This check scopes itself by git because its baseline is committed. Run it from "
            f"inside the repository, or pass --path pointing at the repository root."
        )
    # `dict.fromkeys`, not a list: during a merge conflict `ls-files` emits one line per
    # stage, so an unmerged doc appears three times and every finding in it is counted three
    # times -- the ratchet would go red on a conflict rather than on a doc defect.
    paths = [repo / rel for rel in dict.fromkeys(r for r in out.stdout.split("\0") if r)]
    missing = [p for p in paths if not p.exists()]
    if missing:
        shown = ", ".join(str(p.relative_to(repo)) for p in missing[:5])
        extra = f" (+{len(missing) - 5} more)" if len(missing) > 5 else ""
        print(f"note: {len(missing)} tracked doc(s) are gone from the working tree, skipping: {shown}{extra}")
        print("      `git rm` them to record the deletion, or restore them.")
    return [p for p in paths if p.exists()]


def scan(repo: Path) -> dict[str, list[str]]:
    index = PackageIndex(repo / "mfgarchon")
    results: dict[str, list[str]] = {c: [] for c in CATEGORIES}
    for path in sorted(tracked_markdown(repo)):
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


def report_stats(repo: Path) -> None:
    """Count the exemptions in force, from the tree.

    These numbers change with every rename, which is exactly why they are computed here
    instead of written into the docstring: a measurement in prose is true when typed and
    silently false afterwards, and nobody comes back to update it.
    """
    index = PackageIndex(repo / "mfgarchon")
    # Two shapes, and the second is easy to miss: a name can be recorded as `None` (ambiguous,
    # or **kwargs), or never recorded at all (a class with no own `__init__` returns before
    # recording). The first version of this counter measured only the first shape and reported
    # a fraction of the real exemption -- an undercount in the tool that counts exemptions.
    # Partition over the names that could HAVE a signature -- top-level classes and functions
    # defined in the package. Dividing by every known name conflates "is a constant, so of
    # course it has no signature" with "is a class whose signature could not be read", and
    # reports the exemption as several times larger than it is.
    definitions: set[str] = set()
    for path in sorted((repo / "mfgarchon").rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(errors="replace"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                definitions.add(node.name)
    definitions &= index.known
    # `scan_document` reaches a signature only through `index.known`, so a name recorded in
    # `params` but absent from `known` is never checked and must not count as covered.
    checkable = {n for n, v in index.params.items() if v is not None and n in index.known}
    explicitly_none = {n for n, v in index.params.items() if v is None} & definitions
    never_recorded = definitions - set(index.params)

    own_defs: set[str] = set()
    imported_only: set[str] = set()
    for path in sorted((repo / "mfgarchon").rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(errors="replace"))
        except SyntaxError:
            continue
        index._collect_definitions(tree.body, own_defs)
        own_defs |= index._all_entries(tree)
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                imported_only.update(a.asname or a.name.split(".")[0] for a in node.names if a.name != "*")
    third_party = imported_only - own_defs
    placeheld = sorted(n for n in index.known if PLACEHOLDER.match(n))

    covered = len(checkable & definitions)
    print(f"names the package provides                   {len(index.known)}")
    print(f"modules indexed                              {len(index.provides)}")
    print(f"markdown files tracked by git                {len(tracked_markdown(repo))}")
    print()
    print(f"top-level classes and functions              {len(definitions)}")
    print(f"  of those, kwargs CAN be checked            {covered}")
    print()
    print("exemptions in force among those definitions (each weakens the kwarg check):")
    print(f"  signature unusable -- **kwargs, or defined in two modules   {len(explicitly_none)}")
    print(f"  no signature recorded -- e.g. a class with no own __init__  {len(never_recorded)}")
    assert covered + len(explicitly_none) + len(never_recorded) == len(definitions), (
        "the three buckets must partition the definitions; they do not, so one of them is "
        "double-counting or leaving a gap"
    )
    print()
    print(f"known only via a third-party import (weakens unknown_calls)  {len(third_party)}")
    print(f"real package names matched by PLACEHOLDER and skipped        {len(placeheld)}")
    if placeheld:
        print(f"      {', '.join(placeheld[:8])}{' ...' if len(placeheld) > 8 else ''}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--path", default=".", help="Repository root")
    parser.add_argument("--limit", type=int, default=15, help="Lines printed per category")
    parser.add_argument("--all", action="store_true", help="Print every finding")
    parser.add_argument("--self-test", action="store_true", help="Verify each check still fires")
    parser.add_argument("--stats", action="store_true", help="Count the exemptions currently in force")
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

    if args.stats:
        report_stats(repo)
        sys.exit(0)

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
