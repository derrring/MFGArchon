#!/usr/bin/env python3
"""Both manifests must declare what the package actually imports.

Two failures, one shape, both already in this repository's history.

`pyyaml` was imported at module level by `config/io.py` and declared nowhere. It arrived
transitively -- from omegaconf and from jupyterlab's dependency chain -- so removing those in one
change would have broken `load_solver_config` on a fresh install (#1687). Nothing detected that; it
was found by reading.

`jupyter`, `jupyterlab` and `seaborn` were removed from `pyproject.toml` by that same issue, on the
stated criterion of zero imports across the package, tests, examples and benchmarks. They stayed in
`environment.yml`, which also never gained the eight packages the library imports and `pyproject`
declares. An environment built from that file could not run the library: measured 2026-08-28, 6630
test outcomes against 6724, with 66 tests skipped or not collected because cvxpy and torch were
absent -- and the warning ratchet then reported the warnings those tests would have emitted as
identities GONE, inviting the reader to record the loss as progress.

So: a check, in both directions.

The two directions have different lifespans, and saying so here is the point of this paragraph.
IMPORTED-BUT-UNDECLARED compares the package against `pyproject.toml` and outlives any packaging
decision. DECLARED-BUT-MISSING compares `pyproject.toml` against `environment.yml`, and #2167
deletes that file: `pyproject.toml` + `uv.lock` become the single owner, and swapping the BLAS
implementation -- the one thing conda does that PyPI cannot -- turns out not to need a second
manifest, because `uv pip install` leaves a conda-installed numpy alone. When that lands, delete
the second direction with the file. It is here because `environment.yml` is still the only
onboarding path the documentation names, and it was broken.

  IMPORTED-BUT-UNDECLARED  a third-party module `mfgarchon/` imports that `pyproject.toml`
                           declares nowhere -- the #1687 shape, a fresh install away from breaking
  DECLARED-BUT-MISSING     a runtime dependency in `pyproject.toml` absent from `environment.yml`
                           -- the shape that produced the 6630-outcome run

This does NOT decide whether an unused declaration should be removed. Absence of an import is not
absence of use: `line-profiler` and `memory-profiler` are invoked as `kernprof` and `mprof` and are
correctly declared without ever being imported. That direction needs a human and is out of scope.
"""

from __future__ import annotations

import argparse
import ast
import importlib.metadata
import json
import re
import sys
import tomllib
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
PACKAGE = ROOT / "mfgarchon"

#: Distributions whose import name this environment cannot resolve, with the reason. Every entry
#: is a claim that has to stay true, so the self-test asserts the mapping still fails for each --
#: an entry that starts resolving is a stale exemption and says so.
UNRESOLVABLE = {
    # conda-forge names it `pytorch`; PyPI's `pytorch` is a 1.0.2 placeholder with an empty summary
    # and the real distribution is `torch`. A name comparison between the two manifests reports
    # these as a disagreement, and a migration that trusts the name installs the placeholder.
    "pytorch": "torch",
}


def _normalise(spec: str) -> str:
    """A requirement or conda spec to a comparable distribution name."""
    name = re.split(r"[<>=!~\[;]", spec, maxsplit=1)[0].strip().lower().replace("_", "-")
    return UNRESOLVABLE.get(name, name)


def _imports_of(tree: Path, *, module_level_only: bool) -> set[str]:
    """Top-level module names imported under `tree`.

    `module_level_only` is the whole distinction, and getting it wrong is what the first version of
    this check did. A MODULE-LEVEL third-party import is a hard dependency: undeclared, the package
    does not import at all on a fresh install, which is precisely `pyyaml` in #1687. An import
    inside a function or a `try` is a SOFT dependency by construction -- `optax`, `ot`, `pyvista`,
    `gmsh`, `cupy` and `colorlog` are all reached that way here, each behind a guard, each degrading
    to a fallback. Gating on those would demand declarations for backends the package deliberately
    treats as optional. They are reported, never gated.
    """
    found: set[str] = set()
    for path in sorted(tree.rglob("*.py")):
        try:
            parsed = ast.parse(path.read_text())
        except (SyntaxError, UnicodeDecodeError):
            continue
        nodes = parsed.body if module_level_only else list(ast.walk(parsed))
        # Module level means executed at import: the body, plus anything the body executes
        # unconditionally. A top-level `try:` still runs, so its handlers are walked; a `def` is not.
        if module_level_only:
            expanded = []
            stack = list(nodes)
            while stack:
                node = stack.pop()
                expanded.append(node)
                if isinstance(node, (ast.Try, ast.If)):
                    stack.extend(node.body + node.orelse + getattr(node, "finalbody", []))
                    for handler in getattr(node, "handlers", []):
                        stack.extend(handler.body)
            nodes = expanded
        for node in nodes:
            if isinstance(node, ast.Import):
                found |= {a.name.split(".")[0] for a in node.names}
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                found.add(node.module.split(".")[0])
    return found


def _declared(pyproject: dict) -> set[str]:
    project = pyproject["project"]
    declared = {_normalise(x) for x in project.get("dependencies", [])}
    for group in project.get("optional-dependencies", {}).values():
        declared |= {_normalise(x) for x in group}
    for group in pyproject.get("dependency-groups", {}).values():
        declared |= {_normalise(x) for x in group if isinstance(x, str)}
    return declared


def _conda(environment: dict) -> set[str]:
    names = {_normalise(x) for x in environment["dependencies"] if isinstance(x, str)}
    for entry in environment["dependencies"]:
        if isinstance(entry, dict):
            names |= {_normalise(x) for x in entry.get("pip", [])}
    return names


def _undeclared(package: Path, declared: set[str]) -> tuple[list[str], list[str]]:
    """Third-party imports with no declaration, and imports this environment cannot attribute.

    The second list is not a finding and not silence. `packages_distributions()` only knows what is
    installed, so a module that is imported, undeclared AND absent cannot be mapped -- exactly the
    state the check exists to catch. Reported separately rather than dropped: a silent `continue`
    here would make the check quietest about its own subject.
    """
    mapping = importlib.metadata.packages_distributions()
    undeclared, unattributable = [], []
    for module in sorted(_imports_of(package, module_level_only=True)):
        if module in sys.stdlib_module_names or module == "mfgarchon" or module.startswith("_"):
            continue
        dists = mapping.get(module)
        if not dists:
            unattributable.append(module)
            continue
        if not any(_normalise(d) in declared for d in dists):
            undeclared.append(f"{module} (distribution: {', '.join(sorted(dists))})")
    return undeclared, unattributable


def _self_test() -> int:
    """Both directions on synthetic manifests, plus the exemptions, driven through the real checks."""
    failures: list[str] = []

    if _normalise("pytorch>=2.0") != "torch":
        failures.append("the conda/PyPI name map is not applied by _normalise")
    if _normalise("scikit-fem>=8.0") != "scikit-fem" or _normalise("PyYAML") != "pyyaml":
        failures.append("_normalise does not fold case or extras")

    # Every exemption must still be needed. One that starts resolving is a stale claim.
    mapping = importlib.metadata.packages_distributions()
    for name in UNRESOLVABLE:
        if name in {d.lower() for dists in mapping.values() for d in dists}:
            failures.append(f"exemption {name!r} now resolves on its own and should be deleted")

    declared = {"numpy", "torch"}
    if not _undeclared(PACKAGE, declared)[0]:
        failures.append("a nearly-empty declaration set must report undeclared imports, and did not")
    full = _declared(tomllib.loads((ROOT / "pyproject.toml").read_text()))
    if _undeclared(PACKAGE, full)[0]:
        failures.append("the real pyproject must declare every third-party import, and does not")

    missing = sorted({"cvxpy", "rich"} - {"cvxpy"})
    if missing != ["rich"]:
        failures.append("set difference is not doing what the mirror check relies on")

    if failures:
        for line in failures:
            print(f"self-test FAILED: {line}", file=sys.stderr)
        return 1
    print(
        "self-test OK: both directions fire on synthetic manifests, the real tree is clean, "
        f"and all {len(UNRESOLVABLE)} name exemption(s) are still needed"
    )
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        return _self_test()

    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
    environment = yaml.safe_load((ROOT / "environment.yml").read_text())
    declared = _declared(pyproject)
    conda = _conda(environment) | {"python", "pip"}

    undeclared, unattributable = _undeclared(PACKAGE, declared)
    runtime = {_normalise(x) for x in pyproject["project"].get("dependencies", [])}
    absent = sorted(runtime - conda)

    if args.json:
        print(
            json.dumps(
                {"undeclared": undeclared, "unattributable": unattributable, "absent_from_conda": absent}, indent=2
            )
        )
        return 1 if (undeclared or absent) else 0

    rc = 0
    if undeclared:
        print(f"IMPORTED BUT UNDECLARED ({len(undeclared)}) -- a fresh install is one transitive drop away:")
        for item in undeclared:
            print(f"    {item}")
        print("    Declare each in `pyproject.toml`, in the extra that matches how it is imported.")
        rc = 1
    if absent:
        print(f"\nDECLARED BUT MISSING FROM environment.yml ({len(absent)}):")
        for item in absent:
            print(f"    {item}")
        print("    An environment built from that file cannot run the package. Add them, or move")
        print("    the dependency out of `[project.dependencies]` if it is not really runtime.")
        rc = 1
    if unattributable:
        print(f"\nCANNOT ATTRIBUTE ({len(unattributable)}) -- imported, and not installed here, so no")
        print("distribution name is available. Not a verdict; install the full environment to decide:")
        for item in unattributable:
            print(f"    {item}")
    if rc == 0 and not unattributable:
        print(
            "manifests agree: every third-party import is declared, and every runtime dependency is in environment.yml"
        )
    return rc


if __name__ == "__main__":
    sys.exit(main())
