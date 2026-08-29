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
#: conda spellings that differ from the PyPI distribution name. Applied symmetrically by
#: `_normalise` so the two manifests compare on one vocabulary. PyPI `pytorch` is a 1.0.2 placeholder
#: from 2019; the real distribution is `torch`. Dies with `environment.yml` in #2167.
CONDA_TO_PYPI = {
    "pytorch": "torch",
}

#: Import names whose distribution is spelled differently. Only needed when the distribution is not
#: installed, because `packages_distributions()` answers for the ones that are. Each entry is
#: asserted still necessary by `--self-test`: the fallback uses the import name, so an entry whose
#: import name already resolves is dead weight that will outlive its reason.
IMPORT_TO_DISTRIBUTION = {
    "skfem": "scikit-fem",
    "ot": "POT",
    "yaml": "pyyaml",
    "PIL": "pillow",
}


def _normalise(spec: str) -> str:
    """A requirement or conda spec to a comparable distribution name."""
    name = re.split(r"[<>=!~\[;]", spec, maxsplit=1)[0].strip().lower().replace("_", "-")
    return CONDA_TO_PYPI.get(name, name)


def _catches_import_error(handler: ast.ExceptHandler) -> bool:
    """Whether this `except` clause would swallow a failed import."""
    names = {"ImportError", "ModuleNotFoundError", "Exception", "BaseException"}
    node = handler.type
    if node is None:  # bare `except:`
        return True
    parts = node.elts if isinstance(node, ast.Tuple) else [node]
    return any(isinstance(x, ast.Name) and x.id in names for x in parts)


def _imports_of(tree: Path) -> tuple[set[str], set[str]]:
    """Top-level import names in `tree`, split into UNGUARDED and GUARDED.

    Only the unguarded ones are gated, and the distinction is the whole check.

    `pyyaml` was imported at module level by `config/io.py`, declared nowhere, and arriving
    transitively; dropping the packages that carried it would have raised ImportError at
    `import mfgarchon.config.io` on a fresh install (#1687). **That is what an undeclared dependency
    costs, and it costs it only when the import is unguarded.** A module-level
    `try: import cvxpy / except ImportError: HAVE_CVXPY = False` cannot break an install: the module
    sets a flag and carries on. Gating those is how a check acquires false findings that teach
    people to ignore it.

    Measured on this tree: `yaml`, `numpy` and `rich` have unguarded module-level imports; `cvxpy`,
    `torch`, `networkx`, `colorlog`, `optax` and `ot` are guarded at every module-level site. The
    first group must be declared. The second is reported, never gated.

    Imports inside a function body are excluded from both: those are deliberate lazy loads (#1930)
    and there are hundreds of them.
    """
    unguarded: set[str] = set()
    guarded: set[str] = set()

    def record(node: ast.AST, target: set[str]) -> None:
        if isinstance(node, ast.Import):
            target.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            target.add(node.module.split(".")[0])

    def walk(body: list[ast.stmt], *, shielded: bool) -> None:
        for node in body:
            record(node, guarded if shielded else unguarded)
            # Every compound statement whose body runs at import time. `ast.TryStar` is NOT an
            # `ast.Try`, so `except*` needs its own arm or both halves are lost; a class body
            # executes at class-creation time, which is import time; module-level `for`/`while` and
            # `match` are rare and do execute. Function bodies are deliberately absent -- those are
            # lazy loads (#1930) and there are hundreds of them.
            if isinstance(node, ast.Try | ast.TryStar):
                covered = shielded or any(_catches_import_error(h) for h in node.handlers)
                walk(node.body, shielded=covered)
                for handler in node.handlers:
                    walk(handler.body, shielded=True)
                walk(node.orelse, shielded=shielded)
                walk(node.finalbody, shielded=shielded)
            elif isinstance(node, ast.If | ast.For | ast.AsyncFor | ast.While):
                walk(node.body, shielded=shielded)
                walk(node.orelse, shielded=shielded)
            elif isinstance(node, ast.ClassDef):
                walk(node.body, shielded=shielded)
            elif isinstance(node, ast.Match):
                for case in node.cases:
                    walk(case.body, shielded=shielded)
            elif isinstance(node, ast.With | ast.AsyncWith):
                # `contextlib.suppress(ImportError)` is a guard by another spelling. Only the
                # context expressions are inspected: dumping the whole node would let an unrelated
                # `ImportError` anywhere in the body read as a guard.
                heads = ast.Module(body=[ast.Expr(value=i.context_expr) for i in node.items], type_ignores=[])
                text = ast.dump(heads)
                walk(node.body, shielded=shielded or ("suppress" in text and "ImportError" in text))

    for path in sorted(tree.rglob("*.py")):
        try:
            walk(ast.parse(path.read_text()).body, shielded=False)
        except SyntaxError as exc:  # a file that cannot be parsed is not a file with no imports
            raise SystemExit(f"CANNOT RUN: {path.relative_to(tree.parent)} does not parse: {exc}") from exc
    return unguarded, guarded - unguarded


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
    """(gated, advisory): unguarded third-party imports no manifest declares, and guarded ones.

    **The verdict must not depend on what happens to be installed here.**
    `packages_distributions()` only knows the current environment, so an earlier version routed any
    module it could not map into a list that never set the exit code -- and the modules it cannot
    map are exactly the undeclared, uninstalled ones the check exists to catch. Under CI's own
    install six more lost their mapping, so 9 of 23 third-party module-level imports were
    structurally invisible while the check printed green.

    The fallback closes it: an import name that maps to no installed distribution is checked under
    its own name, which is right whenever import and distribution names agree -- true for every
    dependency here except the `IMPORT_TO_DISTRIBUTION` entries, each asserted still needed by
    `--self-test`.
    """
    mapping = importlib.metadata.packages_distributions()
    unguarded, guarded = _imports_of(package)

    def check(names: set[str]) -> list[str]:
        out = []
        for module in sorted(names):
            if module in sys.stdlib_module_names or module == "mfgarchon" or module.startswith("_"):
                continue
            dists = mapping.get(module) or [IMPORT_TO_DISTRIBUTION.get(module, module)]
            if not any(_normalise(d) in declared for d in dists):
                shown = ", ".join(sorted(dists))
                out.append(module if shown == module else f"{module} (distribution: {shown})")
        return out

    return check(unguarded), check(guarded)


def _self_test() -> int:
    """Both directions on synthetic manifests, plus the exemptions, driven through the real checks."""
    failures: list[str] = []

    if _normalise("pytorch>=2.0") != "torch":
        failures.append("the conda/PyPI name map is not applied by _normalise")
    if _normalise("scikit-fem>=8.0") != "scikit-fem" or _normalise("PyYAML") != "pyyaml":
        failures.append("_normalise does not fold case or extras")

    # Each map entry must still be NEEDED, which is a claim about the manifests and the import
    # names -- not about what happens to be installed here. The previous version asked whether a
    # distribution literally named `pytorch` was installed, which is unrelated: it stayed green when
    # the exemption was made dead by renaming the conda entry, and when the entry was deleted
    # outright. Worse, installing the 1.0.2 PyPI placeholder -- the hazard the map exists for --
    # made it advise deleting the guard.
    conda_names = _conda(yaml.safe_load((ROOT / "environment.yml").read_text()))
    for conda_name, pypi_name in CONDA_TO_PYPI.items():
        if conda_name == pypi_name:
            failures.append(f"CONDA_TO_PYPI[{conda_name!r}] maps a name to itself")
        if _normalise(conda_name) not in conda_names:
            failures.append(
                f"CONDA_TO_PYPI[{conda_name!r}] is unused: environment.yml no longer names it, so the "
                "entry has outlived its reason and should be deleted"
            )

    # An IMPORT_TO_DISTRIBUTION entry is needed only when the import name would NOT resolve on its
    # own. The fallback in `_undeclared` uses the import name, so an entry whose import name is
    # already a declared distribution does nothing and will outlive its reason silently.
    full = _declared(tomllib.loads((ROOT / "pyproject.toml").read_text()))
    for import_name, dist in IMPORT_TO_DISTRIBUTION.items():
        if import_name == dist:
            failures.append(f"IMPORT_TO_DISTRIBUTION[{import_name!r}] maps a name to itself")
        if _normalise(import_name) in full:
            failures.append(
                f"IMPORT_TO_DISTRIBUTION[{import_name!r}] is dead: {import_name!r} is itself declared, "
                "so the fallback already resolves it"
            )

    declared = {"numpy", "torch"}
    if not _undeclared(PACKAGE, declared)[0]:
        failures.append("a nearly-empty declaration set must report undeclared imports, and did not")
    if _undeclared(PACKAGE, full)[0]:
        failures.append("the real pyproject must declare every third-party import, and does not")

    # The verdict must not move with the environment. Re-run the real comparison with the installed
    # mapping emptied -- that is CI, where the backends are absent -- and it must give the same
    # answer. Before the fallback existed, nine of twenty-three imports vanished into an advisory
    # list here and the check printed green.
    real_mapping = importlib.metadata.packages_distributions
    try:
        importlib.metadata.packages_distributions = dict
        blind = _undeclared(PACKAGE, full)[0]
    finally:
        importlib.metadata.packages_distributions = real_mapping
    if blind:
        failures.append(
            "with no distribution mapping available the check reports "
            f"{len(blind)} undeclared import(s) it does not report here: {blind[:4]}. "
            "The verdict depends on what is installed."
        )

    missing = sorted({"cvxpy", "rich"} - {"cvxpy"})
    if missing != ["rich"]:
        failures.append("set difference is not doing what the mirror check relies on")

    if failures:
        for line in failures:
            print(f"self-test FAILED: {line}", file=sys.stderr)
        return 1
    print(
        "self-test OK: both directions fire on synthetic manifests, the real tree is clean, "
        f"the verdict does not move with the environment, and all "
        f"{len(CONDA_TO_PYPI) + len(IMPORT_TO_DISTRIBUTION)} name-map entries are still needed"
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

    undeclared, optional = _undeclared(PACKAGE, declared)
    runtime = {_normalise(x) for x in pyproject["project"].get("dependencies", [])}
    absent = sorted(runtime - conda)

    if args.json:
        print(
            json.dumps(
                {"undeclared": undeclared, "guarded_and_undeclared": optional, "absent_from_conda": absent}, indent=2
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
    if optional:
        print(f"\nGUARDED AND UNDECLARED ({len(optional)}) -- reported, never gated:")
        for item in optional:
            print(f"    {item}")
        print("    Each is behind a module-level `try: import .. except ImportError`, so its absence")
        print("    cannot break an install. Declaring them in an extra would still be an improvement.")
    if rc == 0:
        print(
            "manifests agree: every third-party import is declared, and every runtime dependency is in environment.yml"
        )
    return rc


if __name__ == "__main__":
    sys.exit(main())
