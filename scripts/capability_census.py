"""What each class DECLARES about the boundary conditions it accepts.

The 2026-08-13 design census had four lanes -- claim-vs-actual, dead surface,
own-geometry-vocabulary, import structure -- and every one looks for reality falling **short** of
a claim. It found 77 over-claims. Nobody counted the other direction, and the other direction is
what made #1975 wrong twice: `FPFEMSolver` implements a general Robin wall and declares nothing,
so a census keyed on `_SUPPORTED_BC_TYPES` reported the capability absent.

Over-claiming makes a USER's code fail loudly. Under-claiming makes a MAINTAINER delete or "fix"
something that works.

Population: every concrete class under a named root, discovered by `walk_packages` +
`issubclass`. **The predicate must not be the property audited**, so no declaration takes part in
choosing the population; each declaration is a column and "declares nothing" is a recorded row.

`getattr` is not enough: it finds a base-class default and reports it as a declaration. The MRO is
walked instead, and an inherited permissive default is reported separately -- it is a claim nobody
made deliberately. `honors_inhomogeneous_neumann` defaults to `True` on `BaseMFGSolver`.

`cls.__module__ != module.__name__` is deliberate: it collapses cross-module re-exports, so a
class re-exported from a package `__init__` is one row rather than two.

## What this cannot say

  - A capability announced by a method, a registry entry or `__getattr__` is invisible here.
  - A class outside every root's subclass tree is reached by nothing; the known ones are listed in
    `OUTSIDE_EVERY_PREDICATE` and are **named, not discovered**, because a population predicate is
    itself a claim about scope. That list is incomplete.
  - "Declares nothing" is not "has no capability" -- it is the case where capability cannot be
    read off the class at all, which is the blind spot #1975 fell into.

A second lane, measuring which wall each FP path imposes, was removed. Its findings are recorded
in #1975. The measurement needed a discrimination rule for the LIMIT behaviour of a numerical
scheme, four attempts failed to state one correctly, and 41% of a 32-mutation sweep survived the
ratchet built over it -- including the pins for three of the defects it claimed to have fixed. A
ratchet over an unstated rule pins nothing.
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import pkgutil
import sys
from typing import Any

# --------------------------------------------------------------------------------------
# Lane 1 -- declarations
# --------------------------------------------------------------------------------------

#: Class-level attributes this repository uses to announce a capability or a constraint.
DECLARATIONS = (
    "_SUPPORTED_BC_TYPES",
    "honors_inhomogeneous_neumann",
    "discretization_type",
)

#: Roots whose concrete subclasses form each population.
ROOTS = {
    "solver": ("mfgarchon.alg.base_solver", "BaseMFGSolver"),
    "applicator": ("mfgarchon.geometry.boundary.protocols", "BaseBCApplicator"),
    "backend": ("mfgarchon.backends.base_backend", "BaseBackend"),
    "geometry": ("mfgarchon.geometry.base", "Geometry"),
}

#: Classes doing a root's job without subclassing it, so no predicate above reaches them.
#: Named rather than discovered -- that is the honest form of a scope claim.
OUTSIDE_EVERY_PREDICATE = {
    "ParticleApplicator": "mfgarchon.geometry.boundary.applicator_particle",
    "HJBHowardSolver": "mfgarchon.alg.numerical.hjb_solvers.hjb_howard",
    "ImplicitHeatSolver": "mfgarchon.alg.numerical.pde_solvers.implicit_heat",
}


def _walk(package: str) -> tuple[list[tuple[str, type]], list[tuple[str, str]]]:
    pkg = importlib.import_module(package)
    classes: list[tuple[str, type]] = []
    failures: list[tuple[str, str]] = []
    for mod in pkgutil.walk_packages(pkg.__path__, prefix=f"{package}."):
        try:
            module = importlib.import_module(mod.name)
        except Exception as exc:
            failures.append((mod.name, f"{type(exc).__name__}: {exc}"))
            continue
        classes.extend(
            (name, cls)
            for name, cls in inspect.getmembers(module, inspect.isclass)
            if cls.__module__ == module.__name__
        )
    return classes, failures


def declaration_matrix(package: str = "mfgarchon") -> dict[str, Any]:
    """One row per concrete class under a root, with own vs inherited declarations separated."""
    roots, missing = {}, []
    for label, (module_path, name) in ROOTS.items():
        try:
            roots[label] = getattr(importlib.import_module(module_path), name)
        except Exception as exc:
            missing.append((label, f"{module_path}.{name}", f"{type(exc).__name__}: {exc}"))

    classes, import_failures = _walk(package)
    rows: list[dict] = []
    excluded_by_name: list[str] = []
    for name, cls in classes:
        labels = sorted(lbl for lbl, base in roots.items() if issubclass(cls, base))
        if not labels:
            continue
        if inspect.isabstract(cls):
            continue
        if name.startswith(("Base", "_")):
            # A NAME deciding the population, in a script whose other lane was deleted for
            # exactly that. Kept because these are intended-abstract bases, but recorded:
            # `inspect.isabstract` is False for them (empty `__abstractmethods__`), and all four
            # OWN `discretization_type` (protocols.py:338/625/653/682); three appear as the `<-`
            # owner below, the fourth having no surviving subclass in the population.
            excluded_by_name.append(name)
            continue
        # Same class bound twice in its own module (`NetworkFPSolver = FPNetworkSolver`,
        # fp_network.py:606) yields two rows unless collapsed. The removed conservation lane
        # carried that knowledge; lane 1 inherited the bug when it went.
        if any(r["cls_id"] == id(cls) for r in rows):
            next(r for r in rows if r["cls_id"] == id(cls))["names"].append(name)
            continue
        own, inherited = {}, {}
        for decl in DECLARATIONS:
            owner = next((k.__name__ for k in cls.__mro__ if decl in k.__dict__), None)
            if owner is None:
                continue
            # `owner == cls.__name__`, not `owner == name`: `name` is whichever binding
            # `inspect.getmembers` yielded first (alphabetical), so an alias sorting earlier
            # would report the class as inheriting its own declaration.
            target = own if owner == cls.__name__ else inherited
            target[decl] = {"value": str(getattr(cls, decl)), "from": owner}
        rows.append(
            {
                "name": name,
                "names": [name],
                "cls_id": id(cls),
                "module": cls.__module__,
                "roles": labels,
                "own": sorted(own),
                "own_values": {k: v["value"] for k, v in own.items()},
                "inherited": inherited,
                "declares_nothing": not own,
            }
        )

    rows.sort(key=lambda r: (r["roles"], len(r["own"]), r["name"]))

    outside = {}
    for name, path in OUTSIDE_EVERY_PREDICATE.items():
        try:
            cls = getattr(importlib.import_module(path), name)
            outside[name] = sorted(lbl for lbl, base in roots.items() if issubclass(cls, base))
        except Exception as exc:
            outside[name] = [f"UNAVAILABLE: {type(exc).__name__}"]

    return {
        "rows": rows,
        "roots_missing": missing,
        "import_failures": import_failures,
        "outside_every_predicate": outside,
        "excluded_by_name_prefix": sorted(set(excluded_by_name)),
    }


def _print_declarations(result: dict[str, Any]) -> None:
    for label, path, err in result["roots_missing"]:
        print(f"  ROOT UNAVAILABLE  {label}: {path} -> {err}")
    if result["import_failures"]:
        print(f"\n=== modules that would not import ({len(result['import_failures'])}) ===")
        for name, err in result["import_failures"]:
            print(f"  {name}: {err}")

    rows = result["rows"]
    print(f"\n=== population: {len(rows)} concrete classes ===")
    print(f"{'role':11s} {'class':32s} {'own':>3s}  own declarations / [inherited]")
    for r in rows:
        own = ", ".join(r["own"]) or "-- DECLARES NOTHING --"
        inh = ", ".join(f"{k}={v['value']}<-{v['from']}" for k, v in r["inherited"].items())
        print(f"{','.join(r['roles']):11s} {r['name']:32s} {len(r['own']):3d}  {own}" + (f"   [{inh}]" if inh else ""))

    print("\n=== declares nothing of its own ===")
    for role in sorted({x for r in rows for x in r["roles"]}):
        members = [r for r in rows if role in r["roles"]]
        silent = [r for r in members if r["declares_nothing"]]
        print(f"  {role:11s} {len(silent):3d} of {len(members):3d}  ({100 * len(silent) / len(members):.0f}%)")
    print("  Not evidence of no capability -- it is the case where capability cannot be read off")
    print("  the class at all, which is the blind spot #1975 fell into.")

    print("\n=== declarations that arrive by inheritance, grouped by what is inherited ===")
    for decl in DECLARATIONS:
        inheritors = [r for r in rows if decl in r["inherited"]]
        owners = [r["name"] for r in rows if decl in r["own"]]
        if not (inheritors or owners):
            continue
        print(f"  {decl}: own {len(owners)}, inherited {len(inheritors)}")
        groups: dict[tuple[str, str], list[str]] = {}
        for r in inheritors:
            e = r["inherited"][decl]
            groups.setdefault((str(e["from"]), str(e["value"])), []).append(r["names"][0])
        for (src, val), who in sorted(groups.items(), key=lambda kv: -len(kv[1])):
            shown = val if not val.startswith("<property object") else "<property>"
            print(f"      {len(who):3d}  <- {src:26s} = {shown[:46]}")
    print("  A count alone cannot say whether an inherited declaration is permissive. Grouping")
    print("  can: an inherited `True` on honors_inhomogeneous_neumann is a claim nobody made")
    print("  deliberately; an inherited `False`, or a restrictive frozenset, or a `property`")
    print("  object, is not -- and a bare count previously reported all four as the same thing.")

    if result["excluded_by_name_prefix"]:
        print(f"\n=== concrete, under a root, dropped by a NAME prefix: {len(result['excluded_by_name_prefix'])} ===")
        for nm in result["excluded_by_name_prefix"]:
            print(f"  {nm}")

    print("\n=== outside every population predicate (named, not discovered) ===")
    for name, reached in sorted(result["outside_every_predicate"].items()):
        print(f"  {name:22s} reached by: {reached or 'NOTHING'}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--package", default="mfgarchon")
    ap.add_argument("--json", metavar="FILE")
    args = ap.parse_args()

    import mfgarchon

    print(f"module root: {mfgarchon.__file__}\n")
    result = declaration_matrix(args.package)
    _print_declarations(result)

    if args.json:
        with open(args.json, "w") as fh:
            json.dump(result, fh, indent=2)
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
