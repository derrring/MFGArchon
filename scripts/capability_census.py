#!/usr/bin/env python
"""What each class DECLARES, and what the FP paths actually DO at a wall.

Two lanes, both answering a question the 2026-08-13 design census could not.

That census had four lanes -- claim-vs-actual, dead surface, own-geometry-vocabulary, import
structure -- and every one looks for reality falling **short** of a claim. It found 77 over-claims.
Nobody counted the other direction, and the other direction is what made #1975 wrong twice:

  - `FPFEMSolver` implements a general Robin wall and declares nothing, so a census keyed on
    `_SUPPORTED_BC_TYPES` reported the capability absent.
  - the default FDM scheme imposes `J.n = 0` structurally, by zeroing the total face flux, with no
    branch naming it -- so a sweep over branch names reported the wrong wall.

Over-claiming makes a USER's code fail loudly. Under-claiming makes a MAINTAINER delete or "fix"
something that works: the fix #1975 prescribed would have taken two correct FP paths from
-7.4e-15 to -79.5% mass drift.

## Lane 1 -- `declarations`

Population: every concrete class under a named root, discovered by `walk_packages` +
`issubclass`. **The predicate must not be the property audited**, so no declaration takes part in
choosing the population; each declaration is a column and "declares nothing" is a recorded row.

`getattr` is not enough: it finds a base-class default and reports it as a declaration. The MRO is
walked instead, and an inherited permissive default is reported separately -- it is a claim nobody
made deliberately. `honors_inhomogeneous_neumann` defaults to `True` on `BaseMFGSolver`.

## Lane 2 -- `conservation`

Population: every concrete class implementing `solve_fp_system`. Again not a declaration.

Oracle: mass conservation at a wall with wall-normal drift. `J.n = 0` is `m*v_n - D*d_n m = 0`;
with a normal drift it is NOT `d_n m = 0`, and a conserved quantity separates them --

    imposes J.n = 0    ->  mass conserved, `d_n m != 0` at the wall
    imposes d_n m = 0  ->  mass leaks at a rate proportional to `m_wall * v_n`

Two controls, because "conserves" is otherwise unfalsifiable:
  1. **zero drift** -- every path must conserve. One that leaks here is broken for an unrelated
     reason and its drifted number says nothing about its wall. This control fired on the first
     run: `FPSLJacobianSolver` gains mass at `O(h)` with no drift at all.
  2. **a known-good reference** -- `divergence_upwind` on the raw assembly. If the harness reports
     that leaking, the harness is wrong and every row is void.

## What neither lane can say

  - A capability announced by a method, a registry entry or `__getattr__` is invisible to lane 1.
  - Lane 2 measures each solver **at its default configuration**. Non-default advection schemes are
    reachable and differ: `gradient_upwind` loses 75% where `divergence_upwind` loses nothing.
  - A class outside every root's subclass tree is reached by nothing here; the known ones are
    listed in `OUTSIDE_EVERY_PREDICATE` and are **named, not discovered**, because a population
    predicate is itself a claim about scope.
  - `NOT MEASURED` is not a pass. It is a path whose wall nobody in this repository can currently
    observe -- the state that let #1975 be filed on a false premise.
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import pkgutil
import sys
from typing import Any

import numpy as np

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
    rows = []
    for name, cls in classes:
        labels = sorted(lbl for lbl, base in roots.items() if issubclass(cls, base))
        if not labels or inspect.isabstract(cls) or name.startswith(("Base", "_")):
            continue
        own, inherited = {}, {}
        for decl in DECLARATIONS:
            owner = next((k.__name__ for k in cls.__mro__ if decl in k.__dict__), None)
            if owner is None:
                continue
            target = own if owner == name else inherited
            target[decl] = {"value": str(getattr(cls, decl)), "from": owner}
        rows.append(
            {
                "name": name,
                "module": cls.__module__,
                "roles": labels,
                "own": sorted(own),
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
    }


# --------------------------------------------------------------------------------------
# Lane 2 -- conservation at a drifted wall
# --------------------------------------------------------------------------------------

NX, STEPS, DT, SIGMA, DRIFT = 81, 200, 1e-3, 0.3, 3.2
TOL = 1e-3  # percent; the conserving paths measure at 1e-4 % or better


def fp_solver_population() -> list[tuple[str, type]]:
    """Every concrete class implementing `solve_fp_system` -- the METHOD is the predicate."""
    import mfgarchon.alg as alg_pkg

    found: dict[str, type] = {}
    for mod in pkgutil.walk_packages(alg_pkg.__path__, prefix="mfgarchon.alg."):
        try:
            module = importlib.import_module(mod.name)
        except Exception:
            continue
        for name, cls in inspect.getmembers(module, inspect.isclass):
            if cls.__module__ != module.__name__ or inspect.isabstract(cls):
                continue
            if name.startswith(("Base", "_")):
                continue
            fn = getattr(cls, "solve_fp_system", None)
            if callable(fn) and not getattr(fn, "__isabstractmethod__", False):
                found[name] = cls
    return sorted(found.items())


def _problem(drift: float):
    from mfgarchon import Conditions, MFGProblem, Model
    from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
    from mfgarchon.geometry import TensorProductGrid
    from mfgarchon.geometry.boundary import no_flux_bc

    return MFGProblem(
        model=Model(
            hamiltonian=SeparableHamiltonian(
                control_cost=QuadraticControlCost(control_cost=1.0),
                coupling=lambda m: 0.0 * m,
                coupling_dm=lambda m: 0.0,
            ),
            sigma=SIGMA,
        ),
        domain=TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[NX], boundary_conditions=no_flux_bc(dimension=1)),
        conditions=Conditions(
            u_terminal=lambda x: -drift * x,
            m_initial=lambda x: np.exp(-50 * (x - 0.5) ** 2),
            T=STEPS * DT,
        ),
        Nt=STEPS,
    )


def _initial_density() -> tuple[np.ndarray, np.ndarray, float]:
    x = np.linspace(0.0, 1.0, NX)
    h = 1.0 / (NX - 1)
    m0 = np.exp(-50 * (x - 0.5) ** 2)
    return x, m0 / (m0.sum() * h), h


def reference_drift_pct() -> float:
    """Control 2: the raw assembly path, which #1975 measured at -0.0000%."""
    from mfgarchon.alg.numerical.fp_solvers.fp_fdm_time_stepping import solve_timestep_full_nd
    from mfgarchon.geometry import TensorProductGrid
    from mfgarchon.geometry.boundary import no_flux_bc

    grid = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[NX], boundary_conditions=no_flux_bc(dimension=1))
    x, m, h = _initial_density()
    start = m.sum() * h
    for _ in range(STEPS):
        m = solve_timestep_full_nd(
            M_current=m,
            U_current=-DRIFT * x,
            problem=object(),
            dt=DT,
            sigma=SIGMA,
            coupling_coefficient=1.0,
            spacing=(h,),
            grid=grid,
            ndim=1,
            shape=(NX,),
            boundary_conditions=no_flux_bc(dimension=1),
            advection_scheme="divergence_upwind",
        )
    return 100.0 * (m.sum() * h - start) / start


def _one_run(cls, drift: float) -> tuple[str, float | None, str]:
    try:
        problem = _problem(drift)
    except Exception as exc:
        return "harness_fail", None, f"{type(exc).__name__}: {exc}"
    try:
        solver = cls(problem)
    except Exception as exc:
        return "construct_fail", None, f"{type(exc).__name__}: {str(exc).splitlines()[0][:140]}"

    x, m0, h = _initial_density()
    u = np.tile(-drift * x, (STEPS + 1, 1))
    try:
        m = solver.solve_fp_system(m0, u)
    except TypeError:
        try:
            m = solver.solve_fp_system(m0, potential_field=u)
        except Exception as exc:
            return "solve_fail", None, f"{type(exc).__name__}: {str(exc).splitlines()[0][:140]}"
    except Exception as exc:
        return "solve_fail", None, f"{type(exc).__name__}: {str(exc).splitlines()[0][:140]}"

    m = np.asarray(m)
    if m.ndim != 2 or not np.all(np.isfinite(m)):
        return "solve_fail", None, f"shape {m.shape}, finite={bool(np.all(np.isfinite(m)))}"
    start, end = m[0].sum() * h, m[-1].sum() * h
    if start <= 0:
        return "solve_fail", None, f"initial mass {start:.3e}"
    return "ok", 100.0 * (end - start) / start, ""


def conservation_verdicts() -> dict[str, Any]:
    rows = []
    for name, cls in fp_solver_population():
        v0, d0, e0 = _one_run(cls, 0.0)
        v1, d1, e1 = _one_run(cls, DRIFT)
        if v0 != "ok" or v1 != "ok":
            verdict = "NOT_MEASURED"
            detail = e0 or e1
        elif abs(d0) > TOL:
            verdict = "CONTROL_FAILED"
            detail = f"leaks {d0:+.4f}% with no drift; the drifted number says nothing about its wall"
        elif abs(d1) < TOL:
            verdict = "CONSERVES"
            detail = "imposes J.n = 0"
        else:
            verdict = "LEAKS"
            detail = f"{d1:+.4f}% at a drifted wall -- imposes d_n m = 0, not J.n = 0"
        rows.append({"class": name, "no_drift_pct": d0, "drift_pct": d1, "verdict": verdict, "detail": detail})
    return {"reference_drift_pct": reference_drift_pct(), "rows": rows}


# --------------------------------------------------------------------------------------


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

    print("\n=== claimed only by inheriting a permissive default ===")
    for decl in DECLARATIONS:
        inheritors = [r["name"] for r in rows if decl in r["inherited"]]
        owners = [r["name"] for r in rows if decl in r["own"]]
        if inheritors or owners:
            print(f"  {decl}: own {len(owners)}, INHERITED {len(inheritors)}")
    print("  An inherited permissive default is a claim nobody made deliberately.")

    print("\n=== outside every population predicate (named, not discovered) ===")
    for name, reached in sorted(result["outside_every_predicate"].items()):
        print(f"  {name:22s} reached by: {reached or 'NOTHING'}")


def _print_conservation(result: dict[str, Any]) -> int:
    ref = result["reference_drift_pct"]
    ok = abs(ref) < TOL
    print(f"=== control 2: reference path drift {ref:+.4f}%  {'HARNESS OK' if ok else 'HARNESS SUSPECT'} ===")
    if not ok:
        print("  the reference is supposed to conserve exactly; every row below is void")

    print(f"\n{'class':32s} {'no drift':>12s} {'drift':>12s}   verdict")
    for r in result["rows"]:
        cell = lambda v: f"{v:+11.4f}%" if v is not None else f"{'--':>12s}"  # noqa: E731
        print(
            f"{r['class']:32s} {cell(r['no_drift_pct']):>12s} {cell(r['drift_pct']):>12s}   {r['verdict']}: {r['detail']}"
        )

    counts: dict[str, int] = {}
    for r in result["rows"]:
        counts[r["verdict"]] = counts.get(r["verdict"], 0) + 1
    print("\n=== " + ", ".join(f"{v} {k}" for k, v in sorted(counts.items())) + " ===")
    print("NOT_MEASURED is not a pass. It is a path whose wall nobody here can currently observe.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("lane", choices=("declarations", "conservation", "both"), nargs="?", default="both")
    ap.add_argument("--json", metavar="FILE")
    args = ap.parse_args()

    import mfgarchon

    print(f"module root: {mfgarchon.__file__}\n")
    out: dict[str, Any] = {}
    if args.lane in ("declarations", "both"):
        out["declarations"] = declaration_matrix()
        _print_declarations(out["declarations"])
    if args.lane in ("conservation", "both"):
        if out:
            print()
        out["conservation"] = conservation_verdicts()
        _print_conservation(out["conservation"])

    if args.json:
        with open(args.json, "w") as fh:
            json.dump(out, fh, indent=2)
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
