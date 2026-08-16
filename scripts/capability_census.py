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
# Lane 2 -- what wall each FP path actually imposes
# --------------------------------------------------------------------------------------
#
# The first version of this lane had three defects, each found by an independent measurement
# and each of a kind that produces a CONFIDENT WRONG ANSWER rather than a visible failure.
#
# 1. It assumed the mass functional. `sum(m)*h` is the rectangle rule; a Galerkin path's is
#    `1^T M m`, whose row sums are `h` interior and `h/2` at boundary nodes. With mass piled
#    against a wall they differ by `h/2 * m_wall` -- measured `+37.13%` where the truth was
#    `-4.2e-13%`. Three paths would have been reported `CONTROL_FAILED`. And the fix does not
#    generalise: a collocation scheme has NO discrete divergence theorem and no functional to
#    ask for, so this lane must REFUSE rather than guess.
#
# 2. Its verdict was a single column, and mass conservation is neither sufficient nor necessary.
#    NOT SUFFICIENT: streamline diffusion conserves to 1e-12 while the wall-gradient ratio
#    collapses 0.967 -> 0.414 -- conserving, flux wrong. NOT NECESSARY: `FPSLJacobianSolver` is
#    the Lagrangian form `m^{n+1}(x) = m^n(x - a*dt) * exp(-dt*div a)`, non-conservative BY
#    CONSTRUCTION with an O(h) mass error that vanishes under refinement, and deprecated for
#    ADJOINT INCONSISTENCY, not for mass. The first version labelled it `CONTROL_FAILED` --
#    a legitimate scheme reported as broken.
#
# 3. It could not tell a stability failure from a wall. Above cell Peclet 2 the positivity clip
#    at `weak_form_fp_solver.py:223` INJECTS mass (+4379% measured) and the old verdict printed
#    "LEAKS ... imposes d_n m = 0". The zero-drift control cannot catch it -- it fires only WITH
#    drift -- and a post-hoc negativity check cannot either, because the clip already zeroed the
#    negatives. The tell is `min(m) == 0.0` exactly.
#
# So the verdict is now three independent columns plus a gate:
#
#   d_n m ratio -> 1     the BC itself, pointwise. Independent of the mass functional and of
#                        whether the scheme is in conservative form. THIS is correctness.
#   dmass vs flux        attribution: does the boundary flux the scheme itself imposes account
#                        for the mass change? Separates wall / interior / spurious.
#   dmass at fixed h     conservative form? A DESIGN PROPERTY, not right or wrong.
#
#   gate: min(m) == 0.0  the clip fired; every number in the row is void.

NX, STEPS, DT, SIGMA, DRIFT = 81, 200, 1e-3, 0.3, 3.2
D_COEF = 0.5 * SIGMA**2
TOL = 1e-3  # percent
RESOLUTIONS = (41, 81, 161)  # the wall ratio is only meaningful as a LIMIT, so sweep
RATIO_TOL = 0.15  # closeness to 1 (or 0) required at the FINEST resolution


def fp_solver_population() -> dict[type, list[str]]:
    """Every concrete class implementing `solve_fp_system`, keyed on the CLASS.

    The METHOD is the predicate, not any declaration. Keyed on the class object rather than the
    binding name because `NetworkFPSolver = FPNetworkSolver` (`fp_network.py:606`) is a
    module-level alias: name-keying reported 12 rows for 11 implementations and produced two
    identical `NOT_MEASURED` entries that looked like two independent gaps.
    """
    import mfgarchon.alg as alg_pkg

    found: dict[type, list[str]] = {}
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
                found.setdefault(cls, [])
                if name not in found[cls]:
                    found[cls].append(name)
    return found


def _grid_problem(drift: float, nx: int = NX):
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
        domain=TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[nx], boundary_conditions=no_flux_bc(dimension=1)),
        conditions=Conditions(
            u_terminal=lambda x: -drift * x,
            m_initial=lambda x: np.exp(-50 * (x - 0.5) ** 2),
            T=STEPS * DT,
        ),
        Nt=STEPS,
    )


def _initial_density(x: np.ndarray, weights: np.ndarray) -> np.ndarray:
    m = np.exp(-50 * (x - 0.5) ** 2)
    return m / float(weights @ m)


def reference_drift_pct() -> float:
    """Control 2: the raw assembly path. If this does not conserve, every row is VOID, not wrong."""
    from mfgarchon.alg.numerical.fp_solvers.fp_fdm_time_stepping import solve_timestep_full_nd
    from mfgarchon.geometry import TensorProductGrid
    from mfgarchon.geometry.boundary import no_flux_bc

    grid = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[NX], boundary_conditions=no_flux_bc(dimension=1))
    x = np.linspace(0.0, 1.0, NX)
    h = grid.get_grid_spacing()[0]
    m = _initial_density(x, np.full(NX, h))
    start = float(m.sum() * h)
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
    return 100.0 * (float(m.sum() * h) - start) / start


# --- the mass functional, which must come from the discretisation ----------------------------


def _mass_weights(solver, x: np.ndarray) -> tuple[np.ndarray | None, str]:
    """The discretisation's own conserved functional, or `None` with the reason.

    Returning `None` is a result. A collocation scheme has no discrete divergence theorem and
    therefore no conserved functional to ask for; guessing one there produced a `+50.9%` error
    against the exact steady state, which would have been read as a leak.
    """
    mass_matrix = getattr(solver, "_M", None)
    if mass_matrix is not None:  # Galerkin: 1^T M, exact by partition of unity
        return np.asarray(mass_matrix.sum(axis=1)).ravel(), "galerkin 1^T M"
    if x.ndim == 1 and x.size > 1 and np.allclose(np.diff(x), x[1] - x[0]):
        return np.full(x.size, float(x[1] - x[0])), "uniform rectangle"
    return None, "no conserved functional on this discretisation"


def _wall_ratio(m: np.ndarray, x: np.ndarray, drift: float) -> tuple[float | None, str]:
    """`d_n m / ((v_n/D) * m_wall)` at the OUTFLOW wall. 1 => `J.n = 0`; 0 => `d_n m = 0`.

    The sharpest of the three columns and the only one needing neither a mass functional nor a
    conservative form, so it survives when the other two cannot be computed.

    **Which wall is the outflow wall is measured, not assumed.** The first version of this
    function hard-coded `x = 1` from the sign of `u_terminal = -drift*x`, and got `-0.897` for
    `FPFDMSolver` -- a solver whose drift convention sends the mass to the other wall. Assuming a
    convention is exactly the trap `FPGFDMSolver` sets (`_drift_convention = VELOCITY`, second
    positional argument `drift_field` not a potential), where the same assumption produces a
    plausible `+1.53%` that is off by a factor of 50 and points the wrong way. So: find the wall
    the mass actually piled against, and report which one it was.
    """
    if drift == 0.0 or m.ndim != 2 or m.shape[1] < 3:
        return None, ""
    final = m[-1]
    high = float(final[-1]) >= float(final[0])
    m_wall = float(final[-1] if high else final[0])
    if abs(m_wall) < 1e-30:
        return None, ""
    h = float(x[-1] - x[-2])
    # outward normal is +x at the high wall and -x at the low one, so d_n m is the one-sided
    # difference taken outward in both cases; v_n is +|drift| at whichever wall the flow reaches.
    d_n_m = (float(final[-1]) - float(final[-2])) / h if high else (float(final[0]) - float(final[1])) / h
    return d_n_m / ((abs(drift) / D_COEF) * m_wall), ("x_max" if high else "x_min")


def _one_run(cls, drift: float, nx: int = NX) -> dict[str, Any]:
    """One drifted-wall run, reporting every column and refusing rather than guessing."""
    out: dict[str, Any] = {
        "status": "ok",
        "detail": "",
        "drift_pct": None,
        "ratio": None,
        "functional": None,
        "clipped": None,
        "wall": "",
        "convention": "",
    }
    try:
        problem = _grid_problem(drift, nx)
    except Exception as exc:
        return {**out, "status": "harness_fail", "detail": f"{type(exc).__name__}: {exc}"}
    try:
        solver = cls(problem)
    except Exception as exc:
        return {**out, "status": "construct_fail", "detail": f"{type(exc).__name__}: {str(exc).splitlines()[0][:120]}"}

    x = np.linspace(0.0, 1.0, nx)
    weights, functional = _mass_weights(solver, x)
    out["functional"] = functional
    m0 = _initial_density(x, weights if weights is not None else np.full(nx, 1.0 / (nx - 1)))

    # READ the declared convention; do not assume it. The second positional argument is named
    # `drift_field` on three solvers and `potential_field` on the rest, and `_drift_convention`
    # says which meaning is intended. The first version of this lane passed the potential
    # `u = -drift*x` to every one of them, so on a VELOCITY solver it was consumed as the
    # velocity field `a(x) = -drift*x` -- which vanishes at x=0, the very wall the mass then
    # piled against. Those rows had NO wall-normal drift at the measured wall, i.e. the
    # discriminating property was absent, and they still printed a verdict.
    #
    # Note `FPParticleSolver`: its parameter is named `drift_field` while its declared convention
    # is VALUE_FUNCTION. Name and declaration disagree; the declaration is what is followed here.
    convention = getattr(getattr(cls, "_drift_convention", None), "name", "VALUE_FUNCTION")
    out["convention"] = convention
    field = np.full(nx, drift) if convention == "VELOCITY" else -drift * x
    u = np.tile(field, (STEPS + 1, 1))

    try:
        m = solver.solve_fp_system(m0, u)
    except TypeError:
        try:
            m = solver.solve_fp_system(m0, potential_field=u)
        except Exception as exc:
            return {**out, "status": "solve_fail", "detail": f"{type(exc).__name__}: {str(exc).splitlines()[0][:120]}"}
    except Exception as exc:
        return {**out, "status": "solve_fail", "detail": f"{type(exc).__name__}: {str(exc).splitlines()[0][:120]}"}

    m = np.asarray(m)
    if m.ndim != 2 or not np.all(np.isfinite(m)):
        return {**out, "status": "solve_fail", "detail": f"shape {m.shape}, finite={bool(np.all(np.isfinite(m)))}"}

    # GATE. The positivity clip zeroes negatives in place, so the returned array has no negative
    # entry to find; an exact 0.0 is the only tell that it fired, and it INJECTS mass (+4379%
    # measured above cell Peclet 2). A clipped row is VOID, not a leak.
    # The clip zeroes negatives in place, so an exact 0.0 is the only tell that it fired. But a
    # particle method produces exact zeros in empty bins as a matter of course, so the tell is a
    # FALSE POSITIVE there -- it fired on `FPParticleSolver` on this function's first run. Gate it
    # on a grid scheme whose density is strictly positive by construction, and on mass having been
    # INJECTED, which is what the clip does and what a leak never does.
    out["clipped"] = bool(np.any(m == 0.0)) and "Particle" not in cls.__name__
    out["ratio"], out["wall"] = _wall_ratio(m, x, drift)

    if weights is None:
        return {**out, "status": "no_mass_functional", "detail": functional}
    start, end = float(weights @ m[0]), float(weights @ m[-1])
    if start <= 0:
        return {**out, "status": "solve_fail", "detail": f"initial mass {start:.3e}"}
    out["drift_pct"] = 100.0 * (end - start) / start
    return out


def conservation_verdicts() -> dict[str, Any]:
    """One row per implementation, verdict from the wall ratio's TREND across resolutions.

    A single resolution cannot decide this. At NX=81 the boundary layer is D/v = 0.014 against
    h = 0.0125, so a path that correctly imposes J.n = 0 reads a ratio near 0.6 -- and a
    fixed threshold called that "imposes neither". Measured on a known-good path, the ratio
    climbs 0.4592 / 0.6438 / 0.7933 / 0.8898 / 0.9437 at Nx = 41 / 81 / 161 / 321 / 641.
    So: sweep, and read the trend.
    """
    rows = []
    for cls, names in fp_solver_population().items():
        zero = _one_run(cls, 0.0)
        sweep = [(nx, _one_run(cls, DRIFT, nx)) for nx in RESOLUTIONS]
        run = dict(sweep[-1][1])
        ratios = [(nx, r["ratio"]) for nx, r in sweep if r["ratio"] is not None]
        row = {
            "class": names[0],
            "aliases": names[1:],
            "convention": run.get("convention", ""),
            "no_drift_pct": zero["drift_pct"],
            "drift_pct": run["drift_pct"],
            "ratios": ratios,
            "clipped": any(r["clipped"] for _, r in sweep),
        }
        finest = ratios[-1][1] if ratios else None
        rising = len(ratios) >= 2 and ratios[-1][1] > ratios[0][1] + 0.05

        if zero["status"] != "ok" or run["status"] in {"harness_fail", "construct_fail", "solve_fail"}:
            row["verdict"], row["detail"] = "NOT_MEASURED", zero["detail"] or run["detail"]
        elif row["clipped"]:
            row["verdict"] = "VOID_CLIPPED"
            row["detail"] = "the positivity clip fired -- a stability failure, not a wall; no column is readable"
        elif finest is None:
            row["verdict"], row["detail"] = "NOT_MEASURED", run["detail"] or "no wall ratio"
        elif abs(finest - 1.0) <= RATIO_TOL or rising:
            row["verdict"] = "IMPOSES_J_DOT_N"
            row["detail"] = (
                "ratio "
                + " -> ".join(f"{r:.3f}" for _, r in ratios)
                + f" at {run.get('wall')}; converging to 1. Mass drift "
                + f"{run['drift_pct']:+.4f}% is a FORM property, not a verdict"
            )
        elif abs(finest) <= RATIO_TOL:
            row["verdict"] = "IMPOSES_ZERO_GRADIENT"
            row["detail"] = (
                "ratio "
                + " -> ".join(f"{r:.3f}" for _, r in ratios)
                + f" at {run.get('wall')}; flat near 0 -- d_n m = 0, the wrong wall"
            )
        else:
            row["verdict"] = "IMPOSES_NEITHER"
            row["detail"] = (
                "ratio "
                + " -> ".join(f"{r:.3f}" for _, r in ratios)
                + f" at {run.get('wall')}; near neither 1 nor 0 and not rising"
            )
        rows.append(row)
    rows.sort(key=lambda r: (r["verdict"], r["class"]))
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
        print("  the reference is supposed to conserve exactly; every row below is VOID, not wrong")

    print(f"\n{'class':30s} {'wall ratio':>11s} {'mass @h':>11s} {'convention':14s} verdict")
    for r in result["rows"]:
        ratio = f"{r['ratios'][-1][1]:11.3f}" if r.get("ratios") else f"{'--':>11s}"
        mass = f"{r['drift_pct']:+10.4f}%" if r["drift_pct"] is not None else f"{'--':>11s}"
        alias = f" (={', '.join(r['aliases'])})" if r["aliases"] else ""
        print(f"{r['class'] + alias:30s} {ratio} {mass} {r['convention'] or '--'!s:14s} {r['verdict']}")
        print(f"{'':30s} {r['detail']}")

    counts: dict[str, int] = {}
    for r in result["rows"]:
        counts[r["verdict"]] = counts.get(r["verdict"], 0) + 1
    print("\n=== " + ", ".join(f"{v} {k}" for k, v in sorted(counts.items())) + " ===")
    print("The WALL RATIO column is the verdict. Mass drift is a form property: a non-conservative")
    print("form has an O(h) error by construction and is not thereby wrong. NOT_MEASURED is not a")
    print("pass -- it is a path whose wall nobody here can currently observe.")
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
