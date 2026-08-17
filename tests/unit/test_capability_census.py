"""Ratchet over `scripts/capability_census.py`. Refs #1975, #1977.

Two lanes, both answering a question the 2026-08-13 design census could not: all four of its lanes
look for reality falling *short* of a claim, and it found 77 over-claims. Nobody counted the other
direction — capability that exists undeclared, and a wall imposed with nothing naming it — and
that direction made #1975 wrong twice.

**Lane 2's instrument was rebuilt after eight defects, and every one produced a confident verdict
rather than a failure.** Four were found by independent measurement of the paths it could not
construct; four more only by re-running it against a path whose answer was already known:

1. it assumed the mass functional (`sum(m)*h` vs a Galerkin `1^T M`, `+37%` vs `-4e-13%`);
2. its verdict was sign-blind (a `d_n m = 0` wall *loses* mass; the clip *gains* it);
3. it could not tell a stability failure from a wall;
4. it double-counted a module-level alias, reporting 12 rows for 11 implementations;
5. it assumed which wall was the outflow wall;
6. its clip gate false-positived on a particle method, where exact zeros are ordinary;
7. **it assumed the drift convention** — `_drift_convention` is a declared class attribute reading
   `VELOCITY` on three solvers, and passing them a potential made the wall-normal drift vanish at
   the very wall the mass reached, so the discriminating property was absent and a verdict printed
   anyway;
8. it thresholded a quantity that is only meaningful as a limit.

Everything here is imported from the script rather than restated, so the two cannot drift.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "capability_census.py"


@pytest.fixture(scope="module")
def census():
    spec = importlib.util.spec_from_file_location("capability_census", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["capability_census"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def declarations(census):
    return census.declaration_matrix()


@pytest.fixture(scope="module")
def conservation(census):
    return census.conservation_report()


# =============================================================================
# Lane 1 -- who declares nothing
# =============================================================================

_DECLARES_NOTHING = {
    "solver": {
        "FPFEMSolver",
        "FPNetworkSolver",
        "FPSLAdjointSolver",
        "HJBFEMSolver",
        "MeshlessGalerkinFPSolver",
        "MeshlessGalerkinHJBSolver",
        "NetworkFPSolver",
        "NetworkHJBSolver",
        "NetworkPolicyIterationHJBSolver",
        "PenaltyHJBSolver",
        "PrimalDualMFGSolver",
        "SinkhornMFGSolver",
        "VariationalMFGSolver",
        "WassersteinMFGSolver",
        "WeakFormFPSolver",
        "WeakFormHJBSolver",
    },
    "applicator": {"FDMApplicator", "GraphApplicator", "ImplicitApplicator", "MeshfreeApplicator"},
    "backend": {"JAXBackend", "NumPyBackend", "NumbaBackend", "TorchBackend"},
    "geometry": {
        "ComplementDomain",
        "CustomNetwork",
        "DifferenceDomain",
        "GridNetwork",
        "Hyperrectangle",
        "Hypersphere",
        "IntersectionDomain",
        "MazeGeometry",
        "Mesh1D",
        "Mesh2D",
        "Mesh3D",
        "RandomNetwork",
        "ScaleFreeNetwork",
        "TensorProductGrid",
        "UnionDomain",
    },
}


@pytest.mark.parametrize("role", sorted(_DECLARES_NOTHING))
def test_the_set_of_classes_declaring_nothing_is_unchanged(declarations, role):
    """Leaving this set is #1977 progress; joining it is a capability shipped undeclared."""
    got = {r["name"] for r in declarations["rows"] if role in r["roles"] and r["declares_nothing"]}
    want = _DECLARES_NOTHING[role]
    if got == want:
        return
    raise AssertionError(
        f"{role}: classes declaring nothing changed.\n"
        f"  newly silent ({sorted(got - want)}) -- a capability shipped without a declaration.\n"
        f"  now declaring ({sorted(want - got)}) -- #1977 progress; record it here."
    )


def test_no_module_in_the_package_fails_to_import(declarations):
    """The population is only as complete as the walk, and `walk_packages` yields a name whether
    or not the module imports."""
    assert declarations["import_failures"] == [], (
        f"modules that would not import: {declarations['import_failures']}. Every class they "
        "define is missing from every row of this census."
    )


def test_the_roots_still_exist(declarations):
    assert declarations["roots_missing"] == [], f"unresolvable roots: {declarations['roots_missing']}"


def test_the_classes_outside_every_predicate_are_still_outside_it(declarations):
    """Named rather than discovered — a population predicate is itself a claim about scope, and
    this is the part no mechanism recovers."""
    assert declarations["outside_every_predicate"] == {
        "HJBHowardSolver": [],
        "ImplicitHeatSolver": [],
        "ParticleApplicator": [],
    }


def test_the_permissive_default_is_still_claimed_by_inheritance(declarations):
    """`honors_inhomogeneous_neumann` defaults to True on `BaseMFGSolver`, so a solver that never
    mentions it claims to honour an inhomogeneous Neumann flux — a claim made by a default, not by
    anyone, and invisible at every call site.

    Two inheritance chains meaning opposite things: from `BaseMFGSolver` it is that default; from
    a sibling (`FPSLAdjointSolver` <- `FPSLSolver`) it is a deliberate `False`.
    """
    rows = declarations["rows"]
    field = "honors_inhomogeneous_neumann"
    owners = [r["name"] for r in rows if field in r["own"]]
    from_base = {
        r["name"]: r["inherited"][field]["value"]
        for r in rows
        if r["inherited"].get(field, {}).get("from") == "BaseMFGSolver"
    }
    from_sibling = {
        r["name"]: (r["inherited"][field]["from"], r["inherited"][field]["value"])
        for r in rows
        if field in r["inherited"] and r["inherited"][field]["from"] != "BaseMFGSolver"
    }

    assert len(owners) == 6, f"solvers stating it themselves: {sorted(owners)}"
    assert len(from_base) == 19, f"solvers claiming True by the permissive default: {sorted(from_base)}"
    assert set(from_base.values()) == {"True"}
    assert from_sibling == {"FPSLAdjointSolver": ("FPSLSolver", "False")}


# =============================================================================
# Lane 2 -- the wall-ratio sequence, reported and pinned; no verdict
# =============================================================================

#: Rows the harness can construct and run. The ratio sequence is pinned loosely -- it is a
#: measurement of a numerical scheme, not a constant -- and the ORDER of magnitude and sign are
#: what a regression would move.
_MEASURED = {
    "FPFDMSolver": (0.35, 0.69),
    "FPFVMSolver": (0.39, 1.11),
    "FPParticleSolver": (-0.01, 0.01),
    "FPSLJacobianSolver": (0.00, 0.05),
    "FPSLSolver": (-0.01, 0.47),
}

#: Not a pass. Each needs a geometry or an argument this harness does not build.
_NOT_MEASURED = {
    "FPFEMSolver",
    "FPGFDMSolver",
    "FPNetworkSolver",
    "MeshlessGalerkinFPSolver",
    "WeakFormFPSolver",
}

#: Same class under two names, and one empty deprecated SUBCLASS. Class-keying collapses the
#: first; `_deprecation_meta["alias_for"]` collapses the second. Two identical rows read as two
#: independent confirmations, which is why both are folded rather than reported twice.
_COLLAPSED = {"FPNetworkSolver": ["NetworkFPSolver"], "FPSLSolver": ["FPSLAdjointSolver"]}


def _row(conservation, name):
    matching = [r for r in conservation["rows"] if r["class"] == name]
    assert matching, f"{name} is not in the population at all"
    return matching[0]


def test_the_harness_reference_path_conserves(conservation):
    """Control. If the reference stops conserving, every row is VOID rather than wrong."""
    ref = conservation["reference_drift_pct"]
    assert abs(ref) < 1e-3, f"reference path drifted {ref:+.4f}% -- the harness is broken"


def test_the_population_is_exactly_the_two_sets(conservation):
    got = {r["class"] for r in conservation["rows"]}
    want = set(_MEASURED) | _NOT_MEASURED
    assert got == want, f"population changed: added {sorted(got - want)}, removed {sorted(want - got)}"


def test_the_alias_and_the_deprecated_subclass_are_both_collapsed(conservation):
    got = {r["class"]: r["aliases"] for r in conservation["rows"] if r["aliases"]}
    assert got == _COLLAPSED, (
        f"collapsing changed: {got}. Class-keying catches `X = Y`; `class X(Y): pass` needs the "
        "`_deprecation_meta['alias_for']` fold, and without it two identical rows appear."
    )


@pytest.mark.parametrize("solver", sorted(_MEASURED))
def test_the_wall_ratio_sequence_is_unchanged(solver, conservation):
    """**No verdict.** The sequence is the observation.

    A previous version classified each row from the trend across three resolutions. Independent
    review showed the rule cannot do that: `FPFVMSolver` reads 0.392 / 0.649 / **1.106** and keeps
    going to 12.699 at nx=1281, so a "within 0.15 of 1" clause fires on a value the sequence merely
    transits; `FPSLSolver` reads 0.998 at nx=201 and then 1.926, 3.500. Three points cannot
    separate approach from transit and neither can six.

    So this pins the first and last of the sequence and leaves the reading to a person.
    """
    row = _row(conservation, solver)
    assert row["status"] == "MEASURED", f"{solver} is now {row['status']}: {row['detail']}"
    first, last = row["ratios"][0][1], row["ratios"][-1][1]
    want_first, want_last = _MEASURED[solver]
    assert first == pytest.approx(want_first, abs=0.05), f"{solver} first ratio moved: {row['ratios']}"
    assert last == pytest.approx(want_last, abs=0.05), f"{solver} last ratio moved: {row['ratios']}"


def test_mass_drift_is_reported_beside_the_ratio_and_not_as_a_verdict(conservation):
    """Mass conservation is neither sufficient nor necessary, so it must never be the status.

    NOT SUFFICIENT: streamline diffusion conserves to 1e-12 while the wall ratio collapses.
    NOT NECESSARY: `FPSLJacobianSolver` is the Lagrangian form, non-conservative BY CONSTRUCTION
    with an O(h) error, deprecated for adjoint inconsistency rather than for mass. It carries a
    real drift and is still `MEASURED`; that is the separation this asserts.
    """
    jac = _row(conservation, "FPSLJacobianSolver")
    assert jac["status"] == "MEASURED"
    assert jac["drift_pct"] is not None
    assert abs(jac["drift_pct"]) > 1.0, "the one row where the two axes visibly disagree lost its drift"


@pytest.mark.parametrize("solver", sorted(_NOT_MEASURED))
def test_these_paths_remain_unobservable_by_this_harness(solver, conservation):
    """**NOT_MEASURED is not a pass.** `FPFEMSolver` is in this set and is the class that made
    #1975 wrong the first time."""
    row = _row(conservation, solver)
    assert row["status"] == "NOT_MEASURED", f"{solver} is now measurable: {row['detail']}"


def test_the_declared_drift_convention_is_read_and_not_assumed(census):
    """`_drift_convention` reads VELOCITY on three solvers whose second positional argument is
    `drift_field`. Passing them a potential made the wall-normal drift vanish at the wall the mass
    reached -- the discriminating property absent while a verdict printed anyway."""
    velocity = {
        names[0]
        for cls, names in census.fp_solver_population().items()
        if getattr(getattr(cls, "_drift_convention", None), "name", None) == "VELOCITY"
    }
    assert velocity == {"FPFDMSolver", "FPFVMSolver", "FPGFDMSolver"}, (
        f"the VELOCITY-convention set changed to {sorted(velocity)}; the harness feeds each solver "
        "according to this declaration, so a change here changes what was measured."
    )
