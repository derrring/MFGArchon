"""Ratchet over `scripts/capability_census.py`. Refs #1975, #1977.

The census answers two questions the 2026-08-13 design census could not, because all four of its
lanes look for reality falling *short* of a claim. These look the other way — at capability that
exists and is not declared, and at a wall imposed with nothing naming it. Both directions of that
blind spot produced a wrong issue (#1975, filed on a false premise and corrected twice).

Everything here is imported from the script rather than restated, so the two cannot drift. The
frozen sets below are the ratchet: a change is information, and each failure message says which
direction it went and what that means.
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
    sys.modules["capability_census"] = module  # dataclass/typing lookups need it registered
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def declarations(census):
    return census.declaration_matrix()


@pytest.fixture(scope="module")
def conservation(census):
    return census.conservation_verdicts()


# =============================================================================
# Lane 1 -- who declares nothing
# =============================================================================

# Measured 2026-08-17. Recorded as a POPULATION, not as an absence: a class here is one whose
# capability cannot be read off the class at all, which is the state that let #1975 report
# FPFEMSolver's Robin implementation as absent.
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
    """The population is only as complete as the walk. A module that will not import is a hole in
    it, and a silent one — `walk_packages` yields the name either way."""
    assert declarations["import_failures"] == [], (
        f"modules that would not import: {declarations['import_failures']}. Every class they "
        "define is missing from every row of this census."
    )


def test_the_roots_still_exist(declarations):
    """A root that cannot be resolved silently empties its whole lane."""
    assert declarations["roots_missing"] == [], f"unresolvable roots: {declarations['roots_missing']}"


def test_the_classes_outside_every_predicate_are_still_outside_it(declarations):
    """These do a root's job without subclassing it, so no predicate reaches them. Named rather
    than discovered — a population predicate is itself a claim about scope, and this is the part
    no mechanism recovers."""
    assert declarations["outside_every_predicate"] == {
        "HJBHowardSolver": [],
        "ImplicitHeatSolver": [],
        "ParticleApplicator": [],
    }, (
        "a class outside every population predicate moved. If it is now a subclass, delete it from "
        "OUTSIDE_EVERY_PREDICATE — the census reaches it on its own."
    )


def test_the_permissive_default_is_still_claimed_by_inheritance(declarations):
    """`honors_inhomogeneous_neumann` defaults to True on `BaseMFGSolver`, so a solver that never
    mentions it claims to honour an inhomogeneous Neumann flux. That claim was made by a default,
    not by anyone.

    Two inheritance chains, and they mean opposite things — a first draft of this test asserted
    every inherited value was True and failed on the second:

      - from `BaseMFGSolver`: the permissive default nobody chose;
      - from a sibling solver (`FPSLAdjointSolver` <- `FPSLSolver`): a deliberate `False`, one
        level up, which is a real declaration made by someone.

    This does not assert the default is wrong. It asserts the counts do not drift unnoticed,
    because an inherited claim is invisible at every call site.
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
    assert set(from_base.values()) == {"True"}, (
        f"the permissive default is no longer uniformly True: {from_base}. That changes what "
        f"{len(from_base)} solvers claim, at no call site."
    )
    assert from_sibling == {"FPSLAdjointSolver": ("FPSLSolver", "False")}, (
        f"inheritance from a sibling solver changed: {from_sibling}. Unlike the base default, "
        "this one is a choice someone made -- treat a change here as a real declaration moving."
    )


# =============================================================================
# Lane 2 -- which FP paths impose J.n = 0
# =============================================================================

_CONSERVES = {
    "FPFDMSolver",
    "FPFVMSolver",
    "FPParticleSolver",
    "FPSLAdjointSolver",
    "FPSLSolver",
}
#: `FPSLJacobianSolver` gains mass at O(h) with NO drift at all (+0.1396 / +0.0689 / +0.0343 % at
#: Nx = 41 / 81 / 161, ratio 2.01), so its drifted figure says nothing about its wall. Deprecated
#: — this is the quantified reason the deprecation notes never gave.
_CONTROL_FAILED = {"FPSLJacobianSolver"}
#: Not a pass. Each needs a geometry this harness does not build: an unstructured mesh, collocation
#: points, a network, or an explicit discretization.
_NOT_MEASURED = {
    "FPFEMSolver",
    "FPGFDMSolver",
    "FPNetworkSolver",
    "MeshlessGalerkinFPSolver",
    "NetworkFPSolver",
    "WeakFormFPSolver",
}


def test_the_harness_reference_path_conserves(conservation):
    """Control 2. `divergence_upwind` on the raw assembly must conserve exactly; if it does not,
    every verdict below is void rather than merely wrong."""
    ref = conservation["reference_drift_pct"]
    assert abs(ref) < 1e-3, f"reference path drifted {ref:+.4f}% -- the harness is broken, not the solvers"


@pytest.mark.parametrize("solver", sorted(_CONSERVES))
def test_these_paths_conserve_mass_at_a_wall_with_normal_drift(conservation, solver):
    """The oracle #1975 was missing: mass conservation is a law of the equation, computed without
    reference to any scheme's internals, so it answers a question about behaviour that no census
    over declarations or branch names can answer.

    Conserving here means the path imposes `J.n = 0`. Several do it **structurally**, by zeroing
    the total face flux or by reflecting particle paths, with no branch naming the condition —
    which is exactly why reading the dispatch gave the wrong answer.
    """
    row = next(r for r in conservation["rows"] if r["class"] == solver)
    assert row["verdict"] == "CONSERVES", (
        f"{solver} was conserving and now reports {row['verdict']}: {row['detail']}. "
        "It has stopped imposing J.n = 0 at a wall with wall-normal drift."
    )


@pytest.mark.parametrize("solver", sorted(_CONTROL_FAILED))
def test_these_paths_fail_the_zero_drift_control(conservation, solver):
    """Pinned so the control's own finding does not evaporate. A path that leaks with no drift is
    not a wall problem, and reading its drifted number as one is the mistake the control exists to
    stop."""
    row = next(r for r in conservation["rows"] if r["class"] == solver)
    assert row["verdict"] == "CONTROL_FAILED", (
        f"{solver} now reports {row['verdict']}. If it conserves, that is a fix -- move it to _CONSERVES and say so."
    )


@pytest.mark.parametrize("solver", sorted(_NOT_MEASURED))
def test_these_paths_remain_unobservable_by_this_harness(conservation, solver):
    """**A NOT_MEASURED row is not a pass.** It is a path whose wall nobody in this repository can
    currently observe, and that state is what let #1975 be filed on a false premise: `FPFEMSolver`
    is in this set and implements the very thing the issue said was absent.

    A solver leaving this set is the harness gaining reach — record the verdict it now returns.
    """
    row = next(r for r in conservation["rows"] if r["class"] == solver)
    assert row["verdict"] == "NOT_MEASURED", (
        f"{solver} is now measurable and reports {row['verdict']}: {row['detail']}. Move it to the matching set."
    )


def test_the_population_is_exactly_the_three_sets(conservation):
    """No solver may be silently absent from all three. The population comes from the presence of
    `solve_fp_system`, not from any declaration — so a new FP solver appears here whether or not it
    announces anything."""
    got = {r["class"] for r in conservation["rows"]}
    want = _CONSERVES | _CONTROL_FAILED | _NOT_MEASURED
    assert got == want, (
        f"FP solver population changed: added {sorted(got - want)}, removed {sorted(want - got)}. "
        "A new FP solver must be classified before this file means anything."
    )
