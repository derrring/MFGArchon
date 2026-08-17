"""Ratchet over `scripts/capability_census.py`. Refs #1975, #1977.

The census answers a question the 2026-08-13 design census could not: all four of its lanes look
for reality falling *short* of a claim and it found 77 over-claims; nobody counted the other
direction, and capability that exists undeclared is what made #1975 wrong.

Everything is imported from the script rather than restated, so the two cannot drift. The frozen
sets are the ratchet: leaving them is #1977 progress, joining them is a capability shipped without
a declaration, and each failure message says which.

A second lane over the FP wall condition was removed with the script's. Its findings are in #1975.
The measurement needed a discrimination rule for the LIMIT behaviour of a numerical scheme; four
attempts failed to state one, and 41% of a 32-mutation sweep survived the ratchet built over it --
including the pins for three of the defects that ratchet claimed to have fixed.
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
    """Leaving this set is #1977 progress; joining it is a capability shipped undeclared.

    **This pins WHETHER a class declares, not WHAT it declares.** Measured: widening a gated
    solver's `_SUPPORTED_BC_TYPES` to include `ROBIN` leaves this file green. The declared VALUES
    are pinned per solver in `tests/unit/test_alg/test_solver_bc_support_census_1975.py`; keeping
    a second copy of those sets here would give two frozen copies of one measurement, which is the
    divergence this census exists to find.
    """
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
