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


#: Every solver that OWNS `honors_inhomogeneous_neumann` refuses an inhomogeneous flux. A `True`
#: here is a live claim the #1686 gate stops enforcing, and nothing else in the tree pins it.
_HONORS_INHOMOGENEOUS_NEUMANN_OWN = {
    "FPFDMSolver": "False",
    "FPFVMSolver": "False",
    "FPGFDMSolver": "False",
    "FPParticleSolver": "False",
    "FPSLJacobianSolver": "False",
    "FPSLSolver": "False",
}


def test_the_solvers_that_own_the_inhomogeneous_neumann_flag_all_refuse(declarations):
    """A capability whose VALUE nothing else asserts. Joining or leaving this set is #1686 news."""
    got = {
        r["names"][0]: r["own_values"]["honors_inhomogeneous_neumann"]
        for r in declarations["rows"]
        if "honors_inhomogeneous_neumann" in r["own"]
    }
    assert got == _HONORS_INHOMOGENEOUS_NEUMANN_OWN, f"changed: {got}"


@pytest.mark.parametrize("role", sorted(_DECLARES_NOTHING))
def test_the_set_of_classes_declaring_nothing_is_unchanged(declarations, role):
    """Leaving this set is #1977 progress; joining it is a capability shipped undeclared.

    **This pins WHETHER a class declares, not WHAT it declares.** Measured: widening a gated
    solver's `_SUPPORTED_BC_TYPES` to include `ROBIN` leaves this file green.

    ~~The declared VALUES will be pinned per solver by PR #1976.~~ [CORRECTED 2026-08-17] -- true
    of ONE of the three declarations. #1976 pins `_SUPPORTED_BC_TYPES` per solver (verified: the
    same ROBIN mutation fails 2 of its 29 tests), so that hole has a cover and **#1976 must merge
    first**; a second frozen copy here would give two copies of one measurement, the divergence
    this census exists to find.

    `honors_inhomogeneous_neumann` had NO cover in either file -- flipping `FPFDMSolver`'s own
    value `False -> True`, a solver silently claiming to honour a flux it does not honour, passed
    8 of 8 here and 29 of 29 there. Since nothing else pins it, there is no duplication to create,
    and the test below now does. `discretization_type`'s two owners declare it as a `property`
    object whose `str()` carries a memory address, so its value is not pinnable this way and stays
    uncovered.

    An unwritten sibling of the same hole: a declaration set on the INSTANCE
    (`self._SUPPORTED_BC_TYPES = ...` in `__init__`) is invisible here while the runtime gate
    would honour it -- `_validate_bc_support` reads `getattr(self, "supported_bc_types")` and
    `fp_fdm.py:121` returns `self._SUPPORTED_BC_TYPES`. No class does this today.
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
    """The emptiness guard comes first: `roots_missing == []` is also true of a census with no
    roots at all, and so is the outside-predicate test below. Measured with `ROOTS = {}`: 5 of the
    8 tests went red and these two stayed green, each on a vacuous truth."""
    assert len(declarations["rows"]) > 40, f"population collapsed to {len(declarations['rows'])} rows"
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
    assert len(from_base) == 18, f"solvers claiming True by the permissive default: {sorted(from_base)}"
    assert set(from_base.values()) == {"True"}
    assert from_sibling == {"FPSLAdjointSolver": ("FPSLSolver", "False")}
