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
    return census.conservation_verdicts()


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
# Lane 2 -- which wall each FP path imposes
# =============================================================================

#: The wall ratio `d_n m / ((v_n/D) m_wall)` RISES toward 1 under refinement. Measured at
#: Nx = 41 / 81 / 161. These paths impose `J.n = 0`; several do it structurally, with no branch
#: naming the condition, which is what made a sweep over branch names read the wrong answer.
_IMPOSES_J_DOT_N = {"FPFDMSolver", "FPFVMSolver", "FPSLSolver", "FPSLAdjointSolver"}

#: The ratio FALLS toward 0: the wall converges to `d_n m = 0`, which is the wrong condition when
#: the drift is not tangential. Its `O(h)` mass error is a SEPARATE and legitimate fact — the
#: Lagrangian form is non-conservative by construction — and the previous instrument fused the two
#: into one `CONTROL_FAILED`.
_IMPOSES_ZERO_GRADIENT = {"FPSLJacobianSolver"}

#: Measured, but the witness does not transfer. A particle method's density is a binned/KDE
#: reconstruction, so a one-sided difference on it is not the `d_n m` the ratio is defined from,
#: and the reflection lives in the paths rather than in the density. Recorded rather than given a
#: verdict: a witness that does not apply is not evidence either way.
_WITNESS_NOT_APPLICABLE = {"FPParticleSolver"}

#: Not a pass. Each needs a geometry or an argument this harness does not build. Independent
#: measurement (recipes in #1975) found `FPFEMSolver` and `MeshlessGalerkinFPSolver` impose
#: `J.n = 0` as the NATURAL BC of the weak form, `FPNetworkSolver` conserves structurally with no
#: boundary in this sense, `WeakFormFPSolver` is not independently constructible, and
#: `FPGFDMSolver` imposes NEITHER condition -- its `_boundary_type` is resolved, gate-validated,
#: stored at `fp_gfdm.py:209` and never read again.
_NOT_MEASURED = {
    "FPFEMSolver",
    "FPGFDMSolver",
    "FPNetworkSolver",
    "MeshlessGalerkinFPSolver",
    "WeakFormFPSolver",
}


def _row(conservation, name):
    return next(r for r in conservation["rows"] if r["class"] == name)


def test_the_harness_reference_path_conserves(conservation):
    """Control 2. If the reference stops conserving, every verdict is VOID rather than wrong."""
    ref = conservation["reference_drift_pct"]
    assert abs(ref) < 1e-3, f"reference path drifted {ref:+.4f}% -- the harness is broken, not the solvers"


def test_the_population_is_exactly_the_four_sets(conservation):
    """Keyed on the class, not the binding name: `NetworkFPSolver = FPNetworkSolver` is a
    module-level alias (`fp_network.py:606`) and name-keying reported it as a second, independent
    unmeasured path."""
    got = {r["class"] for r in conservation["rows"]}
    want = _IMPOSES_J_DOT_N | _IMPOSES_ZERO_GRADIENT | _WITNESS_NOT_APPLICABLE | _NOT_MEASURED
    assert got == want, f"FP solver population changed: added {sorted(got - want)}, removed {sorted(want - got)}."


def test_the_alias_is_still_reported_as_an_alias(conservation):
    assert _row(conservation, "FPNetworkSolver")["aliases"] == ["NetworkFPSolver"], (
        "the alias collapsed or moved; a name-keyed population would double-count it again"
    )


@pytest.mark.parametrize("solver", sorted(_IMPOSES_J_DOT_N))
def test_these_paths_converge_to_the_flux_wall(conservation, solver):
    """The verdict is the ratio's TREND, not its value. At Nx=81 the boundary layer is
    `D/v = 0.014` against `h = 0.0125`, so a correct path reads ~0.5-0.7 there — a fixed threshold
    called that "imposes neither", which is defect 8 above."""
    row = _row(conservation, solver)
    assert row["verdict"] == "IMPOSES_J_DOT_N", f"{solver} now reports {row['verdict']}: {row['detail']}"
    ratios = [r for _, r in row["ratios"]]
    assert ratios[-1] > ratios[0], f"{solver}'s wall ratio stopped rising under refinement: {ratios}"


@pytest.mark.parametrize("solver", sorted(_IMPOSES_ZERO_GRADIENT))
def test_these_paths_converge_to_the_zero_gradient_wall(conservation, solver):
    """The wrong wall under a normal drift, and a finding independent of the mass column."""
    row = _row(conservation, solver)
    assert row["verdict"] == "IMPOSES_ZERO_GRADIENT", (
        f"{solver} now reports {row['verdict']}: {row['detail']}. If its ratio now rises toward 1 "
        "that is a fix -- move it to _IMPOSES_J_DOT_N and say so."
    )


def test_a_non_conservative_form_is_not_thereby_wrong(conservation):
    """Mass conservation is neither sufficient nor necessary, so it must never be the verdict.

    NOT SUFFICIENT: streamline diffusion conserves to 1e-12 while the wall ratio collapses
    0.967 -> 0.414. NOT NECESSARY: `FPSLJacobianSolver` is the Lagrangian form
    `m^{n+1}(x) = m^n(x - a dt) exp(-dt div a)`, non-conservative by construction with an O(h)
    error that halves under refinement, and deprecated for ADJOINT INCONSISTENCY, not for mass.

    This asserts the two axes stay separable: a path may carry a real mass drift and still be
    judged on its wall.
    """
    row = _row(conservation, "FPSLJacobianSolver")
    assert row["drift_pct"] is not None, "FPSLJacobianSolver's mass column went unmeasured"
    assert abs(row["drift_pct"]) > 1.0, (
        "FPSLJacobianSolver's mass drift vanished; it is the only path where the two axes visibly "
        "disagree, and this test is what keeps them from being fused back into one verdict"
    )
    assert row["verdict"] == "IMPOSES_ZERO_GRADIENT", "the verdict must come from the wall, not the mass"


@pytest.mark.parametrize("solver", sorted(_WITNESS_NOT_APPLICABLE))
def test_the_wall_witness_does_not_transfer_to_a_particle_density(conservation, solver):
    """Recorded, not judged. The ratio is defined from `d_n m` on a represented density; a binned
    or KDE-reconstructed one is a different object, and the reflection lives in the paths. A
    witness that does not apply is not evidence either way — treating its output as a verdict is
    the same error as reading `NOT_MEASURED` as a pass."""
    row = _row(conservation, solver)
    assert row["ratios"], f"{solver} produced no ratio at all; the row's premise changed"
    assert all(abs(r) < 1e-9 for _, r in row["ratios"]), (
        f"{solver}'s ratio is no longer identically zero ({row['ratios']}). If the density is now "
        "represented rather than binned, this witness may apply -- re-derive before judging."
    )


@pytest.mark.parametrize("solver", sorted(_NOT_MEASURED))
def test_these_paths_remain_unobservable_by_this_harness(conservation, solver):
    """**NOT_MEASURED is not a pass.** It is a path whose wall this harness cannot observe.
    `FPFEMSolver` is in this set and is the class that made #1975 wrong the first time."""
    row = _row(conservation, solver)
    assert row["verdict"] == "NOT_MEASURED", (
        f"{solver} is now measurable and reports {row['verdict']}: {row['detail']}. Move it to the matching set."
    )


def test_the_declared_drift_convention_is_read_and_not_assumed(conservation, census):
    """Defect 7, pinned. `_drift_convention` reads `VELOCITY` on three solvers; passing them a
    potential made the wall-normal drift vanish at the wall the mass reached, so the
    discriminating property was absent while a verdict printed anyway."""
    # Read from the CLASS, not from the measured row: the declaration exists whether or not the
    # harness can construct the solver, and reading it off the measurement made FPGFDMSolver --
    # which is NOT_MEASURED -- silently drop out of the set. Same shape as defect 7 itself.
    velocity = {
        names[0]
        for cls, names in census.fp_solver_population().items()
        if getattr(getattr(cls, "_drift_convention", None), "name", None) == "VELOCITY"
    }
    assert velocity == {"FPFDMSolver", "FPFVMSolver", "FPGFDMSolver"}, (
        f"the set of VELOCITY-convention solvers changed to {sorted(velocity)}; the harness feeds "
        "each solver according to this declaration, so a change here changes what was measured."
    )
    # Name and declaration disagree here, and the declaration is what the harness follows.
    import inspect

    from mfgarchon.alg.numerical.fp_solvers.fp_particle import FPParticleSolver

    params = list(inspect.signature(FPParticleSolver.solve_fp_system).parameters)
    assert params[2] == "drift_field", "the disagreement this pins has moved"
    assert getattr(FPParticleSolver._drift_convention, "name", None) == "VALUE_FUNCTION", (
        "FPParticleSolver's parameter is named `drift_field` while its declared convention is "
        "VALUE_FUNCTION; if that is reconciled, record it -- it is a live trap for any harness."
    )
