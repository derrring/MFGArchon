"""The deprecation guide must not tell a reader to migrate to a name it elsewhere deprecates.

`drift_field` is the destination on `FPFDMSolver.solve_fp_system` (replacing `velocity_field`,
where it means the optimal control a*) and is simultaneously deprecated in favour of
`potential_field` on eight other FP solvers (where it means the value function U). The generated
guide listed all nine rows with nothing marking them as different quantities, so the reader's
reasonable conclusion -- that the name is on its way out everywhere -- is the wrong one.

Both parameters exist on both solver families, so the wrong migration is accepted silently.
Measured on a 21-point 1D problem, sigma = 0.3, T = 0.2, with a constant optimal control
alpha = 1.0 and initial mass centred at 0.3:

    drift_field=alpha      -> final centroid 0.5055   (correct: the control advects the mass)
    potential_field=alpha  -> final centroid 0.3151   (the solver computes -c*grad(alpha) = 0,
                                                       so the advection vanishes; pure diffusion)

A 37.7% error in the transported centroid, no exception and no warning. Issues #1043, #1044.
"""

from __future__ import annotations

import importlib.util
import pathlib

import pytest
from _pytest.outcomes import Skipped

_SCRIPTS = pathlib.Path(__file__).resolve().parents[2] / "scripts"


def _load(stem: str):
    spec = importlib.util.spec_from_file_location(stem, _SCRIPTS / f"{stem}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# The ratchet owns the frozen-paradigm scope rule (`FROZEN`, `is_frozen`); this file reads it
# rather than restating it. Module scope, because `_live_holes` is called from a fixture and from
# a test, and every frozen module name below is BUILT from `FROZEN` -- check_frozen_areas.py counts
# string literals naming a frozen package, so spelling one here would register as a new test
# against a frozen prototype (CLAUDE.md).
_RATCHET = _load("check_internal_deprecation")


def _live_holes(unimportable: dict[str, str]) -> list[str]:
    """The modules in an incomplete scan that the live library needed, frozen paradigms excluded."""
    return sorted(module for module in unimportable if not _RATCHET.is_frozen(module))


def _scan_or_skip(scan):
    """Run `scan`, and turn an incomplete-but-frozen-only tree into a skip rather than an error.

    A free function rather than an inline `try` in the fixture, so a test can drive it with a
    fabricated `IncompleteScanError` and check BOTH branches. Inline, the two ways this can rot --
    skipping unconditionally, or refusing unconditionally -- are invisible: independent review
    mutated each of them and every environment stayed green, including the authoritative gate.
    That is the failure this guard exists to prevent, one level up from where it prevents it.
    """
    from mfgarchon.utils.deprecation import IncompleteScanError

    try:
        return scan()
    except IncompleteScanError as exc:
        if _live_holes(exc.unimportable):
            raise
        pytest.skip(
            f"the package cannot be read in full here, so the shipped guide cannot be generated: "
            f"{len(exc.unimportable)} frozen-paradigm module(s) need optional extras. "
            f"Install `.[nn]` to run this, or rely on ./scripts/local_ci.sh, the authoritative "
            f"gate, which runs in a complete environment."
        )


@pytest.fixture(scope="module")
def gen():
    return _load("generate_deprecation_guide")


@pytest.fixture(scope="module")
def registry(gen):
    """The real deprecation registry, not a synthetic one.

    A hand-built fixture would be generated from this file's *description* of the defect, and the
    description drops whatever made the defect possible -- here, that the colliding name is a
    parameter whose owner path has to be split off before the identifiers compare equal.

    Which is why this needs a tree it can read in FULL, and says so instead of erroring. The guide
    is user-facing, so `scan_all_deprecations` refuses a partial walk outright (#1713) -- in an
    environment without `[nn]` the document under test cannot be produced at all, and asserting
    about the one a partial walk would yield answers a question nobody ships. The nightly runs
    `.[dev,numerical]` on purpose, keeping a 2 GB wheel off the critical path (deprecation-check.yml),
    and errored here every night from #1830 until this guard (#1836).

    The hole has to be entirely inside the frozen paradigms. A LIVE module that will not import is
    a real breakage, and turning that into a skip is how a suite goes quiet about the thing it was
    built to catch. That decision lives in `_scan_or_skip`, where a test can reach it.
    """
    return gen.deduplicate(_scan_or_skip(gen.scan_all_deprecations))


def test_the_live_collision_is_detected(gen, registry):
    collisions = gen.find_name_collisions(registry)
    assert "drift_field" in collisions, (
        "drift_field is the destination on FPFDMSolver and deprecated on eight other FP solvers; "
        "if this is empty the detector stopped seeing the case it was written for"
    )

    sides = collisions["drift_field"]
    destinations = {row["method"] for row in sides["replaces"]}
    deprecated = {row["method"] for row in sides["deprecated_in"]}
    assert "FPFDMSolver.solve_fp_system" in destinations
    assert deprecated >= {"FPSLSolver.solve_fp_system", "WeakFormFPSolver.solve_fp_system"}
    assert destinations & deprecated == set(), "a method cannot be on both sides for one name"

    # The pairing must survive into the row, or the reader has to reconstruct it.
    assert {row["other"] for row in sides["replaces"]} == {"velocity_field"}
    assert {row["other"] for row in sides["deprecated_in"]} == {"potential_field"}


def test_an_unambiguous_registry_produces_no_section(gen):
    """Positive control: the section must be absent when there is nothing to warn about.

    Without this, "the section is present" is satisfied by a detector that flags everything, and
    the guide would carry a permanent warning that readers learn to skip.
    """
    clean = [
        {"name": "Solver.method.old_name", "replacement": "new_name", "type": "parameter", "since": "v0.1.0"},
        {"name": "Solver.method.other_old", "replacement": "other_new", "type": "parameter", "since": "v0.1.0"},
    ]
    assert gen.find_name_collisions(clean) == {}
    assert gen.format_collisions({}) == []
    assert "Do not migrate these across solvers" not in gen.generate_guide(clean)


def _raising(unimportable):
    """A stand-in for `scan_all_deprecations` that fails the way the real one does."""
    from mfgarchon.utils.deprecation import IncompleteScanError

    def scan():
        raise IncompleteScanError(unimportable)

    return scan


# The nightly's own failure, minus the module that made it interesting. `.[dev,numerical]` cannot
# import 20 modules; nineteen were frozen and the twentieth, `backends/numba_backend`, is LIVE --
# it is why this branch installs numba rather than only guarding the test.
#
# Built from a SYNTHETIC frozen prefix rather than from `_RATCHET.FROZEN[0]`. The two packages that
# populated that tuple were deleted, so `FROZEN` is now empty and indexing it raises -- but the
# scoping mechanism it feeds is still the right shape for the next thing that gets frozen, and what
# this file tests is the CLASSIFIER, not whichever package happens to be in scope. Coupling the
# classifier's verification to a live package name is what made these tests go red on a deletion
# that did not touch the classifier at all.
_FROZEN_PREFIX = "mfgarchon.alg.a_frozen_paradigm"
_FROZEN_ONLY = {
    f"{_FROZEN_PREFIX}.core.networks": "ModuleNotFoundError: No module named 'torch'",
    f"{_FROZEN_PREFIX}.core.utils": "ModuleNotFoundError: No module named 'torch'",
    f"{_FROZEN_PREFIX}.nn.feedforward": "ModuleNotFoundError: No module named 'torch'",
}
_LIVE_HOLE = "mfgarchon.backends.numba_backend"


@pytest.fixture(autouse=True)
def _a_frozen_paradigm_exists(monkeypatch):
    """Give the classifier something frozen to classify.

    `FROZEN` is empty in the live package -- the two paradigms that populated it were deleted -- so
    "a frozen hole" is not a reachable category without one. Patching it here tests the CLASSIFIER
    under a controlled scope instead of under whatever the package happens to be carrying, which is
    what these tests were always about; reading the ambient value only made them hostage to it.
    """
    monkeypatch.setattr(_RATCHET, "FROZEN", (_FROZEN_PREFIX,))


def test_the_scan_guard_classifies_the_holes():
    """The classifier alone: frozen holes are not live ones, and a live hole is reported."""
    assert _live_holes(_FROZEN_ONLY) == []
    assert _live_holes({**_FROZEN_ONLY, _LIVE_HOLE: "ImportError: Numba required"}) == [_LIVE_HOLE]


def test_the_scan_guard_skips_on_a_frozen_only_tree():
    """Driving the guard, not the classifier under it.

    Asserting on `_live_holes` alone leaves the `try/except` untested, and independent review
    measured what that costs: mutating the guard to skip unconditionally -- the exact degradation
    this file's docstrings warn about -- left every environment green, the authoritative gate
    included.
    """
    with pytest.raises(Skipped):
        _scan_or_skip(_raising(_FROZEN_ONLY))


def test_the_scan_guard_refuses_a_tree_with_a_live_hole():
    """The half that must NOT be a skip, asserted so that a skip fails rather than passes.

    `pytest.raises(IncompleteScanError)` is the obvious spelling and it is not enough: under a
    guard that skips unconditionally, `Skipped` propagates out of this call and pytest records the
    test as SKIPPED -- exit 0, invisible in the summary, which is what the mutation would exploit.
    So the skip is caught explicitly and turned into a failure.
    """
    from mfgarchon.utils.deprecation import IncompleteScanError

    scan = _raising({**_FROZEN_ONLY, _LIVE_HOLE: "ImportError: Numba required for Numba backend"})
    try:
        _scan_or_skip(scan)
    except Skipped:
        pytest.fail(
            f"the guard skipped a tree whose live library it could not read ({_LIVE_HOLE}); a real "
            f"breakage is now reported as 'not installed', which is how a suite goes quiet"
        )
    except IncompleteScanError:
        return
    pytest.fail("the guard neither raised nor skipped on an unreadable tree")


def test_a_name_deprecated_only_at_a_different_owner_still_collides(gen):
    """The owner path must be split off before comparing, which is what the real case needs.

    `FPFDMSolver.solve_fp_system.drift_field` and `FPSLSolver.solve_fp_system.drift_field` are
    distinct registry keys; comparing the dotted names finds no collision at all. Mutating
    `_short_name` to return the full name reddens this.
    """
    items = [
        {"name": "A.solve.velocity_field", "replacement": "drift_field", "type": "parameter", "since": "v0.1.0"},
        {"name": "B.solve.drift_field", "replacement": "potential_field", "type": "parameter", "since": "v0.1.0"},
    ]
    assert set(gen.find_name_collisions(items)) == {"drift_field"}


def test_the_guide_carries_the_warning_and_flags_every_affected_row(gen, registry):
    """The section alone is not enough: a reader scanning one row must see the ambiguity there."""
    guide = gen.generate_guide(registry)
    assert "## Do not migrate these across solvers" in guide

    section_at = guide.index("## Do not migrate these across solvers")
    first_listing = guide.index("## Deprecated since")
    assert section_at < first_listing, "the warning must precede the rows it qualifies, not follow them"

    rows = [ln for ln in guide.splitlines() if ln.startswith("- **`") and "drift_field" in ln]
    assert len(rows) >= 9, f"expected the FDM destination row plus eight deprecations, found {len(rows)}"
    unflagged = [ln for ln in rows if "Do not migrate these across solvers" not in ln]
    assert not unflagged, "these rows carry the ambiguity with no marker:\n" + "\n".join(unflagged)


def test_the_flag_fires_on_the_destination_row_not_only_the_deprecated_ones(gen, registry):
    """The `velocity_field -> drift_field` row is the one that makes the name look canonical.

    Flagging only the rows whose *deprecated* name collides leaves that row clean, and it is the
    row a FDM user reads. The check is `{short_name, replacement} & collisions`, not `short_name`.
    """
    guide = gen.generate_guide(registry)
    # The listing row, not the summary table's -- the table names both sides by construction.
    fdm_row = next(ln for ln in guide.splitlines() if ln.startswith("- **`velocity_field`**") and "FPFDMSolver" in ln)
    assert "Do not migrate these across solvers" in fdm_row, fdm_row
