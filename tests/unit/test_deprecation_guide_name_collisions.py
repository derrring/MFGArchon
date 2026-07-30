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

_GENERATOR = pathlib.Path(__file__).resolve().parents[2] / "scripts" / "generate_deprecation_guide.py"


def _load():
    spec = importlib.util.spec_from_file_location("generate_deprecation_guide", _GENERATOR)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def gen():
    return _load()


@pytest.fixture(scope="module")
def registry(gen):
    """The real deprecation registry, not a synthetic one.

    A hand-built fixture would be generated from this file's *description* of the defect, and the
    description drops whatever made the defect possible -- here, that the colliding name is a
    parameter whose owner path has to be split off before the identifiers compare equal.
    """
    return gen.deduplicate(gen.scan_all_deprecations())


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
