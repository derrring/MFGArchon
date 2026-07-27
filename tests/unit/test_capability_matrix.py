"""Pinning tests for the capability ratchet (scripts/capability_matrix.py).

The matrix records what the public solve surface can actually do, and its value comes
entirely from two properties that are easy to erode:

1. ``--check-baseline`` fails in BOTH directions. One-directional checking lets a
   recovery land unrecorded; the next baseline then encodes it as if it had always
   held, and the gate loses the ability to say when anything was fixed.
2. The ``--self-test`` mutation breaks the invariant the oracles measure. The first
   mutation written for it was a constant density scale, which left every drift cell
   PASS -- correctly, since ``max|mass(t) - mass(0)|`` is invariant under a uniform
   rescaling. A control that cannot fail proves nothing, which is the same defect the
   matrix exists to catch (Issues #1714, #1715).

These tests exercise the comparison and mutation logic directly. They do not solve
anything: the cells themselves are exercised by running the script.
"""

import importlib.util
import json
from pathlib import Path

import pytest

import numpy as np

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "capability_matrix.py"


@pytest.fixture(scope="module")
def cm():
    spec = importlib.util.spec_from_file_location("capability_matrix", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _baseline(**statuses):
    return {name: {"status": s, "artifact": {}} for name, s in statuses.items()}


# ---------------------------------------------------------------------------
# Baseline comparison -- the ratchet
# ---------------------------------------------------------------------------


def test_identical_statuses_report_no_problem(cm):
    base = _baseline(a="PASS", b="UNSUPPORTED")
    assert cm.compare_to_baseline({"a": "PASS", "b": "UNSUPPORTED"}, base) == []


def test_regression_is_caught(cm):
    problems = cm.compare_to_baseline({"a": "FAIL"}, _baseline(a="PASS"))
    assert len(problems) == 1
    assert "REGRESSION" in problems[0]


def test_recovery_is_also_caught(cm):
    """The direction that a one-way ratchet would silently allow."""
    problems = cm.compare_to_baseline({"a": "PASS"}, _baseline(a="UNSUPPORTED"))
    assert len(problems) == 1
    assert "RECOVERED" in problems[0]


def test_shift_between_two_non_pass_states_is_caught(cm):
    """UNSUPPORTED -> FAIL is neither regression nor recovery, and still must not pass.

    This is the transition that would hide a solver going from "refuses to run" to
    "runs and returns a wrong answer" -- strictly worse, and invisible to a check
    that only compares against PASS.
    """
    problems = cm.compare_to_baseline({"a": "FAIL"}, _baseline(a="UNSUPPORTED"))
    assert len(problems) == 1
    assert "SHIFT" in problems[0]


def test_deleted_cell_is_caught(cm):
    """Deleting a cell must not be a way to make a red baseline go green."""
    problems = cm.compare_to_baseline({}, _baseline(a="PASS"))
    assert len(problems) == 1
    assert "DISAPPEARED" in problems[0]


def test_new_cell_is_caught(cm):
    problems = cm.compare_to_baseline({"a": "PASS", "b": "PASS"}, _baseline(a="PASS"))
    assert len(problems) == 1
    assert "NEW cell" in problems[0]


def test_shipped_baseline_covers_every_declared_cell(cm):
    """A cell added without regenerating the baseline is a cell nothing watches."""
    with open(_SCRIPT.parent / "capability_baseline.json") as fh:
        baseline = json.load(fh)["cells"]
    assert set(baseline) == set(cm.CELLS), (
        "scripts/capability_baseline.json is out of sync with CELLS; regenerate with --write-baseline"
    )


def test_every_mass_oracle_cell_is_a_declared_cell(cm):
    assert set(cm.CELLS) >= cm.MASS_ORACLE_CELLS


def test_currently_failing_density_cells_are_still_registered_for_the_self_test(cm):
    """A cell that is not PASS today has an oracle the self-test cannot yet exercise.

    ``regime_switching/non_negativity`` raises before producing a density (#1681), so
    nothing has proved its oracle reads what it claims to. Listing it in
    MASS_ORACLE_CELLS is what makes the self-test pick it up the moment it recovers --
    without that, the recovery would land with an unproven oracle, which is exactly
    the state #1715 measured in the agreement tests.
    """
    density_cells = {
        "fdm_upwind/mass_conservation",
        "sl_linear/mass_conservation",
        "fdm_centered/mass_conservation",
        "fvm_muscl/mass_conservation",
        "fvm_vs_fdm/agreement",
        "regime_switching/non_negativity",
    }
    assert density_cells <= cm.MASS_ORACLE_CELLS, (
        f"unregistered density cells: {sorted(density_cells - cm.MASS_ORACLE_CELLS)}"
    )


# ---------------------------------------------------------------------------
# The self-test mutation -- the control has to be able to fail
# ---------------------------------------------------------------------------


def _drift(M):
    mass = M.sum(axis=1)
    return float(np.abs(mass - mass[0]).max())


def test_ramp_mutation_injects_drift(cm, monkeypatch):
    M = np.ones((5, 4))
    assert _drift(M) == 0.0
    monkeypatch.setattr(cm, "_DENSITY_MUTATION", 0.10)
    mutated = cm._apply_mutation(M)
    assert _drift(mutated) == pytest.approx(0.4, rel=1e-12)  # 10% of mass 4.0


def test_ramp_mutation_leaves_the_first_slice_alone(cm, monkeypatch):
    """t=0 is the reference the drift is measured against; moving it would mask drift."""
    M = np.ones((5, 4))
    monkeypatch.setattr(cm, "_DENSITY_MUTATION", 0.10)
    np.testing.assert_allclose(cm._apply_mutation(M)[0], M[0])


def test_a_constant_scale_would_not_inject_drift(cm):
    """Pins the reason the mutation is a ramp.

    If someone replaces the ramp with ``M * factor``, the self-test goes back to
    reporting every drift cell INERT -- not because the cells broke, but because the
    control cannot break them. This test states the arithmetic so that change is a
    deliberate one.
    """
    M = np.ones((5, 4))
    assert _drift(M * 1.5) == 0.0


def test_mutation_is_off_by_default(cm):
    """A leaked mutation would make every cell FAIL and the baseline meaningless."""
    assert cm._DENSITY_MUTATION is None
    M = np.arange(20, dtype=float).reshape(5, 4)
    np.testing.assert_array_equal(cm._apply_mutation(M), M)


def test_mutation_applies_along_time_for_higher_dimensional_density(cm, monkeypatch):
    """2-D densities are (nt, nx, ny); the ramp must broadcast over time, not space."""
    M = np.ones((4, 3, 2))
    monkeypatch.setattr(cm, "_DENSITY_MUTATION", 0.30)
    mutated = cm._apply_mutation(M)
    assert mutated.shape == M.shape
    np.testing.assert_allclose(mutated[0], 1.0)
    np.testing.assert_allclose(mutated[-1], 1.30)


# ---------------------------------------------------------------------------
# Harness-error classification
# ---------------------------------------------------------------------------


def test_harness_breakage_is_error_not_unsupported(cm, monkeypatch):
    """A wrong signature must never read as "the library does not support this".

    This is the defect the gfdm_rbf cell shipped with on its first run: a missing
    positional argument surfaced as UNSUPPORTED, indistinguishable from the
    NotImplementedError the cell exists to record.
    """

    def broken():
        raise TypeError("f() missing 1 required positional argument: 'collocation_points'")

    monkeypatch.setattr(cm, "CELLS", {"broken/cell": broken})
    assert cm.evaluate()["broken/cell"]["status"] == "ERROR"


def test_a_refusal_by_the_code_under_test_is_unsupported(cm, monkeypatch):
    def refuses():
        raise NotImplementedError("derivative_method='rbf' is not supported (Issue #1553)")

    monkeypatch.setattr(cm, "CELLS", {"refuses/cell": refuses})
    result = cm.evaluate()["refuses/cell"]
    assert result["status"] == "UNSUPPORTED"
    assert result["artifact"]["exception"] == "NotImplementedError"
    assert "#1553" in result["artifact"]["message"]
