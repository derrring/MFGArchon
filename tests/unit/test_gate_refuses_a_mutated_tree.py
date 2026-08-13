"""The gate refuses to report on a tree a killed discrimination sweep left mutated (#1849).

`scripts/test_discrimination.py` edits production source in place and restores it in a `finally`.
That survives an exception and SIGINT; it survives neither SIGKILL nor a harness timeout. Observed
twice: once in a worktree (#1849), and once in the MAIN checkout on 2026-08-13, where the leftover
was `hjb_residual_norm` with its load-bearing `sqrt(dx)` deleted -- the exact convention two tests
written that hour existed to guard, so the visible symptom would have been "the new tests are
wrong" and the natural response would have been to weaken them.

The script's own `_assert_clean_tree()` runs at ITS startup, so it protects the next sweep and
nothing else. This guard sits at the point of consumption instead: however the sweep died, the
gate refuses.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
GATE = REPO / "scripts" / "local_ci.sh"


def test_the_gate_greps_for_the_marker_at_the_point_of_consumption():
    body = GATE.read_text()
    assert "# MUTATED" in body, "the gate must look for the marker a killed sweep leaves behind"
    assert "cannot_run" in body.split("# MUTATED")[1][:400], (
        "a mutated tree is an ENVIRONMENT failure -- nothing was measured -- so it must exit "
        "through cannot_run (exit 2), not as a normal red gate (exit 1)"
    )


def test_every_mutation_carries_the_marker_the_guard_greps_for():
    """The guard is complete only if no mutation can land without the marker.

    Measured rather than assumed: 24 of 24 carry it. If a future axis is added without one, the
    guard silently stops covering it and this is the only thing that says so.
    """
    sys.path.insert(0, str(REPO / "scripts"))
    import test_discrimination as td

    missing = [m.name for m in td.MUTATIONS if "MUTATED" not in m.new]
    assert not missing, f"mutations whose `new` text carries no marker, so the gate cannot see them: {missing}"


def test_the_guard_actually_refuses(tmp_path):
    """Behavioural, not textual: plant a marker, run the gate, require exit 2.

    Uses --fast so nothing heavy runs; the guard sits before every check either way. The planted
    file is restored under try/finally, and the assertion on the restore is part of the test --
    a test for this defect that itself leaves the tree dirty would be an unusually poor joke.
    """
    victim = REPO / "mfgarchon" / "geometry" / "boundary" / "bc_utils.py"
    original = victim.read_text()
    try:
        m = re.search(r"( +return \"reflect\")", original)
        assert m, "the anchor moved; update this test rather than deleting it"
        victim.write_text(original[: m.start()] + '        return "clamp"  # MUTATED' + original[m.end() :])
        proc = subprocess.run(["bash", str(GATE), "--fast"], cwd=REPO, capture_output=True, text=True, timeout=300)
    finally:
        victim.write_text(original)
    assert victim.read_text() == original, "the test must not leave the tree mutated"
    assert proc.returncode == 2, f"expected GATE CANNOT RUN (exit 2), got {proc.returncode}"
    assert "GATE CANNOT RUN" in proc.stdout
    assert "bc_utils.py" in proc.stdout, "the refusal must name the file, or recovery is a search"
