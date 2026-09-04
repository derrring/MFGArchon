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
import shutil
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

    Planted in a throwaway git worktree, never in the checkout this test is running inside (#2244).
    The earlier version mutated `mfgarchon/geometry/boundary/bc_utils.py` in place for however long
    the subprocess took -- 12s measured -- and under `-n auto` every other worker was reading that
    file at the same time. `test_discrimination_ratchet`'s anchor `[4]` is precisely the text this
    plants over, so it read zero matches and the gate went red on a healthy tree, with a message
    ("the source moved; update the mutation") pointing at a file that was fine.

    The worktree also has to supply the SCRIPT, not just the tree: `local_ci.sh` line 15 is
    `cd "$(dirname "$0")/.."`, so it pins its own repo root and ignores the caller's cwd. Passing
    `cwd=` at the old call site was decorative. The `gate package` line in its output is the
    receipt for which tree was actually measured, and the assertions below read it in both
    directions -- naming the worktree is not enough on its own, since a run that named the main
    checkout as well would satisfy it.

    The working tree's `scripts/` is copied in, so what runs is the gate as it stands rather than
    the gate as last committed.
    """
    worktree = tmp_path / "isolated"
    add = subprocess.run(
        ["git", "worktree", "add", "--detach", str(worktree), "HEAD"],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert add.returncode == 0, f"could not create the worktree: {add.stderr}"
    try:
        # The worktree is checked out at HEAD, so without this the test would measure the
        # COMMITTED gate and an uncommitted change to the guard would pass. `scripts/` is 1.3M;
        # copying it means the isolation costs the tree, not the script under development.
        shutil.rmtree(worktree / "scripts")
        shutil.copytree(REPO / "scripts", worktree / "scripts")

        victim = worktree / "mfgarchon" / "geometry" / "boundary" / "bc_utils.py"
        original = victim.read_text()
        m = re.search(r"( +return \"reflect\")", original)
        assert m, "the anchor moved; update this test rather than deleting it"
        victim.write_text(original[: m.start()] + '        return "clamp"  # MUTATED' + original[m.end() :])
        proc = subprocess.run(
            ["bash", str(worktree / "scripts" / "local_ci.sh"), "--fast"],
            cwd=worktree,
            capture_output=True,
            text=True,
            timeout=300,
        )
    finally:
        subprocess.run(
            ["git", "worktree", "remove", "--force", str(worktree)],
            cwd=REPO,
            capture_output=True,
            timeout=120,
        )

    assert proc.returncode == 2, f"expected GATE CANNOT RUN (exit 2), got {proc.returncode}"
    assert "GATE CANNOT RUN" in proc.stdout
    assert "bc_utils.py" in proc.stdout, "the refusal must name the file, or recovery is a search"
    assert "mutation marker" in proc.stdout, (
        "exit 2 is also the environment-failure code, so the code alone does not say the mutation "
        "guard is what fired -- an unresolvable interpreter would satisfy every assertion above"
    )
    # Both directions, because either alone can pass while the isolation is gone.
    assert str(worktree / "mfgarchon") in proc.stdout, (
        f"the gate did not report measuring the worktree package. Its own `gate package` line is "
        f"the receipt for which tree it read:\n{proc.stdout}"
    )
    assert str(REPO / "mfgarchon") not in proc.stdout, (
        "the gate reported measuring the MAIN checkout, so the isolation is gone and this test is "
        "mutating the tree its own suite is reading -- the #2244 defect, back"
    )


def test_the_guard_test_leaves_the_running_checkout_untouched():
    """The property #2244 bought, asserted rather than assumed.

    A test that plants a marker in the tree it runs inside races every reader in the suite, not
    just the one that caught it. This pins that the file the previous test plants over is clean
    here -- if the isolation regresses, the two tests together still cannot both pass.
    """
    victim = REPO / "mfgarchon" / "geometry" / "boundary" / "bc_utils.py"
    assert "# MUTATED" not in victim.read_text(), (
        "the running checkout carries a mutation marker; the guard test is planting in place again"
    )
