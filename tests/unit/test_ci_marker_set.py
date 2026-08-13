"""The gate's marker set has one owner, and it honours what `pytest.ini` declares.

Issue #1909. Two defects, both latent rather than biting:

1. `pytest.ini` declares `manual` as "Runs in NO automatic tier", and `scripts/local_ci.sh` --
   the authoritative gate -- did not exclude it. The contract held only by coincidence: all ten
   `manual` tests happened to also carry `slow`, which `not slow` removed. Marking a test
   `manual` without `slow` would have run it in the gate, silently, and the whole point of the
   marker is that nothing reports when such a test breaks.

2. The marker string was written twice, byte-identical, in `local_ci.sh` and in
   `scripts/test_discrimination.py`, bound by a comment saying they must match. Diverging them
   measures every kill count in `discrimination_baseline.json` against a different population
   than the gate runs, and nothing fails.
"""

from __future__ import annotations

import configparser
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
MARKER_FILE = REPO / "scripts" / "ci_markers.txt"

# A marker whose pytest.ini description says it runs in no automatic tier MUST be excluded by the
# gate. Matched on the declaration rather than on a hard-coded list, so a future marker with the
# same contract is covered the day it is declared.
NON_AUTOMATIC = re.compile(r"runs in no automatic tier", re.IGNORECASE)


def _declared_markers() -> dict[str, str]:
    cfg = configparser.ConfigParser()
    cfg.read(REPO / "pytest.ini")
    out: dict[str, str] = {}
    for line in cfg["pytest"]["markers"].strip().splitlines():
        if ":" in line:
            name, _, desc = line.partition(":")
            out[name.strip()] = desc.strip()
    return out


def test_the_marker_set_is_a_file_not_a_restated_string():
    """One owner. Both readers must reach for it rather than carry a copy."""
    assert MARKER_FILE.exists(), f"{MARKER_FILE} is the single owner of the gate's marker set"
    ci = (REPO / "scripts" / "local_ci.sh").read_text()
    disc = (REPO / "scripts" / "test_discrimination.py").read_text()
    assert "ci_markers.txt" in ci, "local_ci.sh must read the marker file, not restate the string"
    assert "ci_markers.txt" in disc, "test_discrimination.py must read the marker file, not restate it"
    # The literal itself must appear in neither: a leftover copy is what this test exists to stop.
    stale = "not slow and not benchmark and not experimental"
    assert stale not in ci, "local_ci.sh still carries a literal marker set"
    assert stale not in disc, "test_discrimination.py still carries a literal marker set"


@pytest.mark.parametrize("marker", sorted(n for n, d in _declared_markers().items() if NON_AUTOMATIC.search(d)))
def test_a_marker_declared_non_automatic_is_excluded_by_the_gate(marker: str):
    """`manual` today. The parametrisation follows the declaration, so a new one is covered too."""
    markers = MARKER_FILE.read_text().strip()
    assert f"not {marker}" in markers, (
        f"pytest.ini declares `{marker}` as running in no automatic tier, but the gate's marker "
        f"set does not exclude it: {markers!r}"
    )


def test_the_discrimination_script_and_the_gate_agree_at_runtime():
    """Not a string comparison: the value the script actually holds, against the file."""
    import sys

    sys.path.insert(0, str(REPO / "scripts"))
    import test_discrimination as td

    assert MARKER_FILE.read_text().strip() == td.MARKERS
