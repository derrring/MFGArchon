"""Issue #2062: every name the writer can emit must round-trip through `LogAnalyzer`.

The oracle is a ROUND TRIP, computed independently of the regex: write a record through the real
`MFGFormatter` — the object that decides the on-disk shape — and read the file back through the
real `LogAnalyzer`. Neither side is stubbed, so the test cannot pass by agreeing with itself about
a format.

What it caught: `([^-]+?)` for the logger name could not match a hyphen by construction, so any
line whose logger name carried one was skipped and `parse_log_file` merely reported a lower count.
Every workflow logger is named `mfg_workflow.<uuid>`, and a uuid always has hyphens, so those files
parsed to zero entries — a silent narrowing on the read side, indistinguishable from a quiet run.
"""

from __future__ import annotations

import logging

import pytest

from mfgarchon.utils.mfg_logging import LogAnalyzer
from mfgarchon.utils.mfg_logging.logger import MFGFormatter

# (label, logger name). The first two are the POSITIVE CONTROL: they contain no hyphen and parsed
# correctly before the fix, so if the harness itself were broken they would fail too and the
# hyphenated rows would prove nothing.
NAMES = [
    ("control, no hyphen", "MFGSolver"),
    ("control, underscores", "mfg_workflow_manager"),
    ("workflow logger, uuid", "mfg_workflow.48151c9d-a6d5-4d95-9fbc-124b7fe75a67"),
    ("user-chosen name", "my-experiment"),
    ("hyphen at the end", "trailing-"),
    ("dotted package path", "mfgarchon.alg.numerical.fp_solvers.fp_gfdm"),
]

LEVELS = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


def _write(tmp_path, name, level, message, include_location):
    """One record, through the writer that owns the format."""
    path = tmp_path / "probe.log"
    record = logging.LogRecord(name, getattr(logging, level), "/x/y.py", 42, message, None, None)
    record.funcName = "f"
    path.write_text(MFGFormatter(use_colors=False, include_location=include_location).format(record) + "\n")
    return path


@pytest.mark.parametrize(("label", "name"), NAMES, ids=[n[0] for n in NAMES])
@pytest.mark.parametrize("level", LEVELS)
def test_a_name_the_writer_emits_is_a_name_the_reader_parses(tmp_path, label, name, level):
    path = _write(tmp_path, name, level, "a message", include_location=False)

    analyzer = LogAnalyzer(str(path))
    analyzer.parse_log_file()

    assert len(analyzer.entries) == 1, f"{label}: {name!r} at {level} parsed to {len(analyzer.entries)} entries"
    entry = analyzer.entries[0]
    assert entry["logger"] == name, "the padding must be stripped and the name returned whole"
    assert entry["level"] == level
    assert entry["message"] == "a message"


@pytest.mark.parametrize("include_location", [False, True])
def test_an_empty_message_still_parses(tmp_path, include_location):
    """The second hole #2062 named. Before the fix the line ended in `-` with nothing after it,
    which a literal trailing space in the pattern could not match."""
    path = _write(tmp_path, "my-experiment", "INFO", "", include_location=include_location)

    analyzer = LogAnalyzer(str(path))
    analyzer.parse_log_file()

    assert len(analyzer.entries) == 1
    assert analyzer.entries[0]["message"] == ""
    if include_location:
        assert analyzer.entries[0]["location"] == "y.py:42"


@pytest.mark.parametrize(
    "message",
    ["  leading spaces", "\tleading tab", "has - a dash", "ends with dash -", "[brackets] inside"],
    ids=["leading-spaces", "leading-tab", "inner-dash", "trailing-dash", "brackets"],
)
def test_the_message_is_returned_verbatim(tmp_path, message):
    """The level separator is `\\s?` and not `\\s+` for this reason: measured, a greedy `\\s+`
    eats the message's own leading whitespace, and on an empty message under
    include_location=True it swallows the `[location]` field into the message."""
    path = _write(tmp_path, "my-experiment", "INFO", message, include_location=True)

    analyzer = LogAnalyzer(str(path))
    analyzer.parse_log_file()

    assert len(analyzer.entries) == 1
    assert analyzer.entries[0]["message"] == message, "the message was altered in transit"
    assert analyzer.entries[0]["location"] == "y.py:42", "the location was swallowed into the message"
