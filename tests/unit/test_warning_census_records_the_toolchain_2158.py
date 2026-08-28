"""The census writer is a guard's feeding path, and it had no test (#2158).

`scripts/check_warnings.py` reports what else moved when the warning identity set moves. Every
one of its cases builds the baseline and the census by hand, so the writer that produces them in
a real run -- `pytest_terminal_summary` in `tests/conftest.py` -- was covered by nothing: review
measured four mutations surviving, including deleting the `toolchain` key from the payload
outright, which makes the attribution silently cease to exist while every self-test stays green.

The duplicate-`.dist-info` case cannot be reached from an end-to-end run at all: it needs two
records for one name on `sys.path`, which is why the reader is a module-level function here
rather than a closure inside the hook.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "_census_conftest", Path(__file__).resolve().parent.parent / "conftest.py"
)
_CONFTEST = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_CONFTEST)


class _Reporter:
    """The only thing the hook reads off the terminal reporter."""

    def __init__(self, stats):
        self.stats = stats


def test_an_installed_distribution_resolves():
    """Positive control. Without it every assertion below passes on a query form that finds nothing."""
    assert _CONFTEST._installed_version("pytest") == pytest.__version__


def test_an_absent_distribution_is_none():
    assert _CONFTEST._installed_version("mfgarchon-no-such-distribution") is None


def test_two_dist_info_records_are_reported_as_ambiguous_rather_than_picked(tmp_path, monkeypatch):
    """The defect the reader exists for, constructed.

    `importlib.metadata.version` returns the first match in `sys.path` order and says nothing
    about the rest. A stale record beside a current one then records a version no code in the run
    used -- inventing a package move, or hiding a real one behind the stale pair.
    """
    for version in ("1.0", "2.0"):
        record = tmp_path / f"zzfake-{version}.dist-info"
        record.mkdir()
        (record / "METADATA").write_text(f"Metadata-Version: 2.1\nName: zzfake\nVersion: {version}\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.metadata.MetadataPathFinder.invalidate_caches()

    # The behaviour being defended against, asserted rather than assumed: the stdlib call picks
    # one of the two and reports it as fact.
    assert importlib.metadata.version("zzfake") in {"1.0", "2.0"}
    assert _CONFTEST._installed_version("zzfake") == "ambiguous:1.0|2.0"


def test_the_recorded_names_are_distribution_names_not_import_names():
    """`scikit-fem` imports as `skfem`, and the wrong one records null -- indistinguishable from absent."""
    skfem = pytest.importorskip("skfem")
    assert skfem is not None
    assert _CONFTEST._installed_version("scikit-fem") is not None
    assert _CONFTEST._installed_version("skfem") is None


def test_the_census_payload_carries_the_toolchain(tmp_path, monkeypatch):
    """End to end through the hook: the key exists, is populated, and holds one entry per name."""
    census = tmp_path / "census.json"
    monkeypatch.setenv("MFGARCHON_WARNING_CENSUS", str(census))
    monkeypatch.delenv("PYTEST_XDIST_WORKER", raising=False)

    _CONFTEST.pytest_terminal_summary(_Reporter({"passed": [None] * 7}), 0, None)

    payload = json.loads(census.read_text())
    # Written out rather than read from `TOOLCHAIN_NAMES`, which is the thing under test: keyed on
    # the constant, this assertion followed it and deleting `osqp` from the list survived. Each
    # name here has a warrant recorded beside the constant; changing the set must be deliberate.
    assert set(payload["toolchain"]) == {"python", "pytest", "numpy", "scipy", "scikit-fem", "torch", "cvxpy", "osqp"}
    assert payload["toolchain"]["python"] == ".".join(str(n) for n in sys.version_info[:3])
    assert payload["toolchain"]["pytest"] == pytest.__version__
    # A corrupt record is recorded as such and must never reach the artifact silently.
    assert "unreadable" not in payload["toolchain"].values()
    assert payload["tests_run"] == 7


def test_the_hook_writes_nothing_without_the_environment_variable(tmp_path, monkeypatch):
    """The control for the case above: it must be the env var doing the work, not the stub."""
    census = tmp_path / "census.json"
    monkeypatch.delenv("MFGARCHON_WARNING_CENSUS", raising=False)
    _CONFTEST.pytest_terminal_summary(_Reporter({"passed": [None] * 7}), 0, None)
    assert not census.exists()
    assert not os.environ.get("MFGARCHON_WARNING_CENSUS")
