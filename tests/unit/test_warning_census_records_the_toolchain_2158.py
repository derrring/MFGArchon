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
import platform
import sys
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "_census_conftest", Path(__file__).resolve().parent.parent / "conftest.py"
)
_CONFTEST = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_CONFTEST)


class _WarningReport:
    """`terminalreporter.stats["warnings"]` holds these; the hook reads `.message`.

    Captured from a live run: the real stats keys are `['', 'failed', 'passed', 'skipped',
    'warnings', 'xfailed']`. A stub with no `warnings` key leaves ~55 lines unexecuted -- the regex,
    the site-packages/stdlib path normalisation, the digit normalisation, the 40-character
    truncation, the kind field, the occurrence count. Seven mutations of that block survived before
    this class existed.
    """

    def __init__(self, message):
        self.message = message


class _Reporter:
    """The only thing the hook reads off the terminal reporter."""

    def __init__(self, stats):
        self.stats = stats


#: Four outcome classes with DISTINCT counts. `{"passed": [None] * 7}` left the other three at
#: `stats.get(k, []) -> []` either way, so dropping `skipped`, or dropping three of the four, from
#: the `tests_run` sum survived -- a mutation this file's own PR table listed as killed.
_STATS = {"passed": [None] * 11, "failed": [None] * 3, "xfailed": [None] * 2, "skipped": [None] * 5}
_TESTS_RUN = 21


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

    _CONFTEST.pytest_terminal_summary(_Reporter(dict(_STATS)), 0, None)

    payload = json.loads(census.read_text())
    # Written out rather than read from `TOOLCHAIN_NAMES`, which is the thing under test: keyed on
    # the constant, this assertion followed it and deleting `osqp` from the list survived. Each
    # name here has a warrant recorded beside the constant; changing the set must be deliberate.
    assert set(payload["toolchain"]) == {
        "python",
        "platform",
        "pytest",
        "numpy",
        "scipy",
        "scikit-fem",
        "torch",
        "cvxpy",
        "osqp",
        "numpy-blas",
        "scipy-blas",
    }
    # Both halves of `platform`. Asserting only `startswith(sys.platform)` left the machine half --
    # the axis that actually selects the wheel and hence the BLAS -- covered by nothing: dropping
    # `platform.machine()` passed the whole suite.
    assert payload["toolchain"]["platform"] == f"{sys.platform}-{platform.machine()}"
    assert payload["toolchain"]["numpy-blas"] is not None, "the real numpy must report a BLAS"
    assert payload["toolchain"]["python"] == ".".join(str(n) for n in sys.version_info[:3])
    assert payload["toolchain"]["pytest"] == pytest.__version__
    # A corrupt record is recorded as such and must never reach the artifact silently.
    assert "unreadable" not in payload["toolchain"].values()
    assert payload["tests_run"] == _TESTS_RUN


@pytest.mark.parametrize(
    ("mode", "metadata"),
    [
        ("no Version header", "Metadata-Version: 2.1\nName: zzprobe\n"),
        ("empty Version value", "Metadata-Version: 2.1\nName: zzprobe\nVersion: \n"),
        ("empty METADATA", ""),
        ("no METADATA file", None),
    ],
)
def test_a_record_that_will_not_say_its_version_is_not_an_absent_package(tmp_path, monkeypatch, mode, metadata):
    """`importlib.metadata` swallows the read error and hands back an empty message.

    Four of five corruption modes therefore returned None -- the value that means "not installed" --
    and the report prints `numpy 2.4.6 -> <absent>` for an installed, working numpy, under the note
    saying its tests did not run. Measured before the fix; `chmod 000`, the case `unreadable` is
    named for, was among them.
    """
    record = tmp_path / "zzprobe-1.0.dist-info"
    record.mkdir()
    if metadata is not None:
        (record / "METADATA").write_text(metadata)
    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.metadata.MetadataPathFinder.invalidate_caches()

    assert _CONFTEST._installed_version("zzprobe") == "unreadable", mode
    # The control: with no record at all the answer must still be None, or "unreadable" would just
    # be the new name for absent.
    assert _CONFTEST._installed_version("zzprobe-no-such-record") is None


def test_the_census_records_the_identities_it_was_given(tmp_path, monkeypatch):
    """The extraction block -- regex, path normalisation, digit normalisation, truncation, kind.

    None of it ran before: the stub had no `warnings` key, so `stats.get("warnings", [])` was empty
    and ~55 lines were dead. Seven mutations of that block survived, including emptying the identity
    list outright.
    """
    census = tmp_path / "census.json"
    monkeypatch.setenv("MFGARCHON_WARNING_CENSUS", str(census))
    monkeypatch.delenv("PYTEST_XDIST_WORKER", raising=False)
    root = Path(_CONFTEST.__file__).resolve().parent.parent

    stats = dict(_STATS)
    stats["warnings"] = [
        _WarningReport(f"{root}/mfgarchon/probe.py:12: DeprecationWarning: used 3 times over 45 items\n  x = 1\n"),
        _WarningReport(f"{root}/mfgarchon/probe.py:99: DeprecationWarning: used 7 times over 12 items\n  y = 2\n"),
        _WarningReport(f"{root}/mfgarchon/other.py:5: RuntimeWarning: divide by zero encountered in log\n"),
        # Longer than the 40-character key. Without one, widening the truncation to 60 changes
        # nothing observable and the mutation survives -- measured. 40 is load-bearing: the
        # docstring beside it argues at length for 40 over 60 on stability grounds.
        _WarningReport(
            f"{root}/mfgarchon/long.py:1: UserWarning: "
            "this message is deliberately longer than forty characters so the cut is visible\n"
        ),
    ]
    _CONFTEST.pytest_terminal_summary(_Reporter(stats), 0, None)
    payload = json.loads(census.read_text())

    keys = sorted(payload["identities"])
    assert len(keys) == 3, f"digits must normalise so the two probe.py warnings collapse: {keys}"
    assert payload["occurrences"] == 4
    # Line numbers are deliberately absent from the key -- they move under any edit.
    assert not any(":12" in k or ":99" in k for k in keys)
    fields = [k.split("\t") for k in keys]
    assert [f[0] for f in fields] == ["mfgarchon/long.py", "mfgarchon/other.py", "mfgarchon/probe.py"], fields
    assert sorted(f[1] for f in fields) == ["DeprecationWarning", "RuntimeWarning", "UserWarning"]
    by_file = {f[0]: f[2] for f in fields}
    assert by_file["mfgarchon/probe.py"] == "used N times over N items"
    # The cut, pinned exactly. A `<= 40` bound passes for any wider truncation when every message
    # in the fixture is already shorter than the limit.
    assert by_file["mfgarchon/long.py"] == "this message is deliberately longer than"
    assert len(by_file["mfgarchon/long.py"]) == 40


class _FakeModule:
    """Just enough of numpy/scipy's surface for `_blas_of`."""

    def __init__(self, payload):
        self._payload = payload

    def show_config(self, mode="stdout"):
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


def _cfg(name):
    return {"Build Dependencies": {"blas": {"name": name}}}


def test_the_blas_name_is_lowercased():
    """numpy and scipy disagree on the case of the SAME implementation.

    Measured on one PyPI environment: numpy reports `accelerate`, scipy reports `Accelerate`.
    Compared raw across a re-baseline, that reads as a package moving when nothing did.
    """
    assert _CONFTEST._blas_of(_FakeModule(_cfg("Accelerate"))) == "accelerate"
    assert _CONFTEST._blas_of(_FakeModule(_cfg("accelerate"))) == "accelerate"
    assert _CONFTEST._blas_of(_FakeModule(_cfg("OpenBLAS"))) == "openblas"


@pytest.mark.parametrize(
    "payload",
    [
        AttributeError("no show_config"),
        KeyError("Build Dependencies"),
        {"Build Dependencies": {}},
        _cfg(None),
    ],
    ids=["no-show-config", "no-key", "no-blas-entry", "empty-name"],
)
def test_a_module_that_will_not_say_gives_none_rather_than_killing_the_census(payload):
    """This runs inside `pytest_terminal_summary`. An exception here loses the whole census."""
    assert _CONFTEST._blas_of(_FakeModule(payload)) is None


def test_numpy_blas_reads_numpy_and_scipy_blas_reads_scipy(tmp_path, monkeypatch):
    """Nothing bound the two fields to their modules: swapping the right-hand sides passed.

    Driven THROUGH the hook, not by calling `_blas_of` twice. The swap is in the payload assembly,
    so a test that exercises the function cannot see it -- measured, the first version of this case
    called `_blas_of(numpy)` and `_blas_of(scipy)` directly and the mutation survived it. Testing
    the function is not testing the wiring.

    The stated warrant for reading both is that they are built independently and can differ. In the
    one environment where they visibly do -- PyPI, numpy `accelerate` against scipy `Accelerate` --
    lowercasing erases the difference, so a swap is invisible there too. The case that would expose
    it is the case the pair exists for.
    """
    import numpy
    import scipy

    monkeypatch.setattr(numpy, "show_config", lambda mode="stdout": _cfg("NUMPYMARK"))
    monkeypatch.setattr(scipy, "show_config", lambda mode="stdout": _cfg("SCIPYMARK"))
    census = tmp_path / "census.json"
    monkeypatch.setenv("MFGARCHON_WARNING_CENSUS", str(census))
    monkeypatch.delenv("PYTEST_XDIST_WORKER", raising=False)

    _CONFTEST.pytest_terminal_summary(_Reporter(dict(_STATS)), 0, None)
    toolchain = json.loads(census.read_text())["toolchain"]
    assert toolchain["numpy-blas"] == "numpymark"
    assert toolchain["scipy-blas"] == "scipymark"


def test_a_non_string_blas_name_does_not_kill_the_census():
    """`.lower()` outside the try turns this into an AttributeError raised out of the summary hook.

    Bytes is the worse one: it returns successfully and `json.dumps` then raises past the payload's
    `except OSError`, eating pytest's whole summary.
    """
    for payload in (_cfg(1), _cfg(b"openblas"), _cfg(["openblas"]), _cfg({"a": 1})):
        assert _CONFTEST._blas_of(_FakeModule(payload)) is None


def test_the_real_numpy_reports_something():
    """Positive control. Without it every case above passes on a reader that always returns None."""
    import numpy

    assert _CONFTEST._blas_of(numpy) is not None


def test_the_hook_writes_nothing_without_the_environment_variable(tmp_path, monkeypatch):
    """The control for the case above: it must be the env var doing the work, not the stub."""
    census = tmp_path / "census.json"
    monkeypatch.delenv("MFGARCHON_WARNING_CENSUS", raising=False)
    _CONFTEST.pytest_terminal_summary(_Reporter(dict(_STATS)), 0, None)
    assert not census.exists()
    assert not os.environ.get("MFGARCHON_WARNING_CENSUS")
