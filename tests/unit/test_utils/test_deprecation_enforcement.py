"""
Test deprecation enforcement mechanisms.

Tests the @deprecated decorator and AST-based checker to ensure
the deprecation lifecycle policy is enforced correctly.

Created: 2026-01-20 (Issue #616)
Reference: docs/development/DEPRECATION_LIFECYCLE_POLICY.md
"""

from __future__ import annotations

import warnings

import pytest

from mfgarchon.utils.deprecation import (
    _removable_by_policy,
    audit_all_deprecations,
    check_removal_readiness,
    deprecated,
    deprecated_parameter,
    get_deprecated_parameters,
    get_deprecation_metadata,
)


def test_deprecated_decorator_issues_warning():
    """Verify @deprecated decorator issues DeprecationWarning."""

    @deprecated(
        since="v0.17.0",
        replacement="use new_function()",
    )
    def old_function():
        return "result"

    with pytest.warns(DeprecationWarning, match="old_function.*deprecated.*v0.17.0"):
        result = old_function()

    assert result == "result"


def test_deprecated_decorator_stores_metadata():
    """Verify @deprecated stores discoverable metadata."""

    @deprecated(
        since="v0.17.0",
        replacement="use new_function()",
        reason="Renamed for clarity",
        removal_blockers=["internal_usage", "equivalence_test"],
    )
    def old_function():
        return "result"

    meta = get_deprecation_metadata(old_function)

    assert meta is not None
    assert meta["since"] == "v0.17.0"
    assert meta["replacement"] == "use new_function()"
    assert meta["reason"] == "Renamed for clarity"
    assert meta["symbol"] == "old_function"
    assert "removal_blockers" in meta
    assert "internal_usage" in meta["removal_blockers"]
    assert "equivalence_test" in meta["removal_blockers"]


def test_deprecated_parameter_decorator():
    """Verify @deprecated_parameter marks parameters."""

    @deprecated_parameter(
        param_name="old_param",
        since="v0.17.0",
        replacement="new_param",
    )
    def my_function(new_param: str = "default", old_param: str | None = None):
        if old_param is not None:
            warnings.warn(
                "old_param is deprecated. Use new_param instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            new_param = old_param
        return new_param

    # Check metadata stored
    params = get_deprecated_parameters(my_function)
    assert len(params) == 1
    assert params[0]["param"] == "old_param"
    assert params[0]["since"] == "v0.17.0"
    assert params[0]["replacement"] == "new_param"

    # Check warning issued when using old parameter
    with pytest.warns(DeprecationWarning, match="old_param.*deprecated"):
        result = my_function(old_param="value")

    assert result == "value"

    # No warning when using new parameter
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # Turn warnings into errors
        result = my_function(new_param="value")

    assert result == "value"


def test_deprecated_function_redirects_correctly():
    """Verify deprecated function redirects to new implementation."""

    def new_function(x: int) -> int:
        return x * 2

    @deprecated(
        since="v0.17.0",
        replacement="use new_function()",
    )
    def old_function(x: int) -> int:
        # Deprecated function MUST redirect to new function
        return new_function(x)

    # Verify both give same result
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # Suppress warning for this test
        result_old = old_function(5)

    result_new = new_function(5)

    assert result_old == result_new == 10


def test_get_deprecation_metadata_returns_none_for_non_deprecated():
    """Verify get_deprecation_metadata returns None for regular functions."""

    def regular_function():
        return "result"

    meta = get_deprecation_metadata(regular_function)
    assert meta is None


def test_get_deprecated_parameters_returns_empty_for_non_deprecated():
    """Verify get_deprecated_parameters returns empty list for regular functions."""

    def regular_function(param: str = "default"):
        return param

    params = get_deprecated_parameters(regular_function)
    assert params == []


def test_multiple_deprecated_parameters():
    """Verify function can have multiple deprecated parameters."""

    @deprecated_parameter(
        param_name="old_param1",
        since="v0.16.0",
        replacement="new_param1",
    )
    @deprecated_parameter(
        param_name="old_param2",
        since="v0.17.0",
        replacement="new_param2",
    )
    def my_function(
        new_param1: str = "default1",
        new_param2: str = "default2",
        old_param1: str | None = None,
        old_param2: str | None = None,
    ):
        # Redirection logic
        if old_param1 is not None:
            new_param1 = old_param1
        if old_param2 is not None:
            new_param2 = old_param2
        return new_param1, new_param2

    params = get_deprecated_parameters(my_function)
    assert len(params) == 2
    # Note: Decorators apply bottom-up, so old_param2 is added first
    param_names = {p["param"] for p in params}
    assert "old_param1" in param_names
    assert "old_param2" in param_names


def test_removal_readiness_with_blockers():
    """Verify check_removal_readiness correctly evaluates blockers."""

    @deprecated(
        since="v0.17.0",
        replacement="use new_function()",
        removal_blockers=["internal_usage", "equivalence_test", "migration_docs"],
    )
    def old_function():
        return "result"

    # No blockers cleared - not ready
    status = check_removal_readiness(old_function, "v0.20.0", completed_blockers=[])
    assert not status["ready"]
    assert not status["blockers_cleared"]
    assert len(status["remaining_blockers"]) == 3

    # Some blockers cleared - still not ready
    status = check_removal_readiness(old_function, "v0.20.0", completed_blockers=["internal_usage"])
    assert not status["ready"]
    assert not status["blockers_cleared"]
    assert len(status["remaining_blockers"]) == 2

    # All blockers cleared - ready
    status = check_removal_readiness(
        old_function,
        "v0.20.0",
        completed_blockers=["internal_usage", "equivalence_test", "migration_docs"],
    )
    assert status["ready"]
    assert status["blockers_cleared"]
    assert len(status["remaining_blockers"]) == 0


def test_removal_readiness_non_deprecated():
    """Verify check_removal_readiness handles non-deprecated objects."""

    def regular_function():
        return "result"

    status = check_removal_readiness(regular_function, "v0.20.0")
    assert not status["ready"]
    assert "Not a deprecated object" in status["blocking_reasons"]


# ---------------------------------------------------------------------------
# Removal policy: 3 minor versions OR 6 months (the single source of truth).
# ---------------------------------------------------------------------------


def test_removal_policy_three_minor_versions():
    """Age-eligible once >=3 minor versions have elapsed (no date needed)."""
    eligible, reason = _removable_by_policy("v0.17.0", None, "v0.20.0")
    assert eligible
    assert ">= 3" in reason

    eligible, reason = _removable_by_policy("v0.18.0", None, "v0.20.0")
    assert not eligible  # only 2 minor versions, no date path
    assert "< 3" in reason


def test_removal_policy_major_aware_minor_diff():
    """A major bump counts as 100 minor steps, so v0.x -> v1.0 is age-eligible."""
    eligible, reason = _removable_by_policy("v0.19.0", None, "v1.0.0")
    assert eligible
    assert ">= 3" in reason


def test_removal_policy_six_month_date_path():
    """The date path makes a deprecation eligible even when too few versions elapsed."""
    # Same minor version (0 version steps) but an old date -> eligible via 6-month rule.
    eligible, reason = _removable_by_policy("v0.19.0", "2020-01-01", "v0.19.0")
    assert eligible
    assert "6 months" in reason

    # No date and too few versions -> not eligible, and the reason states the date is absent.
    eligible, reason = _removable_by_policy("v0.19.0", None, "v0.19.0")
    assert not eligible
    assert "no deprecated_on date" in reason


def test_check_removal_readiness_uses_date_path():
    """check_removal_readiness honors the 6-month date path through the decorator."""

    # removal_blockers=[] isolates the age logic from the blocker gate.
    @deprecated(since="v0.19.0", deprecated_on="2020-01-01", replacement="use new()", removal_blockers=[])
    def old_dated():
        return "result"

    # Version criterion alone would fail (0 minor steps), date criterion carries it.
    status = check_removal_readiness(old_dated, "v0.19.0", completed_blockers=[])
    assert status["minimum_age_met"]
    assert status["ready"]

    # Same age, but the default 3-item blocker checklist gates readiness until cleared.
    @deprecated(since="v0.19.0", deprecated_on="2020-01-01", replacement="use new()")
    def old_dated_blocked():
        return "result"

    gated = check_removal_readiness(old_dated_blocked, "v0.19.0", completed_blockers=[])
    assert gated["minimum_age_met"]
    assert not gated["ready"]  # age met, blockers remain


def test_deprecated_on_stored_in_metadata():
    """deprecated_on round-trips into the deprecation metadata."""

    @deprecated(since="v0.19.0", deprecated_on="2024-01-01", replacement="use new()")
    def g():
        return 1

    meta = get_deprecation_metadata(g)
    assert meta is not None
    assert meta["deprecated_on"] == "2024-01-01"


# ---------------------------------------------------------------------------
# deprecated_parameter: warn iff the user actually passed the parameter, not
# whenever its (non-None) default is present. Regression guard for the
# bound.apply_defaults() false-positive bug.
# ---------------------------------------------------------------------------


def test_deprecated_parameter_no_warn_on_default():
    """A non-None default must NOT trigger the deprecation warning by itself."""

    @deprecated_parameter(param_name="mode", since="v0.18.0", replacement="strategy")
    def f(strategy: str = "auto", mode: str = "auto") -> str:
        return strategy

    # User never passes `mode`; its default "auto" is non-None. Old code warned here.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert f(strategy="x") == "x"


def test_deprecated_parameter_warns_when_passed():
    """Passing the deprecated parameter (kw or positional) must warn."""

    @deprecated_parameter(param_name="mode", since="v0.18.0", replacement="strategy")
    def f(strategy: str = "auto", mode: str = "auto") -> str:
        return strategy

    with pytest.warns(DeprecationWarning, match="mode.*deprecated"):
        f(mode="auto")

    with pytest.warns(DeprecationWarning, match="mode.*deprecated"):
        f("x", "auto")  # mode supplied positionally


# ---------------------------------------------------------------------------
# audit_all_deprecations: delegates age to the policy, dedups, categorizes.
# ---------------------------------------------------------------------------


def test_audit_all_categorizes_and_delegates():
    """ready / not_ready / active partition via the shared policy + blocker check."""
    import types

    mod = types.ModuleType("fake_dep_module")

    @deprecated(since="v0.10.0", replacement="use new()", removal_blockers=[])
    def ancient():  # age-eligible, no blockers -> ready
        return 1

    @deprecated(since="v0.17.0", replacement="use new()", removal_blockers=["equivalence_test"])
    def blocked():  # age-eligible (3 minor), blocker remains -> not_ready
        return 2

    @deprecated(since="v0.20.0", replacement="use new()")
    def fresh():  # 0 minor steps, no date -> active
        return 3

    mod.ancient = ancient
    mod.blocked = blocked
    mod.fresh = fresh

    report = audit_all_deprecations(mod, current_version="v0.20.0", completed_blockers=[])
    names = {bucket: {it["name"] for it in report[bucket]} for bucket in ("ready", "not_ready", "active")}
    assert "ancient" in names["ready"]
    assert "blocked" in names["not_ready"]
    assert "fresh" in names["active"]

    # Clearing the blocker promotes `blocked` to ready.
    report2 = audit_all_deprecations(mod, current_version="v0.20.0", completed_blockers=["equivalence_test"])
    assert "blocked" in {it["name"] for it in report2["ready"]}


def test_audit_all_default_version_and_dedup():
    """Default current_version resolves; output keys are unique (dedup works)."""
    import mfgarchon

    # allow_incomplete: this asserts the dedup and version invariants, not that the walk saw the
    # whole tree, and without torch eighteen frozen-paradigm modules do not import. The census
    # completeness question belongs to scripts/check_internal_deprecation.py, which refuses.
    report = audit_all_deprecations(mfgarchon, allow_incomplete=True)  # no current_version -> installed
    all_items = report["ready"] + report["not_ready"] + report["active"]
    assert all_items, "expected at least one live deprecation in mfgarchon"
    # Every item carries the resolved version, identical across the run.
    versions = {it["current_version"] for it in all_items}
    assert len(versions) == 1
    # Dedup invariant: each (name, type, since) appears at most once across all buckets.
    keys = [(it.get("name"), it.get("type"), it.get("since")) for it in all_items]
    assert len(keys) == len(set(keys))


# ---------------------------------------------------------------------------
# The package walk: one broken subpackage must not decide what the count is.
# ---------------------------------------------------------------------------


@pytest.fixture
def broken_tree(tmp_path):
    """A package whose middle subpackage raises AttributeError at import.

    The three subpackages are named so the walk reaches them in this order, because the defect is
    positional: `pkgutil.walk_packages` re-raises anything that is not `ImportError`, so with no
    `onerror` the walk ENDS at `b_broken` and `c_last` is never yielded. That is not a hypothetical
    shape -- it is `alg/reinforcement/multi_population` without torch, where `nn = None` is used as
    a base class (#1773), and it cost 268 of 428 modules and 31 of 72 deprecations (#1713).
    """
    import importlib
    import sys

    root = tmp_path / "broken_pkg"
    (root / "a_first").mkdir(parents=True)
    (root / "b_broken").mkdir()
    (root / "c_last").mkdir()

    decorated = (
        "from mfgarchon.utils.deprecation import deprecated\n"
        "@deprecated(since='v0.10.0', replacement='use new_{n}()', removal_blockers=[])\n"
        "def old_{n}():\n"
        "    return '{n}'\n"
    )
    (root / "__init__.py").write_text("")
    (root / "a_first" / "__init__.py").write_text(decorated.format(n="a"))
    (root / "b_broken" / "__init__.py").write_text("nn = None\n\n\nclass Boom(nn.Module):\n    pass\n")
    (root / "b_broken" / "child.py").write_text(decorated.format(n="b"))
    (root / "c_last" / "__init__.py").write_text(decorated.format(n="c"))

    sys.path.insert(0, str(tmp_path))
    importlib.invalidate_caches()
    try:
        yield importlib.import_module("broken_pkg")
    finally:
        sys.path.remove(str(tmp_path))
        for name in [m for m in sys.modules if m == "broken_pkg" or m.startswith("broken_pkg.")]:
            del sys.modules[name]


def test_walk_survives_a_subpackage_that_raises(broken_tree):
    """Everything AFTER the broken subpackage is still scanned.

    Revert the `onerror` argument in `scan_deprecated` and this fails on `old_c`: that symbol is
    the mutation detector, not `old_a`, which the truncating version also finds.
    """
    from mfgarchon.utils.deprecation import scan_deprecated

    found = {item["name"] for item in scan_deprecated(broken_tree, allow_incomplete=True)}

    assert "old_a" in found, "the subpackage before the break should always have been found"
    assert "old_c" in found, "the walk stopped at the broken subpackage instead of continuing past it"
    assert "old_b" not in found, "b_broken/child.py is genuinely unreachable -- do not claim otherwise"


def test_scan_refuses_an_incomplete_tree_by_default(broken_tree):
    """The default is a refusal that names the module, not a smaller number."""
    from mfgarchon.utils.deprecation import IncompleteScanError, scan_deprecated

    with pytest.raises(IncompleteScanError) as excinfo:
        scan_deprecated(broken_tree)

    assert "broken_pkg.b_broken" in excinfo.value.unimportable
    assert "AttributeError" in excinfo.value.unimportable["broken_pkg.b_broken"]
    # The message has to carry the module name too: `.unimportable` is only read by a caller that
    # already knows to look, and a bare traceback is what sends someone to the baseline instead.
    assert "b_broken" in str(excinfo.value)


def test_audit_refuses_an_incomplete_tree_by_default(broken_tree):
    """The refusal is not bypassed by going through the audit wrapper."""
    from mfgarchon.utils.deprecation import IncompleteScanError

    with pytest.raises(IncompleteScanError):
        audit_all_deprecations(broken_tree, current_version="v0.20.0")

    report = audit_all_deprecations(broken_tree, current_version="v0.20.0", allow_incomplete=True)
    assert {"old_a", "old_c"} <= {it["name"] for bucket in report.values() for it in bucket}


def test_audit_keeps_every_module_a_symbol_was_found_in(tmp_path):
    """The dedup must not let one discovery site stand for all of them.

    `audit_all_deprecations` dedups on (name, type, since) and the survivor is whichever copy the
    walk reached first. A caller filtering by module -- the ratchet's live-vs-frozen scope -- would
    then be judging walk order: measured on the real package, one re-export of a live deprecation
    from `alg/neural/__init__.py` moved the count from 64 back to 63 and the ratchet went green
    over a symbol that had just been added.
    """
    import importlib
    import sys

    root = tmp_path / "reexport_pkg"
    (root / "home").mkdir(parents=True)
    (root / "elsewhere").mkdir()
    (root / "__init__.py").write_text("")
    (root / "home" / "__init__.py").write_text(
        "from mfgarchon.utils.deprecation import deprecated\n"
        "@deprecated(since='v0.10.0', replacement='use new()', removal_blockers=[])\n"
        "def shared():\n"
        "    return 1\n"
    )
    (root / "elsewhere" / "__init__.py").write_text("from reexport_pkg.home import shared  # noqa: F401\n")

    sys.path.insert(0, str(tmp_path))
    importlib.invalidate_caches()
    try:
        report = audit_all_deprecations(
            importlib.import_module("reexport_pkg"), current_version="v0.20.0", completed_blockers=[]
        )
    finally:
        sys.path.remove(str(tmp_path))
        for name in [m for m in sys.modules if m == "reexport_pkg" or m.startswith("reexport_pkg.")]:
            del sys.modules[name]

    entries = [it for bucket in report.values() for it in bucket if it["name"] == "shared"]
    assert len(entries) == 1, "the dedup should still collapse the re-export into one entry"
    assert set(entries[0]["modules"]) == {"reexport_pkg.home", "reexport_pkg.elsewhere"}, (
        "both discovery sites must survive the dedup, or a scope filter judges whichever came first"
    )


def test_scan_of_a_complete_tree_reports_no_holes(tmp_path):
    """Positive control: `unimportable` is empty when nothing is broken.

    Without this, every assertion above is satisfied by a scan that flags everything.
    """
    import importlib
    import sys

    from mfgarchon.utils.deprecation import scan_deprecated

    root = tmp_path / "sound_pkg"
    (root / "sub").mkdir(parents=True)
    (root / "__init__.py").write_text("")
    (root / "sub" / "__init__.py").write_text(
        "from mfgarchon.utils.deprecation import deprecated\n"
        "@deprecated(since='v0.10.0', replacement='use new()', removal_blockers=[])\n"
        "def old_sound():\n"
        "    return 1\n"
    )

    sys.path.insert(0, str(tmp_path))
    importlib.invalidate_caches()
    try:
        # No allow_incomplete: a sound tree must not need the escape hatch.
        assert "old_sound" in {item["name"] for item in scan_deprecated(importlib.import_module("sound_pkg"))}
    finally:
        sys.path.remove(str(tmp_path))
        for name in [m for m in sys.modules if m == "sound_pkg" or m.startswith("sound_pkg.")]:
            del sys.modules[name]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
