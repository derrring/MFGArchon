"""The deprecation ratchet's SCOPE decision must fail when it is wrong, and nothing checked it.

`tests/unit/test_utils/test_deprecation_enforcement.py` pins the library half of this fix -- that
`audit_all_deprecations` keeps every module a symbol was found in. The script half, which turns
those modules into an in-scope/out-of-scope verdict, had no control at all. Measured: reverting
`in_scope()` to the exact bug independent review reported --

    live = [e for e in audited if not is_frozen(e["module"])]

-- left 912 tests green and the ratchet printing `63 / Matches baseline`, while a live deprecation
re-exported from a frozen package's `__init__` was invisible to it. A ratchet whose own scope rule
cannot go red is the defect this ratchet exists to catch, one level up.

Fabricated entries rather than the real package on purpose: the assertions then do not depend on
what `mfgarchon` happens to contain today, and they run without importing it.

Every module name is BUILT from the script's own `FROZEN`, never spelled out. That is single-source
hygiene, and it is also required: `check_frozen_areas.py` counts string literals naming a frozen
package, by design and for good reasons, so spelling them here would register as new tests against
a frozen prototype (CLAUDE.md) when nothing here imports or exercises one.

Shaped after `tests/unit/test_check_frozen_areas.py`, which pins its sibling ratchet's comparison
for the same reason.
"""

from __future__ import annotations

import importlib.util
import pathlib

import pytest

_SCRIPT = pathlib.Path(__file__).resolve().parents[2] / "scripts" / "check_internal_deprecation.py"


def _load():
    spec = importlib.util.spec_from_file_location("check_internal_deprecation", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Module scope, not a fixture: `parametrize` is evaluated at collection time and cannot read one.
CHECK = _load()
# Synthetic, not read from `CHECK.FROZEN`. The two paradigms that populated that tuple were
# deleted, so it is empty and unpacking it raises at COLLECTION time -- taking the whole file with
# it. What these tests exercise is `is_frozen`'s prefix rule, and that rule is worth testing
# whether or not anything is currently frozen; reading the ambient tuple only tied the rule's
# verification to the package list. Each test that needs the rule to fire patches FROZEN itself.
FROZEN_A = "mfgarchon.alg.a_frozen_paradigm"
FROZEN_B = "mfgarchon.alg.another_frozen_paradigm"
DEEP_FROZEN = FROZEN_B + ".algorithms.some_algorithm"
LIVE = "mfgarchon.utils.numerical"


@pytest.fixture(autouse=True)
def _frozen_scope(monkeypatch):
    """Two frozen paradigms exist, for the duration of each test in this file.

    `is_frozen` answers False for everything when `FROZEN` is empty, which is correct for the live
    package and useless for testing the prefix rule.
    """
    monkeypatch.setattr(CHECK, "FROZEN", (FROZEN_A, FROZEN_B))


def _entry(name: str, *modules: str) -> dict:
    """The fields `in_scope` reads. `module` is set to the FIRST site deliberately.

    That is what the real dedup leaves behind -- whichever copy the walk reached first -- and the
    frozen packages sort before every other top-level subpackage, so a frozen site arriving first
    is the ordinary case here, not a contrived one.
    """
    return {"name": name, "type": "function", "since": "v0.17.0", "module": modules[0], "modules": list(modules)}


def test_a_symbol_the_frozen_paradigms_re_export_is_still_in_scope():
    """The mutation detector: judging `module` alone drops this one and the ratchet goes green."""
    live, dropped = CHECK.in_scope([_entry("shared", FROZEN_A, LIVE)])

    assert [e["name"] for e in live] == ["shared"], (
        "a deprecation that also lives outside the frozen paradigms must be counted; "
        "dropping it is how a newly added symbol went invisible"
    )
    assert dropped == 0


def test_a_symbol_only_in_the_frozen_paradigms_is_out_of_scope():
    """Positive control: the exclusion must still exclude, or the fix is just 'stop filtering'."""
    live, dropped = CHECK.in_scope([_entry("frozen_only", FROZEN_A, DEEP_FROZEN)])

    assert live == []
    assert dropped == 1


def test_scope_splits_a_mixed_census_without_losing_any_entry():
    """`len(in) + dropped == len(audited)`: the out-of-scope count is reported, not swallowed."""
    audited = [
        _entry("live_only", LIVE),
        _entry("frozen_only", DEEP_FROZEN),
        _entry("re_exported", FROZEN_A, LIVE),
    ]
    live, dropped = CHECK.in_scope(audited)

    assert {e["name"] for e in live} == {"live_only", "re_exported"}
    assert dropped == 1
    assert len(live) + dropped == len(audited)


@pytest.mark.parametrize(
    ("module", "frozen"),
    [
        (FROZEN_A, True),
        (FROZEN_A + ".submodule", True),
        (FROZEN_B, True),
        (DEEP_FROZEN, True),
        # Siblings whose names merely START with a frozen one. `startswith` on the bare prefix
        # swallows these, and swallows them silently, which is the failure mode being guarded.
        (FROZEN_A + "_ops", False),
        (FROZEN_A + "x.y", False),
        (FROZEN_B + "_learning_utils", False),
        (LIVE, False),
        ("mfgarchon", False),
    ],
)
def test_is_frozen_matches_package_containment_not_string_prefix(module, frozen):
    assert CHECK.is_frozen(module) is frozen


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
