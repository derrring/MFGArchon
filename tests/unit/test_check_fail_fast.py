"""Pinning tests for the fail-fast ratchet (scripts/check_fail_fast.py).

The ratchet is the CI mechanism that keeps `except Exception`, bare `except:`,
silent `pass` handlers and `hasattr()` from growing. It was previously regex-based
and therefore measured something other than what it claimed:

- every *bound* handler (``except Exception as e:``) was invisible -- 104 of 115
  real broad handlers, including the ones that swallow numerics;
- bound/multi-line silent-pass handlers were invisible -- 35 of 95;
- ``hasattr`` mentions inside docstrings and comments were counted as calls --
  40 spurious entries, which inflated the baseline and created headroom to add
  real violations without tripping CI.

Each test below fails if detection reverts to textual scanning.
"""

import importlib.util
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "check_fail_fast.py"


def _load_checker():
    spec = importlib.util.spec_from_file_location("check_fail_fast", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def checker():
    return _load_checker()


def _scan(checker, tmp_path: Path, source: str) -> dict[str, list[str]]:
    (tmp_path / "sample.py").write_text(source, encoding="utf-8")
    return checker.check_fail_fast_violations(str(tmp_path))


def _counts(checker, tmp_path: Path, source: str) -> dict[str, int]:
    return {category: len(items) for category, items in _scan(checker, tmp_path, source).items()}


# --- broad handlers -----------------------------------------------------------


@pytest.mark.parametrize(
    "handler",
    [
        "except Exception:",
        "except Exception as e:",  # regex-invisible: the bound form
        "except BaseException:",
        "except BaseException as exc:",
        "except (ValueError, Exception):",  # regex-invisible: tuple form
        "except (ValueError, Exception) as err:",
        "except err.Exception:",  # qualified form -- pins the ast.Attribute branch
        "except pkg.mod.BaseException as e:",
        "except (ValueError, err.Exception):",
    ],
)
def test_broad_handler_forms_all_counted(checker, tmp_path, handler):
    counts = _counts(checker, tmp_path, f"def f():\n    try:\n        g()\n    {handler}\n        h()\n")
    assert counts["broad_except"] == 1, f"{handler!r} was not counted as a broad handler"


def test_qualified_narrow_handler_is_not_broad(checker, tmp_path):
    """The Attribute branch must match on the attribute name, not merely be qualified."""
    source = "def f():\n    try:\n        g()\n    except np.linalg.LinAlgError as e:\n        h(e)\n"
    assert _counts(checker, tmp_path, source)["broad_except"] == 0


def test_narrow_handler_is_not_broad(checker, tmp_path):
    counts = _counts(checker, tmp_path, "def f():\n    try:\n        g()\n    except ValueError as e:\n        h(e)\n")
    assert counts["broad_except"] == 0


def test_multiline_broad_handler_counted(checker, tmp_path):
    source = "def f():\n    try:\n        g()\n    except (\n        ValueError,\n        Exception,\n    ) as e:\n        h(e)\n"
    assert _counts(checker, tmp_path, source)["broad_except"] == 1


# --- silent pass --------------------------------------------------------------


@pytest.mark.parametrize(
    "handler",
    [
        "except Exception:",
        "except ValueError as e:",  # regex-invisible: whitespace in the clause
        "except (KeyError, IndexError) as e:",
    ],
)
def test_silent_pass_counted_for_bound_and_tuple_handlers(checker, tmp_path, handler):
    counts = _counts(checker, tmp_path, f"def f():\n    try:\n        g()\n    {handler}\n        pass\n")
    assert counts["silent_pass"] == 1, f"silent pass under {handler!r} was not counted"


def test_handler_with_body_is_not_silent_pass(checker, tmp_path):
    source = "def f():\n    try:\n        g()\n    except ValueError:\n        log()\n        pass\n"
    assert _counts(checker, tmp_path, source)["silent_pass"] == 0


# --- bare except --------------------------------------------------------------


def test_bare_except_counted(checker, tmp_path):
    counts = _counts(checker, tmp_path, "def f():\n    try:\n        g()\n    except:\n        h()\n")
    assert counts["bare_except"] == 1
    assert counts["broad_except"] == 0, "a bare except must not double-count as broad"


# --- hasattr ------------------------------------------------------------------


def test_hasattr_in_docstring_and_comment_not_counted(checker, tmp_path):
    source = '''def f(obj):
    """Legacy code used hasattr(obj, "x") -- replaced by a Protocol.

    >>> y = obj.x if hasattr(obj, "x") else None
    """
    # hasattr(obj, "y") was removed in #543
    return obj.x
'''
    assert _counts(checker, tmp_path, source)["hasattr"] == 0, "prose mentioning hasattr was counted as a call"


def test_real_hasattr_call_counted(checker, tmp_path):
    assert _counts(checker, tmp_path, 'def f(o):\n    return hasattr(o, "x")\n')["hasattr"] == 1


def test_two_hasattr_calls_on_one_line_count_twice(checker, tmp_path):
    source = 'def f(o):\n    return hasattr(o, "x") and hasattr(o, "y")\n'
    assert _counts(checker, tmp_path, source)["hasattr"] == 2


def test_shadowed_hasattr_attribute_access_not_counted(checker, tmp_path):
    assert _counts(checker, tmp_path, "def f(mod, o):\n    return mod.hasattr(o)\n")["hasattr"] == 0


# --- reported locations -------------------------------------------------------


def test_reported_line_numbers_are_correct(checker, tmp_path):
    """The location is what a developer follows to fix the violation; pin it, not just the count."""
    source = (
        "def f(o):\n"  # 1
        "    try:\n"  # 2
        "        g()\n"  # 3
        "    except Exception as e:\n"  # 4  <- broad
        "        pass\n"  # 5  (silent pass, same handler)
        "\n"  # 6
        "\n"  # 7
        "def h(o):\n"  # 8
        '    return hasattr(o, "x")\n'  # 9  <- hasattr
    )
    results = _scan(checker, tmp_path, source)

    assert results["broad_except"][0].endswith("sample.py:4: Broad 'except Exception:'")
    assert results["silent_pass"][0].endswith("sample.py:4: Silent 'pass' in except block")
    assert results["hasattr"][0].endswith("sample.py:9: hasattr() call")


def test_violations_found_in_nested_and_async_scopes(checker, tmp_path):
    """Handlers are found wherever they sit, not only at module/function top level."""
    source = (
        "class C:\n"
        "    async def m(self, o):\n"
        "        try:\n"
        "            await g()\n"
        "        except Exception as e:\n"
        "            h(e)\n"
        "        return hasattr(o, 'x')\n"
    )
    counts = _counts(checker, tmp_path, source)
    assert counts["broad_except"] == 1
    assert counts["hasattr"] == 1


# --- unparseable input must not be silently reported as clean -----------------


def test_syntax_error_raises_rather_than_reporting_zero(checker, tmp_path):
    (tmp_path / "broken.py").write_text("def f(:\n", encoding="utf-8")
    with pytest.raises(SyntaxError):
        checker.check_fail_fast_violations(str(tmp_path))


# --- the shipped baseline must match reality ---------------------------------


def test_repo_baseline_is_current(checker):
    """The committed baseline must equal live counts, or the ratchet is measuring fiction.

    This fires in both directions on purpose. An inflated baseline is *headroom* --
    exactly defect 3 of #1629, where 40 phantom `hasattr` entries would have let 40
    real ones be added with CI green. So fixing a violation must come with a baseline
    regeneration in the same commit; the ratchet itself only prints an advisory.
    """
    import json

    repo_root = _SCRIPT.parent.parent
    baseline_path = repo_root / "scripts" / "fail_fast_baseline.json"
    baseline = json.loads(baseline_path.read_text())
    results = checker.check_fail_fast_violations(str(repo_root / "mfgarchon"))
    live = {category: len(items) for category, items in results.items()}

    assert live == baseline, (
        f"Committed fail-fast baseline is stale.\n"
        f"  committed: {baseline}\n"
        f"  live:      {live}\n"
        f"If you fixed violations (counts went down), regenerate it in the same commit:\n"
        f"  python scripts/check_fail_fast.py --path mfgarchon --write-baseline scripts/fail_fast_baseline.json"
    )
