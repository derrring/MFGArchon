#!/usr/bin/env python3
"""Count tests whose assertions a well-formed WRONG answer would satisfy.

Not "inert under mutation" -- that selects for *tests something else*, and every one of the five
tests #1715 named that way turned out to be a genuine cross-path pin. This selects structurally:
a test asserting only `is not None` / `isfinite` / `.shape` / `len` / `isinstance` cannot separate
a right answer from a wrong one of the right shape, for ANY input.

Reports; does not gate. The count is a review queue, not a delete list -- measured 2026-08-12, the
flagged set contains capability cells ("can this configuration run at all", a close-out `CLAUDE.md`
explicitly allows) and negative controls for fail-loud guards ("this input must NOT raise", without
which the guard could reject everything and its `pytest.raises` siblings would still pass). Both
are assertion-free by nature and both are worth keeping.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# Helpers that carry the assertion themselves. `warns` and `raises` matter: a test whose whole
# content is `with pytest.raises(...)` has no `assert` node and is not weak.
STRONG_HELPERS = (
    "assert_allclose",
    "assert_array_equal",
    "assert_array_almost_equal",
    "assert_almost_equal",
    "assert_array_less",
    "approx",
    "raises",
    "warns",
    "assert_frame_equal",
    "assert_series_equal",
)
WEAK_CALLS = {"isfinite", "isinstance", "len", "hasattr", "callable", "id", "type", "bool", "any", "all"}
# There is no frozen paradigm left to exclude. `alg/neural` and `alg/reinforcement` were the
# only two, and they were DELETED rather than frozen (CLAUDE.md; 23,118 lines, 60 files), taking
# `scripts/check_frozen_areas.py` -- the AST decider this module deferred to -- with them.
#
# The exclusion is retired rather than re-homed. Kept, it would need a new owner for a set that
# is now empty, and an empty exclusion reads as though it does something: exactly the inert-filter
# defect that #1905 was fixing when it introduced the deferral. The denominator is now every
# collected test function in the tree.
#
# History, because the number moves twice and both moves are real:
#   name-based filter   1037 of 5371 = 19.3%   (12 files by filename substring)
#   AST decider         1049 of 5393 = 19.5%   (14 files by import; 40 functions wrongly excluded
#                                               by the name filter came back, 20 wrongly kept left)
#   no exclusion        this run                (the remaining 145 functions come back)
# Found while merging: the deferral and the deletion of its owner were on two branches that
# text-merged with zero conflicts and a broken tree.


def _weak(node: ast.Assert) -> bool:
    t = node.test
    if isinstance(t, ast.Compare):
        if any(isinstance(o, (ast.Is, ast.IsNot)) for o in t.ops):
            return True
        src = ast.dump(t)
        structural = any(f"attr='{a}'" in src for a in ("shape", "dtype", "ndim"))
        return (structural or "func=Name(id='len'" in src) and not any(
            isinstance(o, (ast.Lt, ast.Gt, ast.LtE, ast.GtE)) for o in t.ops
        )
    if isinstance(t, ast.Call):
        name = getattr(t.func, "id", None) or getattr(t.func, "attr", None)
        return name in WEAK_CALLS
    if isinstance(t, ast.Name):
        return True
    if isinstance(t, ast.UnaryOp) and isinstance(t.op, ast.Not):
        # `assert not <closeness>(a, b)` is a SEPARATION assertion -- two things must DIFFER --
        # and it is the strongest class this repo has, not the weakest ("byte-identity is the
        # defect, not the pass"). Calling every `not` weak inverted that doctrine on 70 tests,
        # among them `test_coupling_affects_solution` and
        # `test_fp_velocity_consumes_cross_density_1071`. Only a bare `assert not x` is weak.
        # Found by review (#1905).
        inner = ast.dump(t.operand)
        return not any(
            f"attr='{h}'" in inner or f"id='{h}'" in inner
            for h in ("array_equal", "allclose", "isclose", "array_equiv", "approx", "array_almost_equal")
        )
    return False


def _collected_tests(tree: ast.Module):
    """Module-level and class-level test functions only.

    A `def test_helper()` nested INSIDE a test is not collected by pytest and must not be counted.
    Walking every FunctionDef counted them, which double-counted three files and inflated the total
    -- found while reading the output, which is the only reason this note exists.
    """
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test_"):
            yield node
        elif isinstance(node, ast.ClassDef):
            for sub in node.body:
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)) and sub.name.startswith("test_"):
                    yield sub


def scan(root: Path):
    weak, total = [], 0
    for f in sorted(root.rglob("test_*.py")):
        try:
            tree = ast.parse(f.read_text())
        except SyntaxError:
            continue
        for fn in _collected_tests(tree):
            total += 1
            dumped = ast.dump(fn)
            if any(f"attr='{h}'" in dumped or f"id='{h}'" in dumped for h in STRONG_HELPERS):
                continue
            asserts = [n for n in ast.walk(fn) if isinstance(n, ast.Assert)]
            if not asserts:
                weak.append((f, fn.name, "no assertion"))
            elif all(_weak(a) for a in asserts):
                weak.append((f, fn.name, f"{len(asserts)} weak"))
    return weak, total


_CONTROL_WEAK = """
def test_only_shape():
    result = compute()
    assert result is not None
    assert result.shape == (3, 3)
    assert np.all(np.isfinite(result))
"""

_CONTROL_STRONG = """
def test_the_value_is_right():
    result = compute()
    assert np.allclose(result, EXACT, atol=1e-12)
"""


def self_test() -> int:
    """Positive control, two-sided. A count is evidence only if the classifier still discriminates.

    One-sided is not enough in either direction here. A classifier that flagged EVERYTHING would
    satisfy "does it catch a weak test" while making the percentage meaningless, and one that had
    gone inert would report a falling number that reads as the suite improving -- which is the
    direction this ratchet is watched in.
    """
    import tempfile

    failures = []
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        (root / "test_weak_control.py").write_text(_CONTROL_WEAK)
        weak, total = scan(root)
        if total != 1:
            failures.append(f"  collection: saw {total} test functions in a file with exactly 1")
        if not weak:
            failures.append("  weak: did not flag a test asserting only shape/finiteness/not-None")

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        (root / "test_strong_control.py").write_text(_CONTROL_STRONG)
        weak, total = scan(root)
        if total != 1:
            failures.append(f"  collection: saw {total} test functions in a file with exactly 1")
        if weak:
            failures.append(f"  strong: flagged a test that compares against an exact value: {weak}")

    if failures:
        print("SELF-TEST FAILED -- the classifier no longer separates weak assertions from strong:")
        print("\n".join(failures))
        return 1
    print("self-test OK -- flags the weak control, leaves the strong one alone")
    return 0


def main() -> int:
    if "--self-test" in sys.argv:
        return self_test()

    weak, total = scan(REPO / "tests")
    print(
        f"assertion strength : {len(weak)} of {total} defined test functions assert only what a "
        f"well-formed WRONG answer satisfies = {100 * len(weak) / total:.1f}%"
    )
    print("                     (a review queue, not a delete list -- capability cells and")
    print("                      fail-loud negative controls are assertion-free by nature)")
    if "--list" in sys.argv:
        for f, name, why in weak:
            print(f"  {f.relative_to(REPO)}::{name}  [{why}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
