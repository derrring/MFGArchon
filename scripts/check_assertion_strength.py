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
# Frozen paradigms are out of scope by default (CLAUDE.md § FROZEN). These are TEST-tree names,
# not source paths: `alg/neural` and `alg/reinforcement` name the SOURCE layout and match ZERO
# files under `tests/`, so the first version of this filter excluded nothing while its comment and
# its own test both said otherwise -- and that test asserted the constant contains itself, which is
# the tautological shape this script exists to count. Found by review (#1905).
def _is_frozen(path: Path) -> bool:
    """Frozen membership has exactly one owner, and it is not this file.

    `scripts/check_frozen_areas.py` decides it by AST -- imports and string literals reaching
    `FROZEN_PACKAGES` -- with a docstring explaining at length why a name match is insufficient
    and citing a file it measurably missed. This module answered the same question by filename
    substring, and the two disagreed on six files:

        frozen by NAME   : 12        frozen by IMPORT : 14

        excluded by name but NOT frozen (2 files, 40 test functions)
            tests/unit/test_alg/test_rl_stub_metrics_1688.py            (4)
            tests/unit/test_utils/test_neural/test_normalization.py     (36)
        frozen by import but LEFT IN the denominator (4 files, 20 test functions)
            tests/unit/test_alg/test_adaptive_training_canonical_mode_1572.py   (5)
            tests/unit/test_alg/test_mean_field_rl_requires_pop_state_1508.py   (1)
            tests/unit/test_alg/test_solver_construction_smoke_887.py           (5)
            tests/unit/test_config/test_enum_conversions.py                     (9)

    36 of those were dropped from the denominator because a DIRECTORY in their path is spelled
    `test_neural`, while CLAUDE.md freezes `alg/neural` and `alg/reinforcement` only -- `utils/neural`
    is not frozen, and the commit that deletes both frozen packages leaves that file alive, which
    is independent confirmation. Found by re-review (#1905), which also noted that the previous
    repair had entrenched the wrong set by pinning it in a test.
    """
    return bool(_frozen_areas()._references(path))


def _frozen_areas():
    """Imported lazily and by path: `scripts/` is not a package, so a plain import would depend
    on how this script was invoked."""
    global _CFA
    if _CFA is None:
        import importlib.util

        spec = importlib.util.spec_from_file_location("check_frozen_areas", REPO / "scripts" / "check_frozen_areas.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _CFA = module
    return _CFA


_CFA = None


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
        if _is_frozen(f):
            continue
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


def main() -> int:
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
