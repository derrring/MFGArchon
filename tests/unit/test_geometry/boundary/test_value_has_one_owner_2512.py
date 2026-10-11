"""Row B2's structural guard: a boundary value is decoded by one owner (#2512, row B2).

The owner is `types.boundary_value`, reached through `types.segment_value` for a segment. This guard fails if a
module decodes a value itself (audit session ruling, 2026-10-09, #2512 comment 6073690080).

**Population.** Every module under `mfgarchon/`, except `applicator_graph.py`, whose node values are row N2.

**A value read** is ``<x>.value`` or ``<x>.default_value``, or ``getattr(<x>, "value"[, default])``, as D17-D19
read it; ``<x>.bc_type.value`` and the like are enum reads, not values. A name assigned in the same function from a
value read, or from a conditional or boolean expression over one (D10's ``seg.value if ... else 0.0``), carries it.

**A raw decode** of a value read or a carried name is: a call of it; a numeric conversion of it as the first argument
(``float``, ``int``, ``complex``, ``np.asarray``, ``np.array``); or a store of it into an array or a mapping, as D12
wrote ``field[on_boundary] = bc_value``. These are the shapes the census's decoders D1-D21 use (#2512 comment
6104689754). On b10b9e65, with no allowance, the guard flags 16 of them; the other five are the blind spots below,
and D9, which decodes no ``.value`` itself but reads through D1.

**Allowed**, each by file and function, so a moved line does not reopen the question: the readers row B2's second
half routes, each removed from the list then; and an attribute named ``value`` that is not a boundary value. The
owner needs no allowance: it receives the value as a parameter. An allowance that covers no site fails the guard,
so a routed reader cannot leave its name behind for the next one.

**Known blind spots**, as the other structural guards state theirs (audit session ruling, 2026-10-11: the census's
shapes are the criterion, since an AST guard cannot be complete over Python syntax):
- the census's second predicate, ``if callable(v): v(...)`` on a value passed into a helper (D5, D16, D22-D26), and a
  raw value returned or copied for a caller to decode (D14, D21): the dataflow is within one function; row B2's
  second half extends the guard to the second predicate before the row is met;
- names carried through a second name (``f = seg.value; g = f``), an annotated assignment, a walrus, or tuple
  unpacking;
- arithmetic on a value (``seg.value * h``), ``np.full``'s fill value, ``np.float64(...)``, ``.value.__call__``, and
  module-level code.
None of these is a shape a census decoder uses.
"""

from __future__ import annotations

import ast
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from tests.structural_guard import FUNCTIONS, REPO, own_nodes, planted_scan

if TYPE_CHECKING:
    from collections.abc import Mapping

PACKAGE = Path("mfgarchon")
EXCLUDED = {"mfgarchon/geometry/boundary/applicator_graph.py"}  # row N2
VALUE_ATTRS = frozenset({"value", "default_value"})
ENUM_HOLDERS = frozenset({"bc_type", "default_bc", "type", "kind", "math_type"})
CONVERT = frozenset({"float", "int", "complex", "asarray", "array"})
ALLOWED = {
    # Row B2's second half routes these (#2512 comment 6104689754); remove each from this list when it does.
    ("mfgarchon/alg/numerical/fp_solvers/fp_fdm_time_stepping.py", "_get_dirichlet_value_at_point"): (
        "B2b: FP-FDM reads its Dirichlet at t = 0 (#2535)"
    ),
    # An attribute named `value` that is not a boundary value: a cvxpy Problem's optimal value.
    ("mfgarchon/alg/numerical/gfdm_components/joint_socp.py", "solve_joint_socp_at_stencil"): "cvxpy",
    ("mfgarchon/alg/numerical/gfdm_components/joint_socp.py", "solve_relaxed_joint_socp_at_stencil"): "cvxpy",
}


def _is_value_read(node: ast.AST) -> bool:
    if isinstance(node, ast.Attribute):
        return (
            node.attr in VALUE_ATTRS
            and isinstance(node.ctx, ast.Load)
            and not (isinstance(node.value, ast.Attribute) and node.value.attr in ENUM_HOLDERS)
        )
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "getattr"
        and len(node.args) >= 2
        and isinstance(node.args[1], ast.Constant)
        and node.args[1].value in VALUE_ATTRS
    )


def _over(node: ast.AST, test) -> bool:
    """``node`` passes ``test``, or is a conditional or boolean expression one of whose results does."""
    if isinstance(node, ast.IfExp):
        return _over(node.body, test) or _over(node.orelse, test)
    if isinstance(node, ast.BoolOp):
        return any(_over(value, test) for value in node.values)
    return test(node)


def raw_decodes(scope: ast.AST) -> list[int]:
    """Lines in ``scope`` that call, convert or store a boundary value, directly or through a carried name."""
    nodes = list(own_nodes(scope))
    carried = {
        target.id
        for node in nodes
        if isinstance(node, ast.Assign) and _over(node.value, _is_value_read)
        for target in node.targets
        if isinstance(target, ast.Name)
    }

    def valueish(node: ast.AST) -> bool:
        return _over(node, lambda n: _is_value_read(n) or (isinstance(n, ast.Name) and n.id in carried))

    lines = []
    for node in nodes:
        if isinstance(node, ast.Call):
            callee = node.func.id if isinstance(node.func, ast.Name) else getattr(node.func, "attr", None)
            if valueish(node.func) or (callee in CONVERT and node.args and valueish(node.args[0])):
                lines.append(node.lineno)
        elif isinstance(node, ast.Assign) and any(isinstance(t, ast.Subscript) for t in node.targets):
            if valueish(node.value):
                lines.append(node.lineno)
    return lines


def scan(root: Path, allowed: Mapping[tuple[str, str], str] = ALLOWED) -> tuple[list[str], list[str]]:
    """``(population, sites)`` for the package under ``root``; a site is ``path:line: in function``."""
    population, sites = [], []
    for path in sorted((root / PACKAGE).rglob("*.py")):
        rel = path.relative_to(root).as_posix()
        if rel in EXCLUDED:
            continue
        population.append(rel)
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, FUNCTIONS):
                name = getattr(node, "name", "<lambda>")
                if (rel, name) not in allowed:
                    sites.extend(f"{rel}:{line}: in {name}" for line in raw_decodes(node))
    return population, sorted(set(sites))


@pytest.fixture(scope="module")
def package_copy(tmp_path_factory) -> Path:
    """A copy of `mfgarchon/` to plant rivals in; the tests add a module and remove it."""
    root = tmp_path_factory.mktemp("b2_guard")
    shutil.copytree(REPO / PACKAGE, root / PACKAGE, ignore=shutil.ignore_patterns("__pycache__"))
    return root


def test_every_boundary_value_is_decoded_by_the_owner():
    population, sites = scan(REPO)
    assert any(path.endswith("geometry/boundary/types.py") for path in population), population
    assert len(population) > 100, f"the population looks truncated: {len(population)} modules"
    assert sites == [], "a boundary value decoded outside `types.boundary_value` (#2512 row B2):\n" + "\n".join(sites)


def test_every_allowance_still_covers_a_site():
    """An allowance for a reader that has been routed, or is gone, would hide the next one given its name."""
    _, sites = scan(REPO, allowed={})
    covered = {(site.split(":")[0], site.rsplit(" in ", 1)[1]) for site in sites}
    assert set(ALLOWED) <= covered, sorted(set(ALLOWED) - covered)


@pytest.mark.parametrize(
    ("planted", "why"),
    [
        ("def _rival(segment, time):\n    return segment.value(time)\n", "a call"),
        ("def _rival(bc):\n    return float(bc.default_value)\n", "a conversion of the default"),
        ("def _rival(segment, time):\n    g = segment.value\n    return g(time)\n", "a call through a name"),
        ("def _rival(segment):\n    g = getattr(segment, 'value', 0.0)\n    return float(g)\n", "D17-D19's getattr"),
        (
            "def _rival(seg):\n    value = seg.value if not callable(seg.value) else 0.0\n    return float(value)\n",
            "D10's conditional expression",
        ),
        ("def _rival(seg):\n    value = seg.value or 0.0\n    return float(value)\n", "a boolean expression"),
        (
            "def _rival(bc, field, idx):\n    v = bc.default_value\n    field[idx] = v / 2.0 if v else v\n",
            "D12's store into an array",
        ),
    ],
    ids=["call", "convert-default", "through-a-name", "getattr", "ifexp", "boolop", "store"],
)
def test_a_planted_rival_is_found(package_copy, planted, why):
    target = package_copy / PACKAGE / "geometry" / "boundary" / "_planted_rival.py"
    sites = planted_scan(scan, package_copy, target, planted)
    assert any("_planted_rival.py" in site for site in sites), (why, sites)


def test_an_enum_value_is_not_a_boundary_value(package_copy):
    target = package_copy / PACKAGE / "geometry" / "boundary" / "_planted_rival.py"
    planted = "def _not_a_rival(segment):\n    return float(segment.bc_type.value == 'dirichlet')\n"
    assert not any("_planted_rival.py" in site for site in planted_scan(scan, package_copy, target, planted))
