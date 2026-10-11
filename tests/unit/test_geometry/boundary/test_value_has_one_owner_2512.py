"""Row B2's structural guard: a boundary value is decoded by one owner (#2512, row B2).

The owner is `types.boundary_value`, reached through `types.segment_value` for a segment. This guard fails if a
module decodes a value itself (audit session ruling, 2026-10-09, #2512 comment 6073690080).

**Population.** Every module under `mfgarchon/`, except `applicator_graph.py`, whose node values are row N2.

**A raw decode** is a call of ``<x>.value`` or ``<x>.default_value``, or a numeric conversion of one (``float``,
``int``, ``complex``, ``np.asarray``, ``np.array``, ``np.full``, ``np.full_like``), directly or through a name
assigned from one in the same function. ``<x>.bc_type.value`` and the like are enum reads, not values.

**Allowed**, each by file and function, so a moved line does not reopen the question: the readers row B2's second
half routes, each removed from the list then; and an attribute named ``value`` that is not a boundary value. The
owner needs no allowance: it receives the value as a parameter. An allowance that covers no site fails the guard,
so a routed reader cannot leave its name behind for the next one.

**Out of scope**, as the other structural guards say of their own shapes: the dataflow is within one function,
so a value passed to a helper that decodes it there is not followed. The census that found five such decoders by
a second predicate is #2512 comment 6104689754.
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
CONVERT = frozenset({"float", "int", "complex", "asarray", "array", "full", "full_like"})
ALLOWED = {
    # Row B2's second half routes these (#2512 comment 6104689754); remove each from this list when it does.
    ("mfgarchon/alg/numerical/fp_solvers/fp_fdm_time_stepping.py", "_get_dirichlet_value_at_point"): (
        "B2b: FP-FDM reads its Dirichlet at t = 0 (#2535)"
    ),
    # An attribute named `value` that is not a boundary value: a cvxpy Problem's optimal value.
    ("mfgarchon/alg/numerical/gfdm_components/joint_socp.py", "solve_joint_socp_at_stencil"): "cvxpy",
    ("mfgarchon/alg/numerical/gfdm_components/joint_socp.py", "solve_relaxed_joint_socp_at_stencil"): "cvxpy",
}


def _is_value_attr(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Attribute)
        and node.attr in VALUE_ATTRS
        and isinstance(node.ctx, ast.Load)
        and not (isinstance(node.value, ast.Attribute) and node.value.attr in ENUM_HOLDERS)
    )


def raw_decodes(scope: ast.AST) -> list[int]:
    """Lines in ``scope`` that call or convert a boundary value, directly or through a name assigned from one."""
    nodes = list(own_nodes(scope))
    carried = {
        target.id
        for node in nodes
        if isinstance(node, ast.Assign) and _is_value_attr(node.value)
        for target in node.targets
        if isinstance(target, ast.Name)
    }

    def valueish(node: ast.AST) -> bool:
        return _is_value_attr(node) or (isinstance(node, ast.Name) and node.id in carried)

    lines = []
    for node in nodes:
        if not isinstance(node, ast.Call):
            continue
        callee = node.func.id if isinstance(node.func, ast.Name) else getattr(node.func, "attr", None)
        if valueish(node.func) or (callee in CONVERT and node.args and valueish(node.args[0])):
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
    ],
    ids=["call", "convert-default", "through-a-name"],
)
def test_a_planted_rival_is_found(package_copy, planted, why):
    target = package_copy / PACKAGE / "geometry" / "boundary" / "_planted_rival.py"
    sites = planted_scan(scan, package_copy, target, planted)
    assert any("_planted_rival.py" in site for site in sites), (why, sites)


def test_an_enum_value_is_not_a_boundary_value(package_copy):
    target = package_copy / PACKAGE / "geometry" / "boundary" / "_planted_rival.py"
    planted = "def _not_a_rival(segment):\n    return float(segment.bc_type.value == 'dirichlet')\n"
    assert not any("_planted_rival.py" in site for site in planted_scan(scan, package_copy, target, planted))
