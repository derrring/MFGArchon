"""Row B9's structural guard: which BC object is "the BC" of a problem is decided by one owner (#2530, #2512 row B9).

The owner is `MFGProblem.get_boundary_conditions` (`ConditionsMixin`, `core/mfg_components.py`); a solver handed its
own BC uses that one (`BaseMFGSolver._lookup_boundary_conditions`). Plan: #2530 comment 6133804095.

**Rival forms**, read by AST over every module:
- a load of a shared channel: ``<..>components.boundary_conditions``, ``<..>geometry.boundary_conditions``,
  ``<..>problem.boundary_conditions``, a ``<..>geometry.get_boundary_conditions()`` call, a
  ``get_boundary_handler()`` call -- the holder matched by the end of its name, so ``network_geometry`` and
  ``hjb_geometry`` count;
- an operator built on the grid's BC instead of the caller's: ``get_gradient_operator``, ``get_laplacian_operator``
  or ``value_gradient`` with no ``bc=`` (or 4th positional for ``value_gradient``).

At 79fae339 these matched 11 members, by function: the five resolution chains (the base lookup, FP-FDM, FP-FVM, the
particle solver, and the owner's own order, which ranked a components BC above a mesh's) and six operator calls in
HJB-FDM (four), HJB-SL and FP-SL. After B9, none.

**Allowed, each by file and function, with its site count**, so a moved line does not reopen the question and a new
rival inside an allowed function still fails. An allowance that covers no site fails. These are found at both shas
and are the class line's control.

**Known blind spots**, as the other structural guards state theirs:
- a holder named otherwise (``g = problem.geometry; g.boundary_conditions``), or ``grid.get_boundary_conditions()``,
  which the grid's own operators read and which this cannot tell from a solver's read;
- a computed ``getattr``, ``vars(...)``, ``__dict__``;
- an operator class built directly (``LaplacianOperator(..., bc=None)``);
- an allowed function whose read is swapped for another of the same count.
"""

from __future__ import annotations

import ast
import shutil
from typing import TYPE_CHECKING

import pytest

from tests.structural_guard import REPO

if TYPE_CHECKING:
    from pathlib import Path

CHANNELS = ("geometry", "components", "problem")
OPERATORS = frozenset({"get_gradient_operator", "get_laplacian_operator", "value_gradient"})

_PHI = "pads or differentiates a level-set function, a geometry field no solver reaches"
#: (file, function) -> (site count, why it is allowed).
ALLOWED = {
    ("mfgarchon/core/mfg_components.py", "get_boundary_conditions"): (4, "the owner"),
    ("mfgarchon/core/mfg_components.py", "using_resolved_bc"): (
        3,
        "a dependent: sets the provider-resolved BC on the channels that declare it, and restores them",
    ),
    ("mfgarchon/core/mfg_problem.py", "__init__"): (
        2,
        "D2's hand-over: the components BC becomes the legacy grid's declaration; it picks nothing",
    ),
    ("mfgarchon/core/stochastic/stochastic_problem.py", "create_conditional_problem"): (
        1,
        "a dependent: copies the components BC into the conditional problem, whose owner reads it",
    ),
    ("mfgarchon/extensions/topology.py", "__init__"): (1, "a graph's node BCs, which the graph geometry owns (#1471)"),
    ("mfgarchon/operators/differential/gradient.py", "value_gradient"): (
        1,
        "the operators' callee: the BC its caller hands it, else the grid's",
    ),
    ("mfgarchon/operators/differential/interface_jump.py", "_compute_gradient_jump_nd"): (
        1,
        "takes no BC from any caller, and no solver reaches it",
    ),
    ("mfgarchon/geometry/level_set/core.py", "get_normal"): (1, _PHI),
    ("mfgarchon/geometry/level_set/curvature.py", "compute_curvature"): (1, _PHI),
    ("mfgarchon/geometry/level_set/reinitialization.py", "_godunov_gradient_magnitude"): (1, _PHI),
    ("mfgarchon/geometry/level_set/reinitialization.py", "<module>"): (2, "the module's __main__ demonstration"),
}


def _holder(node: ast.AST) -> str:
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Name):
        return node.id
    return ""


def _kind(node: ast.AST) -> str | None:
    if isinstance(node, ast.Attribute) and node.attr == "boundary_conditions" and isinstance(node.ctx, ast.Load):
        holder = _holder(node.value)
        if holder.endswith(CHANNELS):
            return f"{holder}.boundary_conditions"
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
        if node.func.attr == "get_boundary_conditions" and _holder(node.func.value).endswith("geometry"):
            return "geometry.get_boundary_conditions()"
        if node.func.attr == "get_boundary_handler":
            return "get_boundary_handler()"
    if isinstance(node, ast.Call):
        name = _holder(node.func)
        if name in OPERATORS and not any(keyword.arg == "bc" for keyword in node.keywords):
            if not (name == "value_gradient" and len(node.args) >= 4):
                return f"{name} without bc="
    return None


def scan(root: Path) -> dict[tuple[str, str], list[str]]:
    """Rival sites under ``root``, grouped by (file, innermost enclosing function)."""
    sites: dict[tuple[str, str], list[str]] = {}
    for path in sorted((root / "mfgarchon").rglob("*.py")):
        rel = path.relative_to(root).as_posix()
        tree = ast.parse(path.read_text(), rel)
        function_of: dict[int, str] = {}
        for fn in ast.walk(tree):  # outer functions are visited first, so an inner one overwrites
            if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for node in ast.walk(fn):
                    function_of[id(node)] = fn.name
        for node in ast.walk(tree):
            kind = _kind(node)
            if kind:
                key = (rel, function_of.get(id(node), "<module>"))
                sites.setdefault(key, []).append(f"{rel}:{node.lineno}: {kind}")
    return sites


def _violations(sites: dict[tuple[str, str], list[str]]) -> list[str]:
    out = []
    for key, found in sites.items():
        if key not in ALLOWED:
            out += found
        elif len(found) != ALLOWED[key][0]:
            out.append(f"{key}: {len(found)} rival sites, {ALLOWED[key][0]} allowed: {found}")
    return out


def test_the_bc_is_resolved_only_through_the_owner():
    violations = _violations(scan(REPO))
    assert not violations, "a rival to the problem's BC owner (#2530):\n" + "\n".join(violations)


def test_every_allowance_still_covers_a_site():
    sites = scan(REPO)
    assert [key for key in ALLOWED if key not in sites] == []


@pytest.fixture(scope="module")
def package_copy(tmp_path_factory):
    root = tmp_path_factory.mktemp("b9_guard")
    shutil.copytree(REPO / "mfgarchon", root / "mfgarchon", ignore=shutil.ignore_patterns("__pycache__"))
    return root


PLANTED = {
    "a components read in an FP solver": (
        "mfgarchon/alg/numerical/fp_solvers/fp_fdm.py",
        "def _rival(problem):\n    return problem.components.boundary_conditions\n",
    ),
    "a geometry read in an HJB solver": (
        "mfgarchon/alg/numerical/hjb_solvers/hjb_semi_lagrangian.py",
        "def _rival(problem):\n    return problem.geometry.get_boundary_conditions()\n",
    ),
    "a geometry attribute in the base lookup's module": (
        "mfgarchon/alg/base_solver.py",
        "def _rival(self):\n    return self.problem.geometry.boundary_conditions\n",
    ),
    "the boundary handler": (
        "mfgarchon/alg/numerical/fp_solvers/fp_fvm.py",
        "def _rival(problem):\n    return problem.geometry.get_boundary_handler()\n",
    ),
    "an operator on the grid's BC, in a solver": (
        "mfgarchon/alg/numerical/hjb_solvers/hjb_fdm.py",
        "def _rival(self):\n    return self.problem.geometry.get_gradient_operator(scheme='central')\n",
    ),
    "value_gradient on the grid's BC, in a solver": (
        "mfgarchon/alg/numerical/fp_solvers/fp_particle.py",
        "def _rival(geometry, U):\n    return value_gradient(geometry, U, 0.0)\n",
    ),
    "an operator on the grid's BC, outside alg/": (
        "mfgarchon/operators/differential/divergence.py",
        "def _rival(grid):\n    return grid.get_laplacian_operator(order=2)\n",
    ),
}


@pytest.mark.parametrize("form", list(PLANTED))
def test_a_planted_rival_is_found(package_copy, form):
    rel, planted = PLANTED[form]
    target = package_copy / rel
    original = target.read_text()
    try:
        target.write_text(original + "\n\n" + planted)
        violations = _violations(scan(package_copy))
    finally:
        target.write_text(original)
    assert any(rel in v for v in violations), (form, violations)


def test_a_new_read_inside_an_allowed_function_is_found(package_copy):
    """The allowance pins its site count, so a second components read in the conditional problem fails too."""
    rel = "mfgarchon/core/stochastic/stochastic_problem.py"
    target = package_copy / rel
    original = target.read_text()
    anchor = 'component_kwargs["boundary_conditions"] = self.components.boundary_conditions\n'
    assert original.count(anchor) == 1
    try:
        target.write_text(
            original.replace(anchor, anchor + "                _ = self.components.boundary_conditions\n")
        )
        violations = _violations(scan(package_copy))
    finally:
        target.write_text(original)
    assert any("create_conditional_problem" in v for v in violations), violations
