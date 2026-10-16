"""Row B1's structural guard: which segment governs a boundary location is decided by one owner (#2512).

The owner is `BoundaryConditions.segment_for_face` / `segment_for_corner` / `segment_for_point`, in `conditions.py`,
with the field-level tests in `types.py` (plan: #2512 comment 6124387645).

**Rival forms**, from the plan's row fill, read by AST over every module but the owner's two:
- a load of a placement field: ``.boundary``, ``.face``, ``.region``, ``.region_name``, ``.sdf_region``,
  ``.normal_direction``, ``.priority``;
- ``getattr(<x>, "<placement field>")``;
- a ``matches_point(`` call;
- a uniform-only shortcut, ``segments[0]``, on an attribute or a local name;
- a face name built by an f-string, ``f"{axis}_min"`` and the like (#2628).

**Allowed, each by file and function, with its site count**, so a moved line does not reopen the question and a new
rival inside an allowed function still fails. An allowance that covers no site fails, so a function moved onto the
owner cannot leave its name behind. Two kinds:
- a reader the later stages move onto the owner: B1b (FEM), B1c (GFDM, meshless), B1d (particle), B1e (the rest);
- a site that matches a rival form and reads no segment: another object's ``region`` mask, a table of region or
  index names, a metric's column name.
After B1a the FDM family has no allowance at all (scan: 41 rival sites in it at 1052ec62, 0 after).

**Known blind spots**, as the other structural guards state theirs:
- a computed ``getattr`` name, ``vars(seg)``, ``seg.__dict__``;
- a BC copied on attach, or mutated after its first resolve: the owner memoises covers per object;
- a face name built without an f-string (``"x" + "_min"``, ``"%s_min" % axis``, ``str.format``);
- an allowed function whose rival is swapped for another of the same count.
"""

from __future__ import annotations

import ast
import re
import shutil
from typing import TYPE_CHECKING

import pytest

from tests.structural_guard import REPO

if TYPE_CHECKING:
    from pathlib import Path

OWNER = {"mfgarchon/geometry/boundary/conditions.py", "mfgarchon/geometry/boundary/types.py"}
PLACEMENT = frozenset({"boundary", "face", "region", "region_name", "sdf_region", "normal_direction", "priority"})
FACE_NAME = re.compile(r"\}_(min|max)\b|_\{(side|lo|hi|s|end)\}")

_NOT_A_SEGMENT = "matches a rival form and reads no segment"
#: (file, function) -> (site count, why it is allowed).
ALLOWED = {
    ("mfgarchon/alg/base_solver.py", "get_domain_constraints"): (1, "B1e: one constraint per segment, from its region"),
    ("mfgarchon/alg/numerical/adjoint/diagnostics.py", "_get_boundary_indices"): (4, f"{_NOT_A_SEGMENT}: index keys"),
    ("mfgarchon/alg/numerical/adjoint/operators.py", "get_boundary_indices_from_geometry"): (
        4,
        f"{_NOT_A_SEGMENT}: index keys",
    ),
    ("mfgarchon/geometry/base.py", "get_boundary_regions"): (2, f"{_NOT_A_SEGMENT}: a geometry's region table"),
    ("mfgarchon/geometry/graph/network_geometry.py", "get_boundary_regions"): (2, f"{_NOT_A_SEGMENT}: region table"),
    ("mfgarchon/geometry/implicit/point_cloud.py", "get_boundary_regions"): (2, f"{_NOT_A_SEGMENT}: region table"),
    ("mfgarchon/geometry/boundary/constraints.py", "project"): (4, f"{_NOT_A_SEGMENT}: an obstacle's region mask"),
    ("mfgarchon/geometry/boundary/constraints.py", "is_feasible"): (5, f"{_NOT_A_SEGMENT}: an obstacle's region mask"),
    ("mfgarchon/geometry/boundary/constraints.py", "get_active_set"): (
        4,
        f"{_NOT_A_SEGMENT}: an obstacle's region mask",
    ),
    ("mfgarchon/geometry/boundary/constraints.py", "__repr__"): (4, f"{_NOT_A_SEGMENT}: an obstacle's region mask"),
    ("mfgarchon/utils/data/polars_integration.py", "analyze_parameter_effects"): (4, f"{_NOT_A_SEGMENT}: metric names"),
    ("mfgarchon/utils/numerical/flux_diagnostics.py", "_get_boundary_names"): (2, f"{_NOT_A_SEGMENT}: labels"),
}


def _kind(node: ast.AST) -> str | None:
    if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant) and node.slice.value == 0:
        holder = node.value
        if (isinstance(holder, ast.Attribute) and holder.attr == "segments") or (
            isinstance(holder, ast.Name) and holder.id == "segments"
        ):
            return "segments[0]"
    if isinstance(node, ast.Attribute) and node.attr in PLACEMENT and isinstance(node.ctx, ast.Load):
        return f".{node.attr}"
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "getattr":
        if len(node.args) >= 2 and isinstance(node.args[1], ast.Constant) and node.args[1].value in PLACEMENT:
            return f"getattr {node.args[1].value!r}"
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "matches_point":
        return "matches_point("
    if isinstance(node, ast.JoinedStr) and FACE_NAME.search(ast.unparse(node)):
        return "f-string face name"
    return None


def scan(root: Path) -> dict[tuple[str, str], list[str]]:
    """Rival sites under ``root``, outside the owner, grouped by (file, innermost enclosing function)."""
    sites: dict[tuple[str, str], list[str]] = {}
    for path in sorted((root / "mfgarchon").rglob("*.py")):
        rel = path.relative_to(root).as_posix()
        if rel in OWNER:
            continue
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


def test_placement_is_read_only_through_the_owner():
    violations = _violations(scan(REPO))
    assert not violations, "a rival to the face owner (#2512 row B1):\n" + "\n".join(violations)


def test_every_allowance_still_covers_a_site():
    sites = scan(REPO)
    assert [key for key in ALLOWED if key not in sites] == []


@pytest.fixture(scope="module")
def package_copy(tmp_path_factory):
    root = tmp_path_factory.mktemp("b1_guard")
    shutil.copytree(REPO / "mfgarchon", root / "mfgarchon", ignore=shutil.ignore_patterns("__pycache__"))
    return root


PLANTED = {
    "placement field, FDM family": (
        "mfgarchon/alg/numerical/hjb_solvers/hjb_fdm.py",
        "def _rival(seg):\n    return seg.region_name\n",
    ),
    "getattr, FDM family": (
        "mfgarchon/geometry/boundary/bc_utils.py",
        "def _rival(seg):\n    return getattr(seg, 'boundary', None)\n",
    ),
    "matches_point, FDM family": (
        "mfgarchon/alg/numerical/fp_solvers/fp_fdm_time_stepping.py",
        "def _rival(seg, x):\n    return seg.matches_point(x, None, None)\n",
    ),
    "face name, FDM family": (
        "mfgarchon/operators/differential/gradient.py",
        "def _rival(bc, axis):\n    return bc.get_bc_type_at_boundary(f'axis{axis}_min')\n",
    ),
    "uniform shortcut on a name": (
        "mfgarchon/utils/numerical/tensor_calculus.py",
        "def _rival(bc):\n    segments = bc.segments\n    return segments[0]\n",
    ),
    "a new reader in a family still allowed some": (
        "mfgarchon/alg/numerical/fem/bc_adapter.py",
        "def _rival(seg):\n    return seg.region_name\n",
    ),
    "a module with no allowance": ("mfgarchon/utils/drift_helpers.py", "def _rival(bc):\n    return bc.segments[0]\n"),
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


def test_a_new_rival_inside_an_allowed_function_is_found(package_copy):
    """The allowance pins its site count, so a second rival in an allowed function fails too."""
    rel = "mfgarchon/alg/base_solver.py"
    target = package_copy / rel
    original = target.read_text()
    anchor = "        # Handle segments if available\n"
    assert original.count(anchor) == 1
    try:
        target.write_text(original.replace(anchor, anchor + "        _ = [seg.priority for seg in ()]\n"))
        violations = _violations(scan(package_copy))
    finally:
        target.write_text(original)
    assert any("get_domain_constraints" in v for v in violations), violations
