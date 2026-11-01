"""Row N1's structural guard: the graph and continuum BC alphabets do not cross (#2292, #2512 row N1).

A graph node's condition is a `GraphBCType`, a continuum segment's a `BCType`. The two share the value strings
"dirichlet" and "neumann" while being different members, so a reader that decodes one alphabet with the other's
members, or either by its value string, takes a condition of the other kind for one of its own, or drops it: #2292's
`NodeBC(bc_type=BCType.DIRICHLET)` matched none of `GraphApplicator.apply`'s `GraphBCType` comparisons and was
skipped. The acceptance gate refuses the crossing where a condition is handed over
(`test_applicator_bc_type_contract_1948.py`); this guard holds each reader to one alphabet, so a condition that
reaches a reader some other way is still not read as the other kind.

**Rival forms**, read by AST over every module of the package, in a comparison or a ``match`` case:
- a `BCType` member in the graph owner, `applicator_graph.py`, the one module that decodes a `GraphBCType`;
- a `GraphBCType` member anywhere else;
- a string against an expression that loads ``.bc_type``: ``bc.bc_type == "absorbing"`` is never true of a member,
  and ``bc.bc_type.value == "dirichlet"`` is true of both alphabets.

**Allowed, each by file and function, with its site count**, as row B1's guard allows: a continuum reader whose
segments `BCSegment` admits only with a `BCType` (#2292), so its string reads one alphabet.

**Known blind spots**:
- a decode that is not a comparison: a mapping keyed by a member or by a value string, as the owner's `_WRITES` is;
- a type held under another name (``kind = bc.bc_type``, ``default_bc.value``), or a member reached through an alias
  or by value (``GraphBCType("source")``);
- ``isinstance``, a refusal's test rather than a decode, is not read.
"""

from __future__ import annotations

import ast
import shutil
from typing import TYPE_CHECKING

import pytest

from tests.structural_guard import REPO

if TYPE_CHECKING:
    from pathlib import Path

OWNER = "mfgarchon/geometry/boundary/applicator_graph.py"

#: (file, function) -> (site count, why it is allowed).
ALLOWED = {
    ("mfgarchon/alg/numerical/fp_solvers/fp_fdm_time_stepping.py", "_declares_dirichlet"): (
        1,
        "a continuum segment's type, which BCSegment admits only as a BCType (#2292)",
    ),
}


def _members(node: ast.AST, enum: str) -> list[str]:
    """Every ``<enum>.<member>`` anywhere in the operand ``node``: itself, a display's element, or deeper, inside a
    call such as ``frozenset({...})``, which a display-only reading missed (#2292 review 2, S7)."""
    return [
        f"{enum}.{item.attr}"
        for item in ast.walk(node)
        if isinstance(item, ast.Attribute) and isinstance(item.value, ast.Name) and item.value.id == enum
    ]


def _has_string(node: ast.AST) -> bool:
    items = node.elts if isinstance(node, (ast.Tuple, ast.List, ast.Set)) else [node]
    return any(isinstance(item, ast.Constant) and isinstance(item.value, str) for item in items)


def _loads_bc_type(node: ast.AST) -> bool:
    return any(isinstance(sub, ast.Attribute) and sub.attr == "bc_type" for sub in ast.walk(node))


def _comparisons(tree: ast.AST):
    """Each comparison as its operands: a ``Compare``, and a ``match`` as its subject against each case value."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Compare):
            yield node, [node.left, *node.comparators]
        elif isinstance(node, ast.Match):
            for case in node.cases:
                for pattern in ast.walk(case.pattern):
                    if isinstance(pattern, ast.MatchValue):
                        yield pattern, [node.subject, pattern.value]


def _kinds(rel: str, operands: list[ast.AST]) -> list[str]:
    kinds = []
    if rel == OWNER:
        kinds += [f"{member} in the graph owner" for op in operands for member in _members(op, "BCType")]
    else:
        kinds += [f"{member} outside the graph owner" for op in operands for member in _members(op, "GraphBCType")]
    if any(_has_string(op) for op in operands) and any(_loads_bc_type(op) for op in operands):
        kinds.append("a .bc_type read by a string")
    return kinds


def scan(root: Path) -> tuple[dict[tuple[str, str], list[str]], int]:
    """Rival sites under ``root``, grouped by (file, innermost enclosing function), and the number of comparisons
    with a `GraphBCType` member in the owner, which is where the graph alphabet is decoded."""
    sites: dict[tuple[str, str], list[str]] = {}
    owner_decodes = 0
    for path in sorted((root / "mfgarchon").rglob("*.py")):
        rel = path.relative_to(root).as_posix()
        tree = ast.parse(path.read_text(), rel)
        function_of: dict[int, str] = {}
        for fn in ast.walk(tree):  # outer functions are visited first, so an inner one overwrites
            if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for node in ast.walk(fn):
                    function_of[id(node)] = fn.name
        for node, operands in _comparisons(tree):
            if rel == OWNER and any(_members(op, "GraphBCType") for op in operands):
                owner_decodes += 1
            for kind in _kinds(rel, operands):
                key = (rel, function_of.get(id(node), "<module>"))
                sites.setdefault(key, []).append(f"{rel}:{node.lineno}: {kind}")
    return sites, owner_decodes


def _violations(sites: dict[tuple[str, str], list[str]]) -> list[str]:
    out = []
    for key, found in sites.items():
        if key not in ALLOWED:
            out += found
        elif len(found) != ALLOWED[key][0]:
            out.append(f"{key}: {len(found)} rival sites, {ALLOWED[key][0]} allowed: {found}")
    return out


def test_each_alphabet_is_decoded_by_its_own_readers():
    sites, owner_decodes = scan(REPO)
    violations = _violations(sites)
    assert not violations, "a BC alphabet crossing (#2512 row N1):\n" + "\n".join(violations)
    # The control: the owner decodes the graph alphabet here. Were it to move, nothing above could fire for it.
    assert owner_decodes > 0, f"{OWNER} compares no GraphBCType member; the graph owner has moved"


def test_every_allowance_still_covers_a_site():
    sites, _ = scan(REPO)
    assert [key for key in ALLOWED if key not in sites] == []


@pytest.fixture(scope="module")
def package_copy(tmp_path_factory):
    root = tmp_path_factory.mktemp("n1_guard")
    shutil.copytree(REPO / "mfgarchon", root / "mfgarchon", ignore=shutil.ignore_patterns("__pycache__"))
    return root


PLANTED = {
    "a BCType compare in the graph owner": (OWNER, "def _rival(bc):\n    return bc.bc_type == BCType.DIRICHLET\n"),
    "a string compare in the graph owner": (OWNER, "def _rival(bc):\n    return bc.bc_type.value == 'dirichlet'\n"),
    "a string compare in a network solver": (
        "mfgarchon/alg/numerical/network_solvers/fp_network.py",
        "def _rival(node_bc):\n    return node_bc.bc_type == 'absorbing'\n",
    ),
    "graph members compared in a network solver": (
        "mfgarchon/alg/numerical/network_solvers/hjb_network.py",
        "def _rival(node_bc):\n    return node_bc.bc_type in (GraphBCType.SOURCE, GraphBCType.ABSORBING)\n",
    ),
    "a graph member inside a call operand": (
        "mfgarchon/alg/numerical/network_solvers/fp_network.py",
        "def _rival(node_bc):\n    return node_bc.bc_type in frozenset({GraphBCType.SOURCE})\n",
    ),
    "a graph member as a match case in a continuum reader": (
        "mfgarchon/geometry/boundary/bc_utils.py",
        "def _rival(seg):\n    match seg.bc_type:\n        case GraphBCType.ABSORBING:\n            return 1\n    return 0\n",
    ),
}


@pytest.mark.parametrize("form", list(PLANTED))
def test_a_planted_rival_is_found(package_copy, form):
    rel, planted = PLANTED[form]
    target = package_copy / rel
    original = target.read_text()
    try:
        target.write_text(original + "\n\n" + planted)
        violations = _violations(scan(package_copy)[0])
    finally:
        target.write_text(original)
    assert any(rel in v for v in violations), (form, violations)


def test_a_new_rival_inside_an_allowed_function_is_found(package_copy):
    """The allowance pins its site count, so a second string read in the allowed function fails too."""
    rel = "mfgarchon/alg/numerical/fp_solvers/fp_fdm_time_stepping.py"
    target = package_copy / rel
    original = target.read_text()
    anchor = '    default = getattr(boundary_conditions, "default_bc", None)\n'
    assert original.count(anchor) == 1
    try:
        planted = '    _ = [seg for seg in () if seg.bc_type.name == "NEUMANN"]\n'
        target.write_text(original.replace(anchor, planted + anchor))
        violations = _violations(scan(package_copy)[0])
    finally:
        target.write_text(original)
    assert any("_declares_dirichlet" in v for v in violations), violations
