"""Row B3's structural guard: every FP solver reads the shared BC through one owner (#2512, row B3).

The owner is `BaseFPSolver._fp_view_of_shared` (`bc_utils.fp_view_of_shared_bc`); `get_boundary_conditions()`
ends in it, and a solver that resolves the shared BC through its own chain must pass the result to it. This
guard fails if one reads the shared BC and uses the raw value (audit session ruling, 2026-10-09, #2512
comment 6073690080).

**Population.** Every module under `mfgarchon/alg/` that defines a class deriving, transitively, from
`BaseFPSolver`.

**A shared read** is `<x>.boundary_conditions` loaded from anything but `self`, `<x>.get_boundary_conditions()`
called on anything but `self`, or any `get_boundary_handler()`. `self.boundary_conditions` is the solver's
own attribute, and `self.get_boundary_conditions()` is the owner path itself.

**The rule.** Each shared read must reach the owner's argument in the same function: inside it, or through
names and `self` attributes assigned from it, transitively. That is per read: a function that translates
one read and uses another raw is a rival. A read whose parent is a comparison (`... is None`) is a test of
the value, not a use of it.

**Out of scope**, as the C1 guard says of its own shapes: the dataflow is flow-insensitive and within one
function, so a read stored on `self` in one method and translated in another, a read passed through a
helper of its own, or a reassignment between the read and the owner is not followed.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from tests.structural_guard import ALG, FUNCTIONS, REPO, copy_alg, name_of, own_nodes, planted_scan

OWNER = "_fp_view_of_shared"


def _is_self(node: ast.AST) -> bool:
    return isinstance(node, ast.Name) and node.id == "self"


def _key(node: ast.AST) -> str | None:
    """A name, or a ``self`` attribute: what a value can be carried in within one function."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute) and _is_self(node.value):
        return "self." + node.attr
    return None


def shared_read(node: ast.AST) -> bool:
    if isinstance(node, ast.Attribute) and node.attr == "boundary_conditions" and isinstance(node.ctx, ast.Load):
        return not _is_self(node.value)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
        if node.func.attr == "get_boundary_conditions":
            return not _is_self(node.func.value)
        return node.func.attr == "get_boundary_handler"
    return False


def raw_reads(scope: ast.AST) -> list[int]:
    """Lines of the shared reads in ``scope`` that do not reach the owner's argument in ``scope``."""
    nodes = list(own_nodes(scope))
    parents = {child: node for node in nodes for child in ast.iter_child_nodes(node)}
    owner_args = [
        arg
        for node in nodes
        if isinstance(node, ast.Call) and name_of(node.func) == OWNER
        for arg in [*node.args, *(keyword.value for keyword in node.keywords)]
    ]
    inside_owner = {id(sub) for arg in owner_args for sub in ast.walk(arg)}
    owner_keys = {key for arg in owner_args for sub in ast.walk(arg) if (key := _key(sub))}
    assigns = [
        (node.targets if isinstance(node, ast.Assign) else [node.target], node.value)
        for node in nodes
        if isinstance(node, (ast.Assign, ast.AnnAssign)) and node.value is not None
    ]

    def carried_into(holds) -> set[str]:
        """The keys assigned from a value that contains a node ``holds`` accepts."""
        return {
            key
            for targets, value in assigns
            if any(holds(sub) for sub in ast.walk(value))
            for target in targets
            if (key := _key(target))
        }

    def reaches(read: ast.AST) -> bool:
        if id(read) in inside_owner:
            return True
        carried = carried_into(lambda sub: sub is read)
        frontier = set(carried)
        while frontier:
            if frontier & owner_keys:
                return True
            current = frozenset(frontier)
            frontier = carried_into(lambda sub, keys=current: _key(sub) in keys) - carried
            carried |= frontier
        return False

    return sorted(
        node.lineno
        for node in nodes
        if shared_read(node) and not isinstance(parents.get(node), ast.Compare) and not reaches(node)
    )


def _scopes(tree: ast.AST):
    """``(name, scope)`` for every function and lambda in ``tree``, nested ones included."""
    for node in ast.walk(tree):
        if isinstance(node, FUNCTIONS):
            yield getattr(node, "name", "<lambda>"), node


def fp_classes(trees: dict[Path, ast.Module]) -> set[str]:
    """``BaseFPSolver`` and every class deriving from it, transitively, across ``trees``."""
    names = {"BaseFPSolver"}
    changed = True
    while changed:
        changed = False
        for tree in trees.values():
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.ClassDef)
                    and node.name not in names
                    and any(name_of(b) in names for b in node.bases)
                ):
                    names.add(node.name)
                    changed = True
    return names


def scan(root: Path) -> tuple[list[str], list[str]]:
    """``(population, rival sites)`` for the tree at ``root``; a site is ``path:line: in function``."""
    trees = {path.relative_to(root): ast.parse(path.read_text()) for path in sorted((root / ALG).rglob("*.py"))}
    classes = fp_classes(trees)
    population, sites = [], []
    for rel, tree in trees.items():
        if not any(isinstance(node, ast.ClassDef) and node.name in classes for node in ast.walk(tree)):
            continue
        population.append(rel.as_posix())
        for name, scope in _scopes(tree):
            sites.extend(f"{rel.as_posix()}:{line}: in {name}" for line in raw_reads(scope))
    return population, sorted(sites)


@pytest.fixture(scope="module")
def alg_copy(tmp_path_factory) -> Path:
    """A copy of `mfgarchon/alg` to plant rivals in; the tests add files and restore what they edit."""
    return copy_alg(tmp_path_factory, "b3_guard")


def _planted_scan(alg_copy: Path, target: Path, planted: str) -> list[str]:
    return planted_scan(scan, alg_copy, target, planted)


def test_every_fp_solver_reads_the_shared_bc_through_its_owner():
    population, sites = scan(REPO)
    solvers = {
        "fp_fdm.py",
        "fp_fvm.py",
        "fp_gfdm.py",
        "fp_particle.py",
        "fp_semi_lagrangian_adjoint.py",
        "fp_fem_solver.py",
        "weak_form_fp_solver.py",
        "fp_network.py",
    }
    assert solvers <= {Path(p).name for p in population}, "the population no longer reaches the FP solvers"
    assert sites == [], (
        "each site reads the shared BC and uses it raw. Pass it to self._fp_view_of_shared, or read it through "
        "self.get_boundary_conditions() (#2512, row B3):\n" + "\n".join(sites)
    )


def test_the_rule_fires_on_real_code_outside_the_population():
    """The zero above is not a rule that cannot match. The base chain every solver shares,
    `BaseNumericalSolver._lookup_boundary_conditions`, reads the shared BC raw, and `BaseFPSolver`'s
    override passes its result to the owner; base_solver.py defines no FP solver, so it is outside the
    population."""
    tree = ast.parse((REPO / ALG / "base_solver.py").read_text())
    found = {name for name, scope in _scopes(tree) if raw_reads(scope)}
    assert "_lookup_boundary_conditions" in found, found


_PLANT = "class _Planted(FPFDMSolver):\n    def _planted(self):\n{body}"
_CLASS_LINE = 2


@pytest.mark.parametrize(
    ("body", "raw"),
    [
        # The ruling's control: one read translated, another used raw.
        (
            "raw = self.problem.geometry.boundary_conditions\n"
            "view = self._fp_view_of_shared(self.problem.components.boundary_conditions)\n"
            "return raw",
            [1],
        ),
        ("return self.problem.geometry.boundary_conditions", [1]),
        ("bc = self.problem.geometry.get_boundary_conditions()\nreturn bc", [1]),
        ("return self.problem.geometry.get_boundary_handler()", [1]),
        ("bc = self.problem.boundary_conditions\nreturn self._fp_view_of_shared(bc)", []),
        ("bc = self.problem.boundary_conditions\nview = bc\nreturn self._fp_view_of_shared(bc=view)", []),
        ("self.bc = self.problem.boundary_conditions\nself.bc = self._fp_view_of_shared(self.bc)", []),
        ("if self.problem.boundary_conditions is None:\n    return None\nreturn self.get_boundary_conditions()", []),
        ("return self.boundary_conditions", []),
    ],
    ids=[
        "translate-another-use-raw",
        "returned-raw",
        "geometry-accessor-raw",
        "handler",
        "translated",
        "translated-through-two-names",
        "translated-through-self",
        "a-test-of-the-value",
        "own-attribute",
    ],
)
def test_each_shape_is_judged(alg_copy, body, raw):
    """``raw`` lists the lines of ``body`` (1-based) that the guard must report, and nothing else."""
    module = ALG / "numerical" / "fp_solvers" / "fp_fdm.py"
    indented = "".join(f"        {line}\n" for line in body.splitlines())
    original_lines = (REPO / module).read_text().count("\n") + 1  # appended after the original and a blank line
    first = original_lines + 1 + _CLASS_LINE
    sites = _planted_scan(alg_copy, alg_copy / module, _PLANT.format(body=indented))
    assert sites == [f"{module.as_posix()}:{first + line}: in _planted" for line in raw], sites


def test_a_new_fp_solver_anywhere_in_alg_is_in_the_population(alg_copy):
    """The population predicate moves: a module that subclasses an FP solver is held to the owner wherever it lives."""
    rel = ALG / "planted_fp.py"
    raw = "    def _planted(self):\n        return self.problem.geometry.boundary_conditions\n"
    assert _planted_scan(alg_copy, alg_copy / rel, "class _NotFP:\n" + raw) == [], "not an FP solver: not B3's"
    sites = _planted_scan(alg_copy, alg_copy / rel, "class _FP(FPParticleSolver):\n" + raw)
    assert sites == [f"{rel.as_posix()}:3: in _planted"], sites
