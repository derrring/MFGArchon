"""C1's structural guard: nothing in C1's population measures a sweep's change itself (#2512, #2555).

C1 governs the coupling-level outer `tolerance`: the one docs/user/CONVENTIONS.md § 7 names as the outer
loop's, taken by `MFGProblem.solve`, the coupling iterators and `PicardConfig`. Its owners are
`MFGProblem.spatial_measure` (the measure), `sweep_change` (the change) and `check_convergence_criteria`
(the verdict). #2570's behavioural pins show that today's six iterators use them. They cannot stop a new
consumer from computing the change itself, and this guard does (audit session ruling, 2026-10-09, #2512
comment 6073690080).

**Population.** Every module under `mfgarchon/alg/numerical/coupling/`, and any other module under
`mfgarchon/alg/` that references `PicardConfig`, `check_convergence_criteria` or `sweep_change`, so a new
consumer cannot escape by living elsewhere.

**Rivals in the population**, where a difference is a binary subtraction of two non-constant operands, or
a name assigned from one in the same function (a ratio minus 1 is a diagnostic, not a change between two
states):
- a ``norm`` of a difference;
- ``max``, ``amax`` or ``nanmax`` of ``abs`` of a difference, or ``abs(...).max()``;
- ``sum``, ``nansum`` or ``mean`` of a squared difference, as a call or a method, where squared is
  ``d**2``, ``d * d`` or ``square(d)``;
- ``d @ d``, ``dot(d, d)`` or ``vdot(d, d)``.

**Anywhere under** `mfgarchon/alg/`: a call to the deprecated `calculate_error` or
`calculate_l2_convergence_metrics`.

Other spellings, such as an L1 sum of ``abs(d)`` or a difference passed through a helper first, are out
of scope: this guard names the shapes it covers and does not claim more.

**One function is excluded, by name:** `AndersonAccelerator.update`. Its residual norm and its stagnation
test decide how to mix: restart, skip, safeguard. They do not decide whether the outer loop has converged,
which stays with `check_convergence_criteria`. A pin below holds the exclusion to that one function and
requires it to still hold a site the rules match.

**Not C1 by definition.** These are separate algorithms with their own parameters, not exemptions; both
are entries in ledger #2573:
- `HomotopyContinuation._corrector` stops on its own `corr_tol`.
- `PrimalDualMFGSolver` stops on an optimisation residual and its own density change.

Neither module is in the population, and both are rivals by shape, which the whole-`alg/` control below
shows.
"""

from __future__ import annotations

import ast
import shutil
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
ALG = Path("mfgarchon") / "alg"
COUPLING = ALG / "numerical" / "coupling"
OWNERS = frozenset({"PicardConfig", "check_convergence_criteria", "sweep_change"})
DEPRECATED = frozenset({"calculate_error", "calculate_l2_convergence_metrics"})
#: (module, qualified function name) pairs whose sites are not C1 rivals; see the module docstring.
EXCLUDED = frozenset({((COUPLING / "anderson_acceleration.py").as_posix(), "AndersonAccelerator.update")})
_FUNCTIONS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)


def _name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _references_an_owner(tree: ast.AST) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, (ast.Name, ast.Attribute)) and _name(node) in OWNERS:
            return True
        if isinstance(node, ast.ImportFrom) and any(alias.name in OWNERS for alias in node.names):
            return True
    return False


def _own_nodes(scope: ast.AST):
    """The nodes of ``scope`` itself: a nested function or lambda is its own scope and is not entered."""
    stack = list(ast.iter_child_nodes(scope))
    while stack:
        node = stack.pop()
        yield node
        if not isinstance(node, _FUNCTIONS):
            stack.extend(ast.iter_child_nodes(node))


class _Scope:
    """One function (or the module body) and the names it, or an enclosing function, assigns from a difference."""

    def __init__(self, node: ast.AST, inherited: frozenset[str] = frozenset()):
        self.node = node
        self.named: set[str] = set(inherited)
        for sub in _own_nodes(node):
            if isinstance(sub, ast.Assign) and self._is_inline_difference(sub.value):
                self.named.update(t.id for t in sub.targets if isinstance(t, ast.Name))
            elif isinstance(sub, ast.AnnAssign) and sub.value is not None and self._is_inline_difference(sub.value):
                if isinstance(sub.target, ast.Name):
                    self.named.add(sub.target.id)

    @staticmethod
    def _is_inline_difference(node: ast.AST) -> bool:
        return (
            isinstance(node, ast.BinOp)
            and isinstance(node.op, ast.Sub)
            and not isinstance(node.left, ast.Constant)
            and not isinstance(node.right, ast.Constant)
        )

    def difference(self, node: ast.AST) -> bool:
        return self._is_inline_difference(node) or (isinstance(node, ast.Name) and node.id in self.named)

    def abs_of_difference(self, node: ast.AST) -> bool:
        return (
            isinstance(node, ast.Call)
            and _name(node.func) in {"abs", "absolute"}
            and bool(node.args)
            and self.difference(node.args[0])
        )

    def squared_difference(self, node: ast.AST) -> bool:
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
            return self.difference(node.left) and isinstance(node.right, ast.Constant) and node.right.value == 2
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
            return self.difference(node.left) and self.difference(node.right)
        return (
            isinstance(node, ast.Call)
            and _name(node.func) == "square"
            and bool(node.args)
            and self.difference(node.args[0])
        )

    def rivals(self) -> set[tuple[int, str]]:
        found: set[tuple[int, str]] = set()
        for node in _own_nodes(self.node):
            if isinstance(node, ast.BinOp) and isinstance(node.op, ast.MatMult):
                if self.difference(node.left) and self.difference(node.right):
                    found.add((node.lineno, "d @ d"))
            if not isinstance(node, ast.Call):
                continue
            fn, args = _name(node.func), node.args
            method_of = node.func.value if isinstance(node.func, ast.Attribute) else None
            if fn == "norm" and args and self.difference(args[0]):
                found.add((node.lineno, "norm of a difference"))
            if fn in {"max", "amax", "nanmax"} and args and self.abs_of_difference(args[0]):
                found.add((node.lineno, "max-abs of a difference"))
            if fn == "max" and method_of is not None and self.abs_of_difference(method_of):
                found.add((node.lineno, "max-abs of a difference"))
            if fn in {"sum", "nansum", "mean"} and args and self.squared_difference(args[0]):
                found.add((node.lineno, "sum of a squared difference"))
            if fn in {"sum", "mean"} and method_of is not None and self.squared_difference(method_of):
                found.add((node.lineno, "sum of a squared difference"))
            if fn in {"dot", "vdot"} and len(args) == 2 and all(self.difference(a) for a in args):
                found.add((node.lineno, f"{fn}(d, d)"))
        return found


def scopes(tree: ast.Module):
    """``(qualified name, scope)`` for the module body and every function in it, nested ones included."""
    out = []

    def visit(node: ast.AST, prefix: str, inherited: frozenset[str]) -> None:
        scope = _Scope(node, inherited)
        out.append((prefix or "<module>", scope))
        names = frozenset(scope.named) if isinstance(node, _FUNCTIONS) else frozenset()
        for child in _own_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                visit(child, _qualname(tree, child), names)

    visit(tree, "", frozenset())
    return out


def _qualname(tree: ast.Module, target: ast.AST) -> str:
    def walk(node: ast.AST, path: list[str]):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if child is target:
                    return ".".join([*path, child.name])
                found = walk(child, [*path, child.name])
                if found:
                    return found
            else:
                found = walk(child, path)
                if found:
                    return found
        return None

    return walk(tree, []) or "<lambda>"


def in_population(rel: Path, tree: ast.AST) -> bool:
    return COUPLING in rel.parents or _references_an_owner(tree)


def scan(root: Path) -> tuple[list[str], list[str]]:
    """``(population, rival sites)`` for the tree at ``root``; a site is ``path:line: what``."""
    population, sites = [], set()
    for path in sorted((root / ALG).rglob("*.py")):
        rel = path.relative_to(root)
        tree = ast.parse(path.read_text(), filename=str(rel))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and _name(node.func) in DEPRECATED:
                sites.add(f"{rel.as_posix()}:{node.lineno}: calls {_name(node.func)}")
        if not in_population(rel, tree):
            continue
        population.append(rel.as_posix())
        for qualname, scope in scopes(tree):
            if (rel.as_posix(), qualname) in EXCLUDED:
                continue
            for line, what in scope.rivals():
                sites.add(f"{rel.as_posix()}:{line}: {what}")
    return population, sorted(sites)


@pytest.fixture(scope="module")
def alg_copy(tmp_path_factory) -> Path:
    """A copy of `mfgarchon/alg` to plant rivals in; the tests add files and restore what they edit."""
    root = tmp_path_factory.mktemp("c1_guard")
    shutil.copytree(REPO / ALG, root / ALG)
    return root


def test_no_module_in_c1s_population_measures_a_change_itself():
    population, sites = scan(REPO)
    iterators = {
        "fixed_point_iterator.py",
        "block_iterators.py",
        "fictitious_play.py",
        "multi_population_iterator.py",
        "graph_mfg_solver.py",
        "regime_switching_iterator.py",
    }
    assert iterators <= {Path(p).name for p in population}, "the population no longer reaches the six iterators"
    assert sites == [], "route each through sweep_change and check_convergence_criteria (C1, #2555):\n" + "\n".join(
        sites
    )


def test_the_rules_fire_on_real_code_outside_the_population():
    """The zero above is not rules that cannot match: over every module of alg/, they find real loops."""
    sites = set()
    for path in sorted((REPO / ALG).rglob("*.py")):
        for _, scope in scopes(ast.parse(path.read_text())):
            sites.update(f"{path.relative_to(REPO).as_posix()}:{line}" for line, _ in scope.rivals())
    assert any(s.startswith("mfgarchon/alg/numerical/continuation/homotopy.py:") for s in sites)
    assert any(s.startswith("mfgarchon/alg/optimization/variational_solvers/primal_dual_solver.py:") for s in sites)


_INLINE = "def _planted(U_new, U_old):\n    return np.linalg.norm(U_new - U_old)\n"
_NAMED = "def _planted(M_new, M_old):\n    delta = M_new - M_old\n    return np.max(np.abs(delta))\n"


def _plant(target: Path, original: str, planted: str) -> int:
    """Append ``planted`` to a copied module and return the line of its last statement, the rival."""
    text = original + "\n\n" + planted
    target.write_text(text)
    return len(text.rstrip("\n").splitlines())


@pytest.mark.parametrize(("planted", "what"), [(_INLINE, "norm of a difference"), (_NAMED, "max-abs of a difference")])
def test_a_rival_planted_in_an_iterator_is_found(alg_copy, planted, what):
    target = alg_copy / COUPLING / "fixed_point_iterator.py"
    original = target.read_text()
    try:
        line = _plant(target, original, planted)
        assert scan(alg_copy)[1] == [f"{(COUPLING / 'fixed_point_iterator.py').as_posix()}:{line}: {what}"]
    finally:
        target.write_text(original)


def test_a_new_consumer_outside_coupling_is_in_the_population(alg_copy):
    """The population predicate moves: a module that takes PicardConfig is held to C1 wherever it lives."""
    rel = ALG / "planted_consumer.py"
    body = "import numpy as np\n\n\ndef _planted(U_new, U_old):\n    return np.linalg.norm(U_new - U_old)\n"
    try:
        (alg_copy / rel).write_text(body)
        assert scan(alg_copy)[1] == [], "without an owner reference the module is not C1's"
        (alg_copy / rel).write_text("from mfgarchon.config import PicardConfig\n" + body)
        assert scan(alg_copy)[1] == [f"{rel.as_posix()}:6: norm of a difference"]
    finally:
        (alg_copy / rel).unlink(missing_ok=True)


@pytest.mark.parametrize(
    ("planted", "what"),
    [
        ("d = a - b\nx = np.max(np.abs(d))", "max-abs of a difference"),
        ("x = np.abs(a - b).max()", "max-abs of a difference"),
        ("x = np.sqrt(np.sum((a - b) ** 2) * dx)", "sum of a squared difference"),
        ("d = a - b\nx = (d * d).sum()", "sum of a squared difference"),
        ("x = np.mean(np.square(a - b))", "sum of a squared difference"),
        ("d = a - b\nx = d @ d", "d @ d"),
        ("d = a - b\nx = np.dot(d, d)", "dot(d, d)"),
        ("d = a - b\nx = np.vdot(d, d)", "vdot(d, d)"),
        ("x = calculate_error(a, b)", "calls calculate_error"),
    ],
)
def test_each_covered_shape_is_found(alg_copy, planted, what):
    rel = COUPLING / "planted_shape.py"
    body = "def _planted(a, b, dx):\n" + "".join(f"    {line}\n" for line in planted.splitlines())
    try:
        (alg_copy / rel).write_text(body)
        sites = scan(alg_copy)[1]
        assert len(sites) == 1, sites
        assert sites[0].endswith(f": {what}")
    finally:
        (alg_copy / rel).unlink(missing_ok=True)


def test_the_anderson_exclusion_is_one_live_function(alg_copy):
    """It still names a function that exists and holds a matched site, and it reaches no other function."""
    module, qualname = next(iter(EXCLUDED))
    tree = ast.parse((REPO / module).read_text())
    held = dict(scopes(tree))
    assert qualname in held, f"{qualname} is gone from {module}: delete the exclusion"
    assert held[qualname].rivals(), f"{qualname} no longer matches a rule: delete the exclusion"
    target = alg_copy / module
    original = target.read_text()
    try:
        line = _plant(target, original, _NAMED)
        assert scan(alg_copy)[1] == [f"{module}:{line}: max-abs of a difference"]
    finally:
        target.write_text(original)


def test_a_ratio_minus_one_is_not_a_change():
    """A difference needs two non-constant operands: a mass diagnostic is not a sweep's change."""
    scope = _Scope(ast.parse("x = np.max(np.abs(mass / m0 - 1.0))"))
    assert scope.rivals() == set()
