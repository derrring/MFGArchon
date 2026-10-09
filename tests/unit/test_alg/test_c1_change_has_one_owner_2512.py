"""C1's structural guard: nothing in C1's population measures a sweep's change, or decides on one, itself.

C1 (#2555) governs the outer coupling tolerance of docs/user/CONVENTIONS.md § 7, taken by `MFGProblem.solve`,
the coupling iterators and `PicardConfig`: it bounds the change one sweep makes, the map's output against
its input, Phi(x) - x, relative and in the problem's own measure. Its owners are `MFGProblem.spatial_measure` (the measure), `sweep_change` (the change),
`worst_sweep_change` (its max over a multi-field sweep's fields, #2578) and `check_convergence_criteria`
(the verdict). #2570's behavioural pins show that today's six iterators use
them. They cannot stop a new consumer from computing the change itself, and this guard does (audit session
ruling, 2026-10-09, #2512 comment 6073690080).

**Population.** Every module under `mfgarchon/alg/numerical/coupling/`, and any other module under
`mfgarchon/alg/` that references `PicardConfig`, its `picard` attribute, `BaseCouplingIterator`,
`check_convergence_criteria`, `sweep_change` or `worst_sweep_change`. "Takes the outer tolerance" is approximated by those names,
not implemented: a module that receives the tolerance only as a plain `tolerance=` argument, or names
`PicardConfig` only in a string annotation, is outside it.

**A difference** is a binary subtraction of two non-constant operands (a ratio minus 1 is a diagnostic,
not a change between two states); ``.ravel()``, ``.flatten()`` or ``.reshape(...)`` of one; or a name
assigned directly from a subtraction in the same function or an enclosing one. **Rivals in the population:**
- a ``norm`` of a difference;
- ``max``, ``amax`` or ``nanmax`` of ``abs`` or ``absolute`` of a difference, or ``abs(...).max()``;
- ``sum``, ``nansum`` or ``mean`` of a squared difference, or of a product with one as a factor
  (``(a - b)**2 * dx``), as a call or a method, where squared is ``d**2``, ``d * d`` or ``square(d)``;
- ``d @ d``, ``dot(d, d)`` or ``vdot(d, d)``;
- an aggregation of a change over fields, which is `worst_sweep_change`'s (#2578): ``max``, ``min``,
  ``amax``, ``amin``, ``nanmax`` or ``nanmin``, as a call or a method, over a comprehension that reads one
  of `sweep_change`'s keys (``l2distu_rel`` and the rest, by subscript or ``.get``), or over a name the
  function fills with such reads (assigned a comprehension, appended or extended), or a running
  ``w = max(w, <key read>)``. A ``max`` between U's and M's change of one sweep is not over fields, and
  is not matched;
- what the guard treats as a convergence verdict outside `check_convergence_criteria`: every ``<``,
  ``<=``, ``>`` or ``>=`` comparison with exactly one side whose name contains "tol", and ``allclose`` /
  ``isclose`` with a tol-named ``atol`` or ``rtol`` keyword. That is a shape, not a definition of a
  verdict: it also matches a validation such as ``tol <= 0``, and it misses ``err < tol * scale``.

**Anywhere under** `mfgarchon/alg/`: a call to a metric that computes a change its own way. Among them are
the deprecated `calculate_l2_convergence_metrics`, and #2566's `calculate_error`, `compute_norm`,
`MFGConvergenceChecker` and `create_convergence_checker`, the factory of all three of #2566's checkers.

Every function, method, nested function and lambda is a scope. Other spellings, such as an L1 sum of
``abs(d)``, a difference passed through a helper or stored on an attribute first, or a tuple or augmented
assignment, are out of scope: this guard names the shapes it covers and does not claim more.

**Excluded, by qualified name, each pinned below:**
- `check_convergence_criteria`, the verdict's owner.
- `AndersonAccelerator.update`. Its residual norm and its stagnation test decide how to mix: restart,
  skip, safeguard. They do not decide whether the outer loop has converged.

`NewtonMFGSolver` stops on the Picard residual Phi(x) - x, the quantity this convention governs, and since
#2565 it decides through the owners like the other iterators, so it carries no exclusion.

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
OWNERS = frozenset(
    {
        "PicardConfig",
        "picard",
        "BaseCouplingIterator",
        "check_convergence_criteria",
        "sweep_change",
        "worst_sweep_change",
    }
)
#: The keys of `sweep_change`'s result: a max or min over fields of one of them is an aggregation (#2578).
CHANGE_KEYS = frozenset({"l2distu_rel", "l2distu_abs", "l2distm_rel", "l2distm_abs"})
_AGGREGATES = frozenset({"max", "min", "amax", "amin", "nanmax", "nanmin"})
_COMPREHENSIONS = (ast.GeneratorExp, ast.ListComp, ast.SetComp)
#: Metrics that compute a change their own way: the deprecated wrapper, and #2566's helpers.
RIVAL_METRICS = frozenset(
    {
        "calculate_l2_convergence_metrics",
        "calculate_error",
        "compute_norm",
        "MFGConvergenceChecker",
        "create_convergence_checker",
    }
)
#: Each excludes that one function only. See the module docstring.
EXCLUDED_FUNCTIONS = frozenset(
    {
        ((COUPLING / "fixed_point_utils.py").as_posix(), "check_convergence_criteria"),
        ((COUPLING / "anderson_acceleration.py").as_posix(), "AndersonAccelerator.update"),
    }
)
_FUNCTIONS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)
_ORDER = (ast.Lt, ast.LtE, ast.Gt, ast.GtE)


def _name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _tol_named(node: ast.AST) -> bool:
    name = _name(node)
    return name is not None and "tol" in name.lower()


def _reads_a_change_key(node: ast.AST) -> bool:
    """``node`` reads one of `sweep_change`'s keys, by subscript or by ``.get``, anywhere inside it."""
    for sub in ast.walk(node):
        if isinstance(sub, ast.Subscript) and isinstance(sub.slice, ast.Constant) and sub.slice.value in CHANGE_KEYS:
            return True
        if (
            isinstance(sub, ast.Call)
            and _name(sub.func) == "get"
            and sub.args
            and isinstance(sub.args[0], ast.Constant)
            and sub.args[0].value in CHANGE_KEYS
        ):
            return True
    return False


def _comprehension_of_change_keys(node: ast.AST) -> bool:
    return any(isinstance(sub, _COMPREHENSIONS) and _reads_a_change_key(sub.elt) for sub in ast.walk(node))


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
        #: Names this scope fills with per-field change values: a comprehension of key reads, or appended to.
        self.per_field: set[str] = set()
        for sub in _own_nodes(node):
            if isinstance(sub, ast.Assign) and _comprehension_of_change_keys(sub.value):
                self.per_field.update(t.id for t in sub.targets if isinstance(t, ast.Name))
            if (
                isinstance(sub, ast.Call)
                and isinstance(sub.func, ast.Attribute)
                and sub.func.attr in {"append", "extend"}
                and isinstance(sub.func.value, ast.Name)
                and any(_reads_a_change_key(a) for a in sub.args)
            ):
                self.per_field.add(sub.func.value.id)
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
        if self._is_inline_difference(node) or (isinstance(node, ast.Name) and node.id in self.named):
            return True
        return (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in {"ravel", "flatten", "reshape"}
            and self.difference(node.func.value)
        )

    def abs_of_difference(self, node: ast.AST) -> bool:
        return (
            isinstance(node, ast.Call)
            and _name(node.func) in {"abs", "absolute"}
            and bool(node.args)
            and self.difference(node.args[0])
        )

    def squared_difference(self, node: ast.AST) -> bool:
        """``d**2``, ``d * d`` or ``square(d)``, alone or times any weight."""
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
            return self.difference(node.left) and isinstance(node.right, ast.Constant) and node.right.value == 2
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
            if self.difference(node.left) and self.difference(node.right):
                return True
            return self.squared_difference(node.left) or self.squared_difference(node.right)
        return (
            isinstance(node, ast.Call)
            and _name(node.func) == "square"
            and bool(node.args)
            and self.difference(node.args[0])
        )

    def over_fields(self, node: ast.AST) -> bool:
        """``node`` holds one change value per field: a comprehension of key reads, or a name filled with them."""
        if isinstance(node, ast.Name) and node.id in self.per_field:
            return True
        return _comprehension_of_change_keys(node)

    def rivals(self) -> set[tuple[int, str]]:
        found: set[tuple[int, str]] = set()
        for node in _own_nodes(self.node):
            if (
                isinstance(node, ast.Assign)
                and isinstance(node.value, ast.Call)
                and _name(node.value.func) in _AGGREGATES
            ):
                targets = {t.id for t in node.targets if isinstance(t, ast.Name)}
                running = [a for a in node.value.args if isinstance(a, ast.Name) and a.id in targets]
                if running and any(_reads_a_change_key(a) for a in node.value.args):
                    found.add((node.lineno, "an aggregation of a change over fields"))
            if isinstance(node, ast.BinOp) and isinstance(node.op, ast.MatMult):
                if self.difference(node.left) and self.difference(node.right):
                    found.add((node.lineno, "d @ d"))
            if isinstance(node, ast.Compare) and any(isinstance(op, _ORDER) for op in node.ops):
                if sum(_tol_named(side) for side in [node.left, *node.comparators]) == 1:
                    found.add((node.lineno, "a verdict against a tolerance"))
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
            if fn in _AGGREGATES and (
                any(self.over_fields(a) for a in args) or (method_of is not None and self.over_fields(method_of))
            ):
                found.add((node.lineno, "an aggregation of a change over fields"))
            if fn in {"allclose", "isclose"} and any(
                k.arg in {"atol", "rtol"} and _tol_named(k.value) for k in node.keywords
            ):
                found.add((node.lineno, "a verdict by allclose"))
        return found


def _qualname(tree: ast.Module, target: ast.AST) -> str:
    def walk(node: ast.AST, path: list[str]):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (*_FUNCTIONS, ast.ClassDef)):
                name = getattr(child, "name", "<lambda>")
                if child is target:
                    return ".".join([*path, name])
                found = walk(child, [*path, name])
                if found:
                    return found
            else:
                found = walk(child, path)
                if found:
                    return found
        return None

    found = walk(tree, [])
    if found is None:
        raise AssertionError(f"no qualified name for {ast.dump(target)[:80]}")
    return found


def scopes(tree: ast.Module):
    """``(qualified name, scope)`` for the module body and every function and lambda in it, nested ones included."""
    out = []

    def visit(node: ast.AST, qualname: str, inherited: frozenset[str]) -> None:
        scope = _Scope(node, inherited)
        out.append((qualname, scope))
        names = frozenset(scope.named) if isinstance(node, _FUNCTIONS) else frozenset()
        for child in _own_nodes(node):
            if isinstance(child, _FUNCTIONS):
                visit(child, _qualname(tree, child), names)

    visit(tree, "<module>", frozenset())
    return out


def _excluded(module: str, qualname: str) -> bool:
    return (module, qualname) in EXCLUDED_FUNCTIONS


def in_population(rel: Path, tree: ast.AST) -> bool:
    return COUPLING in rel.parents or _references_an_owner(tree)


def scan(root: Path) -> tuple[list[str], list[str]]:
    """``(population, rival sites)`` for the tree at ``root``; a site is ``path:line: what``."""
    population, sites = [], set()
    for path in sorted((root / ALG).rglob("*.py")):
        rel = path.relative_to(root)
        module = rel.as_posix()
        tree = ast.parse(path.read_text(), filename=str(rel))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and _name(node.func) in RIVAL_METRICS:
                sites.add(f"{module}:{node.lineno}: calls {_name(node.func)}")
        if not in_population(rel, tree):
            continue
        population.append(module)
        for qualname, scope in scopes(tree):
            if _excluded(module, qualname):
                continue
            for line, what in scope.rivals():
                sites.add(f"{module}:{line}: {what}")
    return population, sorted(sites)


@pytest.fixture(scope="module")
def alg_copy(tmp_path_factory) -> Path:
    """A copy of `mfgarchon/alg` to plant rivals in; the tests add files and restore what they edit."""
    root = tmp_path_factory.mktemp("c1_guard")
    shutil.copytree(REPO / ALG, root / ALG)
    return root


def _planted_scan(alg_copy: Path, target: Path, anchor: str | None, planted: str) -> list[str]:
    """Plant ``planted`` in a copied module, before ``anchor`` or at the end, scan, and restore the module."""
    original = target.read_text() if target.exists() else None
    try:
        if anchor is None:
            target.write_text((original + "\n\n" if original else "") + planted)
        else:
            assert original is not None, anchor
            assert original.count(anchor) == 1, anchor
            target.write_text(original.replace(anchor, planted + anchor))
        return scan(alg_copy)[1]
    finally:
        if original is None:
            target.unlink(missing_ok=True)
        else:
            target.write_text(original)


def _function(name: str, indent: str = "", args: str = "a, b") -> str:
    body = [f"def {name}({args}):", "    delta = a - b", "    return np.max(np.abs(delta))"]
    return "".join(f"{indent}{line}\n" for line in body) + "\n"


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
    assert sites == [], (
        "each site measures a change, or decides on one, outside C1's owners. Route it through sweep_change, "
        "worst_sweep_change and check_convergence_criteria (C1, #2555, #2578); if it is not the outer loop's "
        "change, name it as an exclusion "
        "in this file with the reason:\n" + "\n".join(sites)
    )


def test_the_rules_fire_on_real_code_outside_the_population():
    """The zero above is not rules that cannot match: over every module of alg/, they find real loops."""
    sites = set()
    for path in sorted((REPO / ALG).rglob("*.py")):
        for _, scope in scopes(ast.parse(path.read_text())):
            sites.update(f"{path.relative_to(REPO).as_posix()}:{line}" for line, _ in scope.rivals())
    assert any(s.startswith("mfgarchon/alg/numerical/continuation/homotopy.py:") for s in sites)
    assert any(s.startswith("mfgarchon/alg/optimization/variational_solvers/primal_dual_solver.py:") for s in sites)


@pytest.mark.parametrize(
    ("planted", "what"),
    [
        ("def _planted(U_new, U_old):\n    return np.linalg.norm(U_new - U_old)\n", "norm of a difference"),
        (_function("_planted"), "max-abs of a difference"),
        ("def _planted(err, tol):\n    return err < tol\n", "a verdict against a tolerance"),
    ],
    ids=["inline", "named", "criterion"],
)
def test_a_rival_planted_in_an_iterator_is_found(alg_copy, planted, what):
    module = (COUPLING / "fixed_point_iterator.py").as_posix()
    sites = _planted_scan(alg_copy, alg_copy / module, None, planted)
    assert len(sites) == 1, sites
    assert sites[0].startswith(f"{module}:")
    assert sites[0].endswith(f": {what}")


def test_a_new_consumer_outside_coupling_is_in_the_population(alg_copy):
    """The population predicate moves: a module that takes PicardConfig is held to C1 wherever it lives."""
    rel = ALG / "planted_consumer.py"
    body = "import numpy as np\n\n\ndef _planted(U_new, U_old):\n    return np.linalg.norm(U_new - U_old)\n"
    assert _planted_scan(alg_copy, alg_copy / rel, None, body) == [], "without an owner reference it is not C1's"
    for reference in ("from mfgarchon.config import PicardConfig\n", "from x import BaseCouplingIterator\n"):
        sites = _planted_scan(alg_copy, alg_copy / rel, None, reference + body)
        assert sites == [f"{rel.as_posix()}:6: norm of a difference"], (reference, sites)


@pytest.mark.parametrize(
    ("planted", "what"),
    [
        ("d = a - b\nx = np.max(np.abs(d))", "max-abs of a difference"),
        ("x = np.abs(a - b).max()", "max-abs of a difference"),
        ("x = np.amax(np.abs(a - b))", "max-abs of a difference"),
        ("x = np.nanmax(np.absolute(a - b))", "max-abs of a difference"),
        ("x = np.max(np.abs((a - b).ravel()))", "max-abs of a difference"),
        ("x = np.linalg.norm((a - b).reshape(-1))", "norm of a difference"),
        ("f = lambda u, v: np.linalg.norm(u - v)", "norm of a difference"),
        ("x = np.sqrt(np.sum((a - b) ** 2) * dx)", "sum of a squared difference"),
        ("x = np.sum((a - b) ** 2 * dx)", "sum of a squared difference"),
        ("d = a - b\nx = (d * d).sum()", "sum of a squared difference"),
        ("x = np.mean(np.square(a - b))", "sum of a squared difference"),
        ("x = np.nansum((a - b) ** 2)", "sum of a squared difference"),
        ("d = a - b\nx = d @ d", "d @ d"),
        ("d = a - b\nx = np.dot(d, d)", "dot(d, d)"),
        ("d = a - b\nx = np.vdot(d, d)", "vdot(d, d)"),
        ("x = np.allclose(a, b, atol=tol)", "a verdict by allclose"),
        ('x = max(c["l2distm_rel"] for c in a)', "an aggregation of a change over fields"),
        ('x = min([c["l2distu_abs"] for c in a])', "an aggregation of a change over fields"),
        ('x = np.nanmax([c.get("l2distm_rel") for c in a])', "an aggregation of a change over fields"),
        ('x = np.array([c["l2distm_abs"] for c in a]).max()', "an aggregation of a change over fields"),
        (
            'e = [max(c["l2distu_rel"], c["l2distm_rel"]) for c in a]\nx = np.amax(e)',
            "an aggregation of a change over fields",
        ),
        ('e = []\nfor c in a:\n    e.append(c["l2distm_rel"])\nx = max(e)', "an aggregation of a change over fields"),
        ('w = 0.0\nfor c in a:\n    w = max(w, c["l2distm_rel"])', "an aggregation of a change over fields"),
        ("x = calculate_error(a, b)", "calls calculate_error"),
        ("x = calculate_l2_convergence_metrics(a, b)", "calls calculate_l2_convergence_metrics"),
        ("x = compute_norm(a)", "calls compute_norm"),
        ("x = MFGConvergenceChecker()", "calls MFGConvergenceChecker"),
        ("x = create_convergence_checker()", "calls create_convergence_checker"),
    ],
)
def test_each_covered_shape_is_found(alg_copy, planted, what):
    body = "def _planted(a, b, dx, tol):\n" + "".join(f"    {line}\n" for line in planted.splitlines())
    sites = _planted_scan(alg_copy, alg_copy / COUPLING / "planted_shape.py", None, body)
    assert len(sites) == 1, sites
    assert sites[0].endswith(f": {what}")


def test_each_exclusion_names_a_live_scope():
    """An exclusion of a function that is gone, or that no longer matches a rule, is deleted."""
    for module, name in sorted(EXCLUDED_FUNCTIONS):
        held = scopes(ast.parse((REPO / module).read_text()))
        inside = [scope for q, scope in held if q == name or q.startswith(name + ".")]
        assert inside, f"{name} is gone from {module}: delete the exclusion"
        assert any(scope.rivals() for scope in inside), f"{name} no longer matches a rule: delete the exclusion"


@pytest.mark.parametrize(
    ("module", "anchor", "planted"),
    [
        # another method of the excluded Anderson function's class
        ("anderson_acceleration.py", "    def reset(self):", _function("_planted", "    ", "self, a, b")),
        # a function nested inside the excluded one
        ("anderson_acceleration.py", "        residual_flat = f_flat - x_flat", _function("_planted", "        ")),
        # an `update` of another class in the same module
        ("anderson_acceleration.py", None, "class _Other:\n" + _function("update", "    ", "self, a, b")),
        # the same qualified name in another module
        ("fixed_point_iterator.py", None, "class AndersonAccelerator:\n" + _function("update", "    ", "self, a, b")),
        # a function of the same name as the verdict's owner, in another module
        ("fixed_point_iterator.py", None, _function("check_convergence_criteria")),
    ],
)
def test_an_exclusion_reaches_nothing_else(alg_copy, module, anchor, planted):
    sites = _planted_scan(alg_copy, alg_copy / COUPLING / module, anchor, planted)
    assert len(sites) == 1, sites
    assert sites[0].endswith(": max-abs of a difference")


def test_a_ratio_minus_one_is_not_a_change():
    """A difference needs two non-constant operands: a mass diagnostic is not a sweep's change."""
    scope = _Scope(ast.parse("x = np.max(np.abs(mass / m0 - 1.0))"))
    assert scope.rivals() == set()


@pytest.mark.parametrize(
    "planted",
    [
        'x = max(c["l2distu_rel"], c["l2distm_rel"])',
        'w = worst_sweep_change(a, "node")\nx = max(w["l2distu_rel"], w["l2distm_rel"])',
        "x = max(err for err in a)",
    ],
    ids=["u-against-m", "of-the-aggregate", "no-change-key"],
)
def test_a_max_that_is_not_over_fields_of_a_change_is_not_an_aggregation(alg_copy, planted):
    """The aggregation rule's scope: U against M of one change, or a max over values that are not a change."""
    body = "def _planted(a, b, dx, tol):\n" + "".join(f"    {line}\n" for line in planted.splitlines())
    assert _planted_scan(alg_copy, alg_copy / COUPLING / "planted_shape.py", None, body) == []
