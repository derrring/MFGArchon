"""The structural guard for #2644: an inner iteration under `alg/` that can stop without converging hands its failure to
one owner, `base_hjb.report_inner_solve_failure` (directly, or through `BaseHJBSolver._inner_solve_failed`).

**Population**, by structure and never by what a loop does on exhaustion: plan_B's two census predicates, ported
here.
- P1, a convergence exit: a loop that can leave through `break` or `return` under a tolerance comparison. That is a
  name matching `tol|eps|threshold|thresh|atol|rtol|tolerance|criterion`, an `allclose`/`isclose`, a name derived
  from one, a `converg` name, or a package helper whose returned value carries one.
- P2, a budget: a loop whose bound names a budget (`max|iter|budget|sweep|...`) and that has any conditional early
  exit.
- Both are read over every module of `mfgarchon/alg/`; helpers are resolved over the whole package.

**Rule**, per function holding such a loop: it calls the owner, or it is allowed below, by file and function with its
loop count. A count pin means a new loop inside an allowed function still fails. An allowance that covers no site
fails, so a function that moves cannot leave its name behind.

**Known blind spots**, as plan_B's census states them: tolerances under other names; fixed-count iterations with no
test; recursion; iterations inside third-party code and callables passed in; anything outside `mfgarchon/alg/` (the
geometry projections and `core/` minimisers are #2644's B2 rows); and a member that calls the owner on one path and
returns its iterate in silence on another.
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

TOL = re.compile(r"(tol|eps|threshold|thresh|atol|rtol|tolerance|criterion)", re.I)
CONV = re.compile(r"converg", re.I)
BUDGET = re.compile(r"(max|iter|budget|sweep|n_steps|num_steps|maxit|limit|n_sub|substep|n_corr|attempt)", re.I)
OWNER = {"_inner_solve_failed", "report_inner_solve_failure"}
LOOPS = (ast.For, ast.While, ast.AsyncFor)
SCOPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)

_OUTER = "an outer coupling or MFG loop: its verdict reaches the caller in SolverResult.converged, out of the class"
#: (file under mfgarchon/alg/, function) -> (loop count, why it does not route through the owner).
ALLOWED = {
    ("numerical/coupling/block_iterators.py", "BlockIterator.solve"): (1, _OUTER),
    ("numerical/coupling/fictitious_play.py", "FictitiousPlayIterator.solve"): (1, _OUTER),
    ("numerical/coupling/fixed_point_iterator.py", "FixedPointIterator.solve"): (1, _OUTER),
    ("numerical/coupling/graph_mfg_solver.py", "GraphMFGSolver.solve"): (1, _OUTER),
    ("numerical/coupling/multi_population_iterator.py", "MultiPopulationIterator.solve"): (1, _OUTER),
    ("numerical/coupling/regime_switching_iterator.py", "RegimeSwitchingIterator.solve"): (1, _OUTER),
    ("numerical/coupling/newton_mfg_solver.py", "NewtonMFGSolver._run_picard_warmup"): (1, f"{_OUTER}; a warm-up"),
    ("optimization/optimal_transport/wasserstein_solver.py", "WassersteinMFGSolver.solve"): (1, _OUTER),
    ("optimization/variational_solvers/primal_dual_solver.py", "PrimalDualMFGSolver.solve"): (1, _OUTER),
    ("numerical/continuation/homotopy.py", "HomotopyContinuation._corrector"): (
        1,
        "returns its converged flag to trace",
    ),
    ("numerical/continuation/homotopy.py", "HomotopyContinuation.trace"): (1, "B2 row 14: reads no inner verdict"),
    ("optimization/optimal_transport/sinkhorn_solver.py", "SinkhornMFGSolver.solve"): (1, "B2 row 15: unreachable"),
    ("optimization/optimal_transport/sinkhorn_solver.py", "SinkhornMFGSolver._sinkhorn_standard"): (1, "B2 row 15"),
    ("optimization/optimal_transport/sinkhorn_solver.py", "SinkhornMFGSolver._sinkhorn_log_domain"): (1, "B2 row 15"),
    ("numerical/hjb_solvers/hjb_weno.py", "HJBWENOSolver._advance_full_interval"): (
        1,
        "raises by name when its substep budget is spent",
    ),
    ("numerical/gfdm_components/joint_socp.py", "PrecomputedJointSocpStencils._solve_with_bisection"): (
        1,
        "returns its status, which _precompute reads",
    ),
    ("numerical/gfdm_components/joint_socp.py", "PrecomputedJointSocpStencils._precompute"): (
        1,
        "an infeasible row goes to the relaxed SOCP, refused unless opted in (#2741)",
    ),
    ("numerical/gfdm_components/positive_qp_fan.py", "select_guarded_rays"): (
        1,
        "returns None, and its caller records the row as unplaced with a reason",
    ),
    ("numerical/gfdm_components/neighborhood_builder.py", "NeighborhoodBuilder.build_neighborhood_structure"): (
        1,
        "a stencil-size search bounded by max_delta, not an iteration toward a root",
    ),
    ("numerical/hjb_solvers/hjb_sl_characteristics.py", "apply_boundary_conditions_1d"): (
        2,
        "folds a position into the box at most max_reflections times, then clamps it: a placement, not a root",
    ),
    ("numerical/hjb_solvers/hjb_sl_characteristics.py", "apply_boundary_conditions_nd"): (2, "as the 1-D fold"),
    ("numerical/fp_solvers/fp_particle_bc.py", "reflect_at_first_contacts"): (
        1,
        "a documented, counted reflection cap",
    ),
}


def _idents(node: ast.AST) -> list[str]:
    out = []
    for n in ast.walk(node):
        if isinstance(n, ast.Name):
            out.append(n.id)
        elif isinstance(n, ast.Attribute):
            out.append(n.attr)
        elif isinstance(n, ast.Constant) and isinstance(n.value, str):
            out.append(n.value)
        elif isinstance(n, ast.keyword) and n.arg:
            out.append(n.arg)
    return out


def _own(nodes):
    """Every node under ``nodes`` outside nested functions, lambdas and classes."""
    stack = list(nodes)
    while stack:
        n = stack.pop()
        yield n
        stack.extend(c for c in ast.iter_child_nodes(n) if not isinstance(c, SCOPES))


def _called(node: ast.AST) -> list[str]:
    return [
        n.func.attr if isinstance(n.func, ast.Attribute) else n.func.id
        for n in ast.walk(node)
        if isinstance(n, ast.Call) and isinstance(n.func, (ast.Attribute, ast.Name))
    ]


def _tol_compare(node: ast.AST) -> bool:
    for n in ast.walk(node):
        if isinstance(n, ast.Compare) and any(isinstance(o, (ast.Lt, ast.LtE, ast.Gt, ast.GtE)) for o in n.ops):
            if any(TOL.search(i) for i in _idents(n)):
                return True
        if isinstance(n, ast.Call) and _called(n)[:1] and _called(n)[0] in ("allclose", "isclose"):
            return True
    return False


def _converg(node: ast.AST) -> bool:
    return any(CONV.search(i) for i in _idents(node) if not i.startswith("__"))


def _derived(func: ast.AST) -> set[str]:
    """Names assigned, in ``func``, from a tolerance comparison, a `converg` name, or another derived name."""
    assigns = []
    for n in _own(func.body):
        if isinstance(n, ast.Assign):
            targets = [t for t in n.targets for t in (t.elts if isinstance(t, ast.Tuple) else [t])]
            assigns.append(([t.id if isinstance(t, ast.Name) else getattr(t, "attr", "") for t in targets], n.value))
        elif isinstance(n, (ast.AugAssign, ast.AnnAssign)) and n.value is not None:
            assigns.append(([getattr(n.target, "id", getattr(n.target, "attr", ""))], n.value))
    derived: set[str] = set()
    changed = True
    while changed:
        changed = False
        for names, value in assigns:
            if _tol_compare(value) or _converg(value) or any(i in derived for i in _idents(value)):
                for name in names:
                    if name and name not in derived:
                        derived.add(name)
                        changed = True
    return derived


def _helpers(trees: list[ast.AST]) -> set[str]:
    """Functions whose returned value carries a convergence test, a helper calling a helper included."""
    funcs = [f for t in trees for f in ast.walk(t) if isinstance(f, (ast.FunctionDef, ast.AsyncFunctionDef))]
    helpers: set[str] = set()
    changed = True
    while changed:
        changed = False
        for f in funcs:
            if f.name in helpers:
                continue
            derived = _derived(f)
            for r in _own(f.body):
                if isinstance(r, ast.Return) and r.value is not None:
                    v = r.value
                    if (
                        _tol_compare(v)
                        or _converg(v)
                        or any(i in derived for i in _idents(v))
                        or set(_called(v)) & helpers
                    ):
                        helpers.add(f.name)
                        changed = True
                        break
    return helpers


def _exits(branch, depth=0) -> bool:
    """Does ``branch`` hold a break binding to the enclosing loop (depth 0), or a return?"""
    for n in branch:
        if isinstance(n, SCOPES):
            continue
        if (isinstance(n, ast.Break) and depth == 0) or isinstance(n, ast.Return):
            return True
        inner = depth + 1 if isinstance(n, LOOPS) else depth
        if _exits(list(ast.iter_child_nodes(n)) if not isinstance(n, LOOPS) else n.body + n.orelse, inner):
            return True
    return False


def _early_exits(loop: ast.AST, test) -> bool:
    """An `if` in ``loop``'s own body whose test passes ``test`` and whose branch leaves ``loop``."""
    stack = [(c, 0) for c in loop.body]
    while stack:
        n, depth = stack.pop()
        if isinstance(n, SCOPES):
            continue
        if isinstance(n, ast.If) and test(n.test):
            branch = n.body + n.orelse
            if _exits(branch) if depth == 0 else any(isinstance(r, ast.Return) for r in _own(branch)):
                return True
        stack.extend((c, depth + 1 if isinstance(n, LOOPS) else depth) for c in ast.iter_child_nodes(n))
    return False


def scan(root: Path) -> dict[tuple[str, str], dict]:
    """The population under ``root``/mfgarchon/alg: (file, function) -> its loop lines, and whether it calls the owner."""
    package = sorted((root / "mfgarchon").rglob("*.py"))
    trees = {p: ast.parse(p.read_text(), str(p)) for p in package}
    helpers = _helpers(list(trees.values()))
    found: dict[tuple[str, str], dict] = {}
    for path, tree in trees.items():
        rel = (
            path.relative_to(root / "mfgarchon" / "alg").as_posix()
            if "alg" in path.relative_to(root).parts[1:2]
            else None
        )
        if rel is None:
            continue
        for cls_or_mod in [tree, *[c for c in ast.walk(tree) if isinstance(c, ast.ClassDef)]]:
            owner_name = "" if cls_or_mod is tree else f"{cls_or_mod.name}."
            for func in cls_or_mod.body:
                if not isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                derived = _derived(func)

                def convergence(test, derived=derived):
                    return (
                        _tol_compare(test)
                        or _converg(test)
                        or bool(set(_idents(test)) & derived)
                        or bool(set(_called(test)) & helpers)
                    )

                lines = []
                for loop in _own(func.body):
                    if not isinstance(loop, LOOPS):
                        continue
                    bound = loop.test if isinstance(loop, ast.While) else loop.iter
                    p1 = (isinstance(loop, ast.While) and convergence(loop.test)) or _early_exits(loop, convergence)
                    p2 = any(BUDGET.search(i) for i in _idents(bound)) and (
                        isinstance(loop, ast.While) or _early_exits(loop, lambda _t: True)
                    )
                    if p1 or p2:
                        lines.append(loop.lineno)
                if lines:
                    found[(rel, owner_name + func.name)] = {
                        "lines": sorted(lines),
                        "owner": bool(set(_called(func)) & OWNER),
                    }
    return found


def _violations(found: dict[tuple[str, str], dict]) -> list[str]:
    out = []
    for key, site in sorted(found.items()):
        if site["owner"]:
            continue
        if key not in ALLOWED:
            out.append(f"{key} loops at {site['lines']} stop without the owner, and are not allowed")
        elif len(site["lines"]) != ALLOWED[key][0]:
            out.append(f"{key}: {len(site['lines'])} loops, {ALLOWED[key][0]} allowed: {site['lines']}")
    return out


MEMBERS = {
    ("numerical/hjb_solvers/base_hjb.py", "solve_hjb_timestep_newton"),
    ("numerical/hjb_solvers/hjb_gfdm.py", "HJBGFDMSolver._solve_timestep"),
    ("numerical/hjb_solvers/hjb_howard.py", "HJBHowardSolver._howard_step"),
    ("numerical/network_solvers/hjb_network.py", "NetworkPolicyIterationHJBSolver._policy_iteration_step"),
    ("numerical/weak_form_hjb_solver.py", "WeakFormHJBSolver._solve_timestep_newton"),
}


@pytest.fixture(scope="module")
def found():
    return scan(REPO)


def test_every_inner_iteration_under_alg_reports_to_the_owner(found):
    violations = _violations(found)
    assert not violations, "an inner iteration that bypasses the owner (#2644):\n" + "\n".join(violations)


def test_the_census_finds_the_members_and_each_routes_through_the_owner(found):
    """The control: plan_B's census found these at ee0879d0. Were the predicate to stop seeing one, the rule above
    could not fire for it."""
    assert {key for key in MEMBERS if key in found and found[key]["owner"]} == MEMBERS


def test_every_allowance_still_covers_a_site(found):
    assert [key for key in ALLOWED if key not in found] == []


@pytest.fixture(scope="module")
def package_copy(tmp_path_factory):
    root = tmp_path_factory.mktemp("guard_2644")
    shutil.copytree(REPO / "mfgarchon", root / "mfgarchon", ignore=shutil.ignore_patterns("__pycache__"))
    return root


_NEWTON = "\n\ndef _rival(u, f, max_iterations, tolerance):\n    for _ in range(max_iterations):\n        step = f(u)\n        u = u - step\n        if abs(step) < tolerance:\n            break\n    return u\n"
PLANTED = {
    "a new Newton loop with no owner call": ("mfgarchon/alg/numerical/hjb_solvers/hjb_fdm.py", None, _NEWTON),
    "a member that stops calling the owner": (
        "mfgarchon/alg/numerical/hjb_solvers/hjb_howard.py",
        "report_inner_solve_failure(",
        "print(",
    ),
    "a second loop in an allowed function": (
        "mfgarchon/alg/numerical/gfdm_components/positive_qp_fan.py",
        None,
        "\n\ndef select_guarded_rays(rays, h6, cap, parameters):\n    for _ in range(parameters.max_expansions):\n        if rays is None:\n            return None\n    for _ in range(parameters.max_expansions):\n        if rays is None:\n            return None\n",
    ),
}


@pytest.mark.parametrize("form", list(PLANTED))
def test_a_planted_rival_is_found(package_copy, form):
    rel, old, new = PLANTED[form]
    target = package_copy / rel
    original = target.read_text()
    try:
        if old is None:
            target.write_text(original + new)
        else:
            assert old in original, (rel, old)
            target.write_text(original.replace(old, new))
        violations = _violations(scan(package_copy))
    finally:
        target.write_text(original)
    stem = rel.split("/alg/")[1]
    assert any(stem in v for v in violations), (form, violations)
