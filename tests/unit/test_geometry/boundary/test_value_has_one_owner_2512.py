"""Row B2's structural guard: a boundary value is decoded by one owner (#2512, row B2).

The owner is `types.boundary_value`, reached through `types.segment_value` for a segment. This guard fails if a
module decodes a value itself (audit session ruling, 2026-10-09, #2512 comment 6073690080).

**Population.** Every module under `mfgarchon/`, except `applicator_graph.py`, whose node values, and whose t = 0
read of a callable graph value, are rows N1/N2.

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
- a raw value returned or copied for a caller to decode (D14, D21): the dataflow is within one function. The census's
  second predicate, ``if callable(v): v(...)`` on a value passed into a helper (D5, D16, D22-D26), has its own check
  below (B2b-2a);
- names carried through a second name (``f = seg.value; g = f``), an annotated assignment, a walrus, or tuple
  unpacking;
- arithmetic on a value (``seg.value * h``), an augmented store (``a[i] += seg.value``), an attribute store
  (``o.x = seg.value``), and indexing a value (``seg.value[0]``);
- a value wrapped in a list (``np.array([seg.value])``), passed by keyword (``np.asarray(a=seg.value)``), or as a
  conversion's later argument (``np.full``'s fill value), and ``np.float64(...)``;
- ``.value.__call__``, a closure over a carried name, a loop target, and module-level code.
None of these is a shape a census decoder uses (review 2 of #2598 planted each one).
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
# Rows N1/N2: node values, and `GraphApplicator.apply`/`apply_fp`/`apply_hjb` reading a callable node value at t = 0
# when given no t, as `apply_direct`/`apply_edge_direct` do at their default, which the manifest's N1/N2 comment
# carries.
EXCLUDED = {"mfgarchon/geometry/boundary/applicator_graph.py"}
VALUE_ATTRS = frozenset({"value", "default_value"})
ENUM_HOLDERS = frozenset({"bc_type", "default_bc", "type", "kind", "math_type"})
CONVERT = frozenset({"float", "int", "complex", "asarray", "array"})
ALLOWED = {
    # An attribute named `value` that is not a boundary value: a cvxpy Problem's optimal value.
    ("mfgarchon/alg/numerical/gfdm_components/joint_socp.py", "solve_joint_socp_at_stencil"): "cvxpy",
    ("mfgarchon/alg/numerical/gfdm_components/joint_socp.py", "solve_relaxed_joint_socp_at_stencil"): "cvxpy",
    ("mfgarchon/alg/numerical/gfdm_components/positive_qp_fan.py", "_positive_laplacian_row"): "cvxpy",
    ("mfgarchon/alg/numerical/gfdm_components/positive_qp_fan.py", "_mirror_row"): "cvxpy",
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


# Row B2's second half (B2b-1): a Robin segment's alpha and beta have one reader, `types.segment_robin_coefficients`.
# Any read of ``<x>.alpha``/``<x>.beta``, or ``getattr(<x>, "alpha"|"beta", ...)``, outside the functions below fails,
# so a reader that takes a coefficient with its own default or its own float() is found. Allowed by file and function:
COEFFICIENT_READS_ALLOWED = {
    # Classifiers of a segment, which read no number: validation.
    ("mfgarchon/geometry/boundary/conditions.py", "validate_values"): "a None check",
    # A `ResolvedBC`, which the owner already read (`resolution._resolve_passthrough`).
    ("mfgarchon/geometry/boundary/resolution.py", "<module>"): "ResolvedBC",
    ("mfgarchon/geometry/boundary/resolution.py", "resolved_bc_to_calculator"): "ResolvedBC",
    ("mfgarchon/geometry/boundary/resolution.py", "to_boundary_conditions"): "ResolvedBC",
    # Attributes named alpha or beta that are not Robin coefficients.
    ("mfgarchon/alg/numerical/adjoint/operators.py", "build_diffusion_matrix_1d"): "a stencil's alpha",
    ("mfgarchon/alg/numerical/coupling/anderson_acceleration.py", "_compute_aa_step_qr"): "Anderson's damping",
    ("mfgarchon/alg/numerical/coupling/anderson_acceleration.py", "update"): "Anderson's damping",
    ("mfgarchon/alg/numerical/pde_solvers/implicit_heat.py", "<module>"): "a diffusivity",
    ("mfgarchon/alg/numerical/pde_solvers/implicit_heat.py", "__repr__"): "a diffusivity",
    ("mfgarchon/alg/numerical/pde_solvers/implicit_heat.py", "_build_system_matrices"): "a diffusivity",
    ("mfgarchon/alg/numerical/pde_solvers/implicit_heat.py", "get_cfl_number"): "a diffusivity",
}


def coefficient_reads(scope: ast.AST) -> list[int]:
    lines = []
    for node in own_nodes(scope):
        if (isinstance(node, ast.Attribute) and node.attr in ("alpha", "beta") and isinstance(node.ctx, ast.Load)) or (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "getattr"
            and len(node.args) >= 2
            and isinstance(node.args[1], ast.Constant)
            and node.args[1].value in ("alpha", "beta")
        ):
            lines.append(node.lineno)
    return lines


def coefficient_scan(root: Path, allowed: Mapping[tuple[str, str], str] = COEFFICIENT_READS_ALLOWED):
    """``(population, sites)``: every coefficient read outside ``allowed``, module level included."""
    population, sites = [], []
    for path in sorted((root / PACKAGE).rglob("*.py")):
        rel = path.relative_to(root).as_posix()
        if rel in EXCLUDED:
            continue
        population.append(rel)
        tree = ast.parse(path.read_text())
        scopes = [(tree, "<module>")] + [
            (n, getattr(n, "name", "<lambda>")) for n in ast.walk(tree) if isinstance(n, FUNCTIONS)
        ]
        for scope, name in scopes:
            if (rel, name) not in allowed:
                sites.extend(f"{rel}:{line}: in {name}" for line in coefficient_reads(scope))
    return population, sorted(set(sites))


def test_every_robin_coefficient_is_read_by_the_owner():
    population, sites = coefficient_scan(REPO)
    assert len(population) > 100, f"the population looks truncated: {len(population)} modules"
    assert sites == [], "a Robin coefficient read outside `types.segment_robin_coefficients`:\n" + "\n".join(sites)


def test_every_coefficient_allowance_still_covers_a_read():
    _, sites = coefficient_scan(REPO, allowed={})
    covered = {(site.split(":")[0], site.rsplit(" in ", 1)[1]) for site in sites}
    assert set(COEFFICIENT_READS_ALLOWED) <= covered, sorted(set(COEFFICIENT_READS_ALLOWED) - covered)


@pytest.mark.parametrize(
    "planted",
    [
        "def _rival(seg):\n    return float(seg.alpha)\n",
        "def _rival(seg):\n    return getattr(seg, 'beta', 0.0)\n",
    ],
    ids=["float", "getattr-default"],
)
def test_a_planted_coefficient_reader_is_found(package_copy, planted):
    target = package_copy / PACKAGE / "geometry" / "boundary" / "_planted_rival.py"
    sites = planted_scan(coefficient_scan, package_copy, target, planted)
    assert any("_planted_rival.py" in site for site in sites), sites


# Row B2's second half (B2b-2a): the census's second predicate. ``if callable(v): v(...)``, or ``v(...) if callable(v)
# else ...``, guarding a call of the tested expression, is how D5, D16 and D22-D26 decoded a value passed into a helper.
# Any such site outside the functions below fails, whatever ``v`` is: the allowance names each one, with its class.
# Population, function scopes and blind spots as for the first predicate: module-level code is not scanned, and a
# callable test that guards no call of the same expression (``g = v if callable(v) else ...``) is not a site. Also
# unseen: a negated test whose ``else`` calls (``if not callable(g): ... else: g(x)``), an early return before the
# call, and ``callable(g) and g(x)``. No census decoder uses these shapes (review 1 of #2603).
CALLABLE_CALLS_ALLOWED = {
    # Not a decode: #2597's translator wraps a deprecated dict's callable as g(x), and the owner calls the wrapper.
    ("mfgarchon/alg/numerical/hjb_solvers/hjb_gfdm.py", "_boundary_conditions_from_dict"): (
        "wraps for the owner: #2597's dict translator"
    ),
    # Not a boundary value: a callable probe of an accessor, a getter or an operator.
    ("mfgarchon/alg/numerical/adjoint/verification.py", "verify_duality"): "non-boundary: L_hjb, L_fp_adjoint",
    ("mfgarchon/alg/numerical/coupling/fixed_point_utils.py", "read_inner_solve_failures"): "non-boundary: a method",
    ("mfgarchon/alg/numerical/coupling/multi_population_iterator.py", "__init__"): "non-boundary: a population getter",
    ("mfgarchon/alg/numerical/fp_solvers/fp_fdm_time_stepping.py", "solve_timestep_tensor_explicit"): (
        "non-boundary: a tensor field"
    ),
    ("mfgarchon/core/mfg_problem.py", "diffusion"): "non-boundary: volatility",
    ("mfgarchon/core/mfg_problem.py", "spatial_measure"): "non-boundary: get_spacing",
    ("mfgarchon/geometry/boundary/invariants.py", "mass_drift"): "non-boundary: integrate",
    ("mfgarchon/geometry/level_set/core.py", "evolve_step"): "non-boundary: a velocity",
    ("mfgarchon/utils/pde_coefficients.py", "resolve_diffusion_source"): "non-boundary: a source",
    ("mfgarchon/visualization/convergence_plots.py", "plot_from_monitor"): "non-boundary: get_plot_data",
    ("mfgarchon/workflow/common.py", "serialize_value"): "non-boundary: to_dict",
}


def _callable_targets(test: ast.AST) -> set[str]:
    return {
        ast.unparse(node.args[0])
        for node in ast.walk(test)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "callable" and node.args
    }


def callable_calls(scope: ast.AST) -> list[int]:
    """Lines in ``scope`` where a ``callable(v)`` test guards a call of ``v`` in the branch it selects."""
    lines = []
    for node in own_nodes(scope):
        if isinstance(node, (ast.If, ast.IfExp)):
            targets = _callable_targets(node.test)
            branch = node.body if isinstance(node, ast.If) else [node.body]
            called = {ast.unparse(c.func) for b in branch for c in ast.walk(b) if isinstance(c, ast.Call)}
            if targets & called:
                lines.append(node.lineno)
    return lines


def callable_scan(root: Path, allowed: Mapping[tuple[str, str], str] = CALLABLE_CALLS_ALLOWED):
    """``(population, sites)``, as `scan`, for the second predicate."""
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
                    sites.extend(f"{rel}:{line}: in {name}" for line in callable_calls(node))
    return population, sorted(set(sites))


def test_no_callable_value_is_decoded_outside_the_owner():
    population, sites = callable_scan(REPO)
    assert len(population) > 100, f"the population looks truncated: {len(population)} modules"
    assert sites == [], "a `callable(v)` test guards a call of v outside the owner (#2512 row B2):\n" + "\n".join(sites)


def test_every_callable_allowance_still_covers_a_site():
    _, sites = callable_scan(REPO, allowed={})
    covered = {(site.split(":")[0], site.rsplit(" in ", 1)[1]) for site in sites}
    assert set(CALLABLE_CALLS_ALLOWED) <= covered, sorted(set(CALLABLE_CALLS_ALLOWED) - covered)


@pytest.mark.parametrize(
    "planted",
    [
        "def _rival(bc_values, point):\n    if callable(bc_values):\n        return bc_values(point)\n    return float(bc_values)\n",
        "def _rival(g, point):\n    return g(point) if callable(g) else float(g)\n",
    ],
    ids=["if", "ifexp"],
)
def test_a_planted_callable_decode_is_found(package_copy, planted):
    target = package_copy / PACKAGE / "geometry" / "boundary" / "_planted_rival.py"
    assert any("_planted_rival.py" in site for site in planted_scan(callable_scan, package_copy, target, planted))


# Row B2's second half (B2b-2b-i): no reader gives the owner a literal time. A reader with no time passes none, and
# the owner refuses a value that declares t (#2512 row B2, #2535); a literal t = 0 was how FP-FDM read every step at
# the initial time. A site is a call of `segment_value` or `boundary_value` with ``t=`` a numeric literal, or of
# `get_bc_value_at_boundary` with ``time=`` one. Blind spots: a literal passed positionally, through a name, or
# through a helper that forwards its own parameter (FP-FDM's level-0 write passes ``time=0.0`` to `_get_bc_value`,
# which is t_0, not a default); a negative literal, ``t=-0.0``, which is a unary minus, not a constant; a call
# outside a function or lambda body, as in the other checks here; `BCSegment.get_value`'s ``t=``, which is not
# checked, since `NodeBC.get_value` shares its name; and a signature default forwarded to the owner,
# ``def f(..., time=0.0)`` passing ``t=time``, which reads t = 0 for every caller that omits it: the next check's.
_TIME_KEYWORDS = {"segment_value": "t", "boundary_value": "t", "get_bc_value_at_boundary": "time"}


def time_literals(scope: ast.AST) -> list[int]:
    """Lines in ``scope`` that hand the owner, or `get_bc_value_at_boundary`, a numeric literal time."""
    lines = []
    for node in own_nodes(scope):
        if isinstance(node, ast.Call):
            name = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", None)
            keyword = _TIME_KEYWORDS.get(name)
            for kw in node.keywords:
                if (
                    kw.arg == keyword
                    and isinstance(kw.value, ast.Constant)
                    and isinstance(kw.value.value, (int, float))
                    and not isinstance(kw.value.value, bool)
                ):
                    lines.append(node.lineno)
    return lines


def time_literal_scan(root: Path) -> tuple[list[str], list[str]]:
    population, sites = [], []
    for path in sorted((root / PACKAGE).rglob("*.py")):
        rel = path.relative_to(root).as_posix()
        if rel in EXCLUDED:
            continue
        population.append(rel)
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, FUNCTIONS):
                sites.extend(f"{rel}:{line}: in {getattr(node, 'name', '<lambda>')}" for line in time_literals(node))
    return population, sorted(set(sites))


def test_no_reader_gives_the_owner_a_literal_time():
    population, sites = time_literal_scan(REPO)
    assert len(population) > 100, f"the population looks truncated: {len(population)} modules"
    assert sites == [], "a literal time handed to the owner (#2512 row B2, #2535):\n" + "\n".join(sites)


@pytest.mark.parametrize(
    "planted",
    [
        "def _rival(segment):\n    return segment_value(segment, t=0.0, x=None, where='w')\n",
        "def _rival(bc):\n    return bc.get_bc_value_at_boundary('x_min', time=0)\n",
    ],
    ids=["owner", "accessor"],
)
def test_a_planted_literal_time_is_found(package_copy, planted):
    target = package_copy / PACKAGE / "geometry" / "boundary" / "_planted_rival.py"
    assert any("_planted_rival.py" in site for site in planted_scan(time_literal_scan, package_copy, target, planted))


# Row B2's second half (B2b-2b-iii, transitive since B2b-2b-iv): no reader gives the owner a default time, directly
# or through any chain of calls. A function whose ``t``, ``time`` or ``current_time`` defaults to a number and reaches
# the owner reads t = 0 for every caller that omits it, in silence: HJB-WENO's ghost fill relied on it at every stage
# (#2605). Such a default is ``None``, which the owner refuses for a value that declares t, or the parameter is
# required.
#
# **Predicate.** A name *reaches the owner* if it is `segment_value`, `boundary_value` or `get_bc_value_at_boundary`,
# or a function that passes one of its own time parameters, by name, to a call of a name that reaches the owner, or a
# class one of whose methods stores a time parameter on ``self`` (each such class at 375122e5 hands that attribute to
# the ghost fill or `laplacian_with_bc` when it is applied), or whose ``__init__`` passes its time on to a name that
# reaches the owner. Constructing such a class with a time is reading at that time, and ``cls(...)``,
# ``type(obj)(...)`` and ``obj.__class__(...)`` are read as constructing the class they are written in, whatever
# ``obj`` is (a false positive at worst; none at this head). A class reaches through its own ``__init__``: a
# ``super().__init__`` into a storing base counts only because the bare name ``__init__`` reaches, which is how
# `NormalDerivOperator.from_axis` is found. Nested defs and lambdas are
# entered, so a closure that forwards its outer function's time counts. A *member* is a
# function with a numeric time default that passes that parameter to a call of a name that reaches the owner. A
# *storer* is a function with a numeric time default that stores that parameter on ``self``: an operator that keeps
# its time and later hands it to the ghost fill.
#
# **Count, by this predicate.** At 375122e5 it found 43 members and 7 storers.
# - Of the 43, 31 reach the owner through functions and 12 through a class's constructor, two of them as ``cls(...)``
#   (`DiffusionOperator.from_volatility`, `NormalDerivOperator.from_axis`).
# - `applicator_graph.py` is outside the population (row N2), so its three are not among them.
# - At this head it finds only the five in `TIME_DEFAULTS_ALLOWED`.
# - The 12 private members left the population by losing the default: their time is keyword-only and required, and
#   every production caller passes it.
# - The other 26 members and the 7 storers default to ``None``: 31 definitions, since two storers' constructors also
#   forward into a third.
#
# **Blind spots:**
# - Callees are matched by bare name. A chain can pass through a same-named function that does not reach the owner;
#   that is a false positive, and the reason `applicator_graph.py`'s `NodeBC.get_value` is outside the population.
#   A same-named function that does reach the owner, called on an object that does not, is the same false positive.
# - A parameter renamed or aliased before the call (``tt = time``), passed inside a container or ``**kwargs``, or
#   through dynamic dispatch (``getattr``, a callback).
# - A time kept other than as a ``self`` attribute in the same function: in a container, a property or a later
#   method, or on another object (``op.time = time``, as `HJBFDMSolver` sets its cached Laplacian's inside a function
#   whose time is required). A ``self`` store is caught whole, annotated, or as one element of a tuple assignment.
# - A time passed on transformed (``float(time)``, ``time + dt``), which is a call of something else, or an
#   expression.
# - A subclass with no ``__init__`` of its own (``class _Sub(_Base): pass``) constructed with a time, where only the
#   base stores it: the subclass's name does not reach. There is no instance at this head.
# - A default that is not a numeric literal (a named constant, a negative literal), a time parameter with another
#   name (``t_now=0.0``), and code outside a function body.
_TIME_PARAMETERS = frozenset({"t", "time", "current_time"})
# The five public members whose time is also the Hamiltonian's, where ``None`` cannot be evaluated: #2612 deprecates
# omitting it, and removes each from here when the deprecation ends.
TIME_DEFAULTS_ALLOWED = {
    ("mfgarchon/alg/numerical/hjb_solvers/base_hjb.py", "compute_hjb_residual"): "H's time too (#2612)",
    ("mfgarchon/alg/numerical/hjb_solvers/base_hjb.py", "compute_hjb_jacobian"): "H's time too (#2612)",
    ("mfgarchon/alg/numerical/hjb_solvers/base_hjb.py", "newton_hjb_step"): "H's time too (#2612)",
    ("mfgarchon/alg/numerical/hjb_solvers/base_hjb.py", "solve_hjb_timestep_newton"): "H's time too (#2612)",
    ("mfgarchon/alg/numerical/hjb_solvers/hjb_fdm.py", "build_linearized_operator"): "H's time too (#2612)",
}


def _time_parameters(scope: ast.FunctionDef | ast.AsyncFunctionDef) -> tuple[set[str], set[str]]:
    """``scope``'s time parameters, and those of them that default to a number."""
    args = scope.args
    positional = args.posonlyargs + args.args
    names = {a.arg for a in positional + args.kwonlyargs if a.arg in _TIME_PARAMETERS}
    defaults = dict(
        zip([a.arg for a in positional[len(positional) - len(args.defaults) :]], args.defaults, strict=True)
    )
    defaults.update({a.arg: d for a, d in zip(args.kwonlyargs, args.kw_defaults, strict=True) if d is not None})
    numeric = {
        name
        for name, default in defaults.items()
        if name in names
        and isinstance(default, ast.Constant)
        and isinstance(default.value, (int, float))
        and not isinstance(default.value, bool)
    }
    return names, numeric


def _callee(func: ast.AST, owner_class: str | None) -> str | None:
    """The bare name a call reaches. Inside a class, ``cls(...)``, ``type(self)(...)`` and ``self.__class__(...)``
    construct the enclosing class, so they reach its name."""
    if owner_class is not None and (
        (isinstance(func, ast.Name) and func.id == "cls")
        or (isinstance(func, ast.Call) and isinstance(func.func, ast.Name) and func.func.id == "type")
        or (isinstance(func, ast.Attribute) and func.attr == "__class__")
    ):
        return owner_class
    return func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)


def _callees_given(scope: ast.AST, names: set[str], owner_class: str | None = None) -> set[str]:
    """The bare names of the calls in ``scope``, nested scopes included, that are passed one of ``names``."""
    callees = set()
    for node in ast.walk(scope):
        if isinstance(node, ast.Call):
            passed = list(node.args) + [kw.value for kw in node.keywords]
            if any(isinstance(a, ast.Name) and a.id in names for a in passed):
                callee = _callee(node.func, owner_class)
                if callee is not None:
                    callees.add(callee)
    return callees


def _is_self_attribute(target: ast.AST) -> bool:
    return isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and target.value.id == "self"


def _stores_on_self(scope: ast.AST, names: set[str]) -> bool:
    """``scope`` stores one of ``names`` on a ``self`` attribute: ``self.x = time``, ``self.x: float = time``, or one
    element of a tuple assignment, ``self.bc, self.x = bc, time``."""
    for node in ast.walk(scope):
        if isinstance(node, ast.AnnAssign) and _is_self_attribute(node.target):
            pairs = [(node.target, node.value)]
        elif isinstance(node, ast.Assign):
            pairs = [(t, node.value) for t in node.targets]
            pairs += [
                (t, v)
                for t in node.targets
                if isinstance(t, ast.Tuple) and isinstance(node.value, ast.Tuple)
                for t, v in zip(t.elts, node.value.elts, strict=False)
            ]
        else:
            continue
        if any(_is_self_attribute(t) and isinstance(v, ast.Name) and v.id in names for t, v in pairs):
            return True
    return False


def time_default_scan(
    root: Path, allowed: Mapping[tuple[str, str], str] = TIME_DEFAULTS_ALLOWED
) -> tuple[list[str], list[str]]:
    """``(population, sites)``: the members and storers outside ``allowed``, each ``path:line: in function``."""
    population, timed, storing_classes, initializers = [], [], set(), []
    for path in sorted((root / PACKAGE).rglob("*.py")):
        rel = path.relative_to(root).as_posix()
        if rel in EXCLUDED:
            continue
        population.append(rel)
        tree = ast.parse(path.read_text())
        owner = {
            id(method): node.name
            for node in ast.walk(tree)
            if isinstance(node, ast.ClassDef)
            for method in node.body
            if isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                names, numeric = _time_parameters(node)
                if names:
                    callees = _callees_given(node, names, owner.get(id(node)))
                    timed.append((rel, node, names, numeric, callees, owner.get(id(node))))
                    if node.name == "__init__" and id(node) in owner:
                        initializers.append((owner[id(node)], callees))
                    if id(node) in owner and _stores_on_self(node, names):
                        storing_classes.add(owner[id(node)])
    # A class reaches the owner when one of its methods stores a time on `self`, or its `__init__` passes a time on
    # to a name that reaches it; constructing it, `cls(...)` included, then reads at the time it is given.
    reaches = set(_TIME_KEYWORDS) | storing_classes
    grew = True
    while grew:
        before = len(reaches)
        reaches |= {node.name for _, node, _, _, callees, _ in timed if callees & reaches}
        reaches |= {cls for cls, callees in initializers if callees & reaches}
        grew = len(reaches) > before
    sites = []
    for rel, node, _, numeric, _, owner_class in timed:
        if not numeric or (rel, node.name) in allowed:
            continue
        if _callees_given(node, numeric, owner_class) & reaches:
            sites.append(f"{rel}:{node.lineno}: in {node.name}")
        if _stores_on_self(node, numeric):
            sites.append(f"{rel}:{node.lineno}: in {node.name} (stores it)")
    return population, sorted(set(sites))


def test_no_reader_gives_the_owner_a_default_time():
    population, sites = time_default_scan(REPO)
    assert len(population) > 100, f"the population looks truncated: {len(population)} modules"
    assert sites == [], (
        f"{len(sites)} numeric default time(s) handed to the owner, directly or through a chain of calls, or stored "
        "(#2512 row B2, #2605):\n" + "\n".join(sites)
    )


def test_the_time_default_allowance_is_exactly_its_members():
    """The allowance names the five members left, by the predicate's own count; one that no longer reaches the owner,
    or has lost its default, must leave it, so the count of members falls with #2612."""
    _, sites = time_default_scan(REPO, allowed={})
    members = {(site.split(":")[0], site.rsplit(" in ", 1)[1]) for site in sites}
    assert members == set(TIME_DEFAULTS_ALLOWED), (
        f"the predicate finds {len(members)} member(s); the allowance names {len(TIME_DEFAULTS_ALLOWED)}",
        sorted(members ^ set(TIME_DEFAULTS_ALLOWED)),
    )


@pytest.mark.parametrize(
    "planted",
    [
        "def _rival(segment, time=0.0):\n    return segment_value(segment, t=time, x=None, where='w')\n",
        "def _rival(bc, t=0):\n    return bc.get_bc_value_at_boundary('x_min', t)\n",
        "def _rival(bc, time=0.0):\n    return _rival_middle(bc, time)\n\n\n"
        "def _rival_middle(bc, time):\n    return _rival_inner(bc, time=time)\n\n\n"
        "def _rival_inner(bc, time):\n    return segment_value(bc.segments[0], t=time, x=None, where='w')\n",
        "def _rival(u, bc, time=0.0):\n    def step(v):\n        return pad_array_with_ghosts(v, bc, time=time)\n"
        "    return step(u)\n",
        "class _Rival:\n    def __init__(self, bc, time=0.0):\n        self.bc, self.time = bc, None\n"
        "        self.time = time\n",
        "class _RivalOperator:\n    def __init__(self, bc, time=None):\n        self.time = time\n\n\n"
        "def _rival(bc, time=0.0):\n    return _RivalOperator(bc, time=time)\n",
        "class _RivalOperator:\n    def __init__(self, bc, time=None):\n        self.time = time\n\n"
        "    @classmethod\n    def make(cls, bc, time=0.0):\n        return cls(bc, time=time)\n",
        "class _Rival:\n    def __init__(self, bc, time=0.0):\n        self.time: float = time\n",
        "class _Rival:\n    def __init__(self, bc, time=0.0):\n        self.bc, self.time = bc, time\n",
    ],
    ids=[
        "owner",
        "accessor",
        "two calls deep",
        "a closure",
        "a storer",
        "into a storer",
        "cls() constructor",
        "annotated store",
        "tuple store",
    ],
)
def test_a_planted_default_time_is_found(package_copy, planted):
    target = package_copy / PACKAGE / "geometry" / "boundary" / "_planted_rival.py"
    assert any("_planted_rival.py" in site for site in planted_scan(time_default_scan, package_copy, target, planted))
