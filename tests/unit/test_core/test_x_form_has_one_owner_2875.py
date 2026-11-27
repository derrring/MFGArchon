"""#2875: the form of the ``x`` a body is handed has one owner, ``types/callable_protocols.py`` (the approver's rulings in
#2875 comments 7216531026 and 7216681260).

The owner builds that ``x`` as `Coordinates`, a tuple subclass, and `BoundCallable` refuses any other tuple. A public
method a caller calls with its own ``x`` relays it in the caller's form, and those relay sites are named below.

What this file reads, by AST over ``mfgarchon/``:
- only the owner module constructs `Coordinates`;
- no ``x`` a call is handed is a tuple or list its caller builds itself: a display, a comprehension, a ``tuple(...)`` or
  ``list(...)`` call, a call of a package function that returns one it builds, or a name bound to one of these in the
  function or in one enclosing it. That covers ``x=``, the ``"x"`` entry of a dict a call unpacks with ``**`` (inline,
  or bound to a name, in an if-expression too), and the argument handed by position to a parameter named ``x`` of a
  package function whose name is unique; a lambda's body is read as its function's;
- the owner is not handed such a tuple to mark, since `coordinate_tuple` and `caller_coordinates` take a tuple as it is;
- each named relay site hands on the ``x`` it was given, through the owner where it reaches a bound body, and builds
  none, and each still relays.

It cannot see an ``x`` handed to a parameter named otherwise (`base_hjb`'s refused rows go to ``rows``), a method name
several classes share, or a tuple built two calls away. Those doors are held by the binder's refusal where the body is
bound, and by ``test_a_control_has_ps_shape_2875.py::test_each_read_hands_x_in_its_owners_form`` where it is a
Hamiltonian's method.
Each scan runs on planted sources first, which it must flag.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

import numpy as np

import mfgarchon
from mfgarchon.core.hamiltonian import HamiltonianBase
from mfgarchon.types import callable_protocols as cp

PACKAGE = Path(mfgarchon.__file__).resolve().parent
OWNER = "types/callable_protocols.py"
BUILT = (ast.Tuple, ast.List, ast.ListComp, ast.GeneratorExp)
MARKS_AS_IT_IS = {"coordinate_tuple", "caller_coordinates"}

# Where a caller's own x reaches a body: each hands on what it was given (the approver's ruling of 2026-11-27)
RELAYS = {
    "core/hamiltonian.py": [
        "CongestionHamiltonian.optimal_control",
        "DualHamiltonian.__call__",
        "DualHamiltonian.__call__.neg_objective",
        "DualHamiltonian._find_optimal_alpha",
        "DualHamiltonian._find_optimal_alpha.neg_objective",
        "DualLagrangian.__call__",
        "DualLagrangian.__call__.neg_objective",
        "DualLagrangian._find_optimal_p",
        "DualLagrangian._find_optimal_p.neg_objective",
        "DualLagrangian.dm",
        "HamiltonianBase._finite_diff_dm",
        "HamiltonianBase._finite_diff_dp",
        "HamiltonianBase._finite_diff_dx",
        "HamiltonianBase.dm",
        "HamiltonianBase.dp",
        "HamiltonianBase.jacobian_fd",
        "HamiltonianBase.optimal_control",
        "LagrangianBase.conjugate_argmax.neg_objective",
        "LagrangianBase.conjugate_argmax.neg_objective_nd",
        "LagrangianBase.evaluate_hamiltonian",
        "LagrangianBase.optimal_control",
        "LagrangianBase.proximal.objective",
        "LagrangianBase.proximal.objective_nd",
        "SeparableHamiltonian.dx",
        "SeparableLagrangian.__call__",
        "SeparableLagrangian.evaluate_hamiltonian",
    ],
}


def _name(call: ast.Call) -> str:
    return getattr(call.func, "id", getattr(call.func, "attr", ""))


def _functions(tree: ast.AST):
    """Every function, with its dotted name inside its classes and functions, and the functions enclosing it."""

    def walk(node, stack, enclosing):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                yield ".".join([*stack, child.name]), child, enclosing
                yield from walk(child, [*stack, child.name], [child, *enclosing])
            elif isinstance(child, ast.ClassDef):
                yield from walk(child, [*stack, child.name], enclosing)
            else:
                yield from walk(child, stack, enclosing)

    yield from walk(tree, [], [])


def _own_nodes(fn: ast.AST):
    """The nodes of ``fn`` outside the functions and classes defined in it; a lambda's body is read as ``fn``'s."""
    for child in ast.iter_child_nodes(fn):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        yield child
        yield from _own_nodes(child)


def _bound_to(fn: ast.AST, name: str) -> list[ast.AST]:
    return [
        node.value
        for node in _own_nodes(fn)
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets)
    ]


class PackageIndex:
    """What a call site's callee does with what it is handed, across the package: the functions that return a tuple they
    build (outside the owner), and each uniquely named function's parameters."""

    def __init__(self, sources: dict[str, str]):
        defs: dict[str, list[tuple[str, ast.FunctionDef]]] = {}
        for path, source in sources.items():
            for _, fn, _ in _functions(ast.parse(source)):
                defs.setdefault(fn.name, []).append((path, fn))
        unique = {name: found[0] for name, found in defs.items() if len(found) == 1}
        self.builders = {
            name
            for name, (path, fn) in unique.items()
            if path != OWNER
            and any(
                isinstance(r, ast.Return) and r.value is not None and _built(r.value, set()) for r in _own_nodes(fn)
            )
        }
        self.params = {name: [a.arg for a in (*fn.args.posonlyargs, *fn.args.args)] for name, (_, fn) in unique.items()}


def _built(node: ast.AST, builders: set[str]) -> bool:
    if isinstance(node, ast.IfExp):
        return _built(node.body, builders) or _built(node.orelse, builders)
    return isinstance(node, BUILT) or (isinstance(node, ast.Call) and _name(node) in {"tuple", "list", *builders})


def _bindings(chain: list[ast.AST], name: str) -> list[ast.AST]:
    """What ``name`` is bound to in the innermost of ``chain`` (a function and those enclosing it) that binds it."""
    for fn in chain:
        bound = _bound_to(fn, name)
        if bound:
            return bound
    return []


def _built_here(value: ast.AST, chain: list[ast.AST], builders: set[str]) -> bool:
    if _built(value, builders):
        return True
    return isinstance(value, ast.Name) and any(_built(v, builders) for v in _bindings(chain, value.id))


def _x_by_position(call: ast.Call, index: PackageIndex) -> ast.AST | None:
    """The argument ``call`` hands by position to a parameter named x, for a callee the index names uniquely."""
    params = index.params.get(_name(call))
    if params is None:
        return None
    if isinstance(call.func, ast.Attribute) and params[:1] in (["self"], ["cls"]):
        params = params[1:]
    for argument, param in zip(call.args, params, strict=False):
        if param == "x" and not isinstance(argument, ast.Starred):
            return argument
    return None


def deciding_sites(source: str, path: str, index: PackageIndex | None = None) -> list[str]:
    """The places in ``source`` that decide x's form outside its owner. ``index`` names the package's tuple-building
    helpers and its functions' parameters; by default, those ``source`` itself defines."""
    if path == OWNER:
        return []
    tree = ast.parse(source)
    index = index or PackageIndex({path: source})
    builders = index.builders
    found = []
    scopes = [([], tree)] + [([fn, *enclosing], fn) for _, fn, enclosing in _functions(tree)]
    for chain, scope in scopes:
        for node in _own_nodes(scope):
            if not isinstance(node, ast.Call):
                continue
            where = f"{path}:{node.lineno}"
            if _name(node) == "Coordinates":
                found.append(f"{where} constructs Coordinates")
            if _name(node) in MARKS_AS_IT_IS and node.args and _built_here(node.args[0], chain, builders):
                found.append(f"{where} has {_name(node)} mark a tuple it built")
            for keyword in node.keywords:
                if keyword.arg == "x" and _built_here(keyword.value, chain, builders):
                    found.append(f"{where} hands x as a tuple it built: {ast.unparse(keyword.value)[:60]}")
                if keyword.arg is None:
                    unpacked = keyword.value
                    held = _bindings(chain, unpacked.id) if isinstance(unpacked, ast.Name) else [unpacked]
                    for value in held:
                        for dict_node in (n for n in ast.walk(value) if isinstance(n, ast.Dict)):
                            for key, entry in zip(dict_node.keys, dict_node.values, strict=True):
                                if (
                                    isinstance(key, ast.Constant)
                                    and key.value == "x"
                                    and _built_here(entry, chain, builders)
                                ):
                                    found.append(f"{where} holds x as a tuple it built: {ast.unparse(entry)[:60]}")
            positional = _x_by_position(node, index)
            if positional is not None and _built_here(positional, chain, builders):
                found.append(f"{where} hands x by position as a tuple it built: {ast.unparse(positional)[:60]}")
    return found


def relay_defects(source: str, names: list[str]) -> list[str]:
    """For each named relay site in ``source``: missing, relaying nothing, or handing on an x it built."""
    functions = {name: fn for name, fn, _ in _functions(ast.parse(source))}
    defects = []
    for name in names:
        fn = functions.get(name)
        if fn is None:
            defects.append(f"{name}: no such function")
            continue
        handed = [
            k.value for node in _own_nodes(fn) if isinstance(node, ast.Call) for k in node.keywords if k.arg == "x"
        ]
        if not handed:
            defects.append(f"{name}: relays no x")
        for value in handed:
            if isinstance(value, ast.Name) and value.id == "x":
                continue
            if isinstance(value, ast.Call) and _name(value) == "caller_coordinates":
                continue
            bound = _bound_to(fn, value.id) if isinstance(value, ast.Name) else []
            if bound and all(isinstance(v, ast.Call) and _name(v) == "moved" for v in bound):
                continue
            defects.append(f"{name}: hands on x as {ast.unparse(value)[:60]}")
    return defects


@pytest.mark.parametrize(
    ("planted", "flagged"),
    [
        ("Coordinates((1.0, 2.0))", "constructs Coordinates"),
        ("body(x=(a, b))", "hands x as a tuple it built"),
        ("body(x=tuple(p))", "hands x as a tuple it built"),
        ("body(x=[a[k] for k in range(2)])", "hands x as a tuple it built"),
        ("def door(p):\n    pts = tuple(p)\n    return body(x=pts)", "hands x as a tuple it built"),
        ('body(**{"t": 0.0, "x": tuple(q)})', "holds x as a tuple it built"),
        (
            'def door(t, q):\n    slots = {"t": t, "x": tuple(q)}\n    return body(**slots)',
            "holds x as a tuple it built",
        ),
        (
            'def door(t, q):\n    slots = {"t": t} if q is None else {"t": t, "x": tuple(q)}\n    return body(**slots)',
            "holds x",
        ),
        ("def points(g):\n    return tuple(g)\n\n\nbody(x=points(q))", "hands x as a tuple it built"),
        (
            "def read(o, x, t):\n    return o(x=x, t=t)\n\n\ndef door(p):\n    x = tuple(p)\n    return read(f, x, 0.0)",
            "by position",
        ),
        (
            "def read(o, x):\n    return o(x=x)\n\n\ndef door(p):\n    x = tuple(p)\n    return lambda: read(f, x)",
            "by position",
        ),
        (
            "def prox(x=None):\n    x = tuple(z) if x is None else x\n\n    def objective(a):\n        return body(x=x)\n\n    return objective",
            "hands x as a tuple it built",
        ),
        ("coordinate_tuple(tuple(p), 2)", "has coordinate_tuple mark a tuple it built"),
        ("caller_coordinates((a, b))", "has caller_coordinates mark a tuple it built"),
    ],
)
def test_the_scan_flags_a_planted_construction(planted, flagged):
    found = deciding_sites(planted, "a/door.py")
    assert len(found) == 1, found
    assert flagged in found[0], found


def test_the_scan_passes_what_the_owner_builds_and_a_name_handed_on():
    clean = (
        "body(x=coordinate_tuple(p, 2))\nbody(x=point_tuple(q))\nbody(x=line_coordinates(s))\nbody(x=x)\n"
        'frame.agg({"x": ["mean", "std"]})\n'
    )
    assert deciding_sites(clean, "a/door.py") == []
    assert deciding_sites("Coordinates((1.0,))", OWNER) == []


def test_the_relay_check_flags_a_relay_that_builds_and_one_that_relays_nothing():
    source = (
        "class H:\n"
        "    def dp(self, x):\n        return self(x=tuple(x))\n"
        "    def dm(self, x):\n        return 0.0\n"
        "    def ok(self, x):\n        return self(x=x)\n"
    )
    assert relay_defects(source, ["H.dp", "H.dm", "H.ok", "H.gone"]) == [
        "H.dp: hands on x as tuple(x)",
        "H.dm: relays no x",
        "H.gone: no such function",
    ]


def test_only_the_owner_decides_the_form_of_x():
    sources = {str(path.relative_to(PACKAGE)): path.read_text() for path in sorted(PACKAGE.rglob("*.py"))}
    index = PackageIndex(sources)
    found = []
    for path, source in sources.items():
        found += deciding_sites(source, path, index)
    assert found == [], "\n".join(found)


@pytest.mark.parametrize("path", sorted(RELAYS))
def test_each_relay_site_hands_on_the_x_it_was_given(path):
    assert relay_defects((PACKAGE / path).read_text(), RELAYS[path]) == []


def test_the_binder_refuses_a_tuple_its_door_built():
    """The binder reads x the owner built and refuses a tuple a door built itself, by name. Mutant: the refusal of an
    unmarked tuple removed."""
    bound = cp.bind_user_callable(lambda x: x[0], cp.REGION_SLOTS, role="r")
    assert bound(x=cp.point_tuple([0.25, 0.5])) == 0.25
    with pytest.raises(TypeError, match=r"a tuple it built itself as x, not one from coordinate_tuple"):
        bound(x=(0.25, 0.5))


class _Recorder(HamiltonianBase):
    def __init__(self):
        super().__init__()
        self.seen = []

    def __call__(self, t, x, p, m):
        self.seen.append(type(x))
        return float(np.sum(np.asarray(x, dtype=float) ** 2))


@pytest.mark.parametrize(
    ("x", "form"),
    [
        (np.array([0.3, 0.6]), np.ndarray),
        ((0.3, 0.6), tuple),
        (cp.point_tuple([0.3, 0.6]), cp.Coordinates),
    ],
    ids=["array", "plain tuple", "owner's coordinates"],
)
def test_a_finite_difference_hands_on_the_callers_form(x, form):
    """The approver's ruling of 2026-11-27: a point the library derives from the caller's x keeps the caller's form, an
    array an array of its shape and a tuple a tuple of its kind. Mutant: the perturbed point built as a plain tuple, as
    before, which turns an array into a tuple."""
    H = _Recorder()
    grad = H._finite_diff_dx(x, m=1.0, p=np.zeros(2), t=0.0)
    np.testing.assert_allclose(grad, [0.6, 1.2], rtol=1e-5)
    assert set(H.seen) == {form}
