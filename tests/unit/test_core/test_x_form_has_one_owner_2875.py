"""#2875: the form of the ``x`` the library builds from its own points and hands a user body in reach has one owner,
``types/callable_protocols.py`` (plan v2, #2875 comment 7219629907, approved in 7219680642).

What this file reads, by AST over ``mfgarchon/`` outside the owner module. An *x position* is:
- the ``x=`` argument of a call, the ``"x"`` entry of a dict a call unpacks with ``**`` (inline, bound to a name, or in
  an if-expression), and the ``"x"`` entry of a dict display handed by position to a binder's ``check_on``;
- the first argument, by position, of `coordinate_tuple` and `point_at`, which take a tuple as it is, and of
  `point_tuple`, which rebuilds a tuple's entries as floats;
- an argument handed, by position or by name, to a parameter named ``x`` or to a parameter that carries an x on: one its
  function hands, as it is, to an x position. The carry is followed through calls, closures and the arguments a
  ``functools.partial`` is made with, to a fixpoint.

No x position holds a value built outside the owner: a tuple or list display, a comprehension, a ``tuple(...)`` or
``list(...)`` call, ``np.meshgrid(...)``, the ``next(...)`` of a ``zip(...)`` or ``itertools.product(...)``, a call of a
package function that returns one of these, a call of one that returns one of its own arguments (followed to that
argument), or a name bound to any of these by a plain or an annotated assignment in the function or one enclosing it. At
the owner's first arguments only a tuple counts, since the owner reads a list as an array. The first argument of
`coordinate_tuple` and `point_at` is also the enclosing function's own parameter, or an allowance in ``OWNER_ARGUMENTS``
names its expression; a tuple built outside the owner fails there before any allowance is read, so an allowance that
names a variable does not cover a tuple bound to it. A lambda's body is read as its function's.

A caller's own ``x``, handed to a public method, is outside the fact and reaches the body as main hands it. Those relay
sites are named in ``RELAYS``: each hands on its own parameter, or a form main gives it, named with its expression in
``DERIVED`` (a value built from it, which the scan flags) or ``RELAY_FORMS`` (the owner called on it, or an arithmetic
step from it). An allowance in ``OWNER_ARGUMENTS``, ``DERIVED`` or ``RELAY_FORMS`` that covers nothing fails.

What the scan cannot follow: a callee named through ``getattr`` or found in a registry; a function or method name that
several definitions share (followed neither as a callee nor as a helper); the arguments a ``functools.partial`` object is
later called with; ``*args`` and ``**kwargs`` forwarding; a class's constructor; a value kept on an attribute; a carrier
parameter its own function rebinds; a lambda bound to a name and called as a carrier; a name bound by a tuple target, a
``for`` target or a walrus; a tuple-returning call outside the package other than those named above; a dict handed to
``check_on`` by a name or by keyword; and the owner's first argument passed by keyword.
The raw Hamiltonian doors are also held at runtime, by
``test_a_control_has_ps_shape_2875.py::test_each_read_hands_x_in_its_owners_form``.
Each rule runs on planted sources first, which it must flag.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

import pytest

import mfgarchon

PACKAGE = Path(mfgarchon.__file__).resolve().parent
OWNER = "types/callable_protocols.py"
BUILT = (ast.Tuple, ast.List, ast.ListComp, ast.GeneratorExp)
AS_IT_IS = {"coordinate_tuple", "point_at"}
REBUILDS = {"point_tuple"}
OWNER_FUNCTIONS = {*AS_IT_IS, *REBUILDS, "grid_coordinates", "line_coordinates"}
TUPLE_ITERATORS = {"zip", "product"}

# Where a caller's own x reaches a body (the approver's rulings in 7216681260, 7219629907 and on review 2)
RELAYS = {
    "alg/optimization/variational_problem.py": [
        "VariationalMFGProblem._setup_jax_functions.jax_lagrangian",
        "VariationalMFGProblem._setup_jax_functions.jax_terminal_cost",
        "VariationalMFGProblem.compute_lagrangian_derivatives",
        "VariationalMFGProblem.convert_to_hamiltonian.hamiltonian_dm",
        "VariationalMFGProblem.evaluate_lagrangian",
        "VariationalMFGProblem.evaluate_terminal_cost",
    ],
    "core/hamiltonian.py": [
        "CongestionHamiltonian._value",
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
        "SeparableHamiltonian._value",
        "SeparableHamiltonian.dx",
        "SeparableLagrangian.__call__",
        "SeparableLagrangian.evaluate_hamiltonian",
    ],
    "core/mfg_problem.py": [
        "MFGProblem.__init__.vola_value",
        "MFGProblem.diffusion.diffusion",
        "MFGProblem._compose_state_penalty_into_potential.composed_potential",
    ],
    "core/stochastic/stochastic_problem.py": [
        "StochasticMFGProblem.H_conditional",
        "StochasticMFGProblem.g_conditional",
        "StochasticMFGProblem.create_conditional_problem._FrozenNoiseHamiltonian.__call__",
    ],
    "utils/drift_helpers.py": ["composite_drift._composite_drift"],
}

# The points main builds from a caller's x at a relay site, by the value handed and its expression: the finite-difference
# steps, and a point on a line taken as the one-entry tuple (x,)
_ON_A_LINE = {"(x,)": "(x,)"}
_STEPS = {"x_plus": "(*x[:i], x[i] + eps, *x[i + 1:])", "x_minus": "(*x[:i], x[i] - eps, *x[i + 1:])"}
DERIVED = {
    "alg/optimization/variational_problem.py": {
        "VariationalMFGProblem._setup_jax_functions.jax_lagrangian": _ON_A_LINE,
        "VariationalMFGProblem._setup_jax_functions.jax_terminal_cost": _ON_A_LINE,
        "VariationalMFGProblem.compute_lagrangian_derivatives": _ON_A_LINE,
        "VariationalMFGProblem.convert_to_hamiltonian.hamiltonian_dm": _ON_A_LINE,
        "VariationalMFGProblem.evaluate_lagrangian": _ON_A_LINE,
        "VariationalMFGProblem.evaluate_terminal_cost": _ON_A_LINE,
    },
    "core/hamiltonian.py": {"HamiltonianBase._finite_diff_dx": _STEPS, "SeparableHamiltonian.dx": _STEPS},
}

# Main's other forms at a relay site, which the scan does not flag, by the value handed and its expression: the owner
# called on the caller's x (a batch read point by point, the dimension stated, the stand-in where none is given), and an
# arithmetic step from it
_AT_A_POINT = {"point_at(x, i)": "point_at(x, i)"}
_STAND_IN = {"x": "point_tuple(np.zeros(max(d, 1)))"}
RELAY_FORMS = {
    "alg/optimization/variational_problem.py": {
        "VariationalMFGProblem.compute_lagrangian_derivatives": {"x + eps": "x + eps", "x - eps": "x - eps"},
    },
    "core/hamiltonian.py": {
        "HamiltonianBase.dm": _AT_A_POINT,
        "HamiltonianBase.dp": _AT_A_POINT,
        "LagrangianBase.proximal.objective": _STAND_IN,
        "LagrangianBase.proximal.objective_nd": _STAND_IN,
    },
    "core/stochastic/stochastic_problem.py": {
        "StochasticMFGProblem.g_conditional": {"x": "coordinate_tuple(x, self.dimension)"},
    },
}

# Arguments of coordinate_tuple and point_at that are not the enclosing function's own parameter, by expression: each an
# array of the library's own points, the owner's own output (_call_problem_source), or the caller's own x on the line
# that rebinds it to the owner's (g_conditional)
OWNER_ARGUMENTS = {
    "alg/numerical/fem/fp_fem_solver.py": {"FPFEMSolver._build_advection": {"np.moveaxis(x_qp, 0, -1)"}},
    "alg/numerical/fp_solvers/fp_fvm.py": {"FPFVMSolver._face_velocity_from_potential": {"seam"}},
    "alg/numerical/fp_solvers/fp_particle.py": {"FPParticleSolver._solve_fp_system_callable_drift": {"particles_t"}},
    "alg/numerical/hjb_solvers/base_hjb.py": {"_refuse_a_non_finite_hjb_term": {"points[bad]"}},
    "alg/numerical/hjb_solvers/hjb_gfdm.py": {"HJBGFDMSolver._solve_backward_howard": {"_pts"}},
    "alg/numerical/hjb_solvers/hjb_howard.py": {"HJBHowardSolver._howard_step": {"pts"}},
    "alg/numerical/meshless_galerkin/fp_solver.py": {
        "MeshlessGalerkinFPSolver._build_advection": {"self._disc.dof_coordinates"}
    },
    "alg/numerical/weak_form_hjb_solver.py": {
        "WeakFormHJBSolver._solve_timestep_newton": {"x_grid"},
        "WeakFormHJBSolver.solve_hjb_system": {"self._disc.dof_coordinates"},
    },
    "core/hamiltonian.py": {"_state_x": {"state.x"}},
    "core/mfg_components.py": {"ConditionsMixin._condition_on_grid": {"grid"}},
    "core/mfg_problem.py": {"_call_problem_source": {"coordinates"}},
    "core/stochastic/stochastic_problem.py": {"StochasticMFGProblem.g_conditional": {"x"}},
    "operators/interaction/energy_functionals.py": {"potential_reader.read": {"points"}},
    "utils/manufactured.py": {"_drift_coefficient": {"np.zeros_like(probes)"}, "fp_source.source": {"pts"}},
    "utils/pde_coefficients.py": {"points_of": {"np.stack(grid, axis=-1)"}},
    "utils/validation/components.py": {"_evaluate_callable_on_grid": {"points"}},
    "utils/validation/functions.py": {
        "_read_source": {"np.asarray(x, dtype=float)"},
        "_refuse_a_non_finite_source": {"probe"},
    },
}


@dataclass(frozen=True)
class Site:
    path: str
    line: int
    function: str
    value: str
    derivation: str
    what: str

    def __str__(self) -> str:
        return f"{self.path}:{self.line} in {self.function or '<module>'} {self.what}: {self.value}"


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


def _params(fn: ast.AST) -> list[str]:
    args = fn.args
    return [a.arg for a in (*args.posonlyargs, *args.args, *args.kwonlyargs)]


def _positional(fn: ast.AST) -> list[str]:
    return [a.arg for a in (*fn.args.posonlyargs, *fn.args.args)]


def _bound_to(fn: ast.AST, name: str) -> list[ast.AST]:
    """What a plain or an annotated assignment in ``fn`` binds ``name`` to."""
    bound = []
    for node in _own_nodes(fn):
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            bound.append(node.value)
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            if isinstance(node.target, ast.Name) and node.target.id == name:
                bound.append(node.value)
    return bound


def _bindings(chain: list[ast.AST], name: str) -> list[ast.AST]:
    """What ``name`` is bound to in the innermost of ``chain`` (a function and those enclosing it) that binds it or takes
    it as a parameter; nothing, for a parameter it does not rebind."""
    for fn in chain:
        bound = _bound_to(fn, name)
        if bound or name in _params(fn):
            return bound
    return []


def _parameter_of(chain: list[ast.AST], name: str) -> ast.AST | None:
    """The function in ``chain`` whose parameter ``name`` is, where no function inside it rebinds the name."""
    for fn in chain:
        if _bound_to(fn, name):
            return None
        if name in _params(fn):
            return fn
    return None


def _derivation(value: ast.AST, chain: list[ast.AST]) -> str:
    """The expression ``value`` stands for: what a name is bound to, or the value itself."""
    bound = _bindings(chain, value.id) if isinstance(value, ast.Name) else []
    return " | ".join(sorted({ast.unparse(v) for v in bound})) if bound else ast.unparse(value)


class PackageIndex:
    """What a call's callee does with what it is handed, across the package outside the owner: the functions that return
    a value they build, those that return one of their parameters as it is, and the parameters that carry an x on, with
    the kind of x position each reaches ("x", or "owner" for `point_tuple`'s first argument)."""

    def __init__(self, sources: dict[str, str]):
        self.scopes = []
        defs: dict[str, list[ast.AST]] = {}
        for path, source in sources.items():
            if path == OWNER:
                continue
            tree = ast.parse(source)
            self.scopes.append((path, "", [], tree))
            for qual, fn, enclosing in _functions(tree):
                self.scopes.append((path, qual, [fn, *enclosing], fn))
                defs.setdefault(fn.name, []).append(fn)
        self.unique = {name: found[0] for name, found in defs.items() if len(found) == 1}
        returns = {
            name: [r.value for r in _own_nodes(fn) if isinstance(r, ast.Return) and r.value is not None]
            for name, fn in self.unique.items()
        }
        self.passes = {
            name: {v.id for v in values if isinstance(v, ast.Name) and _parameter_of([fn], v.id) is fn}
            for name, values in returns.items()
            if (fn := self.unique[name])
        }
        self.builders: set[str] = set()
        self.carriers: dict[str, dict[str, str]] = {}
        changed = True
        while changed:
            changed = False
            for name, values in returns.items():
                fn = self.unique[name]
                if name not in self.builders and any(self.built(v, [fn], tuples_only=False) for v in values):
                    self.builders.add(name)
                    changed = True
            for _, _, chain, scope in self.scopes:
                for node in _own_nodes(scope):
                    if not isinstance(node, ast.Call):
                        continue
                    for value, kind in self.x_positions(node, chain):
                        holder = _parameter_of(chain, value.id) if isinstance(value, ast.Name) else None
                        if holder is None or self.unique.get(holder.name) is not holder:
                            continue
                        kind = "x" if kind == "as it is" else kind
                        carried = self.carriers.setdefault(holder.name, {})
                        if carried.get(value.id) not in ("x", kind):
                            carried[value.id] = "x" if "x" in (kind, carried.get(value.id)) else kind
                            changed = True

    def _kind(self, callee: str, param: str) -> str | None:
        return "x" if param == "x" else self.carriers.get(callee, {}).get(param)

    def x_positions(self, call: ast.Call, chain: list[ast.AST]):
        """Each value ``call`` hands to an x position, with the kind of that position: "x", "owner" (`point_tuple`'s
        first argument) or "as it is" (the first argument of `coordinate_tuple` or `point_at`)."""
        name = _name(call)
        dicts = []
        for keyword in call.keywords:
            if keyword.arg == "x":
                yield keyword.value, "x"
            elif keyword.arg is None:
                unpacked = keyword.value
                dicts += _bindings(chain, unpacked.id) if isinstance(unpacked, ast.Name) else [unpacked]
        if name == "check_on" and call.args:
            dicts.append(call.args[0])
        for value in dicts:
            for dict_node in (n for n in ast.walk(value) if isinstance(n, ast.Dict)):
                for key, entry in zip(dict_node.keys, dict_node.values, strict=True):
                    if isinstance(key, ast.Constant) and key.value == "x":
                        yield entry, "x"
        if name in AS_IT_IS | REBUILDS and call.args and not isinstance(call.args[0], ast.Starred):
            yield call.args[0], "as it is" if name in AS_IT_IS else "owner"
        args, method = call.args, isinstance(call.func, ast.Attribute)
        if name == "partial" and call.args:
            target = call.args[0]
            name = getattr(target, "id", getattr(target, "attr", ""))
            args, method = call.args[1:], isinstance(target, ast.Attribute)
        callee = self.unique.get(name)
        if callee is None:
            return
        positional = _positional(callee)
        if method and positional[:1] in (["self"], ["cls"]):
            positional = positional[1:]
        for argument, param in zip(args, positional, strict=False):
            if isinstance(argument, ast.Starred):
                break
            kind = self._kind(name, param)
            if kind:
                yield argument, kind
        for keyword in call.keywords:
            if keyword.arg not in (None, "x"):
                kind = self._kind(name, keyword.arg)
                if kind:
                    yield keyword.value, kind

    def built(self, value: ast.AST, chain: list[ast.AST], *, tuples_only: bool, depth: int = 0) -> bool:
        """Whether ``value`` is built outside the owner: a tuple only, where ``tuples_only``."""
        if depth > 8:
            return False
        if isinstance(value, ast.IfExp):
            return self.built(value.body, chain, tuples_only=tuples_only, depth=depth + 1) or self.built(
                value.orelse, chain, tuples_only=tuples_only, depth=depth + 1
            )
        if isinstance(value, ast.Tuple) or (not tuples_only and isinstance(value, BUILT)):
            return True
        if isinstance(value, ast.Name):
            return any(
                self.built(v, chain, tuples_only=tuples_only, depth=depth + 1) for v in _bindings(chain, value.id)
            )
        if not isinstance(value, ast.Call):
            return False
        name = _name(value)
        if name in ("tuple", "meshgrid") or (name == "list" and not tuples_only) or name in self.builders:
            return True
        if name == "next" and value.args and isinstance(value.args[0], ast.Call):
            return _name(value.args[0]) in TUPLE_ITERATORS
        callee = self.unique.get(name)
        passed = self.passes.get(name, set())
        if callee is None or not passed:
            return False
        positional = _positional(callee)
        if isinstance(value.func, ast.Attribute) and positional[:1] in (["self"], ["cls"]):
            positional = positional[1:]
        handed = [a for a, p in zip(value.args, positional, strict=False) if p in passed]
        handed += [k.value for k in value.keywords if k.arg in passed]
        return any(self.built(a, chain, tuples_only=tuples_only, depth=depth + 1) for a in handed)


def deciding_sites(source: str, path: str, index: PackageIndex | None = None) -> list[Site]:
    """The places in ``source`` that decide x's form outside its owner. ``index`` reads the package's helpers and
    carriers; by default, those ``source`` itself defines."""
    if path == OWNER:
        return []
    tree = ast.parse(source)
    index = index or PackageIndex({path: source})
    found = []
    scopes = [("", [], tree)] + [(qual, [fn, *enclosing], fn) for qual, fn, enclosing in _functions(tree)]
    for qual, chain, scope in scopes:
        for node in _own_nodes(scope):
            if not isinstance(node, ast.Call):
                continue
            for value, kind in index.x_positions(node, chain):
                if kind == "as it is" and index.built(value, chain, tuples_only=True):
                    what = "hands its owner a tuple it built"
                elif kind == "as it is":
                    own = isinstance(value, ast.Name) and chain and _parameter_of(chain[:1], value.id) is chain[0]
                    if own:
                        continue
                    what = "hands its owner a value that is not its own parameter"
                elif index.built(value, chain, tuples_only=kind == "owner"):
                    what = "hands its owner a tuple it built" if kind == "owner" else "hands x as a value it built"
                else:
                    continue
                found.append(Site(path, node.lineno, qual, ast.unparse(value), _derivation(value, chain), what))
    return found


def against_allowances(
    found: list[Site], derived: dict[str, dict[str, dict[str, str]]], arguments: dict[str, dict[str, set[str]]]
) -> tuple[list[Site], list[str]]:
    """The sites no allowance covers, and the allowances that cover no site: ``derived`` by the value handed and its
    expression, ``arguments`` by the owner's argument."""
    covered = set()
    left = []
    for site in found:
        if site.what.startswith("hands its owner a value that is not"):
            if site.value in arguments.get(site.path, {}).get(site.function, set()):
                covered.add((site.path, site.function, site.value))
                continue
        elif derived.get(site.path, {}).get(site.function, {}).get(site.value) == site.derivation:
            covered.add((site.path, site.function, site.value))
            continue
        left.append(site)
    allowed = [(p, f, v) for p, fs in derived.items() for f, vs in fs.items() for v in vs]
    allowed += [(p, f, v) for p, fs in arguments.items() for f, vs in fs.items() for v in vs]
    unused = [f"{p}: {f} hands no {v}" for p, f, v in sorted(allowed) if (p, f, v) not in covered]
    return left, unused


def relay_defects(
    source: str,
    path: str,
    names: list[str],
    index: PackageIndex | None = None,
    derived: dict[str, dict[str, dict[str, str]]] | None = None,
    forms: dict[str, dict[str, dict[str, str]]] | None = None,
) -> tuple[list[str], set[tuple[str, str, str]]]:
    """For each named relay site in ``source``: missing, relaying nothing, or handing on anything but its own parameter or
    a point an allowance names by its expression. Returns the defects and the ``forms`` allowances it used."""
    index = index or PackageIndex({path: source})
    derived = DERIVED if derived is None else derived
    forms = RELAY_FORMS if forms is None else forms
    functions = {qual: [fn, *enclosing] for qual, fn, enclosing in _functions(ast.parse(source))}
    defects, used = [], set()
    for name in names:
        chain = functions.get(name)
        if chain is None:
            defects.append(f"{name}: no such function")
            continue
        handed = [
            value
            for node in _own_nodes(chain[0])
            if isinstance(node, ast.Call)
            for value, _ in index.x_positions(node, chain)
        ]
        if not handed:
            defects.append(f"{name}: relays no x")
        for value in handed:
            text, expression = ast.unparse(value), _derivation(value, chain)
            if isinstance(value, ast.Name) and _parameter_of(chain, value.id) is not None:
                continue
            if derived.get(path, {}).get(name, {}).get(text) == expression:
                continue
            if forms.get(path, {}).get(name, {}).get(text) == expression:
                used.add((path, name, text))
                continue
            defects.append(f"{name}: hands on x as {text}" + (f" = {expression}" if expression != text else ""))
    return defects, used


def _package_sources() -> dict[str, str]:
    return {str(path.relative_to(PACKAGE)): path.read_text() for path in sorted(PACKAGE.rglob("*.py"))}


@pytest.mark.parametrize(
    ("planted", "flagged"),
    [
        ("body(x=(a, b))", "hands x as a value it built"),
        ("body(x=tuple(p))", "hands x as a value it built"),
        ("body(x=[a[k] for k in range(2)])", "hands x as a value it built"),
        ("def door(p):\n    pts = tuple(p)\n    return body(x=pts)", "hands x as a value it built"),
        ("def door(p):\n    pts: tuple = tuple(p)\n    return body(x=pts)", "hands x as a value it built"),
        ("body(x=np.meshgrid(a, b))", "hands x as a value it built"),
        ("body(x=next(zip(a, b)))", "hands x as a value it built"),
        ('body(**{"t": 0.0, "x": tuple(q)})', "hands x as a value it built"),
        ('bound.check_on({"t": 0.0, "x": (grid,)})', "hands x as a value it built"),
        (
            'def door(t, q):\n    slots = {"t": t, "x": tuple(q)}\n    return body(**slots)',
            "hands x as a value it built",
        ),
        (
            'def door(t, q):\n    slots = {"t": t} if q is None else {"t": t, "x": tuple(q)}\n    return body(**slots)',
            "hands x as a value it built",
        ),
        ("def points(g):\n    return tuple(g)\n\n\nbody(x=points(q))", "hands x as a value it built"),
        (
            "def read(o, x, t):\n    return o(x=x, t=t)\n\n\ndef door(p):\n    x = tuple(p)\n    return read(f, x, 0.0)",
            "hands x as a value it built",
        ),
        (
            "def read(o, x):\n    return o(x=x)\n\n\ndef door(p):\n    x = tuple(p)\n    return lambda: read(f, x)",
            "hands x as a value it built",
        ),
        (
            "def prox(x=None):\n    x = tuple(z) if x is None else x\n\n    def objective(a):\n        return body(x=x)\n\n    return objective",
            "hands x as a value it built",
        ),
        ("point_tuple(tuple(p))", "hands its owner a tuple it built"),
        ("coordinate_tuple(np.meshgrid(a, b), 2)", "hands its owner a tuple it built"),
        ("def door(s):\n    pts = tuple(s)\n    return coordinate_tuple(pts, 2)", "hands its owner a tuple it built"),
        ("def door(p):\n    q = p\n    return point_at(q, 0)", "hands its owner a value that is not its own parameter"),
        # The owner handed a tuple through a helper that returns its argument (review 1's mut_launder)
        (
            "def launder(v):\n    return v\n\n\ndef door(p):\n    built = tuple(p)\n    return point_tuple(launder(built))",
            "hands its owner a tuple it built",
        ),
        # The carry through a plain call: rows goes to a parameter named otherwise, which hands it on as x
        (
            "def terms(o, rows):\n    return o(x=rows)\n\n\ndef door(p):\n    rows = tuple(p)\n    return terms(f, rows)",
            "hands x as a value it built",
        ),
        # The carry through functools.partial: rows is bound to a parameter that hands it on as x
        (
            "def read(o, at):\n    return o(x=at)\n\n\ndef terms(o, rows):\n    return partial(read, o, rows)\n\n\n"
            "def door(p):\n    rows = tuple(p)\n    return terms(f, rows)",
            "hands x as a value it built",
        ),
        # The carry into the owner: a parameter handed to coordinate_tuple carries, so its caller's tuple is read there
        (
            "def wrap(points, d):\n    return coordinate_tuple(points, d)\n\n\ndef door(p):\n    return wrap(tuple(p), 2)",
            "hands x as a value it built",
        ),
    ],
)
def test_the_scan_flags_a_planted_construction(planted, flagged):
    found = deciding_sites(planted, "a/door.py")
    assert len(found) == 1, [str(s) for s in found]
    assert flagged in found[0].what, str(found[0])


def test_the_scan_passes_what_the_owner_builds_and_a_name_handed_on():
    clean = (
        "def door(p, q, s, x):\n    body(x=coordinate_tuple(p, 2))\n    body(x=point_tuple(q))\n"
        "    body(x=line_coordinates(s))\n    body(x=x)\n\n\n"
        "point_tuple([a, b])\n"
        'frame.agg({"x": ["mean", "std"]})\n'
        "def wrap(p, d):\n    return coordinate_tuple(p, d)\n\n\n"
        "def at(x, i):\n    return point_at(x, i)\n\n\n"
        "def terms(o, rows):\n    return o(x=rows)\n\n\ndef caller(p, d):\n    return terms(f, coordinate_tuple(p, d))\n"
    )
    assert deciding_sites(clean, "a/door.py") == []
    assert deciding_sites("body(x=(1.0,))", OWNER) == []


def test_an_allowance_that_covers_nothing_fails():
    source = "def step(x):\n    x_plus = (x[0] + 1.0,)\n    return body(x=x_plus)\n\n\ndef rows(g):\n    return point_at(g.T, 0)"
    found = deciding_sites(source, "a/door.py")
    derived = {"a/door.py": {"step": {"x_plus": "(x[0] + 1.0,)"}}}
    arguments = {"a/door.py": {"rows": {"g.T"}}}
    assert against_allowances(found, derived, arguments) == ([], [])
    other_step = {"a/door.py": {"step": {"x_plus": "(x[0] - 1.0,)"}}}
    assert [s.value for s in against_allowances(found, other_step, arguments)[0]] == ["x_plus"]
    left, unused = against_allowances(
        found, {"a/door.py": {"step": {"x_plus": "(x[0] + 1.0,)", "x_minus": "(x[0] - 1.0,)"}}}, {"a/door.py": {}}
    )
    assert [s.value for s in left] == ["g.T"]
    assert unused == ["a/door.py: step hands no x_minus"]
    # An allowance that names a variable does not cover a tuple bound to it (review 3's MF-1)
    built = deciding_sites("def door(s):\n    pts = tuple(s)\n    return coordinate_tuple(pts, 2)", "a/door.py")
    left, unused = against_allowances(built, {}, {"a/door.py": {"door": {"pts"}}})
    assert [(s.value, s.what) for s in left] == [("pts", "hands its owner a tuple it built")]
    assert unused == ["a/door.py: door hands no pts"]


def test_the_relay_check_flags_a_relay_that_builds_converts_or_relays_nothing():
    source = (
        "class H:\n"
        "    def dp(self, x):\n        return self(x=tuple(x))\n"
        "    def dm(self, x):\n        return 0.0\n"
        "    def ok(self, x):\n        return self(x=x)\n"
        "    def through(self, x):\n        return self(x=point_tuple(x))\n"
        "    def step(self, x):\n        x_plus = (x[0] + 1.0,)\n        return self(x=x_plus)\n"
        "    def at(self, x, i):\n        return self(x=point_at(x, i))\n"
    )
    names = ["H.dp", "H.dm", "H.ok", "H.through", "H.step", "H.at", "H.gone"]
    allowed = {"a/h.py": {"H.step": {"x_plus": "(x[0] - 1.0,)"}}}
    forms = {"a/h.py": {"H.at": {"point_at(x, i)": "point_at(x, i)"}}}
    defects, used = relay_defects(source, "a/h.py", names, derived=allowed, forms=forms)
    assert defects == [
        "H.dp: hands on x as tuple(x)",
        "H.dm: relays no x",
        "H.through: hands on x as point_tuple(x)",
        "H.step: hands on x as x_plus = (x[0] + 1.0,)",
        "H.gone: no such function",
    ]
    assert used == {("a/h.py", "H.at", "point_at(x, i)")}


def test_only_the_owner_decides_the_form_of_x():
    sources = _package_sources()
    index = PackageIndex(sources)
    found = [site for path, source in sources.items() for site in deciding_sites(source, path, index)]
    left, unused = against_allowances(found, DERIVED, OWNER_ARGUMENTS)
    assert left == [], "\n".join(str(s) for s in left)
    assert unused == [], "\n".join(unused)


def test_each_relay_site_hands_on_the_x_it_was_given():
    sources = _package_sources()
    index = PackageIndex(sources)
    defects, used = [], set()
    for path, names in RELAYS.items():
        found, hit = relay_defects(sources[path], path, names, index)
        defects += found
        used |= hit
    unused = [
        f"{p}: {f} hands no {v}"
        for p, fs in RELAY_FORMS.items()
        for f, vs in fs.items()
        for v in vs
        if (p, f, v) not in used
    ]
    assert defects == [], "\n".join(defects)
    assert unused == [], "\n".join(unused)
