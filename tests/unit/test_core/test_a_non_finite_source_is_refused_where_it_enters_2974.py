"""#2974: a caller-supplied value or callable that is non-finite where the solvers will read it is refused where it
enters, by name.

`_refuse_a_non_finite_source` reads once, where `MFGProblem` is built: the arrays and scalars whole, and each callable at
the state the solve is certain to read, x the probe nodes and t at 0, T/2 and T, with m `m_initial`'s values and v
`u_terminal`'s. A scalar held by an object that exists before any problem (a control cost, a kernel, a boundary segment,
`QuadraticMFGHamiltonian`) is refused where that object is built, and a volatility override handed to a solver by
`resolve_volatility_override`.

A Hamiltonian's own `__call__` is read at one state by `validate_hamiltonian`: t 0, its sample node, p 0, and m
`m_initial`'s value there. The owner reads that node too, so a source the Hamiltonian holds that is non-finite at that
state is refused by the owner first, by name, on a grid whose middle index is not its centre as well.

What is not refused: a source finite at those reads and non-finite elsewhere (a coupling NaN only at m = 0.5 or at m = 1,
where `m_initial` takes neither, builds), an input no solver reads (`coupling_dm`), and an infinity that means something
(`BoundedControlCost(max_control=inf)`, unbounded).

Every value parameter of a `HamiltonianBase` or `LagrangianBase` subclass is declared beside the class: a source the owner
reads (`HELD_SOURCES`), a scalar its constructor refuses (`CHECKED_SCALARS`), or neither, with why (`NOT_HELD`). The last
tests read every such class in the library from its source, nested ones included.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import pathlib
import re
import warnings

import pytest

import numpy as np
from scipy.sparse.linalg import LinearOperator

import mfgarchon
from mfgarchon import Conditions, MFGProblem, Model
from mfgarchon.alg.numerical.fp_solvers import FPFDMSolver
from mfgarchon.alg.numerical.hjb_solvers import HJBFDMSolver
from mfgarchon.core.coupling_costs import PotentialCost
from mfgarchon.core.hamiltonian import (
    BoundedControlCost,
    CongestionHamiltonian,
    QuadraticControlCost,
    QuadraticMFGHamiltonian,
    SeparableHamiltonian,
    SeparableLagrangian,
    _value_parameter,
)
from mfgarchon.core.mfg_components import MFGComponents
from mfgarchon.extensions.topology import NetworkMFGComponents, NetworkMFGProblem
from mfgarchon.geometry import Hyperrectangle, TensorProductGrid
from mfgarchon.geometry import boundary as bnd
from mfgarchon.geometry.graph.network_geometry import GridNetwork
from mfgarchon.operators.interaction import kernels
from mfgarchon.utils.validation import ValidationError

NAN, INF = float("nan"), float("inf")
_POINTS = {1: 11, 2: 36}
# `_rising` is 1 at no node by design, so its mass is not 1 and the problem says so; the mass is not these pins' subject.
_MASS_IS_NOT_THE_SUBJECT = pytest.mark.filterwarnings("ignore:initial density mass:UserWarning")


def _grid(dim, bc=None):
    return TensorProductGrid(
        bounds=[(0.0, 1.0)] * dim,
        Nx_points=[11] if dim == 1 else [6, 6],
        boundary_conditions=bc or bnd.no_flux_bc(dimension=dim),
    )


def _u_terminal(x):
    value = np.sin(np.pi * np.asarray(x[0], dtype=float))
    return value * np.cos(np.pi * np.asarray(x[1], dtype=float)) if len(x) > 1 else value


def _h(**sources):
    return SeparableHamiltonian(QuadraticControlCost(control_cost=1.0), **sources)


def _problem(dim, *, model=None, m_initial=None, u_terminal=None, T=0.5, bc=None, **problem_kw):
    return MFGProblem(
        model=Model(**{"hamiltonian": _h(), "volatility": 0.2, **(model or {})}),
        domain=_grid(dim, bc),
        conditions=Conditions(m_initial=m_initial or (lambda x: 1.0), u_terminal=u_terminal or _u_terminal, T=T),
        Nt=5,
        **problem_kw,
    )


def _components_problem(dim, *, m_initial=None, u_terminal=None, T=0.5, lagrangian=None, **problem_kw):
    return MFGProblem(
        geometry=_grid(dim),
        T=T,
        Nt=5,
        components=MFGComponents(
            **({"lagrangian": lagrangian} if lagrangian is not None else {"hamiltonian": _h()}),
            m_initial=(lambda x: 1.0) if m_initial is None else m_initial,
            u_terminal=_u_terminal if u_terminal is None else u_terminal,
        ),
        **problem_kw,
    )


def _network_problem(T=0.5, **components):
    network = GridNetwork(width=3, height=3)
    network.create_network()
    return NetworkMFGProblem(geometry=network, T=T, Nt=5, components=NetworkMFGComponents(**components))


def _at_x(value):
    return lambda x: np.full(np.shape(np.asarray(x[0], dtype=float)), value)


def _at_tx(value):
    return lambda t, x: np.full(np.shape(np.asarray(x[0], dtype=float)), value)


def _of_m(value):
    return lambda m: np.full(np.shape(m), value)


def _wall(value):
    return lambda x: np.full(np.shape(x)[1:], value) if np.ndim(x) >= 2 else value


def _source(value):
    return lambda t, x, v, m: np.full(np.shape(m), value)


def _nan_only_at(value):
    """A coupling NaN where m is ``value`` and finite at every other m."""
    return lambda m: np.where(np.isclose(m, value), NAN, 0.5 * np.asarray(m, dtype=float))


def _rising(x):
    """An initial density that is 1 nowhere and takes one value at one node of each grid below: 2 + x on [0, 1], and
    2 + x + 7y on the unit square, unique at the 6 x 6 grid's middle index, (0.6, 0)."""
    return 2.0 + np.asarray(x[0], dtype=float) + (7.0 * np.asarray(x[1], dtype=float) if len(x) > 1 else 0.0)


def _nonlocal(dim, value):
    n = _POINTS[dim]
    return LinearOperator((n, n), matvec=lambda v: np.full(np.shape(v), value), dtype=float)


def _read(name):
    """The owner's refusal of one source, by its name."""
    return rf"{re.escape(name)} is non-finite where the solvers read it"


# (id, build(dim), exception, what the refusal names). One row per source the owner reads on a grid, at 1-D and 2-D.
GRID_REFUSALS = [
    (
        "potential NaN",
        lambda d: _problem(d, model={"hamiltonian": _h(potential=_at_tx(NAN))}),
        ValidationError,
        _read("SeparableHamiltonian's potential"),
    ),
    (
        "potential +inf",
        lambda d: _problem(d, model={"hamiltonian": _h(potential=_at_tx(INF))}),
        ValidationError,
        _read("SeparableHamiltonian's potential"),
    ),
    (
        "coupling NaN",
        lambda d: _problem(d, model={"hamiltonian": _h(coupling=_of_m(NAN))}),
        ValidationError,
        _read("SeparableHamiltonian's coupling"),
    ),
    (
        "coupling NaN only at m_initial at validate_hamiltonian's node",
        lambda d: _problem(
            d, m_initial=_rising, model={"hamiltonian": _h(coupling=_nan_only_at(2.5 if d == 1 else 2.6))}
        ),
        ValidationError,
        _read("SeparableHamiltonian's coupling"),
    ),
    (
        "a Lagrangian's potential NaN",
        lambda d: _components_problem(
            d, lagrangian=SeparableLagrangian(QuadraticControlCost(control_cost=1.0), potential=_at_tx(NAN))
        ),
        ValidationError,
        _read("SeparableLagrangian's potential"),
    ),
    (
        "QuadraticMFGHamiltonian potential NaN",
        lambda d: _problem(d, model={"hamiltonian": QuadraticMFGHamiltonian(potential=_at_tx(NAN))}),
        ValidationError,
        _read("QuadraticMFGHamiltonian's potential"),
    ),
    (
        "congestion_factor NaN",
        lambda d: _problem(
            d,
            model={
                "hamiltonian": CongestionHamiltonian(
                    QuadraticControlCost(control_cost=1.0), congestion_factor=_of_m(NAN)
                )
            },
        ),
        ValidationError,
        _read("CongestionHamiltonian's congestion_factor"),
    ),
    ("volatility NaN", lambda d: _problem(d, model={"volatility": NAN}), ValidationError, _read("volatility")),
    ("volatility +inf", lambda d: _problem(d, model={"volatility": INF}), ValidationError, _read("volatility")),
    (
        "volatility callable NaN",
        lambda d: _problem(d, model={"volatility": _at_x(NAN)}),
        ValidationError,
        _read("volatility"),
    ),
    (
        "volatility array NaN",
        lambda d: _problem(
            d, model={"volatility": np.full((11,) if d == 1 else (6, 6), NAN), "volatility_kind": "field"}
        ),
        ValidationError,
        _read("volatility"),
    ),
    (
        "coupling_cost NaN",
        lambda d: _problem(d, model={"coupling_cost": PotentialCost(_at_x(NAN))}),
        ValidationError,
        _read("coupling_cost"),
    ),
    (
        "terminal_coupling +inf",
        lambda d: _problem(d, model={"terminal_coupling": PotentialCost(_at_x(INF))}),
        ValidationError,
        _read("terminal_coupling"),
    ),
    ("m_initial NaN", lambda d: _problem(d, m_initial=_at_x(NAN)), ValidationError, _read("m_initial")),
    ("u_terminal +inf", lambda d: _problem(d, u_terminal=_at_x(INF)), ValidationError, _read("u_terminal")),
    ("T NaN", lambda d: _problem(d, T=NAN), ValidationError, _read("T")),
    ("T +inf", lambda d: _problem(d, T=INF), ValidationError, _read("T")),
    (
        "source_term_hjb NaN",
        lambda d: _problem(d, source_term_hjb=_source(NAN)),
        ValidationError,
        _read("source_term_hjb"),
    ),
    (
        "source_term_fp NaN",
        lambda d: _problem(d, source_term_fp=_source(NAN)),
        ValidationError,
        _read("source_term_fp"),
    ),
    ("state_penalty NaN", lambda d: _problem(d, state_penalty=_wall(NAN)), ValidationError, _read("state_penalty")),
    (
        "state_penalty_scale +inf",
        lambda d: _problem(d, state_penalty=_wall(1.0), state_penalty_scale=INF),
        ValidationError,
        _read("state_penalty_scale"),
    ),
    (
        "nonlocal_operator NaN",
        lambda d: _problem(d, nonlocal_operator=_nonlocal(d, NAN)),
        ValidationError,
        _read("nonlocal_operator"),
    ),
    (
        "components m_initial array NaN",
        lambda d: _components_problem(d, m_initial=np.full((11,) if d == 1 else (6, 6), NAN)),
        ValidationError,
        _read("m_initial"),
    ),
    (
        "components u_terminal array NaN",
        lambda d: _components_problem(d, u_terminal=np.full((11,) if d == 1 else (6, 6), NAN)),
        ValidationError,
        _read("u_terminal"),
    ),
    (
        "components diffusion +inf",
        lambda d: _components_problem(d, diffusion=INF),
        ValidationError,
        _read("volatility"),
    ),
    (
        "a Dirichlet value g(t) NaN",
        lambda d: _problem(d, bc=bnd.dirichlet_bc(value=lambda t: NAN, dimension=d)),
        NotImplementedError,
        r"BCSegment 'uniform' \(what its callable returned at t=0\) has a non-finite value nan",
    ),
    (
        "a Dirichlet value NaN",
        lambda d: _problem(d, bc=bnd.dirichlet_bc(value=NAN, dimension=d)),
        NotImplementedError,
        r"BCSegment 'uniform' has a non-finite value nan",
    ),
    (
        "a Robin alpha NaN",
        lambda d: _problem(d, bc=bnd.robin_bc(value=0.0, alpha=NAN, beta=1.0, dimension=d)),
        NotImplementedError,
        r"BCSegment 'uniform' \(its alpha\) has a non-finite value nan",
    ),
    (
        "a flux value +inf",
        lambda d: _problem(d, bc=bnd.flux_bc(value=INF, dimension=d)),
        NotImplementedError,
        r"BCSegment 'uniform' has a non-finite value inf",
    ),
]


@_MASS_IS_NOT_THE_SUBJECT
@pytest.mark.parametrize("dim", [1, 2])
@pytest.mark.parametrize(
    ("build", "error", "names"), [row[1:] for row in GRID_REFUSALS], ids=[row[0] for row in GRID_REFUSALS]
)
def test_a_non_finite_source_on_a_grid_is_refused_by_name(dim, build, error, names):
    with pytest.raises(error, match=names):
        build(dim)


NETWORK_REFUSALS = [
    (
        "node_potential_func NaN",
        {"node_potential_func": lambda t, node: NAN},
        _read("NetworkHamiltonian's node_potential_func"),
    ),
    (
        "node_potential_func +inf",
        {"node_potential_func": lambda t, node: INF},
        _read("NetworkHamiltonian's node_potential_func"),
    ),
    (
        "node_interaction_func NaN",
        {"node_interaction_func": lambda t, node, m: NAN},
        _read("NetworkHamiltonian's node_interaction_func"),
    ),
    (
        "hamiltonian_func NaN",
        {
            "hamiltonian_func": lambda t, node, neighbors, p, m: NAN,
            "hamiltonian_dm_func": lambda t, node, neighbors, p, m: 0.0,
        },
        _read("NetworkHamiltonian's hamiltonian_func"),
    ),
    ("T NaN", {"T": NAN}, _read("T")),
]


@pytest.mark.parametrize(
    ("given", "names"), [row[1:] for row in NETWORK_REFUSALS], ids=[row[0] for row in NETWORK_REFUSALS]
)
def test_a_non_finite_source_on_a_network_is_refused_by_name(given, names):
    """Read in the node form the network solvers hand: an integer node, and m and p one entry per node."""
    with pytest.raises(ValidationError, match=names):
        _network_problem(**given)


@pytest.mark.parametrize(
    ("solve", "given", "names"),
    [
        (
            "hjb",
            NAN,
            r"HJB-FDM \(solve_hjb_system_backward\): a scalar volatility must be finite and non-negative, got nan",
        ),
        (
            "hjb",
            INF,
            r"HJB-FDM \(solve_hjb_system_backward\): a scalar volatility must be finite and non-negative, got inf",
        ),
        ("fp", NAN, r"FPFDMSolver\.solve_fp_system: a scalar volatility must be finite and non-negative, got nan"),
        ("fp", np.where(np.arange(11) == 4, NAN, 0.2), r"the volatility= array holds 1 non-finite values of 11"),
    ],
    ids=["HJB scalar NaN", "HJB scalar +inf", "FP scalar NaN", "FP array, one NaN"],
)
def test_a_non_finite_volatility_override_is_refused_by_name(solve, given, names):
    """The volatility a solver is handed at the solve, beside the problem's (#2618)."""
    problem = _problem(1)
    if solve == "hjb":
        run = lambda: HJBFDMSolver(problem).solve_hjb_system(  # noqa: E731
            np.ones((6, 11)), np.zeros(11), np.zeros((6, 11)), volatility=given
        )
    else:
        run = lambda: FPFDMSolver(problem).solve_fp_system(  # noqa: E731
            np.ones(11), drift_field=np.zeros((6, 11)), volatility=given
        )
    with pytest.raises(ValueError, match=names):
        run()


def test_a_finite_mass_is_read_whole():
    """Every entry finite, the total infinite: the mass check reads the total, a state built from the entries, and
    refuses it (`MFGProblem`'s tier 1). On [0, 10] the measure is 10, so entries of 1e308 integrate past the largest
    double; on [0, 1] the total is at most the largest entry, and finite."""
    domain = TensorProductGrid(bounds=[(0.0, 10.0)], Nx_points=[11], boundary_conditions=bnd.no_flux_bc(dimension=1))
    conditions = Conditions(m_initial=lambda x: 1e308, u_terminal=_u_terminal, T=0.5)
    with pytest.raises(ValueError, match=r"m_initial has total mass inf"):
        MFGProblem(model=Model(hamiltonian=_h(), volatility=0.2), domain=domain, conditions=conditions, Nt=5)


def test_a_bounded_control_refuses_a_nan_bound():
    with pytest.raises(ValueError, match=r"max_control must be positive, got nan"):
        BoundedControlCost(control_cost=1.0, max_control=NAN)


KERNEL_PARAMETERS = [
    (kernel, name)
    for kernel in (
        kernels.GaussianKernel,
        kernels.TentKernel,
        kernels.WendlandKernel,
        kernels.DipoleKernel,
        kernels.PowerLawKernel,
    )
    for name, parameter in inspect.signature(kernel).parameters.items()
    if isinstance(parameter.default, float)
]


@pytest.mark.parametrize(("kernel", "name"), KERNEL_PARAMETERS, ids=[f"{k.__name__}.{n}" for k, n in KERNEL_PARAMETERS])
def test_a_kernel_refuses_a_non_finite_parameter_by_name(kernel, name):
    with pytest.raises(ValueError, match=rf"{kernel.__name__}: {name} must be finite, got nan"):
        kernel(**{name: NAN})


def _coupling_nan_at_a_half(m):
    return np.where(np.isclose(m, 0.5), NAN, 0.5 * m)


def _on_an_implicit_domain():
    """A ``Hyperrectangle`` draws its points at each read, so no node pairs with its entry in ``m_initial`` and the
    callables are not read there (`_draws_its_nodes`)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # the density's mass on this measure is not this pin's subject
        return MFGProblem(
            model=Model(hamiltonian=_h(potential=_at_tx(1.0)), volatility=0.2, coupling_cost=PotentialCost(_at_x(1.0))),
            domain=Hyperrectangle(np.array([[0.0, 1.0]])),
            conditions=Conditions(m_initial=lambda x: 1.0, u_terminal=lambda x: 0.0, T=0.5),
            Nt=4,
        )


@_MASS_IS_NOT_THE_SUBJECT
@pytest.mark.parametrize(
    "build",
    [
        lambda: _problem(1),
        lambda: _problem(2),
        lambda: _network_problem(),
        lambda: _problem(
            1, model={"hamiltonian": SeparableHamiltonian(BoundedControlCost(control_cost=1.0, max_control=INF))}
        ),
        lambda: _problem(1, model={"hamiltonian": _h(coupling=_coupling_nan_at_a_half)}),
        lambda: _problem(1, m_initial=_rising, model={"hamiltonian": _h(coupling=_nan_only_at(1.0))}),
        lambda: _problem(1, model={"hamiltonian": _h(coupling=_of_m(0.0), coupling_dm=_of_m(NAN))}),
        lambda: _problem(
            1, model={"volatility": _at_x(0.2)}, bc=bnd.dirichlet_bc(value=lambda t: 1.0 + t, dimension=1)
        ),
        _on_an_implicit_domain,
    ],
    ids=[
        "1-D",
        "2-D",
        "network",
        "max_control +inf, unbounded",
        "coupling NaN only at m = 0.5, off m_initial",
        "coupling NaN only at m = 1, off m_initial",
        "coupling_dm NaN, which no grid solver reads",
        "finite callables",
        "an implicit domain",
    ],
)
def test_a_source_finite_where_it_is_read_builds(build):
    build()


# --- Every value parameter of a Hamiltonian or a Lagrangian is declared (#2974's amendment A). ---

_ROOTS = ("HamiltonianBase", "LagrangianBase")
_DECLARATIONS = ("HELD_SOURCES", "CHECKED_SCALARS", "NOT_HELD")
_UNKNOWN = object()


def _declared_names(statement, where):
    """The names one declaration in a class body lists, which must be a literal: a dict's keys, or a tuple's items."""
    value = statement.value
    if isinstance(value, ast.Dict):
        items = value.keys
    elif isinstance(value, (ast.Tuple, ast.List, ast.Set)):
        items = value.elts
    else:
        raise AssertionError(f"{where}: declare it as a literal, so that it can be read from the source")
    names = [item.value for item in items if isinstance(item, ast.Constant) and isinstance(item.value, str)]
    assert len(names) == len(items), f"{where}: every entry is named by a string literal"
    return set(names)


def _classes(sources):
    """Every class in ``sources`` ({path: text}), nested ones included: by name, its bases' names, its own constructor's
    parameters as (name, annotation, default), or None without one, and the names each declaration in its body lists."""
    found = {}
    for path, text in sources.items():
        for node in ast.walk(ast.parse(text)):
            if not isinstance(node, ast.ClassDef):
                continue
            declared = {kind: set() for kind in _DECLARATIONS}
            parameters = None
            for statement in node.body:
                target = (
                    statement.target
                    if isinstance(statement, ast.AnnAssign)
                    else (
                        statement.targets[0]
                        if isinstance(statement, ast.Assign) and len(statement.targets) == 1
                        else None
                    )
                )
                if isinstance(target, ast.Name) and target.id in _DECLARATIONS:
                    declared[target.id] = _declared_names(statement, f"{path}: {node.name}.{target.id}")
                if isinstance(statement, ast.FunctionDef) and statement.name == "__init__":
                    arguments = statement.args
                    positional = arguments.posonlyargs + arguments.args
                    defaults = [None] * (len(positional) - len(arguments.defaults)) + arguments.defaults
                    pairs = list(zip(positional[1:], defaults[1:], strict=True)) + list(
                        zip(arguments.kwonlyargs, arguments.kw_defaults, strict=True)
                    )
                    parameters = [
                        (
                            argument.arg,
                            ast.unparse(argument.annotation) if argument.annotation else "",
                            _literal(default),
                        )
                        for argument, default in pairs
                    ]
            bases = tuple(ast.unparse(base).rsplit(".", 1)[-1] for base in node.bases)
            found.setdefault(node.name, []).append(
                {"where": f"{path}:{node.lineno}", "bases": bases, "parameters": parameters, "declared": declared}
            )
    return found


def _literal(default):
    if default is None:
        return inspect.Parameter.empty
    try:
        return ast.literal_eval(default)
    except ValueError:
        return _UNKNOWN


def _one(found, name):
    entries = found.get(name, [])
    assert len(entries) <= 1, f"{name} is defined {len(entries)} times: {[e['where'] for e in entries]}"
    return entries[0] if entries else None


def _ancestry(found, name):
    """``name`` and the classes it derives from that ``found`` holds, nearest first."""
    order, pending = [], [name]
    while pending:
        current = pending.pop(0)
        entry = _one(found, current)
        if entry is None or current in order:
            continue
        order.append(current)
        pending.extend(entry["bases"])
    return order


def _operators(found):
    """The roots and every class that derives from one, by its bases' names."""
    derived = set(_ROOTS)
    while True:
        more = {name for name, entries in found.items() if any(set(e["bases"]) & derived for e in entries)} - derived
        if not more:
            return sorted(derived & set(found))
        derived |= more


def _undeclared(found):
    """(class, parameter, why) for each value parameter of an operator class that no declaration along its ancestry
    lists, each name listed by two kinds of declaration, and each name a class declares that its own constructor does
    not take."""
    problems = []
    for name in _operators(found):
        ancestry = _ancestry(found, name)
        constructor = next(
            (_one(found, a)["parameters"] for a in ancestry if _one(found, a)["parameters"] is not None), []
        )
        merged = {kind: set().union(*(_one(found, a)["declared"][kind] for a in ancestry)) for kind in _DECLARATIONS}
        for parameter, annotation, default in constructor:
            kinds = [kind for kind in _DECLARATIONS if parameter in merged[kind]]
            if not kinds and _value_parameter(annotation, default):
                problems.append((name, parameter, "undeclared"))
            if len(kinds) > 1:
                problems.append((name, parameter, f"declared as {' and '.join(kinds)}"))
        taken = {parameter for parameter, _, _ in constructor}
        for kind, names in _one(found, name)["declared"].items():
            problems.extend((name, stale, f"in {kind}, not a parameter") for stale in sorted(names - taken))
    return problems


def _library():
    root = pathlib.Path(mfgarchon.__file__).parent
    return _classes({str(path.relative_to(root.parent)): path.read_text() for path in sorted(root.rglob("*.py"))})


def test_every_value_parameter_of_an_operator_is_declared():
    """A class that gains a number, an array or a callable among its constructor's parameters declares it: read by the
    owner, refused by the constructor, or not held, with why. Undeclared, the owner would not read it, and a non-finite
    one would reach the solve. The classification is `_value_parameter`'s, beside the declarations."""
    found = _library()
    operators = _operators(found)
    # The population: classes in several modules, one defined inside a function.
    assert {
        "SeparableHamiltonian",
        "QuadraticMFGHamiltonian",
        "DualLagrangian",
        "NetworkHamiltonian",
        "_FrozenNoiseHamiltonian",
    } <= set(operators), operators
    assert _undeclared(found) == []


def test_the_declaration_check_finds_an_undeclared_parameter():
    """The check, on a planted class: a float it flags, a container and a declared index it does not, and a declaration
    of a name the constructor does not take it reports."""
    planted = """
class HamiltonianBase:
    NOT_HELD = {"population_index": "an index"}
    def __init__(self, population_index: int = 0): ...

class Planted(HamiltonianBase):
    CHECKED_SCALARS = ("gone",)
    def __init__(self, control_cost: ControlCostBase, strength: float = 1.0, population_index: int = 0): ...
"""
    assert _undeclared(_classes({"planted.py": planted})) == [
        ("Planted", "strength", "undeclared"),
        ("Planted", "gone", "in CHECKED_SCALARS, not a parameter"),
    ]


def _checked_scalars():
    rows = []
    for name, entries in _library().items():
        for entry in entries:
            for scalar in sorted(entry["declared"]["CHECKED_SCALARS"]):
                module = entry["where"].split(":")[0].removesuffix(".py").replace("/", ".")
                rows.append((module, name, scalar))
    return rows


def test_a_scalar_the_constructor_checks_is_declared():
    assert ("mfgarchon.core.hamiltonian", "QuadraticMFGHamiltonian", "coupling_coefficient") in _checked_scalars()


@pytest.mark.parametrize(("module", "name", "scalar"), _checked_scalars())
def test_a_declared_scalar_is_refused_by_its_constructor(module, name, scalar):
    """A declaration of a checked scalar is a claim about the constructor, so it is run: NaN, refused by name."""
    operator = getattr(importlib.import_module(module), name)
    with pytest.raises(ValueError, match=rf"{name}: {scalar} must be finite, got nan"):
        operator(**{scalar: NAN})
