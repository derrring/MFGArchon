"""#2974: a caller-supplied value or callable that is non-finite where the solvers will read it is refused where it
enters, by name.

`_refuse_a_non_finite_source` reads once, where `MFGProblem` is built: the arrays and scalars whole, and each callable only
at states every `NumericalScheme` that solves the problem hands it.
- Times: the interior time nodes; t_0 as well for the Hamiltonian's sources, a callable volatility and a boundary
  callable; T alone for the terminal coupling. The two source terms and the coupling cost at the interior nodes alone.
- Space: the probe nodes off the boundary.
- m and v: the first Picard iteration's guesses, the initial density and the terminal value at it (`u_terminal`
  without a terminal coupling).
- On a network: a node source at t_0, and a `NodeBC`'s callable value at its nodes.
The R1 tests record both sides, the owner's reads and `problem.solve(scheme=...)`'s under every member of the enum, and
hold the first inside the second (#2974's R1). A scalar held by an object that exists before any problem (a control
cost, a kernel, a boundary segment, `QuadraticMFGHamiltonian`) is refused where that object is built, in what its
readers read. A volatility override handed to a solver is refused by `resolve_volatility_override`.

A Hamiltonian's own `__call__` is read at one state by `validate_hamiltonian`: t 0, the owner's centre node, p 0, and m
`m_initial`'s value there. So a source the Hamiltonian holds that is non-finite at that state is refused by the owner
first, by name.

What is not refused:
- a source finite at those reads and non-finite elsewhere, at a time, a node, an m or a v the solve does not hand it at
  entry. A coupling NaN only at m = 0.5 or at m = 1, where `m_initial` takes neither, builds; so does a source
  non-finite only at T/2, which is no time node here;
- a source non-finite only at a state some scheme that solves the problem is not handed (the HJB source or the coupling
  cost at t_0, the FP's source at T), and a FP source that takes v;
- an input no solver reads (`coupling_dm`);
- a network source that takes p or m, which the solvers are handed by their caller;
- an infinity that means something (`BoundedControlCost(max_control=inf)`, unbounded).

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
from mfgarchon.geometry.boundary.applicator_graph import GraphBCConfig, NodeBC
from mfgarchon.geometry.boundary.types import FIELDS_READ
from mfgarchon.geometry.graph.network_geometry import GridNetwork
from mfgarchon.operators.interaction import kernels
from mfgarchon.types.schemes import NumericalScheme
from mfgarchon.utils.validation import ValidationError
from mfgarchon.utils.validation.functions import _refuse_a_non_finite_source

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


def _network_problem(T=0.5, node_bcs=(), **components):
    network = GridNetwork(width=3, height=3)
    network.create_network()
    if node_bcs:
        network.set_boundary_conditions(GraphBCConfig(node_bcs=tuple(node_bcs)))
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
    """An initial density that is 1 nowhere and takes one value at one node of each grid below: 2 + x on [0, 1], 2.5 at
    its centre, and 2 + x + 7y on the unit square, 5.2 at the 6 x 6 grid's centre node (0.4, 0.4) alone."""
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
        "coupling NaN only at m_initial at the centre node, validate_hamiltonian's",
        lambda d: _problem(
            d, m_initial=_rising, model={"hamiltonian": _h(coupling=_nan_only_at(2.5 if d == 1 else 5.2))}
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
        lambda d: _problem(d, source_term_fp=lambda t, x, m: np.full(np.shape(m), NAN)),
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
    ("T NaN", {"T": NAN}, _read("T")),
]


@pytest.mark.parametrize(
    ("given", "names"), [row[1:] for row in NETWORK_REFUSALS], ids=[row[0] for row in NETWORK_REFUSALS]
)
def test_a_non_finite_source_on_a_network_is_refused_by_name(given, names):
    """Read in the node form the network HJB solvers hand, an integer node, at t_0, the one grid time both read."""
    with pytest.raises(ValidationError, match=names):
        _network_problem(**given)


def test_a_node_source_is_read_at_a_node_a_node_bc_governs():
    """A Dirichlet NodeBC does not keep its node's potential out of the solve: policy iteration hands the potential there
    at every step, and a NaN there stops it converging. So the owner reads that node too, by name."""

    def potential(t, node):
        return NAN if int(node) == 8 else 0.0

    exit_bc = NodeBC(nodes=[8], value=0.5, name="exit")
    with pytest.raises(ValidationError, match=_read("NetworkHamiltonian's node_potential_func")):
        _network_problem(node_bcs=[exit_bc], node_potential_func=potential)


def test_a_node_bcs_callable_value_is_refused_by_name():
    """Read at its nodes at the HJB side's times, which policy iteration hands it, as the graph applicator reads it."""
    exit_bc = NodeBC(nodes=[0], value=lambda t, node: NAN, name="exit")
    with pytest.raises(
        NotImplementedError, match=r"NodeBC 'exit': value \(what its callable returned at node 0, t=0\)"
    ):
        _network_problem(node_bcs=[exit_bc])


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


# What each type's readers read, as #2974's rows measured it: the condition's datum, and Robin's two weights. Written out
# here, not read from `FIELDS_READ`, so a wrong entry there fails instead of moving the expectation with it.
_READ_BY_ITS_READERS = {
    bnd.BCType.DIRICHLET: {"value"},
    bnd.BCType.NEUMANN: {"value"},
    bnd.BCType.ROBIN: {"value", "alpha", "beta"},
    bnd.BCType.PERIODIC: set(),
    bnd.BCType.REFLECTING: set(),
    bnd.BCType.NO_FLUX: set(),
    bnd.BCType.FLUX: {"value"},
    bnd.BCType.EXTRAPOLATION_LINEAR: set(),
    bnd.BCType.EXTRAPOLATION_QUADRATIC: set(),
}
_SEGMENT_FIELDS = [(bc_type, name) for bc_type in bnd.BCType for name in ("value", "alpha", "beta")]


@pytest.mark.parametrize(("bc_type", "name"), _SEGMENT_FIELDS, ids=[f"{t.name}.{n}" for t, n in _SEGMENT_FIELDS])
def test_a_segment_refuses_a_nan_constant_in_a_field_its_type_reads_and_nowhere_else(bc_type, name):
    """Every `BCType` is classified, and a segment of that type built with a NaN in one field is refused where it is
    built if its type's readers read that field, and builds if they do not. At 21ef740b a Dirichlet segment's NaN
    alpha or beta, and a NO_FLUX segment's, solved as the finite twin under every scheme that takes the segment, and so
    did a NO_FLUX value under SL and FEM (FDM refuses it where it reads it)."""
    assert set(_READ_BY_ITS_READERS) == set(bnd.BCType) == set(FIELDS_READ)

    def build():
        return bnd.BCSegment(name="wall", bc_type=bc_type, boundary="x_max", **{name: NAN})

    if name in _READ_BY_ITS_READERS[bc_type]:
        label = "" if name == "value" else rf" \(its {name}\)"
        with pytest.raises(NotImplementedError, match=rf"BCSegment 'wall'{label} has a non-finite value nan"):
            build()
    else:
        build()


def test_a_dirichlet_segment_does_not_read_a_weight_too_large_for_a_float():
    """At 21ef740b a Dirichlet segment with alpha 10**400 solved as its twin with alpha 1."""
    bnd.BCSegment(name="wall", bc_type=bnd.BCType.DIRICHLET, boundary="x_max", value=0.0, alpha=10**400)


# --- Review 2: what each read is handed, and when. ---


class _CostReadingTheBuild:
    """A coupling cost whose `bind` reads what construction has built by then: the initial mass, and the Hamiltonian the
    solvers read, at x = 0.95 where the wall below is 1 (p = 0, m = 1)."""

    def bind(self, problem):
        self.mass = float(problem.initial_mass)
        hamiltonian = problem.hamiltonian_class
        self.h = float(np.asarray(hamiltonian(t=0.0, x=(np.array(0.95),), p=np.zeros(1), m=1.0)).reshape(-1)[0])
        return PotentialCost(lambda x: np.zeros(np.shape(np.asarray(x[0], dtype=float)))).bind(problem)


def test_a_coupling_cost_is_bound_after_the_wall_and_the_mass():
    """The costs are bound where 21ef740b binds them, after the soft wall is composed into H and the initial mass is
    measured, so a cost's `bind` reads both, and H(0.95) is the wall's -1.0. Bound before the owner's first reads, as
    01dcbb8a bound them, the same `bind` raised an AttributeError on `initial_mass` and saw H(0.95) = 0."""
    cost = _CostReadingTheBuild()
    problem = _problem(
        1,
        model={"coupling_cost": cost},
        state_penalty=lambda x: np.where(np.asarray(x[0], dtype=float) > 0.9, 1.0, 0.0),
    )
    assert cost.mass == float(problem.initial_mass)
    assert cost.h == -1.0


class _TerminalNaNAt:
    """A terminal coupling whose derivative is NaN at the time ``t_star`` alone, and 0 at every other."""

    def __init__(self, t_star):
        self.t_star = t_star

    def bind(self, problem):
        energy = PotentialCost(lambda x: np.zeros(np.shape(np.asarray(x[0], dtype=float)))).bind(problem)
        t_star = self.t_star

        class Bound:
            def __getattr__(self, attr):
                return getattr(energy, attr)

            def flat_derivative(self, m, *, t=0.0):
                out = np.asarray(energy.flat_derivative(m, t=t), dtype=float)
                return np.full(np.shape(out), NAN) if float(t) == t_star else out

        return Bound()


def test_the_terminal_coupling_is_read_at_the_time_the_solve_reads_it():
    """T 0.1 at Nt 11: 11 * (0.1 / 11) is not 0.1, so the time grid's last node, formed as Nt * dt, lies one ulp from T.
    The solve reads G' through `terminal_value`, at T, and so does the owner: a G' NaN at T is refused, naming the
    terminal coupling, and one NaN at Nt * dt alone builds."""
    T, steps = 0.1, 11
    assert steps * (T / steps) != T

    def build(t_star):
        return MFGProblem(
            model=Model(hamiltonian=_h(), volatility=0.2, terminal_coupling=_TerminalNaNAt(t_star)),
            domain=_grid(1),
            conditions=Conditions(m_initial=lambda x: 1.0, u_terminal=_u_terminal, T=T),
            Nt=steps,
        )

    with pytest.raises(ValidationError, match=_read("terminal_coupling")):
        build(T)
    build(steps * (T / steps))


ONE_ELEMENT_ARRAYS = [
    ("GaussianKernel amplitude", lambda: kernels.GaussianKernel(amplitude=np.array([0.5]))),
    ("GaussianKernel length_scale", lambda: kernels.GaussianKernel(length_scale=np.array([0.2]))),
    ("PowerLawKernel exponent", lambda: kernels.PowerLawKernel(exponent=np.array([0.5]))),
    ("BoundedControlCost max_control", lambda: BoundedControlCost(control_cost=1.0, max_control=np.array([1.0]))),
]


@pytest.mark.parametrize("build", [row[1] for row in ONE_ELEMENT_ARRAYS], ids=[row[0] for row in ONE_ELEMENT_ARRAYS])
def test_a_one_element_array_is_read_as_its_number(build):
    """At 21ef740b each, in a solved problem, gives U bit for bit as its scalar does; 01dcbb8a raised a TypeError."""
    build()


def test_a_one_element_array_that_is_nan_is_refused_by_name():
    with pytest.raises(ValueError, match="GaussianKernel: amplitude must be a number"):
        kernels.GaussianKernel(amplitude=np.array([NAN]))


def test_a_callable_returning_no_number_is_read_as_its_reader_reads_it():
    """A return that holds no number is decided by what the solve's reader makes of it. The HJB source's is not this
    fact: at 21ef740b the solve raised a TypeError on it, and 01dcbb8a refused it as non-finite; here it builds, as at
    21ef740b. The wall's reader, the composed potential, reads a None as NaN (`wall_values`), so a None wall is
    non-finite where the solver reads it: at 21ef740b it solved to a U 0.54 off its twin's with `converged` False, and
    it is refused here, naming the wall."""
    _problem(1, source_term_hjb=lambda t, x: None)
    with pytest.raises(ValidationError, match=_read("state_penalty")):
        _problem(1, state_penalty=lambda x: None)


# The read set from below, each a state the documented reads include and no other row pins (the probe nodes here are
# 0.5, the centre, 0.6, 0.2 and 0.9; Nt 5, so t_{Nt // 2} is 0.2 and t_{Nt - 1} is 0.4).
LOWER_BOUND = [
    (
        "source_term_hjb NaN only at t_{Nt // 2}",
        lambda: _problem(1, source_term_hjb=lambda t, x: _at_x(_only_at(0.2)(t))(x)),
        ValidationError,
        _read("source_term_hjb"),
    ),
    (
        "source_term_hjb NaN only at t_{Nt - 1}",
        lambda: _problem(1, source_term_hjb=lambda t, x: _at_x(_only_at(0.4)(t))(x)),
        ValidationError,
        _read("source_term_hjb"),
    ),
    (
        "potential NaN only at a probe node off the centre",
        lambda: _problem(
            1,
            model={
                "hamiltonian": _h(
                    potential=lambda t, x: np.where(np.isclose(np.asarray(x[0], dtype=float), 0.9), NAN, 0.0)
                )
            },
        ),
        ValidationError,
        _read("SeparableHamiltonian's potential"),
    ),
    (
        "a Dirichlet g(t) NaN only at an interior time",
        lambda: _problem(
            1, bc=bnd.dirichlet_bc(value=lambda t: NAN if abs(float(t) - 0.2) < 1e-12 else 0.0, dimension=1)
        ),
        NotImplementedError,
        r"BCSegment 'uniform' \(what its callable returned at t=0.2\) has a non-finite value nan",
    ),
]


@pytest.mark.parametrize(
    ("build", "error", "names"), [row[1:] for row in LOWER_BOUND], ids=[row[0] for row in LOWER_BOUND]
)
def test_a_state_the_documented_reads_include_is_read(build, error, names):
    with pytest.raises(error, match=names):
        build()


def test_a_source_a_derived_operator_holds_is_named_once_by_the_operator_given():
    """A Lagrangian's potential is held again by the Hamiltonian derived from it; read once, it is named once, by the
    operator the caller gave."""
    with pytest.raises(ValidationError) as refused:
        _components_problem(
            1, lagrangian=SeparableLagrangian(QuadraticControlCost(control_cost=1.0), potential=_at_tx(NAN))
        )
    assert "SeparableLagrangian's potential" in str(refused.value)
    assert "SeparableHamiltonian's potential" not in str(refused.value)


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


_KERNEL_SCALES = {"length_scale", "rep_scale", "att_scale", "softening"}
_AMPLITUDES_AND_EXPONENTS = [(kernel, name) for kernel, name in KERNEL_PARAMETERS if name not in _KERNEL_SCALES]


@pytest.mark.parametrize(("kernel", "name"), KERNEL_PARAMETERS, ids=[f"{k.__name__}.{n}" for k, n in KERNEL_PARAMETERS])
def test_a_kernel_refuses_a_nan_parameter_by_name(kernel, name):
    with pytest.raises(ValueError, match=rf"{kernel.__name__}: {name} must be a number, got nan"):
        kernel(**{name: NAN})


@pytest.mark.parametrize(
    ("kernel", "name"), _AMPLITUDES_AND_EXPONENTS, ids=[f"{k.__name__}.{n}" for k, n in _AMPLITUDES_AND_EXPONENTS]
)
def test_a_kernel_refuses_an_infinite_amplitude_or_exponent_by_name(kernel, name):
    """At 21ef740b each of these built, and its solve returned a finite U with `converged` False."""
    with pytest.raises(ValueError, match=rf"{kernel.__name__}: {name} must be finite, got inf"):
        kernel(**{name: INF})


_KERNEL_LIMITS = [
    (kernels.GaussianKernel, "length_scale", lambda r: np.ones_like(r)),
    (kernels.TentKernel, "length_scale", lambda r: np.ones_like(r)),
    (kernels.WendlandKernel, "length_scale", lambda r: np.ones_like(r)),
    (kernels.DipoleKernel, "rep_scale", lambda r: 1.0 + kernels.DipoleKernel(rep_amplitude=0.0).profile(r)),
    (kernels.DipoleKernel, "att_scale", lambda r: kernels.DipoleKernel(att_amplitude=0.0).profile(r) - 0.5),
    (kernels.PowerLawKernel, "softening", lambda r: np.zeros_like(r)),
]


@pytest.mark.parametrize(
    ("kernel", "name", "limit"), _KERNEL_LIMITS, ids=[f"{k.__name__}.{n}" for k, n, _ in _KERNEL_LIMITS]
)
def test_an_infinite_scale_builds_the_kernel_it_tends_to(kernel, name, limit):
    """An infinite length scale is the constant kernel, and an infinite softening the zero one, with the defaults'
    amplitudes (1, and Dipole's 1 and 0.5). At 21ef740b each solved bit for bit as that limit, written as its own kernel."""
    assert {n for k, n, _ in _KERNEL_LIMITS} == _KERNEL_SCALES
    r = np.linspace(0.0, 2.0, 9)
    np.testing.assert_array_equal(kernel(**{name: INF}).profile(r), limit(r))


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
    """The check, on a planted class. It flags a float, an ``ArrayLike``, an ``Any`` or unannotated parameter with a
    ``None`` default, a callable whose annotation names a container it returns, and an annotation that names neither with
    a number for its default. It does not flag a container, a string, the same annotation with a string for its default,
    an unannotated bool, or a declared index. It reports a declaration of a name the constructor does not take.

    Its limits. It reads class statements and ``__init__`` signatures from the source, so it does not see a value
    taken through ``**kwargs``, a dataclass field (no ``__init__`` is written), a class whose base is an alias of an
    operator class (``Base = HamiltonianBase``), or a class built with ``type()``. It reads the annotation's text, so a
    value or a container behind an alias it has no mark for is misread, and an annotation naming neither, such as an
    enum, is a value only when its default is a number."""
    planted = """
class HamiltonianBase:
    NOT_HELD = {"population_index": "an index"}
    def __init__(self, population_index: int = 0): ...

class Planted(HamiltonianBase):
    CHECKED_SCALARS = ("gone",)
    def __init__(
        self,
        control_cost: ControlCostBase,
        strength: float = 1.0,
        weights: ArrayLike = None,
        anything: Any = None,
        bare=None,
        factory: Callable[[], ControlCostBase] | None = None,
        label: str = "planted",
        kind: Kind = 0,
        kind_name: Kind = "a",
        flag=True,
        population_index: int = 0,
    ): ...
"""
    assert _undeclared(_classes({"planted.py": planted})) == [
        ("Planted", "strength", "undeclared"),
        ("Planted", "weights", "undeclared"),
        ("Planted", "anything", "undeclared"),
        ("Planted", "bare", "undeclared"),
        ("Planted", "factory", "undeclared"),
        ("Planted", "kind", "undeclared"),
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


class _OwnCallNaNAtTheCentre(SeparableHamiltonian):
    """A Hamiltonian whose own __call__ is NaN at the 6 x 6 grid's centre node (0.4, 0.4) alone, and its parts finite."""

    def __call__(self, t, x, p, m):
        value = np.asarray(super().__call__(t=t, x=x, p=p, m=m), dtype=float)
        at = np.isclose(np.asarray(x[0], dtype=float), 0.4) & np.isclose(np.asarray(x[1], dtype=float), 0.4)
        return np.where(at, NAN, value)


@_MASS_IS_NOT_THE_SUBJECT
def test_validate_hamiltonian_reads_an_own_call_at_the_owners_centre_node():
    """Its state is inside the owner's reads: t 0, the centre probe node off the boundary, p 0, m m_initial's value there.
    On a 6 x 6 grid that node is (0.4, 0.4), where the grid's middle index, (0.6, 0), is a boundary node."""
    with pytest.raises(ValidationError, match=r"Hamiltonian returned"):
        _problem(2, model={"hamiltonian": _OwnCallNaNAtTheCentre(QuadraticControlCost(control_cost=1.0))})


def test_on_an_implicit_domain_the_owner_reads_no_callable():
    """An implicit domain draws its points afresh at each read of its grid (#2778), so no node pairs with an entry of
    m_initial and the owner reads no callable there. A Hamiltonian's NaN potential is then refused by validate_hamiltonian,
    through its __call__, and not by the owner's read of the potential, which would read a random draw."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # the density's mass on this measure is not this pin's subject
        with pytest.raises(ValidationError, match=r"Hamiltonian returned NaN or Inf at sample point"):
            MFGProblem(
                model=Model(hamiltonian=_h(potential=_at_tx(NAN)), volatility=0.2),
                domain=Hyperrectangle(np.array([[0.0, 1.0]])),
                conditions=Conditions(m_initial=lambda x: 1.0, u_terminal=lambda x: 0.0, T=0.5),
                Nt=4,
            )


# --- States the solve does not read are not refused. ---


def _only_at(t_star, value=0.0):
    """A value non-finite (NaN) at time ``t_star`` alone, and ``value`` at every other time."""
    return lambda t: NAN if abs(float(t) - t_star) < 1e-12 else value


def _one_over_x(x):
    with np.errstate(divide="ignore"):
        return 1.0 / np.asarray(x[0], dtype=float)


NOT_READ = [
    ("source_term_hjb NaN only at T", lambda: _problem(1, source_term_hjb=lambda t, x: _at_x(_only_at(0.5)(t))(x))),
    (
        "source_term_hjb NaN only at T/2, no time node",
        lambda: _problem(1, source_term_hjb=lambda t, x: _at_x(_only_at(0.25)(t))(x)),
    ),
    ("source_term_fp NaN only at t_0", lambda: _problem(1, source_term_fp=lambda t, x: _at_x(_only_at(0.0)(t))(x))),
    (
        "potential NaN only at T",
        lambda: _problem(1, model={"hamiltonian": _h(potential=lambda t, x: _at_x(_only_at(0.5)(t))(x))}),
    ),
    (
        "volatility NaN only at T",
        lambda: _problem(1, model={"volatility": lambda t, x: _at_x(_only_at(0.5, 0.2)(t))(x)}),
    ),
    (
        "potential 1/x, infinite at a Dirichlet node",
        lambda: _problem(
            1,
            model={"hamiltonian": _h(potential=lambda t, x: _one_over_x(x))},
            bc=bnd.dirichlet_bc(value=0.0, dimension=1),
        ),
    ),
    (
        "potential 1/x under no-flux, infinite at a boundary node",
        lambda: _problem(1, model={"hamiltonian": _h(potential=lambda t, x: _one_over_x(x))}),
    ),
    (
        "source_term_fp taking v, NaN where v is not u_terminal",
        lambda: _problem(
            1,
            source_term_fp=lambda t, x, v, m: np.where(
                np.isclose(v, np.sin(np.pi * np.asarray(x[0], dtype=float))), 0.0, NAN
            ),
        ),
    ),
    (
        "network hamiltonian_func NaN, which takes p and m",
        lambda: _network_problem(
            hamiltonian_func=lambda t, node, neighbors, p, m: NAN,
            hamiltonian_dm_func=lambda t, node, neighbors, p, m: 0.0,
        ),
    ),
    (
        "network node_interaction_func NaN, which takes m",
        lambda: _network_problem(node_interaction_func=lambda t, node, m: NAN),
    ),
]


@pytest.mark.parametrize("build", [row[1] for row in NOT_READ], ids=[row[0] for row in NOT_READ])
def test_a_state_the_solve_is_not_handed_at_entry_is_not_refused(build):
    """The read set's limits, each a problem the owner must build. Each is non-finite only where the default solve is not
    handed it at entry, or where its value does not reach the solution (a Dirichlet node), or is a node source taking
    the caller's p or m. A wider read set refuses them, and at 21ef740b the ones on a grid solve as their finite twin."""
    build()


class _CostNaNAtTheStart:
    """A coupling cost whose derivative is NaN at t = 0 alone."""

    def bind(self, problem):
        energy = PotentialCost(lambda x: np.zeros(np.shape(np.asarray(x[0], dtype=float)))).bind(problem)

        class Bound:
            def __getattr__(self, attr):
                return getattr(energy, attr)

            def flat_derivative(self, m, *, t=0.0):
                out = np.asarray(energy.flat_derivative(m, t=t), dtype=float)
                return np.full(np.shape(out), NAN) if abs(float(t)) < 1e-12 else out

        return Bound()


NOT_READ_BY_EVERY_SCHEME = [
    (
        "source_term_hjb NaN only at t_0, which SL is not handed",
        lambda: _problem(1, source_term_hjb=lambda t, x: _at_x(_only_at(0.0)(t))(x)),
    ),
    (
        "coupling_cost NaN only at t_0, which SL is not handed",
        lambda: _problem(1, model={"coupling_cost": _CostNaNAtTheStart()}),
    ),
    (
        "source_term_fp NaN only at T, which FVM is not handed",
        lambda: _problem(1, source_term_fp=lambda t, x: _at_x(_only_at(0.5)(t))(x)),
    ),
    (
        "source_term_fp taking v, NaN at every state",
        lambda: _problem(1, source_term_fp=lambda t, x, v, m: np.full(np.shape(m), NAN)),
    ),
]


@pytest.mark.parametrize(
    "build", [row[1] for row in NOT_READ_BY_EVERY_SCHEME], ids=[row[0] for row in NOT_READ_BY_EVERY_SCHEME]
)
def test_a_state_a_scheme_that_solves_the_problem_is_not_handed_is_not_refused(build):
    """The read set is what every `NumericalScheme` that solves the problem hands. The semi-Lagrangian schemes never read
    the HJB source or the coupling cost at t_0, and the finite-volume ones never read the FP's source at T, so each of
    these builds; a FP source that takes v is handed u_terminal at T alone, so none of its states is read. At 21ef740b
    the first three solved as their finite twins under those schemes, and the default scheme reached all four. What the
    solve reaches is the point-of-use checks' (#2975, #2976)."""
    build()


# --- R1: every state the owner reads a callable at is one each scheme that solves the problem hands it. ---

# Construction runs other checks too (the consistency check reads its own probes), so the owner is run alone, on the
# built problem, while its reads are noted: "off" while the problem is built, "owner", then "solve".
_PHASE = ["off"]
_SEEN: dict[str, set] = {"owner": set(), "solve": set()}
_CTX: dict = {}


def _t(t):
    return None if t is None else round(float(np.asarray(t).reshape(-1)[0]), 10)


def _flag(values, reference, node, k):
    if values is None:
        return None
    handed = np.asarray(values, dtype=float).reshape(-1)
    if handed.size == reference.size:
        return bool(np.isclose(handed[node], reference[node]))
    return bool(np.isclose(handed[k if handed.size > 1 else 0], reference[node]))


def _note(name, *, t=None, x=None, m=None, v=None, node=None):
    """One call's states, one per node it names: (name, t, node, m is m_initial's there, v is u_terminal's there)."""
    if _PHASE[0] == "off":
        return
    seen = _SEEN[_PHASE[0]]
    if node is not None:
        seen.add((name, _t(t), int(node), None, None))
        return
    if x is None:
        if m is not None:  # a density term: the m values it is handed
            seen.update(
                (name, None, None, round(float(value), 10), None) for value in np.asarray(m, dtype=float).ravel()
            )
        else:
            seen.add((name, _t(t), None, None, None))
        return
    nodes = _CTX["nodes"]
    points = np.stack([np.atleast_1d(np.asarray(c, dtype=float)).reshape(-1) for c in x], axis=1)
    for k, point in enumerate(points):
        hit = np.flatnonzero(np.all(np.isclose(nodes, point, rtol=0, atol=1e-12), axis=1))
        i = int(hit[0]) if hit.size else -1
        seen.add(
            (
                name,
                _t(t),
                i,
                _flag(m, _CTX["m0"], i, k) if i >= 0 else None,
                _flag(v, _CTX["vT"], i, k) if i >= 0 else None,
            )
        )


class _NotedCost:
    """A coupling cost whose bound energy notes each derivative it is asked for: (t, whether m is m_initial)."""

    def __init__(self, name):
        self.name = name

    def bind(self, problem):
        energy = PotentialCost(lambda x: np.zeros(np.shape(np.asarray(x[0], dtype=float)))).bind(problem)
        name = self.name

        class Noted:
            def __getattr__(self, attr):
                return getattr(energy, attr)

            def flat_derivative(self, m, *, t=0.0):
                if _PHASE[0] != "off":
                    _SEEN[_PHASE[0]].add((name, _t(t), None, bool(np.allclose(np.ravel(m), _CTX["m0"])), None))
                return energy.flat_derivative(m, t=t)

        return Noted()


def _potential(t, x):
    _note("potential", t=t, x=x)
    return np.zeros(np.shape(np.asarray(x[0], dtype=float)))


def _hamiltonian_group(costs=True):
    def coupling(m):
        _note("coupling", m=m)
        return 0.5 * np.asarray(m, dtype=float)

    def wall(x):
        _note("state_penalty", x=x)
        return np.zeros(np.shape(x)[1:]) if np.ndim(x) >= 2 else 0.0

    # Off a grid a coupling cost binds on the solver's own measure, and the problem binds none for the owner to read.
    noted = {"coupling_cost": _NotedCost("coupling_cost"), "terminal_coupling": _NotedCost("terminal_coupling")}
    model = Model(hamiltonian=_h(potential=_potential, coupling=coupling), volatility=0.5, **(noted if costs else {}))
    return {"model": model, "state_penalty": wall}, {
        "potential",
        "coupling",
        "state_penalty",
        *(noted if costs else ()),
    }


def _volatility_group():
    def volatility(t, x, m):
        _note("volatility", t=t, x=x, m=m)
        return np.full(np.shape(np.asarray(x[0], dtype=float)), 0.5)

    return {"model": Model(hamiltonian=_h(potential=_potential), volatility=volatility)}, {"potential", "volatility"}


def _with_a_terminal_coupling(kwargs):
    """A terminal coupling whose derivative is not zero, G'(m) = 0.3 + 0.2 x, so the first iteration hands the HJB v at
    u_terminal + G'(m_initial), not u_terminal."""
    kwargs["model"] = Model(
        hamiltonian=_h(potential=_potential),
        volatility=0.5,
        terminal_coupling=PotentialCost(lambda x: 0.3 + 0.2 * np.asarray(x[0], dtype=float)),
    )
    return kwargs


def _source_group(fp_takes_v, terminal=False):
    def source_hjb(t, x, v, m):
        _note("source_term_hjb", t=t, x=x, v=v, m=m)
        return np.zeros(np.shape(np.asarray(x[0], dtype=float)))

    def source_fp(t, x, m):
        _note("source_term_fp", t=t, x=x, m=m)
        return np.zeros(np.shape(np.asarray(x[0], dtype=float)))

    def source_fp_v(t, x, v, m):
        _note("source_term_fp", t=t, x=x, v=v, m=m)
        return np.zeros(np.shape(np.asarray(x[0], dtype=float)))

    # A source_term_fp that takes v is handed u_terminal at T alone, which no finite-volume scheme reads: not read.
    kwargs = {"source_term_hjb": source_hjb, "source_term_fp": source_fp_v if fp_takes_v else source_fp}
    return (
        _with_a_terminal_coupling(kwargs) if terminal else kwargs,
        {"potential", "source_term_hjb", *(() if fp_takes_v else ("source_term_fp",))},
    )


def _nonlocal_group(terminal=False):
    def matvec(v):
        # The operand has no time, node or m to tell the states apart, so it is named: the first iteration's v, u_terminal
        # where that is another, or another iterate.
        if _PHASE[0] != "off":
            flat = np.ravel(v)
            operand = (
                "first" if np.allclose(flat, _CTX["vT"]) else "u_terminal" if np.allclose(flat, _CTX["uT"]) else "other"
            )
            _SEEN[_PHASE[0]].add(("nonlocal_operator", None, None, None, operand))
        return np.zeros(np.shape(v))

    kwargs = {"nonlocal_operator": LinearOperator((11, 11), matvec=matvec, dtype=float)}
    return _with_a_terminal_coupling(kwargs) if terminal else kwargs, {"potential", "nonlocal_operator"}


def _boundary_group():
    def g(t):
        _note("Dirichlet g(t)", t=t)
        return 0.0

    return {"bc": bnd.dirichlet_bc(value=g, dimension=1)}, {"potential", "Dirichlet g(t)"}


def _mesh_1d(bc=None):
    from mfgarchon.geometry.meshes.mesh_1d import Mesh1D

    mesh = Mesh1D(bounds=(0.0, 1.0), num_elements=16)
    mesh.generate_mesh()
    mesh.boundary_conditions = bc or bnd.no_flux_bc(dimension=1)
    return mesh


# (id, domain: 1 or 2 for the grid, "mesh" for a Mesh1D of 16 cells, which FEM solves on; Nt; group)
GRID_SETUPS = [
    ("1-D, Nt 5, the Hamiltonian's sources, wall, costs", 1, 5, _hamiltonian_group),
    ("1-D, Nt 5, a volatility(t, x, m)", 1, 5, _volatility_group),
    ("1-D, Nt 5, the source terms, the FP's taking v", 1, 5, lambda: _source_group(True)),
    ("1-D, Nt 5, the FP's source without v", 1, 5, lambda: _source_group(False)),
    ("1-D, Nt 5, nonlocal_operator", 1, 5, _nonlocal_group),
    ("1-D, Nt 5, the source terms beside a terminal coupling", 1, 5, lambda: _source_group(True, terminal=True)),
    ("1-D, Nt 5, nonlocal_operator beside a terminal coupling", 1, 5, lambda: _nonlocal_group(terminal=True)),
    ("1-D, Nt 5, a Dirichlet g(t)", 1, 5, _boundary_group),
    ("1-D, Nt 1, the Hamiltonian's sources, wall, costs", 1, 1, _hamiltonian_group),
    ("1-D, Nt 1, the FP's source without v", 1, 1, lambda: _source_group(False)),
    ("1-D, Nt 4, the FP's source without v", 1, 4, lambda: _source_group(False)),
    ("6 x 6, Nt 5, the Hamiltonian's sources, wall, costs", 2, 5, _hamiltonian_group),
    ("6 x 6, Nt 5, the FP's source without v", 2, 5, lambda: _source_group(False)),
    ("6 x 6, Nt 5, the source terms beside a terminal coupling", 2, 5, lambda: _source_group(True, terminal=True)),
    ("Mesh1D, Nt 5, the Hamiltonian's sources and wall", "mesh", 5, lambda: _hamiltonian_group(costs=False)),
    ("Mesh1D, Nt 5, the FP's source without v", "mesh", 5, lambda: _source_group(False)),
    ("Mesh1D, Nt 5, a Dirichlet g(t)", "mesh", 5, _boundary_group),
    (
        "Mesh1D, Nt 5, the source terms beside a terminal coupling",
        "mesh",
        5,
        lambda: _source_group(True, terminal=True),
    ),
]


def _refusal(scheme, domain, names):
    """Why ``scheme`` does not solve the setup, as a pattern its message matches, or None where it solves it (#2974's
    record). A member this does not name is expected to solve every setup, so a new one fails here until it is read."""
    name = scheme.name
    if name == "GFDM":
        return "has no FP half"
    if name == "MESHLESS_GALERKIN":  # problem.solve cannot hand it its collocation points
        return "collocation_points"
    if name.startswith("FEM_"):
        if domain != "mesh":
            return "requires an unstructured mesh"
        # P2's DOFs are the vertices and the edge midpoints, never the problem's points; and it binds a declared
        # coupling cost or terminal coupling on P1 only, which it refuses first.
        return "points that are not those of the problem being solved|on P1 only" if name == "FEM_P2" else None
    if domain == "mesh":
        return "Cannot use .* with this problem"
    if name.startswith("SL_"):
        if "volatility" in names:
            return "reads one constant volatility"
        if "Dirichlet g(t)" in names:
            return "does not support boundary condition type"
    if name.startswith("FVM_"):
        if "volatility" in names:
            return "supports only scalar/None volatility"
        if "Dirichlet g(t)" in names:
            return "does not support Dirichlet BC"
    return None


@_MASS_IS_NOT_THE_SUBJECT
@pytest.mark.parametrize("scheme", list(NumericalScheme), ids=[scheme.name for scheme in NumericalScheme])
@pytest.mark.parametrize(
    ("domain", "steps", "group"), [row[1:] for row in GRID_SETUPS], ids=[row[0] for row in GRID_SETUPS]
)
def test_every_state_the_owner_reads_is_one_each_scheme_hands(domain, steps, group, scheme):
    """Each callable notes every state it is handed, while the owner reads the built problem and then while
    problem.solve(scheme=...) runs, as (time, node, m is m_initial's there, v is the first iteration's there: the terminal
    value at m_initial, which is u_terminal without a terminal coupling). For every
    `NumericalScheme` member that solves the setup, what the owner read must be among what the solve read: a scheme that
    stops reading one of the owner's states, or an owner that reads one a scheme does not, fails here. A member that does
    not solve it must refuse it with the message `_refusal` names.

    The control: the potential, and the wall where the setup has one, are handed every node by every scheme, so their
    recorded node sets must be the whole grid or mesh. A recorder that misreads a point it is handed shrinks the record
    to a plausible subset; this fails instead."""
    kwargs, names = group()
    if steps == 1:  # no interior time node, the only times every scheme hands these
        names -= {"coupling_cost", "source_term_hjb", "source_term_fp"}
    if domain == "mesh" and "model" in kwargs and kwargs["model"].terminal_coupling is not None:
        # Off a grid the problem binds no G, and the v the first iteration hands is the HJB solver's binding's.
        names -= {"source_term_hjb", "nonlocal_operator"}
    bc = kwargs.pop("bc", None)
    for side in _SEEN.values():
        side.clear()
    _PHASE[0] = "off"
    dim = 1 if domain == "mesh" else domain
    problem = MFGProblem(
        model=kwargs.pop("model", None) or Model(hamiltonian=_h(potential=_potential), volatility=0.5),
        domain=_mesh_1d(bc) if domain == "mesh" else _grid(dim, bc),
        conditions=Conditions(
            m_initial=lambda x: 1.0 + 0.5 * np.cos(np.pi * np.asarray(x[0], dtype=float)),
            u_terminal=lambda x: 0.1 * _u_terminal(x),
            T=0.5,
        ),
        Nt=steps,
        **kwargs,
    )
    _CTX.update(
        nodes=np.asarray(problem.geometry.get_spatial_grid(), dtype=float).reshape(-1, dim),
        m0=np.asarray(problem.m_initial, dtype=float).ravel(),
        vT=_first_terminal_value(problem),
        uT=np.asarray(problem.u_terminal, dtype=float).ravel(),
    )
    solve = lambda: problem.solve(scheme=scheme, max_iterations=2, verbose=False)  # noqa: E731
    refusal = _refusal(scheme, domain, names)
    if refusal is not None:
        with pytest.raises((NotImplementedError, TypeError, ValueError), match=refusal):
            _read_as_the_owner_then_solve(problem, solve)
        return
    _read_as_the_owner_then_solve(problem, solve)
    read = {state[0] for state in _SEEN["owner"]}
    assert names <= read, f"the owner read none of {sorted(names - read)}"
    every_node = set(range(len(_CTX["nodes"])))
    for name in {"potential", "state_penalty"} & names:
        handed = {state[2] for state in _SEEN["solve"] if state[0] == name} - {-1}
        assert handed == every_node, f"{name} recorded at {len(handed)} of {len(every_node)} nodes under {scheme.name}"
    unread = sorted(_SEEN["owner"] - _SEEN["solve"], key=str)
    assert not unread, f"{len(unread)} states the owner read that {scheme.name} was not handed, first {unread[:3]}"


def _first_terminal_value(problem):
    """The v every row of the iterator's cold start holds: ``terminal_value(get_m_init())``, or u_terminal where G is
    bound by the solver and not the problem (off a grid), where what takes v is not read at all."""
    try:
        return np.asarray(problem.terminal_value(problem.get_m_init()), dtype=float).ravel()
    except (NotImplementedError, ValueError):
        return np.asarray(problem.u_terminal, dtype=float).ravel()


def _read_as_the_owner_then_solve(problem, solve):
    _PHASE[0] = "owner"
    try:
        _refuse_a_non_finite_source(problem)
        _refuse_a_non_finite_source(problem, after_binding=True)
        _PHASE[0] = "solve"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            solve()
    finally:
        _PHASE[0] = "off"


@pytest.mark.parametrize("solver", ["NetworkHJBSolver", "NetworkPolicyIterationHJBSolver"])
def test_every_state_the_owner_reads_on_a_network_is_one_its_hjb_solvers_hand(solver):
    """The network's: a node source at t_0, and a NodeBC's callable value at its nodes at the HJB side's times, which
    policy iteration hands it (NetworkHJBSolver refuses a node BC). Each solver is driven as a caller drives it, with the
    problem's own arrays."""
    from mfgarchon.alg.numerical.network_solvers import hjb_network

    def node_potential(t, node):
        _note("node_potential_func", t=t, node=node)
        return 0.0

    def exit_value(t, node):
        _note("NodeBC value", t=t, node=node)
        return 0.5

    policy = solver == "NetworkPolicyIterationHJBSolver"
    for side in _SEEN.values():
        side.clear()
    _PHASE[0] = "off"
    problem = _network_problem(
        node_potential_func=node_potential,
        node_bcs=[NodeBC(nodes=[8], value=exit_value, name="exit")] if policy else (),
    )
    M = np.tile(np.asarray(problem.m_initial, dtype=float), (6, 1))
    U_T = np.asarray(problem.u_terminal, dtype=float)
    _read_as_the_owner_then_solve(
        problem, lambda: getattr(hjb_network, solver)(problem).solve_hjb_system(M, U_T, np.zeros((6, 9)))
    )
    names = {"node_potential_func", "NodeBC value"} if policy else {"node_potential_func"}
    assert names <= {state[0] for state in _SEEN["owner"]}, f"the owner read none of {sorted(names)}"
    unread = sorted(_SEEN["owner"] - _SEEN["solve"], key=str)
    assert not unread, f"{len(unread)} states the owner read that {solver} was not handed, first {unread[:3]}"
