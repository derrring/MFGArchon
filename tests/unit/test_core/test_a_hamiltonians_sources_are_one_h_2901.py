"""A Hamiltonian's sources are one H: a user's Hamiltonian whose sources disagree is refused at construction (#2901).

The library reads a Hamiltonian through several sources: `__call__` in the HJB residual, `dp` in its Jacobian,
`optimal_control` in the Fokker–Planck drift, the doors over them on a batch and on a point, for a separable H its cost,
the potential the soft wall is composed into, and on a network the rates. Before #2901 a problem checked `dp` and `dm`
against `__call__` (`validate_hamiltonian_consistency`, #1642), and nothing else, so a subclass whose `optimal_control`
was another H's was solved as a mixture with no error. The user ruled (A) on 2026-11-11: refuse it at construction.
Three owners decide it: the consistency check (the methods agree at a point, at its probes), the batch owner
`refuse_a_hamiltonian_whose_reads_disagree` (a door's batch read agrees with its point read), and `_why_not_as_it_ships`
(the cost stands for H only as the library ships it).

- **The control against -dp** (a mutation-verified pin). `Mix` and the issue's reproduction are refused at
  construction, and the message names the pair that disagrees and the probe (t, x, m, p) where it does: against dp and
  H when dp agrees with H, against H alone when it does not.
- **The probes vary t, x, m and p** (a mutation-verified pin). A control that departs from -dp only at T/2, only at T,
  only at an axis's first or last end node, only for p < 0, or only off the diagonal p_1 = p_2, is refused; so are a dp
  wrong only after t = 0 and a dm wrong only at an end node. The probe times are the problem's own T (a row at T = 2).
  On a grid with an even number of nodes the middle probe is the node nearest the centre, so there are still 2d + 1.
- **A probe that cannot be evaluated is skipped, not the check** (a mutation-verified pin). A wrong dp, and `Mix`, whose
  H raises at x = 0 are refused; so is `Mix` whose control raises at p = 0, or at t = T. A correct H whose control raises
  at p = 0 is accepted with a warning. An `AssertionError` is not an evaluation error and propagates.
- **Each door against its method, at a point** (a mutation-verified pin), and **each door on a batch against a point**
  (the batch owner): a door doubled at every input, and one doubled only on batches of two or more, are refused.
- **A network Hamiltonian's sources** (a mutation-verified pin): its rates against its dp at every node, and its dp and
  dm against differences of `__call__` at a bounded set of nodes. Each wrong source is refused, and so is a source wrong
  only at t = T or only at one probed node.
- **The soft wall** (a mutation-verified pin): composed into a subclass whose `__call__` does not read the potential, it
  is refused; a subclass that changes nothing is walled.
- **`canonical_cs` and `_control_cost_lambda` read the cost only as shipped** (a mutation-verified pin).
- **HJB-GFDM's DMP warning reads the Hamiltonian's dp at the step's density** (a mutation-verified pin), on a congestion
  H with lambda != 1, where |grad u| / lambda, the dp at m = 1 and the dp at the step's density all differ.
- **The control every reader must accept** (a necessary oracle): the library Hamiltonians of eight kinds, in 1-D and 2-D.
"""

from __future__ import annotations

import logging
import re
import warnings

import pytest

import numpy as np

from mfgarchon import Conditions, MFGProblem, Model
from mfgarchon.core import hamiltonian as hm
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import no_flux_bc
from mfgarchon.utils.validation import ValidationError, validate_custom_functions


def f(m):
    return 0.5 * m


def _a(v):
    return np.asarray(v, dtype=float)


def _momentum(dim, scale=0.4157):
    """The first nonzero momentum probe, as the messages print it: a scale times the signs (+1, -1, ...)."""
    return np.array2string(scale * np.array([(-1.0) ** k for k in range(dim)]), precision=4)


class _OneHalfQuadratic(hm.SeparableHamiltonian):
    """`__call__` and `dp` are the lambda = 1/3 Hamiltonian, 1.5|p|^2 - f(m) and 3p, and they agree."""

    def __call__(self, t, x, p, m):
        return 1.5 * np.sum(_a(p) ** 2, axis=-1) - f(_a(m))

    def dp(self, t, x, p, m):
        return 3.0 * _a(p)


class _Mix(_OneHalfQuadratic):
    """And its optimal control is the lambda = 1/2 one."""

    def optimal_control(self, t, x, p, m):
        return -2.0 * _a(p)


class _WrongDpAndControl(hm.SeparableHamiltonian):
    """The library's H, a dp twice its derivative, and a control that is minus neither: it disagrees with H and dp."""

    def dp(self, t, x, p, m):
        return 2.0 * _a(super().dp(t=t, x=x, p=p, m=m))

    def optimal_control(self, t, x, p, m):
        return -3.0 * _a(p)


def _scaled_control(factor):
    """The library's H and dp, and its control times factor(t, x, p): they agree wherever factor is 1."""

    class Scaled(hm.SeparableHamiltonian):
        def optimal_control(self, t, x, p, m):
            return factor(t, x, _a(p)) * _a(super().optimal_control(t=t, x=x, p=p, m=m))

    return Scaled


_AFTER_T0 = _scaled_control(lambda t, x, p: 1.0 + 4.0 * t)
_ONLY_AT_T = _scaled_control(lambda t, x, p: 1.0 + 40.0 * max(t - 0.4, 0.0))
_AWAY_FROM_THE_MIDDLE = _scaled_control(lambda t, x, p: 1.0 + 3.0 * (_a(x[0]) - 0.5) ** 2)
_ONLY_AT_THE_LAST_END = _scaled_control(lambda t, x, p: 1.0 + 12.0 * np.maximum(_a(x[0]) - 0.75, 0.0))


class _AbsControl(hm.SeparableHamiltonian):
    """The library's H and dp, and the control -|p|: they agree for p >= 0 only."""

    def optimal_control(self, t, x, p, m):
        return -np.abs(_a(p))


class _SwappedControl(hm.SeparableHamiltonian):
    """The library's H and dp, and the control with its components swapped: they agree on p_1 = p_2 only."""

    def optimal_control(self, t, x, p, m):
        return -_a(p)[..., ::-1]


class _DpAfterT0(hm.SeparableHamiltonian):
    """The library's H and control, and dp times 1 + 4t: dp is wrong after t = 0 only."""

    def dp(self, t, x, p, m):
        return (1.0 + 4.0 * t) * _a(super().dp(t=t, x=x, p=p, m=m))


class _DmAtTheFirstEnd(hm.SeparableHamiltonian):
    """The library's H, dp and control, and dm doubled where x_1 < 0.25: wrong at the first end node only."""

    def dm(self, t, x, p, m):
        return np.where(_a(x[0]) < 0.25, 2.0, 1.0) * _a(super().dm(t=t, x=x, p=p, m=m))


def _door_doubled(door):
    class Doubled(hm.SeparableHamiltonian):
        pass

    setattr(Doubled, door, lambda self, state: 2.0 * _a(getattr(hm.SeparableHamiltonian, door)(self, state)))
    return Doubled


def _door_doubled_on_batches(door):
    class DoubledOnBatches(hm.SeparableHamiltonian):
        pass

    def read(self, state):
        value = _a(getattr(hm.SeparableHamiltonian, door)(self, state))
        return 2.0 * value if np.ndim(state.p) == 2 and np.shape(state.p)[0] > 1 else value

    setattr(DoubledOnBatches, door, read)
    return DoubledOnBatches


class _PointReadControl(hm.SeparableHamiltonian):
    """H = (1 + x_1)|p|^2/2 - f(m), and a control that reads x as one point: at one point every source agrees, and on a
    batch the control takes the first node's x at every node (#2875's class)."""

    def __call__(self, t, x, p, m):
        return 0.5 * (1.0 + _a(x[0])) * np.sum(_a(p) ** 2, axis=-1) - f(_a(m))

    def dp(self, t, x, p, m):
        scale = 1.0 + _a(x[0])
        return scale[..., None] * _a(p) if np.ndim(p) == 2 else scale * _a(p)

    def optimal_control(self, t, x, p, m):
        return -(1.0 + float(np.ravel(_a(x[0]))[0])) * _a(p)


class _RaisesAtTheFirstEnd(hm.SeparableHamiltonian):
    """The library's H raising where x_1 = 0, an end node."""

    def __call__(self, t, x, p, m):
        if np.any(_a(x[0]) == 0.0):
            raise ValueError("x outside (0, 1]")
        return super().__call__(t=t, x=x, p=p, m=m)


class _WrongDpRaisingAtTheFirstEnd(_RaisesAtTheFirstEnd):
    def dp(self, t, x, p, m):
        return 2.0 * _a(super().dp(t=t, x=x, p=p, m=m))


class _MixRaisingAtTheFirstEnd(_RaisesAtTheFirstEnd):
    def optimal_control(self, t, x, p, m):
        return -2.0 * _a(p)


def _control_raising(where, error, wrong):
    """A control that raises ``error`` where ``where(t, p)``, and is minus dp times 2 (``wrong``) or 1 elsewhere."""

    class Raising(hm.SeparableHamiltonian):
        def optimal_control(self, t, x, p, m):
            if where(t, _a(p)):
                raise error("a control not defined here")
            return (2.0 if wrong else 1.0) * _a(super().optimal_control(t=t, x=x, p=p, m=m))

    return Raising


class _WrittenOut(hm.SeparableHamiltonian):
    """H = |p|^2/2 - f(m) written out: the library's H with no potential, not reading the one the wall is composed into."""

    def __call__(self, t, x, p, m):
        return 0.5 * np.sum(_a(p) ** 2, axis=-1) - f(_a(m))


class _Plain(hm.SeparableHamiltonian):
    """A subclass that changes nothing."""


class _PlainCost(hm.QuadraticControlCost):
    """A cost subclass that changes nothing."""


class _QuadraticLagrangian(hm.LagrangianBase):
    """A user's Lagrangian, not a SeparableLagrangian, so the library forms its Hamiltonian numerically."""

    def __call__(self, t, x, alpha, m):
        return 0.5 * np.sum(_a(alpha) ** 2, axis=-1)


def _quiet(fn):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        logging.disable(logging.CRITICAL)
        try:
            return fn()
        finally:
            logging.disable(logging.NOTSET)


def _grid(dim=1, n=11):
    return TensorProductGrid(
        bounds=[(0.0, 1.0)] * dim, Nx_points=[n] * dim, boundary_conditions=no_flux_bc(dimension=dim)
    )


def _problem(h, dim=1, T=0.5, **kwargs):
    return MFGProblem(
        model=Model(hamiltonian=h, volatility=0.3),
        domain=_grid(dim),
        conditions=Conditions(
            m_initial=lambda x: 0.2 + 0.0 * _a(x[0]),
            u_terminal=lambda x: 2.0 * (_a(x[0]) - 0.8) ** 2,
            T=T,
        ),
        Nt=5,
        **kwargs,
    )


def _quadratic(cost=1.0):
    return hm.QuadraticControlCost(control_cost=cost)


def _refused(make, dim=1, T=0.5):
    with pytest.raises(ValidationError) as caught:
        _quiet(lambda: _problem(make(), dim, T))
    return str(caught.value)


@pytest.mark.parametrize("dim", [1, 2])
@pytest.mark.parametrize(
    ("make", "control"),
    [
        (lambda: _Mix(_quadratic(), coupling=f), "0.8314"),
        (lambda: _OneHalfQuadratic(_quadratic(), coupling=f), "0.4157"),  # the issue's reproduction
    ],
    ids=["Mix", "the issue's reproduction"],
)
def test_a_control_that_is_not_minus_dp_is_refused_at_construction(make, control, dim):
    probe = "[0.5]" if dim == 1 else "[0.5 0.5]"
    message = (
        rf"optimal_control disagrees with dp and with H at t=0, x={re.escape(probe)}, m=0\.5, "
        rf"p={re.escape(_momentum(dim))}: -optimal_control\[0\]={control}, dp\[0\]=1\.2471, and the central difference "
        rf"of H is 1\.2471"
    )
    with pytest.raises(ValidationError, match=message):
        _quiet(lambda: _problem(make(), dim))


def test_a_control_that_disagrees_with_dp_and_h_alike_is_named_against_h():
    """dp is wrong too, so the control is named against H alone."""
    message = _refused(lambda: _WrongDpAndControl(_quadratic(), coupling=f))
    assert re.search(r"optimal_control disagrees with H at t=0, x=\[0\.5\]", message), message
    assert "optimal_control disagrees with dp and with H" not in message


@pytest.mark.parametrize("dim", [1, 2])
@pytest.mark.parametrize(
    ("cls", "t", "x"),
    [
        (_AFTER_T0, "0.25", lambda dim: [0.5] * dim),
        (_ONLY_AT_T, "0.5", lambda dim: [0.5] * dim),
        (_AWAY_FROM_THE_MIDDLE, "0", lambda dim: [0.0] + [0.5] * (dim - 1)),
        (_ONLY_AT_THE_LAST_END, "0", lambda dim: [1.0] + [0.5] * (dim - 1)),
    ],
    ids=["after t = 0", "only at T", "away from the middle", "only at the last end node"],
)
def test_a_control_that_departs_from_minus_dp_off_one_point_and_time_is_refused(cls, t, x, dim):
    """T = 0.5, so T/2 = 0.25; along axis 0 the end nodes are x = 0 and x = 1, the other axes at the middle."""
    probe = np.array2string(np.array(x(dim), dtype=float), precision=4)
    message = (
        rf"optimal_control disagrees with dp and with H at t={re.escape(t)}, x={re.escape(probe)}, m=0\.5, "
        rf"p={re.escape(_momentum(dim))}: "
    )
    with pytest.raises(ValidationError, match=message):
        _quiet(lambda: _problem(cls(_quadratic(), coupling=f), dim))


def test_the_probe_times_are_the_problems_own():
    """At T = 2 the middle time is 1, which a check with T = 0.5 hard-coded would not probe."""
    message = _refused(
        lambda: _scaled_control(lambda t, x, p: 1.0 + 4.0 * max(t - 0.6, 0.0))(_quadratic(), coupling=f), T=2.0
    )
    assert re.search(r"optimal_control disagrees with dp and with H at t=1, x=\[0\.5\]", message), message


@pytest.mark.parametrize(
    ("make", "dim", "momentum"),
    [
        (lambda: _AbsControl(_quadratic(), coupling=f), 1, "[-1.3691]"),
        (lambda: _AbsControl(_quadratic(), coupling=f), 2, None),
        (lambda: _SwappedControl(_quadratic(), coupling=f), 2, None),
    ],
    ids=["-|p| in 1-D", "-|p| in 2-D", "components swapped in 2-D"],
)
def test_the_momenta_take_both_signs_on_every_component(make, dim, momentum):
    """Every probe used to be a(1, ..., 1) with a >= 0, where both of these agree with -dp."""
    message = _refused(make, dim)
    assert "optimal_control disagrees with dp and with H" in message, message
    if momentum is not None:
        assert f"p={momentum}" in message, message


@pytest.mark.parametrize(
    ("make", "where"),
    [
        (lambda: _DpAfterT0(_quadratic(), coupling=f), r"dH_dp\[0\] is not the derivative of H at t=0\.25"),
        (lambda: _DmAtTheFirstEnd(_quadratic(), coupling=f), r"dH_dm is not the derivative of H at t=0, x=\[0\.\]"),
    ],
    ids=["dp wrong after t = 0", "dm wrong at the first end node"],
)
def test_dp_and_dm_are_compared_at_every_probe(make, where):
    assert re.search(where, _refused(make)), _refused(make)


def test_an_even_grid_still_has_2d_plus_1_probes_around_an_interior_middle():
    from mfgarchon.utils.validation.functions import _get_sample_inputs, _probe_points

    grid = TensorProductGrid(bounds=[(0.0, 1.0)] * 2, Nx_points=[10, 10], boundary_conditions=no_flux_bc(dimension=2))
    points = _probe_points(grid, _get_sample_inputs(grid, "x")[0])
    assert len(points) == 5
    assert all(0.0 < c < 1.0 for c in points[0]), points[0]


def test_an_end_node_is_found_on_a_domain_far_from_the_origin():
    """On [1e7, 1e7 + 10] every node is within np.isclose's default relative tolerance of an end."""
    from mfgarchon.utils.validation.functions import _get_sample_inputs, _probe_points

    grid = TensorProductGrid(bounds=[(1e7, 1e7 + 10.0)], Nx_points=[11], boundary_conditions=no_flux_bc(dimension=1))
    points = _probe_points(grid, _get_sample_inputs(grid, "x")[0])
    assert sorted(float(q[0]) for q in points) == [1e7, 1e7 + 5.0, 1e7 + 10.0]


@pytest.mark.parametrize(
    "make",
    [
        lambda: _WrongDpRaisingAtTheFirstEnd(_quadratic(), coupling=f),
        lambda: _MixRaisingAtTheFirstEnd(_quadratic(), coupling=f),
        lambda: _control_raising(lambda t, p: not np.any(p), ValueError, wrong=True)(_quadratic(), coupling=f),
        lambda: _control_raising(lambda t, p: t == 0.5, ValueError, wrong=True)(_quadratic(), coupling=f),
    ],
    ids=[
        "a wrong dp whose H raises at x = 0",
        "Mix whose H raises at x = 0",
        "Mix whose control raises at p = 0",
        "Mix whose control raises at t = T",
    ],
)
def test_a_probe_that_cannot_be_evaluated_is_skipped_not_the_check(make):
    message = _refused(make)
    assert "Could not" in message or "disagree" in message or "not the derivative" in message
    assert "[ERROR]" in message, message


@pytest.mark.parametrize(
    "error", [ValueError, ZeroDivisionError, KeyError], ids=["ValueError", "ArithmeticError", "KeyError"]
)
def test_a_control_that_cannot_be_evaluated_somewhere_is_warned_about_not_refused(error):
    h = _control_raising(lambda t, p: not np.any(p), error, wrong=False)(_quadratic(), coupling=f)
    result = _quiet(lambda: validate_custom_functions(h, h.dm, h.dp, _grid(), optimal_control=h.optimal_control))
    assert result.is_valid, [str(i) for i in result.issues]
    assert any("Could not compare the optimal control with dH/dp at t=0" in str(i) for i in result.issues)


def test_an_assertion_error_is_not_an_evaluation_error_and_propagates():
    h = _control_raising(lambda t, p: not np.any(p), AssertionError, wrong=False)
    with pytest.raises(AssertionError, match="a control not defined here"):
        _quiet(lambda: _problem(h(_quadratic(), coupling=f)))


@pytest.mark.parametrize(
    ("door", "method"),
    [
        ("evaluate_H", "__call__"),
        ("evaluate_dp", "dp"),
        ("evaluate_optimal_control", "optimal_control"),
        ("dH_dm", "dm"),
    ],
)
def test_a_door_that_disagrees_with_its_method_is_refused_at_construction(door, method):
    message = _refused(lambda: _door_doubled(door)(_quadratic(), coupling=f))
    assert re.search(rf"{door} and {re.escape(method)} disagree at t=0, x=\[0\.5\]", message), message


class _DoorRaisingAtZero(hm.SeparableHamiltonian):
    """A control door that raises at p = 0, the first probe, and returns twice the control everywhere else."""

    def evaluate_optimal_control(self, state):
        if not np.any(_a(state.p)):
            raise ValueError("a door not defined at p = 0")
        return 2.0 * _a(super().evaluate_optimal_control(state))


class _DualDoorDoubled(hm.DualHamiltonian):
    """A point-only Hamiltonian (``reads_a_batch`` False) whose control door returns twice its control."""

    def evaluate_optimal_control(self, state):
        return 2.0 * _a(super().evaluate_optimal_control(state))


def test_a_door_that_cannot_be_evaluated_at_one_probe_is_compared_at_the_others():
    message = _refused(lambda: _DoorRaisingAtZero(_quadratic(), coupling=f))
    assert "evaluate_optimal_control and optimal_control disagree" in message, message


def test_a_point_only_hamiltonians_door_is_compared_at_one_point():
    """A point-only Hamiltonian is handed one point, not a batch of one, so its door and method are read as it reads."""
    message = _refused(lambda: _DualDoorDoubled(lagrangian=_QuadraticLagrangian()))
    assert "evaluate_optimal_control and optimal_control disagree" in message, message


@pytest.mark.parametrize(
    ("make", "door"),
    [
        (
            lambda: _door_doubled_on_batches("evaluate_optimal_control")(_quadratic(), coupling=f),
            "evaluate_optimal_control",
        ),
        (lambda: _door_doubled_on_batches("dH_dm")(_quadratic(), coupling=f), "dH_dm"),
        (lambda: _PointReadControl(_quadratic(), coupling=f), "evaluate_optimal_control"),
    ],
    ids=["control door doubled on batches", "dH_dm door doubled on batches", "a control that reads x as a point"],
)
def test_a_door_whose_batch_read_disagrees_with_its_point_read_is_refused(make, door):
    with pytest.raises(ValueError, match=rf"\.{door}, at \d samples, reads differently together"):
        _quiet(lambda: _problem(make()))


def _network():
    pytest.importorskip("igraph")
    from mfgarchon.extensions.topology import NetworkMFGProblem
    from mfgarchon.geometry.graph.network_geometry import GridNetwork

    net = GridNetwork(width=3, height=3)
    net.create_network()
    problem = _quiet(lambda: NetworkMFGProblem(geometry=net, T=0.5, Nt=5))
    return problem.hamiltonian_class, problem.geometry


def _network_check(h, geometry):
    return _quiet(
        lambda: validate_custom_functions(
            h, h.dm, h.dp, geometry, optimal_control=h.optimal_control, times=(0.0, 0.25, 0.5)
        )
    )


def _network_subclass(method, factor):
    """The library's NetworkHamiltonian with ``method`` scaled by factor(t, node)."""
    from mfgarchon.extensions.topology import NetworkHamiltonian

    class Scaled(NetworkHamiltonian):
        pass

    def scaled(self, t, x, p, m):
        return factor(t, int(np.ravel(x)[0])) * _a(getattr(NetworkHamiltonian, method)(self, t=t, x=x, p=p, m=m))

    setattr(Scaled, method, scaled)
    return Scaled


@pytest.mark.parametrize(
    ("method", "factor", "message"),
    [
        ("optimal_control", lambda t, i: 2.0, "optimal_control and dp disagree on the network at t=0, node"),
        (
            "optimal_control",
            lambda t, i: 2.0 if t == 0.5 else 1.0,
            "optimal_control and dp disagree on the network at t=0.5",
        ),
        (
            "optimal_control",
            lambda t, i: 2.0 if i == 3 else 1.0,
            "optimal_control and dp disagree on the network at t=0, node 3",
        ),
        ("__call__", lambda t, i: 2.0, "is not the derivative of the network's H at t=0, node 0"),
        ("dm", lambda t, i: 2.0, "dm is not the derivative of the network's H at t=0, node 0"),
        ("dm", lambda t, i: 2.0 if t == 0.5 else 1.0, "dm is not the derivative of the network's H at t=0.5"),
        ("dm", lambda t, i: 2.0 if i == 4 else 1.0, "dm is not the derivative of the network's H at t=0, node 4"),
    ],
    ids=[
        "rates doubled",
        "rates doubled at T only",
        "rates doubled at node 3 only",
        "__call__ doubled",
        "dm doubled",
        "dm doubled at T only",
        "dm doubled at node 4 only",
    ],
)
def test_a_network_hamiltonian_whose_sources_disagree_is_refused(method, factor, message):
    library, geometry = _network()
    accepted = _network_check(library, geometry)
    assert accepted.is_valid, [str(i) for i in accepted.issues]
    wrong = _network_subclass(method, factor)(network_data=library.network_data)
    refused = _network_check(wrong, geometry)
    assert not refused.is_valid
    assert any(message in str(i) for i in refused.issues), [str(i) for i in refused.issues]


def test_a_wall_composed_into_a_call_that_does_not_read_the_potential_is_refused():
    def wall(x):
        return _a(x[0]) > 0.85

    with pytest.raises(ValidationError, match=r"the soft wall composed into _WrittenOut's potential does not reach"):
        _quiet(lambda: _problem(_WrittenOut(_quadratic(), coupling=f), state_penalty=wall, state_penalty_scale=5.0))
    walled = _quiet(lambda: _problem(_Plain(_quadratic(), coupling=f), state_penalty=wall, state_penalty_scale=5.0))
    h, x_wall = walled.hamiltonian_class, (np.array(1.0),)
    unwalled = _Plain(_quadratic(), coupling=f)
    moved = float(h(t=0.0, x=x_wall, p=np.array([0.3]), m=0.5)) - float(
        unwalled(t=0.0, x=x_wall, p=np.array([0.3]), m=0.5)
    )
    assert moved == pytest.approx(-5.0)


@pytest.mark.parametrize(
    ("make", "reason"),
    [
        (lambda: _Plain(_quadratic(), coupling=f), r"its class is _Plain"),
        (
            lambda: hm.SeparableHamiltonian(_PlainCost(control_cost=1.0), coupling=f),
            r"its control cost is a _PlainCost",
        ),
    ],
    ids=["a subclass that changes nothing", "a QuadraticControlCost subclass that changes nothing"],
)
def test_canonical_cs_reads_the_cost_only_as_the_library_ships_it(make, reason):
    from mfgarchon.alg.numerical.hjb_solvers import HJBSemiLagrangianSolver

    problem = _quiet(lambda: _problem(make()))
    message = (
        r"HJBSemiLagrangianSolver\(diffusion_method='canonical_cs'\), run on a 1-D problem: .* this problem's \w+ is not "
        rf"\({reason}"
    )
    with pytest.raises(NotImplementedError, match=message):
        HJBSemiLagrangianSolver(problem, diffusion_method="canonical_cs")


def test_the_control_cost_lambda_is_read_only_as_the_library_ships_it():
    from mfgarchon.alg.numerical.hjb_solvers import HJBFDMSolver

    solver = HJBFDMSolver(_quiet(lambda: _problem(_Plain(_quadratic(), coupling=f))))
    with pytest.raises(
        NotImplementedError,
        match=r"^HJBFDMSolver reads the control cost's lambda_ .* this problem's _Plain is not \(its class is _Plain\)",
    ):
        solver._control_cost_lambda()


def _g(x0):
    return 3.0 * np.sin(np.pi * _a(x0))


class _ExtraTerm(hm.SeparableHamiltonian):
    """The library's H minus g(x) in `__call__`: g is p- and m-free, so dp, dm and the control are the library's, and
    construction accepts it; the cost's Lagrangian does not carry g."""

    def __call__(self, t, x, p, m):
        return _a(super().__call__(t=t, x=x, p=p, m=m)) - _g(x[0])


class _PlainL1(hm.L1ControlCost):
    """An L1 cost subclass that changes nothing."""


class _TwiceTheLagrangian(hm.QuadraticControlCost):
    """The quadratic cost's evaluate, dp and control, and twice its Lagrangian."""

    def lagrangian(self, alpha):
        return 2.0 * super().lagrangian(alpha)


class _SaysItIsNotSmooth(hm.SeparableHamiltonian):
    """A quadratic H that states it is not smooth: the route follows the Hamiltonian's statement, not its cost's."""

    def is_smooth(self):
        return False


def _l1():
    return hm.L1ControlCost(control_cost=1.0)


def test_a_lagrangian_is_derived_from_the_cost_only_as_the_library_ships_both():
    """User ruling, 2026-11-11: the cost stands for H only as the library ships both, for the readers of its Lagrangian."""
    assert isinstance(
        _quiet(lambda: _problem(hm.SeparableHamiltonian(_l1(), coupling=f))).lagrangian_class, hm.SeparableLagrangian
    )
    assert _quiet(lambda: _problem(_ExtraTerm(_l1(), coupling=f))).lagrangian_class is None
    assert (
        _quiet(lambda: _problem(hm.SeparableHamiltonian(_PlainL1(control_cost=1.0), coupling=f))).lagrangian_class
        is None
    )


@pytest.mark.parametrize(
    ("make", "reason"),
    [
        (lambda: _Plain(_l1(), coupling=f), "its class is _Plain"),
        (lambda: hm.SeparableHamiltonian(_PlainL1(control_cost=1.0), coupling=f), "its control cost is a _PlainL1"),
        (lambda: _ExtraTerm(_l1(), coupling=f), "its class is _ExtraTerm"),
    ],
    ids=["a subclass that changes nothing", "an L1 cost subclass that changes nothing", "the g(x) subclass"],
)
def test_the_semi_lagrangian_dpp_path_reads_the_cost_only_as_shipped(make, reason):
    from mfgarchon.alg.numerical.hjb_solvers import HJBSemiLagrangianSolver

    problem = _quiet(lambda: _problem(make()))
    with pytest.raises(
        NotImplementedError,
        match=rf"HJBSemiLagrangianSolver, run on a 1-D problem with a non-smooth \w+: its DPP path .* \({re.escape(reason)}",
    ):
        HJBSemiLagrangianSolver(problem)


@pytest.mark.parametrize(
    ("make", "reason"),
    [
        (lambda: _Plain(_quadratic(), coupling=f), "its class is _Plain"),
        (lambda: hm.SeparableHamiltonian(_PlainCost(control_cost=1.0), coupling=f), "its control cost is a _PlainCost"),
        (
            lambda: hm.SeparableHamiltonian(_TwiceTheLagrangian(control_cost=1.0), coupling=f),
            "its control cost is a _TwiceTheLagrangian",
        ),
        (lambda: _ExtraTerm(_quadratic(), coupling=f), "its class is _ExtraTerm"),
    ],
    ids=[
        "a subclass that changes nothing",
        "a cost subclass that changes nothing",
        "a cost overriding only lagrangian",
        "the g(x) subclass",
    ],
)
def test_howard_reads_the_cost_only_as_shipped(make, reason):
    from mfgarchon.alg.numerical.hjb_solvers import HJBGFDMSolver

    problem = _quiet(lambda: _problem(make()))
    x = np.linspace(0.0, 1.0, 11)
    solver = _quiet(lambda: HJBGFDMSolver(problem, collocation_points=x[:, None], delta=0.35, inner_solver="howard"))
    m = np.ones((6, 11))
    with pytest.raises(
        NotImplementedError,
        match=rf"HJBGFDMSolver\(inner_solver='howard'\), run on a 1-D problem: .* \({re.escape(reason)}",
    ):
        _quiet(lambda: solver.solve_hjb_system(m, np.zeros(11), np.zeros_like(m)))


@pytest.mark.parametrize(
    ("make", "reason"),
    [
        (lambda: _Plain(_quadratic(), coupling=f), "its class is _Plain"),
        (lambda: hm.SeparableHamiltonian(_PlainCost(control_cost=1.0), coupling=f), "its control cost is a _PlainCost"),
    ],
    ids=["a subclass that changes nothing", "a cost subclass that changes nothing"],
)
def test_positive_qp_fan_reads_the_cost_only_as_shipped(make, reason):
    pytest.importorskip("cvxpy", reason="positive_qp_fan requires cvxpy (the `numerical` extra)")
    from mfgarchon.alg.numerical.hjb_solvers import HJBGFDMSolver

    xs = np.linspace(0.0, 1.0, 9)
    points = np.array([[a, b] for a in xs for b in xs])
    boundary = np.array([i for i, (a, b) in enumerate(points) if a in (0.0, 1.0) or b in (0.0, 1.0)])
    problem = _quiet(lambda: _problem(make(), dim=2))
    with pytest.raises(ValueError, match=rf"positive_qp_fan'\) needs a SeparableHamiltonian .* \({re.escape(reason)}"):
        _quiet(
            lambda: HJBGFDMSolver(
                problem,
                collocation_points=points,
                boundary_indices=boundary,
                delta=0.3,
                inner_solver="howard",
                monotonicity_scheme="positive_qp_fan",
            )
        )


def test_the_fp_route_reads_the_hamiltonians_own_smoothness(monkeypatch):
    """The Hamiltonian's is_smooth(), not its cost's: a quadratic H that states it is not smooth takes the velocity
    route, on FVM, which forms no drift from U."""
    import sys

    import mfgarchon.alg.numerical.coupling.fixed_point_utils as fpu
    from mfgarchon.alg.numerical.fp_solvers.fp_fvm import FPFVMSolver
    from mfgarchon.alg.numerical.hjb_solvers import HJBFDMSolver

    original, seen = fpu.resolve_fp_drift_kwargs, []

    def recording(*args, **kwargs):
        out = original(*args, **kwargs)
        seen.append(tuple(sorted(out[0])))
        return out

    for module in list(sys.modules.values()):
        if module is not None and getattr(module, "resolve_fp_drift_kwargs", None) is original:
            monkeypatch.setattr(module, "resolve_fp_drift_kwargs", recording)
    for make, key in (
        (lambda: hm.SeparableHamiltonian(_quadratic(), coupling=f), "potential_field"),
        (lambda: _SaysItIsNotSmooth(_quadratic(), coupling=f), "drift_field"),
    ):
        seen.clear()
        problem = _quiet(lambda make=make: _problem(make()))
        _quiet(
            lambda problem=problem: problem.solve(
                hjb_solver=HJBFDMSolver(problem), fp_solver=FPFVMSolver(problem), max_iterations=1, verbose=False
            )
        )
        assert set(seen) == {(key,)}, seen


class _NarrowL1(hm.L1ControlCost):
    """An L1 cost whose admissible set alone is narrowed: the Hamiltonian's evaluate and dp are the parent's."""

    def effective_domain(self):
        return (-0.25, 0.25)


def _lagrangian_components(lagrangian):
    from mfgarchon.core.mfg_components import MFGComponents

    return MFGComponents(
        lagrangian=lagrangian,
        m_initial=lambda x: 1.0 + 0.0 * _a(x[0]),
        u_terminal=lambda x: (_a(x[0]) - 0.5) ** 2,
    )


def _components_problem(components):
    return MFGProblem(geometry=_grid(1, 21), T=0.5, Nt=10, volatility=0.3, components=components)


def test_a_hamiltonian_and_a_lagrangian_given_together_are_refused():
    """As Model refuses them: every reader but HJB-SL's DPP path read H, the DPP path read L, and nothing compared."""
    from mfgarchon.core.mfg_components import MFGComponents

    with pytest.raises(
        ValueError, match=r"Provide hamiltonian OR lagrangian, not both\. They are dual descriptions of the same game"
    ):
        MFGComponents(
            hamiltonian=hm.SeparableHamiltonian(hm.L1ControlCost(lambda_=0.5)),
            lagrangian=hm.SeparableLagrangian(control_cost=_NarrowL1(lambda_=0.5)),
            m_initial=lambda x: 1.0,
            u_terminal=lambda x: 0.0,
        )


def test_a_copy_of_a_lagrangian_defined_components_carries_one_description():
    """The derived Hamiltonian is held privately, so dataclasses.replace hands __post_init__ the Lagrangian alone."""
    import dataclasses

    components = _lagrangian_components(hm.SeparableLagrangian(control_cost=_quadratic()))
    assert components.hamiltonian is None
    copied = dataclasses.replace(components, m_initial=lambda x: 2.0 + 0.0 * _a(x[0]))
    assert isinstance(_quiet(lambda: _components_problem(copied)).hamiltonian_class, hm.SeparableHamiltonian)


def test_a_lagrangian_whose_cost_is_not_as_shipped_is_refused_on_the_dpp_path():
    """Its Hamiltonian, derived from it, shares the cost: HJB-FDM read the parent's evaluate and dp, the DPP path the
    narrowed domain (measured: max|U_SL - U_SL,shipped| 1.3e-02, against 0 for HJB-FDM)."""
    from mfgarchon.alg.numerical.hjb_solvers import HJBSemiLagrangianSolver

    problem = _quiet(
        lambda: _components_problem(_lagrangian_components(hm.SeparableLagrangian(control_cost=_NarrowL1(lambda_=0.5))))
    )
    with pytest.raises(
        NotImplementedError,
        match=r"its DPP path solves a Lagrangian built on the control cost.*its control cost is a _NarrowL1",
    ):
        HJBSemiLagrangianSolver(problem)


def test_a_lagrangian_defined_problem_with_a_shipped_cost_runs_on_the_dpp_path():
    from mfgarchon.alg.numerical.hjb_solvers import HJBSemiLagrangianSolver

    problem = _quiet(
        lambda: _components_problem(
            _lagrangian_components(hm.SeparableLagrangian(control_cost=hm.L1ControlCost(lambda_=0.5)))
        )
    )
    solver = HJBSemiLagrangianSolver(problem)
    assert solver._use_dpp
    x = np.linspace(0.0, 1.0, 21)
    m = np.ones((11, 21))
    assert np.all(np.isfinite(_quiet(lambda: solver.solve_hjb_system(m, (x - 0.5) ** 2, np.zeros_like(m)))))


def test_a_users_lagrangian_base_is_unaffected():
    """A user's LagrangianBase is one source by construction: its Hamiltonian is the Legendre dual of it."""
    from mfgarchon.alg.numerical.hjb_solvers import HJBSemiLagrangianSolver

    problem = _quiet(lambda: _components_problem(_lagrangian_components(_QuadraticLagrangian())))
    assert isinstance(problem.hamiltonian_class, hm.DualHamiltonian)
    HJBSemiLagrangianSolver(problem)


def test_the_dmp_warning_reads_the_hamiltonians_dp_at_the_steps_density(mfg_caplog):
    """On a congestion H with lambda = 0.5, max|dp| at the step's density lies apart from |grad u| / lambda and from dp at
    m = 1, so a threshold between them warns or stays silent only for the head's read."""
    from mfgarchon.alg.numerical.hjb_solvers import HJBGFDMSolver

    h = hm.CongestionHamiltonian(
        control_cost=_quadratic(0.5),
        congestion_factor=lambda m: 1 + 3.0 * m,
        congestion_factor_dm=lambda m: 3.0 + 0 * m,
    )
    problem = _quiet(lambda: _problem(h))
    x = np.linspace(0.0, 1.0, 11)
    solver = _quiet(
        lambda: HJBGFDMSolver(
            problem,
            x[:, None],
            delta=0.25,
            monotonicity_scheme="joint_socp",
            monotonicity_application="precompute",
            check_dmp=True,
        )
    )
    _quiet(lambda: solver.assemble_hjb_iteration_matrix(np.zeros((solver.n_points, 1))))
    u, m = 2.0 * x, np.full(11, 2.0)
    grad = np.stack([solver._D_grad[0] @ u], axis=1)
    head = float(np.max(np.abs(h.dp(t=0.0, x=(x,), p=grad, m=m))))
    rivals = (float(np.max(np.abs(grad))) / 0.5, float(np.max(np.abs(h.dp(t=0.0, x=(x,), p=grad, m=np.ones(11))))))
    assert head < min(rivals)
    for threshold, warns in ((0.5 * (head + min(rivals)), False), (0.5 * head, True)):
        solver._dmp_alpha_crit, solver._dmp_warned = threshold, False
        mfg_caplog.clear()
        with mfg_caplog.at_level(logging.WARNING, logger="mfgarchon.alg.numerical.hjb_solvers.hjb_gfdm"):
            solver._maybe_warn_dmp(u, m, 0.0)
        assert any("DMP not guaranteed" in r for r in mfg_caplog.messages) is warns, (threshold, head, rivals)


@pytest.mark.parametrize("dim", [1, 2])
@pytest.mark.parametrize(
    "make",
    [
        lambda: hm.SeparableHamiltonian(_quadratic(0.7), coupling=f),
        lambda: hm.SeparableHamiltonian(hm.L1ControlCost(control_cost=1.0), coupling=f),
        lambda: hm.SeparableHamiltonian(hm.BoundedControlCost(control_cost=1.0, max_control=2.0), coupling=f),
        lambda: hm.QuadraticMFGHamiltonian(),
        lambda: hm.CongestionHamiltonian(
            control_cost=_quadratic(),
            congestion_factor=lambda m: 1 + 0.8 * m,
            congestion_factor_dm=lambda m: 0.8 + 0 * m,
        ),
        lambda: hm.SeparableHamiltonian(hm.L1ControlCost(control_cost=1.0)).regularize(0.1),
        lambda: hm.SeparableLagrangian(control_cost=_quadratic()).as_hamiltonian(),
        lambda: _QuadraticLagrangian().legendre_transform(),
    ],
    ids=[
        "quadratic",
        "L1",
        "bounded",
        "QuadraticMFGHamiltonian",
        "congestion",
        "Moreau-Yosida",
        "as_hamiltonian",
        "Dual",
    ],
)
def test_every_library_hamiltonian_is_accepted(make, dim):
    """The control every reader must accept: the library's own Hamiltonians, whose sources agree."""
    assert _quiet(lambda: _problem(make(), dim)).hamiltonian_class is not None
