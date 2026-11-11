"""A Hamiltonian's sources are one H: a user's Hamiltonian whose sources disagree is refused at construction (#2901).

The library reads a Hamiltonian through several sources: `__call__` in the HJB residual, `dp` in its Jacobian,
`optimal_control` in the Fokker–Planck drift, the doors over them on a batch and on a point, for a separable H its cost,
the potential the soft wall is composed into, and on a network the rates. Before #2901 a problem checked `dp` and `dm`
against `__call__` (`validate_hamiltonian_consistency`, #1642), and `evaluate_H` and `evaluate_dp` on a batch against a
point (#1884), and nothing else, so a subclass whose `optimal_control` was another H's was solved as a mixture with no
error. The user ruled (A) on 2026-11-11: refuse it at construction. Three owners decide it: the consistency check (the
methods agree at a point, at its probes), the batch owner `refuse_a_hamiltonian_whose_reads_disagree` (a door's batch
read agrees with its point read), and `_why_not_as_it_ships` (the cost stands for H only as the library ships it).
Stage 1 of #2901: the Lagrangian side is its stage 2.

- **The control against -dp** (a mutation-verified pin). `Mix` and the issue's reproduction are refused at
  construction, and the message names the pair that disagrees and the probe (t, x, m, p) where it does: against dp and
  H when dp agrees with H, against H alone when it does not.
- **The probes vary t, x, m and p** (a mutation-verified pin). A control that departs from -dp only at T/2, only at T,
  only at an axis's first or last end node, only for p < 0, or only off the diagonal p_1 = p_2, is refused; so are a dp
  wrong only after t = 0, a dm wrong only at an end node, and a dm wrong only at m = 1. The probe times are the
  problem's own T (a row at T = 2). On a grid with an even number of nodes the middle probe is the node nearest the
  centre, so there are still 2d + 1.
- **Each comparison has its own guard** (a mutation-verified pin). A wrong dp, and `Mix`, whose H raises at x = 0 are
  refused; so is `Mix` whose control raises at p = 0, the first probe of its comparison, or at t = T. A correct H whose
  control raises at p = 0 is accepted with a warning, and a wrong dm raising at p = 0 is warned about and refused at
  the other probes. A point-only Hamiltonian whose dm raises at every probe is refused, beside its doubled control or
  alone. An `AssertionError` is not an evaluation error and propagates.
- **Each door against its method, at a point** (a mutation-verified pin), and **each door on a batch against a point**
  (the batch owner, at 0, T/2 and T): a door doubled at every input, one doubled only on batches of two or more, and a
  control that reads x as a point, are refused, also where they differ only after t = 0. A door that cannot be
  evaluated at one probe is warned about, and one that raises at every probe is refused on a point-only Hamiltonian.
- **A network Hamiltonian's sources** (a mutation-verified pin): its rates against its dp, and its dp and dm against
  differences of `__call__`, at a bounded set of nodes. Each wrong source is refused, dp against `__call__` included,
  and so is a source wrong only at t = T or only at one probed node, on a network whose four probed nodes are
  distinct. A node that raises does not discard a witness found at another.
- **The soft wall** (a mutation-verified pin): composed into a subclass whose `__call__` does not read the potential, it
  is refused, also for an interior wall between D2's x probes, a subclass that drops it only after t = 0, and one whose
  H raises at a probe; a subclass that changes nothing is walled.
- **The cost stands for H only as shipped** (a mutation-verified pin): `canonical_cs`, `_control_cost_lambda`, HJB-SL's
  DPP path, Howard and positive_qp_fan refuse subclasses, a cost instance with its own `effective_domain`, and a user's
  HamiltonianBase carrying a cost; a user's cost that `regularize()` wraps derives no Lagrangian. Howard still runs on
  `SeparableHamiltonian` and `QuadraticMFGHamiltonian`.
- **HJB-GFDM's DMP warning reads the Hamiltonian's dp at the step's density and time** (a mutation-verified pin), on an
  H whose dp depends on both, and both call sites, Newton's and the loop after Howard's sweep, hand it the step's
  density row and its time.
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


class _DmRightAtOneHalf(hm.SeparableHamiltonian):
    """The library's H, dp and control, and dm doubled where m > 0.75: right at m = 0.5, wrong at the probe m = 1."""

    def dm(self, t, x, p, m):
        return np.where(_a(m) > 0.75, 2.0, 1.0) * _a(super().dm(t=t, x=x, p=p, m=m))


class _WrittenOutH(hm.HamiltonianBase):
    """H = |p|^2/2 - m/2 written out, every method its own; subclasses break one."""

    def __call__(self, t, x, p, m):
        return 0.5 * np.sum(_a(p) ** 2, axis=-1) - 0.5 * _a(m)

    def dp(self, t, x, p, m):
        return _a(p)

    def optimal_control(self, t, x, p, m):
        return -_a(p)

    def dm(self, t, x, p, m):
        return -0.5 * np.ones_like(_a(m))


class _PointOnlyDmRaises(_WrittenOutH):
    """A point-only Hamiltonian (``reads_a_batch`` False) with no dH/dm: the batch owner does not read it."""

    reads_a_batch = False

    def dm(self, t, x, p, m):
        raise NotImplementedError("no dH/dm written")


class _PointOnlyDmRaisesControlDoubled(_PointOnlyDmRaises):
    def optimal_control(self, t, x, p, m):
        return -2.0 * _a(p)


class _DmRaisingAtZeroMomentumWrongElsewhere(hm.SeparableHamiltonian):
    """dm raises at p = 0, the first probe of each (t, x, m), and is doubled elsewhere."""

    def dm(self, t, x, p, m):
        if not np.any(_a(p)):
            raise ValueError("dm not defined at p = 0")
        return 2.0 * _a(super().dm(t=t, x=x, p=p, m=m))


class _ControlPointReadAfterT0(hm.SeparableHamiltonian):
    """H = (1 + 2 t x_1)|p|^2/2 - f(m): every method agrees at a point, and the control reads x as one point. At t = 0 the
    x-dependence vanishes, so only a batch read after t = 0 differs (review 2 of #2952)."""

    def _c(self, t, x):
        return 1.0 + 2.0 * float(t) * _a(x[0])

    def __call__(self, t, x, p, m):
        return 0.5 * self._c(t, x) * np.sum(_a(p) ** 2, axis=-1) - f(_a(m))

    def dp(self, t, x, p, m):
        c = self._c(t, x)
        return c[..., None] * _a(p) if np.ndim(p) == 2 else c * _a(p)

    def optimal_control(self, t, x, p, m):
        return -(1.0 + 2.0 * float(t) * float(np.ravel(_a(x[0]))[0])) * _a(p)


class _ControlDoorDoubledOnBatchesAfterT0(hm.SeparableHamiltonian):
    def evaluate_optimal_control(self, state):
        value = _a(super().evaluate_optimal_control(state))
        return 2.0 * value if np.ndim(state.p) == 2 and np.shape(state.p)[0] > 1 and float(state.t) > 0 else value


class _CarriesCost(hm.HamiltonianBase):
    """A user's HamiltonianBase that carries a shipped QuadraticControlCost and writes the same H out by hand."""

    def __init__(self):
        super().__init__()
        self.control_cost = _quadratic()

    def __call__(self, t, x, p, m):
        return 0.5 * np.sum(_a(p) ** 2, axis=-1) - f(_a(m))

    def dp(self, t, x, p, m):
        return _a(p)

    def optimal_control(self, t, x, p, m):
        return -_a(p)

    def dm(self, t, x, p, m):
        return -0.5 * np.ones_like(_a(m))


def _narrowed(cost):
    """A shipped cost whose instance holds its own effective_domain, the control set HJB-SL's DPP path reads."""
    cost.effective_domain = lambda: (-0.25, 0.25)
    return cost


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


class _WrongDpRaisingAtTheFirstProbe(hm.SeparableHamiltonian):
    """dp doubled, and H raising at the first probe of the check, the middle node at t = 0."""

    def __call__(self, t, x, p, m):
        if float(t) == 0.0 and np.any(_a(x[0]) == 0.5):
            raise ValueError("H not defined at the middle at t = 0")
        return super().__call__(t=t, x=x, p=p, m=m)

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


class _WrittenOutAfterT0(hm.SeparableHamiltonian):
    """The library's H at t = 0, and H written out without the potential after it."""

    def __call__(self, t, x, p, m):
        if float(t) == 0.0:
            return super().__call__(t=t, x=x, p=p, m=m)
        return 0.5 * np.sum(_a(p) ** 2, axis=-1) - f(_a(m))


class _WrittenOutRaisingAtTheFirstEnd(_WrittenOut):
    def __call__(self, t, x, p, m):
        if np.any(_a(x[0]) == 0.0):
            raise ValueError("x outside (0, 1]")
        return super().__call__(t=t, x=x, p=p, m=m)


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
        (
            lambda: _DmRightAtOneHalf(_quadratic(), coupling=f),
            r"dH_dm is not the derivative of H at t=0, x=\[0\.5\], m=1,",
        ),
    ],
    ids=["dp wrong after t = 0", "dm wrong at the first end node", "dm wrong at m = 1 only"],
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
        lambda: _WrongDpRaisingAtTheFirstProbe(_quadratic(), coupling=f),
        lambda: _control_raising(lambda t, p: not np.any(p), ValueError, wrong=True)(_quadratic(), coupling=f),
        lambda: _control_raising(lambda t, p: t == 0.5, ValueError, wrong=True)(_quadratic(), coupling=f),
    ],
    ids=[
        "a wrong dp whose H raises at x = 0",
        "Mix whose H raises at x = 0",
        "a wrong dp whose H raises at the first probe",
        "Mix whose control raises at p = 0",
        "Mix whose control raises at t = T",
    ],
)
def test_a_probe_that_cannot_be_evaluated_is_skipped_not_the_check(make):
    """The failing probe is warned about and the witness still refuses. In the first-probe row and the p = 0 row the
    failure comes first in its comparison's order, so a check that stopped at its first failure would find no witness."""
    h = make()
    result = _quiet(
        lambda: validate_custom_functions(
            h, h.dm, h.dp, _grid(), optimal_control=h.optimal_control, times=(0.0, 0.25, 0.5)
        )
    )
    assert not result.is_valid, [str(i) for i in result.issues]
    assert any("Could not evaluate" in str(i) for i in result.issues), [str(i) for i in result.issues]


@pytest.mark.parametrize(
    "error", [ValueError, ZeroDivisionError, KeyError], ids=["ValueError", "ArithmeticError", "KeyError"]
)
def test_a_control_that_cannot_be_evaluated_somewhere_is_warned_about_not_refused(error):
    h = _control_raising(lambda t, p: not np.any(p), error, wrong=False)(_quadratic(), coupling=f)
    result = _quiet(lambda: validate_custom_functions(h, h.dm, h.dp, _grid(), optimal_control=h.optimal_control))
    assert result.is_valid, [str(i) for i in result.issues]
    assert any("Could not evaluate optimal_control at t=0" in str(i) for i in result.issues)


def test_an_assertion_error_is_not_an_evaluation_error_and_propagates():
    h = _control_raising(lambda t, p: not np.any(p), AssertionError, wrong=False)
    with pytest.raises(AssertionError, match="a control not defined here"):
        _quiet(lambda: _problem(h(_quadratic(), coupling=f)))


@pytest.mark.parametrize(
    ("make", "names"),
    [
        (_PointOnlyDmRaisesControlDoubled, ("dH_dm could not be evaluated at any probe", "optimal_control disagrees")),
        (_PointOnlyDmRaises, ("dH_dm could not be evaluated at any probe",)),
    ],
    ids=["dm raising everywhere, control doubled", "dm raising everywhere, the rest right"],
)
def test_a_method_that_cannot_be_evaluated_at_any_probe_is_refused_on_a_point_only_hamiltonian(make, names):
    """A point-only Hamiltonian's doors are not read at construction by the batch owner, so the consistency check is
    the only reader: a dm raising everywhere no longer discards the other comparisons, and is refused itself."""
    message = _refused(make)
    for name in names:
        assert name in message, message


def test_a_dm_that_cannot_be_evaluated_somewhere_is_warned_about_and_compared_at_the_others():
    h = _DmRaisingAtZeroMomentumWrongElsewhere(_quadratic(), coupling=f)
    result = _quiet(lambda: validate_custom_functions(h, h.dm, h.dp, _grid(), optimal_control=h.optimal_control))
    assert any("Could not evaluate dH_dm at t=0" in str(i) for i in result.issues), [str(i) for i in result.issues]
    assert any("dH_dm is not the derivative of H" in str(i) for i in result.issues), [str(i) for i in result.issues]
    assert not result.is_valid


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


class _DoorRaisingAtZeroOnly(hm.SeparableHamiltonian):
    """A control door that raises at p = 0 and is the method everywhere else."""

    def evaluate_optimal_control(self, state):
        if not np.any(_a(state.p)):
            raise ValueError("a door not defined at p = 0")
        return super().evaluate_optimal_control(state)


class _PointOnlyDoorRaises(_WrittenOutH):
    """A point-only Hamiltonian whose control door raises at every point, while its method is right."""

    reads_a_batch = False

    def evaluate_optimal_control(self, state):
        raise NotImplementedError("no control door written")


def test_a_door_that_cannot_be_evaluated_somewhere_is_warned_about_not_refused():
    h = _DoorRaisingAtZeroOnly(_quadratic(), coupling=f)
    result = _quiet(lambda: validate_custom_functions(h, h.dm, h.dp, _grid(), optimal_control=h.optimal_control))
    assert result.is_valid, [str(i) for i in result.issues]
    assert any(
        "Could not compare evaluate_optimal_control with optimal_control at t=0" in str(i) for i in result.issues
    ), [str(i) for i in result.issues]


def test_a_door_that_cannot_be_evaluated_at_any_probe_is_refused_on_a_point_only_hamiltonian():
    message = _refused(_PointOnlyDoorRaises)
    assert "evaluate_optimal_control could not be evaluated at any probe at which optimal_control could" in message, (
        message
    )


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
        (lambda: _ControlPointReadAfterT0(_quadratic(), coupling=f), "evaluate_optimal_control"),
        (lambda: _ControlDoorDoubledOnBatchesAfterT0(_quadratic(), coupling=f), "evaluate_optimal_control"),
    ],
    ids=[
        "control door doubled on batches",
        "dH_dm door doubled on batches",
        "a control that reads x as a point",
        "a control that reads x as a point, seen after t = 0 only",
        "control door doubled on batches after t = 0 only",
    ],
)
def test_a_door_whose_batch_read_disagrees_with_its_point_read_is_refused(make, door):
    with pytest.raises(ValueError, match=rf"\.{door}, at \d samples, reads differently together"):
        _quiet(lambda: _problem(make()))


def _network(width=3, height=3):
    pytest.importorskip("igraph")
    from mfgarchon.extensions.topology import NetworkMFGProblem
    from mfgarchon.geometry.graph.network_geometry import GridNetwork

    net = GridNetwork(width=width, height=height)
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
        ("__call__", lambda t, i: 2.0, r"dp\[\d+\] is not the derivative of the network's H at t=0, node \d+"),
        ("dm", lambda t, i: 2.0, "dm is not the derivative of the network's H at t=0, node 0"),
        ("dm", lambda t, i: 2.0 if t == 0.5 else 1.0, "dm is not the derivative of the network's H at t=0.5"),
    ],
    ids=[
        "rates doubled",
        "rates doubled at T only",
        "__call__ doubled",
        "dm doubled",
        "dm doubled at T only",
    ],
)
def test_a_network_hamiltonian_whose_sources_disagree_is_refused(method, factor, message):
    library, geometry = _network()
    accepted = _network_check(library, geometry)
    assert accepted.is_valid, [str(i) for i in accepted.issues]
    wrong = _network_subclass(method, factor)(network_data=library.network_data)
    refused = _network_check(wrong, geometry)
    assert not refused.is_valid
    assert any(re.search(message, str(i)) for i in refused.issues), [str(i) for i in refused.issues]


@pytest.mark.parametrize("method", ["optimal_control", "dm"])
@pytest.mark.parametrize(("node", "which"), [(11, "the last"), (6, "the middle, n // 2"), (5, "the highest-degree")])
def test_each_probed_network_node_is_compared(method, node, which):
    """On a 4 x 3 grid network the probed nodes are 0, 11, 6 and 5, four distinct; a source wrong at one of them alone
    is refused, naming it. A node not probed is not seen."""
    library, geometry = _network(4, 3)
    wrong = _network_subclass(method, lambda t, i: 2.0 if i == node else 1.0)(network_data=library.network_data)
    refused = _network_check(wrong, geometry)
    assert not refused.is_valid
    assert any(f"at t=0, node {node}" in str(i) for i in refused.issues), [str(i) for i in refused.issues]


def test_a_network_node_that_cannot_be_evaluated_does_not_discard_a_witness():
    """Rates doubled at node 0 and raising at the last node: the witness at node 0 stands."""
    from mfgarchon.extensions.topology import NetworkHamiltonian

    library, geometry = _network()

    class Wrong(NetworkHamiltonian):
        def optimal_control(self, t, x, p, m):
            node = int(np.ravel(x)[0])
            if node == self.num_nodes - 1:
                raise ValueError("no rates at the last node")
            return (2.0 if node == 0 else 1.0) * _a(super().optimal_control(t=t, x=x, p=p, m=m))

    refused = _network_check(Wrong(network_data=library.network_data), geometry)
    assert not refused.is_valid
    assert any("optimal_control and dp disagree on the network at t=0, node 0" in str(i) for i in refused.issues)
    assert any("Could not compare the network's rates with dp" in str(i) for i in refused.issues)


@pytest.mark.parametrize("dim", [1, 2])
def test_an_interior_wall_between_the_probes_is_read_back(dim):
    """A wall on 0.15 < x_1 < 0.35, the nodes 0.2 and 0.3, is zero at every x of D2's probes (the middle and the end
    nodes); the wall check also reads the grid node where it is largest."""

    def wall(x):
        return (_a(x[0]) > 0.15) & (_a(x[0]) < 0.35)

    with pytest.raises(ValidationError, match=r"the soft wall composed into _WrittenOut's potential does not reach"):
        _quiet(
            lambda: _problem(_WrittenOut(_quadratic(), coupling=f), dim, state_penalty=wall, state_penalty_scale=5.0)
        )


@pytest.mark.parametrize(
    ("make", "where"),
    [
        (lambda: _WrittenOutAfterT0(_quadratic(), coupling=f), "at t=0.25,"),
        (lambda: _WrittenOutRaisingAtTheFirstEnd(_quadratic(), coupling=f), "at t=0, x=[1.]"),
    ],
    ids=["dropped after t = 0 only", "H raising at the first end node"],
)
def test_the_wall_is_read_back_at_every_probe(make, where):
    def wall(x):
        return _a(x[0]) > 0.85

    with pytest.raises(ValidationError) as caught:
        _quiet(lambda: _problem(make(), state_penalty=wall, state_penalty_scale=5.0))
    assert "does not reach its __call__" in str(caught.value), str(caught.value)
    assert where in str(caught.value), str(caught.value)


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
        (
            lambda: hm.SeparableHamiltonian(_narrowed(_l1()), coupling=f),
            "its control cost's instance holds ['effective_domain'] of its own",
        ),
    ],
    ids=[
        "a subclass that changes nothing",
        "an L1 cost subclass that changes nothing",
        "the g(x) subclass",
        "a cost instance with its own effective_domain",
    ],
)
def test_the_semi_lagrangian_dpp_path_reads_the_cost_only_as_shipped(make, reason):
    from mfgarchon.alg.numerical.hjb_solvers import HJBSemiLagrangianSolver

    problem = _quiet(lambda: _problem(make()))
    with pytest.raises(
        NotImplementedError,
        match=rf"HJBSemiLagrangianSolver, run on a 1-D problem with a non-smooth \w+: it solves a non-smooth separable "
        rf"Hamiltonian on its DPP path, .* \({re.escape(reason)}",
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
        (
            lambda: hm.SeparableHamiltonian(_narrowed(_quadratic()), coupling=f),
            "its control cost's instance holds ['effective_domain'] of its own",
        ),
        (_CarriesCost, "its class is _CarriesCost"),
    ],
    ids=[
        "a subclass that changes nothing",
        "a cost subclass that changes nothing",
        "a cost overriding only lagrangian",
        "the g(x) subclass",
        "a cost instance with its own effective_domain",
        "a user's HamiltonianBase carrying a QuadraticControlCost",
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
        (
            lambda: hm.SeparableHamiltonian(_narrowed(_quadratic()), coupling=f),
            "its control cost's instance holds ['effective_domain'] of its own",
        ),
    ],
    ids=[
        "a subclass that changes nothing",
        "a cost subclass that changes nothing",
        "a cost instance with its own effective_domain",
    ],
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


@pytest.mark.parametrize(
    "make",
    [lambda: hm.SeparableHamiltonian(_quadratic(), coupling=f), hm.QuadraticMFGHamiltonian],
    ids=["SeparableHamiltonian", "QuadraticMFGHamiltonian"],
)
def test_howard_still_runs_on_the_library_hamiltonians(make):
    """The class rule's other side, which no refusal row can show: the shipped classes still run."""
    from mfgarchon.alg.numerical.hjb_solvers import HJBGFDMSolver

    problem = _quiet(lambda: _problem(make()))
    x = np.linspace(0.0, 1.0, 11)
    solver = _quiet(lambda: HJBGFDMSolver(problem, collocation_points=x[:, None], delta=0.35, inner_solver="howard"))
    m = np.ones((6, 11))
    assert np.all(np.isfinite(_quiet(lambda: solver.solve_hjb_system(m, (x - 0.5) ** 2, np.zeros_like(m)))))


def test_a_regularized_user_cost_is_still_a_users_cost():
    """``regularize()`` wraps the cost in its Moreau-Yosida envelope; the cost it wraps decides, so no Lagrangian is
    derived from a user's cost through it."""
    assert hm.why_the_cost_is_not_the_hamiltonians(hm.SeparableHamiltonian(_l1()).regularize(0.1)) is None
    user = hm.SeparableHamiltonian(_PlainL1(control_cost=1.0), coupling=f).regularize(0.1)
    why = hm.why_the_cost_is_not_the_hamiltonians(user)
    assert why == (
        "its control cost regularizes a cost that is not as shipped (its control cost is a _PlainL1, not a "
        "QuadraticControlCost or L1ControlCost or BoundedControlCost or _MoreauYosidaControlCost)"
    ), why
    assert _quiet(lambda: _problem(user)).lagrangian_class is None
    shipped = hm.SeparableHamiltonian(_l1(), coupling=f).regularize(0.1)
    assert isinstance(_quiet(lambda: _problem(shipped)).lagrangian_class, hm.SeparableLagrangian)


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
        match=r"it solves a non-smooth separable Hamiltonian on its DPP path, .*its control cost is a _NarrowL1",
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


class _TimeAndDensityScaled(hm.HamiltonianBase):
    """H = (1 + 4t)(1 + 3m)|p|^2/2 - m/2, every method its own: dp depends on the time and the density."""

    def _c(self, t, m):
        return (1.0 + 4.0 * float(t)) * (1.0 + 3.0 * _a(m))

    def __call__(self, t, x, p, m):
        return 0.5 * self._c(t, m) * np.sum(_a(p) ** 2, axis=-1) - 0.5 * _a(m)

    def dp(self, t, x, p, m):
        c = self._c(t, m)
        return c[..., None] * _a(p) if np.ndim(p) == 2 else c * _a(p)

    def optimal_control(self, t, x, p, m):
        return -self.dp(t=t, x=x, p=p, m=m)

    def dm(self, t, x, p, m):
        return 1.5 * (1.0 + 4.0 * float(t)) * np.sum(_a(p) ** 2, axis=-1) - 0.5


def _dmp_solver(h, **kwargs):
    from mfgarchon.alg.numerical.hjb_solvers import HJBGFDMSolver

    problem = _quiet(lambda: _problem(h))
    x = np.linspace(0.0, 1.0, 11)
    return x, _quiet(
        lambda: HJBGFDMSolver(
            problem,
            x[:, None],
            delta=0.25,
            monotonicity_scheme="joint_socp",
            monotonicity_application="precompute",
            check_dmp=True,
            **kwargs,
        )
    )


def test_the_dmp_warning_reads_the_hamiltonians_dp_at_the_steps_density_and_time(mfg_caplog):
    """At t = 0.25 and m = 2, max|dp| lies above |grad u| (the control cost's reading at lambda = 1), dp at m = 1, and
    dp at t = 0, so a threshold between them warns only for the head's read."""
    h = _TimeAndDensityScaled()
    x, solver = _dmp_solver(h)
    _quiet(lambda: solver.assemble_hjb_iteration_matrix(np.zeros((solver.n_points, 1))))
    u, m, t = 2.0 * x, np.full(11, 2.0), 0.25
    grad = np.stack([solver._D_grad[0] @ u], axis=1)
    head = float(np.max(np.abs(h.dp(t=t, x=(x,), p=grad, m=m))))
    rivals = (
        float(np.max(np.abs(grad))),
        float(np.max(np.abs(h.dp(t=t, x=(x,), p=grad, m=np.ones(11))))),
        float(np.max(np.abs(h.dp(t=0.0, x=(x,), p=grad, m=m)))),
    )
    assert head > max(rivals)
    for threshold, warns in ((0.5 * (head + max(rivals)), True), (1.5 * head, False)):
        solver._dmp_alpha_crit, solver._dmp_warned = threshold, False
        mfg_caplog.clear()
        with mfg_caplog.at_level(logging.WARNING, logger="mfgarchon.alg.numerical.hjb_solvers.hjb_gfdm"):
            solver._maybe_warn_dmp(u, m, t)
        assert any("DMP not guaranteed" in r for r in mfg_caplog.messages) is warns, (threshold, head, rivals)


@pytest.mark.parametrize("inner_solver", ["newton", "howard"])
def test_the_dmp_warning_is_handed_the_steps_density_and_time(inner_solver, monkeypatch):
    """Both call sites, the Newton step's and the loop after Howard's sweep, hand the warning the step's density row and
    its time: each density row is distinct, so a row and its time identify each other."""
    x, solver = _dmp_solver(hm.SeparableHamiltonian(_quadratic(), coupling=f), inner_solver=inner_solver)
    seen = []
    monkeypatch.setattr(solver, "_maybe_warn_dmp", lambda u, m, t: seen.append((np.array(m, dtype=float), float(t))))
    rows = 1.0 + 0.1 * np.arange(6)[:, None] + 0.0 * x
    _quiet(lambda: solver.solve_hjb_system(rows, (x - 0.5) ** 2, np.zeros_like(rows)))
    assert seen
    dt = 0.5 / 5
    for m, t in seen:
        n = int(np.argmin(np.abs(rows[:, 0] - m[0])))
        assert np.allclose(m, rows[n]), (m[0], t, n)
        assert t == pytest.approx(n * dt), (m[0], t, n)
    assert any(t > 0 for _, t in seen)


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
