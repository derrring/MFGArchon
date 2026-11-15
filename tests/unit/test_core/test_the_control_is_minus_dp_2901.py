"""A Hamiltonian's control is -dp, and the consistency check reads every source through one reader (#2901, stage 1c-i).

The library reads a Hamiltonian through several sources: the HJB solvers read `__call__` and `dp`, and the FP solvers
read `optimal_control`. Before 1c-i the construction check `validate_hamiltonian_consistency` compared `dp` and `dm` with
`__call__` at one point at t = 0, and nothing compared the control, so a subclass whose `optimal_control` was another
H's was solved as a mixture with no error.

- **The control against -dp.** `Mix` and the issue's reproduction are refused at construction, and the message names
  the pair that disagrees and the probe (t, x, m, p) where it does: against dp and H when dp agrees with H, against H
  alone when it does not.
- **The probes vary t, x, m and p.** A control that departs from -dp only at T/2, only at T, only at an axis's end node,
  only for p < 0, or only off the diagonal is refused; so are a dp wrong only after t = 0, a dm wrong only at an end node,
  and a dm wrong only at m = 1. The probe times are the problem's own T (a row at T = 2). On a grid with an even number
  of nodes the middle probe is the node nearest the centre, so there are still 2d + 1. In 3-D each component is probed
  at a magnitude of its own, so a 1<->3 swap is refused, with the 1<->2 swap as the control.
- **Each comparison has its own guard.** A probe at which a source cannot be evaluated is warned about and skipped for
  that comparison alone, so a witness found at another probe still refuses. H, dp or the control raising at every probe
  at which it could be read is refused; dm raising everywhere is warned about and not refused, since no grid solver
  reads dm. An `AssertionError` is not an evaluation error and propagates.
- **What a source returns goes through the owner of its shape** (#2875). A value or a dm of shape (1,) at a point, and a
  dp or a control of shape (1, d), are refused at construction, naming the method and the shapes; the same values in
  the contract's shapes are accepted. A shape refusal at one probe is not skipped. A return of the right shape that
  holds no number, `None`, is a probe the source could not be evaluated at, not a NaN value.
- **The control every reader must accept** (a necessary oracle): the library's own Hamiltonians, in 1-D and 2-D.
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
    """The first nonzero momentum probe, as the messages print it: component k a scale times (-1)**k (1 + 0.3 k)."""
    return np.array2string(scale * np.array([(-1.0) ** k * (1.0 + 0.3 * k) for k in range(dim)]), precision=4)


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
    """H = |p|^2/2 - m/2 written out, every method its own, read one point at a time; subclasses break one."""

    reads_a_batch = False

    def __call__(self, t, x, p, m):
        return 0.5 * np.sum(_a(p) ** 2, axis=-1) - 0.5 * _a(m)

    def dp(self, t, x, p, m):
        return _a(p)

    def optimal_control(self, t, x, p, m):
        return -_a(p)

    def dm(self, t, x, p, m):
        return -0.5 * np.ones_like(_a(m))


class _DmRaises(_WrittenOutH):
    def dm(self, t, x, p, m):
        raise NotImplementedError("no dH/dm written")


class _DmRaisesControlDoubled(_DmRaises):
    def optimal_control(self, t, x, p, m):
        return -2.0 * _a(p)


class _DmRaisingAtZeroMomentumWrongElsewhere(hm.SeparableHamiltonian):
    """dm raises at p = 0, the first probe of each (t, x, m), and is doubled elsewhere."""

    def dm(self, t, x, p, m):
        if not np.any(_a(p)):
            raise ValueError("dm not defined at p = 0")
        return 2.0 * _a(super().dm(t=t, x=x, p=p, m=m))


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


def _problem(h, dim=1, T=0.5):
    return MFGProblem(
        model=Model(hamiltonian=h, volatility=0.3),
        domain=_grid(dim),
        conditions=Conditions(
            m_initial=lambda x: 0.2 + 0.0 * _a(x[0]),
            u_terminal=lambda x: 2.0 * (_a(x[0]) - 0.8) ** 2,
            T=T,
        ),
        Nt=5,
    )


def _quadratic(cost=1.0):
    return hm.QuadraticControlCost(control_cost=cost)


def _refused(make, dim=1, T=0.5):
    with pytest.raises(ValidationError) as caught:
        _quiet(lambda: _problem(make(), dim, T))
    return str(caught.value)


def _checked(h):
    """The consistency check as a problem runs it, at T = 0.5, on the 11-node grid."""
    return _quiet(
        lambda: validate_custom_functions(
            h, h.dm, h.dp, _grid(), optimal_control=h.optimal_control, times=(0.0, 0.25, 0.5)
        )
    )


# --- The control against -dp ---------------------------------------------------------------------------------------


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


# --- The probes vary t, x, m and p ---------------------------------------------------------------------------------


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


def _swapped(first, second):
    class Swapped(hm.SeparableHamiltonian):
        def optimal_control(self, t, x, p, m):
            value = _a(super().optimal_control(t=t, x=x, p=p, m=m)).copy()
            value[..., [first, second]] = value[..., [second, first]]
            return value

    return Swapped


@pytest.mark.parametrize(("first", "second"), [(0, 2), (0, 1)], ids=["1<->3", "1<->2"])
def test_each_component_is_probed_at_a_magnitude_of_its_own(first, second):
    """With the signs alone, components 1 and 3 were equal at every probe in 3-D, so their swap was not seen (#2901 review
    3, M1); the 1<->2 swap is the control."""
    message = _refused(lambda: _swapped(first, second)(_quadratic(), coupling=f), dim=3)
    assert "optimal_control disagrees with dp and with H" in message, message


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
    message = _refused(make)
    assert re.search(where, message), message


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


# --- Each comparison has its own guard -----------------------------------------------------------------------------


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
    result = _checked(make())
    assert not result.is_valid, [str(i) for i in result.issues]
    assert any("Could not evaluate" in str(i) for i in result.issues), [str(i) for i in result.issues]


@pytest.mark.parametrize(
    "error", [ValueError, ZeroDivisionError, KeyError], ids=["ValueError", "ArithmeticError", "KeyError"]
)
def test_a_control_that_cannot_be_evaluated_somewhere_is_warned_about_not_refused(error):
    result = _checked(_control_raising(lambda t, p: not np.any(p), error, wrong=False)(_quadratic(), coupling=f))
    assert result.is_valid, [str(i) for i in result.issues]
    assert any("Could not evaluate optimal_control at t=0" in str(i) for i in result.issues)


def test_an_assertion_error_is_not_an_evaluation_error_and_propagates():
    h = _control_raising(lambda t, p: not np.any(p), AssertionError, wrong=False)
    with pytest.raises(AssertionError, match="a control not defined here"):
        _quiet(lambda: _problem(h(_quadratic(), coupling=f)))


class _DpRaisingEverywhere(_WrittenOutH):
    def dp(self, t, x, p, m):
        raise NotImplementedError("no dp written")


class _ControlRaisingEverywhere(_WrittenOutH):
    def optimal_control(self, t, x, p, m):
        raise NotImplementedError("no control written")


class _HRaisingEverywhere(_WrittenOutH):
    def __call__(self, t, x, p, m):
        raise NotImplementedError("no H written")


class _HRaisingAtTheFirstEndControlEverywhere(_ControlRaisingEverywhere):
    """H raises at x = 0, one of the three probe points, so it can be read at 36 of the 54 probes."""

    def __call__(self, t, x, p, m):
        if float(_a(x[0])) == 0.0:
            raise ValueError("H not defined at x = 0")
        return super().__call__(t=t, x=x, p=p, m=m)


@pytest.mark.parametrize(
    ("make", "message"),
    [
        (_DpRaisingEverywhere, "dH_dp could not be evaluated at any probe at which H could (54 of them)"),
        (
            _ControlRaisingEverywhere,
            "optimal_control could not be evaluated at any probe at which H could (54 of them)",
        ),
        (_HRaisingEverywhere, "The Hamiltonian could not be evaluated at any of the 54 probes"),
        (
            _HRaisingAtTheFirstEndControlEverywhere,
            "optimal_control could not be evaluated at any probe at which H could (36 of them)",
        ),
    ],
    ids=["dp", "the control", "H", "the control, counted where H could be read"],
)
def test_a_source_the_solvers_read_that_raises_at_every_probe_is_refused(make, message):
    """1-D: 3 probe points, 3 times and 6 (m, p), so 54 probes. A method is counted against the probes at which H could
    be read, so one raising wherever H can be read is refused even where H itself raises at some probes."""
    assert message in _refused(make)


def test_a_dm_that_raises_at_every_probe_is_warned_about_and_the_rest_still_compared():
    """No grid solver reads dm (#2901's census), so a dm that cannot be evaluated anywhere is warned about, and does not
    discard the other comparisons: the same H with its control doubled is refused."""
    result = _checked(_DmRaises())
    assert result.is_valid, [str(i) for i in result.issues]
    assert any("Could not evaluate dH_dm at t=0" in str(i) for i in result.issues), [str(i) for i in result.issues]
    message = _refused(_DmRaisesControlDoubled)
    assert "optimal_control disagrees with dp and with H" in message, message


def test_a_dm_that_cannot_be_evaluated_somewhere_is_warned_about_and_compared_at_the_others():
    result = _checked(_DmRaisingAtZeroMomentumWrongElsewhere(_quadratic(), coupling=f))
    assert any("Could not evaluate dH_dm at t=0" in str(i) for i in result.issues), [str(i) for i in result.issues]
    assert any("dH_dm is not the derivative of H" in str(i) for i in result.issues), [str(i) for i in result.issues]
    assert not result.is_valid


class _DpRaisingAtZeroControlWrongThere(hm.SeparableHamiltonian):
    def dp(self, t, x, p, m):
        if not np.any(_a(p)):
            raise ValueError("dp not defined at p = 0")
        return super().dp(t=t, x=x, p=p, m=m)

    def optimal_control(self, t, x, p, m):
        value = _a(super().optimal_control(t=t, x=x, p=p, m=m))
        return value + 0.3 if not np.any(_a(p)) else value


class _ControlRaisingAtZeroDpWrongThere(hm.SeparableHamiltonian):
    def optimal_control(self, t, x, p, m):
        if not np.any(_a(p)):
            raise ValueError("control not defined at p = 0")
        return super().optimal_control(t=t, x=x, p=p, m=m)

    def dp(self, t, x, p, m):
        value = _a(super().dp(t=t, x=x, p=p, m=m))
        return value + 0.3 if not np.any(_a(p)) else value


@pytest.mark.parametrize(
    ("make", "message"),
    [
        (
            lambda: _DpRaisingAtZeroControlWrongThere(_quadratic(), coupling=f),
            "optimal_control disagrees with H at t=0",
        ),
        (
            lambda: _ControlRaisingAtZeroDpWrongThere(_quadratic(), coupling=f),
            "dH_dp[0] is not the derivative of H at t=0",
        ),
    ],
    ids=["dp raising, the control wrong there", "the control raising, dp wrong there"],
)
def test_one_source_raising_at_a_probe_leaves_the_others_compared_there(make, message):
    assert message in _refused(make)


# --- What a source returns goes through the owner of its shape -----------------------------------------------------


def _row(p):
    """One point's vector as a (1, d) row, and a batch as it is."""
    return _a(p).reshape(1, -1) if _a(p).ndim == 1 else _a(p)


class _ValueOfOneEntry(_WrittenOutH):
    def __call__(self, t, x, p, m):
        return np.atleast_1d(super().__call__(t=t, x=x, p=p, m=m))


class _DmOfOneEntry(_WrittenOutH):
    def dm(self, t, x, p, m):
        return np.atleast_1d(super().dm(t=t, x=x, p=p, m=m))


class _DpRow(_WrittenOutH):
    def dp(self, t, x, p, m):
        return _row(p)


class _ControlRow(_WrittenOutH):
    def optimal_control(self, t, x, p, m):
        return -_row(p)


@pytest.mark.parametrize("dim", [1, 2])
@pytest.mark.parametrize(
    ("make", "location", "method", "returned"),
    [
        (_ValueOfOneEntry, "hamiltonian", "__call__", lambda dim: (1,)),
        (_DmOfOneEntry, "dH_dm", "dm", lambda dim: (1,)),
        (_DpRow, "dH_dp", "dp", lambda dim: (1, dim)),
        (_ControlRow, "optimal_control", "optimal_control", lambda dim: (1, dim)),
    ],
    ids=["a value of shape (1,)", "a dm of shape (1,)", "a dp row", "a control row"],
)
def test_a_return_in_another_shape_than_its_contract_is_refused_at_construction(make, location, method, returned, dim):
    """Each is the right value in a shape the owner of its return's shape refuses (#2875; strict, the approver's ruling
    of 2026-11-15): one value per point has m's shape, (), and dp and the control have p's, (d,)."""
    message = _refused(make, dim)
    expected = (
        rf"\[ERROR\] \({location}\) validate_hamiltonian_consistency: {make.__name__}\.{re.escape(method)} was handed "
        rf"p of shape \({dim},\) and m of shape \(\), and returned shape {re.escape(str(returned(dim)))}\."
    )
    assert re.search(expected, message), message


@pytest.mark.parametrize("dim", [1, 2])
def test_the_same_values_in_the_contracts_shapes_are_accepted(dim):
    """The control for the row above: the same H with every return in its contract's shape."""
    assert _quiet(lambda: _problem(_WrittenOutH(), dim)).hamiltonian_class is not None


class _TwoValuesAtTheLastEnd(_WrittenOutH):
    """A point H that returns two values at x = 1 and one everywhere else."""

    def __call__(self, t, x, p, m):
        value = super().__call__(t=t, x=x, p=p, m=m)
        return np.array([value, value]) if float(_a(x[0])) == 1.0 else value


def test_a_return_of_another_shape_at_one_probe_is_refused_not_skipped():
    """A shape error breaks the source's contract, so it is refused at the one probe where it occurs, not skipped."""
    message = _refused(_TwoValuesAtTheLastEnd)
    assert "_TwoValuesAtTheLastEnd.__call__ was handed p of shape (1,) and m of shape (), and returned shape (2,)" in (
        message
    ), message
    assert "At t=0, x=[1.], m=0.5, p=[0.]." in message, message


class _ValueNoneAtTheLastEnd(_WrittenOutH):
    """H returns None at x = 1: shape (), as one value at a point has, and no number."""

    def __call__(self, t, x, p, m):
        return None if float(_a(x[0])) == 1.0 else super().__call__(t=t, x=x, p=p, m=m)


class _ControlNoneAtTheLastEnd(_WrittenOutH):
    """The control returns [None] at x = 1: shape (1,), p's shape in 1-D, and no number."""

    def optimal_control(self, t, x, p, m):
        if float(_a(x[0])) == 1.0:
            return np.array([None], dtype=object)
        return super().optimal_control(t=t, x=x, p=p, m=m)


@pytest.mark.parametrize(
    ("make", "warned"),
    [
        (_ValueNoneAtTheLastEnd, "Could not evaluate the Hamiltonian at t=0, x=[1.]"),
        (_ControlNoneAtTheLastEnd, "Could not evaluate optimal_control at t=0, x=[1.]"),
    ],
    ids=["the value", "the control"],
)
def test_a_return_that_holds_no_number_is_a_probe_not_evaluated_not_a_nan(make, warned):
    """numpy reads None in an object array as NaN; the reader does not, so the probe is warned about as one the source
    could not be evaluated at, naming the TypeError, and the other probes are compared."""
    result = _checked(make())
    issues = [str(i) for i in result.issues]
    assert result.is_valid, issues
    assert any(warned in i and "TypeError" in i for i in issues), issues
    assert not any("non-finite" in i for i in issues), issues


# --- The control every reader must accept --------------------------------------------------------------------------


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
