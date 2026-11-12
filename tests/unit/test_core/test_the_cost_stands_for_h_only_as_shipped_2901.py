"""The control cost, and a Lagrangian built on it, stand for the Hamiltonian only as the library ships both (#2901, stage 1b).

User ruling, 2026-11-11: the class rule. A reader that takes a quantity from the cost, or from the Lagrangian derived from
it, in place of the Hamiltonian's own methods reads it only for a ``SeparableHamiltonian`` or ``QuadraticMFGHamiltonian``
and a cost exactly as the library ships them; anything else is refused, saying what was run. One owner decides it,
#2914's `_why_not_as_it_ships`, read through `why_the_cost_is_not_the_hamiltonians`.

- **The readers refuse a Hamiltonian or a cost that is not as shipped** (mutation-verified pins): HJB-SL's DPP path and
  ``diffusion_method='canonical_cs'``, ``BaseHJBSolver._control_cost_lambda``, HJB-GFDM's ``inner_solver='howard'`` and
  ``monotonicity_scheme='positive_qp_fan'``, and ``HJBHowardSolver(discretisation='guarded_fan')``. Before, each decided
  by a test of its own, and a subclass whose ``__call__`` carried another term was solved without it.
- **``MFGComponents`` derives a Lagrangian only there** (a mutation-verified pin), through a ``regularize()`` wrapper too.
- **The FP route asks the Hamiltonian, not its cost, whether it is smooth** (a mutation-verified pin).
- **HJB-GFDM's DMP warning reads |grad u| / lambda only where the cost stands for H** (mutation-verified pins): there it
  fires as before, and elsewhere it is not run, saying so, where it read the cost's lambda, or problem.lambda_. The
  solve is the same either way.
- **The shipped classes still run** (necessary oracles): Howard on the library's Hamiltonians, a Lagrangian-defined
  problem with a shipped cost on the DPP path, and a user's ``LagrangianBase``.
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


def f(m):
    return 0.5 * m


def _a(v):
    return np.asarray(v, dtype=float)


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
        bounds=[(0.0, 1.0)] * dim,
        Nx_points=[n] * dim,
        boundary_conditions=no_flux_bc(dimension=dim),
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


def _own_lagrangian(cost):
    """A shipped cost whose instance holds its own lagrangian, twice the class's: Howard and the fan read it."""
    shipped = type(cost).lagrangian
    cost.lagrangian = lambda alpha: 2.0 * shipped(cost, alpha)
    return cost


@pytest.mark.parametrize(
    ("make", "reason"),
    [
        (lambda: _Plain(_quadratic(), coupling=f), r"its class is _Plain"),
        (
            lambda: hm.SeparableHamiltonian(_PlainCost(control_cost=1.0), coupling=f),
            r"its control cost is a _PlainCost",
        ),
    ],
    ids=[
        "a subclass that changes nothing",
        "a QuadraticControlCost subclass that changes nothing",
    ],
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
        _quiet(lambda: _problem(hm.SeparableHamiltonian(_l1(), coupling=f))).lagrangian_class,
        hm.SeparableLagrangian,
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
        (
            lambda: hm.SeparableHamiltonian(_PlainL1(control_cost=1.0), coupling=f),
            "its control cost is a _PlainL1",
        ),
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
        (
            lambda: hm.SeparableHamiltonian(_PlainCost(control_cost=1.0), coupling=f),
            "its control cost is a _PlainCost",
        ),
        (
            lambda: hm.SeparableHamiltonian(_TwiceTheLagrangian(control_cost=1.0), coupling=f),
            "its control cost is a _TwiceTheLagrangian",
        ),
        (lambda: _ExtraTerm(_quadratic(), coupling=f), "its class is _ExtraTerm"),
        (
            lambda: hm.SeparableHamiltonian(_narrowed(_quadratic()), coupling=f),
            "its control cost's instance holds ['effective_domain'] of its own",
        ),
        (
            lambda: hm.SeparableHamiltonian(_own_lagrangian(_quadratic()), coupling=f),
            "its control cost's instance holds ['lagrangian'] of its own",
        ),
        (_CarriesCost, "its class is _CarriesCost"),
    ],
    ids=[
        "a subclass that changes nothing",
        "a cost subclass that changes nothing",
        "a cost overriding only lagrangian",
        "the g(x) subclass",
        "a cost instance with its own effective_domain",
        "a cost instance with its own lagrangian",
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
        (
            lambda: hm.SeparableHamiltonian(_PlainCost(control_cost=1.0), coupling=f),
            "its control cost is a _PlainCost",
        ),
        (
            lambda: hm.SeparableHamiltonian(_narrowed(_quadratic()), coupling=f),
            "its control cost's instance holds ['effective_domain'] of its own",
        ),
        (
            lambda: hm.SeparableHamiltonian(_own_lagrangian(_quadratic()), coupling=f),
            "its control cost's instance holds ['lagrangian'] of its own",
        ),
    ],
    ids=[
        "a subclass that changes nothing",
        "a cost subclass that changes nothing",
        "a cost instance with its own effective_domain",
        "a cost instance with its own lagrangian",
    ],
)
def test_positive_qp_fan_reads_the_cost_only_as_shipped(make, reason):
    pytest.importorskip("cvxpy", reason="positive_qp_fan requires cvxpy (the `numerical` extra)")
    from mfgarchon.alg.numerical.hjb_solvers import HJBGFDMSolver

    xs = np.linspace(0.0, 1.0, 9)
    points = np.array([[a, b] for a in xs for b in xs])
    boundary = np.array([i for i, (a, b) in enumerate(points) if a in (0.0, 1.0) or b in (0.0, 1.0)])
    problem = _quiet(lambda: _problem(make(), dim=2))
    with pytest.raises(
        ValueError,
        match=rf"positive_qp_fan'\) needs a SeparableHamiltonian .* \({re.escape(reason)}",
    ):
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
    [
        lambda: hm.SeparableHamiltonian(_quadratic(), coupling=f),
        hm.QuadraticMFGHamiltonian,
    ],
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
                hjb_solver=HJBFDMSolver(problem),
                fp_solver=FPFVMSolver(problem),
                max_iterations=1,
                verbose=False,
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


@pytest.mark.parametrize(
    ("cost", "refused"),
    [
        (lambda: _PlainCost(), "its control cost is a _PlainCost"),
        (_l1, "its control cost is a L1ControlCost, not a QuadraticControlCost"),
        (lambda: hm.QuadraticControlCost(), None),
    ],
    ids=[
        "a QuadraticControlCost subclass that changes nothing",
        "a shipped L1 cost",
        "QuadraticControlCost, shipped",
    ],
)
def test_the_guarded_fan_reads_the_cost_only_as_shipped(cost, refused):
    """The guarded fan takes its policy and its Lagrangian from the cost it is handed, so it reads that cost only as the
    library ships it. The shipped cost passes the check and reaches the next one, which asks for a positive_qp_fan
    provider; none is given here."""
    from mfgarchon.alg.numerical.hjb_solvers.hjb_howard import HJBHowardSolver

    problem = _quiet(lambda: _problem(hm.SeparableHamiltonian(_quadratic(), coupling=f)))
    if refused:
        expected = (
            ValueError,
            r"needs control_cost=QuadraticControlCost\(\.\.\.\) as the library ships it: .* \(" + re.escape(refused),
        )
    else:
        expected = (
            RuntimeError,
            r"reads its Laplacian, its fans and its boundary type check from a stencil_provider",
        )
    with pytest.raises(expected[0], match=expected[1]):
        HJBHowardSolver(
            problem,
            stencil_provider=None,
            discretisation="guarded_fan",
            control_cost=cost(),
        )


def _dmp_solve(make_problem, check_dmp, caplog):
    """HJB-GFDM on joint_socp over the problem's own grid nodes, with the DMP warning on or off, and the warnings it
    logged. Not `_quiet`: that disables logging, and the warning is the record read here."""
    pytest.importorskip("cvxpy", reason="joint_socp requires cvxpy (the `numerical` extra)")
    from mfgarchon.alg.numerical.hjb_solvers import HJBGFDMSolver

    problem = _quiet(make_problem)
    x = np.asarray(problem.geometry.coordinates[0], dtype=float)
    solver = _quiet(
        lambda: HJBGFDMSolver(
            problem,
            x[:, None],
            delta=0.25,
            monotonicity_scheme="joint_socp",
            monotonicity_application="precompute",
            check_dmp=check_dmp,
        )
    )
    m = np.ones((problem.Nt + 1, x.size))
    caplog.clear()
    with (
        warnings.catch_warnings(),
        caplog.at_level(logging.WARNING, logger="mfgarchon.alg.numerical.hjb_solvers.hjb_gfdm"),
    ):
        warnings.simplefilter("ignore")
        u = solver.solve_hjb_system(m, (x - 0.5) ** 2, np.zeros_like(m))
    return np.asarray(u, dtype=float), list(caplog.messages)


def test_the_dmp_warning_fires_where_the_cost_stands_for_h(mfg_caplog):
    """A shipped quadratic Hamiltonian: the warning reads |grad u| / lambda, fires as it did, and the solve with it on is
    the solve with it off."""
    make = lambda: _problem(hm.SeparableHamiltonian(_quadratic(), coupling=f))  # noqa: E731
    u_off, _ = _dmp_solve(make, False, mfg_caplog)
    u_on, messages = _dmp_solve(make, True, mfg_caplog)
    np.testing.assert_array_equal(u_on, u_off)
    assert any("DMP not guaranteed" in r for r in messages), messages
    assert not any("DMP check not run" in r for r in messages), messages


@pytest.mark.parametrize(
    "make",
    [
        lambda: _components_problem(_lagrangian_components(_QuadraticLagrangian())),
        lambda: _problem(hm.SeparableHamiltonian(_PlainCost(control_cost=1.0), coupling=f)),
        lambda: _problem(hm.SeparableHamiltonian(_l1(), coupling=f)),
    ],
    ids=[
        "a user's LagrangianBase (DualHamiltonian)",
        "a QuadraticControlCost subclass that changes nothing",
        "a shipped L1 cost",
    ],
)
def test_the_dmp_warning_is_not_run_where_the_cost_does_not_stand_for_h(make, mfg_caplog):
    """No cost, or a cost whose lambda does not give the drift as the library ships it: |grad u| / lambda is not known
    to be the drift, so the warning is not run, says so once, and the solve is the one without it. It fired there
    before, on the cost's lambda or problem.lambda_."""
    u_off, _ = _dmp_solve(make, False, mfg_caplog)
    u_on, messages = _dmp_solve(make, True, mfg_caplog)
    np.testing.assert_array_equal(u_on, u_off)
    not_run = [r for r in messages if "DMP check not run" in r]
    assert len(not_run) == 1, messages
    assert "is not as the library ships it" in not_run[0], messages
    assert not any("DMP not guaranteed" in r for r in messages), messages
