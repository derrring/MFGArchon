#!/usr/bin/env python3
"""Issue #2207: a problem-level source must reach the solver through EVERY coupling loop.

WHAT THIS PINS, AND WHY IT IS NOT A HAPPY-PATH ASSERTION
--------------------------------------------------------
The mutation is "the problem field is absent": solve the same problem with and without it and
compare. A loop that composes and forwards moves; a loop that drops it returns a bit-identical
answer. Before #2207 that separated the loops here into two groups -- measured, ``FixedPointIterator``
moved ``max|U|`` by 8.75 having called the source 15 times, while ``FictitiousPlayIterator`` and
``BlockGaussSeidelIterator`` both returned ``0.000000e+00`` having called it **0 times**. Both were
solving a different equation than the one posed, with no error and no warning.

THREE FIELDS, NOT ONE -- AND WHY
--------------------------------
``MFGProblem`` carries three composable source fields and they travel different channels:
``source_term_hjb`` and ``nonlocal_operator`` through ``compose_hjb_source``, ``source_term_fp``
through ``compose_fp_source``. The first version of this file exercised only ``source_term_hjb``,
and an adversarial review measured what that cost: disabling the FP composition entirely
(``compose_fp_source(...)`` -> ``None`` inside the owner) left this file **4 passed**. Half of what
the change claims was unpinned by the test written to pin it. Each field now has a row, and each is
measured on the array it actually moves -- an HJB source on ``U``, an FP source on ``M``.

The call counter is the second half and is not redundant with the norm. A norm of zero has two
causes -- the source never arrived, or it arrived and cancelled -- and only the counter separates
them. It is also what would catch a regression that composes and forwards correctly while the
solver discards downstream: the norm would go to zero and look exactly like the defect this file
exists for, while the counter would stay non-zero and say where to look.

WHY d = 1
---------
The property is source-term plumbing -- named in AGENTS.md as fully expressed in one dimension.
Nothing here is directional: no normal/tangential split, no axis pairing, no corner. A 2-D fixture
would buy runtime and no discrimination.

THE LOOPS THAT HOLD ONE SOLVER PER POPULATION, REGIME OR NOISE PATH (#2666)
---------------------------------------------------------------------------
``MultiPopulationIterator``, ``RegimeSwitchingIterator`` and ``CommonNoiseMFGSolver`` dropped the
problem's source on both halves -- measured, ``Δ = 0`` with the source called 0 times -- because none
of them reached the owner this file pins. Their rows are below the #2207 rows and assert more than
``|Δ| > 0``: each compares against the source's exact effect at convergence. A constant HJB source
``c`` shifts ``U`` by ``c (T - t)`` and leaves ``M`` alone (docs/user/CONVENTIONS.md § 2: the source keeps
its sign); a constant FP source under no-flux grows the trapezoid mass by ``c t`` -- except across
regimes, where the mass moves between them and the oracle is the scheme's own recurrence (see that
test). Regimes add the problem's source to their cross-regime term in the one ``source_term``
channel, which is the combination #2207 left open, decided by #2666.

NOT COVERED, AND NAMED RATHER THAN IMPLIED
------------------------------------------
``GraphMFGSolver`` is not a row. It composes in its own ``_compose_hjb_source`` /
``_compose_fp_source`` from ``source_composition``'s primitives and honoured both halves when #2666
measured it, so no change of #2666 reaches it; #2173 proposes folding ``MultiPopulationProblem`` into
``MFGProblem``, a related but distinct question.
"""

from __future__ import annotations

import inspect

import pytest

import numpy as np

from mfgarchon.alg.numerical.coupling.base_mfg import BaseCouplingIterator
from mfgarchon.alg.numerical.coupling.block_iterators import BlockGaussSeidelIterator, BlockJacobiIterator
from mfgarchon.alg.numerical.coupling.fictitious_play import FictitiousPlayIterator
from mfgarchon.alg.numerical.coupling.fixed_point_iterator import FixedPointIterator
from mfgarchon.alg.numerical.coupling.multi_population_iterator import MultiPopulationIterator
from mfgarchon.alg.numerical.coupling.regime_switching_iterator import RegimeSwitchingIterator
from mfgarchon.alg.numerical.fp_solvers.fp_fdm import FPFDMSolver
from mfgarchon.alg.numerical.hjb_solvers.hjb_fdm import HJBFDMSolver
from mfgarchon.alg.numerical.stochastic import CommonNoiseMFGSolver
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.core.mfg_components import MFGComponents
from mfgarchon.core.mfg_problem import MFGProblem
from mfgarchon.core.model import Conditions, Model
from mfgarchon.core.multi_population import MultiPopulationProblem
from mfgarchon.core.regime_switching import RegimeSwitchingConfig
from mfgarchon.core.stochastic import OrnsteinUhlenbeckProcess, StochasticMFGProblem
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import no_flux_bc

_N = 21
_ITERATIONS = 3

#: Large against every field this fixture produces, so a solver that receives it cannot return an
#: answer within rounding of the sourceless one. A small source would make a real drop and a real
#: forward differ by an amount no threshold could separate.
_MAGNITUDE = 50.0

#: Looser than the inner Newton's own default. A tighter outer tolerance makes the solver warn that
#: the outer loop asks for more than the inner one delivers -- true, and not this file's subject.
_TOLERANCE = 1e-6

_LOOPS = [FixedPointIterator, FictitiousPlayIterator, BlockGaussSeidelIterator, BlockJacobiIterator]

#: The loop whose row must move for any other row to mean anything. It has composed and forwarded a
#: source since #1424, so a zero here indicts the fixture, not the library.
_POSITIVE_CONTROL = FixedPointIterator


class _CountingSource:
    """The problem-level ``(x, m, v, t)`` callback, constant ``value``, counting its own invocations."""

    def __init__(self, value: float = _MAGNITUDE) -> None:
        self.value = value
        self.calls = 0

    def __call__(self, t, x, v, m):
        self.calls += 1
        a = np.asarray(x, dtype=float)
        return np.full(a.shape[0] if a.ndim == 2 else a.size, self.value)


def _nonlocal_operator():
    """A dense ASYMMETRIC (N, N) kernel, applied as ``K @ v_t`` (#1259).

    Asymmetric because a symmetric kernel could not express a transpose defect at all -- a necessary
    condition, and AGENTS.md's reason for breaking the symmetry of the 2-D MMS pair (#2016). It is
    NOT sufficient and this file does not exploit it: every assertion here is ``|delta| > 0`` against
    a no-kernel baseline, which a transposed kernel satisfies too. Measured -- mutating
    ``nonlocal_operator @ v_t`` to ``.T @ v_t`` leaves this file at 17 passed. The asymmetry is kept
    so a future provenance pin (#2211) has a fixture that can separate the two; claiming it buys
    discrimination here would be the "necessary read as sufficient" error.
    """
    i = np.arange(_N)[:, None]
    j = np.arange(_N)[None, :]
    return (_MAGNITUDE / _N) * np.exp(-0.5 * np.abs(i - 2 * j) / _N)


def _u_terminal(x):
    """NOT identically zero, and that is load-bearing rather than decoration.

    The nonlocal term is ``K @ v_t`` (#1259). With a zero terminal cost and zero coupling this
    problem has ``U == 0`` everywhere, so ``K @ v_t == 0`` for EVERY kernel and the nonlocal row
    cannot fail -- measured: ``max|U| = 0.0`` on the first draft of this fixture, which failed the
    nonlocal row on all four loops INCLUDING the positive control. That is the file's own control
    doing its job: when the control row fails, the fixture is wrong and not the library.
    """
    a = np.asarray(x, dtype=float)
    return (a - 0.5) ** 2


def _m_initial(x):
    """Mass 1 on this node-centred grid to within a ulp, with no normalising constant to keep in
    step with #2145: under TRAPEZOID weights the cosine contributes 0 over a whole period, so the
    constant carries the whole mass. Under the plain node sum it does not -- `np.sum(m) * dx` is
    1.075 here, a 7.5% error -- and which measure is meant is the whole subject of #2145, so the
    weighting is stated rather than left to "sums to 0". ``np.trapezoid`` returns exactly ``1.0``; the library's own ``_measure_initial_density()``
    returns ``0.9999999999999999``, which is what matters because that is the value its warning
    tests. An earlier draft of this line said "exactly 1", which is true of one of those two
    measures and was written without running the other."""
    return 1.0 + 0.5 * np.cos(2.0 * np.pi * np.asarray(x, dtype=float))


def _problem(**source_fields):
    """Built through the model/domain/conditions form on purpose: the legacy keyword form warns, and #2203 is already
    open against the two fixtures that still use it. A third would be a third."""
    grid = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[_N], boundary_conditions=no_flux_bc(dimension=1))
    return MFGProblem(
        model=Model(
            hamiltonian=SeparableHamiltonian(
                control_cost=QuadraticControlCost(control_cost=1.0),
                coupling=lambda m: np.asarray(m) * 0.0,
                coupling_dm=lambda m: np.asarray(m) * 0.0,
            ),
            volatility=0.3,
        ),
        domain=grid,
        conditions=Conditions(m_initial=_m_initial, u_terminal=_u_terminal, T=0.2),
        Nt=5,
        coupling_coefficient=0.0,
        **source_fields,
    )


def _solve(loop_cls, **source_fields):
    """``(U, M)`` from one coupling loop."""
    problem = _problem(**source_fields)
    loop = loop_cls(problem, HJBFDMSolver(problem), FPFDMSolver(problem))
    result = loop.solve(max_iterations=_ITERATIONS, tolerance=_TOLERANCE)
    if isinstance(result, tuple):
        return np.asarray(result[0], dtype=float), np.asarray(result[1], dtype=float)
    return np.asarray(loop.U, dtype=float), np.asarray(loop.M, dtype=float)


#: (field name, factory, which array the field moves). The pairing matters: an FP source leaves U
#: untouched at this coupling strength, so measuring it on U would report a correct forward as a
#: drop -- the same fixture mistake in the opposite direction.
_FIELDS = [
    ("source_term_hjb", _CountingSource, "U"),
    ("source_term_fp", _CountingSource, "M"),
    ("nonlocal_operator", _nonlocal_operator, "U"),
]


def _measure(loop_cls, field, factory, which):
    """``(max|Δ| on the array this field moves, invocation count or None)``."""
    u0, m0 = _solve(loop_cls)
    obj = factory()
    u1, m1 = _solve(loop_cls, **{field: obj})
    a, b = (u0, u1) if which == "U" else (m0, m1)
    calls = getattr(obj, "calls", None)  # a nonlocal_operator is a matrix and counts nothing
    return float(np.nanmax(np.abs(b - a))), calls


@pytest.mark.parametrize(("field", "factory", "which"), _FIELDS, ids=[f[0] for f in _FIELDS])
@pytest.mark.parametrize("loop_cls", _LOOPS, ids=lambda c: c.__name__)
def test_the_problems_source_reaches_the_solver(loop_cls, field, factory, which):
    delta, calls = _measure(loop_cls, field, factory, which)

    if calls is not None:
        assert calls > 0, (
            f"{loop_cls.__name__} never invoked the problem's {field}. The coupling loop did not "
            f"compose it (Issue #2207): pass the iterates to _build_hjb_kwargs / _build_fp_kwargs "
            f"rather than calling the solver directly."
        )
    assert delta > 0.0, (
        f"{loop_cls.__name__}: {field} left {which} bit-identical (max|d{which}| = {delta:.6e}"
        + (f", callback invoked {calls} times" if calls is not None else "")
        + "). The problem posed and the problem solved are different."
    )


def test_the_measurement_can_distinguish_a_drop_from_a_forward():
    """Control on the instrument, in the same run as the rows above.

    If the fixture were mis-built -- a source too small to move the answer, ``_ITERATIONS`` too low
    for it to propagate, the wrong array compared -- every row above would report a drop and the
    file would fail for a reason having nothing to do with the coupling loops.
    """
    for field, factory, which in _FIELDS:
        delta, calls = _measure(_POSITIVE_CONTROL, field, factory, which)
        broken = (
            f"The instrument is broken, not the library: {_POSITIVE_CONTROL.__name__} has forwarded "
            f"a composed source since #1424, and this fixture measured {field} -> "
            f"max|d{which}| = {delta:.6e}. Every other row in this file is uninterpretable."
        )
        assert delta > 0.0, broken
        if calls is not None:
            assert calls > 0, broken


@pytest.mark.parametrize("builder", ["_build_hjb_kwargs", "_build_fp_kwargs"])
@pytest.mark.parametrize("name", ["M", "U"])
def test_both_iterates_are_required_not_optional(builder, name):
    """Issue #2207: the omission this change removes must not be re-expressible.

    A default of ``None`` on EITHER iterate restores the hole -- a loop that forgets it composes
    nothing, raises nothing, and solves without the source. Asserted on the signature rather than
    through a ``pytest.raises(TypeError, match=...)``, because an adversarial review measured that
    ``match="M"`` is satisfied by the message for a missing ``M`` alone: giving ``U`` a default left
    the whole file green, so half the mechanism was unpinned by the test that named it.
    """
    param = inspect.signature(getattr(BaseCouplingIterator, builder)).parameters[name]
    assert param.default is inspect.Parameter.empty, (
        f"BaseCouplingIterator.{builder} gave '{name}' a default. The composition is then skippable "
        f"by omission, which is exactly the #2207 hole one level up."
    )
    assert param.kind is inspect.Parameter.KEYWORD_ONLY


@pytest.mark.parametrize("loop_cls", [BlockGaussSeidelIterator, BlockJacobiIterator], ids=lambda c: c.__name__)
def test_the_strict_adjoint_path_refuses_an_fp_source(loop_cls):
    """#2207: the one path that has no route for a source must refuse it, not drop it.

    `BlockIterator._solve_fp` returns to `_solve_fp_strict_adjoint` BEFORE reaching the builders,
    so the owner is unreachable from `adjoint_mode != "off"` and the source has nowhere to go --
    that path assembles and steps its own FP operator instead of calling the FP solver. Dropping it
    silently is #1424; dropping it HERE while the HJB source still arrives through the builder is
    worse than either, because the two halves of one problem would then be solved from different
    equations.

    Three arms, and the last two are the controls without which the first proves nothing about
    over-firing: an FP source must raise; no source at all must still run; and an HJB source must
    also still run, because the guard's predicate is `source_term_fp` and nothing else.

    That last reason is stated as the traced fact and not as a story about routing. An earlier draft
    of this docstring said the HJB source "enters through `build_linearized_operator`, not the FP
    transport operator" -- invented, and false in both halves: that method's signature is
    `(self, U, M, time)` and the string `source` does not occur in its body or in either of its
    dimensional helpers. The HJB source actually travels `_build_hjb_kwargs` -> `compose_hjb_source`
    -> `solve_hjb_system(source_term=...)` from `_solve_hjb`, the other half of the block step, which
    never reaches the FP operator this path assembles. What makes THIS arm pass is narrower and is
    what the mutation proves: point the guard at `source_term_hjb` and both parametrisations raise.

    This test exists because three review rounds passed over the branch's only user-visible
    behaviour change with no oracle named -- AGENTS.md makes naming one mandatory, and "verified by
    a probe I ran" corresponds to no committed artifact.
    """
    p_src = _problem(source_term_fp=_CountingSource())
    solver = loop_cls(p_src, HJBFDMSolver(p_src), FPFDMSolver(p_src), adjoint_mode="jacobian_transpose")
    with pytest.raises(NotImplementedError, match="cannot carry an FP source term"):
        solver.solve(max_iterations=2, tolerance=_TOLERANCE)

    for label, fields in (("no source", {}), ("hjb source only", {"source_term_hjb": _CountingSource()})):
        q = _problem(**fields)
        loop = loop_cls(q, HJBFDMSolver(q), FPFDMSolver(q), adjoint_mode="jacobian_transpose")
        # try/except, not `assert result is not None`: the solve returns a SolverResult, so that
        # assertion can never fire and the authored diagnostic below never printed. The arm
        # discriminates by "did not raise", so the raise is what has to be caught.
        try:
            loop.solve(max_iterations=2, tolerance=_TOLERANCE)
        except NotImplementedError as exc:
            pytest.fail(
                f"the guard over-fired on '{label}': this configuration carries no FP source and "
                f"must still run, but the strict-adjoint path refused it -- {exc}"
            )


# --- #2666: the loops that hold one solver per population, regime or noise path ----------------------

_T, _NT = 0.2, 5  # _problem's horizon
_TIMES = np.arange(_NT + 1) * (_T / _NT)
_TRAPEZOID = np.r_[0.5, np.ones(_N - 2), 0.5] / (_N - 1)

#: Solved to convergence, so the exact effect of the source is what the iterate carries. The largest residual
#: #2666 measured on these rows is 5.494e-12 (the common-noise mean's dM under an HJB source); a dropped source
#: leaves c T = 1 unexplained.
_CONVERGED = {"max_iterations": 200, "tolerance": 1e-12}
_EXACT = 1e-9


def _assert_the_sources_exact_effect(field, c, u0, u1, m0, m1, where):
    """A constant source's whole effect: on the array it moves, exactly; on the other, none."""
    du, dm = np.asarray(u1) - np.asarray(u0), np.asarray(m1) - np.asarray(m0)
    if field == "source_term_hjb":
        moved, still = np.max(np.abs(du - c * (_T - _TIMES)[:, None])), np.max(np.abs(dm))
        expect = f"dU = c (T - t) with c = {c}"
    else:
        moved, still = np.max(np.abs(dm @ _TRAPEZOID - c * _TIMES)), np.max(np.abs(du))
        expect = f"the mass grows by c t with c = {c}"
    assert moved <= _EXACT, f"{where}: {field} should give {expect}; off by {moved:.3e} (a dropped source: {c * _T})"
    assert still <= _EXACT, f"{where}: {field} moved the other field by {still:.3e}"


@pytest.mark.parametrize("field", ["source_term_hjb", "source_term_fp"])
def test_each_population_solves_with_its_own_source(field):
    """MultiPopulationIterator, K = 2, with a different c per population, so a source applied to the wrong one shows."""
    values = (5.0, 3.0)

    def solve(sources):
        problems = [_problem(**({field: s} if s is not None else {})) for s in sources]
        result = MultiPopulationIterator(
            MultiPopulationProblem(populations=problems, population_names=["a", "b"]),
            [HJBFDMSolver(p) for p in problems],
            [FPFDMSolver(p) for p in problems],
            relaxation=0.5,
        ).solve(**_CONVERGED)
        assert result.converged
        return result.U, result.M

    u0, m0 = solve([None, None])
    sources = [_CountingSource(v) for v in values]
    u1, m1 = solve(sources)
    for k, source in enumerate(sources):
        assert source.calls > 0, f"population {k}'s {field} was never called"
        _assert_the_sources_exact_effect(field, source.value, u0[k], u1[k], m0[k], m1[k], f"population {k}")


_Q = np.array([[-0.1, 0.1], [0.2, -0.2]])


def _regime_solve(*per_regime_fields):
    """Two regimes, each problem built with its own fields; none by default."""
    problems = [_problem(**fields) for fields in (per_regime_fields or ({}, {}))]
    result = RegimeSwitchingIterator(
        problems,
        RegimeSwitchingConfig(transition_matrix=_Q),
        [HJBFDMSolver(p) for p in problems],
        [FPFDMSolver(p) for p in problems],
        max_iterations=_CONVERGED["max_iterations"],
        tolerance=_CONVERGED["tolerance"],
        damping=0.5,
    ).solve()
    assert result.converged
    return result.values, result.densities


def test_each_regime_adds_its_own_hjb_source_to_the_cross_regime_term():
    """The same c in both regimes, so the cross-regime term Q (u_k - u_j) is unchanged by the shift and the exact
    effect is dU = c (T - t) in each. One source object per regime, each counted: a loop that composed one regime's
    problem for every regime would call one of them twice as often and the other never."""
    u0, m0 = _regime_solve()
    sources = [_CountingSource(5.0), _CountingSource(5.0)]
    u1, m1 = _regime_solve(*({"source_term_hjb": s} for s in sources))
    for k, source in enumerate(sources):
        assert source.calls > 0, f"regime {k}'s source_term_hjb was never called"
        _assert_the_sources_exact_effect("source_term_hjb", source.value, u0[k], u1[k], m0[k], m1[k], f"regime {k}")


def test_each_regime_adds_its_own_fp_source_under_the_integrating_factor():
    """The oracle is this scheme's mass recurrence, per regime, which holds to rounding.

    Derived from the scheme, not the code. The diagonal outflow is carried by ``n^k = e^{q_k t} m^k`` (#1681), so the
    FP solve sees ``d_t n^k - L^k[n^k] = e^{q_k t} (sum_{j != k} Q_jk m^j + S^k)``. FPFDMSolver takes a source at the
    NEW time, ``n_{n+1} = n_n + dt (L n_{n+1} + source(t_{n+1}))``, and ``L`` conserves the trapezoid mass under
    no-flux. Recovering ``m = e^{-q_k t} n`` and integrating, with ``S^k = c_k``:

        M^k_{n+1} = e^{-q_k dt} M^k_n + dt (sum_{j != k} Q_jk M^j_{n+1} + c_k).

    Each regime has its own source and its own c (5 and 3), so a source applied to the wrong regime shows.

    It holds under that hypothesis only: FPFDMSolver's implicit source time. Measured on this fixture for #2666, regimes
    0 and 1: residual 8.882e-15 and 6.439e-15 with the sources, 2.220e-16 without, and 1.777e-03 and 1.836e-03 for the
    same recurrence with the source at t_n. A source left outside the factor misses it by 3.960e-03 and 4.705e-03, a
    dropped one by 2.000e-01 and 1.200e-01, and regime 0's source composed for both regimes by 8.000e-02 in regime 1.

    Why not "the total mass grows by sum_k c_k t": that holds for the PDE and not here. Summing the recurrence over k
    leaves ``dt sum_j q_j (M^j_{n+1} - M^j_n)``, the scheme's non-conservation #1798 records, and on this fixture the
    total is off by 8.900e-03 -- the size of the error a source not scaled by ``e^{q_k t}`` makes, so it cannot see
    that one.
    """
    u0, _ = _regime_solve()
    sources = [_CountingSource(5.0), _CountingSource(3.0)]
    u1, m1 = _regime_solve(*({"source_term_fp": s} for s in sources))
    mass = np.array([np.asarray(m) @ _TRAPEZOID for m in m1])  # (K, Nt + 1)
    dt = _T / _NT
    for k, source in enumerate(sources):
        assert source.calls > 0, f"regime {k}'s source_term_fp was never called"
        q_k = -_Q[k, k]
        inflow = sum(_Q[j, k] * mass[j, 1:] for j in range(2) if j != k)
        predicted = np.exp(-q_k * dt) * mass[k, :-1] + dt * (inflow + source.value)
        residual = np.max(np.abs(mass[k, 1:] - predicted))
        assert residual <= 1e-12, (
            f"regime {k}: the mass misses the recurrence M_(n+1) = e^(-q dt) M_n + dt (inflow_(n+1) + c) by "
            f"{residual:.3e} (a dropped source misses it by dt c = {dt * source.value}). If FPFDMSolver's source time "
            "changed -- a Crank-Nicolson source, say -- this fires on a correct solver: re-derive the recurrence for "
            "the new time, do not loosen the bound."
        )
        assert np.max(np.abs(np.asarray(u1[k]) - np.asarray(u0[k]))) <= _EXACT, f"regime {k}: an FP source moved U"


class _ConvergedConditionalSolve:
    """A conditional-solver factory that solves to ``_CONVERGED``, so the mean carries the exact effect. The inner
    Newton is set at least as tight as the outer loop, which otherwise warns that it cannot get below the inner floor."""

    def __init__(self, problem):
        self.loop = FixedPointIterator(
            problem, HJBFDMSolver(problem, newton_tolerance=_CONVERGED["tolerance"]), FPFDMSolver(problem)
        )

    def solve(self):
        return self.loop.solve(**_CONVERGED)


@pytest.mark.parametrize("field", ["source_term_hjb", "source_term_fp"])
def test_every_conditional_problem_carries_the_source(field):
    """CommonNoiseMFGSolver: each noise path's conditional problem is the parent rebuilt with the noise frozen, so it
    carries the parent's source. Every conditional solve shifts by the same exact effect, and so does the mean."""

    def solve(**fields):
        components = MFGComponents(
            hamiltonian=SeparableHamiltonian(
                control_cost=QuadraticControlCost(control_cost=1.0),
                coupling=lambda m: np.asarray(m) * 0.0,
                coupling_dm=lambda m: np.asarray(m) * 0.0,
            ),
            u_terminal=_u_terminal,
            m_initial=_m_initial,
        )
        problem = StochasticMFGProblem(
            geometry=TensorProductGrid(
                bounds=[(0.0, 1.0)], Nx_points=[_N], boundary_conditions=no_flux_bc(dimension=1)
            ),
            T=_T,
            Nt=_NT,
            volatility=0.3,
            noise_process=OrnsteinUhlenbeckProcess(kappa=1.0, mu=0.0, sigma=0.1),
            conditional_hamiltonian=lambda x, p, m, theta: 0.5 * np.sum(p**2, axis=-1),
            components=components,
            **fields,
        )
        result = CommonNoiseMFGSolver(
            problem,
            num_noise_samples=2,
            conditional_solver_factory=_ConvergedConditionalSolve,
            variance_reduction=False,
            parallel=False,
            seed=7,
        ).solve(verbose=False)
        return [result.u_mean, *result.u_samples], [result.m_mean, *result.m_samples]

    u0, m0 = solve()
    source = _CountingSource(5.0)
    u1, m1 = solve(**{field: source})
    assert source.calls > 0
    # The conditional Hamiltonian ignores theta, so each noise path's solve carries the exact effect, not only
    # their mean: a path that dropped the source would be hidden in a mean only by a path that doubled it.
    for i, where in enumerate(["the conditional mean", *(f"noise path {k}" for k in range(len(u0) - 1))]):
        _assert_the_sources_exact_effect(field, source.value, u0[i], u1[i], m0[i], m1[i], where)


@pytest.mark.parametrize("field", ["source_term_hjb", "nonlocal_operator"])
def test_the_standalone_hjb_solve_refuses_a_source_it_cannot_apply(field):
    """``BaseHJBSolver.solve()`` passed no source, so a declared one was dropped (#2666). It now refuses, and the route
    its message names was run once for #2666, on a problem like ``_problem`` with ``m_initial = 1``:
    ``solve_hjb_system(..., source_term=compose_hjb_source(...))`` moved U by exactly c (T - t), to 1.110e-16. The
    control is the same problem without the field, which still solves."""
    fields = {field: _CountingSource(5.0) if field == "source_term_hjb" else _nonlocal_operator()}
    with pytest.raises(NotImplementedError, match=rf"solve\(\) has no route.*{field}.*#2666"):
        HJBFDMSolver(_problem(**fields)).solve()
    assert np.all(np.isfinite(HJBFDMSolver(_problem()).solve()))


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
