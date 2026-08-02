"""
Unit tests for RegimeSwitchingIterator.

Tests the Markov-switching MFG iterator that solves K coupled HJB-FP
systems with inter-regime transition terms (#973).
"""

import pytest

import numpy as np

from mfgarchon.alg.numerical.coupling.regime_switching_iterator import (
    RegimeSwitchingIterator,
    RegimeSwitchingResult,
)
from mfgarchon.alg.numerical.fp_solvers import FPFDMSolver
from mfgarchon.alg.numerical.hjb_solvers import HJBFDMSolver
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.core.mfg_components import MFGComponents
from mfgarchon.core.mfg_problem import MFGProblem
from mfgarchon.core.regime_switching import RegimeSwitchingConfig
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import no_flux_bc


def _make_problem(coupling_strength: float = 1.0, sigma: float = 0.3, T: float = 0.5, Nt: int = 10) -> MFGProblem:
    """Create a simple 1D MFG problem for one regime.

    ``T``/``Nt`` are constructor parameters because ``dt`` is a plain attribute computed
    once in ``__init__``: assigning ``problem.T`` afterwards leaves ``dt`` at its old value
    and the problem then reports a horizon its own time grid does not span. Measured:
    ``T=0.5, Nt=10`` gives ``dt=0.05``, and ``p.T = 1.0`` leaves ``dt=0.05``.
    """
    H = SeparableHamiltonian(
        control_cost=QuadraticControlCost(control_cost=1.0),
        coupling=lambda m: coupling_strength * m,
        coupling_dm=lambda m: coupling_strength,
    )
    components = MFGComponents(
        hamiltonian=H,
        u_terminal=lambda x: 0.0,
        m_initial=lambda x: 1.0,
    )
    return MFGProblem(
        geometry=TensorProductGrid(
            bounds=[(0.0, 1.0)], Nx_points=[31 + 1], boundary_conditions=no_flux_bc(dimension=1)
        ),
        T=T,
        Nt=Nt,
        sigma=sigma,
        components=components,
    )


def _make_2regime_system():
    """Create a 2-regime system with transition matrix."""
    p1 = _make_problem(coupling_strength=1.0)
    p2 = _make_problem(coupling_strength=0.5)
    Q = np.array([[-0.1, 0.1], [0.2, -0.2]])
    config = RegimeSwitchingConfig(transition_matrix=Q)
    hjb1, hjb2 = HJBFDMSolver(p1), HJBFDMSolver(p2)
    fp1, fp2 = FPFDMSolver(p1), FPFDMSolver(p2)
    return [p1, p2], config, [hjb1, hjb2], [fp1, fp2]


class TestRegimeSwitchingInstantiation:
    """Test RegimeSwitchingIterator construction."""

    def test_basic_2regime(self):
        problems, config, hjbs, fps = _make_2regime_system()
        iterator = RegimeSwitchingIterator(
            problems=problems,
            regime_config=config,
            hjb_solvers=hjbs,
            fp_solvers=fps,
        )
        assert iterator._max_iter == 50
        assert iterator._damping == 0.5

    def test_custom_parameters(self):
        problems, config, hjbs, fps = _make_2regime_system()
        iterator = RegimeSwitchingIterator(
            problems=problems,
            regime_config=config,
            hjb_solvers=hjbs,
            fp_solvers=fps,
            max_iterations=20,
            tolerance=1e-3,
            damping=0.3,
        )
        assert iterator._max_iter == 20
        assert iterator._tol == 1e-3
        assert iterator._damping == 0.3

    def test_mismatched_counts_raises(self):
        problems, config, hjbs, fps = _make_2regime_system()
        with pytest.raises(ValueError, match="Need 2 problems"):
            RegimeSwitchingIterator(
                problems=[problems[0]],
                regime_config=config,
                hjb_solvers=hjbs,
                fp_solvers=fps,
            )

    def test_mismatched_hjb_solvers_raises(self):
        problems, config, hjbs, fps = _make_2regime_system()
        with pytest.raises(ValueError, match="Need 2 HJB"):
            RegimeSwitchingIterator(
                problems=problems,
                regime_config=config,
                hjb_solvers=[hjbs[0]],
                fp_solvers=fps,
            )

    def test_mismatched_fp_solvers_raises(self):
        problems, config, hjbs, fps = _make_2regime_system()
        with pytest.raises(ValueError, match="Need 2 FP"):
            RegimeSwitchingIterator(
                problems=problems,
                regime_config=config,
                hjb_solvers=hjbs,
                fp_solvers=[fps[0]],
            )


class TestRegimeSwitchingSolve:
    """Test solve() method produces valid results."""

    def test_returns_result_type(self):
        problems, config, hjbs, fps = _make_2regime_system()
        iterator = RegimeSwitchingIterator(
            problems=problems,
            regime_config=config,
            hjb_solvers=hjbs,
            fp_solvers=fps,
            max_iterations=3,
        )
        result = iterator.solve()
        assert isinstance(result, RegimeSwitchingResult)

    def test_result_shapes(self):
        problems, config, hjbs, fps = _make_2regime_system()
        iterator = RegimeSwitchingIterator(
            problems=problems,
            regime_config=config,
            hjb_solvers=hjbs,
            fp_solvers=fps,
            max_iterations=3,
        )
        result = iterator.solve()
        assert len(result.values) == 2
        assert len(result.densities) == 2
        Nt = problems[0].Nt
        Nx = problems[0].geometry.get_grid_shape()[0]
        assert result.values[0].shape == (Nt + 1, Nx)
        assert result.values[1].shape == (Nt + 1, Nx)

    def test_solutions_are_finite(self):
        problems, config, hjbs, fps = _make_2regime_system()
        iterator = RegimeSwitchingIterator(
            problems=problems,
            regime_config=config,
            hjb_solvers=hjbs,
            fp_solvers=fps,
            max_iterations=5,
        )
        result = iterator.solve()
        for k in range(2):
            assert np.all(np.isfinite(result.values[k])), f"Non-finite values in regime {k}"
            assert np.all(np.isfinite(result.densities[k])), f"Non-finite densities in regime {k}"

    def test_error_history_recorded(self):
        problems, config, hjbs, fps = _make_2regime_system()
        iterator = RegimeSwitchingIterator(
            problems=problems,
            regime_config=config,
            hjb_solvers=hjbs,
            fp_solvers=fps,
            max_iterations=5,
        )
        result = iterator.solve()
        assert len(result.error_history) > 0
        assert len(result.error_history) <= 5

    def test_iterations_reported(self):
        problems, config, hjbs, fps = _make_2regime_system()
        iterator = RegimeSwitchingIterator(
            problems=problems,
            regime_config=config,
            hjb_solvers=hjbs,
            fp_solvers=fps,
            max_iterations=3,
        )
        result = iterator.solve()
        assert result.iterations > 0
        assert result.iterations <= 3


class TestRegimeSwitchingUpdateSchemes:
    """Test Jacobi vs Gauss-Seidel update schemes."""

    def test_gauss_seidel_default(self):
        problems, config, hjbs, fps = _make_2regime_system()
        iterator = RegimeSwitchingIterator(
            problems=problems,
            regime_config=config,
            hjb_solvers=hjbs,
            fp_solvers=fps,
        )
        assert iterator._update_scheme == "gauss_seidel"

    def test_jacobi_scheme(self):
        problems, config, hjbs, fps = _make_2regime_system()
        iterator = RegimeSwitchingIterator(
            problems=problems,
            regime_config=config,
            hjb_solvers=hjbs,
            fp_solvers=fps,
            update_scheme="jacobi",
            max_iterations=3,
        )
        result = iterator.solve()
        assert isinstance(result, RegimeSwitchingResult)
        assert np.all(np.isfinite(result.values[0]))


class TestRegimeSwitchingGetResults:
    """Test get_results() interface."""

    def test_get_results_after_solve(self):
        problems, config, hjbs, fps = _make_2regime_system()
        iterator = RegimeSwitchingIterator(
            problems=problems,
            regime_config=config,
            hjb_solvers=hjbs,
            fp_solvers=fps,
            max_iterations=3,
        )
        iterator.solve()
        U, _M = iterator.get_results()
        assert U.shape[0] == problems[0].Nt + 1

    def test_get_results_before_solve_raises(self):
        problems, config, hjbs, fps = _make_2regime_system()
        iterator = RegimeSwitchingIterator(
            problems=problems,
            regime_config=config,
            hjb_solvers=hjbs,
            fp_solvers=fps,
        )
        with pytest.raises(RuntimeError, match="No solution computed"):
            iterator.get_results()


class TestRegimeSwitchingCrossTermSign:
    """Inter-regime coupling sign (#1251, 2026-06-10 audit).

    The DPP chain term ``sum_j Q[k,j](v^k - v^j)`` sits on the HJB LHS, and the HJB
    solver subtracts the source (``Phi_U -= source_term``), so the source must equal
    ``-cross``. A ``+cross`` source flips the inter-regime value coupling.

    The FP side carries the same signs -- inflow adds, outflow removes -- but since
    Issue #1681 they arrive on two different channels: the inflow through ``source_term``,
    the outflow through the integrating factor, because a lagged sink defeats the scheme's
    positivity. What has to stay pinned is the sign of each, not which channel it used.
    """

    def _iterator(self):
        problems, config, hjbs, fps = _make_2regime_system()
        return (
            RegimeSwitchingIterator(problems=problems, regime_config=config, hjb_solvers=hjbs, fp_solvers=fps),
            problems,
            config.transition_matrix,
        )

    def test_hjb_source_is_negative_cross_term(self):
        iterator, problems, Q = self._iterator()
        Nt, N = problems[0].Nt, 5
        x = np.linspace(0.0, 1.0, N)
        v0, v1 = 2.0, 1.0
        Us_full = [v0 * np.ones((Nt + 1, N)), v1 * np.ones((Nt + 1, N))]

        src0 = iterator._make_hjb_source(0, 2, Q, Us_full, [None, None])
        np.testing.assert_allclose(src0(0.0, x), -Q[0, 1] * (v0 - v1) * np.ones(N))

        src1 = iterator._make_hjb_source(1, 2, Q, Us_full, [None, None])
        np.testing.assert_allclose(src1(0.0, x), -Q[1, 0] * (v1 - v0) * np.ones(N))

    def test_fp_inflow_enters_the_source_with_a_positive_sign(self):
        iterator, problems, Q = self._iterator()
        Nt, N = problems[0].Nt, 5
        x = np.linspace(0.0, 1.0, N)
        m0, m1 = 0.7, 0.3
        Ms = [m0 * np.ones((Nt + 1, N)), m1 * np.ones((Nt + 1, N))]

        src0 = iterator._make_fp_source(0, 2, Q, Ms)
        # At t=0 the integrating factor is 1, so the source is the bare inflow.
        np.testing.assert_allclose(src0(0.0, x), Q[1, 0] * m1 * np.ones(N))
        assert (src0(0.0, x) > 0).all(), "inflow must be non-negative; the scheme's positivity rests on it"

    def test_inflow_is_pre_scaled_by_the_integrating_factor_at_every_t(self):
        """Both halves of the substitution, pinned as an identity rather than an error bound.

        ``m^k = exp(-q_k t) n^k`` only reproduces the intended equation if the source carries
        ``exp(+q_k t)``. Dropping that scaling while keeping the recovery factor damps the
        INFLOW instead of the outflow -- silently wrong physics, and the whole unit suite
        passed under it. The closed-form test could not see it because at t=0 the factor is 1
        and this class had checked nowhere else. So assert the factor directly, at t>0.
        """
        iterator, problems, Q = self._iterator()
        Nt, N = problems[0].Nt, 5
        x = np.linspace(0.0, 1.0, N)
        m0, m1 = 0.7, 0.3
        Ms = [m0 * np.ones((Nt + 1, N)), m1 * np.ones((Nt + 1, N))]

        q_0 = iterator._outflow_rate(0, 2, Q)
        src0 = iterator._make_fp_source(0, 2, Q, Ms)
        for t in (problems[0].dt, 3 * problems[0].dt, problems[0].T):
            np.testing.assert_allclose(
                src0(t, x),
                np.exp(q_0 * t) * Q[1, 0] * m1 * np.ones(N),
                err_msg=f"source at t={t} is not exp(+q_k t)-scaled; the substitution is half-applied",
            )
        # And the two halves must cancel: source * recovery == the un-substituted inflow.
        t_grid = np.arange(Nt + 1) * problems[0].dt
        scaled = np.array([src0(t, x) for t in t_grid])
        np.testing.assert_allclose(
            iterator._undo_integrating_factor(0, q_0, scaled),
            Q[1, 0] * m1 * np.ones((Nt + 1, N)),
        )

    def test_fp_outflow_is_a_decay_with_the_rate_the_generator_names(self):
        """Same sign as before #1681 (mass leaves regime k), now on the factor channel."""
        iterator, problems, Q = self._iterator()
        q_0 = iterator._outflow_rate(0, 2, Q)
        assert q_0 == pytest.approx(Q[0, 1]), "outflow rate must be the off-diagonal row sum"
        assert q_0 > 0, "a positive q_k is what makes the factor exp(-q_k t) a sink"

        # A constant field must come back monotonically reduced, by exactly exp(-q_0 t).
        Nt, N = problems[0].Nt, 5
        N_k = np.ones((Nt + 1, N))
        recovered = iterator._undo_integrating_factor(0, q_0, N_k)
        t_grid = np.arange(Nt + 1) * problems[0].dt
        np.testing.assert_allclose(recovered, np.exp(-q_0 * t_grid)[:, None] * N_k)
        assert recovered[-1].max() < recovered[0].min(), "outflow sign flipped: regime k gained mass"

    def test_source_and_factor_reconstruct_the_original_right_hand_side(self):
        """The split is a change of channel, not of equation.

        ``d/dt[exp(-q t) n] = exp(-q t) (n' - q n)``, so with ``n' = L n + exp(q t) * inflow``
        the recovered ``m`` solves ``m' = L m + inflow - q m`` -- the right-hand side this
        class pinned before #1681. Checked at t=0, where the factor is 1 and the two forms
        must agree term by term.
        """
        iterator, problems, Q = self._iterator()
        Nt, N = problems[0].Nt, 5
        x = np.linspace(0.0, 1.0, N)
        m0, m1 = 0.7, 0.3
        Ms = [m0 * np.ones((Nt + 1, N)), m1 * np.ones((Nt + 1, N))]

        inflow = iterator._make_fp_source(0, 2, Q, Ms)(0.0, x)
        outflow = iterator._outflow_rate(0, 2, Q) * m0 * np.ones(N)
        np.testing.assert_allclose(inflow - outflow, (Q[1, 0] * m1 - Q[0, 1] * m0) * np.ones(N))


class TestRegimeFPDriftConvention:
    """FP drift-convention routing (#1315, Refs #1043).

    After the v0.18.6 rename ``drift_field`` is the velocity ``alpha*``, not the value
    function. For a smooth separable ``H`` the FP solver must receive ``U`` via
    ``potential_field`` and derive ``alpha*`` itself; passing ``U`` as ``drift_field`` bypasses
    ``resolve_fp_drift_kwargs`` and converges to a wrong equilibrium (silent-wrong-physics).
    This pins the per-regime FP solve to route ``U`` via ``potential_field`` so the bypass
    cannot silently reopen.
    """

    def test_fp_receives_U_via_potential_field_not_drift(self):
        problems, config, hjbs, fps = _make_2regime_system()
        iterator = RegimeSwitchingIterator(
            problems=problems,
            regime_config=config,
            hjb_solvers=hjbs,
            fp_solvers=fps,
            max_iterations=1,
        )

        captured: list[tuple[int, list[str]]] = []
        for k, fp in enumerate(fps):
            original = fp.solve_fp_system

            def spy(*args, _original=original, _k=k, **kwargs):
                captured.append((_k, sorted(kwargs)))
                return _original(*args, **kwargs)

            fp.solve_fp_system = spy

        iterator.solve()

        assert captured, "FP solver was never called"
        for k, keys in captured:
            assert "potential_field" in keys, (
                f"regime {k}: value function not routed via potential_field (kwargs={keys}); "
                "#1315 drift-convention bypass reopened"
            )
            assert "drift_field" not in keys, (
                f"regime {k}: value function wrongly passed as drift_field (velocity alpha*); "
                "#1315 silent-wrong-equilibrium"
            )


class TestDiagonalOutflowIsNotALaggedSource:
    """Issue #1681: the diagonal transition term must not reach the FP solver as a source.

    ``d_t m^k - L^k[m^k] = sum_j Q[j,k] m^j - q_k m^k``. A positivity-preserving ``L^k``
    keeps ``m^k >= 0`` against a **non-negative** source; the second term is neither
    non-negative nor proportional to the density the current step actually has, and
    passing it lagged drove ``divergence_upwind`` to ``-1.0e-03`` on a plain LQ problem.
    The fix carries it in an integrating factor instead, so these tests pin the split,
    not the symptom.
    """

    def _solve(self, Nt=10, T=1.0, max_iterations=3):
        problems = [_make_problem(coupling_strength=c, sigma=0.1, T=T, Nt=Nt) for c in (1.0, 0.5)]
        Q = np.array([[-0.1, 0.1], [0.2, -0.2]])
        iterator = RegimeSwitchingIterator(
            problems=problems,
            regime_config=RegimeSwitchingConfig(transition_matrix=Q),
            hjb_solvers=[HJBFDMSolver(p) for p in problems],
            fp_solvers=[FPFDMSolver(p) for p in problems],
            max_iterations=max_iterations,
            tolerance=1e-4,
            damping=0.5,
        )
        return problems, Q, iterator.solve()

    def test_every_regime_density_stays_non_negative(self):
        """The invariant the lagged sink broke. Reverting the split reddens this."""
        _problems, _Q, result = self._solve()
        for k, dens in enumerate(result.densities):
            M = np.asarray(dens, dtype=float)
            assert np.isfinite(M).all(), f"regime {k}: non-finite density"
            assert M.min() >= 0.0, (
                f"regime {k}: density went to {M.min():.4e}. The diagonal outflow is back in "
                "the FP source term (Issue #1681); divergence_upwind cannot preserve "
                "positivity against a lagged negative source."
            )

    def test_regime_masses_track_the_markov_chain_closed_form(self):
        """External oracle: mass transfer obeys ``M(t) = M(0) expm(Qt)``, independent of x.

        Integrating the FP system over a no-flux domain kills the transport term, so the
        regime masses solve the Markov chain's own ODE. That is not a second discretisation
        of the same scheme -- it is the continuous law the scheme is supposed to reproduce,
        so this cannot go tautological the way a path-A-vs-path-B agreement test does.
        """
        from scipy.linalg import expm

        problems, Q, result = self._solve()
        dx = problems[0].geometry.get_grid_spacing()[0]
        masses = np.array([np.asarray(d, dtype=float).sum(axis=1) * dx for d in result.densities])
        t_grid = np.arange(problems[0].Nt + 1) * problems[0].dt
        exact = np.array([masses[:, 0] @ expm(Q * t) for t in t_grid]).T

        rel = np.abs(masses - exact).max() / np.abs(exact).max()
        # 2.3e-03 measured at this fixture (Nt=10, 3 Picard iterations). The residual is the
        # lagged INFLOW plus the piecewise-constant-in-time source, and it does not clean up
        # under dt-refinement alone at fixed iteration count -- recorded, not hidden.
        assert rel < 5e-3, f"regime masses deviate {rel:.3e} from M(0) @ expm(Qt)"

        # Direction is part of the claim: stationary pi = [2/3, 1/3] from equal initial
        # masses means regime 0 must GAIN and regime 1 must LOSE. A sign-flipped transfer
        # would keep the magnitude and fail here.
        assert masses[0, -1] > masses[0, 0], "regime 0 should gain mass (pi_0 = 2/3)"
        assert masses[1, -1] < masses[1, 0], "regime 1 should lose mass (pi_1 = 1/3)"

    def test_outflow_horizon_beyond_float64_accuracy_is_refused(self):
        """The integrating factor spans exp(q_k T); past the limit it must stop, not degrade."""
        problems = [_make_problem(coupling_strength=c, sigma=0.1) for c in (1.0, 0.5)]
        for p in problems:
            p.T, p.Nt = 100.0, 10
        Q = np.array([[-2.0, 2.0], [1.0, -1.0]])  # q_0 * T = 200 > 50
        with pytest.raises(ValueError, match=r"q_k\*T"):
            RegimeSwitchingIterator(
                problems=problems,
                regime_config=RegimeSwitchingConfig(transition_matrix=Q),
                hjb_solvers=[HJBFDMSolver(p) for p in problems],
                fp_solvers=[FPFDMSolver(p) for p in problems],
            )
