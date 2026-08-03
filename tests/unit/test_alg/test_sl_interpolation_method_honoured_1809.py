"""The SL solver must refuse an interpolation_method it will not honour (#1809, #1664).

`interpolation_method` was stored unvalidated, and both interpolators default to linear for
anything they do not recognise. Two consequences, both silent:

- **Dimension-dependent meaning.** The 1D path is ``if method == "cubic": PCHIP else: linear``,
  so it recognises two methods; the nD path recognises five. Measured on
  ``U = [0, 10, 0, 10, 0]`` at ``x = 0.30`` (nodes at 0.25 and 0.50): ``nearest`` returns
  **8.0 in 1D** and **10.0 in nD** -- the same public argument naming a different interpolant
  depending on dimension. A typo collapsed to linear at both.
- **Grid too small for the method.** ``RegularGridInterpolator`` needs 4 points per axis for
  ``cubic`` and 6 for ``quintic``. Below that it raises, and the fallback chain substituted
  ``RBF -> nearest neighbour`` behind ``logger.debug`` -- the caller asked for cubic and could
  receive nearest (#1664).

The oracle here is not a numerical law, so these are **convention pins**: each is verified by
mutating the guard and observing it redden, and the accept-cases exist because a guard that
refuses everything passes every refuse-case and looks correct.
"""

from __future__ import annotations

import pytest

import numpy as np

from mfgarchon import MFGProblem
from mfgarchon.alg.numerical.hjb_solvers import HJBSemiLagrangianSolver
from mfgarchon.alg.numerical.hjb_solvers.hjb_sl_interpolation import (
    HONOURED_METHODS_ND,
    MIN_POINTS_PER_AXIS_ND,
    honoured_methods,
    interpolate_value_1d,
    interpolate_value_nd,
    min_points_per_axis,
)
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.core.mfg_components import MFGComponents
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import no_flux_bc


def _problem(dimension: int, n: int) -> MFGProblem:
    return MFGProblem(
        geometry=TensorProductGrid(
            bounds=[(0.0, 1.0)] * dimension,
            Nx_points=[n] * dimension,
            boundary_conditions=no_flux_bc(dimension=dimension),
        ),
        T=0.5,
        Nt=5,
        sigma=0.3,
        components=MFGComponents(
            m_initial=lambda x: 1.0,
            u_terminal=lambda x: 0.0,
            hamiltonian=SeparableHamiltonian(
                control_cost=QuadraticControlCost(control_cost=1.0),
                coupling=lambda m: m,
                coupling_dm=lambda m: 1.0,
            ),
        ),
    )


class TestTheDivergenceThatMotivatesTheGuard:
    """Pin the measured fact itself, so the guard's reason cannot rot away from it."""

    def test_nearest_means_different_things_in_1d_and_nd(self):
        xg = np.linspace(0.0, 1.0, 5)
        u = np.array([0.0, 10.0, 0.0, 10.0, 0.0])
        q = 0.30  # between nodes at 0.25 and 0.50

        assert interpolate_value_1d(u, q, xg, method="nearest") == pytest.approx(8.0), (
            "the 1D path recognises only 'cubic'; everything else is linear"
        )
        u2 = np.repeat(u[:, None], 5, axis=1)
        assert interpolate_value_nd(u2, np.array([q, 0.5]), (xg, np.linspace(0, 1, 5)), (5, 5), method="nearest") == (
            pytest.approx(10.0)
        ), "the nD path honours 'nearest'"

    def test_the_honoured_sets_differ_by_dimension(self):
        assert honoured_methods(1) < honoured_methods(2), "1D honours strictly fewer methods"
        assert "nearest" in honoured_methods(2)
        assert "nearest" not in honoured_methods(1)


class TestUnhonouredMethodsAreRefused:
    @pytest.mark.parametrize("method", ["nearest", "slinear", "quintic"], ids=["nearest", "slinear", "quintic"])
    def test_an_nd_only_method_is_refused_in_1d(self, method):
        with pytest.raises(ValueError, match=r"not honoured at dimension 1"):
            HJBSemiLagrangianSolver(_problem(1, 21), interpolation_method=method)

    @pytest.mark.parametrize("dimension", [1, 2])
    def test_a_typo_is_refused_rather_than_silently_linear(self, dimension):
        with pytest.raises(ValueError, match=r"not honoured at dimension"):
            HJBSemiLagrangianSolver(_problem(dimension, 11), interpolation_method="cubis")


class TestGridTooSmallForTheMethodIsRefused:
    @pytest.mark.parametrize(("method", "n"), [("cubic", 3), ("quintic", 5)], ids=["cubic-on-3", "quintic-on-5"])
    def test_below_the_per_axis_minimum_is_refused(self, method, n):
        need = min_points_per_axis(method, 2)
        assert n < need, "fixture must actually be below the minimum"
        # The stated number, not `\d+`: a message that announces a different minimum than the
        # one enforced is a silent drift, and `\d+` matched it happily.
        with pytest.raises(ValueError, match=rf"needs at least {need} points per axis"):
            HJBSemiLagrangianSolver(_problem(2, n), interpolation_method=method)

    @pytest.mark.parametrize(("method", "n"), [("cubic", 4), ("quintic", 6)], ids=["cubic-at-4", "quintic-at-6"])
    def test_exactly_at_the_minimum_is_accepted(self, method, n):
        """The boundary of the refusal, on the accepting side -- an off-by-one here refuses valid work."""
        assert n == min_points_per_axis(method, 2)
        HJBSemiLagrangianSolver(_problem(2, n), interpolation_method=method)


class TestHonouredConfigurationsStillConstruct:
    """Negative controls. A guard that raises unconditionally passes every test above."""

    @pytest.mark.parametrize(
        ("dimension", "n", "method"),
        [
            (1, 21, "linear"),
            (1, 21, "cubic"),
            (2, 11, "linear"),
            (2, 11, "slinear"),
            (2, 11, "nearest"),
            (2, 11, "cubic"),
            (2, 11, "quintic"),
        ],
    )
    def test_every_honoured_combination_constructs(self, dimension, n, method):
        solver = HJBSemiLagrangianSolver(_problem(dimension, n), interpolation_method=method)
        assert solver.interpolation_method == method


class TestTheMinimaAreAnchoredToScipy:
    """For the per-axis minima, scipy IS the external oracle -- so consult it, not our table.

    Every other assertion in this file re-reads the same dict the guard reads, which cannot
    detect the dict being wrong. If scipy ever changed a minimum, the guard would accept a grid
    the interpolator then refuses at runtime, and #1664's silent RBF-to-nearest degradation would
    reopen behind a green suite. This is the one part of the change with a law outside the code.
    """

    @pytest.mark.parametrize("method", sorted(HONOURED_METHODS_ND))
    def test_scipy_agrees_with_the_nd_table(self, method):
        from scipy.interpolate import RegularGridInterpolator

        need = MIN_POINTS_PER_AXIS_ND[method]

        def rgi_works(n: int) -> bool:
            coords = (np.linspace(0.0, 1.0, n),) * 2
            try:  # exactly the construction hjb_sl_interpolation uses
                RegularGridInterpolator(coords, np.zeros((n, n)), method=method, bounds_error=False, fill_value=None)(
                    np.array([[0.5, 0.5]])
                )
            except Exception:
                return False
            return True

        assert rgi_works(need), f"table says {method} needs {need} points, scipy refuses that"
        if need > 1:
            assert not rgi_works(need - 1), f"table says {method} needs {need}, but scipy accepts {need - 1}"


class TestTheRefusalNamesTheRightAxis:
    """The axis number is what the message tells the user to act on, so pin it.

    Every other refusal fixture is isotropic `(n, n)`, where reporting the first or the last
    short axis is indistinguishable.
    """

    # The third case is the discriminating one. With a single short axis, reporting the first or
    # the last short axis selects the same element, and a mutation swapping them survives. Two
    # short axes of DIFFERENT lengths separate them: first is (axis 0, n=3), last is (axis 1, n=2).
    @pytest.mark.parametrize(
        ("shape", "expected_axis", "expected_n"),
        [((3, 20), 0, 3), ((20, 3), 1, 3), ((3, 2), 0, 3)],
        ids=["short-first", "short-last", "both-short-different-lengths"],
    )
    def test_the_short_axis_is_identified(self, shape, expected_axis, expected_n):
        problem = MFGProblem(
            geometry=TensorProductGrid(
                bounds=[(0.0, 1.0), (0.0, 1.0)],
                Nx_points=list(shape),
                boundary_conditions=no_flux_bc(dimension=2),
            ),
            T=0.5,
            Nt=5,
            sigma=0.3,
            components=MFGComponents(
                m_initial=lambda x: 1.0,
                u_terminal=lambda x: 0.0,
                hamiltonian=SeparableHamiltonian(
                    control_cost=QuadraticControlCost(control_cost=1.0),
                    coupling=lambda m: m,
                    coupling_dm=lambda m: 1.0,
                ),
            ),
        )
        with pytest.raises(ValueError, match=rf"axis {expected_axis} has {expected_n}"):
            HJBSemiLagrangianSolver(problem, interpolation_method="cubic")


class TestOneDimensionalMinimaFollowTheTruePathNotRegularGridInterpolator:
    """1D uses PCHIP/CubicSpline, which never touch RegularGridInterpolator.

    Applying the nD table here refused `cubic` at n=3, where cubic is genuinely honoured --
    measured on a non-linear profile, PCHIP 0.7477 against linear 0.9256. At n=2 the two agree
    exactly, so refusing there is right. The 1D boundary is 3.
    """

    def test_cubic_is_accepted_at_three_points_in_1d(self):
        assert min_points_per_axis("cubic", 1) == 3
        HJBSemiLagrangianSolver(_problem(1, 3), interpolation_method="cubic")

    def test_cubic_is_refused_at_two_points_in_1d_where_it_is_linear(self):
        xg = np.linspace(0.0, 1.0, 2)
        u = np.array([0.0, 7.0])
        assert interpolate_value_1d(u, 0.3, xg, method="cubic") == pytest.approx(
            interpolate_value_1d(u, 0.3, xg, method="linear")
        ), "at 2 points cubic IS linear, which is why it is refused"
        with pytest.raises(ValueError, match=r"needs at least 3 points per axis at dimension 1"):
            HJBSemiLagrangianSolver(_problem(1, 2), interpolation_method="cubic")


class TestAnUnrecognisedMethodRaisesAtND:
    """The changelog ships this as a behaviour change, so it owes a test (#1814 review, N1).

    `interpolate_value_nd` used to return the LINEAR value for any method it did not recognise --
    the catch-all `else` that made a typo indistinguishable from a deliberate choice. It now
    raises. Unreachable through the solver, which validates at construction, so this is the only
    thing that exercises it.
    """

    def test_an_unknown_method_raises_rather_than_returning_linear(self):
        xg = np.linspace(0.0, 1.0, 5)
        u = np.repeat(np.array([0.0, 10.0, 0.0, 10.0, 0.0])[:, None], 5, axis=1)
        with pytest.raises(ValueError, match="not defined"):
            interpolate_value_nd(u, np.array([0.30, 0.5]), (xg, xg), (5, 5), method="bogus")

    def test_at_one_axis_it_does_not_raise_and_the_changelog_says_so(self):
        """The claim above is scoped to >=2 axes, and this is why -- so it cannot silently widen.

        `sl_backend`'s `dimension` selects WHICH SL MACHINERY applies, and the 1D branch is total
        over method strings, so a one-axis grid takes the characteristic-path pair and absorbs
        everything RegularGridInterpolator would have refused. That is `interpolate_value_1d`'s
        long-standing behaviour, now reached through both names rather than the two disagreeing.
        Unreachable through the solver -- `_grid_shape` exists only when `dimension > 1`.
        """
        xg = np.linspace(0.0, 1.0, 5)
        u = np.array([0.0, 10.0, 0.0, 10.0, 0.0])
        assert interpolate_value_nd(u, np.array([0.30]), (xg,), (5,), method="bogus") == pytest.approx(8.0)
        assert interpolate_value_nd(u, np.array([0.30]), (xg,), (5,), method="quintic") == pytest.approx(8.0)

    def test_an_honoured_method_still_returns_a_value(self):
        """Negative control: a guard that raises on everything passes the test above."""
        xg = np.linspace(0.0, 1.0, 5)
        u = np.repeat(np.array([0.0, 10.0, 0.0, 10.0, 0.0])[:, None], 5, axis=1)
        assert interpolate_value_nd(u, np.array([0.30, 0.5]), (xg, xg), (5, 5), method="nearest") == pytest.approx(10.0)
