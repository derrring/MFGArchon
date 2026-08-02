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
    MIN_POINTS_PER_AXIS,
    honoured_methods,
    interpolate_value_1d,
    interpolate_value_nd,
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
        assert n < MIN_POINTS_PER_AXIS[method], "fixture must actually be below the minimum"
        with pytest.raises(ValueError, match=r"needs at least \d+ points per axis"):
            HJBSemiLagrangianSolver(_problem(2, n), interpolation_method=method)

    @pytest.mark.parametrize(("method", "n"), [("cubic", 4), ("quintic", 6)], ids=["cubic-at-4", "quintic-at-6"])
    def test_exactly_at_the_minimum_is_accepted(self, method, n):
        """The boundary of the refusal, on the accepting side -- an off-by-one here refuses valid work."""
        assert n == MIN_POINTS_PER_AXIS[method]
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
