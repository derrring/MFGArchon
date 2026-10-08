"""The outer coupling tolerance: one measure, one norm, one criterion, one compared pair (#2555).

CONVENTIONS.md § 9: the outer tolerance bounds the relative discrete L2 change one sweep makes, in the
problem's own spatial measure, with no absolute criterion unless one is asked for. The change is the
map's output against its input, never the damped step (#1684 item 7).
"""

from __future__ import annotations

import logging
import warnings

import pytest

import numpy as np

from mfgarchon import Conditions, Model
from mfgarchon.alg.numerical.coupling.block_iterators import BlockIterator
from mfgarchon.alg.numerical.coupling.fictitious_play import FictitiousPlayIterator
from mfgarchon.alg.numerical.coupling.fixed_point_iterator import FixedPointIterator
from mfgarchon.alg.numerical.coupling.graph_coupling import AdjacencyCoupling
from mfgarchon.alg.numerical.coupling.graph_mfg_solver import GraphMFGSolver
from mfgarchon.alg.numerical.coupling.multi_population_iterator import MultiPopulationIterator
from mfgarchon.alg.numerical.coupling.regime_switching_iterator import RegimeSwitchingIterator
from mfgarchon.alg.numerical.fp_solvers import FPFDMSolver
from mfgarchon.alg.numerical.hjb_solvers import HJBFDMSolver
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.core.mfg_components import MFGComponents
from mfgarchon.core.mfg_problem import MFGProblem
from mfgarchon.core.multi_population import MultiPopulationProblem
from mfgarchon.core.regime_switching import RegimeSwitchingConfig
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import no_flux_bc
from mfgarchon.utils.convergence.convergence_metrics import RELATIVE_CHANGE_FLOOR, l2_change

_NX, _NT = 12, 5


def _problem() -> MFGProblem:
    """A 1-D coupled problem whose map moves: a bump of mass 1 drifting toward a terminal-cost well."""
    x = np.linspace(0.0, 1.0, _NX)
    mass = np.sum(np.exp(-30 * (x - 0.3) ** 2) * np.r_[0.5, np.ones(_NX - 2), 0.5]) / (_NX - 1)
    return MFGProblem(
        geometry=TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[_NX], boundary_conditions=no_flux_bc(dimension=1)),
        T=0.2,
        Nt=_NT,
        volatility=0.3,
        components=MFGComponents(
            hamiltonian=SeparableHamiltonian(
                control_cost=QuadraticControlCost(control_cost=1.0), coupling=lambda m: m, coupling_dm=lambda m: 1.0
            ),
            u_terminal=lambda x: (np.asarray(x, dtype=float) - 0.7) ** 2,
            m_initial=lambda x: np.exp(-30 * (np.asarray(x, dtype=float) - 0.3) ** 2) / mass,
        ),
    )


def _solve(kind: str, relaxation: float, tolerance: float, absolute_tolerance: float | None) -> bool:
    """Run one coupling iterator for at most 3 sweeps and return its verdict."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        logging.disable(logging.WARNING)
        try:
            if kind in ("fixed_point", "block", "fictitious_play"):
                p = _problem()
                hjb, fp = HJBFDMSolver(p), FPFDMSolver(p)
                if kind == "fixed_point":
                    it = FixedPointIterator(p, hjb, fp, relaxation=relaxation)
                    r = it.solve(max_iterations=3, tolerance=tolerance, absolute_tolerance=absolute_tolerance)
                elif kind == "block":
                    it = BlockIterator(p, hjb, fp, method="gauss_seidel", relaxation=relaxation)
                    r = it.solve(
                        max_iterations=3, tolerance=tolerance, absolute_tolerance=absolute_tolerance, verbose=False
                    )
                else:
                    r = FictitiousPlayIterator(p, hjb, fp).solve(
                        max_iterations=3, tolerance=tolerance, absolute_tolerance=absolute_tolerance
                    )
                return bool(r.converged)
            if kind == "multi_population":
                ps = [_problem(), _problem()]
                multi = MultiPopulationProblem(populations=ps, population_names=["a", "b"])
                it = MultiPopulationIterator(
                    multi, [HJBFDMSolver(q) for q in ps], [FPFDMSolver(q) for q in ps], relaxation=relaxation
                )
                return bool(
                    it.solve(max_iterations=3, tolerance=tolerance, absolute_tolerance=absolute_tolerance).converged
                )
            if kind == "graph":
                ps = [_problem() for _ in range(3)]
                adjacency = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=float)
                return GraphMFGSolver(
                    ps,
                    AdjacencyCoupling(adjacency, alpha=0.05, beta=0.02),
                    [HJBFDMSolver(q) for q in ps],
                    [FPFDMSolver(q) for q in ps],
                    max_iterations=3,
                    tolerance=tolerance,
                    damping=relaxation,
                    absolute_tolerance=absolute_tolerance,
                ).solve().converged
            assert kind == "regime"
            ps = [_problem(), _problem()]
            return RegimeSwitchingIterator(
                ps,
                RegimeSwitchingConfig(transition_matrix=np.array([[-0.1, 0.1], [0.2, -0.2]])),
                [HJBFDMSolver(q) for q in ps],
                [FPFDMSolver(q) for q in ps],
                max_iterations=3,
                tolerance=tolerance,
                damping=relaxation,
                absolute_tolerance=absolute_tolerance,
            ).solve().converged
        finally:
            logging.disable(logging.NOTSET)


_DAMPED = ["fixed_point", "block", "multi_population", "graph", "regime"]


@pytest.mark.parametrize("kind", _DAMPED)
def test_damping_cannot_buy_the_verdict(kind):
    """#1684 item 7, over every iterator that damps. The counterfactual is the relaxation factor.

    The damped step is ``theta * (map - old)``. Measured on this fixture at relaxation 0.01, the block,
    graph and regime iterators used to report a change of 6.1e-03, 2.2e-02 and 1.2e-02, so a tolerance
    of 0.1 was met within two sweeps; the map's own relative change stays above 0.69 for all five
    iterators over the first three sweeps. One pin over the shared owner's callers rather than one per
    iterator: the owner cannot see which pair a caller hands it, so a pin on the owner alone would pass
    a caller that hands it the damped iterate.
    """
    assert _solve(kind, relaxation=0.01, tolerance=0.1, absolute_tolerance=None) is False, (
        f"{kind} reported convergence at relaxation 0.01: its change is the damped step, not the map's "
        "output against its input (#1684 item 7, #2555)"
    )


@pytest.mark.parametrize("kind", [*_DAMPED, "fictitious_play"])
def test_an_absolute_bound_applies_only_when_asked_for(kind):
    """Each iterator stops on the relative change alone, and on both when ``absolute_tolerance`` is set.

    At relaxation 1 every iterator's relative change falls below 0.5 by the third sweep on this
    fixture; an absolute bound of 1e-12 is not met by then.
    """
    assert _solve(kind, relaxation=1.0, tolerance=0.5, absolute_tolerance=None) is True
    assert _solve(kind, relaxation=1.0, tolerance=0.5, absolute_tolerance=1e-12) is False


def _graded_grid():
    x = np.sort(np.r_[0.0, np.random.default_rng(0).uniform(0.0, 1.0, _NX - 2), 1.0])
    return TensorProductGrid(
        bounds=[(0.0, 1.0)],
        Nx_points=[_NX],
        spacing_type="custom",
        custom_coordinates=[x],
        boundary_conditions=no_flux_bc(dimension=1),
    )


@pytest.mark.parametrize(
    ("geometry", "area"),
    [
        (lambda: TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[_NX], boundary_conditions=no_flux_bc(dimension=1)), 1.0),
        (
            lambda: TensorProductGrid(
                bounds=[(0.0, 1.0), (0.0, 10.0)], Nx_points=[_NX, 7], boundary_conditions=no_flux_bc(dimension=2)
            ),
            10.0,
        ),
        (_graded_grid, 1.0),
    ],
    ids=["1d", "2d_tall", "graded"],
)
def test_the_change_is_the_l2_norm_in_the_geometrys_own_measure(geometry, area):
    """A constant change is integrated exactly by any quadrature, so its norm is known in closed form.

    Before #2555 the absolute value took the first axis's spacing alone, which on the 2-D extent is off
    by a factor of about sqrt(N_y / L_y), and a graded grid raised from ``get_grid_spacing()[0]``.
    """
    problem = MFGProblem(
        model=Model(
            hamiltonian=SeparableHamiltonian(
                control_cost=QuadraticControlCost(control_cost=1.0), coupling=lambda m: m, coupling_dm=lambda m: 1.0
            ),
            volatility=0.3,
        ),
        domain=geometry(),
        conditions=Conditions(m_initial=lambda p: 1.0 / area, u_terminal=lambda p: 0.0, T=0.2),
        Nt=_NT,
    )
    shape = (problem.Nt + 1, *problem.geometry.get_grid_shape())
    absolute, relative = l2_change(np.full(shape, 3.0), np.full(shape, 1.0), problem.spatial_measure().integrate, problem.dt)
    assert absolute == pytest.approx(2.0 * np.sqrt(area * problem.dt * (problem.Nt + 1)), rel=1e-12)
    assert relative == pytest.approx(2.0 / 3.0, rel=1e-12)


def test_below_the_floor_the_relative_change_is_the_absolute_one():
    """A map whose output has no norm has no relative change; the absolute one stands in for it."""
    integrate = lambda f: np.sum(f, axis=-1) * 0.1  # noqa: E731
    new, old = np.zeros((3, 4)), np.full((3, 4), 1e-3)
    absolute, relative = l2_change(new, old, integrate, 0.5)
    assert RELATIVE_CHANGE_FLOOR == 1e-12
    assert absolute == pytest.approx(np.sqrt(0.5 * 3 * 4 * 0.1 * 1e-6), rel=1e-12)
    assert relative == absolute
