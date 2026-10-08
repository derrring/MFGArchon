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
from mfgarchon.utils.convergence.convergence_metrics import (
    RELATIVE_CHANGE_FLOOR,
    calculate_l2_convergence_metrics,
    l2_change,
    sweep_change,
)

_NX, _NT = 12, 5


def _problem(data: str = "moving") -> MFGProblem:
    """A 1-D coupled problem on 12 points, with one of five data sets.

    - ``moving``: a bump of mass 1 drifting toward a terminal-cost well; both fields move.
    - ``heavy``: mass 10 with coupling 0.1 m, so f(m) and the dynamics are ``moving``'s: the relative
      change is the same and the absolute change ten times ``moving``'s.
    - ``static_u``: the bump with no coupling and no terminal cost, so u = 0 exactly and only m moves.
    - ``static_m``: a uniform density with coupling 5 m and terminal cost 1, so u = 1 + 5 (T - t) is flat,
      m stays uniform to rounding (relative change 1.9e-16 per sweep), and only u moves. The terminal row anchors u's norm away from 0, so a damped
      step's relative size does not divide its damping factor back out.
    - ``static``: a uniform density with no coupling and no terminal cost: neither field moves (u = 0
      exactly, m to rounding).
    """
    x = np.linspace(0.0, 1.0, _NX)
    mass = np.sum(np.exp(-30 * (x - 0.3) ** 2) * np.r_[0.5, np.ones(_NX - 2), 0.5]) / (_NX - 1)
    bump = lambda x: np.exp(-30 * (np.asarray(x, dtype=float) - 0.3) ** 2) / mass  # noqa: E731
    flat = lambda x: 1.0 + 0.0 * np.asarray(x, dtype=float)  # noqa: E731
    well = lambda x: (np.asarray(x, dtype=float) - 0.7) ** 2  # noqa: E731
    coupled = data in ("moving", "heavy", "static_m")
    c = {"heavy": 0.1, "static_m": 5.0}.get(data, 1.0)
    return MFGProblem(
        geometry=TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[_NX], boundary_conditions=no_flux_bc(dimension=1)),
        T=0.2,
        Nt=_NT,
        volatility=0.3,
        components=MFGComponents(
            hamiltonian=SeparableHamiltonian(
                control_cost=QuadraticControlCost(control_cost=1.0),
                coupling=(lambda m: c * m) if coupled else (lambda m: 0.0 * np.asarray(m)),
                coupling_dm=(lambda m: c) if coupled else (lambda m: 0.0 * np.asarray(m)),
            ),
            u_terminal=well
            if data in ("moving", "heavy")
            else (lambda x: (1.0 if data == "static_m" else 0.0) + 0.0 * np.asarray(x)),
            m_initial={"heavy": lambda x: 10.0 * bump(x)}.get(data, bump if data in ("moving", "static_u") else flat),
        ),
    )


def _solve(
    kind: str,
    relaxation: float,
    tolerance: float,
    absolute_tolerance: float | None,
    data: str = "moving",
    mixed: bool = False,
    sweeps: int = 3,
) -> bool:
    """Run one coupling iterator for at most ``sweeps`` sweeps and return its verdict.

    ``mixed`` gives a multi-field iterator ``static`` fields with one ``data`` field among them, second of
    two or in the middle of three, and no coupling between fields.

    A regime problem on ``static_u`` data switches one way, from regime 1 into regime 0 at rate 5: regime
    1's density is static after its first sweep, while regime 0's keeps moving because its inflow reads
    regime 1's previous trajectory.
    """

    def fields(n):
        if not mixed:
            return [_problem(data) for _ in range(n)]
        return [_problem(data) if i == 1 else _problem("static") for i in range(n)]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        logging.disable(logging.WARNING)
        try:
            if kind in ("fixed_point", "block", "fictitious_play"):
                p = _problem(data)
                hjb, fp = HJBFDMSolver(p), FPFDMSolver(p)
                if kind == "fixed_point":
                    it = FixedPointIterator(p, hjb, fp, relaxation=relaxation)
                    r = it.solve(max_iterations=sweeps, tolerance=tolerance, absolute_tolerance=absolute_tolerance)
                elif kind == "block":
                    it = BlockIterator(p, hjb, fp, method="gauss_seidel", relaxation=relaxation)
                    r = it.solve(
                        max_iterations=sweeps, tolerance=tolerance, absolute_tolerance=absolute_tolerance, verbose=False
                    )
                else:
                    r = FictitiousPlayIterator(p, hjb, fp).solve(
                        max_iterations=sweeps, tolerance=tolerance, absolute_tolerance=absolute_tolerance
                    )
                return bool(r.converged)
            if kind == "multi_population":
                ps = fields(3 if mixed else 2)
                multi = MultiPopulationProblem(populations=ps, population_names=["a", "b", "c"][: len(ps)])
                it = MultiPopulationIterator(
                    multi, [HJBFDMSolver(q) for q in ps], [FPFDMSolver(q) for q in ps], relaxation=relaxation
                )
                return bool(
                    it.solve(
                        max_iterations=sweeps, tolerance=tolerance, absolute_tolerance=absolute_tolerance
                    ).converged
                )
            if kind == "graph":
                ps = fields(3)
                adjacency = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=float)
                strength = 0.0 if mixed else 1.0
                return (
                    GraphMFGSolver(
                        ps,
                        AdjacencyCoupling(adjacency, alpha=0.05 * strength, beta=0.02 * strength),
                        [HJBFDMSolver(q) for q in ps],
                        [FPFDMSolver(q) for q in ps],
                        max_iterations=sweeps,
                        tolerance=tolerance,
                        damping=relaxation,
                        absolute_tolerance=absolute_tolerance,
                    )
                    .solve()
                    .converged
                )
            assert kind == "regime"
            ps = fields(2)
            # No switching for `mixed`; one-way switching from regime 1 into regime 0 for `static_u` (above); a
            # symmetric generator between two identical uniform densities for `static_m`.
            if mixed:
                q = np.zeros((2, 2))
            elif data == "static_u":
                q = np.array([[0.0, 0.0], [5.0, -5.0]])
            elif data == "static_m":
                q = np.array([[-0.1, 0.1], [0.1, -0.1]])
            else:
                q = np.array([[-0.1, 0.1], [0.2, -0.2]])
            return (
                RegimeSwitchingIterator(
                    ps,
                    RegimeSwitchingConfig(transition_matrix=q),
                    [HJBFDMSolver(q_) for q_ in ps],
                    [FPFDMSolver(q_) for q_ in ps],
                    max_iterations=sweeps,
                    tolerance=tolerance,
                    damping=relaxation,
                    absolute_tolerance=absolute_tolerance,
                )
                .solve()
                .converged
            )
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

    On the ``heavy`` data at relaxation 1, every iterator's relative change is below 0.5 by sweep 2
    (0.379 for the single-pair iterators; regime measures from its second sweep), while their absolute
    change there is 2.39. So an iterator that compared the absolute change by default,
    against the same tolerance, would not stop within two sweeps.
    """
    assert _solve(kind, relaxation=1.0, tolerance=0.5, absolute_tolerance=None, data="heavy", sweeps=2) is True
    assert _solve(kind, relaxation=1.0, tolerance=0.5, absolute_tolerance=1e-12, data="heavy", sweeps=2) is False


@pytest.mark.parametrize(
    ("kind", "data"),
    [
        *[pytest.param(kind, "static_u", id=f"{kind}-only_m_moves") for kind in _DAMPED],
        *[pytest.param(kind, "static_m", id=f"{kind}-only_u_moves") for kind in _DAMPED if kind != "multi_population"],
    ],
)
def test_each_field_is_measured_before_damping(kind, data):
    """#1684 item 7, one field at a time, which is the shape the multi-population defect had.

    One field is static (u = 0 exactly, or a uniform m that stays uniform to rounding) and the other moves:
    m by about 0.2 relative per sweep, u by 0.39 (fixed point, block) to 0.96 (graph, regime). At relaxation
    0.01 the moving field's damped step is a hundredth of that, so an iterator that measured only that
    field's damped step would stop at a tolerance of 0.1. A pin that damps both fields at once cannot see a
    one-field revert. `MultiPopulationIterator` never damps u, so it has no u half to revert.
    """
    assert _solve(kind, relaxation=0.01, tolerance=0.1, absolute_tolerance=None, data=data) is False


@pytest.mark.parametrize(
    ("kind", "data", "tolerance", "sweeps", "mixed"),
    [
        pytest.param("graph", "static_u", 0.15, 3, True, id="graph-m_moves"),
        pytest.param("graph", "static_m", 0.6, 3, True, id="graph-u_moves"),
        pytest.param("multi_population", "static_u", 0.15, 3, True, id="multi_population-m_moves"),
        pytest.param("multi_population", "static_m", 0.6, 1, True, id="multi_population-u_moves"),
        pytest.param("regime", "static_u", 0.15, 3, False, id="regime-m_moves"),
        pytest.param("regime", "static_m", 0.6, 3, True, id="regime-u_moves"),
    ],
)
def test_the_largest_field_change_decides(kind, data, tolerance, sweeps, mixed):
    """A multi-field iterator takes the max over fields, of each field's change (#2555 ruling (ii)).

    One field moves in one quantity (m by about 0.19, or u by about 0.95, relative per sweep at relaxation
    0.01) and the others are static: zero-coupling fields around it in the middle of three (graph,
    populations) or second of two (regime's u case), or the source regime of one-way switching (regime's
    m case). Each tolerance sits below the moving field's change and above the mean over fields, so a min,
    a mean, or reading one end field would stop where the max does not. Populations run one sweep for u,
    because `MultiPopulationIterator` does not damp u and reaches u's fixed point at the second sweep.
    """
    assert (
        _solve(
            kind, relaxation=0.01, tolerance=tolerance, absolute_tolerance=None, data=data, mixed=mixed, sweeps=sweeps
        )
        is False
    )


@pytest.mark.parametrize("route", ["keyword", "nt_reentry", "config"])
def test_problem_solve_threads_the_absolute_bound(route):
    """`MFGProblem.solve(absolute_tolerance=)`, its `Nt=` re-entry and `PicardConfig` each reach the verdict.

    The ``heavy`` data through the default scheme at relaxation 1 meets a relative tolerance of 0.5 at
    sweep 2; an absolute bound of 1e-12 is not met within three sweeps.
    """
    from mfgarchon.config import MFGSolverConfig
    from mfgarchon.config.core import PicardConfig

    x = np.linspace(0.0, 1.0, _NX)
    mass = np.sum(np.exp(-30 * (x - 0.3) ** 2) * np.r_[0.5, np.ones(_NX - 2), 0.5]) / (_NX - 1)

    def run(absolute_tolerance):
        problem = MFGProblem(
            model=Model(
                hamiltonian=SeparableHamiltonian(
                    control_cost=QuadraticControlCost(control_cost=1.0),
                    coupling=lambda m: 0.1 * m,
                    coupling_dm=lambda m: 0.1 + 0.0 * np.asarray(m),
                ),
                volatility=0.3,
            ),
            domain=TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[_NX], boundary_conditions=no_flux_bc(dimension=1)),
            conditions=Conditions(
                m_initial=lambda x: 10.0 * np.exp(-30 * (np.asarray(x, dtype=float) - 0.3) ** 2) / mass,
                u_terminal=lambda x: (np.asarray(x, dtype=float) - 0.7) ** 2,
                T=0.2,
            ),
            Nt=_NT,
        )
        picard = {"max_iterations": 3, "tolerance": 0.5, "relaxation": 1.0, "verbose": False}
        if route == "config":
            return problem.solve(
                config=MFGSolverConfig(picard=PicardConfig(**picard, absolute_tolerance=absolute_tolerance))
            )
        nt = {"nt_reentry": _NT + 1}.get(route)
        return problem.solve(
            Nt=nt, config=MFGSolverConfig(picard=PicardConfig(**picard)), absolute_tolerance=absolute_tolerance
        )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert run(None).converged is True
        assert run(1e-12).converged is False


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
        (
            lambda: TensorProductGrid(
                bounds=[(0.0, 1.0)], Nx_points=[_NX], boundary_conditions=no_flux_bc(dimension=1)
            ),
            1.0,
        ),
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
    absolute, relative = l2_change(
        np.full(shape, 3.0), np.full(shape, 1.0), problem.spatial_measure().integrate, problem.dt
    )
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


@pytest.mark.parametrize("shape", [(_NT + 1, _NX), (_NT + 1, 5, 7)], ids=["1d", "2d"])
def test_the_deprecated_scalar_spacing_form_is_the_owner_on_a_uniform_measure(shape):
    """The deprecation's equivalence test: old function against owner, and against the pre-#2555 formula.

    ``calculate_l2_convergence_metrics(..., Dx, Dt)`` is ``sweep_change`` on the measure that weighs every
    node ``Dx``, which is the ``||diff||_2 * sqrt(Dx * Dt)`` it computed before #2555.
    """
    rng = np.random.default_rng(0)
    U, U0, M, M0 = (rng.standard_normal(shape) for _ in range(4))
    Dx, Dt = 0.1, 0.04
    with pytest.warns(DeprecationWarning, match="sweep_change"):
        old = calculate_l2_convergence_metrics(U, U0, M, M0, Dx, Dt)
    new = sweep_change(U, U0, M, M0, lambda f: np.sum(f, axis=tuple(range(1, f.ndim))) * Dx, Dt)
    assert old == pytest.approx(new, rel=1e-13)
    assert old["l2distu_abs"] == pytest.approx(np.linalg.norm(U - U0) * np.sqrt(Dx * Dt), rel=1e-13)
    assert old["l2distm_rel"] == pytest.approx(np.linalg.norm(M - M0) / np.linalg.norm(M), rel=1e-13)
