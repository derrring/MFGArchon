"""FP-GFDM is withdrawn: it refuses construction, and the alternative its refusal names holds a wall (#2583).

FP-GFDM built no wall: no flux or ghost row, and an operator with no boundary-condition argument, so on a bounded
domain the drift's flux crossed the boundary. What that did to mass depended on the drift (it stayed at 1 under a
uniform drift and moved to 0.599 or 1.629 under others), so a mass check could not be relied on to see it. Every
domain it accepted was bounded (grids, which always carry a BC, and implicit domains, which carry none) or periodic
without being wrapped, so the refusal is unconditional at construction (user ruling 2026-10-09). The rebuild is
#2584.

The alternative is pinned by a property of its result, not by completion: completion alone passed FP-GFDM's own
wall-less solve.
"""

from __future__ import annotations

import pytest

import numpy as np

from mfgarchon import Conditions, MFGProblem, Model
from mfgarchon.alg.numerical import FPFDMSolver, HJBGFDMSolver
from mfgarchon.alg.numerical.fp_solvers import FPGFDMSolver
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import neumann_bc, no_flux_bc
from mfgarchon.geometry.implicit import Hyperrectangle
from mfgarchon.types import NumericalScheme

N = 21
POINTS = np.linspace(0.0, 1.0, N).reshape(-1, 1)
REFUSAL = r"FPGFDMSolver is withdrawn: FP-GFDM built no wall.*#2583.*#2584"


def _problem(domain, u_terminal=lambda x: 0.0 * np.asarray(x, dtype=float)) -> MFGProblem:
    return MFGProblem(
        model=Model(
            hamiltonian=SeparableHamiltonian(control_cost=QuadraticControlCost(control_cost=1.0)), volatility=0.3
        ),
        domain=domain,
        conditions=Conditions(
            m_initial=lambda x: 1.0 + 0.0 * np.asarray(x, dtype=float)[..., 0], u_terminal=u_terminal, T=0.5
        ),
        Nt=10,
    )


def _grid(bc):
    return TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[N], boundary_conditions=bc)


@pytest.mark.parametrize(
    "domain",
    [
        lambda: _grid(no_flux_bc(dimension=1)),
        lambda: _grid(neumann_bc(dimension=1)),
        lambda: Hyperrectangle(np.array([[0.0, 1.0]])),
    ],
    ids=["grid-no_flux", "grid-neumann", "implicit-no_bc"],
)
def test_construction_is_refused_with_the_reason(domain):
    """The two grid cases were #1822's strict-xfail rows while FP-GFDM declared them; the implicit domain
    carries no BC and used to solve wall-less too."""
    with pytest.raises(NotImplementedError, match=REFUSAL):
        FPGFDMSolver(_problem(domain()), collocation_points=POINTS)


@pytest.mark.parametrize("hjb_config", [{"collocation_points": POINTS}, None], ids=["with-points", "no-points"])
def test_the_gfdm_pair_refuses_with_the_reason_before_building_anything(hjb_config):
    """`scheme_factory`'s GFDM pair refuses before it builds the HJB half. Without collocation points, building the
    HJB first raised its own `TypeError` and the reason never showed (review 1 of #2585)."""
    from mfgarchon.factory import create_paired_solvers

    with pytest.raises(NotImplementedError, match=REFUSAL):
        create_paired_solvers(_problem(_grid(no_flux_bc(dimension=1))), NumericalScheme.GFDM, hjb_config=hjb_config)


def test_safe_mode_gfdm_reaches_the_reason():
    """`problem.solve(scheme=GFDM)` has no way to pass collocation points; it raised the HJB's `TypeError` before
    this change, and reaches the withdrawal's reason now that the pair refuses first."""
    with pytest.raises(NotImplementedError, match=REFUSAL):
        _problem(_grid(no_flux_bc(dimension=1))).solve(scheme=NumericalScheme.GFDM, verbose=False)


@pytest.mark.parametrize("route", ["dual scheme", "GFDM HJB with FP-FDM"])
def test_the_alternatives_the_refusal_names_hold_a_wall(route):
    """Both alternatives the refusal names for a grid without obstacles, run on a terminal cost that drives agents
    into the x_max wall: the density piles up against it and the mass stays.

    Measured at this commit: m(T) at x_max over the midpoint is 34.0 (FDM_UPWIND) and 27.2 (GFDM HJB with
    FP-FDM), at mass 1.000000. With FP-FDM's wall rows made transparent (review 1 of #2585) the ratios are 7.31
    and 10.84 at mass 0.848 and 0.863, so on this coupled fixture it is the mass line that separates a wall-less
    FP. The pile-up inequality alone does not.

    The refusal says the second pair is not dual and that Expert Mode warns about it, so that warning is
    asserted too."""
    problem = _problem(_grid(no_flux_bc(dimension=1)), u_terminal=lambda x: -2.0 * np.asarray(x, dtype=float))
    if route == "dual scheme":
        result = problem.solve(scheme=NumericalScheme.FDM_UPWIND, max_iterations=20, verbose=False)
    else:
        with pytest.warns(UserWarning, match=r"DUALITY MISMATCH WARNING"):
            result = problem.solve(
                hjb_solver=HJBGFDMSolver(problem, collocation_points=POINTS),
                fp_solver=FPFDMSolver(problem),
                max_iterations=20,
                verbose=False,
            )
    M = np.asarray(result.M)
    assert M[-1, -1] > 3.0 * M[-1, N // 2] > 3.0 * M[-1, 0]
    weights = np.r_[0.5, np.ones(N - 2), 0.5] / (N - 1)
    assert abs(float(np.sum(M[-1] * weights)) - 1.0) < 1e-6


def test_on_an_implicit_domain_both_alternatives_refuse():
    """The refusal says that neither alternative runs on an implicit domain. This goes red when either starts
    accepting one, which is when that sentence must change."""
    with pytest.raises(ValueError, match="Cannot use HJBFDMSolver with this problem"):
        _problem(Hyperrectangle(np.array([[0.0, 1.0]]))).solve(scheme=NumericalScheme.FDM_UPWIND, verbose=False)
    with pytest.raises(ValueError, match="Cannot use FPFDMSolver with this problem"):
        FPFDMSolver(_problem(Hyperrectangle(np.array([[0.0, 1.0]]))))
