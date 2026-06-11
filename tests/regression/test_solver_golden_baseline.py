"""Golden-master byte-identity baseline for the coupled MFG solve.

Purpose: a regression gate for refactors that re-route the sigma / lambda / optimal-control
derivation through the Hamiltonian single source (Issue #1071) — and for any other change
intended to be numerically transparent on the paper paths. The coupled solve is bit-reproducible
run-to-run on a given machine (verified, max|d|=0.0), so a #1071-class refactor that is
byte-identical for lambda=1 reproduces the baseline exactly. The gate asserts to ``atol=1e-9``
(rtol=0) — tight enough that any real drift fails, with a 1e-9 floor only to absorb cross-platform
last-bit BLAS differences between the fixture's origin (macOS) and CI (Linux).

The gate is sensitive to the lambda / alpha* derivation #1071 touches: solving the SAME problem
with control_cost lambda=1 vs lambda=2 gives ``max|dU| ~ 4e-4`` (>> 0), so a silent lambda/alpha
desync of the kind that powered the #1247 Howard defects would be caught here.

If the baseline legitimately changes (e.g. an intentional algorithm improvement), regenerate the
fixture with ``python tests/regression/test_solver_golden_baseline.py`` and review the diff.

NOTE: ``MFGProblem.solve()`` currently routes through the HJB-FDM + FP-FDM path. When #1071 (or
other work) migrates the GFDM path (the largest sigma/lambda read surface, 29 sites in hjb_gfdm),
ADD a GFDM golden here (a collocation-based HJBGFDMSolver + FPFDMSolver baseline).
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np

from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.core.mfg_components import MFGComponents
from mfgarchon.core.mfg_problem import MFGProblem
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import no_flux_bc

_FIXTURE = Path(__file__).parent / "fixtures" / "solver_golden_lq_fdm.npz"


def _make_lq_problem(control_cost: float = 1.0) -> MFGProblem:
    """Deterministic small 1D LQ MFG (the paper-path FDM convention, lambda configurable)."""
    grid = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[21], boundary_conditions=no_flux_bc(dimension=1))
    components = MFGComponents(
        hamiltonian=SeparableHamiltonian(
            control_cost=QuadraticControlCost(control_cost=control_cost),
            coupling=lambda m: 0.1 * m,
            coupling_dm=lambda m: 0.1,
        ),
        m_initial=lambda x: np.exp(-30 * (np.atleast_1d(x)[0] - 0.5) ** 2),
        u_terminal=lambda x: 0.0,
    )
    return MFGProblem(geometry=grid, T=0.2, Nt=10, sigma=0.3, components=components)


def _solve(problem: MFGProblem):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return problem.solve(max_iterations=3, tolerance=1e-12, verbose=False)


def test_coupled_solve_matches_golden_baseline():
    """The lambda=1 LQ coupled solve is byte-identical to the committed golden baseline."""
    res = _solve(_make_lq_problem(1.0))
    U = np.asarray(res.U)
    M = np.asarray(res.M)

    ref = np.load(_FIXTURE)
    # rtol=0, atol=1e-9: the solve is bit-reproducible on a given machine, so a #1071-class
    # refactor that is byte-identical for lambda=1 reproduces this exactly. The 1e-9 floor only
    # absorbs cross-platform last-bit BLAS/LAPACK differences (the fixture is generated on macOS,
    # CI runs on Linux); it is still 6 orders below the ~4e-4 lambda/alpha drift the gate exists to
    # catch (see test_baseline_is_sensitive_to_control_cost).
    np.testing.assert_allclose(U, ref["U"], rtol=0, atol=1e-9, err_msg="HJB U drifted from the golden baseline")
    np.testing.assert_allclose(M, ref["M"], rtol=0, atol=1e-9, err_msg="FP M drifted from the golden baseline")


def test_baseline_is_sensitive_to_control_cost():
    """Sanity: the gate actually responds to the lambda/alpha derivation it is meant to protect.

    If lambda=1 and lambda=2 gave the same U, the golden test above would not catch a #1071
    lambda-desync regression. They must differ.
    """
    u1 = np.asarray(_solve(_make_lq_problem(1.0)).U)
    u2 = np.asarray(_solve(_make_lq_problem(2.0)).U)
    assert np.max(np.abs(u1 - u2)) > 1e-6, "gate insensitive to control_cost — golden test is vacuous"


if __name__ == "__main__":
    # Regenerate the golden fixture (run after an INTENTIONAL, reviewed baseline change).
    _FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    res = _solve(_make_lq_problem(1.0))
    np.savez(_FIXTURE, U=np.asarray(res.U), M=np.asarray(res.M))
    print(f"regenerated {_FIXTURE}")
