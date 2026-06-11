"""
Pinning test for Issue #1285 (warm-start NameError sub-bug).

FixedPointIterator.solve() and FictitiousPlayIterator.solve() defined
M_initial / U_terminal only inside the cold-start else-branch.  On the
warm-start path both names were referenced but never defined -> NameError.

The fix hoists _get_initial_and_terminal_conditions() unconditionally before
the warm/cold branch (mirrors BlockIterator, which was already correct).
"""

from __future__ import annotations

import numpy as np

from mfgarchon.alg.numerical.coupling import FictitiousPlayIterator, FixedPointIterator
from mfgarchon.alg.numerical.fp_solvers import FPFDMSolver
from mfgarchon.alg.numerical.hjb_solvers import HJBFDMSolver
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.core.mfg_components import MFGComponents
from mfgarchon.core.mfg_problem import MFGProblem
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import no_flux_bc


def _make_problem() -> MFGProblem:
    """Minimal 1D MFG problem with Gaussian initial density."""
    geometry = TensorProductGrid(
        bounds=[(0.0, 1.0)],
        Nx_points=[11],
        boundary_conditions=no_flux_bc(dimension=1),
    )
    components = MFGComponents(
        m_initial=lambda x: np.exp(-10 * (x - 0.5) ** 2),
        u_terminal=lambda x: 0.0,
        hamiltonian=SeparableHamiltonian(
            control_cost=QuadraticControlCost(control_cost=1.0),
            coupling=lambda m: m,
            coupling_dm=lambda m: 1.0,
        ),
    )
    return MFGProblem(
        geometry=geometry,
        T=0.2,
        Nt=4,
        sigma=0.2,
        components=components,
    )


def _warm_arrays(problem: MFGProblem) -> tuple[np.ndarray, np.ndarray]:
    """Return trivial warm-start arrays of the right shape."""
    shape = tuple(problem.geometry.get_grid_shape())
    Nt1 = problem.Nt + 1
    U_warm = np.zeros((Nt1, *shape))
    M_warm = np.ones((Nt1, *shape)) / np.prod(shape)
    return U_warm, M_warm


class TestWarmStartNameError1285:
    """Issue #1285: warm-start path must not raise NameError."""

    def test_fixed_point_iterator_warm_start(self):
        """FixedPointIterator.solve() must complete without NameError on warm path."""
        problem = _make_problem()
        hjb = HJBFDMSolver(problem)
        fp = FPFDMSolver(problem)

        solver = FixedPointIterator(problem, hjb, fp, relaxation=0.5)
        U_warm, M_warm = _warm_arrays(problem)
        solver.set_warm_start_data(U_warm, M_warm)

        # Must not raise NameError; 2 iterations is sufficient to exercise the path.
        result = solver.solve(max_iterations=2, tolerance=1e-10)
        assert result.U is not None
        assert result.M is not None
        assert result.U.shape[0] == problem.Nt + 1
        assert result.M.shape[0] == problem.Nt + 1
        assert np.all(np.isfinite(result.U))
        assert np.all(np.isfinite(result.M))

    def test_fictitious_play_iterator_warm_start(self):
        """FictitiousPlayIterator.solve() must complete without NameError on warm path."""
        problem = _make_problem()
        hjb = HJBFDMSolver(problem)
        fp = FPFDMSolver(problem)

        solver = FictitiousPlayIterator(problem, hjb, fp, learning_rate_schedule="harmonic")
        U_warm, M_warm = _warm_arrays(problem)
        solver.set_warm_start_data(U_warm, M_warm)

        # Must not raise NameError; 2 iterations is sufficient to exercise the path.
        result = solver.solve(max_iterations=2, tolerance=1e-10)
        assert result.U is not None
        assert result.M is not None
        assert result.U.shape[0] == problem.Nt + 1
        assert result.M.shape[0] == problem.Nt + 1
        assert np.all(np.isfinite(result.U))
        assert np.all(np.isfinite(result.M))

    def test_cold_start_still_works_fixed_point(self):
        """Cold-start path must be unaffected by the hoist."""
        problem = _make_problem()
        hjb = HJBFDMSolver(problem)
        fp = FPFDMSolver(problem)

        solver = FixedPointIterator(problem, hjb, fp, relaxation=0.5)
        result = solver.solve(max_iterations=3, tolerance=1e-10)
        assert result.U is not None
        assert np.all(np.isfinite(result.U))

    def test_cold_start_still_works_fictitious_play(self):
        """Cold-start path must be unaffected by the hoist."""
        problem = _make_problem()
        hjb = HJBFDMSolver(problem)
        fp = FPFDMSolver(problem)

        solver = FictitiousPlayIterator(problem, hjb, fp)
        result = solver.solve(max_iterations=3, tolerance=1e-10)
        assert result.U is not None
        assert np.all(np.isfinite(result.U))
