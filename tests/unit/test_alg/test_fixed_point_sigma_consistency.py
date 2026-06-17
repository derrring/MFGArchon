"""Issue #1082: FixedPointIterator warns on HJB-FP volatility mismatch.

When user passes `volatility_field=X` to FixedPointIterator AND
`problem.sigma=Y` with `X != Y`, HJB sees Y, FP sees X. Picard fixed point
corresponds to neither the original nor a coherent augmented MFG. Same
trap pattern as Issue #811.

This validates that the warning fires for scalar mismatch (the simplest
case) and stays silent for non-scalar / callable / matched cases.
"""

from __future__ import annotations

import warnings

import pytest

from mfgarchon.alg.numerical.coupling import FixedPointIterator
from mfgarchon.alg.numerical.fp_solvers import FPFDMSolver
from mfgarchon.alg.numerical.hjb_solvers import HJBFDMSolver
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.core.mfg_components import MFGComponents
from mfgarchon.core.mfg_problem import MFGProblem
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import no_flux_bc


def _make_problem(sigma=0.3):
    H = SeparableHamiltonian(
        control_cost=QuadraticControlCost(control_cost=1.0),
        coupling=lambda m: m,
        coupling_dm=lambda m: 1.0,
    )
    components = MFGComponents(
        hamiltonian=H,
        u_terminal=lambda x: 0.0,
        m_initial=lambda x: 1.0,
    )
    return MFGProblem(
        geometry=TensorProductGrid(
            bounds=[(0.0, 1.0)], Nx_points=[11 + 1], boundary_conditions=no_flux_bc(dimension=1)
        ),
        T=0.2,
        Nt=5,
        sigma=sigma,
        components=components,
    )


def test_warns_on_scalar_mismatch():
    """Issue #1082: scalar volatility_field != problem.sigma warns."""
    problem = _make_problem(sigma=0.3)
    hjb = HJBFDMSolver(problem)
    fp = FPFDMSolver(problem)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        FixedPointIterator(problem, hjb, fp, volatility_field=0.5)  # mismatch!
        sigma_warns = [x for x in w if "volatility_field" in str(x.message)]

    assert len(sigma_warns) == 1, f"expected 1 mismatch warning, got {len(sigma_warns)}"
    assert "0.5" in str(sigma_warns[0].message)
    assert "0.3" in str(sigma_warns[0].message)


def test_silent_when_matched():
    """No warning when volatility_field equals problem.sigma."""
    problem = _make_problem(sigma=0.3)
    hjb = HJBFDMSolver(problem)
    fp = FPFDMSolver(problem)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        FixedPointIterator(problem, hjb, fp, volatility_field=0.3)
        sigma_warns = [x for x in w if "volatility_field" in str(x.message)]

    assert len(sigma_warns) == 0


def test_silent_when_volatility_field_none():
    """No warning when volatility_field is None (default — uses problem.sigma)."""
    problem = _make_problem(sigma=0.3)
    hjb = HJBFDMSolver(problem)
    fp = FPFDMSolver(problem)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        FixedPointIterator(problem, hjb, fp, volatility_field=None)
        sigma_warns = [x for x in w if "volatility_field" in str(x.message)]

    assert len(sigma_warns) == 0


def test_silent_when_callable():
    """No warning when volatility_field is a callable (intentional override,
    e.g., LLF augmentation). User opting in to non-scalar volatility."""
    problem = _make_problem(sigma=0.3)
    hjb = HJBFDMSolver(problem)
    fp = FPFDMSolver(problem)

    def callable_vol(t, x, m):
        return 0.5

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        FixedPointIterator(problem, hjb, fp, volatility_field=callable_vol)
        sigma_warns = [x for x in w if "volatility_field" in str(x.message)]

    # Callable case: silent (LLF / regularization is the intended use)
    assert len(sigma_warns) == 0


def test_warns_when_newton_tolerance_looser_than_picard():
    """Issue #1081: a Newton tolerance looser than the Picard tolerance is a convergence
    floor — each inner HJB solve injects a ~newton_tolerance residual, so Picard cannot
    drop below it and reports 'max iterations' without converging. Warn at solve()."""
    problem = _make_problem(sigma=0.3)
    hjb = HJBFDMSolver(problem, newton_tolerance=1e-4)
    fp = FPFDMSolver(problem)
    iterator = FixedPointIterator(problem, hjb, fp)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        iterator.solve(max_iterations=1, tolerance=1e-8)
        tol_warns = [x for x in w if "newton_tolerance" in str(x.message)]

    assert len(tol_warns) == 1, f"expected 1 tolerance-floor warning, got {len(tol_warns)}"
    assert "looser" in str(tol_warns[0].message)


def test_silent_when_newton_tolerance_tight_enough():
    """No tolerance-floor warning when newton_tolerance <= picard tolerance (the equal
    1e-6 defaults are fine; a tighter Newton tolerance is also fine)."""
    problem = _make_problem(sigma=0.3)
    hjb = HJBFDMSolver(problem, newton_tolerance=1e-8)
    fp = FPFDMSolver(problem)
    iterator = FixedPointIterator(problem, hjb, fp)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        iterator.solve(max_iterations=1, tolerance=1e-6)
        tol_warns = [x for x in w if "newton_tolerance" in str(x.message)]

    assert len(tol_warns) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
