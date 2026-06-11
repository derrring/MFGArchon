"""Pinning test: v0.25.0 removal of deprecated HJBGFDMSolver constructor params.

Issue #1070 — Remove deprecated FixedPointIterator damping_* params and
qp_optimization_level (past the deprecation window).

Cluster B: HJBGFDMSolver three removed params:
  - NiterNewton          → max_newton_iterations
  - l2errBoundNewton     → newton_tolerance
  - qp_optimization_level → monotonicity_scheme + monotonicity_application

PINNING BEHAVIOUR (pre/post fix):
  - PRE-FIX: passing deprecated kwargs emits DeprecationWarning (they are still in
    the signature). These tests fail because they assert TypeError is raised.
  - POST-FIX: deprecated kwargs are no longer in the signature, so Python raises
    TypeError: __init__() got an unexpected keyword argument. Tests pass.

Canonical kwargs (max_newton_iterations, newton_tolerance, monotonicity_scheme,
monotonicity_application) must still work correctly and be byte-identical to
their pre-removal counterparts.
"""

from __future__ import annotations

import pytest

from mfgarchon.alg.numerical.hjb_solvers import HJBGFDMSolver
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.core.mfg_components import MFGComponents
from mfgarchon.core.mfg_problem import MFGProblem
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import no_flux_bc


def _make_problem_and_points(n: int = 11):
    """Minimal 1D MFG problem + collocation points."""
    grid = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[n], boundary_conditions=no_flux_bc(dimension=1))
    H = SeparableHamiltonian(
        control_cost=QuadraticControlCost(control_cost=1.0),
        coupling=lambda m: 0.0 * m,
        coupling_dm=lambda m: 0.0 * m,
    )
    comps = MFGComponents(
        m_initial=lambda x: 1.0,
        u_terminal=lambda x: 0.0,
        hamiltonian=H,
    )
    problem = MFGProblem(geometry=grid, T=0.2, Nt=5, sigma=0.3, components=comps)
    pts = problem.geometry.get_spatial_grid().reshape(-1, 1)
    return problem, pts


class TestRemovedDeprecatedParams:
    """Locked-in v0.25.0 removal: NiterNewton, l2errBoundNewton, qp_optimization_level
    must NOT be accepted by HJBGFDMSolver.__init__ after removal."""

    @pytest.fixture(scope="class")
    def setup(self):
        return _make_problem_and_points()

    def test_NiterNewton_raises_type_error(self, setup):
        """NiterNewton= no longer in signature → TypeError: unexpected kwarg."""
        problem, pts = setup
        with pytest.raises(TypeError, match="NiterNewton"):
            HJBGFDMSolver(problem, pts, NiterNewton=40)

    def test_l2errBoundNewton_raises_type_error(self, setup):
        """l2errBoundNewton= no longer in signature → TypeError: unexpected kwarg."""
        problem, pts = setup
        with pytest.raises(TypeError, match="l2errBoundNewton"):
            HJBGFDMSolver(problem, pts, l2errBoundNewton=1e-5)

    def test_qp_optimization_level_raises_type_error(self, setup):
        """qp_optimization_level= no longer in signature → TypeError: unexpected kwarg."""
        problem, pts = setup
        with pytest.raises(TypeError, match="qp_optimization_level"):
            HJBGFDMSolver(problem, pts, monotonicity_scheme="none", qp_optimization_level="auto")

    def test_qp_optimization_level_standalone_raises_type_error(self, setup):
        """qp_optimization_level= as sole kwarg also raises TypeError."""
        problem, pts = setup
        with pytest.raises(TypeError, match="qp_optimization_level"):
            HJBGFDMSolver(problem, pts, qp_optimization_level="none")


class TestCanonicalParamsStillWork:
    """Canonical replacements for the removed deprecated params must still work
    correctly and produce solvers with the expected internal state."""

    @pytest.fixture(scope="class")
    def setup(self):
        return _make_problem_and_points()

    def test_max_newton_iterations_accepted(self, setup):
        """max_newton_iterations= (canonical) is accepted and stored correctly."""
        problem, pts = setup
        solver = HJBGFDMSolver(problem, pts, max_newton_iterations=40, monotonicity_scheme="none")
        assert solver.max_newton_iterations == 40

    def test_newton_tolerance_accepted(self, setup):
        """newton_tolerance= (canonical) is accepted and stored correctly."""
        problem, pts = setup
        solver = HJBGFDMSolver(problem, pts, newton_tolerance=1e-5, monotonicity_scheme="none")
        assert solver.newton_tolerance == 1e-5

    def test_monotonicity_scheme_none_accepted(self, setup):
        """monotonicity_scheme='none' selects the bare Wendland-Taylor path."""
        problem, pts = setup
        solver = HJBGFDMSolver(problem, pts, monotonicity_scheme="none")
        assert solver.monotonicity_scheme == "none"
        assert solver.qp_optimization_level == "none"  # internal attribute still present

    def test_monotonicity_scheme_qp_m_matrix_adaptive(self, setup):
        """monotonicity_scheme='qp_m_matrix' + application='adaptive'."""
        problem, pts = setup
        solver = HJBGFDMSolver(problem, pts, monotonicity_scheme="qp_m_matrix", monotonicity_application="adaptive")
        assert solver.monotonicity_scheme == "qp_m_matrix"
        assert solver.monotonicity_application == "adaptive"
        assert solver.qp_optimization_level == "auto"  # internal alias

    def test_monotonicity_scheme_qp_m_matrix_always(self, setup):
        """monotonicity_scheme='qp_m_matrix' + application='always'."""
        problem, pts = setup
        solver = HJBGFDMSolver(problem, pts, monotonicity_scheme="qp_m_matrix", monotonicity_application="always")
        assert solver.monotonicity_scheme == "qp_m_matrix"
        assert solver.monotonicity_application == "always"

    def test_legacy_attribute_NiterNewton_gone(self, setup):
        """After removal, solver instance no longer exposes .NiterNewton attribute."""
        problem, pts = setup
        solver = HJBGFDMSolver(problem, pts, max_newton_iterations=30, monotonicity_scheme="none")
        assert not hasattr(solver, "NiterNewton"), (
            "solver.NiterNewton attribute still present; remove self.NiterNewton = ... "
            "assignment from HJBGFDMSolver.__init__"
        )

    def test_legacy_attribute_l2errBoundNewton_gone(self, setup):
        """After removal, solver instance no longer exposes .l2errBoundNewton attribute."""
        problem, pts = setup
        solver = HJBGFDMSolver(problem, pts, newton_tolerance=1e-7, monotonicity_scheme="none")
        assert not hasattr(solver, "l2errBoundNewton"), (
            "solver.l2errBoundNewton attribute still present; remove self.l2errBoundNewton = ... "
            "assignment from HJBGFDMSolver.__init__"
        )
