#!/usr/bin/env python3
"""Issue #1071: the control-cost weight lambda comes from the Hamiltonian single source.

The HJB solvers historically read a scalar placeholder ``problem.lambda_`` (via
``HJBGFDMSolver._get_lambda_value()``) that was NEVER synced with the components
Hamiltonian's ``QuadraticControlCost.lambda_`` -- the desync that powered the #1247
Howard defects. After #1071 the canonical source is ``hamiltonian_class.control_cost``,
read through ``BaseHJBSolver._control_cost_lambda()`` /
``BaseHJBSolver._hamiltonian_control_cost()``.

These tests pin:
1. The fix: with the Hamiltonian carrying lambda=2 and ``problem.lambda_`` left at its
   default placeholder, the solver now derives lambda=2 (from H), not 1 (the placeholder).
2. The equivalence (deprecation policy): for the matched lambda=1 case, old and new agree.
3. The legacy fallback: with NO Hamiltonian class, ``problem.lambda_`` is the sole source.
4. Fail-loud: the strict accessor raises when there is no Hamiltonian class.
"""

import pytest

import numpy as np

from mfgarchon.alg.numerical.hjb_solvers import HJBFDMSolver, HJBGFDMSolver
from mfgarchon.alg.numerical.hjb_solvers.base_hjb import BaseHJBSolver
from mfgarchon.core.derivatives import DerivativeTensors
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.core.mfg_components import MFGComponents
from mfgarchon.core.mfg_problem import MFGProblem
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import no_flux_bc


def _problem_with_lambda(control_lambda: float) -> MFGProblem:
    """1D MFG problem whose Hamiltonian's control cost carries ``control_lambda``.

    ``problem.lambda_`` is deliberately left unset (default), so it stays at the scalar
    placeholder the old ``_get_lambda_value()`` would have read.
    """
    grid = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[21], boundary_conditions=no_flux_bc(dimension=1))
    components = MFGComponents(
        hamiltonian=SeparableHamiltonian(control_cost=QuadraticControlCost(lambda_=control_lambda)),
        m_initial=lambda x: 1.0,
        u_terminal=lambda x: 0.0,
    )
    return MFGProblem(geometry=grid, T=0.2, Nt=10, sigma=0.3, components=components)


def _gfdm_solver(problem: MFGProblem) -> HJBGFDMSolver:
    pts = np.linspace(0.0, 1.0, 21).reshape(-1, 1)
    return HJBGFDMSolver(problem, collocation_points=pts, delta=0.2)


def test_control_cost_lambda_derived_from_hamiltonian_not_placeholder():
    """#1247-root fix: lambda is read from the Hamiltonian, not the problem.lambda_ placeholder."""
    problem = _problem_with_lambda(2.0)

    # The placeholder source the OLD _get_lambda_value() read was getattr(problem, "lambda_", 1.0).
    # It is NOT 2.0 -- this is exactly the desync #1071 removes.
    placeholder = getattr(problem, "lambda_", None)
    assert placeholder in (None, 1.0), f"expected unset/1.0 placeholder, got {placeholder}"

    # The migrated accessor derives lambda from the Hamiltonian single source.
    solver = _gfdm_solver(problem)
    assert solver._control_cost_lambda() == 2.0
    # And the strict accessor exposes the canonical control-cost object.
    assert solver._hamiltonian_control_cost().lambda_ == 2.0


def test_base_helper_is_shared_across_solvers():
    """The helper lives on BaseHJBSolver, so every HJB solver derives the same lambda from H."""
    problem = _problem_with_lambda(2.0)
    fdm = HJBFDMSolver(problem)
    gfdm = _gfdm_solver(problem)
    assert fdm._control_cost_lambda() == 2.0
    assert gfdm._control_cost_lambda() == 2.0


def test_matched_case_is_equivalent_to_placeholder():
    """Deprecation equivalence: for the matched lambda=1 case, H-derived == old placeholder."""
    problem = _problem_with_lambda(1.0)
    solver = _gfdm_solver(problem)
    # Old path: getattr(problem, "lambda_", 1.0) -> 1.0 (None -> 1.0). New path: H.control_cost -> 1.0.
    old_placeholder = getattr(problem, "lambda_", 1.0)
    if old_placeholder is None:
        old_placeholder = 1.0
    assert solver._control_cost_lambda() == old_placeholder == 1.0


class _NoHamProblem:
    """Stand-in for the legacy LQ fast path: a problem with no Hamiltonian class.

    A real MFGProblem always carries a Hamiltonian (Issue #670 requires components with a
    hamiltonian for u_terminal), so the no-Hamiltonian branch is exercised with a stand-in
    -- the same pattern the Howard suite uses (_MockProblem with hamiltonian_class=None).
    """

    hamiltonian_class = None

    def __init__(self, lambda_: float | None):
        self.lambda_ = lambda_


class _Solver:
    def __init__(self, problem):
        self.problem = problem


def test_legacy_fallback_without_hamiltonian_class():
    """No Hamiltonian class -> problem.lambda_ is the sole source (legacy LQ fast path)."""
    solver = _Solver(_NoHamProblem(lambda_=3.0))
    assert BaseHJBSolver._control_cost_lambda(solver) == 3.0
    # Unset lambda_ falls back to 1.0 (matches the old _get_lambda_value default).
    assert BaseHJBSolver._control_cost_lambda(_Solver(_NoHamProblem(lambda_=None))) == 1.0


def test_strict_accessor_fails_loud_without_hamiltonian_class():
    """The strict accessor raises -- a solver must not invent a control cost (Issue #1071)."""
    solver = _Solver(_NoHamProblem(lambda_=3.0))
    with pytest.raises(ValueError, match="hamiltonian_class is None"):
        BaseHJBSolver._hamiltonian_control_cost(solver)


def test_nonpositive_lambda_rejected():
    """A non-positive control cost is rejected (division by lambda requires lambda > 0)."""
    with pytest.raises(ValueError, match="must be positive"):
        QuadraticControlCost(lambda_=-1.0)


# --- #1071 Phase 4: GFDM per-point FD-Jacobian fast path sources lambda from the
# --- Hamiltonian single source, not a direct problem.lambda_ read (hjb_gfdm.py:_compute_dH_dp_fd).


class _ControlCost:
    def __init__(self, lambda_: float):
        self.lambda_ = lambda_


class _Ham:
    def __init__(self, lambda_: float):
        self.control_cost = _ControlCost(lambda_)


class _FastPathProblem:
    """``is_custom=False`` (so the GFDM LQ fast path runs) but the Hamiltonian's control
    cost (canonical lambda) disagrees with the ``problem.lambda_`` placeholder.

    This divergent state is NOT reachable through the public API -- ``is_custom=False``
    implies the problem carries no Hamiltonian class, so the two always agree. It is
    constructed here purely to PIN that the fast path reads lambda from the canonical
    single source, discriminating the single-source code from the pre-fix direct read.
    """

    is_custom = False
    dimension = 1

    def __init__(self, placeholder_lambda: float, canonical_lambda: float):
        self.lambda_ = placeholder_lambda
        self.hamiltonian_class = _Ham(canonical_lambda)


class _FastPathSolver:
    """Minimal stand-in carrying the inherited ``_control_cost_lambda`` for unbound-call
    invocation of ``HJBGFDMSolver._compute_dH_dp_fd``."""

    _control_cost_lambda = BaseHJBSolver._control_cost_lambda

    def __init__(self, problem):
        self.problem = problem


def test_fd_jacobian_fast_path_sources_lambda_from_hamiltonian_not_placeholder():
    """#1071 Phase 4 (discriminating): the GFDM per-point FD-Jacobian fast path derives
    lambda from the Hamiltonian single source, not ``problem.lambda_``.

    Pins against reverting ``hjb_gfdm.py:_compute_dH_dp_fd`` to a direct
    ``getattr(self.problem, "lambda_")`` read: canonical lambda=2 must give dH/dp = p/2,
    whereas the pre-fix placeholder read (lambda=1) gave p/1.
    """
    solver = _FastPathSolver(_FastPathProblem(placeholder_lambda=1.0, canonical_lambda=2.0))
    derivs = DerivativeTensors.from_arrays(grad=np.array([0.6]), hess=np.zeros((1, 1)))

    out = HJBGFDMSolver._compute_dH_dp_fd(solver, point_idx=0, m_at_x=1.0, derivs=derivs)

    np.testing.assert_allclose(out, np.array([0.3]))  # p / canonical lambda = 0.6 / 2
    assert not np.allclose(out, np.array([0.6])), (
        "fast path read the problem.lambda_ placeholder (p/1) instead of control_cost.lambda_ (p/2)"
    )


def test_fd_jacobian_fast_path_byte_identical_on_reachable_legacy_path():
    """#1071 Phase 4 (equivalence): on the only construction that actually reaches the
    fast path -- no Hamiltonian class, ``problem.lambda_`` the sole source -- the
    single-source routing reproduces the old direct-read result exactly (p / lambda_)."""
    solver = _FastPathSolver(_NoHamProblem(lambda_=2.0))
    derivs = DerivativeTensors.from_arrays(grad=np.array([0.6]), hess=np.zeros((1, 1)))

    out = HJBGFDMSolver._compute_dH_dp_fd(solver, point_idx=0, m_at_x=1.0, derivs=derivs)

    np.testing.assert_allclose(out, np.array([0.3]))  # p / problem.lambda_ = 0.6 / 2
