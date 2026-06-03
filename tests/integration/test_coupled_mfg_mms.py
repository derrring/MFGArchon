#!/usr/bin/env python3
"""
Coupled MFG (HJB-FP) Method-of-Manufactured-Solutions (MMS) EOC validation.

This is the missing audit-item-C test. The existing coupled tests
(``TestCoupledHJBFPValidation`` in ``test_mms_validation.py``) check only:

  1. Picard self-consistency residual (-> 0 for ANY fixed point), and
  2. mass conservation (~1 for ANY mass-conserving scheme).

Neither verifies that the converged ``(u_h, m_h)`` is the *correct* solution of
the coupled system. A wrong-but-self-consistent discretization passes both:

  - the sigma->D factor bug (#1152) inflated/deflated the diffusion coefficient
    but still converged to a self-consistent fixed point and conserved mass, and
  - the no-flux wall-leak bug (#1151) routed the FP flux through a
    non-conservative gradient form yet still produced a smooth fixed point.

Both shipped because no test compared the converged fields to a *known* exact
coupled solution. This test does exactly that, via MMS:

  * manufacture a smooth, periodic, genuinely time-dependent pair (u*, m*),
  * use an ACTIVE bidirectional coupling (HJB sees m* through f(m); FP sees
    grad u* through the drift v = -c * grad U), and
  * inject the analytic source terms S_HJB, S_FP that make (u*, m*) the EXACT
    solution of the source-augmented coupled system, then
  * run the real FixedPointIterator (the production solve path) at a sequence
    of grid sizes and assert the empirical convergence order (EOC) for BOTH
    u and m.

WHAT THIS CATCHES THAT SELF-CONSISTENCY TESTS CANNOT
----------------------------------------------------
A factor error in the diffusion coefficient (the sigma->D bug class, #1152),
a wrong coupling sign, a wrong drift coefficient, or a non-conservative FP flux
(#1151) all break the *rate of convergence to the exact pair*. With a wrong
discretization the error stops decreasing (or decreases at the wrong order) as
the grid refines, so the EOC assertion fails even though Picard still converges
to a (wrong) self-consistent fixed point and mass is still conserved.

VERIFIED CONVENTIONS (working tree, not memory)
-----------------------------------------------
HJB residual (source SUBTRACTED): base_hjb.py:661 / :705 ``Phi_U -= source_term``;
  effective continuous equation ``-d_t u + H(x, m, grad u) - (sigma^2/2) Lap u = S_HJB``.
  => S_HJB = -d_t u* + H(x, m*, grad u*) - (sigma^2/2) Lap u*  (the continuous LHS).
H = H_control(p) + V + f(m), coupling ADDED: hamiltonian.py:2144; with
  QuadraticControlCost, H_control(p) = |p|^2/(2*lambda) (hamiltonian.py:365).
FP RHS (source ADDED): fp_fdm_time_stepping.py:513 (explicit) and :1184 (implicit
  MFG-coupled, the path FPFDMSolver uses); effective continuous equation
  ``d_t m + div(alpha* m) - (sigma^2/2) Lap m = S_FP``.
  => S_FP = d_t m* + div(alpha* m*) - (sigma^2/2) Lap m*.
FP drift: alpha = -coupling_coefficient * grad(U) (fp_fdm_alg_divergence_upwind.py:158,178),
  coupling_coefficient default 0.5 (mfg_problem.py:261). It is an INDEPENDENT knob
  from lambda; we set coupling_coefficient = 1/lambda = 1.0 so the drift the solver
  builds agrees with -grad u* / lambda regardless of which drift path is selected
  (the SeparableHamiltonian + smooth control branch passes potential_field=U_new and
  the FP solver forms v = -coupling_coefficient * grad U: fixed_point_iterator.py:755-763).
sigma vs D: D = sigma^2/2 (mfg_problem.py:36, fp_fdm_time_stepping.py:475). Pass sigma
  via sigma=; the (2*pi^2*sigma^2) coefficients below already encode (sigma^2/2)*k^2.

FALSE-SAFETY GUARDS encoded here
--------------------------------
* S_HJB and S_FP depend ONLY on (x, t); they ignore the (m, v) arguments. This is
  mandatory: FixedPointIterator hardcodes v = zeros for the HJB source
  (fixed_point_iterator.py:263), so a v-dependent S_HJB would silently be wrong.
* The coupling is ACTIVE: c_f > 0 so the HJB residual genuinely contains f(m_current)
  (cancelled by the +c_f*m* term in S_HJB at the fixed point), and the FP drift
  genuinely contains grad U (cross term in S_FP). With c_f = 0 the test would
  degenerate into two decoupled MMS and could not catch a cross-coupling bug.
* periodic BC keeps boundaries exact (sin/cos manufactured pair), avoiding the
  no-flux conservative-Laplacian boundary handling (#1075) so the measured error
  is purely interior discretization error.

Manufactured pair (k = 2*pi, domain [0,1], periodic):
    u*(t,x) = b * exp(-t) * sin(k x)          (value function; sign-indefinite OK)
    m*(t,x) = 1 + a * exp(-t) * cos(k x)      (density; m* in [1-a, 1+a] > 0)
    f(m)    = c_f * m                          (active linear congestion)
"""

from __future__ import annotations

from typing import ClassVar

import pytest

import numpy as np

from mfgarchon.alg.numerical.coupling import FixedPointIterator
from mfgarchon.alg.numerical.fp_solvers import FPFDMSolver
from mfgarchon.alg.numerical.hjb_solvers import HJBFDMSolver
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.core.mfg_components import MFGComponents
from mfgarchon.core.mfg_problem import MFGProblem
from mfgarchon.geometry import TensorProductGrid, periodic_bc

# Reuse the existing MMS base. pytest's default (prepend) import mode puts this
# test's directory on sys.path, so the sibling module is importable by bare name;
# the package-qualified path is kept as a fallback for other runners.
try:
    from test_mms_validation import ManufacturedSolution
except ModuleNotFoundError:  # pragma: no cover - runner-dependent
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from test_mms_validation import ManufacturedSolution

K = 2.0 * np.pi


class CoupledSinusoid1D(ManufacturedSolution):
    """
    Coupled-MFG manufactured solution with active bidirectional coupling.

    u*(t,x) = b e^{-t} sin(k x)
    m*(t,x) = 1 + a e^{-t} cos(k x)
    f(m)    = c_f m,    H(x,m,p) = |p|^2/(2 lambda) + f(m)

    Drift used by the FP solver: alpha* = -c * grad u*,  c = coupling_coefficient.
    We set c = 1/lambda so alpha* = -grad u*/lambda = optimal_control(grad u*),
    making the drift channel robust to the iterator's drift-path selection.

    Source terms (continuous LHS of each equation evaluated on the exact pair):

      S_HJB = -d_t u* + |d_x u*|^2/(2 lambda) + c_f m* - (sigma^2/2) d_xx u*
      S_FP  =  d_t m* + d_x(alpha* m*)        - (sigma^2/2) d_xx m*

    with alpha* = -c d_x u*. Both depend only on (t, x).
    """

    def __init__(
        self,
        sigma: float = 0.25,
        a: float = 0.2,
        b: float = 0.15,
        c_f: float = 0.3,
        lambda_: float = 1.0,
        coupling_coefficient: float = 1.0,
        k: float = K,
    ):
        super().__init__(dimension=1)
        self.sigma = sigma
        self.a = a
        self.b = b
        self.c_f = c_f
        self.lam = lambda_
        self.c = coupling_coefficient  # FP drift coefficient
        self.k = k
        self.D = 0.5 * sigma**2

    # --- exact fields -----------------------------------------------------
    # Shape-preserving (scalar -> scalar, array -> array) so they can serve both
    # as MFGComponents IC/TC callables (invoked per-point with a scalar x_i,
    # mfg_components.py:797-799) and as grid evaluators in the error metric.
    def u_star(self, t: float, x):
        x = np.asarray(x, dtype=float)
        return self.b * np.exp(-t) * np.sin(self.k * x)

    def m_star(self, t: float, x):
        x = np.asarray(x, dtype=float)
        return 1.0 + self.a * np.exp(-t) * np.cos(self.k * x)

    # ManufacturedSolution.solution dispatches to m* by convention (density side).
    def solution(self, t: float, x: np.ndarray) -> np.ndarray:
        return self.m_star(t, x)

    # --- source terms (signature the iterator expects: (x, m, v, t)) ------
    def hjb_source(self, x: np.ndarray, m, v, t: float) -> np.ndarray:
        """S_HJB(t,x). Ignores m, v (FixedPointIterator passes v=zeros)."""
        x = np.atleast_1d(x).ravel()
        k, a, b, c_f, lam, sigma = self.k, self.a, self.b, self.c_f, self.lam, self.sigma
        e1 = np.exp(-t)
        e2 = np.exp(-2.0 * t)
        sin = np.sin(k * x)
        cos = np.cos(k * x)
        # -d_t u* = +b e^{-t} sin(kx)
        term_dt = b * e1 * sin
        # |d_x u*|^2/(2 lambda) = (k b e^{-t} cos)^2 / (2 lam) = (k^2 b^2 / (2 lam)) e^{-2t} cos^2
        term_ctrl = (k**2 * b**2 / (2.0 * lam)) * e2 * cos**2
        # +f(m*) = c_f (1 + a e^{-t} cos)
        term_coupling = c_f * (1.0 + a * e1 * cos)
        # -(sigma^2/2) d_xx u* = -(sigma^2/2)(-k^2 b e^{-t} sin) = (sigma^2/2) k^2 b e^{-t} sin
        term_diff = 0.5 * sigma**2 * k**2 * b * e1 * sin
        return term_dt + term_ctrl + term_coupling + term_diff

    def fp_source(self, x: np.ndarray, m, v, t: float) -> np.ndarray:
        """S_FP(t,x). Ignores m, v; uses analytic alpha* = -c d_x u*."""
        x = np.atleast_1d(x).ravel()
        k, a, b, c, sigma = self.k, self.a, self.b, self.c, self.sigma
        e1 = np.exp(-t)
        e2 = np.exp(-2.0 * t)
        sin = np.sin(k * x)
        cos = np.cos(k * x)
        # d_t m* = -a e^{-t} cos
        term_dt = -a * e1 * cos
        # div(alpha* m*) where alpha* = -c d_x u* = -c k b e^{-t} cos:
        #   d_x(alpha* m*) = (d_x alpha*) m* + alpha* (d_x m*)
        #   d_x alpha* = +c k^2 b e^{-t} sin ; d_x m* = -a k e^{-t} sin
        # sum = c k^2 b e^{-t} sin + 2 c k^2 a b e^{-2t} sin cos
        term_adv = c * k**2 * b * e1 * sin + 2.0 * c * k**2 * a * b * e2 * sin * cos
        # -(sigma^2/2) d_xx m* = (sigma^2/2) k^2 a e^{-t} cos
        term_diff = 0.5 * sigma**2 * k**2 * a * e1 * cos
        return term_dt + term_adv + term_diff


def _build_problem(mfg: CoupledSinusoid1D, Nx: int, Nt: int, T: float) -> MFGProblem:
    bc = periodic_bc(dimension=1)
    geometry = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[Nx], boundary_conditions=bc)
    components = MFGComponents(
        m_initial=lambda x: mfg.m_star(0.0, x),
        u_terminal=lambda x: mfg.u_star(T, x),
        hamiltonian=SeparableHamiltonian(
            control_cost=QuadraticControlCost(control_cost=mfg.lam),
            coupling=lambda m: mfg.c_f * m,
            coupling_dm=lambda m: mfg.c_f,
        ),
    )
    return MFGProblem(
        geometry=geometry,
        T=T,
        Nt=Nt,
        sigma=mfg.sigma,
        coupling_coefficient=mfg.c,  # = 1/lambda; aligns FP drift with -grad u*/lambda
        components=components,
        source_term_hjb=mfg.hjb_source,
        source_term_fp=mfg.fp_source,
    )


def _solve_coupled(mfg: CoupledSinusoid1D, Nx: int, Nt: int, T: float):
    problem = _build_problem(mfg, Nx, Nt, T)
    hjb_solver = HJBFDMSolver(problem)
    fp_solver = FPFDMSolver(problem)
    # relaxation=1.0 (undamped Picard): empirically converges in ~14-19 outer
    # iterations for the parameters used here. relaxation=0.5/0.8 reach the SAME
    # fixed point but take far more iterations (>100), making the test
    # impractically slow; the converged (u_h, m_h) is relaxation-independent
    # (verified relax in {0.8,1.0} give byte-identical eu/em), so 1.0 is correct.
    iterator = FixedPointIterator(
        problem,
        hjb_solver=hjb_solver,
        fp_solver=fp_solver,
        relaxation=1.0,
    )
    # Converge Picard hard (tol = the inner HJB Newton floor, 1e-6, with ample
    # max_iterations) so the OUTER residual floor does not mask the discretization
    # error we are trying to measure -- the whole point of an MMS-vs-exact test
    # versus a self-consistency test. 1e-6 is ~4 orders below the spatial
    # discretization error (~1e-2), so the discretization error dominates, and it
    # matches the Newton floor so the outer/inner tolerance warning does not fire.
    result = iterator.solve(max_iterations=200, tolerance=1e-6, verbose=False)
    assert result.converged, (
        f"Picard did not converge at Nx={Nx} (iters={result.iterations}); "
        "EOC measurement requires a converged outer iteration."
    )
    x_grid = problem.geometry.coordinates[0]
    return result.U, result.M, x_grid


def _eoc(errors: np.ndarray) -> np.ndarray:
    errors = np.asarray(errors)
    ratios = errors[:-1] / errors[1:]
    return ratios


@pytest.mark.integration
class TestCoupledMMSConvergence:
    """
    Coupled HJB-FP MMS EOC test (audit item C).

    Parameters (validated, not guessed): a=0.2, b=0.15, c_f=0.3, sigma=0.25,
    lambda=1.0, coupling_coefficient=1.0 (=1/lambda), T=0.2. These keep the
    advective drift modest relative to diffusion so the undamped Picard converges
    in ~14-19 iterations and the FP density stays well-behaved, while still
    exercising an ACTIVE bidirectional coupling (c_f>0 and a non-zero grad-u
    drift cross term).

    Grid sequence: Nx in [31, 61]. A 121 point would push each coupled solve to
    ~3 min (the solve is O(Nt * Nx * Newton-per-step) and Nt=4*Nx), so the test
    uses the affordable [31, 61] pair. Time refinement: Nt = 4*Nx.

    EMPIRICALLY MEASURED (this exact configuration, verified before committing
    the threshold):
        u: errors [2.873e-2, 1.625e-2] -> ratio 1.768 (order 0.822)
        m: errors [2.585e-1, 1.469e-1] -> ratio 1.760 (order 0.816)
    Both Picard iterations converged (14-19 outer iterations).

    Threshold: ratios > 1.5 for BOTH u and m -- the precedent set by the
    single-equation source MMS tests (test_mms_validation.py:399 and :796). With
    the measured 1.76 it leaves ~17% margin. We do NOT assert the naive order-1
    ratio of 2 (coarse pre-asymptotic regime + upwind numerical diffusion) nor
    2nd order (the upwind drift forbids it; O(h^2) would false-fail).

    Decoupled cross-check (recorded for reviewers): feeding each solver the EXACT
    other field reproduces the same ~order-1 rate at the FULL stiffness
    a=0.4/b=0.3/sigma=0.3 with FP minM exactly 1-a=0.6 (no clipping), confirming
    both source terms reproduce the exact continuous solution; the coupled
    high-stiffness undershoot is a Picard transient, not a source-term error.
    """

    RESOLUTIONS: ClassVar[list[int]] = [31, 61]
    NT_FACTOR = 4
    RATIO_THRESHOLD = 1.5

    def _run_sweep(self):
        mfg = CoupledSinusoid1D()  # validated defaults
        T = 0.2
        err_u, err_m = [], []
        for Nx in self.RESOLUTIONS:
            Nt = self.NT_FACTOR * Nx
            U, M, x = _solve_coupled(mfg, Nx, Nt, T)
            # Sanity: density stays positive; clipping to 0 would contaminate EOC.
            assert np.all(M > 0.0), f"Nx={Nx}: non-positive density min(M)={np.min(M):.3e}"
            u_exact_0 = mfg.u_star(0.0, x)
            m_exact_T = mfg.m_star(T, x)
            eu = np.sqrt(np.mean((U[0, :] - u_exact_0) ** 2))
            em = np.sqrt(np.mean((M[-1, :] - m_exact_T) ** 2))
            err_u.append(eu)
            err_m.append(em)
        return np.array(err_u), np.array(err_m), mfg

    @pytest.mark.slow
    def test_coupled_mms_eoc_u_and_m(self):
        err_u, err_m, _ = self._run_sweep()
        ratios_u = _eoc(err_u)
        ratios_m = _eoc(err_m)
        msg = (
            f"\nResolutions: {self.RESOLUTIONS}"
            f"\n u errors: {err_u}, ratios: {ratios_u}"
            f"\n m errors: {err_m}, ratios: {ratios_m}"
        )
        # A sigma->D factor error (#1152), wrong coupling sign/coefficient, or a
        # non-conservative FP flux (#1151) breaks one or both rates while leaving
        # Picard self-consistency + mass conservation intact (the existing tests).
        assert np.all(ratios_u > self.RATIO_THRESHOLD), "HJB (u) EOC too low." + msg
        assert np.all(ratios_m > self.RATIO_THRESHOLD), "FP (m) EOC too low." + msg

    def test_coupling_is_active(self):
        """
        Guard against silent decoupling: confirm the coupling channel actually
        perturbs the manufactured source. Fast, algebra-only (no solve).
        """
        mfg = CoupledSinusoid1D(c_f=0.3)
        mfg0 = CoupledSinusoid1D(c_f=0.0)
        x = np.linspace(0.0, 1.0, 41)
        diff = mfg.hjb_source(x, None, None, 0.1) - mfg0.hjb_source(x, None, None, 0.1)
        expected = mfg.c_f * mfg.m_star(0.1, x)
        assert np.allclose(diff, expected), "Coupling term missing from S_HJB"
        assert mfg.c > 0.0
        assert np.any(np.abs(diff) > 0.0)


if __name__ == "__main__":
    t = TestCoupledMMSConvergence()
    eu, em, mfg = t._run_sweep()
    ru, rm = _eoc(eu), _eoc(em)
    print(f"Resolutions: {t.RESOLUTIONS}")
    print(f"u  errors : {eu}  ratios: {ru}")
    print(f"m  errors : {em}  ratios: {rm}")
    assert np.all(ru > t.RATIO_THRESHOLD), "u EOC below threshold"
    assert np.all(rm > t.RATIO_THRESHOLD), "m EOC below threshold"
    print("Coupled MMS EOC passed.")
