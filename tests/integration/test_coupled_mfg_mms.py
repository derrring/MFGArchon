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
HJB residual (source SUBTRACTED): ``Phi_U -= source_term`` in base_hjb.py, both the batch
  and per-point paths; effective continuous equation
  ``-d_t u + H(x, m, grad u) - (sigma^2/2) Lap u = S_HJB``.
  => S_HJB = -d_t u* + H(x, m*, grad u*) - (sigma^2/2) Lap u*  (the continuous LHS).
H = H_control(p) + V + f(m), coupling ADDED (hamiltonian.py); with QuadraticControlCost,
  H_control(p) = |p|^2/(2*lambda).
FP RHS (source ADDED): fp_fdm_time_stepping.py, both the explicit and the implicit
  MFG-coupled path FPFDMSolver uses; effective continuous equation
  ``d_t m + div(alpha* m) - (sigma^2/2) Lap m = S_FP``.
  => S_FP = d_t m* + div(alpha* m*) - (sigma^2/2) Lap m*.
FP drift: alpha* = H.optimal_control(grad u*) = -grad u*/lambda for this quadratic-MINIMIZE
  cost, and the source is assembled from that owner rather than from a hand-written scale.
  ~~It is an INDEPENDENT knob from lambda; we set coupling_coefficient = 1/lambda = 1.0 so the
  drift the solver builds agrees with -grad u*/lambda.~~ [SUPERSEDED 2026-08-31]
  SUPERSEDED-BY: #2201. `coupling_coefficient` is INERT on THIS PROBLEM'S SOLVER PATH, so the
  agreement was never contingent on setting it. The scope is the claim: a quadratic-MINIMIZE
  SeparableHamiltonian solved through the FDM FP/HJB families, which resolve the drift through
  `fp_drift_coefficient(problem)` -- it returns 1/control_cost.lambda_ for such a Hamiltonian and
  never reaches the `coupling_coefficient` fallback (#1420 / G-017). It is NOT a package-wide
  universal, and an earlier draft of this block claimed one: an AST census finds 10 call sites in
  7 files, and the velocity-channel FP families (FVM / FEM / meshless-Galerkin FP, and the
  network solvers) resolve the drift through `H.optimal_control` and call that helper ZERO times.
  The FP scope word is load-bearing: `meshless_galerkin/hjb_solver.py:118` IS one of those 10
  call sites, so dropping it makes the sentence contradict its own census --
  `utils/pde_coefficients.py:47-50` already says so. Outside the scope above the fallback is live:
  a non-separable Hamiltonian returns `coupling_coefficient` itself.
  Measured at the SOLVE, which is the level the claim is about: at lambda = 1.0, a full coupled
  solve at Nx=21/Nt=40 is bit-identical for `coupling_coefficient` = 1.0 / 7.0 / 0.5 / -3.0,
  max|dU| = max|dM| = 0.000e+00, against a control where sigma 0.25 -> 0.26 moves the same solve.
  Pinned by `test_the_drift_scale_is_inert_at_the_solve` below. This is the same "wrong source, right number" misattribution that
  `test_coupled_mms_2d_no_flux.py` had already corrected in itself -- right value, wrong reason,
  which is exactly why it left no trace.
sigma vs D: D = sigma^2/2 -- `diffusion_from_volatility` is the one converter. Pass sigma
  via sigma=; the (2*pi^2*sigma^2) coefficients this file used to spell out
  already encoded (sigma^2/2)*k^2 -- they are gone since #2201, and the diffusion term now comes
  from the shared assembly, which resolves sigma through `diffusion_from_volatility`.

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
from mfgarchon.utils.manufactured import ManufacturedPair, check_pair, fp_source, hjb_source

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

    Drift used by the FP solver: alpha* = optimal_control(grad u*) = -grad u*/lambda.
    It is read from the Hamiltonian, not from `coupling_coefficient`, which is inert
    here -- see the header. `c` is retained only because `_build_problem` passes it to
    `MFGProblem`, where MFGProblem's own default (0.5) would be equally inert and more
    confusing.

    Source terms (continuous LHS of each equation evaluated on the exact pair):

      S_HJB = -d_t u* + |d_x u*|^2/(2 lambda) + c_f m* - (sigma^2/2) d_xx u*
      S_FP  =  d_t m* + d_x(alpha* m*)        - (sigma^2/2) d_xx m*

    with alpha* = -d_x u*/lambda. Both depend only on (t, x). The assembly itself lives in
    `mfgarchon.utils.manufactured`; only the pair and its derivatives are written here.
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
        # NOT the FP drift coefficient, despite the argument's name: nothing reads this for the
        # drift. It reaches `MFGProblem(coupling_coefficient=...)` and stops there -- see the
        # [SUPERSEDED] block in the header and the solve-level pin below.
        self.c = coupling_coefficient
        self.k = k
        self.D = 0.5 * sigma**2
        self._build_sources()

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
    # The ASSEMBLY is owned by `mfgarchon.utils.manufactured` (#2201). What stays here is the pair
    # and its analytic derivatives; what left is the arithmetic that turned them into S_HJB / S_FP,
    # which this file used to state alongside the sign conventions in its own header -- two
    # statements of one convention, which is how they drift apart.
    def _build_sources(self):
        k, lam = self.k, self.lam
        pair = ManufacturedPair(
            u=lambda t, x: self.u_star(t, x[..., 0]),
            u_t=lambda t, x: -self.b * np.exp(-t) * np.sin(k * x[..., 0]),
            grad_u=lambda t, x: (self.b * k * np.exp(-t) * np.cos(k * x[..., 0]))[:, None],
            hess_u=lambda t, x: (-self.b * k**2 * np.exp(-t) * np.sin(k * x[..., 0]))[:, None, None],
            m=lambda t, x: self.m_star(t, x[..., 0]),
            m_t=lambda t, x: -self.a * np.exp(-t) * np.cos(k * x[..., 0]),
            grad_m=lambda t, x: (-self.a * k * np.exp(-t) * np.sin(k * x[..., 0]))[:, None],
            hess_m=lambda t, x: (-self.a * k**2 * np.exp(-t) * np.cos(k * x[..., 0]))[:, None, None],
            name="coupled_sinusoid_1d",
        )
        # Deliberately a SEPARATE object from the one `_build_problem` puts on the problem, and it
        # must stay separate: a mutant that perturbs the problem's Hamiltonian must NOT reach the
        # source, or it becomes self-consistent and the study converges cleanly on the wrong
        # equation. Measured on the 2D sibling: mutating lambda on BOTH sides gives EOC u
        # 0.918/0.985 and PASSES, where the one-sided mutant gives 0.253/0.123 and fails.
        # They must agree on the VALUES. Here both read `mfg.lam`, so there is no literal to
        # drift; the 2D sibling, where one site was a literal, pins it with
        # `test_the_source_and_the_solver_agree_on_the_coefficients`.
        hamiltonian = SeparableHamiltonian(
            control_cost=QuadraticControlCost(lambda_=lam),
            coupling=lambda m: self.c_f * m,
            coupling_dm=lambda _m: self.c_f,
        )
        self.pair = pair
        self._hjb = hjb_source(pair, hamiltonian, self.sigma)
        self._fp = fp_source(pair, hamiltonian, self.sigma)

    def hjb_source(self, x: np.ndarray, m, v, t: float) -> np.ndarray:
        """S_HJB(t,x). Ignores m, v (FixedPointIterator passes v=zeros)."""
        return self._hjb(t, np.atleast_1d(x).reshape(-1, 1))

    def fp_source(self, x: np.ndarray, m, v, t: float) -> np.ndarray:
        """S_FP(t,x). Ignores m, v; the drift comes from the Hamiltonian's own optimal_control."""
        return self._fp(t, np.atleast_1d(x).reshape(-1, 1))


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
        # INERT on this solver path -- it aligns nothing. Kept only because MFGProblem's own
        # default (0.5) would be equally inert and more confusing.
        coupling_coefficient=mfg.c,
        components=components,
        source_term_hjb=mfg.hjb_source,
        source_term_fp=mfg.fp_source,
    )


def _solve_coupled(mfg: CoupledSinusoid1D, Nx: int, Nt: int, T: float):
    problem = _build_problem(mfg, Nx, Nt, T)
    hjb_solver = HJBFDMSolver(problem)
    fp_solver = FPFDMSolver(problem)
    # relaxation=1.0 (undamped Picard): empirically converges in 10 outer
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
    in 10 iterations and the FP density stays well-behaved, while still
    exercising an ACTIVE bidirectional coupling (c_f>0 and a non-zero grad-u
    drift cross term).

    Grid sequence: Nx in [31, 61]. A 121 point would push each coupled solve to
    ~3 min (the solve is O(Nt * Nx * Newton-per-step) and Nt=4*Nx), so the test
    uses the affordable [31, 61] pair. Time refinement: Nt = 4*Nx.

    EMPIRICALLY MEASURED (this exact configuration, verified before committing
    the threshold):
        ~~u: errors [2.873e-2, 1.625e-2] -> ratio 1.768 (order 0.822)~~
        ~~m: errors [2.585e-1, 1.469e-1] -> ratio 1.760 (order 0.816)~~
        u: errors [4.3479e-03, 2.2531e-03] -> ratio 1.9298 (order 0.948)
        m: errors [2.2768e-02, 1.2589e-02] -> ratio 1.8085 (order 0.855)
    [SUPERSEDED 2026-08-31] SUPERSEDED-BY: #2201. Re-measured on the current tree; the struck
    figures were stale by 6.6x in u and 11.4x in m. NOT caused by the #2201 migration -- the
    pre-migration file measures the same current values -- but this block claims to have been
    "verified before committing the threshold", and a false measurement is worse than none. The
    conclusion strengthens rather than reverses: the real ratios are ABOVE the recorded ones, so
    the margin over the 1.5 threshold is ~21% on the binding field (m), not the ~17% stated below.
    Both Picard iterations converged (~~14-19~~ **10** outer iterations).
    [SUPERSEDED 2026-09-01] SUPERSEDED-BY: #2201. Re-measured: 10 at Nx = 21, 31 and 61, with a
    control that fires -- the same probe gives 4 / 10 / 16 at tolerance 1e-3 / 1e-6 / 1e-9, so the
    instrument is not pinned at 10. Also stale at base d1a4c473, so not caused by the #2201
    migration -- but it sat inside the block whose note says "re-measured on the current tree",
    which had re-measured the error rows above it and not this line. Two further copies at :265
    and :305 carried the same figure and are corrected.

    Threshold: ratios > 1.5 for BOTH u and m -- the precedent set by the
    single-equation source MMS tests (test_mms_validation.py:399 and :796). With
    the measured 1.81 on m (the binding field) it leaves ~21% margin. We do NOT assert the naive order-1
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
        # `assert mfg.c > 0.0` stood here and became vacuous at #2201, which removed its only
        # reader: the drift now comes from the Hamiltonian, so `c` constrains nothing this test is
        # about. It is deleted rather than re-pointed at `c_f`: this test constructs
        # `CoupledSinusoid1D(c_f=0.3)` above, so `assert mfg.c_f > 0.0` would restate a literal the
        # test itself supplied and guard only against __init__ dropping it. The substantive
        # assertion is the allclose on `diff` above -- it fails if the coupling term is missing
        # from S_HJB, which is what this test is named for.
        assert np.any(np.abs(diff) > 0.0)

    def test_the_drift_scale_follows_lambda_not_coupling_coefficient(self):
        """The header used to call `coupling_coefficient` an INDEPENDENT knob set to match 1/lambda.
        It is inert ON THIS PROBLEM'S SOLVER PATH: the FDM FP/HJB families resolve the drift
        through `fp_drift_coefficient`, which returns 1/control_cost.lambda_ for a
        quadratic-MINIMIZE SeparableHamiltonian and never reaches the `coupling_coefficient`
        fallback. Not a package-wide universal -- the velocity-channel FP solvers (FVM, FEM,
        meshless-Galerkin FP) and the network solvers resolve the drift through
        `H.optimal_control` and never call that helper.

        This matters for the source, not just the prose: an assembly that scaled the transport by
        `c` would silently manufacture a different equation than the solver integrates the moment
        anyone changed that argument, and the EOC study would still converge."""
        from mfgarchon.utils.pde_coefficients import fp_drift_coefficient

        for lam, cc, expected in ((1.0, 1.0, 1.0), (1.0, 7.0, 1.0), (2.0, 1.0, 0.5), (2.0, 0.5, 0.5)):
            problem = _build_problem(CoupledSinusoid1D(lambda_=lam, coupling_coefficient=cc), 21, 10, 0.2)
            assert fp_drift_coefficient(problem) == pytest.approx(expected), (
                f"drift at lambda={lam}, coupling_coefficient={cc} resolved to "
                f"{fp_drift_coefficient(problem)}, expected 1/lambda = {expected}"
            )
            assert problem.coupling_coefficient == pytest.approx(cc)  # the argument did arrive

    def test_the_drift_scale_is_inert_at_the_solve(self):
        """The claim in the header is about the SOLVE, so it is pinned at the solve.

        `test_the_drift_scale_follows_lambda_not_coupling_coefficient` above pins
        `fp_drift_coefficient`'s RETURN VALUE, which is one indirection short of the claim: the
        step from the helper to the solve is "every consuming site routes through the owner", and
        that is precisely the step the package has already changed twice (#1528 moved three FP
        families off it). Measured: patching only the CONSUMING module
        (`fp_fdm_time_stepping.fp_drift_coefficient`) to return `coupling_coefficient` leaves both
        of those tests green while the converged density moves by max|dM| ~ 1.2e+01.

        So this asserts the thing itself -- bit-identical solves across four values of the argument,
        including a negative one that would invert any drift that actually read it.
        """
        reference = _solve_coupled(CoupledSinusoid1D(coupling_coefficient=1.0), 21, 40, 0.1)
        for cc in (7.0, 0.5, -3.0):
            other = _solve_coupled(CoupledSinusoid1D(coupling_coefficient=cc), 21, 40, 0.1)
            for name, a, b in (("U", reference[0], other[0]), ("M", reference[1], other[1])):
                delta = float(np.max(np.abs(np.asarray(a) - np.asarray(b))))
                assert delta == 0.0, (
                    f"coupling_coefficient={cc} moved {name} by {delta:.3e}; it is supposed to be "
                    f"inert on this solver path, and the manufactured source does not use it."
                )

        # POSITIVE CONTROL: a coefficient that IS live must move the same measurement, or the
        # assertions above are satisfied by a solve that never varied for any reason.
        moved = _solve_coupled(CoupledSinusoid1D(sigma=0.26), 21, 40, 0.1)
        control = float(np.max(np.abs(np.asarray(reference[1]) - np.asarray(moved[1]))))
        assert control > 1e-6, f"sigma control moved M by only {control:.3e}; the probe is blind"

    def test_the_manufactured_pair_is_its_own_derivatives(self):
        """The pair's analytic derivatives against a finite difference of u* and m*. Non-circular:
        it never touches the assembled source, so it audits what the assembly takes on trust."""
        mfg = CoupledSinusoid1D()
        check_pair(mfg.pair, 0.12, np.linspace(0.05, 0.95, 61).reshape(-1, 1))


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
