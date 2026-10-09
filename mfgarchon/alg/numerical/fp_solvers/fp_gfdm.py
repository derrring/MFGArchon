"""FP-GFDM is withdrawn (#2583): it built no wall, so it refuses construction until it is rebuilt (#2584).

The strong-form solver that lived here assembled ``div(m * alpha)`` and the Laplacian on scattered
collocation points with a ``TaylorOperator``. No flux or ghost row was ever built, and the operator took no
boundary-condition argument: its ``geometry=`` serves periodic wrapping and the ``obstacle_sdf=`` it was given
serves stencil visibility, and neither imposes a flux condition. So a declared NO_FLUX or NEUMANN wall was
accepted and read by nothing, and the drift's flux crossed the boundary.

What that did to mass depended on the drift, so a mass check could not be relied on to see it. Measured on
1-D, 21 points, sigma = 0.4, T = 0.5, Nt = 10, a uniform initial density and an explicit no-flux wall:
- under a uniform drift +1 toward x_max, FP-FDM piles the density up at the wall (m(T) = 0.0493, 0.5573,
  6.3400 at x = 0, 0.5, 1) while FP-GFDM stayed uniform (1.0000 at all three), both at mass 1.0000;
- under the drifts ``x`` and ``0.5 - x``, FP-GFDM's mass at T was 0.599 and 1.629, i.e. (1 -/+ 0.05)^10:
  a net boundary flux of 0.05 of the mass per explicit step, compounded over the 10 steps.

An implicit domain carrying no BC gave the same uniform solve. Every domain it accepted was bounded, or
periodic without being wrapped: a periodic implicit domain behaved exactly like a non-periodic one.

It ran wherever the caller supplied the velocity: standalone, ``solve_fp_system(drift_field=...)``, and inside
a coupled solve given the iterator's ``drift_field=`` override. The GFDM pair's own coupled solve, which
routes the drift from the value function, did not run: it raised #1420's auto-route error under a smooth
separable Hamiltonian, and a drift-shape ``ValueError`` under a bounded or L1 control cost. User ruling
2026-10-09: refuse rather than solve wall-less.

The class stays, as a refusing stub: the import succeeds and construction fails with the reason, rather
than the import failing with an ImportError, and ``scheme_factory``'s GFDM pair raises the same reason before
it builds anything. The
rebuild, as the discrete adjoint of the GFDM generator that the HJB side already assembles, is #2584. The
deleted body is in git history.
"""

from __future__ import annotations

from typing import Any, NoReturn

from mfgarchon.alg.base_solver import SchemeFamily
from mfgarchon.alg.numerical.fp_solvers.base_fp import BaseFPSolver, DriftConvention
from mfgarchon.utils.pde_coefficients import retired_volatility_keywords

WITHDRAWN = (
    "FPGFDMSolver is withdrawn: FP-GFDM built no wall (no flux or ghost row; its operator took no "
    "boundary-condition argument), so on every bounded domain it accepted the drift's flux crossed the "
    "boundary (#2583). It refuses construction until it is rebuilt as the discrete adjoint of the "
    "GFDM generator (#2584). On a grid without obstacles, use a scheme with a dual FP, e.g. "
    "problem.solve(scheme=NumericalScheme.FDM_UPWIND), or, to keep the GFDM HJB, pair it with an FP solver "
    "that builds walls: problem.solve(hjb_solver=HJBGFDMSolver(problem, collocation_points=...), "
    "fp_solver=FPFDMSolver(problem)), a non-dual pair that Expert Mode warns about. On an implicit domain or "
    "a grid with obstacles neither runs: FDM_UPWIND and FPFDMSolver refuse it."
)


class FPGFDMSolver(BaseFPSolver):
    """Withdrawn (#2583): construction raises ``NotImplementedError`` naming the alternative. Rebuild: #2584."""

    #: The withdrawn solver took a velocity. The coupling layer reads this from instances, and none can exist,
    #: so only tests that inspect the class read it now.
    _drift_convention = DriftConvention.VELOCITY
    _scheme_family = SchemeFamily.GFDM
    #: It honours no boundary condition type (#2583; PERIODIC was withdrawn earlier, #1822).
    _SUPPORTED_BC_TYPES: frozenset = frozenset()
    honors_inhomogeneous_neumann: bool = False

    def __init__(self, problem: Any, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError(WITHDRAWN)

    @retired_volatility_keywords
    def solve_fp_system(self, M_initial: Any = None, drift_field: Any = None) -> NoReturn:
        """Unreachable: construction refuses. Defined so the abstract-class check does not pre-empt that refusal;
        it keeps the two names the drift-routing contract reads (#2377, #1043) and refuses the retired
        volatility names as every entry point does (#2378)."""
        raise NotImplementedError(WITHDRAWN)
