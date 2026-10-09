"""FP-GFDM is withdrawn (#2583): it built no wall, so it refuses construction until it is rebuilt (#2584).

The strong-form solver that lived here assembled ``div(m * alpha)`` and the Laplacian on scattered
collocation points with a ``TaylorOperator`` that takes no boundary argument, and no flux or ghost row was
ever built. So a declared NO_FLUX or NEUMANN wall was accepted and read by nothing: under a drift the full
advective flux crossed the boundary at exactly conserved mass, which a mass check cannot see. Measured on
1-D, 21 points, a uniform drift +1 toward x_max and an explicit no-flux wall: FP-FDM piles the density up
at the wall (m(T) = 0.0493, 0.5573, 6.3400 at x = 0, 0.5, 1) while FP-GFDM stayed uniform (1.0000 x 3),
both at mass 1.0000. The same held on an implicit domain carrying no BC, and every domain GFDM is admitted
on is bounded, so no reachable problem had a correct FP-GFDM solve (user ruling 2026-10-09: refuse rather
than solve wall-less).

The class stays, as a refusing stub, so an import and ``scheme_factory``'s GFDM pair fail loudly with the
reason rather than with an ImportError. The rebuild, as the discrete adjoint of the GFDM generator that the
HJB side already assembles, is #2584. The deleted body is in git history.
"""

from __future__ import annotations

from typing import Any, NoReturn

from mfgarchon.alg.base_solver import SchemeFamily
from mfgarchon.alg.numerical.fp_solvers.base_fp import BaseFPSolver, DriftConvention
from mfgarchon.utils.pde_coefficients import retired_volatility_keywords

WITHDRAWN = (
    "FPGFDMSolver is withdrawn: FP-GFDM built no wall (its operator took no boundary argument), so on every "
    "reachable domain, all of them bounded, the drift's flux crossed the boundary at exactly conserved mass "
    "(#2583). It refuses construction until it is rebuilt as the discrete adjoint of the GFDM generator "
    "(#2584). For a bounded problem, use a scheme with a dual FP, e.g. "
    "problem.solve(scheme=NumericalScheme.FDM_UPWIND); to keep the GFDM HJB, pair it with an FP solver that "
    "builds walls, problem.solve(hjb_solver=HJBGFDMSolver(problem, collocation_points=...), "
    "fp_solver=FPFDMSolver(problem)) or FPParticleSolver, a non-dual pair that Expert Mode warns about."
)


class FPGFDMSolver(BaseFPSolver):
    """Withdrawn (#2583): construction raises ``NotImplementedError`` naming the alternative. Rebuild: #2584."""

    #: Read by the coupling layer's drift routing and its tests (#1420): the withdrawn solver took a velocity.
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
