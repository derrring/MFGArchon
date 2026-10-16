"""
Retired penalty wrapper for the obstacle problem v >= Psi(x) (#2002).

`PenaltyHJBSolver` raises `NotImplementedError` on construction; the message there gives the
reason and the alternative. It was written (#924) to approximate the variational inequality

    min(-dv/dt + H(x, m, Dv), v - Psi(x)) = 0

by the penalized equation

    -dv/dt + H(x, m, Dv) - (1/eps) * max(0, Psi(x) - v) = 0,

whose penalty raises v where it falls below Psi, injected through the inner solver's
`source_term`. That channel is `(t, x) -> array` and cannot carry v, so the term the wrapper
applied was `penalty_parameter * max(0, Psi(x))`, the same where the constraint holds and where it
fails. The obstacle route that runs is
`HJBFDMSolver(problem, constraint=ObstacleConstraint(psi, 'lower'))` (#591), with `psi` the
obstacle sampled on the grid: a projection after each time step, with its limits in #2036 and #2046.

`v >= Psi` bounds the cost-to-go from below. The library minimises (`docs/user/CONVENTIONS.md`
§1), so optimal stopping at a stopping cost Psi is the upper obstacle `v <= Psi`,
`ObstacleConstraint(psi, 'upper')`, not this one.

References:
    - Bensoussan & Lions (1982), "Applications of Variational Inequalities
      in Stochastic Control"
    - Achdou & Capuzzo-Dolcetta (2010), "Mean Field Games: Numerical Methods"
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from mfgarchon.types.callable_protocols import evaluate_solver_source
from mfgarchon.utils.pde_coefficients import retired_volatility_keywords

from .base_hjb import BaseHJBSolver

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray


class PenaltyHJBSolver(BaseHJBSolver):
    """Retired (#2002): construction raises `NotImplementedError`; see the module docstring.

    It was written to enforce v >= Psi(x) by adding a penalty to the inner solver's
    `source_term`, which cannot carry v.

    Parameters
    ----------
    inner_solver : BaseHJBSolver
        The HJB solver it would have wrapped.
    obstacle : Callable[[NDArray], NDArray]
        Obstacle function Psi(x). Receives spatial grid (N, d) or (N,),
        returns array of obstacle values (N,).
    penalty_parameter : float
        Penalty strength 1/eps. Default: 1e4.
    """

    # Inherit scheme family from inner solver
    _scheme_family = None  # Set dynamically from inner solver

    def __init__(
        self,
        inner_solver: BaseHJBSolver,
        obstacle: Callable[[NDArray], NDArray],
        penalty_parameter: float = 1e4,
    ):
        raise NotImplementedError(
            "PenaltyHJBSolver is RETIRED (#2002). It cannot do what it was written to do, and "
            "the reason is structural rather than a bug in its arithmetic.\n\n"
            "Its design is to add the variational inequality `v >= Psi(x)` to ANY inner solver by "
            "injecting a penalty into that solver's `source_term`. A penalty for that constraint "
            "is `max(0, Psi - v)`, which needs the value function. `source_term` has signature "
            "`(t, x) -> array`. There is nowhere for `v` to enter, so what it actually applied "
            "was `penalty_parameter * max(0, Psi(x))` -- positive wherever `Psi > 0` whether or "
            "not the constraint holds, and byte-identical at a node satisfying it and one "
            "violating it. It penalised POSITION, not VIOLATION, and no value of "
            "`penalty_parameter` changes that.\n\n"
            "What to use instead, and its limits:\n"
            "  HJBFDMSolver(problem, constraint=ObstacleConstraint(psi, 'lower'))  -- #591.\n"
            "`ObstacleConstraint.project` does read `u` and does enforce `u >= psi`. HJBFDMSolver "
            "projects after each time step, in 1-D and n-D, and refuses a terminal condition "
            "outside the set (#2036). That is a projection splitting, not a solve of the discrete "
            "variational inequality. It is also the only solver "
            "carrying a `constraint` attribute, so this is an FDM-family capability, not a "
            "general one -- #2046 tracks threading the constraint through the shared timestep "
            "solve, which is where a projection can actually participate in the iteration.\n\n"
            "If your obstacle is a state penalty V(x) rather than a constraint -- alpha-free and "
            "u-free -- it belongs in the Hamiltonian's potential (#1999, #2001), not here."
        )
        # Unreachable. Left in place so that a return behind a `(t, x, v)` channel is a one-line
        # revert.
        super().__init__(inner_solver.problem, getattr(inner_solver, "config", None))
        self._inner = inner_solver
        self._obstacle = obstacle
        self._penalty = penalty_parameter
        # Copy scheme family for trait validation
        self._scheme_family = getattr(inner_solver, "_scheme_family", None)

    def _get_solver_type_id(self) -> str | None:
        """Delegate to inner solver for compatibility checking."""
        return self._inner._get_solver_type_id()

    def solve(self) -> NDArray:
        """Unreachable (construction raises). It returned the inner solver's `solve()`, with no penalty."""
        return self._inner.solve()

    @retired_volatility_keywords
    def solve_hjb_system(
        self,
        M_density: NDArray,
        U_terminal: NDArray,
        U_coupling_prev: NDArray,
        volatility: float | NDArray | None = None,
        source_term: Callable | None = None,
        volatility_kind: str | None = None,
    ) -> NDArray:
        """Solve HJB with obstacle constraint via penalty method.

        .. warning::

           **This docstring described the intended term, not the implemented one (#2002).**
           It said the method "composes the penalty term ``(1/eps) * max(0, Psi - v)``" and
           that "the penalty pushes v upward when it falls below Psi, enforcing the
           variational inequality ``v >= Psi`` in the limit ``eps -> 0``". The code below
           computes ``penalty_parameter * max(0, Psi)`` -- no ``v``, and the knob is the
           reciprocal spelling. Both sentences are withdrawn.

           What is actually applied is a **position** penalty: positive wherever
           ``Psi > 0``, identical at a node satisfying the constraint and one violating it,
           so it cannot push ``v`` anywhere in particular. See the comment at the
           expression itself.

           A constraint-shaped alternative exists elsewhere --
           :meth:`~mfgarchon.geometry.boundary.ObstacleConstraint.project` (#591), which
           ``HJBFDMSolver`` applies when constructed with ``constraint=``. It does read ``u``
           and does enforce ``u >= psi`` on what it returns, which is more than this term can
           say. Since #2036 it is applied after each time step in 1-D and nD alike, so the
           obstacle feeds back through the sweep, and a terminal condition outside the set is
           refused. It is a projection splitting, not a solve of the discrete variational
           inequality (#2046). It is the route that runs; this method is unreachable.
        """
        penalty_param = self._penalty
        obstacle_fn = self._obstacle

        def penalized_source(t: float, x: NDArray) -> NDArray:
            # Start with existing source term if any
            base = evaluate_solver_source(source_term, t=t, x=x) if source_term is not None else np.zeros(x.shape[0])

            # Penalty: (1/eps) * max(0, Psi(x) - v)
            # Note: We evaluate Psi at x but don't have v here.
            # The penalty is based on the obstacle value only.
            # For the full v-dependent penalty, the inner solver's
            # time-stepping loop handles it (source_term is evaluated
            # at each time step with the current v).
            #
            # This works because the HJB solver subtracts source_term
            # from the residual: F(u) = (u-u_next)/dt + H - S = 0
            # With S = (1/eps)*max(0, Psi-u), we get:
            #   F(u) = (u-u_next)/dt + H - (1/eps)*max(0, Psi-u) = 0
            # When u < Psi, the penalty term pushes u upward.
            #
            # However: the source_term signature is (t, x) -> array,
            # not (t, x, v) -> array. The v-dependent penalty must be
            # handled at the time-stepping level inside the solver.
            # For now, we apply a static obstacle penalty.
            #
            # #2002: `max(0, psi)` PENALISES POSITION, NOT VIOLATION. It contains no `v`, so it is
            # positive wherever psi > 0 whether or not `v >= Psi` holds -- verified, unchanged at
            # v = -10, 0, +10. This is not a weaker form of the constraint; it is a different term.
            #
            # And this is NOT the "proper handling" that `source_composition`'s docstring pointed
            # here for. It is the same stub with the reciprocal knob: multiplying by
            # `penalty_parameter` (1e4) where that path divides by `eps` (1e6). At psi = 0.5 the two
            # give 5.000000e+03 and 5.000000e-07 -- 1e10 apart, while a comment there claimed they
            # matched. That claim is withdrawn; this comment is the other half of the correction.
            psi = np.asarray(obstacle_fn(x)).ravel()
            return base + penalty_param * np.maximum(0.0, psi)

        return self._inner.solve_hjb_system(
            M_density,
            U_terminal,
            U_coupling_prev,
            volatility=volatility,
            volatility_kind=volatility_kind,
            source_term=penalized_source,
        )

    @property
    def free_boundary_estimate(self) -> NDArray | None:
        """Unreachable (construction raises). It always returned None: no estimate was implemented."""
        return None

    def validate_solution(self) -> dict[str, Any]:
        """Delegate validation to inner solver."""
        return self._inner.validate_solution()
