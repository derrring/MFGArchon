"""
Penalty method for variational inequality HJB (optimal stopping / entry-exit).

Solves: min(-du/dt + H(x, m, Du), v - Psi(x)) = 0

via penalization: -du/dt + H(x, m, Du) + (1/eps) * max(0, Psi(x) - v) = 0

This is a **wrapper** that decorates any BaseHJBSolver. The penalty term is
injected via the source_term parameter (#921), so the inner solver doesn't
need any modification. Any HJB solver (FDM, SL, GFDM, FEM) gains VI
capability instantly.

Issue #924: Part of Layer 1 (Generalized PDE & Institutional MFG Plan).

Mathematical background:
    The variational inequality arises in MFG with optimal stopping
    (entry/exit dynamics). Agents can choose to exit the game when their
    value function hits the obstacle Psi(x) (e.g., zero scrap value).

    The penalty method approximates the VI by adding a large penalty
    (1/eps) * max(0, Psi - v) to the HJB equation. As eps -> 0, the
    penalized solution converges to the VI solution.

    For MFG entry/exit models (Institutional Proposal Project A):
    - Psi(x) = 0 (exit value) with smooth pasting at the free boundary
    - The free boundary x*(t) separates active from exited firms

References:
    - Bensoussan & Lions (1982), "Applications of Variational Inequalities
      in Stochastic Control"
    - Achdou & Capuzzo-Dolcetta (2010), "Mean Field Games: Numerical Methods"
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from .base_hjb import BaseHJBSolver

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray


class PenaltyHJBSolver(BaseHJBSolver):
    """Wrapper that adds variational inequality constraint to any HJB solver.

    Enforces v >= Psi(x) (for MINIMIZE) or v <= Psi(x) (for MAXIMIZE)
    via a penalty term added to the source_term of the inner solver.

    Parameters
    ----------
    inner_solver : BaseHJBSolver
        The HJB solver to wrap. Can be any concrete solver (FDM, SL, GFDM, etc.)
    obstacle : Callable[[NDArray], NDArray]
        Obstacle function Psi(x). Receives spatial grid (N, d) or (N,),
        returns array of obstacle values (N,).
    penalty_parameter : float
        Penalty strength 1/eps. Larger values give sharper enforcement
        but may cause numerical stiffness. Default: 1e4.
        Typical range: 1e3 (soft) to 1e6 (hard).

    Example
    -------
    >>> from mfgarchon.alg.numerical.hjb_solvers import HJBFDMSolver
    >>> inner = HJBFDMSolver(problem)
    >>> # Entry-exit: firms exit when value drops to zero
    >>> penalty_solver = PenaltyHJBSolver(
    ...     inner_solver=inner,
    ...     obstacle=lambda x: np.zeros(x.shape[0]),  # Psi = 0
    ...     penalty_parameter=1e4,
    ... )
    >>> U = penalty_solver.solve_hjb_system(M, U_T, U_prev)
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
            "`ObstacleConstraint.project` does read `u` and does enforce `u >= psi` on the "
            "returned array. It is not yet an obstacle-problem SOLVER: in 1-D the projection runs "
            "after the backward sweep, so the result is the unconstrained solution clipped, and "
            "in n-D the terminal slice is never projected (#2036). It is also the only solver "
            "carrying a `constraint` attribute, so this is an FDM-family capability, not a "
            "general one -- #2046 tracks threading the constraint through the shared timestep "
            "solve, which is where a projection can actually participate in the iteration.\n\n"
            "If your obstacle is a state penalty V(x) rather than a constraint -- alpha-free and "
            "u-free -- it belongs in the Hamiltonian's potential (#1999, #2001), not here."
        )
        # Unreachable. Left in place so the retirement is a one-line revert if #2002 decides the
        # wrapper should return with a `(t, x, v)` channel behind it.
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
        """Solve standalone (delegates to inner solver with penalty)."""
        return self._inner.solve()

    def solve_hjb_system(
        self,
        M_density: NDArray,
        U_terminal: NDArray,
        U_coupling_prev: NDArray,
        volatility_field: float | NDArray | None = None,
        source_term: Callable | None = None,
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
           say. **It is not, however, an obstacle-problem solver, and "correct" overstates it**
           (#2036): in 1D the projection runs after the backward sweep finishes, so the result
           is exactly ``max(U_free, psi)`` -- the unconstrained solution clipped, with no free
           boundary resolved; in nD it runs inside the time loop and does feed back, but the
           terminal slice never passes through it and can violate the constraint. Prefer it
           over this term while #2002 is open, knowing both limits.
        """
        penalty_param = self._penalty
        obstacle_fn = self._obstacle

        def penalized_source(t: float, x: NDArray) -> NDArray:
            # Start with existing source term if any
            base = source_term(t, x) if source_term is not None else np.zeros(x.shape[0])

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
            volatility_field=volatility_field,
            source_term=penalized_source,
        )

    @property
    def free_boundary_estimate(self) -> NDArray | None:
        """Estimate free boundary location after solve.

        Returns spatial points where v approximately equals Psi,
        i.e., the boundary between the continuation and stopping regions.

        Returns None if no solution is available.
        """
        # This would need access to the last solution and obstacle values.
        # Placeholder for future implementation.
        return None

    def validate_solution(self) -> dict[str, Any]:
        """Delegate validation to inner solver."""
        return self._inner.validate_solution()


if __name__ == "__main__":
    """Quick smoke test for PenaltyHJBSolver."""
    from mfgarchon import MFGProblem
    from mfgarchon.alg.numerical.hjb_solvers import HJBFDMSolver
    from mfgarchon.geometry import TensorProductGrid
    from mfgarchon.geometry.boundary import no_flux_bc

    print("Testing PenaltyHJBSolver...")

    from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
    from mfgarchon.core.mfg_components import MFGComponents

    # Simple 1D problem with Hamiltonian and terminal condition
    H = SeparableHamiltonian(
        control_cost=QuadraticControlCost(control_cost=1.0),
        coupling=lambda m: -(m**2),
    )
    components = MFGComponents(
        hamiltonian=H,
        u_terminal=lambda x: 0.0,
        m_initial=lambda x: 1.0,
    )
    geometry = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[51], boundary_conditions=no_flux_bc(dimension=1))
    problem = MFGProblem(geometry=geometry, T=1.0, Nt=20, sigma=0.3, components=components)
    inner = HJBFDMSolver(problem)

    # Obstacle: Psi(x) = 0.5 * sin(pi * x) — agents must stay above this
    def obstacle(x: np.ndarray) -> np.ndarray:
        return 0.5 * np.sin(np.pi * np.atleast_1d(x).ravel())

    solver = PenaltyHJBSolver(inner, obstacle=obstacle, penalty_parameter=1e4)

    # Solve — use problem's own initial/terminal conditions
    Nt = problem.Nt
    grid_shape = problem.geometry.get_grid_shape()
    Nx = grid_shape[0]
    M = np.ones((Nt + 1, Nx)) / Nx
    U_T = problem.get_final_u()
    U_prev = np.zeros((Nt + 1, Nx))

    U = solver.solve_hjb_system(M, U_T, U_prev)

    # Check obstacle constraint
    x_grid = problem.geometry.get_spatial_grid().ravel()
    psi = obstacle(x_grid)
    violation = np.min(U[-1] - psi)
    print(f"  Min(v - Psi) at t=0: {violation:.4f} (should be >= 0 or near 0)")
    print(f"  Solution shape: {U.shape}")
    print(f"  Solution range: [{U.min():.4f}, {U.max():.4f}]")

    assert U.shape[0] == Nt + 1, f"Time dimension mismatch: {U.shape}"
    assert np.all(np.isfinite(U)), "Non-finite values in solution"
    print("Smoke test passed!")
