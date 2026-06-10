"""
Pinning test for issue #1258: mixed-BC Dirichlet wall frozen at IC.

BUG (alg/numerical/fp_solvers/fp_fdm_time_stepping.py):
    _is_dirichlet_at_point returned True for every boundary point when a mixed BC
    contained any Dirichlet segment, because the loop over segments returned True
    without verifying the point lies on that segment's wall.  Additionally,
    _get_bc_type returns None for mixed BC, so the RHS Dirichlet enforcement block
    in solve_timestep_full_nd was never reached.  Net effect: every boundary row
    got a Dirichlet identity matrix row (1/dt on diagonal) but RHS = m_current/dt,
    giving m_next = m_current (frozen at IC) on ALL walls.

FIX:
    _is_dirichlet_at_point now determines which wall(s) the grid point is on from
    multi_idx/shape, then calls get_bc_type_at_boundary per wall.
    _get_dirichlet_value_at_point similarly resolves the wall-specific value.
    The RHS enforcement in solve_timestep_full_nd now also triggers for mixed BC
    that contains any Dirichlet segment, applying the per-point Dirichlet value.
"""

import numpy as np

from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.core.mfg_components import MFGComponents
from mfgarchon.core.mfg_problem import MFGProblem
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import BCSegment, BCType, BoundaryConditions, no_flux_bc


def _make_problem(N: int = 20, Nt: int = 10, sigma: float = 0.3, T: float = 0.5) -> MFGProblem:
    """Minimal 1D MFGProblem for FP FDM testing."""
    domain = TensorProductGrid(
        bounds=[(0.0, 1.0)],
        Nx_points=[N],
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
    return MFGProblem(geometry=domain, T=T, Nt=Nt, sigma=sigma, components=components)


def _make_mixed_bc() -> BoundaryConditions:
    """1D mixed BC: Dirichlet(value=0) at x_min, NO_FLUX at x_max."""
    dirichlet_seg = BCSegment(
        name="left_absorb",
        bc_type=BCType.DIRICHLET,
        value=0.0,
        boundary="x_min",
        priority=1,
    )
    no_flux_seg = BCSegment(
        name="right_wall",
        bc_type=BCType.NO_FLUX,
        priority=0,
    )
    return BoundaryConditions(
        segments=[dirichlet_seg, no_flux_seg],
        dimension=1,
        domain_bounds=np.array([[0.0, 1.0]]),
    )


class TestMixedBCDirichletNotFrozen:
    """
    Pinning tests for issue #1258: mixed-BC Dirichlet wall must reach prescribed value.

    Before the fix, ALL boundary points (including the Dirichlet wall) were frozen
    at the initial condition value because:
    1. _is_dirichlet_at_point returned True for every boundary point (any Dirichlet segment)
    2. _get_bc_type returned None for mixed BC, so the Dirichlet RHS was never applied
    Result: every boundary row solved m_next = m_current (identity mapping).
    """

    def test_dirichlet_wall_reaches_prescribed_value(self):
        """
        x_min wall must reach Dirichlet value=0, not stay frozen at IC.

        With rightward drift and Dirichlet(0) at x_min, the boundary density
        must be forced to 0 at each implicit step, not stay at the IC value.
        This is the primary regression guard for issue #1258.
        """
        N = 20
        Nt = 11  # shape of U_solution = (Nt, N)
        problem = _make_problem(N=N, Nt=Nt - 1)  # Nt-1 because problem.Nt = number of intervals
        bc = _make_mixed_bc()

        # IC with intentionally large value at x_min so a frozen wall is obvious
        x_grid = np.linspace(0, 1, N)
        m_ic = np.exp(-((x_grid - 0.5) ** 2) / (2 * 0.2**2))
        m_ic[0] = 2.0  # Non-zero IC at x_min — must be driven to 0 by Dirichlet BC

        # Rightward drift: U = -v*x  => drift = -coupling * dU/dx = +v (rightward)
        v = 0.5
        U_sol = np.tile(-v * x_grid, (Nt, 1))

        from mfgarchon.alg.numerical.fp_solvers.fp_fdm_time_stepping import (
            solve_fp_nd_full_system,
        )

        m_result = solve_fp_nd_full_system(
            m_initial_condition=m_ic,
            U_solution_for_drift=U_sol,
            problem=problem,
            boundary_conditions=bc,
            show_progress=False,
        )

        # IC preserved at t=0
        assert abs(m_result[0, 0] - 2.0) < 1e-10, f"IC should be 2.0 at x_min, got {m_result[0, 0]}"

        # After implicit steps, x_min wall MUST be at Dirichlet value 0.0
        # (frozen-bug: it would stay at 2.0)
        assert abs(m_result[3, 0]) < 1e-10, (
            f"x_min wall should be 0.0 after 3 steps (Dirichlet BC), got {m_result[3, 0]:.6f}. "
            "This indicates issue #1258: Dirichlet value not applied for mixed BC."
        )
        assert abs(m_result[-1, 0]) < 1e-10, f"x_min wall should be 0.0 at last step, got {m_result[-1, 0]:.6f}."

    def test_interior_not_globally_frozen(self):
        """
        Interior points must evolve — not all frozen at IC.

        A secondary symptom of the bug: with uniform IC and all boundary points
        frozen, the interior can also remain frozen. With the fix, interior points
        must change under diffusion + drift.
        """
        N = 20
        Nt = 11
        problem = _make_problem(N=N, Nt=Nt - 1)
        bc = _make_mixed_bc()

        x_grid = np.linspace(0, 1, N)
        m_ic = np.exp(-((x_grid - 0.5) ** 2) / (2 * 0.2**2))
        m_ic[0] = 2.0

        v = 0.5
        U_sol = np.tile(-v * x_grid, (Nt, 1))

        from mfgarchon.alg.numerical.fp_solvers.fp_fdm_time_stepping import (
            solve_fp_nd_full_system,
        )

        m_result = solve_fp_nd_full_system(
            m_initial_condition=m_ic,
            U_solution_for_drift=U_sol,
            problem=problem,
            boundary_conditions=bc,
            show_progress=False,
        )

        # Interior should change: at least one interior point must differ from IC
        max_interior_change = np.max(np.abs(m_result[3, 1:-1] - m_ic[1:-1]))
        assert max_interior_change > 1e-6, (
            f"Interior should change under drift+diffusion, but max change is {max_interior_change:.2e}. "
            "This suggests the solver is globally frozen."
        )

    def test_no_flux_wall_uses_no_flux_assembly(self):
        """
        x_max wall must NOT be forced to Dirichlet value; it uses no-flux BC.

        Before the fix, _is_dirichlet_at_point incorrectly returned True for the
        x_max (no-flux) wall, giving it a Dirichlet identity row but without the
        correct RHS. With the fix, x_max gets proper no-flux assembly.
        """
        N = 20
        Nt = 11
        problem = _make_problem(N=N, Nt=Nt - 1)
        bc = _make_mixed_bc()

        x_grid = np.linspace(0, 1, N)
        m_ic = np.exp(-((x_grid - 0.5) ** 2) / (2 * 0.2**2))
        m_ic[0] = 0.5  # Non-trivial IC

        v = 0.5
        U_sol = np.tile(-v * x_grid, (Nt, 1))

        from mfgarchon.alg.numerical.fp_solvers.fp_fdm_time_stepping import (
            solve_fp_nd_full_system,
        )

        m_result = solve_fp_nd_full_system(
            m_initial_condition=m_ic,
            U_solution_for_drift=U_sol,
            problem=problem,
            boundary_conditions=bc,
            show_progress=False,
        )

        # x_max should NOT be pinned to Dirichlet 0: it is a no-flux wall, so
        # under rightward drift it should accumulate mass (value > 0).
        assert m_result[-1, -1] > 1e-4, (
            f"x_max (no-flux wall) should accumulate mass under rightward drift, "
            f"got {m_result[-1, -1]:.6f}. "
            "If it is 0, the wall was incorrectly treated as Dirichlet."
        )
