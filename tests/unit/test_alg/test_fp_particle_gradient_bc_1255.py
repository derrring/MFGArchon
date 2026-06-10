"""
Pinning tests for Issue #1255 — three residual holes in the #1246 FP-particle
drift-gradient fix.

(A) fp_particle.py: _compute_gradient_nd used the geometry's own BC, ignoring
    the solver's boundary_conditions override.  With a no-flux geometry BC and a
    periodic solver override the gradient at the wall was wrong by O(1/h).

(B) fp_particle.py: on implicit geometries (Hyperrectangle/CSG torus),
    get_gradient_operator(self) accepted no kwargs, so the scheme="central" call
    raised TypeError — crashing a previously-working torus/zero-drift path.

(C) applicator_fdm.py: _update_ghosts_uniform discarded seg.alpha/seg.beta for
    uniform ROBIN BCs, silently treating them all as pure Neumann (alpha=0,
    beta=1).

Each test below FAILS on the buggy code and PASSES after the fix.

2026-06-10 audit — Issue #1255.
"""

from __future__ import annotations

import pytest

import numpy as np

# ---------------------------------------------------------------------------
# Helpers shared across tests
# ---------------------------------------------------------------------------


def _default_hamiltonian():
    from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian

    return SeparableHamiltonian(
        control_cost=QuadraticControlCost(control_cost=1.0),
        coupling=lambda m: m,
        coupling_dm=lambda m: 1.0,
    )


def _default_components():
    from mfgarchon.core.mfg_components import MFGComponents

    return MFGComponents(
        m_initial=lambda x: np.exp(-10 * (x - 0.5) ** 2),
        u_terminal=lambda x: 0.0,
        hamiltonian=_default_hamiltonian(),
    )


# ===========================================================================
# (A) Solver boundary_conditions override threads into _compute_gradient_nd
# ===========================================================================


class TestGradientBCOverrideThreaded:
    """
    Issue #1255 (A): _compute_gradient_nd must use the solver's
    boundary_conditions, not the geometry's default BC.

    Setup: TensorProductGrid with NO_FLUX BC; solver created with periodic BC
    override.  On u = [0, 1, 2, 3, 4] (spacing=1):

      periodic:  ghost_lo = u[4] = 4  →  du/dx|_{x=0} = (u[1]-ghost_lo)/(2h) = (1-4)/2 = -1.5
      no_flux:   ghost_lo = u[0] = 0  →  du/dx|_{x=0} = (u[1]-ghost_lo)/(2h) = (1-0)/2 = 0.5

    Bug: get_gradient_operator always called self.get_boundary_conditions()
    (geometry BC), so the override had no effect → gradient = 0.5.
    Fix: bc_override is forwarded to get_gradient_operator(bc=bc_override).
    """

    def _make_solver(self):
        from mfgarchon.alg.numerical.fp_solvers import FPParticleSolver
        from mfgarchon.core.mfg_problem import MFGProblem
        from mfgarchon.geometry import TensorProductGrid
        from mfgarchon.geometry.boundary import no_flux_bc, periodic_bc

        # 1-D grid [0,4] with 5 points, spacing = 1.0 — BC is NO_FLUX
        geom = TensorProductGrid(
            bounds=[(0.0, 4.0)],
            Nx_points=[5],
            boundary_conditions=no_flux_bc(dimension=1),
        )
        problem = MFGProblem(
            geometry=geom,
            T=1.0,
            Nt=2,
            sigma=0.1,
            coupling_coefficient=0.5,
            components=_default_components(),
        )
        # Solver overrides BC with periodic — priority-1 source per docs
        solver = FPParticleSolver(
            problem,
            num_particles=10,
            boundary_conditions=periodic_bc(dimension=1),
        )
        return solver

    def test_override_bc_is_stored(self):
        """Solver stores the override BC, not the geometry BC."""
        from mfgarchon.geometry.boundary.conditions import BoundaryConditions
        from mfgarchon.geometry.boundary.types import BCType

        solver = self._make_solver()
        assert isinstance(solver.boundary_conditions, BoundaryConditions)
        assert solver.boundary_conditions.segments[0].bc_type == BCType.PERIODIC

    def test_gradient_uses_periodic_override_not_noflux_geometry(self):
        """
        _compute_gradient_nd must return the periodic-wrap gradient, not the
        no-flux one.  Fails on buggy code (returns 0.5), passes after fix (-1.5).
        """
        solver = self._make_solver()
        u = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        spacings = [1.0]

        grads = solver._compute_gradient_nd(u, spacings, use_backend=False)

        grad_x = grads[0]
        # Periodic: at x=0, du/dx = (u[1] - u[4])/(2*1) = (1-4)/2 = -1.5
        # No-flux: at x=0, du/dx = (u[1] - u[0])/(2*1) = (1-0)/2 = 0.5
        assert abs(grad_x[0] - (-1.5)) < 1e-9, (
            f"Expected -1.5 (periodic override), got {grad_x[0]:.4f}. "
            "Bug: geometry's no-flux BC was used instead of solver's periodic override."
        )


# ===========================================================================
# (B) Implicit geometry (torus) no longer raises TypeError on scheme=
# ===========================================================================


class TestImplicitGeometryGradientNoTypeError:
    """
    Issue #1255 (B): ImplicitGeometry.get_gradient_operator(self) accepts no
    kwargs, so calling it with scheme="central" raised TypeError — crashing the
    torus/zero-drift path.

    Fix: _compute_gradient_nd detects non-TensorProductGrid geometry and falls
    back to gradient_nd (periodic-wrap) instead of calling get_gradient_operator.

    We test with a minimal mock geometry whose get_gradient_operator takes no
    kwargs (reproducing the ImplicitGeometry signature exactly).
    """

    class _MockImplicitGeom:
        """Reproduces the failing ImplicitGeometry.get_gradient_operator signature."""

        dimension = 1
        periodic_dimensions = (0,)  # Fully periodic → sentinel "periodic" in solver

        def get_boundary_conditions(self):
            return None

        def get_gradient_operator(self):  # NO kwargs accepted
            # Placeholder — should never be reached after fix
            raise NotImplementedError("Meshfree gradient not implemented")

    def _make_solver_with_implicit_geom(self):
        """Create a minimal FPParticleSolver whose geometry is implicit."""
        from mfgarchon.alg.numerical.fp_solvers import FPParticleSolver

        geom = self._MockImplicitGeom()

        # Build a minimal mock problem that satisfies FPParticleSolver.__init__
        # without needing a real MFGProblem.
        class _MinimalProblem:
            geometry = geom
            T = 1.0
            Nt = 2
            sigma = 0.1
            coupling_coefficient = 0.5
            dimension = 1

            def get_components(self):
                return None

        problem = _MinimalProblem()

        # FPParticleSolver.__init__ calls super().__init__(problem) which accesses
        # problem attributes — we bypass __init__ and construct the object directly
        # so we can set only what _compute_gradient_nd needs.
        solver = object.__new__(FPParticleSolver)
        solver.problem = problem
        solver.backend = None  # No GPU; numpy only
        solver.boundary_conditions = "periodic"  # Torus sentinel
        return solver

    def test_zero_drift_on_implicit_geom_does_not_raise_type_error(self):
        """
        Calling _compute_gradient_nd on a zero field with implicit geometry must
        not raise TypeError.  Buggy code: TypeError on scheme= kwarg.
        """
        solver = self._make_solver_with_implicit_geom()
        u = np.zeros(5)
        spacings = [1.0]

        # Must not raise TypeError
        grads = solver._compute_gradient_nd(u, spacings, use_backend=False)

        # Zero field → zero gradient in every component
        assert len(grads) == 1
        assert np.allclose(grads[0], 0.0), "Expected zero gradient for zero field"

    def test_nonzero_field_falls_back_to_periodic_wrap(self):
        """
        For a non-zero field, the fallback path (gradient_nd) uses periodic wrap.
        This is the pre-#1246 correct behavior for torus geometries.
        """
        solver = self._make_solver_with_implicit_geom()
        u = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        spacings = [1.0]

        grads = solver._compute_gradient_nd(u, spacings, use_backend=False)

        # gradient_nd uses np.roll periodic wrap — same as pre-#1246 behavior.
        # At x=0: (u[1] - u[-1])/(2h) = (1 - 4)/2 = -1.5
        assert abs(grads[0][0] - (-1.5)) < 1e-9, f"Expected -1.5 (periodic-wrap fallback), got {grads[0][0]:.4f}"


# ===========================================================================
# (C) Uniform Robin ghost fill uses general alpha/beta formula
# ===========================================================================


class TestUniformRobinGhostAlphaBeta:
    """
    Issue #1255 (C): _update_ghosts_uniform discarded seg.alpha/seg.beta for
    ROBIN, silently using alpha=0/beta=1 (pure Neumann).

    We verify by creating a PreallocatedGhostBuffer with robin_bc(alpha=1,
    beta=2, value=5.0, dimension=1) and checking that the ghost values match
    the general Robin formula, not the pure-Neumann one.

    Cell-centred Robin formula (cell centres dx apart, boundary midway):
        alpha * (u_g + u_i)/2 + beta * (u_g - u_i)/dx * sign = g
        => u_g = (g - u_i * (alpha/2 - beta*sign/dx)) / (alpha/2 + beta*sign/dx)

    Low wall (sign = -1):
        coeff_ghost = alpha/2 + beta*(-1)/dx = alpha/2 - beta/dx
        coeff_int   = alpha/2 - beta*(-1)/dx = alpha/2 + beta/dx
        u_ghost_lo  = (g - u_i * coeff_int) / coeff_ghost

    High wall (sign = +1):
        coeff_ghost = alpha/2 + beta*(+1)/dx = alpha/2 + beta/dx
        coeff_int   = alpha/2 - beta*(+1)/dx = alpha/2 - beta/dx
        u_ghost_hi  = (g - u_i * coeff_int) / coeff_ghost

    For alpha=1, beta=2, g=5, dx=1, u_lo=u_hi=1.0:
        Low:  coeff_ghost = 0.5 - 2 = -1.5  coeff_int = 2.5
              u_ghost_lo  = (5 - 1*2.5)/(-1.5) = -2.5/-1.5 ≈ 1.6667
        High: coeff_ghost = 0.5 + 2 = 2.5   coeff_int = -1.5
              u_ghost_hi  = (5 - 1*(-1.5))/2.5 = 6.5/2.5 = 2.6

    Buggy code (alpha=0, beta=1):
        Low:  u_ghost_lo = u_i - dx*g = 1 - 5 = -4.0
        High: u_ghost_hi = u_i + dx*g = 1 + 5 = 6.0
    """

    @pytest.fixture
    def ghost_buffer(self):
        from mfgarchon.geometry.boundary.applicator_fdm import PreallocatedGhostBuffer
        from mfgarchon.geometry.boundary.conditions import robin_bc

        bc = robin_bc(alpha=1.0, beta=2.0, value=5.0, dimension=1)
        buf = PreallocatedGhostBuffer(
            interior_shape=(5,),
            boundary_conditions=bc,
            domain_bounds=np.array([[0.0, 4.0]]),  # dx = 4/(5-1) = 1.0
            ghost_depth=1,
        )
        buf.interior[:] = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
        buf.update_ghosts(time=0.0)
        return buf

    def test_low_wall_ghost_uses_general_robin_not_neumann(self, ghost_buffer):
        """
        Low-wall ghost must use the general Robin formula, not pure Neumann.
        Buggy value: -4.0 (pure Neumann, alpha=0, beta=1).
        Fixed value: (5 - 1*2.5)/(-1.5) = -2.5/-1.5 ≈ 1.6667.
        """
        u_ghost_lo = ghost_buffer.padded[0]  # index 0 = low ghost
        expected = (5.0 - 1.0 * 2.5) / (-1.5)  # ≈ 1.6667
        assert abs(u_ghost_lo - expected) < 1e-9, (
            f"Low ghost: expected {expected:.4f} (general Robin), "
            f"got {u_ghost_lo:.4f}. "
            "Bug: pure-Neumann formula was used (alpha=0, beta=1 hardcoded)."
        )

    def test_high_wall_ghost_uses_general_robin_not_neumann(self, ghost_buffer):
        """
        High-wall ghost must use the general Robin formula, not pure Neumann.
        Buggy value: 6.0 (pure Neumann).
        Fixed value: (5 + 1*1.5)/2.5 = 2.6.
        """
        u_ghost_hi = ghost_buffer.padded[-1]  # index -1 = high ghost
        expected = (5.0 - 1.0 * (-1.5)) / 2.5  # = 6.5/2.5 = 2.6
        assert abs(u_ghost_hi - expected) < 1e-9, (
            f"High ghost: expected {expected:.4f} (general Robin), "
            f"got {u_ghost_hi:.4f}. "
            "Bug: pure-Neumann formula was used (alpha=0, beta=1 hardcoded)."
        )

    def test_uniform_matches_mixed_segment_path(self):
        """
        Cross-check: uniform Robin ghost must equal the mixed-segment path result.

        Create the same Robin BC as a mixed-segment specification (which already
        used the full formula before #1255).  Both paths must agree.
        """
        from mfgarchon.geometry.boundary.applicator_fdm import PreallocatedGhostBuffer
        from mfgarchon.geometry.boundary.conditions import BoundaryConditions, robin_bc
        from mfgarchon.geometry.boundary.types import BCSegment, BCType

        alpha, beta, g_val = 1.0, 2.0, 5.0
        interior = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
        shape = (5,)
        bounds = np.array([[0.0, 4.0]])

        # Uniform path (the one with the bug)
        bc_uniform = robin_bc(alpha=alpha, beta=beta, value=g_val, dimension=1)
        buf_uniform = PreallocatedGhostBuffer(shape, bc_uniform, bounds)
        buf_uniform.interior[:] = interior
        buf_uniform.update_ghosts(time=0.0)

        # Mixed-segment path (was correct before #1255)
        bc_mixed = BoundaryConditions(
            segments=[
                BCSegment(
                    name="left",
                    bc_type=BCType.ROBIN,
                    alpha=alpha,
                    beta=beta,
                    value=g_val,
                    boundary="x_min",
                ),
                BCSegment(
                    name="right",
                    bc_type=BCType.ROBIN,
                    alpha=alpha,
                    beta=beta,
                    value=g_val,
                    boundary="x_max",
                ),
            ],
            dimension=1,
        )
        buf_mixed = PreallocatedGhostBuffer(shape, bc_mixed, bounds)
        buf_mixed.interior[:] = interior
        buf_mixed.update_ghosts(time=0.0)

        assert np.allclose(buf_uniform.padded, buf_mixed.padded, atol=1e-12), (
            f"Uniform path padded:\n{buf_uniform.padded}\n"
            f"Mixed path padded:\n{buf_mixed.padded}\n"
            "Uniform and mixed Robin paths must produce identical ghost values."
        )
