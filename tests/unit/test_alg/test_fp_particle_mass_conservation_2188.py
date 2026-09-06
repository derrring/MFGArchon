"""`mass_conservation_error` reports absorption on the particle FP path (Issue #2188).

`SolverResult.mass_conservation_error` was blind to particle absorption. The grid density
`FPParticleSolver` hands back is a kernel-density reconstruction that integrates to ~1
*whatever particle count it is built from* -- 2000 particles and 9 particles both
reconstruct to a normalised density -- so the generic grid-integral measurement in
`FixedPointIterator` could not see absorption at all. Measured in the issue: an absorbing
boundary that killed 99.6% of the particles reported LESS error (4.7079e-02) than a no-flux
solve that lost none (4.9722e-02); across that pair the reported number was
anti-correlated with the defect the field exists to detect.

The fix: `BaseFPSolver.mass_conservation_error_override()` (default `None`, meaning "use the
generic grid measurement") is overridden by `FPParticleSolver` to report the fraction of
particles absorbed instead, computed from `M_particles_trajectory` -- which still holds the
particle count the grid projection throws away.
"""

from __future__ import annotations

import pytest

import numpy as np

from mfgarchon import Conditions, MFGProblem, Model
from mfgarchon.alg.numerical.coupling.fixed_point_iterator import FixedPointIterator
from mfgarchon.alg.numerical.fp_solvers.fp_particle import FPParticleSolver
from mfgarchon.alg.numerical.hjb_solvers.hjb_fdm import HJBFDMSolver
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import BCSegment, BCType, BoundaryConditions


def _bc(right_type: BCType) -> BoundaryConditions:
    return BoundaryConditions(
        segments=[
            BCSegment(name="left", bc_type=BCType.NEUMANN, value=0.0, boundary="x_min"),
            BCSegment(name="right", bc_type=right_type, value=0.0, boundary="x_max"),
        ],
        dimension=1,
    )


def _problem(bc: BoundaryConditions, nx: int = 21, nt: int = 15, sigma: float = 0.5) -> MFGProblem:
    grid = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[nx - 1], boundary_conditions=bc)
    return MFGProblem(
        model=Model(
            hamiltonian=SeparableHamiltonian(control_cost=QuadraticControlCost(control_cost=1.0), coupling=lambda m: m),
            sigma=sigma,
        ),
        domain=grid,
        conditions=Conditions(u_terminal=lambda x: np.squeeze(0.0 * np.asarray(x)), m_initial=lambda x: 1.0, T=0.5),
        Nt=nt,
    )


def _run(bc: BoundaryConditions, num_particles: int = 800, seed: int = 7) -> tuple:
    """Real solve through the actual `FixedPointIterator` API -- no mocks. Returns
    `(result, fp_solver)` so the trajectory can be inspected alongside the reported field."""
    problem = _problem(bc)
    hjb = HJBFDMSolver(problem)
    fp = FPParticleSolver(problem, num_particles=num_particles, seed=seed, boundary_conditions=bc)
    it = FixedPointIterator(problem, hjb_solver=hjb, fp_solver=fp)
    result = it.solve(max_iterations=3, tolerance=1e-3)
    return result, fp


class TestTheAbsorbingSolveReportsMoreErrorThanTheConservingOne:
    def test_no_flux_reports_zero_and_absorbing_reports_the_true_loss(self):
        """The end-to-end oracle: run both boundary conditions for real, on the same problem
        and the same particle seed, and check the ORDERING the issue's own table got backwards.

        The oracle for the absorbing case is independent of `mass_conservation_error_override`
        itself: it is `1 - surviving/initial`, counted directly from the trajectory the solve
        actually produced, not by calling the function under test on itself.
        """
        no_flux_result, _ = _run(_bc(BCType.NEUMANN))
        absorb_result, absorb_fp = _run(_bc(BCType.DIRICHLET))

        assert no_flux_result.mass_conservation_error == 0.0, (
            "no-flux cannot absorb particles; a nonzero value here means the override is "
            "firing on a trajectory with no absorption"
        )

        traj = absorb_fp.M_particles_trajectory
        assert isinstance(traj, list), "the absorbing BC is expected to produce a variable-length trajectory"
        n0, n_last = len(traj[0]), len(traj[-1])
        assert n_last < n0, "the fixture must actually lose particles, or this test proves nothing"
        oracle = 1.0 - n_last / n0

        assert absorb_result.mass_conservation_error == pytest.approx(oracle, rel=1e-12)
        assert absorb_result.mass_conservation_error > no_flux_result.mass_conservation_error, (
            f"the solve that lost particles ({absorb_result.mass_conservation_error!r}) reported "
            f"less error than the one that lost none ({no_flux_result.mass_conservation_error!r}) "
            f"-- the exact ordering #2188 was filed about"
        )

    def test_the_pre_fix_grid_measurement_understates_the_true_loss(self):
        """A labelled defect pin: recomputes the OLD (grid-integral) formula directly on the
        same absorbing solve's own `self.M`, to show it does not track absorption -- the
        mechanism the issue describes, reproduced rather than asserted.

        The KDE reconstruction is not machine-precision blind to absorption (it is not exactly
        0.0) -- it is blind in the sense that matters: it does not scale with the true loss.
        Measured on this fixture: true loss 0.1025 (82 of 800 particles), grid-only drift
        0.0166 -- a 6x understatement, not a wrong-direction error here, but #2188's own table
        shows the SAME grid-only mechanism landing on the wrong side of a no-flux comparison
        entirely (see the oracle test above, which is the ordering claim). This test pins the
        magnitude gap; that one pins the ordering.

        Retirement condition: this trips if `geometry.integrate` ever starts seeing particle
        count (it structurally cannot, short of `FPParticleSolver` changing what `M` is), or if
        this fixture stops losing particles.
        """
        absorb_result, absorb_fp = _run(_bc(BCType.DIRICHLET))
        traj = absorb_fp.M_particles_trajectory
        n0, n_last = len(traj[0]), len(traj[-1])
        assert n_last < n0, "fixture must lose particles"
        true_loss = 1.0 - n_last / n0

        problem = absorb_fp.problem
        M = np.asarray(absorb_result.M, dtype=float)
        grid_mass = problem.geometry.integrate(M)
        grid_only_drift = float(np.max(np.abs(grid_mass / grid_mass[0] - 1.0)))

        assert grid_only_drift < true_loss / 2.0, (
            f"the grid-only measurement ({grid_only_drift!r}) is no longer far below the true "
            f"particle loss ({true_loss!r}); if this fails, #2188's mechanism no longer holds "
            "and the override may be redundant"
        )


class TestMassConservationErrorOverrideDirectly:
    """Unit-level tests on the override in isolation, covering all three trajectory shapes."""

    def test_a_fixed_shape_array_with_no_absorption_gives_zero(self):
        problem = _problem(_bc(BCType.NEUMANN))
        fp = FPParticleSolver(problem, num_particles=10, seed=1)
        fp.M_particles_trajectory = np.zeros((5, 10))  # (Nt, num_particles)
        assert fp.mass_conservation_error_override() == 0.0

    def test_a_variable_length_list_reports_the_survival_ratio(self):
        problem = _problem(_bc(BCType.DIRICHLET))
        fp = FPParticleSolver(problem, num_particles=10, seed=1)
        fp.M_particles_trajectory = [np.zeros(10), np.zeros(8), np.zeros(5)]
        assert fp.mass_conservation_error_override() == pytest.approx(0.5)  # 1 - 5/10

    def test_a_nan_marked_constant_length_list_gives_the_same_answer_as_compact_removal(self):
        """The representation `preserve_indices=True` produces, found only by running it: a
        list where every entry has the SAME length, with absorbed rows marked NaN in place.
        Counting by `len()` alone reads this as zero absorption -- the exact shape of #2188
        one representation down. Both encodings of the identical physical outcome (5 of 10
        survive) must agree.
        """
        problem = _problem(_bc(BCType.DIRICHLET))

        compact = FPParticleSolver(problem, num_particles=10, seed=1)
        compact.M_particles_trajectory = [np.zeros(10), np.zeros(8), np.zeros(5)]

        fp = FPParticleSolver(problem, num_particles=10, seed=1)
        fp.M_particles_trajectory = [
            np.zeros(10),
            np.array([0.0] * 8 + [np.nan] * 2),
            np.array([0.0] * 5 + [np.nan] * 5),
        ]

        # Pin the VALUE independently first -- 5 of 10 survive, so 0.5 -- then the agreement.
        # Asserting only that the two representations agree would compare the method under test
        # against itself, which pins invariance and no number at all.
        assert fp.mass_conservation_error_override() == pytest.approx(0.5)
        assert fp.mass_conservation_error_override() == compact.mass_conservation_error_override()

    def test_no_trajectory_yet_gives_none(self):
        problem = _problem(_bc(BCType.NEUMANN))
        fp = FPParticleSolver(problem, num_particles=10, seed=1)
        assert fp.M_particles_trajectory is None
        assert fp.mass_conservation_error_override() is None


class TestNonParticleSolversAreUnaffected:
    def test_the_base_class_default_is_none(self):
        """The control for the whole mechanism: every solver that does not override this
        method reports None, meaning `FixedPointIterator` falls through to the unchanged
        grid-integral measurement -- confirmed separately by the pre-existing
        `test_mass_conservation_error_1672.py` suite, which this issue does not touch."""
        from mfgarchon.alg.numerical.fp_solvers.base_fp import BaseFPSolver

        assert BaseFPSolver.mass_conservation_error_override is not FPParticleSolver.mass_conservation_error_override


class TestBothAbsorbingRepresentationsAgreeThroughARealSolve:
    """The `preserve_indices=True` representation, driven through an actual solver.

    The 1-D fixtures above cannot reach it: `preserve_indices=True` raises
    `NotImplementedError` on the 1-D path, so every other test in this file exercises only the
    compact-removal encoding. This class uses the 2-D absorbing corridor -- the same construction
    `test_fp_particle_solver.py::TestFPParticlePreserveIndices` uses -- because it is the only
    harness in the suite that can produce the NaN-marked encoding at all.

    Without this, representation 3 would be verified solely by hand-built arrays, and the claim
    that the two encodings agree on real solver output would rest on nothing committed.
    """

    @staticmethod
    def _corridor():
        Lx, Ly, Nt, T = 4.0, 2.0, 20, 1.0
        bc = BoundaryConditions(
            segments=[
                BCSegment(name="left", bc_type=BCType.NEUMANN, value=0.0, boundary="x_min"),
                BCSegment(name="right", bc_type=BCType.DIRICHLET, value=0.0, boundary="x_max"),
                BCSegment(name="bottom", bc_type=BCType.NEUMANN, value=0.0, boundary="y_min"),
                BCSegment(name="top", bc_type=BCType.NEUMANN, value=0.0, boundary="y_max"),
            ],
            dimension=2,
        )
        grid = TensorProductGrid([(0, Lx), (0, Ly)], Nx_points=[21, 11], boundary_conditions=bc)
        problem = MFGProblem(
            model=Model(
                hamiltonian=SeparableHamiltonian(
                    control_cost=QuadraticControlCost(control_cost=1.0), coupling=lambda m: m
                ),
                sigma=0.3,
            ),
            domain=grid,
            conditions=Conditions(
                # 1 / |domain| so the grid-measure mass is exactly 1: this corridor is 4x2, and a
                # uniform 1.0 would carry mass 8 and trip the #1887 "mass is not 1" warning, which
                # the gate's warnings ratchet correctly refuses as a NEW identity from a new test.
                m_initial=lambda p: 1.0 / (Lx * Ly),
                u_terminal=lambda p: 0.0,
                T=T,
            ),
            Nt=Nt,
        )
        return problem, bc

    def _solve(self, preserve_indices: bool, seed: int = 42) -> float:
        np.random.seed(seed)
        problem, bc = self._corridor()
        solver = FPParticleSolver(
            problem,
            num_particles=500,
            density_mode="query_only",
            boundary_conditions=bc,
            preserve_indices=preserve_indices,
        )
        init = np.random.uniform([0.5, 0.5], [1.5, 1.5], (500, 2))
        solver.solve_fp_system(
            initial_particles=init,
            drift_field=lambda t, x, m: np.column_stack([np.full(len(x), 2.0), np.zeros(len(x))]),
            volatility_field=0.3,
            drift_needs_density=False,
            show_progress=False,
        )
        return solver.mass_conservation_error_override()

    def test_the_two_encodings_of_absorption_report_the_same_loss(self):
        """Same seed, same physics, two internal encodings of "who is still alive" -- the reported
        loss must not depend on which one the solver happened to store.

        This is the test that would have caught the first draft of this fix, which counted
        `len()` per slice: under `preserve_indices=True` the length is constant by construction,
        so it read 0.0 absorption while the compact encoding read the true loss.
        """
        compact = self._solve(preserve_indices=False)
        nan_marked = self._solve(preserve_indices=True)

        assert compact > 0.0, "the corridor must actually absorb particles, or this proves nothing"
        assert nan_marked == pytest.approx(compact, rel=1e-12), (
            f"the NaN-marked encoding reported {nan_marked!r} where compact removal reported "
            f"{compact!r} on identical physics -- the encoding is leaking into the measurement"
        )
