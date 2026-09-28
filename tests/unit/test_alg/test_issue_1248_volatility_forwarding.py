"""Pinning tests for Issue #1248: volatility_field silently lost.

D1: MFGProblem.solve() must forward problem.volatility_field to the
    FixedPointIterator so both HJB and FP solvers see the full SDE volatility
    (array or callable) rather than the mean-scalar placeholder stored in
    problem.sigma.

    Pinning: MFGProblem(sigma=spatially_varying_array).solve() must produce a
    density that is NOT allclose to MFGProblem(sigma=mean(array)).solve().
    Before the fix the two are byte-identical (both solve with the mean scalar).

D2: FPParticleSolver.solve_fp_system(drift_field=ndarray, volatility_field=...) must not
    drop the override for the problem's volatility. Before the fix it did, and both solves
    produced the same density.

    Pinning: a scalar override that differs from the problem's changes the density. An array
    or a callable override is refused by name: this path consumes one scalar sigma, and
    ~~the array's mean~~ [SUPERSEDED 2026-09-27, #2376] a representative value is no longer
    substituted for a field. SUPERSEDED-BY: test_volatility_is_never_collapsed_2376.py.
"""

from __future__ import annotations

import pytest

import numpy as np

from mfgarchon.alg.numerical.fp_solvers.fp_particle import FPParticleSolver
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.core.mfg_components import MFGComponents
from mfgarchon.core.mfg_problem import MFGProblem
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import no_flux_bc

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

Nx = 20  # small enough for fast tests
T = 0.3
Nt = 6


def _geometry() -> TensorProductGrid:
    return TensorProductGrid(
        bounds=[(0.0, 1.0)],
        Nx_points=[Nx],
        boundary_conditions=no_flux_bc(dimension=1),
    )


def _components() -> MFGComponents:
    H = SeparableHamiltonian(control_cost=QuadraticControlCost(control_cost=1.0))
    return MFGComponents(
        m_initial=lambda x: np.exp(-20.0 * (x - 0.5) ** 2),
        u_terminal=lambda x: 0.5 * (x - 0.5) ** 2,
        hamiltonian=H,
    )


def _m_initial_normalised() -> np.ndarray:
    x = np.linspace(0.0, 1.0, Nx)
    m = np.exp(-20.0 * (x - 0.5) ** 2)
    return m / (m.sum() / Nx)  # L1 normalise on the grid


# ---------------------------------------------------------------------------
# D1: MFGProblem.solve() must forward volatility_field to FixedPointIterator
# ---------------------------------------------------------------------------


class TestD1SolveMustForwardVolatilityField:
    """Issue #1248 D1 — problem.solve() forwards volatility_field to the iterator."""

    @staticmethod
    def _sigma_array() -> np.ndarray:
        # Piecewise: sigma=0.15 on left half, sigma=0.45 on right half.
        # The mean is 0.30; the spatial variation is large enough to produce
        # a measurably different solution from the constant-sigma solve.
        sigma = np.empty(Nx)
        sigma[: Nx // 2] = 0.15
        sigma[Nx // 2 :] = 0.45
        return sigma

    def test_array_sigma_solve_differs_from_mean_sigma_solve(self):
        """After fix: spatial-array sigma produces a density different from mean-sigma solve.

        Before fix: FixedPointIterator.volatility_field == None, HJB receives
        volatility_field=None and falls back to problem.sigma (the mean 0.30
        placeholder).  Both solves are byte-identical.  After fix, HJB receives
        the full spatial array so the two solves diverge.
        """
        sigma_arr = self._sigma_array()
        sigma_mean = float(np.mean(sigma_arr))

        geo = _geometry()
        comp = _components()

        problem_array = MFGProblem(
            geometry=geo, components=comp, T=T, Nt=Nt, volatility=sigma_arr, volatility_kind="field"
        )
        problem_mean = MFGProblem(geometry=geo, components=comp, T=T, Nt=Nt, volatility=sigma_mean)

        result_array = problem_array.solve()
        result_mean = problem_mean.solve()

        m_array = result_array.M[-1]
        m_mean = result_mean.M[-1]

        # After fix the spatial-sigma solve must diverge from the mean-sigma solve.
        assert not np.allclose(m_array, m_mean, atol=1e-6), (
            "D1 regression: MFGProblem(volatility=array).solve() produced the same "
            "density as MFGProblem(volatility=mean(array)).solve() — volatility_field "
            "was not forwarded to the FixedPointIterator (Issue #1248 D1)."
        )

    def test_scalar_sigma_solve_is_unchanged(self):
        """Passing a scalar sigma must still work correctly after the fix.

        Forwarding volatility_field=float to the iterator is identical to the
        prior behaviour where the iterator used problem.sigma directly.
        """
        geo = _geometry()
        comp = _components()
        problem = MFGProblem(geometry=geo, components=comp, T=T, Nt=Nt, volatility=0.25)
        result = problem.solve()
        assert result.M is not None
        assert result.M.shape[0] == Nt + 1
        # Mass conservation, on the geometry's own measure and against the mass the solve STARTED
        # with. Two things were wrong with `result.M[-1].sum() / Nx` compared to 1.0 at tolerance
        # 0.3: `sum()/Nx` is a point average, not an integral on any grid, and after #1887 the
        # library no longer rescales `m_initial`, so 1.0 was a property of the old constructor
        # rather than of this solve. A 30% band around a number nothing produces is not a check.
        mass = np.asarray(geo.integrate(result.M), dtype=float)
        drift = float(np.max(np.abs(mass / mass[0] - 1.0)))
        assert drift < 1e-9, f"scalar-sigma solve did not conserve mass: drift {drift:.3e}"


# ---------------------------------------------------------------------------
# D2: FPParticleSolver must use array volatility_field (not problem.sigma)
# ---------------------------------------------------------------------------


class TestD2ParticleVolatilityFieldNotDropped:
    """Issue #1248 D2 — FPParticleSolver uses array volatility_field with ndarray drift."""

    def test_scalar_volatility_override_with_ndarray_drift_is_not_dropped(self):
        """A scalar override that differs from the problem's changes the density.

        Before the fix the grid-drift path fell back to the problem's volatility regardless of
        the override, so both solves used 0.1 and produced byte-identical densities.
        """
        geo = _geometry()
        comp = _components()
        problem = MFGProblem(geometry=geo, components=comp, T=T, Nt=Nt, volatility=0.1)

        solver = FPParticleSolver(problem, num_particles=500)

        m0 = _m_initial_normalised()
        # Simple value-function array for drift (Nt+1 time slices)
        U_arr = np.tile(0.3 * (np.linspace(0.0, 1.0, Nx) - 0.5) ** 2, (Nt + 1, 1))

        # Reference: no volatility_field override -> the problem's 0.1
        np.random.seed(42)
        m_ref = solver.solve_fp_system(m0, drift_field=U_arr)

        np.random.seed(42)
        m_new = solver.solve_fp_system(m0, drift_field=U_arr, volatility=0.5)

        assert not np.allclose(m_new, m_ref, atol=1e-6), (
            "D2 regression: FPParticleSolver with ndarray drift_field + a scalar "
            "volatility_field produced the same density as the problem-volatility "
            "solve -- the override was silently dropped (Issue #1248 D2)."
        )

    @pytest.mark.parametrize(
        "override",
        [np.full(Nx, 0.5), lambda t, x, m: 0.5 * np.ones_like(np.atleast_1d(x))],
        ids=["array", "callable"],
    )
    def test_non_scalar_volatility_with_ndarray_drift_is_refused_by_name(self, override):
        """An array or callable override on the grid-drift path is refused, not dropped or averaged.

        The path consumes one scalar sigma. Averaging an array was the #1248 stopgap; #2376
        replaced it with a refusal that names the path and the callable-drift alternative.
        """
        geo = _geometry()
        comp = _components()
        problem = MFGProblem(geometry=geo, components=comp, T=T, Nt=Nt, volatility=0.1)
        solver = FPParticleSolver(problem, num_particles=200)

        m0 = _m_initial_normalised()
        U_arr = np.tile(0.3 * (np.linspace(0.0, 1.0, Nx) - 0.5) ** 2, (Nt + 1, 1))

        kind = {"volatility_kind": "field"} if isinstance(override, np.ndarray) else {}
        with pytest.raises(NotImplementedError, match="FPParticleSolver's grid-drift path"):
            solver.solve_fp_system(m0, drift_field=U_arr, volatility=override, **kind)
