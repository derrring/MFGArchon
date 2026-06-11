"""Pinning tests for Issue #1248: volatility_field silently lost.

D1: MFGProblem.solve() must forward problem.volatility_field to the
    FixedPointIterator so both HJB and FP solvers see the full SDE volatility
    (array or callable) rather than the mean-scalar placeholder stored in
    problem.sigma.

    Pinning: MFGProblem(sigma=spatially_varying_array).solve() must produce a
    density that is NOT allclose to MFGProblem(sigma=mean(array)).solve().
    Before the fix the two are byte-identical (both solve with the mean scalar).

D2: FPParticleSolver.solve_fp_system(drift_field=ndarray, volatility_field=array)
    must use the supplied array's mean as effective sigma, not problem.sigma.

    Pinning: solve with ndarray drift + array volatility_field whose mean differs
    from problem.sigma must produce a density NOT allclose to the reference solve
    that uses only problem.sigma.  Before the fix both produce the same density
    because the array is silently dropped and problem.sigma is used for both.
"""

from __future__ import annotations

import warnings

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

        problem_array = MFGProblem(geometry=geo, components=comp, T=T, Nt=Nt, sigma=sigma_arr)
        problem_mean = MFGProblem(geometry=geo, components=comp, T=T, Nt=Nt, sigma=sigma_mean)

        result_array = problem_array.solve()
        result_mean = problem_mean.solve()

        m_array = result_array.M[-1]
        m_mean = result_mean.M[-1]

        # After fix the spatial-sigma solve must diverge from the mean-sigma solve.
        assert not np.allclose(m_array, m_mean, atol=1e-6), (
            "D1 regression: MFGProblem(sigma=array).solve() produced the same "
            "density as MFGProblem(sigma=mean(array)).solve() — volatility_field "
            "was not forwarded to the FixedPointIterator (Issue #1248 D1)."
        )

    def test_scalar_sigma_solve_is_unchanged(self):
        """Passing a scalar sigma must still work correctly after the fix.

        Forwarding volatility_field=float to the iterator is identical to the
        prior behaviour where the iterator used problem.sigma directly.
        """
        geo = _geometry()
        comp = _components()
        problem = MFGProblem(geometry=geo, components=comp, T=T, Nt=Nt, sigma=0.25)
        result = problem.solve()
        assert result.M is not None
        assert result.M.shape[0] == Nt + 1
        # Basic mass-conservation sanity check (loose tolerance for solver variation)
        total_mass = result.M[-1].sum() / Nx
        assert abs(total_mass - 1.0) < 0.3, f"Scalar-sigma solve: final mass = {total_mass:.4f}, expected ~1.0"


# ---------------------------------------------------------------------------
# D2: FPParticleSolver must use array volatility_field (not problem.sigma)
# ---------------------------------------------------------------------------


class TestD2ParticleVolatilityFieldNotDropped:
    """Issue #1248 D2 — FPParticleSolver uses array volatility_field with ndarray drift."""

    def test_array_volatility_with_ndarray_drift_differs_from_problem_sigma(self):
        """After fix: solve with array volatility != problem.sigma produces different density.

        Before fix: line 1409 in fp_particle.py always fell back to
        effective_sigma = self.problem.sigma regardless of the supplied array,
        so both solves used 0.1 and produced byte-identical densities.  After
        fix, effective_sigma = mean(sigma_array) = 0.5, which differs from
        problem.sigma = 0.1, so the densities diverge.
        """
        geo = _geometry()
        comp = _components()
        # problem.sigma = 0.1 (the small baseline)
        problem = MFGProblem(geometry=geo, components=comp, T=T, Nt=Nt, sigma=0.1)

        solver = FPParticleSolver(problem, num_particles=500)

        m0 = _m_initial_normalised()
        # Simple value-function array for drift (Nt+1 time slices)
        U_arr = np.tile(0.3 * (np.linspace(0.0, 1.0, Nx) - 0.5) ** 2, (Nt + 1, 1))

        # Reference: no volatility_field override → uses problem.sigma = 0.1
        np.random.seed(42)
        m_ref = solver.solve_fp_system(m0, drift_field=U_arr)

        # Test: supply array volatility whose mean (0.5) differs from problem.sigma (0.1).
        sigma_arr = np.full(Nx, 0.5)
        np.random.seed(42)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)  # expected collapse-to-mean warning
            m_new = solver.solve_fp_system(m0, drift_field=U_arr, volatility_field=sigma_arr)

        # After fix the two densities must differ: mean(0.5) >> 0.1 so diffusion is much
        # stronger and the density spreads measurably more.
        assert not np.allclose(m_new, m_ref, atol=1e-6), (
            "D2 regression: FPParticleSolver with ndarray drift_field + array "
            "volatility_field produced the same density as the problem.sigma "
            "solve — the array volatility_field was silently dropped (Issue #1248 D2)."
        )

    def test_array_volatility_with_ndarray_drift_emits_warning(self):
        """Collapsing array volatility to its mean on the grid-drift path must warn.

        The collapse (mean(sigma_array)) is a necessary approximation on the CPU/GPU
        grid-drift paths; a UserWarning informs the caller that true per-point
        volatility is unavailable on this path.
        """
        geo = _geometry()
        comp = _components()
        problem = MFGProblem(geometry=geo, components=comp, T=T, Nt=Nt, sigma=0.1)
        solver = FPParticleSolver(problem, num_particles=200)

        m0 = _m_initial_normalised()
        U_arr = np.tile(0.3 * (np.linspace(0.0, 1.0, Nx) - 0.5) ** 2, (Nt + 1, 1))
        sigma_arr = np.full(Nx, 0.5)

        np.random.seed(42)
        with pytest.warns(UserWarning, match="collapsed to mean"):
            solver.solve_fp_system(m0, drift_field=U_arr, volatility_field=sigma_arr)

    def test_callable_volatility_with_ndarray_drift_raises(self):
        """Callable volatility_field + ndarray drift_field must raise NotImplementedError.

        Before the fix this silently used problem.sigma.  After the fix we
        raise (fail-fast) because we cannot reduce a callable to a scalar without
        evaluating it over the domain — and this path does not support per-point
        evaluation.  Users should pass a callable drift_field instead.
        """
        geo = _geometry()
        comp = _components()
        problem = MFGProblem(geometry=geo, components=comp, T=T, Nt=Nt, sigma=0.1)
        solver = FPParticleSolver(problem, num_particles=200)

        m0 = _m_initial_normalised()
        U_arr = np.tile(0.3 * (np.linspace(0.0, 1.0, Nx) - 0.5) ** 2, (Nt + 1, 1))

        def sigma_callable(t, x, m):
            if hasattr(x, "__len__"):
                return 0.5 * np.ones_like(x)
            return 0.5

        with pytest.raises(NotImplementedError, match="callable volatility_field"):
            solver.solve_fp_system(m0, drift_field=U_arr, volatility_field=sigma_callable)
