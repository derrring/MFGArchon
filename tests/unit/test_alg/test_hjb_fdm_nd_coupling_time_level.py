#!/usr/bin/env python3
"""Issue #1423: the nD HJB-FDM backward loop must couple U^n to the density at the SAME time
level, M[n] — matching the continuous coupling H(x, ∇u, m(t_n)) and the 1D path's "BUG #7 FIX".
Previously the nD path used M[n+1] (a silent O(dt) cross-path off-by-one vs 1D / Howard).

White-box pinning: spy on _solve_single_timestep, feed a density whose time slices are all distinct,
and assert that when solving U^n the solver receives M[n] (not M[n+1]). With the pre-fix code this
test fails (it would receive M[n+1]).
"""

import warnings

import pytest

import numpy as np

from mfgarchon.alg.numerical.hjb_solvers import HJBFDMSolver
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.core.mfg_components import MFGComponents
from mfgarchon.core.mfg_problem import MFGProblem
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import no_flux_bc


def _problem_2d(N=6, T=0.2, Nt=5, sigma=0.1):
    comp = MFGComponents(
        m_initial=lambda x: 1.0,
        u_terminal=lambda x: 0.0,
        hamiltonian=SeparableHamiltonian(
            control_cost=QuadraticControlCost(control_cost=1.0),
            coupling=lambda m: m,
            coupling_dm=lambda m: 1.0,
        ),
    )
    geom = TensorProductGrid(
        bounds=[(0.0, 1.0), (0.0, 1.0)],
        Nx_points=[N + 1, N + 1],
        boundary_conditions=no_flux_bc(dimension=2),
    )
    return MFGProblem(geometry=geom, components=comp, T=T, Nt=Nt, sigma=sigma)


class TestNDCouplingTimeLevel:
    def test_nd_couples_un_to_mn(self):
        problem = _problem_2d()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            solver = HJBFDMSolver(problem, solver_type="fixed_point")

        nt_points = problem.Nt + 1
        shape = solver.shape
        # Density with strictly distinct, identifiable time slices: M[k] == full of value (k+1).
        M = np.stack([np.full(shape, float(k + 1)) for k in range(nt_points)])
        U_terminal = np.zeros(shape)
        U_prev = np.zeros((nt_points, *shape))

        captured: dict[int, float] = {}
        orig = solver._solve_single_timestep

        def spy(U_next, M_coupling, U_guess, sigma_at_n, Sigma_at_n, **kw):
            # Record the constant value of the coupling slice; infer which time level it is.
            captured[len(captured)] = float(np.asarray(M_coupling).flat[0])
            return orig(U_next, M_coupling, U_guess, sigma_at_n, Sigma_at_n, **kw)

        solver._solve_single_timestep = spy
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            solver.solve_hjb_system(M, U_terminal, U_coupling_prev=U_prev)

        # The backward loop runs n = Nt-1, Nt-2, ..., 0. For each, the coupling slice must be M[n],
        # whose constant value is (n+1). So the sequence of captured values must be Nt, Nt-1, ..., 1.
        expected = [float(n + 1) for n in range(nt_points - 2, -1, -1)]
        got = [captured[i] for i in range(len(captured))]
        assert got == expected, (
            f"nD HJB-FDM coupled U^n to the wrong density time level (Issue #1423). "
            f"Expected M[n] values {expected}, got {got}. "
            f"A pre-fix M[n+1] would yield {[float(n + 2) for n in range(nt_points - 2, -1, -1)]}."
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
