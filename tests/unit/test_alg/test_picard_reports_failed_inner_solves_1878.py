"""A coupled solve does not report convergence over HJB time steps that are not roots (#1878).

On the 1-D smoke fixture (21 nodes, 10 steps, sigma = 0, FDM_UPWIND) Picard met its criteria at sweep 38 at
6c0610d2 while three of the ten backward steps of that sweep had returned iterates at residuals 7.5e+01 to
2.8e+02 against a Newton tolerance of 1e-06. The fixed point of that map is not a solution of the discrete
MFG system.

Two claims, two oracles:

- **The report.** Where inner solves cannot converge, the result says ``converged=False`` and names the steps.
  Made certain by asking for a Newton tolerance no float iterate can meet, with a budget of 3 steps, so it
  holds whatever the inner solver's globalization does.
- **The answer.** The returned ``U`` and ``M`` satisfy the discrete HJB at every backward step, measured from
  the arrays alone through the scheme's residual -- not from what Newton says about itself.
"""

from __future__ import annotations

import warnings

import numpy as np

from mfgarchon import Conditions, MFGProblem, Model
from mfgarchon.alg.numerical.fp_solvers import FPFDMSolver
from mfgarchon.alg.numerical.hjb_solvers import HJBFDMSolver, base_hjb
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.geometry import TensorProductGrid, no_flux_bc
from mfgarchon.utils.pde_coefficients import get_spatial_grid


def _smoke_problem() -> MFGProblem:
    """`scripts/capability_matrix.py::_smoke_problem`, through the Model/Conditions API.

    Its initial density integrates to 0.546 on purpose, as the capability fixture's does, so the constructor's
    sub-unit-mass warning is expected and silenced here.
    """
    grid = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[21], boundary_conditions=no_flux_bc(dimension=1))
    hamiltonian = SeparableHamiltonian(
        control_cost=QuadraticControlCost(control_cost=1.0), coupling=lambda m: m, coupling_dm=lambda m: 1.0
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="initial density mass", category=UserWarning)
        return MFGProblem(
            model=Model(hamiltonian=hamiltonian, sigma=0.0),
            domain=grid,
            conditions=Conditions(
                m_initial=lambda x: np.exp(-10 * (np.asarray(x) - 0.5) ** 2), u_terminal=lambda x: 0.0, T=1.0
            ),
            Nt=10,
        )


def _solve(problem: MFGProblem, **hjb_options):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return problem.solve(
            hjb_solver=HJBFDMSolver(problem, **hjb_options),
            fp_solver=FPFDMSolver(problem),
            max_iterations=150,
            verbose=False,
        )


def _hjb_residual_per_step(problem: MFGProblem, U: np.ndarray, M: np.ndarray) -> np.ndarray:
    """max|F_n(U[n]; U[n+1], M[n])| for every backward step n, from the returned arrays."""
    bc = problem.geometry.get_boundary_conditions()
    bounds = problem.geometry.get_bounds()
    domain_bounds = np.array([[bounds[0][0], bounds[1][0]]])
    diffusion = problem.get_diffusion_coefficient_field(override=None, field_name="volatility_field", dimension=1)
    grid = get_spatial_grid(problem)
    out = []
    for n in range(U.shape[0] - 1):
        sigma_at_n = diffusion.evaluate_at(timestep_idx=n, grid=grid, density=M[n], dt=problem.dt)
        residual = base_hjb.compute_hjb_residual(
            U[n],
            U[n + 1],
            M[n],
            problem,
            n,
            None,
            sigma_at_n,
            True,
            bc=bc,
            domain_bounds=domain_bounds,
            current_time=n * problem.dt,
        )
        out.append(float(np.abs(np.asarray(residual, dtype=float)).max()))
    return np.asarray(out)


def test_a_fixed_point_over_inner_solves_that_cannot_converge_is_not_reported_converged():
    problem = _smoke_problem()
    result = _solve(problem, newton_tolerance=1e-300, max_newton_iterations=3)
    failures = result.metadata["inner_hjb_failures"]
    assert result.converged is False, "Picard reported convergence over inner solves that cannot meet their tolerance"
    assert result.metadata["convergence_reason"].startswith("inner_hjb_not_converged"), result.metadata[
        "convergence_reason"
    ]
    assert sorted(failure.t_idx for failure in failures) == list(range(problem.Nt)), failures
    # The criteria were met, so the loop stops there: more sweeps of the same map only re-certify its fixed point.
    assert result.iterations < 150, f"Picard ran its whole budget ({result.iterations}) after its criteria were met"


def test_a_converged_result_satisfies_the_discrete_hjb_at_every_step():
    """The answer half, and its presence control: the same fixture at the default tolerance converges.

    At 6c0610d2 the non-decrease guard in `solve_hjb_timestep_newton` stopped Newton after its first,
    overshooting step and returned an iterate worse than its start, so t_idx 7-9 carried residuals up to 8.7e+02.
    """
    problem = _smoke_problem()
    result = _solve(problem)
    assert result.converged is True, result.metadata["convergence_reason"]
    assert result.metadata["inner_hjb_failures"] == []
    residuals = _hjb_residual_per_step(problem, np.asarray(result.U, dtype=float), np.asarray(result.M, dtype=float))
    assert residuals.max() < 1e-4, (
        f"the returned arrays leave discrete HJB residuals {np.array2string(residuals, precision=2)}"
    )


def test_the_nd_path_records_its_non_converged_steps_too():
    """A 2-D value-iteration solve with a one-iteration budget records every time step; a converging one records none."""
    grid = TensorProductGrid(bounds=[(0.0, 1.0)] * 2, Nx_points=[9, 7], boundary_conditions=no_flux_bc(dimension=2))
    hamiltonian = SeparableHamiltonian(
        control_cost=QuadraticControlCost(control_cost=1.0), coupling=lambda m: m, coupling_dm=lambda m: 1.0
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        problem = MFGProblem(
            model=Model(hamiltonian=hamiltonian, sigma=0.3),
            domain=grid,
            conditions=Conditions(m_initial=lambda x: 1.0, u_terminal=lambda x: 0.0, T=0.1),
            Nt=4,
        )
        recorded = {}
        for label, options in (("starved", {"max_newton_iterations": 1, "newton_tolerance": 1e-300}), ("default", {})):
            solver = HJBFDMSolver(problem, solver_type="fixed_point", **options)
            solver.solve_hjb_system(np.ones((5, 9, 7)), np.zeros((9, 7)), np.zeros((5, 9, 7)))
            recorded[label] = solver.inner_solve_failures()
    assert sorted(failure.t_idx for failure in recorded["starved"]) == [0, 1, 2, 3], recorded["starved"]
    assert recorded["default"] == (), recorded["default"]
