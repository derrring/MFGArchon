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

import pytest

import numpy as np

from mfgarchon import Conditions, MFGProblem, Model
from mfgarchon.alg.numerical.coupling import FixedPointIterator
from mfgarchon.alg.numerical.fp_solvers import FPFDMSolver
from mfgarchon.alg.numerical.hjb_solvers import HJBFDMSolver, base_hjb
from mfgarchon.alg.numerical.hjb_solvers.base_hjb import InnerSolveFailure
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
        # An aggregating coupling (a negative cost, #2375 ruling 3): this file's measurements were taken on it.
        control_cost=QuadraticControlCost(control_cost=1.0),
        coupling=lambda m: -m,
        coupling_dm=lambda m: -1.0,
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


@pytest.mark.parametrize("stop", ["criteria", "callback"])
def test_a_fixed_point_over_inner_solves_that_cannot_converge_is_not_reported_converged(stop: str):
    """Both exits that evaluate the criteria refuse: the criteria check, and a callback that stops the loop.

    The callback runs before the criteria check in each sweep, so it can only stop a sweep that meets them if
    they are met at once: a tolerance of 10 does that at the first sweep.
    """
    problem = _smoke_problem()
    iterator = FixedPointIterator(
        problem,
        hjb_solver=HJBFDMSolver(problem, newton_tolerance=1e-300, max_newton_iterations=3),
        fp_solver=FPFDMSolver(problem),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if stop == "callback":
            result = iterator.solve(
                max_iterations=150, tolerance=10.0, iteration_callback=lambda *_: False, verbose=False
            )
        else:
            result = iterator.solve(max_iterations=150, verbose=False)
    failures = result.metadata["inner_hjb_failures"]
    assert result.converged is False, "Picard reported convergence over inner solves that cannot meet their tolerance"
    assert "inner_hjb_not_converged" in result.metadata["convergence_reason"], result.metadata["convergence_reason"]
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
        # An aggregating coupling (a negative cost, #2375 ruling 3): this file's measurements were taken on it.
        control_cost=QuadraticControlCost(control_cost=1.0),
        coupling=lambda m: -m,
        coupling_dm=lambda m: -1.0,
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


class _ReportsOnSomeSolves(HJBFDMSolver):
    """Solves for real and REPORTS a fabricated failure on the solves ``rule`` picks, by call index."""

    def __init__(self, problem, rule):
        super().__init__(problem)
        self.rule, self.calls, self.flags = rule, 0, []

    def solve_hjb_system(
        self, M_density, U_terminal, U_coupling_prev, volatility_field=None, source_term=None, **kwargs
    ):
        out = super().solve_hjb_system(
            M_density, U_terminal, U_coupling_prev, volatility_field=volatility_field, source_term=source_term, **kwargs
        )
        self.flags.append(self.rule(self.calls))
        self.calls += 1
        return out

    def inner_solve_failures(self):
        if not self.flags or not self.flags[-1]:
            return ()
        return (InnerSolveFailure(t_idx=0, residual=1.0, tolerance=1e-6, steps=1, reason="fabricated"),)


@pytest.mark.parametrize(
    ("label", "rule"),
    [
        ("first_solve_only", lambda call: call == 0),
        ("odd_solves", lambda call: call % 2 == 1),
        ("every_solve", lambda call: True),
    ],
)
def test_the_verdict_reads_the_sweep_that_met_the_criteria(label, rule):
    """The report of the sweep that stopped the loop decides: not the first sweep's, not the previous sweep's.

    ``first_solve_only`` and ``odd_solves`` flag an earlier sweep and must not refuse (at this tolerance the
    stopping solve has an even index); ``every_solve`` flags the stopping sweep too and must refuse.
    """
    problem = _smoke_problem()
    hjb = _ReportsOnSomeSolves(problem, rule)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = FixedPointIterator(problem, hjb_solver=hjb, fp_solver=FPFDMSolver(problem)).solve(
            max_iterations=150, tolerance=1e-3, verbose=False
        )
    assert hjb.flags, "the stub never solved"
    assert result.converged is (not hjb.flags[-1]), (label, len(hjb.flags), result.metadata["convergence_reason"])


class _DuckTypedHJB:
    """A plain object with ``solve_hjb_system``: the coupling layer duck-types, so this is a supported solver."""

    def __init__(self, problem):
        self.inner = HJBFDMSolver(problem)

    def solve_hjb_system(
        self, M_density, U_terminal, U_coupling_prev, volatility_field=None, source_term=None, **kwargs
    ):
        return self.inner.solve_hjb_system(
            M_density, U_terminal, U_coupling_prev, volatility_field=volatility_field, source_term=source_term, **kwargs
        )


def test_a_solver_without_the_report_is_not_tracked_rather_than_an_error():
    problem = _smoke_problem()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = FixedPointIterator(problem, hjb_solver=_DuckTypedHJB(problem), fp_solver=FPFDMSolver(problem)).solve(
            max_iterations=3, verbose=False
        )
    assert result.metadata["inner_hjb_failures"] is None
