"""An inner nonlinear iteration that exhausts its budget hands its failure to one owner (#2644).

The owner, `base_hjb.report_inner_solve_failure`, raises a `ConvergenceError` by default; under
`on_inner_failure="record"` it keeps the iterate, records the step for `inner_solve_failures()`, and warns. A coupled
solve records for its sweeps and reads the record at its verdict (#1878). Before, the members said nothing, or wrote a
log line no caller could see, and the record had one producer and one reader, so a `BlockIterator` over failed inner
steps reported `converged=True`.

Each test is an oracle: a verdict, an identity, or a residual recomputed outside the solver.
"""

from __future__ import annotations

import warnings

import pytest

import numpy as np

from mfgarchon import Conditions, MFGProblem, Model
from mfgarchon.alg.numerical.coupling.block_iterators import BlockIterator
from mfgarchon.alg.numerical.coupling.fixed_point_iterator import FixedPointIterator
from mfgarchon.alg.numerical.fp_solvers import FPFDMSolver
from mfgarchon.alg.numerical.hjb_solvers import ConvergenceError, HJBFDMSolver, HJBGFDMSolver
from mfgarchon.alg.numerical.hjb_solvers.base_hjb import InnerSolveFailure
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import BCSegment, BCType, BoundaryConditions, no_flux_bc


def _problem(n=41, nt=40, dimension=1):
    grid = TensorProductGrid(
        bounds=[(0.0, 1.0)] * dimension, Nx_points=[n] * dimension, boundary_conditions=no_flux_bc(dimension=dimension)
    )
    model = Model(
        hamiltonian=SeparableHamiltonian(control_cost=QuadraticControlCost(control_cost=1.0), coupling=lambda m: m),
        volatility=0.4,
    )
    cond = Conditions(
        m_initial=lambda x: np.exp(-40 * (np.asarray(x, dtype=float)[..., 0] - 0.3) ** 2),
        u_terminal=lambda x: 2.0 * (np.asarray(x, dtype=float)[..., 0] - 0.7) ** 2,
        T=0.5,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return MFGProblem(model=model, domain=grid, conditions=cond, Nt=nt)


@pytest.mark.parametrize("coupler", ["fixed point", "block gauss-seidel"])
@pytest.mark.parametrize("budget", [1, 30])
def test_a_coupled_solve_is_not_converged_over_a_failed_inner_step(coupler, budget):
    """At a one-step Newton budget every step of the last sweep fails; at the default budget none does. Measured at
    ee0879d0, the fixed point reported converged=False over the failed steps and BlockIterator converged=True: only the
    fixed point read the record. The solver's own policy is restored after the solve."""
    p = _problem()
    hjb = HJBFDMSolver(p, max_newton_iterations=budget)
    build = {
        "fixed point": lambda: FixedPointIterator(p, hjb, FPFDMSolver(p)),
        "block gauss-seidel": lambda: BlockIterator(p, hjb, FPFDMSolver(p), method="gauss_seidel", relaxation=0.5),
    }[coupler]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result = build().solve(max_iterations=60, tolerance=1e-4, verbose=False)
    failed = hjb.inner_solve_failures()
    assert result.converged == (not failed), (result.converged, len(failed))
    assert bool(failed) == (budget == 1), len(failed)
    assert result.metadata["inner_hjb_failures"] == list(failed)  # on every result, converged or not (#2644)
    assert hjb.on_inner_failure == "raise"


def test_newtons_result_lists_its_inner_failures_as_every_coupler_does():
    """Newton returns ``(U, M, info)``, and ``info`` carries ``inner_hjb_failures`` in the shape every coupler's
    metadata does, whatever the verdict: here a one-step budget, whose failed steps it lists (#2644 review 2, F4)."""
    from mfgarchon.alg.numerical.coupling.newton_mfg_solver import NewtonMFGSolver

    p = _problem(n=9, nt=4)
    hjb = HJBFDMSolver(p, max_newton_iterations=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, _, info = NewtonMFGSolver(p, hjb, FPFDMSolver(p), picard_warmup=1).solve(
            max_iterations=2, tolerance=1e-4, verbose=False
        )
    assert isinstance(info["inner_hjb_failures"], list), info
    assert info["inner_hjb_failures"], info["convergence_reason"]


def _terminal_problem(terminal):
    """The 1-D problem of `_problem` with the given terminal: steep (30(x-0.7)^2) fails 2 steps at a 3-step budget."""
    model = Model(
        hamiltonian=SeparableHamiltonian(control_cost=QuadraticControlCost(control_cost=1.0), coupling=lambda m: m),
        volatility=0.4,
    )
    cond = Conditions(
        m_initial=lambda x: np.exp(-40 * (np.asarray(x, dtype=float)[..., 0] - 0.3) ** 2),
        u_terminal=lambda x: terminal(np.asarray(x, dtype=float)[..., 0]),
        T=0.5,
    )
    grid = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[41], boundary_conditions=no_flux_bc(dimension=1))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return MFGProblem(model=model, domain=grid, conditions=cond, Nt=40)


@pytest.mark.parametrize("coupler", ["graph", "regime switching", "multi-population"])
def test_a_solver_shared_by_two_nodes_is_read_for_each_of_its_solves(coupler):
    """The repo builds `GraphMFGSolver([p, p], ..., [h, h], ...)`: one HJB solver object for two nodes. Its latest record
    is the second node's, so with a steep terminal at node 0 (2 failed steps at a 3-step budget) and a flat one at node
    1 (none), reading each solver's latest record certified the result, converged=True at 658d143f (#2644 review 1,
    M1). The sweep's record is each solve's, so the two failed steps of node 0 are read and the result is refused."""
    from mfgarchon.alg.numerical.coupling.graph_coupling import AdjacencyCoupling
    from mfgarchon.alg.numerical.coupling.graph_mfg_solver import GraphMFGSolver
    from mfgarchon.alg.numerical.coupling.multi_population_iterator import MultiPopulationIterator
    from mfgarchon.alg.numerical.coupling.regime_switching_iterator import RegimeSwitchingIterator
    from mfgarchon.core.multi_population import MultiPopulationProblem
    from mfgarchon.core.regime_switching import RegimeSwitchingConfig

    steep, flat = _terminal_problem(lambda x: 30.0 * (x - 0.7) ** 2), _terminal_problem(lambda x: 0.0 * x)
    hjb = HJBFDMSolver(steep, max_newton_iterations=3)
    fps = [FPFDMSolver(steep), FPFDMSolver(flat)]
    build = {
        "graph": lambda: GraphMFGSolver(
            [steep, flat], AdjacencyCoupling(0.05 * (np.ones((2, 2)) - np.eye(2))), [hjb, hjb], fps
        ),
        "regime switching": lambda: RegimeSwitchingIterator(
            [steep, flat],
            RegimeSwitchingConfig(transition_matrix=np.array([[-0.1, 0.1], [0.2, -0.2]])),
            [hjb, hjb],
            fps,
        ),
        "multi-population": lambda: MultiPopulationIterator(
            MultiPopulationProblem(populations=[steep, flat], population_names=["steep", "flat"]), [hjb, hjb], fps
        ),
    }[coupler]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = build().solve()
    assert not result.converged
    assert len(result.metadata["inner_hjb_failures"]) == 2, result.metadata.get("convergence_reason")


class _ForwardingWrapper:
    """A supported duck-typed HJB solver that forwards its report to the solver it wraps, so it logs no solve itself."""

    def __init__(self, inner):
        self.inner, self.problem = inner, inner.problem

    def solve_hjb_system(self, *args, **kwargs):
        return self.inner.solve_hjb_system(*args, **kwargs)

    def inner_solve_failures(self):
        return self.inner.inner_solve_failures()

    @property
    def on_inner_failure(self):
        return self.inner.on_inner_failure

    @on_inner_failure.setter
    def on_inner_failure(self, policy):
        self.inner.on_inner_failure = policy


class _ReportsAFailureItsLogDoesNotHold(HJBFDMSolver):
    """Solves and logs for real, and reports a failure no solve recorded."""

    def inner_solve_failures(self):
        return (InnerSolveFailure(t_idx=0, residual=1.0, tolerance=1e-6, steps=1, reason="fabricated", measure="m"),)


@pytest.mark.parametrize("case", ["one slot, read by its report", "a shared wrapper", "a shared report not its log"])
def test_a_solvers_report_decides_and_a_shared_solver_is_read_only_through_it(case):
    """`inner_solve_failures()` is the report (#2644 review 2, F2 and F3). An object in one slot is read by it: at
    fc43d3d4 a BlockIterator read the private record instead, and certified over the failure the repo's own double
    reports. An object in several slots is read from the log only where the log is that report's history; a forwarding
    wrapper that logs nothing, shared, certified at 658d143f and fc43d3d4, and a report the log does not hold would
    have been read past. Both are refused by name instead."""
    from mfgarchon.alg.numerical.coupling.graph_coupling import AdjacencyCoupling
    from mfgarchon.alg.numerical.coupling.graph_mfg_solver import GraphMFGSolver
    from tests.unit.test_alg.test_picard_reports_failed_inner_solves_1878 import _ReportsOnSomeSolves, _smoke_problem

    if case == "one slot, read by its report":
        problem = _smoke_problem()
        hjb = _ReportsOnSomeSolves(problem, lambda call: True)
        solver = BlockIterator(problem, hjb, FPFDMSolver(problem), method="gauss_seidel", relaxation=0.5)
        expected = "inner_hjb_not_converged"
        kwargs = {"max_iterations": 150, "tolerance": 1e-3, "verbose": False}
    else:
        steep, flat = _terminal_problem(lambda x: 30.0 * (x - 0.7) ** 2), _terminal_problem(lambda x: 0.0 * x)
        if case == "a shared wrapper":
            hjb = _ForwardingWrapper(HJBFDMSolver(steep, max_newton_iterations=3))
        else:
            steep = _terminal_problem(lambda x: 0.0 * x)
            hjb = _ReportsAFailureItsLogDoesNotHold(steep)
        coupling = AdjacencyCoupling(0.05 * (np.ones((2, 2)) - np.eye(2)))
        solver = GraphMFGSolver([steep, flat], coupling, [hjb, hjb], [FPFDMSolver(steep), FPFDMSolver(flat)])
        expected = "cannot be read one by one"
        kwargs = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = solver.solve(**kwargs)
    assert not result.converged
    assert expected in result.metadata["convergence_reason"], result.metadata["convergence_reason"]


def test_a_failure_with_no_record_begun_is_refused_under_record():
    """A member that forgot to begin its record raised standalone, but inside a coupler it warned and dropped the
    failure, so the coupler read an empty record and certified (#2644 review 1, S9)."""
    from mfgarchon.alg.numerical.hjb_solvers.base_hjb import InnerSolveFailure, report_inner_solve_failure

    failure = InnerSolveFailure(t_idx=0, residual=1.0, tolerance=1e-6, steps=1, reason="budget exhausted", measure="m")
    for policy in ("record", "fallback"):
        with pytest.raises(RuntimeError, match="began none"):
            report_inner_solve_failure(failure, policy=policy, record=None, solver="AMember")


def test_the_1d_newton_refuses_an_exhausted_budget_by_default():
    """At a one-step budget HJB-FDM's 1-D Newton exhausts its first step, t_idx 39. By default that is a
    ConvergenceError, not a returned iterate; no test held the 1-D raise before (#2644 review 1, M2)."""
    x = np.linspace(0.0, 1.0, 41)
    with pytest.raises(ConvergenceError, match="did not converge at t_idx=39"):
        HJBFDMSolver(_problem(), max_newton_iterations=1).solve_hjb_system(
            np.ones((41, 41)), 2.0 * (x - 0.7) ** 2, np.zeros((41, 41))
        )
    # The module-level entry raises at its own defaults too, where it raised a RuntimeError at fc43d3d4 (#2644 review
    # 2, F1); and it reads its budget through the one check.
    from mfgarchon.alg.numerical.hjb_solvers import base_hjb

    with pytest.raises(ConvergenceError, match="did not converge at t_idx=39"):
        base_hjb.solve_hjb_system_backward(
            np.ones((41, 41)), 2.0 * (x - 0.7) ** 2, np.zeros((41, 41)), _problem(), max_newton_iterations=1
        )
    with pytest.raises(TypeError, match="an integer is required"):
        base_hjb.solve_hjb_system_backward(
            np.ones((41, 41)), 2.0 * (x - 0.7) ** 2, np.zeros((41, 41)), _problem(), max_newton_iterations=20.0
        )


def test_an_owner_raise_inside_the_nd_newton_reaches_the_caller_as_itself(monkeypatch):
    """HJB-FDM's nD Newton catches RuntimeError around the generic solver to fall back or re-raise, and a
    ConvergenceError is a RuntimeError: an owner's raise from inside it was caught and replaced (#2644)."""
    p = _problem(n=9, nt=4, dimension=2)
    solver = HJBFDMSolver(p, solver_type="newton")
    planted = ConvergenceError("an inner solve", 1, 1.0, "raised by an inner solve's owner")

    def raising(*args, **kwargs):
        raise planted

    monkeypatch.setattr(solver, "_evaluate_hamiltonian_nd", raising)
    with pytest.raises(ConvergenceError) as raised:
        solver.solve_hjb_system(np.ones((5, 9, 9)), np.zeros((9, 9)), np.zeros((5, 9, 9)))
    assert raised.value is planted


def test_network_policy_iteration_refuses_an_exhausted_budget_and_records_it_when_asked():
    """It returned its last iterate in silence. A one-iteration budget exhausts every step of this fixture, the default
    none (measured at ee0879d0: 20 of 20, and 0 of 20)."""
    from mfgarchon.alg.numerical.network_solvers.hjb_network import NetworkPolicyIterationHJBSolver
    from mfgarchon.extensions.topology import NetworkMFGComponents, NetworkMFGProblem
    from mfgarchon.geometry.graph.network_geometry import GridNetwork

    net = GridNetwork(width=5, height=1)
    net.create_network()
    comps = NetworkMFGComponents(node_potential_func=lambda t, i: 2.0 * i - 0.4 * i * i)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        problem = NetworkMFGProblem(geometry=net, T=0.5, Nt=20, components=comps)
    m, g = np.ones((21, 5)) / 5, np.array([0.0, 0.0, 0.0, 0.0, 10.0])
    with pytest.raises(ConvergenceError, match="policy iteration budget exhausted"):
        NetworkPolicyIterationHJBSolver(problem, max_policy_iterations=1).solve_hjb_system(M_density=m, U_terminal=g)
    recording = NetworkPolicyIterationHJBSolver(problem, max_policy_iterations=1, on_inner_failure="record")
    with pytest.warns(RuntimeWarning, match="policy iteration budget exhausted"):
        recording.solve_hjb_system(M_density=m, U_terminal=g)
    assert sorted(f.t_idx for f in recording.inner_solve_failures()) == list(range(20))
    default = NetworkPolicyIterationHJBSolver(problem)
    default.solve_hjb_system(M_density=m, U_terminal=g)
    assert default.inner_solve_failures() == ()


def test_the_weak_form_newton_refuses_an_exhausted_budget_and_records_it_when_asked():
    """It logged on a logger that does not propagate and returned the iterate. #2642's P1 problem, at one Newton
    iteration a step: refused by default; under "record", every recorded step is a failure the record names."""
    from mfgarchon.alg.numerical.fem.hjb_fem_solver import HJBFEMSolver
    from mfgarchon.geometry.boundary import dirichlet_bc
    from mfgarchon.geometry.meshes.mesh_1d import Mesh1D

    mesh = Mesh1D(bounds=(0.0, 1.0), num_elements=16)
    mesh.generate_mesh()
    mesh.boundary_conditions = dirichlet_bc(value=0.7, dimension=1)
    model = Model(hamiltonian=SeparableHamiltonian(control_cost=QuadraticControlCost(control_cost=1.0)), volatility=0.4)
    cond = Conditions(
        m_initial=lambda x: 1.0 + 0.0 * np.asarray(x, dtype=float)[..., 0], u_terminal=lambda x: 0.0, T=0.5
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fem = HJBFEMSolver(MFGProblem(model=model, domain=mesh, conditions=cond, Nt=8), order=1)
    size = fem._disc.dof_coordinates.shape[0]
    args = {
        "M_density": np.ones((9, size)),
        "U_terminal": np.zeros(size),
        "use_newton": True,
        "max_newton_iterations": 1,
    }
    with pytest.raises(ConvergenceError, match="Newton iteration budget exhausted"):
        fem.solve_hjb_system(**args)
    fem.on_inner_failure = "record"
    with pytest.warns(RuntimeWarning, match="Newton iteration budget exhausted"):
        fem.solve_hjb_system(**args)
    assert fem.inner_solve_failures(), "a one-iteration budget recorded no step"
    # a budget below 1 took no step and handed back the terminal value as every step's answer (#2644 review 1, S1)
    with pytest.raises(ValueError, match="budget of at least 1"):
        fem.solve_hjb_system(**{**args, "max_newton_iterations": 0})


def test_hjb_gfdm_records_the_residual_of_the_step_it_returns():
    """The recorded residual is the returned step's, with its boundary rows, recomputed here outside the step: at the
    recorded level, from the returned values (#2599's fixture, the constant terminal, one Newton iteration a step)."""
    L, T, NX, NT = 2.0, 1.0, 21, 10
    x = np.linspace(0.0, L, NX)
    bc = BoundaryConditions(
        dimension=1,
        segments=[
            BCSegment(name="a", bc_type=BCType.DIRICHLET, value=0.7, boundary="x_min"),
            BCSegment(name="b", bc_type=BCType.DIRICHLET, value=0.4, boundary="x_max"),
        ],
    )
    grid = TensorProductGrid(bounds=[(0.0, L)], Nx_points=[NX], boundary_conditions=bc)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        problem = MFGProblem(
            model=Model(
                hamiltonian=SeparableHamiltonian(control_cost=QuadraticControlCost(control_cost=1.0)), volatility=0.5
            ),
            domain=grid,
            conditions=Conditions(m_initial=lambda z: 1.0, u_terminal=lambda z: 0.0, T=T),
            Nt=NT,
        )
    solver = HJBGFDMSolver(
        problem,
        collocation_points=x.reshape(-1, 1),
        delta=3.0 * L / (NX - 1),
        monotonicity_scheme="none",
        max_newton_iterations=1,
        on_inner_failure="record",
    )
    U_T, M = np.full(NX, 0.1), np.ones((NT + 1, NX)) / L
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        U = np.asarray(solver.solve_hjb_system(M_density=M, U_terminal=U_T, U_coupling_prev=np.tile(U_T, (NT + 1, 1))))
    failures = solver.inner_solve_failures()
    assert len(failures) == NT, len(failures)
    H = solver.problem.hamiltonian_class
    for failure in failures:
        n = failure.t_idx
        grad, lap = solver._compute_derivatives_vectorized(U[n])
        pde = solver._compute_hjb_residual_hamiltonian(U[n], U[n + 1], M[n], grad, lap, H, n * T / NT)
        _, residual = solver._apply_boundary_conditions_to_sparse_system(None, pde, n, U[n])
        np.testing.assert_allclose(failure.residual, np.linalg.norm(residual), rtol=1e-12)
        assert failure.residual >= solver.newton_tolerance
