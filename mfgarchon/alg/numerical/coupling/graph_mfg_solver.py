"""
Graph MFG Solver — Picard iteration over N-node graph with per-node PDE solvers.

Solves N coupled MFG systems (one per graph node) with inter-node coupling
injected via GraphCouplingOperator source terms:

    HJB_k: -dv^k/dt + H^k(x, Dv^k, m^k) + S_hjb^k(all nodes) = 0
    FP_k:  dm^k/dt - L^k[m^k] + S_fp^k(all nodes) = 0

Each node has its own MFGProblem, HJB solver, and FP solver. The graph
coupling enters via source_term injection — no solver modification needed.

This is the **Type 2** network MFG solver:
- Type 1 (finite-state ODE): NetworkHJBSolver (existing)
- Type 2 (PDE at each node, graph-coupled): **this module**
- Type 3 (hybrid): Type 2 + RegimeSwitchingIterator per node

Issue #962: GraphMFGSolver — unified orchestrator for network MFG types 2 and 3.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from mfgarchon.alg.numerical.coupling.base_mfg import (
    BaseCouplingIterator,
    assert_paired_solver_geometry,
    assert_paired_solver_sigma,
    damping_argument,
    hjb_terminal_binding,
    refuse_a_capped_exit_in_a_coupled_solve,
)
from mfgarchon.alg.numerical.coupling.fixed_point_utils import (
    BUDGET_SPENT,
    check_convergence_criteria,
    fp_output_stop,
    fp_solver_sig_params,
    resolve_fp_drift_kwargs,
    verdict_reads_inner_failures,
)
from mfgarchon.alg.numerical.coupling.graph_coupling import _get_time_slice
from mfgarchon.alg.numerical.coupling.source_composition import (
    _problem_hjb_source_terms,
    coupling_binding_for,
    hjb_source_fields,
    hjb_source_is_active,
)
from mfgarchon.config.damping import DampingPolicy
from mfgarchon.core.mfg_problem import _call_problem_source
from mfgarchon.types.callable_protocols import evaluate_solver_source
from mfgarchon.utils.convergence import sweep_change, worst_sweep_change
from mfgarchon.utils.mfg_logging import get_logger

from .coupled_result import MassInvariant, SweepRecord, coupled_result, refuse_components_off_one_grid
from .damping_run import DampingRun
from .fixed_point_utils import diverged_value_function
from .iteration_state import IterationState

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

    from mfgarchon.alg.numerical.coupling.graph_coupling import GraphCouplingOperator
    from mfgarchon.alg.numerical.fp_solvers.base_fp import BaseFPSolver
    from mfgarchon.alg.numerical.hjb_solvers.base_hjb import BaseHJBSolver
    from mfgarchon.core.mfg_problem import MFGProblem
    from mfgarchon.utils.solver_result import SolverResult

logger = get_logger(__name__)


class GraphMFGSolver(BaseCouplingIterator):
    """Picard iteration over N-node graph with per-node MFG solvers.

    Each node runs its own HJB-FP solver pair. Inter-node coupling
    is provided by a GraphCouplingOperator that produces source_term
    callables for each node's HJB and FP equations.

    This generalizes RegimeSwitchingIterator: instead of a fixed
    transition matrix Q, any GraphCouplingOperator can define the
    inter-node interaction (adjacency, Laplacian, custom).

    Parameters
    ----------
    problems : list[MFGProblem]
        One MFGProblem per graph node. Each defines its own
        Hamiltonian, sigma, geometry, etc.
    coupling : GraphCouplingOperator
        Inter-node coupling operator (AdjacencyCoupling,
        LaplacianCoupling, or custom).
    hjb_solvers : list[BaseHJBSolver]
        One HJB solver per node.
    fp_solvers : list[BaseFPSolver]
        One FP solver per node.
    max_iterations : int
        Maximum Picard iterations (default 50).
    tolerance : float
        Bound on the largest relative L2 change of a sweep, over nodes and over the value function and
        the density, each node's in its own problem's measure; the change is the sweep's output before
        damping against its input (default 1e-5; docs/user/CONVENTIONS.md § 9, #2555).
    damping : DampingPolicy | None
        Each node's damping of U and M, a `DampingPolicy` (#2997), applied per node: every node's fields adapt on
        their own history. None is ``DampingPolicy.constant(0.5)``.
    absolute_tolerance : float | None
        If given, the largest absolute L2 change must also fall below it (default None).

    Example
    -------
    >>> from mfgarchon.alg.numerical.coupling.graph_coupling import AdjacencyCoupling
    >>> A = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=float)
    >>> coupling = AdjacencyCoupling(A, alpha=0.1, beta=0.05)
    >>> solver = GraphMFGSolver(
    ...     problems=[p1, p2, p3],
    ...     coupling=coupling,
    ...     hjb_solvers=[hjb1, hjb2, hjb3],
    ...     fp_solvers=[fp1, fp2, fp3],
    ... )
    >>> result = solver.solve()
    """

    def __init__(
        self,
        problems: list[MFGProblem],
        coupling: GraphCouplingOperator,
        hjb_solvers: list[BaseHJBSolver],
        fp_solvers: list[BaseFPSolver],
        max_iterations: int = 50,
        tolerance: float = 1e-5,
        damping: DampingPolicy | None = None,
        absolute_tolerance: float | None = None,
    ):
        # Use first problem as representative for base class
        super().__init__(problems[0])
        self._problems = problems
        self._coupling = coupling
        self._hjb = hjb_solvers
        self._fp = fp_solvers
        # Issue #1315: cache each per-node FP solver's solve_fp_system signature so the FP step
        # can route the value function through the drift-convention dispatcher
        # (resolve_fp_drift_kwargs), exactly as the single-solver iterators do via
        # _init_solver_signatures. One set per node because nodes may use different FP solvers.
        self._fp_sig_params_k = [fp_solver_sig_params(fp) for fp in fp_solvers]
        # Issue #1489 (S1): per-node FP drift-input convention, parallel to _fp_sig_params_k, so
        # resolve_fp_drift_kwargs routes each node by convention rather than param presence.
        self._fp_drift_convention_k = [getattr(fp, "_drift_convention", None) for fp in fp_solvers]
        self._max_iter = max_iterations
        self._tol = tolerance
        self._abs_tol = absolute_tolerance
        self.damping = damping_argument(damping, "GraphMFGSolver") or DampingPolicy.constant(0.5)

        # Validate dimensions
        N = coupling.n_nodes
        if len(problems) != N:
            raise ValueError(f"Need {N} problems for {N} nodes, got {len(problems)}")
        if len(hjb_solvers) != N:
            raise ValueError(f"Need {N} HJB solvers for {N} nodes, got {len(hjb_solvers)}")
        if len(fp_solvers) != N:
            raise ValueError(f"Need {N} FP solvers for {N} nodes, got {len(fp_solvers)}")
        # RFC #1574 C14 / Issue #1603: each node HJB-FP pair is an adjoint pair; guard every pair.
        for _k in range(N):
            assert_paired_solver_sigma(hjb_solvers[_k], fp_solvers[_k], f"GraphMFGSolver[node {_k}]")
            assert_paired_solver_geometry(
                hjb_solvers[_k], fp_solvers[_k], f"GraphMFGSolver[node {_k}]", problem=problems[_k]
            )

        # dt enters _get_time_slice when the coupling callable is invoked by each HJB/FP solver at its current time
        # step; the nodes share one time grid, which the solve refuses otherwise (refuse_components_off_one_grid).
        self._dt = problems[0].T / problems[0].Nt

        self._N = N
        self._last_result: SolverResult | None = None

    @verdict_reads_inner_failures(lambda self: list(self._hjb), fp_solvers_of=lambda self: list(self._fp))
    def solve(self) -> SolverResult:
        """Run Picard iteration over N-node graph.

        Returns
        -------
        SolverResult
            U and M of shape (N, Nt+1, *shape), node k's at ``U[k]`` and ``M[k]``, and ``components`` N (#2997). Its
            ``mass_conservation_error`` is the total mass's drift, since a coupling can move mass between nodes. The
            total itself moves by design where the adjacency's weighted in- and out-degrees differ at some node, or a
            custom ``coupling_fp`` adds mass (`MassInvariant.TOTAL`); each node's own drift is in
            ``metadata["per_component"]``. On a diverged HJB, each node the sweep solved holds
            that HJB output as its U, the diverged node's with its NaN kept (#1718); the later nodes' U, and every M,
            are the iterate after the last finite sweep.
        """
        refuse_a_capped_exit_in_a_coupled_solve(self._fp, "GraphMFGSolver.solve")
        N = self._N

        # Each node's iterate from its one owner (#2997): its layout, the grid's shape or the pair's DOF count on a mesh,
        # and its cold start, u(T) at m(0) in every row of U and m(0) with the FP's prescribed values in M (#2991). Its
        # u(T) is read through the node's HJB solver's binding, so a terminal coupling solves off a grid (#1642).
        states = [
            IterationState(
                self._problems[k],
                self._hjb[k],
                self._fp[k],
                reader=f"GraphMFGSolver[node {k}]",
                terminal_binding=hjb_terminal_binding(self._hjb[k]),
                unstructured_mesh=True,
            )
            for k in range(N)
        ]
        refuse_components_off_one_grid(
            self._problems, [state.solution_shape for state in states], reader="GraphMFGSolver", component="node"
        )
        Us_full, Ms = [], []
        for state in states:
            U_k, M_k = state.start()
            Us_full.append(U_k)
            Ms.append(M_k)
        # One damping run per node: each node's fields adapt on that node's own history (#2997 rulings 5 and 9).
        runs = [DampingRun(self.damping) for state in states]

        record = SweepRecord(N)
        fp_output_available = False
        # The reason a spent budget reports, on every route (#3002, ruling 2)
        reason = BUDGET_SPENT
        integrates = [p.spatial_measure().integrate for p in self._problems]
        dts = [p.dt for p in self._problems]

        for iteration in range(self._max_iter):
            # The owner's refusals answer at every sweep, so a BC changed during the solve is refused at the next one
            # (#3005; #3002 stage 2a, ruling A). Stage 2b moves this call into the coupled driver.
            for state in states:
                state.read_owner()
            Us_new: list[NDArray] = [np.empty(0)] * N
            Ms_new: list[NDArray] = [np.empty(0)] * N
            terminals: list[NDArray] = [np.empty(0)] * N

            # --- HJB step: solve N backward equations with coupling ---
            for k in range(N):
                raw_coupling = self._coupling.compute_hjb_source(k, Us_full, Ms, self._dt)
                hjb_source = self._compose_hjb_source(k, Us_full, Ms, raw_coupling)
                # u(T) at the density this node's HJB solve uses (#1642 stage 2): the solve and the write-back after
                # the update both read it.
                terminals[k] = states[k].terminal_at(Ms[k][-1])

                U_k = self._hjb[k].solve_hjb_system(
                    Ms[k],
                    terminals[k],  # terminal condition
                    Us_full[k],  # previous iterate
                    source_term=hjb_source,
                )
                Us_new[k] = U_k

                # Issue #1718: check before the next node is solved. Every node's HJB runs before any
                # FP, so one node's NaN would reach every other node through `compute_fp_source`
                # before anything noticed, and the FP solver would then report a CFL condition for
                # what is an HJB failure.
                diverged = diverged_value_function(
                    U_k,
                    terminals[k],
                    site=f"GraphMFGSolver[node {k}]",
                    # `compute_hjb_source` is handed every node's density, whichever of them a coupling reads (#2845).
                    density=np.concatenate([np.ravel(m) for m in Ms]),
                    iteration=iteration,
                )
                if diverged is not None:
                    Us_new[k] = diverged
                    # Nodes after k were never solved this sweep. Carry the previous iterate for those, a value
                    # function of the node's shape, which is what "this node did not advance" means, so a consumer
                    # looping over values to find the NaN reaches it.
                    values = [Us_new[j] if j <= k else Us_full[j] for j in range(N)]
                    record.diverged()
                    return self._result(values, Ms, record, False, "diverged_nan", fp_output_available)

            # --- FP step: solve N forward equations with coupling ---
            for k in range(N):
                raw_coupling = self._coupling.compute_fp_source(k, Us_new, Ms, self._dt)
                fp_source = self._compose_fp_source(k, Us_new, Ms, raw_coupling)

                m0_k = states[k].m0
                # Issue #1315 (Refs #1043): route the value function through the
                # drift-convention dispatcher instead of passing U as drift_field. After the
                # v0.18.6 rename drift_field is the velocity alpha*, not the U-potential; for a
                # smooth separable H the FP solver must receive U via potential_field and derive
                # the velocity internally. Passing drift_field=U is silent-wrong-equilibrium.
                # Mirrors FixedPointIterator.solve() / fictitious_play (#1299), per-node: each
                # node has its own problem (Hamiltonian/geometry) and FP solver signature.
                fp_kwargs: dict[str, Any] = {"source_term": fp_source}
                fp_sig_params_k = self._fp_sig_params_k[k]
                if fp_sig_params_k is not None:
                    drift_kwargs, use_positional_U = resolve_fp_drift_kwargs(
                        self._problems[k],
                        fp_sig_params_k,
                        None,
                        Us_new[k],
                        Ms[k],
                        drift_convention=self._fp_drift_convention_k[k],
                        fp_solver=self._fp[k],
                    )
                    fp_kwargs.update(drift_kwargs)
                    if use_positional_U:
                        M_k = self._fp[k].solve_fp_system(m0_k, Us_new[k], **fp_kwargs)
                    else:
                        M_k = self._fp[k].solve_fp_system(m0_k, **fp_kwargs)
                else:
                    M_k = self._fp[k].solve_fp_system(m0_k, Us_new[k], source_term=fp_source)
                Ms_new[k] = M_k

            # --- Damping, per node and field (the policy, #2997), then the write-back: level 0 to m(0), and U's
            # terminal row to the u(T) this sweep's HJB solve was given. The next iterate is never an input array
            # (`DampingRun.update`), so the map's outputs stay what the change below measures. ---
            Us_next: list[NDArray] = [np.empty(0)] * N
            Ms_next: list[NDArray] = [np.empty(0)] * N
            for k in range(N):
                U_k, M_k = runs[k].update(iteration, Us_full[k], Us_new[k], Ms[k], Ms_new[k])
                Us_next[k], Ms_next[k] = states[k].write_back(U_k, M_k, terminals[k])
            fp_output_available = True

            # The FP-output rule every coupled route reads, on the nodes' total mass, Graph's invariant (#2997; #3002
            # stage 2a, ruling A), before the change is measured, which refuses a non-finite one (#2578). Stage 2b
            # moves this call into the coupled driver.
            stop = fp_output_stop(
                Us_new, Ms_new, Us_next, Ms_next, Ms, each_component=False, site="GraphMFGSolver", iteration=iteration
            )
            if stop is not None:
                record.diverged()
                return self._result(Us_next, Ms_next, record, False, stop, fp_output_available)

            # --- Convergence: the sweep's output against its input, BEFORE damping ---
            # Gate on BOTH the value function AND the density change (the canonical (u, m) criterion),
            # and on the map's output: after damping the change is `theta` times this one, so the
            # damping factor bought the verdict (#1684 item 7). Each node's change is its relative L2
            # change in its own problem's measure, and the criterion takes the max over nodes (#2555).
            per_node = [sweep_change(Us_new[k], Us_full[k], Ms_new[k], Ms[k], integrates[k], dts[k]) for k in range(N)]
            worst = worst_sweep_change(per_node, "node")
            record.measured(per_node)
            met, criteria_reason = check_convergence_criteria(
                worst["l2distu_rel"],
                worst["l2distm_rel"],
                worst["l2distu_abs"],
                worst["l2distm_abs"],
                self._tol,
                self._abs_tol,
            )
            for k in range(N):
                for message in runs[k].observe(iteration, per_node[k]):
                    logger.warning(f"GraphMFGSolver[node {k}]: {message}")

            Us_full = Us_next
            Ms = Ms_next

            if met:
                return self._result(Us_full, Ms, record, True, criteria_reason, fp_output_available)

        return self._result(Us_full, Ms, record, False, reason, fp_output_available)

    def _result(
        self,
        values: list[NDArray],
        densities: list[NDArray],
        record: SweepRecord,
        converged: bool,
        reason: str,
        fp_output_available: bool,
    ) -> SolverResult:
        self._last_result = coupled_result(
            U=values,
            M=densities,
            record=record,
            converged=converged,
            reason=reason,
            solver_name="GraphMFGSolver",
            damping=self.damping,
            problems=self._problems,
            fp_solvers=self._fp,
            fp_output_available=fp_output_available,
            component_names=[str(k) for k in range(self._N)],
            invariant=MassInvariant.TOTAL,
        )
        return self._last_result

    def _compose_hjb_source(
        self,
        k: int,
        Us_full: list[NDArray],
        Ms: list[NDArray],
        coupling_source: Callable[[float, NDArray], NDArray],
    ) -> Callable[[float, NDArray], NDArray]:
        """Compose per-node problem sources with the graph coupling source.

        The graph coupling source from `GraphCouplingOperator` is layered on top
        of each node problem's own HJB source, the parts `hjb_source_fields` names
        (`source_term_hjb`, a `coupling_cost`, `nonlocal_operator`),
        so that all Layer 1 feature dimensions (Levy, custom sources, graph)
        compose at each node, matching Binding Constraint #1 of the Plan.

        A penalty wrapper (#2002) injected a ``v``-free term through ``source_term``, which cannot carry the
        value function, so it never composed a variational inequality. There is currently NO obstacle
        route for a per-node HJB solver here -- ``constraint=`` exists only on ``HJBFDMSolver``
        (#591), and #2046 tracks threading it through the shared timestep solve.
        """
        p = self._problems[k]
        if not hjb_source_is_active(p):
            return coupling_source
        # A node problem's coupling cost through the problem's own binding, on its grid; off a grid it is refused here,
        # before the first evaluation (#1642 stage 1b-2).
        binding = coupling_binding_for(p, reader="GraphMFGSolver") if "coupling_cost" in hjb_source_fields(p) else None

        dt = self._dt

        def composed(t: float, x_eval: NDArray) -> NDArray:
            s = evaluate_solver_source(coupling_source, t=t, x=x_eval, dimension=p.dimension)
            # The node problem's own parts via the shared single-source primitive (Issue #1382), layered on the
            # graph coupling source and summed in the primitive's own order (source, F's part, nonlocal), which
            # preserves byte-for-byte agreement with the pre-#1382 graph closure. Uses self._dt so the slice index
            # matches the grid couplers.
            for part in _problem_hjb_source_terms(p, Ms[k], Us_full[k], t, x_eval, dt, binding).values():
                s = s + part
            return s

        return composed

    def _compose_fp_source(
        self,
        k: int,
        Us_new: list[NDArray],
        Ms: list[NDArray],
        coupling_source: Callable[[float, NDArray], NDArray],
    ) -> Callable[[float, NDArray], NDArray]:
        """Compose per-node problem sources with the graph coupling source (FP side)."""
        p = self._problems[k]
        has_problem_source = p.source_term_fp is not None
        if not has_problem_source:
            return coupling_source

        dt = self._dt

        def composed(t: float, x_eval: NDArray) -> NDArray:
            s = evaluate_solver_source(coupling_source, t=t, x=x_eval, dimension=p.dimension)
            v_t = _get_time_slice(Us_new[k], t, dt)
            m_t = _get_time_slice(Ms[k], t, dt)
            s = s + _call_problem_source(p, "source_term_fp", t=t, x=x_eval, m=m_t, v=v_t)
            return s

        return composed

    def get_results(self) -> tuple:
        """The last solve's U and M, every node's, of shape (N, Nt+1, *shape) (required by BaseCouplingIterator)."""
        if self._last_result is not None:
            return self._last_result.U, self._last_result.M
        raise RuntimeError("No solution computed yet. Call solve() first.")

    def validate_solution(self) -> dict[str, Any]:
        """Placeholder for solution validation."""
        return {}
