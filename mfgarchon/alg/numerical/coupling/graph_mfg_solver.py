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

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from mfgarchon.alg.numerical.coupling.base_mfg import BaseCouplingIterator
from mfgarchon.alg.numerical.coupling.fixed_point_utils import (
    fp_solver_sig_params,
    resolve_fp_drift_kwargs,
)
from mfgarchon.alg.numerical.coupling.graph_coupling import _get_time_slice

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

    from mfgarchon.alg.numerical.coupling.graph_coupling import GraphCouplingOperator
    from mfgarchon.alg.numerical.fp_solvers.base_fp import BaseFPSolver
    from mfgarchon.alg.numerical.hjb_solvers.base_hjb import BaseHJBSolver
    from mfgarchon.core.mfg_problem import MFGProblem


@dataclass
class GraphMFGResult:
    """Result container for graph MFG solver."""

    values: list[NDArray]
    """Value functions v^k for each node, shape (Nt+1, Nx) each."""

    densities: list[NDArray]
    """Density fields m^k for each node, shape (Nt+1, Nx) each."""

    converged: bool
    """Whether the Picard iteration converged."""

    iterations: int
    """Number of Picard iterations performed."""

    error_history: list[float] = field(default_factory=list)
    """Max error across all nodes per iteration."""

    n_nodes: int = 0
    """Number of nodes in the graph."""


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
        Convergence tolerance on max over nodes of both the value-function and the
        density change, max(|v^k_{n+1} - v^k_n|, |m^k_{n+1} - m^k_n|) (default 1e-5).
    damping : float
        Damping factor for Picard update (default 0.5).

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
        damping: float = 0.5,
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
        self._max_iter = max_iterations
        self._tol = tolerance
        self._damping = damping

        # Validate dimensions
        N = coupling.n_nodes
        if len(problems) != N:
            raise ValueError(f"Need {N} problems for {N} nodes, got {len(problems)}")
        if len(hjb_solvers) != N:
            raise ValueError(f"Need {N} HJB solvers for {N} nodes, got {len(hjb_solvers)}")
        if len(fp_solvers) != N:
            raise ValueError(f"Need {N} FP solvers for {N} nodes, got {len(fp_solvers)}")

        # All nodes must share the same time grid for coupling to be well-defined.
        # dt enters _get_time_slice when the coupling callable is invoked by each
        # HJB/FP solver at its current time step.
        dt0 = problems[0].T / problems[0].Nt
        for k, p in enumerate(problems):
            dt_k = p.T / p.Nt
            if not np.isclose(dt_k, dt0):
                raise ValueError(
                    f"All nodes must share the same dt for graph coupling. "
                    f"Node 0: dt={dt0:.6g}; Node {k}: dt={dt_k:.6g}"
                )
        self._dt = dt0

        self._N = N
        self._last_result: GraphMFGResult | None = None

    def solve(self) -> GraphMFGResult:
        """Run Picard iteration over N-node graph.

        Returns
        -------
        GraphMFGResult
            Contains value functions, densities, convergence info for all nodes.
        """
        N = self._N

        # Initialize: terminal conditions and initial densities
        Us_full = []
        for k in range(N):
            p = self._problems[k]
            u_terminal = p.get_u_terminal()
            U_k = np.zeros((p.Nt + 1, len(u_terminal)))
            U_k[-1] = u_terminal
            Us_full.append(U_k)

        Ms = []
        for k in range(N):
            m_init = self._problems[k].get_m_initial()
            Ms.append(m_init)

        error_history: list[float] = []

        for iteration in range(self._max_iter):
            Us_new: list[NDArray] = [np.empty(0)] * N
            Ms_new: list[NDArray | None] = [None] * N

            # --- HJB step: solve N backward equations with coupling ---
            Ms_expanded = [self._expand_density(k, Ms[k]) for k in range(N)]
            for k in range(N):
                raw_coupling = self._coupling.compute_hjb_source(k, Us_full, Ms_expanded, self._dt)
                hjb_source = self._compose_hjb_source(k, Us_full, Ms_expanded, raw_coupling)

                U_k = self._hjb[k].solve_hjb_system(
                    Ms_expanded[k],
                    Us_full[k][-1],  # terminal condition
                    Us_full[k],  # previous iterate
                    source_term=hjb_source,
                )
                Us_new[k] = U_k

            # --- FP step: solve N forward equations with coupling ---
            for k in range(N):
                raw_coupling = self._coupling.compute_fp_source(k, Us_new, Ms_expanded, self._dt)
                fp_source = self._compose_fp_source(k, Us_new, Ms_expanded, raw_coupling)

                m0_k = Ms[k][0] if isinstance(Ms[k], np.ndarray) and Ms[k].ndim == 2 else Ms[k]
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
                        self._problems[k], fp_sig_params_k, None, Us_new[k], Ms_expanded[k]
                    )
                    fp_kwargs.update(drift_kwargs)
                    if use_positional_U:
                        M_k = self._fp[k].solve_fp_system(m0_k, Us_new[k], **fp_kwargs)
                    else:
                        M_k = self._fp[k].solve_fp_system(m0_k, **fp_kwargs)
                else:
                    M_k = self._fp[k].solve_fp_system(m0_k, Us_new[k], source_term=fp_source)
                Ms_new[k] = M_k

            # --- Damping ---
            theta = self._damping
            for k in range(N):
                Us_new[k] = theta * Us_new[k] + (1 - theta) * Us_full[k]
                if Ms_new[k] is not None:
                    Ms_expanded = self._expand_density(k, Ms[k])
                    if Ms_new[k].shape == Ms_expanded.shape:
                        Ms_new[k] = theta * Ms_new[k] + (1 - theta) * Ms_expanded

            # --- Convergence check ---
            # Gate on BOTH the value function AND the density change (the canonical (u, m)
            # criterion). Previously this checked only U, so a node whose density was still
            # evolving while its value function had stabilized reported converged=True with a
            # non-converged density (same one-field defect as RegimeSwitchingIterator).
            error_U = max(np.max(np.abs(Us_new[k] - Us_full[k])) for k in range(N))
            m_errors = []
            m_comparable = True
            for k in range(N):
                if Ms_new[k] is None:
                    m_comparable = False
                    break
                Ms_exp = self._expand_density(k, Ms[k])
                if Ms_new[k].shape != Ms_exp.shape:
                    m_comparable = False
                    break
                m_errors.append(np.max(np.abs(Ms_new[k] - Ms_exp)))
            error_M = max(m_errors) if (m_comparable and m_errors) else float("inf")
            error = max(error_U, error_M)
            error_history.append(error)

            Us_full = Us_new
            Ms = Ms_new

            if error < self._tol:
                self._last_result = GraphMFGResult(
                    values=Us_full,
                    densities=Ms,
                    converged=True,
                    iterations=iteration + 1,
                    error_history=error_history,
                    n_nodes=N,
                )
                return self._last_result

        self._last_result = GraphMFGResult(
            values=Us_full,
            densities=Ms,
            converged=False,
            iterations=self._max_iter,
            error_history=error_history,
            n_nodes=N,
        )
        return self._last_result

    def _expand_density(self, k: int, m: NDArray) -> NDArray:
        """Expand density to (Nt+1, Nx) if needed."""
        if isinstance(m, np.ndarray) and m.ndim == 2:
            return m
        Nt = self._problems[k].Nt
        return np.tile(m, (Nt + 1, 1))

    def _compose_hjb_source(
        self,
        k: int,
        Us_full: list[NDArray],
        Ms_expanded: list[NDArray],
        coupling_source: Callable[[float, NDArray], NDArray],
    ) -> Callable[[float, NDArray], NDArray]:
        """Compose per-node problem sources with the graph coupling source.

        The graph coupling source from `GraphCouplingOperator` is layered on top
        of any per-node `problem.source_term_hjb` and `problem.nonlocal_operator`
        so that all Layer 1 feature dimensions (Levy, custom sources, graph)
        compose at each node, matching Binding Constraint #1 of the Plan.

        Obstacle / penalty VI composition is handled by wrapping each node's
        HJB solver in PenaltyHJBSolver, not here.
        """
        p = self._problems[k]
        has_problem_source = p.source_term_hjb is not None
        has_nonlocal = p.nonlocal_operator is not None
        if not (has_problem_source or has_nonlocal):
            return coupling_source

        dt = self._dt

        def composed(t: float, x_eval: NDArray) -> NDArray:
            s = coupling_source(t, x_eval)
            if has_problem_source:
                v_t = _get_time_slice(Us_full[k], t, dt)
                m_t = _get_time_slice(Ms_expanded[k], t, dt)
                s = s + p.source_term_hjb(x_eval, m_t, v_t, t)
            if has_nonlocal:
                v_t = _get_time_slice(Us_full[k], t, dt)
                s = s + p.nonlocal_operator @ v_t
            return s

        return composed

    def _compose_fp_source(
        self,
        k: int,
        Us_new: list[NDArray],
        Ms_expanded: list[NDArray],
        coupling_source: Callable[[float, NDArray], NDArray],
    ) -> Callable[[float, NDArray], NDArray]:
        """Compose per-node problem sources with the graph coupling source (FP side)."""
        p = self._problems[k]
        has_problem_source = p.source_term_fp is not None
        if not has_problem_source:
            return coupling_source

        dt = self._dt

        def composed(t: float, x_eval: NDArray) -> NDArray:
            s = coupling_source(t, x_eval)
            v_t = _get_time_slice(Us_new[k], t, dt)
            m_t = _get_time_slice(Ms_expanded[k], t, dt)
            s = s + p.source_term_fp(x_eval, m_t, v_t, t)
            return s

        return composed

    def get_results(self) -> tuple:
        """Get computed solution arrays (required by BaseCouplingIterator)."""
        if self._last_result is not None:
            return self._last_result.values[0], self._last_result.densities[0]
        raise RuntimeError("No solution computed yet. Call solve() first.")

    def validate_solution(self) -> dict[str, Any]:
        """Placeholder for solution validation."""
        return {}
