"""
Graph coupling operators for network MFG systems.

Defines the GraphCouplingOperator Protocol and concrete implementations
for inter-node coupling in graph-structured Mean Field Games.

Three types of network MFG share the same coupling structure:
- Type 1 (finite-state ODE): NetworkHJBSolver handles internally
- Type 2 (PDE at each node, graph-coupled): uses GraphCouplingOperator
- Type 3 (hybrid PDE-ODE): Type 2 + RegimeSwitchingIterator

The coupling enters each node's HJB/FP equations via source_term injection:
    HJB_k: -dv^k/dt + H^k(x, Dv^k, m^k) + S_hjb^k = 0
    FP_k:  dm^k/dt - L^k[m^k] + S_fp^k = 0

where S_hjb^k, S_fp^k encode the inter-node interaction.

Issue #961: GraphCouplingOperator Protocol for inter-node coupling.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np

from mfgarchon.types.callable_protocols import GRAPH_COUPLING_SLOTS, bind_user_callable

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray


@runtime_checkable
class GraphCouplingOperator(Protocol):
    """Protocol for inter-node coupling in graph MFG systems.

    Given the current state of all N nodes (value functions and densities),
    produces source_term callables for each node's HJB and FP equations.

    The source terms have signature (t: float, x: NDArray) -> NDArray,
    compatible with BaseHJBSolver.solve_hjb_system(source_term=...) and
    BaseFPSolver.solve_fp_system(source_term=...).

    Implementations:
        AdjacencyCoupling: sum_j A_{ij} g(v_j, m_j)
        LaplacianCoupling: L_{ij} v_j (diffusive coupling)
    """

    def compute_hjb_source(
        self,
        node_idx: int,
        values: list[NDArray],
        densities: list[NDArray],
        dt: float,
    ) -> Callable[[float, NDArray], NDArray]:
        """Build HJB source term for node_idx given all nodes' state.

        Args:
            node_idx: Index of the node being solved.
            values: Value functions v^k for all nodes, each shape (Nt+1, Nx).
            densities: Densities m^k for all nodes, each shape (Nt+1, Nx) or (Nx,).
            dt: Time step of the (shared) time grid, used to index into
                values/densities at the physical time the solver requests.

        Returns:
            source_term(t, x) -> NDArray compatible with HJB solver.
        """
        ...

    def compute_fp_source(
        self,
        node_idx: int,
        values: list[NDArray],
        densities: list[NDArray],
        dt: float,
    ) -> Callable[[float, NDArray], NDArray]:
        """Build FP source term for node_idx given all nodes' state.

        Args:
            node_idx: Index of the node being solved.
            values: Value functions v^k for all nodes.
            densities: Densities m^k for all nodes.
            dt: Time step of the (shared) time grid.

        Returns:
            source_term(t, x) -> NDArray compatible with FP solver.
        """
        ...

    @property
    def n_nodes(self) -> int:
        """Number of nodes in the graph."""
        ...


class AdjacencyCoupling:
    """Inter-node coupling via adjacency matrix.

    HJB source for node i: S_i(t, x) = sum_j A_{ij} * g_hjb(v_j(t, x), m_j(t, x))
    FP source for node i:  S_i(t, x) = sum_j A_{ij} * g_fp(v_j(t, x), m_j(t, x))

    The coupling functions g_hjb, g_fp define how neighbor states affect
    each node. Common choices:
    - Value difference: g_hjb(v_j, m_j) = alpha * (v_i - v_j)  (migration incentive)
    - Density interaction: g_fp(v_j, m_j) = beta * m_j  (population inflow)

    Parameters
    ----------
    adjacency : NDArray
        Adjacency matrix A, shape (N, N). A_{ij} > 0 if node j affects node i.
        Can be weighted (interaction strength) or binary.
    coupling_hjb : Callable[..., NDArray] | None
        Custom HJB coupling function, read by the names it declares of t, i, j, v_j and m_j.
        Default: alpha * (v_i_at_t - v_j_at_t) (value difference).
    coupling_fp : Callable[..., NDArray] | None
        Custom FP coupling function, read by the same names.
        Default: beta * (m_j_at_t - m_i_at_t) (mass transfer).
    alpha : float
        Default HJB coupling strength (default 0.1).
    beta : float
        Default FP coupling strength (default 0.1).

    Example
    -------
    >>> # 3-node ring graph
    >>> A = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=float)
    >>> coupling = AdjacencyCoupling(A, alpha=0.1, beta=0.05)
    """

    def __init__(
        self,
        adjacency: NDArray,
        coupling_hjb: Callable | None = None,
        coupling_fp: Callable | None = None,
        alpha: float = 0.1,
        beta: float = 0.1,
    ):
        self._A = np.asarray(adjacency, dtype=float)
        N = self._A.shape[0]
        if self._A.shape != (N, N):
            raise ValueError(f"Adjacency must be square, got {self._A.shape}")
        self._N = N
        self._alpha = alpha
        self._beta = beta
        self._coupling_hjb = (
            None
            if coupling_hjb is None
            else bind_user_callable(coupling_hjb, GRAPH_COUPLING_SLOTS, role="coupling_hjb")
        )
        self._coupling_fp = (
            None if coupling_fp is None else bind_user_callable(coupling_fp, GRAPH_COUPLING_SLOTS, role="coupling_fp")
        )

    @property
    def n_nodes(self) -> int:
        return self._N

    def compute_hjb_source(
        self,
        node_idx: int,
        values: list[NDArray],
        densities: list[NDArray],
        dt: float,
    ) -> Callable[[float, NDArray], NDArray]:
        i = node_idx
        A_row = self._A[i]
        alpha = self._alpha
        custom = self._coupling_hjb

        def source(t_eval: float, x: NDArray) -> NDArray:
            Nx = x.shape[0]
            s = np.zeros(Nx)
            for j in range(len(values)):
                if j == i or A_row[j] == 0:
                    continue
                v_i = _get_time_slice(values[i], t_eval, dt)
                v_j = _get_time_slice(values[j], t_eval, dt)
                m_j = _get_time_slice(densities[j], t_eval, dt)
                if custom is not None:
                    s += A_row[j] * custom(t=t_eval, i=i, j=j, v_j=v_j, m_j=m_j)
                else:
                    # Default: value difference (migration incentive)
                    n = min(len(v_i), len(v_j), Nx)
                    s[:n] += A_row[j] * alpha * (v_i[:n] - v_j[:n])
            return s

        return source

    def compute_fp_source(
        self,
        node_idx: int,
        values: list[NDArray],
        densities: list[NDArray],
        dt: float,
    ) -> Callable[[float, NDArray], NDArray]:
        i = node_idx
        A_row = self._A[i]
        beta = self._beta
        custom = self._coupling_fp

        def source(t_eval: float, x: NDArray) -> NDArray:
            Nx = x.shape[0]
            s = np.zeros(Nx)
            for j in range(len(densities)):
                if j == i or A_row[j] == 0:
                    continue
                v_j = _get_time_slice(values[j], t_eval, dt)
                m_i = _get_time_slice(densities[i], t_eval, dt)
                m_j = _get_time_slice(densities[j], t_eval, dt)
                if custom is not None:
                    s += A_row[j] * custom(t=t_eval, i=i, j=j, v_j=v_j, m_j=m_j)
                else:
                    # Default: mass transfer (inflow from j, outflow to j)
                    n = min(len(m_i), len(m_j), Nx)
                    s[:n] += A_row[j] * beta * (m_j[:n] - m_i[:n])
            return s

        return source


class LaplacianCoupling:
    """Diffusive coupling via graph Laplacian.

    HJB source for node i: S_i(t, x) = kappa * sum_j L_{ij} * v_j(t, x)
    FP source for node i:  S_i(t, x) = kappa * sum_j L_{ij} * m_j(t, x)

    where L = D - A is the graph Laplacian (D = degree matrix).
    This models diffusion of value/density across the network.

    Parameters
    ----------
    adjacency : NDArray
        Adjacency matrix A, shape (N, N).
    kappa : float
        Diffusion strength (default 0.1).
    """

    def __init__(self, adjacency: NDArray, kappa: float = 0.1):
        A = np.asarray(adjacency, dtype=float)
        N = A.shape[0]
        if A.shape != (N, N):
            raise ValueError(f"Adjacency must be square, got {A.shape}")
        # Graph Laplacian: L = D - A
        D = np.diag(A.sum(axis=1))
        self._L = D - A
        self._N = N
        self._kappa = kappa

    @property
    def n_nodes(self) -> int:
        return self._N

    def compute_hjb_source(
        self,
        node_idx: int,
        values: list[NDArray],
        densities: list[NDArray],
        dt: float,
    ) -> Callable[[float, NDArray], NDArray]:
        i = node_idx
        L_row = self._L[i]
        kappa = self._kappa

        def source(t_eval: float, x: NDArray) -> NDArray:
            Nx = x.shape[0]
            s = np.zeros(Nx)
            for j in range(len(values)):
                if L_row[j] == 0:
                    continue
                v_j = _get_time_slice(values[j], t_eval, dt)
                n = min(len(v_j), Nx)
                s[:n] += kappa * L_row[j] * v_j[:n]
            return s

        return source

    def compute_fp_source(
        self,
        node_idx: int,
        values: list[NDArray],
        densities: list[NDArray],
        dt: float,
    ) -> Callable[[float, NDArray], NDArray]:
        i = node_idx
        L_row = self._L[i]
        kappa = self._kappa

        def source(t_eval: float, x: NDArray) -> NDArray:
            Nx = x.shape[0]
            s = np.zeros(Nx)
            for j in range(len(densities)):
                if L_row[j] == 0:
                    continue
                m_j = _get_time_slice(densities[j], t_eval, dt)
                n = min(len(m_j), Nx)
                s[:n] += kappa * L_row[j] * m_j[:n]
            return s

        return source


def _get_time_slice(arr: NDArray, t: float, dt: float) -> NDArray:
    """Extract spatial slice from (Nt+1, Nx) or (Nx,) array at time t.

    Uses `dt` to convert physical time to array index (round-to-nearest).
    Clamped to valid range [0, Nt].
    """
    if arr.ndim == 1:
        return arr
    if dt <= 0:
        return arr[0]
    n = min(round(t / dt), arr.shape[0] - 1)
    n = max(n, 0)
    return arr[n]
