"""
Joint SOCP-constrained GFDM stencil weights.

Constructive discrete comparison principle for the GFDM HJB Laplacian.
The joint SOCP simultaneously enforces:

    A^T L = e_lap                       (Laplacian 2nd-order consistency)
    A^T D[d,:] = e_grad_d                (gradient 2nd-order consistency)
    L_j >= eps_pos for j != center        (M-matrix on -Δ_h)
    ||D[:, j]||_2 <= C * L_j / h_i        (per-edge cone bound)

The cone closes the comparison-principle proof for the central GFDM scheme
via per-edge absorption. Reference: forthcoming paper §sec:error_structure
(Theorem `thm:joint_socp_feasibility`, Lemma `lem:wendland_stencil_ratio`).

Scope of the guarantee (Issue #1074):
    These constraints are enforced PER STENCIL — each collocation row's
    off-diagonal Laplacian weights satisfy the sign/dominance constraints
    in isolation. This does NOT in general guarantee that the ASSEMBLED HJB
    iteration matrix (I/dt - D*L + alpha*D_grad) is an M-matrix: the signed
    drift term alpha*D_grad can flip an off-diagonal positive once |alpha|
    is large relative to the diffusion D (a Peclet-like condition). The
    discrete maximum principle for the assembled system is therefore NOT
    a-priori guaranteed by per-stencil feasibility. Use
    `monotonicity_enforcer.verify_assembled_m_matrix(...)` (and the
    `critical_drift_for_dmp(...)` threshold) to check the assembled M-matrix
    / DMP property for a specific problem and discretization.

Solver: cvxpy + CLARABEL.

Implementation history: this module ports the audit-major
`gfdm_monotonicity_audit/shared/socp.py` validation experiment into
mfgarchon proper (Issue #XXXX, v0.18.0+ Phase 1B).
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

try:
    import cvxpy as cp

    _CVXPY_AVAILABLE = True
except ImportError:
    cp = None
    _CVXPY_AVAILABLE = False


# =============================================================================
# Wendland kernel + Taylor matrix helpers
# =============================================================================


def wendland_phi31(q: np.ndarray | float) -> np.ndarray | float:
    """Wendland $\\phi_{3,1}$ kernel: $(1-q)_+^4 (4q + 1)$.

    Compactly supported on $[0, 1]$, twice continuously differentiable
    at the support boundary. Standard Wendland C^2 kernel for
    GFDM/RBF-FD methods.
    """
    q = np.asarray(q)
    pos = np.maximum(1.0 - q, 0.0)
    return (pos**4) * (4.0 * q + 1.0)


def wendland_stencil_weights(offsets: np.ndarray, delta: float) -> np.ndarray:
    """Per-neighbor Wendland weights at distance $r_j = \\|\\text{offset}_j\\|$
    over support radius $\\delta$.

    Args:
        offsets: shape (n,) for 1D or (n, d) for d-dim
        delta:   support radius (kernel evaluated at $r/\\delta$)

    Returns:
        weights shape (n,), with $w_j = \\phi_{3,1}(\\|\\text{offset}_j\\|/\\delta)$,
        floored at 1e-12 to avoid singular weighted-LSQ.
    """
    offsets = np.asarray(offsets)
    if offsets.ndim == 1:
        r = np.abs(offsets)
    else:
        r = np.linalg.norm(offsets, axis=-1)
    w = wendland_phi31(r / float(delta))
    return np.maximum(w, 1e-12)


def build_taylor_matrix_1d(offsets: np.ndarray) -> tuple[np.ndarray, list]:
    """Build 2nd-order Taylor matrix A in 1D.

    Row j = (1, dx_j, dx_j^2/2). Column ordering: [(0,), (1,), (2,)].
    """
    multi_indices = [(0,), (1,), (2,)]
    offsets = np.asarray(offsets).reshape(-1)
    n = offsets.shape[0]
    A = np.zeros((n, 3))
    A[:, 0] = 1.0
    A[:, 1] = offsets
    A[:, 2] = 0.5 * offsets**2
    return A, multi_indices


def build_taylor_matrix_2d(offsets: np.ndarray) -> tuple[np.ndarray, list]:
    """Build 2nd-order Taylor matrix A in 2D.

    Row j = (1, dx_j, dy_j, dx_j^2/2, dx_j dy_j, dy_j^2/2).
    Column ordering: [(0,0), (1,0), (0,1), (2,0), (1,1), (0,2)].
    """
    multi_indices = [(0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2)]
    n = offsets.shape[0]
    A = np.zeros((n, 6))
    A[:, 0] = 1.0
    A[:, 1] = offsets[:, 0]
    A[:, 2] = offsets[:, 1]
    A[:, 3] = 0.5 * offsets[:, 0] ** 2
    A[:, 4] = offsets[:, 0] * offsets[:, 1]
    A[:, 5] = 0.5 * offsets[:, 1] ** 2
    return A, multi_indices


# =============================================================================
# Joint SOCP solver — single stencil
# =============================================================================


def solve_joint_socp_at_stencil(
    A: np.ndarray,
    center_idx: int,
    h_i: float,
    C: float,
    eps_pos: float = 0.0,
    solver: str = "CLARABEL",
    dimension: int | None = None,
    wendland_w: np.ndarray | None = None,
) -> dict:
    """Solve the joint SOCP-constrained QP at one stencil. Dimension-agnostic.

    Args:
        A: Taylor matrix shape (n, k). Number of Taylor columns determines the
           ambient dimension: k=3 → 1D, k=6 → 2D. Override via `dimension=`.
        center_idx: index of the center node within the n neighbors.
        h_i: characteristic stencil scale (median neighbor distance).
        C: per-edge kappa upper bound. Smaller = tighter monotonicity.
        eps_pos: minimum required positivity for off-center L_j (default 0.0).
        solver: cvxpy solver name (default CLARABEL, supports SOCP).
        dimension: ambient dimension (1 or 2). If None, inferred from A.shape[1]:
                   3 → 1D, 6 → 2D.
        wendland_w: optional per-neighbor weights, shape (n,). When provided,
            the objective uses $\\sum_j (1/w_j) (L_j^2 + \\|D_{:,j}\\|^2)$ —
            the W^{-1} quadratic that matches mfgarchon's Wendland-Taylor LSQ
            on Taylor coefficients (KKT equivalence).

            Without this, on wide stencils the SOCP picks weights far from
            the Wendland-LSQ pseudo-inverse and degrades numerical accuracy.

            Pass `wendland_stencil_weights(offsets, delta)` to compute.
            None falls back to unweighted (legacy, OK for narrow stencils).

    Returns:
        dict with keys:
            status:    "feasible" | "infeasible" | "solver_error"
            L:         shape (n,) Laplacian weights, or None
            D:         shape (d, n) gradient weights (d = dimension), or None
            kappa_max: max achieved per-edge kappa, or inf
            objective: cvxpy objective value (None for fast-path return)
            via:       "wendland_lsq_fast_path" | "socp_clarabel"
    """
    if not _CVXPY_AVAILABLE:
        return {
            "status": "solver_error",
            "message": "cvxpy not installed; cannot run joint SOCP. pip install cvxpy.",
            "L": None,
            "D": None,
            "kappa_max": np.inf,
            "objective": None,
        }

    n, k = A.shape
    if dimension is None:
        dimension = {3: 1, 6: 2}.get(k)
        if dimension is None:
            raise ValueError(f"Cannot infer dimension from A.shape[1]={k}; pass dimension= explicitly.")

    if dimension == 1:
        e_lap = np.array([0.0, 0.0, 1.0])
        e_grad = [np.array([0.0, 1.0, 0.0])]
    elif dimension == 2:
        e_lap = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 1.0])
        e_grad = [
            np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
        ]
    else:
        raise ValueError(f"Only dimension 1 or 2 supported, got {dimension}")

    # --- Fast path: paper Theorem `thm:joint_socp_feasibility` ---
    # If unconstrained Wendland-LSQ already satisfies the SOCP constraints,
    # return it directly. Avoids ill-conditioned CLARABEL solve at small-h
    # symmetric stencils where L weights scale as 1/h^2 (objective O(1/h^4)).
    if wendland_w is not None:
        try:
            W_diag = np.diag(np.asarray(wendland_w, dtype=float))
            ATA = A.T @ W_diag @ A
            # Issue #1066: use solve() instead of inv() — squares condition number
            # and would silently break SOCP feasibility on marginal stencils.
            # Solve once for [e_lap, e_grad[0], ..., e_grad[d-1]] as columns.
            rhs = np.column_stack([e_lap, *e_grad])  # shape (k, 1+dimension)
            sol = np.linalg.solve(ATA, rhs)  # shape (k, 1+dimension)
            WA = W_diag @ A
            L_lsq = WA @ sol[:, 0]
            D_lsq = (WA @ sol[:, 1:]).T  # shape (dimension, n)

            # Check feasibility
            L_off = np.delete(L_lsq, center_idx)
            m_matrix_ok = bool(np.all(L_off >= eps_pos - 1e-12))
            cone_ok = True
            kappas_lsq = []
            for j in range(n):
                if j == center_idx:
                    continue
                if L_lsq[j] <= 1e-12:
                    cone_ok = False
                    kappas_lsq.append(np.inf)
                    continue
                k_j = h_i * np.linalg.norm(D_lsq[:, j]) / L_lsq[j]
                kappas_lsq.append(k_j)
                if k_j > C + 1e-9:
                    cone_ok = False

            if m_matrix_ok and cone_ok:
                return {
                    "status": "feasible",
                    "L": L_lsq,
                    "D": D_lsq,
                    "kappa_max": float(max(kappas_lsq)) if kappas_lsq else np.nan,
                    "objective": None,
                    "via": "wendland_lsq_fast_path",
                }
        except (np.linalg.LinAlgError, ValueError):
            pass  # fall through to SOCP

    # --- Slow path: cvxpy SOCP ---
    L = cp.Variable(n)
    D = cp.Variable((dimension, n))

    constraints = [e_lap == A.T @ L]
    for d in range(dimension):
        constraints.append(A.T @ D[d, :] == e_grad[d])
    for j in range(n):
        if j == center_idx:
            continue
        constraints.append(L[j] >= eps_pos)
        constraints.append(cp.norm(D[:, j], 2) <= (C / h_i) * L[j])

    if wendland_w is None:
        obj = cp.Minimize(cp.sum_squares(L) + cp.sum_squares(D))
    else:
        if len(wendland_w) != n:
            raise ValueError(f"wendland_w length {len(wendland_w)} != n={n}")
        # Cap 1/w_j to bound conditioning when neighbors are near support edge
        # (q→1, w→0; raw 1/w can blow up to 10^12).
        w = np.asarray(wendland_w, dtype=float)
        MAX_INV_W = 1000.0
        inv_w_raw = 1.0 / w
        inv_w_min = 1.0 / float(w.max())
        inv_w = np.minimum(inv_w_raw, MAX_INV_W * inv_w_min)
        D_sq_per_col = cp.sum(cp.square(D), axis=0)
        obj = cp.Minimize(inv_w @ cp.square(L) + inv_w @ D_sq_per_col)

    prob = cp.Problem(obj, constraints)

    try:
        prob.solve(solver=solver, verbose=False)
    except cp.error.SolverError as e:
        return {
            "status": "solver_error",
            "message": str(e),
            "L": None,
            "D": None,
            "kappa_max": np.inf,
            "objective": None,
        }

    if prob.status not in ("optimal", "optimal_inaccurate"):
        return {
            "status": "infeasible" if prob.status == "infeasible" else prob.status,
            "L": None,
            "D": None,
            "kappa_max": np.inf,
            "objective": None,
        }

    L_val = L.value
    D_val = D.value
    kappas = []
    for j in range(n):
        if j == center_idx:
            continue
        if L_val[j] <= 1e-12:
            kappas.append(np.inf)
        else:
            kappas.append(h_i * np.linalg.norm(D_val[:, j]) / L_val[j])
    kmax = float(np.max(kappas)) if kappas else np.nan

    return {
        "status": "feasible",
        "L": L_val,
        "D": D_val,
        "kappa_max": kmax,
        "objective": float(prob.value),
        "via": "socp_clarabel",
    }


def solve_relaxed_joint_socp_at_stencil(
    A: np.ndarray,
    center_idx: int,
    h_i: float,
    C: float,
    eps_pos: float = 0.0,
    solver: str = "CLARABEL",
    dimension: int | None = None,
    wendland_w: np.ndarray | None = None,
    lambda_M: float = 1.0e4,
    lambda_C: float = 1.0e4,
) -> dict:
    """Always-feasible relaxed joint SOCP.

    Replaces hard constraints `L_j >= 0` and `||D[:,j]|| <= C * L_j / h` with
    slack-penalty soft versions:

        L_j >= -ε_M_j,   ε_M_j >= 0   (penalty: λ_M * ε_M_j²)
        ||D[:,j]|| <= (C/h) * (L_j + ε_C_j),   ε_C_j >= 0   (penalty: λ_C * ε_C_j²)

    The hard equality constraints `A^T L = e_lap`, `A^T D = e_grad` (consistency)
    are preserved. Solution always exists.

    For well-conditioned stencils where `solve_joint_socp_at_stencil` is feasible,
    large penalties (λ_M, λ_C ≥ 1e4) drive ε → 0 and recover the original
    joint_socp solution. For marginally infeasible stencils, the slacks activate
    smoothly, producing a continuous map (cloud geometry → stencil weights).

    This continuity is the key property: it eliminates scheme-switch
    discontinuity between SOCP-feasible and Phase-2 fallback regimes that
    plague hybrid joint_socp + M-matrix-QP architectures on irregular 2D
    clouds (where mirror stencils with similar geometry can land on different
    sides of the SOCP feasibility threshold and receive incompatible weights).

    Returns: dict with same keys as solve_joint_socp_at_stencil + "eps_M_max"
    and "eps_C_max" diagnostics. Status is always "feasible" except on solver
    error.
    """
    if not _CVXPY_AVAILABLE:
        return {
            "status": "solver_error",
            "message": "cvxpy not installed",
            "L": None,
            "D": None,
            "kappa_max": np.inf,
            "objective": None,
        }

    n, k = A.shape
    if dimension is None:
        dimension = {3: 1, 6: 2}.get(k)
        if dimension is None:
            raise ValueError(f"Cannot infer dimension from A.shape[1]={k}")

    if dimension == 1:
        e_lap = np.array([0.0, 0.0, 1.0])
        e_grad = [np.array([0.0, 1.0, 0.0])]
    elif dimension == 2:
        e_lap = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 1.0])
        e_grad = [np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])]
    else:
        raise ValueError(f"Only dimension 1 or 2 supported, got {dimension}")

    L = cp.Variable(n)
    D = cp.Variable((dimension, n))
    eps_M = cp.Variable(n, nonneg=True)  # M-matrix slack
    eps_C = cp.Variable(n, nonneg=True)  # cone slack

    constraints = [e_lap == A.T @ L]
    for d in range(dimension):
        constraints.append(A.T @ D[d, :] == e_grad[d])
    for j in range(n):
        if j == center_idx:
            continue
        # Soft M-matrix: L_j + eps_M_j >= eps_pos  =>  L_j >= eps_pos - eps_M_j
        constraints.append(L[j] + eps_M[j] >= eps_pos)
        # Soft cone: ||D[:,j]||_2 <= (C/h) * (L_j + eps_C_j)
        constraints.append(cp.norm(D[:, j], 2) <= (C / h_i) * (L[j] + eps_C[j]))

    if wendland_w is None:
        base_obj = cp.sum_squares(L) + cp.sum_squares(D)
    else:
        w = np.asarray(wendland_w, dtype=float)
        MAX_INV_W = 1000.0
        inv_w = np.minimum(1.0 / w, MAX_INV_W / float(w.max()))
        D_sq_per_col = cp.sum(cp.square(D), axis=0)
        base_obj = inv_w @ cp.square(L) + inv_w @ D_sq_per_col

    obj = cp.Minimize(base_obj + lambda_M * cp.sum_squares(eps_M) + lambda_C * cp.sum_squares(eps_C))

    prob = cp.Problem(obj, constraints)
    try:
        prob.solve(solver=solver, verbose=False)
    except cp.error.SolverError as e:
        return {
            "status": "solver_error",
            "message": str(e),
            "L": None,
            "D": None,
            "kappa_max": np.inf,
            "objective": None,
        }

    if prob.status not in ("optimal", "optimal_inaccurate"):
        return {"status": prob.status, "L": None, "D": None, "kappa_max": np.inf, "objective": None}

    L_val = L.value
    D_val = D.value
    eps_M_max = float(np.max(eps_M.value))
    eps_C_max = float(np.max(eps_C.value))
    kappas = []
    for j in range(n):
        if j == center_idx:
            continue
        denom = L_val[j] + eps_C.value[j]
        if denom <= 1e-12:
            kappas.append(np.inf)
        else:
            kappas.append(h_i * np.linalg.norm(D_val[:, j]) / denom)
    kmax = float(np.max(kappas)) if kappas else np.nan

    return {
        "status": "feasible",
        "L": L_val,
        "D": D_val,
        "kappa_max": kmax,
        "objective": float(prob.value),
        "via": "relaxed_socp_clarabel",
        "eps_M_max": eps_M_max,
        "eps_C_max": eps_C_max,
    }


# =============================================================================
# Precomputed joint SOCP stencils (precompute application strategy)
# =============================================================================


@dataclass
class JointSocpStencilData:
    """Precomputed joint SOCP weights for a single point's stencil."""

    L: np.ndarray  # Laplacian weights, shape (n_neighbors,)
    D: np.ndarray  # Gradient weights, shape (dimension, n_neighbors)
    neighbor_indices: np.ndarray
    center_in_neighbors: int
    kappa_max: float
    via: str  # "wendland_lsq_fast_path" | "socp_clarabel"


class PrecomputedJointSocpStencils:
    """Cache for precomputed joint SOCP stencil weights.

    Mirrors `PrecomputedMonotoneStencils` but applies to ALL interior nodes
    (not just boundary) and enforces the joint SOCP (M-matrix + per-edge
    cone) rather than just M-matrix.

    Parameters
    ----------
    points : np.ndarray
        Collocation points, shape (n_total, dimension).
    interior_indices : np.ndarray
        Indices of interior nodes where SOCP should be applied. Boundary
        buffer nodes (where (S1)–(S3) of `prop:soft_monotonicity` fail)
        are typically excluded; the paper algorithm uses Phase 2 fallback
        for those.
    delta : float
        Wendland kernel support radius. Used for distance-weighted SOCP
        objective via `wendland_stencil_weights`.
    neighborhoods : dict
        Post-filter stencil dict (typically ``HJBGFDMSolver.neighborhoods``
        built by ``NeighborhoodBuilder``). Single source of truth: stencils
        always trace to these indices (Issue #1102 dual-source bug class —
        legacy fallback to ``op.get_derivative_weights()`` removed in v0.25.0).
    cone_constant_C : float
        Per-edge cone bound: ||D_{ij}||_2 <= C * h_i * L_{ij}. Default 1.0
        (within the paper's $C_\\star \\in [0.5, 1]$ feasibility range for
        Wendland $C^2$).
    eps_pos : float
        Minimum off-center Laplacian weight (M-matrix slack). Default 0.0.
    max_stencil_enlargements : int
        SOCP-infeasibility-triggered adaptive stencil enlargement (Issue #1106).
        When > 0, a stencil that remains infeasible after C-bisection is rebuilt
        with ``enlargement_step`` additional next-nearest cloud points and the
        SOCP retried, up to this many enlargement steps. A larger stencil adds
        Taylor degrees of freedom, so the SOCP feasible set can become non-empty
        for geometrically starved (wall / corner / obstacle-adjacent) stencils
        where C-bisection alone fails (the infeasibility is directional, not a
        cone-magnitude issue). Default 0 (OFF) — the paper / default path is
        byte-identical. The enlarged neighbor set is stored ONLY in this object's
        ``JointSocpStencilData.neighbor_indices`` (consumed by the HJB-GFDM
        differentiation/Jacobian assembly via the same single-source contract);
        it does NOT mutate the shared runtime neighborhoods, so enlargement does
        not cascade into the operator / FP solver / boundary handler.
    enlargement_step : int
        Number of next-nearest cloud points added per enlargement step
        (Issue #1106). Default 2.

    Attributes
    ----------
    stencils : dict[int, JointSocpStencilData]
        Precomputed weights at each feasible interior node.
    stats : dict
        Precomputation statistics: n_feasible, n_infeasible, n_fast_path,
        n_socp, n_enlarged, max_enlargement_steps, time_ms.
    """

    # Extra candidate margin queried from the kD-tree beyond the requested
    # additions, so duplicates already present in the stencil don't starve the
    # enlargement of fresh points.
    _ENLARGE_POOL = 8

    def __init__(
        self,
        points: np.ndarray,
        interior_indices: np.ndarray,
        delta: float,
        neighborhoods: dict,
        cone_constant_C: float = 1.0,
        eps_pos: float = 0.0,
        cone_constant_C_max: float | None = None,
        cone_constant_C_growth: float = 2.0,
        use_relaxed_fallback: bool = False,
        lambda_M: float = 1.0e4,
        lambda_C: float = 1.0e4,
        max_stencil_enlargements: int = 0,
        enlargement_step: int = 2,
    ):
        if not _CVXPY_AVAILABLE:
            raise ImportError("cvxpy is required for joint SOCP. Install with: pip install cvxpy")
        # Single source of truth: neighborhoods + points + delta. With the
        # legacy `op.get_derivative_weights()` fallback removed in v0.25.0,
        # the TaylorOperator reference is unused; constructor no longer takes one.
        self._points = np.asarray(points)
        self._interior_indices = np.asarray(interior_indices)
        self._delta = float(delta)
        self._C = float(cone_constant_C)
        self._eps_pos = float(eps_pos)
        # Single source of truth: stencils always trace to the supplied
        # post-filter neighborhoods (after visibility filter, ghost nodes,
        # adaptive δ-enlargement). The legacy fallback to
        # `op.get_derivative_weights()` was removed in v0.25.0 — it silently
        # produced wrong results on irregular clouds where the visibility
        # filter / adaptive enlargement modified runtime stencils
        # (Issue #1102 dual-source bug class).
        self._neighborhoods = neighborhoods
        # Per-stencil C-bisection cap. None disables bisection.
        # When set, infeasible stencils retry with C *= cone_constant_C_growth
        # until feasible or C exceeds C_max.
        self._C_max = float(cone_constant_C_max) if cone_constant_C_max is not None else None
        self._C_growth = float(cone_constant_C_growth)
        # Always-feasible relaxed-SOCP fallback for stencils that fail
        # C-bisection. When True, n_infeasible should be 0 (every interior
        # point gets joint-SOCP-style (L, D), with slack penalties handling
        # marginally infeasible cases continuously).
        self._use_relaxed_fallback = bool(use_relaxed_fallback)
        self._lambda_M = float(lambda_M)
        self._lambda_C = float(lambda_C)
        # SOCP-infeasibility-triggered adaptive stencil enlargement (Issue #1106).
        # The kD-tree (over the full cloud) is built only when enlargement is
        # enabled, keeping the default path overhead-free.
        self._max_enlargements = int(max_stencil_enlargements)
        self._enlargement_step = int(enlargement_step)
        if self._max_enlargements > 0 and self._enlargement_step < 1:
            raise ValueError(f"enlargement_step must be >= 1 when enlargement is enabled, got {enlargement_step}")
        self._kdtree = cKDTree(self._points) if self._max_enlargements > 0 else None
        self._dimension = self._points.shape[1] if self._points.ndim == 2 else 1

        if self._dimension not in (1, 2):
            raise ValueError(f"Joint SOCP currently supports 1D or 2D, got dimension {self._dimension}")

        self.stencils: dict[int, JointSocpStencilData] = {}
        self.achieved_C: dict[int, float] = {}
        self.stats = {
            "n_interior": len(self._interior_indices),
            "n_feasible": 0,
            "n_infeasible": 0,
            "n_fast_path": 0,
            "n_socp": 0,
            "n_relaxed_C": 0,
            "max_achieved_C": float(self._C),
            "n_relaxed_fallback": 0,  # stencils that needed slack-penalty solve
            "max_eps_M": 0.0,
            "max_eps_C": 0.0,
            "n_enlarged": 0,  # stencils made feasible via adaptive enlargement (#1106)
            "max_enlargement_steps": 0,
            "time_ms": 0.0,
        }
        self._precompute()

    def _build_stencil_arrays(self, center_global: int, nbr: np.ndarray) -> tuple[np.ndarray, int, float, np.ndarray]:
        """Build the Taylor matrix, center position, scale h_i, and Wendland
        weights for the stencil at ``center_global`` over neighbors ``nbr``.

        Returns ``(A, center_in_nbr, h_i, w_neighbor)``. The caller must ensure
        ``center_global`` is present in ``nbr``.
        """
        build_A = build_taylor_matrix_1d if self._dimension == 1 else build_taylor_matrix_2d
        offsets = self._points[nbr] - self._points[center_global]
        offsets_for_taylor = offsets.reshape(-1) if self._dimension == 1 and offsets.ndim == 2 else offsets
        A, _ = build_A(offsets_for_taylor)

        if self._dimension == 1:
            offsets_1d = offsets.reshape(-1)
            dists = np.abs(offsets_1d)
            w_neighbor = wendland_stencil_weights(offsets_1d, self._delta)
        else:
            dists = np.linalg.norm(offsets, axis=1)
            w_neighbor = wendland_stencil_weights(offsets, self._delta)

        nz = dists[dists > 1e-12]
        h_i = float(np.median(nz)) if len(nz) > 0 else self._delta
        center_in_nbr = int(np.where(nbr == int(center_global))[0][0])
        return A, center_in_nbr, h_i, w_neighbor

    def _solve_with_bisection(
        self, A: np.ndarray, center_in_nbr: int, h_i: float, w_neighbor: np.ndarray
    ) -> tuple[dict, float]:
        """Solve the joint SOCP at one stencil, retrying with C-bisection
        (C *= growth up to C_max) while infeasible. Returns ``(res, C_used)``.
        """
        C_try = self._C
        res = solve_joint_socp_at_stencil(
            A, center_in_nbr, h_i, C_try, eps_pos=self._eps_pos, dimension=self._dimension, wendland_w=w_neighbor
        )
        while self._C_max is not None and res["status"] != "feasible" and C_try * self._C_growth <= self._C_max + 1e-12:
            C_try *= self._C_growth
            res = solve_joint_socp_at_stencil(
                A, center_in_nbr, h_i, C_try, eps_pos=self._eps_pos, dimension=self._dimension, wendland_w=w_neighbor
            )
        return res, C_try

    def _enlarge_stencil(self, center_global: int, cur_nbr: np.ndarray) -> np.ndarray | None:
        """Append up to ``enlargement_step`` next-nearest cloud points not already
        in ``cur_nbr`` (Issue #1106).

        Returns the enlarged neighbor-index array (center preserved at its
        position, new indices appended), or ``None`` if the cloud has no further
        points to add. New neighbors inject Taylor degrees of freedom so the SOCP
        feasible set can become non-empty for a geometrically starved stencil.
        """
        existing = {int(x) for x in cur_nbr}
        n_total = len(self._points)
        k_query = min(n_total, len(cur_nbr) + self._enlargement_step + self._ENLARGE_POOL)
        _, idxs = self._kdtree.query(self._points[center_global], k=k_query)
        idxs = np.atleast_1d(np.asarray(idxs))
        new = [int(j) for j in idxs if 0 <= int(j) < n_total and int(j) not in existing]
        if not new:
            return None
        add = new[: self._enlargement_step]
        return np.concatenate([np.asarray(cur_nbr, dtype=int), np.asarray(add, dtype=int)])

    def _precompute(self) -> None:
        t0 = time.time()

        for i in self._interior_indices:
            i = int(i)
            # Post-filter neighborhood (matches runtime exactly).
            # NeighborhoodBuilder convention: center index is one of the entries.
            nh = self._neighborhoods.get(int(i))
            if nh is None:
                continue
            nbr = np.asarray(nh["indices"])
            if not np.any(nbr == i):
                continue

            A, center_in_nbr, h_i, w_neighbor = self._build_stencil_arrays(i, nbr)
            res, C_try = self._solve_with_bisection(A, center_in_nbr, h_i, w_neighbor)

            # SOCP-infeasibility-triggered adaptive stencil enlargement (Issue #1106).
            # When a stencil is infeasible after C-bisection, add next-nearest
            # cloud points and retry: the extra Taylor degrees of freedom can make
            # the SOCP feasible set non-empty for geometrically starved
            # (wall / corner / obstacle-adjacent) stencils where the infeasibility
            # is directional, not a cone-magnitude issue that C-bisection covers.
            # The enlarged stencil is ADOPTED only when it achieves feasibility, so
            # if enlargement does not help, the relaxed fallback below runs on the
            # original (base) stencil exactly as before. Default OFF
            # (max_stencil_enlargements=0) => this block is skipped entirely.
            n_steps_used = 0
            if res["status"] != "feasible" and self._max_enlargements > 0:
                cur_nbr = nbr
                for step in range(1, self._max_enlargements + 1):
                    ext = self._enlarge_stencil(i, cur_nbr)
                    if ext is None:
                        break  # cloud exhausted; no further points to add
                    cur_nbr = ext
                    A_e, center_e, h_e, w_e = self._build_stencil_arrays(i, cur_nbr)
                    res_e, C_e = self._solve_with_bisection(A_e, center_e, h_e, w_e)
                    if res_e["status"] == "feasible":
                        nbr, A, center_in_nbr, h_i, w_neighbor = cur_nbr, A_e, center_e, h_e, w_e
                        res, C_try = res_e, C_e
                        n_steps_used = step
                        break

            if n_steps_used > 0:
                self.stats["n_enlarged"] += 1
                self.stats["max_enlargement_steps"] = max(self.stats["max_enlargement_steps"], n_steps_used)

            # Final resort: always-feasible relaxed SOCP. This eliminates the
            # discrete scheme switch between joint_socp and Phase-2 M-matrix-QP
            # that creates discontinuous discretization on irregular clouds. For
            # well-conditioned stencils (the C-bisection feasible cases) the
            # original joint_socp solution is used; for marginally infeasible
            # stencils the relaxed SOCP smoothly degrades while keeping the
            # equality constraints (consistency).
            #
            # Issue #1106/#1107/#1108 hook: this point is reached only when both
            # C-bisection AND adaptive enlargement (#1106) failed to find an exact
            # feasible solution — i.e. the stencil is geometrically infeasible at
            # every attempted size. Future exact fallbacks plug in HERE, before
            # the slack-penalty relaxed SOCP: #1107 (3rd-order Taylor stencil) and
            # #1108 (M-matrix-only fallback for cone-binding infeasibility).
            if res["status"] != "feasible" and self._use_relaxed_fallback:
                C_relaxed = self._C if self._C_max is None else self._C_max
                res = solve_relaxed_joint_socp_at_stencil(
                    A,
                    center_in_nbr,
                    h_i,
                    C_relaxed,
                    eps_pos=self._eps_pos,
                    dimension=self._dimension,
                    wendland_w=w_neighbor,
                    lambda_M=self._lambda_M,
                    lambda_C=self._lambda_C,
                )

            if res["status"] != "feasible":
                self.stats["n_infeasible"] += 1
                continue

            self.stencils[i] = JointSocpStencilData(
                L=np.asarray(res["L"], dtype=float),
                D=np.asarray(res["D"], dtype=float),
                neighbor_indices=np.asarray(nbr),
                center_in_neighbors=int(center_in_nbr),
                kappa_max=float(res["kappa_max"]),
                via=res.get("via", "socp_clarabel"),
            )
            self.achieved_C[i] = C_try
            self.stats["n_feasible"] += 1
            if C_try > self._C + 1e-12:
                self.stats["n_relaxed_C"] += 1
                if C_try > self.stats["max_achieved_C"]:
                    self.stats["max_achieved_C"] = C_try
            if res.get("via") == "wendland_lsq_fast_path":
                self.stats["n_fast_path"] += 1
            elif res.get("via") == "relaxed_socp_clarabel":
                self.stats["n_relaxed_fallback"] += 1
                self.stats["max_eps_M"] = max(self.stats["max_eps_M"], res.get("eps_M_max", 0.0))
                self.stats["max_eps_C"] = max(self.stats["max_eps_C"], res.get("eps_C_max", 0.0))
            else:
                self.stats["n_socp"] += 1

        self.stats["time_ms"] = (time.time() - t0) * 1000.0

    def has_stencil(self, point_idx: int) -> bool:
        """Whether SOCP-feasible weights were precomputed for this point."""
        return point_idx in self.stencils

    def get_weights_dict(self, point_idx: int) -> dict | None:
        """Return weights in mfgarchon's `get_derivative_weights` format,
        or None if SOCP infeasible at this point.

        Returned dict has keys: neighbor_indices, grad_weights (shape (d, n)),
        lap_weights (shape (n,)), center_idx_in_neighbors.
        """
        s = self.stencils.get(point_idx)
        if s is None:
            return None
        return {
            "neighbor_indices": s.neighbor_indices,
            "grad_weights": s.D,
            "lap_weights": s.L,
            "center_idx_in_neighbors": s.center_in_neighbors,
            "weight_matrix": None,
        }
