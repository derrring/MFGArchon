"""Howard's policy iteration inner solver for HJB on GFDM clouds.

Replaces the Newton inner loop of `HJBGFDMSolver._solve_timestep` when the
Hamilton-Jacobi nonlinearity `H = H(x, ∇u, m)` is strictly convex in `p`.
Resolves Issue #1118 (Newton stalls on `|∇u|²` stiffness — Armijo backtracks
bottom out at MIN_ALPHA, temporal plateau in u_solve).

Issue #1118 fix. Validated in:
    mfg-research/experiments/gfdm_monotonicity_audit/minors/
    {exp08_towel_2d_validation, exp09_obstacle_navigation_full,
     exp11_fixed_vs_adaptive}/

Howard delivers ~57× speedup over Newton inner on irregular 2D clouds with
11–15% k-NN-fallback stencils (per exp09 Phase 7 readme). Five forked
research patches converged on the same skeleton; this class is the
graduation of that pattern.

Algorithm
---------
For each backward time step (Nt-1 → 0):

    Initial policy α⁰ ← -∇U_{n+1} (or warm-started from previous step)

    For k = 0, 1, ..., MAX_ITER:
        1. Build advection operator A_adv(α^k) — sign-aware upwind D_grad
        2. Solve linear system:
               (I/dt - A_adv - diag(σ_i²/2) D_lap) · U^{k+1}
                   = U_n+1/dt + L(x, α^k, m)
           where L is the Legendre dual of H at (x, α, m).
        3. Update policy: α^{k+1} = arg min_α (α · ∇U^{k+1} + L(x, α, m))
        4. If ‖α^{k+1} - α^k‖_∞ / ‖α^k‖_∞ < tol: break

    U_n ← U^{k+1}

Policy iteration converges globally under the Bokanowski-Maroso-Zidani 2009
hypothesis (monotone consistent stable scheme, H strictly convex in p).
Convex-in-p is the required hypothesis — separability is neither necessary
nor sufficient.

References
----------
- Bokanowski, Maroso, Zidani 2009 (J. Sci. Comput.) — global convergence
  for monotone schemes.
- Achdou, Capuzzo-Dolcetta 2010 (SIAM J. Numer. Anal.) — coupled (u,m)
  Newton uses Howard as inner solver.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import TYPE_CHECKING, Literal

import numpy as np
from scipy.sparse import csr_matrix, diags, eye, lil_matrix
from scipy.sparse.linalg import spsolve
from scipy.spatial import cKDTree

from mfgarchon.utils.pde_coefficients import diffusion_from_volatility

if TYPE_CHECKING:
    from mfgarchon.alg.numerical.hjb_solvers.hjb_gfdm import HJBGFDMSolver
    from mfgarchon.core.mfg_problem import MFGProblem


AlphaStarFn = Callable[[np.ndarray, np.ndarray, np.ndarray, int], np.ndarray]
"""Legendre transform: alpha_star(x, p, m, t_idx) -> alpha.

Given collocation points `x` (shape (n, d)), gradient `p = ∇U` (shape (n, d)),
density `m` (shape (n,)), and time index `t_idx`, returns the optimal control
`alpha` (shape (n, d)) that achieves `min_α (α · p + L(x, α, m, t))`.

For LQ `H = |p|²/(2c) + g(x, m)`: `alpha_star(x, p, m, t) = -p/c`.
"""

RunningCostFn = Callable[[int], np.ndarray]
# Control-cost Lagrangian L(alpha) over the collocation cloud (Issue #1071): the
# policy-evaluation running cost. Single source = control_cost.lagrangian; the
# default unit-quadratic (1/2)|alpha|^2 is used when this is None.
ControlLagrangianFn = Callable[[np.ndarray], np.ndarray]
"""Time-indexed running cost L(x, alpha*, m, t) at the optimal policy.

`running_cost(t_idx) -> array of shape (n,)` returning the running cost
evaluated at each collocation point at time `t_idx`. This is the
non-quadratic-in-alpha part of the Lagrangian (potential V(x), congestion
g(x, m), etc.) — the quadratic `(1/2)|alpha|^2` part is computed
internally from the alpha returned by `alpha_star`.

Pass None when the problem has no running cost beyond `(1/2)|alpha|^2`.
"""


# ---------------------------------------------------------------------------
# Static stencil precompute (cached on solver instance)
# ---------------------------------------------------------------------------


def _write_closed_row(A: lil_matrix, i: int, nbr_idx: np.ndarray, weights: np.ndarray) -> None:
    """Write row `i` from a stencil, closing it so the row sums to zero (Issue #2081).

    Two weight conventions reach this function. SOCP stencils satisfy the sum rule by
    construction -- `build_taylor_matrix_*` sets `A[:, 0] = 1.0` and the consistency constraint
    is `e_lap == A.T @ L` with `e_lap[0] == 0.0`, so row 0 of that equality IS `sum_j L_j == 0`,
    and the fast path's weighted-LSQ solve satisfies it exactly too. `TaylorOperator` and
    `UpwindOperator` weights multiply deviations `u_j - u_i` and do not.

    The closure is **mandatory** for the second and **idempotent** for the first: subtracting a
    row sum that is already zero changes nothing. So it is applied unconditionally and there is
    no branch on where the weights came from. Assembled without it, `TaylorOperator` gives a
    Laplacian wrong by O(1e+2) rather than by a rounding error.

    Accumulates rather than assigns: `neighbor_indices` is not guaranteed unique -- a periodic
    geometry maps ghost images back to their originals, so one global index can appear twice in
    a stencil with different displacements and different weights.
    """
    total = 0.0
    for j_local, j_global in enumerate(nbr_idx):
        j = int(j_global)
        if j < 0:  # ghost particle: no column to write
            continue
        w = float(weights[j_local])
        A[i, j] += w
        total += w
    # Folds in the centre's own weight when it is in the stencil: it received `w_c` above, so the
    # diagonal ends at `w_c - sum(all) = -sum(j != c)`, which is the same value the branch for a
    # centre-free stencil produces.
    A[i, i] += -total


def _build_dlap_from_weights(stencils: dict, n_total: int) -> csr_matrix:
    """Assemble the sparse Laplacian from per-point stencil weights, row-closed."""
    A = lil_matrix((n_total, n_total))
    for i, w_dict in stencils.items():
        _write_closed_row(A, i, np.asarray(w_dict["neighbor_indices"]), w_dict["lap_weights"])
    return A.tocsr()


def _build_dgrad_central_from_weights(stencils: dict, n_total: int, axis: int) -> csr_matrix:
    """Assemble the central gradient along `axis` from per-point stencil weights, row-closed."""
    A = lil_matrix((n_total, n_total))
    for i, w_dict in stencils.items():
        grad_w = w_dict["grad_weights"]
        grad_axis = grad_w[axis] if grad_w.ndim == 2 else grad_w
        _write_closed_row(A, i, np.asarray(w_dict["neighbor_indices"]), grad_axis)
    return A.tocsr()


def _build_per_axis_upwind_pair(
    points: np.ndarray, socp_data: dict, n_total: int, axis: int
) -> tuple[csr_matrix, csr_matrix]:
    """Assemble (Dpos, Dneg) one-sided gradient operators along `axis`.

    Dpos uses neighbors with `(x_j - x_i)[axis] > 0` (forward stencil).
    Dneg uses neighbors with `(x_j - x_i)[axis] < 0` (backward stencil).

    Both are weighted least-squares fits on the half-stencil. Sign-aware
    selection (Dpos vs Dneg at each row, by sign of `alpha[axis]`) yields
    a monotone upwind operator. From `exp08_towel_2d_validation/howard_patch_towel_upwind.py`.
    """
    Dpos = lil_matrix((n_total, n_total))
    Dneg = lil_matrix((n_total, n_total))
    for i, w_dict in socp_data.items():
        nbr_idx = np.asarray(w_dict["neighbor_indices"])
        rel = points[nbr_idx] - points[i]
        proj = rel[:, axis] if rel.ndim == 2 else rel  # 1D fallback
        # Forward half-stencil
        pos_mask = proj > 1e-12
        if pos_mask.sum() >= 1:
            sel = proj[pos_mask]
            denom = float(np.sum(sel * sel))
            if denom > 1e-14:
                per_w = sel / denom
                tot = 0.0
                for k_local, j_global in enumerate(nbr_idx[pos_mask]):
                    if j_global == i:
                        continue
                    w = float(per_w[k_local])
                    Dpos[i, int(j_global)] += w
                    tot += w
                Dpos[i, i] -= tot
        # Backward half-stencil
        neg_mask = proj < -1e-12
        if neg_mask.sum() >= 1:
            sel = -proj[neg_mask]  # flip to positive offsets
            denom = float(np.sum(sel * sel))
            if denom > 1e-14:
                per_w = sel / denom
                tot = 0.0
                for k_local, j_global in enumerate(nbr_idx[neg_mask]):
                    if j_global == i:
                        continue
                    w = -float(per_w[k_local])  # negate: Dneg @ u ≈ (u_i - u_left)/|Δx|
                    Dneg[i, int(j_global)] += w
                    tot += w
                Dneg[i, i] -= tot
    return Dpos.tocsr(), Dneg.tocsr()


def _build_upwind_projection(alpha: np.ndarray, points: np.ndarray, socp_data: dict, n_total: int) -> csr_matrix:
    """Assemble advection operator A_adv ≈ alpha · ∇ via projection onto alpha.

    For each row i, select neighbors with `rel · α̂ > 0` (the upwind side
    relative to flow direction `α̂ = α/|α|`), compute the projected
    distance, and weight by `|α| · proj_j / Σ proj_k²`. This is the
    canonical upwind GFDM pattern from `exp09_obstacle_navigation_full/howard_patch.py`.

    Returns sparse `A_adv` such that `A_adv @ u ≈ alpha · ∇u`.
    """
    A = lil_matrix((n_total, n_total))
    # points is always shape (n, d) — caller reshapes 1D to (n, 1) in
    # `_build_static`. alpha is always shape (n, d).
    for i, w_dict in socp_data.items():
        a = alpha[i]  # shape (d,)
        a_norm = float(np.linalg.norm(a))
        if a_norm < 1e-10:
            continue
        d_a = a / a_norm
        nbr_idx = np.asarray(w_dict["neighbor_indices"])
        rel = points[nbr_idx] - points[i]  # shape (k, d)
        proj = rel @ d_a  # shape (k,)
        mask = proj > 0
        if mask.sum() < 1:
            # No neighbour lies on the upwind side of alpha: no sign-correct
            # one-sided advection stencil exists at this node. Skip its advective
            # contribution (pure diffusion, M-matrix preserved), matching
            # `_build_per_axis_upwind_pair`. The previous `mask = np.ones(...)`
            # fallback pulled in downwind neighbours (proj_j < 0 -> w_j < 0),
            # silently emitting negative advective off-diagonals and breaking the
            # unconditional-M-matrix property of `thm:upwind_comparison`. Such
            # degenerate stencils are expected to be excluded by the boundary buffer.
            continue
        sel_proj = proj[mask]
        sel_nbrs = nbr_idx[mask]
        denom = float(np.sum(sel_proj * sel_proj))
        if denom < 1e-14:
            continue
        per_w = sel_proj / denom
        total_self = 0.0
        for k_local, j_global in enumerate(sel_nbrs):
            if j_global == i:
                continue
            w = a_norm * float(per_w[k_local])
            A[i, int(j_global)] += w
            total_self += w
        A[i, i] -= total_self
    return A.tocsr()


# ---------------------------------------------------------------------------
# HJBHowardSolver
# ---------------------------------------------------------------------------


class HJBHowardSolver:
    """Howard's policy iteration inner solver for HJB on GFDM clouds.

    Replaces the Newton inner loop of `HJBGFDMSolver` when the Hamiltonian
    `H(x, p, m)` is strictly convex in `p`. Reads SOCP-corrected
    `D_lap`/`D_grad` weights from a `stencil_provider` (constructed
    `HJBGFDMSolver`).

    Parameters
    ----------
    problem : MFGProblem
        The MFG problem (used for `Nt`, `T`, `sigma`).
    stencil_provider : HJBGFDMSolver
        A constructed `HJBGFDMSolver` with `monotonicity_scheme='joint_socp'`
        and `monotonicity_application='precompute'`. Howard reads
        SOCP-corrected `D_lap`/`D_grad` from the provider's
        `_joint_socp_stencils`. The provider's `collocation_points` and
        boundary-segment classification (`_bc_segment_per_point`) are also
        consumed.
    alpha_star : Callable
        Legendre transform: `alpha_star(x, p, m, t_idx) -> alpha`.
        See module docstring `AlphaStarFn` type alias. For LQ
        `H = |p|²/(2c)`, pass `lambda x, p, m, t: -p / c`. The Hamiltonian
        must be strictly convex in `p` for policy iteration to converge
        (Legendre uniqueness of the optimal control). Separability is
        neither necessary nor sufficient.
    running_cost : Callable | None
        `running_cost(t_idx) -> array of shape (n,)` returning the
        non-quadratic-in-alpha running cost evaluated at each collocation
        point. Pass None when there is no running cost beyond
        `(1/2)|alpha|^2`. The quadratic term is computed internally from
        `alpha_star`'s output.
    discretisation : Literal["upwind_projection", "upwind_per_axis", "central"]
        Choice of `A_adv` assembly:

        - `"upwind_projection"` (default): single-axis upwind along the
          policy direction `α̂` (exp09 / exp11 pattern). Robust on 2D
          irregular clouds.
        - `"upwind_per_axis"`: per-axis sign-aware Dpos/Dneg pair, blended
          row-by-row by sign of `α_axis` (exp08 upwind-variant pattern).
          BS-monotone by construction.
        - `"central"`: bare central gradient. Does NOT preserve monotonicity
          under advection-dominant regime; included for comparison only.
    max_iter : int
        Maximum Howard inner iterations per backward time step.
    tol : float
        Relative `∞`-norm tolerance on policy change `(α^{k+1} − α^k)/α^k`.
    volatility_field : float | np.ndarray | None
        Override $\\sigma$ (constant scalar or a collocation-space array of
        shape `(n,)`). A field row-scales the Laplacian by
        $D_i = \\sigma_i^2/2$. When None, read from `problem` via
        `stencil_provider._get_sigma_value(None)`.

    Raises
    ------
    RuntimeError
        If `stencil_provider` does not have `_joint_socp_stencils`
        populated (missing SOCP precompute).
    """

    def __init__(
        self,
        problem: MFGProblem,
        stencil_provider: HJBGFDMSolver,
        alpha_star: AlphaStarFn,
        running_cost: RunningCostFn | None = None,
        control_lagrangian: ControlLagrangianFn | None = None,
        discretisation: Literal["upwind_projection", "upwind_per_axis", "central"] = "upwind_projection",
        max_iter: int = 20,
        tol: float = 1e-4,
        volatility_field: float | np.ndarray | None = None,
        use_provider_bc_rows: bool = False,
    ):
        # #2066: this REFUSED without `_joint_socp_stencils`, for operators it can obtain
        # elsewhere. Howard needs D_lap, D_grad and an interior/boundary split; none of those is
        # SOCP-specific, and `get_derivative_weights` on the live TaylorOperator/LocalRBFOperator
        # returns the SAME dict keys the SOCP object does -- neighbor_indices, grad_weights,
        # lap_weights, center_idx_in_neighbors -- which is what `_build_dlap_from_weights` consumes.
        #
        # Monotonicity is a real hypothesis (Bokanowski-Maroso-Zidani 2009, see the module
        # docstring) but it is a hypothesis about CONVERGENCE, not about whether the solve can run,
        # and this module already ships `discretisation="central"` documented as "Does NOT preserve
        # monotonicity ... included for comparison only". Refusing here while offering that is
        # inconsistent. Worse, the old gate did not deliver the property it appeared to guard: a
        # `joint_socp` run logs SOCP-infeasible interior nodes falling "through to bare
        # Wendland-Taylor LSQ (NON-MONOTONE)" and this check passes anyway, because it tested for
        # the OBJECT rather than for monotonicity.
        _socp = getattr(stencil_provider, "_joint_socp_stencils", None)
        _generic = getattr(stencil_provider, "_gfdm_operator", None)
        if _socp is None and getattr(_generic, "get_derivative_weights", None) is None:
            raise RuntimeError(
                "HJBHowardSolver: stencil_provider offers neither `_joint_socp_stencils` nor a "
                "`_gfdm_operator` exposing `get_derivative_weights`, so there is no source for "
                "the D_lap/D_grad operators policy evaluation needs. Construct the provider with "
                "monotonicity_scheme='joint_socp', monotonicity_application='precompute' for "
                "monotone stencils, or with any scheme that builds a Taylor/RBF operator (#2066)."
            )
        if _socp is None:
            warnings.warn(
                "HJBHowardSolver: running on non-SOCP stencils. Policy iteration's global "
                "convergence (Bokanowski-Maroso-Zidani 2009) assumes a MONOTONE scheme, and bare "
                "Wendland-Taylor weights are not monotone in general. The solve will run; its "
                "convergence is not covered by that hypothesis. Use "
                "monotonicity_scheme='joint_socp' for the covered case (#2066).",
                UserWarning,
                stacklevel=2,
            )
        if discretisation not in ("upwind_projection", "upwind_per_axis", "central"):
            raise ValueError(
                f"discretisation must be one of {{upwind_projection, upwind_per_axis, central}}, got {discretisation!r}"
            )

        self.problem = problem
        self.stencil_provider = stencil_provider
        self.alpha_star = alpha_star
        self.running_cost = running_cost
        self.control_lagrangian = control_lagrangian
        self.discretisation = discretisation
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self._volatility_field_override = volatility_field
        # Issue #1118 PR2: when True, pull value-form BC rows from the provider's shared
        # `_value_form_bc_rows` (Dirichlet value + real Neumann normal·grad stencil) instead
        # of the self-contained Dirichlet-identity/Neumann-by-nearest scheme below.
        self.use_provider_bc_rows = bool(use_provider_bc_rows)

        self._static: dict | None = None  # lazy-built on first solve

    # ---- Static precompute (lazy) -------------------------------------

    def _build_static(self) -> dict:
        """Build SOCP-stencil-derived operators and BC masks once."""
        s = self.stencil_provider
        pts = np.asarray(s.collocation_points)
        if pts.ndim == 1:
            pts = pts.reshape(-1, 1)
        n = pts.shape[0]
        dimension = pts.shape[1]

        # #2066: SOCP when it is there, the provider's own operator when it is not. Both return
        # the same dict -- neighbor_indices, grad_weights, lap_weights, center_idx_in_neighbors --
        # which is the entire contract `_build_dlap_from_weights` and `_build_dgrad_central_from_weights` need. The
        # SOCP path is byte-identical to before; nothing about an existing run changes.
        socp_obj = getattr(s, "_joint_socp_stencils", None)
        if socp_obj is not None:
            socp_data = {i: socp_obj.get_weights_dict(i) for i in range(n) if socp_obj.has_stencil(i)}
        else:
            _op = s._gfdm_operator
            socp_data = {}
            for i in range(n):
                w = _op.get_derivative_weights(i)
                if w is not None:
                    socp_data[i] = w

        D_lap = _build_dlap_from_weights(socp_data, n)
        D_grad_central = [_build_dgrad_central_from_weights(socp_data, n, d) for d in range(dimension)]

        per_axis_upwind: list[tuple[csr_matrix, csr_matrix]] | None = None
        if self.discretisation == "upwind_per_axis":
            per_axis_upwind = [_build_per_axis_upwind_pair(pts, socp_data, n, d) for d in range(dimension)]

        # #2066: the interior/boundary split was a BY-PRODUCT of SOCP feasibility -- a point the
        # SOCP could not solve silently became a boundary point and got BC treatment instead of
        # interior treatment. The provider carries the real classification in `boundary_indices`,
        # from `_detect_boundary_indices` or from the caller. Prefer it; fall back to the old
        # derivation only when it is absent, so the SOCP path is unchanged where it already worked.
        _declared_boundary = getattr(s, "boundary_indices", None)
        if _declared_boundary is not None and socp_obj is None:
            interior_mask = np.ones(n, dtype=bool)
            interior_mask[np.asarray(_declared_boundary, dtype=int)] = False
            interior_mask &= np.array([i in socp_data for i in range(n)])
        else:
            interior_mask = np.zeros(n, dtype=bool)
            for i in socp_data:
                interior_mask[i] = True
        boundary_idx = np.where(~interior_mask)[0]

        # BC type per boundary point. Reads optional segment classification
        # from the stencil provider; defaults to Neumann-by-extension if
        # the provider hasn't pre-classified.
        is_dirichlet = np.zeros(n, dtype=bool)
        per_pt = getattr(s, "_bc_segment_per_point", None)
        if per_pt is not None:
            for i in boundary_idx:
                seg = per_pt.get(int(i))
                if seg is not None and "DIRICHLET" in str(getattr(seg, "bc_type", "")).upper():
                    is_dirichlet[i] = True

        # Nearest interior point for Neumann-by-extension BC rows.
        int_pts_idx = np.where(interior_mask)[0]
        nearest_int = np.zeros(n, dtype=int)
        if len(boundary_idx) > 0 and len(int_pts_idx) > 0:
            int_tree = cKDTree(pts[int_pts_idx])
            for i in boundary_idx:
                _, j_local = int_tree.query(pts[i], k=1)
                nearest_int[i] = int_pts_idx[j_local]

        return {
            "pts": pts,
            "n": n,
            "dimension": dimension,
            "socp_data": socp_data,
            "D_lap": D_lap,
            "D_grad_central": D_grad_central,
            "per_axis_upwind": per_axis_upwind,
            "boundary_idx": boundary_idx,
            "is_dirichlet": is_dirichlet,
            "nearest_int": nearest_int,
        }

    # ---- A_adv assembly -----------------------------------------------

    def _build_A_adv(self, alpha: np.ndarray, static: dict) -> csr_matrix:
        """Assemble the advection operator A_adv such that A_adv @ u ≈ α·∇u."""
        if self.discretisation == "central":
            n = static["n"]
            A = csr_matrix((n, n))
            for d in range(static["dimension"]):
                A = A + diags(alpha[:, d]) @ static["D_grad_central"][d]
            return A
        if self.discretisation == "upwind_per_axis":
            n = static["n"]
            A = csr_matrix((n, n))
            for d in range(static["dimension"]):
                Dpos, Dneg = static["per_axis_upwind"][d]
                pos_mask = (alpha[:, d] > 0).astype(float)
                D_sel = diags(pos_mask) @ Dpos + diags(1.0 - pos_mask) @ Dneg
                A = A + diags(alpha[:, d]) @ D_sel
            return A
        # upwind_projection
        return _build_upwind_projection(alpha, static["pts"], static["socp_data"], static["n"])

    # ---- One backward step --------------------------------------------

    def _howard_step(
        self,
        u_next: np.ndarray,
        m_n: np.ndarray,
        t_idx: int,
        sigma: float | np.ndarray,
        dt: float,
        static: dict,
        alpha_init: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """One backward time step. Returns (U_n, alpha_at_convergence)."""
        n = static["n"]
        dimension = static["dimension"]
        pts = static["pts"]
        D_lap = static["D_lap"]
        boundary_idx = static["boundary_idx"]
        is_dirichlet = static["is_dirichlet"]
        nearest_int = static["nearest_int"]

        # Initial policy: from previous time step's policy (warm start) or
        # from central gradient of u_next.
        if alpha_init is not None:
            alpha = alpha_init.copy()
        else:
            p = np.zeros((n, dimension))
            for d in range(dimension):
                p[:, d] = static["D_grad_central"][d] @ u_next
            alpha = self.alpha_star(pts, p, m_n, t_idx)

        rc_t = self.running_cost(t_idx) if self.running_cost is not None else None
        u_new = u_next.copy()

        # Issue #1118 PR2: value-form BC rows from the provider's shared builder. Constant
        # across inner iterations at this time step; None -> the legacy self-contained scheme.
        bc_rows = self.stencil_provider._value_form_bc_rows(t_idx) if self.use_provider_bc_rows else None
        if isinstance(sigma, np.ndarray):
            diffusion_operator = diags(diffusion_from_volatility(sigma, kind="field")) @ D_lap
        else:
            diffusion_operator = 0.5 * sigma * sigma * D_lap

        for _iteration in range(self.max_iter):  # named for the #2072 diagnostic below
            A_adv = self._build_A_adv(alpha, static)
            A = eye(n, format="csr") / dt - A_adv - diffusion_operator

            # RHS: u_next/dt + L(alpha) + running_cost. L(alpha) is the control-cost
            # Lagrangian from the single source (control_cost.lagrangian, Issue #1071):
            # L(alpha) = lambda/2 |alpha|^2 for the quadratic cost, so this is byte-identical
            # to the prior hardcoded (1/2)|alpha|^2 at lambda=1 and correct for lambda != 1.
            # Falls back to the unit-quadratic form when no Lagrangian is supplied.
            if self.control_lagrangian is not None:
                L_alpha = self.control_lagrangian(alpha)
            else:
                L_alpha = 0.5 * np.sum(alpha * alpha, axis=1)
            b = u_next / dt + L_alpha
            if rc_t is not None:
                b = b + rc_t

            # Apply BC rows in-place on A.
            A_lil = A.tolil()
            if bc_rows is not None:
                # Value-form rows shared with the Newton path (Issue #1118 PR2): A[i,:]=row,
                # b[i]=bc_target. Honors Dirichlet values + the real Neumann normal·grad stencil.
                for i, (row, target) in bc_rows.items():
                    A_lil[i, :] = row
                    b[i] = target
                # Issue #1254, 2026-06-10 audit: detect stencil-less interior points.
                # boundary_idx = all points without an SOCP stencil.  When
                # use_provider_bc_rows=True the provider must supply a BC row for every such
                # point; a point absent from both socp_data and bc_rows is a geometrically
                # interior node whose SOCP solve failed — Howard has no fallback, so its bare
                # A[i,i]=1/dt row would silently freeze the value and pollute its neighbors.
                uncovered = [int(i) for i in boundary_idx if i not in bc_rows]
                if uncovered:
                    raise RuntimeError(
                        f"HJBHowardSolver (use_provider_bc_rows=True): "
                        f"{len(uncovered)} point(s) in boundary_idx have no SOCP stencil "
                        f"and no provider BC row (first few indices: {uncovered[:8]}). "
                        "A bare A[i,i]=1/dt row would silently freeze those nodes. "
                        "Howard has no Wendland-Taylor fallback; ensure the SOCP solve "
                        "succeeds for all interior points (Issue #1254)."
                    )
            else:
                # Legacy self-contained scheme (homogeneous no-flux only).
                for i in boundary_idx:
                    A_lil[i, :] = 0
                    if is_dirichlet[i]:
                        A_lil[i, i] = 1.0
                        b[i] = 0.0
                    else:
                        A_lil[i, i] = -1.0
                        A_lil[i, int(nearest_int[i])] = 1.0
                        b[i] = 0.0

            u_new = spsolve(A_lil.tocsr(), b)
            if not np.all(np.isfinite(u_new)):
                # #2072 item 3. This warned and returned `u_next.copy()` and RETURNED -- it did not
                # continue the policy loop; what carried on was the outer backward sweep. (`u_next`
                # is the LATER time level, previous only in sweep order.) A fail-silent fallback:
                # the sweep completes, every value is finite, and the returned field is a
                # plausible-looking answer to a different problem -- measured, every time level came
                # back bit-identical to the terminal condition.
                #
                # A Hamiltonian that slips the decomposition guard reaches here by a real mechanism,
                # not a hypothetical one: the wrong alpha* = -dH/dp is cubic in p, the policy
                # iteration diverges, L(alpha) overflows, `b` goes non-finite, and spsolve returns
                # all-NaN -- so this branch fired at EVERY timestep and is what made the result look
                # finite instead of NaN.
                #
                # Nothing depended on it. Settled by MUTATION rather than by counting: replacing
                # this fallback with a hard raise on `main` left every test in the Howard-touching
                # population with an identical outcome. (An earlier version of this comment said
                # "four test files import this class". One does; three merely name it in a
                # docstring, a dict key and an importlib string -- and the count was the wrong
                # predicate anyway, since a test can reach this path through
                # `inner_solver="howard"` without importing the class at all.)
                _bad = int(np.count_nonzero(~np.isfinite(u_new)))
                raise RuntimeError(
                    f"HJBHowardSolver: policy evaluation returned {_bad} non-finite value(s) of "
                    f"{u_new.size} at time index {t_idx}, Howard iteration {_iteration}. The linear solve "
                    f"did not produce a value function, and continuing would substitute the "
                    f"previous timestep and return a finite field that answers a different "
                    f"problem. Check the Hamiltonian's decomposition (inner_solver='howard' "
                    f"assumes a quadratic control cost) and the stencil conditioning; "
                    f"inner_solver='newton' needs no decomposition. (Issue #2072)"
                )

            # Policy update.
            p_new = np.zeros((n, dimension))
            for d in range(dimension):
                p_new[:, d] = static["D_grad_central"][d] @ u_new
            alpha_new = self.alpha_star(pts, p_new, m_n, t_idx)

            # Convergence on policy.
            denom = max(float(np.linalg.norm(alpha, ord=np.inf)), 1e-10)
            rel = float(np.linalg.norm(alpha_new - alpha, ord=np.inf)) / denom
            alpha = alpha_new
            if rel < self.tol:
                break

        return u_new, alpha

    # ---- Public solve --------------------------------------------------

    def solve_hjb_system(
        self,
        M_density: np.ndarray | None,
        U_terminal: np.ndarray,
        cross_density=None,
        **_unused,
    ) -> np.ndarray:
        """Backward sweep using Howard inner.

        Parameters
        ----------
        M_density : np.ndarray | None
            Density field at every time step, shape `(Nt+1, n)`. When None,
            treated as zero (no MFG coupling).
        U_terminal : np.ndarray
            Terminal condition `U(T, x)`, shape `(n,)`.

        Returns
        -------
        np.ndarray
            Value function `U(t, x)`, shape `(Nt+1, n)`. `U[Nt] == U_terminal`.
        """
        # Issue #1071: named explicitly (not swallowed by **_unused) so a multi-population
        # cross-density trajectory fails loud here rather than silently decoupling — it would
        # never reach Howard's policy-iteration evaluation.
        if cross_density is not None:
            raise NotImplementedError(
                "HJBHowardSolver does not support multi-population cross-density coupling "
                "(Issue #1071). Use HJBFDMSolver for multi-population MFG."
            )
        if self._static is None:
            self._static = self._build_static()
        static = self._static
        n = static["n"]

        Nt = int(self.problem.Nt)
        T_final = float(self.problem.T)
        dt = T_final / Nt

        if self._volatility_field_override is None:
            sigma = float(self.stencil_provider._get_sigma_value(None))
        elif np.isscalar(self._volatility_field_override):
            sigma = float(self._volatility_field_override)
        else:
            sigma = np.asarray(self._volatility_field_override, dtype=float)
            if sigma.ndim == 0:
                sigma = float(sigma)
            elif sigma.shape != (n,):
                raise ValueError(
                    "HJBHowardSolver volatility_field must be a scalar or a row-wise "
                    f"field with shape ({n},), got {sigma.shape}."
                )

        U = np.zeros((Nt + 1, n))
        U[Nt] = U_terminal.copy()

        alpha_carry: np.ndarray | None = None
        for nt in range(Nt - 1, -1, -1):
            m_n = np.asarray(M_density[nt]) if M_density is not None else np.zeros(n)
            U[nt], alpha_carry = self._howard_step(U[nt + 1], m_n, nt, sigma, dt, static, alpha_carry)

        return U


__all__ = ["HJBHowardSolver", "AlphaStarFn", "RunningCostFn", "ControlLagrangianFn"]
