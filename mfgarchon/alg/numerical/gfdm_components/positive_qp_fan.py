"""The positive-QP Laplacian and the guarded fan: the operator of ``monotonicity_scheme="positive_qp_fan"`` (#1689).

Both are built once per cloud, for its interior nodes, and each row is admitted by a certificate, never by a
solver's status:

- **The Laplacian row** of a node is the positive P2 row: ``min ||L||^2`` subject to ``A^T L = e_lap`` and
  ``L_j >= 0`` off the centre, over its k nearest nodes, k widened through ``laplacian_pools``. A node on a face
  has every neighbour in a half-plane and no such row, which is why the scheme reads its boundary rows from the
  boundary conditions.
- **The fan** of a node is a set of rays to nearby nodes whose adjacent angular gaps lie in ``[thin, theta0]``.
  Adjacent rays bound its sectors. In a sector with rays ``z0, z1`` a control is ``alpha = l0 z0 + l1 z1`` with
  ``l >= 0``, so its advection row ``l0 (u_j0 - u_i) + l1 (u_j1 - u_i)`` has non-negative off-diagonals and the
  policy-evaluation matrix of ``HJBHowardSolver(discretisation="guarded_fan")`` is an M-matrix on interior rows.
- **The policy** (`cone_policy`) is Howard's exact argmin over a node's sectors: in each sector the control that
  minimises ``alpha . g + L(alpha)``, ``g`` the sector's gradient, then the sector of least value. For a quadratic
  control cost that control is the Euclidean projection of ``alpha* = -g / lambda`` onto the sector's cone, which
  is why the scheme admits ``QuadraticControlCost`` only.

**A reflecting face** (#1689 PR 4) is closed by two rows:

- **The mirror row** (`_mirror_row`) replaces the Laplacian row of every interior node whose nearest reflecting face
  lies within its reach, ``MIRROR_REACH * delta_i``. Over the node's `MIRROR_POOL` nearest nodes (the centre
  included, sampled boundary nodes kept as sources) and their mirror images across that face, it is ``min ||L||^2``
  subject to the P2 moments, ``L >= 0`` off the centre, and the obliqueness certificate of the paper's
  eq:bc_gamma_main (chap_01:479-482): ``gamma_hat = 2 delta_i sum_j L_ghost_ij s_ij >= MIRROR_GAMMA0``, with
  ``s_ij`` the source's inward distance to the face and ``delta_i`` the mirror pool's radius, ``max |x_j - x_i|``.
  Each ghost's weight is then folded onto its source, so the row's off-centre weights stay non-negative and its row
  sum zero. A layer node no mirror row places is refused by name.
- **The value-copy row** ``u_b - u_j = 0`` (`value_copy_rows`) is a sampled boundary node's, ``j`` its nearest
  interior node: the paper's copy row (eq:copy_row, chap_01:363-366), with the M-matrix sign pattern, annihilating
  constants.

Two differences from the paper, which the measurement behind them (#1689 comment 6296766771) made:

- **Copied boundary nodes are stencil sources** of the mirror rows, as they are of the interior rows. The paper
  assumes they are not (chap_01:367, restated at line 493 in def:row_classification and at line 677 in (B)).
- **A mirror row goes on every layer node.** def:row_classification places one only where the plain row fails,
  since its (P) comes before its (R) (chap_01:488-490).

A construction note, not a difference in the theorem's hypotheses: the mirror row's advection is the node's
physical fan, a folded fan with zero reflected coefficients, so (B)'s fan clause (chap_01:677) holds where that
fan is complete. (R) at chap_01:490 builds its fan over the reflected rays as well, which is not ported, so a layer
node with no complete physical fan is refused by name.

Ported from mfg-research's ``alg_beta`` (``two_tier_stencils._posL``, ``guarded_fan``, ``howard_cone_patch``) and
exp15's widened fan; the research keeps its own copy for the paper. Two-dimensional only: a fan is angular.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from itertools import pairwise
from typing import TYPE_CHECKING

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.spatial import cKDTree

from mfgarchon.alg.numerical.gfdm_components import joint_socp
from mfgarchon.alg.numerical.gfdm_components.joint_socp import build_taylor_matrix_2d

if TYPE_CHECKING:
    from collections.abc import Callable

    from mfgarchon.core.hamiltonian import QuadraticControlCost

#: The 2-D Laplacian in the Taylor columns of `build_taylor_matrix_2d`: (1, x, y, x^2/2, xy, y^2/2).
_E_LAP = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 1.0])

#: A Laplacian row is admitted when each moment's residual is at most this times that moment's scale, and every
#: off-centre weight is at least minus this times the row's largest weight, both measured on the row's QP in units of
#: the node's ``h6``. So a cloud's unit of length does not decide it. A numerical tolerance, not a method parameter.
CERTIFICATE_TOL = 1e-8

#: A sector whose rays' cross product is below this times the product of their lengths -- the sine of its aperture
#: -- is degenerate and not emitted.
_DET_TOL = 1e-10

#: The mirror row's pool: the nearest nodes, the centre included (#1689 PR 4). The certificate was measured with it,
#: on exp17's clouds at N_int 400 to 2500 (comment 6296766771); a wider pool would move ``delta_i`` and the reach.
MIRROR_POOL = 21

#: The reach factor kappa: a node is mirrored when its nearest reflecting face lies within ``kappa * delta_i``. The
#: paper's computations take the pool radius 5 delta / 2 and the reach 3 delta / 2 (def:row_classification,
#: chap_01:486), so kappa = 3/5.
MIRROR_REACH = 0.6

#: The obliqueness the mirror row certifies, gamma_0 of eq:bc_gamma_main. The theorem needs one gamma_0 > 0 at every
#: h; on exp17's clouds the largest gamma_hat each layer row can reach stays at or above 1.187 from N_int 400 to 2500.
MIRROR_GAMMA0 = 0.5


@dataclass(frozen=True)
class GuardedFanParameters:
    """Parameters of ``monotonicity_scheme="positive_qp_fan"``. The defaults are the paper's values.

    Attributes
    ----------
    theta0_deg : float
        Largest angular gap between adjacent rays of a fan, in degrees. Default 120. Supported: above
        ``thin_deg`` and below 180. A sector's aperture must stay below 180 degrees: its rays then span a
        pointed cone, and a control in it has non-negative coordinates along them, which is the monotonicity
        argument.
    widened_caps_deg : tuple of float
        Caps tried in order for a node with no fan at ``theta0_deg``, which happens within about 0.3h of a
        boundary. Default 125, 130, ..., 175. Supported: increasing, each above ``theta0_deg`` and below 180.
        Empty: such a node is refused.
    thin_deg : float
        Smallest angular gap between adjacent rays, in degrees. Default 25. Supported: above 0, below
        ``theta0_deg`` and below 120, since a fan has at least three gaps summing to 360. With the widest gap's, its
        sine bounds a control's sector coordinates: the smaller of the two enters the scheme's constant.
    radius_factor : float
        A fan's rays first reach ``radius_factor * h6``, ``h6`` the node's mean distance to its six nearest
        nodes. Default 2. Supported: above 0.
    radius_growth : float
        Each expansion multiplies that radius by this. Default 1.3. Supported: above 1.
    max_expansions : int
        Radii tried at a cap before the next cap. Default 8. Supported: at least 1.
    n_candidates : int
        Nearest nodes queried for a node's rays, the node itself among them. Default 40. Supported: at least 4,
        the node and the three rays of the smallest fan.
    laplacian_pools : tuple of int
        Pool sizes, the k nearest nodes with the centre, tried in order for a node's Laplacian row. Default
        21, 31, 41, 61. Supported: increasing, each at least 2. Whether a pool places a row is its certificate's
        to decide: a regular grid's five-point stencil is a positive P2 row.
    """

    theta0_deg: float = 120.0
    widened_caps_deg: tuple[float, ...] = tuple(float(c) for c in range(125, 180, 5))
    thin_deg: float = 25.0
    radius_factor: float = 2.0
    radius_growth: float = 1.3
    max_expansions: int = 8
    n_candidates: int = 40
    laplacian_pools: tuple[int, ...] = (21, 31, 41, 61)

    def __post_init__(self) -> None:
        where = "GuardedFanParameters"
        if not 0.0 < self.thin_deg < self.theta0_deg < 180.0 or not self.thin_deg < 120.0:
            raise ValueError(
                f"{where}: need 0 < thin_deg < theta0_deg < 180 and thin_deg < 120, got thin_deg={self.thin_deg}, "
                f"theta0_deg={self.theta0_deg}. A sector's aperture must stay below 180 degrees for its rays to span "
                "a pointed cone, in which a control has non-negative coordinates (the monotonicity argument). A fan "
                "needs thin_deg below theta0_deg, and below 120, since it has at least three gaps summing to 360."
            )
        caps = tuple(float(c) for c in self.widened_caps_deg)
        if any(not self.theta0_deg < c < 180.0 for c in caps) or any(b <= a for a, b in pairwise(caps)):
            raise ValueError(
                f"{where}: widened_caps_deg must increase, each above theta0_deg={self.theta0_deg} and below 180, "
                f"got {caps}. A cap of 180 or more admits a sector that spans a half-plane or more, where a control "
                "has no non-negative coordinates."
            )
        if not self.radius_factor > 0.0 or not self.radius_growth > 1.0 or self.max_expansions < 1:
            raise ValueError(
                f"{where}: need radius_factor > 0, radius_growth > 1 and max_expansions >= 1, got "
                f"{self.radius_factor}, {self.radius_growth}, {self.max_expansions}."
            )
        if self.n_candidates < 4:
            raise ValueError(
                f"{where}: n_candidates must be at least 4, the node itself and the three rays of the smallest fan, "
                f"got {self.n_candidates}."
            )
        pools = tuple(int(k) for k in self.laplacian_pools)
        if not pools or pools[0] < 2 or any(b <= a for a, b in pairwise(pools)):
            raise ValueError(
                f"{where}: laplacian_pools must be non-empty and increasing, each at least 2 (the centre and a "
                f"neighbour), got {pools}."
            )


def mean_six_spacing(points: np.ndarray) -> np.ndarray:
    """Each node's mean distance to its six nearest other nodes: the fan's length scale ``h6``."""
    distances, _ = cKDTree(points).query(points, k=7)
    return distances[:, 1:].mean(axis=1)


def select_guarded_rays(
    rays: np.ndarray, spacing: float, cap_deg: float, parameters: GuardedFanParameters
) -> np.ndarray | None:
    """Indices into ``rays`` of a fan whose adjacent angular gaps all lie in ``[thin, cap]``, or None.

    The radius grows from ``radius_factor * spacing`` by ``radius_growth``; at each radius every start ray is
    tried, in both angular directions, and the rays are chained greedily at least ``thin`` apart. Among the
    admissible fans of the first radius that has one, the pick is by the gap spectrum sorted descending, then
    the radius spectrum, then the ray count: quantities unchanged under rotation and reflection, so a cloud and
    its mirror image get mirrored fans (the research's ``tie_break="invariant", bidirectional=True``).
    """
    radius = np.linalg.norm(rays, axis=1)
    nonzero = np.flatnonzero(radius > 1e-12)
    angle = np.arctan2(rays[:, 1], rays[:, 0]) % (2.0 * np.pi)
    cap = np.radians(cap_deg)
    thin = np.radians(parameters.thin_deg)
    for expansion in range(parameters.max_expansions):
        reach = parameters.radius_factor * spacing * parameters.radius_growth**expansion
        candidates = nonzero[radius[nonzero] <= reach]
        if len(candidates) < 3:
            continue
        admissible = []
        for direction in (1, -1):
            directed = angle[candidates] if direction > 0 else (-angle[candidates]) % (2.0 * np.pi)
            ordered = candidates[np.argsort(directed)]
            a = np.sort(directed)
            a_ext = np.r_[a, a + 2.0 * np.pi]
            for start in range(len(ordered)):
                chain = [start]
                last = a_ext[start]
                stop = a_ext[start] + 2.0 * np.pi
                while True:
                    nxt = max(int(np.searchsorted(a_ext, last + thin - 1e-12, side="left")), chain[-1] + 1)
                    if nxt >= start + len(ordered) or a_ext[nxt] > stop - thin + 1e-12:
                        break
                    chain.append(nxt)
                    last = a_ext[nxt]
                if len(chain) < 3:
                    continue
                positions = np.asarray(chain)
                gaps = np.diff(np.r_[a_ext[positions], stop])
                if gaps.min() >= thin - 1e-10 and gaps.max() <= cap + 1e-12:
                    selected = ordered[positions % len(ordered)]
                    score = (
                        tuple(np.sort(gaps)[::-1].round(12)),
                        tuple(np.sort(radius[selected]).round(12)),
                        len(selected),
                    )
                    admissible.append((score, selected))
        if admissible:
            return min(admissible, key=lambda item: item[0])[1]
    return None


def _positive_laplacian_row(
    displacement: np.ndarray, center: int, length: float
) -> tuple[np.ndarray, float, int] | None:
    """The positive P2 Laplacian row over ``displacement`` (the centre at index ``center``), or None.

    The QP is solved on ``displacement / length``, ``length`` the node's ``h6``, and its weights ``w`` returned as
    ``w / length**2``: the row of a cloud scaled by ``s`` is the row scaled by ``1 / s**2``. Unnormalised, CLARABEL
    reported a node of a unit box scaled by 1e-6 to 0.1 infeasible, which it solves at scale 1 (#2750 review 2).
    Admitted only by its certificate, never by the solver's status: finite weights, each moment's residual at most
    `CERTIFICATE_TOL` times its scale ``1 + sum_j |w_j A_jk|``, and every off-centre weight at least
    ``-CERTIFICATE_TOL * max|w|``, all in units of ``length``. Off-centre weights in that band below 0 are then set to
    0 and the centre to minus the off-centre sum, so the row's off-centre weights are exactly non-negative, and the
    residual is measured again. Returns the weights, the largest relative residual, and how many weights were set
    to 0.
    """
    cp = joint_socp.cp
    A, _ = build_taylor_matrix_2d(displacement / length)
    n = len(displacement)
    off = np.flatnonzero(np.arange(n) != center)
    weights = cp.Variable(n)
    problem = cp.Problem(cp.Minimize(cp.sum_squares(weights)), [A.T @ weights == _E_LAP, weights[off] >= 0])
    try:
        problem.solve(solver="CLARABEL")
    except cp.error.SolverError:
        return None
    if weights.value is None:
        return None
    w = np.asarray(weights.value, dtype=float)
    if not np.all(np.isfinite(w)):
        return None

    def relative_residual(w: np.ndarray) -> float:
        return float(np.max(np.abs(A.T @ w - _E_LAP) / (1.0 + np.abs(A.T * w).sum(axis=1))))

    if not (relative_residual(w) <= CERTIFICATE_TOL and w[off].min() >= -CERTIFICATE_TOL * np.abs(w).max()):
        return None
    snapped = int(np.count_nonzero(w[off] < 0.0))
    w[off] = np.maximum(w[off], 0.0)
    w[center] = -w[off].sum()
    residual = relative_residual(w)
    if not residual <= CERTIFICATE_TOL:
        return None
    return w / length**2, residual, snapped


def _mirror_row(
    displacement: np.ndarray, distance: np.ndarray, radius: float, length: float, gamma0: float | None = None
) -> tuple[np.ndarray, float, float, int] | None:
    """A layer node's mirror row (#1689 PR 4), or None: the centre at index 0, then its ``m`` sources, then their
    ghosts, in ``displacement`` (shape ``(1 + 2m, 2)``); ``distance`` each source's inward distance to the mirrored
    face, ``radius`` the pool's radius ``delta_i``.

    The QP ``min ||L||^2`` subject to ``A^T L = e_lap``, ``L >= 0`` off the centre and ``2 delta_i distance . L_ghost
    >= gamma0`` (`MIRROR_GAMMA0` where None, read at the call as the refusal and the report read it; 0 where a
    refusal asks whether the moments alone can be met) is solved in units of
    ``length``, the node's ``h6``, as `_positive_laplacian_row` is; the certificate is scale-free, since ``L`` scales
    as ``1 / length**2`` and ``delta_i`` and ``s`` as ``length``. Admitted only by its certificate: each moment's
    residual at most `CERTIFICATE_TOL` times its scale, every off-centre weight of the QP, before folding, at least
    ``-CERTIFICATE_TOL * max|L|``, and ``gamma_hat >= gamma0 - CERTIFICATE_TOL``. The ghosts are then folded onto
    their sources; a folded weight in the band below 0 is set to 0 and the centre to minus the off-centre sum, so the
    folded row is exactly Metzler with zero row sum. Returns the folded weights (the centre, then the sources), the
    residual, ``gamma_hat`` and how many weights were set to 0.
    """
    gamma0 = MIRROR_GAMMA0 if gamma0 is None else gamma0
    cp = joint_socp.cp
    A, _ = build_taylor_matrix_2d(displacement / length)
    m = len(distance)
    weights = cp.Variable(1 + 2 * m)
    gamma = 2.0 * (radius / length) * (np.asarray(distance) / length) @ weights[1 + m :]
    problem = cp.Problem(
        cp.Minimize(cp.sum_squares(weights)), [A.T @ weights == _E_LAP, weights[1:] >= 0, gamma >= gamma0]
    )
    try:
        problem.solve(solver="CLARABEL")
    except cp.error.SolverError:
        return None
    if weights.value is None:
        return None
    w = np.asarray(weights.value, dtype=float)
    if not np.all(np.isfinite(w)):
        return None
    residual = float(np.max(np.abs(A.T @ w - _E_LAP) / (1.0 + np.abs(A.T * w).sum(axis=1))))
    gamma_hat = float(2.0 * (radius / length) * (np.asarray(distance) / length) @ w[1 + m :])
    if not (
        residual <= CERTIFICATE_TOL
        and w[1:].min() >= -CERTIFICATE_TOL * np.abs(w).max()
        and gamma_hat >= gamma0 - CERTIFICATE_TOL
    ):
        return None
    folded = np.r_[w[0], w[1 : 1 + m] + w[1 + m :]]
    snapped = int(np.count_nonzero(folded[1:] < 0.0))
    folded[1:] = np.maximum(folded[1:], 0.0)
    folded[0] = -folded[1:].sum()
    return folded / length**2, residual, gamma_hat, snapped


def value_copy_rows(points: np.ndarray, interior_indices: np.ndarray, nodes: np.ndarray) -> dict[int, int]:
    """For each sampled boundary node in ``nodes``, its nearest interior node: the value-copy row ``u_b - u_j = 0``
    a reflecting face's boundary node takes (#1689 PR 4, the paper's eq:copy_row)."""
    interior = np.asarray(interior_indices, dtype=int)
    nodes = np.asarray(nodes, dtype=int)
    if not len(nodes):
        return {}
    _, nearest = cKDTree(points[interior]).query(points[nodes], k=1)
    return {int(b): int(interior[j]) for b, j in zip(nodes, np.atleast_1d(nearest), strict=True)}


class PositiveQPFan:
    """The positive Laplacian rows and the guarded fans of a 2-D cloud's interior nodes, with their certificates.

    Parameters
    ----------
    points : np.ndarray
        The collocation points, shape ``(n, 2)``.
    interior_indices : np.ndarray
        The nodes that get a row; the others are boundary nodes, whose rows are their boundary conditions'.
    parameters : GuardedFanParameters
    obstacle_sdf, visibility_samples, visibility_margin
        With ``obstacle_sdf``, a node's Laplacian pool and its ray candidates keep only the nodes it sees, by the
        same filter as the GFDM neighbourhoods.
    reflecting_faces : dict
        The reflecting faces of the box, each an outward unit normal ``n`` (a tuple) mapped to the face's offset
        ``c``, the face being ``n . x = c`` (#1689 PR 4). A node whose nearest of them lies within its reach takes a
        mirror row across it in place of its Laplacian row. Empty: no mirror row.
    reflecting_nodes : np.ndarray
        The sampled boundary nodes on those faces, which take value-copy rows (`copy_rows`).

    Raises
    ------
    ImportError
        Without cvxpy.
    ValueError
        Naming the nodes no Laplacian pool places, and those with no fan at any cap.

    Attributes
    ----------
    laplacian : csr_matrix
        Shape ``(n, n)``: the positive rows of the interior nodes, zero rows elsewhere.
    sectors : dict
        Arrays ``node``, ``j0``, ``j1`` (int) and ``z0``, ``z1`` (rays, shape ``(m, 2)``) and ``zinv`` (shape
        ``(m, 2, 2)``, the inverse of the matrix with columns ``z0``, ``z1``), one entry per sector, grouped by
        node.
    copy_rows : dict
        Each of ``reflecting_nodes`` mapped to its nearest interior node (`value_copy_rows`).
    report : dict
        What the certificates measured: rows per pool, widened fans and the widest cap, the largest residual,
        the weights set to 0; and the mirror rows, the smallest ``gamma_hat`` and how many sit at ``MIRROR_GAMMA0``.
    """

    def __init__(
        self,
        points: np.ndarray,
        interior_indices: np.ndarray,
        parameters: GuardedFanParameters,
        obstacle_sdf: Callable[[np.ndarray], np.ndarray] | None = None,
        visibility_samples: int = 10,
        visibility_margin: float = 0.0,
        reflecting_faces: dict[tuple[float, ...], float] | None = None,
        reflecting_nodes: np.ndarray | None = None,
    ):
        if not joint_socp._cvxpy():
            raise ImportError(
                "monotonicity_scheme='positive_qp_fan' needs cvxpy (with CLARABEL) for its Laplacian rows."
            )
        points = np.asarray(points, dtype=float)
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError(
                f"PositiveQPFan is two-dimensional (a fan is angular); got points of shape {points.shape}."
            )
        if reflecting_faces and obstacle_sdf is not None:
            raise ValueError(
                "PositiveQPFan (monotonicity_scheme='positive_qp_fan'): a reflecting face with an obstacle in the "
                "domain needs exp09's reflected candidate pool, its pairwise images at junctions and the "
                "visibility-enlarged pool, which are not ported (#1689 R-D3); the mirror row reads no visibility. "
                "Use monotonicity_scheme='joint_socp'."
            )
        self.parameters = parameters
        n = len(points)
        tree = cKDTree(points)
        h6 = mean_six_spacing(points)

        def visible(i: int, nodes: np.ndarray) -> np.ndarray:
            if obstacle_sdf is None:
                return nodes
            from mfgarchon.geometry.visibility import filter_visible_neighbors

            others = nodes[nodes != i]
            seen = filter_visible_neighbors(
                points[i], points[others], obstacle_sdf, n_samples=visibility_samples, margin=visibility_margin
            )
            return np.r_[i, others[seen]]

        rows, cols, vals = [], [], []
        per_pool = dict.fromkeys(parameters.laplacian_pools, 0)
        residual_max, snapped_total = 0.0, 0
        unplaced: dict[int, str] = {}
        node, j0, j1, z0, z1 = [], [], [], [], []
        widened: dict[int, float] = {}
        faces = [(np.asarray(normal, dtype=float), float(c)) for normal, c in (reflecting_faces or {}).items()]
        mirrored: dict[int, tuple[float, float]] = {}
        mirror_residual_max = 0.0
        for i in (int(k) for k in np.asarray(interior_indices)):
            layer = self._mirror(i, points, tree, faces, h6[i]) if faces else None
            if isinstance(layer, str):
                unplaced[i] = layer
                continue
            if layer is not None:
                (weights, residual, gamma_hat, snapped), pool = layer
                rows.extend([i] * len(pool))
                cols.extend(pool.tolist())
                vals.extend(weights.tolist())
                mirrored[i] = (gamma_hat, residual)
                mirror_residual_max = max(mirror_residual_max, residual)
                snapped_total += snapped
            for k in () if layer is not None else parameters.laplacian_pools:
                _, pool = tree.query(points[i], k=min(k, n))
                pool = visible(i, np.atleast_1d(pool).astype(int))
                center = int(np.flatnonzero(pool == i)[0])
                placed = _positive_laplacian_row(points[pool] - points[i], center, h6[i])
                if placed is not None:
                    weights, residual, snapped = placed
                    rows.extend([i] * len(pool))
                    cols.extend(pool.tolist())
                    vals.extend(weights.tolist())
                    per_pool[k] += 1
                    residual_max = max(residual_max, residual)
                    snapped_total += snapped
                    break
            else:
                if layer is None:
                    unplaced[i] = f"no positive P2 Laplacian row in the {parameters.laplacian_pools[-1]} nearest nodes"
                    continue

            _, candidates = tree.query(points[i], k=min(parameters.n_candidates, n))
            candidates = visible(i, np.atleast_1d(candidates).astype(int))
            candidates = candidates[candidates != i]
            rays = points[candidates] - points[i]
            for cap in (parameters.theta0_deg, *parameters.widened_caps_deg):
                selected = select_guarded_rays(rays, h6[i], cap, parameters)
                if selected is not None:
                    break
            else:
                unplaced[i] = (
                    f"no fan with every gap at most {max((parameters.theta0_deg, *parameters.widened_caps_deg))} degrees"
                    + (
                        "; a mirror row's advection is its physical fan, since the fan over reflected rays "
                        "(def:row_classification's (R)) is not ported"
                        if layer is not None
                        else ""
                    )
                )
                continue
            if cap > parameters.theta0_deg:
                widened[i] = cap
            fan_nodes, fan_rays = candidates[selected], rays[selected]
            order = np.argsort(np.arctan2(fan_rays[:, 1], fan_rays[:, 0]))
            fan_nodes, fan_rays = fan_nodes[order], fan_rays[order]
            for r in range(len(fan_nodes)):
                a, b = fan_rays[r], fan_rays[(r + 1) % len(fan_nodes)]
                if abs(a[0] * b[1] - a[1] * b[0]) < _DET_TOL * np.linalg.norm(a) * np.linalg.norm(b):
                    continue
                node.append(i)
                j0.append(int(fan_nodes[r]))
                j1.append(int(fan_nodes[(r + 1) % len(fan_nodes)]))
                z0.append(a)
                z1.append(b)

        if unplaced:
            raise ValueError(_refusal(unplaced, points))
        self.laplacian = csr_matrix(coo_matrix((vals, (rows, cols)), shape=(n, n)))
        z0a, z1a = np.asarray(z0, dtype=float).reshape(-1, 2), np.asarray(z1, dtype=float).reshape(-1, 2)
        self.sectors = {
            "node": np.asarray(node, dtype=int),
            "j0": np.asarray(j0, dtype=int),
            "j1": np.asarray(j1, dtype=int),
            "z0": z0a,
            "z1": z1a,
            "zinv": np.linalg.inv(np.stack([z0a, z1a], axis=2)),
        }
        incomplete = _incomplete_fans(self.sectors, np.asarray(interior_indices, dtype=int), n)
        if incomplete:
            raise ValueError(_refusal(dict.fromkeys(incomplete, "its fan's sectors do not cover the plane"), points))
        self.copy_rows = value_copy_rows(
            points, interior_indices, np.zeros(0, dtype=int) if reflecting_nodes is None else reflecting_nodes
        )
        gammas = np.array([g for g, _ in mirrored.values()])
        self.report = {
            "rows_per_pool": per_pool,
            "n_widened": len(widened),
            "widened_cap_max_deg": max(widened.values(), default=None),
            "laplacian_relative_residual_max": residual_max,
            "weights_set_to_zero": snapped_total,
            "n_sectors": len(node),
            "n_mirror_rows": len(mirrored),
            "mirror_gamma_hat_min": float(gammas.min()) if len(gammas) else None,
            "mirror_rows_at_gamma0": int(np.count_nonzero(gammas <= MIRROR_GAMMA0 + 1e-6)),
            "mirror_relative_residual_max": mirror_residual_max,
            "n_value_copy_rows": len(self.copy_rows),
        }

    @staticmethod
    def _mirror(
        i: int, points: np.ndarray, tree: cKDTree, faces: list[tuple[np.ndarray, float]], length: float
    ) -> tuple[tuple[np.ndarray, float, float, int], np.ndarray] | str | None:
        """Node ``i``'s mirror row, if it is a layer node (#1689 PR 4): None when its nearest reflecting face lies
        beyond its reach, ``MIRROR_REACH * delta_i``. Else ``(row, pool)``, the folded row of `_mirror_row` over the
        pool's nodes, the centre first; or the reason, where no row meets the certificate."""
        n = len(points)
        _, pool = tree.query(points[i], k=min(MIRROR_POOL, n))
        pool = np.atleast_1d(pool).astype(int)
        sources = pool[pool != i]
        offset = points[sources] - points[i]
        radius = float(np.linalg.norm(offset, axis=1).max())
        inward = [c - float(normal @ points[i]) for normal, c in faces]
        nearest = int(np.argmin(inward))
        if not inward[nearest] < MIRROR_REACH * radius:
            return None
        normal, c = faces[nearest]
        ghosts = points[sources] - 2.0 * ((points[sources] @ normal) - c)[:, None] * normal
        displacement = np.vstack([np.zeros((1, 2)), offset, ghosts - points[i]])
        row = _mirror_row(displacement, c - points[sources] @ normal, radius, length)
        if row is None:
            face = f"the face {normal.tolist()} . x = {c}"
            # Which part fails: the P2 moments with non-negative weights, or only the obliqueness certificate. Read
            # through its certificate alone, so the solver's note that an infeasible problem's answer is inaccurate
            # says nothing here.
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Solution may be inaccurate", category=UserWarning)
                moments = _mirror_row(displacement, np.zeros(len(sources)), radius, length, gamma0=0.0) is not None
            return f"a layer node of {face}, within {MIRROR_REACH} times its pool radius of it, with " + (
                f"no mirror row across it in its {MIRROR_POOL} nearest nodes that meets the obliqueness certificate "
                f"gamma_hat >= {MIRROR_GAMMA0} (eq:bc_gamma_main)"
                if moments
                else f"no mirror row across it in its {MIRROR_POOL} nearest nodes and their images with non-negative "
                "off-centre weights: no weights meeting the P2 moments were found there"
            )
        return row, np.r_[i, sources]


def _incomplete_fans(sectors: dict, interior: np.ndarray, n: int) -> list[int]:
    """The interior nodes whose sector apertures do not sum to a full turn: a fan with a gap no sector spans."""
    z0, z1 = sectors["z0"], sectors["z1"]
    aperture = np.arctan2(z0[:, 0] * z1[:, 1] - z0[:, 1] * z1[:, 0], np.einsum("ij,ij->i", z0, z1))
    total = np.bincount(sectors["node"], weights=aperture, minlength=n)
    return [int(i) for i in interior if abs(total[i] - 2.0 * np.pi) > 1e-9]


def _refusal(unplaced: dict[int, str], points: np.ndarray) -> str:
    rows = sorted(unplaced)
    shown = "; ".join(f"{i} at {np.round(points[i], 6).tolist()}: {unplaced[i]}" for i in rows[:10])
    return (
        f"HJBGFDMSolver(monotonicity_scheme='positive_qp_fan'): {len(rows)} interior row(s) cannot be built, so the "
        f"scheme cannot give them monotone rows (#1689). First 10: {shown}. A node close to a boundary or an obstacle "
        "with no nodes on its far side has none: add nodes there, or use monotonicity_scheme='joint_socp'."
    )


@dataclass(frozen=True)
class ConePolicy:
    """A policy of ``HJBHowardSolver(discretisation="guarded_fan")``: per node, its sector and the control in it.

    ``control`` has shape ``(n, 2)`` and is zero off the fan's nodes. ``node``, ``j0``, ``j1``, ``lam0``, ``lam1``
    give each fan node's sector and the control's non-negative coordinates along its rays.
    """

    control: np.ndarray
    node: np.ndarray
    j0: np.ndarray
    j1: np.ndarray
    lam0: np.ndarray
    lam1: np.ndarray

    def advection(self, n: int) -> csr_matrix:
        """``A_adv`` with ``(A_adv u)_i = lam0 (u_j0 - u_i) + lam1 (u_j1 - u_i)``: off-diagonals ``lam >= 0``."""
        data = np.concatenate([self.lam0, self.lam1, -(self.lam0 + self.lam1)])
        rows = np.concatenate([self.node, self.node, self.node])
        cols = np.concatenate([self.j0, self.j1, self.node])
        return csr_matrix(coo_matrix((data, (rows, cols)), shape=(n, n)))


def cone_policy(u: np.ndarray, sectors: dict, control_cost: QuadraticControlCost) -> ConePolicy:
    """Howard's exact policy at ``u``: per node, the sector and the control minimising ``alpha . g + L(alpha)``.

    In a sector, ``g = Z^{-T} (u_j0 - u_i, u_j1 - u_i)`` with ``Z = [z0 z1]``, and the control ranges over the
    cone ``{l0 z0 + l1 z1 : l >= 0}``. For ``L(alpha) = lambda / 2 |alpha|^2`` the minimiser is the Euclidean
    projection onto the cone of ``alpha* = control_cost.optimal_control(g)``: ``alpha*`` itself when its
    coordinates are non-negative, else the best of its projections onto either ray and the apex.
    """
    node, j0, j1 = sectors["node"], sectors["j0"], sectors["j1"]
    z0, z1, zi = sectors["z0"], sectors["z1"], sectors["zinv"]
    du0, du1 = u[j0] - u[node], u[j1] - u[node]
    g = np.stack([zi[:, 0, 0] * du0 + zi[:, 1, 0] * du1, zi[:, 0, 1] * du0 + zi[:, 1, 1] * du1], axis=1)
    target = control_cost.optimal_control(g)
    lam0 = zi[:, 0, 0] * target[:, 0] + zi[:, 0, 1] * target[:, 1]
    lam1 = zi[:, 1, 0] * target[:, 0] + zi[:, 1, 1] * target[:, 1]
    outside = (lam0 < 0.0) | (lam1 < 0.0)
    if outside.any():
        t, s0, s1 = target[outside], z0[outside], z1[outside]
        on0 = np.maximum(0.0, np.einsum("ij,ij->i", t, s0) / np.einsum("ij,ij->i", s0, s0))
        on1 = np.maximum(0.0, np.einsum("ij,ij->i", t, s1) / np.einsum("ij,ij->i", s1, s1))
        go = g[outside]
        value = np.stack(
            [
                on0 * np.einsum("ij,ij->i", s0, go) + control_cost.lagrangian(on0[:, None] * s0),
                on1 * np.einsum("ij,ij->i", s1, go) + control_cost.lagrangian(on1[:, None] * s1),
                np.zeros(len(go)),
            ],
            axis=1,
        )
        pick = value.argmin(axis=1)
        lam0[outside] = np.where(pick == 0, on0, 0.0)
        lam1[outside] = np.where(pick == 1, on1, 0.0)
    alpha_sector = lam0[:, None] * z0 + lam1[:, None] * z1
    value = np.einsum("ij,ij->i", alpha_sector, g) + control_cost.lagrangian(alpha_sector)
    order = np.lexsort((value, node))
    first = np.ones(len(order), dtype=bool)
    first[1:] = node[order][1:] != node[order][:-1]
    chosen = order[first]
    alpha = np.zeros((len(u), 2))
    alpha[node[chosen]] = alpha_sector[chosen]
    return ConePolicy(alpha, node[chosen], j0[chosen], j1[chosen], lam0[chosen], lam1[chosen])
