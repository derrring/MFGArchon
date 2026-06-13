"""Conservative finite-volume flux kernels for the Fokker-Planck FVM solver (Issue #422).

These helpers implement the *advective* numerical flux of a cell-centered finite-volume
discretization on a structured (tensor-product) grid. They are the higher-order extension
of the conservative divergence-upwind FDM stencil in
:mod:`fp_fdm_alg_divergence_upwind`: the flux at an interface ``(i+1/2)`` is shared by the two
cells that touch it, so the divergence telescopes and mass is conserved to machine precision.

Two reconstructions are provided:

- **1st-order upwind**: ``m_{i+1/2} = m_i`` if ``alpha_{i+1/2} >= 0`` else ``m_{i+1}``.
- **2nd-order MUSCL**: piecewise-linear with a ``minmod`` slope limiter,
  ``m_{i+1/2} = m_i + sigma_i*dx/2`` from the upwind side, where
  ``sigma_i = minmod((m_i - m_{i-1})/dx, (m_{i+1} - m_i)/dx)``. The minmod limiter is TVD,
  which buys positivity (no negative-density ringing) *and* second order in smooth regions.

The diffusion flux is handled separately by the solver via the conservative finite-volume
:class:`~mfgarchon.operators.differential.laplacian.LaplacianOperator` (single source for the
no-flux/periodic zero-column-sum closure), so only advection lives here.

Boundary closure (per axis):

- ``no_flux`` / ``neumann`` / ``reflecting``: zero flux at the two wall faces -> exact mass
  conservation.
- ``periodic``: the wrap face flux is shared by cell ``N-1`` and cell ``0`` -> exact mass
  conservation.

Mathematical background: LeVeque, *Finite Volume Methods for Hyperbolic Problems* (2002),
ch. 6 (REA algorithm, slope limiters); Toro, *Riemann Solvers* (2009).
"""

from __future__ import annotations

import numpy as np

ZERO_FLUX_BC = frozenset({"no_flux", "neumann", "reflecting"})


def minmod(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    r"""Elementwise minmod limiter.

    ``minmod(a, b) = 0`` if ``a`` and ``b`` have opposite signs (or either is zero),
    otherwise ``sign(a) * min(|a|, |b|)``. This is the TVD slope limiter that prevents
    new extrema (hence negative densities) in the MUSCL reconstruction.
    """
    same_sign = a * b > 0.0
    return np.where(same_sign, np.sign(a) * np.minimum(np.abs(a), np.abs(b)), 0.0)


def _muscl_limited_increments(mm: np.ndarray, periodic: bool) -> np.ndarray:
    r"""Per-cell limited linear increments ``delta_i = sigma_i * dx`` along the last axis.

    ``mm`` has shape ``(..., N)``; the limited increment uses the cell's left/right first
    differences. Interior cells use ``minmod(m_i - m_{i-1}, m_{i+1} - m_i)``. For a periodic
    axis the two end cells wrap; otherwise their increment is set to zero (the reconstruction
    degrades to first order at the wall cell, which keeps the scheme TVD and positive there).
    """
    delta = np.zeros_like(mm)
    dm_left = mm[..., 1:-1] - mm[..., :-2]
    dm_right = mm[..., 2:] - mm[..., 1:-1]
    delta[..., 1:-1] = minmod(dm_left, dm_right)
    if periodic:
        delta[..., 0] = minmod(mm[..., 0] - mm[..., -1], mm[..., 1] - mm[..., 0])
        delta[..., -1] = minmod(mm[..., -1] - mm[..., -2], mm[..., 0] - mm[..., -1])
    return delta


def _interior_face_states(mm: np.ndarray, alpha_int: np.ndarray, scheme: str, periodic: bool):
    """Reconstruct upwind face densities at the ``N-1`` interior faces along the last axis.

    Returns the reconstructed face density ``m_{i+1/2}`` (shape ``(..., N-1)``) selected from
    the upwind side according to the sign of ``alpha_int``.
    """
    if scheme == "upwind":
        return np.where(alpha_int >= 0.0, mm[..., :-1], mm[..., 1:])
    if scheme == "muscl":
        delta = _muscl_limited_increments(mm, periodic)
        m_left = mm[..., :-1] + 0.5 * delta[..., :-1]
        m_right = mm[..., 1:] - 0.5 * delta[..., 1:]
        return np.where(alpha_int >= 0.0, m_left, m_right), delta
    raise ValueError(f"Unknown FVM reconstruction scheme: {scheme!r}. Use 'upwind' or 'muscl'.")


def axis_flux_divergence(
    m: np.ndarray,
    alpha_int: np.ndarray,
    axis: int,
    dx: float,
    scheme: str,
    bc_type: str,
    alpha_wrap: np.ndarray | None = None,
) -> np.ndarray:
    r"""Advective flux divergence ``(F_{i+1/2} - F_{i-1/2})/dx`` along one axis.

    Parameters
    ----------
    m : np.ndarray
        Cell averages, shape ``(*spatial_shape)``.
    alpha_int : np.ndarray
        Interface velocity at the ``N-1`` interior faces along ``axis`` (shape = ``m`` with the
        ``axis`` length reduced by one). Sharing this per-face velocity between neighboring cells
        is what makes the column sums vanish (mass conservation).
    axis : int
        Axis along which to compute the flux divergence.
    dx : float
        Grid spacing along ``axis``.
    scheme : str
        ``"upwind"`` (1st order) or ``"muscl"`` (2nd order, minmod-limited).
    bc_type : str
        Uniform BC type along this axis: a zero-flux type (``no_flux``/``neumann``/
        ``reflecting``) or ``periodic``.
    alpha_wrap : np.ndarray | None
        Interface velocity at the periodic wrap face (between cell ``N-1`` and cell ``0``);
        required iff ``bc_type == "periodic"``. Shape = ``m`` with ``axis`` removed.

    Returns
    -------
    np.ndarray
        Flux-divergence contribution of this axis, shape ``(*spatial_shape)``.
    """
    mm = np.moveaxis(m, axis, -1)
    ai = np.moveaxis(alpha_int, axis, -1)
    periodic = bc_type == "periodic"
    n = mm.shape[-1]

    if scheme == "muscl":
        m_face, delta = _interior_face_states(mm, ai, scheme, periodic)
    else:
        m_face = _interior_face_states(mm, ai, scheme, periodic)
        delta = None
    f_int = ai * m_face

    f_full = np.zeros((*mm.shape[:-1], n + 1), dtype=float)
    f_full[..., 1:n] = f_int

    if periodic:
        if alpha_wrap is None:
            raise ValueError("alpha_wrap is required for a periodic axis.")
        aw = np.moveaxis(alpha_wrap, axis, -1) if alpha_wrap.ndim == m.ndim else alpha_wrap
        # alpha_wrap has the axis removed; broadcast onto the leading dims of mm[..., 0].
        aw = np.asarray(aw, dtype=float)
        if scheme == "muscl":
            m_left_wrap = mm[..., -1] + 0.5 * delta[..., -1]
            m_right_wrap = mm[..., 0] - 0.5 * delta[..., 0]
            m_wrap = np.where(aw >= 0.0, m_left_wrap, m_right_wrap)
        else:
            m_wrap = np.where(aw >= 0.0, mm[..., -1], mm[..., 0])
        f_wrap = aw * m_wrap
        f_full[..., 0] = f_wrap
        f_full[..., n] = f_wrap
    elif bc_type in ZERO_FLUX_BC:
        # Zero wall flux: f_full[..., 0] and f_full[..., n] stay 0.
        pass
    else:
        raise NotImplementedError(
            f"FVM advection boundary closure not implemented for bc_type={bc_type!r}. "
            "Supported: no_flux/neumann/reflecting (zero-flux) and periodic. "
            "Dirichlet advection is deferred (Issue #422 scope note)."
        )

    div = (f_full[..., 1:] - f_full[..., :-1]) / dx
    return np.moveaxis(div, -1, axis)


def advective_divergence(
    m: np.ndarray,
    alpha_faces: list[np.ndarray],
    alpha_wrap: list[np.ndarray | None],
    spacing: list[float],
    scheme: str,
    bc_types: list[str],
) -> np.ndarray:
    """Total advective flux divergence ``sum_d (F^d_{+} - F^d_{-})/dx_d`` over all axes.

    The per-axis contributions are summed (dimension-by-dimension reconstruction on the
    structured grid). Each axis telescopes independently, so the global cell-sum of the result
    is zero for zero-flux/periodic boundaries -> mass is conserved exactly.
    """
    ndim = m.ndim
    div = np.zeros_like(m, dtype=float)
    for d in range(ndim):
        div += axis_flux_divergence(m, alpha_faces[d], d, spacing[d], scheme, bc_types[d], alpha_wrap[d])
    return div
