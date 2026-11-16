"""The volatility <-> diffusion conversion, D = 1/2 Sigma Sigma^T: one convention, one owner (#811, RFC #1596).

``docs/user/CONVENTIONS.md`` § Volatility and diffusion states it. Every solver derives its diffusion through
`diffusion_from_volatility`; `volatility_from_diffusion` is its reverse, under the same ``kind`` contract;
`validate_symmetric_psd` is the admissibility gate for a ``(d, d)`` volatility; `diffusion_from_volatility_torch` is the
same scalar conversion on torch tensors. They read numpy only and know no MFG, so they sit with the generic numerics,
below every layer that reads them (#2938 stage 2d).
"""

from __future__ import annotations

from typing import Any

import numpy as np


def diffusion_from_volatility(
    volatility: float | np.ndarray,
    *,
    kind: str | None = None,
) -> float | np.ndarray:
    r"""Canonical PDE diffusion coefficient ``D`` from SDE volatility ``sigma`` (Issue #811).

    Single source of truth for the volatility -> diffusion conversion documented in
    ``docs/user/CONVENTIONS.md`` § Volatility and diffusion. Volatility is a tensor in general
    (the noise matrix :math:`\Sigma`, shape ``(d, k)``); the scalar :math:`\sigma` is the
    isotropic special case. The diffusion coefficient is

    .. math::  D = \tfrac{1}{2}\,\Sigma\Sigma^\top

    i.e. half the noise covariance :math:`\Sigma\Sigma^\top` (Oksendal, Karatzas-Shreve) --
    **not** :math:`\Sigma^\top\Sigma`; only :math:`\Sigma\Sigma^\top` has the correct ``(d, d)``
    shape for a ``(d, k)`` volatility. For a square symmetric :math:`\Sigma` they coincide.

    A **scalar** ``sigma`` is unambiguous: ``D = sigma**2 / 2``. For an **array** ``sigma``
    the shape alone cannot distinguish a ``(d, d)`` noise tensor from an ``(Nx, Ny)`` spatial
    field, so ``kind`` is **required** -- the converter refuses to guess (fail-loud, the same
    silent-convention class this single-source converter exists to eliminate):

    - ``kind="field"``  -> isotropic per-point volatility; ``D = sigma**2 / 2`` elementwise
      (any spatial shape). Byte-identical to the literal ``0.5 * sigma**2`` (IEEE: ``0.5*x == x/2``).
    - ``kind="tensor"`` -> the trailing two axes are the noise matrix :math:`\Sigma` ``(d, k)``;
      ``D = 0.5 * Sigma @ Sigma.T`` over the last two axes (any leading axes are spatial).
      Requires ``ndim >= 2``. A diagonal-anisotropic volatility is passed as a ``(d, d)``
      diagonal matrix here, not a ``(d,)`` vector.

    Raises ``ValueError`` for an array with ``kind is None`` (ambiguous), an unknown ``kind``,
    or ``kind="tensor"`` with ``ndim < 2``.
    """
    arr = np.asarray(volatility, dtype=float)
    if arr.ndim == 0:
        return 0.5 * float(arr) ** 2  # scalar isotropic: unambiguous, kind not needed
    if kind is None:
        raise ValueError(
            "array volatility is ambiguous: pass kind='field' (isotropic per-point sigma, "
            "D = sigma^2/2 elementwise) or kind='tensor' (trailing (d,k) is the noise matrix "
            "Sigma, D = 1/2 Sigma Sigma^T). Refusing to guess a (d,d) tensor vs a spatial "
            "field (Issue #811)."
        )
    if kind == "field":
        return 0.5 * arr**2
    if kind == "tensor":
        if arr.ndim < 2:
            raise ValueError(
                f"kind='tensor' needs a (d,k) or (*spatial,d,k) noise matrix (ndim>=2), got ndim={arr.ndim}."
            )
        return 0.5 * np.matmul(arr, np.swapaxes(arr, -1, -2))
    raise ValueError(f"kind must be 'field' or 'tensor' (or None for scalar sigma), got {kind!r}.")


def validate_symmetric_psd(
    S: np.ndarray,
    *,
    name: str = "sigma tensor",
    tolerance: float = 1e-10,
) -> None:
    r"""Consumer-side admissibility gate for a ``(d, d)`` tensor VOLATILITY input (RFC #1596).

    A ``(d, d)`` volatility is the **symmetric** standard-deviation matrix ``S`` -- the symmetric
    square root of the covariance ``C`` (``C = S S^T = S^2``), so the diffusion is
    ``D = 1/2 S S^T = 1/2 C``. An asymmetric or non-PSD ``(d, d)`` input is caller confusion
    (passing a Cholesky factor, a raw covariance, or a malformed matrix), so this fails loud.

    Grid consumers that assemble a per-axis + cross-derivative diffusion on a tensor grid
    (``adi_diffusion_step``, ``DiffusionOperator`` tensor path) call this immediately before
    :func:`diffusion_from_volatility` ``(kind="tensor")``. The converter itself stays
    symmetry-agnostic: ``D = 1/2 S S^T`` is the covariance of *any* noise matrix, and the
    particle solver legitimately drives particles with an asymmetric ``(d, k)`` noise matrix
    (RFC #1596: symmetry is a consumer admissibility constraint, not a kernel identity).

    Raises ``ValueError`` if ``S`` is not square ``(d, d)``, not symmetric within ``tolerance``,
    or has an eigenvalue below ``-tolerance``.
    """
    S = np.asarray(S, dtype=float)
    if S.ndim != 2 or S.shape[0] != S.shape[1]:
        raise ValueError(f"{name} must be a square (d, d) matrix, got shape {S.shape}.")
    max_asymmetry = float(np.max(np.abs(S - S.T))) if S.size else 0.0
    if max_asymmetry > tolerance:
        raise ValueError(
            f"{name} must be symmetric: it is the standard-deviation matrix (symmetric square "
            f"root of the covariance), not a covariance or a Cholesky factor (RFC #1596). "
            f"Max asymmetry |S - S^T| = {max_asymmetry:.3e} > tolerance {tolerance:.2e}."
        )
    min_eigenvalue = float(np.min(np.linalg.eigvalsh(S))) if S.size else 0.0
    if min_eigenvalue < -tolerance:
        raise ValueError(
            f"{name} must be positive semi-definite (a std-dev matrix has non-negative "
            f"eigenvalues). Found min eigenvalue {min_eigenvalue:.6e} < 0 (RFC #1596)."
        )


def volatility_from_diffusion(
    D: float | np.ndarray,
    *,
    kind: str | None = None,
) -> float | np.ndarray:
    r"""SDE volatility :math:`\Sigma` from a PDE diffusion ``A`` -- the reverse of
    :func:`diffusion_from_volatility`, under the same ``kind`` contract (#2375 ruling 6).

    - scalar ``D``: :math:`\sigma = \sqrt{2D}`; ``kind`` is not needed.
    - ``kind="field"``: :math:`\sigma = \sqrt{2D}` elementwise.
    - ``kind="tensor"``: the trailing two axes are ``A``, shape ``(d, d)``, possibly preceded by grid
      axes. :math:`\Sigma` is the **symmetric** square root of :math:`2A` (via ``eigh``), not a
      Cholesky factor: consumers validate a ``(d, d)`` volatility as symmetric PSD
      (:func:`validate_symmetric_psd`, RFC #1596), so the round trip volatility -> diffusion ->
      volatility must land on a symmetric matrix.

    An array without ``kind`` is refused for the reason :func:`diffusion_from_volatility` gives: a
    ``(d, d)`` tensor and a ``d x d`` spatial field have the same shape.

    Raises ``ValueError`` for a negative scalar or field, a tensor that is not positive
    semi-definite, an array with ``kind is None``, an unknown ``kind``, or a ``kind="tensor"``
    array whose trailing two axes are not square.
    """
    arr = np.asarray(D, dtype=float)
    if arr.ndim == 0:
        value = float(arr)
        if value < 0:
            raise ValueError(f"Diffusion coefficient must be non-negative, got {value}")
        return float(np.sqrt(2.0 * value))
    if kind is None:
        raise ValueError(
            "array diffusion is ambiguous: pass kind='field' (isotropic per-point D, sigma = sqrt(2D) "
            "elementwise) or kind='tensor' (trailing (d,d) is the diffusion matrix A, Sigma its "
            "symmetric square root times sqrt(2)). Refusing to guess a (d,d) tensor vs a spatial field "
            "(#2375 ruling 6)."
        )
    if kind == "field":
        if np.any(arr < 0):
            raise ValueError("Diffusion coefficient array must be non-negative")
        return np.sqrt(2.0 * arr)
    if kind == "tensor":
        if arr.ndim < 2 or arr.shape[-1] != arr.shape[-2]:
            raise ValueError(f"kind='tensor' needs a (d,d) or (*spatial,d,d) diffusion matrix, got shape {arr.shape}.")
        two_D = 2.0 * arr
        eigvals, eigvecs = np.linalg.eigh(0.5 * (two_D + np.swapaxes(two_D, -1, -2)))  # symmetrize guards fp asymmetry
        if float(np.min(eigvals)) < -1e-10:
            raise ValueError(
                f"Diffusion tensor must be positive semi-definite to take a real square root; "
                f"got min eigenvalue {float(np.min(eigvals)):.6e} (RFC #1596)."
            )
        sqrt_eigvals = np.sqrt(np.clip(eigvals, 0.0, None))
        return (eigvecs * sqrt_eigvals[..., None, :]) @ np.swapaxes(eigvecs, -1, -2)
    raise ValueError(f"kind must be 'field' or 'tensor' (or None for a scalar D), got {kind!r}.")


def diffusion_from_volatility_torch(volatility: Any) -> Any:
    r"""Canonical PDE diffusion coefficient ``D`` from SDE volatility ``sigma`` for torch tensors.

    Mirrors the scalar contract of :func:`diffusion_from_volatility` (``D = 0.5 * sigma**2``)
    but accepts any numeric type — plain Python floats, NumPy scalars, and PyTorch tensors —
    preserving the autograd computation graph when ``sigma`` is a ``torch.Tensor``.

    Issue #1189/#1193: single source of the sigma->D conversion for torch-based solvers
    (PINNs, torch_backend).  A pure Python implementation (no explicit torch import) so this
    module stays importable when PyTorch is absent; torch tensors are handled transparently
    because ``*`` and ``**`` dispatch to tensor ops automatically.

    :math:`D = \frac{1}{2}\sigma^2` — byte-identical to ``0.5 * sigma**2`` for all
    IEEE-754 double-precision normal floats; the constant 0.5 is exactly representable,
    so ``0.5 * x`` and ``x / 2`` coincide at the bit level.

    Parameters
    ----------
    volatility : float, numpy scalar, or torch.Tensor
        Volatility coefficient (SDE noise amplitude).

    Returns
    -------
    Same type as input (float, numpy scalar, or torch.Tensor)
        Diffusion coefficient ``D = 0.5 * sigma**2``.
    """
    return 0.5 * volatility**2
