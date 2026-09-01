"""Manufactured solutions: one owner for turning an exact pair into MFG source terms.

The method of manufactured solutions is the only EXTERNAL oracle a scheme has. Pick exact
``u*`` and ``m*``, substitute them into the MFG system, and whatever is left over is fed back as a
source term; the scheme must then reproduce ``u*`` and ``m*`` and its error must fall at its order.
Everything else this repository can check compares one implementation against another, and it has
repeatedly measured two implementations agreeing while both were wrong.

WHAT THIS OWNS
--------------
The ASSEMBLY of a source from an exact pair, for a **coupled HJB-FP pair** on a separable
Hamiltonian:

    S_hjb = -d_t u + H(x, grad u, m) - tr(D . Hess u)
    S_fp  =  d_t m + div(m * alpha*)  - tr(D . Hess m)

where ``alpha*`` is the optimal control (the FP drift) and ``D`` is the diffusion tensor. That is
where a sign goes wrong quietly (this repository already has #1645, a DPP sign bug), so a
:class:`ManufacturedPair` carries its own analytic derivatives -- written once per family -- and
the two functions below own the assembly.

Two fixtures manufacture a coupled pair and are the callers this is written for:
``tests/integration/test_coupled_mms_2d_no_flux.py`` and ``tests/integration/test_coupled_mfg_mms.py``.
Other MMS fixtures in this repository are NOT coupled pairs -- they manufacture ``m`` alone with a
prescribed velocity, or ``u`` alone with a constant density, or have no source at all -- and
``test_mms_validation.ManufacturedSolution`` remains the owner of the velocity-driven FP family,
which takes a velocity directly and is strictly more general on that side than this module.
This module does not subsume it and does not try to.

NOTHING IS RE-DERIVED HERE
--------------------------
Every convention this assembly needs already has a single owner in this package, and each is called
rather than restated -- the failure this module exists to prevent is a source built from a
different equation than the solver integrates, and re-deriving a convention is exactly how that
happens:

- ``H``            -> the problem's own :meth:`HamiltonianBase.evaluate_H`.
- ``alpha*``       -> the problem's own :meth:`optimal_control`, which is sense-aware. Deriving
  ``alpha* = -grad u / lambda`` here instead would be silently sign-flipped for a MAXIMIZE cost.
- ``sigma -> D``   -> :func:`mfgarchon.utils.pde_coefficients.diffusion_from_volatility`, including
  its ``kind`` argument and its refusal to guess. A ``(d, d)`` volatility is the symmetric
  standard-deviation matrix ``S`` with ``D = (1/2) S S^T`` (RFC #1596), NOT a covariance -- reading
  it as one is a silent factor of ``S``.

Only ``div(alpha*)`` is not directly available from an owner. It is obtained as
``div(alpha*) = c * tr(Hess u)`` for a drift linear in ``p``, and the coefficient ``c`` is
MEASURED from the Hamiltonian's own ``optimal_control`` rather than assumed -- see
:func:`_drift_coefficient`. A drift that is not linear in ``p`` (an L1 or regularized cost) fails
that measurement and is refused.

WHY THE HESSIAN AND NOT THE LAPLACIAN
-------------------------------------
The diffusion term is written ``tr(D . Hess)``, not ``(sigma^2 / 2) Lap``. For isotropic
``D = (sigma^2/2) I`` the two agree exactly, so nothing is lost. What is gained is the off-diagonal:
``sum_{i != j} D_ij d2u/dx_i dx_j`` is precisely the cross-derivative term of #2198, and a
Laplacian-shaped source cannot express it at all.

Note what this does NOT buy on its own: under an isotropic ``D`` every off-diagonal Hessian entry
is multiplied by exactly zero, so a wrong cross-derivative in a pair is invisible to the assembled
source. :func:`check_pair` is the check that sees it, and it does not go through the assembly.

SCOPE
-----
- Separable Hamiltonians whose optimal control is linear in ``p``. Verified by measurement against
  the Hamiltonian's own ``optimal_control``, not assumed from the cost's type.
- **Constant** ``Sigma``. For spatially varying diffusion the FP term is
  ``(1/2) sum_ij d2(sigma_ij m)/dx_i dx_j``, which is not ``tr(D . Hess m)``. Constant ``Sigma`` is
  sufficient for the two to coincide, not necessary -- a divergence-free ``Sigma(x)`` would also do
  -- but a varying ``Sigma`` is refused here rather than approximated, and the refusal is enforced
  by requiring an array ``sigma`` to be ``(d, d)`` and to declare ``sigma_kind="tensor"``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from mfgarchon.core.hamiltonian import HEvalState, SeparableHamiltonian
from mfgarchon.utils.pde_coefficients import diffusion_from_volatility, validate_symmetric_psd

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from mfgarchon.core.hamiltonian import HamiltonianBase

# A field callable: (t, x) -> (N,) for scalars, (t, x) -> (N, d) for gradients, (t, x) -> (N, d, d)
# for Hessians, with x of shape (N, d). The same convention `base_hjb.py` already uses for a source
# term, so a pair's own derivatives and the assembled source speak one language.
Field = Callable[[float, "NDArray"], "NDArray"]


@dataclass(frozen=True)
class ManufacturedPair:
    """An exact ``(u*, m*)`` together with the analytic derivatives a source assembly needs.

    Every field is a callable of ``(t, x)`` with ``x`` of shape ``(N, d)``. Gradients return
    ``(N, d)``, Hessians ``(N, d, d)``, everything else ``(N,)``.

    The derivatives are supplied rather than computed. Differentiating numerically would put the
    differentiation's own truncation error into a source that is supposed to be exact, and the
    resulting convergence study would measure that error rather than the scheme's.

    Nothing here checks that the derivatives ARE the derivatives -- the assembly cannot, since it
    consumes them and has no independent access to ``u`` and ``m``. :func:`check_pair` is that
    check, it is finite-difference based, and a pair should pass it once when it is written.
    """

    u: Field
    u_t: Field
    grad_u: Field
    hess_u: Field
    m: Field
    m_t: Field
    grad_m: Field
    hess_m: Field
    name: str = "unnamed"


# --------------------------------------------------------------------------------------------
# Shape contract
# --------------------------------------------------------------------------------------------
# The module pins a convention; enforcing it is not optional. An (N, 1) where an (N,) is expected
# broadcasts to (N, N) with no error and feeds the solver a source of the wrong size -- the exact
# failure `test_coupled_mms_2d_no_flux._split` was written against.


def _points(x: NDArray) -> NDArray:
    """Validate the evaluation points and return them as a float ``(N, d)`` array."""
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 2:
        raise ValueError(
            f"manufactured sources take points of shape (N, d); got shape {arr.shape}. A flat (N,) "
            f"array of 1-D points is read as N points of dimension d -- pass x.reshape(-1, 1)."
        )
    return arr


def _scalar(value: NDArray, n: int, what: str) -> NDArray:
    arr = np.asarray(value, dtype=float)
    if arr.shape != (n,):
        raise ValueError(f"{what} must return shape ({n},) for {n} points; got {arr.shape}.")
    return arr


def _gradient(value: NDArray, n: int, dim: int, what: str) -> NDArray:
    arr = np.asarray(value, dtype=float)
    if arr.shape != (n, dim):
        raise ValueError(f"{what} must return shape ({n}, {dim}); got {arr.shape}.")
    return arr


def _hessian(value: NDArray, n: int, dim: int, what: str) -> NDArray:
    arr = np.asarray(value, dtype=float)
    if arr.shape != (n, dim, dim):
        raise ValueError(f"{what} must return shape ({n}, {dim}, {dim}); got {arr.shape}.")
    return arr


# --------------------------------------------------------------------------------------------
# The two conventions, each taken from its owner
# --------------------------------------------------------------------------------------------


def _diffusion_tensor(sigma: float | NDArray, dim: int, sigma_kind: str | None) -> NDArray:
    """The constant ``(d, d)`` diffusion tensor ``D``, via this package's volatility owner.

    ``D`` already carries the ``1/2``: :func:`diffusion_from_volatility` returns ``D = (1/2) S S^T``
    for a tensor volatility and ``sigma**2 / 2`` for a scalar one, so the assemblies below contract
    ``D`` with a Hessian directly and no factor of a half appears in this module at all.

    A ``(d,)`` array is refused. Everywhere else in this package a 1-D ``sigma`` array of length
    ``Nx`` is a spatially varying field (``base_hjb``, ``fp_fdm_time_stepping``), which this module
    does not support; a diagonal-anisotropic volatility is passed as a ``(d, d)`` diagonal matrix,
    which is the same instruction :func:`diffusion_from_volatility` gives.
    """
    arr = np.asarray(sigma, dtype=float)
    if arr.ndim == 0:
        if sigma_kind is not None:
            raise ValueError(f"a scalar sigma is unambiguous; drop sigma_kind={sigma_kind!r}.")
        return float(diffusion_from_volatility(arr)) * np.eye(dim)
    if sigma_kind != "tensor":
        raise ValueError(
            f"an array sigma must declare sigma_kind='tensor' and be the (d, d) symmetric "
            f"standard-deviation matrix S (D = 1/2 S S^T, RFC #1596); got sigma_kind="
            f"{sigma_kind!r} with shape {arr.shape}. A spatially varying sigma -- the meaning a 1-D "
            f"array carries elsewhere in this package -- is out of scope: the FP diffusion is then "
            f"(1/2) sum_ij d2(sigma_ij m)/dx_i dx_j, not tr(D . Hess m). A diagonal-anisotropic "
            f"volatility is np.diag([s1, ..., sd]), not [s1, ..., sd]."
        )
    if arr.shape != (dim, dim):
        raise ValueError(f"a tensor sigma must have shape ({dim}, {dim}) for {dim}-D points; got {arr.shape}.")
    validate_symmetric_psd(arr, name="manufactured sigma tensor")
    return np.asarray(diffusion_from_volatility(arr, kind="tensor"), dtype=float)


# Deterministic, zero-free probes with distinct magnitudes and both signs, so that a drift which is
# merely odd, or linear on one ray, cannot pass the linearity measurement below.
def _drift_probes(dim: int) -> NDArray:
    base = np.arange(1, dim + 1, dtype=float)
    return np.stack([base, -0.5 * base[::-1], np.full(dim, 2.0), -np.full(dim, 0.25) * base])


def _drift_coefficient(hamiltonian: HamiltonianBase, dim: int) -> float:
    """Measure ``c`` in ``alpha* = c * p`` from the Hamiltonian's own ``optimal_control``.

    ``div(alpha*)`` is the one quantity the assembly needs that no owner exposes directly. For a
    drift linear in ``p`` it is ``c * tr(Hess u)``, and ``c`` is read out of the drift owner here
    rather than re-derived from ``lambda`` and the optimization sense -- re-deriving it is what
    silently flips the transport term for a MAXIMIZE cost.

    :class:`SeparableHamiltonian` is required because its ``optimal_control`` depends on ``p``
    alone; for a Hamiltonian whose drift also depends on ``x``, ``m`` or ``t`` a single constant
    ``c`` measured at one point would not describe it.
    """
    if not isinstance(hamiltonian, SeparableHamiltonian):
        raise NotImplementedError(
            f"the FP assembly needs div(alpha*), which this module obtains as c * tr(Hess u) for a "
            f"drift linear in p. That requires a SeparableHamiltonian, whose optimal control "
            f"depends on p alone; got {type(hamiltonian).__name__}, whose drift may vary with x, m "
            f"or t. Supply div(alpha*) explicitly, or extend this module -- do not assemble a "
            f"source from a drift the solver does not use."
        )
    probes = _drift_probes(dim)
    alpha = np.asarray(
        hamiltonian.optimal_control(np.zeros_like(probes), np.ones(len(probes)), probes, 0.0),
        dtype=float,
    )
    if alpha.shape != probes.shape:
        raise NotImplementedError(f"optimal_control returned shape {alpha.shape} for probes {probes.shape}.")
    coefficient = float(alpha.flat[0] / probes.flat[0])
    if not np.allclose(alpha, coefficient * probes, rtol=1e-12, atol=1e-14):
        raise NotImplementedError(
            f"the optimal control of this {type(hamiltonian.control_cost).__name__} is not linear "
            f"in p, so div(alpha*) is not c * tr(Hess u) and this module cannot assemble the FP "
            f"transport term for it. Max deviation from the best linear fit: "
            f"{float(np.max(np.abs(alpha - coefficient * probes))):.3e}."
        )
    return coefficient


# --------------------------------------------------------------------------------------------
# The two assemblies
# --------------------------------------------------------------------------------------------


def hjb_source(
    pair: ManufacturedPair,
    hamiltonian: HamiltonianBase,
    sigma: float | NDArray,
    *,
    sigma_kind: str | None = None,
) -> Field:
    """``S_hjb = -d_t u + H(x, grad u, m) - tr(D . Hess u)``.

    ``H`` is evaluated through the problem's own Hamiltonian, so the manufactured equation is the
    one the solver integrates rather than a re-implementation that could drift from it.
    """

    def source(t: float, x: NDArray) -> NDArray:
        pts = _points(x)
        n, dim = pts.shape
        diffusion_tensor = _diffusion_tensor(sigma, dim, sigma_kind)
        p = _gradient(pair.grad_u(t, pts), n, dim, "grad_u")
        m = _scalar(pair.m(t, pts), n, "m")
        h = np.asarray(hamiltonian.evaluate_H(HEvalState(x=pts, p=p, m=m, t=t)), dtype=float).ravel()
        h = _scalar(h, n, "evaluate_H")
        hess = _hessian(pair.hess_u(t, pts), n, dim, "hess_u")
        diffusion = np.einsum("ij,nij->n", diffusion_tensor, hess)
        u_t = _scalar(pair.u_t(t, pts), n, "u_t")
        return -u_t + h - diffusion

    return source


def fp_source(
    pair: ManufacturedPair,
    hamiltonian: HamiltonianBase,
    sigma: float | NDArray,
    *,
    sigma_kind: str | None = None,
) -> Field:
    """``S_fp = d_t m + div(m * alpha*) - tr(D . Hess m)``.

    The transport expands as ``div(m alpha*) = grad m . alpha* + m div(alpha*)``. ``alpha*`` is the
    optimal control taken from the Hamiltonian itself -- note that it is ``-grad u / lambda`` for a
    quadratic MINIMIZE cost and ``+grad u / lambda`` for MAXIMIZE, which is why it is read from the
    owner and not written out here. ``div(alpha*) = c tr(Hess u)`` with ``c`` measured by
    :func:`_drift_coefficient`; that Laplacian is the divergence of the DRIFT, not a diffusion term,
    and is unaffected by ``Sigma``.
    """
    # Measured once, at build time: a Hamiltonian this module cannot assemble for must fail when the
    # source is built, not midway through a solve. Both refusals -- non-separable, and a drift that
    # is not linear in p -- are dimension-independent, so a 1-D probe settles them before the
    # evaluation points (and hence the real dimension) exist.
    _drift_coefficient(hamiltonian, 1)

    def source(t: float, x: NDArray) -> NDArray:
        pts = _points(x)
        n, dim = pts.shape
        diffusion_tensor = _diffusion_tensor(sigma, dim, sigma_kind)
        coefficient = _drift_coefficient(hamiltonian, dim)
        grad_u = _gradient(pair.grad_u(t, pts), n, dim, "grad_u")
        m = _scalar(pair.m(t, pts), n, "m")
        alpha = _gradient(
            np.asarray(hamiltonian.optimal_control(pts, m, grad_u, t), dtype=float), n, dim, "optimal_control"
        )
        hess_u = _hessian(pair.hess_u(t, pts), n, dim, "hess_u")
        grad_m = _gradient(pair.grad_m(t, pts), n, dim, "grad_m")
        hess_m = _hessian(pair.hess_m(t, pts), n, dim, "hess_m")
        div_alpha = coefficient * np.einsum("nii->n", hess_u)
        transport = (grad_m * alpha).sum(axis=-1) + m * div_alpha
        diffusion = np.einsum("ij,nij->n", diffusion_tensor, hess_m)
        return _scalar(pair.m_t(t, pts), n, "m_t") + transport - diffusion

    return source


# --------------------------------------------------------------------------------------------
# Auditing a pair
# --------------------------------------------------------------------------------------------


# Step sizes chosen where truncation and round-off balance for a central difference in IEEE double:
# eps^(1/3) for a first derivative, eps^(1/4) for a second. Using one step for both makes the second
# derivatives round-off dominated (at h = 1e-6 the second-difference noise is O(1e-4) relative).
_FIRST_STEP = 6e-6
_SECOND_STEP = 1.2e-4


def _relative(analytic: NDArray, numeric: NDArray) -> float:
    scale = float(np.max(np.abs(numeric)))
    error = float(np.max(np.abs(analytic - numeric)))
    return error / scale if scale > 0.0 else error


def pair_derivative_errors(pair: ManufacturedPair, t: float, x: NDArray) -> dict[str, float]:
    """Relative error of each analytic derivative in ``pair`` against a finite difference of ``u`` and ``m``.

    This is the pair's only NON-CIRCULAR check. A residual built from the pair's own derivatives is
    an algebraic identity -- the source is DEFINED as that residual, so it returns zero whatever the
    convention, including a deliberately sign-flipped one. Differencing ``u`` and ``m`` themselves
    does not share the assembly, so it audits the six DERIVATIVE callables the assembly takes on
    trust. Six, not eight: ``ManufacturedPair`` carries eight callables, but ``u`` and ``m`` are the
    finite-difference ORACLE here rather than things being audited.

    What it catches: a wrong analytic derivative, including the off-diagonal Hessian entries that an
    isotropic ``Sigma`` multiplies by exactly zero and that no assembled source can see.
    What it does NOT catch: an error in the assembly convention (a sign, a factor, a misplaced
    ``lambda``), since both sides of that comparison would share it -- that is what
    :func:`hjb_source` / :func:`fp_source` delegating to this package's owners is for -- nor any
    mismatch between the manufactured equation and the one the solver discretizes, which only a
    convergence study sees.

    Returns a ``{name: relative_error}`` mapping over the six derivative fields; the caller decides
    the tolerance, since it depends on how oscillatory the family is.
    """
    pts = _points(x)
    n, dim = pts.shape
    h1, h2 = _FIRST_STEP, _SECOND_STEP

    def shifted(field: Field, axis: int, step: float) -> tuple[NDArray, NDArray]:
        offset = np.zeros(dim)
        offset[axis] = step
        return field(t, pts + offset), field(t, pts - offset)

    errors: dict[str, float] = {}
    for name, value, first, second in (
        ("u", pair.u, pair.grad_u, pair.hess_u),
        ("m", pair.m, pair.grad_m, pair.hess_m),
    ):
        base = _scalar(value(t, pts), n, name)

        forward, backward = value(t + h1, pts), value(t - h1, pts)
        time_field = pair.u_t if name == "u" else pair.m_t
        errors[f"{name}_t"] = _relative(_scalar(time_field(t, pts), n, f"{name}_t"), (forward - backward) / (2.0 * h1))

        gradient = _gradient(first(t, pts), n, dim, f"grad_{name}")
        numeric_gradient = np.empty((n, dim))
        for axis in range(dim):
            plus, minus = shifted(value, axis, h1)
            numeric_gradient[:, axis] = (plus - minus) / (2.0 * h1)
        errors[f"grad_{name}"] = _relative(gradient, numeric_gradient)

        hessian = _hessian(second(t, pts), n, dim, f"hess_{name}")
        numeric_hessian = np.empty((n, dim, dim))
        for axis in range(dim):
            plus, minus = shifted(value, axis, h2)
            numeric_hessian[:, axis, axis] = (plus - 2.0 * base + minus) / h2**2
            for other in range(axis + 1, dim):
                offset_i, offset_j = np.zeros(dim), np.zeros(dim)
                offset_i[axis] = h2
                offset_j[other] = h2
                mixed = (
                    value(t, pts + offset_i + offset_j)
                    - value(t, pts + offset_i - offset_j)
                    - value(t, pts - offset_i + offset_j)
                    + value(t, pts - offset_i - offset_j)
                ) / (4.0 * h2**2)
                numeric_hessian[:, axis, other] = mixed
                numeric_hessian[:, other, axis] = mixed
        errors[f"hess_{name}"] = _relative(hessian, numeric_hessian)

    return errors


def check_pair(pair: ManufacturedPair, t: float, x: NDArray, *, tolerance: float = 1e-4) -> None:
    """Raise if any analytic derivative in ``pair`` disagrees with a finite difference of ``u`` / ``m``.

    See :func:`pair_derivative_errors` for what this does and does not catch. A pair should pass
    this once, where it is written -- it is a check on the pair, not on the scheme, so it does not
    belong in a convergence study's inner loop.
    """
    errors = pair_derivative_errors(pair, t, x)
    failed = {name: value for name, value in errors.items() if not (value <= tolerance)}
    if failed:
        detail = ", ".join(f"{name}={value:.3e}" for name, value in sorted(failed.items()))
        raise ValueError(
            f"manufactured pair {pair.name!r}: analytic derivatives disagree with a finite "
            f"difference of u / m beyond tolerance {tolerance:.1e} -- {detail}. All errors: "
            f"{ {name: f'{value:.3e}' for name, value in sorted(errors.items())} }."
        )
