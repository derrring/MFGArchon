"""
Spectral diagnostics for a discretized operator with boundary conditions applied.

**This module reports numbers. It does not return a stability verdict.**

That is deliberate, and it is what this module is for. Its predecessor claimed to implement
"GKS (Gustafsson-Kreiss-Sundström) stability analysis" and returned ``stable: bool`` from three
``pde_type`` branches. It was not GKS -- GKS is a normal-mode/Laplace-transform theory for
hyperbolic initial-boundary value problems, producing a Kreiss determinant condition on the
boundary scheme, whereas what was computed is the spectrum of a semi-discrete operator. All three
branches were defective in different ways (Issue #1859), which is the signature of one function
made to serve three incompatible theories:

- parabolic sampled the wrong end of the spectrum on its sparse path, reporting stable for an
  operator with max Re(lambda) = +0.1;
- hyperbolic compared max|Im(lambda)| against 10*max|lambda|, both reduced from the same array --
  a tautology, since |Im z| <= |z| for every complex z, so no input returned False;
- elliptic answered a whole-spectrum question ("do all eigenvalues share a sign") from a
  one-ended truncated sample.

A survey of the field found no production PDE library ships a GKS/Kreiss-Lopatinskii checker. The
one research package that does (``boundaryscheme``, Boutin-Le Barbenchon-Seguin) is restricted to
1D scalar constant-coefficient one-step explicit schemes and does not enforce its own CFL
hypothesis. What production codes ship instead is exactly what is here: the ingredients, as
numbers, with no verdict attached -- see Trixi.jl's ``jacobian_ad_forward``, whose documented use
records a measured growth rate and wraps it in no boolean at all.

**What the two abscissas mean.** For ``du/dt = L u``:

- The **spectral abscissa** ``alpha(L) = max Re(lambda)`` is a *necessary* condition only.
  ``alpha > 0`` proves unbounded growth. ``alpha <= 0`` does NOT bound ``||exp(tL)||`` at finite
  t when L is non-normal -- and every one-sided boundary stencil produces a non-normal L.
- The **numerical abscissa** ``omega(L) = lambda_max((L + L^H)/2)`` is *sufficient*:
  ``||exp(tL)||_2 <= exp(t*omega)`` for all t >= 0. So ``omega <= 0`` proves non-expansiveness.

``alpha <= omega`` always. Between them the answer is genuinely open, and transient growth lives
there; that gap is why a single boolean was the wrong return type.

Both are reported. Neither is thresholded here.

**References**:

[1] Gustafsson, B., Kreiss, H. O., & Oliger, J. (1995). Time Dependent Problems and Difference
    Methods. Wiley. -- for the GKS theory this module does NOT implement.
[2] Trefethen, L. N., & Embree, M. (2005). Spectra and Pseudospectra. Princeton. -- for why the
    spectral abscissa is insufficient for non-normal operators.

Created: 2026-01-18 (Issue #593 Phase 4.2). Rescoped 2026-08-08 (Issue #1859).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.sparse import csr_matrix, issparse
from scipy.sparse.linalg import ArpackError, eigs, eigsh

from mfgarchon.utils.mfg_logging import get_logger

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = get_logger(__name__)


@dataclass
class SpectralDiagnostics:
    """
    Measured spectral quantities for an operator. No verdict field, by design.

    Attributes:
        spectral_abscissa: max Re(lambda). Necessary only -- see module docstring.
        numerical_abscissa: lambda_max((L + L^H)/2), or None if not requested. Sufficient:
            ||exp(tL)||_2 <= exp(t * numerical_abscissa).
        max_abs_imag: max |Im(lambda)| over the computed eigenvalues.
        min_real_part: min Re(lambda) over the computed eigenvalues.
        eigenvalues: The eigenvalues actually computed.
        full_spectrum: True when every eigenvalue was computed. When False, `spectral_abscissa`
            is still the true maximum (the solver targets that end), but `min_real_part` and any
            statement about the whole spectrum are NOT supported by this data.
        operator_size: N.
        description: Caller-supplied label, for reporting.
    """

    spectral_abscissa: float
    numerical_abscissa: float | None
    max_abs_imag: float
    min_real_part: float
    eigenvalues: NDArray[np.complex128]
    full_spectrum: bool
    operator_size: int
    description: str

    def __str__(self) -> str:
        """Report the numbers and the theorem each one supports. No conclusion is drawn."""
        seen = f"all {self.operator_size}" if self.full_spectrum else f"{len(self.eigenvalues)} of {self.operator_size}"
        omega = "not computed" if self.numerical_abscissa is None else f"{self.numerical_abscissa:+.6e}"
        return (
            f"Spectral diagnostics: {self.description}\n"
            f"  eigenvalues computed:  {seen}\n"
            f"  spectral abscissa:     max Re(lambda)          = {self.spectral_abscissa:+.6e}\n"
            f"      > 0 proves growth; <= 0 does NOT bound ||exp(tL)|| for non-normal L\n"
            f"  numerical abscissa:    lambda_max((L+L^H)/2)   = {omega}\n"
            f"      <= 0 proves ||exp(tL)||_2 <= 1 for all t >= 0\n"
            f"  max |Im(lambda)|:      {self.max_abs_imag:.6e}\n"
            f"  min Re(lambda):        {self.min_real_part:+.6e}"
            + ("" if self.full_spectrum else "   (of the computed subset only)")
        )


def spectral_diagnostics(
    operator: csr_matrix | NDArray,
    description: str = "unknown",
    num_eigenvalues: int | None = None,
    full_spectrum: bool = False,
    with_numerical_abscissa: bool = True,
    max_dense_size: int = 2000,
) -> SpectralDiagnostics:
    """
    Measure spectral quantities of a discretized operator with BCs applied.

    Args:
        operator: Sparse or dense (N, N) matrix -- the spatial discretization including BCs.
        description: Label for reporting.
        num_eigenvalues: How many to compute on the sparse path. Default min(50, N-2).
        full_spectrum: Force the dense solver so every eigenvalue is computed. Required for any
            statement about the whole spectrum (definiteness, `min_real_part`); a truncated solve
            samples one end, and the other end is where a sign change would sit. Costs O(N^3):
            about 2s and 32MB at N=2000, measured.
        with_numerical_abscissa: Compute lambda_max((L + L^H)/2). This is the only quantity here
            that bounds ||exp(tL)||, so it is on by default.
        max_dense_size: Refuse `full_spectrum` above this N rather than silently truncating.

    Returns:
        SpectralDiagnostics -- numbers, no verdict. Interpretation is the caller's, and the
        module docstring states what each quantity does and does not prove.

    Raises:
        ValueError: if `full_spectrum` is requested and N > max_dense_size.

    Example:
        >>> diag = spectral_diagnostics(A, description="Neumann BC (2nd-order FDM)")
        >>> if diag.numerical_abscissa <= 0:
        ...     pass  # proven non-expansive in the 2-norm
        >>> if diag.spectral_abscissa > 0:
        ...     pass  # proven to grow
    """
    if not issparse(operator):
        operator = csr_matrix(operator)

    n = operator.shape[0]
    if num_eigenvalues is None:
        num_eigenvalues = min(50, n - 2)

    if full_spectrum and max_dense_size < n:
        raise ValueError(
            f"full_spectrum=True needs the dense solver, but N={n} exceeds "
            f"max_dense_size={max_dense_size}. Raise it to accept the O(N^3) cost, or use the "
            f"truncated solve and do not draw whole-spectrum conclusions from it."
        )

    use_dense = full_spectrum or n <= 100
    saw_everything = use_dense

    try:
        if use_dense:
            eigenvalues = np.linalg.eigvals(operator.toarray())
        else:
            # 'LR' -- largest REAL part, which is what the spectral abscissa is. Asking for
            # 'LM' (largest magnitude) returns the most negative eigenvalues of a discretized
            # Laplacian and misses the end that decides growth entirely (Issue #1859).
            eigenvalues, _ = eigs(operator, k=num_eigenvalues, which="LR", tol=1e-6)
    except Exception:
        logger.warning("Sparse eigensolver failed for N=%d, falling back to dense solver", n)
        eigenvalues = np.linalg.eigvals(operator.toarray())
        saw_everything = True

    numerical_abscissa: float | None = None
    if with_numerical_abscissa:
        # Hermitian part; its largest eigenvalue is the logarithmic norm in the 2-norm.
        symmetric_part = (operator + operator.conjugate().transpose()) * 0.5
        if use_dense or n <= 100:
            numerical_abscissa = float(np.linalg.eigvalsh(symmetric_part.toarray()).max())
        else:
            try:
                numerical_abscissa = float(
                    eigsh(symmetric_part, k=1, which="LA", tol=1e-6, return_eigenvectors=False)[0]
                )
            except ArpackError:
                # Deliberately not silenced into `None`. This is the only quantity here that
                # bounds ||exp(tL)||, so dropping it quietly would leave the caller holding the
                # necessary condition alone -- which is the state that made the retired API
                # misleading in the first place. Fall back to the exact solver when that is
                # affordable, and otherwise let the failure through.
                if max_dense_size < n:
                    raise
                logger.warning("ARPACK failed on the symmetric part at N=%d; using dense", n)
                numerical_abscissa = float(np.linalg.eigvalsh(symmetric_part.toarray()).max())

    real_parts = eigenvalues.real
    return SpectralDiagnostics(
        spectral_abscissa=float(real_parts.max()),
        numerical_abscissa=numerical_abscissa,
        max_abs_imag=float(np.abs(eigenvalues.imag).max()),
        min_real_part=float(real_parts.min()),
        eigenvalues=eigenvalues,
        full_spectrum=saw_everything,
        operator_size=n,
        description=description,
    )
