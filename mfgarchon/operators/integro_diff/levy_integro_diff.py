"""
Lévy integro-differential operator for jump-diffusion MFG.

Computes the non-local operator:

    J[v](x_i) = sum_k w_k [v(x_i + z_k) - v(x_i) - z_k * Dv(x_i)]

where {z_k, w_k} are quadrature nodes/weights for the Lévy measure nu.

Inherits from scipy.sparse.linalg.LinearOperator, following the same
pattern as operators/differential/ (LaplacianOperator, etc.).

Issue #923: Part of Layer 1 (Generalized PDE & Institutional MFG Plan).

Design constraints (from Dev Plan Rev 4):
- Adjoint: J* = W^{-1} J^T W where W = grid integration weights
- Tests MUST include non-uniform grid adjoint consistency
- File named levy_integro_diff.py to avoid confusion with interface_jump.py
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.sparse.linalg import LinearOperator

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from .levy_measures import LevyMeasure


class LevyIntegroDiffOperator(LinearOperator):
    """Non-local integro-differential operator for jump-diffusion processes.

    Computes J[v](x) on a 1D grid using Gauss-Legendre quadrature
    for the integral against the Lévy measure.

    Inherits LinearOperator: supports ``J @ v_flat`` and ``J(v)``.
    For finite-activity measures, ``as_sparse()`` returns explicit
    sparse matrix (enables implicit time stepping).

    Parameters
    ----------
    grid_points : NDArray
        1D spatial grid, shape (N,). Must be sorted.
    levy_measure : LevyMeasure
        Jump size distribution (GaussianJumps, CompoundPoissonJumps, etc.).
    intensity : float
        Overall jump intensity multiplier (default 1.0).
    compensate : bool
        Whether to include the compensator term -z*Dv(x) (default True).
        Set False for finite-activity processes where compensator is optional.
    quadrature_order : int
        Number of Gauss-Legendre quadrature points (default 32).
    interpolation : str
        Method for evaluating v at off-grid points: 'cubic' or 'linear'.

    Example
    -------
    >>> from mfgarchon.operators.integro_diff import LevyIntegroDiffOperator, GaussianJumps
    >>> x = np.linspace(0, 1, 101)
    >>> J = LevyIntegroDiffOperator(x, GaussianJumps(mu=0, sigma=0.1))
    >>> v = np.sin(np.pi * x)
    >>> Jv = J @ v  # Apply non-local operator
    """

    def __init__(
        self,
        grid_points: NDArray,
        levy_measure: LevyMeasure,
        intensity: float = 1.0,
        compensate: bool = True,
        quadrature_order: int = 32,
        interpolation: str = "cubic",
    ):
        N = len(grid_points)
        super().__init__(dtype=np.float64, shape=(N, N))
        self._grid = np.asarray(grid_points, dtype=np.float64)
        self._levy = levy_measure
        self._intensity = intensity
        self._compensate = compensate
        self._interp_method = interpolation

        # Precompute quadrature nodes and weights on Lévy measure support
        z_min, z_max = levy_measure.support_bounds()
        # Gauss-Legendre on [z_min, z_max]
        nodes_ref, weights_ref = np.polynomial.legendre.leggauss(quadrature_order)
        # Transform from [-1, 1] to [z_min, z_max]
        self._z_nodes = 0.5 * (z_max - z_min) * nodes_ref + 0.5 * (z_max + z_min)
        self._z_weights = 0.5 * (z_max - z_min) * weights_ref
        # Evaluate Lévy density at quadrature points
        self._nu_values = levy_measure.density(self._z_nodes)

        # Precompute grid spacing for gradient approximation
        self._dx = np.gradient(self._grid)

        # Precompute integration weights for adjoint (trapezoidal rule)
        self._integration_weights = np.ones(N)
        if N > 1:
            dx_arr = np.diff(self._grid)
            self._integration_weights[0] = dx_arr[0] / 2
            self._integration_weights[-1] = dx_arr[-1] / 2
            self._integration_weights[1:-1] = (dx_arr[:-1] + dx_arr[1:]) / 2

        # Cache for sparse matrix (built on first call to as_sparse)
        self._sparse_cache = None

    def _matvec(self, v_flat: NDArray) -> NDArray:
        """Apply J[v]. Core computation via quadrature + interpolation."""
        v = np.asarray(v_flat).ravel()
        N = len(self._grid)
        result = np.zeros(N)

        # Build interpolator for v at off-grid points
        if self._interp_method == "cubic":
            interp = CubicSpline(self._grid, v, bc_type="not-a-knot", extrapolate=True)
        else:
            from scipy.interpolate import interp1d

            interp = interp1d(self._grid, v, kind="linear", fill_value="extrapolate")

        # Compute gradient Dv for compensator term
        if self._compensate:
            dv = np.gradient(v, self._grid)

        # Quadrature: sum over jump sizes z_k
        for k in range(len(self._z_nodes)):
            z_k = self._z_nodes[k]
            w_k = self._z_weights[k]
            nu_k = self._nu_values[k]

            if abs(nu_k * w_k) < 1e-15:
                continue

            # v(x + z_k) via interpolation
            x_shifted = self._grid + z_k
            v_shifted = interp(x_shifted)

            # J[v](x) += w_k * nu_k * [v(x+z) - v(x) - z*Dv(x)]
            integrand = v_shifted - v
            if self._compensate:
                integrand -= z_k * dv

            result += w_k * nu_k * integrand

        return self._intensity * result

    def apply_adjoint(self, m: NDArray) -> NDArray:
        """Apply J*[m] for Fokker-Planck equation. Preserves total mass.

        The L^2-adjoint with grid integration weights W:
            J* = W^{-1} J^T W

        This ensures <J[v], m>_W = <v, J*[m]>_W for the discrete inner product
        <f, g>_W = sum_i W_i f_i g_i.

        Binding Constraint #2 from Dev Plan Rev 4.
        """
        W = self._integration_weights
        # J* = W^{-1} J^T (W m)
        Wm = W * np.asarray(m).ravel()
        JT_Wm = self._rmatvec(Wm)
        return JT_Wm / W

    def _rmatvec(self, v: NDArray) -> NDArray:
        """Apply J^T v (matrix transpose, NOT L^2 adjoint)."""
        # Build the sparse matrix and use its transpose
        J = self.as_sparse()
        return J.T @ v

    def as_sparse(self):
        """Assemble J as explicit sparse matrix.

        For finite-activity Lévy measures, J is a dense-banded matrix
        (each row has ~Q non-zero entries where Q = quadrature_order).
        Stored as CSR for efficient matrix-vector products.

        Enables implicit time stepping: (I - dt*J) v^{n+1} = rhs.
        """
        if self._sparse_cache is not None:
            return self._sparse_cache

        import scipy.sparse

        N = len(self._grid)
        # Build full matrix column by column (or use identity vectors)
        J_dense = np.zeros((N, N))
        e = np.zeros(N)
        for j in range(N):
            e[:] = 0.0
            e[j] = 1.0
            J_dense[:, j] = self._matvec(e)

        self._sparse_cache = scipy.sparse.csr_matrix(J_dense)
        return self._sparse_cache

    @property
    def grid(self) -> NDArray:
        """The spatial grid points."""
        return self._grid

    @property
    def integration_weights(self) -> NDArray:
        """Grid integration weights (trapezoidal rule)."""
        return self._integration_weights
