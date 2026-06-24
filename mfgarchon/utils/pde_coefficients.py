"""
Utilities for handling PDE coefficients (drift, diffusion) in MFG solvers.

This module provides unified handling of scalar, array, and callable coefficients,
eliminating code duplication across HJB, FP, and coupling solvers.
"""

from __future__ import annotations

import warnings
from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np

from mfgarchon.utils.deprecation import deprecated_alias

if TYPE_CHECKING:
    from collections.abc import Callable

    from mfgarchon.core.mfg_problem import MFGProblem


def fp_drift_coefficient(problem: Any) -> float:
    """Single-source the MFG drift coefficient ``c`` (drift ``= -c·∇U``) from the Hamiltonian's
    control law, not the independent ``coupling_coefficient`` field (Issue #1420 / gotcha G-017).

    For a smooth separable quadratic control cost ``H_control = |p|²/(2·control_cost)`` (MINIMIZE),
    the optimal feedback is ``α* = -∇U/control_cost``, so ``c = 1/control_cost``. ``control_cost`` is
    owned by the Hamiltonian (``problem.hamiltonian_class.control_cost.lambda_``) — the single source
    also consumed by ``H.optimal_control`` and the HJB Jacobian. The legacy ``coupling_coefficient``
    attribute is a private copy that must equal ``1/control_cost`` but silently diverged from it
    (``MFGProblem`` default 0.5), making the coupled solve converge to the wrong fixed point
    (G-017; exp16 Tier-2 had the Towel equilibrium ~4-5x too wide).

    Falls back to the legacy ``coupling_coefficient`` attribute when there is no quadratic MINIMIZE
    separable Hamiltonian to source from: a non-``SeparableHamiltonian`` Hamiltonian (e.g.
    ``QuadraticMFGHamiltonian``, which carries its own ``coupling_coefficient``) or a non-Hamiltonian
    direct solve. Non-smooth / congestion / MAXIMIZE control costs never reach the ``-c·∇U`` path
    (``resolve_fp_drift_kwargs`` routes them to the velocity ``drift_field`` channel), and
    ``CongestionHamiltonian`` is not a ``SeparableHamiltonian`` so it is excluded here by type. A
    ``coupling_coefficient`` of ``None`` (absent / explicitly unset) falls back to ``1.0``.
    """
    from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian

    h_class = getattr(problem, "hamiltonian_class", None)
    if (
        isinstance(h_class, SeparableHamiltonian)
        and isinstance(h_class.control_cost, QuadraticControlCost)
        and h_class.control_cost.sign == 1  # OptimizationSense.MINIMIZE
    ):
        return 1.0 / h_class.control_cost.lambda_
    cc = getattr(problem, "coupling_coefficient", None)
    return float(cc) if cc is not None else 1.0


def diffusion_from_volatility(
    sigma: float | np.ndarray,
    *,
    kind: str | None = None,
) -> float | np.ndarray:
    r"""Canonical PDE diffusion coefficient ``D`` from SDE volatility ``sigma`` (Issue #811).

    Single source of truth for the volatility -> diffusion conversion documented in
    ``NAMING_CONVENTIONS.md`` "Volatility vs Diffusion". Volatility is a tensor in general
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
    arr = np.asarray(sigma, dtype=float)
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


def diffusion_from_volatility_torch(sigma: Any) -> Any:
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
    sigma : float, numpy scalar, or torch.Tensor
        Volatility coefficient (SDE noise amplitude).

    Returns
    -------
    Same type as input (float, numpy scalar, or torch.Tensor)
        Diffusion coefficient ``D = 0.5 * sigma**2``.
    """
    return 0.5 * sigma**2


_VOLATILITY_LEGACY_KEY_WARNED: set[str] = set()


def resolve_volatility(
    problem_params: dict[str, Any],
    *,
    legacy_key: str | None = None,
    legacy_is_squared: bool = False,
    default: float,
) -> Any:
    r"""Resolve the SDE volatility ``sigma`` from a backend ``problem_params`` dict.

    Single source for the diffusion-coefficient lookup shared by the four computational
    backends (numpy / jax / torch / numba). Before this resolver the backends read the
    coefficient under **three different keys with three different semantics and defaults**
    -- numba ``"sigma"`` (default 1.0), numpy/jax ``"sigma_sq"`` (``sigma**2``, default 0.01),
    torch ``"diffusion"`` (the volatility despite the name, default 0.1) -- so a single
    ``problem_params`` dict could not be ported across backends without remapping
    (Issue #1282 item 3; coordinate with Issue #1189's ``"sigma"`` canonicalization).

    Resolution order (canonical first):

    1. ``problem_params["sigma"]`` -- the canonical SDE volatility key
       (``NAMING_CONVENTIONS.md`` "Volatility vs Diffusion"). Returned verbatim; the caller
       computes ``D = sigma**2 / 2``. No warning.
    2. ``problem_params[legacy_key]`` -- the backend's historical key, when ``legacy_key`` is
       given and present. ``legacy_is_squared=True`` means the stored value is ``sigma**2``
       (the numpy/jax ``"sigma_sq"`` key), so ``sqrt`` is applied; ``legacy_is_squared=False``
       means the stored value is already ``sigma`` (the torch ``"diffusion"`` key). Emits a
       **one-time** ``DeprecationWarning`` (per legacy key, per process) naming the canonical
       ``"sigma"`` key.
    3. ``default`` -- the backend's historical no-key default, so callers that omit the
       parameter see no behavior change.

    Returns ``sigma`` (not ``D``); each backend applies its own ``sigma -> D`` conversion
    (``0.5 * sigma**2`` for numpy/jax/numba, :func:`diffusion_from_volatility_torch` for torch)
    so the autograd / JIT computation graph is preserved. The square root uses ``value ** 0.5``
    rather than ``np.sqrt`` so torch tensors and jax tracers pass through their own dispatch
    unchanged.

    Parameters
    ----------
    problem_params : dict
        Backend problem-parameter dict.
    legacy_key : str | None
        The backend's historical dict key, or ``None`` (numba, already canonical).
    legacy_is_squared : bool
        ``True`` if ``problem_params[legacy_key]`` stores ``sigma**2`` (numpy/jax ``"sigma_sq"``).
    default : float
        Volatility used when neither the canonical nor the legacy key is present. Required
        (keyword-only) so each backend explicitly preserves its prior no-key default.

    Returns
    -------
    sigma : float | torch.Tensor | jax.Array
        The SDE volatility (same type as the stored value on the legacy/canonical path).
    """
    if "sigma" in problem_params:
        return problem_params["sigma"]
    if legacy_key is not None and legacy_key in problem_params:
        if legacy_key not in _VOLATILITY_LEGACY_KEY_WARNED:
            _VOLATILITY_LEGACY_KEY_WARNED.add(legacy_key)
            warnings.warn(
                f"problem_params key {legacy_key!r} is deprecated; use the canonical 'sigma' "
                f"key holding the SDE volatility (D = sigma**2/2) for cross-backend portability "
                f"(Issue #1282).",
                DeprecationWarning,
                stacklevel=2,
            )
        value = problem_params[legacy_key]
        return value**0.5 if legacy_is_squared else value
    return default


def scalar_diffusion_from_volatility(volatility_field: Any, fallback_sigma: Any) -> float:
    """Single scalar PDE diffusion ``D`` for solvers that assemble ``D * K`` with a scalar ``D``
    (the weak-form / FEM family).

    Routes the SDE-volatility -> PDE-diffusion conversion through the single source
    :func:`diffusion_from_volatility` (``D = sigma^2 / 2``; Issue #811) so the ``0.5 * sigma**2``
    formula is not re-copied per solver:

    - ``volatility_field is None`` -> ``D`` from ``fallback_sigma`` (the problem's ``sigma``);
    - scalar ``volatility_field`` -> ``D`` from that scalar;
    - array ``volatility_field`` -> ``D`` from ``mean(volatility_field)``, with a warning. These
      solvers cannot represent a spatially-varying field (``D`` multiplies the assembled stiffness
      as one scalar), so the field is collapsed to its mean -- made loud here rather than silent
      (Issue #1079-adjacent). Use an FDM/GFDM path with ``coefficient_field`` for true varying ``D``.

    Byte-identical to the prior inline ``0.5 * sigma**2`` / ``0.5 * mean(sigma)**2`` copies.
    """
    if volatility_field is None:
        return float(diffusion_from_volatility(fallback_sigma))
    if np.ndim(volatility_field) == 0:
        return float(diffusion_from_volatility(float(volatility_field)))
    warnings.warn(
        "Weak-form/FEM solver uses a single scalar diffusion D; the spatially-varying volatility "
        "field is collapsed to its mean (D = mean(sigma)^2 / 2). For a true varying-coefficient "
        "diffusion use an FDM/GFDM FP path with coefficient_field (Issue #1079).",
        UserWarning,
        stacklevel=2,
    )
    return float(diffusion_from_volatility(float(np.mean(volatility_field))))


class CoefficientMode(Enum):
    """
    Specifies which variables a callable coefficient depends on.

    Use this when your coefficient function doesn't depend on all (t, x, m) variables.

    Examples
    --------
    Time-only volatility:
    >>> sigma_t = lambda t: 0.1 + 0.05 * t
    >>> field = CoefficientField(sigma_t, 0.1, "volatility", mode=CoefficientMode.TIME)

    Space-only volatility:
    >>> sigma_x = lambda x: 0.1 * np.exp(-np.linalg.norm(x)**2)
    >>> field = CoefficientField(sigma_x, 0.1, "volatility", mode=CoefficientMode.SPACE)

    Density-dependent (degenerate) volatility:
    >>> sigma_m = lambda m: 0.1 * np.sqrt(m + 1e-6)
    >>> field = CoefficientField(sigma_m, 0.1, "volatility", mode=CoefficientMode.DENSITY)
    """

    FULL = "full"  # σ(t, x, m) - all three variables
    TIME = "time"  # σ(t) - time only
    SPACE = "space"  # σ(x) - space only
    DENSITY = "density"  # σ(m) - density only
    TIME_SPACE = "time_space"  # σ(t, x) - time and space
    TIME_DENSITY = "time_density"  # σ(t, m) - time and density
    SPACE_DENSITY = "space_density"  # σ(x, m) - space and density


class CoefficientField:
    """
    Unified interface for scalar, array, and callable PDE coefficients.

    Handles extraction and validation of diffusion and drift coefficients
    at specific timesteps during PDE solving.

    Parameters
    ----------
    field : None | float | ndarray | Callable
        The coefficient field:
        - None: Use default from problem
        - float: Constant coefficient
        - ndarray: Precomputed spatially/temporally varying coefficient
        - Callable: State-dependent coefficient with signature (t, x, m) -> float | ndarray
                   OR with keyword-only signature (*, t, x, m) for explicit dependencies
    default_value : float | ndarray
        Default value to use when field is None (typically problem.sigma or problem.drift)
    field_name : str
        Name of coefficient for error messages (e.g., "volatility_field", "drift_field")
    dimension : int
        Spatial dimension (1 for 1D, 2 for 2D, etc.)
    mode : CoefficientMode | str | None, optional
        Specifies which variables the callable depends on. Required for legacy positional
        callables that don't use all (t, x, m) arguments. Not needed for keyword-only callables.

    Examples
    --------
    Scalar diffusion:
    >>> field = CoefficientField(0.1, problem.sigma, "volatility_field", dimension=1)
    >>> sigma = field.evaluate_at(timestep=5, grid=x_coords, density=m)

    Array diffusion:
    >>> sigma_array = np.ones((Nt, Nx)) * 0.1
    >>> field = CoefficientField(sigma_array, problem.sigma, "volatility_field", dimension=1)
    >>> sigma = field.evaluate_at(timestep=5, grid=x_coords, density=m)

    Callable diffusion (modern keyword-only style):
    >>> sigma_tm = lambda *, t, m: 0.1 * t * np.sqrt(m)
    >>> field = CoefficientField(sigma_tm, problem.sigma, "volatility_field", dimension=1)
    >>> sigma = field.evaluate_at(timestep=5, grid=x_coords, density=m)

    Callable diffusion (legacy positional style with mode):
    >>> sigma_t = lambda t: 0.1 + 0.05 * t
    >>> field = CoefficientField(sigma_t, problem.sigma, "volatility", mode="time")
    >>> sigma = field.evaluate_at(timestep=5, grid=x_coords, density=m)

    Porous medium diffusion:
    >>> def porous_medium(*, m):
    ...     return 0.1 * np.sqrt(m + 1e-6)
    >>> field = CoefficientField(porous_medium, problem.sigma, "volatility_field", dimension=1)
    >>> sigma = field.evaluate_at(timestep=5, grid=x_coords, density=m)
    """

    def __init__(
        self,
        field: None | float | np.ndarray | Callable,
        default_value: float | np.ndarray,
        field_name: str = "coefficient",
        dimension: int = 1,
        mode: CoefficientMode | str | None = None,
    ):
        self.field = field
        self.default = default_value
        self.name = field_name
        self.dimension = dimension
        self.mode = CoefficientMode(mode) if isinstance(mode, str) else mode

        # Cache type checks
        self._is_none = field is None
        self._is_scalar = isinstance(field, (int, float))
        self._is_array = isinstance(field, np.ndarray)
        self._is_callable = callable(field)

    def evaluate_at(
        self,
        timestep_idx: int,
        grid: np.ndarray | tuple[np.ndarray, ...],
        density: np.ndarray,
        dt: float | None = None,
    ) -> float | np.ndarray:
        """
        Evaluate coefficient at specific timestep and state.

        Parameters
        ----------
        timestep_idx : int
            Timestep index for evaluation
        grid : ndarray | tuple[ndarray, ...]
            Spatial grid coordinates:
            - 1D: ndarray of shape (Nx,)
            - nD: tuple of coordinate arrays
        density : ndarray
            Density field at current timestep
        dt : float | None
            Timestep size (needed for computing physical time)

        Returns
        -------
        float | ndarray
            Evaluated coefficient (scalar or array matching density shape)
        """
        if self._is_none:
            return self.default

        elif self._is_scalar:
            return float(self.field)

        elif self._is_callable:
            return self._evaluate_callable(timestep_idx, grid, density, dt)

        elif self._is_array:
            return self._extract_from_array(timestep_idx, density.shape)

        else:
            raise TypeError(f"{self.name} must be None, float, ndarray, or Callable, got {type(self.field)}")

    def _evaluate_callable(
        self,
        timestep_idx: int,
        grid: np.ndarray | tuple[np.ndarray, ...],
        density: np.ndarray,
        dt: float | None,
    ) -> float | np.ndarray:
        """
        Evaluate callable coefficient with flexible signature support.

        Tries keyword-only arguments first (modern style), then falls back to
        mode-based positional arguments (legacy style).
        """
        # Compute physical time
        t_current = timestep_idx * dt if dt is not None else timestep_idx

        # Try keyword-only arguments first (modern, preferred style)
        try:
            result = self.field(t=t_current, x=grid, m=density)
            return self._validate_callable_output(result, density.shape, timestep_idx)
        except TypeError as e:
            # Check if it's a keyword-only callable that needs partial arguments
            error_msg = str(e)
            if "unexpected keyword argument" in error_msg or "missing" in error_msg:
                # Keyword-only callable - infer which arguments it needs
                # Try different combinations
                for args_dict in [
                    {"t": t_current, "m": density},
                    {"t": t_current, "x": grid},
                    {"x": grid, "m": density},
                    {"t": t_current},
                    {"x": grid},
                    {"m": density},
                ]:
                    try:
                        result = self.field(**args_dict)
                        return self._validate_callable_output(result, density.shape, timestep_idx)
                    except TypeError:
                        continue
            # Not keyword-only, fall through to mode-based evaluation

        # Mode-based evaluation (legacy positional arguments)
        if self.mode is None:
            # Default: assume full signature for backward compatibility
            result = self.field(t_current, grid, density)
        elif self.mode == CoefficientMode.FULL:
            result = self.field(t_current, grid, density)
        elif self.mode == CoefficientMode.TIME:
            result = self.field(t_current)
        elif self.mode == CoefficientMode.SPACE:
            result = self.field(grid)
        elif self.mode == CoefficientMode.DENSITY:
            result = self.field(density)
        elif self.mode == CoefficientMode.TIME_SPACE:
            result = self.field(t_current, grid)
        elif self.mode == CoefficientMode.TIME_DENSITY:
            result = self.field(t_current, density)
        elif self.mode == CoefficientMode.SPACE_DENSITY:
            result = self.field(grid, density)
        else:
            raise ValueError(f"Unknown coefficient mode: {self.mode}")

        return self._validate_callable_output(result, density.shape, timestep_idx)

    def _validate_callable_output(self, output: Any, expected_shape: tuple, timestep_idx: int) -> float | np.ndarray:
        """
        Validate callable coefficient output.

        Parameters
        ----------
        output : Any
            Output from callable coefficient
        expected_shape : tuple
            Expected shape (matching density)
        timestep_idx : int
            Current timestep index for error messages

        Returns
        -------
        float | ndarray
            Validated output (scalar or array)

        Raises
        ------
        TypeError
            If output is not float or ndarray
        ValueError
            If output shape doesn't match expected or contains NaN/Inf
        """
        # Handle scalar output
        if isinstance(output, (int, float)):
            return float(output)

        # Handle array output
        elif isinstance(output, np.ndarray):
            # Check shape
            if output.shape != expected_shape:
                raise ValueError(
                    f"Callable {self.name} returned array with shape {output.shape}, "
                    f"expected {expected_shape} (matching density shape) at timestep {timestep_idx}"
                )

            # Check for NaN/Inf
            if np.any(np.isnan(output)) or np.any(np.isinf(output)):
                raise ValueError(
                    f"Callable {self.name} returned NaN or Inf values at timestep {timestep_idx}. "
                    f"Check your coefficient function implementation."
                )

            return output

        else:
            raise TypeError(
                f"Callable {self.name} must return float or ndarray, got {type(output)} at timestep {timestep_idx}"
            )

    def _extract_from_array(self, timestep_idx: int, expected_shape: tuple) -> np.ndarray:
        """
        Extract coefficient from precomputed array.

        Handles both spatially-varying (ndim = dimension) and
        spatiotemporal (ndim = dimension + 1) arrays.

        Parameters
        ----------
        timestep_idx : int
            Current timestep index
        expected_shape : tuple
            Expected spatial shape

        Returns
        -------
        ndarray
            Extracted coefficient array

        Raises
        ------
        ValueError
            If array dimensions are incompatible
        """
        field_ndim = self.field.ndim

        # Spatially varying only: shape matches expected_shape
        if field_ndim == self.dimension:
            if self.field.shape != expected_shape:
                raise ValueError(
                    f"Spatial {self.name} array has shape {self.field.shape}, "
                    f"expected {expected_shape} (matching grid shape)"
                )
            return self.field

        # Spatiotemporal: shape is (Nt, *spatial_shape)
        elif field_ndim == self.dimension + 1:
            # Extract at timestep
            extracted = self.field[timestep_idx, ...]

            if extracted.shape != expected_shape:
                raise ValueError(
                    f"Spatiotemporal {self.name} array at timestep {timestep_idx} "
                    f"has shape {extracted.shape}, expected {expected_shape}"
                )
            return extracted

        else:
            raise ValueError(
                f"{self.name} array must have {self.dimension} dimensions (spatial) or "
                f"{self.dimension + 1} dimensions (spatiotemporal), got {field_ndim} dimensions"
            )

    def is_callable(self) -> bool:
        """Check if coefficient is callable (state-dependent)."""
        return self._is_callable

    def is_constant(self) -> bool:
        """Check if coefficient is constant (None or scalar)."""
        return self._is_none or self._is_scalar

    def is_array(self) -> bool:
        """Check if coefficient is precomputed array."""
        return self._is_array

    def validate_tensor_psd(
        self,
        sigma_tensor: float | np.ndarray,
        tolerance: float = 1e-10,
    ) -> None:
        """
        Validate that diffusion tensor is positive semi-definite (PSD).

        Works for all coefficient types:
        - Scalar σ²: Always PSD (if ≥ 0)
        - 1D tensor (1×1 matrix): Check value ≥ 0
        - Diagonal tensor: Check all diagonal entries ≥ 0
        - Full tensor (d×d): Check symmetry and eigenvalues ≥ 0

        Parameters
        ----------
        sigma_tensor : float | ndarray
            Diffusion coefficient or tensor to validate:
            - Scalar: σ² ≥ 0
            - Constant tensor: (d, d) array
            - Spatially varying: (N1, ..., Nd, d, d) array
            - Spatiotemporal: (Nt, N1, ..., Nd, d, d) array
        tolerance : float, optional
            Numerical tolerance for eigenvalue checking (default: 1e-10)

        Raises
        ------
        ValueError
            If tensor contains NaN/Inf, is not symmetric, or has negative eigenvalues

        Examples
        --------
        Scalar diffusion:
        >>> field = CoefficientField(0.1, 0.05, "volatility_field", dimension=2)
        >>> field.validate_tensor_psd(0.1)  # Pass

        Full tensor:
        >>> Sigma = np.array([[0.1, 0.02], [0.02, 0.1]])
        >>> field.validate_tensor_psd(Sigma)  # Pass (symmetric PSD)
        """
        # Scalar case: just check non-negative
        if isinstance(sigma_tensor, (int, float)):
            if sigma_tensor < 0:
                raise ValueError(f"{self.name} scalar must be non-negative, got {sigma_tensor}")
            return

        # Array case
        if not isinstance(sigma_tensor, np.ndarray):
            raise TypeError(f"{self.name} must be float or ndarray, got {type(sigma_tensor)}")

        # Check for NaN/Inf
        if not np.all(np.isfinite(sigma_tensor)):
            raise ValueError(f"{self.name} contains NaN or Inf values")

        # Determine tensor structure from shape
        shape = sigma_tensor.shape

        # Scalar-like array (0D or shape ())
        if sigma_tensor.ndim == 0 or shape == ():
            if sigma_tensor < 0:
                raise ValueError(f"{self.name} must be non-negative, got {sigma_tensor}")
            return

        # Single tensor: shape (d, d)
        if sigma_tensor.ndim == 2:
            self._check_single_tensor_psd(sigma_tensor, tolerance)
            return

        # Spatially varying or spatiotemporal: shape (..., d, d)
        if sigma_tensor.ndim >= 3 and shape[-2] == shape[-1]:
            tensor_dim = shape[-1]
            # Flatten spatial/temporal dimensions
            num_tensors = np.prod(shape[:-2])
            reshaped = sigma_tensor.reshape(num_tensors, tensor_dim, tensor_dim)

            # Check each tensor at each grid point
            for idx in range(num_tensors):
                try:
                    self._check_single_tensor_psd(reshaped[idx], tolerance)
                except ValueError as e:
                    # Add location information to error
                    multi_idx = np.unravel_index(idx, shape[:-2])
                    raise ValueError(f"{self.name} at grid point {multi_idx}: {e}") from None
            return

        # 1D array of diagonal values (for diagonal tensors): shape (d,)
        if sigma_tensor.ndim == 1:
            if np.any(sigma_tensor < 0):
                neg_indices = np.where(sigma_tensor < 0)[0]
                raise ValueError(
                    f"{self.name} diagonal entries must be non-negative. "
                    f"Found negative values at indices {neg_indices.tolist()}"
                )
            return

        # Spatially varying diagonal: shape (N1, ..., Nd, d)
        # Just check all values are non-negative
        if np.any(sigma_tensor < 0):
            raise ValueError(f"{self.name} contains negative values (all entries must be ≥ 0)")

    def has_mixed_derivatives(
        self,
        sigma_tensor: float | np.ndarray,
        tolerance: float = 1e-10,
    ) -> bool:
        """
        Check if diffusion tensor has off-diagonal terms (mixed derivatives).

        Standard ADI methods cannot handle mixed derivatives (∂²u/∂x∂y terms).
        Use this to detect when Craig-Sneyd or Hundsdorfer-Verwer schemes are needed.

        Parameters
        ----------
        sigma_tensor : float | ndarray
            Diffusion coefficient or tensor:
            - Scalar: No mixed derivatives (isotropic)
            - Diagonal vector (d,): No mixed derivatives
            - Full tensor (d, d): Check off-diagonal entries
            - Spatially varying (..., d, d): Check all tensors
        tolerance : float, optional
            Threshold for considering off-diagonal entries as zero (default: 1e-10)

        Returns
        -------
        bool
            True if tensor has significant off-diagonal terms (mixed derivatives)
            False if scalar, diagonal, or nearly-diagonal

        Examples
        --------
        Isotropic diffusion (no mixed):
        >>> field.has_mixed_derivatives(0.1)
        False

        Diagonal anisotropic (no mixed):
        >>> field.has_mixed_derivatives(np.array([0.1, 0.2]))
        False

        Full tensor with correlation (has mixed):
        >>> Sigma = np.array([[0.1, 0.02], [0.02, 0.1]])
        >>> field.has_mixed_derivatives(Sigma)
        True
        """
        # Scalar: isotropic, no mixed derivatives
        if isinstance(sigma_tensor, (int, float)):
            return False

        if not isinstance(sigma_tensor, np.ndarray):
            return False

        # 0D array: scalar-like
        if sigma_tensor.ndim == 0:
            return False

        # 1D array: diagonal entries only, no mixed derivatives
        if sigma_tensor.ndim == 1:
            return False

        # 2D array: single (d, d) tensor
        if sigma_tensor.ndim == 2:
            return self._tensor_has_off_diagonal(sigma_tensor, tolerance)

        # Higher dimensional: spatially varying tensors (..., d, d)
        if sigma_tensor.ndim >= 3 and sigma_tensor.shape[-2] == sigma_tensor.shape[-1]:
            tensor_dim = sigma_tensor.shape[-1]
            num_tensors = int(np.prod(sigma_tensor.shape[:-2]))
            reshaped = sigma_tensor.reshape(num_tensors, tensor_dim, tensor_dim)

            # Check if ANY tensor has off-diagonal terms
            return any(self._tensor_has_off_diagonal(reshaped[idx], tolerance) for idx in range(num_tensors))

        return False

    def _tensor_has_off_diagonal(self, tensor: np.ndarray, tolerance: float) -> bool:
        """Check if a single (d, d) tensor has significant off-diagonal entries."""
        off_diag = tensor - np.diag(np.diag(tensor))
        return np.max(np.abs(off_diag)) > tolerance

    def _check_single_tensor_psd(self, tensor: np.ndarray, tolerance: float) -> None:
        """
        Check that a single d×d tensor is symmetric and positive semi-definite.

        Parameters
        ----------
        tensor : ndarray
            Tensor of shape (d, d)
        tolerance : float
            Numerical tolerance for symmetry and eigenvalue checks

        Raises
        ------
        ValueError
            If tensor is not symmetric or has negative eigenvalues
        """
        # Check symmetry
        symmetric_diff = np.abs(tensor - tensor.T)
        max_asymmetry = np.max(symmetric_diff)

        if max_asymmetry > tolerance:
            raise ValueError(
                f"{self.name} must be symmetric. "
                f"Max asymmetry |Σ - Σᵀ| = {max_asymmetry:.2e} > tolerance {tolerance:.2e}"
            )

        # Check positive semi-definite via eigenvalues
        eigenvalues = np.linalg.eigvalsh(tensor)  # Hermitian/symmetric eigenvalues
        min_eigenvalue = np.min(eigenvalues)

        if min_eigenvalue < -tolerance:
            raise ValueError(
                f"{self.name} must be positive semi-definite. "
                f"Found negative eigenvalue: λ_min = {min_eigenvalue:.6e} < 0. "
                f"All eigenvalues: {eigenvalues}"
            )


def check_adi_compatibility(
    sigma: float | np.ndarray,
    tolerance: float = 1e-10,
) -> tuple[bool, str]:
    """
    Check if diffusion coefficient is compatible with standard ADI schemes.

    Standard ADI (Alternating Direction Implicit) cannot handle mixed derivatives
    (off-diagonal diffusion tensor terms like ∂²u/∂x∂y). This function detects
    when modified schemes (Craig-Sneyd, Hundsdorfer-Verwer) are needed.

    Parameters
    ----------
    sigma : float | ndarray
        Diffusion coefficient:
        - Scalar: σ² (isotropic) - ADI OK
        - Vector (d,): diagonal [σ₁², σ₂², ...] - ADI OK
        - Matrix (d, d): full tensor Σ - check off-diagonal
        - Spatially varying (..., d, d): check all tensors
    tolerance : float, optional
        Threshold for off-diagonal entries (default: 1e-10)

    Returns
    -------
    tuple[bool, str]
        (is_compatible, message)
        - is_compatible: True if standard ADI can be used
        - message: Description of diffusion type

    Examples
    --------
    >>> ok, msg = check_adi_compatibility(0.1)
    >>> print(ok, msg)
    True isotropic (scalar σ²)

    >>> ok, msg = check_adi_compatibility(np.array([0.1, 0.2]))
    >>> print(ok, msg)
    True diagonal anisotropic

    >>> Sigma = np.array([[0.1, 0.02], [0.02, 0.1]])
    >>> ok, msg = check_adi_compatibility(Sigma)
    >>> print(ok, msg)
    False full tensor with off-diagonal terms (mixed derivatives)
    """
    # Scalar
    if isinstance(sigma, (int, float)):
        return True, "isotropic (scalar σ²)"

    if not isinstance(sigma, np.ndarray):
        return True, f"unknown type {type(sigma)}, assuming compatible"

    # 0D array
    if sigma.ndim == 0:
        return True, "isotropic (scalar σ²)"

    # 1D array: diagonal
    if sigma.ndim == 1:
        return True, "diagonal anisotropic"

    # 2D array: (d, d) tensor
    if sigma.ndim == 2:
        off_diag = sigma - np.diag(np.diag(sigma))
        max_off = np.max(np.abs(off_diag))
        if max_off <= tolerance:
            return True, "diagonal tensor"
        else:
            return False, f"full tensor with off-diagonal terms (max={max_off:.2e}, mixed derivatives)"

    # Higher dimensional: spatially varying
    if sigma.ndim >= 3 and sigma.shape[-2] == sigma.shape[-1]:
        tensor_dim = sigma.shape[-1]
        num_tensors = int(np.prod(sigma.shape[:-2]))
        reshaped = sigma.reshape(num_tensors, tensor_dim, tensor_dim)

        max_off_diag = 0.0
        for idx in range(num_tensors):
            off_diag = reshaped[idx] - np.diag(np.diag(reshaped[idx]))
            max_off_diag = max(max_off_diag, np.max(np.abs(off_diag)))

        if max_off_diag <= tolerance:
            return True, "spatially varying diagonal tensor"
        else:
            return (
                False,
                f"spatially varying tensor with off-diagonal terms (max={max_off_diag:.2e}, mixed derivatives)",
            )

    return True, f"unknown structure (shape={sigma.shape}), assuming compatible"


def get_spatial_grid(problem: MFGProblem) -> np.ndarray | tuple[np.ndarray, ...]:
    """
    Get spatial grid coordinates for coefficient evaluation.

    Uses geometry-based API for spatial grid access.

    Parameters
    ----------
    problem : MFGProblem
        MFG problem instance

    Returns
    -------
    ndarray | tuple[ndarray, ...]
        Spatial coordinates:
        - 1D: ndarray of shape (Nx_points,)
        - nD: tuple of coordinate arrays for each dimension

    Examples
    --------
    1D problem:
    >>> grid = get_spatial_grid(problem)  # ndarray of x-coordinates

    2D problem:
    >>> grid = get_spatial_grid(problem)  # (x_coords, y_coords)
    """
    # Geometry-based API
    if hasattr(problem, "geometry") and hasattr(problem.geometry, "coordinates"):
        coords = problem.geometry.coordinates
        # For 1D, return single array; for nD, return tuple of arrays
        if len(coords) == 1:
            return coords[0]
        return tuple(coords)
    else:
        raise AttributeError("Problem must have geometry.coordinates attribute")


class _DriftDispatcher:
    """
    Internal data dispatcher for drift/velocity in FP equation (Issue #641).

    This is an **internal** class for FP solver plumbing. For user-facing
    MFG drift computation, use `DriftField` instead.

    Handles data dispatch (when/how to get drift values):
    1. Zero drift (None): Pure diffusion, no advection
    2. Array drift (ndarray): Precomputed U field, gradient computed in solver
    3. Callable drift (callable): Custom velocity function α(t, x, m)

    Parameters
    ----------
    drift_field : None | ndarray | Callable
        Drift specification:
        - None: Zero drift (pure diffusion)
        - ndarray: Precomputed U field, shape (Nt, *spatial_shape)
        - Callable: State-dependent drift α(t, x, m) -> velocity
    Nt : int
        Number of timesteps (used for zero drift allocation)
    spatial_shape : tuple
        Spatial grid shape
    dimension : int
        Spatial dimension
    """

    def __init__(
        self,
        drift_field: None | np.ndarray | Callable,
        Nt: int,
        spatial_shape: tuple,
        dimension: int = 1,
    ):
        self.field = drift_field
        self.Nt = Nt
        self.spatial_shape = spatial_shape
        self.dimension = dimension

        # Cache type checks
        self._is_none = drift_field is None
        self._is_array = isinstance(drift_field, np.ndarray)
        self._is_callable = callable(drift_field) and not isinstance(drift_field, np.ndarray)

        # For zero drift, create lazily
        self._zero_U: np.ndarray | None = None

    def is_callable(self) -> bool:
        """Check if using callable (state-dependent) drift."""
        return self._is_callable

    def is_zero(self) -> bool:
        """Check if using zero drift (pure diffusion)."""
        return self._is_none

    def is_array(self) -> bool:
        """Check if using precomputed array drift."""
        return self._is_array

    def get_U_at(self, timestep_idx: int) -> np.ndarray:
        """
        Get U slice at specific timestep for array/MFG drift.

        For implicit solvers that compute gradients internally.

        Parameters
        ----------
        timestep_idx : int
            Timestep index

        Returns
        -------
        ndarray
            U values at timestep, shape matches spatial_shape

        Raises
        ------
        ValueError
            If called on callable drift (use evaluate_velocity_at instead)
        """
        if self._is_callable:
            raise ValueError("Cannot get U slice for callable drift. Use evaluate_velocity_at() instead.")

        if self._is_none:
            # Lazy creation of zero U field
            if self._zero_U is None:
                self._zero_U = np.zeros((self.Nt, *self.spatial_shape))
            return self._zero_U[timestep_idx]

        if self._is_array:
            return self.field[timestep_idx]

        raise TypeError(f"Unexpected drift_field type: {type(self.field)}")

    def evaluate_velocity_at(
        self,
        timestep_idx: int,
        grid: np.ndarray | tuple[np.ndarray, ...],
        density: np.ndarray,
        dt: float | None = None,
    ) -> np.ndarray:
        """
        Evaluate velocity field at specific timestep for callable drift.

        For explicit solvers that use velocity directly.

        Parameters
        ----------
        timestep_idx : int
            Timestep index
        grid : ndarray | tuple[ndarray, ...]
            Spatial coordinates
        density : ndarray
            Current density field
        dt : float | None
            Timestep size for computing physical time

        Returns
        -------
        ndarray
            Velocity field at timestep

        Raises
        ------
        ValueError
            If called on non-callable drift (use get_U_at instead)
        """
        if not self._is_callable:
            raise ValueError("Cannot evaluate velocity for non-callable drift. Use get_U_at() instead.")

        # Compute physical time
        t_current = timestep_idx * dt if dt is not None else float(timestep_idx)

        # Call the velocity function
        result = self.field(t_current, grid, density)

        # Validate output
        if isinstance(result, (int, float)):
            result = np.full(self.spatial_shape, float(result))
        elif not isinstance(result, np.ndarray):
            raise TypeError(f"Drift callable must return float or ndarray, got {type(result)}")

        # Check for NaN/Inf
        if np.any(np.isnan(result)) or np.any(np.isinf(result)):
            raise ValueError(f"Drift callable returned NaN or Inf at timestep {timestep_idx}")

        return result


class _DriftArrayDispatcher(_DriftDispatcher):
    """Wraps a precomputed velocity array as a callable-like dispatcher.

    Issue #919: Allows FP solvers to receive velocity directly (not potential U).
    Reports is_callable()=True so the time-stepping loop routes through
    the explicit drift path (solve_timestep_explicit_with_drift), which
    accepts velocity directly.

    Parameters
    ----------
    velocity_array : np.ndarray
        Precomputed velocity field α*(t, x).
        1D: shape (Nt+1, Nx) — scalar velocity at each point.
        nD: shape (Nt+1, ndim, N1, N2, ...) — vector velocity per dimension.
    """

    def __init__(self, velocity_array: np.ndarray, Nt: int, spatial_shape: tuple, dimension: int = 1):
        # Don't call super().__init__ with the array — override behavior entirely
        self.field = velocity_array
        self.Nt = Nt
        self.spatial_shape = spatial_shape
        self.dimension = dimension
        self._is_none = False
        self._is_array = False  # Not a U-potential array
        self._is_callable = True  # Route through callable path
        self._zero_U = None

    def evaluate_velocity_at(
        self,
        timestep_idx: int,
        grid: np.ndarray | tuple[np.ndarray, ...],
        density: np.ndarray,
        dt: float | None = None,
    ) -> np.ndarray:
        """Return velocity slice at timestep (no computation needed)."""
        idx = min(timestep_idx, self.field.shape[0] - 1)
        return self.field[idx]

    def get_U_at(self, timestep_idx: int) -> np.ndarray:
        raise ValueError("_DriftArrayDispatcher provides velocity, not potential U.")


class DriftField:
    """
    MFG drift field: computes optimal control α* from value function U (Issue #623).

    Encapsulates the complete MFG coupling:
    1. Extract U slice at timestep
    2. Compute gradient ∇U using geometry
    3. Apply optimal control formula via ControlCostBase

    This is the user-facing class for MFG drift computation. Use this when you have:
    - A value function U from HJB solver
    - A control cost specification (quadratic, L1, bounded, etc.)
    - A geometry that provides gradient computation

    Parameters
    ----------
    U_solution : ndarray
        Value function from HJB solver, shape (Nt, *spatial_shape)
    control_cost : ControlCostBase
        Control cost specification (determines optimal_control formula)
    geometry : BaseGeometry
        Geometry providing gradient computation
    coupling_coefficient : float, optional
        Scaling factor for drift (default: 1.0)

    Examples
    --------
    Standard MFG coupling:
    >>> from mfgarchon.core import QuadraticControlCost, OptimizationSense
    >>> cost = QuadraticControlCost(sense=OptimizationSense.MINIMIZE, control_cost=1.0)
    >>> drift = DriftField(U_solution, cost, problem.geometry)
    >>> velocity = drift.get_velocity_at(k, density)

    L1 (bang-bang) control:
    >>> from mfgarchon.core import L1ControlCost
    >>> cost = L1ControlCost(control_cost=0.5)
    >>> drift = DriftField(U_solution, cost, problem.geometry)
    >>> velocity = drift.get_velocity_at(k, density)  # Returns ±1 or 0
    """

    def __init__(
        self,
        U_solution: np.ndarray,
        control_cost: Any,  # ControlCostBase, use Any to avoid import cycle
        geometry: Any,  # BaseGeometry
        coupling_coefficient: float = 1.0,
    ):
        self.U_solution = U_solution
        self.control_cost = control_cost
        self.geometry = geometry
        self.coupling_coefficient = coupling_coefficient

        # Cache dimensions
        self.Nt = U_solution.shape[0]
        self.spatial_shape = U_solution.shape[1:]

    def get_U_at(self, timestep_idx: int) -> np.ndarray:
        """Get value function slice at timestep."""
        return self.U_solution[timestep_idx]

    def get_gradient_at(self, timestep_idx: int) -> np.ndarray:
        """
        Compute gradient ∇U at timestep.

        Uses geometry's gradient computation if available,
        otherwise falls back to finite differences.
        """
        U_k = self.get_U_at(timestep_idx)

        # Try geometry-based gradient
        if hasattr(self.geometry, "compute_gradient"):
            return self.geometry.compute_gradient(U_k)

        # Fallback: simple finite differences
        spacing = self.geometry.get_grid_spacing()
        ndim = len(self.spatial_shape)

        if ndim == 1:
            # Central differences for interior, one-sided at boundaries
            grad = np.zeros_like(U_k)
            grad[1:-1] = (U_k[2:] - U_k[:-2]) / (2 * spacing[0])
            grad[0] = (U_k[1] - U_k[0]) / spacing[0]
            grad[-1] = (U_k[-1] - U_k[-2]) / spacing[0]
            return grad
        else:
            # For nD, return tuple of gradients per dimension
            grads = []
            for dim in range(ndim):
                grad = np.gradient(U_k, spacing[dim], axis=dim)
                grads.append(grad)
            return np.stack(grads, axis=0)

    def get_velocity_at(self, timestep_idx: int, density: np.ndarray | None = None) -> np.ndarray:
        """
        Compute optimal control velocity α* at timestep.

        This is the primary method for MFG coupling:
        α* = control_cost.optimal_control(∇U)

        Parameters
        ----------
        timestep_idx : int
            Timestep index
        density : ndarray | None
            Current density (for state-dependent control costs)
            Currently unused, reserved for future extensions

        Returns
        -------
        ndarray
            Optimal control velocity α*
        """
        # Get gradient
        grad_U = self.get_gradient_at(timestep_idx)

        # Apply coupling coefficient
        momentum = self.coupling_coefficient * grad_U

        # Apply optimal control formula
        return self.control_cost.optimal_control(momentum)

    def is_callable(self) -> bool:
        """DriftField is never callable (it's U-based)."""
        return False

    def is_zero(self) -> bool:
        """Check if all U values are zero."""
        return np.allclose(self.U_solution, 0)


# Backward compatibility alias (deprecated in v0.17.1)
MFGDriftField = deprecated_alias("MFGDriftField", DriftField, "v0.17.1")
