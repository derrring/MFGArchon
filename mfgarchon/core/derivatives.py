"""
Derivative Tensor Representation.

Standard representation for derivatives of any order in any dimension.
This replaces the legacy dict[tuple[int,...], float] format.

Tensor Convention:
    - Order p derivative in dimension d has shape (d,) * p = (d, d, ..., d)
    - Tensor index [i, j, k, ...] corresponds to ∂ᵖu/∂xᵢ∂xⱼ∂xₖ...

Examples:
    d=2 (2D):
        order 0: float                    u
        order 1: shape (2,)               [u_x, u_y]
        order 2: shape (2, 2)             [[u_xx, u_xy], [u_xy, u_yy]]
        order 3: shape (2, 2, 2)          [[[u_xxx, u_xxy], [u_xxy, u_xyy]],
                                           [[u_xxy, u_xyy], [u_xyy, u_yyy]]]

    d=3 (3D):
        order 1: shape (3,)               [u_x, u_y, u_z]
        order 2: shape (3, 3)             Hessian matrix
        order 3: shape (3, 3, 3)          Third-order tensor

Usage:
    from mfgarchon.core.derivatives import DerivativeTensors

    # Create from arrays
    grad = np.array([0.5, 0.3])
    hess = np.array([[0.1, 0.05], [0.05, 0.2]])
    derivs = DerivativeTensors.from_arrays(grad=grad, hess=hess)

    # Access
    derivs.grad           # array([0.5, 0.3])
    derivs.hess           # array([[0.1, 0.05], [0.05, 0.2]])
    derivs[1]             # Same as derivs.grad
    derivs[2]             # Same as derivs.hess
    derivs[2][0, 1]       # 0.05 (mixed partial ∂²u/∂x∂y)
    derivs.laplacian      # 0.3 (trace of Hessian)

    # In Hamiltonian
    def hamiltonian(x_idx, m, derivs: DerivativeTensors) -> float:
        return 0.5 * np.sum(derivs.grad ** 2)

See Also:
    archon-notes/development/guides/NAMING_CONVENTIONS.md (mfg-research, private) - Gradient Notation Standard section
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass
class DerivativeTensors:
    """
    Container for derivative tensors up to arbitrary order.

    Attributes:
        dimension: Spatial dimension d
        tensors: Dict mapping order p to tensor of shape (d,) * p

    The tensor at order p has shape (d, d, ..., d) with p dimensions.
    Index [i₁, i₂, ..., iₚ] gives ∂ᵖu/∂x_{i₁}∂x_{i₂}...∂x_{iₚ}
    """

    dimension: int
    tensors: dict[int, NDArray | float] = field(default_factory=dict)

    def __getitem__(self, order: int) -> NDArray | float:
        """Get derivative tensor of given order."""
        if order in self.tensors:
            return self.tensors[order]
        raise AttributeError(f"Derivative tensor of order {order} missing.")

    def __setitem__(self, order: int, tensor: NDArray | float) -> None:
        """Set derivative tensor of given order."""
        self.tensors[order] = tensor

    def __contains__(self, order: int) -> bool:
        """Check if derivative of given order is available."""
        return order in self.tensors

    @property
    def value(self) -> float:
        """Function value u (order 0)."""
        if 0 in self.tensors:
            return float(self.tensors[0])
        raise AttributeError("Function value (order 0) missing.")

    @property
    def grad(self) -> NDArray:
        """Gradient ∇u, shape (d,)."""
        if 1 in self.tensors:
            return self.tensors[1]
        raise AttributeError("Gradient (order 1) missing.")

    @property
    def hess(self) -> NDArray:
        """Hessian ∇²u, shape (d, d)."""
        if 2 in self.tensors:
            return self.tensors[2]
        raise AttributeError("Hessian (order 2) missing.")

    @property
    def third(self) -> NDArray:
        """Third-order derivatives, shape (d, d, d)."""
        if 3 in self.tensors:
            return self.tensors[3]
        raise AttributeError("Third-order derivatives (order 3) missing.")

    @property
    def laplacian(self) -> float:
        """Laplacian Δu = tr(∇²u) = Σᵢ ∂²u/∂xᵢ²."""
        if 2 in self.tensors:
            return float(np.trace(self.tensors[2]))
        raise AttributeError("Laplacian calculation failed: second-order derivatives (Hessian) missing.")

    @property
    def grad_norm_squared(self) -> float:
        """Squared gradient norm |∇u|² = Σᵢ (∂u/∂xᵢ)²."""
        if 1 in self.tensors:
            return float(np.sum(self.tensors[1] ** 2))
        raise AttributeError("Gradient norm calculation failed: first-order derivatives (gradient) missing.")

    @property
    def max_order(self) -> int:
        """Maximum derivative order available."""
        if not self.tensors:
            return -1
        return max(self.tensors.keys())

    @classmethod
    def from_arrays(
        cls,
        grad: NDArray | None = None,
        hess: NDArray | None = None,
        third: NDArray | None = None,
        value: float | None = None,
        **higher_order: NDArray,
    ) -> DerivativeTensors:
        """
        Create DerivativeTensors from arrays.

        Args:
            grad: Gradient array, shape (d,)
            hess: Hessian array, shape (d, d)
            third: Third-order tensor, shape (d, d, d)
            value: Function value (scalar)
            **higher_order: Additional tensors as order4=..., order5=..., etc.

        Returns:
            DerivativeTensors instance

        Example:
            >>> grad = np.array([0.5, 0.3])
            >>> hess = np.array([[0.1, 0.05], [0.05, 0.2]])
            >>> derivs = DerivativeTensors.from_arrays(grad=grad, hess=hess)
        """
        # Infer dimension from first available tensor
        dimension = None
        if grad is not None:
            dimension = len(grad)
        elif hess is not None:
            dimension = hess.shape[0]
        elif third is not None:
            dimension = third.shape[0]
        elif higher_order:
            first_tensor = next(iter(higher_order.values()))
            dimension = first_tensor.shape[0]

        if dimension is None:
            raise ValueError("At least one derivative tensor must be provided")

        tensors: dict[int, NDArray | float] = {}

        if value is not None:
            tensors[0] = value
        if grad is not None:
            tensors[1] = np.asarray(grad)
        if hess is not None:
            tensors[2] = np.asarray(hess)
        if third is not None:
            tensors[3] = np.asarray(third)

        # Handle higher-order tensors (order4, order5, etc.)
        for key, tensor in higher_order.items():
            if key.startswith("order"):
                order = int(key[5:])
                tensors[order] = np.asarray(tensor)

        return cls(dimension=dimension, tensors=tensors)

    @classmethod
    def from_gradient(cls, grad: NDArray, value: float | None = None) -> DerivativeTensors:
        """
        Create DerivativeTensors with only gradient (most common case).

        Args:
            grad: Gradient array, shape (d,)
            value: Optional function value

        Returns:
            DerivativeTensors with dimension inferred from grad
        """
        return cls.from_arrays(grad=grad, value=value)

    @classmethod
    def zeros(cls, dimension: int, max_order: int = 1) -> DerivativeTensors:
        """
        Create zero-initialized DerivativeTensors.

        Args:
            dimension: Spatial dimension d
            max_order: Maximum derivative order to include

        Returns:
            DerivativeTensors with zero tensors up to max_order
        """
        tensors: dict[int, NDArray | float] = {0: 0.0}

        for order in range(1, max_order + 1):
            shape = (dimension,) * order
            tensors[order] = np.zeros(shape)

        return cls(dimension=dimension, tensors=tensors)

    def with_gradient(self, grad: NDArray) -> DerivativeTensors:
        """Return new instance with updated gradient."""
        new_tensors = self.tensors.copy()
        new_tensors[1] = np.asarray(grad)
        return DerivativeTensors(dimension=self.dimension, tensors=new_tensors)

    def with_hessian(self, hess: NDArray) -> DerivativeTensors:
        """Return new instance with updated Hessian."""
        new_tensors = self.tensors.copy()
        new_tensors[2] = np.asarray(hess)
        return DerivativeTensors(dimension=self.dimension, tensors=new_tensors)


# =============================================================================
# Conversion from legacy formats (one-time migration)
# =============================================================================


def from_multi_index_dict(
    derivs: dict[tuple[int, ...], float],
    dimension: int | None = None,
) -> DerivativeTensors:
    """
    Convert legacy dict[tuple[int,...], float] to DerivativeTensors.

    Legacy format: {(1, 0): u_x, (0, 1): u_y, (2, 0): u_xx, ...}
    New format: DerivativeTensors with grad=[u_x, u_y], hess=[[u_xx, ...], ...]

    Args:
        derivs: Legacy multi-index dict
        dimension: Spatial dimension (inferred if not provided)

    Returns:
        DerivativeTensors instance

    Example:
        >>> old = {(1, 0): 0.5, (0, 1): 0.3, (2, 0): 0.1, (0, 2): 0.2, (1, 1): 0.05}
        >>> new = from_multi_index_dict(old)
        >>> new.grad  # array([0.5, 0.3])
        >>> new.hess  # array([[0.1, 0.05], [0.05, 0.2]])
    """
    if not derivs:
        raise ValueError("Empty derivatives dict")

    # Infer dimension from key length
    if dimension is None:
        first_key = next(iter(derivs.keys()))
        dimension = len(first_key)

    # Group by order (sum of multi-index)
    by_order: dict[int, dict[tuple[int, ...], float]] = {}
    for key, val in derivs.items():
        order = sum(key)
        if order not in by_order:
            by_order[order] = {}
        by_order[order][key] = val

    tensors: dict[int, NDArray | float] = {}

    # Order 0: scalar value
    if 0 in by_order:
        zero_key = tuple([0] * dimension)
        tensors[0] = by_order[0].get(zero_key, 0.0)

    # Order 1: gradient vector
    if 1 in by_order:
        grad = np.zeros(dimension)
        for key, val in by_order[1].items():
            # Key like (1, 0, 0) means ∂u/∂x₀
            idx = key.index(1)  # Find which dimension
            grad[idx] = val
        tensors[1] = grad

    # Order 2: Hessian matrix
    if 2 in by_order:
        hess = np.zeros((dimension, dimension))
        for key, val in by_order[2].items():
            # Key like (2, 0) means ∂²u/∂x₀², (1, 1) means ∂²u/∂x₀∂x₁
            indices = _multi_index_to_tensor_indices(key)
            if len(indices) == 2:
                i, j = indices
                hess[i, j] = val
                hess[j, i] = val  # Symmetric
        tensors[2] = hess

    # Order 3+: higher-order tensors
    for order in sorted(by_order.keys()):
        if order <= 2:
            continue
        shape = (dimension,) * order
        tensor = np.zeros(shape)
        for key, val in by_order[order].items():
            indices = _multi_index_to_tensor_indices(key)
            # Set value at all permutations (symmetric tensor)
            _set_symmetric(tensor, indices, val)
        tensors[order] = tensor

    return DerivativeTensors(dimension=dimension, tensors=tensors)


def _multi_index_to_tensor_indices(multi_index: tuple[int, ...]) -> list[int]:
    """
    Convert multi-index (α₁, α₂, ..., αd) to tensor indices [i, j, k, ...].

    Example:
        (2, 1, 0) in 3D means ∂³u/∂x₀²∂x₁ → indices [0, 0, 1]
        (1, 1) in 2D means ∂²u/∂x₀∂x₁ → indices [0, 1]
    """
    indices = []
    for dim, count in enumerate(multi_index):
        indices.extend([dim] * count)
    return indices


def _set_symmetric(tensor: NDArray, indices: list[int], value: float) -> None:
    """Set value at all permutations of indices (for symmetric tensor)."""
    from itertools import permutations

    for perm in set(permutations(indices)):
        tensor[perm] = value


def to_multi_index_dict(
    derivs: DerivativeTensors,
    max_order: int | None = None,
) -> dict[tuple[int, ...], float]:
    """
    Convert DerivativeTensors back to legacy dict format.

    Provided for backward compatibility during migration.

    Args:
        derivs: DerivativeTensors instance
        max_order: Maximum order to include (default: all available)

    Returns:
        Legacy format dict[tuple[int,...], float]
    """
    if max_order is None:
        max_order = derivs.max_order

    d = derivs.dimension
    result: dict[tuple[int, ...], float] = {}

    # Order 0
    if 0 in derivs and derivs[0] is not None:
        result[tuple([0] * d)] = float(derivs[0])

    # Order 1: gradient
    if 1 in derivs and derivs[1] is not None:
        grad = derivs[1]
        for i in range(d):
            key = tuple(1 if j == i else 0 for j in range(d))
            result[key] = float(grad[i])

    # Order 2: Hessian
    if 2 in derivs and derivs[2] is not None:
        hess = derivs[2]
        for i in range(d):
            for j in range(i, d):
                key = [0] * d
                key[i] += 1
                key[j] += 1
                result[tuple(key)] = float(hess[i, j])

    # Order 3+: higher-order tensors
    for order in range(3, max_order + 1):
        if order not in derivs or derivs[order] is None:
            continue
        tensor = derivs[order]
        # Iterate over unique multi-indices
        for multi_idx in _generate_multi_indices(d, order):
            indices = _multi_index_to_tensor_indices(multi_idx)
            result[multi_idx] = float(tensor[tuple(indices)])

    return result


def _generate_multi_indices(dimension: int, order: int):
    """Generate all multi-indices of given order in given dimension."""
    from itertools import combinations_with_replacement

    # Generate combinations and convert to multi-index
    for combo in combinations_with_replacement(range(dimension), order):
        multi_idx = [0] * dimension
        for dim in combo:
            multi_idx[dim] += 1
        yield tuple(multi_idx)
