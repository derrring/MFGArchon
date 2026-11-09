"""
Learning Rate and Damping Schedules for Iterative Methods.

Provides decaying learning rate / damping schedules for iterative algorithms.
These are cross-paradigm infrastructure, consumed by:

- ``alg.numerical.coupling`` — Picard damping, Fictitious Play step size
- ``alg.neural`` — training learning rate decay (future)
- ``alg.optimization`` — primal-dual step size (future)

Architectural note (Issue #985):
    Schedules are iteration *infrastructure* (parameter control policy),
    not algorithms. They belong in ``utils/``, not ``alg/``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable


class LearningRateSchedule(Protocol):
    """Protocol for learning rate schedules."""

    def __call__(self, iteration: int) -> float:
        """
        Compute learning rate for given iteration.

        Args:
            iteration: Current iteration number (0-indexed)

        Returns:
            Learning rate alpha in (0, 1]
        """
        ...


def harmonic_schedule(k: int) -> float:
    """
    Harmonic learning rate: alpha(k) = 1 / (k + 1).

    This is the standard schedule for Fictitious Play with proven
    convergence for potential Mean Field Games.

    Properties:
        - Sum diverges: sum(1/(k+1)) = infinity (ensures exploration)
        - Sum of squares converges: sum(1/(k+1)^2) < infinity (ensures convergence)

    Args:
        k: Iteration number (0-indexed)

    Returns:
        Learning rate 1/(k+1)

    Example:
        >>> [harmonic_schedule(k) for k in range(5)]
        [1.0, 0.5, 0.333..., 0.25, 0.2]
    """
    return 1.0 / (k + 1)


def sqrt_schedule(k: int) -> float:
    """
    Square root learning rate: alpha(k) = 1 / sqrt(k + 1).

    Decays slower than harmonic, providing faster initial progress
    but potentially slower final convergence.

    Args:
        k: Iteration number (0-indexed)

    Returns:
        Learning rate 1/sqrt(k+1)

    Example:
        >>> [sqrt_schedule(k) for k in range(5)]
        [1.0, 0.707..., 0.577..., 0.5, 0.447...]
    """
    return 1.0 / np.sqrt(k + 1)


def polynomial_schedule(k: int, power: float = 0.6) -> float:
    """
    Polynomial learning rate: alpha(k) = 1 / (k + 1)^power.

    Interpolates between constant (power=0), sqrt (power=0.5),
    and harmonic (power=1) schedules.

    Args:
        k: Iteration number (0-indexed)
        power: Decay exponent (default 0.6)

    Returns:
        Learning rate 1/(k+1)^power

    Example:
        >>> [polynomial_schedule(k, power=0.6) for k in range(5)]
        [1.0, 0.659..., 0.517..., 0.435..., 0.380...]
    """
    return 1.0 / (k + 1) ** power


def constant_schedule(alpha: float = 0.5) -> Callable[[int], float]:
    """
    Constant learning rate (for standard Picard iteration).

    Args:
        alpha: Fixed learning rate (default 0.5)

    Returns:
        Schedule function that always returns alpha

    Example:
        >>> schedule = constant_schedule(0.7)
        >>> [schedule(k) for k in range(5)]
        [0.7, 0.7, 0.7, 0.7, 0.7]
    """

    def _schedule(k: int) -> float:
        return alpha

    return _schedule


def exponential_schedule(initial: float = 1.0, decay: float = 0.99) -> Callable[[int], float]:
    """
    Exponential decay: alpha(k) = initial * decay^k.

    Common in deep learning but generally not recommended for
    MFG/game-theoretic settings due to premature convergence.

    Args:
        initial: Initial learning rate
        decay: Decay factor per iteration

    Returns:
        Schedule function

    Example:
        >>> schedule = exponential_schedule(1.0, 0.9)
        >>> [schedule(k) for k in range(5)]
        [1.0, 0.9, 0.81, 0.729, 0.656...]
    """

    def _schedule(k: int) -> float:
        return initial * (decay**k)

    return _schedule


def warmup_schedule(
    warmup_steps: int = 10,
    base_schedule: Callable[[int], float] = harmonic_schedule,
) -> Callable[[int], float]:
    """
    Linear warmup followed by base schedule.

    Starts with small learning rate and linearly increases to 1.0
    over warmup_steps, then follows base_schedule.

    Args:
        warmup_steps: Number of warmup iterations
        base_schedule: Schedule to use after warmup

    Returns:
        Schedule function with warmup

    Example:
        >>> schedule = warmup_schedule(5, harmonic_schedule)
        >>> [schedule(k) for k in range(10)]
        [0.2, 0.4, 0.6, 0.8, 1.0, 0.166..., 0.142..., ...]
    """

    def _schedule(k: int) -> float:
        if k < warmup_steps:
            return (k + 1) / warmup_steps
        else:
            return base_schedule(k - warmup_steps)

    return _schedule


# Registry of named schedules for easy access
LEARNING_RATE_SCHEDULES: dict[str, Callable[[int], float]] = {
    "harmonic": harmonic_schedule,
    "sqrt": sqrt_schedule,
    "polynomial": lambda k: polynomial_schedule(k, power=0.6),
    "constant": constant_schedule(0.5),
}


def get_schedule(name: str | Callable[[int], float]) -> Callable[[int], float]:
    """
    Get a learning rate schedule by name or return custom callable.

    Args:
        name: Schedule name ("harmonic", "sqrt", "polynomial", "constant")
              or a custom callable

    Returns:
        Learning rate schedule function

    Raises:
        ValueError: If named schedule not found

    Example:
        >>> schedule = get_schedule("harmonic")
        >>> schedule(0)
        1.0
        >>> custom = get_schedule(lambda k: 0.1)
        >>> custom(100)
        0.1
    """
    if callable(name):
        return name

    if name not in LEARNING_RATE_SCHEDULES:
        raise ValueError(f"Unknown learning rate schedule: {name}. Available: {list(LEARNING_RATE_SCHEDULES.keys())}")

    return LEARNING_RATE_SCHEDULES[name]
