"""How a coupling route damps its iterate: one policy per field, set by the user on every route (#2997, stage 1).

A sweep maps the iterate ``(U_old, M_old)`` to the map's output ``(U_new, M_new)``; the next iterate of each field is
``theta_k * new + (1 - theta_k) * old``. `FieldDamping` decides ``theta_k`` for one field, and `DampingPolicy` holds
one for U and one for M, with Anderson acceleration as an option on U. The two fields are independent: what U's policy
decides never reads M's history, and the other way round.

``theta_k`` for a field with base ``theta``, floor ``floor`` and sweep ``k`` (0-based):

- ``schedule``: ``"constant"`` gives ``theta``; ``"harmonic"`` ``theta / (k + 1)``; ``"sqrt"``
  ``theta / sqrt(k + 1)``; ``"polynomial"`` ``theta / (k + 1)**0.6``; ``"exponential"`` ``theta**(k + 1)``; a callable
  ``f`` gives ``theta * f(k)``. Each decaying one is clamped below at ``floor``; ``"constant"`` is not. ``"none"``
  takes the map's output whole (``theta_k = 1``): the field is not damped.
- ``adaptive``: the base ``theta`` reacts to the field's own relative change after each sweep. It decays by
  ``adaptive_decay`` (not below ``floor``) when the change grows by more than 1.2 times, and recovers by 1.05 (not above
  the initial ``theta``) after three decreasing sweeps (#583). The schedule then applies to the adapted base.
- ``first_sweep``: ``"damped"`` applies the above at ``k = 0``; ``"whole"`` takes the map's output whole at ``k = 0``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable

SCHEDULES = ("none", "constant", "harmonic", "sqrt", "polynomial", "exponential")


def _scheduled(schedule: str | Callable[[int], float], theta: float, k: int, floor: float) -> float:
    """``theta_k`` for a base ``theta`` at sweep ``k`` under ``schedule``, before the first-sweep rule."""
    if callable(schedule):
        return max(theta * float(schedule(k)), floor)
    if schedule == "none":
        return 1.0
    if schedule == "constant":
        return theta
    if schedule == "harmonic":
        value = theta / (k + 1)
    elif schedule == "sqrt":
        value = theta / np.sqrt(k + 1)
    elif schedule == "polynomial":
        value = theta / (k + 1) ** 0.6
    elif schedule == "exponential":
        value = theta ** (k + 1)
    else:
        raise ValueError(f"unknown damping schedule {schedule!r}: one of {SCHEDULES}, or a callable k -> factor")
    return max(float(value), floor)


@dataclass(frozen=True)
class FieldDamping:
    """One field's damping: its base ``theta`` in (0, 1], ``schedule``, ``floor``, whether it adapts, and its first
    sweep (the module docstring states the rule)."""

    theta: float = 0.5
    schedule: str | Callable[[int], float] = "constant"
    floor: float = 0.05
    adaptive: bool = False
    adaptive_decay: float = 0.5
    first_sweep: Literal["damped", "whole"] = "damped"

    def __post_init__(self) -> None:
        if not 0.0 < self.theta <= 1.0:
            raise ValueError(f"FieldDamping: theta must be in (0, 1], got {self.theta!r}")
        if not (callable(self.schedule) or self.schedule in SCHEDULES):
            raise ValueError(f"FieldDamping: schedule must be one of {SCHEDULES} or a callable, got {self.schedule!r}")
        if not 0.0 <= self.floor <= 1.0:
            raise ValueError(f"FieldDamping: floor must be in [0, 1], got {self.floor!r}")
        if not 0.0 < self.adaptive_decay < 1.0:
            raise ValueError(f"FieldDamping: adaptive_decay must be in (0, 1), got {self.adaptive_decay!r}")
        if self.first_sweep not in ("damped", "whole"):
            raise ValueError(f"FieldDamping: first_sweep must be 'damped' or 'whole', got {self.first_sweep!r}")

    @property
    def damped(self) -> bool:
        """Whether this field is damped at all: its schedule is not ``"none"``."""
        return self.schedule != "none"


@dataclass(frozen=True)
class AndersonOptions:
    """Anderson acceleration on U: the accelerator's memory ``depth`` and mixing ``beta``."""

    depth: int = 5
    beta: float = 1.0


@dataclass(frozen=True)
class DampingPolicy:
    """U's and M's damping (`FieldDamping` each), and Anderson acceleration on U or None."""

    U: FieldDamping = field(default_factory=FieldDamping)
    M: FieldDamping = field(default_factory=FieldDamping)
    anderson: AndersonOptions | None = None

    @classmethod
    def constant(cls, theta: float = 0.5, theta_M: float | None = None) -> DampingPolicy:
        """``theta`` on U, and ``theta_M`` (``theta`` when None) on M, every sweep: FixedPoint's default at 0.5."""
        return cls(U=FieldDamping(theta=theta), M=FieldDamping(theta=theta if theta_M is None else theta_M))

    @classmethod
    def fictitious_play(
        cls,
        alpha0: float = 1.0,
        schedule: str | Callable[[int], float] = "harmonic",
        *,
        floor: float = 0.0,
        damp_value_function: bool = False,
    ) -> DampingPolicy:
        """Fictitious play: M averaged with ``alpha_k = alpha0 * schedule(k)``, no lower than ``floor``, and U the
        best response taken whole, or averaged the same way when ``damp_value_function``."""
        belief = FieldDamping(theta=alpha0, schedule=schedule, floor=floor)
        return cls(U=belief if damp_value_function else FieldDamping(schedule="none"), M=belief)

    def run(self) -> DampingRun:
        """The per-solve state that applies this policy: the adapted bases, the histories and the accelerator."""
        return DampingRun(self)


class DampingRun:
    """One solve's application of a `DampingPolicy`: the per-field adapted base, change history and theta history."""

    def __init__(self, policy: DampingPolicy) -> None:
        self.policy = policy
        self._base = {"U": policy.U.theta, "M": policy.M.theta}
        self._changes: dict[str, list[float]] = {"U": [], "M": []}
        self.thetas: dict[str, list[float]] = {"U": [], "M": []}
        self.adapted: list[dict[str, Any]] = []
        self._anderson = None
        if policy.anderson is not None:
            from .anderson_acceleration import AndersonAccelerator

            self._anderson = AndersonAccelerator(depth=policy.anderson.depth, beta=policy.anderson.beta)

    def theta(self, name: str, k: int) -> float:
        """``theta_k`` of field ``name`` ("U" or "M") at sweep ``k``."""
        spec = self.policy.U if name == "U" else self.policy.M
        if k == 0 and spec.first_sweep == "whole":
            return 1.0
        return _scheduled(spec.schedule, self._base[name], k, spec.floor)

    def update(self, k: int, U_old: Any, U_new: Any, M_old: Any, M_new: Any) -> tuple[Any, Any]:
        """The next iterate after sweep ``k``: each field blended by its ``theta_k``, U through Anderson when set."""
        theta_U, theta_M = self.theta("U", k), self.theta("M", k)
        self.thetas["U"].append(theta_U)
        self.thetas["M"].append(theta_M)
        if self._anderson is not None:
            U = self._anderson.update(U_old.flatten(), U_new.flatten(), method="type1").reshape(U_old.shape)
        else:
            U = theta_U * U_new + (1 - theta_U) * U_old
        M = theta_M * M_new + (1 - theta_M) * M_old
        return U, M

    def observe(self, k: int, change_U: float, change_M: float) -> list[str]:
        """Record sweep ``k``'s relative change of each field; each adaptive field adapts its base on its own history.
        Returns a message per decay, for the caller to log."""
        messages = []
        for name, change in (("U", change_U), ("M", change_M)):
            history = self._changes[name]
            history.append(float(change))
            spec = self.policy.U if name == "U" else self.policy.M
            if not spec.adaptive or len(history) < 2:
                continue
            ratio = history[-1] / history[-2] if history[-2] > 0 else 1.0
            if ratio > 1.2:
                self._base[name] = max(self._base[name] * spec.adaptive_decay, spec.floor)
                messages.append(
                    f"Adaptive damping: {name} change grew {ratio:.2f}x; its base theta is now {self._base[name]:.4f}."
                )
            elif len(history) >= 3 and all(history[-(i + 1)] < history[-(i + 2)] for i in range(2)):
                self._base[name] = min(self._base[name] * 1.05, spec.theta)
        if self.policy.U.adaptive or self.policy.M.adaptive:
            self.adapted.append({"iteration": k + 1, "theta_U": self._base["U"], "theta_M": self._base["M"]})
        return messages
