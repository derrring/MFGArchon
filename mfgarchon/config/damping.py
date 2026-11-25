"""How a coupling route damps its iterate: one policy per field, set by the user on every route (#2997, stage 1).

The policy is data and lives here, in the configuration layer, so that ``PicardConfig.damping`` can hold it; the
coupling package applies it to a solve (`mfgarchon.alg.numerical.coupling.damping_run.DampingRun`).

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

import numbers
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable

SCHEDULES = ("none", "constant", "harmonic", "sqrt", "polynomial", "exponential")


def scheduled_theta(schedule: str | Callable[[int], float], theta: float, k: int, floor: float) -> float:
    """``theta_k`` for a base ``theta`` at sweep ``k`` under ``schedule``, before the first-sweep rule."""
    if not isinstance(schedule, str):  # a callable k -> factor, the one other form FieldDamping admits
        return max(theta * float(schedule(k)), floor)
    if schedule == "none":
        return 1.0
    if schedule == "constant":
        return theta
    # theta times the schedule's factor, each written as the factor itself is, so the arithmetic is the rounding the
    # routes had: FixedPoint's base * schedule(k) and FictitiousPlay's alpha0 * schedule(k).
    if schedule == "harmonic":
        value = theta * (1.0 / (k + 1))
    elif schedule == "sqrt":
        value = theta * (1.0 / np.sqrt(k + 1))
    elif schedule == "polynomial":
        value = theta * (1.0 / (k + 1) ** 0.6)
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
        if not isinstance(self.adaptive, (bool, np.bool_)):
            raise TypeError(f"FieldDamping: adaptive must be a bool, got {self.adaptive!r}")
        object.__setattr__(self, "adaptive", bool(self.adaptive))
        if not 0.0 < self.adaptive_decay < 1.0:
            raise ValueError(f"FieldDamping: adaptive_decay must be in (0, 1), got {self.adaptive_decay!r}")
        if self.first_sweep not in ("damped", "whole"):
            raise ValueError(f"FieldDamping: first_sweep must be 'damped' or 'whole', got {self.first_sweep!r}")


@dataclass(frozen=True)
class AndersonOptions:
    """Anderson acceleration on U: the accelerator's memory ``depth`` (at least 1) and mixing ``beta`` in (0, 1]."""

    depth: int = 5
    beta: float = 1.0

    def __post_init__(self) -> None:
        if isinstance(self.depth, bool) or not isinstance(self.depth, numbers.Integral) or self.depth < 1:
            raise ValueError(f"AndersonOptions: depth must be an int >= 1, got {self.depth!r}")
        self._check_mixing()

    def _check_mixing(self) -> None:
        """``beta``, the accelerator's mixing, in (0, 1]. It is named like a Robin coefficient, so #2512's coefficient scan
        allows this method by name, and only this one."""
        if isinstance(self.beta, (bool, np.bool_)):
            raise ValueError(f"AndersonOptions: beta must be a real number in (0, 1], not a bool, got {self.beta!r}")
        if not 0.0 < self.beta <= 1.0:
            raise ValueError(f"AndersonOptions: beta must be a real number in (0, 1], got {self.beta!r}")


@dataclass(frozen=True)
class DampingPolicy:
    """U's and M's damping (`FieldDamping` each), and Anderson acceleration on U or None. With Anderson set, U's next
    iterate is the accelerator's, so U's `FieldDamping` is not read; M's still is."""

    U: FieldDamping = field(default_factory=FieldDamping)
    M: FieldDamping = field(default_factory=FieldDamping)
    anderson: AndersonOptions | None = None

    @classmethod
    def from_mapping(cls, mapping: dict[str, Any]) -> DampingPolicy:
        """The policy a mapping states, as ``PicardConfig.to_yaml`` writes it: ``U`` and ``M`` each a `FieldDamping`'s
        keywords, and ``anderson`` an `AndersonOptions`' keywords or null. A key outside these is refused by name."""
        unknown = sorted(set(mapping) - {"U", "M", "anderson"})
        if unknown:
            raise ValueError(f"DampingPolicy: a policy states U, M and anderson, not {unknown}")
        try:
            anderson = mapping.get("anderson")
            return cls(
                U=FieldDamping(**mapping.get("U", {})),
                M=FieldDamping(**mapping.get("M", {})),
                anderson=None if anderson is None else AndersonOptions(**anderson),
            )
        except TypeError as exc:
            raise ValueError(f"DampingPolicy: {exc}") from exc

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
