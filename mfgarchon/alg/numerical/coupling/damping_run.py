"""One solve's application of a `DampingPolicy` (#2997): the per-field adapted bases, the histories, and Anderson's
memory on U. The policy itself, and the arithmetic of its schedules, is `mfgarchon.config.damping`."""

from __future__ import annotations

from typing import Any

from mfgarchon.config.damping import DampingPolicy, scheduled_theta


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

    def base(self, name: str) -> float:
        """Field ``name``'s base theta now: the initial one, or where adaptation has moved it."""
        return self._base[name]

    def theta(self, name: str, k: int) -> float:
        """``theta_k`` of field ``name`` ("U" or "M") at sweep ``k``."""
        spec = self.policy.U if name == "U" else self.policy.M
        if k == 0 and spec.first_sweep == "whole":
            return 1.0
        return scheduled_theta(spec.schedule, self._base[name], k, spec.floor)

    def update(self, k: int, U_old: Any, U_new: Any, M_old: Any, M_new: Any) -> tuple[Any, Any]:
        """The next iterate after sweep ``k``: each field blended by its ``theta_k``, U through Anderson when set. A
        field whose ``theta_k`` is 1 takes the map's output itself."""
        theta_U, theta_M = self.theta("U", k), self.theta("M", k)
        self.thetas["U"].append(theta_U)
        self.thetas["M"].append(theta_M)
        if self._anderson is not None:
            U = self._anderson.update(U_old.flatten(), U_new.flatten(), method="type1").reshape(U_old.shape)
        else:
            U = U_new if theta_U == 1.0 else theta_U * U_new + (1 - theta_U) * U_old
        M = M_new if theta_M == 1.0 else theta_M * M_new + (1 - theta_M) * M_old
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
