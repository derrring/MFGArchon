"""
Hamiltonian and Lagrangian abstractions for MFG optimal control.

Issues: #623 (original), #651 (duality), #667 (auto-diff), #673 (class-based API)

This module provides the mathematical foundation for coupling HJB and FP solvers
through a formal optimal control interface.

Mathematical Background
-----------------------
In MFG, agents minimize a cost functional:

    J[α] = E[∫₀ᵀ L(x, α, m) dt + g(x(T), m(T))]

where:
    - L(x, α, m): Running cost (Lagrangian)
    - α: Control (velocity/drift)
    - m: Population density

The HJB equation uses the Hamiltonian H, related to L via Legendre transform:

    H(x, p, m) = sup_α { p·α - L(x, α, m) }

The optimal control satisfies:

    α* = argmax_α { p·α - L(x, α, m) } = -∂H/∂p

Architecture (v0.17.2+)
-----------------------
Two complementary class hierarchies:

1. **Hamiltonian** (Issue #673): Full MFG Hamiltonian H(x, m, p, t)
   - Clean callable API: `H(x, m, p, t)` or `H(x, m, derivs, t)`
   - Auto-computed derivatives via `dp()` and `dm()` (Issue #667)
   - Supports state-dependent terms (congestion, potential)

2. **ControlCostBase** (original): Pure control cost L(α) or H(p)
   - Simpler interface for control-only Hamiltonians
   - `optimal_control(p)`, `lagrangian(α)`, `hamiltonian(p)`
   - Can be composed into Hamiltonian

3. **Lagrangian** (Issue #651): Running cost L(x, α, m, t)
   - Legendre transform to Hamiltonian
   - Duality: L ↔ H via `to_hamiltonian()` and `to_lagrangian()`

Design Philosophy
-----------------
Users typically think in terms of **running cost** (Lagrangian), but solvers
need the **Hamiltonian** and **optimal control formula**. This module:

1. Accepts either Lagrangian or Hamiltonian specification
2. Provides `optimal_control(p)` - the single source of truth for drift
3. Handles sign conventions via `OptimizationSense`
4. Auto-computes derivatives when not provided (Issue #667)

For common cases (quadratic, L1), closed-form formulas are provided.
For general cases, numerical Legendre transform is available.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Protocol

import numpy as np

# Issue #700: Import HamiltonianJacobians for jacobian_fd() method
from mfgarchon.types import HamiltonianJacobians

if TYPE_CHECKING:
    from numpy.typing import NDArray


class OptimizationSense(Enum):
    """
    Optimization direction for MFG problems.

    This enum captures the fundamental difference between:
    - Control theory: Agents MINIMIZE cost (α moves downhill on U)
    - Economics: Agents MAXIMIZE utility (α moves uphill on U)

    The sign convention in `optimal_control(p)` depends on this choice.

    Examples
    --------
    Cost minimization (standard MFG):
    >>> hamiltonian = QuadraticHamiltonian(sense=OptimizationSense.MINIMIZE)
    >>> alpha = hamiltonian.optimal_control(grad_U)  # Returns -grad_U/lambda

    Utility maximization (economics):
    >>> hamiltonian = QuadraticHamiltonian(sense=OptimizationSense.MAXIMIZE)
    >>> alpha = hamiltonian.optimal_control(grad_U)  # Returns +grad_U/lambda
    """

    MINIMIZE = "minimize"  # Control theory / HJB-FP: min ∫L dt, α = -∂H/∂p
    MAXIMIZE = "maximize"  # RL / Economics: max ∫R dt, α = +∂H/∂p


class ControlCostBase(ABC):
    """
    Kinetic/control cost component for MFG Hamiltonians.

    Internal component used by SeparableHamiltonian, CongestionHamiltonian, etc.
    NOT a standalone Hamiltonian -- use HamiltonianBase for full H(x,m,p,t).

    Subclasses must implement: optimal_control(), dp(), evaluate(), lagrangian().

    Issue #898: Redesigned interface. Key changes from pre-v0.19:
    - ``evaluate(p)`` replaces ``hamiltonian(p)`` -- always returns finite values
    - ``dp(p)`` added -- gradient/subdifferential of H w.r.t. p
    - ``lambda_`` replaces ``.control_cost`` attribute (naming collision fix)
    - ``regularize(epsilon)`` -- Moreau-Yosida smoothing
    - ``proximal(tau, z)`` -- proximal of Lagrangian for ADMM/variational solvers

    Parameters
    ----------
    sense : OptimizationSense
        Whether agents minimize cost or maximize utility
    control_cost : float
        Control cost weight lambda. Deprecated: use ``lambda_`` instead.
    lambda_ : float or None
        Control cost weight lambda. Preferred over ``control_cost``.
    """

    def __init__(
        self,
        sense: OptimizationSense = OptimizationSense.MINIMIZE,
        control_cost: float | None = None,
        *,
        lambda_: float | None = None,
    ):
        # Handle lambda_ vs control_cost (Issue #898: deprecation)
        if lambda_ is not None and control_cost is not None:
            raise ValueError("Cannot specify both 'lambda_' and 'control_cost'. Use 'lambda_' only.")
        if lambda_ is not None:
            lam = lambda_
        elif control_cost is not None:
            lam = control_cost
        else:
            lam = 1.0  # default

        if lam <= 0:
            raise ValueError(f"lambda_ must be positive, got {lam}")

        self.sense = sense
        self._lambda = lam
        # Sign convention: MINIMIZE -> alpha = -dH/dp, MAXIMIZE -> alpha = +dH/dp
        self.sign = 1 if sense == OptimizationSense.MINIMIZE else -1
        # Issue #1068: explicit None-init avoids hasattr() in regularize().
        # Subclasses (_MoreauYosidaControlCost) override this with the original base.
        self.base: ControlCostBase | None = None

    @property
    def lambda_(self) -> float:
        """Control cost weight lambda."""
        return self._lambda

    @property
    def control_cost(self) -> float:
        """Control cost weight lambda.

        .. deprecated:: 0.19.0
            Use ``lambda_`` instead. Will be removed in v0.25.0.
        """
        import warnings

        warnings.warn(
            "ControlCostBase.control_cost is deprecated, use .lambda_ instead. Will be removed in v0.25.0.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._lambda

    # === PRIMARY interface (solvers use these via HamiltonianBase delegation) ===

    @abstractmethod
    def optimal_control(self, p: np.ndarray) -> np.ndarray:
        """
        Compute optimal control alpha*(p). Single source of truth for drift.

        For cost minimization: alpha* = -dH/dp
        For utility maximization: alpha* = +dH/dp

        Parameters
        ----------
        p : ndarray
            Momentum field (typically nabla U from HJB solution)

        Returns
        -------
        ndarray
            Optimal control field, same shape as p
        """
        ...

    @abstractmethod
    def evaluate(self, p: np.ndarray) -> np.ndarray:
        """
        Evaluate H_control(p). Must return FINITE values for all p.

        This is the numerical evaluation used in the HJB equation.
        For non-smooth costs, returns the value obtained by substituting
        the optimal control: H(p) = p * alpha*(p) - L(alpha*(p)).

        Parameters
        ----------
        p : ndarray
            Momentum field

        Returns
        -------
        ndarray
            Hamiltonian value at each point (always finite)
        """
        ...

    @abstractmethod
    def dp(self, p: np.ndarray) -> np.ndarray:
        """
        Gradient dH/dp (or Clarke subdifferential selection for non-smooth H).

        Must return finite values for all p. For non-smooth H, return a
        well-defined selection from the subdifferential (e.g., 0 at kinks).

        Parameters
        ----------
        p : ndarray
            Momentum field

        Returns
        -------
        ndarray
            Gradient of H w.r.t. p, same shape as p
        """
        ...

    @abstractmethod
    def lagrangian(self, alpha: np.ndarray) -> np.ndarray:
        """
        Evaluate running cost L(alpha).

        Parameters
        ----------
        alpha : ndarray
            Control field

        Returns
        -------
        ndarray
            Running cost at each point
        """
        ...

    # === Admissible control set (Issue #1642, capability B3) ===

    def effective_domain(self) -> tuple[float, float] | None:
        """Admissible control set A = dom(L), i.e. where L(alpha) is finite.

        SINGLE OWNER of "which controls are admissible" (Issue #1642, B3).
        Every consumer that needs to bound a control -- optimization boxes,
        conjugate/Legendre sups, semi-Lagrangian control sweeps -- must read
        it from here rather than re-deriving it. ``SeparableLagrangian.control_bounds()``
        and ``HJBSemiLagrangianSolver._solve_timestep_dpp`` both delegate to this
        method; do not add a second isinstance ladder.

        Scope of that claim, stated exactly: consumers own how densely they
        SAMPLE A (a quadrature choice -- the DPP sweep still selects three points
        for a piecewise-linear L_ctrl against eleven for a quadratic one). What
        they may not do is decide what A IS. Narrowing that last sampling
        isinstance to a control-cost capability is Issue #1651.

        Returns
        -------
        tuple[float, float] | None
            ``(a_min, a_max)`` when A is a box, ``None`` when A = R^d.

        Notes
        -----
        ``lagrangian(alpha)`` does NOT enforce this domain: it evaluates the
        smooth branch everywhere and returns a FINITE value for infeasible
        alpha. ``BoundedControlCost(max_control=2).lagrangian(5.0)`` returns
        12.5 where the true L is +inf. Consumers that maximize
        ``p*alpha - L(alpha)`` must therefore restrict the search to this
        domain, or they silently overshoot -- for ``L1ControlCost(lambda_=0.5)``
        at p=5 the unrestricted sup returns 225.0 against a true H of 4.5.
        Making ``lagrangian()`` itself fail loud outside A is deferred to
        Issue #1644 (capability B4), which owns the guarded conjugate helper.
        """
        return None

    # === DEPRECATED: hamiltonian() -> use evaluate() ===

    def hamiltonian(self, p: np.ndarray) -> np.ndarray:
        """
        Evaluate Hamiltonian H(p).

        .. deprecated:: 0.19.0
            Use ``evaluate(p)`` instead. ``hamiltonian()`` may return +inf
            for non-smooth costs (L1). ``evaluate()`` always returns finite
            values. Will be removed in v0.25.0.
        """
        import warnings

        warnings.warn(
            "ControlCostBase.hamiltonian() is deprecated, use .evaluate() instead. "
            "evaluate() always returns finite values. Will be removed in v0.25.0.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.evaluate(p)

    # === REGULARIZATION ===

    def is_smooth(self) -> bool:
        """Whether H_control is C^1 in p. Default True, override for non-smooth."""
        return True

    def regularize(self, epsilon: float, method: str = "moreau-yosida") -> ControlCostBase:
        """Return a C^{1,1} smooth approximation of this control cost.

        Continuation-friendly: calling regularize() on an already-regularized
        cost re-regularizes the ORIGINAL base cost with the new epsilon.

        Parameters
        ----------
        epsilon : float
            Regularization parameter. Smaller = closer to original.
        method : str
            Regularization method. Currently: "moreau-yosida" (default).

        Returns
        -------
        ControlCostBase
            Smooth version. Returns self if already natively smooth.
        """
        # Issue #1068: use explicit None-init instead of hasattr().
        # self.base is None for plain costs; set to the original cost for wrappers.
        if self.is_smooth() and self.base is None:
            return self
        base = self.base if self.base is not None else self
        return _make_regularized(base, epsilon, method)

    # === VARIATIONAL / ADMM interface ===

    def proximal(self, tau: float, z: np.ndarray) -> np.ndarray:
        """Proximal of the Lagrangian: prox_{tau * L}(z).

        prox_{tau*L}(z) = argmin_alpha { L(alpha) + |alpha - z|^2 / (2*tau) }

        Required for ADMM and Chambolle-Pock variational solvers.
        Closed-form for standard costs. NotImplementedError for custom.

        NOTE: This is prox of L (Lagrangian), not prox of H (Hamiltonian).
        regularize() uses prox_H internally; ADMM uses prox_L.

        Parameters
        ----------
        tau : float
            Step size parameter
        z : ndarray
            Point to compute proximal at

        Returns
        -------
        ndarray
            Proximal point, same shape as z
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not provide an analytic proximal. "
            "Implement proximal() or use a standard cost (Quadratic, L1, Bounded)."
        )


class QuadraticControlCost(ControlCostBase):
    """
    Quadratic control cost: L(alpha) = lambda/2 * |alpha|^2.

    The most common choice in MFG:
    - Lagrangian: L(alpha) = lambda/2 * |alpha|^2
    - Hamiltonian: H(p) = |p|^2 / (2*lambda)
    - Optimal control: alpha* = -p/lambda (MINIMIZE), +p/lambda (MAXIMIZE)

    Parameters
    ----------
    sense : OptimizationSense
        Whether agents minimize cost or maximize utility
    control_cost : float
        Deprecated, use ``lambda_`` instead.
    lambda_ : float
        Control cost weight (higher = more costly to move). Default 1.0.

    Examples
    --------
    >>> cost = QuadraticControlCost(lambda_=2.0)
    >>> cost.evaluate(np.array([1.0, 2.0]))  # [0.25, 1.0]
    >>> cost.dp(np.array([1.0, 2.0]))        # [0.5, 1.0]
    """

    def optimal_control(self, p: np.ndarray) -> np.ndarray:
        """alpha* = -p/lambda (MINIMIZE) or +p/lambda (MAXIMIZE)."""
        return -self.sign * p / self._lambda

    def evaluate(self, p: np.ndarray) -> np.ndarray:
        """H(p) = |p|^2 / (2*lambda)."""
        return 0.5 * np.sum(p**2, axis=-1) / self._lambda

    def dp(self, p: np.ndarray) -> np.ndarray:
        """dH/dp = p / lambda."""
        return p / self._lambda

    def lagrangian(self, alpha: np.ndarray) -> np.ndarray:
        """L(alpha) = lambda/2 * |alpha|^2."""
        return 0.5 * self._lambda * np.sum(alpha**2, axis=-1)

    def proximal(self, tau: float, z: np.ndarray) -> np.ndarray:
        """prox_{tau * lambda/2 * |.|^2}(z) = z / (1 + tau * lambda)."""
        return z / (1 + tau * self._lambda)


class L1ControlCost(ControlCostBase):
    """
    Bounded bang-bang control: L(alpha) = lambda * |alpha| + I_{|alpha|<=1}.

    Models minimum fuel/effort with bounded control magnitude.
    The control is bang-bang: alpha* in {-1, 0, +1}.

    - Lagrangian: L(alpha) = lambda * |alpha| for |alpha| <= 1
    - Hamiltonian: H(p) = max(|p| - lambda, 0)  -- finite for all p
    - Optimal control: alpha* = -sign(p) if |p| > lambda, else 0

    NOTE: This models bounded bang-bang, not unbounded L1. The unbounded
    case (L = lambda*|alpha|, alpha in R) gives H = indicator (0 or +inf),
    which is numerically unusable. The bounded interpretation is consistent
    with optimal_control() returning +/-1.

    Parameters
    ----------
    sense : OptimizationSense
        Whether agents minimize cost or maximize utility
    control_cost : float
        Deprecated, use ``lambda_`` instead.
    lambda_ : float
        Control cost weight (activation threshold). Default 1.0.

    Examples
    --------
    >>> cost = L1ControlCost(lambda_=0.5)
    >>> cost.optimal_control(np.array([0.3, 0.7, -0.8]))
    array([ 0., -1.,  1.])
    >>> cost.evaluate(np.array([0.3, 0.7, -0.8]))
    array([0. , 0.2, 0.3])
    """

    def optimal_control(self, p: np.ndarray) -> np.ndarray:
        """Bang-bang: alpha* = -sign(p) where |p| > lambda, else 0."""
        alpha = np.zeros_like(p)
        active = np.abs(p) > self._lambda
        alpha[active] = -self.sign * np.sign(p[active])
        return alpha

    def evaluate(self, p: np.ndarray) -> np.ndarray:
        """H(p) = max(|p| - lambda, 0). Always finite."""
        return np.maximum(np.abs(np.atleast_1d(p)) - self._lambda, 0.0)

    def dp(self, p: np.ndarray) -> np.ndarray:
        """Clarke subdifferential: sign(p) where |p| > lambda, 0 otherwise."""
        p_arr = np.atleast_1d(p)
        result = np.zeros_like(p_arr, dtype=float)
        active = np.abs(p_arr) > self._lambda
        result[active] = np.sign(p_arr[active])
        return result

    def lagrangian(self, alpha: np.ndarray) -> np.ndarray:
        """L(alpha) = lambda * |alpha|."""
        return self._lambda * np.sum(np.abs(alpha), axis=-1)

    def effective_domain(self) -> tuple[float, float]:
        """A = [-1, 1]. The bang-bang bound, matching optimal_control()'s +/-1.

        This is the class's modelling choice (see the class docstring): the
        unbounded L1 cost has an indicator Hamiltonian and is numerically
        unusable, so A is normalized to the unit interval.
        """
        return (-1.0, 1.0)

    def proximal(self, tau: float, z: np.ndarray) -> np.ndarray:
        """prox_{tau * L}(z): soft-threshold then clip to [-1, 1]."""
        soft = np.sign(z) * np.maximum(np.abs(z) - tau * self._lambda, 0.0)
        return np.clip(soft, -1.0, 1.0)

    def is_smooth(self) -> bool:
        return False


class BoundedControlCost(ControlCostBase):
    """
    Bounded control with quadratic cost: L(alpha) = lambda/2 * |alpha|^2, |alpha| <= alpha_max.

    Models speed-limited agents. Quadratic in the interior, saturates at bounds.

    - Lagrangian: L(alpha) = lambda/2 * |alpha|^2 for |alpha| <= alpha_max, else +inf
    - Hamiltonian: H(p) = |p|^2/(2*lambda) for |p| <= lambda*alpha_max,
                          alpha_max*|p| - lambda*alpha_max^2/2 beyond
    - Optimal control: alpha* = clip(-p/lambda, -alpha_max, alpha_max)

    Parameters
    ----------
    sense : OptimizationSense
        Whether agents minimize cost or maximize utility
    control_cost : float
        Deprecated, use ``lambda_`` instead.
    lambda_ : float
        Control cost weight. Default 1.0.
    max_control : float
        Maximum control magnitude alpha_max. Default 1.0.

    Examples
    --------
    >>> cost = BoundedControlCost(lambda_=1.0, max_control=2.0)
    >>> cost.optimal_control(np.array([1.0, 3.0, 5.0]))
    array([-1., -2., -2.])
    """

    def __init__(
        self,
        sense: OptimizationSense = OptimizationSense.MINIMIZE,
        control_cost: float | None = None,
        max_control: float = 1.0,
        *,
        lambda_: float | None = None,
    ):
        super().__init__(sense, control_cost, lambda_=lambda_)
        if max_control <= 0:
            raise ValueError(f"max_control must be positive, got {max_control}")
        self.max_control = max_control

    def optimal_control(self, p: np.ndarray) -> np.ndarray:
        """alpha* = clip(-p/lambda, -alpha_max, alpha_max)."""
        alpha_unconstrained = -self.sign * p / self._lambda
        return np.clip(alpha_unconstrained, -self.max_control, self.max_control)

    def evaluate(self, p: np.ndarray) -> np.ndarray:
        """H(p) = sup_alpha { p . alpha - L(alpha) }. Always finite.

        Uses the unsigned optimizer (before MINIMIZE/MAXIMIZE sign convention)
        to compute the Hamiltonian value.
        """
        p_arr = np.atleast_1d(p)
        threshold = self._lambda * self.max_control
        # Quadratic region: H = |p|^2 / (2*lambda)
        h = 0.5 * p_arr**2 / self._lambda
        # Saturated region: H = alpha_max * |p| - lambda * alpha_max^2 / 2
        saturated = np.abs(p_arr) > threshold
        h[saturated] = self.max_control * np.abs(p_arr[saturated]) - 0.5 * self._lambda * self.max_control**2
        return h

    def dp(self, p: np.ndarray) -> np.ndarray:
        """dH/dp = p/lambda in unsaturated region, +/-alpha_max at saturation."""
        p_arr = np.atleast_1d(p)
        threshold = self._lambda * self.max_control
        result = p_arr / self._lambda
        saturated = np.abs(p_arr) > threshold
        result[saturated] = self.max_control * np.sign(p_arr[saturated])
        return result

    def lagrangian(self, alpha: np.ndarray) -> np.ndarray:
        """L(alpha) = lambda/2 * |alpha|^2. Finite everywhere; see effective_domain().

        The true L is +inf outside ``effective_domain()``. This method does NOT
        enforce that -- callers must restrict to the domain (Issue #1642, B3).
        """
        return 0.5 * self._lambda * np.sum(alpha**2, axis=-1)

    def effective_domain(self) -> tuple[float, float]:
        """A = [-max_control, max_control]."""
        return (-self.max_control, self.max_control)

    def proximal(self, tau: float, z: np.ndarray) -> np.ndarray:
        """prox_{tau * L}(z) = clip(z / (1 + tau*lambda), -alpha_max, alpha_max)."""
        return np.clip(z / (1 + tau * self._lambda), -self.max_control, self.max_control)

    def is_smooth(self) -> bool:
        return False  # C^1 but not C^2 at saturation boundary


# ============================================================================
# Regularization: Internal implementation (Issue #898)
# ============================================================================


def _make_regularized(base: ControlCostBase, epsilon: float, method: str) -> ControlCostBase:
    """Factory for regularized control costs. Internal, not exported."""
    if method == "moreau-yosida":
        return _MoreauYosidaControlCost(base, epsilon)
    raise ValueError(f"Unknown regularization method: {method!r}. Supported: 'moreau-yosida'")


class _MoreauYosidaControlCost(ControlCostBase):
    """Moreau-Yosida envelope of a ControlCostBase. C^{1,1} smooth. Internal.

    H_eps(p) = inf_q { H(q) + |p-q|^2 / (2*epsilon) }
    dH_eps/dp = (p - prox_{eps*H}(p)) / epsilon

    For L1ControlCost: prox = clip to [-lambda, lambda] (projection).
    For QuadraticControlCost: prox = p * lambda / (lambda + epsilon).
    """

    def __init__(self, base: ControlCostBase, epsilon: float):
        if epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {epsilon}")
        super().__init__(sense=base.sense, lambda_=base._lambda)
        self.base = base
        self.epsilon = epsilon

    def _prox_h(self, p: np.ndarray) -> np.ndarray:
        """Proximal of H (Hamiltonian), not L. Used for Moreau-Yosida."""
        p_arr = np.atleast_1d(p)
        if isinstance(self.base, L1ControlCost):
            # prox of max(|p|-lambda, 0): soft-threshold toward [-lambda, lambda]
            # For p > lambda: prox = p - epsilon * sign(p), but clamped
            # Actually: prox of H where H = max(|p|-lam, 0)
            # prox_{eps*H}(p) = p - eps * dH/dp if |p| > lam+eps, else clip
            lam = self.base._lambda
            result = np.copy(p_arr)
            above = p_arr > lam + self.epsilon
            below = p_arr < -(lam + self.epsilon)
            middle_pos = (p_arr > lam) & ~above
            middle_neg = (p_arr < -lam) & ~below
            result[above] = p_arr[above] - self.epsilon
            result[below] = p_arr[below] + self.epsilon
            result[middle_pos] = lam
            result[middle_neg] = -lam
            # |p| <= lam: prox = p (H=0, gradient=0)
            return result
        elif isinstance(self.base, QuadraticControlCost):
            lam = self.base._lambda
            return p_arr * lam / (lam + self.epsilon)
        elif isinstance(self.base, BoundedControlCost):
            # Proximal of bounded quadratic Hamiltonian
            lam = self.base._lambda
            a_max = self.base.max_control
            threshold = lam * a_max
            # Interior: prox of quadratic = rescale
            result = p_arr * lam / (lam + self.epsilon)
            # Saturated region: prox of linear = shift toward threshold
            sat_pos = p_arr > threshold + self.epsilon * a_max
            sat_neg = p_arr < -(threshold + self.epsilon * a_max)
            result[sat_pos] = p_arr[sat_pos] - self.epsilon * a_max
            result[sat_neg] = p_arr[sat_neg] + self.epsilon * a_max
            return result
        else:
            # General: pointwise scalar optimization
            from scipy.optimize import minimize_scalar

            result = np.empty_like(p_arr, dtype=float)
            for i in range(p_arr.size):
                pi = float(p_arr.flat[i])
                res = minimize_scalar(
                    lambda q, _pi=pi: float(self.base.evaluate(np.array([q]))) + (_pi - q) ** 2 / (2 * self.epsilon)
                )
                result.flat[i] = res.x
            return result

    def optimal_control(self, p: np.ndarray) -> np.ndarray:
        """Smooth optimal control from Moreau-Yosida gradient."""
        return -self.sign * self.dp(p)

    def evaluate(self, p: np.ndarray) -> np.ndarray:
        """H_eps(p) = H(prox(p)) + |p - prox(p)|^2 / (2*eps).

        The Moreau envelope is a functional of the whole momentum vector, so
        both terms are aggregated over the component axis. Base costs disagree
        on that convention (QuadraticControlCost already sums over axis=-1;
        L1ControlCost and BoundedControlCost return one value per component),
        so the base term is summed here rather than trusted to be a total.
        Adding a summed penalty to a per-component base term counted the
        penalty d times -- see tests/unit/test_core/test_hamiltonian_classes.py
        ``TestMoreauYosidaPenaltyAggregation``.
        """
        p_arr = np.atleast_1d(p)
        q = self._prox_h(p_arr)
        base_total = np.sum(self.base.evaluate(q), axis=-1)
        return base_total + np.sum((p_arr - q) ** 2, axis=-1) / (2 * self.epsilon)

    def dp(self, p: np.ndarray) -> np.ndarray:
        """dH_eps/dp = (p - prox(p)) / epsilon. Always Lipschitz."""
        p_arr = np.atleast_1d(p)
        return (p_arr - self._prox_h(p_arr)) / self.epsilon

    def lagrangian(self, alpha: np.ndarray) -> np.ndarray:
        """L_eps(alpha) = L(alpha) + epsilon/2 * |alpha|^2."""
        return self.base.lagrangian(alpha) + self.epsilon / 2 * np.sum(alpha**2, axis=-1)

    def effective_domain(self) -> tuple[float, float] | None:
        """Same domain as the wrapped cost.

        L_eps = L + (eps/2)|.|^2 adds a finite-everywhere term, so
        dom(L_eps) = dom(L). Moreau-Yosida smooths H, it does not enlarge A.
        """
        return self.base.effective_domain()

    def proximal(self, tau: float, z: np.ndarray) -> np.ndarray:
        """prox_{tau * L_eps}(z). L_eps = L + eps/2 |.|^2."""
        # prox of sum: for L + eps/2|.|^2, use Moreau decomposition or direct
        # For L1 base: L_eps = lambda|a| + eps/2|a|^2
        # prox_{tau*L_eps}(z) = soft_threshold(z/(1+tau*eps), tau*lambda/(1+tau*eps))
        if isinstance(self.base, L1ControlCost):
            denom = 1 + tau * self.epsilon
            z_scaled = z / denom
            thresh = tau * self.base._lambda / denom
            soft = np.sign(z_scaled) * np.maximum(np.abs(z_scaled) - thresh, 0.0)
            return np.clip(soft, -1.0, 1.0)
        # General: compose proximals
        # prox_{tau*(L + eps/2|.|^2)}(z) = prox_{tau'*L}(z/(1+tau*eps))
        # where tau' = tau/(1+tau*eps)
        denom = 1 + tau * self.epsilon
        tau_prime = tau / denom
        z_prime = z / denom
        return self.base.proximal(tau_prime, z_prime)

    def is_smooth(self) -> bool:
        return True


# Convenience aliases
QuadraticHamiltonian = QuadraticControlCost
L1Hamiltonian = L1ControlCost
BoundedHamiltonian = BoundedControlCost


# ============================================================================
# MFGOperator: Common base for Hamiltonian and Lagrangian (Issue #651)
# ============================================================================


class MFGOperatorBase(ABC):
    """
    Abstract base class for MFG operators (Hamiltonian and Lagrangian).

    This provides a common interface for both H(x, m, p, t) and L(x, α, m, t),
    enabling symmetric treatment via Legendre transform duality.

    Mathematical Background (Issue #651)
    -------------------------------------
    In optimal control, Hamiltonian and Lagrangian are Legendre duals:

        H(x, p, m, t) = sup_α { p·α - L(x, α, m, t) }   (Legendre transform)
        L(x, α, m, t) = sup_p { p·α - H(x, p, m, t) }   (Inverse transform)

    This duality means:
    - Users can specify either H or L (whichever is more natural)
    - The other is automatically derived via Legendre transform
    - Both have equal mathematical status

    Sign convention for the potential V and the coupling f (Issue #1642, B1)
    ------------------------------------------------------------------------
    Write the control part as L_ctrl(alpha) and H_ctrl(p), which are Legendre
    conjugates of each other, and let W(x, m, t) collect every alpha-independent
    term of L. Then the convention is fixed by the shipped HJB residual, not by
    preference:

        L(x, alpha, m, t) = L_ctrl(alpha) - V(x, t) - f(m)

    Derivation. The dynamic programming principle gives

        (u^n - u^{n+1}) / dt = min_alpha { L(x, alpha, m, t) + p . alpha }

    so with L = L_ctrl + W the PDE reads  -d_t u + H_ctrl(p) - W = 0.  The
    residual assembled in ``alg/numerical/hjb_solvers/base_hjb.py`` is
    ``Phi_U += (u^n - u^{n+1})/dt`` then ``Phi_U += H`` with H = H_ctrl + V + f,
    i.e.  -d_t u + H_ctrl + V + f = 0.  Matching the two gives W = -(V + f).

    Equivalently, and this is the form the round-trip test asserts: since V and
    f do not depend on alpha,

        sup_alpha { p . alpha - L(x, alpha, m, t) } = H_ctrl(p) - W = H(x, m, p, t)

    holds if and only if W = -(V + f). Adding V and f to BOTH H and L breaks
    conjugacy by exactly 2(V + f), constant in p.

    ``canonical_cs`` (``hjb_semi_lagrangian.py``) and the ``Dual*`` pair follow
    this convention. ``SeparableLagrangian.__call__`` does NOT -- it returns
    L_ctrl + V + f and is not self-conjugate against its own
    ``evaluate_hamiltonian``. That fork is tracked as Issue #1645 and recorded
    as strict xfails in ``tests/unit/test_core/test_hl_convention.py``.

    Where this convention is actually pinned, named precisely, because a
    conjugate round trip only discriminates when the two sides source V and f
    INDEPENDENTLY -- a Lagrangian obtained from H via ``legendre_transform()``
    yields H** == H for every convex H whatever the signs are, which asserts
    convexity and not a convention:

    - ``TestSeparableRoundTrip.test_separable_fork_gap_is_exactly_two_v_plus_f``
      -- 3 tests at one (V, f) point, one per control cost. Quantifies the
      Issue #1645 fork as an exact 2(V+f) gap, constant in p.
    - ``TestCongestionRoundTrip`` -- 12 tests, 3 control costs x 4 (V, f) cells,
      conjugating an analytic Lagrangian written out in the test module against
      ``CongestionHamiltonian``.

    ``test_conjugate_of_lagrangian_recovers_hamiltonian`` is not part of that
    pin: its V != 0 rows are strict xfail and its V0_f0 rows are
    non-discriminating, the disputed term being identically zero there.

    Note on the alpha sign. This class documents H = sup_alpha { p . alpha - L },
    while ``optimal_control`` returns the drift alpha* = -dH/dp (MINIMIZE). The
    two coincide for every control cost shipped here because each L_ctrl is even
    in alpha; the (V, f) conclusion above is independent of that choice, since V
    and f are alpha-independent either way.

    Parameters
    ----------
    sense : OptimizationSense
        Whether agents minimize cost (MINIMIZE) or maximize utility (MAXIMIZE)
    finite_diff_eps : float
        Step size for finite difference derivatives (default: 1e-6)
    """

    def __init__(
        self,
        sense: OptimizationSense = OptimizationSense.MINIMIZE,
        finite_diff_eps: float = 1e-6,
        population_index: int = 0,
    ):
        self.sense = sense
        self.finite_diff_eps = finite_diff_eps
        self.population_index = population_index
        # Sign convention: MINIMIZE -> α = -∂H/∂p, MAXIMIZE -> α = +∂H/∂p
        self._sign = 1 if sense == OptimizationSense.MINIMIZE else -1

    @property
    @abstractmethod
    def is_hamiltonian(self) -> bool:
        """Return True if this is a Hamiltonian operator."""
        ...

    @property
    @abstractmethod
    def is_lagrangian(self) -> bool:
        """Return True if this is a Lagrangian operator."""
        ...


# ============================================================================
# Hamiltonian: Full MFG Hamiltonian H(x, m, p, t) - Issue #673
# ============================================================================


@dataclass(frozen=True)
class HEvalState:
    """Pure evaluation state for the single-source Hamiltonian contract (Issue #1071).

    The minimal batch state the Hamiltonian needs to produce ``H``, ``∂H/∂p``, and
    ``∂H/∂m``. It carries physics inputs only — position, momentum, density, time —
    and deliberately holds NO discretization concern (no ``dt``, ``u``, grid, or
    boundary conditions); those stay solver-side. Single-point evaluation is the
    ``N=1`` case.

    Absorbs the former ``HamiltonianState`` (its ``x_idx`` grid-coupling field is
    dropped — operator assembly is the solver's responsibility, not the
    Hamiltonian's). Named ``HEvalState`` to avoid collision with the unrelated
    ``types/solver_types.SolverState`` alias.

    Attributes
    ----------
    x : NDArray
        Position(s), shape ``(N, d)`` (or ``(d,)`` for a single point).
    p : NDArray
        Momentum ``p = ∇u``, shape ``(N, d)``. Exposed as ``grad_u`` alias.
    m : float | NDArray
        Density, shape ``(N,)`` single-population or ``(K, N)`` stacked
        multi-population.
    t : float
        Time (default ``0.0``).
    """

    x: NDArray
    p: NDArray
    m: float | NDArray
    t: float = 0.0

    @property
    def grad_u(self) -> NDArray:
        """Alias for ``p`` — the momentum is the value-function gradient ``∇u``."""
        return self.p


@dataclass(frozen=True)
class HamiltonianValues:
    """Bundled Hamiltonian outputs for the convenience ``evaluate()`` pass (Issue #1071).

    Returned by :meth:`HamiltonianBase.evaluate` — a THIN convenience for explicit
    solvers (WENO/SL Lax-Friedrichs speed) and the future JAX spec that want ``H``
    and ``∂H/∂p`` (and σ) from one sweep. The implicit-Newton hot path does NOT use
    this; it calls the granular :meth:`HamiltonianBase.evaluate_H` /
    :meth:`HamiltonianBase.evaluate_dp` so the residual never computes ``∂H/∂p`` and
    the Jacobian never recomputes ``H``.

    ``∂H/∂m`` is deliberately NOT a field here — it is the separate
    :meth:`HamiltonianBase.dH_dm` (FP-coupling only), kept off this struct so the
    hot bundle stays a fixed-shape pytree.

    Attributes
    ----------
    H : NDArray
        Hamiltonian value, shape ``(N,)``.
    dH_dp : NDArray
        Momentum gradient ``∂H/∂p``, shape ``(N, d)``. Drift ``α* = -sign · ∂H/∂p``.
    sigma : NDArray
        PHYSICAL volatility σ, ALWAYS an array — shape ``(N,)`` (isotropic) or
        ``(N, d, d)`` (anisotropic). This is σ, NOT the diffusion coefficient
        ``D = σ²/2`` (folded at the stencil by ``diffusion_from_volatility``) and
        NOT a numerical viscosity (LLF/SUPG live at the solver level, never here).
    """

    H: NDArray
    dH_dp: NDArray
    sigma: NDArray


class Regularizer(Protocol):
    """Physics-only Hamiltonian regularizer (Issue #1071 scaffold).

    A value object that composes a *physical* regularization onto a Hamiltonian:
    Moreau–Yosida control-cost smoothing, anisotropic volatility, congestion
    mobility — terms whose strength ``epsilon`` vanishes on a chosen schedule, NOT
    with the mesh. Numerical viscosity (Local Lax–Friedrichs #1059, SUPG, SBP-SAT)
    is explicitly OUT of scope: it vanishes with the mesh, so it is a solver-level
    mixin consuming ``∂H/∂p`` + ``h``, never a Hamiltonian-level regularizer.

    Concrete regularizers land in later #1071 phases; this phase ships only the
    protocol + :meth:`HamiltonianBase.with_regularizer` composition hook.
    """

    epsilon: float

    def apply(self, h: HamiltonianBase) -> HamiltonianBase:
        """Return a NEW Hamiltonian with this regularization composed on."""
        ...


class HamiltonianBase(MFGOperatorBase):
    """
    Abstract base for full MFG Hamiltonians H(x, m, p, t).

    This is the primary interface for class-based Hamiltonians in MFG.
    Unlike ControlCostBase (which handles only H(p)), Hamiltonian
    supports full state dependence including position x, density m, and time t.

    Key Features (Issue #673)
    -------------------------
    - Clean callable API: `H(x, m, p, t)` returns Hamiltonian value
    - Auto-differentiation: `dp()` and `dm()` computed automatically (#667)
    - Composable: Can wrap ControlCostBase for control cost component
    - Extensible: Subclass for custom state-dependent Hamiltonians
    - Symmetric duality: `to_lagrangian()` converts to Lagrangian (#651)

    Mathematical Background
    -----------------------
    The HJB equation in MFG is:

        -∂u/∂t + H(x, m, ∇u, t) = 0

    Where H typically has the form:

        H(x, m, p, t) = H_control(p) + V(x, t) + f(m)

    With:
    - H_control(p): Control cost term (e.g., ½|p|²/λ for quadratic)
    - V(x, t): Potential/running cost
    - f(m): Density coupling (e.g., congestion)

    Parameters
    ----------
    sense : OptimizationSense
        Whether agents minimize cost or maximize utility.
        Affects sign of optimal control: α* = ∓∂H/∂p
    finite_diff_eps : float
        Step size for finite difference derivatives (default: 1e-6)

    Examples
    --------
    Basic usage with separable Hamiltonian:

    >>> H = SeparableHamiltonian(
    ...     control_cost=QuadraticControlCost(control_cost=2.0),
    ...     potential=lambda x, t: np.sin(x),
    ...     coupling=lambda m: m**2
    ... )
    >>> x, m, p, t = np.array([0.5]), 0.3, np.array([1.0]), 0.0
    >>> H(x, m, p, t)  # Evaluate Hamiltonian
    >>> H.dp(x, m, p, t)  # Get ∂H/∂p (auto-computed)
    >>> H.dm(x, m, p, t)  # Get ∂H/∂m (auto-computed)
    >>> L = H.legendre_transform()  # Convert to Lagrangian (Issue #651)

    See Also
    --------
    LagrangianBase : Running cost L(x, α, m, t)
    DualHamiltonian : Hamiltonian from Lagrangian via Legendre transform
    DualLagrangian : Lagrangian from Hamiltonian via inverse Legendre
    """

    @property
    def is_hamiltonian(self) -> bool:
        """Return True - this is a Hamiltonian operator."""
        return True

    @property
    def is_lagrangian(self) -> bool:
        """Return False - this is not a Lagrangian operator."""
        return False

    # === SINGLE-SOURCE CONTRACT (Issue #1071) ===

    # Physical volatility σ for the convenience evaluate(); ``None`` until the
    # σ-source unification (#1071 Phase 6) wires it from the problem/coefficient
    # field. The implicit-Newton hot path does NOT use this — base_hjb resolves σ
    # at timestep scope and passes it to the stencil. Set explicitly to call
    # evaluate(). Class-level default keeps object shape stable.
    physical_sigma: float | NDArray | None = None

    def evaluate_H(self, state: HEvalState) -> NDArray:
        """Hamiltonian value ``H`` over a batch (Issue #1071 granular primitive).

        The single home for the batch ``H`` call that every HJB solver used to
        inline as ``np.asarray(H_class(x, m, p, t=t), dtype=float)``. This is the
        residual-path primitive: it computes ``H`` only and never touches
        ``∂H/∂p``. Returns a float ``ndarray`` shaped as ``__call__`` returns it
        (callers ``.ravel()`` / reshape as their assembly needs).
        """
        return np.asarray(self(state.x, state.m, state.p, t=state.t), dtype=float)

    def evaluate_dp(self, state: HEvalState) -> NDArray:
        """Momentum gradient ``∂H/∂p`` over a batch (Issue #1071 granular primitive).

        The single home for the batch ``∂H/∂p`` call that HJB Jacobians used to
        inline as ``np.asarray(H_class.dp(x, m, p, t=t), dtype=float)``. This is the
        Jacobian-path primitive: it computes ``∂H/∂p`` only and never recomputes
        ``H``. Callers keep their own sign convention (drift ``α* = -∂H/∂p``).
        """
        return np.asarray(self.dp(state.x, state.m, state.p, t=state.t), dtype=float)

    def dH_dm(self, state: HEvalState) -> NDArray:
        """Density derivative ``∂H/∂m`` over a batch (Issue #1071, FP-coupling only).

        Separate from the residual/Jacobian primitives — the FP source term, not the
        HJB hot path — so it stays off the :class:`HamiltonianValues` bundle.
        """
        return np.asarray(self.dm(state.x, state.m, state.p, t=state.t), dtype=float)

    def evaluate(self, state: HEvalState) -> HamiltonianValues:
        """Convenience: ``H``, ``∂H/∂p`` and physical σ in one bundle (Issue #1071).

        THIN wrapper composing :meth:`evaluate_H` + :meth:`evaluate_dp` + the
        physical volatility, for explicit solvers (WENO/SL Lax–Friedrichs speed) and
        the future JAX spec. NOT used by the implicit-Newton hot path, which calls
        the granular primitives so the residual never computes ``∂H/∂p`` and the
        Jacobian never recomputes ``H``.
        """
        return HamiltonianValues(
            H=self.evaluate_H(state),
            dH_dp=self.evaluate_dp(state),
            sigma=self._resolve_physical_sigma(state),
        )

    def _resolve_physical_sigma(self, state: HEvalState) -> NDArray:
        """Resolve :attr:`physical_sigma` to a batch array for :meth:`evaluate`.

        Fail-fast when σ has not been set: the σ-source unification (#1071 Phase 6)
        will wire it from the problem; until then, callers of the convenience
        ``evaluate()`` must set ``physical_sigma`` explicitly.
        """
        if self.physical_sigma is None:
            raise ValueError(
                "HamiltonianBase.evaluate() needs a physical volatility but "
                "`physical_sigma` is None. Set it before calling evaluate(); the "
                "σ-source unification (Issue #1071 Phase 6) will wire it from the "
                "problem. The implicit-Newton hot path uses evaluate_H/evaluate_dp "
                "and does not require this."
            )
        sig = np.asarray(self.physical_sigma, dtype=float)
        if sig.ndim == 0:
            n = int(np.asarray(state.x).shape[0])
            return np.full(n, float(sig))
        return sig

    def with_regularizer(self, reg: Regularizer) -> HamiltonianBase:
        """Compose a PHYSICS-ONLY regularizer, returning a NEW Hamiltonian (Issue #1071).

        Functional composition scaffold for physical regularizers (Moreau–Yosida,
        anisotropic, congestion). Numerical viscosity (LLF/SUPG/SBP-SAT) is NOT a
        regularizer — it is a solver-level mixin (see :class:`Regularizer`). Concrete
        regularizers land in later #1071 phases.
        """
        return reg.apply(self)

    @abstractmethod
    def __call__(
        self,
        x: NDArray,
        m: float | NDArray,
        p: NDArray,
        t: float = 0.0,
    ) -> float | NDArray:
        """
        Evaluate Hamiltonian H(x, m, p, t).

        Parameters
        ----------
        x : NDArray
            Position, shape (d,) for d-dimensional problem
        m : float | NDArray
            Density at x
        p : NDArray
            Momentum ∇u at x, shape (d,)
        t : float
            Time (default: 0.0)

        Returns
        -------
        float | NDArray
            Hamiltonian value(s)
        """
        ...

    def dp(
        self,
        x: NDArray,
        m: float | NDArray,
        p: NDArray,
        t: float = 0.0,
    ) -> NDArray:
        """
        Compute ∂H/∂p (gradient w.r.t. momentum).

        This is used for:
        - Optimal control: α* = -sign * ∂H/∂p
        - Jacobian computation in Newton methods
        - Characteristic curves in semi-Lagrangian methods

        Default implementation uses finite differences (Issue #667).
        Override for analytic derivatives.

        Parameters
        ----------
        x : NDArray
            Position, shape (d,)
        m : float | NDArray
            Density at x
        p : NDArray
            Momentum ∇u at x, shape (d,)
        t : float
            Time

        Returns
        -------
        NDArray
            Gradient ∂H/∂p, shape (d,) for single-point, (N, d) for batch
        """
        p_arr = np.asarray(p)
        if p_arr.ndim == 2:
            # Issue #929: Try vectorized batch FD first (2d batch evals),
            # fall back to per-point loop if __call__ doesn't support batch
            eps = self.finite_diff_eps
            N, d = p_arr.shape
            try:
                grad = np.zeros((N, d))
                for i in range(d):
                    p_plus = p_arr.copy()
                    p_plus[:, i] += eps
                    p_minus = p_arr.copy()
                    p_minus[:, i] -= eps
                    H_plus = np.asarray(self(x, m, p_plus, t), dtype=float).ravel()
                    H_minus = np.asarray(self(x, m, p_minus, t), dtype=float).ravel()
                    grad[:, i] = (H_plus - H_minus) / (2 * eps)
                return grad
            except (TypeError, ValueError):
                # __call__ doesn't support batch m — fall back to per-point loop
                m_arr = np.asarray(m)
                return np.stack([self._finite_diff_dp(x[i], float(m_arr.flat[i]), p_arr[i], t) for i in range(N)])
        return self._finite_diff_dp(x, m, p, t)

    def dm(
        self,
        x: NDArray,
        m: float | NDArray,
        p: NDArray,
        t: float = 0.0,
    ) -> float | NDArray:
        """
        Compute ∂H/∂m (derivative w.r.t. density).

        This appears in the FP equation source term and is needed
        for coupling the HJB-FP system correctly.

        Default implementation uses finite differences (Issue #667).
        Override for analytic derivatives.

        Supports both single-point and batch inputs (Issue #775).
        For batch: p.shape = (N, d), returns shape (N,).
        For single-point: p.shape = (d,), returns float.

        Parameters
        ----------
        x : NDArray
            Position, shape (d,) or (N, d)
        m : float | NDArray
            Density at x, scalar or shape (N,)
        p : NDArray
            Momentum ∇u at x, shape (d,) or (N, d)
        t : float
            Time

        Returns
        -------
        float | NDArray
            Derivative ∂H/∂m
        """
        p_arr = np.asarray(p)
        if p_arr.ndim == 2:
            # Issue #929: Try vectorized batch FD first, fall back to per-point
            m_arr = np.asarray(m)
            eps = self.finite_diff_eps
            try:
                H_plus = np.asarray(self(x, m_arr + eps, p_arr, t), dtype=float).ravel()
                H_minus = np.asarray(self(x, m_arr - eps, p_arr, t), dtype=float).ravel()
                return (H_plus - H_minus) / (2 * eps)
            except (TypeError, ValueError):
                return np.array(
                    [self._finite_diff_dm(x[i], float(m_arr.flat[i]), p_arr[i], t) for i in range(p_arr.shape[0])]
                )
        return self._finite_diff_dm(x, m, p, t)

    def dx(
        self,
        x: NDArray,
        m: float | NDArray,
        p: NDArray,
        t: float = 0.0,
    ) -> NDArray:
        """Compute dH/dx (gradient w.r.t. position).

        Required for the full characteristic ODE system:
            dx/dt = dH/dp,  dp/dt = -dH/dx

        Default: central finite differences on __call__.
        Override for analytic (e.g., SeparableHamiltonian differentiates
        only the potential V(x,t)).

        Parameters
        ----------
        x : NDArray
            Position, shape (d,) or (N, d) for batch
        m : float | NDArray
            Density at x
        p : NDArray
            Momentum at x, shape (d,) or (N, d) for batch
        t : float
            Time

        Returns
        -------
        NDArray
            Gradient dH/dx, shape (d,) or (N, d)
        """
        x_arr = np.asarray(x)
        if x_arr.ndim == 2:
            p_arr = np.asarray(p)
            m_arr = np.asarray(m)
            return np.stack(
                [self._finite_diff_dx(x_arr[i], float(m_arr.flat[i]), p_arr[i], t) for i in range(x_arr.shape[0])]
            )
        return self._finite_diff_dx(x, m, p, t)

    def optimal_control(
        self,
        x: NDArray,
        m: float | NDArray,
        p: NDArray,
        t: float = 0.0,
    ) -> NDArray:
        """
        Compute optimal control α* given state and momentum.

        The optimal control satisfies the first-order condition:
        - MINIMIZE: α* = -∂H/∂p (gradient descent on value)
        - MAXIMIZE: α* = +∂H/∂p (gradient ascent on utility)

        Parameters
        ----------
        x : NDArray
            Position
        m : float | NDArray
            Density at x
        p : NDArray
            Momentum ∇u at x
        t : float
            Time

        Returns
        -------
        NDArray
            Optimal control α*, same shape as p
        """
        dH_dp = self.dp(x, m, p, t)
        return -self._sign * dH_dp

    def jacobian_fd(
        self,
        x: NDArray,
        m: float | NDArray,
        p: NDArray,
        dx: float,
        t: float = 0.0,
        scheme: str = "central",
    ) -> HamiltonianJacobians:
        """
        Compute FD Jacobian components for Newton/policy iteration.

        Uses chain rule: ∂H/∂U_j = ∂H/∂p · ∂p/∂U_j

        This method connects the continuous derivative dp() with the discrete
        finite difference Jacobian used by HJB solvers for Newton iteration.

        Issue #700: Unifies Hamiltonian class with HamiltonianJacobians.

        Parameters
        ----------
        x : NDArray
            Position
        m : float | NDArray
            Density at x
        p : NDArray
            Momentum ∇u at x (current gradient estimate)
        dx : float
            Grid spacing
        t : float
            Time (default: 0.0)
        scheme : str
            FD scheme: "central", "upwind_forward", "upwind_backward"
            - "central": p ≈ (U[i+1] - U[i-1])/(2dx)
            - "upwind_forward": p ≈ (U[i+1] - U[i])/dx
            - "upwind_backward": p ≈ (U[i] - U[i-1])/dx

        Returns
        -------
        HamiltonianJacobians
            Dataclass with diagonal, lower, upper tridiagonal components

        Example
        -------
        >>> H = SeparableHamiltonian(control_cost=QuadraticControlCost(1.0))
        >>> jac = H.jacobian_fd(x, m, p, dx=0.01, scheme="central")
        >>> # Use in Newton iteration:
        >>> A_diag = diffusion_diag + jac.diagonal
        >>> A_lower = diffusion_lower + jac.lower
        >>> A_upper = diffusion_upper + jac.upper
        """
        # Get ∂H/∂p from the class method
        dH_dp = self.dp(x, m, p, t)
        # Issue #1068: use np.ndim() instead of hasattr(, "__len__") to detect arrays.
        dH_dp_scalar = float(dH_dp[0]) if np.ndim(dH_dp) > 0 else float(dH_dp)

        if scheme == "central":
            # p ≈ (U[i+1] - U[i-1]) / (2dx)
            # ∂p/∂U[i+1] = +1/(2dx)
            # ∂p/∂U[i-1] = -1/(2dx)
            # ∂p/∂U[i] = 0
            coeff = dH_dp_scalar / (2 * dx)
            return HamiltonianJacobians(
                diagonal=np.array([0.0]),
                lower=np.array([-coeff]),  # ∂H/∂U[i-1]
                upper=np.array([coeff]),  # ∂H/∂U[i+1]
            )
        elif scheme == "upwind_forward":
            # p ≈ (U[i+1] - U[i]) / dx
            # ∂p/∂U[i+1] = +1/dx
            # ∂p/∂U[i] = -1/dx
            coeff = dH_dp_scalar / dx
            return HamiltonianJacobians(
                diagonal=np.array([-coeff]),
                lower=np.array([0.0]),
                upper=np.array([coeff]),
            )
        elif scheme == "upwind_backward":
            # p ≈ (U[i] - U[i-1]) / dx
            # ∂p/∂U[i] = +1/dx
            # ∂p/∂U[i-1] = -1/dx
            coeff = dH_dp_scalar / dx
            return HamiltonianJacobians(
                diagonal=np.array([coeff]),
                lower=np.array([-coeff]),
                upper=np.array([0.0]),
            )
        else:
            raise ValueError(f"Unknown FD scheme: {scheme}. Supported: 'central', 'upwind_forward', 'upwind_backward'")

    def _finite_diff_dp(
        self,
        x: NDArray,
        m: float | NDArray,
        p: NDArray,
        t: float,
    ) -> NDArray:
        """Compute ∂H/∂p using central finite differences."""
        eps = self.finite_diff_eps
        d = p.shape[0] if p.ndim > 0 else 1
        grad = np.zeros(d)

        p_flat = np.atleast_1d(p).astype(float)

        for i in range(d):
            p_plus = p_flat.copy()
            p_minus = p_flat.copy()
            p_plus[i] += eps
            p_minus[i] -= eps

            H_plus = self(x, m, p_plus, t)
            H_minus = self(x, m, p_minus, t)
            grad[i] = (H_plus - H_minus) / (2 * eps)

        return grad

    def _finite_diff_dm(
        self,
        x: NDArray,
        m: float | NDArray,
        p: NDArray,
        t: float,
    ) -> float:
        """Compute ∂H/∂m using central finite differences."""
        eps = self.finite_diff_eps
        m_scalar = float(m) if np.isscalar(m) else float(np.mean(m))

        H_plus = self(x, m_scalar + eps, p, t)
        H_minus = self(x, m_scalar - eps, p, t)

        return float((H_plus - H_minus) / (2 * eps))

    def _finite_diff_dx(
        self,
        x: NDArray,
        m: float | NDArray,
        p: NDArray,
        t: float,
    ) -> NDArray:
        """Compute dH/dx using central finite differences."""
        eps = self.finite_diff_eps
        x_flat = np.atleast_1d(x).astype(float)
        d = len(x_flat)
        grad = np.zeros(d)

        for i in range(d):
            x_plus = x_flat.copy()
            x_minus = x_flat.copy()
            x_plus[i] += eps
            x_minus[i] -= eps

            H_plus = self(x_plus, m, p, t)
            H_minus = self(x_minus, m, p, t)
            grad[i] = (H_plus - H_minus) / (2 * eps)

        return grad

    # === REGULARIZATION (Issue #898) ===

    def is_smooth(self) -> bool:
        """Whether H is C^1 in p. Default True. Subclasses override."""
        return True

    def regularize(self, epsilon: float, method: str = "moreau-yosida") -> HamiltonianBase:
        """Return smoothed Hamiltonian. Subclasses with ControlCostBase override.

        Raises NotImplementedError for HamiltonianBase subclasses that don't
        have a control_cost component (e.g., DualHamiltonian, custom subclasses).
        SeparableHamiltonian and CongestionHamiltonian override this to smooth
        their control cost while preserving structure.

        Parameters
        ----------
        epsilon : float
            Regularization parameter.
        method : str
            Method name (default: "moreau-yosida").

        Returns
        -------
        HamiltonianBase
            Smoothed Hamiltonian. Returns self if already smooth.
        """
        if self.is_smooth():
            return self
        raise NotImplementedError(
            f"{type(self).__name__}.regularize() is not implemented. "
            "Hamiltonians with a control_cost component (SeparableHamiltonian, "
            "CongestionHamiltonian) support this out of the box."
        )

    # Issue #673: to_legacy_func() removed - use class-based API directly

    def legendre_transform(
        self,
        p_bounds: tuple[float, float] | None = None,
        n_search: int = 100,
    ) -> LagrangianBase:
        """
        Convert Hamiltonian to Lagrangian via Legendre transform.

        Computes L(x, α, m, t) = sup_p { p·α - H(x, p, m, t) }

        The Legendre transform is involutive: applying it twice recovers
        the original (up to convexification). This provides symmetric
        duality between H and L (Issue #651).

        Parameters
        ----------
        p_bounds : tuple[float, float] | None
            Bounds on momentum for numerical optimization.
            If None, uses (-10, 10) as default.
        n_search : int
            Number of points for grid search (default: 100)

        Returns
        -------
        LagrangianBase
            The Legendre-transformed Lagrangian (DualLagrangian)

        Examples
        --------
        >>> H = SeparableHamiltonian(control_cost=QuadraticControlCost(control_cost=2.0))
        >>> L = H.legendre_transform()
        >>> # L(α) = ½λ|α|² (recovered from H = ½|p|²/λ)
        >>> H_back = L.legendre_transform()  # Involutive: back to Hamiltonian
        """
        return DualLagrangian(
            hamiltonian=self,
            sense=self.sense,
            p_bounds=p_bounds or (-10.0, 10.0),
            n_search=n_search,
        )


# =============================================================================
# Lagrangian: Running cost L(x, α, m, t) with Legendre transform - Issue #651
# =============================================================================

#: Search box used by the numerical fallbacks when ``control_bounds()`` returns
#: None (A = R^d). It is a solver convenience, NOT a modelling statement: a
#: maximizer/minimizer landing on its edge means the box truncated the answer,
#: so the numerics raise rather than return a silently truncated value.
_FALLBACK_CONTROL_BOUNDS: tuple[float, float] = (-10.0, 10.0)


class LagrangianBase(MFGOperatorBase):
    """
    Abstract base for Lagrangian (running cost) L(x, alpha, m, t).

    First-class MFG specification, parallel to HamiltonianBase.
    Users can specify either H or L; both provide optimal_control().

    Issue #904: Redesigned interface. Key additions:
    - ``optimal_control(x, m, p, t)`` -- same signature as HamiltonianBase
    - ``evaluate_hamiltonian(x, m, p, t)`` -- H value on-the-fly, no DualHamiltonian
    - ``proximal(tau, z)`` -- for ADMM/variational solvers
    - ``control_bounds()`` -- for semi-Lagrangian solver

    Duality: H(x, p, m, t) = sup_alpha { p . alpha - L(x, alpha, m, t) }

    Parameters
    ----------
    sense : OptimizationSense
        Whether agents minimize cost or maximize utility
    """

    @property
    def is_hamiltonian(self) -> bool:
        return False

    @property
    def is_lagrangian(self) -> bool:
        return True

    @abstractmethod
    def __call__(
        self,
        x: NDArray,
        alpha: NDArray,
        m: float | NDArray,
        t: float = 0.0,
    ) -> float | NDArray:
        """Evaluate L(x, alpha, m, t)."""
        ...

    # === Numerical search box: one owner for the control_bounds() fallback ===

    def _resolve_search_bounds(self) -> tuple[tuple[float, float], bool]:
        """Return ((a_min, a_max), used_fallback) for the numerical routines.

        Single owner of the ``control_bounds() or _FALLBACK_CONTROL_BOUNDS``
        decision, shared by conjugate_argmax() and proximal() so the two cannot
        drift apart on either the box or the truncation check.
        """
        declared = self.control_bounds()
        if declared is None:
            return _FALLBACK_CONTROL_BOUNDS, True
        return declared, False

    def _reject_fallback_truncation(
        self,
        alpha: NDArray,
        used_fallback: bool,
        routine: str,
    ) -> None:
        """Raise if a fallback-box search terminated on the box edge.

        When control_bounds() declares a real control set, hitting the edge is a
        genuine active constraint and is left alone. When the box is the
        _FALLBACK_CONTROL_BOUNDS stand-in, the edge carries no modelling meaning:
        the returned value is truncated, not optimal, so it must not be returned
        silently (Issue #1642 review).
        """
        if not used_fallback:
            return
        lo, hi = _FALLBACK_CONTROL_BOUNDS
        # scipy's bounded minimizers stop ~xatol (1e-5) shy of a binding bound;
        # 1e-5 * range gives ~20x margin over that without flagging the interior.
        tol = 1e-5 * (hi - lo)
        alpha_arr = np.atleast_1d(alpha)
        if np.any((alpha_arr <= lo + tol) | (alpha_arr >= hi - tol)):
            raise ValueError(
                f"{type(self).__name__}.{routine}: the numerical search hit the edge of "
                f"the fallback box {_FALLBACK_CONTROL_BOUNDS} at alpha={alpha_arr.tolist()}. "
                "control_bounds() returned None (A = R^d), so this box is a solver "
                "stand-in, not the control set -- the result is truncated, not optimal. "
                "Either override control_bounds() with the real admissible set, or "
                "override conjugate_argmax()/proximal() with the closed form."
            )

    # === Conjugate maximizer: the single owner of the numerical Legendre step ===

    def conjugate_argmax(
        self,
        x: NDArray,
        m: float | NDArray,
        p: NDArray,
        t: float = 0.0,
    ) -> NDArray:
        """argmax_alpha { p . alpha - L(x, alpha, m, t) } -- the maximizer of L*(p).

        **This is THE extension point for analytic subclasses.** Both
        optimal_control() and evaluate_hamiltonian() derive from it, so
        overriding it here supplies the closed form to both at once and keeps
        them on one sign convention. Overriding optimal_control() alone leaves
        evaluate_hamiltonian() on the numerical fallback.

        By the envelope theorem this equals dH/dp, which is NOT alpha*: the optimal
        control is alpha* = -sign * dH/dp (Issue #1642, capability B5). Keeping the
        maximization in one place is what stops optimal_control() and
        evaluate_hamiltonian() from re-forking into private sign conventions --
        before #1642 they held opposite ones, and only the two errors cancelling
        made evaluate_hamiltonian() return the right number.

        Default: 1D scalar optimization via scipy, L-BFGS-B in nD. Raises if the
        search terminates on the edge of the control_bounds() fallback box.
        """
        from scipy.optimize import minimize_scalar

        p_arr = np.atleast_1d(p)
        d = p_arr.shape[-1] if p_arr.ndim >= 2 else (1 if p_arr.ndim == 0 else len(p_arr))

        bounds, used_fallback = self._resolve_search_bounds()

        if d == 1:
            p_val = float(p_arr.flat[0])

            def neg_objective(a):
                alpha = np.array([a])
                return -(p_val * a - float(self(x, alpha, m, t)))

            res = minimize_scalar(neg_objective, bounds=bounds, method="bounded")
            argmax = np.array([res.x])
            self._reject_fallback_truncation(argmax, used_fallback, "conjugate_argmax")
            return argmax

        # nD: scipy.optimize.minimize
        from scipy.optimize import minimize as scipy_minimize

        def neg_objective_nd(alpha):
            return -(np.dot(p_arr.ravel(), alpha) - float(self(x, alpha, m, t)))

        x0 = np.clip(p_arr.ravel(), bounds[0], bounds[1])
        res = scipy_minimize(neg_objective_nd, x0, bounds=[bounds] * d, method="L-BFGS-B")
        self._reject_fallback_truncation(res.x, used_fallback, "conjugate_argmax")
        return res.x

    # === Optimal control (same signature AND same convention as HamiltonianBase) ===

    def optimal_control(
        self,
        x: NDArray,
        m: float | NDArray,
        p: NDArray,
        t: float = 0.0,
    ) -> NDArray:
        """Compute the optimal control alpha*, the drift the agent applies.

        Same convention as HamiltonianBase.optimal_control() (Issue #1642):
        - MINIMIZE: alpha* = -dH/dp
        - MAXIMIZE: alpha* = +dH/dp

        with dH/dp = argmax_alpha { p . alpha - L(x, alpha, m, t) }. Computed
        directly from L without constructing DualHamiltonian.

        Before #1642 this returned the bare argmax -- i.e. +p/lambda under
        MINIMIZE where HamiltonianBase and SeparableLagrangian both return
        -p/lambda -- so a subclass that did not override it got a drift of the
        wrong sign.

        Derives from conjugate_argmax(); **override that instead** for an
        analytic subclass. evaluate_hamiltonian() reads conjugate_argmax()
        directly, so an override placed here alone would not reach it.
        """
        return -self._sign * self.conjugate_argmax(x, m, p, t)

    # === On-the-fly H evaluation ===

    def evaluate_hamiltonian(
        self,
        x: NDArray,
        m: float | NDArray,
        p: NDArray,
        t: float = 0.0,
    ) -> float | NDArray:
        """H(x, m, p, t) = sup_alpha { p . alpha - L(x, alpha, m, t) } = L*(p).

        The convex conjugate, evaluated at its own maximizer. Matches
        DualHamiltonian.__call__ and the always-sup ControlCostBase.evaluate:
        the value does not depend on OptimizationSense (Issue #1185).

        Evaluated at ``conjugate_argmax``, NOT at ``optimal_control`` -- those
        differ by the sense sign, so substituting one for the other flips the
        Hamiltonian's sign under MINIMIZE (Issue #1642). Analytic subclasses
        therefore override ``conjugate_argmax``, which both consumers read.

        Computes the Hamiltonian value on-the-fly without DualHamiltonian.
        """
        dH_dp = self.conjugate_argmax(x, m, p, t)
        p_dot_alpha = float(np.sum(np.atleast_1d(p) * dH_dp))
        return p_dot_alpha - float(self(x, dH_dp, m, t))

    # === ADMM / variational interface ===

    def proximal(
        self,
        tau: float,
        z: np.ndarray,
        x: NDArray | None = None,
        m: float | NDArray | None = None,
        t: float = 0.0,
    ) -> np.ndarray:
        """Proximal of L: prox_{tau*L}(z) = argmin_alpha { L(alpha) + |alpha-z|^2/(2*tau) }.

        Required for ADMM and Chambolle-Pock variational solvers.
        Default: numerical (scipy). Override for closed-form.

        For separable L, only the control cost term matters (V, f don't
        depend on alpha), so x and m are unused.

        Raises if the search terminates on the edge of the control_bounds()
        fallback box -- the same truncation hazard as conjugate_argmax().
        """
        from scipy.optimize import minimize_scalar

        z_arr = np.atleast_1d(z)
        d = len(z_arr) if z_arr.ndim == 1 else 1
        bounds, used_fallback = self._resolve_search_bounds()

        # Dummy x, m if not provided
        if x is None:
            x = np.zeros(max(d, 1))
        if m is None:
            m = 0.0

        if d == 1:
            z_val = float(z_arr.flat[0])

            def objective(a):
                alpha = np.array([a])
                return float(self(x, alpha, m, t)) + (a - z_val) ** 2 / (2 * tau)

            res = minimize_scalar(objective, bounds=bounds, method="bounded")
            prox = np.array([res.x])
            self._reject_fallback_truncation(prox, used_fallback, "proximal")
            return prox

        from scipy.optimize import minimize as scipy_minimize

        def objective_nd(alpha):
            return float(self(x, alpha, m, t)) + np.sum((alpha - z_arr) ** 2) / (2 * tau)

        x0 = np.clip(z_arr, bounds[0], bounds[1])
        res = scipy_minimize(objective_nd, x0, bounds=[bounds] * d, method="L-BFGS-B")
        self._reject_fallback_truncation(res.x, used_fallback, "proximal")
        return res.x

    # === Semi-Lagrangian interface ===

    def control_bounds(self) -> tuple[float, float] | None:
        """Bounds on admissible control set A.

        Returns (a_min, a_max) or None if A = R^d (unbounded).
        Used by semi-Lagrangian solver and numerical optimization.
        """
        return None

    # === Legacy: Legendre transform ===

    def legendre_transform(
        self,
        alpha_bounds: tuple[float, float] | None = None,
        n_search: int = 100,
    ) -> HamiltonianBase:
        """Convert Lagrangian to Hamiltonian via numerical Legendre transform.

        .. note::
            For common control costs, prefer using ``evaluate_hamiltonian()``
            or ``optimal_control()`` directly instead of constructing a
            DualHamiltonian object.
        """
        return DualHamiltonian(
            lagrangian=self,
            sense=self.sense,
            alpha_bounds=alpha_bounds or (-10.0, 10.0),
            n_search=n_search,
        )


class SeparableLagrangian(LagrangianBase):
    """Separable Lagrangian: L(x, alpha, m, t) = L_control(alpha) + V(x, t) + f(m).

    Mirrors SeparableHamiltonian. Uses ControlCostBase for the control term,
    providing closed-form optimal_control, evaluate_hamiltonian, and proximal.

    Parameters
    ----------
    control_cost : ControlCostBase
        Control cost specification (Quadratic, L1, Bounded, etc.)
    potential : callable or None
        V(x, t). If None, V = 0.
    coupling : callable or None
        f(m). If None, f = 0.
    sense : OptimizationSense
        Optimization direction.
    """

    def __init__(
        self,
        control_cost: ControlCostBase,
        potential: callable | None = None,
        coupling: callable | None = None,
        sense: OptimizationSense = OptimizationSense.MINIMIZE,
    ):
        super().__init__(sense=sense)
        self.control_cost = control_cost
        self._potential = potential
        self._coupling = coupling

    def __call__(self, x, alpha, m, t=0.0):
        """L = L_control(alpha) + V(x, t) + f(m)."""
        L_ctrl = self.control_cost.lagrangian(np.atleast_1d(alpha))
        V = float(self._potential(x, t)) if self._potential is not None else 0.0
        f_m = float(self._coupling(m)) if self._coupling is not None else 0.0
        return L_ctrl + V + f_m

    def optimal_control(self, x, m, p, t=0.0):
        """Delegates to control_cost.optimal_control(p). Analytic."""
        return self.control_cost.optimal_control(np.atleast_1d(p))

    def evaluate_hamiltonian(self, x, m, p, t=0.0):
        """H = H_control(p) + V(x,t) + f(m). Uses control_cost.evaluate()."""
        p_arr = np.atleast_1d(p)
        H_ctrl = self.control_cost.evaluate(p_arr)
        if not isinstance(H_ctrl, (int, float)):
            H_ctrl = float(H_ctrl.sum()) if p_arr.ndim < 2 else H_ctrl
        V = float(self._potential(x, t)) if self._potential is not None else 0.0
        f_m = float(self._coupling(m)) if self._coupling is not None else 0.0
        return H_ctrl + V + f_m

    def proximal(self, tau, z, x=None, m=None, t=0.0):
        """Delegates to control_cost.proximal(). V and f don't depend on alpha."""
        return self.control_cost.proximal(tau, np.atleast_1d(z))

    def control_bounds(self):
        """Admissible control set, owned by the control cost (Issue #1642, B3).

        Delegates to ``ControlCostBase.effective_domain()``. This replaced an
        isinstance ladder that re-derived the same answer -- a second owner of
        the admissible set. Do not re-introduce one here.
        """
        return self.control_cost.effective_domain()

    def as_hamiltonian(self) -> SeparableHamiltonian:
        """Return the corresponding SeparableHamiltonian (shared control_cost)."""
        return SeparableHamiltonian(
            control_cost=self.control_cost,
            potential=self._potential,
            coupling=self._coupling,
            sense=self.sense,
        )


class DualHamiltonian(HamiltonianBase):
    """
    Hamiltonian defined via Legendre transform of a Lagrangian.

    This class computes H(x, p, m, t) = sup_α { p·α - L(x, α, m, t) }
    numerically for general Lagrangians. It is the "dual" of a given
    Lagrangian in the sense of Legendre/convex duality.

    For separable quadratic Lagrangians, use SeparableHamiltonian instead
    which has analytic formulas.

    Parameters
    ----------
    lagrangian : LagrangianBase
        The Lagrangian to transform
    sense : OptimizationSense
        Optimization direction
    alpha_bounds : tuple[float, float]
        Bounds on control for optimization
    n_search : int
        Number of points for initial grid search

    Notes
    -----
    The numerical Legendre transform uses a two-stage approach:
    1. Grid search to find approximate optimum
    2. Local refinement using scipy.optimize (if available)

    See Also
    --------
    DualLagrangian : Lagrangian created from Hamiltonian via Legendre transform
    LagrangianBase.legendre_transform : Symmetric operation (L → H)
    """

    def __init__(
        self,
        lagrangian: LagrangianBase,
        sense: OptimizationSense = OptimizationSense.MINIMIZE,
        alpha_bounds: tuple[float, float] = (-10.0, 10.0),
        n_search: int = 100,
    ):
        super().__init__(sense=sense)
        self.lagrangian = lagrangian
        self.alpha_bounds = alpha_bounds
        self.n_search = n_search

    def __call__(
        self,
        x: NDArray,
        m: float | NDArray,
        p: NDArray,
        t: float = 0.0,
    ) -> float | NDArray:
        """
        Compute H via numerical Legendre transform.

        H(x, p, m, t) = sup_α { p·α - L(x, α, m, t) }
        """
        d = p.shape[0] if p.ndim > 0 else 1
        p_flat = np.atleast_1d(p)

        # For 1D, use simple grid search + refinement
        if d == 1:
            alpha_grid = np.linspace(self.alpha_bounds[0], self.alpha_bounds[1], self.n_search)
            values = np.array(
                [float(p_flat[0]) * a - float(self.lagrangian(x, np.array([a]), m, t)) for a in alpha_grid]
            )

            # H is the convex conjugate L*(p) = sup_alpha { p.alpha - L(alpha) },
            # independent of OptimizationSense (matches the d>1 branch below, the
            # always-sup ControlCostBase.evaluate, and dp's argmax). Issue #1185.
            return float(np.max(values))

        # For higher dimensions, use scipy if available
        try:
            from scipy.optimize import minimize as scipy_minimize

            def neg_objective(alpha):
                # Minimize negative of (p·α - L)
                return -(np.dot(p_flat, alpha) - float(self.lagrangian(x, alpha, m, t)))

            # Initial guess: project p onto bounds
            x0 = np.clip(p_flat, self.alpha_bounds[0], self.alpha_bounds[1])
            bounds = [self.alpha_bounds] * d

            result = scipy_minimize(neg_objective, x0, bounds=bounds, method="L-BFGS-B")
            return float(-result.fun)

        except ImportError:
            # Fallback: grid search in each dimension
            from itertools import product

            alpha_1d = np.linspace(self.alpha_bounds[0], self.alpha_bounds[1], 20)
            best_val = -np.inf

            for alpha_tuple in product(alpha_1d, repeat=d):
                alpha = np.array(alpha_tuple)
                val = np.dot(p_flat, alpha) - float(self.lagrangian(x, alpha, m, t))
                best_val = max(best_val, val)

            return float(best_val)

    def dp(
        self,
        x: NDArray,
        m: float | NDArray,
        p: NDArray,
        t: float = 0.0,
    ) -> NDArray:
        """
        Compute ∂H/∂p = α* (optimal control).

        By envelope theorem, ∂H/∂p equals the optimal control α*.
        """
        return self._find_optimal_alpha(x, m, p, t)

    def _find_optimal_alpha(
        self,
        x: NDArray,
        m: float | NDArray,
        p: NDArray,
        t: float,
    ) -> NDArray:
        """Find α* = argmax_α { p·α - L(x, α, m, t) }."""
        d = p.shape[0] if p.ndim > 0 else 1
        p_flat = np.atleast_1d(p)

        if d == 1:
            alpha_grid = np.linspace(self.alpha_bounds[0], self.alpha_bounds[1], self.n_search)
            values = np.array(
                [float(p_flat[0]) * a - float(self.lagrangian(x, np.array([a]), m, t)) for a in alpha_grid]
            )
            best_idx = np.argmax(values)
            return np.array([alpha_grid[best_idx]])

        # Higher dimensions: scipy or grid
        try:
            from scipy.optimize import minimize as scipy_minimize

            def neg_objective(alpha):
                return -(np.dot(p_flat, alpha) - float(self.lagrangian(x, alpha, m, t)))

            x0 = np.clip(p_flat, self.alpha_bounds[0], self.alpha_bounds[1])
            bounds = [self.alpha_bounds] * d
            result = scipy_minimize(neg_objective, x0, bounds=bounds, method="L-BFGS-B")
            return result.x

        except ImportError:
            from itertools import product

            alpha_1d = np.linspace(self.alpha_bounds[0], self.alpha_bounds[1], 20)
            best_val = -np.inf
            best_alpha = np.zeros(d)

            for alpha_tuple in product(alpha_1d, repeat=d):
                alpha = np.array(alpha_tuple)
                val = np.dot(p_flat, alpha) - float(self.lagrangian(x, alpha, m, t))
                if val > best_val:
                    best_val = val
                    best_alpha = alpha

            return best_alpha


class DualLagrangian(LagrangianBase):
    """
    Lagrangian defined via inverse Legendre transform of a Hamiltonian.

    This class computes L(x, α, m, t) = sup_p { p·α - H(x, p, m, t) }
    numerically for general Hamiltonians. It is the "dual" of a given
    Hamiltonian in the sense of Legendre/convex duality.

    For separable quadratic Hamiltonians, the inverse transform gives
    back the quadratic Lagrangian analytically.

    Parameters
    ----------
    hamiltonian : HamiltonianBase
        The Hamiltonian to inverse-transform
    sense : OptimizationSense
        Optimization direction
    p_bounds : tuple[float, float]
        Bounds on momentum for optimization
    n_search : int
        Number of points for initial grid search

    See Also
    --------
    DualHamiltonian : Hamiltonian created from Lagrangian via Legendre transform
    HamiltonianBase.legendre_transform : Symmetric operation (H → L)
    """

    def __init__(
        self,
        hamiltonian: HamiltonianBase,
        sense: OptimizationSense = OptimizationSense.MINIMIZE,
        p_bounds: tuple[float, float] = (-10.0, 10.0),
        n_search: int = 100,
    ):
        super().__init__(sense=sense)
        self.hamiltonian = hamiltonian
        self.p_bounds = p_bounds
        self.n_search = n_search

    def __call__(
        self,
        x: NDArray,
        alpha: NDArray,
        m: float | NDArray,
        t: float = 0.0,
    ) -> float | NDArray:
        """
        Compute L via inverse Legendre transform.

        L(x, α, m, t) = sup_p { p·α - H(x, p, m, t) }
        """
        d = alpha.shape[0] if alpha.ndim > 0 else 1
        alpha_flat = np.atleast_1d(alpha)

        # For 1D, use simple grid search
        if d == 1:
            p_grid = np.linspace(self.p_bounds[0], self.p_bounds[1], self.n_search)
            values = np.array(
                [float(p) * float(alpha_flat[0]) - float(self.hamiltonian(x, m, np.array([p]), t)) for p in p_grid]
            )

            # L is the convex conjugate H*(alpha) = sup_p { p.alpha - H(p) },
            # independent of OptimizationSense (matches the d>1 branch below and
            # d_alpha's argmax). Issue #1185.
            return float(np.max(values))

        # For higher dimensions, use scipy if available
        try:
            from scipy.optimize import minimize as scipy_minimize

            def neg_objective(p):
                return -(np.dot(p, alpha_flat) - float(self.hamiltonian(x, m, p, t)))

            x0 = np.clip(alpha_flat, self.p_bounds[0], self.p_bounds[1])
            bounds = [self.p_bounds] * d
            result = scipy_minimize(neg_objective, x0, bounds=bounds, method="L-BFGS-B")
            return float(-result.fun)

        except ImportError:
            from itertools import product

            p_1d = np.linspace(self.p_bounds[0], self.p_bounds[1], 20)
            best_val = -np.inf

            for p_tuple in product(p_1d, repeat=d):
                p = np.array(p_tuple)
                val = np.dot(p, alpha_flat) - float(self.hamiltonian(x, m, p, t))
                best_val = max(best_val, val)

            return float(best_val)

    def d_alpha(
        self,
        x: NDArray,
        alpha: NDArray,
        m: float | NDArray,
        t: float = 0.0,
    ) -> NDArray:
        """
        Compute ∂L/∂α = p* (optimal momentum).

        By envelope theorem, ∂L/∂α equals the optimal momentum p*.
        """
        return self._find_optimal_p(x, alpha, m, t)

    def dm(
        self,
        x: NDArray,
        alpha: NDArray,
        m: float | NDArray,
        t: float = 0.0,
    ) -> float:
        """Compute ∂L/∂m using finite differences."""
        eps = self.finite_diff_eps
        m_scalar = float(m) if np.isscalar(m) else float(np.mean(m))

        L_plus = self(x, alpha, m_scalar + eps, t)
        L_minus = self(x, alpha, m_scalar - eps, t)

        return float((L_plus - L_minus) / (2 * eps))

    def _find_optimal_p(
        self,
        x: NDArray,
        alpha: NDArray,
        m: float | NDArray,
        t: float,
    ) -> NDArray:
        """Find p* = argmax_p { p·α - H(x, p, m, t) }."""
        d = alpha.shape[0] if alpha.ndim > 0 else 1
        alpha_flat = np.atleast_1d(alpha)

        if d == 1:
            p_grid = np.linspace(self.p_bounds[0], self.p_bounds[1], self.n_search)
            values = np.array(
                [float(p) * float(alpha_flat[0]) - float(self.hamiltonian(x, m, np.array([p]), t)) for p in p_grid]
            )
            best_idx = np.argmax(values)
            return np.array([p_grid[best_idx]])

        # Higher dimensions: scipy or grid
        try:
            from scipy.optimize import minimize as scipy_minimize

            def neg_objective(p):
                return -(np.dot(p, alpha_flat) - float(self.hamiltonian(x, m, p, t)))

            x0 = np.clip(alpha_flat, self.p_bounds[0], self.p_bounds[1])
            bounds = [self.p_bounds] * d
            result = scipy_minimize(neg_objective, x0, bounds=bounds, method="L-BFGS-B")
            return result.x

        except ImportError:
            from itertools import product

            p_1d = np.linspace(self.p_bounds[0], self.p_bounds[1], 20)
            best_val = -np.inf
            best_p = np.zeros(d)

            for p_tuple in product(p_1d, repeat=d):
                p = np.array(p_tuple)
                val = np.dot(p, alpha_flat) - float(self.hamiltonian(x, m, p, t))
                if val > best_val:
                    best_val = val
                    best_p = p

            return best_p


# ============================================================================
# Concrete Implementations
# ============================================================================


class SeparableHamiltonian(HamiltonianBase):
    """
    Separable Hamiltonian: H(x, m, p, t) = H_control(p) + V(x, t) + f(m).

    This is the most common form in MFG, where:
    - H_control(p): Control cost (from ControlCostBase)
    - V(x, t): Potential energy / state cost (see sign-convention note below)
    - f(m): Density coupling term

    The separability allows efficient computation and analytic derivatives.

    Sign convention (Issue #1057, gotcha G-001)
    -------------------------------------------
    `potential` enters the Hamiltonian as ``H = H_control(p) + V(x, t) + f(m)``.
    The class also accepts ``sense=OptimizationSense.MINIMIZE`` (default) which
    flips the sign on ``H_control``'s Legendre-transform direction — but ``V`` is
    currently added to ``H`` **without** a corresponding sense-flip. Empirically,
    research code attracts density to ``x_c`` by writing ``V(x, t) = -C * (x - x_c)**2``
    (inverted parabola, peak at ``x_c``), not the bowl shape that standard MFG
    literature would suggest. This is the de-facto "potential as reward"
    convention.

    The API gap (V not interacting with ``sense``) is tracked as Issue #1060;
    until that lands, the practical guidance is:

    - **Attractive** potential at ``x_c``: ``V(x, t) = -0.5 * C * (x - x_c)**2``
      (inverted parabola, peak at ``x_c``).
    - **Repulsive** potential at ``x_c``: ``V(x, t) = +0.5 * C * (x - x_c)**2``
      (bowl, minimum at ``x_c``).

    Verified by exp08/09 Stage A/B/C runners — they attract density to ``x_c``
    using ``-C₁ * (x - x_c)**2``.

    Related: the dual ``MFGProblem(source_term_hjb=...)`` channel carries the
    OPPOSITE sign — it enters the HJB right-hand side and is cost-signed (a
    positive source repels). See Issue #1057 gotchas G-002/G-003 and the
    ``MFGProblem`` "Sign conventions" docstring note.

    Parameters
    ----------
    control_cost : ControlCostBase
        Control cost specification (quadratic, L1, bounded, etc.)
    potential : Callable[[NDArray, float], float] | None
        Potential V(x, t) added to H. If None, V = 0. See "Sign convention" above —
        write inverted parabola (peak at x_c) for attractive.
    coupling : Callable[[float | NDArray], float | NDArray] | None
        Density coupling f(m). If None, f = 0. Same "added to H" convention.
    coupling_dm : Callable[[float | NDArray], float | NDArray] | None
        Derivative df/dm. If None, computed via finite differences.

    Examples
    --------
    Standard MFG with quadratic control, no potential, m² coupling:

    >>> H = SeparableHamiltonian(
    ...     control_cost=QuadraticControlCost(control_cost=1.0),
    ...     coupling=lambda m: -m**2,
    ...     coupling_dm=lambda m: -2*m,  # Analytic derivative
    ... )
    >>> H(x=np.array([0.5]), m=0.3, p=np.array([1.0]), t=0.0)

    With potential field:

    >>> def potential(x, t):
    ...     return np.sin(2 * np.pi * x[0])  # Periodic potential
    >>>
    >>> H = SeparableHamiltonian(
    ...     control_cost=QuadraticControlCost(),
    ...     potential=potential,
    ... )
    """

    def __init__(
        self,
        control_cost: ControlCostBase,
        potential: callable | None = None,
        coupling: callable | None = None,
        coupling_dm: callable | None = None,
        sense: OptimizationSense = OptimizationSense.MINIMIZE,
        population_index: int = 0,
    ):
        super().__init__(sense=sense, population_index=population_index)
        self.control_cost = control_cost
        self._potential = potential
        self._coupling = coupling
        self._coupling_dm = coupling_dm
        # Issue #929: Cache vectorized-callable detection result
        self._potential_is_vectorized: bool | None = None

    def _evaluate_potential_batch(self, x_batch: NDArray, t: float) -> NDArray:
        """Evaluate V(x, t) at N points with auto-detected vectorization.

        Issue #929: First call probes whether the potential callable supports
        batch input (N, d) -> (N,). If yes, uses fast path. If no, falls back
        to per-point loop. Detection result is cached for subsequent calls.
        """
        N = x_batch.shape[0]

        if self._potential_is_vectorized is None:
            # Probe with small batch (at least 2 points needed for shape check)
            probe_size = min(2, N)
            if probe_size < 2:
                # Single-point batch — can't distinguish scalar from vectorized
                self._potential_is_vectorized = False
            else:
                try:
                    probe = np.asarray(self._potential(x_batch[:2], t), dtype=float)
                    self._potential_is_vectorized = probe.shape == (2,)
                except (TypeError, IndexError, ValueError):
                    self._potential_is_vectorized = False

        if self._potential_is_vectorized:
            return np.asarray(self._potential(x_batch, t), dtype=float)
        return np.array([float(self._potential(x_batch[i], t)) for i in range(N)])

    def __call__(
        self,
        x: NDArray,
        m: float | NDArray,
        p: NDArray,
        t: float = 0.0,
    ) -> float | NDArray:
        """
        Evaluate H = H_control(p) + V(x, t) + f(m).

        Supports both single-point and batch inputs (Issue #775).
        For batch: p.shape = (N, d), returns shape (N,).
        For single-point: p.shape = (d,), returns float.
        """
        p_arr = np.atleast_1d(p)
        is_batch = p_arr.ndim == 2

        # Control cost term: evaluate() always returns finite values (Issue #898)
        H_control = self.control_cost.evaluate(p_arr)
        if not is_batch and isinstance(H_control, np.ndarray):
            H_control = float(H_control.sum())

        # Potential term (Issue #929: auto-detect vectorized callable)
        if self._potential is not None:
            if is_batch:
                V = self._evaluate_potential_batch(x, t)
            else:
                V = float(self._potential(x, t))
        else:
            V = np.zeros(p_arr.shape[0]) if is_batch else 0.0

        # Coupling term
        if self._coupling is not None:
            if is_batch:
                m_arr = np.asarray(m)
                try:
                    f_m = np.asarray(self._coupling(m_arr), dtype=float).ravel()
                    if f_m.shape[0] != p_arr.shape[0]:
                        raise ValueError
                except (TypeError, ValueError):
                    f_m = np.array([float(self._coupling(float(m_arr.flat[i]))) for i in range(p_arr.shape[0])])
            else:
                f_m = float(self._coupling(m))
        else:
            f_m = np.zeros(p_arr.shape[0]) if is_batch else 0.0

        return H_control + V + f_m

    def dp(
        self,
        x: NDArray,
        m: float | NDArray,
        p: NDArray,
        t: float = 0.0,
    ) -> NDArray:
        """
        dH/dp from control cost component. No isinstance dispatch (Issue #898).

        For separable H, dH/dp depends only on p (not x, m, t).
        Delegates to control_cost.dp(p) which each subclass implements.
        """
        return self.control_cost.dp(np.atleast_1d(p))

    def dm(
        self,
        x: NDArray,
        m: float | NDArray,
        p: NDArray,
        t: float = 0.0,
    ) -> float | NDArray:
        """
        Compute ∂H/∂m = df/dm (only coupling term depends on m).

        Supports both single-point and batch inputs (Issue #775).
        For batch: p.shape = (N, d), returns shape (N,).
        For single-point: p.shape = (d,), returns float.
        """
        p_arr = np.asarray(p)
        is_batch = p_arr.ndim == 2

        if self._coupling_dm is not None:
            if is_batch:
                m_arr = np.asarray(m)
                try:
                    result = np.asarray(self._coupling_dm(m_arr), dtype=float).ravel()
                    if result.shape[0] != p_arr.shape[0]:
                        raise ValueError
                except (TypeError, ValueError):
                    result = np.array([float(self._coupling_dm(float(m_arr.flat[i]))) for i in range(p_arr.shape[0])])
                return result
            return float(self._coupling_dm(m))

        if self._coupling is None:
            if is_batch:
                return np.zeros(p_arr.shape[0])
            return 0.0

        # Finite difference fallback (with batch dispatch)
        if is_batch:
            m_arr = np.asarray(m)
            return np.array(
                [self._finite_diff_dm(x[i], float(m_arr.flat[i]), p_arr[i], t) for i in range(p_arr.shape[0])]
            )
        return self._finite_diff_dm(x, m, p, t)

    def dx(
        self,
        x: NDArray,
        m: float | NDArray,
        p: NDArray,
        t: float = 0.0,
    ) -> NDArray:
        """dH/dx = grad_V(x, t) for separable H.

        Only the potential V(x,t) depends on x. The control cost H_kin(p)
        and coupling f(m) are independent of x.

        Uses FD on the potential callable. Returns zero if no potential.
        """
        if self._potential is None:
            x_arr = np.atleast_1d(x)
            return np.zeros(x_arr.shape[-1] if x_arr.ndim >= 2 else len(x_arr))

        # FD on V(x, t)
        eps = self.finite_diff_eps
        x_flat = np.atleast_1d(x).astype(float)
        d = len(x_flat)
        grad = np.zeros(d)
        for i in range(d):
            x_plus = x_flat.copy()
            x_minus = x_flat.copy()
            x_plus[i] += eps
            x_minus[i] -= eps
            grad[i] = (float(self._potential(x_plus, t)) - float(self._potential(x_minus, t))) / (2 * eps)
        return grad

    def optimal_control(
        self,
        x: NDArray,
        m: float | NDArray,
        p: NDArray,
        t: float = 0.0,
    ) -> NDArray:
        """
        Optimal control from the control cost specification.

        For separable Hamiltonians, optimal control depends only on p,
        not on x, m, or t.
        """
        return self.control_cost.optimal_control(np.atleast_1d(p))

    def is_smooth(self) -> bool:
        """Delegates to control cost component."""
        return self.control_cost.is_smooth()

    def regularize(self, epsilon: float, method: str = "moreau-yosida") -> SeparableHamiltonian:
        """Smooth the control cost, preserving separable structure."""
        if self.is_smooth():
            return self
        return SeparableHamiltonian(
            control_cost=self.control_cost.regularize(epsilon, method),
            potential=self._potential,
            coupling=self._coupling,
            coupling_dm=self._coupling_dm,
            sense=self.sense,
        )


class CongestionHamiltonian(HamiltonianBase):
    """
    Non-separable Hamiltonian with density-dependent kinetic cost (Issue #782).

    H(x, m, p, t) = |p|^2 / (2*lambda*c(m)) + V(x, t) + f(m)

    The congestion factor c(m) modifies the kinetic term, making movement
    costlier in high-density regions (multiplicative congestion). Unlike
    SeparableHamiltonian where the kinetic term |p|^2/(2*lambda) is independent
    of density, here c(m) couples density into the velocity cost.

    Parameters
    ----------
    control_cost : ControlCostBase
        Control cost specification (provides lambda for kinetic term)
    congestion_factor : callable
        c(m) -> float or ndarray. Must be positive for all valid densities.
        Example: ``lambda m: 1 + gamma * domain_volume * m``
    congestion_factor_dm : callable or None
        c'(m) -> float or ndarray. Derivative of congestion factor.
        If None, finite differences are used for dm().
    potential : callable or None
        V(x, t) -> float. Spatial potential term.
    coupling : callable or None
        f(m) -> float or ndarray. Additive density coupling term.
    coupling_dm : callable or None
        f'(m) -> float or ndarray. Derivative of coupling.
    sense : OptimizationSense
        MINIMIZE (default) or MAXIMIZE.

    Examples
    --------
    Standard multiplicative congestion (velocity reduction by crowd density):

    >>> gamma, domain_volume = 1.0, 1.0
    >>> H = CongestionHamiltonian(
    ...     control_cost=QuadraticControlCost(control_cost=1.0),
    ...     congestion_factor=lambda m: 1 + gamma * domain_volume * m,
    ...     congestion_factor_dm=lambda m: gamma * domain_volume,
    ... )
    """

    def __init__(
        self,
        control_cost: ControlCostBase,
        congestion_factor: callable,
        congestion_factor_dm: callable | None = None,
        potential: callable | None = None,
        coupling: callable | None = None,
        coupling_dm: callable | None = None,
        sense: OptimizationSense = OptimizationSense.MINIMIZE,
        population_index: int = 0,
    ):
        super().__init__(sense=sense, population_index=population_index)
        self.control_cost = control_cost
        self._congestion_factor = congestion_factor
        self._congestion_factor_dm = congestion_factor_dm
        self._potential = potential
        self._coupling = coupling
        self._coupling_dm = coupling_dm

    def __call__(
        self,
        x: NDArray,
        m: float | NDArray,
        p: NDArray,
        t: float = 0.0,
    ) -> float | NDArray:
        """
        Evaluate H = |p|^2 / (2*lambda*c(m)) + V(x, t) + f(m).

        Supports both single-point and batch inputs (Issue #775).
        For batch: p.shape = (N, d), returns shape (N,).
        For single-point: p.shape = (d,), returns float.
        """
        p_arr = np.atleast_1d(p)
        is_batch = p_arr.ndim == 2

        # Kinetic term: H_control(p) / c(m).  evaluate() always finite (Issue #898)
        H_kinetic = self.control_cost.evaluate(p_arr)
        if not is_batch and isinstance(H_kinetic, np.ndarray):
            H_kinetic = float(H_kinetic.sum())

        # Congestion factor c(m)
        if is_batch:
            m_arr = np.asarray(m)
            try:
                c_m = np.asarray(self._congestion_factor(m_arr), dtype=float).ravel()
                if c_m.shape[0] != p_arr.shape[0]:
                    raise ValueError
            except (TypeError, ValueError):
                c_m = np.array([float(self._congestion_factor(float(m_arr.flat[i]))) for i in range(p_arr.shape[0])])
        else:
            c_m = float(self._congestion_factor(m))

        H_kinetic = H_kinetic / c_m

        # Potential term V(x, t)
        if self._potential is not None:
            if is_batch:
                V = np.array([float(self._potential(x[i], t)) for i in range(x.shape[0])])
            else:
                V = float(self._potential(x, t))
        else:
            V = np.zeros(p_arr.shape[0]) if is_batch else 0.0

        # Coupling term f(m)
        if self._coupling is not None:
            if is_batch:
                m_arr = np.asarray(m)
                try:
                    f_m = np.asarray(self._coupling(m_arr), dtype=float).ravel()
                    if f_m.shape[0] != p_arr.shape[0]:
                        raise ValueError
                except (TypeError, ValueError):
                    f_m = np.array([float(self._coupling(float(m_arr.flat[i]))) for i in range(p_arr.shape[0])])
            else:
                f_m = float(self._coupling(m))
        else:
            f_m = np.zeros(p_arr.shape[0]) if is_batch else 0.0

        return H_kinetic + V + f_m

    def dp(
        self,
        x: NDArray,
        m: float | NDArray,
        p: NDArray,
        t: float = 0.0,
    ) -> NDArray:
        """
        Compute dH/dp = p / (lambda * c(m)).

        For quadratic control cost H_kinetic = |p|^2/(2*lambda*c(m)),
        the derivative is p/(lambda*c(m)).

        Returns shape (d,) for single-point, (N, d) for batch.
        """
        p_arr = np.atleast_1d(p)
        is_batch = p_arr.ndim == 2

        # Congestion factor c(m)
        if is_batch:
            m_arr = np.asarray(m)
            try:
                c_m = np.asarray(self._congestion_factor(m_arr), dtype=float).ravel()
                if c_m.shape[0] != p_arr.shape[0]:
                    raise ValueError
            except (TypeError, ValueError):
                c_m = np.array([float(self._congestion_factor(float(m_arr.flat[i]))) for i in range(p_arr.shape[0])])
        else:
            c_m = float(self._congestion_factor(m))

        # Delegate to control_cost.dp(), then divide by c(m) (Issue #898)
        base_dp = self.control_cost.dp(p_arr)
        if is_batch:
            return base_dp / c_m[:, np.newaxis]
        return base_dp / c_m

    def dm(
        self,
        x: NDArray,
        m: float | NDArray,
        p: NDArray,
        t: float = 0.0,
    ) -> float | NDArray:
        """
        Compute dH/dm = -c'(m) * H_control(p) / c(m)^2 + f'(m).

        Uses control_cost.evaluate(p) instead of hardcoded |p|^2/(2*lambda).
        No isinstance dispatch (Issue #898).
        """
        p_arr = np.asarray(p)
        is_batch = p_arr.ndim == 2

        # Kinetic contribution to dm (requires congestion_factor_dm)
        if self._congestion_factor_dm is not None:
            # H_control(p) via evaluate() -- works for any ControlCostBase
            H_kin = self.control_cost.evaluate(p_arr)
            if not is_batch and isinstance(H_kin, np.ndarray):
                H_kin = float(H_kin.sum())

            if is_batch:
                m_arr = np.asarray(m)
                try:
                    c_m = np.asarray(self._congestion_factor(m_arr), dtype=float).ravel()
                    c_dm = np.asarray(self._congestion_factor_dm(m_arr), dtype=float).ravel()
                    if c_m.shape[0] != p_arr.shape[0] or c_dm.shape[0] != p_arr.shape[0]:
                        raise ValueError
                except (TypeError, ValueError):
                    c_m = np.array(
                        [float(self._congestion_factor(float(m_arr.flat[i]))) for i in range(p_arr.shape[0])]
                    )
                    c_dm = np.array(
                        [float(self._congestion_factor_dm(float(m_arr.flat[i]))) for i in range(p_arr.shape[0])]
                    )

                kinetic_dm = -c_dm * H_kin / c_m**2
            else:
                c_m = float(self._congestion_factor(m))
                c_dm = float(self._congestion_factor_dm(m))
                kinetic_dm = -c_dm * H_kin / c_m**2

            # Coupling contribution f'(m)
            if self._coupling_dm is not None:
                if is_batch:
                    try:
                        coupling_dm_val = np.asarray(self._coupling_dm(m_arr), dtype=float).ravel()
                        if coupling_dm_val.shape[0] != p_arr.shape[0]:
                            raise ValueError
                    except (TypeError, ValueError):
                        coupling_dm_val = np.array(
                            [float(self._coupling_dm(float(m_arr.flat[i]))) for i in range(p_arr.shape[0])]
                        )
                else:
                    coupling_dm_val = float(self._coupling_dm(m))
            else:
                coupling_dm_val = np.zeros(p_arr.shape[0]) if is_batch else 0.0

            result = kinetic_dm + coupling_dm_val
            if not is_batch:
                return float(result)
            return result

        # Fallback to finite differences
        if is_batch:
            m_arr = np.asarray(m)
            return np.array(
                [self._finite_diff_dm(x[i], float(m_arr.flat[i]), p_arr[i], t) for i in range(p_arr.shape[0])]
            )
        return self._finite_diff_dm(x, m, p, t)

    def optimal_control(
        self,
        x: NDArray,
        m: float | NDArray,
        p: NDArray,
        t: float = 0.0,
    ) -> NDArray:
        """
        Optimal control: alpha* = -sign * dH/dp.

        For congestion Hamiltonian, the optimal control depends on density m
        (unlike separable case where it only depends on p).
        """
        return -self._sign * self.dp(x, m, p, t)

    def is_smooth(self) -> bool:
        """Delegates to control cost component."""
        return self.control_cost.is_smooth()

    def regularize(self, epsilon: float, method: str = "moreau-yosida") -> CongestionHamiltonian:
        """Smooth the control cost, preserving congestion structure."""
        if self.is_smooth():
            return self
        return CongestionHamiltonian(
            control_cost=self.control_cost.regularize(epsilon, method),
            congestion_factor=self._congestion_factor,
            congestion_factor_dm=self._congestion_factor_dm,
            potential=self._potential,
            coupling=self._coupling,
            coupling_dm=self._coupling_dm,
            sense=self.sense,
        )


class QuadraticMFGHamiltonian(SeparableHamiltonian):
    """
    Standard quadratic MFG Hamiltonian: H = ½c|p|² - V(x) - m².

    This is the Hamiltonian used when no custom hamiltonian_func is provided
    in MFGComponents. It represents the most common form in MFG literature:

    - Quadratic control cost: H_control = ½c|p|²
    - Optional potential: V(x, t)
    - Quadratic density coupling: f(m) = -m²

    Parameters
    ----------
    coupling_coefficient : float
        Coefficient c in ½c|p|² (default: 1.0)
    potential : Callable | None
        Potential V(x, t) (default: None, meaning V=0)
    sense : OptimizationSense
        Optimization direction

    Notes
    -----
    The default coupling f(m) = -m² gives ∂H/∂m = -2m.
    This Hamiltonian leads to the classical optimal control:
    α* = -c·p (for MINIMIZE sense).
    """

    def __init__(
        self,
        coupling_coefficient: float = 1.0,
        potential: callable | None = None,
        sense: OptimizationSense = OptimizationSense.MINIMIZE,
    ):
        super().__init__(
            control_cost=QuadraticControlCost(
                sense=sense,
                control_cost=1.0 / coupling_coefficient if coupling_coefficient > 0 else 1.0,
            ),
            potential=potential,
            coupling=lambda m: -(m**2),
            coupling_dm=lambda m: -2 * m,
            sense=sense,
        )
        self.coupling_coefficient = coupling_coefficient


# ============================================================================
# Factory and Utilities
# ============================================================================


def create_hamiltonian(
    hamiltonian_type: str = "quadratic",
    **kwargs,
) -> HamiltonianBase:
    """
    Factory function to create Hamiltonians by type name.

    Parameters
    ----------
    hamiltonian_type : str
        One of: "quadratic", "l1", "bounded", "default", "separable"
    **kwargs
        Type-specific parameters

    Returns
    -------
    HamiltonianBase
        The created Hamiltonian

    Examples
    --------
    >>> H = create_hamiltonian("quadratic", control_cost=2.0)
    >>> H = create_hamiltonian("default", coupling_coefficient=0.5)
    """
    sense = kwargs.pop("sense", OptimizationSense.MINIMIZE)

    if hamiltonian_type == "quadratic":
        control_cost = kwargs.get("control_cost", 1.0)
        return SeparableHamiltonian(
            control_cost=QuadraticControlCost(sense=sense, control_cost=control_cost),
            sense=sense,
        )

    elif hamiltonian_type == "l1":
        control_cost = kwargs.get("control_cost", 1.0)
        return SeparableHamiltonian(
            control_cost=L1ControlCost(sense=sense, control_cost=control_cost),
            sense=sense,
        )

    elif hamiltonian_type == "bounded":
        control_cost = kwargs.get("control_cost", 1.0)
        max_control = kwargs.get("max_control", 1.0)
        return SeparableHamiltonian(
            control_cost=BoundedControlCost(sense=sense, control_cost=control_cost, max_control=max_control),
            sense=sense,
        )

    elif hamiltonian_type == "default":
        coupling_coefficient = kwargs.get("coupling_coefficient", 1.0)
        potential = kwargs.get("potential")
        return QuadraticMFGHamiltonian(
            coupling_coefficient=coupling_coefficient,
            potential=potential,
            sense=sense,
        )

    elif hamiltonian_type == "separable":
        control_cost = kwargs.get(
            "control_cost",
            QuadraticControlCost(sense=sense),
        )
        potential = kwargs.get("potential")
        coupling = kwargs.get("coupling")
        coupling_dm = kwargs.get("coupling_dm")
        return SeparableHamiltonian(
            control_cost=control_cost,
            potential=potential,
            coupling=coupling,
            coupling_dm=coupling_dm,
            sense=sense,
        )

    else:
        raise ValueError(
            f"Unknown hamiltonian_type: {hamiltonian_type}. Valid types: quadratic, l1, bounded, default, separable"
        )


if __name__ == "__main__":
    """Quick smoke test for development."""
    print("Testing Hamiltonian abstractions...")
    print("=" * 60)

    # Test QuadraticControlCost
    print("\n1. QuadraticControlCost (MINIMIZE):")
    cost = QuadraticControlCost(sense=OptimizationSense.MINIMIZE, control_cost=2.0)
    p = np.array([1.0, 2.0, -3.0])
    alpha = cost.optimal_control(p)
    print(f"   p = {p}")
    print(f"   α* = {alpha}  (expected: [-0.5, -1.0, 1.5])")
    assert np.allclose(alpha, [-0.5, -1.0, 1.5]), "QuadraticControlCost MINIMIZE failed"

    print("\n2. QuadraticControlCost (MAXIMIZE):")
    cost_max = QuadraticControlCost(sense=OptimizationSense.MAXIMIZE, control_cost=2.0)
    alpha_max = cost_max.optimal_control(p)
    print(f"   α* = {alpha_max}  (expected: [0.5, 1.0, -1.5])")
    assert np.allclose(alpha_max, [0.5, 1.0, -1.5]), "QuadraticControlCost MAXIMIZE failed"

    # Test L1ControlCost
    print("\n3. L1ControlCost (bang-bang):")
    cost_l1 = L1ControlCost(control_cost=1.5)
    p_l1 = np.array([0.5, 2.0, -3.0])
    alpha_l1 = cost_l1.optimal_control(p_l1)
    print(f"   p = {p_l1}, threshold = 1.5")
    print(f"   α* = {alpha_l1}  (expected: [0, -1, 1])")
    assert np.allclose(alpha_l1, [0, -1, 1]), "L1ControlCost failed"

    # Test BoundedControlCost
    print("\n4. BoundedControlCost:")
    cost_bounded = BoundedControlCost(control_cost=1.0, max_control=1.5)
    p_bounded = np.array([1.0, 2.0, 3.0])
    alpha_bounded = cost_bounded.optimal_control(p_bounded)
    print(f"   p = {p_bounded}, max_control = 1.5")
    print(f"   α* = {alpha_bounded}  (expected: [-1, -1.5, -1.5])")
    assert np.allclose(alpha_bounded, [-1, -1.5, -1.5]), "BoundedControlCost failed"

    print("\n" + "=" * 60)
    print("Testing Hamiltonian classes (Issue #673)...")
    print("=" * 60)

    # Test SeparableHamiltonian
    print("\n5. SeparableHamiltonian (quadratic control):")
    H = SeparableHamiltonian(
        control_cost=QuadraticControlCost(control_cost=2.0),
        coupling=lambda m: -(m**2),
        coupling_dm=lambda m: -2 * m,
    )
    x = np.array([0.5])
    m_val = 0.3
    p_val = np.array([1.0])
    t_val = 0.0

    H_val = H(x, m_val, p_val, t_val)
    # H = ½|p|²/λ + f(m) = 0.5 * 1.0 / 2.0 + (-0.09) = 0.25 - 0.09 = 0.16
    print(f"   H(x={x}, m={m_val}, p={p_val}) = {H_val:.4f}")
    print("   Expected: 0.5 * 1.0² / 2.0 - 0.3² = 0.25 - 0.09 = 0.16")
    assert abs(H_val - 0.16) < 1e-10, f"SeparableHamiltonian value failed: {H_val}"

    # Test dp (analytic)
    dp_val = H.dp(x, m_val, p_val, t_val)
    print(f"   ∂H/∂p = {dp_val}  (expected: p/λ = [0.5])")
    assert np.allclose(dp_val, [0.5]), f"SeparableHamiltonian dp failed: {dp_val}"

    # Test dm (analytic)
    dm_val = H.dm(x, m_val, p_val, t_val)
    print(f"   ∂H/∂m = {dm_val:.4f}  (expected: -2m = -0.6)")
    assert abs(dm_val - (-0.6)) < 1e-10, f"SeparableHamiltonian dm failed: {dm_val}"

    # Test optimal control
    alpha_opt = H.optimal_control(x, m_val, p_val, t_val)
    print(f"   α* = {alpha_opt}  (expected: -p/λ = [-0.5])")
    assert np.allclose(alpha_opt, [-0.5]), "SeparableHamiltonian optimal_control failed"

    # Test QuadraticMFGHamiltonian (and backward-compat alias DefaultMFGHamiltonian)
    print("\n6. QuadraticMFGHamiltonian:")
    H_default = QuadraticMFGHamiltonian(coupling_coefficient=1.0)
    H_default_val = H_default(x, m_val, p_val, t_val)
    # H = ½c|p|² - m² = 0.5 * 1.0 * 1.0 - 0.09 = 0.5 - 0.09 = 0.41
    print(f"   H(x={x}, m={m_val}, p={p_val}) = {H_default_val:.4f}")
    print("   Expected: 0.5 * 1.0 * 1.0² - 0.3² = 0.5 - 0.09 = 0.41")
    assert abs(H_default_val - 0.41) < 1e-10, f"QuadraticMFGHamiltonian failed: {H_default_val}"

    # Test factory function
    print("\n7. create_hamiltonian factory:")
    H_factory = create_hamiltonian("quadratic", control_cost=2.0)
    H_factory_val = H_factory(x, m_val, p_val, t_val)
    print("   create_hamiltonian('quadratic', control_cost=2.0)")
    print(f"   H = {H_factory_val:.4f}  (expected: 0.25)")
    # H = ½|p|²/λ = 0.5 * 1.0 / 2.0 = 0.25
    assert abs(H_factory_val - 0.25) < 1e-10, "Factory Hamiltonian failed"

    # Issue #673: to_legacy_func() removed - test class-based API directly
    print("\n8. Class-based Hamiltonian direct calls:")
    H_class = SeparableHamiltonian(
        control_cost=QuadraticControlCost(control_cost=1.0),
        coupling=lambda m: -(m**2),
        coupling_dm=lambda m: -2 * m,
    )

    # Call class-based API directly: H(x, m, p, t)
    p_test = np.array([2.0])  # p = 2.0 in 1D
    m_test = 0.3
    H_direct = H_class(x, m_test, p_test, t_val)
    # H = ½|p|²/λ - m² = 0.5 * 4.0 / 1.0 - 0.09 = 2.0 - 0.09 = 1.91
    print(f"   H(x, m=0.3, p=2.0, t) = {H_direct:.4f}")
    print("   Expected: 0.5 * 2² - 0.3² = 2.0 - 0.09 = 1.91")
    assert abs(H_direct - 1.91) < 1e-10, f"Class-based H failed: {H_direct}"

    dm_direct = H_class.dm(x, m_test, p_test, t_val)
    print(f"   H.dm(x, m=0.3, p, t) = {dm_direct:.4f}  (expected: -0.6)")
    assert abs(dm_direct - (-0.6)) < 1e-10, "Class-based dm failed"

    print("\n" + "=" * 60)
    print("Testing Lagrangian and Legendre transform (Issue #651)...")
    print("=" * 60)

    # Create a simple quadratic Lagrangian
    class TestQuadraticLagrangian(LagrangianBase):
        def __init__(self, lam=1.0):
            super().__init__()
            self.lam = lam

        def __call__(self, x, alpha, m, t=0.0):
            return 0.5 * self.lam * np.sum(alpha**2)

    print("\n9. Lagrangian -> Hamiltonian via Legendre transform:")
    L = TestQuadraticLagrangian(lam=2.0)
    H_legendre = L.legendre_transform()

    # For L = ½λ|α|², the Legendre transform gives H = ½|p|²/λ
    # With λ=2 and p=1: H = 0.5 * 1 / 2 = 0.25
    H_legendre_val = H_legendre(x, m_val, p_val, t_val)
    print("   L(α) = ½ * 2 * |α|² -> H(p) = ½|p|²/2")
    print(f"   H(p=1) = {H_legendre_val:.4f}  (expected: ~0.25)")
    # Allow some tolerance for numerical Legendre transform
    assert abs(H_legendre_val - 0.25) < 0.05, f"Legendre transform failed: {H_legendre_val}"

    print("\n10. Hamiltonian -> Lagrangian via inverse Legendre transform:")
    # Test symmetric duality: H -> L -> H should recover original
    H_orig = SeparableHamiltonian(
        control_cost=QuadraticControlCost(control_cost=2.0),
    )
    L_from_H = H_orig.legendre_transform()

    # For H = ½|p|²/λ, the inverse Legendre transform gives L = ½λ|α|²
    # With λ=2 (control_cost=2): L(α=1) = 0.5 * 2 * 1 = 1.0
    alpha_test = np.array([1.0])
    L_val = L_from_H(x, alpha_test, m_val, t_val)
    print("   H(p) = ½|p|²/2 -> L(α) = ½ * 2 * |α|²")
    print(f"   L(α=1) = {L_val:.4f}  (expected: ~1.0)")
    assert abs(L_val - 1.0) < 0.1, f"Inverse Legendre transform failed: {L_val}"

    print("\n11. Symmetric duality: L -> H -> L should recover original:")
    L_orig = TestQuadraticLagrangian(lam=2.0)
    H_from_L = L_orig.legendre_transform()
    L_recovered = H_from_L.legendre_transform()

    # L_recovered(α=1) should ≈ L_orig(α=1) = 0.5 * 2 * 1 = 1.0
    L_orig_val = L_orig(x, alpha_test, m_val, t_val)
    L_recovered_val = L_recovered(x, alpha_test, m_val, t_val)
    print(f"   L_orig(α=1) = {L_orig_val:.4f}")
    print(f"   L_recovered(α=1) = {L_recovered_val:.4f}")
    assert abs(L_recovered_val - L_orig_val) < 0.2, f"Duality cycle failed: {L_recovered_val} vs {L_orig_val}"
    print("   Duality cycle: L -> H -> L verified!")

    print("\n12. MFGOperator properties:")
    print(f"   H.is_hamiltonian = {H_orig.is_hamiltonian}  (expected: True)")
    print(f"   H.is_lagrangian = {H_orig.is_lagrangian}    (expected: False)")
    print(f"   L.is_hamiltonian = {L_orig.is_hamiltonian}  (expected: False)")
    print(f"   L.is_lagrangian = {L_orig.is_lagrangian}    (expected: True)")
    assert H_orig.is_hamiltonian is True
    assert H_orig.is_lagrangian is False
    assert L_orig.is_hamiltonian is False
    assert L_orig.is_lagrangian is True

    print("\n" + "=" * 60)
    print("All smoke tests passed!")
