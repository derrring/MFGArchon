"""
Model and Conditions classes for MFGArchon API v1.0.

These dataclasses separate the mathematical description of an MFG problem
into orthogonal components:

- Model: Game rules (Hamiltonian/Lagrangian + diffusion + coupling)
- Conditions: Problem data (time horizon + initial/terminal conditions)

Design doc: mfg-research/docs/archon-notes/development/API_V1_DESIGN.md
Issue: derrring/MFGArchon#875
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

    from mfgarchon.core.hamiltonian import HamiltonianBase


@dataclass
class Model:
    """The game rules -- agent dynamics and cost structure.

    Provide EITHER hamiltonian OR lagrangian (dual descriptions of the same game). Only the
    Hamiltonian form can be solved today: the Legendre transform from a Lagrangian is not
    implemented, and ``effective_hamiltonian`` raises for one (docs/user/CONVENTIONS.md
    § Component ownership).

    The Hamiltonian formulation H(t, x, p, m) enters the HJB equation:
        -du/dt + H(t, x, grad(u), m) - tr(A D^2 u) = 0,  A = 1/2 Sigma Sigma^T
    The Lagrangian formulation L(t, x, alpha, m) enters the variational problem:
        min_alpha integral L(t, x, alpha, m) dt

    Args:
        hamiltonian: H(t, x, p, m) -- Hamiltonian formulation.
        lagrangian: L(t, x, alpha, m) -- Lagrangian formulation.
        volatility: The SDE volatility Sigma in dX = alpha dt + Sigma dW -- a float, an array
            (which needs ``volatility_kind``), or a callable. Not the PDE diffusion, which is
            A = 1/2 Sigma Sigma^T and is derived from it (#2375 ruling 6).
        volatility_kind: For an array volatility, ``"field"`` (isotropic per-point sigma) or
            ``"tensor"`` (a square ``(d, d)`` matrix, or ``(*grid, d, d)`` per point; a non-square
            matrix is refused). Required for an array.
        drift_field: Prescribed drift for FP-only problems (no optimization).
        coupling_cost: F(m) -- interaction cost (variational formulation). Read by nothing yet (#2429).
        terminal_coupling: G(x, m(T)) -- terminal cost in objective (variational). Read by
            nothing yet (#2429).
    """

    hamiltonian: HamiltonianBase | None = None
    lagrangian: Callable | None = None
    volatility: float | NDArray | Callable = 0.1
    volatility_kind: str | None = None
    drift_field: Callable | None = None
    coupling_cost: Callable | None = None
    terminal_coupling: Callable | None = None

    @property
    def sigma(self) -> Any:
        """Retired (#2375 ruling 6): reading it raises, naming ``volatility``."""
        raise AttributeError(_RETIRED_SIGMA)

    def __post_init__(self) -> None:
        has_h = self.hamiltonian is not None
        has_l = self.lagrangian is not None
        has_d = self.drift_field is not None

        if not has_h and not has_l and not has_d:
            raise ValueError("Model requires at least one of: hamiltonian, lagrangian, or drift_field")
        if has_h and has_l:
            raise ValueError(
                "Provide hamiltonian OR lagrangian, not both. They are dual descriptions of the same game; supply one."
            )

    @property
    def effective_hamiltonian(self) -> HamiltonianBase:
        """Always available -- derived from Lagrangian if needed."""
        if self.hamiltonian is not None:
            return self.hamiltonian
        if self.lagrangian is not None:
            raise NotImplementedError(
                "Automatic Legendre transform from Lagrangian to Hamiltonian "
                "is not yet implemented. Provide hamiltonian directly, or "
                "implement the transform for your specific Lagrangian."
            )
        raise ValueError("No hamiltonian or lagrangian defined")


# `sigma` is retired on both sides of Model (#2375 ruling 6). Reading it is the raising property above.
# Passing it is refused here, around the generated __init__, which `dataclasses.replace` also calls --
# a retired InitVar would instead be read back by `replace` and by any `getattr`, as `None`.
_RETIRED_SIGMA = (
    "Model.sigma is retired (#2375 ruling 6): the SDE volatility is Model.volatility -- pass "
    "volatility=, with volatility_kind='field' or 'tensor' for an array."
)
_generated_model_init = Model.__init__


@functools.wraps(_generated_model_init)
def _model_init_refusing_sigma(self: Model, *args: Any, **kwargs: Any) -> None:
    if "sigma" in kwargs:
        raise TypeError(_RETIRED_SIGMA.replace("Model.sigma is retired", "Model(sigma=...) is retired", 1))
    _generated_model_init(self, *args, **kwargs)


Model.__init__ = _model_init_refusing_sigma  # type: ignore[method-assign]


@dataclass
class Conditions:
    """Problem data: time horizon + initial/terminal conditions.

    Both u_terminal and m_initial MUST be callables (not arrays).
    This preserves orthogonality with Domain -- the same conditions
    work on any grid resolution.

    The conversion is one-way and it already happens: `MFGProblem` evaluates the
    callable once at construction and stores an array on `problem.m_initial`,
    while `conditions.m_initial` keeps the callable. Measured, a callable and the
    equivalent array produce element-wise identical internal state -- so this rule
    is not about the representation the solvers see. It is about what THIS object
    holds: a callable is `m_0(x)` on the continuum, an array is already a
    projection onto one grid, and a Conditions holding a grid is a Conditions that
    is no longer orthogonal to Domain.

    A density with no closed form -- a KDE, a measured histogram, a previous
    solve's output -- currently has no v1.0 expression. `lambda _x: snapshot` does
    NOT work: the signature validator evaluates it and refuses the array it gets
    back. That gap is real and wants a constructor that ADMITS it is grid-bound,
    not a relaxation of this rule.

    Callable signature:
        1D: f(x) where x shape (N,), returns (N,)
        nD: f(x) where x shape (N, d), returns (N,)

    Args:
        u_terminal: Terminal cost u_T(x). None for variational (u is derived).
        m_initial: Initial density m_0(x).
        T: Time horizon (physics, not discretization).
    """

    u_terminal: Callable | None = None
    m_initial: Callable | None = None
    T: float = 1.0

    def __post_init__(self) -> None:
        if self.m_initial is not None and not callable(self.m_initial):
            raise TypeError(
                f"m_initial must be callable, got {type(self.m_initial).__name__}. "
                "Conditions holds m_0(x) on the continuum; an array is already a projection onto "
                "one grid, and holding one here would make Conditions grid-bound. "
                "Use a function: m_initial=lambda x: np.exp(-5*(x-0.5)**2). "
                "The library validates and reports its mass but does not rescale it (#1887); a "
                "mass of 1 is your modelling decision, not a requirement. If you do want one, "
                "compute the constant yourself against the grid you will solve on -- in 1-D "
                "`x = domain.coordinates[0]; c = float(domain.integrate(f(x)))`, then divide your "
                "closed form by `c`. `coordinates` is a list of per-axis vectors, so that "
                "one-liner is 1-D only: in n-D build the field on `domain.meshgrid(indexing='ij')` "
                "first, or `integrate` refuses the shape. "
                "If your density has no closed form (a KDE, a histogram, a previous solve), there "
                "is no v1.0 way to pass it yet; `lambda _x: array` is rejected by the signature "
                "check, not accepted."
            )
        if self.u_terminal is not None and not callable(self.u_terminal):
            raise TypeError(
                f"u_terminal must be callable, got {type(self.u_terminal).__name__}. "
                "Use a function: u_terminal=lambda x: (x-0.5)**2"
            )
        if self.T <= 0:
            raise ValueError(f"T must be positive, got {self.T}")


@dataclass
class ErgodicConditions:
    """Stationary MFG -- no time horizon, no terminal condition.

    For ergodic (long-time average) MFG problems where
    the solution is time-independent.

    Args:
        m_stationary_guess: Initial guess for stationary density.
        discount_rate: Discount factor for discounted ergodic problems.
    """

    m_stationary_guess: Callable | None = None
    discount_rate: float | None = None
