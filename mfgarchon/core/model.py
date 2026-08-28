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

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

    from mfgarchon.core.hamiltonian import HamiltonianBase


@dataclass
class Model:
    """The game rules -- agent dynamics and cost structure.

    Provide EITHER hamiltonian OR lagrangian (dual descriptions of the same game).
    The other is derived via Legendre transform.

    The Hamiltonian formulation H(x, p, m) enters the HJB equation:
        -du/dt + H(x, grad(u), m) = 0
    The Lagrangian formulation L(x, alpha, m) enters the variational problem:
        min_alpha integral L(x, alpha, m) dt

    Args:
        hamiltonian: H(x, p, m) -- Hamiltonian formulation.
        lagrangian: L(x, alpha, m) -- Lagrangian formulation.
        sigma: Diffusion coefficient (float, array, or callable).
        drift_field: Prescribed drift for FP-only problems (no optimization).
        coupling_cost: F(m) -- interaction cost (variational formulation).
        terminal_coupling: G(x, m(T)) -- terminal cost in objective (variational).
    """

    hamiltonian: HamiltonianBase | None = None
    lagrangian: Callable | None = None
    sigma: float | NDArray | Callable = 0.1
    drift_field: Callable | None = None
    coupling_cost: Callable | None = None
    terminal_coupling: Callable | None = None

    def __post_init__(self) -> None:
        has_h = self.hamiltonian is not None
        has_l = self.lagrangian is not None
        has_d = self.drift_field is not None

        if not has_h and not has_l and not has_d:
            raise ValueError("Model requires at least one of: hamiltonian, lagrangian, or drift_field")
        if has_h and has_l:
            raise ValueError(
                "Provide hamiltonian OR lagrangian, not both. "
                "They are dual descriptions -- the other is derived via Legendre transform."
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
                "Note it must integrate to 1 on YOUR grid -- the library validates and reports the "
                "mass but does not rescale it (#1887) -- so divide by "
                "`domain.integrate(f(domain.coordinates[0]))` when you build it. "
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
