"""
Model and Conditions classes: the model/domain/conditions form of MFGProblem.

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
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

    from mfgarchon.core.hamiltonian import HamiltonianBase
    from mfgarchon.core.mfg_problem import MFGProblem
    from mfgarchon.operators.interaction.energy_functionals import EnergyFunctional


@runtime_checkable
class CouplingCost(Protocol):
    """A coupling cost F[m], the value of ``Model.coupling_cost``: a specification free of any discretization, which
    the library binds to a problem (#1642, the user's ruling 1 of 2026-10-14).

    ``bind(problem)`` returns the ``EnergyFunctional`` of F on that problem's nodes and measure, whose flat derivative
    the HJB source composes. The shipped specifications are ``InteractionCost(kernel)`` and ``PotentialCost(V)`` in
    ``mfgarchon.operators.interaction``, and their sum.
    """

    def bind(self, problem: MFGProblem) -> EnergyFunctional: ...


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
        coupling_cost: The coupling cost F[m], a `CouplingCost` -- ``InteractionCost(kernel)``,
            ``PotentialCost(V)`` or their sum (``mfgarchon.operators.interaction``). ``MFGProblem`` binds it to its grid
            and measure, and the HJB carries its flat derivative ``delta F / delta m`` as a source, cost-signed, summed
            with the Hamiltonian's own coupling ``f`` (#1642). Bound on a ``TensorProductGrid``; a solver that evaluates
            its source at its own points, such as HJB-GFDM's collocation points, reads ``delta F / delta m`` there.
            Refused on FEM and meshless geometries until #1642's stage 1b, and on a network, where no solve takes a
            ``Model`` (#2045).
            In a multi-population problem each population's own ``Model`` carries
            its own F_k, which reads that population's density m_k alone; coupling across populations belongs in each
            H_k, through the stacked densities (#2173). The same holds per regime of a regime-switching problem and per
            node problem of a graph solve: each composes its own F on its own density. A plain ``F[m] -> float`` is
            refused: it would carry its own quadrature, so it is bound to one grid (#1642, the user's ruling 3).
        terminal_coupling: The terminal coupling cost G[m(T)], a `CouplingCost` like ``coupling_cost``. ``MFGProblem``
            binds it to its grid and measure, and the terminal condition is ``u(T, x) = g(x) + delta G / delta m`` at the
            density m(T) the HJB solve uses (`MFGProblem.terminal_value`, #1642 stage 2). Bound on a
            ``TensorProductGrid``; refused on FEM and meshless geometries until #1642's stage 1b, on a network (#2568),
            and in `MultiPopulationIterator`, which with more than one population couples them through their stacked
            densities (#1642).
    """

    hamiltonian: HamiltonianBase | None = None
    lagrangian: Callable | None = None
    volatility: float | NDArray | Callable = 0.1
    volatility_kind: str | None = None
    drift_field: Callable | None = None
    coupling_cost: CouplingCost | None = None
    terminal_coupling: CouplingCost | None = None

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
        self.refuse_what_is_not_bound()

    def refuse_what_is_not_bound(self) -> None:
        """The refusals of what a problem cannot bind, read at construction and again where a problem binds this model,
        since a ``Model`` is a mutable dataclass (#1642): a ``coupling_cost`` or ``terminal_coupling`` that is not a
        `CouplingCost`."""
        for field in ("coupling_cost", "terminal_coupling"):
            _refuse_unbindable_cost(field, getattr(self, field))

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
    solve's output -- currently has no expression on `Conditions`. `lambda _x: snapshot` does
    NOT work: the signature validator evaluates it and refuses the array it gets
    back. That gap is real and wants a constructor that ADMITS it is grid-bound,
    not a relaxation of this rule.

    Callable signature: called once per grid point -- 1-D: ``f(x)`` with ``x`` a float; n-D:
    ``f(x)`` with ``x`` an array of shape ``(d,)`` -- returning one number. A time-first ``f(t, x)``
    is read at ``T`` (``u_terminal``) or ``0`` (``m_initial``); ``f(x, t)`` is refused
    (docs/user/CONVENTIONS.md § Callable signatures).

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
                "is no way to pass it to Conditions yet; `lambda _x: array` is rejected by the signature "
                "check, not accepted."
            )
        if self.u_terminal is not None and not callable(self.u_terminal):
            raise TypeError(
                f"u_terminal must be callable, got {type(self.u_terminal).__name__}. "
                "Use a function: u_terminal=lambda x: (x-0.5)**2"
            )
        if self.T <= 0:
            raise ValueError(f"T must be positive, got {self.T}")


def _refuse_unbindable_cost(field: str, cost: object) -> None:
    """Refuse a ``Model.coupling_cost`` or ``Model.terminal_coupling`` the library cannot bind: a class, an
    ``EnergyFunctional`` already bound to a grid, or anything that is not a `CouplingCost` (#1642). None passes."""
    if cost is None or (not isinstance(cost, type) and isinstance(cost, CouplingCost)):
        return
    from mfgarchon.operators.interaction.energy_functionals import missing_energy_functional_members

    bound = not isinstance(cost, type) and not missing_energy_functional_members(cost)
    if isinstance(cost, type):
        what = f"the class {cost.__name__}, not an instance of it"
    elif bound and field == "coupling_cost":
        what = (
            f"{type(cost).__name__}, an EnergyFunctional -- F already bound to a grid. Give the specification "
            "it was bound from, or pass create_lions_source(energy) as source_term_hjb"
        )
    elif bound:
        what = f"{type(cost).__name__}, an EnergyFunctional -- G already bound to a grid. Give the specification it was bound from"
    else:
        what = type(cost).__name__
    if field == "terminal_coupling":
        raise TypeError(
            f"Model(terminal_coupling=...) takes a coupling cost the library binds to the problem's grid and measure -- "
            f"InteractionCost(kernel), PotentialCost(V) or their sum, from mfgarchon.operators.interaction, or an "
            f"instance of a class of your own whose bind(problem) returns an EnergyFunctional -- not {what}. The "
            f"problem adds its flat derivative delta G / delta m at m(T) to u(T) (#1642). A terminal cost that does not "
            f"depend on m(T) is Conditions(u_terminal=...)."
        )
    raise TypeError(
        f"Model(coupling_cost=...) takes a coupling cost the library binds to the problem's grid and measure -- "
        f"InteractionCost(kernel), PotentialCost(V) or their sum, from mfgarchon.operators.interaction, or an "
        f"instance of a class of your own whose bind(problem) returns an EnergyFunctional -- not {what}. A plain F[m] -> float carries its own quadrature, so it is bound to one grid (#1642). Give "
        f"such an F to the problem's HJB source explicitly, on that grid: MFGProblem(..., domain=grid, "
        f"source_term_hjb=create_lions_source(F, FiniteDifferenceFunctionalDerivative(), "
        f"weights=grid.measure_weights().ravel())), with create_lions_source from "
        f"mfgarchon.alg.numerical.coupling.lions_correction and FiniteDifferenceFunctionalDerivative from "
        f"mfgarchon.utils.functional_calculus; or pass its flat derivative delta F / delta m as source_term_hjb "
        f"directly."
    )
