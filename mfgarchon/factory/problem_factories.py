"""Problem construction, not built. #1920

The previous implementation was 885 lines here plus 556 in `general_mfg_factory.py`, and its
central method **silently discarded the caller's boundary conditions**. Measured before deletion:

    caller asked for                        DIRICHLET (value 7.0)
    problem.geometry.boundary_conditions    NO_FLUX
    problem.get_boundary_conditions()       NO_FLUX

`create_from_hamiltonian` passed `boundary_conditions=` into the components while building the
geometry with a hardcoded `no_flux_bc(dimension=1)`, and solvers resolve the BC from the geometry.
No error, no warning. Every non-no-flux BC on every problem built this way was replaced by a
reflecting wall.

**What a rebuilt factory owes.** A factory that only assembles is an example with extra steps; the
reason to have one in this library is that construction carries invariants a caller will get wrong,
and today none of them is enforced anywhere:

- the caller's `BoundaryConditions` must reach the geometry, which is what solvers read, and not
  only the components (the defect above);
- the control cost's `OptimizationSense` must agree with the Hamiltonian's -- measured, all four
  pairs construct silently and on a mismatch the control cost decides (#1986);
- the drift convention must match the solver family (`_drift_convention`, VELOCITY vs
  VALUE_FUNCTION);
- an unsupported `BCType` must be refused at construction, which needs the solver's declaration,
  its `supported_bc_types` property, AND a `_validate_bc_support` call -- a solver missing any one
  of the three is ungated (#1977).

Until it enforces those, hand assembly is not worse: `MFGProblem(geometry=..., components=...)`
puts the caller in front of the same choices with nothing pretending they are handled.

Nothing in `mfgarchon/` imported this module, and no example used it -- the two that appeared to
define their own local functions of the same names.
"""

from __future__ import annotations

from typing import Any, NoReturn

__all__ = [
    # Classic LQ-MFG conditions (#670). Kept: these are reusable problem DATA -- a terminal cost
    # and an initial density -- not construction, so the defect above does not touch them.
    "lq_mfg_initial_density",
    "lq_mfg_terminal_cost",
    "create_crowd_problem",
    "create_general_mfg_problem",
    "create_highdim_problem",
    "create_lq_problem",
    "create_mfg_problem",
    "create_network_problem",
    "create_standard_problem",
    "create_stochastic_problem",
    "create_variational_problem",
]

_MESSAGE = (
    "mfgarchon.factory problem construction is not built (#1920). The previous implementation "
    "silently replaced the caller's BoundaryConditions with a hardcoded no-flux wall, so it was "
    "removed rather than left to answer wrongly. Build the problem directly -- "
    "MFGProblem(geometry=..., components=...) -- which puts you in front of the same choices "
    "without pretending they are handled. See this module's docstring for what a rebuilt factory "
    "must guarantee before it is worth using."
)


def _unbuilt(*_args: Any, **_kwargs: Any) -> NoReturn:
    raise NotImplementedError(_MESSAGE)


create_crowd_problem = _unbuilt
create_general_mfg_problem = _unbuilt
create_highdim_problem = _unbuilt
create_lq_problem = _unbuilt
create_mfg_problem = _unbuilt
create_network_problem = _unbuilt
create_standard_problem = _unbuilt
create_stochastic_problem = _unbuilt
create_variational_problem = _unbuilt


def lq_mfg_terminal_cost(Lx: float = 1.0):
    """Classic LQ-MFG terminal cost: g(x) = 5*(cos(2πx/L) + 0.4*sin(4πx/L)).

    Args:
        Lx: Domain length (default 1.0)

    Returns:
        Callable that computes terminal cost at position x

    Example:
        >>> problem = MFGProblem(
        ...     geometry=grid,
        ...     u_terminal=lq_mfg_terminal_cost(Lx=1.0),
        ...     m_initial=lq_mfg_initial_density(),
        ... )
    """
    import numpy as np

    def u_terminal(x: float) -> float:
        return 5 * (np.cos(x * 2 * np.pi / Lx) + 0.4 * np.sin(x * 4 * np.pi / Lx))

    return u_terminal


def lq_mfg_initial_density():
    """Classic LQ-MFG initial density: bimodal Gaussian at x=0.2 and x=0.8.

    Returns:
        Callable that computes initial density at position x

    Note:
        MFGProblem will automatically normalize the density to integrate to 1.

    Example:
        >>> problem = MFGProblem(
        ...     geometry=grid,
        ...     u_terminal=lq_mfg_terminal_cost(),
        ...     m_initial=lq_mfg_initial_density(),
        ... )
    """
    import numpy as np

    def m_initial(x: float) -> float:
        return 2 * np.exp(-200 * (x - 0.2) ** 2) + np.exp(-200 * (x - 0.8) ** 2)

    return m_initial
