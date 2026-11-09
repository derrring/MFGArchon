"""Problem construction, not built. #1920

The previous implementation was 885 lines here plus 556 in `general_mfg_factory.py`, and its
central method **silently discarded the caller's boundary conditions**. Measured before deletion:

    caller asked for                        DIRICHLET (value 7.0)
    problem.geometry.boundary_conditions    NO_FLUX
    problem.get_boundary_conditions()       NO_FLUX

`create_from_hamiltonian` passed `boundary_conditions=` into the components while building the
geometry with a hardcoded `no_flux_bc(dimension=1)`. The solvers then split: the HJB solvers resolve
the BC from the geometry and ran on a reflecting wall, while `FPFDMSolver` and `FPFVMSolver` read the
components' BC first and ran on the caller's boundary. No error, no warning: every non-no-flux BC on
every problem built this way gave an HJB and an FP solved under different boundaries.

**What a rebuilt factory owes.** A factory that only assembles is an example with extra steps; the
reason to have one in this library is that construction carries invariants a caller will get wrong,
and today none of them is enforced anywhere:

- the caller's `BoundaryConditions` must reach the geometry, which the HJB solvers read, and not
  only the components (the defect above);
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

__all__ = [
    # Classic LQ-MFG conditions (#670). Kept: these are reusable problem DATA -- a terminal cost
    # and an initial density -- not construction, so the defect above does not touch them.
    "lq_mfg_initial_density",
    "lq_mfg_terminal_cost",
]


def lq_mfg_terminal_cost(Lx: float = 1.0):
    """Classic LQ-MFG terminal cost: g(x) = 5*(cos(2πx/L) + 0.4*sin(4πx/L)).

    Args:
        Lx: Domain length (default 1.0)

    Returns:
        Callable that computes terminal cost at position x

    Example:
        >>> problem = MFGProblem(
        ...     model=Model(hamiltonian=SeparableHamiltonian(control_cost=QuadraticControlCost(lambda_=1.0))),
        ...     domain=grid,
        ...     conditions=Conditions(u_terminal=lq_mfg_terminal_cost(Lx=1.0), m_initial=lq_mfg_initial_density(), T=1.0),
        ...     Nt=50,
        ... )
    """
    import numpy as np

    def u_terminal(x: tuple) -> float:
        return 5 * (np.cos(x[0] * 2 * np.pi / Lx) + 0.4 * np.sin(x[0] * 4 * np.pi / Lx))

    return u_terminal


def lq_mfg_initial_density():
    """Classic LQ-MFG initial density: bimodal Gaussian at x=0.2 and x=0.8.

    Returns:
        Callable that computes initial density at position x

    Note:
        This density is NOT normalised, and `MFGProblem` will not normalise it for you (#1887):
        it measures the mass, names the measure it used, warns if the mass is not 1, and leaves
        your array alone. Measured on the unit interval it integrates to 0.381352 at Nx=11 and
        0.375982 at Nx=101 -- the grid dependence there is the quadrature converging, not the
        density changing. If you want a probability density, divide by `grid.integrate(...)`
        yourself, and note that doing so changes the problem: the coupling f(m) is then evaluated
        at about 2.66 times these values.

    Example:
        >>> problem = MFGProblem(
        ...     model=Model(hamiltonian=SeparableHamiltonian(control_cost=QuadraticControlCost(lambda_=1.0))),
        ...     domain=grid,
        ...     conditions=Conditions(u_terminal=lq_mfg_terminal_cost(Lx=1.0), m_initial=lq_mfg_initial_density(), T=1.0),
        ...     Nt=50,
        ... )
    """
    import numpy as np

    def m_initial(x: tuple) -> float:
        return 2 * np.exp(-200 * (x[0] - 0.2) ** 2) + np.exp(-200 * (x[0] - 0.8) ** 2)

    return m_initial
