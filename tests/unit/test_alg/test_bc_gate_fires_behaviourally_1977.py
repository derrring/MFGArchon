"""A solver is gated iff it REFUSES a BC type it does not support. Declaring is not gating.

Measured 2026-08-17, on `FPFEMSolver`, in three steps:

    _SUPPORTED_BC_TYPES declared, nothing else   PERIODIC constructed
    + `_validate_bc_support(...)` called          PERIODIC constructed
    + `supported_bc_types` property               PERIODIC REFUSED

The gate reads the **property**; `_validate_bc_support` returns early on `supported is None`, and
`supported` comes from the property, not the private attribute. So a solver can declare, and call
the gate, and still be as ungated as one that declares nothing.

The declaration census in `test_solver_bc_support_census_1975.py` and `test_capability_census.py`
reads `_SUPPORTED_BC_TYPES`. It would have called that first state "gated". This file asserts the
behaviour instead, which no combination of the three parts can fake.
"""

from __future__ import annotations

import pytest

import numpy as np

from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.core.mfg_components import MFGComponents
from mfgarchon.core.mfg_problem import MFGProblem
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import BCSegment, BCType, BoundaryConditions

pytestmark = pytest.mark.filterwarnings("ignore")

_N = 21


def _components():
    return MFGComponents(
        m_initial=lambda x: np.ones_like(x),
        u_terminal=lambda x: 0.0 * x,
        hamiltonian=SeparableHamiltonian(control_cost=QuadraticControlCost(control_cost=1.0)),
    )


def _bc(bc_type: BCType, **kw) -> BoundaryConditions:
    return BoundaryConditions(
        dimension=1,
        # Without a default, FPGFDMSolver raises the #1100 ValueError from _resolve_boundary_type
        # before the capability gate is reached -- a different refusal, on a different axis.
        default_bc=bc_type,
        segments=[
            BCSegment(name="L", bc_type=bc_type, boundary="x_min", **kw),
            BCSegment(name="R", bc_type=bc_type, boundary="x_max", **kw),
        ],
    )


def _grid_problem(bc: BoundaryConditions) -> MFGProblem:
    grid = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[_N], boundary_conditions=bc)
    return MFGProblem(geometry=grid, T=0.2, Nt=5, sigma=0.3, components=_components())


#: Solvers constructible on a plain 1-D structured grid, and one BCType each declares it does NOT
#: support. Kept small on purpose: the point is that the gate FIRES, not a coverage matrix.
_GRID_SOLVERS = {
    "FPFDMSolver": (BCType.ROBIN, {"alpha": 1.0, "beta": 1.0, "value": 0.0}),
    "FPFVMSolver": (BCType.DIRICHLET, {"value": 0.0}),
    "FPGFDMSolver": (BCType.PERIODIC, {}),
    "HJBFDMSolver": (BCType.REFLECTING, {}),
    "HJBWENOSolver": (BCType.DIRICHLET, {"value": 0.0}),
}


@pytest.mark.parametrize("name", sorted(_GRID_SOLVERS))
def test_the_gate_refuses_an_unsupported_type_at_construction(name):
    import mfgarchon.alg.numerical.fp_solvers as fp
    import mfgarchon.alg.numerical.hjb_solvers as hjb

    cls = getattr(fp, name, None) or getattr(hjb, name)
    bc_type, kw = _GRID_SOLVERS[name]
    assert bc_type not in cls._SUPPORTED_BC_TYPES, f"{name} now declares {bc_type.name}; pick another"

    problem = _grid_problem(_bc(bc_type, **kw))
    kwargs = {"collocation_points": np.linspace(0.0, 1.0, _N).reshape(-1, 1), "delta": 0.2} if "GFDM" in name else {}
    with pytest.raises(NotImplementedError):
        cls(problem, **kwargs)


def test_declaring_without_the_property_would_not_gate():
    """The state `FPFEMSolver` was in for part of 2026-08-17, reconstructed on a solver that works.

    `_validate_bc_support` reads `supported_bc_types`. Removing the property while leaving
    `_SUPPORTED_BC_TYPES` in place makes the gate silently pass -- which is why the declaration
    census cannot answer this question.
    """
    from mfgarchon.alg.numerical.fp_solvers.fp_fdm import FPFDMSolver

    problem = _grid_problem(_bc(BCType.ROBIN, alpha=1.0, beta=1.0, value=0.0))

    class _DeclaresButHasNoProperty(FPFDMSolver):
        supported_bc_types = None  # type: ignore[assignment]

    with pytest.raises(NotImplementedError):
        FPFDMSolver(problem)
    _DeclaresButHasNoProperty(problem)  # constructs: the declaration alone gates nothing
