"""FP solvers must refuse a Neumann value they do not honour (Issue #1686).

Every FP family declares `BCType.NEUMANN` in `_SUPPORTED_BC_TYPES` and then reads only the type:
`neumann_bc(value=g)` with `g != 0` is applied by the HJB side and silently discarded by the FP
side, so the coupled solve integrates a pair that is not adjoint and still reports
`converged=True`. Measured before this gate: `max|M(g=0) - M(g=-100)| = 0.0` on the FP side while
the HJB side moved by `1.6e+03`.

Declaring the type without honouring the value is the RFC #1574 class -- a declared surface
broader than the honoured code, silent in the gap. Until an inhomogeneous flux wall exists, the
library refuses the problem rather than solving a different one.
"""

from __future__ import annotations

import sys

import pytest

import numpy as np

import mfgarchon.alg.numerical.fp_solvers as _fp_pkg
from mfgarchon import MFGProblem
from mfgarchon.alg.base_solver import BaseMFGSolver
from mfgarchon.alg.numerical.fp_solvers import FPFDMSolver, FPGFDMSolver
from mfgarchon.alg.numerical.hjb_solvers import HJBFDMSolver
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.core.mfg_components import MFGComponents
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import (
    BCSegment,
    BCType,
    BoundaryConditions,
    neumann_bc,
    no_flux_bc,
)


# Derived, not hand-listed. Two earlier attempts got this wrong in opposite ways: the first
# hardcoded five module names and missed `fp_particle` and `fp_semi_lagrangian_adjoint`; the
# second listed `FPSLSolver` and `FPSLAdjointSolver` -- both from the *adjoint* module -- and
# omitted `FPSLJacobianSolver` entirely, so deleting that module's flag left every test green.
# Walking the package makes the list follow the code instead of my memory.
def _fp_families_that_decline_the_value() -> list[type]:
    families = []
    for _name in dir(_fp_pkg):
        obj = getattr(_fp_pkg, _name)
        if (
            isinstance(obj, type)
            and issubclass(obj, BaseMFGSolver)
            and getattr(obj, "honors_inhomogeneous_neumann", True) is False
        ):
            families.append(obj)
    return sorted(families, key=lambda c: c.__name__)


FP_FAMILIES = _fp_families_that_decline_the_value()


def _construct(solver_cls, problem):
    """Build a solver, supplying the extra arguments a family requires.

    FPGFDMSolver needs collocation points; every other family takes the problem alone. Handled
    here rather than by dropping GFDM from the list, since excluding a family would leave it
    unpinned -- and it declares BCType.NEUMANN like the rest.
    """
    if solver_cls is FPGFDMSolver:
        return solver_cls(problem, collocation_points=np.linspace(0.0, 1.0, 21).reshape(-1, 1))
    return solver_cls(problem)


def _problem(bc):
    return MFGProblem(
        geometry=TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[21], boundary_conditions=bc),
        Nt=10,
        T=1.0,
        components=MFGComponents(
            m_initial=lambda x: np.exp(-10 * (np.asarray(x) - 0.5) ** 2).squeeze(),
            u_terminal=lambda x: 0.0,
            hamiltonian=SeparableHamiltonian(control_cost=QuadraticControlCost(control_cost=1.0)),
        ),
    )


@pytest.mark.parametrize("solver_cls", FP_FAMILIES)
def test_fp_refuses_a_neumann_value_it_would_discard(solver_cls):
    """A non-zero Neumann value must raise, naming the value and the issue."""
    with pytest.raises(NotImplementedError, match="honours only the homogeneous case"):
        _construct(solver_cls, _problem(neumann_bc(value=5.0, dimension=1)))


@pytest.mark.parametrize("solver_cls", FP_FAMILIES)
@pytest.mark.parametrize(
    ("bc_factory", "label"),
    [
        (lambda: neumann_bc(value=0.0, dimension=1), "neumann g=0"),
        (lambda: no_flux_bc(dimension=1), "no_flux"),
    ],
)
def test_fp_still_accepts_the_homogeneous_case(solver_cls, bc_factory, label):
    """`g = 0` is the whole of current usage and must remain accepted."""
    _construct(solver_cls, _problem(bc_factory()))


def test_hjb_still_accepts_a_neumann_value():
    """The HJB side honours `du/dn = g` and must not be caught by the FP-side refusal.

    This is what makes the flag per-solver rather than global: the same `BCType.NEUMANN` means
    `du/dn = g` on the HJB side and a prescribed flux `J.n = g` on the FP side.
    """
    HJBFDMSolver(_problem(neumann_bc(value=5.0, dimension=1)))


def test_the_flag_is_what_decides():
    """Pin the mechanism, not just the outcome -- flipping the flag flips the behaviour."""
    problem = _problem(neumann_bc(value=5.0, dimension=1))

    assert FPFDMSolver.honors_inhomogeneous_neumann is False
    assert HJBFDMSolver.honors_inhomogeneous_neumann is True

    class _Claims(FPFDMSolver):
        honors_inhomogeneous_neumann = True

    _Claims(problem)  # must not raise: the refusal is driven by the flag, not by the class


@pytest.mark.parametrize("solver_cls", FP_FAMILIES)
def test_a_time_dependent_neumann_value_is_also_refused(solver_cls):
    """`isinstance(value, (int, float))` alone lets a callable through.

    A first version of this gate guarded with that check, so
    `neumann_bc(value=lambda t: 5.0)` was accepted and discarded silently -- the behaviour the
    gate exists to stop, reintroduced by the guard itself. A callable cannot be shown identically
    zero here, so it is refused rather than assumed homogeneous, and the message says so.
    """
    with pytest.raises(NotImplementedError, match="time-dependent value cannot be checked"):
        _construct(solver_cls, _problem(neumann_bc(value=lambda t: 5.0, dimension=1)))


def test_hjb_still_accepts_a_time_dependent_neumann_value():
    """The HJB side honours `du/dn = g(t)`; the refusal must not reach it."""
    HJBFDMSolver(_problem(neumann_bc(value=lambda t: 5.0, dimension=1)))


def test_every_fp_module_declaring_neumann_is_covered():
    """The derived family list must cover every module that declares BCType.NEUMANN.

    Pins the enumeration itself. Without this, dropping the flag from one module leaves the
    whole file green -- which is exactly what happened to `fp_semi_lagrangian.py`.
    """
    import pathlib

    pkg_dir = pathlib.Path(_fp_pkg.__file__).parent
    declaring = {f.stem for f in pkg_dir.glob("fp_*.py") if "_SUPPORTED_BC_TYPES" in f.read_text()}
    covered = {pathlib.Path(sys.modules[c.__module__].__file__).stem for c in FP_FAMILIES}

    assert declaring == covered, (
        f"modules declaring BCType.NEUMANN {sorted(declaring)} are not the modules covered by "
        f"FP_FAMILIES {sorted(covered)}; a family is unpinned"
    )


def _neumann_segment(value):
    return BoundaryConditions(
        default_bc=BCType.NO_FLUX,
        dimension=1,
        segments=[BCSegment(name="wall", bc_type=BCType.NEUMANN, value=value, boundary="x_min")],
    )


@pytest.mark.parametrize(
    ("value", "should_refuse", "label"),
    [
        (np.zeros(21), False, "all-zero array is g = 0"),
        (np.ones(21), True, "non-zero array"),
        (None, False, "unset value"),
        (0.0, False, "scalar zero"),
    ],
)
def test_values_float_cannot_handle_do_not_escape_as_typeerror(value, should_refuse, label):
    """A capability gate must not raise TypeError from inside `float()`.

    The first version guarded only `None` and `callable`, so an array reached `float()` and
    raised `TypeError: only 0-dimensional arrays can be converted to Python scalars` -- and an
    all-zero array is a legitimate ``g = 0`` that was crashing rather than being accepted.
    """
    problem = _problem(_neumann_segment(value))
    if should_refuse:
        with pytest.raises(NotImplementedError, match="honours only the homogeneous case"):
            FPFDMSolver(problem)
    else:
        FPFDMSolver(problem)


def test_the_uniform_default_is_checked_too():
    """`default_bc=NEUMANN, default_value=g` is a value like any other.

    Checking only `segments` left the original silent discard reachable through the default.
    """
    bc = BoundaryConditions(
        default_bc=BCType.NEUMANN,
        default_value=5.0,
        dimension=1,
        segments=[BCSegment(name="wall", bc_type=BCType.NEUMANN, value=None)],
    )
    with pytest.raises(NotImplementedError, match="honours only the homogeneous case"):
        FPFDMSolver(_problem(bc))


def test_the_message_names_what_it_found():
    """Each refusal must describe the value it actually saw.

    `_describe` returns a string for four categories, only one of which is time-dependent. A
    predicate of `any(isinstance(v, str) ...)` therefore reported "time-dependent" for arrays and
    providers, and dropped the scalar entirely from a mixed segment set -- so the message named
    something the caller had not written.
    """
    array_only = BoundaryConditions(
        default_bc=BCType.NO_FLUX,
        dimension=1,
        segments=[BCSegment(name="w", bc_type=BCType.NEUMANN, value=np.ones(21), boundary="x_min")],
    )
    with pytest.raises(NotImplementedError, match=r"value\(s\) \['<array>'\]"):
        FPFDMSolver(_problem(array_only))

    mixed = BoundaryConditions(
        default_bc=BCType.NO_FLUX,
        dimension=1,
        segments=[
            BCSegment(name="a", bc_type=BCType.NEUMANN, value=5.0, boundary="x_min"),
            BCSegment(name="b", bc_type=BCType.NEUMANN, value=np.ones(21), boundary="x_max"),
        ],
    )
    with pytest.raises(NotImplementedError, match=r"5\.0"):
        FPFDMSolver(_problem(mixed))

    with pytest.raises(NotImplementedError, match="time-dependent value cannot be checked"):
        FPFDMSolver(_problem(neumann_bc(value=lambda t: 5.0, dimension=1)))
