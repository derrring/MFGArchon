"""Every solver that declares PERIODIC must return a periodic field. Issue #1822.

`TensorProductGrid(bounds=[(0, 1)], Nx_points=[21])` is endpoint-inclusive, so under a periodic
boundary condition `x[0]` and `x[-1]` are the same physical point and any field a solver returns
must agree there. This is a property of the continuous problem; no discretisation may violate it
at O(1).

Two invariants, deliberately separate, because each certifies what the other misses:

- **the seam** -- `field[0] == field[-1]`, since they are the same physical point;
- **mass** -- a periodic domain has no boundary, so a periodic FP solve conserves it exactly.

`FPFVMSolver` is the case that forced the split: it satisfies the seam at 2.2e-16 and creates 6.5%
of its own mass. The seam alone would certify it.

This file is a **ratchet, not a bug report**. Six solvers fail the seam and six fail mass today.
Each failure is marked `xfail(strict=True)` with the issue that owns it, so:

- fixing a solver turns its XFAIL into an XPASS, which `strict=True` reports as a failure, forcing
  the marker to be removed in the same change -- the count can only go down;
- a solver that starts declaring PERIODIC without honouring it is caught by the coverage test
  below rather than by nobody.

The order matters and is stated in #1822: this test comes first precisely so per-solver repair has
a number to move. Patching the eight in eight PRs without it is how the class stayed invisible
through #1257 and #1739, each of which fixed one stage of one solver.
"""

from __future__ import annotations

import importlib
import inspect

import pytest

import numpy as np

from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.core.mfg_components import MFGComponents
from mfgarchon.core.mfg_problem import MFGProblem
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import periodic_bc
from mfgarchon.geometry.boundary.types import BCType

NX = 21
NT = 10
SEAM_TOL = 1e-9  # exact zero in exact arithmetic; this admits round-off and meshless quadrature

# The modules searched for PERIODIC-declaring solvers. Listed rather than walked, because a walk
# imports optional-backend modules as a side effect; `test_the_matrix_covers_every_declaring_solver`
# is what stops this list going stale.
_SEARCHED = (
    "mfgarchon.alg.numerical.hjb_solvers.hjb_semi_lagrangian",
    "mfgarchon.alg.numerical.hjb_solvers.hjb_weno",
    "mfgarchon.alg.numerical.hjb_solvers.hjb_gfdm",
    "mfgarchon.alg.numerical.hjb_solvers.hjb_fdm",
    "mfgarchon.alg.numerical.fp_solvers.fp_fdm",
    "mfgarchon.alg.numerical.fp_solvers.fp_fvm",
    "mfgarchon.alg.numerical.fp_solvers.fp_gfdm",
    "mfgarchon.alg.numerical.fp_solvers.fp_particle",
    "mfgarchon.alg.numerical.fp_solvers.fp_semi_lagrangian",
    "mfgarchon.alg.numerical.fp_solvers.fp_semi_lagrangian_adjoint",
)

# name -> (kind, owning issue) for the solvers measured as NOT honouring PERIODIC on ebfc5c96.
# Measured seams, 1D, Nx=21, sigma=0.3, T=0.5, Nt=10, exactly periodic input:
#   HJBSemiLagrangianSolver 7.68e-01   FPParticleSolver   5.78e-01
#   HJBWENOSolver           1.77e-01   FPSLAdjointSolver  1.25e-01
#   HJBFDMSolver            1.53e-01   FPSLSolver         1.25e-01
#   FPFDMSolver             1.11e-01
# HJBGFDMSolver is separate: it declares PERIODIC and raises NotImplementedError for it.
KNOWN_NOT_HONOURED = {
    "HJBWENOSolver": "#1822",
    "HJBFDMSolver": "#1822",
    "HJBGFDMSolver": "#1822 (declares PERIODIC, raises NotImplementedError for it)",
    "FPFDMSolver": "#1822",
    "FPParticleSolver": "#1822",
    "FPSLJacobianSolver": "#1822 (deprecated, retirement tracked in #1756)",
}

# Mass is the second invariant, and it is INDEPENDENT of the seam. Periodic BC has no boundary
# through which mass can leave, so a periodic FP solve conserves it exactly. Measured drift over
# the same solve:
#
#   FPSLSolver / FPSLAdjointSolver  10.77%    FPFVMSolver   6.55%
#   FPParticleSolver                 7.45%    FPGFDMSolver  0.99%
#   FPFDMSolver                      6.35%
#
# `FPFVMSolver` is why this exists: it satisfies the seam invariant to 2.2e-16 and creates 6.5% of
# its own mass, so the seam alone would certify it as honouring PERIODIC.
#
# `FPSLJacobianSolver` conserves to 0.0000% and is NOT listed -- but that is renormalisation by
# fiat, not conservation (RFC #1456 class (b), #1429 S0-11). A passing row here means the number
# is right, not that the mechanism is.
MASS_NOT_CONSERVED = {
    "FPFDMSolver": "#1822",
    "FPFVMSolver": "#1822",
    "FPGFDMSolver": "#1822",
    "FPParticleSolver": "#1822",
    "FPSLSolver": "#1822",
    "FPSLAdjointSolver": "#1822",
}


def _declaring_solvers() -> dict[str, type]:
    """Every class in the searched modules whose `_SUPPORTED_BC_TYPES` contains PERIODIC."""
    found: dict[str, type] = {}
    for module_name in _SEARCHED:
        module = importlib.import_module(module_name)
        for name, cls in inspect.getmembers(module, inspect.isclass):
            if cls.__module__ != module_name:
                continue
            declared = getattr(cls, "_SUPPORTED_BC_TYPES", None)
            if declared and BCType.PERIODIC in declared:
                found[name] = cls
    return found


def _periodic_problem() -> MFGProblem:
    return MFGProblem(
        geometry=TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[NX], boundary_conditions=periodic_bc(dimension=1)),
        T=0.5,
        Nt=NT,
        sigma=0.3,
        components=MFGComponents(
            m_initial=lambda z: 1.0 + 0.5 * np.cos(2 * np.pi * np.asarray(z)),
            u_terminal=lambda z: np.sin(2 * np.pi * np.asarray(z)),
            hamiltonian=SeparableHamiltonian(
                control_cost=QuadraticControlCost(control_cost=1.0),
                coupling=lambda m: m,
                coupling_dm=lambda m: 1.0,
            ),
        ),
    )


def _seam(field: np.ndarray) -> float:
    arr = np.asarray(field)
    if arr.ndim == 1:
        arr = arr[None, :]
    return float(np.abs(arr[:, 0] - arr[:, -1]).max())


def _solve_periodic(cls: type) -> np.ndarray:
    """Run one solve from exactly periodic data and return the field it produced."""
    x = np.linspace(0.0, 1.0, NX)
    u_periodic = np.sin(2 * np.pi * x)
    m_periodic = 1.0 + 0.5 * np.cos(2 * np.pi * x)
    assert _seam(u_periodic) < 1e-15, "the u input itself must be periodic, or the output tells us nothing"
    assert _seam(m_periodic) < 1e-15, "the m input itself must be periodic, or the output tells us nothing"

    problem = _periodic_problem()
    kwargs = {}
    if "collocation_points" in inspect.signature(cls.__init__).parameters:
        kwargs["collocation_points"] = x.reshape(-1, 1)
    solver = cls(problem, **kwargs)

    if hasattr(solver, "solve_hjb_system"):
        return solver.solve_hjb_system(np.tile(m_periodic, (NT + 1, 1)), u_periodic, np.zeros((NT + 1, NX)))
    return solver.solve_fp_system(m_periodic, np.tile(u_periodic, (NT + 1, 1)))


def _parametrised():
    for name, cls in sorted(_declaring_solvers().items()):
        marks = []
        if name in KNOWN_NOT_HONOURED:
            marks.append(
                pytest.mark.xfail(
                    strict=True,
                    reason=f"{name} declares PERIODIC and does not honour it ({KNOWN_NOT_HONOURED[name]})",
                )
            )
        yield pytest.param(name, cls, marks=marks, id=name)


@pytest.mark.parametrize(("name", "cls"), list(_parametrised()))
def test_a_periodic_solve_returns_a_periodic_field(name, cls):
    """x_min and x_max are the same point, so the solver's own output must agree there."""
    field = _solve_periodic(cls)
    assert np.isfinite(np.asarray(field)).all(), f"{name} produced non-finite values"
    seam = _seam(field)
    assert seam < SEAM_TOL, (
        f"{name} declares BCType.PERIODIC but returned a field with a seam of {seam:.4e} between "
        f"x_min and x_max, which are the same physical point on a periodic domain"
    )


def _fp_parametrised():
    """The FP half, which is where mass conservation is a question at all."""
    for name, cls in sorted(_declaring_solvers().items()):
        if not hasattr(cls, "solve_fp_system"):
            continue
        marks = []
        if name in MASS_NOT_CONSERVED:
            marks.append(
                pytest.mark.xfail(
                    strict=True,
                    reason=f"{name} creates or destroys mass on a periodic domain ({MASS_NOT_CONSERVED[name]})",
                )
            )
        yield pytest.param(name, cls, marks=marks, id=name)


@pytest.mark.parametrize(("name", "cls"), list(_fp_parametrised()))
def test_a_periodic_fp_solve_conserves_mass(name, cls):
    """A periodic domain has no boundary, so there is nowhere for mass to go.

    Independent of the seam: `FPFVMSolver` satisfies that one at 2.2e-16 while creating 6.5% of
    its own mass. Certifying PERIODIC support on the seam alone would pass it.
    """
    field = _solve_periodic(cls)
    x = np.linspace(0.0, 1.0, NX)
    initial = float(np.trapezoid(field[0], x))
    final = float(np.trapezoid(field[-1], x))
    assert initial > 0, f"{name} produced zero initial mass; the ratio below would be meaningless"
    drift = abs(final / initial - 1.0)
    assert drift < 1e-9, (
        f"{name} changed total mass by {drift:.4%} over a periodic solve ({initial:.6f} -> "
        f"{final:.6f}), and a periodic domain has no boundary for it to cross"
    )


def _declaring_solvers_from_source() -> set[str]:
    """The same question answered from the package's SOURCE TEXT, not from imports.

    Deliberately a second mechanism. The obvious coverage test -- compare `_parametrised()` against
    `_declaring_solvers()` -- is tautological, because both derive from `_SEARCHED`: dropping a
    module from that tuple removes it from each side and the comparison still passes. Measured:
    deleting `fp_fvm` from `_SEARCHED` left that version of this test green while the matrix
    silently stopped covering a solver. An AST walk over the files cannot be narrowed by editing
    `_SEARCHED`, which is the whole point.
    """
    import ast
    from pathlib import Path

    root = Path(__file__).resolve().parents[3] / "mfgarchon" / "alg" / "numerical"
    names: set[str] = set()
    for path in sorted(root.rglob("*.py")):
        source = path.read_text()
        if "_SUPPORTED_BC_TYPES" not in source:
            continue
        for node in ast.walk(ast.parse(source)):
            if not isinstance(node, ast.ClassDef):
                continue
            for stmt in node.body:
                targets = [stmt.target] if isinstance(stmt, ast.AnnAssign) else getattr(stmt, "targets", [])
                if not any(isinstance(t, ast.Name) and t.id == "_SUPPORTED_BC_TYPES" for t in targets):
                    continue
                if stmt.value is not None and "PERIODIC" in ast.get_source_segment(source, stmt.value):
                    names.add(node.name)
    return names


def test_the_matrix_covers_every_declaring_solver():
    """A new solver cannot declare PERIODIC and quietly sit outside this ratchet.

    Without this, the file measures whatever it happens to list, and a solver added later is
    absent rather than failing -- which reads identically to passing.
    """
    from_source = _declaring_solvers_from_source()
    parametrised = {p.id for p in _parametrised()}
    uncovered = from_source - parametrised
    assert not uncovered, (
        f"these classes declare PERIODIC in their source and are not in this matrix: "
        f"{sorted(uncovered)}. Add them, or the ratchet measures whatever it happens to list."
    )
    # The reverse direction is deliberately NOT asserted. A subclass inherits the declaration
    # without restating it -- `FPSLAdjointSolver` is the deprecated alias of `FPSLSolver` (#710)
    # and reaches a user as a PERIODIC-declaring solver while defining no `_SUPPORTED_BC_TYPES` of
    # its own. It belongs in the matrix and will never appear in an AST walk of declaration sites.


def test_the_known_broken_list_names_only_solvers_that_exist():
    """A stale entry here would silently xfail nothing and hide a regression elsewhere."""
    declaring = set(_declaring_solvers())
    stale = set(KNOWN_NOT_HONOURED) - declaring
    assert not stale, f"KNOWN_NOT_HONOURED names solvers that no longer declare PERIODIC: {sorted(stale)}"
