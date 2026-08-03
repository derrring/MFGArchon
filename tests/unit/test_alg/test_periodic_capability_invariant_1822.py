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


# The datum, and why it is phase-shifted rather than the obvious sin/cos.
#
# `cos(2 pi x)` on linspace(0, 1, N) is mirror-symmetric about x = 0.5, and under a mirror-symmetric
# input a NO_FLUX solve and a PERIODIC solve produce the same field -- so the seam invariant cannot
# tell them apart and any solver passes it for free. Measured: with the symmetric datum
# `FPFVMSolver` scored 2.2e-16 and `FPGFDMSolver` 2.2e-11, and those were this file's only two
# non-xfail rows, i.e. its only positive controls. With the phase shifts below they score 1.79e-01
# and an outright raise. The certification was an artefact of the test data.
#
# Requirements on the datum: exactly periodic (seam 0), strictly positive for the density, and
# neither symmetric nor antisymmetric about the midpoint.
def _U(z):
    z = np.asarray(z)
    return np.sin(2 * np.pi * z + 0.7) + 0.4 * np.cos(4 * np.pi * z)


def _M(z):
    z = np.asarray(z)
    return 1.0 + 0.5 * np.cos(2 * np.pi * z + 1.1)


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

# Measured on the datum above, 1D, Nx=21, sigma=0.3, T=0.5, Nt=10. Each entry pins the FAILURE
# MODE as well as the failure: `raises=` means a solver that starts crashing, or crashing
# differently, is caught rather than xfailing identically to one that merely returns a bad seam.
#
#   HJBFDMSolver    7.42e-01     FPParticleSolver    ~5e-01 (UNSEEDED: varies ~25% per run)
#   HJBWENOSolver   2.64e-01     FPSLJacobianSolver  1.58e+00
#   FPFVMSolver     1.79e-01     FPFDMSolver         1.29e-01
#   HJBGFDMSolver   raises NotImplementedError for PERIODIC
#   FPGFDMSolver    raises ValueError (density goes invalid mid-solve)
#
# Passing: HJBSemiLagrangianSolver (0.0) and FPSLSolver / FPSLAdjointSolver (4.4e-16) -- the three
# repaired in #1824, and the only genuine positive controls this file has ever had.
KNOWN_NOT_HONOURED = {
    "HJBWENOSolver": ("#1822", AssertionError),
    "HJBFDMSolver": ("#1822", AssertionError),
    "HJBGFDMSolver": ("#1822 declares PERIODIC and raises for it", NotImplementedError),
    "FPFDMSolver": ("#1822", AssertionError),
    "FPFVMSolver": ("#1822", AssertionError),
    "FPGFDMSolver": ("#1822 density goes invalid mid-solve", ValueError),
    "FPParticleSolver": ("#1822", AssertionError),
    "FPSLJacobianSolver": ("#1822 deprecated, retirement in #1756", AssertionError),
}

# Mass is the second invariant and it is INDEPENDENT of the seam: a periodic domain has no
# boundary, so a periodic FP solve conserves mass exactly. Measured drift on the datum above:
#
#   FPSLSolver / FPSLAdjointSolver  13.42%    FPFVMSolver      5.76%
#   FPFDMSolver                      5.56%    FPParticleSolver ~5% (UNSEEDED)
#   FPGFDMSolver                     raises
#
# FPSLSolver is the case that forces the split: it satisfies the seam at 4.4e-16 and creates 13%
# of its own mass. Traced in #1820 -- advection conserves exactly at zero drift and the
# Crank-Nicolson step conserves exactly on an identified field, while one splat step creates 1.14%
# at drift 0.3 and 3.81% at drift 1.0. The defect is in splat under drift.
#
# `FPSLJacobianSolver` conserves to 0.0000% and is NOT listed -- that is renormalisation by fiat,
# not conservation (RFC #1456 class (b), #1429 S0-11). A passing row means the number is right,
# not that the mechanism is.
MASS_NOT_CONSERVED = {
    "FPFDMSolver": ("#1822", AssertionError),
    "FPFVMSolver": ("#1822", AssertionError),
    "FPGFDMSolver": ("#1822 density goes invalid mid-solve", ValueError),
    "FPParticleSolver": ("#1822", AssertionError),
    "FPSLSolver": ("#1820 splat under drift", AssertionError),
    "FPSLAdjointSolver": ("#1820 splat under drift", AssertionError),
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
            if declared is None:
                # The public name is what BaseMFGSolver._validate_bc_support actually reads.
                declared = getattr(cls, "supported_bc_types", None)
                if isinstance(declared, property):
                    declared = None
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
            m_initial=_M,
            u_terminal=_U,
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
    u_periodic = _U(x)
    m_periodic = _M(x)
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
            issue, exc = KNOWN_NOT_HONOURED[name]
            marks.append(
                pytest.mark.xfail(
                    strict=True,
                    raises=exc,
                    reason=f"{name} declares PERIODIC and does not honour it ({issue})",
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
            issue, exc = MASS_NOT_CONSERVED[name]
            marks.append(
                pytest.mark.xfail(
                    strict=True,
                    raises=exc,
                    reason=f"{name} does not conserve mass on a periodic domain ({issue})",
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
        if "_SUPPORTED_BC_TYPES" not in source and "supported_bc_types" not in source:
            continue
        for node in ast.walk(ast.parse(source)):
            if not isinstance(node, ast.ClassDef):
                continue
            for stmt in node.body:
                targets = [stmt.target] if isinstance(stmt, ast.AnnAssign) else getattr(stmt, "targets", [])
                # BOTH names. `_validate_bc_support` (base_solver.py) reads the PUBLIC
                # `supported_bc_types`, so a class declaring only that one advertises PERIODIC
                # through the real gate while being invisible to a private-name-only scan.
                if not any(
                    isinstance(t, ast.Name) and t.id in ("_SUPPORTED_BC_TYPES", "supported_bc_types") for t in targets
                ):
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
