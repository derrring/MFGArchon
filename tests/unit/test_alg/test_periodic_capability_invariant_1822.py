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
from mfgarchon.geometry.boundary import dirichlet_bc, neumann_bc, no_flux_bc, periodic_bc
from mfgarchon.geometry.boundary.invariants import bc_residual, mass_drift, seam
from mfgarchon.geometry.boundary.types import BCType


def _residual_of(solved, bc_type) -> float:
    """Adapter: `_solve_with_bc` returns (field, kind, x); the owner takes them separately."""
    field, kind, x = solved
    return bc_residual(field, bc_type, x, kind)


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
    # HJBWENOSolver is GONE from this roster, not silenced: its ghost buffer read the last g
    # interior entries, which on an endpoint-inclusive grid includes the node that IS x[0], so
    # every periodic stencil sat one cell off. Fixed in applicator_fdm._periodic_ghost_slices;
    # seam 2.63e-01 -> exactly 0. The ghost fill itself is pinned against the analytic continuation
    # in tests/unit/test_geometry/test_periodic_ghost_fill_1822.py.
    # FPFDMSolver and FPFVMSolver are GONE from this roster for one shared reason, the same one
    # that removed HJBWENOSolver: both wrapped cell N-1 to cell 0, which on an endpoint-inclusive
    # grid treats the repeated endpoint as its own cell and solves on a torus one cell too long.
    # The seam was the visible half; the invisible half was worse. Against the analytic heat
    # kernel at Nx=21 they were 8.7e-02 of relative error and are now 9.3e-03, converging
    # 9.3e-03 -> 3.8e-03 -> 2.4e-03 over 21/41/81 -- and under the SYMMETRIC datum that error sat
    # behind a seam of 2e-15, so this file's own invariant could not have found it. Fixed in
    # conditions.periodic_axis_span (one owner, four scheme modules and the FVM wrap face route
    # through it) plus the repeated-endpoint constraint row in fp_fdm_time_stepping. Pinned
    # against the heat kernel and a rigid translation, not against the seam, in
    # tests/unit/test_alg/test_periodic_torus_oracle_1822.py.
    # Reason is a pointer, not a diagnosis: the evidence for what produces this seam (the seam
    # enters at the stalled timestep and decays backward, and the stall is periodic-specific) is
    # in #1834, and nothing in THIS change tests it.
    "HJBFDMSolver": ("#1834", AssertionError),
    "FPParticleSolver": ("#1822", AssertionError),
    "FPSLJacobianSolver": ("#1822 deprecated, retirement in #1756", AssertionError),
}

# Mass is the second invariant and it is INDEPENDENT of the seam. But the assertion is
# CONVERGENCE, not exact conservation, and getting that wrong is what the first version of this
# file did: it asserted `drift < 1e-9` and reported six solvers as failing to honour PERIODIC.
#
# Measured drift against refinement (Nx=21/41/81, Nt scaled with it):
#
#   FPFDMSolver       5.56e-02  2.87e-02  1.46e-02
#   FPFVMSolver       5.77e-02  2.93e-02  1.47e-02
#   FPSLSolver        1.34e-01  4.91e-02  2.62e-02
#   FPParticleSolver  5.63e-02  3.38e-02  1.64e-02
#
# Every one halves per refinement. That is O(h) discretisation error in a non-conservative scheme,
# not a capability defect -- these solvers do not claim conservation by construction, and an
# absolute tolerance on a convergent quantity is a tolerance chosen to make a point.
#
# `FPSLJacobianSolver` reports 0.0000% at every resolution, which is the opposite failure: exact by
# renormalisation rather than by conservation (RFC #1456 class (b), #1429 S0-11). A convergence
# oracle cannot see it; it is listed under its own issue rather than certified here.
MASS_NON_CONVERGENT: dict[str, tuple[str, type[Exception]]] = {
    # Empty, and that is the honest state: every FP solver still declaring PERIODIC has a periodic
    # mass error that halves under refinement. The one entry that used to sit here, FPGFDMSolver,
    # never produced a number to converge -- it has since stopped declaring PERIODIC (#1822), so
    # this file no longer asks it a question about periodicity.
}


def _declared_bc_types(cls) -> frozenset | None:
    """What a class advertises, by the name the real gate reads.

    `BaseMFGSolver._validate_bc_support` reads the PUBLIC `supported_bc_types`, so a class
    declaring only that one advertises through the real gate while being invisible to a
    private-name-only scan.
    """
    declared = getattr(cls, "_SUPPORTED_BC_TYPES", None)
    if declared is None:
        declared = getattr(cls, "supported_bc_types", None)
        if isinstance(declared, property):
            declared = None
    return declared or None


def _solvers_declaring_any_bc() -> dict[str, type]:
    """Every class in the searched modules that advertises ANY boundary condition.

    This is what the declared-surface half of the file (#1574) must iterate. Keying it on
    PERIODIC instead was a filter that looked like no filter: every class in `_SEARCHED`
    declared PERIODIC, so the two sets were identical and the narrowing was invisible. The
    first solver to stop declaring it fell out of the surface matrix entirely, taking its
    DIRICHLET and NEUMANN rows with it -- measured when the GFDM pair undeclared PERIODIC in
    #1822: five rows vanished silently, three of them recording live defects
    (FPGFDMSolver-NEUMANN, FPGFDMSolver-NO_FLUX, HJBGFDMSolver-DIRICHLET).
    """
    found: dict[str, type] = {}
    for module_name in _SEARCHED:
        module = importlib.import_module(module_name)
        for name, cls in inspect.getmembers(module, inspect.isclass):
            if cls.__module__ != module_name:
                continue
            if _declared_bc_types(cls):
                found[name] = cls
    return found


def _declaring_solvers() -> dict[str, type]:
    """Every searched class whose declaration contains PERIODIC.

    Narrower than `_solvers_declaring_any_bc` on purpose: the seam and mass invariants are
    statements ABOUT periodicity, so a solver that does not claim it has nothing to answer for
    there. The surface matrix below is the one that must not be scoped this way.
    """
    return {n: c for n, c in _solvers_declaring_any_bc().items() if BCType.PERIODIC in _declared_bc_types(c)}


def _periodic_problem(nx: int = NX, nt: int = NT) -> MFGProblem:
    return MFGProblem(
        geometry=TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[nx], boundary_conditions=periodic_bc(dimension=1)),
        T=0.5,
        Nt=nt,
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


def _problem_with_bc(bc, nx: int, nt: int) -> MFGProblem:
    """The same fixture as `_periodic_problem`, with the boundary condition as a parameter."""
    return MFGProblem(
        geometry=TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[nx], boundary_conditions=bc),
        T=0.5,
        Nt=nt,
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


def _solve_periodic(cls: type, nx: int = NX, nt: int = NT) -> np.ndarray:
    """Run one solve from exactly periodic data and return the field it produced."""
    x = np.linspace(0.0, 1.0, nx)
    u_periodic = _U(x)
    m_periodic = _M(x)
    assert seam(u_periodic) < 1e-15, "the u input itself must be periodic, or the output tells us nothing"
    assert seam(m_periodic) < 1e-15, "the m input itself must be periodic, or the output tells us nothing"

    problem = _periodic_problem(nx=nx, nt=nt)
    kwargs = {}
    if "collocation_points" in inspect.signature(cls.__init__).parameters:
        kwargs["collocation_points"] = x.reshape(-1, 1)
    solver = cls(problem, **kwargs)

    if hasattr(solver, "solve_hjb_system"):
        return solver.solve_hjb_system(np.tile(m_periodic, (nt + 1, 1)), u_periodic, np.zeros((nt + 1, nx)))
    return solver.solve_fp_system(m_periodic, np.tile(u_periodic, (nt + 1, 1)))


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
    measured = seam(field)
    assert measured < SEAM_TOL, (
        f"{name} declares BCType.PERIODIC but returned a field with a seam of {measured:.4e} between "
        f"x_min and x_max, which are the same physical point on a periodic domain"
    )


def _fp_parametrised():
    """The FP half, which is where mass conservation is a question at all."""
    for name, cls in sorted(_declaring_solvers().items()):
        if not hasattr(cls, "solve_fp_system"):
            continue
        marks = []
        if name in MASS_NON_CONVERGENT:
            issue, exc = MASS_NON_CONVERGENT[name]
            marks.append(
                pytest.mark.xfail(
                    strict=True,
                    raises=exc,
                    reason=f"{name} does not conserve mass on a periodic domain ({issue})",
                )
            )
        yield pytest.param(name, cls, marks=marks, id=name)


@pytest.mark.parametrize(("name", "cls"), list(_fp_parametrised()))
def test_a_periodic_fp_solve_conserves_mass_in_the_limit(name, cls):
    """A periodic domain has no boundary, so mass error must vanish as the grid refines.

    CONVERGENCE, not exact conservation. None of these solvers claims conservation by
    construction, so an absolute tolerance would flag O(h) discretisation error as a capability
    defect -- which an earlier version of this file did, for six solvers at once. What is a defect
    is error that stops converging: a scheme leaking a fixed fraction per step is not going to be
    saved by a finer grid.
    """
    drifts = []
    for nx, nt in ((21, 10), (41, 20)):
        field = _solve_periodic(cls, nx=nx, nt=nt)
        drifts.append(mass_drift(field, np.linspace(0.0, 1.0, nx)))

    if drifts[0] < 1e-12:
        # Already exact at the coarse grid. Either genuinely conservative or renormalised; this
        # oracle cannot tell those apart, and says so rather than certifying.
        return
    assert drifts[1] < drifts[0] / 1.5, (
        f"{name} periodic mass error did not converge under refinement: {drifts[0]:.3e} at Nx=21, "
        f"{drifts[1]:.3e} at Nx=41. A convergent scheme roughly halves here"
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


# ---------------------------------------------------------------------------
# The rest of the declared surface (#1574). PERIODIC above is one column of it.
# ---------------------------------------------------------------------------

# Every BC type each solver declares, driven through one solve, against the invariant that type
# means. Measured surface: 39 (solver, BC) pairs -- NEUMANN 11, PERIODIC 11, NO_FLUX 11,
# DIRICHLET 4, ROBIN 1, REFLECTING 1.
#
# The verdict grid, which is what makes this a capability check rather than an accuracy check:
#
#   declared + invariant holds      honest
#   declared + invariant violated   advertises and returns wrong numbers   <- the #1574 class
#   declared + path refuses         contradiction: declares, then raises   <- also the class
#
# Oracles, and why each is absolute or convergent:
#
#   DIRICHLET   u (or m) at the wall equals g. Zero in exact arithmetic, so exactness is the
#               strong form -- but a particle method approximates the wall rather than pinning it
#               (FPParticleSolver: 4.94e-01 at Nx=21, 3.27e-06 at Nx=41), so convergence to zero
#               is accepted and exactness is recorded rather than required.
#   NEUMANN /   HJB: the one-sided du/dn at each wall goes to zero. HJBFDMSolver is exact; SL,
#   NO_FLUX     WENO and GFDM converge at ratio ~1.55 over 21->41.
#               FP: mass is conserved. Convergent, NOT exact -- see the periodic mass note above
#               for why an absolute tolerance here measures the grid instead of the solver.
#
# ROBIN and REFLECTING are declared once each and have no constructor in `mfgarchon.geometry.
# boundary` reachable from this fixture, so they are reported as uncovered rather than passed.
BC_FACTORIES = {
    BCType.NO_FLUX: lambda: no_flux_bc(dimension=1),
    BCType.NEUMANN: lambda: neumann_bc(dimension=1),
    BCType.DIRICHLET: lambda: dirichlet_bc(dimension=1, value=0.0),
    BCType.PERIODIC: lambda: periodic_bc(dimension=1),
}

# (solver, BC) pairs that do not honour what they declare, measured on this fixture.
# Measured seam under refinement (Nx=21/41/81) for the PERIODIC column, which is why this list is
# shorter than the seam column's: a residual that CONVERGES is a scheme consistent with the BC
# without identifying the coincident nodes, and that is a weaker claim than the seam test above
# makes, not a violated one.
#
#   converge:   HJBFDMSolver 7.42e-01 6.51e-01 4.72e-01
#   exact:      HJBSemiLagrangianSolver 0, FPSLSolver / FPSLAdjointSolver 4.4e-16,
#               HJBWENOSolver (was 2.63e-01 2.08e-01 1.34e-01 before its ghost fill was fixed),
#               FPFVMSolver and FPFDMSolver (were 1.79e-01 9.00e-02 4.25e-02 and the same shape,
#               before the wrap was put on the right torus -- #1822)
#   NOT:        FPParticleSolver 5.64e-01 2.08e-01 2.63e-01 (up at 81)
#               FPSLJacobianSolver 1.58e+00 7.64e-03 1.16e-02 (up at 81)
# Unseeded solvers cannot be classified here at all: measured over three trials, FPParticleSolver
# returned monotone=False, False, True on the identical configuration. Marking it xfail asserts a
# failure it does not reliably have; marking it pass asserts the opposite. It is skipped, named,
# and the seeding is the fix.
STOCHASTIC_UNSEEDED = {
    "FPParticleSolver": "no seed parameter; seam varies ~25% run to run",
}

SURFACE_NOT_HONOURED = {
    ("HJBGFDMSolver", "DIRICHLET"): ("#1822 declares DIRICHLET, solve returns NaN", AssertionError),
    ("FPGFDMSolver", "NEUMANN"): ("#1822 density goes invalid mid-solve", ValueError),
    ("FPGFDMSolver", "NO_FLUX"): ("#1822 density goes invalid mid-solve", ValueError),
    # Mass converges 5.62e-02 -> 3.07e-02 and then the Nx=81 solve raises, so the third point
    # that would settle the trend does not exist. Listed under the raise, not under the trend.
    ("FPSLSolver", "NEUMANN"): ("#1822 Nx=81 solve raises", ValueError),
    ("FPSLSolver", "NO_FLUX"): ("#1822 Nx=81 solve raises", ValueError),
    ("FPSLAdjointSolver", "NEUMANN"): ("#1822 Nx=81 solve raises", ValueError),
    ("FPSLAdjointSolver", "NO_FLUX"): ("#1822 Nx=81 solve raises", ValueError),
    ("FPSLJacobianSolver", "PERIODIC"): ("#1822 deprecated, retirement in #1756", AssertionError),
}


def _solve_with_bc(cls, bc_type, nx, nt):
    x = np.linspace(0.0, 1.0, nx)
    problem = _problem_with_bc(BC_FACTORIES[bc_type](), nx, nt)
    kwargs = {}
    if "collocation_points" in inspect.signature(cls.__init__).parameters:
        kwargs["collocation_points"] = x.reshape(-1, 1)
    solver = cls(problem, **kwargs)
    if hasattr(solver, "solve_hjb_system"):
        return solver.solve_hjb_system(np.tile(_M(x), (nt + 1, 1)), _U(x), np.zeros((nt + 1, nx))), "HJB", x
    return solver.solve_fp_system(_M(x), np.tile(_U(x), (nt + 1, 1))), "FP", x


def _surface_params():
    # The WIDER set: a declared BC must be measured whether or not the solver also claims PERIODIC.
    for name, cls in sorted(_solvers_declaring_any_bc().items()):
        declared = getattr(cls, "_SUPPORTED_BC_TYPES", None) or getattr(cls, "supported_bc_types", None) or ()
        for bc_type in sorted(declared, key=lambda t: t.name):
            if bc_type not in BC_FACTORIES:
                continue  # ROBIN / REFLECTING: reported by the coverage test below, not passed
            marks = []
            key = (name, bc_type.name)
            if key in SURFACE_NOT_HONOURED:
                issue, exc = SURFACE_NOT_HONOURED[key]
                marks.append(
                    pytest.mark.xfail(strict=True, raises=exc, reason=f"{name} declares {bc_type.name}: {issue}")
                )
            yield pytest.param(name, cls, bc_type, marks=marks, id=f"{name}-{bc_type.name}")


@pytest.mark.parametrize(("name", "cls", "bc_type"), list(_surface_params()))
def test_a_declared_bc_type_is_honoured(name, cls, bc_type):
    """Declaring a BC type is a claim. This is the measurement that makes it cost something."""
    if name in STOCHASTIC_UNSEEDED:
        pytest.skip(f"{name} is unseeded ({STOCHASTIC_UNSEEDED[name]}); any verdict here is a draw")

    residuals = []
    for nx, nt in ((21, 10), (41, 20), (81, 40)):
        r = _residual_of(_solve_with_bc(cls, bc_type, nx, nt), bc_type)
        assert np.isfinite(r), f"{name} declares {bc_type.name} and the solve produced non-finite values"
        residuals.append(r)
        if len(residuals) == 1 and r < 1e-12:
            return  # exact at the coarse grid: the strong form, no refinement needed

    # THREE points, monotone. Two are not a trend: on 21->41 alone FPSLJacobianSolver improves
    # 1.58e+00 -> 7.64e-03 and then gets WORSE at 81 (1.16e-02), and a two-point check certifies
    # it. HJBFDMSolver is the opposite case -- 7.42e-01, 6.51e-01, 4.72e-01 is genuine, slow
    # convergence that a ratio threshold tuned for the fast cases would have failed.
    trend = f"{residuals[0]:.3e}, {residuals[1]:.3e}, {residuals[2]:.3e} at Nx=21/41/81"
    assert residuals[1] < residuals[0], (
        f"{name} declares {bc_type.name} but its boundary residual grew from Nx=21 to 41: {trend}"
    )
    assert residuals[2] < residuals[1], (
        f"{name} declares {bc_type.name} but its boundary residual grew from Nx=41 to 81: {trend}"
    )


def test_every_declared_pair_is_either_measured_or_named_uncovered():
    """A declared BC with no oracle must be visible as uncovered, not absent.

    ROBIN and REFLECTING are each declared once and have no fixture here. Absent, they would read
    as covered; named, they are a gap someone can close.
    """
    uncovered = {
        (name, t.name)
        for name, cls in _solvers_declaring_any_bc().items()
        for t in (getattr(cls, "_SUPPORTED_BC_TYPES", None) or ())
        if t not in BC_FACTORIES
    }
    assert uncovered == {("HJBGFDMSolver", "ROBIN"), ("FPParticleSolver", "REFLECTING")}, (
        f"the set of declared-but-unmeasured (solver, BC) pairs changed: {sorted(uncovered)}"
    )


def test_the_surface_matrix_measures_every_declared_pair_it_has_a_fixture_for():
    """A solver cannot fall out of the surface matrix by narrowing what it declares.

    This is the guard for a failure this file actually had. `_surface_params` used to iterate
    `_declaring_solvers()` -- the PERIODIC-declaring set -- which looked like no filter at all,
    because every class in `_SEARCHED` declared PERIODIC. The moment the GFDM pair stopped
    declaring it (#1822, since neither ever honoured it), five rows vanished: three of them
    recording live defects (FPGFDMSolver-NEUMANN, FPGFDMSolver-NO_FLUX, HJBGFDMSolver-DIRICHLET).

    Measured, with the matrix keyed back the old way: 0 GFDM rows collected and the file still
    **green** at 34 passed / 10 xfailed, versus 5 rows and 36 / 13 now. A suite that stays green
    while it quietly stops measuring things is the thing this ratchet exists to prevent, so the
    coverage is asserted rather than assumed.
    """
    expected = {
        (name, t.name)
        for name, cls in _solvers_declaring_any_bc().items()
        for t in (_declared_bc_types(cls) or ())
        if t in BC_FACTORIES
    }
    parametrised = {tuple(p.id.split("-", 1)) for p in _surface_params()}
    missing = expected - parametrised
    assert not missing, (
        f"these declared (solver, BC) pairs have a fixture and are NOT measured: {sorted(missing)}. "
        f"A pair that is absent reads exactly like a pair that passed."
    )
