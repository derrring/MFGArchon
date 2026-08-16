"""Which `BCType` each solver accepts, pinned. #1975

**These tests pin a defect, not a correct state.** The Fokker-Planck reflecting wall is Robin in
`m`; no FP solver in this library accepts `BCType.ROBIN`, and the only solver that does is on the
HJB side, where the condition is Neumann. Support is inverted relative to the mathematics.

A test that asserts a defect persists is a hazard -- it can become the thing that defends the
defect. Two properties keep these from being that:

1. Each failure message says which direction the change was and what to do about it. A red result
   here is information, never "revert your change".
2. `test_the_resolver_emits_a_type_no_consumer_accepts` does not pin either side. It pins the
   *disagreement* between `FPResolver`, which emits `ROBIN`, and the solvers, which refuse it. It
   goes green only when the gap closes, from whichever end.

The physics, so the table can be read against something:

    HJB:  d_n u = 0                                  -- true Neumann
    FP :  (sigma^2/2) d_n m + m*(D_pH . n) = 0        -- Robin in m, coefficient a FIELD

Carmona-Delarue I 4.7, (4.169)/(4.170); Cirant, JMPA 103 (2015) p.1295, lines hjn)/kn).
`J.n = 0` reduces to `d_n m = 0` only when `D_pH . n = 0` -- true for p-homogeneous and congestion
Hamiltonians, false for any term linear in p. Every grid FP solver here imposes `d_n m = 0`
unconditionally.

`FPParticleSolver` is the exception and needs no ROBIN branch: Skorokhod reflection of the path
yields the condition on the density by construction. It is the oracle #1975 proposes for the fix.
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil

import pytest

from mfgarchon.geometry.boundary import BCSegment, BCType
from mfgarchon.geometry.boundary.resolution import FPResolver, MathBCType

# Measured 2026-08-16 by reading `_SUPPORTED_BC_TYPES` off every solver class. Frozen here so a
# change to any solver's accepted set is visible in one line of a diff rather than in nobody's.
_CENSUS = {
    "FP": {
        "FPFDMSolver": {"DIRICHLET", "NEUMANN", "NO_FLUX", "PERIODIC"},
        "FPFVMSolver": {"NEUMANN", "NO_FLUX", "PERIODIC"},
        "FPGFDMSolver": {"NEUMANN", "NO_FLUX"},
        "FPParticleSolver": {"DIRICHLET", "NEUMANN", "NO_FLUX", "PERIODIC", "REFLECTING"},
        "FPSLAdjointSolver": {"NEUMANN", "NO_FLUX", "PERIODIC"},
        "FPSLJacobianSolver": {"NEUMANN", "NO_FLUX", "PERIODIC"},
        "FPSLSolver": {"NEUMANN", "NO_FLUX", "PERIODIC"},
    },
    "HJB": {
        "HJBFDMSolver": {"DIRICHLET", "NEUMANN", "NO_FLUX", "PERIODIC"},
        "HJBGFDMSolver": {"DIRICHLET", "NEUMANN", "NO_FLUX", "PERIODIC", "ROBIN"},
        "HJBSemiLagrangianSolver": {"NEUMANN", "NO_FLUX", "PERIODIC"},
        "HJBWENOSolver": {"NEUMANN", "NO_FLUX", "PERIODIC"},
    },
}


def _declared_support() -> dict[str, dict[str, set[str]]]:
    """Every solver class that declares `_SUPPORTED_BC_TYPES`, by side."""
    import mfgarchon.alg.numerical.fp_solvers as fp_pkg
    import mfgarchon.alg.numerical.hjb_solvers as hjb_pkg

    found: dict[str, dict[str, set[str]]] = {"FP": {}, "HJB": {}}
    for pkg, side in ((fp_pkg, "FP"), (hjb_pkg, "HJB")):
        for mod in pkgutil.iter_modules(pkg.__path__):
            module = importlib.import_module(f"{pkg.__name__}.{mod.name}")
            for name, cls in inspect.getmembers(module, inspect.isclass):
                if cls.__module__ != module.__name__:
                    continue  # re-export, not a definition here
                declared = getattr(cls, "_SUPPORTED_BC_TYPES", None)
                if declared is not None:
                    found[side][name] = {t.name for t in declared}
    return found


# =============================================================================
# The census
# =============================================================================


def test_the_solver_population_has_not_changed():
    """A new solver, or a deleted one, must land in `_CENSUS` before the rest of this file means
    anything -- every assertion below reads the frozen table, so a solver absent from it is
    unmeasured rather than passing."""
    found = _declared_support()
    for side in ("FP", "HJB"):
        assert set(found[side]) == set(_CENSUS[side]), (
            f"{side} solvers declaring _SUPPORTED_BC_TYPES changed: "
            f"added {sorted(set(found[side]) - set(_CENSUS[side]))}, "
            f"removed {sorted(set(_CENSUS[side]) - set(found[side]))}. "
            f"Update _CENSUS, then re-read #1975 -- the table is its evidence."
        )


@pytest.mark.parametrize(
    ("side", "solver"),
    [(side, name) for side, solvers in _CENSUS.items() for name in sorted(solvers)],
)
def test_each_solver_accepts_exactly_what_the_census_records(side, solver):
    """Pins one row. Widening is the fix landing and narrowing is a capability lost; the message
    says which, because the two need opposite responses and a bare diff does not distinguish
    them."""
    got = _declared_support()[side][solver]
    want = _CENSUS[side][solver]
    if got == want:
        return
    added, removed = sorted(got - want), sorted(want - got)
    raise AssertionError(
        f"{solver} now accepts {sorted(got)}, census says {sorted(want)}.\n"
        f"  gained: {added or 'nothing'}  -- new capability; update the census and #1975.\n"
        f"  lost:   {removed or 'nothing'}  -- a wall that used to be expressible no longer is."
    )


# =============================================================================
# The gap itself -- this one is not a snapshot
# =============================================================================


def test_the_resolver_emits_a_type_no_consumer_accepts():
    """`FPResolver` translates an impermeable wall into the Robin condition that is actually true
    of it, and hands back a `MathBCType` that every FP solver refuses.

    This is the whole of #1975 in one assertion, and it pins **neither side** -- so it does not
    defend the defect. It goes green when the resolver stops emitting ROBIN (the physics would
    have to have changed, so: unlikely and worth a fight) or when some FP solver starts accepting
    it (the fix). Either way the reader is sent to #1975 rather than to a revert.
    """
    resolved = FPResolver().resolve(
        BCSegment(name="wall", bc_type=BCType.NO_FLUX, boundary="x_min"),
        {"drift": 3.2, "diffusion": 0.045},
    )
    assert resolved.math_type is MathBCType.ROBIN, (
        "FPResolver no longer emits ROBIN for an impermeable wall. The FP flux condition "
        "J.n = 0 is Robin in m whenever D_pH . n != 0, so this is either a real change of "
        "physics or a regression -- see #1975 before updating this test."
    )

    # Read live, NOT from _CENSUS. Against the frozen dict this assertion would compare a
    # constant with itself and survive the very change it exists to notice -- caught by the M1
    # mutation (FPFDMSolver gains ROBIN), which the first draft of this file passed.
    accepting = sorted(s for s, types in _declared_support()["FP"].items() if "ROBIN" in types)
    assert not accepting, (
        f"{accepting} now accept(s) ROBIN. This assertion existed to record that NO FP solver "
        f"did, while FPResolver emits exactly that -- the inversion #1975 tracks. If this is the "
        f"fix landing: close #1975, delete this assertion, and replace it with a mass-conservation "
        f"oracle against FPParticleSolver, which already imposes the correct wall."
    )


def test_the_only_robin_capable_solver_is_on_the_side_that_does_not_need_it():
    """The inversion stated as such: ROBIN is the FP condition and is supported only on HJB.

    Kept separate from the row-by-row census because a reader scanning that table sees eleven
    correct-looking rows; the inversion is a property of the table, not of any row in it.
    """
    live = _declared_support()  # live, for the reason given in the test above
    fp_robin = {s for s, t in live["FP"].items() if "ROBIN" in t}
    hjb_robin = {s for s, t in live["HJB"].items() if "ROBIN" in t}

    assert fp_robin == set(), f"FP side gained ROBIN ({sorted(fp_robin)}) -- see #1975"
    assert hjb_robin == {"HJBGFDMSolver"}, (
        f"HJB solvers accepting ROBIN changed to {sorted(hjb_robin)}. The HJB reflecting "
        "condition is Neumann, so ROBIN there serves a different purpose (#624 adjoint-"
        "consistent coupling); a change here is not the #1975 fix."
    )


def test_the_fp_assembly_reads_none_of_the_robin_coefficients():
    """The gate refusing ROBIN is honest, not an oversight: nothing downstream would read it.

    `_BOUNDARY_HANDLERS` (`fp_fdm_time_stepping.py:110`) is keyed on the *advection scheme*, and
    all four entries are `add_boundary_no_flux_entries_*`. The call site is
    `elif (is_no_flux or not is_uniform) and is_boundary`, so a mixed BC at a boundary point
    routes to the no-flux handler whatever its segments say.

    This assertion, like the one above, pins the **gap** rather than either side. It fails the
    moment the assembly starts reading `(alpha, beta, g)` -- which is the #1975 fix, and the
    reader is sent there rather than to a revert.

    Feller/Wentzell is why the fix is the coefficient vector and not another named branch: the
    canonical boundary operator is one condition with sign-constrained additive coefficients, and
    the named types are degenerate corners of it. Adding a ROBIN branch samples one more point of
    the same continuum.
    """
    import numpy as np

    from mfgarchon.alg.numerical.fp_solvers.fp_fdm_time_stepping import solve_timestep_full_nd
    from mfgarchon.geometry import TensorProductGrid
    from mfgarchon.geometry.boundary import BoundaryConditions, no_flux_bc

    nx = 21
    grid = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[nx], boundary_conditions=no_flux_bc(dimension=1))
    h = grid.get_grid_spacing()[0]
    x = np.linspace(0.0, 1.0, nx)

    # Off-centre, so mass actually reaches one wall before the other and the wall term matters.
    m0 = np.exp(-50 * (x - 0.35) ** 2)
    m0 /= m0.sum() * h
    u0 = -3.2 * x  # constant wall-normal drift, the case where J.n = 0 is NOT d_n m = 0

    def mixed(bc_type, **kw):
        return BoundaryConditions(
            segments=[
                BCSegment(name="lo", bc_type=bc_type, boundary="x_min", **kw),
                BCSegment(name="hi", bc_type=bc_type, boundary="x_max", **kw),
            ],
            dimension=1,
        )

    def step(bc):
        return solve_timestep_full_nd(
            M_current=m0,
            U_current=u0,
            problem=object(),
            dt=1e-3,
            sigma=0.3,
            coupling_coefficient=1.0,
            spacing=(h,),
            grid=grid,
            ndim=1,
            shape=(nx,),
            boundary_conditions=bc,
            advection_scheme="divergence_upwind",
        )

    reference = step(mixed(BCType.NO_FLUX))
    # Two wildly different alphas: if either coefficient were read, one of these would move.
    for alpha in (3.2, 999.0):
        got = step(mixed(BCType.ROBIN, alpha=alpha, beta=-0.045, value=0.0))
        assert np.array_equal(got, reference), (
            f"a ROBIN wall with alpha={alpha} no longer assembles byte-identically to no-flux. "
            "If the FP assembly now reads (alpha, beta, g), that is the #1975 fix: close it, "
            "delete this assertion, and replace it with a mass-conservation oracle against "
            "FPParticleSolver, which already imposes the correct wall by Skorokhod reflection."
        )


# =============================================================================
# The refusal is load-bearing -- it must stay loud
# =============================================================================


def test_an_unsupported_bc_is_refused_rather_than_silently_reinterpreted():
    """The census only means something because the declared set is enforced.

    If `_validate_bc_support` stopped raising, every row above would still pass while the solvers
    quietly applied a different wall -- which is the failure mode #1456 exists to prevent, and it
    would make this whole file decorative.
    """
    from mfgarchon.alg.base_solver import BaseNumericalSolver
    from mfgarchon.alg.numerical.hjb_solvers.hjb_fdm import HJBFDMSolver

    assert "ROBIN" not in _CENSUS["HJB"]["HJBFDMSolver"], "premise of this test changed"
    assert hasattr(BaseNumericalSolver, "_validate_bc_support"), (
        "the gate that turns _SUPPORTED_BC_TYPES from documentation into behaviour is gone; "
        "every assertion in this file is decorative without it"
    )
    assert "_validate_bc_support" in inspect.getsource(HJBFDMSolver.__init__), (
        "HJBFDMSolver no longer calls _validate_bc_support at construction, so its declared "
        "support is no longer enforced"
    )
