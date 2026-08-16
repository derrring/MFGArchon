"""Which wall each FP path actually imposes, and which solvers are outside the capability gate.

This file replaces one that got the physics backwards, so the reasoning is written down.

**The reflecting wall is `J.n = 0`, not `d_n m = 0`.** They coincide only when the drift is
tangential at the wall. The conservative schemes impose the first **structurally** -- by zeroing
the total face flux, advective plus diffusive -- with no BC-type branch naming it; the
`gradient_*` family imposes the second and leaks. `FPParticleSolver` gets `J.n = 0` from Skorokhod
reflection of the path, also without naming it.

That is why the first version of this file was wrong. It read "no BC-type dispatch at the
boundary" as "imposes `d_n m = 0`", measured `ROBIN(alpha=3.2) == no_flux` byte-identically, and
called that a defect. With `u = -3.2x`, coupling 1, sigma 0.3, `FPResolver` emits
`ROBIN(alpha=+3.2, beta=-0.045)` -- **the same numbers** -- so that case *is* the physical wall the
default scheme already imposes and byte-identity there is the correct answer. Only a wall that is
genuinely different (`alpha != v_n`, or `g != 0`) shows a real gap.

So the load-bearing test here is `test_the_conservative_schemes_conserve_mass_at_a_drifted_wall`:
an external oracle, computed independently of any scheme, that says which condition is actually
imposed. A census keyed on declarations cannot say that -- correctness here is structural, not
declared -- and its absence is what let the wrong reading stand.

Refs #1975, #1977, #1979.
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil

import pytest

import numpy as np

from mfgarchon.alg.base_solver import BaseNumericalSolver
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import BCSegment, BCType, BoundaryConditions, no_flux_bc
from mfgarchon.geometry.boundary.resolution import FPResolver, MathBCType

# Measured 2026-08-17 by walking `mfgarchon.alg` with `walk_packages` and keeping concrete
# `BaseNumericalSolver` subclasses -- a predicate INDEPENDENT of whether a class declares
# `_SUPPORTED_BC_TYPES`. The previous version discovered its population by reading that attribute
# inside two non-recursed directories, so it could not report its own blind spot: it missed 5 FP
# and 5 HJB solvers, `FPFEMSolver` among them.
_GATED = {
    "FPFDMSolver": {"DIRICHLET", "NEUMANN", "NO_FLUX", "PERIODIC"},
    "FPFVMSolver": {"NEUMANN", "NO_FLUX", "PERIODIC"},
    "FPGFDMSolver": {"NEUMANN", "NO_FLUX"},
    "FPParticleSolver": {"DIRICHLET", "NEUMANN", "NO_FLUX", "PERIODIC", "REFLECTING"},
    "FPSLAdjointSolver": {"NEUMANN", "NO_FLUX", "PERIODIC"},
    "FPSLJacobianSolver": {"NEUMANN", "NO_FLUX", "PERIODIC"},
    "FPSLSolver": {"NEUMANN", "NO_FLUX", "PERIODIC"},
    "HJBFDMSolver": {"DIRICHLET", "NEUMANN", "NO_FLUX", "PERIODIC"},
    "HJBGFDMSolver": {"DIRICHLET", "NEUMANN", "NO_FLUX", "PERIODIC", "ROBIN"},
    "HJBSemiLagrangianSolver": {"NEUMANN", "NO_FLUX", "PERIODIC"},
    "HJBWENOSolver": {"NEUMANN", "NO_FLUX", "PERIODIC"},
}

# Recorded as a POPULATION, not as an absence. Half the solvers declare nothing, so
# `_validate_bc_support` no-ops on them (`base_solver.py:282`) -- including `FPFEMSolver`, the one
# FP solver that implements a general Robin. #1977.
_UNGATED = {
    "FPFEMSolver",
    "FPNetworkSolver",
    "HJBFEMSolver",
    "MeshlessGalerkinFPSolver",
    "MeshlessGalerkinHJBSolver",
    "NetworkFPSolver",
    "NetworkHJBSolver",
    "NetworkPolicyIterationHJBSolver",
    "PenaltyHJBSolver",
    "WeakFormFPSolver",
    "WeakFormHJBSolver",
}

# Outside even a subclass predicate: plain classes that read BC segments and write Dirichlet
# values and Neumann normal-gradient rows. No population predicate over `BaseNumericalSolver`
# reaches them, which is why they are named rather than discovered.
_NOT_EVEN_SUBCLASSES = {
    "HJBHowardSolver": "mfgarchon.alg.numerical.hjb_solvers.hjb_howard",
    "ImplicitHeatSolver": "mfgarchon.alg.numerical.pde_solvers.implicit_heat",
}


def _solver_population() -> dict[str, bool]:
    """Every concrete `BaseNumericalSolver` subclass -> whether it declares a BC support set.

    `walk_packages`, not `iter_modules`: the latter does not recurse, and `alg/numerical/fem/`
    is one directory below the two the old census walked.
    """
    import mfgarchon.alg as alg_pkg

    found: dict[str, bool] = {}
    for mod in pkgutil.walk_packages(alg_pkg.__path__, prefix="mfgarchon.alg."):
        module = importlib.import_module(mod.name)
        for name, cls in inspect.getmembers(module, inspect.isclass):
            if cls.__module__ != module.__name__ or not issubclass(cls, BaseNumericalSolver):
                continue
            if inspect.isabstract(cls) or name.startswith("Base"):
                continue
            found[name] = getattr(cls, "_SUPPORTED_BC_TYPES", None) is not None
    return found


def _declared(name: str) -> set[str]:
    for mod in pkgutil.walk_packages(importlib.import_module("mfgarchon.alg").__path__, prefix="mfgarchon.alg."):
        module = importlib.import_module(mod.name)
        cls = getattr(module, name, None)
        if cls is not None and inspect.isclass(cls) and cls.__module__ == module.__name__:
            declared = getattr(cls, "_SUPPORTED_BC_TYPES", None)
            if declared is not None:
                return {t.name for t in declared}
    raise AssertionError(f"{name} not found, or declares nothing")


# =============================================================================
# The external oracle -- which wall is actually imposed
# =============================================================================

_SIGMA, _DRIFT, _NX, _STEPS = 0.3, 3.2, 81, 200


def _run(scheme: str) -> tuple[float, float]:
    """One drifted-wall run. Returns (mass drift in %, d_n m at the high wall).

    `u = -drift*x` gives a constant wall-normal velocity, the case where `J.n = 0` and
    `d_n m = 0` are DIFFERENT conditions. A scheme imposing `J.n = 0` conserves mass and has
    `d_n m != 0`; one imposing `d_n m = 0` leaks.
    """
    from mfgarchon.alg.numerical.fp_solvers.fp_fdm_time_stepping import solve_timestep_full_nd

    grid = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[_NX], boundary_conditions=no_flux_bc(dimension=1))
    h = grid.get_grid_spacing()[0]
    x = np.linspace(0.0, 1.0, _NX)
    m = np.exp(-50 * (x - 0.5) ** 2)
    m /= m.sum() * h
    m0 = m.sum() * h

    for _ in range(_STEPS):
        m = solve_timestep_full_nd(
            M_current=m,
            U_current=-_DRIFT * x,
            problem=object(),
            dt=1e-3,
            sigma=_SIGMA,
            coupling_coefficient=1.0,
            spacing=(h,),
            grid=grid,
            ndim=1,
            shape=(_NX,),
            boundary_conditions=no_flux_bc(dimension=1),
            advection_scheme=scheme,
        )
    return 100.0 * (m.sum() * h - m0) / m0, (m[-1] - m[-2]) / h


@pytest.mark.parametrize("scheme", ["divergence_upwind", "divergence_centered"])
def test_the_conservative_schemes_conserve_mass_at_a_drifted_wall(scheme):
    """`J.n = 0`, imposed structurally by zeroing the total face flux -- no BC-type branch.

    **This is the test whose absence let #1975 be filed on a false premise.** It is an external
    oracle: mass conservation is a law of the equation, computed here without reference to any
    scheme's internals, so it cannot go tautological the way a declaration census can.

    `divergence_upwind` is the default. Measured: -0.0000% and -0.0000%.
    """
    drift_pct, dmdx = _run(scheme)
    assert abs(drift_pct) < 1e-6, f"{scheme} leaked {drift_pct:.4f}% at a wall with normal drift"
    assert abs(dmdx) > 1.0, (
        f"{scheme} has d_n m = {dmdx:.4g} at the wall. J.n = 0 requires d_n m = (v/D)*m, which is "
        "large here; a near-zero gradient means the wall became d_n m = 0."
    )


@pytest.mark.parametrize("scheme", ["gradient_upwind", "gradient_centered"])
def test_the_gradient_schemes_impose_a_zero_gradient_wall_and_leak(scheme):
    """The counterpart, pinned so the distinction cannot quietly collapse in either direction.

    These impose `d_n m = 0`, which is the wrong condition when the drift is not tangential, and
    they are documented non-conservative (#1075). Measured: -78.05% and -75.47%.
    """
    drift_pct, _ = _run(scheme)
    assert drift_pct < -10.0, (
        f"{scheme} conserved mass ({drift_pct:.4f}%) at a drifted wall. If it now imposes "
        "J.n = 0 that is a fix worth recording -- see #1075 and #1975 before updating this."
    )


def test_the_two_families_disagree_by_a_large_margin():
    """A control on the pair above: if the fixture stopped driving mass into the wall, both
    families would conserve trivially and both tests would still pass in the wrong way."""
    conservative, _ = _run("divergence_upwind")
    gradient, _ = _run("gradient_upwind")
    assert gradient - conservative < -50.0, (
        "the two schemes no longer separate; the fixture may have stopped exercising the wall"
    )


# =============================================================================
# The real gap: a wall that is NOT the reflecting one
# =============================================================================


def _mixed(bc_type, **kw):
    return BoundaryConditions(
        segments=[
            BCSegment(name="lo", bc_type=bc_type, boundary="x_min", **kw),
            BCSegment(name="hi", bc_type=bc_type, boundary="x_max", **kw),
        ],
        dimension=1,
    )


def _step(bc, scheme="divergence_upwind"):
    from mfgarchon.alg.numerical.fp_solvers.fp_fdm_time_stepping import solve_timestep_full_nd

    grid = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[21], boundary_conditions=no_flux_bc(dimension=1))
    h = grid.get_grid_spacing()[0]
    x = np.linspace(0.0, 1.0, 21)
    m = np.exp(-50 * (x - 0.35) ** 2)
    m /= m.sum() * h
    return solve_timestep_full_nd(
        M_current=m,
        U_current=-3.2 * x,
        problem=object(),
        dt=1e-3,
        sigma=0.3,
        coupling_coefficient=1.0,
        spacing=(h,),
        grid=grid,
        ndim=1,
        shape=(21,),
        boundary_conditions=bc,
        advection_scheme=scheme,
    )


def test_a_wall_that_is_not_the_reflecting_one_is_still_assembled_as_no_flux():
    """`alpha = 999` is a genuinely different wall and must not equal no-flux. It does.

    `alpha = 3.2` is deliberately **excluded**: with this fixture `FPResolver` emits
    `ROBIN(alpha=+3.2, beta=-0.045)` for the impermeable wall, so that case *is* the reflecting
    wall the conservative scheme already imposes and byte-identity there is correct. The previous
    version of this file asserted on it and read the right answer as a defect.

    The mechanism: `_BOUNDARY_HANDLERS` handlers are not passed `boundary_conditions` at all, so
    no configuration can make them read a coefficient. Measured over 768 combinations of ndim,
    npts, sigma, dt, scheme and (alpha, beta, g): zero non-identical cases.
    """
    reference = _step(_mixed(BCType.NO_FLUX))
    got = _step(_mixed(BCType.ROBIN, alpha=999.0, beta=-0.045, value=0.0))

    assert np.array_equal(got, reference), (
        "a ROBIN wall with alpha=999 no longer assembles byte-identically to no-flux. If the FDM "
        "boundary handlers now read (alpha, beta, g), that is the #1975 gap closing: close it and "
        "replace this with a convergence check against the exact d_n m = (alpha/beta)*m relation. "
        "Do NOT close it by making the conservative schemes read alpha for a reflecting wall -- "
        "they already impose J.n = 0 and would double-count."
    )


def test_a_provider_valued_coefficient_is_accepted_without_a_word():
    """#1979. A coupled wall coefficient -- the entire point of the provider layer -- produces a
    no-flux wall and no diagnostic. Pinned so the fix is visible, not so it is defended."""
    from mfgarchon.geometry.boundary.providers import AdjointConsistentProvider

    reference = _step(_mixed(BCType.NO_FLUX))
    got = _step(_mixed(BCType.ROBIN, alpha=AdjointConsistentProvider(side="left", sigma=0.3), beta=-0.045, value=0.0))

    assert np.array_equal(got, reference), (
        "the FDM path now does something with a provider-valued alpha. If it RAISES, that is "
        "#1979 fixed -- delete this and assert the refusal instead."
    )


def test_the_resolver_still_emits_the_condition_its_consumers_impose_structurally():
    """`FPResolver` translates an impermeable wall into `ROBIN(alpha=v_n, beta=-D, g=0)`.

    It has zero library callers, and #1975 records why that is not simply a wiring bug: the
    conservative assemblies already impose that condition, so routing through the resolver would
    be redundant there and double-counting on the FEM weak form (measured: -79.5% mass drift when
    `alpha = v_n` is handed to `assemble_robin_terms`, whose natural BC is already `J.n = 0`).
    """
    resolved = FPResolver().resolve(
        BCSegment(name="wall", bc_type=BCType.NO_FLUX, boundary="x_min"),
        {"drift": _DRIFT, "diffusion": 0.5 * _SIGMA**2},
    )
    assert resolved.math_type is MathBCType.ROBIN
    assert resolved.alpha == pytest.approx(_DRIFT)
    assert resolved.beta == pytest.approx(-0.5 * _SIGMA**2)


# =============================================================================
# The capability gate: who is inside it
# =============================================================================


def test_the_solver_population_is_unchanged():
    """Discovered by `walk_packages` + `issubclass`, a predicate that does NOT depend on the
    declaration being audited. The old census discovered its population by reading
    `_SUPPORTED_BC_TYPES` inside two non-recursed directories and so could not report its own
    blind spot -- it missed 10 solvers.
    """
    found = _solver_population()
    assert set(found) == _GATED.keys() | _UNGATED, (
        f"solver population changed: added {sorted(set(found) - (_GATED.keys() | _UNGATED))}, "
        f"removed {sorted((_GATED.keys() | _UNGATED) - set(found))}. Update the census, then #1977."
    )


def test_exactly_these_solvers_are_outside_the_capability_gate():
    """Recorded as a population, not as an absence. Half the solvers declare nothing, so
    `_validate_bc_support` no-ops on them -- `FPFEMSolver` among them, which is the one FP solver
    implementing a general Robin. A solver LEAVING this set is #1977 being fixed; a solver joining
    it is a new capability shipped without a declaration.
    """
    found = _solver_population()
    ungated = {n for n, gated in found.items() if not gated}
    assert ungated == _UNGATED, (
        f"newly ungated: {sorted(ungated - _UNGATED)} (a solver shipped without declaring its BC "
        f"support); newly gated: {sorted(_UNGATED - ungated)} (#1977 progress -- record it)."
    )


@pytest.mark.parametrize("solver", sorted(_GATED))
def test_each_gated_solver_accepts_exactly_what_the_census_records(solver):
    """Widening is a capability gained, narrowing is one lost; the message says which, because
    the two need opposite responses."""
    got, want = _declared(solver), _GATED[solver]
    if got == want:
        return
    raise AssertionError(
        f"{solver} now accepts {sorted(got)}, census says {sorted(want)}.\n"
        f"  gained: {sorted(got - want) or 'nothing'}  -- new capability; update the census.\n"
        f"  lost:   {sorted(want - got) or 'nothing'}  -- a wall that was expressible no longer is."
    )


@pytest.mark.parametrize(("name", "module"), sorted(_NOT_EVEN_SUBCLASSES.items()))
def test_the_classes_outside_any_subclass_predicate_are_still_outside_it(name, module):
    """These apply BC segments -- Dirichlet values, Neumann normal-gradient stencil rows -- and are
    not `BaseNumericalSolver` subclasses, so no population predicate above reaches them. Named
    rather than discovered, which is the only way a blind spot can be pinned at all."""
    cls = getattr(importlib.import_module(module), name)
    assert not issubclass(cls, BaseNumericalSolver), f"{name} is now a solver subclass -- add it to the census"
    assert getattr(cls, "_SUPPORTED_BC_TYPES", None) is None, f"{name} now declares support -- record it (#1977)"


def test_an_unsupported_bc_raises_at_construction():
    """Behavioural, replacing an `inspect.getsource` string match that was wrong in **both**
    directions: it passed when the gate was disabled (`if False and unsupported:`) and when the
    call was deleted but the string kept, and it failed on a behaviour-preserving refactor that
    moved the call into a helper.
    """
    from mfgarchon import Conditions, MFGProblem, Model
    from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian

    bc = _mixed(BCType.ROBIN, alpha=1.0, beta=1.0, value=0.0)
    model = Model(
        hamiltonian=SeparableHamiltonian(
            control_cost=QuadraticControlCost(control_cost=1.0),
            coupling=lambda m: 0.5 * m,
            coupling_dm=lambda m: 0.5,
        ),
        sigma=0.1,
    )
    domain = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[21], boundary_conditions=bc)
    conditions = Conditions(
        u_terminal=lambda x: (x - 0.5) ** 2,
        m_initial=lambda x: np.exp(-50 * (x - 0.5) ** 2),
        T=0.2,
    )
    problem = MFGProblem(model=model, domain=domain, conditions=conditions, Nt=5)

    with pytest.raises(NotImplementedError, match="ROBIN"):
        problem.solve(max_iterations=1)


def test_no_gated_solver_may_declare_robin_while_the_assembly_ignores_alpha():
    """Couples the two halves, so the naive fix cannot look like the real one.

    Declaring `ROBIN` without teaching the assembly to read the coefficients is exactly the fix
    #1975 warns against, and with the two facts pinned separately that state passes the assembly
    check while the declaration checks report a capability gained. This is the assertion that
    fails on it.
    """
    fp_robin = {n for n in _GATED if n.startswith("FP") and "ROBIN" in _declared(n)}
    coefficients_ignored = np.array_equal(
        _step(_mixed(BCType.ROBIN, alpha=999.0, beta=-0.045, value=0.0)),
        _step(_mixed(BCType.NO_FLUX)),
    )
    assert not (fp_robin and coefficients_ignored), (
        f"{sorted(fp_robin)} declare(s) ROBIN while the FDM boundary handlers still ignore alpha "
        "(a wall with alpha=999 is byte-identical to no-flux). That is a declaration without an "
        "implementation -- the failure mode #1456's gate exists to prevent."
    )
