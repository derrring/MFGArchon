"""Which solvers declare their BC support, and what the FDM boundary assembly reads. #1975 #1977

The reasoning, the history and the physics live in #1975.
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

# --------------------------------------------------------------------------------------
# Population
# --------------------------------------------------------------------------------------


def _population() -> dict[type, list[str]]:
    """Concrete `BaseNumericalSolver` subclasses -> every name they are bound under anywhere in
    `mfgarchon.alg`. `walk_packages` never yields the root, so `mfgarchon.alg` is imported
    explicitly.

    Deliberately no `cls.__module__ != module.__name__` filter: it hid the cross-module alias at
    `network_solvers/__init__.py:19` from the test written to catch that case (#1976 review)."""
    import mfgarchon.alg as alg_pkg

    found: dict[type, list[str]] = {}
    for mod_name in ["mfgarchon.alg"] + [
        m.name for m in pkgutil.walk_packages(alg_pkg.__path__, prefix="mfgarchon.alg.")
    ]:
        module = importlib.import_module(mod_name)
        for name, cls in inspect.getmembers(module, inspect.isclass):
            if not issubclass(cls, BaseNumericalSolver) or inspect.isabstract(cls):
                continue
            found.setdefault(cls, [])
            if name not in found[cls]:
                found[cls].append(name)
    # `names[0]` is the canonical name, and it must be `cls.__name__` rather than whichever
    # binding `inspect.getmembers` yields first: getmembers sorts alphabetically, so the alias
    # `HJBNetworkSolver` would otherwise displace `NetworkHJBSolver` as the row's identity.
    for cls, names in found.items():
        names.sort(key=lambda n: (n != cls.__name__, n))
    return found


_GATED = {
    # #1977: the FEM pair joined on 2026-08-17. Declaring was not enough -- the gate reads the
    # `supported_bc_types` property and had to be CALLED; both were added with the declaration.
    # Behavioural cover: tests/unit/test_alg/test_bc_gate_fires_behaviourally_1977.py
    "FPFEMSolver": {"DIRICHLET", "NEUMANN", "NO_FLUX", "REFLECTING", "ROBIN"},
    "HJBFEMSolver": {"DIRICHLET", "NEUMANN", "NO_FLUX", "REFLECTING", "ROBIN"},
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

#: Declares nothing, so `_validate_bc_support` no-ops on it (`base_solver.py:282`). #1977.
_UNGATED = {
    "FPNetworkSolver",
    "MeshlessGalerkinFPSolver",
    "MeshlessGalerkinHJBSolver",
    "NetworkHJBSolver",
    "NetworkPolicyIterationHJBSolver",
    "PenaltyHJBSolver",
    "WeakFormFPSolver",
    "WeakFormHJBSolver",
}

#: Same class bound under more than one name. `NetworkFPSolver` is a second binding in the
#: defining module (`fp_network.py:606`); `HJBNetworkSolver` is a cross-module re-export alias
#: (`network_solvers/__init__.py:19`) that only becomes visible without a defining-module filter.
_ALIASES = {"FPNetworkSolver": ["NetworkFPSolver"], "NetworkHJBSolver": ["HJBNetworkSolver"]}

#: Do a solver's boundary job without being solver subclasses, so no predicate here reaches them.
#: Named, not discovered, and incomplete: other non-subclass classes branch on a BC type too.
#:
#: ~~"Apply BC segments"~~ [CORRECTED 2026-08-17] -- true of one of the three. `HJBHowardSolver`
#: reads `seg.bc_type` and branches on DIRICHLET (hjb_howard.py:363). `BoundaryHandler` probes
#: `segments` for truthiness only, to classify "mixed" (boundary_handler.py:324). `ImplicitHeatSolver`
#: forwards `bc` to `get_laplacian_operator` (implicit_heat.py:99) and otherwise mentions a segment
#: only inside a `__repr__` f-string (:293). The roster is right; the verb was not.
_NOT_REACHED = {
    "HJBHowardSolver": "mfgarchon.alg.numerical.hjb_solvers.hjb_howard",
    "ImplicitHeatSolver": "mfgarchon.alg.numerical.pde_solvers.implicit_heat",
    "BoundaryHandler": "mfgarchon.alg.numerical.gfdm_components.boundary_handler",
}


def test_the_population_is_exactly_the_two_sets():
    got = {names[0] for names in _population().values()}
    want = _GATED.keys() | _UNGATED
    assert got == want, f"added {sorted(got - want)}, removed {sorted(want - got)}"


def test_the_alias_bindings_are_unchanged():
    got = {names[0]: names[1:] for names in _population().values() if len(names) > 1}
    assert got == _ALIASES, f"alias bindings changed: {got}"


def test_no_solver_is_bound_under_a_name_the_census_does_not_know():
    known = _GATED.keys() | _UNGATED | {a for v in _ALIASES.values() for a in v}
    unknown = {n for names in _population().values() for n in names} - known
    assert not unknown, f"names not in any set: {sorted(unknown)}"


@pytest.mark.parametrize("solver", sorted(_GATED))
def test_each_gated_solver_declares_exactly_this(solver):
    cls = next(c for c, n in _population().items() if n[0] == solver)
    got = {t.name for t in cls._SUPPORTED_BC_TYPES}
    want = _GATED[solver]
    assert got == want, f"gained {sorted(got - want)}, lost {sorted(want - got)}"


def test_exactly_these_solvers_declare_nothing():
    ungated = {n[0] for c, n in _population().items() if getattr(c, "_SUPPORTED_BC_TYPES", None) is None}
    assert ungated == _UNGATED, f"newly silent {sorted(ungated - _UNGATED)}; now declaring {sorted(_UNGATED - ungated)}"


@pytest.mark.parametrize(("name", "module"), sorted(_NOT_REACHED.items()))
def test_the_named_unreached_classes_are_still_unreached(name, module):
    cls = getattr(importlib.import_module(module), name)
    assert not issubclass(cls, BaseNumericalSolver), f"{name} is now a solver subclass"


def test_the_gate_reaches_four_classes_this_file_does_not():
    """The population predicate is `BaseNumericalSolver`; `_validate_bc_support` and
    `honors_inhomogeneous_neumann` live on `BaseMFGSolver`, a strict superclass. An undisclosed
    population predicate is a scope claim, so the gap is computed rather than described."""
    from mfgarchon.alg.base_solver import BaseMFGSolver

    reached = set()
    for mod_name in ["mfgarchon.alg"] + [
        m.name
        for m in pkgutil.walk_packages(importlib.import_module("mfgarchon.alg").__path__, prefix="mfgarchon.alg.")
    ]:
        for _, cls in inspect.getmembers(importlib.import_module(mod_name), inspect.isclass):
            if issubclass(cls, BaseMFGSolver) and not inspect.isabstract(cls):
                reached.add(cls)
    outside = {c.__name__ for c in reached} - {n[0] for n in _population().values()}
    assert outside == {
        "PrimalDualMFGSolver",
        "SinkhornMFGSolver",
        "VariationalMFGSolver",
        "WassersteinMFGSolver",
    }, f"changed: {sorted(outside)}"
    assert all(getattr(c, "_SUPPORTED_BC_TYPES", None) is None for c in reached if c.__name__ in outside)


def test_the_permissive_default_is_claimed_by_inheritance():
    """`honors_inhomogeneous_neumann` defaults to True on `BaseMFGSolver`, so a solver that never
    mentions it claims to honour an inhomogeneous flux."""
    field = "honors_inhomogeneous_neumann"
    own, inherited = set(), {}
    for cls, names in _population().items():
        owner = next((k.__name__ for k in cls.__mro__ if field in k.__dict__), None)
        if owner is None:
            continue
        (own.add(names[0]) if owner == names[0] else inherited.setdefault(owner, set()).add(names[0]))
    assert own == {
        "FPFDMSolver",
        "FPFVMSolver",
        "FPGFDMSolver",
        "FPParticleSolver",
        "FPSLJacobianSolver",
        "FPSLSolver",
    }
    assert set(inherited) == {"BaseMFGSolver", "FPSLSolver"}
    assert all(getattr(cls, field) is True for cls, n in _population().items() if n[0] in inherited["BaseMFGSolver"])


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

    `divergence_upwind` is the default.
    """
    drift_pct, dmdx = _run(scheme)
    assert abs(drift_pct) < 1e-6, f"{scheme} leaked {drift_pct:.4f}% at a wall with normal drift"
    assert abs(dmdx) > 100.0, (
        f"{scheme} has d_n m = {dmdx:.4g} at the wall. J.n = 0 requires d_n m = (v/D)*m, which is "
        "large here; a small gradient means the wall became d_n m = 0."
    )  # 1.0 until 2026-08-17: the gradient_* wall this message says it detects sits ABOVE 1.0 on
    # this fixture, so the threshold named a failure inside its own pass band. Figures in #1975.


@pytest.mark.parametrize("scheme", ["gradient_upwind", "gradient_centered"])
def test_the_gradient_schemes_impose_a_zero_gradient_wall_and_leak(scheme):
    """The counterpart, pinned so the distinction cannot quietly collapse in either direction.

    These impose `d_n m = 0`, which is the wrong condition when the drift is not tangential, and
    they are documented non-conservative (#1075).
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


# --------------------------------------------------------------------------------------
# What the FDM boundary assembly reads
# --------------------------------------------------------------------------------------


def _mixed(bc_type, **kw):
    return BoundaryConditions(
        segments=[
            BCSegment(name="lo", bc_type=bc_type, boundary="x_min", **kw),
            BCSegment(name="hi", bc_type=bc_type, boundary="x_max", **kw),
        ],
        dimension=1,
    )


def _step(bc):
    from mfgarchon.alg.numerical.fp_solvers.fp_fdm_time_stepping import solve_timestep_full_nd

    nx = 21
    grid = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[nx], boundary_conditions=no_flux_bc(dimension=1))
    h = grid.get_grid_spacing()[0]
    x = np.linspace(0.0, 1.0, nx)
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
        shape=(nx,),
        boundary_conditions=bc,
        advection_scheme="divergence_upwind",
    )


def test_the_step_responds_to_a_bc_at_all():
    """Positive control for the two null results below: without it, byte-identity proves nothing
    about coefficients and only that `_step` ignores its argument."""
    assert not np.array_equal(_step(_mixed(BCType.DIRICHLET, value=7.0)), _step(_mixed(BCType.NO_FLUX)))


def test_a_robin_coefficient_the_assembly_cannot_read_gives_the_no_flux_result():
    """`alpha = 999`. `_BOUNDARY_HANDLERS`' handlers take no `boundary_conditions` parameter."""
    assert np.array_equal(
        _step(_mixed(BCType.ROBIN, alpha=999.0, beta=-0.045, value=0.0)),
        _step(_mixed(BCType.NO_FLUX)),
    )


def test_a_provider_valued_coefficient_is_refused_rather_than_dropped():
    """#1979, fixed. This test used to pin the DEFECT -- it asserted that a coupled wall coefficient
    produced a byte-identical no-flux wall with no diagnostic -- and carried its own retirement
    instruction: "if it RAISES, #1979 is fixed -- assert the refusal instead". It raises. This is
    that assertion.

    Recorded so the history is not lost: before the fix, a ROBIN segment carrying an
    `AdjointConsistentProvider` on `alpha` assembled `np.array_equal` to a plain NO_FLUX wall. The
    wall a user wired a coupled coefficient to AVOID was returned as though it were their request.
    """
    from mfgarchon.geometry.boundary.providers import AdjointConsistentProvider

    bc = _mixed(BCType.ROBIN, alpha=AdjointConsistentProvider(side="left", sigma=0.3), beta=-0.045, value=0.0)
    with pytest.raises(NotImplementedError, match=r"provider-valued wall coefficient"):
        _step(bc)

    # Control: a float coefficient must still assemble, or the refusal is a blanket one and every
    # ordinary wall is broken. (It still equals the no-flux result -- the handlers do not read
    # coefficients at all, which is #1979's actual mechanism and is not what this PR changes.)
    got = _step(_mixed(BCType.ROBIN, alpha=1.0, beta=-0.045, value=0.0))
    assert np.array_equal(got, _step(_mixed(BCType.NO_FLUX)))


def test_an_unsupported_bc_raises_at_construction():
    """The gate reads `supported_bc_types` (the protocol property), not the private attribute."""
    from mfgarchon import Conditions, MFGProblem, Model
    from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian

    problem = MFGProblem(
        model=Model(
            hamiltonian=SeparableHamiltonian(
                control_cost=QuadraticControlCost(control_cost=1.0),
                coupling=lambda m: 0.5 * m,
                coupling_dm=lambda m: 0.5,
            ),
            sigma=0.1,
        ),
        domain=TensorProductGrid(
            bounds=[(0.0, 1.0)],
            Nx_points=[21],
            boundary_conditions=_mixed(BCType.ROBIN, alpha=1.0, beta=1.0, value=0.0),
        ),
        conditions=Conditions(
            u_terminal=lambda x: (x - 0.5) ** 2,
            m_initial=lambda x: np.exp(-50 * (x - 0.5) ** 2),
            T=0.2,
        ),
        Nt=5,
    )
    with pytest.raises(NotImplementedError, match="ROBIN"):
        problem.solve(max_iterations=1)


def test_no_fp_solver_declares_robin_while_the_assembly_ignores_the_coefficient():
    """Reads `_SUPPORTED_BC_TYPES`; the #1456 gate reads `supported_bc_types`.

    Today every declaring solver's property is exactly `return self._SUPPORTED_BC_TYPES`, so the
    two coincide -- but that premise is NOT asserted here. A previous revision asserted it by
    comparing `inspect.getsource(...).splitlines()[-1]`, and review measured that open in three
    of five shapes (a multi-line body ending in the forward; a `functools.wraps`-decorated
    property, since `getsource` unwraps; an ungated class gaining a non-`property` attribute,
    which `assert fget is None` accepts while the gate goes live) and closed falsely on one
    (a behaviour-preserving `frozenset(...)` copy). It is the instrument an earlier revision
    deleted for being wrong in both directions, and it was wrong in both directions again, so it
    is gone rather than repaired. **The hole is: a solver can widen its live support through the
    property alone and every assertion here stays green.**
    """
    # Scoped to the solvers that ROUTE THROUGH the FDM assembly. `FPFEMSolver` declares ROBIN
    # (#1977, 2026-08-17) and reads the coefficients itself via `fem/bc_adapter.assemble_robin_terms`
    # -- it never reaches `solve_timestep_full_nd`, so its declaration says nothing about the
    # blindness measured below. Before that declaration existed the two sets coincided and the
    # narrower predicate was invisible.
    _FDM_ASSEMBLY = {"FPFDMSolver"}
    declaring = {
        n[0]
        for c, n in _population().items()
        if n[0] in _FDM_ASSEMBLY and any(t.name == "ROBIN" for t in (getattr(c, "_SUPPORTED_BC_TYPES", None) or ()))
    }
    ignored = np.array_equal(
        _step(_mixed(BCType.ROBIN, alpha=999.0, beta=-0.045, value=0.0)),
        _step(_mixed(BCType.NO_FLUX)),
    )
    assert not (declaring and ignored), (
        f"{sorted(declaring)} declare ROBIN while the assembly is still blind to alpha -- "
        "a declaration without an implementation"
    )
