"""Which solvers declare their BC support, and what the FDM boundary assembly reads. #1975 #1977

Assertions only. The reasoning, the history and the physics live in #1975; four revisions of this
file put them here as prose and four independent reviews found a false statement in that prose
each time. Nothing below is stated that is not computed in this file.
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
    """Concrete `BaseNumericalSolver` subclasses -> the names they are bound under in their OWN
    module. `walk_packages` never yields the root, so `mfgarchon.alg` is imported explicitly."""
    import mfgarchon.alg as alg_pkg

    found: dict[type, list[str]] = {}
    for mod_name in ["mfgarchon.alg"] + [
        m.name for m in pkgutil.walk_packages(alg_pkg.__path__, prefix="mfgarchon.alg.")
    ]:
        module = importlib.import_module(mod_name)
        for name, cls in inspect.getmembers(module, inspect.isclass):
            if cls.__module__ != module.__name__:
                continue  # a re-export; the defining module owns the row
            if not issubclass(cls, BaseNumericalSolver) or inspect.isabstract(cls):
                continue
            found.setdefault(cls, [])
            if name not in found[cls]:
                found[cls].append(name)
    return found


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

#: Declares nothing, so `_validate_bc_support` no-ops on it (`base_solver.py:282`). #1977.
_UNGATED = {
    "FPFEMSolver",
    "FPNetworkSolver",
    "HJBFEMSolver",
    "MeshlessGalerkinFPSolver",
    "MeshlessGalerkinHJBSolver",
    "NetworkHJBSolver",
    "NetworkPolicyIterationHJBSolver",
    "PenaltyHJBSolver",
    "WeakFormFPSolver",
    "WeakFormHJBSolver",
}

#: Same class bound twice in its own module.
_ALIASES = {"FPNetworkSolver": ["NetworkFPSolver"]}

#: Apply BC segments without being solver subclasses, so no predicate here reaches them. Named,
#: not discovered -- and this list is known to be incomplete.
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
    """`names[0]` is whichever binding `inspect.getmembers` yields first, so a new alias that
    sorts earlier would silently replace a canonical name in every set above."""
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


def test_a_provider_valued_coefficient_is_accepted_without_a_word():
    """#1979: a coupled wall coefficient produces a no-flux wall and no diagnostic."""
    from mfgarchon.geometry.boundary.providers import AdjointConsistentProvider

    got = _step(_mixed(BCType.ROBIN, alpha=AdjointConsistentProvider(side="left", sigma=0.3), beta=-0.045, value=0.0))
    assert np.array_equal(got, _step(_mixed(BCType.NO_FLUX))), (
        "the FDM path now does something with a provider-valued alpha; if it RAISES, #1979 is "
        "fixed -- assert the refusal instead"
    )


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


def test_the_gate_reads_what_this_file_reads():
    """The gate reads `supported_bc_types`; every assertion here reads `_SUPPORTED_BC_TYPES`.

    Those are the same thing only while every property just forwards. Measured: all 11 gated
    solvers have a property whose body is exactly `return self._SUPPORTED_BC_TYPES`, and the 11
    ungated ones have no property at all. If either stops being true, this file is measuring
    something the gate does not use -- which is how a solver could widen its live support with
    every assertion here still green.
    """
    import textwrap

    for cls, names in _population().items():
        prop = inspect.getattr_static(cls, "supported_bc_types", None)
        fget = getattr(prop, "fget", None)
        declares = getattr(cls, "_SUPPORTED_BC_TYPES", None) is not None
        if not declares:
            assert fget is None, f"{names[0]} declares nothing but defines the property"
            continue
        assert fget is not None, f"{names[0]} declares but has no supported_bc_types property"
        body = textwrap.dedent(inspect.getsource(fget)).strip().splitlines()[-1].strip()
        assert body == "return self._SUPPORTED_BC_TYPES", (
            f"{names[0]}'s property no longer just forwards ({body!r}); the gate and this file "
            "now read different things"
        )


def test_no_fp_solver_declares_robin_while_the_assembly_ignores_the_coefficient():
    """Justified by the test above: the private attribute is what the gate ends up reading."""
    declaring = {
        n[0]
        for c, n in _population().items()
        if n[0].startswith("FP") and any(t.name == "ROBIN" for t in (getattr(c, "_SUPPORTED_BC_TYPES", None) or ()))
    }
    ignored = np.array_equal(
        _step(_mixed(BCType.ROBIN, alpha=999.0, beta=-0.045, value=0.0)),
        _step(_mixed(BCType.NO_FLUX)),
    )
    assert not (declaring and ignored), (
        f"{sorted(declaring)} declare ROBIN while the assembly is still blind to alpha -- "
        "a declaration without an implementation"
    )
