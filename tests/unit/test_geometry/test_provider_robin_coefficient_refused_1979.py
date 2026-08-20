"""A provider-valued Robin coefficient must be refused, not silently dropped (#1979).

`BCValueProvider` exists so a wall coefficient can be recomputed each Picard iterate -- that is the
whole point of `AdjointConsistentProvider` and of #1970's `NormalDriftProvider`, both of which live
on `alpha`, not on `value`. Two paths consumed such a segment wrongly:

- **FDM**: `_BOUNDARY_HANDLERS` is keyed on the advection scheme and its handlers take no
  `boundary_conditions` argument at all, so nothing read `alpha` and the segment assembled
  byte-identically to a no-flux wall -- the wall the user wired a coefficient to avoid, returned as
  though it were their request.
- **FEM**: `assemble_robin_terms` coerced with `float()`, giving a bare builtin `TypeError` --
  unnamed, ungreppable, and silent about the remedy -- three lines above a `NotImplementedError`
  guard that already did this correctly for `g`.

Refusing is the fix, not reading. The conservative grid schemes already impose `J.n = 0`
structurally (#1975), so teaching those handlers to add `(alpha, beta, g)` would count the drift
twice -- measured there at -79.5% mass against -7.4e-15. A general Robin wall on the grid paths is
#1975; this pins only that the silent case is gone.

Each refusal is paired with a float-coefficient control, because a guard that refuses everything
would pass the `raises` half while breaking every legitimate Robin solve.
"""

from __future__ import annotations

import pytest

import numpy as np

from mfgarchon.geometry.boundary.conditions import BoundaryConditions
from mfgarchon.geometry.boundary.providers import ConstantProvider
from mfgarchon.geometry.boundary.types import BCSegment, BCType


def _no_flux_bc(alpha):
    """The REACHABLE case, and the one #1970 produces.

    Not a ROBIN segment: every grid FP solver refuses ROBIN at construction
    (`_validate_bc_support`, #1456), so that route is already closed and cannot be the user impact.
    An impermeable wall IS Robin in m -- `alpha*m - D*d_n m = 0` -- so its coefficient lives on
    `alpha` of a NO_FLUX segment, which passes the capability gate.
    """
    return BoundaryConditions(
        segments=[
            BCSegment(name="wall", bc_type=BCType.NO_FLUX, boundary="x_min", value=0.0, alpha=alpha),
            BCSegment(name="wall2", bc_type=BCType.NO_FLUX, boundary="x_max", value=0.0),
        ],
        dimension=1,
    )


def _robin_bc(alpha, beta=1.0):
    """For the FEM path, which does accept ROBIN."""
    return BoundaryConditions(
        segments=[BCSegment(name="wall", bc_type=BCType.ROBIN, boundary="x_min", value=0.0, alpha=alpha, beta=beta)],
        dimension=1,
    )


def test_the_provider_is_detected_on_alpha_at_all():
    """Control on the premise: `has_providers` must see a provider on `alpha`, not only on `value`.

    If this ever regresses to `value`-only, both guards below go quiet and pass for the wrong
    reason -- the resolver gate has already been widened once for exactly that (#1979 cites the
    2026-08-16 correction in `has_providers`).
    """
    assert _no_flux_bc(alpha=ConstantProvider(value=2.0)).has_providers() is True
    assert _no_flux_bc(alpha=1.0).has_providers() is False


def _fdm_solver_with(bc):
    """The user-facing path #1979 describes: a solver constructed with a Robin wall."""
    import numpy as np

    from mfgarchon.alg.numerical.fp_solvers.fp_fdm import FPFDMSolver
    from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
    from mfgarchon.core.mfg_components import MFGComponents
    from mfgarchon.core.mfg_problem import MFGProblem
    from mfgarchon.geometry import TensorProductGrid

    nx = 11
    grid = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[nx], boundary_conditions=bc)
    components = MFGComponents(
        hamiltonian=SeparableHamiltonian(control_cost=QuadraticControlCost(control_cost=1.0)),
        m_initial=lambda x: 1.0,
        u_terminal=lambda x: 0.0,
    )
    problem = MFGProblem(geometry=grid, components=components, T=0.1, Nt=4, sigma=1.0)
    return FPFDMSolver(problem, boundary_conditions=bc), np.ones(nx) / nx


class TestFDMPath:
    def test_a_provider_valued_alpha_is_refused(self):
        solver, m0 = _fdm_solver_with(_no_flux_bc(alpha=ConstantProvider(value=2.0)))
        with pytest.raises(NotImplementedError, match=r"provider-valued wall coefficient"):
            solver.solve_fp_system(M_initial=m0, potential_field=np.zeros((5, len(m0))))

    def test_the_message_names_the_coefficient_and_the_remedy(self):
        solver, m0 = _fdm_solver_with(_no_flux_bc(alpha=ConstantProvider(value=2.0)))
        with pytest.raises(NotImplementedError) as exc:
            solver.solve_fp_system(M_initial=m0, potential_field=np.zeros((5, len(m0))))
        text = str(exc.value)
        assert "wall.alpha" in text, "the message must name WHICH coefficient, not just that one exists"
        assert "NO_FLUX" in text, "the message must name the segment type, since ROBIN is refused elsewhere"
        assert "1979" in text
        # The remedy must be a real capability, not a way around the guard. `FPFEMSolver` reads
        # alpha/beta/g in its weak form (established in #1975, closed COMPLETED); resolving the
        # provider is a DOWNGRADE -- it freezes a per-iterate coefficient at one state -- and the
        # message must say so rather than offer it as the fix.
        assert "FPFEMSolver" in text, "the message must name the path that CAN do this"
        assert "with_resolved_providers" in text, "resolving must still be mentioned"
        assert "DOWNGRADE" in text, (
            "offering `with_resolved_providers` as the remedy prescribes the guard's own defeat: it "
            "discards exactly the property a provider exists for"
        )
        assert "1975" not in text.replace("#1975", ""), "avoid a bare 1975 that reads as an open pointer"


class TestTheGuardIsWhereTheHazardIs:
    """#2013 review: the guard sat in `solve_fp_nd_full_system` while `_BOUNDARY_HANDLERS` is
    dispatched from `solve_timestep_full_nd`, so calling that directly walked straight past it --
    the same "the gate is not where the hazard is" shape the guard exists to fix."""

    def test_the_single_owner_is_called_from_both_entry_points(self):
        import inspect

        from mfgarchon.alg.numerical.fp_solvers import fp_fdm_time_stepping as mod

        src = inspect.getsource(mod)
        assert src.count("def _refuse_provider_wall_coefficients(") == 1, "one owner, not a copy per entry"
        # Count CALL SITES, which means excluding the `def` line -- it matches the same text, and a
        # first version of this assertion counted it and read 3 where it expected 2.
        calls = [
            ln
            for ln in src.splitlines()
            if "_refuse_provider_wall_coefficients(boundary_conditions)" in ln and not ln.lstrip().startswith("def ")
        ]
        assert len(calls) == 2, (
            f"both `solve_fp_nd_full_system` and `solve_timestep_full_nd` must call it; the dispatch "
            f"site is the one that matters, because it is where _BOUNDARY_HANDLERS is reached. "
            f"Found {len(calls)}: {calls}"
        )
        # and the call must precede the dispatch, not follow it
        assert src.index(
            "_refuse_provider_wall_coefficients(boundary_conditions)", src.index("def solve_timestep_full_nd")
        ) < src.index("_BOUNDARY_HANDLERS[advection_scheme]("), "the guard must run BEFORE the handlers are dispatched"

    def test_a_provider_on_value_is_refused_too(self):
        """#1686: every FP solver silently drops the value in `neumann_bc(value=g)`. A guard that
        covered only alpha/beta would refuse the coefficient and keep dropping the datum."""
        from mfgarchon.alg.numerical.fp_solvers.fp_fdm_time_stepping import (
            _refuse_provider_wall_coefficients,
        )

        bc = BoundaryConditions(
            segments=[
                BCSegment(name="wall", bc_type=BCType.NEUMANN, boundary="x_min", value=ConstantProvider(value=0.5))
            ],
            dimension=1,
        )
        with pytest.raises(NotImplementedError, match=r"wall\.value"):
            _refuse_provider_wall_coefficients(bc)

        # Control: a float value must pass, or the guard is a blanket refusal.
        ok = BoundaryConditions(
            segments=[BCSegment(name="wall", bc_type=BCType.NEUMANN, boundary="x_min", value=0.5)],
            dimension=1,
        )
        _refuse_provider_wall_coefficients(ok)
        _refuse_provider_wall_coefficients(None)

    def test_a_float_alpha_does_not_trip_the_guard(self):
        """Control: a blanket refusal would pass the two tests above and break every Robin solve."""
        solver, m0 = _fdm_solver_with(_no_flux_bc(alpha=1.0))
        raised = ""
        try:
            solver.solve_fp_system(M_initial=m0, potential_field=np.zeros((5, len(m0))))
        except Exception as exc:  # record any refusal, then judge it below
            raised = str(exc)
        # A different contract refusing this wall is #1975's business, not this guard's. What must
        # not happen is THIS guard firing on a float.
        assert "provider-valued wall coefficient" not in raised, (
            f"a float-valued alpha tripped the #1979 provider guard: {raised}"
        )


class TestFEMPath:
    def _basis(self):
        skfem = pytest.importorskip("skfem")
        mesh = skfem.MeshTri().refined(2)
        return skfem.Basis(mesh, skfem.ElementTriP1()), mesh

    @pytest.mark.parametrize("field", ["alpha", "beta"])
    def test_a_provider_valued_coefficient_is_refused(self, field):
        from mfgarchon.alg.numerical.fem.bc_adapter import assemble_robin_terms

        basis, _ = self._basis()
        kwargs = {"alpha": 1.0, "beta": 1.0, field: ConstantProvider(value=2.0)}
        bc = BoundaryConditions(
            segments=[BCSegment(name="wall", bc_type=BCType.ROBIN, boundary="x_min", value=0.0, **kwargs)],
            dimension=2,
        )
        with pytest.raises(NotImplementedError, match=rf"non-constant {field}"):
            assemble_robin_terms(basis, bc, D=1.0)

    def test_the_refusal_is_not_the_bare_TypeError_it_replaced(self):
        """The defect was `float()` raising a builtin TypeError. Pin that it no longer can."""
        from mfgarchon.alg.numerical.fem.bc_adapter import assemble_robin_terms

        basis, _ = self._basis()
        bc = BoundaryConditions(
            segments=[
                BCSegment(
                    name="wall",
                    bc_type=BCType.ROBIN,
                    boundary="x_min",
                    value=0.0,
                    alpha=ConstantProvider(value=2.0),
                    beta=1.0,
                )
            ],
            dimension=2,
        )
        with pytest.raises(NotImplementedError):
            assemble_robin_terms(basis, bc, D=1.0)
