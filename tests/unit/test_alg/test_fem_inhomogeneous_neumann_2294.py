"""An inhomogeneous Neumann wall contributes its boundary load to both FEM solvers. #2294

`du/dn = g` with `g != 0` owes the weak form a boundary load `D * int_dOmega g phi_i`. It is an
operator augmentation, not a condensation, so it is assembled by `bc_adapter.assemble_robin_terms`
and folded into `M/dt + D*K` through the solvers' `_robin_operator_terms` hook — the arms in
`apply_bc_to_fem_system` are correctly bare `pass`, since there is nothing there to condense.

WAS A RECORDED DEFECT PIN (#2294), retired 2026-09-10 by its own stated condition. The natural-BC
family had no assembly at all: `g` was accepted, stored on the segment, carried to the adapter and
discarded, while both solvers reported `honors_inhomogeneous_neumann = True` — inherited from
`BaseMFGSolver` and declared on neither, which is why the capability census could not see it
(#1975). `NEUMANN(g=0)` and `NEUMANN(g=5)` gave bit-identical fields. Routing the natural-BC load
through `assemble_robin_terms` moved both solvers by `6.185279e-01`, the exact figure this file
recorded as the ROBIN-vs-NEUMANN gap before the fix, so the term now assembled is the term that was
missing.

WHY THE TWO-SPELLINGS COMPARISON IS NO LONGER THE ORACLE. It was, while the fork was open: `NEUMANN`
went one way and `ROBIN(alpha=0, beta=1, g)` another. The fix routes both through one owner, and
agreement between two spellings of one call is then close to tautological — it would pass over a
broken owner, and the better the consolidation the less it proves. It is kept below for the one
thing it still separates (that the synthesized coefficients match a user-spelled Robin) and is
labelled as that, not as evidence the load is right.

`test_the_boundary_load_matches_the_boundary_measure` is what adjudicates the load now. It is an
external oracle: by partition of unity `sum_i int_dOmega phi_i = |dOmega_seg|`, so the assembled
load must sum to `D * g * |dOmega_seg|` with the measure taken from the domain's own bounds. Nothing
in it is computed by the code under test.
"""

from __future__ import annotations

import functools

import pytest

import numpy as np

from mfgarchon import Conditions, MFGProblem, Model
from mfgarchon.alg.numerical.fem.fp_fem_solver import FPFEMSolver
from mfgarchon.alg.numerical.fem.hjb_fem_solver import HJBFEMSolver
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.geometry.boundary import BCSegment, BCType, BoundaryConditions

_G = 5.0

#: The domain is the unit square and the two segments below sit on `x_min` and `x_max`, so the
#: measure the load integrates over is 1 + 1. Taken from the bounds passed to `Mesh2D`, never from
#: the mesh or the assembly — that is what keeps the oracle external.
_BOUNDARY_MEASURE = 2.0


def _problem(segments):
    import skfem

    from mfgarchon.alg.numerical.fem.mesh_adapter import skfem_to_meshdata
    from mfgarchon.geometry.meshes.mesh_2d import Mesh2D

    geometry = Mesh2D(domain_type="rectangle", bounds=(0.0, 1.0, 0.0, 1.0))
    geometry.mesh_data = skfem_to_meshdata(skfem.MeshTri.init_sqsymmetric().refined(2))
    geometry.boundary_conditions = BoundaryConditions(dimension=2, segments=segments)
    hamiltonian = SeparableHamiltonian(
        control_cost=QuadraticControlCost(control_cost=1.0),
        coupling=lambda m: 0.0 * np.asarray(m),
        coupling_dm=lambda m: 0.0 * np.asarray(m),
    )
    # The v1.0 API, not the legacy `MFGProblem(geometry=, components=, ...)`. That form is
    # deprecated, and the warning census gates on it: a new file using it adds a warning identity
    # and turns the gate red, which is #2119 working as intended.
    return MFGProblem(
        model=Model(hamiltonian=hamiltonian, sigma=0.5),
        domain=geometry,
        conditions=Conditions(m_initial=lambda p: 1.0, u_terminal=lambda p: 0.0, T=0.1),
        Nt=3,
    )


def _segments(kind: str, g: float):
    """`NEUMANN(g)` and `ROBIN(alpha=0, beta=1, g)` are the same condition, `du/dn = g`."""
    extra = {"alpha": 0.0, "beta": 1.0} if kind == "robin" else {}
    bc_type = BCType.ROBIN if kind == "robin" else BCType.NEUMANN
    return [
        BCSegment(name=name, bc_type=bc_type, value=g, boundary=boundary, **extra)
        for name, boundary in (("L", "x_min"), ("R", "x_max"))
    ]


@functools.cache
def _solve(solver_name: str, kind: str, g: float):
    """The field each solver returns under one boundary spelling.

    Both are driven by a NON-CONSTANT state, and that is a correctness requirement rather than
    taste: see `test_the_solve_is_not_degenerate`.
    """
    cls = HJBFEMSolver if solver_name == "HJBFEMSolver" else FPFEMSolver
    solver = cls(_problem(_segments(kind, g)), order=1)
    x = solver._disc.dof_coordinates[:, 0]
    steps = np.zeros((4, x.size))
    if solver_name == "HJBFEMSolver":
        return np.asarray(solver.solve_hjb_system(M_density=np.ones((4, x.size)), U_terminal=np.sin(np.pi * x)))
    return np.asarray(solver.solve_fp_system(m_initial=1.0 + 0.5 * np.sin(np.pi * x), U_solution_for_drift=steps))


_SOLVERS = ["HJBFEMSolver", "FPFEMSolver"]


@pytest.mark.parametrize("solver_name", _SOLVERS)
def test_the_solve_is_not_degenerate(solver_name):
    """POSITIVE CONTROL, and the one the obvious version of this file gets wrong.

    Several claims below are a diff coming back as zero, so a solve that returns a constant makes
    them vacuous. This is not hypothetical: with `U_terminal` left at the components' flat `0.0`,
    `HJBFEMSolver` returns a field that is identically zero — spread `0.000000e+00`, `0` of `324`
    DOFs non-zero — and every zero-diff assertion here passes for a reason that has nothing to do
    with #2294 and kept passing after it was fixed.

    An external control on a DIFFERENT condition does not close this. Dirichlet moves the field by
    `5.0` even on that degenerate solve, because it writes boundary DOFs directly rather than
    through the weak form; it reports "the fixture can express a boundary datum" while the interior
    is dead.
    """
    field = _solve(solver_name, "neumann", 0.0)

    assert np.ptp(field) > 1e-6, (
        f"{solver_name} returned a field with spread {np.ptp(field):.3e}: it is constant, so every "
        f"zero-difference assertion in this file is vacuous and none of them measures #2294."
    )
    assert np.count_nonzero(field) > field.size // 2, (
        f"{solver_name} returned {np.count_nonzero(field)} of {field.size} non-zero DOFs; the solve "
        f"is mostly dead and the comparisons below are not measuring a live weak form."
    )


@pytest.mark.parametrize("solver_name", _SOLVERS)
def test_the_boundary_load_matches_the_boundary_measure(solver_name):
    """THE ORACLE. A law the assembly must reproduce, computed without it.

    The finite-element basis is a partition of unity on the boundary, so `sum_i int_dOmega phi_i`
    is the measure of the facets integrated over — `_BOUNDARY_MEASURE`, read off the domain bounds.
    The load `D * (g/beta) * int_dOmega phi_i` must therefore sum to `D * g * |dOmega_seg|` for a
    Neumann segment, where `beta = 1` by the condition's own definition.

    What it separates, none of which the two-spellings comparison can: a wrong `D` scaling, a sign
    error, a `beta` synthesized as anything but 1, a facet set that is not the named wall, and a
    load assembled twice. It caught the last of those during the fix — an oracle run against a raw
    `skfem` mesh rather than the solver's reported 4x this figure, because `_find_segment_facets`
    silently falls back to the whole boundary on an untagged mesh. The solver path tags
    `x_min/x_max/y_min/y_max` and is unaffected, which is why this test drives the solver's own hook
    rather than calling the assembler with a hand-built basis.

    `alpha = 0` is asserted through the boundary mass: a Neumann condition has no zeroth-order term,
    so `A_robin` must carry no entries at all.
    """
    cls = HJBFEMSolver if solver_name == "HJBFEMSolver" else FPFEMSolver
    solver = cls(_problem(_segments("neumann", _G)), order=1)
    D = 0.125

    A_robin, rhs_robin = solver._robin_operator_terms(D)

    assert rhs_robin is not None, f"{solver_name} assembles no boundary load for NEUMANN(g={_G}); that is #2294."
    expected = D * _G * _BOUNDARY_MEASURE
    assert rhs_robin.sum() == pytest.approx(expected, rel=1e-12), (
        f"{solver_name}: the NEUMANN(g={_G}) boundary load sums to {rhs_robin.sum():.12f}, and the "
        f"partition of unity over a boundary of measure {_BOUNDARY_MEASURE} requires "
        f"D*g*|dOmega| = {expected:.12f}. Ratio {rhs_robin.sum() / expected:.6f}."
    )
    assert A_robin.nnz == 0, (
        f"{solver_name}: NEUMANN has no zeroth-order term, so its boundary mass must be empty, but "
        f"A_robin carries {A_robin.nnz} entries. alpha was synthesized as something other than 0."
    )


@pytest.mark.parametrize("solver_name", _SOLVERS)
def test_a_homogeneous_neumann_wall_still_assembles_nothing(solver_name):
    """The regression the fix could have caused, and the reason the g=0 arm returns early.

    Every solve written before #2294 used a homogeneous natural BC. If the new branch assembled a
    zero mass and a zero load for those instead of declining, the hook would return matrices where
    it used to return `(None, None)`, and the caller would add them — arithmetic that is a no-op in
    exact terms and not guaranteed to be one in floating point. `None` keeps those solves on
    literally the same code path they were on.
    """
    cls = HJBFEMSolver if solver_name == "HJBFEMSolver" else FPFEMSolver
    solver = cls(_problem(_segments("neumann", 0.0)), order=1)

    assert solver._robin_operator_terms(0.125) == (None, None), (
        f"{solver_name} now assembles a term for a HOMOGENEOUS Neumann wall. It contributes nothing "
        f"mathematically, so the hook must decline rather than hand back zeros: every pre-#2294 "
        f"solve used this branch and must stay bit-identical."
    )


@pytest.mark.parametrize("solver_name", _SOLVERS)
def test_the_two_spellings_agree_when_the_flux_is_zero(solver_name):
    """`NEUMANN(0)` and `ROBIN(alpha=0, beta=1, 0)` are the same condition and must still agree.

    They reach the assembler differently after the fix — Neumann declines at `g == 0` while Robin
    assembles an all-zero mass and load — so this asserts that the two routes are indistinguishable
    in the result, which is the byte-identity claim the early return above exists to protect.
    """
    neumann = _solve(solver_name, "neumann", 0.0)
    robin = _solve(solver_name, "robin", 0.0)

    assert np.array_equal(neumann, robin), (
        f"{solver_name}: NEUMANN(0) and ROBIN(alpha=0, beta=1, 0) are the same condition but differ "
        f"by {np.max(np.abs(neumann - robin)):.3e}."
    )


@pytest.mark.parametrize("solver_name", _SOLVERS)
def test_the_two_spellings_of_one_condition_agree(solver_name):
    """CONSISTENCY, not an oracle — see the module docstring.

    Both spellings now route through `assemble_robin_terms`, so this cannot adjudicate whether the
    load is right; `test_the_boundary_load_matches_the_boundary_measure` does that. What it still
    separates is the synthesis: `NEUMANN` fabricates `(alpha, beta) = (0, 1)` while `ROBIN` reads
    the pair off the segment, so a synthesis of `beta = -1` or `alpha = 1` shows up here as a
    disagreement between two spellings of one condition.
    """
    neumann = _solve(solver_name, "neumann", _G)
    robin = _solve(solver_name, "robin", _G)

    assert np.allclose(neumann, robin), (
        f"{solver_name}: NEUMANN(g={_G}) and ROBIN(alpha=0, beta=1, g={_G}) are the same condition "
        f"and differ by {np.max(np.abs(neumann - robin)):.6e}."
    )


@pytest.mark.parametrize("solver_name", _SOLVERS)
def test_the_declaration_is_now_true(solver_name):
    """The declaration half. `honors_inhomogeneous_neumann` was `True` while the value was dropped.

    It is still `True` and still declared on neither solver — inherited from `BaseMFGSolver` — but
    it is now accurate. The second assertion is kept from the pin: a sweep over own-class attributes
    sees nothing to check here, which is the blindness #1975 records, so if either solver ever
    declares the flag on its own class that is a deliberate act worth reading.
    """
    cls = HJBFEMSolver if solver_name == "HJBFEMSolver" else FPFEMSolver

    assert getattr(cls, "honors_inhomogeneous_neumann", None) is True
    assert "honors_inhomogeneous_neumann" not in vars(cls), (
        f"{cls.__name__} now declares honors_inhomogeneous_neumann on its own class. If it declares "
        f"False, an inhomogeneous Neumann has regressed; if True, the census can finally see it."
    )
