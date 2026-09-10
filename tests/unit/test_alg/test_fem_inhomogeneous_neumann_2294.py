"""An inhomogeneous Neumann wall reaches the HJB weak form, and the FP weak form refuses it. #2294

`du/dn = g` with `g != 0` owes the weak form a boundary load `D * int_dOmega g phi_i`. It is an
operator augmentation, not a condensation, so it is assembled by `bc_adapter.assemble_robin_terms`
and folded into `M/dt + D*K` through `_robin_operator_terms`.

THE TWO HALVES ARE NOT THE SAME CONDITION, and this file exists as much for that as for the load.

- **HJB** integrates only `-D*Delta u` by parts; `H` is a mass term, not a divergence. The boundary
  term it leaves is `-D*du/dn`, so its natural condition constrains the GRADIENT and `NEUMANN(g)` is
  exactly `ROBIN(alpha=0, beta=1, g)`.
- **FP** assembles `div(v m)` on the volume basis with no facet term (`_build_advection` returns
  `-C.T`). The boundary term it leaves is the TOTAL FLUX `J.n = v m - D grad m`. Adding the same
  load there would impose `J.n = -D*g`, which coincides with `dm/dn = g` only where the drift has no
  normal component at the wall. The refusal is keyed on the SPELLING: `ROBIN(alpha=0, beta=1, g)`
  still assembles on the FP side and gives a load bit-identical to the refused one. Pre-existing,
  disclosed under #1237, filed rather than widened here.

Measured 2026-09-10, driving `FPFEMSolver` with a wall-crossing drift: at `g=0` mass is conserved to
`6.7e-15` where `dm/dn = 0` would leak by `-int (a.n) m`, and at `g=5` the injection rate is exactly
`D*g*|dOmega| = 1.25` and IDENTICAL with and without drift, where the gradient reading would differ
by `int (a.n) m`. So `FPFEMSolver` now declares `honors_inhomogeneous_neumann = False` and the #1686
gate refuses the problem before the solve rather than solving a different one.

WAS A RECORDED DEFECT PIN, retired 2026-09-10 by its own stated condition: the natural-BC family had
no assembly at all, `g` was accepted and discarded, and both solvers claimed
`honors_inhomogeneous_neumann = True`. Routing the load moved both by `6.185279e-01`, the figure this
file recorded as the ROBIN-vs-NEUMANN gap.

WHY THE TWO-SPELLINGS COMPARISON IS NOT THE ORACLE. It was, while the fork was open. Once both
spellings route through one owner, agreement is close to tautological -- it would pass over a broken
owner. `test_the_boundary_load_is_the_partition_of_unity_on_the_named_walls` adjudicates the load
now: the FEM basis is a partition of unity, so on a wall of uniform facet length `h` the load is
`D*g*h` at interior wall nodes and half that at the two end nodes, summing to `D*g*|dOmega_seg|` --
every number taken from the domain's own bounds, none from the code under test.

It pins the load's SUPPORT, its DISTRIBUTION and its TOTAL, because an adversarial review found that
pinning the total alone let three mutations through: a sum-preserving uniform lump, moving the load
to the opposite pair of equal-measure walls, and flipping the sign where the solver consumes it. The
first two are killed here, measured. The third is not, and cannot be: this file stops at
`_robin_operator_terms`, which returns the assembled terms and says nothing about how a solver adds
them. The sign at the point of consumption is pinned by `test_fem_robin_bc.py::TestRobinSolveLoopWiring`
-- `test_hjb_linear_loop_fixed_point` and `test_hjb_newton_loop_fixed_point` both go red when the two
HJB fold-in sites are negated, and `test_fp_forward_loop_fixed_point` covers the FP side. Named rather
than duplicated.
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

#: The domain is the unit square and the two segments sit on `x_min` and `x_max`, so the measure the
#: load integrates over is the sum of their Y-EXTENTS, 1 + 1. Taken from the bounds passed to
#: `Mesh2D`, never from the mesh or the assembly -- that is what keeps the oracle external. Named for
#: the y-extent on purpose: changing the y-bounds invalidates it and changing the x-bounds does not.
_WALL_Y_EXTENT = 1.0
_BOUNDARY_MEASURE = 2 * _WALL_Y_EXTENT

#: `MeshTri.init_sqsymmetric().refined(2)` puts 8 facets of equal length on each wall.
_FACETS_PER_WALL = 8
_H = _WALL_Y_EXTENT / _FACETS_PER_WALL


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
    """`NEUMANN(g)` and `ROBIN(alpha=0, beta=1, g)` are the same condition for the HJB weak form."""
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
    if solver_name == "HJBFEMSolver":
        return np.asarray(solver.solve_hjb_system(M_density=np.ones((4, x.size)), U_terminal=np.sin(np.pi * x)))
    # `potential_field` is the drift channel and is the parameter this solver actually has. An
    # earlier version passed `U_solution_for_drift=`, which `solve_fp_system` swallows through
    # `**kwargs`, so every FP arm ran at ZERO DRIFT -- and zero drift is precisely the regime in
    # which `J.n = 0` and `dm/dn = 0` coincide, i.e. the one configuration that cannot see #2294's
    # FP half. Renaming the parameter alone did not fix that; a value is needed. `U = -x` gives
    # `alpha* = -grad U = (+1, 0)`, so the wall-normal component is non-zero at both x walls.
    return np.asarray(
        solver.solve_fp_system(m_initial=1.0 + 0.5 * np.sin(np.pi * x), potential_field=np.tile(-x, (4, 1)))
    )


_SOLVERS = ["HJBFEMSolver", "FPFEMSolver"]


@pytest.mark.parametrize("solver_name", _SOLVERS)
def test_the_solve_is_not_degenerate(solver_name):
    """POSITIVE CONTROL, and the one the obvious version of this file gets wrong.

    Several claims here are a diff coming back as zero, so a solve that returns a constant makes
    them vacuous. This is not hypothetical: with `U_terminal` left at the components' flat `0.0`,
    `HJBFEMSolver` returns a field that is identically zero -- spread `0.000000e+00`, `0` of `324`
    DOFs non-zero -- and every zero-diff assertion here passes for a reason that has nothing to do
    with #2294.

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


def test_the_boundary_load_is_the_partition_of_unity_on_the_named_walls():
    """THE ORACLE. A law the assembly must reproduce, computed without it. HJB only -- see the FP test.

    The FEM basis is a partition of unity on the boundary, so on a wall of uniform facet length `h`
    the load `D*g*int phi_i` is `D*g*h` at an interior wall node, half that at each of the two end
    nodes, and `D*g*|dOmega_seg|` in total. Everything on the right-hand side comes from the domain
    bounds and the facet count; nothing comes from the code under test.

    THREE THINGS ARE PINNED, and an adversarial review is why it is three rather than one. Pinning
    the total alone let a sum-preserving uniform lump through, and let the load move to `y_min`/`y_max`
    -- an equal-measure pair -- with all twelve tests passing. So:

    * SUPPORT -- non-zero only on DOFs of the named walls (`x = 0` or `x = 1`);
    * DISTRIBUTION -- exactly two distinct magnitudes, in ratio 2:1;
    * TOTAL -- `D*g*|dOmega_seg|`.

    `alpha = 0` is asserted through the boundary mass: a Neumann condition has no zeroth-order term,
    so `A_robin` must carry no entries.
    """
    solver = HJBFEMSolver(_problem(_segments("neumann", _G)), order=1)
    D = 0.125
    A_robin, rhs = solver._robin_operator_terms(D)

    assert rhs is not None, f"HJBFEMSolver assembles no boundary load for NEUMANN(g={_G}); that is #2294."

    x = solver._disc.dof_coordinates[:, 0]
    on_named_walls = np.isclose(x, 0.0) | np.isclose(x, 1.0)
    loaded = ~np.isclose(rhs, 0.0)

    assert not np.any(loaded & ~on_named_walls), (
        f"the load is non-zero on {int(np.sum(loaded & ~on_named_walls))} DOF(s) that are not on "
        f"x_min or x_max. It is being integrated over facets other than the named segments."
    )
    assert np.array_equal(loaded, on_named_walls), (
        f"{int(np.sum(on_named_walls & ~loaded))} DOF(s) on the named walls carry no load; the "
        f"facet set is a subset of the segments, not the segments."
    )

    magnitudes = np.sort(np.unique(np.round(rhs[loaded], 12)))
    assert magnitudes.size == 2, (
        f"expected exactly two distinct load magnitudes (interior wall node and end node), got "
        f"{magnitudes.size}: {magnitudes}. A uniform load over the same facets has one."
    )
    assert magnitudes[1] == pytest.approx(2 * magnitudes[0], rel=1e-12), (
        f"the two magnitudes {magnitudes} are not in ratio 2:1. An end node owns half a facet."
    )
    assert magnitudes[1] == pytest.approx(D * _G * _H, rel=1e-12), (
        f"interior wall node carries {magnitudes[1]:.12f}, and the partition of unity over a facet "
        f"of length {_H} requires D*g*h = {D * _G * _H:.12f}."
    )

    expected_total = D * _G * _BOUNDARY_MEASURE
    assert rhs.sum() == pytest.approx(expected_total, rel=1e-12), (
        f"the load sums to {rhs.sum():.12f}; the partition of unity over a boundary of measure "
        f"{_BOUNDARY_MEASURE} requires D*g*|dOmega| = {expected_total:.12f}."
    )
    assert A_robin.nnz == 0, (
        f"NEUMANN has no zeroth-order term, so its boundary mass must be empty, but A_robin carries "
        f"{A_robin.nnz} entries. alpha was synthesized as something other than 0."
    )


def test_the_fp_solver_refuses_the_neumann_spelling():
    """#2294's other half, named for what it actually checks.

    NOT "the FP weak form refuses the condition": it refuses this SPELLING. `ROBIN(alpha=0, beta=1, g)`
    is the same mathematical condition and still reaches the FP assembly, producing a load vector
    bit-identical to the one refused here (`max|diff| = 0.0`, measured). That path is pre-existing,
    has its own tests, and `weak_form_fp_solver.py` already discloses the total-flux character of an
    inhomogeneous Robin as out of scope for #1237 -- so it is filed, not widened here.

    What this test covers is ONE enforcement path: `_validate_bc_support`, reached from
    `FPFEMSolver.__init__`, refusing on `honors_inhomogeneous_neumann = False`. It does not reach
    `assemble_robin_terms`; `test_the_natural_bc_parameter_gates_the_neumann_arm` covers that.

    `_build_advection` assembles `div(v m)` on the volume basis and returns `-C.T` with no facet
    term, so this weak form's natural boundary condition is the total flux `J.n`. Adding the HJB
    load would impose `J.n = -D*g` while the caller wrote `dm/dn = g` -- equal only where the drift
    has no normal component at the wall, which is exactly the configuration a test is most likely to
    use and least likely to notice.

    Retirement: if FP ever gains the advection facet term (#1237), it can honour the condition, and
    then `honors_inhomogeneous_neumann` goes back to True and this test says so by failing.
    """
    with pytest.raises((NotImplementedError, ValueError)) as excinfo:
        FPFEMSolver(_problem(_segments("neumann", _G)), order=1)

    assert not FPFEMSolver.honors_inhomogeneous_neumann, (
        "FPFEMSolver declares it honours an inhomogeneous Neumann again. If the advection facet "
        "term now exists, delete this test; if not, the declaration is false and #1686's gate is "
        "no longer refusing the problem."
    )
    assert "NEUMANN" in str(excinfo.value).upper()


def test_the_natural_bc_parameter_gates_the_neumann_arm():
    """The new mechanism, pinned at the level it lives on. An adversarial review found it unpinned.

    `test_the_fp_solver_refuses_the_neumann_spelling` never reaches `assemble_robin_terms` -- the
    #1686 gate stops the solve at construction -- so every mutation of the new machinery survived it:
    neutering the `natural_bc != "gradient"` check, flipping `FPFEMSolver`'s call to `"gradient"`, and
    flipping the parameter default. A test named for a mechanism that is satisfied by a different one
    is the failure this file exists to avoid, one level up. So this calls the function directly.
    """
    from mfgarchon.alg.numerical.fem.bc_adapter import assemble_robin_terms

    # The SOLVER's basis, never a hand-built one. `MeshTri.init_sqsymmetric()` carries no boundary
    # tags, so `_find_segment_facets` falls back to the whole perimeter (#2301) and a named wall
    # silently becomes the boundary -- which is 4x the load and reads as a defect in the assembly.
    # That fallback cost this branch a false accusation against correct code once already.
    solver = HJBFEMSolver(_problem(_segments("neumann", _G)), order=1)
    basis, bc = solver._basis, solver._bc

    _, rhs = assemble_robin_terms(basis, bc, 0.125, natural_bc="gradient")
    assert rhs is not None, "the gradient-natural weak form must assemble the load"
    assert rhs.sum() == pytest.approx(0.125 * _G * _BOUNDARY_MEASURE, rel=1e-12)

    with pytest.raises(NotImplementedError, match="TOTAL FLUX"):
        assemble_robin_terms(basis, bc, 0.125, natural_bc="flux")

    # The default is the RESTRICTIVE one on purpose: a caller who forgets gets the refusal, not
    # silently wrong physics. Flipping it to "gradient" is a one-word change with no other symptom.
    with pytest.raises(NotImplementedError, match="TOTAL FLUX"):
        assemble_robin_terms(basis, bc, 0.125)

    # FPFEMSolver's OWN call site, reached by a route production does not take -- and deliberately.
    # `_validate_bc_support` refuses an inhomogeneous Neumann at construction, so the backstop below
    # it is unreachable through the public path and a mutation of the argument survives every
    # behavioural test (measured). Constructing on a homogeneous wall and then swapping the bc is
    # the only way to exercise the second layer, and a backstop nothing exercises is a comment.
    fp = FPFEMSolver(_problem(_segments("neumann", 0.0)), order=1)
    fp._bc = bc
    with pytest.raises(NotImplementedError, match="TOTAL FLUX"):
        fp._robin_operator_terms(0.125)


def test_the_public_factory_path_assembles():
    """The regression a guard for the `default_bc` channel caused, twice, and the reason it is gone.

    Every `*_bc()` factory sets BOTH channels in lockstep: `neumann_bc(5.0)` yields a segment AND
    `default_bc=NEUMANN, default_value=5.0`. A guard reading the default channel alone therefore
    rejects the library's own public API -- measured, it did, and nothing here saw it because these
    tests build `BoundaryConditions(segments=...)` directly and leave `default_bc` None, the one
    shape such a guard cannot see. Its replacement asked whether the segments COVER the boundary and
    under-refused on `boundary="left"`, an alias `parse_boundary_face` resolves and the raw lookup
    does not. Both are removed; the fall-through gap is disclosed and filed.

    What survives is this: the documented way of asking for the condition must assemble it.
    """
    from mfgarchon.alg.numerical.fem.bc_adapter import assemble_robin_terms
    from mfgarchon.geometry.boundary import neumann_bc

    basis = HJBFEMSolver(_problem(_segments("neumann", 0.0)), order=1)._basis

    _, rhs = assemble_robin_terms(basis, neumann_bc(_G, dimension=2), 0.125, natural_bc="gradient")
    assert rhs is not None, "neumann_bc(g) is the library's public way to ask for this condition and it must assemble."
    assert rhs.sum() == pytest.approx(0.125 * _G * 4.0, rel=1e-12), (
        "the factory's segment carries no `boundary`, so it spans the whole boundary -- perimeter 4, "
        "not the 2 of the two named walls."
    )


@pytest.mark.parametrize(
    "value",
    [0.0, -0.0, np.float64(0.0), np.float32(0.0), np.int64(0), np.array(0.0)],
    ids=["float", "neg_zero", "float64", "float32", "int64", "array"],
)
@pytest.mark.parametrize("solver_name", _SOLVERS)
def test_a_homogeneous_neumann_wall_assembles_nothing_whatever_its_dtype(solver_name, value):
    """The regression the fix could have caused, and an adversarial review did find in a first cut.

    Every solve written before #2294 used a homogeneous natural BC. Two ways to break them: assemble
    a zero mass and load instead of declining, so the caller adds matrices where it used to add
    nothing; or decide "is this zero" with `isinstance(g, (int, float))`, which is False for
    `np.float32(0.0)` and turned a homogeneous wall into a raise. The predicate has one owner,
    `bc_utils.describe_inhomogeneous_bc_data`, and this asserts the arm uses it.
    """
    cls = HJBFEMSolver if solver_name == "HJBFEMSolver" else FPFEMSolver
    solver = cls(_problem(_segments("neumann", 0.0)), order=1)
    solver._bc.segments[0].value = value

    assert solver._robin_operator_terms(0.125) == (None, None), (
        f"{solver_name} assembles a term for a HOMOGENEOUS Neumann wall valued "
        f"{value!r} ({type(value).__name__}). It contributes nothing mathematically, so the hook "
        f"must decline rather than hand back zeros, and it must recognise zero whatever its dtype."
    )


@pytest.mark.parametrize("solver_name", _SOLVERS)
def test_the_two_spellings_agree_when_the_flux_is_zero(solver_name):
    """CONTROL, restored -- it was deleted in the previous round without replacement.

    At `g = 0` the two spellings reach the assembler by different routes: Neumann declines at the
    verifiably-zero check while Robin assembles an all-zero mass and load. The results must be
    indistinguishable, which is the byte-identity claim the early return exists to protect. It also
    holds on BOTH solvers, where the `g != 0` comparison is HJB-only -- so this is the only place the
    FP half of the spelling equivalence is still checked.
    """
    neumann = _solve(solver_name, "neumann", 0.0)
    robin = _solve(solver_name, "robin", 0.0)

    assert np.array_equal(neumann, robin), (
        f"{solver_name}: NEUMANN(0) and ROBIN(alpha=0, beta=1, 0) are the same condition but differ "
        f"by {np.max(np.abs(neumann - robin)):.3e}."
    )


def test_the_two_spellings_of_one_condition_agree():
    """CONSISTENCY, not an oracle -- see the module docstring. HJB only.

    Both spellings route through `assemble_robin_terms`, so this cannot adjudicate whether the load
    is right. What it still separates is the SYNTHESIS: `NEUMANN` fabricates `(alpha, beta) = (0, 1)`
    while `ROBIN` reads the pair off the segment, so a synthesis of `beta = -1` or `alpha = 1` shows
    up here. A review confirmed the scope precisely: mutations confined to the Neumann branch are
    killed by this test, and the tautology is exactly co-extensive with mutations that move both.
    """
    neumann = _solve("HJBFEMSolver", "neumann", _G)
    robin = _solve("HJBFEMSolver", "robin", _G)

    assert np.allclose(neumann, robin), (
        f"HJBFEMSolver: NEUMANN(g={_G}) and ROBIN(alpha=0, beta=1, g={_G}) are the same condition "
        f"for this weak form and differ by {np.max(np.abs(neumann - robin)):.6e}."
    )


def test_each_solver_declares_the_capability_it_actually_has():
    """The declaration half, and why no census caught the original defect.

    `honors_inhomogeneous_neumann` was `True` on both solvers while the value was dropped, and
    DECLARED on neither -- inherited from `BaseMFGSolver`, so a sweep over own-class attributes saw
    nothing to check, which is the blindness #1975 records. Now HJB inherits a `True` that is
    finally accurate and FP declares its own `False`, which is what makes #1686's gate fire.
    """
    assert HJBFEMSolver.honors_inhomogeneous_neumann is True
    assert "honors_inhomogeneous_neumann" not in vars(HJBFEMSolver), (
        "HJBFEMSolver now declares the flag on its own class. If it declares False, an "
        "inhomogeneous Neumann has regressed; if True, the census can finally see it."
    )
    assert vars(FPFEMSolver).get("honors_inhomogeneous_neumann") is False, (
        "FPFEMSolver must declare False on its OWN class, not inherit True: its natural boundary "
        "condition is the total flux, so it cannot impose dm/dn = g (#2294)."
    )
