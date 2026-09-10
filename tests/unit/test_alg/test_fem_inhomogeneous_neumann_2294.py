"""Both FEM solvers declare `honors_inhomogeneous_neumann` and drop the flux value. #2294

RECORDED DEFECT, not a contract. `bc_adapter.apply_bc_to_fem_system` has one arm for the natural-BC
family — `BCType.NEUMANN`, `NO_FLUX`, `REFLECTING` — whose body is a bare `pass` under the comment
"Natural BC — no action needed in weak form". That is true for a HOMOGENEOUS Neumann condition and
false for an inhomogeneous one: `beta * du/dn = g` with `g != 0` contributes a boundary load
`(D/beta) * int_dOmega g phi_i` to the weak form, which nothing assembles.

The `ROBIN` arm immediately below it is ALSO a bare `pass`, and saying otherwise sends a reader to
the wrong place. Robin is an operator augmentation rather than a condensation, so its boundary mass
and load are built by `bc_adapter.assemble_robin_terms` and folded in upstream through the solvers'
`_robin_operator_terms` hook. That function — not either arm — is where the natural-BC load has to be
routed, and it is the line the retirement below actually changes.

So `g` is accepted, stored on the segment, carried to the adapter, and discarded — while both
solvers report `honors_inhomogeneous_neumann = True`. Neither declares it; both inherit `True` from
`BaseMFGSolver`, which is why the capability census does not catch it (#1975 records that an
inherited default is invisible to an "own values" sweep).

THE ORACLE IS INTERNAL, so this file needs no analytic solution. `ROBIN(alpha=0, beta=1, g)` IS
`beta * du/dn = g` — the same mathematical condition as `NEUMANN(g)`, spelled onto the arm that
assembles. The two must agree for every `g`. Measured 2026-09-10, they agree bit-for-bit at `g = 0`
and differ by `6.185279e-01` at `g = 5`, and the whole of that difference is the dropped term.

Retirement: assemble the natural-BC load. `test_the_two_spellings_of_one_condition_agree` reports
XPASS(strict) and `test_the_inhomogeneous_neumann_value_is_still_dropped` fails carrying the
instruction to delete it.
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

    Every claim below is a diff coming back as zero, so a solve that returns a constant makes all of
    them vacuous. This is not hypothetical: with `U_terminal` left at the components' flat `0.0`,
    `HJBFEMSolver` returns a field that is identically zero — spread `0.000000e+00`, `0` of `324`
    DOFs non-zero — and every zero-diff assertion here passes for a reason that has nothing to do
    with #2294 and would keep passing after it is fixed.

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
def test_the_two_spellings_agree_when_the_flux_is_zero(solver_name):
    """CONTROL for the defect below, and it is what makes the comparison legitimate.

    `NEUMANN(0)` and `ROBIN(alpha=0, beta=1, 0)` are the same condition at zero flux, and they come
    back bit-identical. Without this, a difference at `g = 5` could just be two code paths that
    disagree in general; with it, the two paths ARE one condition and the entire disagreement is the
    term that only one of them assembles.
    """
    neumann = _solve(solver_name, "neumann", 0.0)
    robin = _solve(solver_name, "robin", 0.0)

    assert np.array_equal(neumann, robin), (
        f"{solver_name}: NEUMANN(0) and ROBIN(alpha=0, beta=1, 0) are the same condition but differ "
        f"by {np.max(np.abs(neumann - robin)):.3e}. The comparison this file rests on is invalid; "
        f"fix this before reading anything below."
    )


@pytest.mark.parametrize(
    "solver_name",
    [
        pytest.param(
            name, marks=pytest.mark.xfail(strict=True, reason=f"#2294: {name} drops the inhomogeneous Neumann value")
        )
        for name in _SOLVERS
    ],
)
def test_the_two_spellings_of_one_condition_agree(solver_name):
    """THE CONTRACT. `du/dn = g` must not depend on which `BCType` spells it.

    Retires by XPASS(strict) the moment the natural-BC load is assembled.
    """
    neumann = _solve(solver_name, "neumann", _G)
    robin = _solve(solver_name, "robin", _G)

    assert np.allclose(neumann, robin), (
        f"{solver_name}: NEUMANN(g={_G}) and ROBIN(alpha=0, beta=1, g={_G}) are the same condition "
        f"and differ by {np.max(np.abs(neumann - robin)):.6e}."
    )


@pytest.mark.parametrize("solver_name", _SOLVERS)
def test_the_inhomogeneous_neumann_value_is_still_dropped(solver_name):
    """RECORDED DEFECT (#2294). Asserts the WRONG behaviour on purpose.

    Bit-identity rather than a tolerance: the value is not approximately ignored, it never reaches
    an assembly at all, so the two solves are the same floating-point computation. A tolerance here
    would also pass for a value that was assembled and merely small, which is a different bug.
    """
    zero_flux = _solve(solver_name, "neumann", 0.0)
    big_flux = _solve(solver_name, "neumann", _G)

    if not np.array_equal(zero_flux, big_flux):
        pytest.fail(
            f"{solver_name} now responds to the inhomogeneous Neumann value: max|diff| = "
            f"{np.max(np.abs(zero_flux - big_flux)):.6e}. That is the #2294 fix. Delete this test, "
            f"and remove the xfail from test_the_two_spellings_of_one_condition_agree."
        )


@pytest.mark.parametrize("solver_name", _SOLVERS)
def test_the_declaration_says_it_honours_what_it_drops(solver_name):
    """The declaration half, and why no census caught this.

    `honors_inhomogeneous_neumann` is `True` on both solvers and DECLARED on neither — it is
    inherited from `BaseMFGSolver`. A sweep over own-class attributes sees nothing to check, which
    is the blindness #1975 already records. Retires with the test above: once the load is assembled
    the declaration becomes true and only this docstring needs deleting.
    """
    cls = HJBFEMSolver if solver_name == "HJBFEMSolver" else FPFEMSolver

    assert getattr(cls, "honors_inhomogeneous_neumann", None) is True
    assert "honors_inhomogeneous_neumann" not in vars(cls), (
        f"{cls.__name__} now declares honors_inhomogeneous_neumann on its own class. If it declares "
        f"False, this file's defect is disclosed rather than fixed and the xfail above should say "
        f"so; if True, the census can finally see it."
    )
