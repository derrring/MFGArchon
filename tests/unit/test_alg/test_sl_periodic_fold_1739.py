"""The SL boundary fold speaks the mapping's vocabulary, and periodic feet actually wrap.

Issue #1739.

Oracle: periodicity itself, not agreement between two of our own code paths. On a periodic
domain `x = 0` and `x = 1` are the same physical point, so `u(t, 0) == u(t, 1)` is a property
of the continuous problem that no discretisation may violate at O(1). It therefore survives a
later consolidation of the folds it exercises, which a path-A-vs-path-B test would not.
"""

import pytest

import numpy as np

from mfgarchon.alg.numerical.hjb_solvers import HJBSemiLagrangianSolver
from mfgarchon.alg.numerical.hjb_solvers.hjb_sl_characteristics import (
    apply_boundary_conditions_1d,
    fold_into_domain,
)
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.core.mfg_components import MFGComponents
from mfgarchon.core.mfg_problem import MFGProblem
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import periodic_bc
from mfgarchon.geometry.boundary.bc_utils import bc_type_to_geometric_operation
from mfgarchon.geometry.boundary.types import BCType

NX = 21


def _periodic_solver(**kwargs):
    components = MFGComponents(
        m_initial=lambda x: 1.0 + 0.0 * x,
        u_terminal=lambda x: np.sin(2 * np.pi * x),
        hamiltonian=SeparableHamiltonian(
            control_cost=QuadraticControlCost(control_cost=1.0),
            coupling=lambda m: m,
            coupling_dm=lambda m: 1.0,
        ),
    )
    geometry = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[NX], boundary_conditions=periodic_bc(dimension=1))
    problem = MFGProblem(geometry=geometry, T=0.5, Nt=10, components=components, sigma=0.3)
    return HJBSemiLagrangianSolver(problem, interpolation_method="linear", **kwargs)


def _seam(solver, monkeypatch) -> float:
    """max |u(t, x_min) - u(t, x_max)| over the solve -- zero for a true periodic solution.

    Carries its own positive control, because a seam of zero has two causes: the fold wraps,
    or the solve never went anywhere. A solver that returns its terminal data untouched has a
    seam of exactly zero and would satisfy the assertion while measuring nothing. So this
    counts the feet that actually left the domain, and refuses to report a seam unless the
    fold was exercised and the value function moved.
    """
    import mfgarchon.alg.numerical.hjb_solvers.hjb_semi_lagrangian as solver_mod

    real_fold = solver_mod.fold_into_domain
    outside = 0

    def counting_fold(x, lo, hi, bc_op):
        nonlocal outside
        outside += int(np.sum((x < np.min(lo)) | (x > np.max(hi))))
        return real_fold(x, lo, hi, bc_op)

    monkeypatch.setattr(solver_mod, "fold_into_domain", counting_fold)

    u_terminal = np.sin(2 * np.pi * np.linspace(0.0, 1.0, NX))
    U = solver.solve_hjb_system(np.ones((11, NX)), u_terminal, np.zeros((11, NX)))

    assert np.isfinite(U).all(), "solve produced non-finite values; the seam number would be meaningless"
    assert outside > 0, (
        "no departure foot left the domain, so the fold was never asked to wrap anything and "
        "a zero seam would say nothing about it"
    )
    assert not np.allclose(U[0], u_terminal), (
        "u at t=0 equals the terminal data: the solve did not evolve, and its seam is inherited "
        "from the input rather than produced by the scheme"
    )
    return float(np.abs(U[:, 0] - U[:, -1]).max())


# ---------------------------------------------------------------------------
# The vocabulary, which is what actually went wrong
# ---------------------------------------------------------------------------


def test_the_fold_accepts_every_operation_the_mapping_can_emit():
    """The pin that would have caught #1739 on the day it was written.

    Producer and consumer of one dispatch vocabulary, checked against each other rather
    than each against its own docstring. A fourth spelling on either side fails here.
    """
    emitted = {bc_type_to_geometric_operation(member.value) for member in BCType}
    emitted |= {bc_type_to_geometric_operation(None)}
    points = np.array([-0.15, 0.5, 1.2])
    for op in sorted(emitted):
        folded = fold_into_domain(points, 0.0, 1.0, op)
        assert np.all((folded >= 0.0) & (folded <= 1.0)), (
            f"fold_into_domain({op!r}) left points outside the domain: {folded}"
        )


def test_an_unrecognised_operation_raises_instead_of_silently_clamping():
    """A fall-through here would make the next vocabulary drift silent. Do not add one."""
    with pytest.raises(ValueError, match="unknown geometric boundary operation"):
        fold_into_domain(np.array([1.2]), 0.0, 1.0, "wrap")


def test_periodic_fold_wraps_and_matches_the_independent_scalar_implementation():
    """`apply_boundary_conditions_1d` is a separate implementation with separate call sites,
    so this is a real cross-check rather than a tautology. Clamping is asserted to be a
    different answer, so a revert to `np.clip` cannot pass.
    """
    points = np.array([-0.15, 1.2, 0.5])
    folded = fold_into_domain(points, 0.0, 1.0, "periodic")
    scalar = np.array([apply_boundary_conditions_1d(float(v), 0.0, 1.0, bc_type="periodic") for v in points])

    np.testing.assert_allclose(folded, scalar, atol=1e-12)
    np.testing.assert_allclose(folded, [0.85, 0.2, 0.5], atol=1e-12)
    assert not np.allclose(folded, np.clip(points, 0.0, 1.0)), "clamping must not be indistinguishable from wrapping"


# ---------------------------------------------------------------------------
# The external oracle: a periodic solution has no seam
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("diffusion_method", "tolerance"),
    [
        # canonical_cs folds every departure, so the seam closes to round-off.
        ("canonical_cs", 1e-12),
        # stochastic leaves a discretisation-scale residual at sigma=0.3.
        ("stochastic", 1e-2),
    ],
)
def test_a_periodic_solve_has_no_seam(diffusion_method, tolerance, monkeypatch):
    """x_min and x_max are the same physical point, so u must agree there.

    Independent of the scheme: this is a property of the continuous problem, which is what
    makes it survive a later consolidation of the folds it exercises.
    """
    seam = _seam(_periodic_solver(diffusion_method=diffusion_method), monkeypatch)
    assert seam < tolerance, (
        f"periodic solve under diffusion_method={diffusion_method!r} left a seam of {seam:.3e} "
        f"between u(t, x_min) and u(t, x_max); the departure fold is not wrapping"
    )
