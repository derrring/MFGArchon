"""The SL boundary fold speaks the mapping's vocabulary, and periodic feet actually wrap.

Issue #1739. Three sites in `hjb_semi_lagrangian.py` dispatched on `bc_op == "wrap"`, a
spelling `bc_type_to_geometric_operation` has never produced -- its alphabet is
`{'reflect', 'periodic', 'clamp'}`. All three branches were unreachable, so every periodic
foot fell through to a clamp or an extrapolation. No exception and no warning: the solve
returned a value function for boundary conditions the problem did not declare.

The oracle here is periodicity itself, not agreement between two of our own code paths: on a
periodic domain `x = 0` and `x = 1` are the same physical point, so `u(t, 0) == u(t, 1)` is a
property of the continuous problem that no discretisation may violate at O(1). Measured on
`diffusion_method='canonical_cs'`, the seam went from 8.67e-01 to 2.45e-16.
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
    geometry = TensorProductGrid(
        bounds=[(0.0, 1.0)], Nx_points=[NX], boundary_conditions=periodic_bc(dimension=1)
    )
    problem = MFGProblem(geometry=geometry, T=0.5, Nt=10, components=components, sigma=0.3)
    return HJBSemiLagrangianSolver(problem, interpolation_method="linear", **kwargs)


def _seam(solver) -> float:
    """max |u(t, x_min) - u(t, x_max)| over the solve -- zero for a true periodic solution."""
    u_terminal = np.sin(2 * np.pi * np.linspace(0.0, 1.0, NX))
    U = solver.solve_hjb_system(np.ones((11, NX)), u_terminal, np.zeros((11, NX)))
    assert np.isfinite(U).all(), "solve produced non-finite values; the seam number would be meaningless"
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
    """`wrap` is the exact spelling that sat dead at three sites for the life of the bug.

    A fall-through would make the next vocabulary drift as quiet as this one was.
    """
    with pytest.raises(ValueError, match="unknown geometric boundary operation"):
        fold_into_domain(np.array([1.2]), 0.0, 1.0, "wrap")


def test_periodic_fold_wraps_and_matches_the_independent_scalar_implementation():
    """`apply_boundary_conditions_1d` wrapped correctly all along; the vectorised sites did not.

    It is a separate implementation with separate call sites, so this is a real cross-check
    rather than a tautology -- and clamping is shown to be a different answer, so an
    accidental revert to `np.clip` cannot pass.
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
        # canonical_cs folds every departure, so the seam closes to round-off (was 8.67e-01).
        ("canonical_cs", 1e-12),
        # stochastic leaves a discretisation-scale residual at sigma=0.3 (was 1.82e+00).
        ("stochastic", 1e-2),
    ],
)
def test_a_periodic_solve_has_no_seam(diffusion_method, tolerance):
    """x_min and x_max are the same physical point, so u must agree there.

    Independent of the scheme: this is a property of the continuous problem, which is what
    makes it survive a later consolidation of the folds it exercises.
    """
    seam = _seam(_periodic_solver(diffusion_method=diffusion_method))
    assert seam < tolerance, (
        f"periodic solve under diffusion_method={diffusion_method!r} left a seam of {seam:.3e} "
        f"between u(t, x_min) and u(t, x_max); the departure fold is not wrapping"
    )
