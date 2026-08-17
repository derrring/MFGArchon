"""The strict-adjoint FP-FDM step stops rather than repairing a diverged solve (#1507, #1683).

The transposed HJB advection operator is not an M-matrix, so at high Péclet the linear
solve undershoots negative. #1507 made that clip **visible** by renormalising to the
pre-step total and warning -- an improvement on the silent version before it, and still
the wrong shape: the returned array was finite, non-negative and exactly mass-conserving,
so every cheap invariant a caller might check was satisfied by the repair rather than by
the physics. Measured on this configuration, the clip discarded **8.39%** of the mass and
the result still reported exact conservation.

#1683 routes the site through `clip_nonnegative_or_raise`, which stops the solve and does
not renormalise. These tests therefore assert the opposite of what they used to.

Note what the previous assertions were: `(m_next >= 0).all()` and
`isclose(m_next.sum(), m0.sum())` are exactly the two properties clip-then-renormalise
produces **by construction** -- they held whether the step was healthy or a 90% clip. A
test that asserts the symptoms of a repair passes over the thing the repair is hiding.
"""

from __future__ import annotations

import pytest

import numpy as np
from scipy import sparse

from mfgarchon.alg.numerical.fp_solvers.fp_fdm import FPFDMSolver
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.core.mfg_problem import MFGComponents, MFGProblem
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import no_flux_bc
from mfgarchon.utils.numerical import mass_fabricated_by_clip


def _fp_solver_and_advection(n=21, drift=40.0):
    geom = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[n], boundary_conditions=no_flux_bc(dimension=1))
    comp = MFGComponents(
        m_initial=lambda x: 1.0,
        u_terminal=lambda x: 0.0,
        hamiltonian=SeparableHamiltonian(
            control_cost=QuadraticControlCost(control_cost=1.0), coupling=lambda m: m, coupling_dm=lambda m: 1.0
        ),
    )
    prob = MFGProblem(geometry=geom, T=0.2, Nt=5, sigma=0.05, components=comp, coupling_coefficient=1.0)
    fp = FPFDMSolver(prob)
    h = 1.0 / (n - 1)
    off = drift / (2 * h)
    a = sparse.diags([-off * np.ones(n - 1), np.zeros(n), off * np.ones(n - 1)], [-1, 0, 1]).tocsr()
    return fp, a, h, n


def _peaked_density(n, h):
    """Peaked density at high Péclet -- the configuration that makes the solve undershoot."""
    m0 = np.zeros(n)
    m0[n // 2] = 1.0 / h
    return m0


def test_a_diverged_strict_adjoint_step_raises_instead_of_renormalising():
    fp, a, h, n = _fp_solver_and_advection()
    with pytest.raises(ValueError, match="would fabricate"):
        fp.solve_fp_step_adjoint_mode(_peaked_density(n, h), a, sigma=0.05)


def test_the_message_names_the_fabricated_fraction_and_a_remedy():
    """A diagnostic that reports a defect without naming a next step is read as noise."""
    fp, a, h, n = _fp_solver_and_advection()
    with pytest.raises(ValueError) as exc:
        fp.solve_fp_step_adjoint_mode(_peaked_density(n, h), a, sigma=0.05)
    message = str(exc.value)
    assert "Strict-adjoint FP step" in message
    assert "%" in message, "the fabricated fraction is the quantity, and must be reported"
    assert "M-matrix" in message
    assert "Reduce dt" in message


def test_the_clip_on_this_configuration_is_large_not_marginal():
    """Pins the measurement the disposition rests on, not merely that something raised.

    Without it, lowering the threshold would satisfy the assertions above while saying
    nothing about whether the clip was ever significant.
    """
    fp, a, h, n = _fp_solver_and_advection()
    with pytest.raises(ValueError) as exc:
        fp.solve_fp_step_adjoint_mode(_peaked_density(n, h), a, sigma=0.05)
    percent = float(str(exc.value).split("would fabricate")[1].split("%")[0])
    assert percent > 1.0, f"expected a large clip on this configuration, message reported {percent}%"


def test_the_step_no_longer_renormalises():
    """Restoring the pre-clip total is what made this class of defect invisible.

    On a healthy step nothing is clipped, so the returned mass is whatever the operator
    produced rather than being forced back to the input total.
    """
    fp, a, _h, n = _fp_solver_and_advection(drift=0.0)
    m0 = np.full(n, 1.0)
    m_next = fp.solve_fp_step_adjoint_mode(m0, a, sigma=0.05)
    assert mass_fabricated_by_clip(m_next) == 0.0
    # The old code multiplied by mass_before / mass_after_clip, which forced this to hold
    # exactly regardless of what the operator did. It is no longer imposed; whether it
    # happens to hold is now a property of the step.
    assert np.isfinite(float(m_next.sum()))
