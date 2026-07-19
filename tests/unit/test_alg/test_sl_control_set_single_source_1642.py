"""The SL DPP control sweep must read the admissible set from its single owner (Issue #1642, B3).

``_solve_timestep_dpp`` used to re-derive the admissible control set A with its own
isinstance ladder -- a hardcoded ``(-1.0, 1.0)`` for ``L1ControlCost`` and a private
``cc.max_control`` read for ``BoundedControlCost``. That made it a SECOND owner of A
alongside ``ControlCostBase.effective_domain()``: the two agreed at the time, so the
defect was drift risk rather than a wrong answer, plus one live blind spot -- a
Moreau-Yosida-wrapped cost matched none of the branches and silently lost its box.

This path is live, not dead: ``_use_dpp`` requires ``not H.is_smooth()``, which holds for
plain ``BoundedControlCost`` and ``L1ControlCost``.

The sweep now reads A from ``effective_domain()`` and keeps only the QUADRATURE decision
locally (three points for a piecewise-linear L_ctrl, eleven for a quadratic one) -- a
genuine structural distinction, not a duplicated quantity. Narrowing that last isinstance
to a control-cost capability is Issue #1651.
"""

from __future__ import annotations

import numpy as np

from mfgarchon import Conditions, MFGProblem, Model
from mfgarchon.alg.numerical.hjb_solvers import HJBSemiLagrangianSolver
from mfgarchon.core.hamiltonian import BoundedControlCost, L1ControlCost, SeparableHamiltonian
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import no_flux_bc

NX, NT = 41, 10


class _NarrowL1(L1ControlCost):
    """An L1 cost whose admissible set is NOT the class default (-1, 1).

    Nothing but ``effective_domain()`` changes, so a consumer that reads the owner
    produces a different answer while one holding the ``(-1.0, 1.0)`` literal produces
    the identical answer. That is the discrimination this module needs.
    """

    def effective_domain(self) -> tuple[float, float]:
        return (-0.25, 0.25)


class _NarrowBounded(BoundedControlCost):
    """Same trick for the quadratic-on-A branch: A shrinks, ``max_control`` does not."""

    def effective_domain(self) -> tuple[float, float]:
        return (-0.2, 0.2)


def _solve(control_cost) -> np.ndarray:
    # H-only construction: MFGComponents derives the SeparableLagrangian that
    # ``problem.lagrangian_class`` returns, which is the route production code takes.
    problem = MFGProblem(
        model=Model(hamiltonian=SeparableHamiltonian(control_cost=control_cost), sigma=0.15),
        domain=TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[NX], boundary_conditions=no_flux_bc(dimension=1)),
        conditions=Conditions(u_terminal=lambda x: (x - 0.5) ** 2, m_initial=lambda x: 1.0, T=0.5),
        Nt=NT,
    )
    solver = HJBSemiLagrangianSolver(problem)
    assert solver._use_dpp, "this test is only meaningful on the DPP path; the gate changed"
    xs = problem.geometry.coordinates[0]
    density = np.ones((NT + 1, NX))
    return solver.solve_hjb_system(density, (xs - 0.5) ** 2, U_coupling_prev=np.zeros((NT + 1, NX)))


def test_dpp_sweep_reads_the_narrowed_domain_for_l1():
    """The load-bearing pin. A hardcoded (-1, 0, 1) candidate set cannot see this change."""
    wide = _solve(L1ControlCost(lambda_=0.5))
    narrow = _solve(_NarrowL1(lambda_=0.5))

    assert np.isfinite(wide).all()
    assert np.isfinite(narrow).all()
    assert not np.allclose(wide, narrow, rtol=0, atol=0), (
        "shrinking effective_domain() from (-1, 1) to (-0.25, 0.25) left the value function "
        "byte-identical: the DPP control sweep is not reading the admissible set from its owner"
    )
    # Restricting the admissible set can only raise the minimized DPP cost.
    assert narrow.max() >= wide.max()


def test_dpp_sweep_reads_the_narrowed_domain_for_bounded():
    """Same pin on the quadratic-on-A branch, where the ladder read cc.max_control directly."""
    wide = _solve(BoundedControlCost(lambda_=1.0, max_control=2.0))
    narrow = _solve(_NarrowBounded(lambda_=1.0, max_control=2.0))

    assert np.isfinite(wide).all()
    assert np.isfinite(narrow).all()
    assert not np.allclose(wide, narrow, rtol=0, atol=0), (
        "shrinking effective_domain() left the value function byte-identical: the DPP sweep "
        "is still reading cc.max_control instead of the owner"
    )


def test_no_second_owner_literals_survive_in_the_sweep():
    """Consolidation pin: the removed ladder's two re-derivations must not come back.

    Scope stated precisely -- this greps the method's source for the specific literals the
    old ladder used. It cannot stop an arbitrary re-fork; the numerical tests above are the
    checks that matter. It does stop the exact regression that was here.
    """
    import inspect

    source = inspect.getsource(HJBSemiLagrangianSolver._solve_timestep_dpp)

    assert "effective_domain()" in source, "the DPP sweep must read the admissible set from its owner"
    assert "np.array([-1.0, 0.0, 1.0])" not in source, "the hardcoded L1 bang-bang literal is back"
    assert "cc.max_control" not in source, "the private max_control re-derivation is back"
