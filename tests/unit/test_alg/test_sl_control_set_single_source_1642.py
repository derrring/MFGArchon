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

Two numerical pins, written in cfb0207c (#1647), solved the DPP path with an L1 and a bounded
cost subclass that narrowed only ``effective_domain()``. Since #2901 (user ruling, 2026-11-11)
the DPP path reads the cost only as the library ships it, and refuses a subclass, so no public
input separates ``effective_domain()`` from ``max_control`` there; they were deleted. A future
shipped cost that separates them brings its own pin.
"""

from __future__ import annotations

from mfgarchon.alg.numerical.hjb_solvers import HJBSemiLagrangianSolver


def test_no_second_owner_literals_survive_in_the_sweep():
    """Consolidation pin: the removed ladder's two re-derivations must not come back.

    Scope stated precisely -- this greps the method's source for the specific literals the
    old ladder used. It cannot stop an arbitrary re-fork. It does stop the exact regression
    that was here.
    """
    import inspect

    source = inspect.getsource(HJBSemiLagrangianSolver._solve_timestep_dpp)

    assert "effective_domain()" in source, "the DPP sweep must read the admissible set from its owner"
    assert "np.array([-1.0, 0.0, 1.0])" not in source, "the hardcoded L1 bang-bang literal is back"
    assert "cc.max_control" not in source, "the private max_control re-derivation is back"
