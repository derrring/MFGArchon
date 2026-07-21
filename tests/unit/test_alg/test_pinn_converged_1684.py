"""`converged` must track convergence, not "an epoch ran" (Issue #1684).

Three PINN solvers reported ``len(training_history["total_loss"]) > 0``. The guard four lines above
each site already raises ``RuntimeError`` when that list is empty, so the expression could only ever
be ``True`` -- a loss of ``1e30`` after one epoch reported convergence.

The training loop has always had the real test (``base_pinn.py``: the loop breaks when
``total_loss < convergence_tolerance``). The defect was that the reporting layer ignored it and
substituted its own. These tests pin the reported flag to the loop's own criterion.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from mfgarchon.alg.neural.pinn_solvers.base_pinn import PINNBase  # noqa: E402


class _Probe:
    """Exercises the property without standing up a network.

    ``converged`` reads only ``best_loss`` and ``config.convergence_tolerance``; binding the real
    property to a stand-in keeps the test on the contract rather than on PINN training, which is
    slow and stochastic and would make this test non-discriminating for the reason #1663 documents.
    """

    converged = PINNBase.converged

    def __init__(self, best_loss, tolerance=1e-6):
        self.best_loss = best_loss
        self.config = type("Cfg", (), {"convergence_tolerance": tolerance})()


@pytest.mark.parametrize(
    ("best_loss", "expected", "why"),
    [
        (1e30, False, "one epoch at a huge loss is what the old tautology called converged"),
        (1e12, False, "a diverging run"),
        (1e-3, False, "improving but still above tolerance"),
        (1e-6, False, "exactly at tolerance is not below it"),
        (1e-12, True, "genuinely converged"),
        (float("inf"), False, "never trained"),
    ],
)
def test_converged_tracks_the_loop_criterion(best_loss, expected, why):
    assert _Probe(best_loss).converged is expected, why


def test_the_flag_moves_when_the_solve_gets_worse():
    """The counterfactual #1684 asks of every candidate: worsen the solve, the flag must move.

    The old expression failed this -- it returned True for every one of these.
    """
    good = _Probe(1e-12).converged
    bad = _Probe(1e30).converged

    assert good is True, "a converged solve must report converged"
    assert bad is False, "a 1e30 loss must not report converged -- the old expression did"


def test_tolerance_is_honoured_rather_than_hardcoded():
    """A stricter tolerance must be able to turn a passing run into a failing one."""
    loss = 1e-7

    assert _Probe(loss, tolerance=1e-6).converged is True
    assert _Probe(loss, tolerance=1e-9).converged is False


def test_all_three_solvers_read_the_owner_rather_than_recomputing():
    """Pin the single-source property itself.

    Each solver previously carried its own copy of the expression. A future edit that reintroduces a
    local computation would pass the behavioural tests above -- they exercise the base property --
    so the absence of a second implementation is asserted directly.
    """
    import ast
    import inspect

    from mfgarchon.alg.neural.pinn_solvers import fp_pinn_solver, hjb_pinn_solver, mfg_pinn_solver

    for module in (mfg_pinn_solver, hjb_pinn_solver, fp_pinn_solver):
        tree = ast.parse(inspect.getsource(module))
        offenders = [
            node.lineno
            for node in ast.walk(tree)
            if isinstance(node, ast.Compare)
            and any(isinstance(c, ast.Attribute) and c.attr == "training_history" for c in ast.walk(node))
        ]
        assert not offenders, (
            f"{module.__name__} compares against training_history at line(s) {offenders}; "
            "read PINNBase.converged instead of recomputing the criterion (#1684)"
        )
