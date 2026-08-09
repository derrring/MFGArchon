r"""The finite-difference Jacobian must not straddle a kink (#1745).

An upwind HJB residual selects between two one-sided differences, so it is
non-differentiable where they tie -- and symmetric initial data puts nodes exactly there.
Measured on the 2-D smoke problem at the stalled iterate: the grid centre's two
neighbours agree to 9e-11, and dF[60]/dU[59] is +13.395 approached from one side and
-7.9998 from the other. A one-sided quotient reports only the branch on its own side; the
step built from it is one the line search then cuts to ~0.03, and the solve creeps to its
iteration budget instead of converging.

The system below is that structure in four unknowns, evaluated at the tie. It is built to
four **structural** properties rather than to a list of mutants, because a mutant list is
what the author already imagined:

1. **Every row and every column separates the two quotients.** Row 0 and row 2 through a
   quadratic (forward reads `2a + h`, central reads `2a`), row 1 through the kink itself,
   row 3 through `u[1]**2` in a column that is otherwise flat. A Jacobian that is central
   on some rows or columns and one-sided on the rest is therefore visible somewhere. An
   earlier revision was linear on rows 0 and 2, and "forward on the even rows" passed the
   whole file while collapsing the real solve from 13 inner solves to 3.
2. **The entries bracket the real Jacobian's dynamic range.** Measured over the 84
   Jacobians the `fdm_upwind_2d` capability cell builds: smallest nonzero `|J| = 0.0356`,
   largest `= 449.78`. Here the range is `1e-4` to `2000`, so any clipping bound or
   sparsification threshold that could alter the real solve falls inside what these
   assertions pin. An earlier revision topped out at 20, and clipping to +-25 -- which
   kills the real cell in one iteration -- passed.
3. **One entry's value depends on the step size.** The central quotient of a cubic is
   `3a^2 + h^2` exactly, so `J[3,3]` reports the epsilon in force. Against a piecewise
   linear system every quotient is exact at every step size and an epsilon test certifies
   nothing.
4. **Each row's residual magnitude is kept near its pinned entries.** Cancellation in a
   quotient scales with `|F_i|`, not with `|J[i,j]|`, so the small `1e-4` entry sits in
   row 1 (`|F| ~ 1`), not beside the 2000 in row 2 (`|F| ~ 1000`). Worst realized error is
   0.36 of its tolerance.

**What no fixed-size miniature can catch:** a mutation keyed on the system's size or
bandwidth. "Forward when n > 8", "forward when n is odd" and "forward outside bandwidth 3"
all pass here and return the real cell to its pre-fix defect, because n = 4 puts every
entry in band. Those are a limit of this oracle; the capability cell is what covers them.
"""

from __future__ import annotations

import pytest

import numpy as np

from mfgarchon.utils.numerical.nonlinear_solvers import NewtonSolver

# u[0] and u[2] are equidistant from u[1], so the two candidate differences tie and the
# selection is a coin flip -- the configuration the 2-D solve reaches on its own.
TIE = np.array([1.0, 0.0, 1.0, 2.0])

# Analytic Jacobian at TIE, with the kink row carrying the mean of its two branch slopes.
# [3,3] is 3*u[3]**2 = 12, plus the central quotient's exact h**2 term.
EXPECTED = np.array(
    [
        [20.0, 0.0, 3.0, 0.0],
        [0.5, -1.0, 0.5, 1e-4],
        [0.0, 0.0, 2000.0, 0.0],
        [0.0, 0.0, 0.0, 12.0],
    ]
)


def _selecting(u: np.ndarray) -> np.ndarray:
    return np.array(
        [
            10.0 * u[0] ** 2 + 3.0 * u[2],
            max(u[0] - u[1], u[2] - u[1]) + 1e-4 * u[3],
            1000.0 * u[2] ** 2,
            u[3] ** 3 + u[1] ** 2,
        ]
    )


def _jacobian(sparse: bool, epsilon: float = 1e-7) -> np.ndarray:
    J = NewtonSolver(sparse=sparse, finite_diff_epsilon=epsilon)._finite_difference_jacobian(_selecting, TIE)
    return np.asarray(J.toarray() if sparse else J)


# Both branches ship: hjb_fdm.py builds its solver with sparse=True, and the sparse
# assembly is a separate code path from the dense one.
@pytest.mark.parametrize("sparse", [False, True])
def test_the_jacobian_at_a_tie_reports_both_branches_not_one(sparse):
    """The kink row's two live columns must carry the mean of the branch slopes, 0.5.

    Perturbing u[0] upwards makes `u[0] - u[1]` the winner (slope 1 in that component);
    perturbing it downwards hands the max to `u[2] - u[1]`, whose slope in u[0] is 0. A
    forward quotient sees only the first and returns 1.
    """
    J = _jacobian(sparse)

    assert np.isclose(J[1, 0], 0.5, rtol=0, atol=1e-6), f"column 0 straddles the kink: {J[1, 0]} (forward gives 1.0)"
    assert np.isclose(J[1, 2], 0.5, rtol=0, atol=1e-6), f"column 2 straddles the kink: {J[1, 2]} (forward gives 1.0)"


@pytest.mark.parametrize("sparse", [False, True])
def test_every_entry_matches_the_analytic_jacobian(sparse):
    """Whole-matrix, not a handful of scalars: an unpinned entry is where a mutant lives.

    `atol` governs the structurally-zero entries, where the two evaluations are
    bit-identical and the quotient is exactly 0, so a one-sided value of order `h` shows
    up. `rtol` governs the large ones, whose realized error is set by `|F_i|` and not by
    the entry.
    """
    J = _jacobian(sparse)

    assert J.shape == EXPECTED.shape
    worst = int(np.argmax(np.abs(J - EXPECTED)))
    assert np.allclose(J, EXPECTED, rtol=1e-8, atol=1e-8), (
        f"entry {divmod(worst, 4)} is {J.flat[worst]}, expected {EXPECTED.flat[worst]}\ngot:\n{J}"
    )


def test_the_kink_columns_are_not_an_artefact_of_one_epsilon():
    """A truncation effect shrinks with epsilon. A kink does not: 0.5 at every step size."""
    for eps in (1e-4, 1e-6, 1e-8):
        J = _jacobian(sparse=False, epsilon=eps)
        assert np.isclose(J[1, 0], 0.5, rtol=0, atol=1e-6), f"eps={eps:g} gave {J[1, 0]}"


def test_the_step_size_used_is_the_one_it_was_given():
    """`J[3,3]` is 3*u[3]**2 + eps**2 exactly, so it reports the epsilon in force.

    The range stops at 1e-4 for DISCRIMINATION, not for noise: the signal being asserted
    is eps**2, so at eps = 1e-5 it is 1e-10, below this atol, and a solver ignoring
    `finite_diff_epsilon` would pass. Cancellation is the looser bound -- 8.9e-16/eps,
    still 5.6x under atol at 1e-5 -- and does not bind until 1e-6.
    """
    for eps in (1e-2, 1e-3, 1e-4):
        J = _jacobian(sparse=False, epsilon=eps)
        expected = 12.0 + eps**2
        assert np.isclose(J[3, 3], expected, rtol=0, atol=1e-9), (
            f"eps={eps:g}: J[3,3] = {J[3, 3]!r}, expected {expected!r} -- "
            "the solver is not using the epsilon it was handed"
        )
