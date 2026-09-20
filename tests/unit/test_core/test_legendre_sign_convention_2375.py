"""The Legendre pairing carries `-p·α`, so `α* = -∂H/∂p` holds for every L. #2375 ruling 5.

WHY NOTHING IN THE LIBRARY COULD HAVE CAUGHT THIS. Every concrete control cost shipped here is
**even in α** -- measured: `QuadraticControlCost`, `L1ControlCost`, `BoundedControlCost` on a
symmetric domain, and the Moreau-Yosida regularisation all give `L(a) == L(-a)` to 1e-14. And
`sup{-p·α - L}` equals `sup{+p·α - L}` exactly when `L(-α) = L(α)`. So the change is byte-identical
across the whole library and a test built on any shipped cost passes before and after while pinning
nothing. The discriminating `L` has to be constructed here, and this file constructs it.

THE SEARCH CONFIGURATION IS PART OF THE ASSERTION, not scaffolding. `DualHamiltonian` locates `α*`
by grid search over `np.linspace(*alpha_bounds, n_search)`, and `n_search` **defaults to 100**. At
the defaults -- `alpha_bounds=(-10, 10)`, step 0.202 -- the even-`L` probe returns -1.9192 instead of
-2.0000, and all three non-even rows shift in the second decimal. Every test below therefore states
`ALPHA_BOUNDS` and `N_SEARCH` explicitly.

    A run whose even-`L` control does not return exactly -2.0000 has measured its own grid,
    not the library.

That is what `test_the_even_lagrangian_pins_the_grid_before_anything_else_is_believed` is for, and it
is not a formality: it is the only thing separating "the convention is wrong" from "the step size is
0.2". Both produce a number near -2 and neither announces which.
"""

from __future__ import annotations

import pytest

import numpy as np

from mfgarchon.core.hamiltonian import DualHamiltonian, DualLagrangian, LagrangianBase

# Stated, not defaulted -- see the module docstring. step = 10/(100001-1) = 1e-4.
ALPHA_BOUNDS = (-5.0, 5.0)
N_SEARCH = 100001
ODD_COEFF = 0.3


class _NonEvenLagrangian(LagrangianBase):
    """`L(a) = 0.5a² + 0.3a`. The smallest L that separates the two conventions.

    A DELIBERATE DUPLICATE of `AsymmetricL` in `test_lagrangian_base_alpha_sign_1642.py`, same
    formula and same 0.3. Not imported because no test module in this repository imports from
    another -- measured, 0 of 250 files -- so importing would set a new precedent to save nine
    lines. The alternative, moving the fixture to `tests/conftest.py`, is the single-source answer
    and is a wider change than this phase; it is the right follow-up if a third file ever needs it.

    **If the 0.3 changes, both must change.** 1642 owns the `LagrangianBase` convention pins;
    this file owns `DualHamiltonian`, `DualLagrangian` and the round trip, which that file does not
    touch. Splitting by class is why they are two files rather than one.

    `0.5a²` alone is even and cannot: the odd term is the entire discriminating content, and its
    coefficient is what the three expected values below are a function of.
    """

    def __call__(self, x, alpha, m, t=0.0):
        a = np.atleast_1d(np.asarray(alpha, dtype=float))
        return float(0.5 * np.sum(a**2) + ODD_COEFF * np.sum(a))


class _EvenLagrangian(LagrangianBase):
    """`L(a) = 0.5a²`. The control: every convention agrees on it, so a disagreement is the grid."""

    def __call__(self, x, alpha, m, t=0.0):
        a = np.atleast_1d(np.asarray(alpha, dtype=float))
        return float(0.5 * np.sum(a**2))


X = np.zeros(1)
M = 1.0


def _dual(lagrangian):
    return DualHamiltonian(lagrangian, alpha_bounds=ALPHA_BOUNDS, n_search=N_SEARCH)


def test_the_even_lagrangian_pins_the_grid_before_anything_else_is_believed():
    """MANDATORY CONTROL, and it must be read before the rows below.

    The two conventions agree exactly on an even `L`, so this asserts the grid resolves the optimum
    and nothing about the convention. If it fails, every other number in this file is a measurement
    of `np.linspace` -- at the constructor defaults the same probe returns -1.9192.

    Analytic: `argmin{p·a + 0.5a²} = -p`, so -2.0 at `p = +2`.
    """
    even = _EvenLagrangian()
    p = np.array([2.0])
    assert float(np.ravel(_dual(even).optimal_control(X, M, p))[0]) == pytest.approx(-2.0, abs=5e-5)
    assert float(np.ravel(even.optimal_control(X, M, p))[0]) == pytest.approx(-2.0, abs=5e-5)


@pytest.mark.parametrize(
    ("p", "expected_alpha_star"),
    [(2.0, -2.3), (-2.0, 1.7), (0.6, -0.9)],
)
def test_a_non_even_lagrangian_gives_the_control_that_actually_minimises_cost(p, expected_alpha_star):
    """THE ARTIFACT. `α*` must minimise `p·α + L(α)` -- the cost-to-go the agent pays.

    Analytic, so no grid supplies the expected value: `d/da (p·a + 0.5a² + 0.3a) = 0` gives
    `α* = -(p + 0.3)`. Before this change the library returned `-(p - 0.3)` -- the right magnitude
    off by twice the odd coefficient, and correct for every even `L`, which is why it survived.

    Both classes are asserted because they are two owners of one convention: `DualHamiltonian`
    through `_find_optimal_alpha`, `LagrangianBase` through `conjugate_argmax`. Pinning one leaves
    the other free to fork, which is the failure #1642 records.
    """
    assert expected_alpha_star == pytest.approx(-(p + ODD_COEFF), abs=1e-12), "expectation is analytic"
    non_even = _NonEvenLagrangian()
    pv = np.array([p])
    assert float(np.ravel(_dual(non_even).optimal_control(X, M, pv))[0]) == pytest.approx(expected_alpha_star, abs=5e-4)
    assert float(np.ravel(non_even.optimal_control(X, M, pv))[0]) == pytest.approx(expected_alpha_star, abs=5e-4)


def test_the_hamiltonian_value_is_the_conjugate_at_minus_p():
    """`H = sup{-p·α - L}` is `L*(-p)`, not `L*(p)` -- the value moved, not just the argmax.

    Closed form for this `L`: `H(p) = 0.5(p + 0.3)²`. Asserting the VALUE and not only the optimiser,
    because `__call__` and `_find_optimal_alpha` carry the objective separately -- three branches
    each -- and flipping one without the other leaves an `H` whose gradient is not its own argmax.
    """
    h = _dual(_NonEvenLagrangian())
    for p in (2.0, -2.0, 0.6):
        assert float(h(X, M, np.array([p]))) == pytest.approx(0.5 * (p + ODD_COEFF) ** 2, abs=5e-4)


def test_the_inverse_transforms_gradient_carries_the_same_sign():
    """PINS `DualLagrangian._find_optimal_p`, which the round-trip test does NOT reach.

    Measured: un-negating that function's return leaves every other test in this file green, because
    the round trip reads `DualLagrangian.__call__` (a sup VALUE) and never its argmax. So `d_alpha`
    was an arm of this change that nothing exercised -- the same gap #2375's phase-1 sibling hit on
    four guard arms, found the same way, by mutating the thing just written.

    Analytic: with `H(p) = 0.5(p + 0.3)²`, the inverse `L(α) = sup_p {-p·α - H(p)}` is maximised at
    `p* = -(α + 0.3)`, and `∂L/∂α = -p* = α + 0.3` -- which is `L'(α)` for the original
    `L = 0.5α² + 0.3α`, as a recovered Legendre pair requires. Un-negated it returns `-(α + 0.3)`,
    wrong by a sign for every α, and exactly zero at α = -0.3.
    """
    h = DualHamiltonian(_NonEvenLagrangian(), alpha_bounds=(-6.0, 6.0), n_search=601)
    inverse = DualLagrangian(h, p_bounds=(-6.0, 6.0), n_search=601)
    for alpha in (1.0, -1.0, 0.5):
        expected = alpha + ODD_COEFF
        # The two candidate answers are +/-(alpha + 0.3), i.e. 2*|alpha + 0.3| >= 0.4 apart, so a
        # 0.02 grid step resolves WHICH sign without resolving the value tightly. Deliberately
        # coarse: a nested double search at 1e-4 cost 42s for no extra discrimination.
        assert abs(2 * expected) > 10 * 5e-2, "the two signs must be far apart relative to tolerance"
        got = float(np.ravel(inverse.d_alpha(X, np.array([alpha]), M))[0])
        assert got == pytest.approx(expected, abs=5e-2)


def test_the_legendre_round_trip_recovers_L_and_not_its_mirror():
    """BOTH sides of the pair carry `-p·α`, or `L → H → L` returns `L(-α)`.

    This is the reason the inverse transform is in the same change. Flipping `H` alone leaves a
    round trip that returns `L(-α)`: a well-formed Lagrangian, equal to `L` for every even `L`, and
    wrong for exactly the non-even ones the convention was changed for. `L(1) = 0.8` against
    `L(-1) = 0.2`, so the two outcomes are 0.6 apart and no grid error explains the gap.

    Coarser grids than the module constants on purpose: this asserts WHICH of two well-separated
    values is returned, and a 1e-4 step over a nested double search costs minutes for no extra
    discrimination.
    """
    lagrangian = _NonEvenLagrangian()
    h = DualHamiltonian(lagrangian, alpha_bounds=(-6.0, 6.0), n_search=1201)
    round_trip = DualLagrangian(h, p_bounds=(-6.0, 6.0), n_search=241)
    for alpha in (1.0, -1.0, 0.5):
        direct = lagrangian(X, np.array([alpha]), M)
        mirror = lagrangian(X, np.array([-alpha]), M)
        assert abs(direct - mirror) > 0.2, "the probe must separate L(a) from L(-a)"
        assert float(round_trip(X, np.array([alpha]), M)) == pytest.approx(direct, abs=2e-2)
