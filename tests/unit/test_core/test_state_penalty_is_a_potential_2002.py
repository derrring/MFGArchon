"""A soft wall is a POTENTIAL, and its sign is not the one you would write (#2002).

`problem.state_penalty(x)` is COST-signed: positive where the region is expensive. The
Hamiltonian's `potential` is REWARD-signed (gotcha G-001, and the sign block above
`source_term_hjb` in `mfg_problem.py`). So the composition SUBTRACTS, and this file exists
because that sign is invisible at every call site that does not do the subtraction.

Measured on a Gaussian wall at x = 0.5, no coupling, `u_terminal = 0`:

    potential amplitude   u(0, mid)
    ------------------- -----------
                    0     +0.000000
                   +5     -1.419462     <- a POSITIVE potential makes the wall CHEAPER
                   -5     +0.555313     <- a cost needs a NEGATIVE potential

That is what `L = L_ctrl - V - f` means operationally, and the repo has been wrong about this
sign twice (#1642 B1, #1645 B2). The composition is in ONE place so the subtraction is written
once; these tests pin the direction rather than the arithmetic, because a future refactor that
"simplifies" the sign will keep every unit passing and silently turn walls into wells.

WHY IT IS A POTENTIAL AND NOT A SOURCE. The term is `alpha`-free and `u`-free -- it depends on
where you are, not on what you do or what the value function says. That is the definition of a
potential, and it is what distinguishes this from the variational inequality `v >= Psi(x)` that
`problem.obstacle` claimed to be and never was. The VI slot is reserved and deliberately
unfinished: `HJBFDMSolver(constraint=ObstacleConstraint(...))` (#591), with #2036 and #2046 on
what it still owes.
"""

from __future__ import annotations

import pytest

import numpy as np

from mfgarchon import Conditions, MFGProblem, Model
from mfgarchon.alg.numerical.hjb_solvers.hjb_fdm import HJBFDMSolver
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.geometry.boundary import no_flux_bc
from mfgarchon.geometry.grids import TensorProductGrid

_N = 21
_NT = 6
_MID = _N // 2


def _wall(x):
    """A cost: expensive in the middle, negligible at the walls."""
    return 5.0 * np.exp(-60.0 * (np.atleast_1d(x) - 0.5) ** 2)


def _problem(**kwargs):
    grid = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[_N], boundary_conditions=no_flux_bc(dimension=1))
    hamiltonian = SeparableHamiltonian(control_cost=QuadraticControlCost(control_cost=1.0))
    return MFGProblem(
        model=Model(hamiltonian=hamiltonian, sigma=0.2),
        domain=grid,
        conditions=Conditions(
            u_terminal=lambda x: np.zeros_like(np.atleast_1d(x)).squeeze(),
            m_initial=lambda x: 1.0,
            T=0.3,
        ),
        Nt=_NT,
        **kwargs,
    )


def _value_at_t0(problem):
    solver = HJBFDMSolver(problem)
    density = np.tile(np.ones(_N) / _N, (problem.Nt + 1, 1))
    return np.asarray(solver.solve_hjb_system(density, np.zeros(_N), np.zeros((problem.Nt + 1, _N))))[0]


@pytest.fixture(scope="module")
def measured():
    return {
        "none": _value_at_t0(_problem()),
        "wall": _value_at_t0(_problem(state_penalty=_wall)),
        "wall3": _value_at_t0(_problem(state_penalty=_wall, state_penalty_scale=3.0)),
    }


def test_without_a_wall_the_value_is_flat(measured):
    """Positive control. Every other test reads a DIFFERENCE against this baseline, and a
    baseline that already had structure would let a wall of the wrong sign look right."""
    baseline = measured["none"]
    assert np.allclose(baseline, baseline[0]), f"baseline is not flat: {baseline}"


def test_a_wall_is_a_cost_not_a_reward(measured):
    """THE SIGN PIN. Delete this and a wall becomes a well, with every other test still green."""
    assert measured["wall"][_MID] > measured["none"][_MID], (
        f"the wall made the middle CHEAPER ({measured['wall'][_MID]:.6f} vs "
        f"{measured['none'][_MID]:.6f}) -- the composition's subtraction has been inverted"
    )


def test_scaling_the_wall_costs_more(measured):
    """`state_penalty_scale` multiplies. A scale that divided, or that was ignored, passes the
    sign test above and fails here."""
    assert measured["wall3"][_MID] > measured["wall"][_MID]


def test_the_wall_is_local(measured):
    """A cost applied everywhere -- a constant folded in by mistake -- would pass both tests
    above. The boundary must barely move."""
    at_edge = abs(measured["wall"][0] - measured["none"][0])
    at_middle = abs(measured["wall"][_MID] - measured["none"][_MID])
    assert at_edge < 0.01 * at_middle, f"edge moved {at_edge:.3e} against middle {at_middle:.3e}"


def test_the_retired_obstacle_field_refuses_and_names_both_successors():
    with pytest.raises(NotImplementedError) as excinfo:
        _problem(obstacle=_wall)

    message = str(excinfo.value)
    assert "RETIRED" in message
    assert "state_penalty" in message, "must name the soft-wall successor"
    assert "constraint=" in message, "must name the reserved VI slot"
    assert "obstacles" in message, "must disambiguate from the geometry field of almost the same name"


def test_a_hamiltonian_that_cannot_carry_a_potential_refuses():
    """Fail loud rather than skip. A soft wall quietly not applied is the whole of #2002."""

    class _NoPotential:
        def __call__(self, x, m, p, t):  # pragma: no cover - never invoked
            return 0.0

    problem = _problem()
    problem.components._hamiltonian_class = _NoPotential()
    problem.state_penalty = _wall

    with pytest.raises(NotImplementedError, match="carries a potential"):
        problem._compose_state_penalty_into_potential()
