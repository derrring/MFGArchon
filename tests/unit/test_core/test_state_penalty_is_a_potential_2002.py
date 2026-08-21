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


# ---------------------------------------------------------------------------------------------
# What the composition must PRESERVE. Each of these was a blocker found in review, and each had
# zero test discrimination before it was found: the composition rebuilt the Hamiltonian from five
# of its six constructor parameters, and updated one of the two objects holding the potential.
# ---------------------------------------------------------------------------------------------


def _hamiltonian_with(population_index=0, potential=None):
    return SeparableHamiltonian(
        control_cost=QuadraticControlCost(control_cost=1.0),
        potential=potential,
        population_index=population_index,
    )


def _problem_with(hamiltonian, **kwargs):
    grid = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[_N], boundary_conditions=no_flux_bc(dimension=1))
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


def test_the_composition_preserves_every_hamiltonian_field():
    """It COPIES; an earlier version rebuilt from five of six constructor parameters.

    `population_index` is the one that was dropped, and it is live -- a multi-population problem
    keyed to population k silently became population 0. Nothing caught it, because nothing here
    used a non-default value.
    """
    composed = _problem_with(_hamiltonian_with(population_index=2), state_penalty=_wall).components._hamiltonian_class

    assert composed.population_index == 2, "population_index was dropped by the rebuild"
    assert isinstance(composed, SeparableHamiltonian)
    assert composed.control_cost.lambda_ == 1.0


def test_the_composition_does_not_mutate_the_callers_hamiltonian():
    """The user's object is theirs. Copy, do not edit in place."""
    original = _hamiltonian_with()
    _problem_with(original, state_penalty=_wall)
    assert original._potential is None, "the caller's Hamiltonian was modified"


def test_the_lagrangian_gets_the_wall_too():
    """`MFGComponents` snapshots the potential into `_lagrangian_class` BEFORE composition.

    `HJBSemiLagrangianSolver` reads that copy whenever the control cost is non-smooth, so a
    composition that updated the Hamiltonian alone left a solver silently unwalled -- the #2002
    defect itself, in a second channel.
    """
    problem = _problem_with(_hamiltonian_with(), state_penalty=_wall)
    lagrangian = problem.components._lagrangian_class

    assert lagrangian is not None
    assert lagrangian._potential is problem.components._hamiltonian_class._potential


def test_a_zero_wall_is_a_no_op_on_a_vectorised_base_potential():
    """The composed closure must return what the BASE returns, or it defeats the vectorisation
    probe: `SeparableHamiltonian` calls the potential on a batch to decide whether it may.

    A base that is not preserved here shows up as a silent per-point fallback at best, and the
    base frozen at `x_batch[0]` at worst.
    """

    def vectorised_base(x, t=0.0):
        column = np.atleast_1d(np.asarray(x, dtype=float))
        values = np.sin(2 * np.pi * (column[..., 0] if column.ndim > 1 else column))
        return float(values[0]) if values.size == 1 else values

    batch = np.linspace(0.0, 1.0, 5).reshape(-1, 1)
    truth = np.sin(2 * np.pi * batch.ravel())

    bare = _problem_with(_hamiltonian_with(potential=vectorised_base)).components._hamiltonian_class
    walled = _problem_with(
        _hamiltonian_with(potential=vectorised_base),
        state_penalty=lambda x: np.zeros_like(np.atleast_1d(x)),
    ).components._hamiltonian_class

    assert np.allclose(bare._evaluate_potential_batch(batch, 0.0), truth), "control: base is wrong"
    assert np.allclose(walled._evaluate_potential_batch(batch, 0.0), truth), "a zero wall changed the base potential"
    assert walled._potential_is_vectorized is bare._potential_is_vectorized, (
        "the composition defeated the vectorisation probe"
    )


def test_the_composed_potential_returns_what_the_base_returns():
    """Shape is the base's contract, not the wrapper's -- asserted directly.

    The vectorisation test above does not discriminate this: with a batch-sized base a stray
    `.squeeze()` is a no-op, so a mutant that squeezes regardless passes it. The failure only
    shows on results of size 1, which is exactly the per-point call every solver makes.
    """
    seen = {}

    def recording_base(x, t=0.0):
        column = np.atleast_1d(np.asarray(x, dtype=float))
        value = np.sin(2 * np.pi * (column[..., 0] if column.ndim > 1 else column))
        # Per-point callers wrap this in float(), so a size-1 answer must be scalar. That is the
        # contract the wrapper has to reproduce, not one it may normalise away.
        out = float(value[0]) if value.size == 1 else value
        seen[np.shape(np.asarray(x))] = np.shape(out)
        return out

    composed = _problem_with(
        _hamiltonian_with(potential=recording_base),
        state_penalty=lambda x: np.zeros_like(np.atleast_1d(x)),
    ).components._hamiltonian_class._potential

    for probe in (np.array([0.3]), np.linspace(0.0, 1.0, 5).reshape(-1, 1)):
        seen.clear()
        out = composed(probe, 0.0)
        base_shape = seen[np.shape(probe)]
        assert np.shape(out) == base_shape, (
            f"composed returned {np.shape(out)} where the base returned {base_shape} for x of "
            f"shape {np.shape(probe)} -- the wrapper changed the base's contract"
        )
