"""Issue #2007: the `gradient_*` non-conservation warning understated the leak by 10x.

It read "leaks O(1e-2), even with zero drift". Measured:

    genuinely zero drift, stationary initial density   -1.7e-14   (no leak at all)
    wall-normal drift A = 0.7, D = 1/8, d = 1          -1.4e-1
    wall-normal drift A = 0.7, D = 1/8, d = 2          -8.9e-1

So the 1.5e-2 figure was a **transient** density, which the string did not say, and a reader
budgeting against O(1e-2) was off by an order of magnitude exactly where the scheme is used.

WHAT THIS FILE DOES NOT DO. #2007 recommends removing these schemes, on the grounds that the
gradient form also drops the `m div(alpha)` term of `div(alpha m)` and so does not discretize the FP
operator even away from the wall -- repointing the wall moves a source-free instance from 5.81e-1 to
8.02e-1. That is a live recommendation, not something settled here:
`test_gradient_centered_still_available_and_leaks` records a standing decision to keep them
explicitly selectable, and overriding a recorded decision is not a warning's job.

So this is the uncontested half: the numbers a caller reads are the numbers that were measured.
"""

from __future__ import annotations

import warnings

import numpy as np

from mfgarchon.alg.numerical.fp_solvers.fp_fdm import FPFDMSolver
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.core.mfg_components import MFGComponents
from mfgarchon.core.mfg_problem import MFGProblem
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import no_flux_bc


def _warning_text(scheme="gradient_upwind", n=21):
    grid = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[n], boundary_conditions=no_flux_bc(dimension=1))
    problem = MFGProblem(
        geometry=grid,
        T=0.2,
        Nt=5,
        sigma=0.5,
        coupling_coefficient=0.0,
        components=MFGComponents(
            m_initial=lambda z: np.ones_like(np.asarray(z, dtype=float)),
            u_terminal=lambda z: np.asarray(z, dtype=float) * 0.0,
            hamiltonian=SeparableHamiltonian(
                control_cost=QuadraticControlCost(control_cost=1.0),
                coupling=lambda m: np.asarray(m) * 0.0,
                coupling_dm=lambda m: np.asarray(m) * 0.0,
            ),
        ),
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        FPFDMSolver(problem, advection_scheme=scheme)
    texts = [str(w.message) for w in caught if "conserve mass" in str(w.message)]
    assert texts, f"no non-conservation warning was emitted for {scheme}"
    return texts[0]


def test_the_warning_states_the_loss_is_unbounded_not_a_magnitude():
    """UPDATED 2026-08-22. The contract this test enforces is unchanged -- the numbers a caller
    reads are the numbers that were measured -- but the numbers changed, and so did their shape.

    The previous version pinned a single driven figure (`-1.4e-1`). That figure was correct for its
    fixture and is still in the message as a retraction, but a lone magnitude reads as a bounded
    error a caller can budget against, and the loss is not bounded: re-measured across cell Peclet
    it runs 0 -> +0.5%, 0.19 -> -23.6%, 0.89 -> -99.97%. So what must be quotable is the
    DEPENDENCE and its limit, not a point on it.
    """
    text = _warning_text()
    assert "UNBOUNDED" in text, "a bounded-sounding magnitude is the thing that misled"
    assert "Peclet" in text, "the axis the loss runs along must be named, or the reader cannot place their own case"
    assert "-99.97%" in text, "the limit case must be quotable, not only the mild one"
    assert "-23.6%" in text, "and an intermediate point, so the trend is visible rather than asserted"
    assert "relaxed uniform field" in text, (
        "what the caller actually gets back at the limit -- 'lost mass' understates a field that "
        "no longer resembles the solution"
    )


def test_the_understated_phrase_is_gone_from_the_live_claim():
    """`O(1e-2), even with zero drift` is the sentence that cost the order of magnitude. It may
    appear as a RETRACTION -- the message says what it used to say -- but not as a live figure.

    Both superseded figures now sit behind "Earlier revisions", which is what this splits on: a
    retraction is only safe while it is unmistakably a retraction."""
    text = _warning_text()
    live = text.split("Earlier revisions")[0]
    assert "O(1e-2)" not in live, (
        "the understated magnitude is back in the part of the message a reader takes as current"
    )


def test_the_second_defect_is_mentioned_because_a_conservative_wall_would_not_fix_it():
    """A reader who only hears 'does not conserve mass' will reach for a conservative wall. The
    gradient form also drops `m div(alpha)`, so that would not make it the FP operator."""
    text = _warning_text()
    assert "div(alpha)" in text, "the interior defect must be named, not only the wall one"
    assert "2007" in text


def test_the_divergence_schemes_do_not_warn():
    """Control. Without it, a warning emitted unconditionally would pass every test above."""
    grid = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[21], boundary_conditions=no_flux_bc(dimension=1))
    problem = MFGProblem(
        geometry=grid,
        T=0.2,
        Nt=5,
        sigma=0.5,
        coupling_coefficient=0.0,
        components=MFGComponents(
            m_initial=lambda z: np.ones_like(np.asarray(z, dtype=float)),
            u_terminal=lambda z: np.asarray(z, dtype=float) * 0.0,
            hamiltonian=SeparableHamiltonian(
                control_cost=QuadraticControlCost(control_cost=1.0),
                coupling=lambda m: np.asarray(m) * 0.0,
                coupling_dm=lambda m: np.asarray(m) * 0.0,
            ),
        ),
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        FPFDMSolver(problem, advection_scheme="divergence_upwind")
    assert not [w for w in caught if "conserve mass" in str(w.message)]
