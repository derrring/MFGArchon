"""The FDM upwind HJB refuses a Hamiltonian its momentum rule is not Godunov for (#2311).

Rouy-Tourin per axis is the Godunov numerical Hamiltonian exactly when H is even in each momentum
component and nondecreasing in its magnitude. #2311 measured three library-constructible Hamiltonians
outside that, each solving to a finite, wrong answer with no warning. Two are refused here; the presence
half keeps a refusal of everything from passing.
"""

from __future__ import annotations

import pytest

import numpy as np

from mfgarchon import Conditions, MFGProblem, Model
from mfgarchon.alg.numerical.hjb_solvers import HJBFDMSolver
from mfgarchon.core.hamiltonian import (
    CongestionHamiltonian,
    DualHamiltonian,
    LagrangianBase,
    QuadraticControlCost,
    SeparableHamiltonian,
)
from mfgarchon.geometry import TensorProductGrid, no_flux_bc


class _ShiftedQuadraticL(LagrangianBase):
    """``L = |alpha - beta|^2 / 2 - offset``, minimised at ``alpha = beta``; ``offset`` adds to ``H``."""

    def __init__(self, beta: float, offset: float = 0.0):
        super().__init__()
        self.beta = beta
        self.offset = offset

    def __call__(self, x, alpha, m, t=0.0):
        return 0.5 * float(np.sum((np.atleast_1d(alpha) - self.beta) ** 2)) - self.offset


def _problem(hamiltonian, dimension: int):
    grid = TensorProductGrid(
        bounds=[(0.0, 1.0)] * dimension, Nx_points=[11] * dimension, boundary_conditions=no_flux_bc(dimension=dimension)
    )
    return MFGProblem(
        model=Model(hamiltonian=hamiltonian, sigma=0.3),
        domain=grid,
        conditions=Conditions(m_initial=lambda p: 1.0, u_terminal=lambda p: 0.0, T=0.1),
        Nt=3,
    )


def _quadratic():
    return SeparableHamiltonian(control_cost=QuadraticControlCost(control_cost=1.0))


@pytest.mark.parametrize("dimension", [1, 2])
@pytest.mark.parametrize(
    ("label", "make", "violation"),
    [
        ("dual_shifted_lagrangian", lambda: DualHamiltonian(_ShiftedQuadraticL(0.7)), "not even"),
        # The asymmetry is 2 * 0.7 * |p| <= 5.6 at the probed momenta; a tolerance relative to |H| would let
        # the constant swallow it.
        ("dual_shifted_under_a_large_constant", lambda: DualHamiltonian(_ShiftedQuadraticL(0.7, 1e7)), "not even"),
        (
            "congestion_negative_factor",
            lambda: CongestionHamiltonian(
                control_cost=QuadraticControlCost(control_cost=1.0),
                congestion_factor=lambda m: -(1.0 + m),
                congestion_factor_dm=lambda m: -1.0,
            ),
            "decreasing",
        ),
    ],
)
def test_a_hamiltonian_outside_the_godunov_condition_is_refused(label, make, violation, dimension):
    with pytest.raises(NotImplementedError, match=rf"#2311.*{violation}"):
        HJBFDMSolver(_problem(make(), dimension))


@pytest.mark.parametrize("dimension", [1, 2])
@pytest.mark.parametrize(
    ("label", "make"),
    [
        ("separable_quadratic", _quadratic),
        ("dual_even_lagrangian", lambda: DualHamiltonian(_ShiftedQuadraticL(0.0))),
        # Even and increasing wherever it is finite; the probe must not evaluate it at vacuum.
        (
            "coupling_singular_at_vacuum",
            lambda: SeparableHamiltonian(
                control_cost=QuadraticControlCost(control_cost=1.0),
                coupling=lambda m: -1.0 / m,
                coupling_dm=lambda m: 1.0 / m**2,
            ),
        ),
    ],
)
def test_a_hamiltonian_inside_it_is_accepted(label, make, dimension):
    HJBFDMSolver(_problem(make(), dimension))
    # And the refusal belongs to the upwind scheme only: centered differencing selects no momentum.
    HJBFDMSolver(_problem(DualHamiltonian(_ShiftedQuadraticL(0.7)), dimension), advection_scheme="gradient_centered")
