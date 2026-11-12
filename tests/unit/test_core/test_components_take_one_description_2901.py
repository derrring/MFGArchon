"""`MFGComponents` takes one description of the game: the Hamiltonian or the Lagrangian (#2901, stage 1a).

User rulings, 2026-11-11: (a) a Hamiltonian and a Lagrangian given together are refused, as `Model` refuses them; (D)
after `MFGComponents(lagrangian=L)` the public `hamiltonian` field holds what the user gave, `None`, and the derived
Hamiltonian lives in `_hamiltonian_class`, read through `MFGProblem.hamiltonian_class`.

- **H and L together are refused**, by `MFGComponents` and by `Model`, through the one owner both read,
  `refuse_two_descriptions` (a mutation-verified pin). Before, the components carried both, every reader but HJB-SL's
  DPP path read H, the DPP path read L, and nothing compared them.
- **A copy of a Lagrangian-defined components carries one description** (a mutation-verified pin). With the derived
  Hamiltonian in the public field, `dataclasses.replace` handed `__post_init__` both, which (a) refuses.
- **A Lagrangian-defined problem builds** (a necessary oracle): its Hamiltonian is derived, analytically for a
  `SeparableLagrangian` and as the Legendre dual of a user's `LagrangianBase`.
"""

from __future__ import annotations

import dataclasses
import logging
import warnings

import pytest

import numpy as np

from mfgarchon import MFGProblem, Model
from mfgarchon.core import hamiltonian as hm
from mfgarchon.core.mfg_components import MFGComponents
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import no_flux_bc


def _a(v):
    return np.asarray(v, dtype=float)


class _QuadraticLagrangian(hm.LagrangianBase):
    """A user's Lagrangian, not a SeparableLagrangian, so the library forms its Hamiltonian numerically."""

    def __call__(self, t, x, alpha, m):
        return 0.5 * np.sum(_a(alpha) ** 2, axis=-1)


def _lagrangian_components(lagrangian):
    return MFGComponents(
        lagrangian=lagrangian, m_initial=lambda x: 1.0 + 0.0 * _a(x[0]), u_terminal=lambda x: (_a(x[0]) - 0.5) ** 2
    )


def _problem(components):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        logging.disable(logging.CRITICAL)
        try:
            grid = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[21], boundary_conditions=no_flux_bc(dimension=1))
            return MFGProblem(geometry=grid, T=0.5, Nt=10, volatility=0.3, components=components)
        finally:
            logging.disable(logging.NOTSET)


@pytest.mark.parametrize(
    "construct",
    [
        lambda h, lag: MFGComponents(hamiltonian=h, lagrangian=lag, m_initial=lambda x: 1.0, u_terminal=lambda x: 0.0),
        lambda h, lag: Model(hamiltonian=h, lagrangian=lag),
    ],
    ids=["MFGComponents", "Model"],
)
def test_a_hamiltonian_and_a_lagrangian_given_together_are_refused(construct):
    with pytest.raises(
        ValueError, match=r"Provide hamiltonian OR lagrangian, not both\. They are dual descriptions of the same game"
    ):
        construct(
            hm.SeparableHamiltonian(hm.QuadraticControlCost(control_cost=1.0)),
            hm.SeparableLagrangian(control_cost=hm.QuadraticControlCost(control_cost=1.0)),
        )


@pytest.mark.parametrize(
    ("lagrangian", "derived"),
    [
        (
            lambda: hm.SeparableLagrangian(control_cost=hm.QuadraticControlCost(control_cost=1.0)),
            hm.SeparableHamiltonian,
        ),
        (lambda: _QuadraticLagrangian(), hm.DualHamiltonian),
    ],
    ids=["SeparableLagrangian", "a user's LagrangianBase"],
)
def test_a_copy_of_a_lagrangian_defined_components_carries_one_description(lagrangian, derived):
    components = _lagrangian_components(lagrangian())
    assert components.hamiltonian is None
    copied = dataclasses.replace(components, m_initial=lambda x: 2.0 + 0.0 * _a(x[0]))
    assert copied.hamiltonian is None
    assert isinstance(_problem(copied).hamiltonian_class, derived)


@pytest.mark.parametrize(
    ("lagrangian", "derived"),
    [
        (
            lambda: hm.SeparableLagrangian(control_cost=hm.QuadraticControlCost(control_cost=1.0)),
            hm.SeparableHamiltonian,
        ),
        (_QuadraticLagrangian, hm.DualHamiltonian),
    ],
    ids=["SeparableLagrangian", "a user's LagrangianBase"],
)
def test_a_lagrangian_defined_problem_builds(lagrangian, derived):
    assert isinstance(_problem(_lagrangian_components(lagrangian())).hamiltonian_class, derived)
