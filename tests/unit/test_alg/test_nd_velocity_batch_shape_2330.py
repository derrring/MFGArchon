"""The nD FP velocity calls optimal_control on (N, d) batches, and refuses a return of any other shape (#2330).

`compute_fp_velocity_field` called `optimal_control` on grid-shaped ``(nx, ny, d)`` arrays, outside the ``(d,)`` /
``(N, d)`` contract `HamiltonianBase` documents. At b93da74a `CongestionHamiltonian` raised a TypeError, and so did
a user subclass written to the documented shapes. A return that was not ``(nx, ny, d)`` was broadcast into every
velocity component: a ``dp`` returning one speed per point gave a velocity 1.98 off the closed form, silently.

Oracle: for ``H = |p|^2 / (2 (1 + lam m))`` the optimal control is ``-p / (1 + lam m)`` at each point's own density.
``U`` is linear with different slopes on the two axes, so `np.gradient` recovers ``p`` exactly and the check is on
the per-point pairing alone; ``m`` depends on both axes differently, so a transposed axis cannot pass.
"""

from __future__ import annotations

import warnings

import pytest

import numpy as np

from mfgarchon import MFGProblem
from mfgarchon.alg.numerical.coupling.fixed_point_utils import compute_fp_velocity_field
from mfgarchon.core.hamiltonian import CongestionHamiltonian, HamiltonianBase, QuadraticControlCost
from mfgarchon.core.mfg_components import MFGComponents
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import no_flux_bc

LAM = 2.0
SLOPES = (0.7, -1.3)


class _PointOrBatch(HamiltonianBase):
    """A user Hamiltonian written to the shapes `HamiltonianBase` documents, with the base finite-difference dp."""

    def __call__(self, x, m, p, t=0.0):
        p = np.asarray(p)
        if p.ndim == 1:
            return 0.5 * float(p @ p) / (1.0 + LAM * float(m))
        return 0.5 * np.sum(p * p, axis=1) / (1.0 + LAM * np.asarray(m).ravel())


class _ScalarSpeed(_PointOrBatch):
    def dp(self, x, m, p, t=0.0):
        return np.linalg.norm(np.asarray(p), axis=-1) / (1.0 + LAM * np.asarray(m))


def _velocity(hamiltonian):
    grid = TensorProductGrid(
        bounds=[(0.0, 1.0), (0.0, 2.0)], Nx_points=[11, 9], boundary_conditions=no_flux_bc(dimension=2)
    )
    x, y = np.meshgrid(*grid.coordinates, indexing="ij")
    U = np.stack([SLOPES[0] * x + SLOPES[1] * y + 0.1 * n for n in range(4)])
    M = np.stack([0.2 + x**2 + 0.3 * y + 0.05 * n for n in range(4)])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        problem = MFGProblem(
            geometry=grid,
            T=0.3,
            Nt=3,
            sigma=0.3,
            components=MFGComponents(m_initial=lambda z: 1.0, u_terminal=lambda z: 0.0, hamiltonian=hamiltonian),
        )
    closed_form = -np.stack([SLOPES[0] / (1 + LAM * M), SLOPES[1] / (1 + LAM * M)], axis=1)
    return compute_fp_velocity_field(problem, U, M, hamiltonian), closed_form


@pytest.mark.parametrize(
    "hamiltonian",
    [
        CongestionHamiltonian(
            control_cost=QuadraticControlCost(control_cost=1.0), congestion_factor=lambda m: 1.0 + LAM * np.asarray(m)
        ),
        _PointOrBatch(),
    ],
    ids=["CongestionHamiltonian", "user_subclass"],
)
def test_the_velocity_is_the_optimal_control_at_each_points_own_density(hamiltonian):
    velocity, closed_form = _velocity(hamiltonian)
    assert velocity.shape == closed_form.shape
    np.testing.assert_allclose(velocity, closed_form, rtol=0, atol=1e-8)


def test_a_control_of_the_wrong_shape_is_refused():
    with pytest.raises(ValueError, match=r"must return one control vector per point, shape \(99, 2\)"):
        _velocity(_ScalarSpeed())
