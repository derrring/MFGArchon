"""The nD FP velocity calls optimal_control on (N, d) batches, and refuses a return of any other shape (#2330).

`compute_fp_velocity_field` called `optimal_control` on grid-shaped ``(nx, ny, d)`` arrays, outside the ``(d,)`` /
``(N, d)`` contract `HamiltonianBase` documents. At b93da74a `CongestionHamiltonian` raised a TypeError, and a user
subclass written to the documented shapes raised a ValueError. A return that broadcast against the grid was spread
into every velocity component: a ``dp`` returning one speed per point gave a velocity 1.98 off the closed form,
silently.

Oracle: for ``H = |p - s x|^2 / (2 (1 + lam m))`` the optimal control is ``-(p - s x) / (1 + lam m)`` at each point's
own position and density. ``U = a x + b y + c x y`` is linear along each axis, so `np.gradient` recovers ``p``
exactly while ``p`` still varies from node to node; ``p``, ``x`` and ``m`` each depend on the two axes differently,
so a mis-paired position, momentum or density, or a transposed axis, cannot pass (review of #2334: with ``U``
linear and no ``x`` term, seven such mutants survived).
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
TWIST = 0.4
SHIFT = np.array([0.5, -0.8])


class _PointOrBatch(HamiltonianBase):
    """A user Hamiltonian written to the shapes `HamiltonianBase` documents, with the base finite-difference dp."""

    def __call__(self, x, m, p, t=0.0):
        q = np.asarray(p) - SHIFT * np.asarray(x)
        if q.ndim == 1:
            return 0.5 * float(q @ q) / (1.0 + LAM * float(m))
        return 0.5 * np.sum(q * q, axis=1) / (1.0 + LAM * np.asarray(m).ravel())


class _ScalarSpeed(_PointOrBatch):
    def dp(self, x, m, p, t=0.0):
        return np.linalg.norm(np.asarray(p), axis=-1) / (1.0 + LAM * np.asarray(m))


class _ComponentsFirst(_PointOrBatch):
    """The right numbers in the wrong layout: ``(d, N)``, which has the size of ``(N, d)``."""

    def dp(self, x, m, p, t=0.0):
        return (np.asarray(p) / (1.0 + LAM * np.asarray(m))[:, None]).T


def _velocity(hamiltonian):
    grid = TensorProductGrid(
        bounds=[(0.0, 1.0), (0.0, 2.0)], Nx_points=[11, 9], boundary_conditions=no_flux_bc(dimension=2)
    )
    x, y = np.meshgrid(*grid.coordinates, indexing="ij")
    U = np.stack([SLOPES[0] * x + SLOPES[1] * y + TWIST * x * y + 0.1 * n for n in range(4)])
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
    p = np.stack([SLOPES[0] + TWIST * y, SLOPES[1] + TWIST * x])
    shift = np.zeros(2) if isinstance(hamiltonian, CongestionHamiltonian) else SHIFT
    q = p - shift[:, None, None] * np.stack([x, y])
    closed_form = -q[None] / (1 + LAM * M)[:, None]
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


@pytest.mark.parametrize("hamiltonian", [_ScalarSpeed(), _ComponentsFirst()], ids=["one_per_point", "components_first"])
def test_a_control_of_the_wrong_shape_is_refused(hamiltonian):
    with pytest.raises(ValueError, match=r"must return one control vector per point, shape \(99, 2\)"):
        _velocity(hamiltonian)
