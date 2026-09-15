"""The nD FP velocity calls optimal_control on (N, d) batches, and refuses a return of any other shape (#2330).

`compute_fp_velocity_field` called `optimal_control` on grid-shaped ``(nx, ny, d)`` arrays, outside the ``(d,)`` /
``(N, d)`` contract `HamiltonianBase` documents. At b93da74a `CongestionHamiltonian` raised a TypeError, and a user
subclass written to the documented shapes raised a ValueError. A return that broadcast against the grid was spread
into every velocity component: a ``dp`` returning one speed per point gave a velocity 1.98 off the closed form,
silently.

Oracle: for ``H = |p - s x|^2 / (2 (1 + lam m))`` the optimal control is ``-(p - s x) / (1 + lam m)`` at each point's
own position and density at that time step. ``U = a x + b y + c x y + 0.1 n x`` is linear along each axis, so
`np.gradient` recovers ``p`` exactly while ``p`` varies from node to node and from step to step; ``p``, ``x`` and ``m``
each depend on the two axes differently, so a position, momentum or density mis-paired in space or in time, or a
transposed axis, cannot pass. (Review of #2334: with ``U`` linear, no ``x`` term and ``p`` constant in time, six
pairing and layout mutants survived, and so did reading the momentum at another step.)

A multi-population cross density must be one value per node to be paired with the batch; any other size is refused.
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


def _velocity(hamiltonian, cross_density=None):
    grid = TensorProductGrid(
        bounds=[(0.0, 1.0), (0.0, 2.0)], Nx_points=[11, 9], boundary_conditions=no_flux_bc(dimension=2)
    )
    x, y = np.meshgrid(*grid.coordinates, indexing="ij")
    steps = np.arange(4)[:, None, None]
    U = SLOPES[0] * x + SLOPES[1] * y + TWIST * x * y + 0.1 * steps * x
    M = 0.2 + x**2 + 0.3 * y + 0.05 * steps
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        problem = MFGProblem(
            geometry=grid,
            T=0.3,
            Nt=3,
            sigma=0.3,
            components=MFGComponents(m_initial=lambda z: 1.0, u_terminal=lambda z: 0.0, hamiltonian=hamiltonian),
        )
    p = np.stack([SLOPES[0] + TWIST * y + 0.1 * steps, np.broadcast_to(SLOPES[1] + TWIST * x, U.shape)], axis=1)
    shift = np.zeros(2) if isinstance(hamiltonian, CongestionHamiltonian) else SHIFT
    q = p - shift[None, :, None, None] * np.stack([x, y])[None]
    velocity = compute_fp_velocity_field(problem, U, M, hamiltonian, cross_density=cross_density)
    density = M if cross_density is None else cross_density
    return velocity, -q / (1 + LAM * density)[:, None]


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


def test_a_one_population_cross_density_is_read_per_node():
    x, y = np.meshgrid(np.linspace(0.0, 1.0, 11), np.linspace(0.0, 2.0, 9), indexing="ij")
    stack = 1.5 - 0.4 * x + 0.1 * y**2 + 0.02 * np.arange(4)[:, None, None]
    velocity, closed_form = _velocity(_PointOrBatch(), cross_density=stack)
    np.testing.assert_allclose(velocity, closed_form, rtol=0, atol=1e-8)


def test_a_cross_density_of_several_populations_is_refused():
    x, y = np.meshgrid(np.linspace(0.0, 1.0, 11), np.linspace(0.0, 2.0, 9), indexing="ij")
    one = 0.2 + x**2 + 0.3 * y + 0.05 * np.arange(4)[:, None, None]
    with pytest.raises(NotImplementedError, match=r"198 values for 99 nodes"):
        _velocity(_PointOrBatch(), cross_density=np.concatenate([one, one + 0.1], axis=-1))
