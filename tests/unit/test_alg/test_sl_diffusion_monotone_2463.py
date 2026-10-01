"""Every diffusion step of the semi-Lagrangian pair keeps a non-negative field non-negative (#2463).

The step is linear, so it maps every non-negative field to a non-negative one exactly when it maps
each unit vector to one. Backward Euler's matrix is an M-matrix at every step size, so its inverse is
entrywise non-negative. Crank-Nicolson's explicit half has diagonal 1 - D dt / dx^2, negative above
diffusion number 1.

Until #2463 both halves used Crank-Nicolson, and on an attractive problem at sigma = 0.5 the FP
half's density went negative from diffusion number 2.5 on, stopping the solve on the positivity
clip. Here the number is 5, where the Crank-Nicolson step goes negative on the first unit vector it
is handed. Each call site is reached through the solver that owns it, so a call site that stops
reading `SL_DIFFUSION_THETA` fails its own case.
"""

from __future__ import annotations

import pytest

import numpy as np

from mfgarchon import Conditions, MFGProblem, Model
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.factory import create_paired_solvers
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import no_flux_bc, periodic_bc
from mfgarchon.types import NumericalScheme

SIGMA = 1.0
DIFFUSION_NUMBER = 5.0


def _pair(dims: int, bc: str):
    n = 11 if dims == 1 else 7
    boundary = no_flux_bc(dimension=dims) if bc == "no_flux" else periodic_bc(dimension=dims)
    grid = TensorProductGrid(bounds=[(0.0, 1.0)] * dims, Nx_points=[n] * dims, boundary_conditions=boundary)
    hamiltonian = SeparableHamiltonian(control_cost=QuadraticControlCost(lambda_=1.0))
    problem = MFGProblem(
        model=Model(hamiltonian=hamiltonian, volatility=SIGMA),
        domain=grid,
        conditions=Conditions(m_initial=lambda x: 1.0, u_terminal=lambda x: 0.0, T=1.0),
        Nt=10,
    )
    hjb, fp = create_paired_solvers(problem, NumericalScheme.SL_LINEAR)
    dx = 1.0 / (n - 1)
    dt = DIFFUSION_NUMBER * dx**2 / (SIGMA**2 / 2.0)
    return hjb, fp, (n,) * dims, dt


def _unit_vectors(shape, periodic: bool):
    """Every node's unit vector; on a periodic grid the two coincident endpoints carry one value."""
    for index in np.ndindex(shape):
        e = np.zeros(shape)
        e[index] = 1.0
        if periodic:
            for axis in range(len(shape)):
                first = np.take(e, 0, axis=axis)
                last = np.take(e, -1, axis=axis)
                shared = np.maximum(first, last)
                idx0 = [slice(None)] * len(shape)
                idx1 = [slice(None)] * len(shape)
                idx0[axis], idx1[axis] = 0, -1
                e[tuple(idx0)] = shared
                e[tuple(idx1)] = shared
        yield index, e


CASES = [(1, "no_flux"), (1, "periodic"), (2, "no_flux")]


@pytest.mark.parametrize(("dims", "bc"), CASES)
def test_the_hjb_half_diffusion_step_keeps_a_non_negative_field_non_negative(dims, bc):
    hjb, _, shape, dt = _pair(dims, bc)
    worst = min(
        (float(hjb._apply_diffusion(e.copy(), dt).min()), index) for index, e in _unit_vectors(shape, bc == "periodic")
    )
    assert worst[0] >= -1e-14, (
        f"the HJB half's {dims}-D {bc} diffusion step at diffusion number {DIFFUSION_NUMBER} turned the unit "
        f"vector at {worst[1]} negative ({worst[0]:.3e}): it is no longer backward Euler (#2463)"
    )


@pytest.mark.parametrize(("dims", "bc"), CASES)
def test_the_fp_half_diffusion_step_keeps_a_density_non_negative(dims, bc):
    """Zero drift, so the splat moves nothing and the step is the diffusion alone. A negative output
    raises here, through the solver's own positivity clip, before the assertion is reached."""
    _, fp, shape, dt = _pair(dims, bc)
    worst = None
    for index, e in _unit_vectors(shape, bc == "periodic"):
        if dims == 1:
            out = fp._adjoint_sl_step_1d(e.copy(), np.zeros(shape), dt, SIGMA)
        else:
            out = fp._adjoint_sl_step_nd(e.copy(), tuple(np.zeros(shape) for _ in range(dims)), dt, SIGMA)
        if worst is None or float(out.min()) < worst[0]:
            worst = (float(out.min()), index)
    assert worst[0] >= -1e-14, (
        f"the FP half's {dims}-D {bc} diffusion step at diffusion number {DIFFUSION_NUMBER} turned the unit "
        f"vector at {worst[1]} negative ({worst[0]:.3e}) (#2463)"
    )
