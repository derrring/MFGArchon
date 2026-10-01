"""The SL pair's diffusion theta reaches every diffusion step in both halves, and the clip says how to set it (#2463).

A theta-step is linear, so it keeps every non-negative field non-negative exactly when it maps each unit
vector to one. It does so up to diffusion number ``positivity_edge(theta)`` = (2 - theta) / (4 (1 - theta)^2):
3/2 for Crank-Nicolson, every step for backward Euler. Each case runs at its theta's edge, or at 5 for
backward Euler, so a call site that ignores the theta it was given fails the backward-Euler case.

The theta enters through `SLConfig.diffusion_theta`, as a Safe-Mode user sets it, and reaches the FP half
only through the pair factory -- so the FP cases also read the hand-off.
"""

from __future__ import annotations

import pytest

import numpy as np

from mfgarchon import Conditions, MFGProblem, Model
from mfgarchon.config import MFGSolverConfig
from mfgarchon.config.mfg_methods import SLConfig
from mfgarchon.config.translator import hjb_config_to_kwargs
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.factory import create_paired_solvers
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import no_flux_bc, periodic_bc
from mfgarchon.types import NumericalScheme

SIGMA = 1.0

#: (theta, diffusion number): Crank-Nicolson at its edge, 3/2; backward Euler, which has none, at 5.
EDGES = [(1.0, 5.0), (0.5, 1.5)]
GEOMETRIES = [(1, "no_flux"), (1, "periodic"), (2, "no_flux"), (2, "periodic")]


def _config(theta: float) -> MFGSolverConfig:
    cfg = MFGSolverConfig()
    cfg.hjb.sl = SLConfig(diffusion_theta=theta)
    return cfg


def _pair(dims: int, bc: str, theta: float, number: float):
    n = 11 if dims == 1 else 7
    boundary = no_flux_bc(dimension=dims) if bc == "no_flux" else periodic_bc(dimension=dims)
    grid = TensorProductGrid(bounds=[(0.0, 1.0)] * dims, Nx_points=[n] * dims, boundary_conditions=boundary)
    problem = MFGProblem(
        model=Model(hamiltonian=SeparableHamiltonian(control_cost=QuadraticControlCost(lambda_=1.0)), volatility=SIGMA),
        domain=grid,
        conditions=Conditions(m_initial=lambda x: 1.0, u_terminal=lambda x: 0.0, T=1.0),
        Nt=10,
    )
    kwargs = hjb_config_to_kwargs(_config(theta).hjb, NumericalScheme.SL_LINEAR)
    hjb, fp = create_paired_solvers(problem, NumericalScheme.SL_LINEAR, hjb_config=kwargs)
    dt = number * (1.0 / (n - 1)) ** 2 / (SIGMA**2 / 2.0)
    return hjb, fp, (n,) * dims, dt


def _unit_vectors(shape, periodic: bool):
    """Every node's unit vector; on a periodic grid the two coincident endpoints carry one value."""
    for index in np.ndindex(shape):
        e = np.zeros(shape)
        e[index] = 1.0
        if periodic:
            for axis in range(len(shape)):
                first, last = [slice(None)] * len(shape), [slice(None)] * len(shape)
                first[axis], last[axis] = 0, -1
                shared = np.maximum(e[tuple(first)], e[tuple(last)])
                e[tuple(first)] = shared
                e[tuple(last)] = shared
        yield index, e


@pytest.mark.parametrize(("theta", "number"), EDGES)
@pytest.mark.parametrize(("dims", "bc"), GEOMETRIES)
def test_the_hjb_half_diffusion_step_keeps_a_non_negative_field_non_negative(dims, bc, theta, number):
    hjb, _, shape, dt = _pair(dims, bc, theta, number)
    assert hjb.diffusion_theta == theta
    worst = min(
        (float(hjb._apply_diffusion(e.copy(), dt).min()), index) for index, e in _unit_vectors(shape, bc == "periodic")
    )
    assert worst[0] >= -1e-14, (
        f"the HJB half's {dims}-D {bc} diffusion step at theta {theta}, diffusion number {number}, turned the unit "
        f"vector at {worst[1]} negative ({worst[0]:.3e}): a call site is not using the theta it was given (#2463)"
    )


@pytest.mark.parametrize(("theta", "number"), EDGES)
@pytest.mark.parametrize(("dims", "bc"), GEOMETRIES)
def test_the_fp_half_diffusion_step_keeps_a_density_non_negative(dims, bc, theta, number):
    """Zero drift, so the splat moves nothing and the step is the diffusion alone. A negative output raises
    here, through the solver's own positivity clip, before the assertion is reached."""
    _, fp, shape, dt = _pair(dims, bc, theta, number)
    assert fp.diffusion_theta == theta, "the pair factory did not hand the HJB half's theta to the FP half"
    worst = None
    for index, e in _unit_vectors(shape, bc == "periodic"):
        if dims == 1:
            out = fp._adjoint_sl_step_1d(e.copy(), np.zeros(shape), dt, SIGMA)
        else:
            out = fp._adjoint_sl_step_nd(e.copy(), tuple(np.zeros(shape) for _ in range(dims)), dt, SIGMA)
        if worst is None or float(out.min()) < worst[0]:
            worst = (float(out.min()), index)
    assert worst[0] >= -1e-14, (
        f"the FP half's {dims}-D {bc} diffusion step at theta {theta}, diffusion number {number}, turned the unit "
        f"vector at {worst[1]} negative ({worst[0]:.3e}) (#2463)"
    )


def test_the_clip_names_a_theta_that_lets_the_solve_through():
    """The advice is executed, not only printed. #2463's case: attractive coupling f(m) = -m, sigma = 0.5,
    21 points, Nt = 10, diffusion number 5. At the default Crank-Nicolson the FP density goes negative and
    the clip stops the solve; its message names diffusion_theta=1.0, and with that set the solve converges."""
    grid = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[21], boundary_conditions=no_flux_bc(dimension=1))
    x = grid.coordinates[0]
    mass = grid.integrate(np.exp(-10 * (x - 0.5) ** 2))
    hamiltonian = SeparableHamiltonian(
        control_cost=QuadraticControlCost(lambda_=1.0), coupling=lambda m: -m, coupling_dm=lambda m: -1.0
    )
    problem = MFGProblem(
        model=Model(hamiltonian=hamiltonian, volatility=0.5),
        domain=grid,
        conditions=Conditions(
            m_initial=lambda y: np.exp(-10 * (np.asarray(y) - 0.5) ** 2) / mass, u_terminal=lambda y: 0.0, T=1.0
        ),
        Nt=10,
    )
    with pytest.raises(ValueError, match=r"diffusion_theta=1\.0"):
        problem.solve(scheme=NumericalScheme.SL_LINEAR, max_iterations=300, tolerance=1e-8, verbose=False)

    result = problem.solve(
        scheme=NumericalScheme.SL_LINEAR, config=_config(1.0), max_iterations=300, tolerance=1e-8, verbose=False
    )
    assert result.converged
    assert np.asarray(result.M).min() >= 0.0


def test_the_periodic_fp_step_stops_on_a_negative_density_rather_than_returning_it():
    """The FP half's periodic 1-D path returned its diffusion result unchecked until #2463, the only path
    that did. Crank-Nicolson at diffusion number 5 turns a unit vector negative there; the step must stop."""
    _, fp, shape, dt = _pair(1, "periodic", 0.5, 5.0)
    e = np.zeros(shape)
    e[5] = 1.0
    with pytest.raises(ValueError, match="FP-SL positivity clip"):
        fp._adjoint_sl_step_1d(e, np.zeros(shape), dt, SIGMA)


@pytest.mark.parametrize(("theta", "number", "named"), [(0.5, 1.25, False), (0.5, 2.0, True), (1.0, 10.0, False)])
def test_the_clip_blames_the_diffusion_step_only_past_its_edge(theta, number, named):
    """A negative density at diffusion number 1.25 is not Crank-Nicolson's doing: its edge is 3/2. Naming
    diffusion_theta there would send the user to a remedy that does not help (a source_term measured it).
    At backward Euler the diffusion step is never the cause, so it is never named."""
    _, fp, shape, _ = _pair(1, "no_flux", theta, 1.0)
    m = np.ones(shape)
    m[3] = -0.5
    with pytest.raises(ValueError) as raised:
        fp._clip_nonneg(m, diffusion_number=number)
    assert ("diffusion_theta=1.0" in str(raised.value)) is named
