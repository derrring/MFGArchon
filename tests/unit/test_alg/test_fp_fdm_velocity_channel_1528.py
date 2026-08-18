"""The nD FDM velocity channel: reachability and honesty (Issue #1528 phase 2).

Two defects, both on `solve_fp_nd_full_system`:

1. The scalar drift coefficient was resolved unconditionally at function scope,
   before the drift channel was consulted. `fp_drift_coefficient` raises for any
   Hamiltonian whose optimal control is not ``-grad(U)/control_cost`` (MAXIMIZE,
   non-quadratic, regularized), so a MAXIMIZE problem could not run through the
   nD FDM solver *even when supplying* ``velocity_field`` -- the channel that
   exists precisely to carry a precomputed alpha* for those Hamiltonians.

Only ``divergence_upwind`` actually reads ``interface_velocity``, so only that
scheme leaves the coefficient unread; every other path derives its drift from U
and still resolves it.

2. Supplying a velocity to a scheme that cannot read it proceeded silently, and it did
   **not** fall back to ``-c*grad(U)`` as this docstring and the code comments used to
   claim. The ``velocity_field is not None`` branch replaces U with a zero-U dispatcher,
   so *both* drift channels were discarded and the solve ran at zero drift -- returning a
   pure-diffusion density that looks converged and conserves mass. Measured before the
   guard: ``gradient_upwind`` with a velocity was bit-identical to the pure-diffusion
   reference (``|B-C| = 0.000e+00``) while differing from the U-driven run by ``2.1e-2``.
   That was the reachable path for every non-separable Hamiltonian, since
   ``resolve_fp_drift_kwargs`` routes those down the velocity channel precisely because
   ``-c*grad(U)`` cannot represent their drift. It now raises (#1632).

Each test fails if the fix is reverted.
"""

import pytest

import numpy as np

from mfgarchon import Conditions, MFGProblem, Model
from mfgarchon.alg.numerical.fp_solvers.fp_fdm_time_stepping import solve_fp_nd_full_system
from mfgarchon.core.hamiltonian import (
    OptimizationSense,
    QuadraticControlCost,
    SeparableHamiltonian,
)
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import no_flux_bc

N = 8
NT = 4


def _problem(sense: OptimizationSense = OptimizationSense.MINIMIZE) -> MFGProblem:
    hamiltonian = SeparableHamiltonian(
        control_cost=QuadraticControlCost(lambda_=1.0, sense=sense),
        sense=sense,
    )
    return MFGProblem(
        model=Model(hamiltonian=hamiltonian, sigma=0.2),
        domain=TensorProductGrid(
            bounds=[(0.0, 1.0), (0.0, 1.0)],
            Nx_points=[N, N],
            boundary_conditions=no_flux_bc(dimension=2),
        ),
        conditions=Conditions(u_terminal=lambda x: 0.0, m_initial=lambda x: 1.0, T=0.2),
        Nt=NT,
    )


def _uniform_density() -> np.ndarray:
    m0 = np.ones((N, N))
    return m0 / m0.sum()


def _velocity(vx: float = 0.0, vy: float = 0.0) -> np.ndarray:
    vel = np.zeros((NT + 1, 2, N, N))
    vel[:, 0, ...] = vx
    vel[:, 1, ...] = vy
    return vel


# --- defect 1: the coefficient must not be resolved where it is not consumed ---


@pytest.mark.parametrize("sense", [OptimizationSense.MINIMIZE, OptimizationSense.MAXIMIZE])
@pytest.mark.parametrize("scheme", ["divergence_upwind", "flux"])
def test_velocity_channel_runs_for_both_senses(sense, scheme):
    """MAXIMIZE previously raised NotImplementedError from the eager coefficient read.

    Parametrized over the legacy alias too: the skip resolves `flux` -> `divergence_upwind`
    before testing membership, so keying it on the raw scheme name would silently un-do the
    widening for anyone spelling the scheme the old way.
    """
    result = solve_fp_nd_full_system(
        _uniform_density(), None, _problem(sense), velocity_field=_velocity(vx=0.3), advection_scheme=scheme
    )

    assert result.shape == (NT + 1, N, N)
    assert np.isfinite(result).all()
    assert result[-1].sum() == pytest.approx(1.0, abs=1e-9), "no-flux walls must conserve mass"


def test_maximize_still_rejected_on_the_u_channel():
    """The guard is not weakened: deriving the drift from U for MAXIMIZE is still wrong physics."""
    u_solution = np.zeros((NT + 1, N, N))
    with pytest.raises((NotImplementedError, ValueError)):
        solve_fp_nd_full_system(_uniform_density(), u_solution, _problem(OptimizationSense.MAXIMIZE))


def test_u_channel_unchanged_for_minimize():
    """Regression: the path that legitimately consumes the coefficient still resolves it."""
    u_solution = np.zeros((NT + 1, N, N))
    u_solution[:] = np.add.outer(np.linspace(0.0, 1.0, N) ** 2, np.zeros(N))

    result = solve_fp_nd_full_system(_uniform_density(), u_solution, _problem())

    assert np.isfinite(result).all()
    assert result[-1].sum() == pytest.approx(1.0, abs=1e-9)


@pytest.mark.parametrize("scheme", ["gradient_centered", "gradient_upwind", "divergence_centered"])
def test_velocity_on_a_non_consuming_scheme_raises(scheme):
    """Issue #1632: a velocity these schemes cannot read must not be silently dropped.

    Catches the reappearance of a *silent* wrong answer, not a crash. Before the guard
    this call returned a finite, mass-conserving, converged-looking density computed at
    zero drift, because the `velocity_field is not None` branch had already swapped U for
    a zero-U dispatcher. Nothing in the output distinguished it from a correct solve --
    which is why an assertion on finiteness or mass cannot serve as the pin here.
    """
    with pytest.raises(NotImplementedError, match="1632"):
        solve_fp_nd_full_system(
            _uniform_density(), None, _problem(), velocity_field=_velocity(vx=0.3), advection_scheme=scheme
        )


@pytest.mark.parametrize("scheme", ["gradient_centered", "gradient_upwind", "divergence_centered"])
def test_a_zero_velocity_is_not_an_error(scheme):
    """A zero velocity with NO U to displace is the one accepted case.

    The ground is not "a zero velocity is harmless" -- it is not the velocity that gets
    discarded. The `velocity_field is not None` branch replaces U with a zero-U dispatcher
    whatever the velocity's magnitude, so the only safe case is one where there was no U to
    lose. `FPFDMSolver` reaches every scheme this way on its diffusion-only paths
    (`_internal_velocity` is set whenever `drift_field` is an ndarray), which is why raising
    unconditionally broke the torus and mass-leak suites and had to be narrowed.
    """
    result = solve_fp_nd_full_system(
        _uniform_density(), None, _problem(), velocity_field=_velocity(), advection_scheme=scheme
    )
    assert np.isfinite(result).all()


@pytest.mark.parametrize("scheme", ["gradient_centered", "gradient_upwind", "divergence_centered"])
def test_a_zero_velocity_alongside_a_real_u_still_raises(scheme):
    """The narrowing must not open the hole it was narrowing around.

    A zero velocity supplied *with* a value function is not harmless: the zero-U dispatcher
    displaces that U, so the solve runs at zero drift and produces exactly the silent
    pure-diffusion answer this guard exists to stop -- measured at 2.1e-2 from the correct
    answer, the same magnitude as the defect itself. Keying the guard on the velocity's
    magnitude alone would accept it.
    """
    u_solution = np.zeros((NT + 1, N, N))
    u_solution[:] = np.add.outer(np.linspace(0.0, 1.0, N) ** 2, np.zeros(N))
    with pytest.raises(NotImplementedError, match="1632"):
        solve_fp_nd_full_system(
            _uniform_density(), u_solution, _problem(), velocity_field=_velocity(), advection_scheme=scheme
        )


def test_a_misspelled_scheme_reports_itself_as_such():
    """The velocity guard must not pre-empt scheme-name validation and mis-attribute a typo."""
    with pytest.raises(ValueError, match="Unknown advection_scheme"):
        solve_fp_nd_full_system(
            _uniform_density(), None, _problem(), velocity_field=_velocity(vx=0.3), advection_scheme="not_a_scheme"
        )


@pytest.mark.parametrize("scheme", ["gradient_centered", "gradient_upwind", "divergence_centered"])
def test_the_raise_names_the_scheme_and_the_way_out(scheme):
    """The diagnostic must be actionable and greppable, not merely raised."""
    with pytest.raises(NotImplementedError) as exc:
        solve_fp_nd_full_system(
            _uniform_density(), None, _problem(), velocity_field=_velocity(vx=0.3), advection_scheme=scheme
        )
    message = str(exc.value)
    assert scheme in message, "the offending scheme must be named"
    assert "divergence_upwind" in message, "the accept-list must be shown"
    assert "U_solution_for_drift" in message, "the internal parameter must be named"
    assert "potential_field" in message, "the PUBLIC parameter a caller can actually type must be named"


def test_callable_drift_channel_also_runs_for_maximize():
    """The callable-drift channel never consumes the coefficient either, so it widens too.

    `solve_timestep_explicit_with_drift` takes no coupling coefficient, so lazy
    resolution legitimately skips it here as well -- a second (correct) widening
    beyond the velocity_field case the issue was filed for.
    """
    result = solve_fp_nd_full_system(
        _uniform_density(),
        None,
        _problem(OptimizationSense.MAXIMIZE),
        drift_field=lambda t, x, m: np.zeros((2, N, N)),
    )
    assert np.isfinite(result).all()
    assert result[-1].sum() == pytest.approx(1.0, abs=1e-9)


# --- the accept-list must be truthful ---------------------------------------


def test_consuming_scheme_actually_honors_the_velocity():
    """Pins that the accept-list is truthful: a different velocity must change the answer.

    Without this, the guard could 'pass' by whitelisting a scheme that also ignores
    the parameter -- rejecting the honest schemes and silently dropping the velocity
    on the accepted one.
    """
    still = solve_fp_nd_full_system(
        _uniform_density(), None, _problem(), velocity_field=_velocity(), advection_scheme="divergence_upwind"
    )
    moving = solve_fp_nd_full_system(
        _uniform_density(),
        None,
        _problem(),
        velocity_field=_velocity(vx=2.5, vy=-1.7),
        advection_scheme="divergence_upwind",
    )

    assert not np.array_equal(still, moving), "velocity_field was discarded by the accepted scheme"
    assert np.abs(still - moving).max() > 1e-3
