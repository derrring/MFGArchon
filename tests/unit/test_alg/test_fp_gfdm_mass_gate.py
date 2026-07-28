"""The GFDM FP path stops instead of renormalising over a clip (#1683).

It clipped, renormalised to the initial mass, and warned only above 1% drift. Every
configuration therefore returned a final mass of exactly 1.0000 -- including one measured
to clip **61%** of the present mass at a single step. Reporting perfect conservation over
that is the defect, not the diagnostic that was missing.

Migrating this path broke **no existing test**, which is the other half of the finding: a
public solver whose plausible configurations fabricate most of their mass had no coverage
of that behaviour at all. These are that coverage.

Two mechanisms drive it, and they call for opposite changes -- measured on a 21-point
grid, `sigma=0.5` clips 61% at `dt*D/dx^2 = 2.5` (five times the explicit-diffusion
limit), while `sigma=0.1` with a steep drift clips 9.6% at `dt*D/dx^2 = 0.1`, where the
driver is advection. The remedy text names both rather than guessing which one bound.
"""

from __future__ import annotations

import pytest

import numpy as np

from mfgarchon.alg.numerical.fp_solvers.fp_gfdm import FPGFDMSolver
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.core.mfg_components import MFGComponents
from mfgarchon.core.mfg_problem import MFGProblem
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import no_flux_bc

N = 21
NT = 10
T = 0.5


def _solver(sigma):
    grid = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[N], boundary_conditions=no_flux_bc(dimension=1))
    problem = MFGProblem(
        geometry=grid,
        Nt=NT,
        T=T,
        sigma=sigma,
        components=MFGComponents(
            m_initial=lambda x: np.exp(-30 * (np.asarray(x) - 0.5) ** 2),
            u_terminal=lambda x: 0.0,
            hamiltonian=SeparableHamiltonian(
                control_cost=QuadraticControlCost(control_cost=1.0),
                coupling=lambda m: m,
                coupling_dm=lambda m: 1.0,
            ),
        ),
    )
    return FPGFDMSolver(problem, collocation_points=np.linspace(0, 1, N).reshape(-1, 1))


def _inputs(drift_scale):
    x = np.linspace(0, 1, N)
    m0 = np.exp(-30 * (x - 0.5) ** 2)
    m0 /= m0.sum()
    return m0, np.tile(drift_scale * (x - 0.5) ** 2, (NT + 1, 1))


def test_a_diffusion_limited_configuration_stops():
    """sigma=0.5 gives dt*D/dx^2 = 2.5, five times the explicit limit.

    It used to return final mass 1.0000 over a 61% clip -- a configuration a user would
    reasonably pick, reporting perfect conservation.
    """
    m0, drift = _inputs(1.0)
    with pytest.raises(ValueError, match="would fabricate"):
        _solver(0.5).solve_fp_system(m0, drift)


def test_an_advection_driven_configuration_stops():
    """sigma=0.1 with a steep drift: dt*D/dx^2 = 0.1, so diffusion is not the binding limit."""
    m0, drift = _inputs(25.0)
    with pytest.raises(ValueError, match="would fabricate"):
        _solver(0.1).solve_fp_system(m0, drift)


def test_the_remedy_names_both_mechanisms():
    """They point opposite ways, so naming only one sends half the readers backwards."""
    m0, drift = _inputs(1.0)
    with pytest.raises(ValueError) as exc:
        _solver(0.5).solve_fp_system(m0, drift)
    message = str(exc.value)
    assert "GFDM FP solve: at t_idx=" in message
    assert "dt*D/dx^2" in message
    assert "drift is" in message
    assert "measure which one binds" in message


def test_a_converging_configuration_still_runs():
    """The gate must not stop a solve that does not clip.

    sigma=0.3 with a moderate drift clips nothing across all ten steps -- the régime this
    path is usable in, and the one a threshold set too tight would destroy.
    """
    m0, drift = _inputs(5.0)
    result = _solver(0.3).solve_fp_system(m0, drift)
    assert np.all(np.isfinite(result))
    assert result.min() >= 0.0


def test_the_scheme_does_not_conserve_mass_and_now_says_so(record_property):
    """Records #1752: removing the renormalisation exposed a defect larger than the clip.

    This configuration clips **nothing** across all ten steps, so no positivity repair is
    involved -- and its mass still goes 1.000000 -> 2.794967, a 179% gain. The per-step
    `M *= mass_initial / mass_current` was not masking the clip; it was masking the
    scheme. Every configuration returned exactly the initial mass because it was forced
    to.

    The assertion is the measurement, not the desired behaviour. It is written to fail if
    the drift **improves**, so fixing #1752 cannot land silently: a conservative
    discretisation would bring this near 1.0 and turn this test red, which is when it
    should be deleted.
    """
    m0, drift = _inputs(5.0)
    result = _solver(0.3).solve_fp_system(m0, drift)
    final = float(result[-1].sum())
    record_property("gfdm_final_mass", final)
    assert final > 2.0, (
        f"final mass {final:.6f} -- if this dropped toward {float(m0.sum()):.1f} the scheme "
        f"became conservative and #1752 is fixed; delete this test rather than relax it"
    )


def test_the_drift_is_reported_at_warning_level(caplog):
    """The renormalisation's removal left this line as the only signal for the drift.

    It was `logger.debug`, which is off by default -- a 179% mass error that nothing
    printed. Pinned because a diagnostic nobody reads is the same failure as no
    diagnostic, and log levels are the kind of thing a later edit lowers without noticing.
    """
    import logging

    m0, drift = _inputs(5.0)
    with caplog.at_level(logging.WARNING, logger="mfgarchon.alg.numerical.fp_solvers.fp_gfdm"):
        _solver(0.3).solve_fp_system(m0, drift)
    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert warnings, "the drift was not reported at WARNING or above"
    assert "1752" in warnings[0].getMessage(), "the message must name the issue tracking the defect"
