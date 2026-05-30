"""The adjoint semi-Lagrangian FP positivity clip is fail-LOUD, not fail-silent.

``FPSLSolver`` clips negative density (``np.maximum(m, 0)``) at four points:
cubic/quintic splatting (which oscillates) and the Crank-Nicolson / ADI diffusion
step (which is not monotone). That clip deletes negative mass and therefore INJECTS
probability, silently violating conservation. The solver now routes every clip through
``_clip_nonneg`` and emits one warning per ``solve_fp_system`` call when the injected
mass exceeds a relative threshold -- the same diagnostic added to ``WeakFormFPSolver``
in Issue #1147. The warning is behaviour-additive (the clipped values are unchanged).
"""

from __future__ import annotations

import logging

import numpy as np

from mfgarchon import MFGProblem
from mfgarchon.alg.numerical.fp_solvers.fp_semi_lagrangian_adjoint import FPSLSolver
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.core.mfg_problem import MFGComponents
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import no_flux_bc

_CLIP_LOGGER = "mfgarchon.alg.numerical.fp_solvers.fp_semi_lagrangian_adjoint"


def _capture_warnings(logger_name):
    """Attach a record-collecting handler (mfgarchon loggers do not propagate to caplog)."""
    records: list[logging.LogRecord] = []

    class _Collector(logging.Handler):
        def emit(self, record):
            records.append(record)

    logger = logging.getLogger(logger_name)
    handler = _Collector()
    logger.addHandler(handler)
    prev_level = logger.level
    logger.setLevel(logging.WARNING)
    return records, logger, handler, prev_level


def _problem(n=41, nt=20):
    grid = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[n], boundary_conditions=no_flux_bc(dimension=1))
    H = SeparableHamiltonian(control_cost=QuadraticControlCost(control_cost=1.0))
    comp = MFGComponents(hamiltonian=H, m_initial=lambda x: np.ones_like(x), u_terminal=lambda x: x * 0)
    return MFGProblem(geometry=grid, components=comp, T=0.5, Nt=nt, sigma=0.3, coupling_coefficient=0.5)


def _run(fp, m0, U, logger_name=_CLIP_LOGGER):
    records, logger, handler, prev = _capture_warnings(logger_name)
    try:
        fp.solve_fp_system(m0, potential_field=U)
    finally:
        logger.removeHandler(handler)
        logger.setLevel(prev)
    return [r.getMessage() for r in records]


def test_clip_warns_when_cubic_splatting_injects_mass():
    """Cubic splatting under a steep confining drift oscillates; the clip then injects
    mass and the solver warns (once)."""
    n, nt = 41, 20
    prob = _problem(n=n, nt=nt)
    fp = FPSLSolver(prob, interpolation_method="cubic")
    x = np.linspace(0.0, 1.0, n)
    m0 = np.exp(-60 * (x - 0.4) ** 2)
    m0 /= m0.sum() * (x[1] - x[0])
    # Steep confining potential -> velocity alpha = -grad(U) is large -> cubic splat rings.
    U = np.tile(25.0 * (x - 0.5) ** 2, (nt + 1, 1))

    msgs = _run(fp, m0, U)
    assert any("positivity clip injected mass" in m for m in msgs), f"expected a clip warning, got {msgs}"


def test_no_clip_warning_for_linear_pure_diffusion():
    """Linear splatting preserves positivity and Crank-Nicolson diffusion of a smooth
    Gaussian (cell-Peclet stable) does not undershoot, so the clip never injects mass and
    no warning fires -- the warning is a real signal, not noise."""
    n, nt = 41, 20
    prob = _problem(n=n, nt=nt)
    fp = FPSLSolver(prob, interpolation_method="linear")
    x = np.linspace(0.0, 1.0, n)
    m0 = np.exp(-40 * (x - 0.5) ** 2)
    m0 /= m0.sum() * (x[1] - x[0])
    U = np.zeros((nt + 1, n))  # no drift -> pure diffusion

    msgs = _run(fp, m0, U)
    assert not any("positivity clip injected mass" in m for m in msgs), f"unexpected clip warning: {msgs}"


def test_clip_warning_fires_at_most_once_per_solve():
    """The diagnostic is once-per-solve: the flag resets each solve_fp_system call but
    does not spam across the (Nt) time steps within a single solve."""
    n, nt = 41, 20
    prob = _problem(n=n, nt=nt)
    fp = FPSLSolver(prob, interpolation_method="cubic")
    x = np.linspace(0.0, 1.0, n)
    m0 = np.exp(-60 * (x - 0.4) ** 2)
    m0 /= m0.sum() * (x[1] - x[0])
    U = np.tile(25.0 * (x - 0.5) ** 2, (nt + 1, 1))

    msgs = _run(fp, m0, U)
    n_clip = sum("positivity clip injected mass" in m for m in msgs)
    assert n_clip <= 1, f"warning should fire at most once per solve, fired {n_clip} times"
    # And the flag resets: a second solve can warn again.
    msgs2 = _run(fp, m0, U)
    assert sum("positivity clip injected mass" in m for m in msgs2) <= 1
