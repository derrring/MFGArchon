"""The weak-form FP positivity clip is fail-LOUD, not fail-silent.

``WeakFormFPSolver.solve_fp_system`` clips negative density (``np.maximum(M, 0)``)
because the Galerkin/MLS advection is not an M-matrix. That clip deletes negative
mass and therefore INJECTS probability, silently violating conservation. The solver
now emits one warning per solve when the injected mass exceeds a relative threshold,
so the M-matrix violation is visible instead of hidden (kernel fail-fast). Shared by
FEM and meshless; the warning is behaviour-additive (no path changes).
"""

from __future__ import annotations

import logging

import numpy as np

from mfgarchon import MFGProblem
from mfgarchon.alg.numerical.meshless_galerkin.fp_solver import MeshlessGalerkinFPSolver
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.core.mfg_problem import MFGComponents
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import no_flux_bc

_CLIP_LOGGER = "mfgarchon.alg.numerical.weak_form_fp_solver"


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


def _meshless_fp(n=21):
    grid = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[n], boundary_conditions=no_flux_bc(dimension=1))
    H = SeparableHamiltonian(control_cost=QuadraticControlCost(control_cost=1.0))
    comp = MFGComponents(hamiltonian=H, m_initial=lambda x: np.ones_like(x), u_terminal=lambda x: x * 0)
    problem = MFGProblem(geometry=grid, components=comp, T=0.5, Nt=20, sigma=0.3, coupling_coefficient=0.5)
    cloud = np.linspace(0.0, 1.0, n)[:, None]
    return MeshlessGalerkinFPSolver(problem, collocation_points=cloud, delta=3.5 / (n - 1))


def test_clip_warns_when_it_injects_mass():
    """A steep drift makes the central Galerkin advection undershoot; the clip then
    injects mass and the solver warns (once)."""
    fp = _meshless_fp()
    x = fp._disc.dof_coordinates[:, 0]
    m0 = np.exp(-40 * (x - 0.5) ** 2)
    m0 /= float((fp._M @ m0).sum())
    steep_drift = np.tile(10.0 * (x - 0.5) ** 2, (fp.problem.Nt + 1, 1))  # large gradient -> large velocity

    records, logger, handler, prev = _capture_warnings(_CLIP_LOGGER)
    try:
        fp.solve_fp_system(m0, drift_field=steep_drift)
    finally:
        logger.removeHandler(handler)
        logger.setLevel(prev)

    msgs = [r.getMessage() for r in records]
    assert any("positivity clip injected mass" in m for m in msgs), f"expected a clip warning, got {msgs}"


def test_no_clip_warning_on_pure_diffusion():
    """Pure diffusion (no drift) does not undershoot, so the clip never injects mass and
    no warning is emitted -- the warning is a real signal, not noise."""
    fp = _meshless_fp()
    x = fp._disc.dof_coordinates[:, 0]
    m0 = np.exp(-40 * (x - 0.5) ** 2)
    m0 /= float((fp._M @ m0).sum())

    records, logger, handler, prev = _capture_warnings(_CLIP_LOGGER)
    try:
        fp.solve_fp_system(m0, drift_field=None)
    finally:
        logger.removeHandler(handler)
        logger.setLevel(prev)

    assert not any("positivity clip injected mass" in r.getMessage() for r in records)
