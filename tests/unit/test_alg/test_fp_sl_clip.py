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

import pytest

import numpy as np

from mfgarchon import MFGProblem
from mfgarchon.alg.numerical.fp_solvers.fp_semi_lagrangian_adjoint import FPSLSolver
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.core.mfg_problem import MFGComponents
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import no_flux_bc


def _problem(n=41, nt=20):
    grid = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[n], boundary_conditions=no_flux_bc(dimension=1))
    H = SeparableHamiltonian(control_cost=QuadraticControlCost(control_cost=1.0))
    comp = MFGComponents(hamiltonian=H, m_initial=lambda x: np.ones_like(x), u_terminal=lambda x: x * 0)
    return MFGProblem(geometry=grid, components=comp, T=0.5, Nt=nt, sigma=0.3, coupling_coefficient=0.5)


def test_a_clip_that_injects_mass_stops_the_solve():
    """Cubic splatting undershoots; the solve now stops rather than repairing (#1683).

    It used to warn once per solve and return the clipped density, so a run whose mass
    the clip had moved came back finite, non-negative, and indistinguishable from a
    healthy one. Measured across five configurations of this scheme, four clip nothing
    at all and the fifth clips 11.9% while drifting 10.2% in mass -- there is no régime
    between round-off and failure here, which is why one threshold separates them.
    """
    n, nt = 41, 20
    prob = _problem(n=n, nt=nt)
    fp = FPSLSolver(prob, interpolation_method="cubic")
    x = np.linspace(0.0, 1.0, n)
    m0 = np.exp(-60 * (x - 0.4) ** 2)
    m0 /= m0.sum() * (x[1] - x[0])
    U = np.tile(25.0 * (x - 0.5) ** 2, (nt + 1, 1))
    with pytest.raises(ValueError, match="would fabricate"):
        fp.solve_fp_system(m0, U)


def test_the_stop_names_the_interpolation_and_a_remedy():
    """Naming the defect without a next step gets the diagnostic read as noise.

    The remedy also states the counter-intuitive part: on this scheme *refining* can make
    it worse, because the departure point then spans more cells.
    """
    n, nt = 41, 20
    prob = _problem(n=n, nt=nt)
    fp = FPSLSolver(prob, interpolation_method="cubic")
    x = np.linspace(0.0, 1.0, n)
    m0 = np.exp(-60 * (x - 0.4) ** 2)
    m0 /= m0.sum() * (x[1] - x[0])
    U = np.tile(25.0 * (x - 0.5) ** 2, (nt + 1, 1))
    with pytest.raises(ValueError) as exc:
        fp.solve_fp_system(m0, U)
    message = str(exc.value)
    assert "FP-SL positivity clip" in message
    assert "linear interpolation" in message
    assert "coarsen" in message


def test_linear_pure_diffusion_is_not_stopped_by_the_mass_gate():
    """Linear splatting preserves positivity and Crank-Nicolson diffusion of a smooth Gaussian
    (cell-Peclet stable) does not undershoot, so the clip never injects mass -- the gate is a real
    signal, not noise.

    Asserted on the RETURN, not on a log line. The previous form asserted the absence of
    "positivity clip injected mass" on this solver's logger, and that assertion could not fail:
    the module has zero `warning` calls (its only log statement is an init-time `info`), and the
    string exists only in `weak_form_fp_solver.py`. This solver stops by raising -- which its two
    sibling tests assert -- so what "no clip" looks like here is that the call returns.
    """
    n, nt = 41, 20
    prob = _problem(n=n, nt=nt)
    fp = FPSLSolver(prob, interpolation_method="linear")
    x = np.linspace(0.0, 1.0, n)
    m0 = np.exp(-40 * (x - 0.5) ** 2)
    m0 /= m0.sum() * (x[1] - x[0])
    U = np.zeros((nt + 1, n))  # no drift -> pure diffusion

    result = fp.solve_fp_system(m0, potential_field=U)  # must not raise: the gate did not fire

    assert result.shape == (nt + 1, n)
    assert np.all(result >= 0.0), "a positivity clip would have been needed"
    dx = x[1] - x[0]
    # 1e-9, not a looser band: measured, this configuration conserves to 3.55e-15, so the
    # threshold sits six orders above the answer and still refuses anything the gate would have
    # stopped. What it does NOT catch is a scheme that drifts below 1e-9 -- for that the gate
    # itself is the instrument, and "did not raise" above is the assertion that pins it.
    assert abs(result[-1].sum() * dx - 1.0) < 1e-9, "mass moved without the gate noticing"
