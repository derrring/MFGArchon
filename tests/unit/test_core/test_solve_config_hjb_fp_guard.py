"""solve(config=...) fails LOUD when config.hjb / config.fp can't be honored.

In Safe/Auto mode, ``MFGProblem.solve`` builds solvers via ``create_paired_solvers``
and currently only applies ``config.picard``. The composite ``config.hjb`` /
``config.fp`` subtrees are not yet threaded into the factory (Issue #1155). Rather
than silently ignore a user-supplied hjb/fp config (wrong-config-ignored), solve()
raises ``NotImplementedError`` when a non-default hjb/fp config reaches the factory
paths. Expert Mode (explicit hjb_solver/fp_solver) bypasses the guard, and a
picard-only or fully-default config solves normally.
"""

from __future__ import annotations

import pytest

import numpy as np

from mfgarchon import MFGProblem
from mfgarchon.config import FPConfig, HJBConfig, MFGSolverConfig, PicardConfig
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.core.mfg_problem import MFGComponents
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import no_flux_bc
from mfgarchon.types import NumericalScheme


def _problem(n=11, nt=4):
    grid = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[n], boundary_conditions=no_flux_bc(dimension=1))
    H = SeparableHamiltonian(control_cost=QuadraticControlCost(control_cost=1.0))
    comp = MFGComponents(hamiltonian=H, m_initial=lambda x: np.ones_like(x), u_terminal=lambda x: 0.5 * (x - 0.5) ** 2)
    return MFGProblem(geometry=grid, components=comp, T=0.5, Nt=nt, sigma=0.3, coupling_coefficient=0.5)


def test_nondefault_hjb_config_raises_in_safe_mode():
    cfg = MFGSolverConfig(hjb=HJBConfig(method="gfdm"))
    with pytest.raises(NotImplementedError, match=r"config\.hjb.*#1155"):
        _problem().solve(scheme=NumericalScheme.FDM_UPWIND, config=cfg)


def test_nondefault_fp_config_raises_in_auto_mode():
    cfg = MFGSolverConfig(fp=FPConfig(method="fdm"))
    with pytest.raises(NotImplementedError, match=r"config\.fp.*#1155"):
        _problem().solve(config=cfg)  # auto mode (no scheme/solvers)


def test_both_nondefault_named_in_message():
    cfg = MFGSolverConfig(hjb=HJBConfig(method="gfdm"), fp=FPConfig(method="fdm"))
    with pytest.raises(NotImplementedError) as exc:
        _problem().solve(scheme=NumericalScheme.FDM_UPWIND, config=cfg)
    msg = str(exc.value)
    assert "config.hjb" in msg
    assert "config.fp" in msg


def test_picard_only_config_does_not_raise():
    """A non-default picard config (the honored subtree) solves normally."""
    cfg = MFGSolverConfig(picard=PicardConfig(max_iterations=2, tolerance=1e-3))
    _problem().solve(scheme=NumericalScheme.FDM_UPWIND, config=cfg)  # no NotImplementedError


def test_default_config_does_not_raise():
    _problem().solve(scheme=NumericalScheme.FDM_UPWIND, config=MFGSolverConfig(), max_iterations=2)


def test_expert_mode_bypasses_the_guard():
    """Expert Mode builds solvers directly, so config.hjb/.fp are legitimately ignored."""
    from mfgarchon.factory import create_paired_solvers

    prob = _problem()
    hjb, fp = create_paired_solvers(prob, NumericalScheme.FDM_UPWIND)
    cfg = MFGSolverConfig(hjb=HJBConfig(method="gfdm"))  # non-default but irrelevant in Expert Mode
    prob.solve(hjb_solver=hjb, fp_solver=fp, config=cfg, max_iterations=2)  # no NotImplementedError
