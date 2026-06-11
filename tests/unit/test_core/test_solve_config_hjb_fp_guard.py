"""solve(config=...) fails LOUD when a config.hjb / config.fp value can't be honored.

In Safe/Auto mode, ``MFGProblem.solve`` builds solvers via the config translator
(``hjb_config_to_kwargs`` / ``fp_config_to_kwargs``, Issue #1155, threaded by #1309).
Consistent config subtrees are now honored; what fails loud is a config value that
*conflicts* with the resolved scheme — e.g. ``config.hjb.method='gfdm'`` under a
``FDM_UPWIND`` scheme — which raises ``NotImplementedError`` rather than silently
ignoring the user's intent. Expert Mode (explicit hjb_solver/fp_solver) bypasses the
translator, and a picard-only or fully-default config solves normally.

These are the ``solve()``-level integration checks; the translator unit semantics
(threading + per-field conflict/unmapped raises) are covered by
``tests/unit/test_config/test_config_translator.py``.
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


def test_conflicting_fp_config_raises_in_safe_mode():
    """A config.fp.method conflicting with the resolved scheme fails loud through solve().

    Post-#1309 a *consistent* fp config (e.g. method='fdm' under FDM_UPWIND) is threaded and
    honored; only a genuine scheme/method conflict raises. FDM_UPWIND selects an FDM FP solver,
    so fp.method='fem' conflicts. The hjb subtree is default here, so the fp guard is what fires.
    """
    cfg = MFGSolverConfig(fp=FPConfig(method="fem"))
    with pytest.raises(NotImplementedError, match=r"config\.fp.*#1155"):
        _problem().solve(scheme=NumericalScheme.FDM_UPWIND, config=cfg)


def test_consistent_fp_config_is_honored():
    """A config.fp consistent with the scheme is threaded (no raise) — the #1309 behavior.

    fp.method='fdm' matches the FDM_UPWIND FP solver class, so it is honored, not rejected. This
    pins that the guard fires on conflict only, not on any non-default fp subtree (the old, now
    superseded, not-yet-threaded behavior).
    """
    cfg = MFGSolverConfig(fp=FPConfig(method="fdm"))
    _problem().solve(scheme=NumericalScheme.FDM_UPWIND, config=cfg, max_iterations=2)  # no raise


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
