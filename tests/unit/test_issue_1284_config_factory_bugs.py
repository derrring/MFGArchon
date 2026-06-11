"""
Pinning tests for Issue #1284 config/factory bugs.

Four bugs fixed (2026-06-11 survey):
  1. general_mfg_factory.py: create_template_config round-trip raises TypeError
     (BoundaryConditions has no 'type' ctor param; it's a @property).
  2. parameter_sweep.py: create_random_sweep produces n_samples^k Cartesian
     combos instead of n_samples paired tuples.
  3. solver_factory.py: _update_config_with_kwargs raises AttributeError when
     hjb.method=='fdm' and 'delta' kwarg is present (gfdm sub-config is None).
  4. general_mfg_factory.py: missing 'solver' section silently injects sigma=1.0
     instead of raising.

Each test fails on the unfixed code and passes after the fix.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Bug #1 — round-trip create_template_config -> create_from_config_dict
# ---------------------------------------------------------------------------


def test_template_config_round_trips_without_error():
    """create_template_config output must round-trip through create_from_config_dict."""
    from mfgarchon.factory.general_mfg_factory import GeneralMFGFactory

    factory = GeneralMFGFactory()
    with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        factory.create_template_config(tmp_path)
        # Must not raise TypeError: __init__() got an unexpected keyword argument 'type'
        problem = factory.create_from_config_file(tmp_path)
        assert problem is not None
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def test_create_from_config_dict_periodic_bc_no_error():
    """create_from_config_dict with boundary_conditions:{type:periodic} must not raise TypeError."""

    from mfgarchon.factory.general_mfg_factory import GeneralMFGFactory
    from mfgarchon.geometry import BoundaryConditions

    factory = GeneralMFGFactory()
    config = {
        "hamiltonian": {"type": "separable", "control_cost": 1.0, "coupling_coefficient": 1.0},
        "domain": {"xmin": 0.0, "xmax": 1.0, "Nx": 11},
        "time": {"T": 0.1, "Nt": 5},
        "solver": {"sigma": 0.5, "coupling_coefficient": 1.0},
        "boundary_conditions": {"type": "periodic"},
        "functions": {
            "m_initial": "lambda x: np.exp(-10 * (x - 0.5)**2)",
            "u_terminal": "lambda x: x**2",
        },
    }
    # Must not raise TypeError: BoundaryConditions.__init__() got unexpected keyword argument 'type'
    problem = factory.create_from_config_dict(config)
    assert problem is not None
    # Verify the BC was constructed properly (not a raw dict)
    bc = problem.components.boundary_conditions
    assert isinstance(bc, BoundaryConditions)


# ---------------------------------------------------------------------------
# Bug #2 — create_random_sweep cardinality
# ---------------------------------------------------------------------------


def test_create_random_sweep_yields_n_samples_not_product():
    """create_random_sweep(params, n_samples=5) with k=3 params -> 5 combos, not 125."""
    from mfgarchon.workflow.parameter_sweep import create_random_sweep

    params = {
        "sigma": (0.1, 1.0),
        "alpha": (0.5, 2.0),
        "beta": (0.01, 0.5),
    }
    sweep = create_random_sweep(params, n_samples=5)
    assert sweep.total_combinations == 5, (
        f"Expected 5 combinations, got {sweep.total_combinations}. "
        "Likely Cartesian product was used instead of paired tuples."
    )
    assert len(sweep.parameter_combinations) == 5


def test_create_random_sweep_each_combo_has_all_params():
    """Each sampled tuple must contain one value per parameter."""
    from mfgarchon.workflow.parameter_sweep import create_random_sweep

    params = {"sigma": (0.1, 1.0), "alpha": (0.5, 2.0)}
    sweep = create_random_sweep(params, n_samples=8)
    for combo in sweep.parameter_combinations:
        assert set(combo.keys()) == {"sigma", "alpha"}, f"Combo missing keys: {combo}"
        assert 0.1 <= combo["sigma"] <= 1.0
        assert 0.5 <= combo["alpha"] <= 2.0


# ---------------------------------------------------------------------------
# Bug #3 — _update_config_with_kwargs AttributeError when gfdm is None
# ---------------------------------------------------------------------------


def test_update_config_with_delta_kwarg_fdm_method_no_error():
    """SolverFactory._update_config_with_kwargs must not raise when hjb.method='fdm'."""
    from mfgarchon.config import MFGSolverConfig
    from mfgarchon.factory.solver_factory import SolverFactory

    config = MFGSolverConfig()
    # Default method is 'fdm'; gfdm sub-config is None
    assert config.hjb.method == "fdm"
    assert config.hjb.gfdm is None

    # Must not raise AttributeError: 'NoneType' object has no attribute 'delta'
    updated = SolverFactory._update_config_with_kwargs(config, delta=0.5)
    assert updated is not None  # config was returned (gfdm kwarg silently ignored for fdm method)


# ---------------------------------------------------------------------------
# Bug #4 — missing 'solver' section must raise, not silently default sigma=1.0
# ---------------------------------------------------------------------------


def test_create_from_config_dict_missing_solver_raises():
    """create_from_config_dict without 'solver' section must raise, not inject sigma=1.0."""
    from mfgarchon.factory.general_mfg_factory import GeneralMFGFactory

    factory = GeneralMFGFactory()
    config = {
        "hamiltonian": {"type": "separable", "control_cost": 1.0, "coupling_coefficient": 1.0},
        "domain": {"xmin": 0.0, "xmax": 1.0, "Nx": 11},
        "time": {"T": 0.1, "Nt": 5},
        # 'solver' section deliberately absent
    }
    with pytest.raises((KeyError, ValueError)):
        factory.create_from_config_dict(config)
