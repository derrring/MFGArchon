"""
Pinning tests for Issue #1284 config bugs.

The three that exercised `GeneralMFGFactory` are gone with it (#1920): its
`create_from_hamiltonian` built the geometry with a hardcoded `no_flux_bc` and discarded the
caller's BoundaryConditions, so the class was deleted rather than repaired. What they asserted --
that a config round-trips, that a periodic BC does not raise, that a missing solver does -- was
about the config layer reaching a factory that no longer exists.

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

# ---------------------------------------------------------------------------
# Bug #1 — round-trip create_template_config -> create_from_config_dict
# ---------------------------------------------------------------------------


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
