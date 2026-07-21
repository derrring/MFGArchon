"""
Pinning test for GitHub issue #1303:

MFGPINNSolver.get_results() (mfg_pinn_solver.py) built its result dict with

    "training_strategy": self.config.training_strategy,

but PINNConfig has no `training_strategy` field, so get_results() raised

    AttributeError: 'PINNConfig' object has no attribute 'training_strategy'

once the training-history guard was satisfied (i.e. after solve()).  The
sibling HJBPINNSolver.get_results / FPPINNSolver.get_results end their
metadata at `solver_type` / `device` / `config` and carry no
`training_strategy` key; the fix drops the bogus entry to match that
established shape (no new config surface invented).

Pre-fix: this test's get_results() call raises AttributeError.
Post-fix: get_results() returns a dict with the documented keys.
"""

from __future__ import annotations

import pytest

try:
    import torch  # noqa: F401

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

pytestmark = [pytest.mark.optional_torch, pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")]


def _make_problem():
    """Return a minimal 1-D MFGProblem suitable for PINN construction tests."""
    import numpy as np

    from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
    from mfgarchon.core.mfg_problem import MFGComponents, MFGProblem
    from mfgarchon.geometry import Hyperrectangle

    geo = Hyperrectangle(bounds=[(0.0, 1.0)])
    H = SeparableHamiltonian(
        control_cost=QuadraticControlCost(control_cost=1.0),
        coupling=lambda m: np.zeros_like(m),
    )
    components = MFGComponents(
        hamiltonian=H,
        u_terminal=lambda x: np.zeros_like(x) if hasattr(x, "__len__") else 0.0,
        m_initial=lambda x: np.ones_like(x) if hasattr(x, "__len__") else 1.0,
    )
    return MFGProblem(T=1.0, geometry=geo, sigma=0.1, components=components)


def _tiny_config():
    from mfgarchon.alg.neural.pinn_solvers.base_pinn import PINNConfig

    return PINNConfig(hidden_layers=[8, 8], device="cpu")


def _make_solver_with_history():
    """Construct MFGPINNSolver and populate the minimal training history.

    get_results() guards on a non-empty training_history["total_loss"]
    (raising RuntimeError before solve()).  We seed one entry of each
    tracked loss so the guard passes and execution reaches the metadata
    block where the #1303 AttributeError lived.
    """
    from mfgarchon.alg.neural.pinn_solvers.mfg_pinn_solver import MFGPINNSolver

    solver = MFGPINNSolver(_make_problem(), config=_tiny_config())
    solver.training_history["total_loss"] = [1.0]
    solver.training_history["hjb_loss"] = [0.5]
    solver.training_history["fp_loss"] = [0.3]
    solver.training_history["coupling_loss"] = [0.2]
    # `converged` reads best_loss, not the history length (Issue #1684). Seeding only the history
    # left best_loss at inf, so this fixture described a solver that had trained and not converged
    # while asserting it had -- the assertion passed because the old flag was a tautology.
    solver.best_loss = 1.0
    return solver


def test_pinn_config_has_no_training_strategy_field():
    """Guards the root cause: PINNConfig must not silently grow the field.

    The fix is to drop the reference, not to add config surface.  If a
    future change adds a real `training_strategy` field this test should be
    revisited deliberately rather than the attribute reappearing by accident.
    """
    config = _tiny_config()
    assert not hasattr(config, "training_strategy"), (
        "PINNConfig unexpectedly has a training_strategy field; #1303 fix assumed it does not exist"
    )


def test_get_results_returns_dict_without_raising():
    """The #1303 case: get_results() must not raise AttributeError."""
    solver = _make_solver_with_history()

    results = solver.get_results()

    assert isinstance(results, dict)
    # No leftover bogus key from the AttributeError site.
    assert "training_strategy" not in results
    # Sane, documented keys are present.
    for key in (
        "u",
        "m",
        "x_grid",
        "t_grid",
        "training_history",
        "converged",
        "final_loss",
        "epochs_trained",
        "solver_type",
        "device",
        "config",
    ):
        assert key in results, f"get_results() missing expected key {key!r}"
    assert results["solver_type"] == "MFG_PINN"
    # best_loss=1.0 against the default tolerance 1e-6: trained, not converged. Asserting the
    # False case is the point -- the old expression could not produce it (Issue #1684).
    assert results["converged"] is False
    assert results["final_loss"] == 1.0
    assert results["epochs_trained"] == 1


def test_get_results_metadata_shape_matches_siblings():
    """MFG get_results metadata keys must be a superset of the HJB/FP shape.

    HJB and FP get_results both end metadata at solver_type/device/config;
    MFG adds the per-equation loss fields.  None of the three carries
    training_strategy.
    """
    solver = _make_solver_with_history()
    results = solver.get_results()

    sibling_metadata = {"solver_type", "device", "config"}
    assert sibling_metadata <= set(results), "MFG get_results dropped a metadata key present in HJB/FP siblings"
