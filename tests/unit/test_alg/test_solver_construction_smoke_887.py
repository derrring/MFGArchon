"""
Construction smoke tests for the optimisation / DGM solver families
(Phase B of issue #887; demote pinned by issue #1342).

Background
----------
#887 lists several "concrete" solver classes that had zero dedicated test
coverage.  The PINN family (HJB/FP/MFG PINN) is already covered by
``test_pinn_all_solvers_construct_1314.py`` and ``test_pinn_init_chain_1290.py``.
This file adds the same kind of cheap construction gate for the *remaining*
untested solver families:

  * ``SinkhornMFGSolver``      (optimal transport)
  * ``WassersteinMFGSolver``   (optimal transport)
  * ``VariationalMFGSolver``   (direct variational)
  * ``PrimalDualMFGSolver``    (variational primal-dual)
  * ``MFGDGMSolver``           (Deep Galerkin Method; the concrete
                                ``BaseDGMSolver`` subclass)

DEMOTE TO EXPERIMENTAL (#1342)
------------------------------
The "Phase 1: Algorithm Reorganization Foundation" refactor (commit
``c8f12ad9``, 2025-09-29) added ``@abstractmethod`` hooks on the shared base
classes:

  * ``BaseMFGSolver.validate_solution``
  * ``BaseOptimizationSolver.compute_objective`` / ``compute_gradient``
  * ``BaseNeuralSolver.build_networks`` / ``compute_loss`` / ``train_step``

The optimisation and DGM subclasses were never updated to implement them, so
for ~9 months ``SolverCls(problem)`` raised the cryptic::

    TypeError: Can't instantiate abstract class <Solver> without an
    implementation for abstract methods ...

These five solvers have full ``solve()`` bodies but the hooks are unimplemented
and there is no reference to validate them, so the v1.0 decision (#1342) is to
**demote them to clearly-experimental**, not to complete them.  Each now:

  * carries a "**Experimental — not production-ready (Issue #1342).**" docstring;
  * overrides the missing abstract hooks with stubs raising a clear
    ``NotImplementedError`` (so the class is concrete, not abstract); and
  * guards ``__init__`` so construction fails LOUD AND CLEAR with the same
    actionable "experimental ... #1342" message instead of the bare TypeError.

The tests below pin that clean demote: they assert construction raises a
``NotImplementedError`` mentioning "experimental" and "#1342".  They flip to a
hard failure the moment a solver is re-broken (bare TypeError) OR completed
(constructs successfully) — either of which should be a deliberate, reviewed
change that updates this pin.

None of the five are wired into the production factory / ``NumericalScheme``
dispatch; they are import-only experimental classes.

Refs #887, #1342.
"""

from __future__ import annotations

import pytest

import numpy as np

try:
    import torch  # noqa: F401

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Shared assertion
# ---------------------------------------------------------------------------


def _assert_experimental_1342(excinfo: pytest.ExceptionInfo) -> None:
    """The demote must surface a clear, actionable #1342 message.

    Guards against regressing to the bare ``Can't instantiate abstract class``
    TypeError (caught by ``raises=NotImplementedError``) and against a vague
    message that omits the issue reference or the experimental status.
    """
    message = str(excinfo.value)
    assert "#1342" in message, f"message must reference issue #1342, got: {message!r}"
    assert "experimental" in message.lower(), f"message must say 'experimental', got: {message!r}"


# ---------------------------------------------------------------------------
# Tiny problem fixtures
# ---------------------------------------------------------------------------


def _make_mfg_problem():
    """Minimal 1-D ``MFGProblem`` (shape shared with the PINN construct tests)."""
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
    return MFGProblem(T=0.1, geometry=geo, sigma=0.1, components=components)


def _make_variational_problem():
    """Minimal 1-D ``VariationalMFGProblem`` for the variational solvers."""
    from mfgarchon.alg.optimization.variational_problem import VariationalMFGProblem

    return VariationalMFGProblem(xmin=0.0, xmax=1.0, Nx=8, T=0.1, Nt=4, sigma=0.1)


# ---------------------------------------------------------------------------
# Optimal-transport solvers (POT-backed)
# ---------------------------------------------------------------------------


def test_sinkhorn_mfg_solver_experimental():
    """Sinkhorn OT solver is demoted: construction raises a clear #1342 error."""
    from mfgarchon.alg.optimization.optimal_transport.sinkhorn_solver import (
        SinkhornMFGSolver,
        SinkhornSolverConfig,
    )

    config = SinkhornSolverConfig(num_time_steps=4, num_spatial_points=8)
    with pytest.raises(NotImplementedError) as excinfo:
        SinkhornMFGSolver(_make_mfg_problem(), config=config)
    _assert_experimental_1342(excinfo)


def test_wasserstein_mfg_solver_experimental():
    """Wasserstein OT solver is demoted: construction raises a clear #1342 error."""
    from mfgarchon.alg.optimization.optimal_transport.wasserstein_solver import (
        WassersteinMFGSolver,
        WassersteinSolverConfig,
    )

    config = WassersteinSolverConfig(num_time_steps=4, num_spatial_points=8)
    with pytest.raises(NotImplementedError) as excinfo:
        WassersteinMFGSolver(_make_mfg_problem(), config=config)
    _assert_experimental_1342(excinfo)


# ---------------------------------------------------------------------------
# Variational solvers
# ---------------------------------------------------------------------------


def test_variational_mfg_solver_experimental():
    """Direct variational solver is demoted: construction raises a clear #1342 error."""
    from mfgarchon.alg.optimization.variational_solvers import VariationalMFGSolver

    with pytest.raises(NotImplementedError) as excinfo:
        VariationalMFGSolver(_make_variational_problem())
    _assert_experimental_1342(excinfo)


def test_primal_dual_mfg_solver_experimental():
    """Primal-dual variational solver is demoted: construction raises a clear #1342 error."""
    from mfgarchon.alg.optimization.variational_solvers import PrimalDualMFGSolver

    with pytest.raises(NotImplementedError) as excinfo:
        PrimalDualMFGSolver(_make_variational_problem())
    _assert_experimental_1342(excinfo)


# ---------------------------------------------------------------------------
# Deep Galerkin Method (concrete BaseDGMSolver subclass)
# ---------------------------------------------------------------------------


@pytest.mark.optional_torch
@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
def test_mfg_dgm_solver_experimental():
    """Deep Galerkin Method MFG solver is demoted: construction raises a clear #1342 error."""
    from mfgarchon.alg.neural.dgm.base_dgm import DGMConfig
    from mfgarchon.alg.neural.dgm.mfg_dgm_solver import MFGDGMSolver

    config = DGMConfig(hidden_layers=[8, 8])
    with pytest.raises(NotImplementedError) as excinfo:
        MFGDGMSolver(_make_mfg_problem(), config=config)
    _assert_experimental_1342(excinfo)
