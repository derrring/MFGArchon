"""
Construction + basic-solve smoke tests for the optimisation / DGM solver
families (Phase B of issue #887).

Background
----------
#887 lists several "concrete" solver classes that had zero dedicated test
coverage.  The PINN family (HJB/FP/MFG PINN) is already covered by
``test_pinn_all_solvers_construct_1314.py`` and ``test_pinn_init_chain_1290.py``
— those construct tests previously surfaced the #1290 / #1314 ``__init__``
crashes.  This file adds the same kind of cheap construction gate for the
*remaining* untested solver families:

  * ``SinkhornMFGSolver``      (optimal transport)
  * ``WassersteinMFGSolver``   (optimal transport)
  * ``VariationalMFGSolver``   (direct variational)
  * ``PrimalDualMFGSolver``    (variational primal-dual)
  * ``MFGDGMSolver``           (Deep Galerkin Method; the concrete
                                ``BaseDGMSolver`` subclass)

NEWLY-SURFACED BUG (xfail below)
--------------------------------
Every one of these classes is currently **un-instantiable**.  The
"Phase 1: Algorithm Reorganization Foundation" refactor (commit ``c8f12ad9``,
2025-09-29) added ``@abstractmethod`` hooks on the shared base classes:

  * ``BaseMFGSolver.validate_solution``
  * ``BaseOptimizationSolver.compute_objective`` / ``compute_gradient``
  * ``BaseNeuralSolver.build_networks`` / ``compute_loss`` / ``train_step``

The optimisation and DGM subclasses were never updated to implement them, so
``SolverCls(problem)`` raises::

    TypeError: Can't instantiate abstract class <Solver> without an
    implementation for abstract methods 'compute_gradient',
    'compute_objective', 'validate_solution'

(`MFGDGMSolver` is missing ``build_networks``/``compute_loss``/``train_step``/
``validate_solution`` instead.)  These solvers have full ``solve()`` bodies,
configs and result dataclasses, so they are clearly *intended* to be concrete
— the abstract-method gap is a latent bug, not a deliberate design.

The construction tests below are therefore marked ``xfail(strict=True,
raises=TypeError)``: they pass (as XFAIL) today, pin the exact current failure,
and will flip to a hard failure the moment a solver is made constructible —
forcing the stale ``xfail`` to be removed and a real solve smoke test to be
added.

A secondary defect is masked by the abstract-class TypeError and will only
surface once the abstract methods are implemented: ``BaseVariationalSolver``
reads ``problem.geometry.get_grid_shape()`` / ``get_grid_spacing()``, but
``VariationalMFGProblem`` exposes ``xmin/xmax/Nx/x/dx`` and has no ``geometry``
attribute.  The variational construction tests pass the intended
``VariationalMFGProblem`` so the gate keeps failing until *both* defects are
fixed.

Refs #887 (Phase B; full numerical-correctness coverage remains).
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

_ABSTRACT_OPT_REASON = (
    "NEWLY-SURFACED BUG (#887): {cls} is abstract — BaseMFGSolver."
    "validate_solution and BaseOptimizationSolver.compute_objective/"
    "compute_gradient (added by the 2025-09-29 c8f12ad9 reorg) are never "
    "implemented, so construction raises TypeError. Remove this xfail once the "
    "abstract methods are implemented and add a real solve smoke test."
)


@pytest.mark.xfail(
    raises=TypeError,
    strict=True,
    reason=_ABSTRACT_OPT_REASON.format(cls="SinkhornMFGSolver"),
)
def test_sinkhorn_mfg_solver_constructs():
    """Tiny-grid construction of the Sinkhorn optimal-transport solver."""
    from mfgarchon.alg.optimization.optimal_transport.sinkhorn_solver import (
        SinkhornMFGSolver,
        SinkhornSolverConfig,
    )

    config = SinkhornSolverConfig(num_time_steps=4, num_spatial_points=8)
    SinkhornMFGSolver(_make_mfg_problem(), config=config)


@pytest.mark.xfail(
    raises=TypeError,
    strict=True,
    reason=_ABSTRACT_OPT_REASON.format(cls="WassersteinMFGSolver"),
)
def test_wasserstein_mfg_solver_constructs():
    """Tiny-grid construction of the Wasserstein optimal-transport solver."""
    from mfgarchon.alg.optimization.optimal_transport.wasserstein_solver import (
        WassersteinMFGSolver,
        WassersteinSolverConfig,
    )

    config = WassersteinSolverConfig(num_time_steps=4, num_spatial_points=8)
    WassersteinMFGSolver(_make_mfg_problem(), config=config)


# ---------------------------------------------------------------------------
# Variational solvers
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    raises=TypeError,
    strict=True,
    reason=_ABSTRACT_OPT_REASON.format(cls="VariationalMFGSolver"),
)
def test_variational_mfg_solver_constructs():
    """Tiny construction of the direct variational solver."""
    from mfgarchon.alg.optimization.variational_solvers import VariationalMFGSolver

    VariationalMFGSolver(_make_variational_problem())


@pytest.mark.xfail(
    raises=TypeError,
    strict=True,
    reason=_ABSTRACT_OPT_REASON.format(cls="PrimalDualMFGSolver"),
)
def test_primal_dual_mfg_solver_constructs():
    """Tiny construction of the primal-dual variational solver."""
    from mfgarchon.alg.optimization.variational_solvers import PrimalDualMFGSolver

    PrimalDualMFGSolver(_make_variational_problem())


# ---------------------------------------------------------------------------
# Deep Galerkin Method (concrete BaseDGMSolver subclass)
# ---------------------------------------------------------------------------


@pytest.mark.optional_torch
@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
@pytest.mark.xfail(
    raises=TypeError,
    strict=True,
    reason=(
        "NEWLY-SURFACED BUG (#887): MFGDGMSolver (the concrete BaseDGMSolver "
        "subclass) is abstract — BaseNeuralSolver.build_networks/compute_loss/"
        "train_step and BaseMFGSolver.validate_solution (added by the "
        "2025-09-29 c8f12ad9 reorg) are never implemented, so construction "
        "raises TypeError. Remove this xfail once the abstract methods are "
        "implemented and add a real solve smoke test."
    ),
)
def test_mfg_dgm_solver_constructs():
    """Tiny construction of the Deep Galerkin Method MFG solver."""
    from mfgarchon.alg.neural.dgm.base_dgm import DGMConfig
    from mfgarchon.alg.neural.dgm.mfg_dgm_solver import MFGDGMSolver

    config = DGMConfig(hidden_layers=[8, 8])
    MFGDGMSolver(_make_mfg_problem(), config=config)
