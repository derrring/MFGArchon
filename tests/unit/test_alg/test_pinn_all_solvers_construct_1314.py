"""
Pinning test for GitHub issue #1314 (Refs #1290):

FPPINNSolver._initialize_networks called
    create_mfg_networks(architecture_type="standard", separate_networks=False, ...)
expecting a dict back, but create_mfg_networks takes network_type= (not
architecture_type=/separate_networks=) and returns a single nn.Module.  The
unexpected kwargs flowed through **kwargs into FeedForwardNetwork.__init__,
raising

    TypeError: FeedForwardNetwork.__init__() got an unexpected keyword argument
    'architecture_type'

#1290 fixed the same bug class in HJBPINNSolver and MFGPINNSolver but missed
FPPINNSolver.  This test constructs ALL THREE concrete PINN solvers on one tiny
1-D problem so the whole family is covered by a single gate — it would have
caught both #1290 (HJB, MFG) and #1314 (FP).

Fixed in PR that closes #1314:
  fp_pinn_solver.py: _initialize_networks uses network_type="feedforward",
  problem_type="fp", activation guarded to the allowed literal set, and builds
  the {"m_net": ...} dict explicitly.
"""

from __future__ import annotations

import pytest

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

pytestmark = [pytest.mark.optional_torch, pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")]


# ---------------------------------------------------------------------------
# Minimal MFGProblem fixture (shared shape with test_pinn_init_chain_1290.py)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Source-level guard: FP must not reintroduce the bad kwargs.
# ---------------------------------------------------------------------------


def test_fp_initialize_networks_uses_correct_api():
    """
    FPPINNSolver._initialize_networks must use network_type= (not
    architecture_type=/separate_networks=) and build the m_net dict manually.
    """
    import inspect

    from mfgarchon.alg.neural.pinn_solvers.fp_pinn_solver import FPPINNSolver

    src = inspect.getsource(FPPINNSolver._initialize_networks)
    assert "architecture_type=" not in src, (
        "FPPINNSolver._initialize_networks must not use architecture_type= kwarg "
        "(not a parameter of create_mfg_networks)"
    )
    assert "separate_networks=" not in src, (
        "FPPINNSolver._initialize_networks must not use separate_networks= kwarg "
        "(not a parameter of create_mfg_networks)"
    )
    assert "network_type=" in src, "FPPINNSolver._initialize_networks must use network_type= parameter"
    assert "m_net" in src, "FPPINNSolver._initialize_networks must build m_net (FP density network)"


# ---------------------------------------------------------------------------
# Behavioural pinning: all three solvers must construct without error.
# ---------------------------------------------------------------------------


class TestAllPINNSolversConstruct:
    """
    HJB, FP and MFG PINN solvers must each construct on a tiny 1-D problem.
    Pre-#1314, the FP case raised TypeError from the architecture_type= kwarg.
    """

    def test_hjb_pinn_solver_constructs(self):
        from mfgarchon.alg.neural.pinn_solvers.hjb_pinn_solver import HJBPINNSolver

        solver = HJBPINNSolver(_make_problem(), config=_tiny_config())
        assert "u_net" in solver.networks
        assert solver.u_net is solver.networks["u_net"]

    def test_fp_pinn_solver_constructs(self):
        """The #1314 case: FPPINNSolver(problem) must build without TypeError."""
        from mfgarchon.alg.neural.pinn_solvers.fp_pinn_solver import FPPINNSolver

        solver = FPPINNSolver(_make_problem(), config=_tiny_config())
        assert "m_net" in solver.networks, "FP solver must register m_net (not u_net)"
        assert solver.m_net is solver.networks["m_net"]
        # FP builds only the density network — no stray u_net key.
        assert "u_net" not in solver.networks
        # Optimizer must be wired to the m_net key created above.
        assert "m_net" in solver.optimizers

    def test_mfg_pinn_solver_constructs(self):
        from mfgarchon.alg.neural.pinn_solvers.mfg_pinn_solver import MFGPINNSolver

        solver = MFGPINNSolver(_make_problem(), config=_tiny_config())
        assert "u_net" in solver.networks
        assert "m_net" in solver.networks


class TestFPPINNForwardSmoke:
    """Construction succeeded — verify the density network actually computes."""

    def test_fp_pinn_forward_pass(self):
        """FPPINNSolver.forward(t, x) returns a finite, positive (N, 1) density."""
        from mfgarchon.alg.neural.pinn_solvers.fp_pinn_solver import FPPINNSolver

        solver = FPPINNSolver(_make_problem(), config=_tiny_config())

        n = 10
        t = torch.rand(n, 1)
        x = torch.rand(n, 1)
        m = solver.forward(t, x)

        assert m.shape == (n, 1), f"Expected (10, 1), got {tuple(m.shape)}"
        assert m.isfinite().all(), "FP forward output contains non-finite values"
        # forward applies exp(.) so the density is strictly positive.
        assert (m > 0).all(), "FP density must be positive"
