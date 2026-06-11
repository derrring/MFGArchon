"""
Pinning tests for GitHub issue #1290:

PINNBase.__init__ called super().__init__(problem) without the required
`config` argument, crashing every HJBPINNSolver / MFGPINNSolver construction
with TypeError.  Additionally PINNBase did not implement the abstract methods
`build_networks`, `compute_loss`, and `validate_solution` declared by
BaseNeuralSolver / BaseMFGSolver, making all concrete subclasses still
abstract and therefore non-instantiable.

Fixed in PR that closes #1290 (2026-06-11 survey):
  1. base_pinn.py: resolve config before super().__init__; pass both args.
  2. base_pinn.py: add concrete build_networks / compute_loss / validate_solution.
  3. hjb_pinn_solver.py / mfg_pinn_solver.py: fix _initialize_networks to use
     the correct create_mfg_networks API (network_type= not architecture_type=;
     returns nn.Module not dict).
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.optional_torch

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

pytestmark = [pytest.mark.optional_torch, pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")]


# ---------------------------------------------------------------------------
# Minimal MFGProblem fixture
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


def _tiny_pinn_config():
    """Return a PINNConfig with minimal architecture (fast to construct)."""
    from mfgarchon.alg.neural.pinn_solvers.base_pinn import PINNConfig

    return PINNConfig(hidden_layers=[8, 8], max_epochs=2, device="cpu")


# ---------------------------------------------------------------------------
# Bug-present evidence: document what the error was before the fix.
# These tests verify the fix at the source level (no solver construction needed).
# ---------------------------------------------------------------------------


class TestBugEvidence:
    """Source-level checks that the fixed code addresses the two root causes."""

    def test_pinnbase_passes_config_to_super(self):
        """
        After fix, PINNBase.__init__ resolves config and calls
        super().__init__(problem, _pinn_config).  Verify by inspecting
        the source — the call must include both positional arguments.
        """
        import inspect

        from mfgarchon.alg.neural.pinn_solvers.base_pinn import PINNBase

        src = inspect.getsource(PINNBase.__init__)
        # The fixed line calls super().__init__ with two args: problem + config variable
        assert "super().__init__(problem, _pinn_config)" in src, (
            "PINNBase.__init__ must call super().__init__(problem, _pinn_config); "
            "the bug was calling super().__init__(problem) without config"
        )

    def test_pinnbase_implements_build_networks(self):
        """PINNBase must have a concrete build_networks method."""
        import inspect

        from mfgarchon.alg.neural.pinn_solvers.base_pinn import PINNBase

        method = getattr(PINNBase, "build_networks", None)
        assert method is not None, "PINNBase.build_networks must exist"
        # Must not be abstract
        assert not getattr(method, "__isabstractmethod__", False), (
            "PINNBase.build_networks must be concrete, not abstract"
        )
        # Docstring should mention _initialize_networks
        src = inspect.getsource(method)
        assert "_initialize_networks" in src, "build_networks should delegate to _initialize_networks"

    def test_pinnbase_implements_compute_loss(self):
        """PINNBase must have a concrete compute_loss method."""
        from mfgarchon.alg.neural.pinn_solvers.base_pinn import PINNBase

        method = getattr(PINNBase, "compute_loss", None)
        assert method is not None, "PINNBase.compute_loss must exist"
        assert not getattr(method, "__isabstractmethod__", False), (
            "PINNBase.compute_loss must be concrete, not abstract"
        )

    def test_pinnbase_implements_validate_solution(self):
        """PINNBase must have a concrete validate_solution method."""
        import inspect

        from mfgarchon.alg.neural.pinn_solvers.base_pinn import PINNBase

        method = getattr(PINNBase, "validate_solution", None)
        assert method is not None, "PINNBase.validate_solution must exist"
        assert not getattr(method, "__isabstractmethod__", False), (
            "PINNBase.validate_solution must be concrete, not abstract"
        )
        src = inspect.getsource(method)
        assert "evaluate_convergence" in src, "validate_solution should delegate to evaluate_convergence"

    def test_hjb_initialize_networks_uses_correct_api(self):
        """
        HJBPINNSolver._initialize_networks must use network_type= (not
        architecture_type=) and build a dict manually (create_mfg_networks
        returns a single nn.Module).
        """
        import inspect

        from mfgarchon.alg.neural.pinn_solvers.hjb_pinn_solver import HJBPINNSolver

        src = inspect.getsource(HJBPINNSolver._initialize_networks)
        # Check that architecture_type= is NOT used as a keyword argument
        # (comments referencing the old name are OK, but the kwarg must not appear)
        assert "architecture_type=" not in src, (
            "HJBPINNSolver._initialize_networks must not use architecture_type= kwarg (not in create_mfg_networks)"
        )
        assert "separate_networks=" not in src, (
            "HJBPINNSolver._initialize_networks must not use separate_networks= kwarg (not in create_mfg_networks)"
        )
        assert "network_type=" in src, "HJBPINNSolver._initialize_networks must use network_type= parameter"

    def test_mfg_initialize_networks_uses_correct_api(self):
        """
        MFGPINNSolver._initialize_networks must use network_type= and
        create both u_net and m_net individually.
        """
        import inspect

        from mfgarchon.alg.neural.pinn_solvers.mfg_pinn_solver import MFGPINNSolver

        src = inspect.getsource(MFGPINNSolver._initialize_networks)
        # Check that architecture_type= is NOT used as a keyword argument
        assert "architecture_type=" not in src, (
            "MFGPINNSolver._initialize_networks must not use architecture_type= kwarg"
        )
        assert "separate_networks=" not in src, (
            "MFGPINNSolver._initialize_networks must not use separate_networks= kwarg"
        )
        assert "u_net" in src, "MFGPINNSolver._initialize_networks must build u_net"
        assert "m_net" in src, "MFGPINNSolver._initialize_networks must build m_net"


# ---------------------------------------------------------------------------
# Pinning tests: construction must succeed (previously raised TypeError)
# ---------------------------------------------------------------------------


class TestPINNConstruction:
    """
    HJBPINNSolver(problem) and MFGPINNSolver(problem) must construct without
    any TypeError.  These tests fail on the pre-fix code and pass after.
    """

    def test_hjb_pinn_solver_constructs_no_config(self):
        """
        HJBPINNSolver(problem) — no explicit config — must construct.
        Pre-fix: TypeError from super().__init__(problem) missing config arg
        + TypeError from abstract methods not implemented.
        """
        from mfgarchon.alg.neural.pinn_solvers.hjb_pinn_solver import HJBPINNSolver

        problem = _make_problem()
        # Use cpu device to avoid MPS/CUDA-related skips in CI
        from mfgarchon.alg.neural.pinn_solvers.base_pinn import PINNConfig

        solver = HJBPINNSolver(problem, config=PINNConfig(hidden_layers=[8, 8], device="cpu"))
        assert solver is not None
        assert solver.problem is problem

    def test_hjb_pinn_solver_config_wired_to_base(self):
        """
        After construction, solver.config must be a PINNConfig and must be the
        same object that BaseMFGSolver received (verified via identity).
        """
        from mfgarchon.alg.neural.pinn_solvers.base_pinn import PINNConfig
        from mfgarchon.alg.neural.pinn_solvers.hjb_pinn_solver import HJBPINNSolver

        problem = _make_problem()
        cfg = PINNConfig(hidden_layers=[8, 8], device="cpu")
        solver = HJBPINNSolver(problem, config=cfg)
        assert solver.config is cfg, "solver.config must be the PINNConfig passed by the caller"

    def test_hjb_pinn_solver_default_config_is_pinn_config(self):
        """When no config is given, solver.config must be a PINNConfig instance."""
        from mfgarchon.alg.neural.pinn_solvers.base_pinn import PINNConfig
        from mfgarchon.alg.neural.pinn_solvers.hjb_pinn_solver import HJBPINNSolver

        problem = _make_problem()
        solver = HJBPINNSolver(problem, config=PINNConfig(hidden_layers=[8, 8], device="cpu"))
        assert isinstance(solver.config, PINNConfig)

    def test_mfg_pinn_solver_constructs_no_config(self):
        """
        MFGPINNSolver(problem) — no explicit config — must construct.
        Pre-fix: same TypeError chain as HJBPINNSolver.
        """
        from mfgarchon.alg.neural.pinn_solvers.base_pinn import PINNConfig
        from mfgarchon.alg.neural.pinn_solvers.mfg_pinn_solver import MFGPINNSolver

        problem = _make_problem()
        solver = MFGPINNSolver(problem, config=PINNConfig(hidden_layers=[8, 8], device="cpu"))
        assert solver is not None
        assert solver.problem is problem

    def test_mfg_pinn_solver_config_wired_to_base(self):
        """solver.config must be the PINNConfig passed by the caller."""
        from mfgarchon.alg.neural.pinn_solvers.base_pinn import PINNConfig
        from mfgarchon.alg.neural.pinn_solvers.mfg_pinn_solver import MFGPINNSolver

        problem = _make_problem()
        cfg = PINNConfig(hidden_layers=[8, 8], device="cpu")
        solver = MFGPINNSolver(problem, config=cfg)
        assert solver.config is cfg

    def test_hjb_pinn_networks_initialised(self):
        """After construction, solver.networks must contain 'u_net'."""
        from mfgarchon.alg.neural.pinn_solvers.base_pinn import PINNConfig
        from mfgarchon.alg.neural.pinn_solvers.hjb_pinn_solver import HJBPINNSolver

        problem = _make_problem()
        solver = HJBPINNSolver(problem, config=PINNConfig(hidden_layers=[8, 8], device="cpu"))
        assert "u_net" in solver.networks
        assert solver.u_net is solver.networks["u_net"]

    def test_mfg_pinn_networks_initialised(self):
        """After construction, solver.networks must contain 'u_net' and 'm_net'."""
        from mfgarchon.alg.neural.pinn_solvers.base_pinn import PINNConfig
        from mfgarchon.alg.neural.pinn_solvers.mfg_pinn_solver import MFGPINNSolver

        problem = _make_problem()
        solver = MFGPINNSolver(problem, config=PINNConfig(hidden_layers=[8, 8], device="cpu"))
        assert "u_net" in solver.networks
        assert "m_net" in solver.networks
        assert solver.u_net is solver.networks["u_net"]
        assert solver.m_net is solver.networks["m_net"]


# ---------------------------------------------------------------------------
# Forward-pass smoke test: construction succeeded — check networks compute
# ---------------------------------------------------------------------------


class TestPINNForwardSmoke:
    """Quick forward pass after construction — ensures networks are wired."""

    def test_hjb_pinn_forward_pass(self):
        """HJBPINNSolver.forward(t, x) returns finite tensor of shape (N, 1)."""
        from mfgarchon.alg.neural.pinn_solvers.base_pinn import PINNConfig
        from mfgarchon.alg.neural.pinn_solvers.hjb_pinn_solver import HJBPINNSolver

        problem = _make_problem()
        solver = HJBPINNSolver(problem, config=PINNConfig(hidden_layers=[8, 8], device="cpu"))

        n = 10
        t = torch.rand(n, 1)
        x = torch.rand(n, 1)
        u = solver.forward(t, x)

        assert u.shape == (n, 1), f"Expected (10,1), got {u.shape}"
        assert u.isfinite().all(), "Forward output contains non-finite values"

    def test_mfg_pinn_forward_pass(self):
        """MFGPINNSolver.forward(t, x) returns dict with finite u and m tensors."""
        from mfgarchon.alg.neural.pinn_solvers.base_pinn import PINNConfig
        from mfgarchon.alg.neural.pinn_solvers.mfg_pinn_solver import MFGPINNSolver

        problem = _make_problem()
        solver = MFGPINNSolver(problem, config=PINNConfig(hidden_layers=[8, 8], device="cpu"))

        n = 10
        t = torch.rand(n, 1)
        x = torch.rand(n, 1)
        out = solver.forward(t, x)

        assert isinstance(out, dict), "MFGPINNSolver.forward should return a dict"
        assert "u" in out, "forward output must contain key 'u'"
        assert "m" in out, "forward output must contain key 'm'"
        assert out["u"].shape == (n, 1)
        assert out["m"].shape == (n, 1)
        assert out["u"].isfinite().all()
        assert out["m"].isfinite().all()

    def test_build_networks_callable(self):
        """build_networks() (BaseNeuralSolver interface) must be callable without error."""
        from mfgarchon.alg.neural.pinn_solvers.base_pinn import PINNConfig
        from mfgarchon.alg.neural.pinn_solvers.hjb_pinn_solver import HJBPINNSolver

        problem = _make_problem()
        solver = HJBPINNSolver(problem, config=PINNConfig(hidden_layers=[8, 8], device="cpu"))
        # Re-initialise networks — should not raise
        solver.build_networks()
        assert "u_net" in solver.networks
