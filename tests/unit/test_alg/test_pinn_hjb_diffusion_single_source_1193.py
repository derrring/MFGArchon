"""
Pinning test for GitHub issue #1193 (Refs #1189):

hjb_pinn_solver.compute_pde_residual computed the HJB viscous term with an
inline literal::

    viscous_term = (self.sigma**2 / 2) * u_xx

instead of routing the sigma -> diffusion conversion through the single-source
converter ``diffusion_from_volatility_torch`` (D = sigma**2/2) that the sibling
PINN solvers already use (fp_pinn_solver.py, mfg_pinn_solver.py). PR #1298
migrated FP + MFG but missed HJB-PINN (it was added by #1288 the same day).

For a scalar sigma the converter IS ``sigma**2 / 2`` (the constant 0.5 is
exactly representable in IEEE-754, so ``0.5 * x`` and ``x / 2`` coincide at the
bit level), so this is a pure single-source refactor with zero behaviour change.
The discriminating pin is therefore *source-level*: the inline literal must be
gone and the converter call must be present. A solver-level numeric guard
documents the byte-equivalence (D(sigma) recovered from the residual equals the
converter output).

Fixed in PR that closes #1193:
  hjb_pinn_solver.py imports diffusion_from_volatility_torch and the viscous
  term is ``diffusion_from_volatility_torch(self.sigma) * u_xx``.
"""

from __future__ import annotations

import inspect
import re

import pytest

try:
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

pytestmark = [pytest.mark.optional_torch, pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")]


# ---------------------------------------------------------------------------
# Source-level pin: the discriminator (FAILS on pre-fix source, PASSES after).
# ---------------------------------------------------------------------------


def test_hjb_viscous_term_uses_single_source_converter():
    """compute_pde_residual must call diffusion_from_volatility_torch, not the
    inline ``sigma**2 / 2`` literal."""
    from mfgarchon.alg.neural.pinn_solvers.hjb_pinn_solver import HJBPINNSolver

    src = inspect.getsource(HJBPINNSolver.compute_pde_residual)
    assert "diffusion_from_volatility_torch(self.sigma)" in src, (
        "HJB viscous term must route sigma->D through diffusion_from_volatility_torch "
        "(single source, Issue #1193), mirroring fp_pinn_solver / mfg_pinn_solver."
    )


def test_no_inline_sigma_squared_over_two_literal_in_file():
    """No ``sigma**2 / 2`` style literal may remain anywhere in the file: the
    conversion is owned by the single-source converter."""
    import mfgarchon.alg.neural.pinn_solvers.hjb_pinn_solver as mod

    file_src = inspect.getsource(mod)
    # Match sigma ** 2 / 2 with arbitrary internal whitespace, optional self.,
    # e.g. "(self.sigma**2 / 2)", "sigma ** 2 / 2".
    pattern = re.compile(r"(?:self\.)?sigma\s*\*\*\s*2\s*/\s*2")
    hits = pattern.findall(file_src)
    assert not hits, (
        f"Inline sigma**2/2 literal(s) found in hjb_pinn_solver.py: {hits}. "
        "Use diffusion_from_volatility_torch(self.sigma) instead (Issue #1193)."
    )


def test_converter_is_imported():
    """The module imports the single-source converter (mirrors the siblings)."""
    import mfgarchon.alg.neural.pinn_solvers.hjb_pinn_solver as mod

    assert hasattr(mod, "diffusion_from_volatility_torch"), (
        "hjb_pinn_solver must import diffusion_from_volatility_torch from "
        "mfgarchon.utils.pde_coefficients (mirror fp_pinn_solver / mfg_pinn_solver)."
    )


# ---------------------------------------------------------------------------
# Solver-level numeric guard: the viscous coefficient equals the converter
# output (documents the byte-equivalence; tied to the live solver method).
# ---------------------------------------------------------------------------


def _make_problem():
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


def test_residual_viscous_coefficient_equals_converter():
    """Recover D(sigma) from the solver's HJB residual and assert it equals
    diffusion_from_volatility_torch(sigma) exactly.

    With u = x^2 (u_xx = 2) and a sigma-independent Hamiltonian / density,
    res(sigma) - res(0) = -D(sigma) * u_xx, hence
    D(sigma) = (res(0) - res(sigma)) / 2.
    """
    from mfgarchon.alg.neural.pinn_solvers.hjb_pinn_solver import HJBPINNSolver
    from mfgarchon.utils.pde_coefficients import diffusion_from_volatility_torch

    # Defined inside the (torch-guarded) test, not at module level: a module-level
    # `nn.Module` subclass would execute at pytest COLLECTION and raise NameError on a
    # CI runner without the optional torch dependency, erroring the whole suite.
    class _ParabolaNet(nn.Module):
        """u(t, x) = x^2 so that u_t = 0, u_x = 2x, u_xx = 2 (constant)."""

        def forward(self, inp: torch.Tensor) -> torch.Tensor:
            return inp[:, 1:2] ** 2

    solver = HJBPINNSolver(_make_problem(), config=_tiny_config())
    # Deterministic, sigma-independent u and density so only the viscous term
    # carries sigma dependence.
    solver.u_net = _ParabolaNet()
    solver.networks["u_net"] = solver.u_net
    solver.get_density = lambda t, x: torch.ones_like(x)  # type: ignore[method-assign]

    n = 16
    t = torch.rand(n, 1, dtype=solver.dtype)
    x = torch.rand(n, 1, dtype=solver.dtype) * 0.8 + 0.1  # interior

    u_xx = 2.0  # exact for u = x^2

    for sigma in (0.1, 0.5, 1.3):
        solver.sigma = 0.0
        res0 = solver.compute_pde_residual(t, x)["hjb"].detach()
        solver.sigma = sigma
        res_s = solver.compute_pde_residual(t, x)["hjb"].detach()

        d_implied = ((res0 - res_s) / u_xx).mean().item()
        d_expected = float(diffusion_from_volatility_torch(sigma))
        assert d_implied == pytest.approx(d_expected, rel=1e-6, abs=1e-9), (
            f"sigma={sigma}: viscous coefficient {d_implied} != converter {d_expected}"
        )
