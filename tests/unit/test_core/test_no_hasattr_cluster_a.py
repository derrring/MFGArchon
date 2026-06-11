"""Pinning tests for Issue #1068 Cluster A: no hasattr() in core/mfg_components.py
and core/hamiltonian.py.

Two kinds of tests:
1. Structural: verify the targeted function bodies contain no 'hasattr(' calls.
2. Behavioral: verify capability dispatch still routes correctly after the refactor.
"""

from __future__ import annotations

import ast
import inspect
import textwrap

import numpy as np

# ---------------------------------------------------------------------------
# Helper: extract source of a function and check for hasattr()
# ---------------------------------------------------------------------------


def _assert_no_hasattr(func, *, label: str) -> None:
    """Assert that `func`'s source contains no bare hasattr() call."""
    src = textwrap.dedent(inspect.getsource(func))
    tree = ast.parse(src)
    violations = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "hasattr":
            violations.append(ast.unparse(node))
    assert not violations, (
        f"{label} still contains hasattr() calls: {violations}\n"
        "Refs #1068 — replace with isinstance/Protocol or explicit None-init."
    )


# ---------------------------------------------------------------------------
# Structural tests
# ---------------------------------------------------------------------------


class TestNoHashattrStructural:
    """Verify that the targeted functions contain no hasattr() calls."""

    def test_mfg_components_post_init_no_hasattr(self):
        """MFGComponents.__post_init__ must not use hasattr() (lines 153, 161)."""
        from mfgarchon.core.mfg_components import MFGComponents

        _assert_no_hasattr(MFGComponents.__post_init__, label="MFGComponents.__post_init__")

    def test_mfg_components_get_hjb_jacobian_no_hasattr(self):
        """HamiltonianMixin.get_hjb_hamiltonian_jacobian_contrib must not use hasattr() (line 640)."""
        from mfgarchon.core.mfg_components import HamiltonianMixin

        _assert_no_hasattr(
            HamiltonianMixin.get_hjb_hamiltonian_jacobian_contrib,
            label="HamiltonianMixin.get_hjb_hamiltonian_jacobian_contrib",
        )

    def test_mfg_components_using_resolved_bc_no_hasattr(self):
        """ConditionsMixin.using_resolved_bc must not use hasattr() (line 972)."""
        from mfgarchon.core.mfg_components import ConditionsMixin

        _assert_no_hasattr(
            ConditionsMixin.using_resolved_bc,
            label="ConditionsMixin.using_resolved_bc",
        )

    def test_hamiltonian_regularize_no_hasattr(self):
        """ControlCostBase.regularize must not use hasattr() (line 298)."""
        from mfgarchon.core.hamiltonian import ControlCostBase

        _assert_no_hasattr(ControlCostBase.regularize, label="ControlCostBase.regularize")

    def test_hamiltonian_jacobian_fd_no_hasattr(self):
        """HamiltonianBase.jacobian_fd must not use hasattr() (line 1110)."""
        from mfgarchon.core.hamiltonian import HamiltonianBase

        _assert_no_hasattr(HamiltonianBase.jacobian_fd, label="HamiltonianBase.jacobian_fd")


# ---------------------------------------------------------------------------
# Behavioral tests — capability dispatch must still work identically
# ---------------------------------------------------------------------------


class TestCapabilityDispatchBehavior:
    """Verify that the refactored dispatch routes the same as the old hasattr dispatch."""

    def test_separable_lagrangian_yields_separable_hamiltonian(self):
        """MFGComponents with SeparableLagrangian → H via as_hamiltonian() (analytic path)."""
        from mfgarchon.core.hamiltonian import (
            QuadraticControlCost,
            SeparableHamiltonian,
            SeparableLagrangian,
        )
        from mfgarchon.core.mfg_components import MFGComponents

        L = SeparableLagrangian(
            control_cost=QuadraticControlCost(lambda_=2.0),
            coupling=lambda m: -(m**2),
        )
        comp = MFGComponents(
            lagrangian=L,
            m_initial=lambda x: np.ones_like(x) / 10.0,
            u_terminal=lambda x: np.zeros_like(x),
        )
        # Must have derived a SeparableHamiltonian (not a DualHamiltonian)
        assert isinstance(comp.hamiltonian, SeparableHamiltonian), (
            f"Expected SeparableHamiltonian, got {type(comp.hamiltonian).__name__}. "
            "The analytic as_hamiltonian() path was not taken."
        )

    def test_general_lagrangian_uses_legendre_transform(self):
        """MFGComponents with generic LagrangianBase → H via legendre_transform()."""
        from mfgarchon.core.hamiltonian import DualHamiltonian, LagrangianBase

        class _SimpleLagrangian(LagrangianBase):
            def __call__(self, x, alpha, m, t=0.0):
                return 0.5 * np.sum(alpha**2)

            def optimal_control(self, x, m, p, t=0.0):
                return np.asarray(p)

        L = _SimpleLagrangian()
        from mfgarchon.core.mfg_components import MFGComponents

        comp = MFGComponents(
            lagrangian=L,
            m_initial=lambda x: np.ones_like(x) / 10.0,
            u_terminal=lambda x: np.zeros_like(x),
        )
        # General Lagrangian without as_hamiltonian() → DualHamiltonian
        assert isinstance(comp.hamiltonian, DualHamiltonian), (
            f"Expected DualHamiltonian, got {type(comp.hamiltonian).__name__}."
        )

    def test_separable_hamiltonian_derives_lagrangian(self):
        """MFGComponents with SeparableHamiltonian → _lagrangian_class set."""
        from mfgarchon.core.hamiltonian import (
            QuadraticControlCost,
            SeparableHamiltonian,
            SeparableLagrangian,
        )
        from mfgarchon.core.mfg_components import MFGComponents

        H = SeparableHamiltonian(
            control_cost=QuadraticControlCost(lambda_=1.0),
            coupling=lambda m: -m,
        )
        comp = MFGComponents(
            hamiltonian=H,
            m_initial=lambda x: np.ones_like(x) / 10.0,
            u_terminal=lambda x: np.zeros_like(x),
        )
        # Must have derived a SeparableLagrangian from the separable hamiltonian
        assert isinstance(comp._lagrangian_class, SeparableLagrangian), (
            f"Expected _lagrangian_class to be SeparableLagrangian, got {type(comp._lagrangian_class).__name__}."
        )

    def test_non_separable_hamiltonian_no_lagrangian_derived(self):
        """MFGComponents with non-SeparableHamiltonian → _lagrangian_class stays None."""
        from mfgarchon.core.hamiltonian import (
            HamiltonianBase,
        )
        from mfgarchon.core.mfg_components import MFGComponents

        class _MinimalH(HamiltonianBase):
            """Minimal concrete Hamiltonian without control_cost/_potential."""

            def __call__(self, x, m, derivs_or_p, t=0.0):
                return np.zeros(1)

            def dp(self, x, m, derivs_or_p, t=0.0):
                return np.zeros_like(np.atleast_1d(derivs_or_p))

            def dm(self, x, m, derivs_or_p, t=0.0):
                return 0.0

            def optimal_control(self, x, m, derivs_or_p, t=0.0):
                return np.zeros_like(np.atleast_1d(derivs_or_p))

        H = _MinimalH()
        comp = MFGComponents(
            hamiltonian=H,
            m_initial=lambda x: np.ones_like(x) / 10.0,
            u_terminal=lambda x: np.zeros_like(x),
        )
        assert comp._lagrangian_class is None, (
            f"Expected _lagrangian_class=None for non-separable H, got {comp._lagrangian_class!r}."
        )

    def test_control_cost_regularize_smooth_no_base(self):
        """Smooth ControlCostBase with base=None → returns self (no re-wrap)."""
        from mfgarchon.core.hamiltonian import QuadraticControlCost

        cc = QuadraticControlCost(lambda_=1.0)
        # QuadraticControlCost is smooth; base is None → regularize returns self
        result = cc.regularize(epsilon=0.1)
        # Smooth + no base: returns self
        assert result is cc, "Smooth QuadraticControlCost.regularize() should return self when not already wrapped."

    def test_control_cost_regularize_l1_wraps(self):
        """Non-smooth L1ControlCost → regularize returns a Moreau-Yosida wrapper."""
        from mfgarchon.core.hamiltonian import L1ControlCost

        cc = L1ControlCost(lambda_=1.0)
        result = cc.regularize(epsilon=0.1)
        # Non-smooth: must be a distinct wrapped object
        assert result is not cc
        # The wrapper's base must be the original
        assert result.base is cc  # type: ignore[attr-defined]

    def test_control_cost_regularize_strips_epsilon_and_re_wraps(self):
        """Regularizing an already-regularized cost re-wraps the ORIGINAL base."""
        from mfgarchon.core.hamiltonian import L1ControlCost

        cc = L1ControlCost(lambda_=1.0)
        wrapped1 = cc.regularize(epsilon=0.5)
        wrapped2 = wrapped1.regularize(epsilon=0.1)
        # wrapped2 must wrap the original cc, not wrapped1
        assert wrapped2.base is cc, (  # type: ignore[attr-defined]
            "Re-regularization must re-wrap the original base, not the previous wrapper."
        )

    def test_jacobian_fd_scalar_dhdp(self):
        """jacobian_fd works when dH_dp returns a scalar (ndim=0)."""
        from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian

        H = SeparableHamiltonian(control_cost=QuadraticControlCost(lambda_=1.0))
        x = np.array([0.5])
        m = 0.3
        p = np.array([1.0])
        # Should not raise regardless of whether dp returns array or scalar
        jac = H.jacobian_fd(x, m, p, dx=0.01, scheme="central")
        assert jac is not None

    def test_temp_resolved_bc_initialized_to_none(self):
        """MFGProblem._temp_resolved_bc is initialized to None (explicit None-init)."""
        from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
        from mfgarchon.core.mfg_components import MFGComponents
        from mfgarchon.geometry import TensorProductGrid
        from mfgarchon.geometry.boundary import no_flux_bc

        H = SeparableHamiltonian(control_cost=QuadraticControlCost(lambda_=1.0))
        grid = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[10], boundary_conditions=no_flux_bc(dimension=1))

        from mfgarchon.core.mfg_problem import MFGProblem

        comp = MFGComponents(
            hamiltonian=H,
            m_initial=lambda x: np.ones_like(x),
            u_terminal=lambda x: np.zeros_like(x),
        )
        problem = MFGProblem(
            geometry=grid,
            components=comp,
            sigma=0.1,
            T=1.0,
            Nt=5,
        )
        # Must be explicitly initialized, not set at runtime
        assert hasattr(problem, "_temp_resolved_bc"), (
            "MFGProblem must declare _temp_resolved_bc in __init__ (not lazily)."
        )
        assert problem._temp_resolved_bc is None, "_temp_resolved_bc must start as None."
