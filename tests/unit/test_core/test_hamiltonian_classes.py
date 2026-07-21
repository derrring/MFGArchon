"""
Unit tests for class-based Hamiltonian and Lagrangian (Issues #651, #667, #673).

Tests the new class hierarchy:
- MFGOperatorBase: Common base for H and L
- HamiltonianBase: Full MFG Hamiltonian H(x, m, p, t)
- LagrangianBase: Full MFG Lagrangian L(x, α, m, t)
- Legendre transform duality between H and L
"""

import pytest

import numpy as np

from mfgarchon.core.hamiltonian import (
    # Control cost classes
    BoundedControlCost,
    CongestionHamiltonian,
    # Dual classes (Legendre transform)
    DualHamiltonian,
    DualLagrangian,
    HamiltonianBase,
    L1ControlCost,
    LagrangianBase,
    OptimizationSense,
    QuadraticControlCost,
    QuadraticMFGHamiltonian,
    SeparableHamiltonian,
    create_hamiltonian,
)


class TestControlCostBase:
    """Tests for original ControlCostBase hierarchy."""

    def test_quadratic_control_cost_minimize(self):
        """Test quadratic control cost with MINIMIZE sense."""
        cost = QuadraticControlCost(sense=OptimizationSense.MINIMIZE, control_cost=2.0)
        p = np.array([1.0, 2.0, -3.0])

        alpha = cost.optimal_control(p)

        # α* = -p/λ for MINIMIZE
        expected = np.array([-0.5, -1.0, 1.5])
        np.testing.assert_allclose(alpha, expected)

    def test_quadratic_control_cost_maximize(self):
        """Test quadratic control cost with MAXIMIZE sense (RL convention)."""
        cost = QuadraticControlCost(sense=OptimizationSense.MAXIMIZE, control_cost=2.0)
        p = np.array([1.0, 2.0, -3.0])

        alpha = cost.optimal_control(p)

        # α* = +p/λ for MAXIMIZE
        expected = np.array([0.5, 1.0, -1.5])
        np.testing.assert_allclose(alpha, expected)

    def test_l1_control_cost_bang_bang(self):
        """Test L1 control cost gives bang-bang control."""
        cost = L1ControlCost(control_cost=1.5)
        p = np.array([0.5, 2.0, -3.0])

        alpha = cost.optimal_control(p)

        # α* = -sign(p) where |p| > λ, else 0
        expected = np.array([0.0, -1.0, 1.0])
        np.testing.assert_allclose(alpha, expected)

    def test_bounded_control_cost_clipping(self):
        """Test bounded control cost clips at max_control."""
        cost = BoundedControlCost(control_cost=1.0, max_control=1.5)
        p = np.array([1.0, 2.0, 3.0])

        alpha = cost.optimal_control(p)

        # Unconstrained would be -p, but clipped at ±1.5
        expected = np.array([-1.0, -1.5, -1.5])
        np.testing.assert_allclose(alpha, expected)


class TestMFGOperatorBase:
    """Tests for MFGOperatorBase common interface."""

    def test_hamiltonian_is_hamiltonian(self):
        """Test HamiltonianBase.is_hamiltonian returns True."""
        H = SeparableHamiltonian(control_cost=QuadraticControlCost())
        assert H.is_hamiltonian is True
        assert H.is_lagrangian is False

    def test_lagrangian_is_lagrangian(self):
        """Test LagrangianBase.is_lagrangian returns True."""

        class TestLagrangian(LagrangianBase):
            def __call__(self, x, alpha, m, t=0.0):
                return 0.5 * np.sum(alpha**2)

        L = TestLagrangian()
        assert L.is_lagrangian is True
        assert L.is_hamiltonian is False


class TestHamiltonianBase:
    """Tests for HamiltonianBase interface."""

    @pytest.fixture
    def simple_hamiltonian(self):
        """Create a simple separable Hamiltonian for testing."""
        return SeparableHamiltonian(
            control_cost=QuadraticControlCost(control_cost=2.0),
            coupling=lambda m: -(m**2),
            coupling_dm=lambda m: -2 * m,
        )

    def test_hamiltonian_evaluation(self, simple_hamiltonian):
        """Test Hamiltonian __call__ evaluation."""
        x = np.array([0.5])
        m = 0.3
        p = np.array([1.0])
        t = 0.0

        H_val = simple_hamiltonian(x, m, p, t)

        # H = ½|p|²/λ + f(m) = 0.5 * 1.0 / 2.0 + (-0.09) = 0.16
        expected = 0.5 * 1.0 / 2.0 - 0.3**2
        assert abs(H_val - expected) < 1e-10

    def test_hamiltonian_dp_analytic(self, simple_hamiltonian):
        """Test ∂H/∂p computation (analytic for quadratic)."""
        x = np.array([0.5])
        m = 0.3
        p = np.array([1.0])
        t = 0.0

        dp = simple_hamiltonian.dp(x, m, p, t)

        # For quadratic H_control = ½|p|²/λ, ∂H/∂p = p/λ
        expected = np.array([0.5])  # p/λ = 1/2
        np.testing.assert_allclose(dp, expected)

    def test_hamiltonian_dm_analytic(self, simple_hamiltonian):
        """Test ∂H/∂m computation (analytic when coupling_dm provided)."""
        x = np.array([0.5])
        m = 0.3
        p = np.array([1.0])
        t = 0.0

        dm = simple_hamiltonian.dm(x, m, p, t)

        # df/dm = -2m = -0.6
        assert abs(dm - (-0.6)) < 1e-10

    def test_hamiltonian_optimal_control(self, simple_hamiltonian):
        """Test optimal control computation."""
        x = np.array([0.5])
        m = 0.3
        p = np.array([1.0])
        t = 0.0

        alpha = simple_hamiltonian.optimal_control(x, m, p, t)

        # For quadratic: α* = -p/λ (MINIMIZE)
        expected = np.array([-0.5])
        np.testing.assert_allclose(alpha, expected)

    def test_hamiltonian_finite_diff_dm(self):
        """Test finite difference fallback for dm when coupling_dm not provided."""
        H = SeparableHamiltonian(
            control_cost=QuadraticControlCost(control_cost=1.0),
            coupling=lambda m: -(m**2),
            # No coupling_dm provided - should use finite diff
        )
        x = np.array([0.5])
        m = 0.3
        p = np.array([1.0])
        t = 0.0

        dm = H.dm(x, m, p, t)

        # Should approximate -2m = -0.6
        assert abs(dm - (-0.6)) < 0.01  # Allow some FD error


class TestLagrangianBase:
    """Tests for LagrangianBase interface."""

    @pytest.fixture
    def quadratic_lagrangian(self):
        """Create a quadratic Lagrangian for testing."""

        class QuadraticLagrangian(LagrangianBase):
            def __init__(self, lam=1.0):
                super().__init__()
                self.lam = lam

            def __call__(self, x, alpha, m, t=0.0):
                return 0.5 * self.lam * np.sum(alpha**2)

        return QuadraticLagrangian(lam=2.0)

    def test_lagrangian_evaluation(self, quadratic_lagrangian):
        """Test Lagrangian __call__ evaluation."""
        x = np.array([0.5])
        alpha = np.array([1.0])
        m = 0.3
        t = 0.0

        L_val = quadratic_lagrangian(x, alpha, m, t)

        # L = ½λ|α|² = 0.5 * 2 * 1 = 1.0
        assert abs(L_val - 1.0) < 1e-10

    def test_lagrangian_to_hamiltonian(self, quadratic_lagrangian):
        """Test Legendre transform L -> H."""
        H = quadratic_lagrangian.legendre_transform()

        assert isinstance(H, DualHamiltonian)

        # For L = ½λ|α|², H = ½|p|²/λ
        # With λ=2, p=1: H = 0.5 * 1 / 2 = 0.25
        x = np.array([0.5])
        m = 0.3
        p = np.array([1.0])
        t = 0.0

        H_val = H(x, m, p, t)
        assert abs(H_val - 0.25) < 0.05  # Allow numerical tolerance


class TestLegendreDuality:
    """Tests for Legendre transform duality (Issue #651)."""

    def test_hamiltonian_to_lagrangian(self):
        """Test inverse Legendre transform H -> L."""
        H = SeparableHamiltonian(
            control_cost=QuadraticControlCost(control_cost=2.0),
        )
        L = H.legendre_transform()

        assert isinstance(L, DualLagrangian)

        # For H = ½|p|²/λ, L = ½λ|α|²
        # With λ=2, α=1: L = 0.5 * 2 * 1 = 1.0
        x = np.array([0.5])
        alpha = np.array([1.0])
        m = 0.3
        t = 0.0

        L_val = L(x, alpha, m, t)
        assert abs(L_val - 1.0) < 0.1  # Allow numerical tolerance

    def test_duality_cycle_l_h_l(self):
        """Test L -> H -> L recovers original."""

        class QuadraticLagrangian(LagrangianBase):
            def __init__(self, lam=1.0):
                super().__init__()
                self.lam = lam

            def __call__(self, x, alpha, m, t=0.0):
                return 0.5 * self.lam * np.sum(alpha**2)

        L_orig = QuadraticLagrangian(lam=2.0)
        H_from_L = L_orig.legendre_transform()
        L_recovered = H_from_L.legendre_transform()

        x = np.array([0.5])
        alpha = np.array([1.0])
        m = 0.3
        t = 0.0

        L_orig_val = L_orig(x, alpha, m, t)
        L_recovered_val = L_recovered(x, alpha, m, t)

        # Should recover original within numerical tolerance
        assert abs(L_recovered_val - L_orig_val) < 0.2

    def test_dual_hamiltonian_maximize_returns_sup_not_inf(self):
        """Issue #1185: the Legendre conjugate H = L*(p) = sup_a {p.a - L(a)} is a
        supremum by definition, independent of OptimizationSense. Pre-fix the 1D
        ``DualHamiltonian.__call__`` returned the *infimum* under MAXIMIZE (the value at
        a control bound), disagreeing with its own ``dp`` argmax and the d>1 scipy branch.
        """

        class _QuadL(LagrangianBase):
            def __init__(self, lam, sense):
                super().__init__(sense=sense)
                self.lam = lam

            def __call__(self, x, alpha, m, t=0.0):
                return 0.5 * self.lam * np.sum(np.atleast_1d(alpha) ** 2)

        x, m, p, t = np.array([0.5]), 0.3, np.array([1.0]), 0.0
        H_min = _QuadL(2.0, OptimizationSense.MINIMIZE).legendre_transform()
        H_max = _QuadL(2.0, OptimizationSense.MAXIMIZE).legendre_transform()

        # H = sup_a {p a - 0.5 lam a^2} = p^2/(2 lam) = 0.25 (lam=2, p=1); NOT the
        # bound infimum -110 the pre-fix MAXIMIZE branch returned.
        assert abs(H_max(x, m, p, t) - 0.25) < 0.05
        # sense must not change the conjugate value
        assert abs(H_max(x, m, p, t) - H_min(x, m, p, t)) < 1e-9
        # 1D must agree with the d>1 branch under MAXIMIZE: H([1,1]) = |p|^2/(2 lam) = 0.5
        assert abs(H_max(x, m, np.array([1.0, 1.0]), t) - 0.5) < 0.05
        # envelope consistency: H(p) == p.alpha* - L(alpha*), alpha* = dp
        a = float(H_max.dp(x, m, p, t)[0])
        assert abs(H_max(x, m, p, t) - (1.0 * a - 0.5 * 2.0 * a**2)) < 1e-2

    def test_dual_lagrangian_maximize_returns_sup_not_inf(self):
        """Issue #1185: the symmetric DualLagrangian L = H*(a) = sup_p {p.a - H(p)} must
        also be the supremum under MAXIMIZE (not the infimum)."""
        x, alpha, m, t = np.array([0.5]), np.array([1.0]), 0.3, 0.0
        H = SeparableHamiltonian(
            control_cost=QuadraticControlCost(control_cost=2.0),
            sense=OptimizationSense.MAXIMIZE,
        )
        L = H.legendre_transform()  # DualLagrangian(sense=MAXIMIZE)
        # L = H*(alpha) = 0.5 lam alpha^2 = 1.0 at lam=2, alpha=1
        assert abs(L(x, alpha, m, t) - 1.0) < 0.1


class TestQuadraticMFGHamiltonian:
    """Tests for QuadraticMFGHamiltonian."""

    def test_default_hamiltonian_matches_mfg_problem_default(self):
        """Test QuadraticMFGHamiltonian gives same result as MFGProblem default."""
        H = QuadraticMFGHamiltonian(coupling_coefficient=1.0)

        x = np.array([0.5])
        m = 0.3
        p = np.array([1.0])
        t = 0.0

        H_val = H(x, m, p, t)

        # H = ½c|p|² - m² = 0.5 * 1.0 * 1.0 - 0.09 = 0.41
        expected = 0.5 * 1.0 * 1.0**2 - 0.3**2
        assert abs(H_val - expected) < 1e-10

    def test_default_hamiltonian_dm(self):
        """Test QuadraticMFGHamiltonian.dm() = -2m."""
        H = QuadraticMFGHamiltonian(coupling_coefficient=1.0)

        x = np.array([0.5])
        m = 0.3
        p = np.array([1.0])
        t = 0.0

        dm = H.dm(x, m, p, t)

        assert abs(dm - (-0.6)) < 1e-10


class TestCreateHamiltonian:
    """Tests for create_hamiltonian factory function."""

    def test_create_quadratic(self):
        """Test creating quadratic Hamiltonian via factory."""
        H = create_hamiltonian("quadratic", control_cost=2.0)

        assert isinstance(H, SeparableHamiltonian)

        x = np.array([0.5])
        m = 0.0
        p = np.array([1.0])
        t = 0.0

        H_val = H(x, m, p, t)
        # H = ½|p|²/λ = 0.5 * 1 / 2 = 0.25
        assert abs(H_val - 0.25) < 1e-10

    def test_create_default(self):
        """Test creating default MFG Hamiltonian via factory."""
        H = create_hamiltonian("default", coupling_coefficient=2.0)

        assert isinstance(H, QuadraticMFGHamiltonian)
        assert H.coupling_coefficient == 2.0

    def test_create_invalid_raises(self):
        """Test that invalid type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown hamiltonian_type"):
            create_hamiltonian("invalid_type")


# Issue #673: TestToLegacyFunc removed - to_legacy_func() method deleted
# Class-based Hamiltonian API is now called directly (no legacy function wrappers)


class TestBatchPolymorphism:
    """Tests for batch-polymorphic Hamiltonian __call__/dp/dm (Issue #775)."""

    @pytest.fixture
    def separable_2d(self):
        """2D separable Hamiltonian with coupling and potential."""
        return SeparableHamiltonian(
            control_cost=QuadraticControlCost(control_cost=2.0),
            potential=lambda x, t: float(np.sum(x**2)),
            coupling=lambda m: -(m**2),
            coupling_dm=lambda m: -2 * m,
        )

    @pytest.fixture
    def batch_data_2d(self):
        """Batch inputs: N=5 points in d=2."""
        N, d = 5, 2
        rng = np.random.default_rng(42)
        x = rng.standard_normal((N, d))
        m = rng.uniform(0.1, 2.0, size=N)
        p = rng.standard_normal((N, d))
        return x, m, p

    def test_separable_call_batch(self, separable_2d, batch_data_2d):
        """Batch __call__ returns shape (N,)."""
        x, m, p = batch_data_2d
        result = separable_2d(x, m, p, t=0.0)
        assert isinstance(result, np.ndarray)
        assert result.shape == (5,)

    def test_separable_dp_batch(self, separable_2d, batch_data_2d):
        """Batch dp returns shape (N, d)."""
        x, m, p = batch_data_2d
        result = separable_2d.dp(x, m, p, t=0.0)
        assert isinstance(result, np.ndarray)
        assert result.shape == (5, 2)

    def test_separable_dm_batch(self, separable_2d, batch_data_2d):
        """Batch dm returns shape (N,)."""
        x, m, p = batch_data_2d
        result = separable_2d.dm(x, m, p, t=0.0)
        assert isinstance(result, np.ndarray)
        assert result.shape == (5,)

    def test_separable_batch_matches_pointwise(self, separable_2d, batch_data_2d):
        """Batch results match stacked single-point results."""
        x, m, p = batch_data_2d
        N = x.shape[0]

        # Batch
        H_batch = separable_2d(x, m, p)
        dp_batch = separable_2d.dp(x, m, p)
        dm_batch = separable_2d.dm(x, m, p)

        # Pointwise
        H_pw = np.array([separable_2d(x[i], m[i], p[i]) for i in range(N)])
        dp_pw = np.stack([separable_2d.dp(x[i], m[i], p[i]) for i in range(N)])
        dm_pw = np.array([separable_2d.dm(x[i], m[i], p[i]) for i in range(N)])

        np.testing.assert_allclose(H_batch, H_pw, rtol=1e-12)
        np.testing.assert_allclose(dp_batch, dp_pw, rtol=1e-12)
        np.testing.assert_allclose(dm_batch, dm_pw, rtol=1e-12)

    def test_base_dp_batch_fallback(self):
        """Custom Hamiltonian with no analytic dp uses FD batch path."""

        class CustomH(HamiltonianBase):
            def __call__(self, x, m, p, t=0.0):
                # H = sum(p^2) + m (non-separable, no analytic dp)
                p_arr = np.atleast_1d(p)
                return float(np.sum(p_arr**2)) + float(m)

        H = CustomH()
        N, d = 4, 2
        rng = np.random.default_rng(7)
        x = rng.standard_normal((N, d))
        m = rng.uniform(0.1, 2.0, size=N)
        p = rng.standard_normal((N, d))

        result = H.dp(x, m, p, t=0.0)
        assert result.shape == (N, d)

        # Verify against pointwise
        dp_pw = np.stack([H.dp(x[i], m[i], p[i]) for i in range(N)])
        np.testing.assert_allclose(result, dp_pw, rtol=1e-10)

    def test_base_dm_batch_fallback(self):
        """Custom Hamiltonian with no analytic dm uses FD batch path."""

        class CustomH(HamiltonianBase):
            def __call__(self, x, m, p, t=0.0):
                p_arr = np.atleast_1d(p)
                return float(np.sum(p_arr**2)) + float(m) ** 2

        H = CustomH()
        N, d = 4, 2
        rng = np.random.default_rng(7)
        x = rng.standard_normal((N, d))
        m = rng.uniform(0.1, 2.0, size=N)
        p = rng.standard_normal((N, d))

        result = H.dm(x, m, p, t=0.0)
        assert isinstance(result, np.ndarray)
        assert result.shape == (N,)

        # Verify against pointwise
        dm_pw = np.array([H.dm(x[i], m[i], p[i]) for i in range(N)])
        np.testing.assert_allclose(result, dm_pw, rtol=1e-10)

    def test_single_point_unchanged(self):
        """Single-point calls return same types as before (backward compat)."""
        H = SeparableHamiltonian(
            control_cost=QuadraticControlCost(control_cost=2.0),
            coupling=lambda m: -(m**2),
            coupling_dm=lambda m: -2 * m,
        )
        x = np.array([0.5, 1.0])
        m = 0.3
        p = np.array([1.0, -0.5])

        H_val = H(x, m, p, t=0.0)
        assert isinstance(H_val, float)

        dp_val = H.dp(x, m, p, t=0.0)
        assert isinstance(dp_val, np.ndarray)
        assert dp_val.shape == (2,)

        dm_val = H.dm(x, m, p, t=0.0)
        assert isinstance(dm_val, float)

    def test_dm_no_coupling_batch(self):
        """Batch dm with no coupling returns zeros array."""
        H = SeparableHamiltonian(control_cost=QuadraticControlCost(control_cost=1.0))
        N, d = 3, 2
        x = np.zeros((N, d))
        m = np.ones(N)
        p = np.ones((N, d))

        result = H.dm(x, m, p)
        assert isinstance(result, np.ndarray)
        assert result.shape == (N,)
        np.testing.assert_array_equal(result, np.zeros(N))

    def test_quadratic_mfg_batch(self):
        """QuadraticMFGHamiltonian works in batch mode."""
        H = QuadraticMFGHamiltonian(coupling_coefficient=1.0)
        N, d = 5, 2
        rng = np.random.default_rng(99)
        x = rng.standard_normal((N, d))
        m = rng.uniform(0.1, 2.0, size=N)
        p = rng.standard_normal((N, d))

        H_batch = H(x, m, p)
        assert isinstance(H_batch, np.ndarray)
        assert H_batch.shape == (N,)

        # Verify pointwise
        H_pw = np.array([H(x[i], m[i], p[i]) for i in range(N)])
        np.testing.assert_allclose(H_batch, H_pw, rtol=1e-12)


class TestCongestionHamiltonian:
    """Tests for CongestionHamiltonian (Issue #782)."""

    @pytest.fixture
    def congestion_1d(self):
        """1D congestion Hamiltonian: H = |p|^2/(2*lambda*(1+m)) + V(x)."""
        return CongestionHamiltonian(
            control_cost=QuadraticControlCost(control_cost=2.0),
            congestion_factor=lambda m: 1.0 + m,
            congestion_factor_dm=lambda m: np.ones_like(m) if isinstance(m, np.ndarray) else 1.0,
            potential=lambda x, t: float(np.sum(x**2)),
        )

    @pytest.fixture
    def congestion_2d(self):
        """2D congestion Hamiltonian with coupling."""
        gamma, vol = 0.5, 2.0
        return CongestionHamiltonian(
            control_cost=QuadraticControlCost(control_cost=1.0),
            congestion_factor=lambda m: 1.0 + gamma * vol * m,
            congestion_factor_dm=lambda m: np.full_like(m, gamma * vol) if isinstance(m, np.ndarray) else gamma * vol,
            coupling=lambda m: -(m**2),
            coupling_dm=lambda m: -2 * m,
        )

    @pytest.fixture
    def batch_data_2d(self):
        """Batch inputs: N=5 points in d=2."""
        N, d = 5, 2
        rng = np.random.default_rng(42)
        x = rng.standard_normal((N, d))
        m = rng.uniform(0.1, 2.0, size=N)
        p = rng.standard_normal((N, d))
        return x, m, p

    def test_single_point_call(self, congestion_1d):
        """Single-point __call__ returns float."""
        x = np.array([0.5])
        m = 1.0
        p = np.array([3.0])
        result = congestion_1d(x, m, p)
        assert isinstance(result, float)
        # Manual: |p|^2/(2*lambda*c(m)) + V(x)
        # = 9/(2*2*(1+1)) + 0.25 = 9/8 + 0.25 = 1.375
        expected = 9.0 / (2 * 2.0 * 2.0) + 0.25
        assert result == pytest.approx(expected)

    def test_single_point_dp(self, congestion_1d):
        """Single-point dp returns correct shape and value."""
        x = np.array([0.5])
        m = 1.0
        p = np.array([3.0])
        result = congestion_1d.dp(x, m, p)
        assert result.shape == (1,)
        # dH/dp = p / (lambda * c(m)) = 3 / (2 * 2) = 0.75
        expected = 3.0 / (2.0 * 2.0)
        np.testing.assert_allclose(result, [expected])

    def test_single_point_dm(self, congestion_1d):
        """Single-point dm returns correct value."""
        x = np.array([0.5])
        m = 1.0
        p = np.array([3.0])
        result = congestion_1d.dm(x, m, p)
        assert isinstance(result, float)
        # dH/dm = -c'(m)*|p|^2/(2*lambda*c(m)^2)
        # = -1*9/(2*2*4) = -9/16 = -0.5625
        expected = -1.0 * 9.0 / (2.0 * 2.0 * 4.0)
        assert result == pytest.approx(expected)

    def test_batch_call_shape(self, congestion_2d, batch_data_2d):
        """Batch __call__ returns shape (N,)."""
        x, m, p = batch_data_2d
        result = congestion_2d(x, m, p)
        assert isinstance(result, np.ndarray)
        assert result.shape == (5,)

    def test_batch_dp_shape(self, congestion_2d, batch_data_2d):
        """Batch dp returns shape (N, d)."""
        x, m, p = batch_data_2d
        result = congestion_2d.dp(x, m, p)
        assert isinstance(result, np.ndarray)
        assert result.shape == (5, 2)

    def test_batch_dm_shape(self, congestion_2d, batch_data_2d):
        """Batch dm returns shape (N,)."""
        x, m, p = batch_data_2d
        result = congestion_2d.dm(x, m, p)
        assert isinstance(result, np.ndarray)
        assert result.shape == (5,)

    def test_batch_matches_pointwise(self, congestion_2d, batch_data_2d):
        """Batch results match stacked single-point results."""
        x, m, p = batch_data_2d
        N = x.shape[0]

        H_batch = congestion_2d(x, m, p)
        dp_batch = congestion_2d.dp(x, m, p)
        dm_batch = congestion_2d.dm(x, m, p)

        H_pw = np.array([congestion_2d(x[i], m[i], p[i]) for i in range(N)])
        dp_pw = np.stack([congestion_2d.dp(x[i], m[i], p[i]) for i in range(N)])
        dm_pw = np.array([congestion_2d.dm(x[i], m[i], p[i]) for i in range(N)])

        np.testing.assert_allclose(H_batch, H_pw, rtol=1e-12)
        np.testing.assert_allclose(dp_batch, dp_pw, rtol=1e-12)
        np.testing.assert_allclose(dm_batch, dm_pw, rtol=1e-12)

    def test_unit_congestion_matches_separable(self):
        """With c(m)=1, CongestionHamiltonian reduces to SeparableHamiltonian."""
        cc = QuadraticControlCost(control_cost=2.0)
        coupling = lambda m: -(m**2)  # noqa: E731
        coupling_dm = lambda m: -2 * m  # noqa: E731

        H_cong = CongestionHamiltonian(
            control_cost=cc,
            congestion_factor=lambda m: 1.0,
            congestion_factor_dm=lambda m: 0.0,
            coupling=coupling,
            coupling_dm=coupling_dm,
        )
        H_sep = SeparableHamiltonian(
            control_cost=cc,
            coupling=coupling,
            coupling_dm=coupling_dm,
        )

        rng = np.random.default_rng(123)
        x = rng.standard_normal(2)
        m = 1.5
        p = rng.standard_normal(2)

        np.testing.assert_allclose(H_cong(x, m, p), H_sep(x, m, p), rtol=1e-12)
        np.testing.assert_allclose(H_cong.dp(x, m, p), H_sep.dp(x, m, p), rtol=1e-12)
        np.testing.assert_allclose(H_cong.dm(x, m, p), H_sep.dm(x, m, p), rtol=1e-12)

    def test_analytic_dm_matches_finite_diff(self):
        """Analytic dm agrees with finite-difference fallback."""
        cc = QuadraticControlCost(control_cost=1.5)
        gamma = 0.8

        H_analytic = CongestionHamiltonian(
            control_cost=cc,
            congestion_factor=lambda m: 1.0 + gamma * m,
            congestion_factor_dm=lambda m: gamma,
        )
        H_fd = CongestionHamiltonian(
            control_cost=cc,
            congestion_factor=lambda m: 1.0 + gamma * m,
            congestion_factor_dm=None,  # Force finite-difference fallback
        )

        x = np.array([1.0, -0.5])
        m = 2.0
        p = np.array([0.7, -1.2])

        dm_analytic = H_analytic.dm(x, m, p)
        dm_fd = H_fd.dm(x, m, p)
        np.testing.assert_allclose(dm_analytic, dm_fd, rtol=1e-4)

    def test_optimal_control(self, congestion_1d):
        """Optimal control is -sign * dp."""
        x = np.array([0.5])
        m = 1.0
        p = np.array([3.0])
        alpha = congestion_1d.optimal_control(x, m, p)
        dp_val = congestion_1d.dp(x, m, p)
        # MINIMIZE sense: alpha = -dp
        np.testing.assert_allclose(alpha, -dp_val)


# ============================================================================
# Issue #898: New ControlCostBase interface tests
# ============================================================================


class TestControlCostEvaluate:
    """Tests for evaluate() — must return finite values for all inputs."""

    def test_quadratic_evaluate(self):
        cost = QuadraticControlCost(lambda_=2.0)
        p = np.array([1.0, 2.0, 3.0])
        result = cost.evaluate(p)
        # sum(p^2) / (2*lambda) = (1+4+9) / 4 = 3.5
        expected = 0.5 * np.sum(p**2) / 2.0
        np.testing.assert_allclose(result, expected)

    def test_l1_evaluate_below_threshold(self):
        cost = L1ControlCost(lambda_=0.5)
        p = np.array([0.3, -0.2])
        result = cost.evaluate(p)
        np.testing.assert_allclose(result, [0.0, 0.0])

    def test_l1_evaluate_above_threshold(self):
        cost = L1ControlCost(lambda_=0.5)
        p = np.array([0.7, -0.8, 1.5])
        result = cost.evaluate(p)
        expected = np.maximum(np.abs(p) - 0.5, 0.0)
        np.testing.assert_allclose(result, expected)

    def test_l1_evaluate_always_finite(self):
        cost = L1ControlCost(lambda_=1.0)
        p = np.array([0.0, 0.5, 1.0, 5.0, 100.0])
        result = cost.evaluate(p)
        assert np.all(np.isfinite(result))

    def test_bounded_evaluate_unsaturated(self):
        cost = BoundedControlCost(lambda_=1.0, max_control=2.0)
        p = np.array([1.0])  # |p| < lambda * alpha_max = 2
        result = cost.evaluate(p)
        np.testing.assert_allclose(result, [0.5])

    def test_bounded_evaluate_saturated(self):
        cost = BoundedControlCost(lambda_=1.0, max_control=2.0)
        p = np.array([5.0])  # |p| > lambda * alpha_max = 2
        result = cost.evaluate(p)
        expected = 2.0 * 5.0 - 0.5 * 1.0 * 4.0  # a_max*|p| - lam*a_max^2/2
        np.testing.assert_allclose(result, [expected])


class TestControlCostDp:
    """Tests for dp() — gradient / subdifferential."""

    def test_quadratic_dp(self):
        cost = QuadraticControlCost(lambda_=2.0)
        p = np.array([1.0, 2.0])
        np.testing.assert_allclose(cost.dp(p), [0.5, 1.0])

    def test_l1_dp_below_threshold(self):
        cost = L1ControlCost(lambda_=0.5)
        p = np.array([0.3, -0.2])
        np.testing.assert_allclose(cost.dp(p), [0.0, 0.0])

    def test_l1_dp_above_threshold(self):
        cost = L1ControlCost(lambda_=0.5)
        p = np.array([0.7, -0.8])
        np.testing.assert_allclose(cost.dp(p), [1.0, -1.0])

    def test_bounded_dp_unsaturated(self):
        cost = BoundedControlCost(lambda_=1.0, max_control=2.0)
        p = np.array([1.0])
        np.testing.assert_allclose(cost.dp(p), [1.0])

    def test_bounded_dp_saturated(self):
        cost = BoundedControlCost(lambda_=1.0, max_control=2.0)
        p = np.array([5.0, -5.0])
        np.testing.assert_allclose(cost.dp(p), [2.0, -2.0])


class TestControlCostProximal:
    """Tests for proximal() — prox of Lagrangian for ADMM."""

    def test_quadratic_proximal(self):
        cost = QuadraticControlCost(lambda_=2.0)
        z = np.array([3.0])
        result = cost.proximal(1.0, z)
        np.testing.assert_allclose(result, [1.0])  # 3 / (1 + 2)

    def test_l1_proximal_soft_threshold(self):
        cost = L1ControlCost(lambda_=0.5)
        z = np.array([1.0, 0.2, -0.8])
        result = cost.proximal(1.0, z)
        # soft threshold: sign(z) * max(|z| - tau*lambda, 0), clipped to [-1, 1]
        expected = np.sign(z) * np.maximum(np.abs(z) - 0.5, 0.0)
        expected = np.clip(expected, -1.0, 1.0)
        np.testing.assert_allclose(result, expected)

    def test_bounded_proximal(self):
        cost = BoundedControlCost(lambda_=1.0, max_control=2.0)
        z = np.array([1.0, 5.0])
        result = cost.proximal(0.5, z)
        # z / (1 + tau*lam), clipped to [-2, 2]
        expected = np.clip(z / 1.5, -2.0, 2.0)
        np.testing.assert_allclose(result, expected)


class TestControlCostRegularize:
    """Tests for regularize() — Moreau-Yosida smoothing."""

    def test_quadratic_regularize_returns_self(self):
        cost = QuadraticControlCost(lambda_=1.0)
        result = cost.regularize(0.1)
        assert result is cost  # already smooth

    def test_l1_regularize_returns_smooth(self):
        cost = L1ControlCost(lambda_=0.5)
        smooth = cost.regularize(0.1)
        assert smooth.is_smooth()
        assert smooth is not cost

    def test_l1_regularize_evaluate_is_smooth(self):
        """Regularized L1 should have smooth evaluate near the kink."""
        cost = L1ControlCost(lambda_=0.5)
        smooth = cost.regularize(0.1)
        # At p = 0.5 (the kink), evaluate should be smooth
        p_near = np.array([0.45, 0.5, 0.55])
        vals = smooth.evaluate(p_near)
        assert np.all(np.isfinite(vals))
        # Gradient should be continuous (no jump)
        grads = smooth.dp(p_near)
        assert np.all(np.isfinite(grads))

    def test_l1_regularize_dp_lipschitz(self):
        """Regularized dp should be Lipschitz (no discontinuity)."""
        cost = L1ControlCost(lambda_=0.5)
        smooth = cost.regularize(0.1)
        p = np.linspace(0.0, 1.0, 100)
        grads = np.array([smooth.dp(np.array([pi]))[0] for pi in p])
        # Check Lipschitz: |grad[i+1] - grad[i]| <= C * dp
        diffs = np.abs(np.diff(grads))
        dp = p[1] - p[0]
        lip_const = np.max(diffs / dp)
        assert lip_const < 1.0 / 0.1 + 1.0  # Lipschitz const <= 1/epsilon

    def test_continuation_re_regularize(self):
        """regularize().regularize() should use original base cost."""
        cost = L1ControlCost(lambda_=0.5)
        smooth1 = cost.regularize(0.1)
        smooth2 = smooth1.regularize(0.01)
        assert smooth2.epsilon == 0.01
        assert smooth2.base is cost  # base is the ORIGINAL L1, not the first smoothed

    def test_bounded_regularize(self):
        cost = BoundedControlCost(lambda_=1.0, max_control=2.0)
        smooth = cost.regularize(0.1)
        assert smooth.is_smooth()


class TestMoreauYosidaPenaltyAggregation:
    """The Moreau penalty must be counted once, not once per component.

    H_eps(p) = H(prox(p)) + |p - prox(p)|^2 / (2*eps) is a functional of the whole
    momentum vector. Base control costs disagree on whether evaluate() returns one
    value per component (L1, bounded) or a total already summed over axis=-1
    (quadratic), so adding a summed penalty to a per-component base term multiplied
    the penalty by the dimension: exact at d=1, wrong for every d>1. Every other
    test in this class is 1-D, where the defect is invisible.
    """

    EPSILON = 0.1

    @staticmethod
    def _bases():
        return {
            "l1": L1ControlCost(lambda_=0.5),
            "bounded": BoundedControlCost(lambda_=1.0, max_control=1.0),
        }

    @pytest.mark.parametrize("dimension", [1, 2, 3])
    # 0.2 sits inside the prox region (zero penalty), the rest span the
    # soft-threshold and saturated branches where the penalty is nonzero.
    @pytest.mark.parametrize("p_magnitude", [0.2, 0.52, 0.8, 2.0])
    @pytest.mark.parametrize("base_id", ["l1", "bounded"])
    def test_envelope_equals_base_total_plus_one_penalty(self, base_id, p_magnitude, dimension):
        base = self._bases()[base_id]
        smooth = base.regularize(self.EPSILON)
        p = np.full(dimension, p_magnitude)
        q = smooth._prox_h(p)

        expected = np.sum(base.evaluate(q)) + np.sum((p - q) ** 2) / (2 * self.EPSILON)
        assert float(np.sum(smooth.evaluate(p))) == pytest.approx(expected, rel=0.0, abs=1e-15)

    @pytest.mark.parametrize("p_magnitude", [0.52, 0.8, 2.0])
    @pytest.mark.parametrize("base_id", ["l1", "bounded"])
    def test_envelope_is_additive_over_components(self, base_id, p_magnitude):
        """A constant momentum vector must cost exactly d times the 1-D value.

        Reference-free: H_eps is separable here, so H_eps(full(d, v)) = d*H_eps([v])
        with no appeal to prox. A penalty counted per-component gives d^2 instead of
        d, so this separates the two aggregations without recomputing either.
        """
        smooth = self._bases()[base_id].regularize(self.EPSILON)
        one_d = float(np.sum(smooth.evaluate(np.full(1, p_magnitude))))

        for dimension in (2, 3, 5):
            got = float(np.sum(smooth.evaluate(np.full(dimension, p_magnitude))))
            assert got == pytest.approx(dimension * one_d, rel=1e-14)

    @pytest.mark.parametrize("base_id", ["l1", "bounded"])
    def test_batch_evaluate_returns_one_value_per_point(self, base_id):
        """Batch input (N, d) must reduce over components, as the callers assume.

        SeparableHamiltonian.__call__ sums evaluate()'s result only for
        single-point input; for batch it adds V and f(m) of shape (N,) directly.
        """
        base = self._bases()[base_id]
        smooth = base.regularize(self.EPSILON)
        p_batch = np.array([[0.8, 0.8], [1.2, 0.3]])
        q_batch = smooth._prox_h(p_batch)

        values = smooth.evaluate(p_batch)
        expected = np.sum(base.evaluate(q_batch), axis=-1) + np.sum((p_batch - q_batch) ** 2, axis=-1) / (
            2 * self.EPSILON
        )
        assert values.shape == (2,)
        np.testing.assert_allclose(values, expected, rtol=0.0, atol=1e-15)


class TestControlCostLambda:
    """Tests for lambda_ attribute and deprecation."""

    def test_lambda_via_keyword(self):
        cost = QuadraticControlCost(lambda_=3.0)
        assert cost.lambda_ == 3.0

    def test_lambda_via_positional(self):
        """Legacy: positional control_cost still works."""
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            cost = QuadraticControlCost(sense=OptimizationSense.MINIMIZE, control_cost=3.0)
            assert cost.lambda_ == 3.0

    def test_control_cost_property_deprecated(self):
        import warnings

        cost = QuadraticControlCost(lambda_=2.0)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            val = cost.control_cost
            assert val == 2.0
            assert any("deprecated" in str(x.message).lower() for x in w)

    def test_hamiltonian_method_deprecated(self):
        import warnings

        cost = QuadraticControlCost(lambda_=2.0)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            cost.hamiltonian(np.array([1.0]))
            assert any("deprecated" in str(x.message).lower() for x in w)

    def test_cannot_specify_both(self):
        with pytest.raises(ValueError, match="Cannot specify both"):
            QuadraticControlCost(control_cost=1.0, lambda_=2.0)


class TestSeparableHamiltonianDp:
    """Test that SeparableHamiltonian.dp() delegates correctly (no isinstance)."""

    def test_dp_with_quadratic(self):
        H = SeparableHamiltonian(control_cost=QuadraticControlCost(lambda_=2.0))
        x, m, p = np.array([0.5]), 0.3, np.array([1.0])
        np.testing.assert_allclose(H.dp(x, m, p, 0.0), [0.5])

    def test_dp_with_l1(self):
        H = SeparableHamiltonian(control_cost=L1ControlCost(lambda_=0.5))
        x, m = np.array([0.5]), 0.3
        # Below threshold
        np.testing.assert_allclose(H.dp(x, m, np.array([0.3]), 0.0), [0.0])
        # Above threshold
        np.testing.assert_allclose(H.dp(x, m, np.array([0.7]), 0.0), [1.0])

    def test_dp_with_bounded(self):
        H = SeparableHamiltonian(control_cost=BoundedControlCost(lambda_=1.0, max_control=2.0))
        x, m = np.array([0.5]), 0.3
        # Unsaturated
        np.testing.assert_allclose(H.dp(x, m, np.array([1.0]), 0.0), [1.0])
        # Saturated
        np.testing.assert_allclose(H.dp(x, m, np.array([5.0]), 0.0), [2.0])


class TestHamiltonianBaseRegularize:
    """Test HamiltonianBase.regularize() behavior."""

    def test_smooth_hamiltonian_returns_self(self):
        H = SeparableHamiltonian(control_cost=QuadraticControlCost(lambda_=1.0))
        assert H.regularize(0.1) is H

    def test_nonsmooth_separable_regularize(self):
        H = SeparableHamiltonian(
            control_cost=L1ControlCost(lambda_=0.5),
            coupling=lambda m: -(m**2),
            coupling_dm=lambda m: -2 * m,
        )
        Hs = H.regularize(0.1)
        assert isinstance(Hs, SeparableHamiltonian)
        assert Hs.is_smooth()
        # Coupling preserved
        assert Hs._coupling is H._coupling
        assert Hs._coupling_dm is H._coupling_dm

    def test_nonsmooth_congestion_regularize(self):
        H = CongestionHamiltonian(
            control_cost=L1ControlCost(lambda_=0.5),
            congestion_factor=lambda m: 1.0 + m,
            congestion_factor_dm=lambda m: 1.0,
        )
        Hs = H.regularize(0.1)
        assert isinstance(Hs, CongestionHamiltonian)
        assert Hs.is_smooth()

    def test_dual_hamiltonian_regularize_raises(self):
        """DualHamiltonian has no control_cost — regularize raises."""

        H = SeparableHamiltonian(control_cost=QuadraticControlCost(lambda_=1.0))
        L = H.legendre_transform()
        H_dual = L.legendre_transform()
        # DualHamiltonian is smooth (quadratic base) so returns self
        assert H_dual.regularize(0.1) is H_dual


# ============================================================================
# Issue #904: LagrangianBase redesign
# ============================================================================


class TestSeparableLagrangian:
    """Tests for SeparableLagrangian with ControlCostBase."""

    def test_quadratic_evaluate(self):
        from mfgarchon.core.hamiltonian import SeparableLagrangian

        L = SeparableLagrangian(control_cost=QuadraticControlCost(lambda_=2.0))
        x, m = np.array([0.5]), 0.3
        # L(alpha=1) = lambda/2 * |alpha|^2 = 1.0
        np.testing.assert_allclose(L(x, np.array([1.0]), m, 0.0), 1.0)

    def test_quadratic_optimal_control(self):
        from mfgarchon.core.hamiltonian import SeparableLagrangian

        L = SeparableLagrangian(control_cost=QuadraticControlCost(lambda_=2.0))
        x, m = np.array([0.5]), 0.3
        # alpha* = -p/lambda = -2/2 = -1
        np.testing.assert_allclose(L.optimal_control(x, m, np.array([2.0]), 0.0), [-1.0])

    def test_quadratic_evaluate_hamiltonian(self):
        from mfgarchon.core.hamiltonian import SeparableLagrangian

        L = SeparableLagrangian(control_cost=QuadraticControlCost(lambda_=2.0))
        x, m = np.array([0.5]), 0.3
        # H(p=2) = |p|^2 / (2*lambda) = 4/4 = 1
        np.testing.assert_allclose(L.evaluate_hamiltonian(x, m, np.array([2.0]), 0.0), 1.0)

    def test_quadratic_proximal(self):
        from mfgarchon.core.hamiltonian import SeparableLagrangian

        L = SeparableLagrangian(control_cost=QuadraticControlCost(lambda_=2.0))
        # prox_{tau*L}(z) = z / (1 + tau*lambda) = 3/3 = 1
        np.testing.assert_allclose(L.proximal(1.0, np.array([3.0])), [1.0])

    def test_l1_optimal_control(self):
        from mfgarchon.core.hamiltonian import SeparableLagrangian

        L = SeparableLagrangian(control_cost=L1ControlCost(lambda_=0.5))
        x, m = np.array([0.5]), 0.3
        # |p| = 0.7 > lambda = 0.5, so alpha* = -sign(p) = -1
        np.testing.assert_allclose(L.optimal_control(x, m, np.array([0.7]), 0.0), [-1.0])

    def test_l1_evaluate_hamiltonian(self):
        from mfgarchon.core.hamiltonian import SeparableLagrangian

        L = SeparableLagrangian(control_cost=L1ControlCost(lambda_=0.5))
        x, m = np.array([0.5]), 0.3
        # H(p=0.7) = max(|0.7| - 0.5, 0) = 0.2
        np.testing.assert_allclose(L.evaluate_hamiltonian(x, m, np.array([0.7]), 0.0), 0.2, atol=1e-10)

    def test_l1_control_bounds(self):
        from mfgarchon.core.hamiltonian import SeparableLagrangian

        L = SeparableLagrangian(control_cost=L1ControlCost(lambda_=0.5))
        assert L.control_bounds() == (-1.0, 1.0)

    def test_bounded_control_bounds(self):
        from mfgarchon.core.hamiltonian import SeparableLagrangian

        L = SeparableLagrangian(control_cost=BoundedControlCost(lambda_=1.0, max_control=2.0))
        assert L.control_bounds() == (-2.0, 2.0)

    def test_as_hamiltonian(self):
        from mfgarchon.core.hamiltonian import SeparableLagrangian

        L = SeparableLagrangian(
            control_cost=QuadraticControlCost(lambda_=2.0),
            coupling=lambda m: -(m**2),
        )
        H = L.as_hamiltonian()
        assert isinstance(H, SeparableHamiltonian)
        # Same control cost object
        assert H.control_cost is L.control_cost

    def test_optimal_control_matches_hamiltonian(self):
        """L.optimal_control and H.optimal_control should agree."""
        from mfgarchon.core.hamiltonian import SeparableLagrangian

        cc = L1ControlCost(lambda_=0.5)
        L = SeparableLagrangian(control_cost=cc)
        H = SeparableHamiltonian(control_cost=cc)

        x, m, p, t = np.array([0.5]), 0.3, np.array([0.7]), 0.0
        np.testing.assert_allclose(
            L.optimal_control(x, m, p, t),
            H.optimal_control(x, m, p, t),
        )


class TestLagrangianBaseNumerical:
    """Test LagrangianBase default numerical methods (non-separable case)."""

    def test_numerical_optimal_control(self):
        """Custom Lagrangian uses scipy fallback for optimal_control.

        Issue #1642 (capability B5) corrected the expected value here from +1 to -1.
        This test previously asserted the conjugate maximizer and called it alpha*:
        its stationarity condition ``p = dL/dalpha = alpha^3 -> alpha = p^(1/3) = 1``
        solves ``argmax_alpha {p.alpha - L}``, which is ``dH/dp``, not the control.
        Under MINIMIZE the optimal control is ``alpha* = -dH/dp`` -- the convention
        ``HamiltonianBase.optimal_control`` and ``SeparableLagrangian.optimal_control``
        have always used, and which the base class now matches.
        """
        from mfgarchon.core.hamiltonian import LagrangianBase, OptimizationSense

        class QuarticLagrangian(LagrangianBase):
            def __call__(self, x, alpha, m, t=0.0):
                return 0.25 * np.sum(alpha**4)

        L = QuarticLagrangian(sense=OptimizationSense.MINIMIZE)
        x, m, p = np.array([0.0]), 0.0, np.array([1.0])
        # dH/dp solves p = alpha^3 -> alpha = p^(1/3) = 1; alpha* = -dH/dp = -1.
        np.testing.assert_allclose(L.optimal_control(x, m, p, 0.0), [-1.0], atol=0.05)
        # MAXIMIZE takes the other branch: alpha* = +dH/dp = +1.
        L_max = QuarticLagrangian(sense=OptimizationSense.MAXIMIZE)
        np.testing.assert_allclose(L_max.optimal_control(x, m, p, 0.0), [1.0], atol=0.05)
        # The Hamiltonian value is sense-independent: H = p.(dH/dp) - L(dH/dp) = 0.75.
        np.testing.assert_allclose(L.evaluate_hamiltonian(x, m, p, 0.0), 0.75, atol=0.05)
        np.testing.assert_allclose(L_max.evaluate_hamiltonian(x, m, p, 0.0), 0.75, atol=0.05)

    def test_numerical_proximal(self):
        """Custom Lagrangian uses scipy fallback for proximal."""
        from mfgarchon.core.hamiltonian import LagrangianBase, OptimizationSense

        class QuadLagrangian(LagrangianBase):
            def __call__(self, x, alpha, m, t=0.0):
                return 0.5 * np.sum(alpha**2)

        L = QuadLagrangian(sense=OptimizationSense.MINIMIZE)
        # prox_{tau*L}(z) = z/(1+tau) for L=|a|^2/2
        result = L.proximal(1.0, np.array([3.0]))
        np.testing.assert_allclose(result, [1.5], atol=0.05)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
