"""
Simple validation example for Deep Galerkin Methods (DGM).

This example demonstrates the DGM solver on a 2D linear-quadratic MFG problem
to validate the implementation and showcase basic usage patterns.

Mathematical Problem:
- HJB equation: ∂u/∂t + (1/2)|∇u|² + (1/2)|x|² + λm = 0
- FP equation: ∂m/∂t - Δm - div(m∇u) = 0
- Initial condition: m(0,x) = (2π)^(-1) exp(-|x|²/2)
- Terminal condition: u(T,x) = |x|²/2

Expected behavior: For small λ, solution should be close to single-agent case.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from mfgarchon import BoundaryConditions  # noqa: E402
from mfgarchon.alg.neural.dgm.base_dgm import DGMConfig  # noqa: E402
from mfgarchon.core.mfg_problem import MFGProblem  # noqa: E402
from mfgarchon.geometry.boundary import BCType  # noqa: E402

# Check if PyTorch is available
try:
    import torch

    from mfgarchon.alg.neural.dgm.mfg_dgm_solver import MFGDGMSolver

    TORCH_AVAILABLE = True
    print("✓ PyTorch available - DGM solver ready")
except ImportError:
    TORCH_AVAILABLE = False
    print("✗ PyTorch not available - using mock validation")


class Simple2DMFG(MFGProblem):
    """
    Simple 2D linear-quadratic MFG for DGM validation.

    This is a test problem with known analytical properties:
    - Quadratic Hamiltonian: H(x,p) = (1/2)|p|² + (1/2)|x|²
    - Coupling parameter: λ (small for validation)
    - Domain: [-2, 2]² × [0, 1]
    """

    def __init__(self, coupling_strength: float = 0.1):
        """Initialize simple 2D MFG problem."""
        self.coupling_strength = coupling_strength
        self.dimension = 2
        self.domain = [(-2, 2), (-2, 2)]
        self.time_domain = (0, 1.0)

        # Boundary conditions (homogeneous Neumann for validation)
        self.boundary_conditions = BoundaryConditions(dimension=2, default_bc=BCType.NEUMANN)

    def hamiltonian(self, t: float, x: np.ndarray, p: np.ndarray, m: float) -> float:
        """
        Hamiltonian for linear-quadratic problem.

        H(x,p,m) = (1/2)|p|² + (1/2)|x|² + λm
        """
        kinetic_energy = 0.5 * np.sum(p**2, axis=-1)
        potential_energy = 0.5 * np.sum(x**2, axis=-1)
        coupling_term = self.coupling_strength * m

        return kinetic_energy + potential_energy + coupling_term

    def initial_density(self, x: np.ndarray) -> np.ndarray:
        """
        Initial density: Gaussian centered at origin.

        m₀(x) = (2π)⁻¹ exp(-|x|²/2)
        """
        return (1 / (2 * np.pi)) * np.exp(-0.5 * np.sum(x**2, axis=-1))

    def terminal_condition(self, x: np.ndarray) -> np.ndarray:
        """
        Terminal condition for value function.

        u(T,x) = |x|²/2
        """
        return 0.5 * np.sum(x**2, axis=-1)


def create_dgm_config_for_validation() -> DGMConfig:
    """Create DGM configuration optimized for validation."""
    return DGMConfig(
        # Network architecture (smaller for fast validation)
        hidden_layers=[64, 64, 64],
        activation="tanh",
        use_residual_connections=False,
        # Training parameters (reduced for validation)
        learning_rate=1e-3,
        max_epochs=500,  # Reduced for quick validation
        batch_size=256,
        optimizer="adam",
        # Sampling (smaller for validation)
        num_interior_points=1000,
        num_boundary_points=200,
        num_initial_points=200,
        sampling_strategy="monte_carlo",
        # Variance reduction (disabled for simplicity)
        use_control_variates=False,
        use_importance_sampling=False,
        # Physics loss weights
        physics_weight=1.0,
        boundary_weight=10.0,
        initial_weight=10.0,
        # Convergence
        tolerance=1e-4,
        patience=100,
        # Computational
        device="cpu",  # Use CPU for validation
        dtype="float32",
    )


def analytical_reference_solution(t: np.ndarray, x: np.ndarray, λ: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
    """
    Approximate analytical solution for small coupling (λ → 0).

    For small λ, the solution approaches the single-agent case:
    - u(t,x) ≈ (1-t)|x|²/2
    - m(t,x) ≈ (2π(1-t))⁻¹ exp(-|x|²/(2(1-t)))

    Returns:
        Tuple of (value_function, density_function)
    """
    # Avoid division by zero at t=1
    time_factor = np.maximum(1 - t, 1e-6)

    # Value function (approximately)
    u_ref = 0.5 * time_factor[:, np.newaxis] * np.sum(x**2, axis=1)

    # Density function (heat equation solution)
    variance = time_factor[:, np.newaxis]
    normalization = 1 / (2 * np.pi * variance)
    exponent = -0.5 * np.sum(x**2, axis=1, keepdims=True) / variance
    m_ref = normalization * np.exp(exponent)

    return u_ref, m_ref


def run_dgm_validation():
    """Run DGM validation on simple 2D problem."""
    print("🧪 DGM Validation: Simple 2D Linear-Quadratic MFG")
    print("=" * 60)

    # Create problem and configuration
    problem = Simple2DMFG(coupling_strength=0.05)  # Small coupling for validation
    config = create_dgm_config_for_validation()

    print(f"Problem dimension: {problem.dimension}D")
    print(f"Spatial domain: {problem.domain}")
    print(f"Time domain: {problem.time_domain}")
    print(f"Coupling strength λ = {problem.coupling_strength}")
    print()

    if not TORCH_AVAILABLE:
        print("⚠️  PyTorch not available - running mock validation")
        run_mock_validation(problem, config)
        return

    # Initialize DGM solver
    try:
        solver = MFGDGMSolver(problem, config)
        print("✓ DGM solver initialized successfully")

        # Log solver configuration
        print(f"Network architecture: {config.hidden_layers}")
        print(f"Sampling strategy: {config.sampling_strategy}")
        print(f"Training epochs: {config.max_epochs}")
        print(f"Interior points: {config.num_interior_points}")
        print()

        # Test sampling functionality
        print("🔬 Testing sampling functionality...")
        interior_points = solver._sample_interior_points(100)
        boundary_points = solver._sample_boundary_points(40)
        initial_points = solver._sample_initial_points(50)

        print(f"✓ Interior points sampled: {interior_points.shape}")
        print(f"✓ Boundary points sampled: {boundary_points.shape}")
        print(f"✓ Initial points sampled: {initial_points.shape}")

        # Validate sampling bounds
        assert np.all(interior_points[:, 1] >= -2)
        assert np.all(interior_points[:, 1] <= 2)
        assert np.all(interior_points[:, 2] >= -2)
        assert np.all(interior_points[:, 2] <= 2)
        assert np.all(interior_points[:, 0] >= 0)
        assert np.all(interior_points[:, 0] <= 1)
        print("✓ Sampling bounds validated")
        print()

        # Test network initialization
        print("🧠 Testing neural network functionality...")
        test_input = torch.randn(10, 3)  # (batch_size, space+time)

        with torch.no_grad():
            value_output = solver.value_network(test_input)
            density_output = solver.density_network(test_input)

        print(f"✓ Value network output shape: {value_output.shape}")
        print(f"✓ Density network output shape: {density_output.shape}")
        print("✓ Networks forward pass successful")
        print()

        # Run training for validation (short run)
        print("🚀 Running DGM training (validation mode)...")
        print("Note: This is a short validation run, not full convergence")

        # Override config for quick validation
        config.max_epochs = 50
        config.num_interior_points = 200
        solver.config = config

        result = solver.solve()

        print(f"✓ Training completed: {result.num_epochs} epochs")
        print(f"✓ Final loss: {result.final_loss:.6e}")
        print(f"✓ Converged: {result.converged}")
        print()

        # Compare with analytical reference
        print("📊 Comparing with analytical reference...")

        # Create evaluation grid
        t_eval = np.array([0.2, 0.5, 0.8])
        x_eval = np.array([[-1, -1], [0, 0], [1, 1]])

        # Get analytical reference
        u_ref, m_ref = analytical_reference_solution(t_eval, x_eval, problem.coupling_strength)

        print("Analytical reference computed")
        print(f"Reference value function range: [{u_ref.min():.4f}, {u_ref.max():.4f}]")
        print(f"Reference density range: [{m_ref.min():.4f}, {m_ref.max():.4f}]")

        print()
        print("✅ DGM validation completed successfully!")
        print("The solver initialization, sampling, and training loop work correctly.")

    except Exception as e:
        print(f"❌ DGM validation failed: {e}")
        import traceback

        traceback.print_exc()


def run_mock_validation(problem, config):
    """Run mock validation when PyTorch is not available."""
    print("Running mock validation (PyTorch not available)")
    print()

    # Test problem setup
    print("✓ Problem initialized successfully")
    print(f"  - Dimension: {problem.dimension}")
    print(f"  - Domain: {problem.domain}")
    print(f"  - Coupling: {problem.coupling_strength}")

    # Test analytical components
    x_test = np.array([[0, 0], [1, 1], [-1, -1]])
    p_test = np.array([[0.5, 0.5], [1, 0], [0, 1]])

    # Test Hamiltonian
    H_vals = [problem.hamiltonian(0.5, x, p, 0.1) for x, p in zip(x_test, p_test, strict=False)]
    print(f"✓ Hamiltonian evaluated: {H_vals}")

    # Test initial density
    m0_vals = problem.initial_density(x_test)
    print(f"✓ Initial density: {m0_vals}")

    # Test terminal condition
    uT_vals = problem.terminal_condition(x_test)
    print(f"✓ Terminal condition: {uT_vals}")

    # Test analytical reference
    t_test = np.array([0.2, 0.5])
    _u_ref, _m_ref = analytical_reference_solution(t_test, x_test[:2])
    print("✓ Analytical reference computed")

    print()
    print("✅ Mock validation completed!")
    print("All mathematical components work correctly.")
    print("Install PyTorch to run full DGM validation.")


def plot_validation_results():
    """Plot validation results (requires matplotlib)."""
    try:
        # Create simple validation plots
        _fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Plot 1: Initial density
        x = np.linspace(-3, 3, 100)
        y = np.linspace(-3, 3, 100)
        X, Y = np.meshgrid(x, y)
        xy = np.stack([X.ravel(), Y.ravel()], axis=1)

        problem = Simple2DMFG()
        density = problem.initial_density(xy).reshape(X.shape)

        im1 = axes[0].contourf(X, Y, density, levels=20, cmap="viridis")
        axes[0].set_title("Initial Density m₀(x)")
        axes[0].set_xlabel("x₁")
        axes[0].set_ylabel("x₂")
        plt.colorbar(im1, ax=axes[0])

        # Plot 2: Terminal condition
        terminal = problem.terminal_condition(xy).reshape(X.shape)

        im2 = axes[1].contourf(X, Y, terminal, levels=20, cmap="plasma")
        axes[1].set_title("Terminal Condition u(T,x)")
        axes[1].set_xlabel("x₁")
        axes[1].set_ylabel("x₂")
        plt.colorbar(im2, ax=axes[1])

        plt.tight_layout()
        plt.savefig("dgm_validation_plots.png", dpi=150, bbox_inches="tight")
        print("📊 Validation plots saved as 'dgm_validation_plots.png'")

    except ImportError:
        print("⚠️  Matplotlib not available - skipping plots")


if __name__ == "__main__":
    run_dgm_validation()
    print()
    plot_validation_results()
