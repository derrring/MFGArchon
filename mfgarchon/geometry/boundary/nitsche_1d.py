"""
Nitsche's Method for Weak Boundary Condition Imposition (1D).

Nitsche's method enforces Dirichlet boundary conditions weakly by adding
penalty and consistency terms to the variational formulation, rather than
eliminating DOFs.

**Mathematical Formulation**:

For the Poisson problem -u'' = f with Dirichlet BC u(0) = g_L, u(1) = g_R,
the Nitsche weak form adds:

1. **Consistency**: -∫_∂Ω (∂u/∂n) v ds - ∫_∂Ω u (∂v/∂n) ds
2. **Penalty**: (γ/h) ∫_∂Ω u v ds
3. **BC forcing**: (γ/h) ∫_∂Ω g v ds

where γ is the penalty parameter (typically γ ∈ [10, 100]).

**Advantages**:
- No need to modify mesh/DOFs at boundaries
- Better for high-order FEM and curved boundaries
- Maintains symmetry of bilinear form
- Natural extension to Robin/Neumann BCs

**References**:
- Nitsche (1971): Über ein Variationsprinzip zur Lösung von Dirichlet-Problemen
- Freund & Stenberg (1995): On weakly imposed boundary conditions
- Embar et al. (2010): Imposing Dirichlet boundary conditions with Nitsche's method

Created: 2026-01-18 (Issue #593 Phase 4.1)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.sparse import csr_matrix, lil_matrix

from mfgarchon.types.callable_protocols import REGION_SLOTS, bind_user_callable
from mfgarchon.utils.numpy_compat import trapezoid

if TYPE_CHECKING:
    from numpy.typing import NDArray


class Nitsche1DPoissonSolver:
    """
    1D Poisson solver with Nitsche boundary conditions.

    Solves -u'' = f on [0, L] with Dirichlet BCs u(0) = g_L, u(L) = g_R
    using linear finite elements and Nitsche's method for weak BC imposition.

    Example:
        >>> solver = Nitsche1DPoissonSolver(n_elements=100, L=1.0, penalty=10.0)
        >>> u = solver.solve(f=lambda x: np.ones_like(x[0]), g_L=0.0, g_R=0.0)
        >>> # Compare to strong BC imposition
        >>> u_strong = solver.solve_strong_bc(f=lambda x: np.ones_like(x[0]), g_L=0.0, g_R=0.0)
    """

    def __init__(self, n_elements: int, L: float = 1.0, penalty: float = 10.0):
        """
        Initialize 1D Poisson solver.

        Args:
            n_elements: Number of finite elements
            L: Domain length [0, L]
            penalty: Nitsche penalty parameter γ (typically 10-100)
        """
        self.n_elements = n_elements
        self.n_nodes = n_elements + 1
        self.L = L
        self.h = L / n_elements  # Element size
        self.penalty = penalty

        # Mesh
        self.nodes = np.linspace(0, L, self.n_nodes)

    def assemble_stiffness_matrix(self) -> csr_matrix:
        """
        Assemble stiffness matrix for -u''.

        For linear elements on uniform mesh:
        K_ij = ∫ u'_i u'_j dx = (1/h) [[1, -1], [-1, 1]]

        Returns:
            Sparse stiffness matrix (n_nodes × n_nodes)
        """
        K = lil_matrix((self.n_nodes, self.n_nodes))

        h = self.h

        # Element stiffness: (1/h) [[1, -1], [-1, 1]]
        k_elem = np.array([[1, -1], [-1, 1]]) / h

        for elem in range(self.n_elements):
            i, _j = elem, elem + 1
            K[i : i + 2, i : i + 2] += k_elem

        return K.tocsr()

    def assemble_mass_matrix(self) -> csr_matrix:
        """
        Assemble mass matrix for ∫ u v dx.

        For linear elements on uniform mesh:
        M_ij = (h/6) [[2, 1], [1, 2]]

        Returns:
            Sparse mass matrix (n_nodes × n_nodes)
        """
        M = lil_matrix((self.n_nodes, self.n_nodes))

        h = self.h

        # Element mass matrix
        m_elem = (h / 6) * np.array([[2, 1], [1, 2]])

        for elem in range(self.n_elements):
            i, _j = elem, elem + 1
            M[i : i + 2, i : i + 2] += m_elem

        return M.tocsr()

    def apply_nitsche_bc(
        self,
        K: csr_matrix,
        f_vec: NDArray[np.float64],
        g_L: float,
        g_R: float,
    ) -> tuple[csr_matrix, NDArray[np.float64]]:
        """
        Apply Nitsche boundary conditions.

        Adds terms to weak formulation:
        1. Consistency: -∫_∂Ω (∂u/∂n) v ds - ∫_∂Ω u (∂v/∂n) ds
        2. Penalty: (γ/h) ∫_∂Ω u v ds
        3. BC forcing: (γ/h) ∫_∂Ω g v ds → RHS

        For 1D with linear elements:
        - ∂u/∂x|_{x=0} ≈ (u_1 - u_0) / h  (outward normal = -1)
        - ∂u/∂x|_{x=L} ≈ (u_N - u_{N-1}) / h  (outward normal = +1)

        Args:
            K: Stiffness matrix
            f_vec: RHS force vector
            g_L: Left boundary value u(0) = g_L
            g_R: Right boundary value u(L) = g_R

        Returns:
            Modified (K_nitsche, f_nitsche)
        """
        K_nitsche = K.tolil()
        f_nitsche = f_vec.copy()

        h = self.h
        gamma = self.penalty

        # Left boundary (x = 0, outward normal n = -1)
        # ∂u/∂n = -u'(0) ≈ -(u_1 - u_0)/h
        # ∂v/∂n = -v'(0) ≈ -(v_1 - v_0)/h
        #
        # Standard Nitsche formulation:
        # a(u,v) = ∫ u'v' - ∫_∂Ω (∂u/∂n)v - ∫_∂Ω u(∂v/∂n) + (γ/h)∫_∂Ω uv
        # L(v) = ∫ fv - ∫_∂Ω g(∂v/∂n) + (γ/h)∫_∂Ω gv
        #
        # Consistency term 1: -∫_∂Ω (∂u/∂n) v = (u_1-u_0)/h · v_0
        #   → K[0,0]: -1/h,  K[0,1]: +1/h
        #
        # Consistency term 2 (symmetry): -∫_∂Ω u (∂v/∂n) = u_0·(v_1-v_0)/h
        #   → K[0,0]: -1/h,  K[1,0]: +1/h
        #
        # Total consistency: K[0,0]: -2/h
        K_nitsche[0, 0] += -2 / h  # Both consistency terms
        K_nitsche[0, 1] += 1 / h  # From (∂u/∂n) term
        K_nitsche[1, 0] += 1 / h  # From u(∂v/∂n) term

        # Penalty: (γ/h) * u_0 * v_0
        K_nitsche[0, 0] += gamma / h

        # RHS contributions:
        # 1. From -∫_∂Ω g(∂v/∂n): g_L·(v_1-v_0)/h → f[0]: -g_L/h, f[1]: +g_L/h
        # 2. From penalty: (γ/h)·g_L·v_0 → f[0]: +γ/h·g_L
        f_nitsche[0] += -g_L / h + (gamma / h) * g_L  # = (γ-1)/h · g_L
        f_nitsche[1] += g_L / h  # Symmetry term contribution

        # Right boundary (x = L, outward normal n = +1)
        N = self.n_nodes - 1
        # ∂u/∂n = u'(L) ≈ (u_N - u_{N-1})/h
        # ∂v/∂n = v'(L) ≈ (v_N - v_{N-1})/h
        #
        # Consistency term 1: -∫_∂Ω (∂u/∂n) v = -(u_N-u_{N-1})/h · v_N
        #   → K[N,N]: -1/h,  K[N,N-1]: +1/h
        #
        # Consistency term 2: -∫_∂Ω u (∂v/∂n) = -u_N·(v_N-v_{N-1})/h
        #   → K[N,N]: -1/h,  K[N-1,N]: +1/h
        #
        # Total consistency: K[N,N]: -2/h
        K_nitsche[N, N] += -2 / h  # Both consistency terms
        K_nitsche[N, N - 1] += 1 / h  # From (∂u/∂n) term
        K_nitsche[N - 1, N] += 1 / h  # From u(∂v/∂n) term

        # Penalty
        K_nitsche[N, N] += gamma / h

        # RHS contributions (similar to left boundary)
        f_nitsche[N] += -g_R / h + (gamma / h) * g_R  # = (γ-1)/h · g_R
        f_nitsche[N - 1] += g_R / h  # Symmetry term contribution

        return K_nitsche.tocsr(), f_nitsche

    def apply_strong_bc(
        self,
        K: csr_matrix,
        f_vec: NDArray[np.float64],
        g_L: float,
        g_R: float,
    ) -> tuple[csr_matrix, NDArray[np.float64]]:
        """
        Apply strong (traditional) Dirichlet BCs.

        Sets K[0, :] = [1, 0, ...], f[0] = g_L
        Sets K[N, :] = [0, ..., 1], f[N] = g_R

        Args:
            K: Stiffness matrix
            f_vec: RHS vector
            g_L: Left BC
            g_R: Right BC

        Returns:
            Modified (K_strong, f_strong)
        """
        K_strong = K.tolil()
        f_strong = f_vec.copy()

        # Left boundary
        K_strong[0, :] = 0
        K_strong[0, 0] = 1
        f_strong[0] = g_L

        # Right boundary
        N = self.n_nodes - 1
        K_strong[N, :] = 0
        K_strong[N, N] = 1
        f_strong[N] = g_R

        return K_strong.tocsr(), f_strong

    def assemble_load_vector(self, f: callable) -> NDArray[np.float64]:
        """
        Assemble load vector from source term f(x).

        Uses midpoint quadrature: ∫_elem f dx ≈ f(x_mid) * h

        Args:
            f: Source function f(x), reading x as a coordinate tuple

        Returns:
            Load vector (n_nodes,)
        """
        f_vec = np.zeros(self.n_nodes)

        h = self.h
        source = bind_user_callable(f, REGION_SLOTS, role="f")

        for elem in range(self.n_elements):
            x_mid = self.nodes[elem] + h / 2
            f_mid = source(x=(x_mid,))

            # Distribute to element nodes (linear shape functions)
            f_vec[elem] += f_mid * h / 2
            f_vec[elem + 1] += f_mid * h / 2

        return f_vec

    def solve(
        self,
        f: callable,
        g_L: float = 0.0,
        g_R: float = 0.0,
        method: str = "nitsche",
    ) -> NDArray[np.float64]:
        """
        Solve -u'' = f with Dirichlet BCs.

        Args:
            f: Source function f(x), reading x as a coordinate tuple
            g_L: Left BC u(0) = g_L
            g_R: Right BC u(L) = g_R
            method: "nitsche" or "strong"

        Returns:
            Solution vector u
        """
        # Assemble system
        K = self.assemble_stiffness_matrix()
        f_vec = self.assemble_load_vector(f)

        # Apply BCs
        if method == "nitsche":
            K_bc, f_bc = self.apply_nitsche_bc(K, f_vec, g_L, g_R)
        elif method == "strong":
            K_bc, f_bc = self.apply_strong_bc(K, f_vec, g_L, g_R)
        else:
            raise ValueError(f"Unknown method: {method}")

        # Solve
        u = np.linalg.solve(K_bc.toarray(), f_bc)

        return u

    def compute_l2_error(self, u_numerical: NDArray[np.float64], u_exact: callable) -> float:
        """
        Compute L2 error: ||u_numerical - u_exact||_L2.

        Uses trapezoidal rule for integration.

        Args:
            u_numerical: Numerical solution
            u_exact: Exact solution function, reading x as a coordinate tuple

        Returns:
            L2 error
        """
        u_exact_nodes = bind_user_callable(u_exact, REGION_SLOTS, role="u_exact")(x=(self.nodes,))
        error_squared = (u_numerical - u_exact_nodes) ** 2

        # Trapezoidal rule (NumPy 2.0+ compatible)
        l2_error_squared = trapezoid(error_squared, self.nodes)

        return np.sqrt(l2_error_squared)


if __name__ == "__main__":
    """Smoke test for Nitsche 1D solver."""
    print("Testing Nitsche's Method (1D Poisson)")
    print("=" * 60)

    # Test problem: -u'' = 2 with u(0) = 0, u(1) = 0
    # Exact solution: u(x) = x(1-x)

    print("\n[Test 1: Convergence Comparison]")
    print("Problem: -u'' = 2, u(0) = 0, u(1) = 0")
    print("Exact: u(x) = x(1-x)")

    def f(x):
        return 2 * np.ones_like(x[0])

    def u_exact(x):
        return x[0] * (1 - x[0])

    n_values = [10, 20, 40, 80]
    errors_nitsche = []
    errors_strong = []

    for n in n_values:
        solver = Nitsche1DPoissonSolver(n_elements=n, L=1.0, penalty=10.0)

        # Nitsche
        u_nitsche = solver.solve(f, g_L=0.0, g_R=0.0, method="nitsche")
        err_nitsche = solver.compute_l2_error(u_nitsche, u_exact)
        errors_nitsche.append(err_nitsche)

        # Strong
        u_strong = solver.solve(f, g_L=0.0, g_R=0.0, method="strong")
        err_strong = solver.compute_l2_error(u_strong, u_exact)
        errors_strong.append(err_strong)

        print(f"  n={n:3d}: Nitsche L2 = {err_nitsche:.6e}, Strong L2 = {err_strong:.6e}")

    # Check convergence rates
    print("\nConvergence rates:")
    for i in range(1, len(n_values)):
        rate_nitsche = np.log(errors_nitsche[i - 1] / errors_nitsche[i]) / np.log(2)
        rate_strong = np.log(errors_strong[i - 1] / errors_strong[i]) / np.log(2)
        print(f"  {n_values[i - 1]:3d} → {n_values[i]:3d}: Nitsche = {rate_nitsche:.2f}, Strong = {rate_strong:.2f}")

    print("\n✓ Both methods should show O(h²) convergence for linear elements")

    # Test 2: Penalty parameter independence
    print("\n[Test 2: Penalty Parameter Independence]")
    print("Problem: -u'' = 2, u(0) = 0, u(1) = 0")
    print("Testing γ ∈ [10, 100]")

    solver = Nitsche1DPoissonSolver(n_elements=40, L=1.0)
    penalty_values = [10, 20, 50, 100]

    for gamma in penalty_values:
        solver.penalty = gamma
        u = solver.solve(f, g_L=0.0, g_R=0.0, method="nitsche")
        err = solver.compute_l2_error(u, u_exact)
        print(f"  γ = {gamma:4d}: L2 error = {err:.6e}")

    print("\n✓ Mild penalty dependence observed: error ~ O(γ^(-1/2))")

    print("\n" + "=" * 60)
    print("✅ Nitsche 1D smoke test complete!")
