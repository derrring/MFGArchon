"""
GKS (Gustafsson-Kreiss-Sundström) Stability Analysis for BC Discretizations.

The GKS condition validates that boundary condition discretizations do not
introduce numerical instabilities when combined with interior discretizations.

**Theory**:

For a PDE ∂u/∂t = Lu with boundary conditions, GKS stability requires that
the combined discretization (interior + boundary) has eigenvalues satisfying:

- **Parabolic**: Re(λ) ≤ 0 (dissipative)
- **Hyperbolic**: |Im(λ)| bounded (no exponential growth)
- **Elliptic**: All eigenvalues have consistent sign

**Usage** (Developer Tool):

This module is for **validating BC implementations**, not for runtime checking.
Typical workflow:

1. Implement new BC discretization
2. Run GKS validation on representative test problems
3. Document results in BC stability verification notes
4. Add as optional CI check

**References**:

[1] Gustafsson, B., Kreiss, H. O., & Oliger, J. (1995). Time Dependent Problems
    and Difference Methods. Wiley.
[2] Kreiss, H. O., & Lorenz, J. (1989). Initial-Boundary Value Problems and the
    Navier-Stokes Equations. Academic Press.

Created: 2026-01-18 (Issue #593 Phase 4.2)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
from scipy.sparse import csr_matrix, issparse
from scipy.sparse.linalg import eigs

from mfgarchon.utils.mfg_logging import get_logger

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = get_logger(__name__)


@dataclass
class GKSResult:
    """
    Result of GKS stability analysis.

    Attributes:
        stable: Whether the BC discretization is GKS-stable
        eigenvalues: Computed eigenvalues of combined operator
        criterion: Stability criterion used
        max_real_part: Maximum real part of eigenvalues
        max_imag_part: Maximum imaginary part of eigenvalues
        pde_type: Type of PDE (parabolic, hyperbolic, elliptic)
        bc_description: Description of BC being tested
    """

    stable: bool
    eigenvalues: NDArray[np.complex128]
    criterion: str
    max_real_part: float
    max_imag_part: float
    pde_type: Literal["parabolic", "hyperbolic", "elliptic"]
    bc_description: str

    def __str__(self) -> str:
        """Human-readable summary."""
        status = "✅ STABLE" if self.stable else "❌ UNSTABLE"
        return (
            f"GKS Analysis: {status}\n"
            f"  PDE type: {self.pde_type}\n"
            f"  BC: {self.bc_description}\n"
            f"  Criterion: {self.criterion}\n"
            f"  max(Re(λ)): {self.max_real_part:.6e}\n"
            f"  max(Im(λ)): {self.max_imag_part:.6e}\n"
            f"  Eigenvalues computed: {len(self.eigenvalues)}"
        )


def check_gks_stability(
    operator: csr_matrix | NDArray,
    pde_type: Literal["parabolic", "hyperbolic", "elliptic"],
    bc_description: str = "unknown",
    tol: float = 1e-8,
    num_eigenvalues: int | None = None,
    max_dense_size: int = 2000,
) -> GKSResult:
    """
    Check GKS stability condition for a discretized PDE with BCs.

    This function computes eigenvalues of the combined spatial operator
    (including boundary conditions) and verifies the appropriate stability
    criterion for the given PDE type.

    Args:
        operator: Sparse or dense matrix representing the spatial discretization
            with boundary conditions applied. Shape: (N, N) where N is the number
            of DOFs.
        pde_type: Type of PDE being analyzed
            - "parabolic": Heat equation, diffusion (requires Re(λ) ≤ 0)
            - "hyperbolic": NOT IMPLEMENTED -- raises. See Raises below (Issue #1859).
            - "elliptic": Poisson, steady-state (requires consistent sign). Always uses the
              full spectrum, so it is bounded by ``max_dense_size``.
        bc_description: Human-readable description of the BC (for reporting)
        tol: Tolerance for numerical errors in eigenvalue real parts
        num_eigenvalues: Number of eigenvalues to compute (default: min(50, N-2)).
            Applies to the sparse path only, i.e. parabolic with N > 100.
        max_dense_size: Largest N for which ``pde_type="elliptic"`` will run. Definiteness is a
            statement about every eigenvalue, so that branch refuses rather than sampling one
            end of the spectrum. N=2000 costs about 2s and 32MB.

    Raises:
        NotImplementedError: for ``pde_type="hyperbolic"``. The previous criterion was a
            tautology that reported every operator stable, and it is not repairable per-operator:
            boundedness of exp(t·A) is invariant under A → c·A, so no single-operator predicate
            can reference dx. Use :func:`check_gks_convergence` for cross-grid behaviour.
        ValueError: for ``pde_type="elliptic"`` when ``N > max_dense_size``, or for an unknown
            ``pde_type``.

    Returns:
        GKSResult containing stability verdict and eigenvalue data

    Example:
        >>> # Build a 1D Laplacian operator (e.g., via geometry.get_laplacian_operator())
        >>> from mfgarchon.geometry.boundary import BoundaryConditions, BCType
        >>> # Build 1D Laplacian with Neumann BC
        >>> N = 50
        >>> A = build_laplacian_1d(N, dx=0.02, bc_type=BCType.NEUMANN)
        >>> result = check_gks_stability(
        ...     operator=A,
        ...     pde_type="parabolic",
        ...     bc_description="Neumann BC (2nd-order FDM)",
        ... )
        >>> print(result)
        GKS Analysis: ✅ STABLE
          PDE type: parabolic
          BC: Neumann BC (2nd-order FDM)
          Criterion: Re(λ) ≤ 1e-08
          max(Re(λ)): -2.456e-10
          max(Im(λ)): 1.234e-15
          Eigenvalues computed: 50

    Notes:
        - This is a **discrete stability** check, not PDE well-posedness
        - For PDE-level analysis, see L-S (Lopatinskii-Shapiro) methods (Issue #535)
        - Eigenvalue computation can be expensive for large N (O(N³) worst-case)
        - For production use, run once per BC type and document results
    """
    # Convert to sparse if needed
    if not issparse(operator):
        operator = csr_matrix(operator)

    N = operator.shape[0]

    # Determine number of eigenvalues to compute
    if num_eigenvalues is None:
        num_eigenvalues = min(50, N - 2)  # Leave margin for eigs() safety

    if pde_type == "hyperbolic":
        # Refusing, rather than returning a reassuring True (Issue #1859).
        #
        # The previous criterion was `max|Im(lambda)| <= 10 * max|lambda|`, both reduced from the
        # same array. Since |Im z| <= |z| for every complex z, that is implied by the definitions:
        # it returned stable=True for EVERY input, on the dense path as well as the sparse one.
        # Measured before removal: exponentially unstable transport operators amplifying by 7.8e42,
        # and A = 1e6 * I, all reported stable; over 200000 random matrices the worst observed
        # ratio was 0.9999999996 against a threshold of 10. The one escape hatch is closed too --
        # non-finite input raises in the eigensolver before the comparison is reached.
        #
        # It is not repaired to a weaker form because the intended check is not answerable from a
        # single operator at all. Two independent reasons:
        #   1. Boundedness of exp(t*A) is invariant under A -> c*A (a time reparametrization),
        #      so no per-operator predicate can legitimately reference dx.
        #   2. ||A|| = O(1/dx) already holds by CONSISTENCY, so the intended inequality is
        #      satisfied by every consistent scheme whether or not it is stable.
        # Cross-grid scaling belongs in `check_gks_convergence`, which is where a sequence of
        # operators at different dx is available.
        #
        # A three-valued replacement (energy-norm certificate + Kreiss-constant refutation) was
        # derived and rejected under adversarial review: at default settings it reports "unstable"
        # for a textbook 1-D acoustic system, and its two tiers answer in different norms, so an
        # operator can be certified non-expansive and refuted as amplifying at the same time.
        raise NotImplementedError(
            "No single-operator hyperbolic stability criterion is implemented. The previous check "
            "(|Im(lambda)| <= 10*||A||) was a tautology -- |Im z| <= |z| for every complex z -- so "
            "it reported stable=True for every input including exponentially unstable operators. "
            "Growth for non-normal A is a pseudospectral question, not a spectral one, and "
            "dx-scaling requires a sequence of operators: see check_gks_convergence. (Issue #1859)"
        )

    if pde_type == "elliptic" and max_dense_size < N:
        # `elliptic` asks whether the WHOLE spectrum shares a sign. A truncated solve samples one
        # end, and the opposite end is exactly where an indefiniteness would sit -- measured, an
        # operator with one negative eigenvalue among 201 was reported definite, because the 50
        # largest-magnitude eigenvalues lie in [1.40e5, 1.63e5] while the sign flip is at -14.80.
        #
        # So this branch uses the full spectrum below, and refuses above the size where that is
        # affordable rather than returning a verdict it cannot support. Two sparse methods were
        # derived and both were rejected under adversarial review: an LDL inertia count returns
        # `definite` for the periodic Laplacian, which has an exact null vector, and returns
        # `indefinite` for 18 of 27 upwind advection-diffusion operators whose spectra are
        # entirely positive; shift-invert misses interior-magnitude sign flips, which is precisely
        # the case a BC validator exists to catch.
        raise ValueError(
            f"pde_type='elliptic' needs the full spectrum (definiteness is a statement about ALL "
            f"eigenvalues), but N={N} exceeds max_dense_size={max_dense_size}. A truncated sparse "
            f"solve samples one end and cannot see a sign flip at the other. Raise max_dense_size "
            f"to accept the O(N^3) cost (N=2000 takes about 2s and 32MB), or reduce the problem. "
            f"(Issue #1859)"
        )

    # Which end of the spectrum to sample is decided by the criterion, not by convention:
    # `eigs` returns only `num_eigenvalues` of N, so asking for the wrong end silently omits
    # exactly the eigenvalues the test is about (Issue #1859).
    #
    # parabolic tests max Re(lambda), so it needs the largest REAL part ('LR'). Under the
    # previous 'LM' the returned values were the most negative ones (~ -4/dx^2 for a
    # discretized Laplacian), so max_real_part came back large and negative and `stable` was
    # True for every operator once N > 100 -- including one with Re(lambda) = +0.1.
    #
    # elliptic never reaches the sparse path -- it is refused above `max_dense_size` and takes
    # the full spectrum below it -- so 'LR' is the only choice left to make here.
    which = "LR"

    # `elliptic` is forced dense at any N: a one-ended sample cannot answer a whole-spectrum
    # question, and its verdict must not depend on which side of an internal size threshold the
    # input lands on. Measured before this change: the SAME mathematical object returned
    # definite=False (correct) at N=50 and N=100, and definite=True (wrong) at N=101/200/400.
    use_dense = N <= 100 or pde_type == "elliptic"

    try:
        if use_dense:
            # Dense solver: exact, and the only one that sees every eigenvalue
            eigenvalues = np.linalg.eigvals(operator.toarray())
        else:
            # For large problems, use sparse solver
            eigenvalues, _ = eigs(operator, k=num_eigenvalues, which=which, tol=1e-6)
    except Exception:
        # If sparse solver fails (common for small N or ill-conditioned), use dense
        logger.warning("Sparse eigensolver failed for N=%d, falling back to dense solver", N)
        eigenvalues = np.linalg.eigvals(operator.toarray())

    # Extract real and imaginary parts
    real_parts = eigenvalues.real
    imag_parts = eigenvalues.imag

    max_real_part = float(real_parts.max())
    max_imag_part = float(np.abs(imag_parts).max())

    # Check stability criterion based on PDE type
    if pde_type == "parabolic":
        # Parabolic: All eigenvalues must have non-positive real part (dissipative)
        criterion = f"Re(λ) ≤ {tol:.0e}"
        stable = max_real_part <= tol

    elif pde_type == "elliptic":
        # Elliptic: Eigenvalues should have consistent sign (definite operator)
        # Check if all eigenvalues are either positive or negative
        num_positive = np.sum(real_parts > tol)
        num_negative = np.sum(real_parts < -tol)
        criterion = "All Re(λ) same sign (definite operator)"
        stable = (num_positive == len(eigenvalues)) or (num_negative == len(eigenvalues))

    else:
        error_msg = f"Unknown PDE type: {pde_type}"
        raise ValueError(error_msg)

    return GKSResult(
        stable=stable,
        eigenvalues=eigenvalues,
        criterion=criterion,
        max_real_part=max_real_part,
        max_imag_part=max_imag_part,
        pde_type=pde_type,
        bc_description=bc_description,
    )


def check_gks_convergence(
    operator_sequence: list[csr_matrix | NDArray],
    grid_sizes: list[float],
    pde_type: Literal["parabolic", "hyperbolic", "elliptic"],
    bc_description: str = "unknown",
) -> dict[str, NDArray]:
    """
    Check GKS stability under mesh refinement.

    Verifies that stability is maintained as grid is refined (dx → 0).
    This is stronger than single-grid GKS: it ensures the discretization
    is **uniformly stable**.

    Args:
        operator_sequence: List of operators on progressively finer grids
        grid_sizes: Corresponding grid spacings [dx₁, dx₂, ..., dxₙ]
        pde_type: Type of PDE
        bc_description: Description of BC

    Returns:
        Dictionary with keys:
            - "stable": Boolean array indicating stability at each refinement
            - "max_real_parts": Array of max(Re(λ)) values
            - "max_imag_parts": Array of max(Im(λ)) values
            - "grid_sizes": Copy of input grid sizes

    Example:
        >>> operators = [build_laplacian_1d(N, dx) for N, dx in [(25, 0.04), (50, 0.02), (100, 0.01)]]
        >>> grid_sizes = [0.04, 0.02, 0.01]
        >>> convergence = check_gks_convergence(
        ...     operators, grid_sizes, "parabolic", "Neumann BC"
        ... )
        >>> assert all(convergence["stable"]), "GKS stability lost under refinement"
    """
    results = []
    for operator, dx in zip(operator_sequence, grid_sizes, strict=True):
        result = check_gks_stability(operator, pde_type, f"{bc_description} (dx={dx:.3e})")
        results.append(result)

    return {
        "stable": np.array([r.stable for r in results]),
        "max_real_parts": np.array([r.max_real_part for r in results]),
        "max_imag_parts": np.array([r.max_imag_part for r in results]),
        "grid_sizes": np.array(grid_sizes),
    }


if __name__ == "__main__":
    """
    Smoke test for GKS validation.

    Tests standard 1D Laplacian with Dirichlet BC.
    """
    print("GKS Stability Validation - Smoke Test")
    print("=" * 60)

    # Test 1: 1D Laplacian with Dirichlet BC (should be GKS-stable)
    print("\n[Test 1: 1D Laplacian + Dirichlet BC]")

    N = 50
    dx = 1.0 / (N - 1)

    # Build 2nd-order Laplacian: -u'' ≈ (u_{i-1} - 2u_i + u_{i+1}) / dx²
    diag = -2 * np.ones(N)
    off_diag = np.ones(N - 1)
    A = (
        csr_matrix(
            (
                np.concatenate([diag, off_diag, off_diag]),
                (
                    np.concatenate([np.arange(N), np.arange(N - 1), np.arange(1, N)]),
                    np.concatenate([np.arange(N), np.arange(1, N), np.arange(N - 1)]),
                ),
            ),
            shape=(N, N),
        )
        / dx**2
    )

    # Apply Dirichlet BC: Set first and last rows to identity
    A = A.tolil()
    A[0, :] = 0
    A[0, 0] = 1
    A[-1, :] = 0
    A[-1, -1] = 1
    A = A.tocsr()

    result = check_gks_stability(A, pde_type="parabolic", bc_description="Dirichlet BC (strong imposition)")

    print(result)
    print(f"\nMin eigenvalue: {result.eigenvalues.real.min():.6e}")
    print(f"Max eigenvalue: {result.eigenvalues.real.max():.6e}")

    # Test 2: Convergence under refinement
    print("\n[Test 2: GKS Stability Under Mesh Refinement]")

    grid_sizes = [1 / (N - 1) for N in [25, 50, 100]]
    operators = []

    for dx in grid_sizes:
        N_refined = int(1 / dx) + 1
        diag_r = -2 * np.ones(N_refined)
        off_diag_r = np.ones(N_refined - 1)
        A_refined = (
            csr_matrix(
                (
                    np.concatenate([diag_r, off_diag_r, off_diag_r]),
                    (
                        np.concatenate(
                            [
                                np.arange(N_refined),
                                np.arange(N_refined - 1),
                                np.arange(1, N_refined),
                            ]
                        ),
                        np.concatenate(
                            [
                                np.arange(N_refined),
                                np.arange(1, N_refined),
                                np.arange(N_refined - 1),
                            ]
                        ),
                    ),
                ),
                shape=(N_refined, N_refined),
            )
            / dx**2
        )

        # Apply Dirichlet
        A_refined = A_refined.tolil()
        A_refined[0, :] = 0
        A_refined[0, 0] = 1
        A_refined[-1, :] = 0
        A_refined[-1, -1] = 1
        operators.append(A_refined.tocsr())

    convergence = check_gks_convergence(operators, grid_sizes, "parabolic", "Dirichlet BC")

    print(f"Grid sizes: {convergence['grid_sizes']}")
    print(f"Stable: {convergence['stable']}")
    print(f"max(Re(λ)): {convergence['max_real_parts']}")

    if all(convergence["stable"]):
        print("\n✅ GKS stability preserved under refinement")
    else:
        print("\n❌ GKS stability lost during refinement")

    print("\n" + "=" * 60)
    print("✅ GKS smoke test complete!")
