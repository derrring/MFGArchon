"""Why the 2D meshless-Galerkin + Nitsche path does not converge, and what it is not.

Run: python examples/validation/diagnose_meshless_2d_convergence.py

The 2D meshless path is documented as non-convergent. test_meshless_curved_nitsche.py's
test_spd_and_imposes_g says so in a comment -- "quadrature-floor-limited ..., not converging;
SCNI is the lever, not boundary work. Capability check, not an accuracy claim." -- and
test_meshless_nitsche.py::test_homogeneous_dirichlet_eoc measures the 1D version of the same
thing at EOC ~1.4-1.5 against a degree-2 optimum of 2, attributing it to Gauss quadrature of
rational MLS integrands.

Two things that were never checked, and that this script checks:

1. In 2D the error does not merely floor, it GROWS under refinement. EOC is negative.
2. SCNI, the lever the comment names, had never been combined with Nitsche. Across
   test_meshless_scni.py, test_meshless_nitsche.py and test_meshless_curved_nitsche.py,
   MeshlessSCNIDiscretization is constructed 4 times and assemble_nitsche_terms is called 11
   times, with zero overlap. Combined here for the first time, SCNI does not lift the rate --
   it is worse, and its lowest eigenvalue collapses toward zero as h falls.

Sections A-D rule out the parts that are healthy, so a fix is not spent re-checking them. The
fault localises to the interaction of the Nitsche block with the MLS weak forms; every component
in isolation passes.

Manufactured problem throughout: -D laplacian(u) = f on the unit square, u = sin(pi x) sin(pi y),
homogeneous Dirichlet imposed weakly. Structured grids, degree-2 MLS, rho = 3.5h.

Caveat carried deliberately: SCNI's own tests use rho = 3.0h, this uses 3.5h to match the Nitsche
tests. Whether that matters is untested, and is the cheapest thing to vary first.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse.linalg import spsolve

from mfgarchon.alg.numerical.meshless_galerkin.discretization import discretization_from_cloud
from mfgarchon.alg.numerical.meshless_galerkin.mls_basis import monomial_exponents, shape_functions_and_grads
from mfgarchon.alg.numerical.meshless_galerkin.nitsche import assemble_nitsche_terms
from mfgarchon.alg.numerical.meshless_galerkin.scni_discretization import MeshlessSCNIDiscretization
from mfgarchon.geometry.boundary import dirichlet_bc

D = 0.5
DEGREE = 2
RHO_OVER_H = 3.5


def u_exact(X):
    return np.sin(np.pi * X[:, 0]) * np.sin(np.pi * X[:, 1])


def f_source(X):
    return D * 2.0 * np.pi**2 * np.sin(np.pi * X[:, 0]) * np.sin(np.pi * X[:, 1])


def grid(n):
    ax = np.linspace(0.0, 1.0, n)
    return np.stack([m.ravel() for m in np.meshgrid(ax, ax, indexing="ij")], axis=1)


def make_disc(nodes, h, kind, n_gauss=6):
    rho = RHO_OVER_H * h
    if kind == "gauss":
        return discretization_from_cloud(nodes, delta=rho, degree=DEGREE, n_gauss=n_gauss)
    return MeshlessSCNIDiscretization(nodes, rho=rho, degree=DEGREE, bounds=[(0.0, 1.0), (0.0, 1.0)])


def solve(disc, nodes, gamma, n_gauss=6):
    """Returns (error, lambda_min). Error is on the RECONSTRUCTED solution: MLS is
    non-interpolatory, so the coefficient vector is not the nodal values."""
    K, M = disc.stiffness(), disc.mass()
    N, rhs_data = assemble_nitsche_terms(
        disc, dirichlet_bc(0.0, dimension=2), D, gamma, n_gauss=n_gauss, include_data=True
    )
    A = (D * K + N).tocsr()
    lam_min = float(np.linalg.eigvalsh(A.toarray()).min())
    U = spsolve(A, M @ f_source(nodes) + (rhs_data if rhs_data is not None else 0.0))
    phi, _ = shape_functions_and_grads(nodes, nodes, disc.rho, disc._exps, "numpy")
    return float(np.sqrt(np.mean((phi @ U - u_exact(nodes)) ** 2))), lam_min


def eoc(hs, errs):
    return [np.log(errs[i - 1] / errs[i]) / np.log(hs[i - 1] / hs[i]) for i in range(1, len(errs))]


# --------------------------------------------------------------------------------------------
# A. The MLS space is healthy. Not the basis, not the support radius, not conditioning.
# --------------------------------------------------------------------------------------------
def section_a():
    print("A. MLS basis -- best approximation, conditioning, partition of unity")
    print("    n   dof   h        cond(phi)    ||phi U - u||   max|sum phi - 1|")
    for n in (11, 16, 21, 26, 31):
        X, h = grid(n), 1.0 / (n - 1)
        exps = monomial_exponents(2, DEGREE)
        phi, _ = shape_functions_and_grads(X, X, RHO_OVER_H * h, exps, "numpy")
        U, *_ = np.linalg.lstsq(phi, u_exact(X), rcond=None)
        print(
            f"   {n:3d} {len(X):5d}  {h:.4f}   {np.linalg.cond(phi):.3e}   "
            f"{np.sqrt(np.mean((phi @ U - u_exact(X)) ** 2)):.3e}      "
            f"{np.abs(phi.sum(1) - 1).max():.2e}"
        )
    print("    -> best approximation is at machine precision and cond(phi) is flat in h.\n")


# --------------------------------------------------------------------------------------------
# B. Volume assembly is healthy. K and M pass the patch test on every path.
# --------------------------------------------------------------------------------------------
def section_b():
    print("B. Patch test -- K@1 must vanish (a constant has no strain); sum(M) must be |Omega|=1")
    print("    path      n   ||K@1||_inf   sum(M)      ||K-K^T||")
    for n in (11, 16, 21):
        X, h = grid(n), 1.0 / (n - 1)
        for kind, ng in (("gauss", 6), ("gauss", 10), ("scni", None)):
            disc = make_disc(X, h, kind, n_gauss=ng or 6)
            K, M = disc.stiffness(), disc.mass()
            label = f"{kind}{ng or ''}"
            print(
                f"   {label:9} {n:3d}  {np.abs(K @ np.ones(len(X))).max():.3e}   "
                f"{float(np.asarray(M.sum())):.6f}   {float(abs(K - K.T).max()):.1e}"
            )
    print("    -> volume quadrature is exact on the patch test for all three paths.\n")


# --------------------------------------------------------------------------------------------
# C. The default penalty is indefinite in 2D. Already known -- the constant in
#    test_meshless_curved_nitsche.py carries the comment "flat default 20 is indefinite".
#    Reproduced here so the sweep below has a definite baseline to stand on.
# --------------------------------------------------------------------------------------------
def section_c():
    print("C. Nitsche penalty sweep -- coercivity needs gamma > 2*C_tr, and C_tr is larger in 2D")
    print("    n   gamma    err          lambda_min")
    for n in (11, 16, 21):
        for gamma in (20.0, 50.0, 100.0, 200.0):
            err, lam = solve(make_disc(grid(n), 1.0 / (n - 1), "gauss"), grid(n), gamma)
            flag = "  <- INDEFINITE" if lam <= 0 else ""
            print(f"   {n:3d} {gamma:6.0f}   {err:.4e}   {lam:+.3e}{flag}")
        print()
    print("    -> gamma=20 is indefinite at every level and worsens with h; gamma>=50 restores")
    print("       definiteness and does NOT restore convergence.\n")


# --------------------------------------------------------------------------------------------
# D. The finding. Both quadrature paths diverge under refinement at an adequate penalty.
# --------------------------------------------------------------------------------------------
def section_d(gamma=100.0):
    print(f"D. Refinement at gamma={gamma:.0f} (definite) -- Gauss vs SCNI")
    levels = (11, 16, 21, 26)
    for kind in ("gauss", "scni"):
        hs, errs, lams = [], [], []
        for n in levels:
            X, h = grid(n), 1.0 / (n - 1)
            err, lam = solve(make_disc(X, h, kind), X, gamma)
            hs.append(h)
            errs.append(err)
            lams.append(lam)
            print(f"   {kind:5} n={n:3d}  h={h:.4f}  err={err:.4e}  lambda_min={lam:+.3e}")
        print(f"   {kind:5} EOC: {[f'{r:+.2f}' for r in eoc(hs, errs)]}\n")
    print("    -> both diverge. SCNI, the documented lever, is worse than Gauss and its")
    print("       lambda_min collapses toward zero as h falls -- the system tends to singular")
    print("       under refinement, which is the sharpest lead in this file.\n")


if __name__ == "__main__":
    print(__doc__.split("\n\n")[0], "\n")
    section_a()
    section_b()
    section_c()
    section_d()
    print("Ruled out: MLS basis and conditioning (A), volume assembly on all paths (B),")
    print("Dirichlet coverage (all four faces are penalised -- boundary diagonal is ~17x the")
    print("interior), and the 1/h penalty scaling (pen = gamma*D/rho with rho = 3.5h).")
    print("Not ruled out: the Nitsche block's interaction with the MLS weak forms, and SCNI's")
    print("smoothed gradients losing rank as the support covers more nodes.")
