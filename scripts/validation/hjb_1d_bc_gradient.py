"""Standalone validation for Issue #1384 — 1D HJB boundary gradient must be BC-aware.

The default single-population ``HJBFDMSolver`` carried ``backend=NumPyBackend`` (not
``None``), so ``base_hjb`` computed the Hamiltonian momentum ``p = du/dx`` with periodic
``% Nx`` wraparound at the boundary regardless of the actual boundary condition. This
probe contrasts the legacy ``% Nx`` gradient against the BC-aware
``_compute_gradient_array_1d`` on a known function ``U = x**2`` (analytic ``du/dx = 2x``):

- legacy ``% Nx`` boundary gradient is periodic garbage (left ~ -9.975 vs analytic 0.0);
- ``neumann`` / ``no_flux`` give the BC-consistent boundary momentum;
- every BC choice leaves the interior unchanged to <= 1 ULP (2.2e-16) — the fix's blast
  radius is the two boundary points, and only for non-periodic BC.

Run: ``python scripts/validation/hjb_1d_bc_gradient.py``
"""

import numpy as np

from mfgarchon.alg.numerical.hjb_solvers import base_hjb as B
from mfgarchon.geometry.boundary import dirichlet_bc, neumann_bc, no_flux_bc, periodic_bc

# Known function U = x^2 on [0,1], analytic grad = 2x
Nx = 21
x = np.linspace(0.0, 1.0, Nx)
Dx = x[1] - x[0]
U = x**2
analytic = 2.0 * x


def legacy_central(U, Dx, Nx):
    g = np.empty(Nx)
    for i in range(Nx):
        d = B._calculate_derivatives(U, i, Dx, Nx, upwind=False, precomputed_gradient=None)
        g[i] = d[(1,)]
    return g


def legacy_upwind(U, Dx, Nx):
    g = np.empty(Nx)
    for i in range(Nx):
        d = B._calculate_derivatives(U, i, Dx, Nx, upwind=True, precomputed_gradient=None)
        g[i] = d[(1,)]
    return g


leg_c = legacy_central(U, Dx, Nx)
leg_u = legacy_upwind(U, Dx, Nx)

variants = {
    "bc=None central": B._compute_gradient_array_1d(U, Dx, bc=None, upwind=False),
    "neumann central": B._compute_gradient_array_1d(U, Dx, bc=neumann_bc(dimension=1), upwind=False),
    "no_flux central": B._compute_gradient_array_1d(U, Dx, bc=no_flux_bc(dimension=1), upwind=False),
    "dirichlet central": B._compute_gradient_array_1d(U, Dx, bc=dirichlet_bc(0.0, dimension=1), upwind=False),
    "periodic central": B._compute_gradient_array_1d(U, Dx, bc=periodic_bc(dimension=1), upwind=False),
}

np.set_printoptions(precision=4, suppress=True, linewidth=160)
print(f"Dx={Dx:.4f}  analytic grad at boundaries: left={analytic[0]:.4f} right={analytic[-1]:.4f}")
print()
print("=== boundary gradient values (i=0 and i=Nx-1) vs analytic [0.0, 2.0] ===")
print(f"{'variant':22s}  left(i=0)   right(i=Nx-1)   |err|_left   |err|_right")


def row(name, g):
    el = abs(g[0] - analytic[0])
    er = abs(g[-1] - analytic[-1])
    print(f"{name:22s}  {g[0]:9.4f}   {g[-1]:9.4f}     {el:9.4f}   {er:9.4f}")


row("legacy %Nx central", leg_c)
row("legacy %Nx upwind", leg_u)
for k, v in variants.items():
    row(k, v)

print()
print("=== interior byte-identity vs legacy %Nx central (indices 1..Nx-2) ===")
for k, v in variants.items():
    intr = np.allclose(v[1:-1], leg_c[1:-1], rtol=0, atol=0)
    maxd = np.max(np.abs(v[1:-1] - leg_c[1:-1]))
    print(f"{k:22s}  interior_byte_identical={intr}   max_interior_diff={maxd:.2e}")

print()
print("=== full-array byte-identity vs legacy %Nx central (all indices) ===")
for k, v in variants.items():
    full = np.array_equal(v, leg_c)
    maxd = np.max(np.abs(v - leg_c))
    print(f"{k:22s}  full_byte_identical={full}   max_full_diff={maxd:.2e}")
