"""Issue #1145 (Bug A): ``MeshlessGalerkinDiscretization.gradient_projection`` must
return the WEAK-form derivative ``R_d[i,j] = int phi_i (d phi_j / d x_d) dx`` mandated
by the ``WeakFormDiscretization`` protocol -- NOT the strong pointwise derivative
``d phi_j / d x_d (x_i)``.

The solver recovers the nodal gradient via the mass-lumped projection
``G_d = M_lumped^{-1} R_d`` (``weak_form_hjb_solver.py``, ``meshless_galerkin/fp_solver.py``).
Returning the strong form made that ``M_lumped^{-1}`` a spurious second factor, scaling
the recovered gradient by ~``1/dx`` (20x at h=1/20) -- which inflated the FP advection
velocity and blew up the coupled MFG solve.
"""

from __future__ import annotations

import pytest

import numpy as np

from mfgarchon.alg.numerical.meshless_galerkin.discretization import MeshlessGalerkinDiscretization
from mfgarchon.alg.numerical.meshless_galerkin.quadrature import tensor_gauss


def _disc(d: int, n_per: int):
    ax = np.linspace(0.0, 1.0, n_per)
    mesh = np.meshgrid(*([ax] * d), indexing="ij")
    nodes = np.stack([m.ravel() for m in mesh], axis=1)
    h = 1.0 / (n_per - 1)
    rho = 3.5 * h if d == 1 else 2.6 * h
    pts, wts = tensor_gauss([(0.0, 1.0)] * d, n_cells=n_per - 1, n_gauss=4)
    return MeshlessGalerkinDiscretization(nodes, rho, 2, pts, wts, backend="numpy"), nodes


def _nodal_gradient_operators(disc):
    """Mirror the solver: G_d = M_lumped^{-1} R_d (weak_form_hjb_solver.py:90)."""
    m_lumped_inv = 1.0 / disc.mass().toarray().sum(axis=1)
    return [m_lumped_inv[:, None] * r.toarray() for r in disc.gradient_projection()]


@pytest.mark.parametrize(("d", "n_per"), [(1, 11), (2, 7)])
def test_nodal_gradient_reproduces_linear_field(d, n_per):
    """M_lumped^{-1} R_d applied to u = x_e reproduces delta_{ec} (linear-exact)."""
    disc, nodes = _disc(d, n_per)
    grad_ops = _nodal_gradient_operators(disc)
    for e in range(d):
        for c in range(d):
            grad = grad_ops[e] @ nodes[:, c]
            err = np.max(np.abs(grad - (1.0 if e == c else 0.0)))
            assert err < 1e-9, f"d/dx_{e} of x_{c}: max err {err:.2e}"


def test_gradient_projection_is_weak_form_not_strong():
    """Regression for #1145. The raw operator R_0 is the WEAK form, so
    R_0 @ (x) = int phi_i (d x / dx) = int phi_i = the consistent-mass row sums
    (which are O(h)), NOT the strong-form delta = 1 it would be if R_0 returned the
    pointwise nodal derivative."""
    disc, nodes = _disc(1, 11)
    r0 = disc.gradient_projection()[0].toarray()
    m_lumped = disc.mass().toarray().sum(axis=1)
    raw = r0 @ nodes[:, 0]  # int phi_i d/dx (x) = int phi_i = M_lumped_i
    np.testing.assert_allclose(raw, m_lumped, rtol=1e-9, atol=1e-12)
    # ... which is O(h), nowhere near the strong-form 1.0 (the #1145 bug).
    assert np.max(np.abs(raw)) < 0.5
