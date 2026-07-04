"""Issue #1485: the Gauss-quadrature assembly path fails loud on a near-singular MLS moment matrix
(garbage shape functions np.linalg.solve does not flag), while the SCNI path stays exempt (its nodal
smoothing tolerates poor pointwise conditioning) — the guard lives on the caller flag, not the shared basis."""

from __future__ import annotations

import pytest

import numpy as np

from mfgarchon.alg.numerical.meshless_galerkin.mls_basis import monomial_exponents, shape_functions_and_grads


def test_well_conditioned_cloud_does_not_fire():
    nodes = np.linspace(0.0, 1.0, 11).reshape(-1, 1)
    qp = np.linspace(0.05, 0.95, 20).reshape(-1, 1)
    shape_functions_and_grads(qp, nodes, 0.3, monomial_exponents(1, 2), "numpy", check_conditioning=True)  # no raise


def test_degenerate_support_fires_on_gauss_path():
    bad_nodes = np.array([[0.0], [0.001], [0.002]])  # ~collinear/coincident -> rank-deficient P
    qp = np.array([[0.5]])
    with pytest.raises(np.linalg.LinAlgError, match="ill-conditioned"):
        shape_functions_and_grads(qp, bad_nodes, 1.0, monomial_exponents(1, 2), "numpy", check_conditioning=True)


def test_scni_path_is_exempt_on_same_degenerate_cloud():
    bad_nodes = np.array([[0.0], [0.001], [0.002]])
    qp = np.array([[0.5]])
    # SCNI calls without check_conditioning (default False) -> must NOT raise
    shape_functions_and_grads(qp, bad_nodes, 1.0, monomial_exponents(1, 2), "numpy")
