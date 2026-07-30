"""Issue #1485: the Gauss-quadrature assembly path fails loud on a near-singular MLS moment matrix
(garbage shape functions np.linalg.solve does not flag), while the SCNI path stays exempt (its nodal
smoothing tolerates poor pointwise conditioning) — the guard lives on the caller flag, not the shared basis."""

from __future__ import annotations

import pytest

import numpy as np

from mfgarchon.alg.numerical.meshless_galerkin.mls_basis import monomial_exponents, shape_functions_and_grads


def _grid_2d(n):
    axis = np.linspace(0.0, 1.0, n)
    return np.stack([coordinate.ravel() for coordinate in np.meshgrid(axis, axis, indexing="ij")], axis=1)


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


@pytest.mark.parametrize("degree", [2, 4])
def test_shifted_scaled_moments_are_translation_and_scale_invariant(degree):
    nodes = _grid_2d(9)
    evaluation_points = np.array([[0.18, 0.27], [0.51, 0.62], [0.83, 0.74]])
    rho = 0.65
    exponents = monomial_exponents(2, degree)
    phi, gradient = shape_functions_and_grads(
        evaluation_points,
        nodes,
        rho,
        exponents,
        "numpy",
        check_conditioning=True,
    )

    scale = 3.7
    translation = np.array([1.0e6, -2.0e6])
    transformed_phi, transformed_gradient = shape_functions_and_grads(
        scale * evaluation_points + translation,
        scale * nodes + translation,
        scale * rho,
        exponents,
        "numpy",
        check_conditioning=True,
    )

    assert np.allclose(transformed_phi, phi, rtol=2e-9, atol=2e-10)
    assert np.allclose(scale * transformed_gradient, gradient, rtol=2e-8, atol=2e-9)


def test_degree_four_values_and_gradients_reproduce_a_boundary_polynomial():
    nodes = _grid_2d(11)
    evaluation_points = np.array([[0.0, 0.0], [0.05, 0.0], [1.0, 0.5], [0.37, 0.41]])
    phi, gradient = shape_functions_and_grads(
        evaluation_points,
        nodes,
        0.55,
        monomial_exponents(2, 4),
        "numpy",
        check_conditioning=True,
    )
    x_nodes, y_nodes = nodes.T
    nodal_values = x_nodes**4 - 0.7 * x_nodes**2 * y_nodes**2 + 0.2 * y_nodes**3
    x_eval, y_eval = evaluation_points.T
    exact_values = x_eval**4 - 0.7 * x_eval**2 * y_eval**2 + 0.2 * y_eval**3
    exact_gradient = np.column_stack(
        [
            4.0 * x_eval**3 - 1.4 * x_eval * y_eval**2,
            -1.4 * x_eval**2 * y_eval + 0.6 * y_eval**2,
        ]
    )

    assert np.allclose(phi @ nodal_values, exact_values, rtol=1e-10, atol=2e-12)
    reconstructed_gradient = np.column_stack([gradient[:, :, direction] @ nodal_values for direction in range(2)])
    assert np.allclose(reconstructed_gradient, exact_gradient, rtol=1e-9, atol=2e-11)


def test_empty_evaluation_batch_preserves_output_contract():
    phi, gradient = shape_functions_and_grads(
        np.empty((0, 2)),
        _grid_2d(5),
        0.75,
        monomial_exponents(2, 2),
        "numpy",
        check_conditioning=True,
    )

    assert phi.shape == (0, 25)
    assert gradient.shape == (0, 25, 2)


def test_numpy_and_jax_use_the_same_shifted_scaled_moments():
    pytest.importorskip("jax")
    nodes = _grid_2d(7)
    evaluation_points = np.array([[0.22, 0.31], [0.57, 0.68]])
    exponents = monomial_exponents(2, 3)
    numpy_phi, numpy_gradient = shape_functions_and_grads(
        evaluation_points,
        nodes,
        0.75,
        exponents,
        "numpy",
        check_conditioning=True,
    )
    jax_phi, jax_gradient = shape_functions_and_grads(
        evaluation_points,
        nodes,
        0.75,
        exponents,
        "jax",
    )

    assert np.allclose(jax_phi, numpy_phi, rtol=2e-10, atol=2e-12)
    assert np.allclose(jax_gradient, numpy_gradient, rtol=2e-9, atol=2e-11)
