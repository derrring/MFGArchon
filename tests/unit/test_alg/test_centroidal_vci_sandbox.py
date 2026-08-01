"""Admission tests for the experimental centroidal VCI-SCNI operator sandbox."""

from __future__ import annotations

from functools import cache

import pytest

import numpy as np
from scipy import linalg

from mfgarchon.alg.numerical.meshless_galerkin.centroidal_vci_sandbox import (
    BoundaryCompleteCVTCloud,
    CentroidalVCISCNIOperatorSandbox,
    PairedMFGOperatorSandbox,
    _polynomial_data,
    boundary_complete_cvt_rectangle,
)
from mfgarchon.alg.numerical.meshless_galerkin.discretization import MeshlessGalerkinDiscretization
from mfgarchon.alg.numerical.meshless_galerkin.mls_basis import (
    monomial_exponents,
    shape_functions_and_grads,
)
from mfgarchon.alg.numerical.meshless_galerkin.quadrature import tensor_gauss
from mfgarchon.alg.numerical.meshless_galerkin.scni_discretization import MeshlessSCNIDiscretization
from mfgarchon.alg.numerical.meshless_galerkin.voronoi_cells import clipped_voronoi_cells

BOUNDS = [(0.0, 1.0), (0.0, 1.0)]


def _smooth_elliptic_coefficient(points: np.ndarray) -> np.ndarray:
    return 1.0 + 0.2 * np.sin(2.0 * np.pi * points[:, 0]) * np.cos(2.0 * np.pi * points[:, 1])


def _smooth_elliptic_coefficient_gradient(points: np.ndarray) -> np.ndarray:
    return np.column_stack(
        [
            0.4 * np.pi * np.cos(2.0 * np.pi * points[:, 0]) * np.cos(2.0 * np.pi * points[:, 1]),
            -0.4 * np.pi * np.sin(2.0 * np.pi * points[:, 0]) * np.sin(2.0 * np.pi * points[:, 1]),
        ]
    )


def _grid(n: int) -> np.ndarray:
    axis = np.linspace(0.0, 1.0, n)
    return np.stack([coordinate.ravel() for coordinate in np.meshgrid(axis, axis, indexing="ij")], axis=1)


def _jittered_grid(n: int, seed: int) -> np.ndarray:
    nodes = _grid(n)
    interior = (nodes[:, 0] > 0.0) & (nodes[:, 0] < 1.0) & (nodes[:, 1] > 0.0) & (nodes[:, 1] < 1.0)
    perturbation = np.random.default_rng(seed).uniform(-1.0, 1.0, size=(int(interior.sum()), 2))
    nodes[interior] += 0.15 / (n - 1) * perturbation
    return nodes


@cache
def _boundary_complete_case(
    resolution: int,
    seed: int,
) -> tuple[BoundaryCompleteCVTCloud, CentroidalVCISCNIOperatorSandbox]:
    cloud = boundary_complete_cvt_rectangle(resolution, BOUNDS, seed=seed)
    sandbox = CentroidalVCISCNIOperatorSandbox(
        cloud.nodes,
        rho=3.0 * cloud.nominal_spacing,
        bounds=BOUNDS,
    )
    return cloud, sandbox


@cache
def _boundary_complete_vc4_case(
    resolution: int,
    seed: int,
) -> tuple[BoundaryCompleteCVTCloud, CentroidalVCISCNIOperatorSandbox]:
    cloud = boundary_complete_cvt_rectangle(resolution, BOUNDS, seed=seed)
    sandbox = CentroidalVCISCNIOperatorSandbox(
        cloud.nodes,
        rho=3.0 * cloud.nominal_spacing,
        bounds=BOUNDS,
        vci_degree=4,
        vci_support_radius=4.0 * cloud.nominal_spacing,
    )
    return cloud, sandbox


@cache
def _boundary_complete_stabilized_vc4_case(
    resolution: int,
    seed: int,
) -> tuple[BoundaryCompleteCVTCloud, CentroidalVCISCNIOperatorSandbox]:
    cloud = boundary_complete_cvt_rectangle(resolution, BOUNDS, seed=seed)
    sandbox = CentroidalVCISCNIOperatorSandbox(
        cloud.nodes,
        rho=4.5 * cloud.nominal_spacing,
        bounds=BOUNDS,
        degree=4,
        vci_degree=4,
        vci_support_radius=4.0 * cloud.nominal_spacing,
        trial_gradient_mode="pointwise",
        test_gradient_base="trial",
        polynomial_null_stabilization=0.1,
        stabilization_support_radius=4.5 * cloud.nominal_spacing,
    )
    return cloud, sandbox


@cache
def _boundary_complete_edge_stabilized_vc4_case(
    resolution: int,
    seed: int,
) -> tuple[BoundaryCompleteCVTCloud, CentroidalVCISCNIOperatorSandbox]:
    cloud = boundary_complete_cvt_rectangle(resolution, BOUNDS, seed=seed)
    sandbox = CentroidalVCISCNIOperatorSandbox(
        cloud.nodes,
        rho=4.5 * cloud.nominal_spacing,
        bounds=BOUNDS,
        degree=4,
        vci_degree=4,
        vci_support_radius=4.0 * cloud.nominal_spacing,
        trial_gradient_mode="pointwise",
        test_gradient_base="trial",
        polynomial_null_stabilization=0.1,
        stabilization_metric="edge_energy",
    )
    return cloud, sandbox


@cache
def _boundary_complete_variable_edge_stabilized_vc4_case(
    resolution: int,
    seed: int,
) -> tuple[BoundaryCompleteCVTCloud, CentroidalVCISCNIOperatorSandbox]:
    cloud = boundary_complete_cvt_rectangle(resolution, BOUNDS, seed=seed)
    sandbox = CentroidalVCISCNIOperatorSandbox(
        cloud.nodes,
        rho=4.5 * cloud.nominal_spacing,
        bounds=BOUNDS,
        degree=4,
        vci_degree=4,
        vci_support_radius=4.0 * cloud.nominal_spacing,
        trial_gradient_mode="pointwise",
        test_gradient_base="trial",
        polynomial_null_stabilization=0.1,
        stabilization_metric="edge_energy",
        elliptic_coefficient=_smooth_elliptic_coefficient,
        elliptic_coefficient_gradient=_smooth_elliptic_coefficient_gradient,
    )
    return cloud, sandbox


@cache
def _degree_four_reference_stiffness(resolution: int, seed: int) -> np.ndarray:
    cloud, _candidate = _boundary_complete_edge_stabilized_vc4_case(resolution, seed)
    points, weights = tensor_gauss(BOUNDS, n_cells=resolution - 1, n_gauss=4)
    reference = MeshlessGalerkinDiscretization(
        cloud.nodes,
        rho=4.5 * cloud.nominal_spacing,
        degree=4,
        quad_points=points,
        quad_weights=weights,
    )
    return reference.stiffness().toarray()


@cache
def _degree_four_variable_reference_stiffness(resolution: int, seed: int) -> np.ndarray:
    cloud, _candidate = _boundary_complete_variable_edge_stabilized_vc4_case(resolution, seed)
    points, weights = tensor_gauss(BOUNDS, n_cells=resolution - 1, n_gauss=4)
    _, gradients = shape_functions_and_grads(
        points,
        cloud.nodes,
        4.5 * cloud.nominal_spacing,
        monomial_exponents(2, 4),
        "numpy",
        check_conditioning=True,
    )
    return np.einsum(
        "q,q,qid,qjd->ij",
        weights,
        _smooth_elliptic_coefficient(points),
        gradients,
        gradients,
    )


@cache
def _edge_stabilization_dual_data(resolution: int, seed: int):
    _cloud, sandbox = _boundary_complete_edge_stabilized_vc4_case(resolution, seed)
    one = np.ones(sandbox.n_dof)
    mass = sandbox.mass().toarray()
    gauge_basis = linalg.null_space((mass @ one)[None, :])
    reference = gauge_basis.T @ _degree_four_reference_stiffness(resolution, seed) @ gauge_basis
    return sandbox.stabilization().toarray(), gauge_basis, linalg.cho_factor(reference, check_finite=False)


@cache
def _degree_four_evaluation_matched_common_case(resolution: int, seed: int):
    cloud, candidate = _boundary_complete_stabilized_vc4_case(resolution, seed)
    evaluation_budget = candidate.centroid_evaluation_count + candidate.edge_evaluation_count
    common_cells = max(1, int(np.floor(np.sqrt(evaluation_budget) / 4)))
    common_points, common_weights = tensor_gauss(BOUNDS, n_cells=common_cells, n_gauss=4)
    common = MeshlessGalerkinDiscretization(
        cloud.nodes,
        rho=4.5 * cloud.nominal_spacing,
        degree=4,
        quad_points=common_points,
        quad_weights=common_weights,
    )
    return candidate, common.stiffness().toarray(), common.mass().toarray(), len(common_points), evaluation_budget


def _solve_neumann_system(K, M, load, left_null):
    one = np.ones(len(K))
    mean_constraint = M @ one
    saddle = np.block(
        [
            [K, left_null[:, None]],
            [mean_constraint[None, :], np.zeros((1, 1))],
        ]
    )
    augmented_load = np.append(load, 0.0)
    augmented_solution = np.linalg.solve(saddle, augmented_load)
    relative_residual = np.linalg.norm(saddle @ augmented_solution - augmented_load) / np.linalg.norm(load)
    gauge_defect = float(abs(mean_constraint @ augmented_solution[:-1]))
    return augmented_solution[:-1], relative_residual, gauge_defect


def _physical_poisson_errors(
    reconstructed_values,
    reconstructed_gradient,
    exact_values,
    exact_gradient,
    weights,
):
    value_error = reconstructed_values - exact_values
    value_error -= float(weights @ value_error / np.sum(weights))
    l2_error = float(np.sqrt(weights @ value_error**2))
    h1_error = float(np.sqrt(weights @ np.sum((reconstructed_gradient - exact_gradient) ** 2, axis=1)))
    return l2_error, h1_error


@cache
def _matched_cost_arms(resolution: int, seed: int):
    cloud, sandbox = _boundary_complete_case(resolution, seed)
    rho = 3.0 * cloud.nominal_spacing
    E = sandbox.value_operator().toarray()
    W = sandbox.weights
    trial_gradients = [gradient.toarray() for gradient in sandbox.trial_gradient()]
    M = sandbox.mass().toarray()

    plain_scni = MeshlessSCNIDiscretization(
        cloud.nodes,
        rho=rho,
        degree=2,
        bounds=BOUNDS,
        n_edge_gauss=4,
    )
    # Halving every oriented edge count also pairs physical-boundary edges, so this
    # undercounts an optimized VCI implementation and biases the MLS-evaluation
    # budget in VCI's favor. Local moment-solve cost is not included.
    optimistic_vci_evaluation_budget = sandbox.centroid_evaluation_count + (sandbox.edge_evaluation_count + 1) // 2
    common_cells = max(1, int(np.floor(np.sqrt(optimistic_vci_evaluation_budget) / 4)))
    common_points, common_weights = tensor_gauss(BOUNDS, n_cells=common_cells, n_gauss=4)
    common = MeshlessGalerkinDiscretization(
        cloud.nodes,
        rho,
        2,
        common_points,
        common_weights,
    )
    _, pointwise_gradients = shape_functions_and_grads(
        sandbox.evaluation_points,
        cloud.nodes,
        rho,
        monomial_exponents(2, 2),
        "numpy",
        check_conditioning=True,
    )
    arms = {
        "plain_scni": (
            plain_scni.stiffness().toarray(),
            plain_scni.mass().toarray(),
            trial_gradients,
        ),
        "centroidal_scni": (
            sum(gradient.T @ (W[:, None] * gradient) for gradient in trial_gradients),
            M,
            trial_gradients,
        ),
        "vc2": (
            sandbox.stiffness().toarray(),
            M,
            trial_gradients,
        ),
        "common_quadrature": (
            common.stiffness().toarray(),
            common.mass().toarray(),
            [pointwise_gradients[:, :, direction] for direction in range(2)],
        ),
    }
    return cloud, sandbox, E, W, arms, len(common_points), optimistic_vci_evaluation_budget


@pytest.fixture(scope="module")
def structured_sandbox() -> CentroidalVCISCNIOperatorSandbox:
    n = 7
    return CentroidalVCISCNIOperatorSandbox(_grid(n), rho=3.0 / (n - 1), bounds=BOUNDS)


@pytest.fixture(scope="module")
def jittered_sandbox() -> CentroidalVCISCNIOperatorSandbox:
    n = 7
    return CentroidalVCISCNIOperatorSandbox(
        _jittered_grid(n, seed=0),
        rho=3.2 / (n - 1),
        bounds=BOUNDS,
    )


def _hamiltonian(_points: np.ndarray, momentum: np.ndarray) -> np.ndarray:
    return 0.5 * np.sum(momentum**2, axis=1) + 0.1 * np.sum(momentum**4, axis=1)


def _hamiltonian_gradient(_points: np.ndarray, momentum: np.ndarray) -> np.ndarray:
    return momentum + 0.4 * momentum**3


def _stabilization_flux(_points: np.ndarray, momentum: np.ndarray) -> np.ndarray:
    squared_norm = np.sum(momentum**2, axis=1)
    return 0.03 * squared_norm[:, None] * momentum


def _stabilization_jacobian(_points: np.ndarray, momentum: np.ndarray) -> np.ndarray:
    count, dim = momentum.shape
    squared_norm = np.sum(momentum**2, axis=1)
    jacobian = 0.06 * np.einsum("qd,qe->qde", momentum, momentum)
    jacobian += 0.03 * squared_norm[:, None, None] * np.eye(dim)[None, :, :]
    return jacobian.reshape(count, dim, dim)


class TestCentroidalVCGeometry:
    def test_degree_four_polynomial_derivatives_are_independently_pinned(self):
        points = np.array([[0.2, -0.1], [0.8, 0.7]])
        center = np.array([0.1, -0.3])
        scales = np.array([0.5, 2.0])
        values, gradients, laplacians = _polynomial_data(points, center, scales, degree=4)
        sx = (points[:, 0] - center[0]) / scales[0]
        sy = (points[:, 1] - center[1]) / scales[1]

        assert values.shape == (2, 15)
        assert np.allclose(values[:, 11], sx**3 * sy)
        assert np.allclose(gradients[:, 11, 0], 3.0 * sx**2 * sy / scales[0])
        assert np.allclose(gradients[:, 11, 1], sx**3 / scales[1])
        assert np.allclose(laplacians[:, 11], 6.0 * sx * sy / scales[0] ** 2)
        assert np.allclose(values[:, 12], sx**2 * sy**2)
        assert np.allclose(gradients[:, 12, 0], 2.0 * sx * sy**2 / scales[0])
        assert np.allclose(gradients[:, 12, 1], 2.0 * sx**2 * sy / scales[1])
        assert np.allclose(
            laplacians[:, 12],
            2.0 * sy**2 / scales[0] ** 2 + 2.0 * sx**2 / scales[1] ** 2,
        )

    def test_degree_three_vci_branch_is_admitted(self):
        n = 7
        sandbox = CentroidalVCISCNIOperatorSandbox(
            _grid(n),
            rho=3.0 / (n - 1),
            bounds=BOUNDS,
            vci_degree=3,
            vci_support_radius=4.0 / (n - 1),
        )
        diagnostics = sandbox.diagnostics
        assert diagnostics.local_rank_min == 9
        assert diagnostics.corrected_patch_defect < 1e-12
        assert diagnostics.stiffness_nullity == 1

    def test_polygon_centroids_are_exact_on_a_cartesian_tiling(self):
        cells = clipped_voronoi_cells(_grid(3), BOUNDS)
        assert np.allclose(cells[0].centroid, [0.125, 0.125], atol=1e-14)
        assert np.allclose(cells[4].centroid, [0.5, 0.5], atol=1e-14)
        assert all(cell.area > 0.0 for cell in cells)

    def test_structured_cloud_passes_polynomial_and_stability_gates(self, structured_sandbox):
        diagnostics = structured_sandbox.diagnostics
        assert diagnostics.local_rank_min == 5
        assert diagnostics.quadratic_gradient_defect < 1e-10
        assert diagnostics.plain_patch_defect > 1e-4
        assert diagnostics.corrected_patch_defect < 1e-12
        assert diagnostics.correction_ratio_max < 0.2
        assert diagnostics.mass_condition < 1e3
        assert diagnostics.gauge_coercivity_min > 1.0
        assert diagnostics.gauge_smallest_singular_value > 1.0
        assert diagnostics.stiffness_nullity == 1

    def test_jittered_cloud_preserves_the_full_gate(self, jittered_sandbox):
        diagnostics = jittered_sandbox.diagnostics
        assert diagnostics.local_rank_min == 5
        assert diagnostics.local_condition_max < 20.0
        assert diagnostics.quadratic_gradient_defect < 1e-9
        assert diagnostics.corrected_patch_defect < 1e-12
        assert diagnostics.correction_ratio_max < 0.2
        assert diagnostics.gauge_coercivity_min > 1.0
        assert diagnostics.stiffness_nullity == 1

    def test_sdf_chord_geometry_includes_boundary_flux_in_the_patch(self):
        center = np.array([0.5, 0.5])
        radius = 0.4
        n = 11
        axis = np.linspace(0.1, 0.9, n)
        grid = np.stack(
            [coordinate.ravel() for coordinate in np.meshgrid(axis, axis, indexing="ij")],
            axis=1,
        )
        sdf = lambda points: np.linalg.norm(np.atleast_2d(points) - center, axis=1) - radius  # noqa: E731
        nodes = grid[sdf(grid) <= 1e-14]
        sandbox = CentroidalVCISCNIOperatorSandbox(
            nodes,
            rho=3.0 * 0.8 / (n - 1),
            bounds=[(0.1, 0.9), (0.1, 0.9)],
            sdf=sdf,
        )
        assert sandbox.diagnostics.quadratic_gradient_defect < 1e-9
        assert sandbox.diagnostics.corrected_patch_defect < 1e-12
        assert sandbox.diagnostics.stiffness_nullity == 1

    def test_boundary_complete_cvt_is_deterministic_and_keeps_fitted_boundary(self):
        first = boundary_complete_cvt_rectangle(7, BOUNDS, seed=4, iterations=10)
        second = boundary_complete_cvt_rectangle(7, BOUNDS, seed=4, iterations=10)
        assert np.array_equal(first.nodes, second.nodes)
        assert np.array_equal(first.boundary_mask, second.boundary_mask)
        assert len(first.nodes) == 7**2
        assert np.count_nonzero(first.boundary_mask) == 4 * (7 - 1)
        boundary = first.nodes[first.boundary_mask]
        on_box = (boundary[:, 0] == 0.0) | (boundary[:, 0] == 1.0) | (boundary[:, 1] == 0.0) | (boundary[:, 1] == 1.0)
        assert np.all(on_box)
        assert not first.nodes.flags.writeable
        assert not first.boundary_mask.flags.writeable

    @pytest.mark.parametrize(
        ("options", "message"),
        [
            ({"resolution": 4}, "resolution >= 5"),
            ({"resolution": 7, "iterations": -1}, "iterations >= 0"),
            ({"resolution": 7, "jitter_fraction": 0.5}, "jitter_fraction"),
            ({"resolution": 7, "relaxation": 0.0}, "relaxation"),
            ({"resolution": 7, "bounds": [(0.0, 1.0)]}, r"shape \(2, 2\)"),
            ({"resolution": 7, "bounds": [(0.0, 1.0), (1.0, 0.0)]}, "finite increasing bounds"),
        ],
    )
    def test_boundary_complete_cvt_rejects_invalid_family_parameters(self, options, message):
        arguments = {"bounds": BOUNDS, **options}
        with pytest.raises(ValueError, match=message):
            boundary_complete_cvt_rectangle(**arguments)

    @pytest.mark.parametrize(
        ("resolution", "seed"),
        [(resolution, seed) for resolution in (7, 9, 11, 13, 17, 21) for seed in range(3)],
    )
    def test_boundary_complete_cvt_family_passes_refinement_gate(self, resolution, seed):
        cloud, sandbox = _boundary_complete_case(resolution, seed)
        diagnostics = sandbox.diagnostics
        assert cloud.max_interior_centroid_offset_ratio < 0.025
        assert cloud.min_separation_ratio > 0.8
        assert cloud.min_cell_area_ratio > 0.25
        assert cloud.max_cell_area_ratio < 1.6
        assert cloud.max_cell_diameter_ratio < 1.5
        assert diagnostics.local_rank_min == 5
        assert diagnostics.local_condition_max < 15.0
        assert diagnostics.quadratic_gradient_defect < 1e-8
        assert diagnostics.corrected_patch_defect < 1e-12
        assert diagnostics.correction_ratio_max < 0.2
        assert diagnostics.mass_condition < 250.0
        assert diagnostics.gauge_coercivity_min > 9.0
        assert diagnostics.gauge_smallest_singular_value > 9.0
        assert diagnostics.stiffness_nullity == 1

    @pytest.mark.parametrize(
        ("wave_numbers", "minimum_l2_rate", "minimum_h1_rate"),
        [
            ((1, 1), 1.95, 2.2),
            ((2, 1), 1.75, 2.15),
        ],
    )
    @pytest.mark.parametrize("seed", range(3))
    def test_boundary_complete_cvt_poisson_neumann_converges(
        self,
        wave_numbers,
        minimum_l2_rate,
        minimum_h1_rate,
        seed,
    ):
        records = []
        kx, ky = wave_numbers
        for resolution in (7, 9, 11, 13, 17, 21):
            cloud, sandbox = _boundary_complete_case(resolution, seed)
            points = sandbox.evaluation_points
            exact_values = np.cos(kx * np.pi * points[:, 0]) * np.cos(ky * np.pi * points[:, 1])
            forcing = (kx**2 + ky**2) * np.pi**2 * exact_values
            exact_gradient = np.column_stack(
                [
                    -kx * np.pi * np.sin(kx * np.pi * points[:, 0]) * np.cos(ky * np.pi * points[:, 1]),
                    -ky * np.pi * np.cos(kx * np.pi * points[:, 0]) * np.sin(ky * np.pi * points[:, 1]),
                ]
            )

            E = sandbox.value_operator().toarray()
            M = sandbox.mass().toarray()
            K = sandbox.stiffness().toarray()
            left_null = sandbox.left_null_vector()
            load = E.T @ (sandbox.weights * forcing)
            solution, relative_saddle_residual, gauge_defect = _solve_neumann_system(K, M, load, left_null)
            assert relative_saddle_residual < 1e-12
            assert gauge_defect < 1e-12

            reconstructed_values = E @ solution
            reconstructed_gradient = np.column_stack([gradient @ solution for gradient in sandbox.trial_gradient()])
            l2_error, h1_error = _physical_poisson_errors(
                reconstructed_values,
                reconstructed_gradient,
                exact_values,
                exact_gradient,
                sandbox.weights,
            )
            compatibility_ratio = float(abs(left_null @ load) / (np.linalg.norm(left_null) * np.linalg.norm(load)))
            records.append((cloud.nominal_spacing, l2_error, h1_error, compatibility_ratio))

        spacings, l2_errors, h1_errors, compatibility_ratios = map(np.asarray, zip(*records, strict=True))
        assert np.all(np.diff(l2_errors) < 0.0)
        assert np.all(np.diff(h1_errors) < 0.0)
        assert l2_errors[-1] < 0.12 * l2_errors[0]
        assert h1_errors[-1] < 0.08 * h1_errors[0]
        assert np.max(compatibility_ratios) < 4e-3
        assert compatibility_ratios[-1] < 2e-4
        l2_rate = float(np.polyfit(np.log(spacings), np.log(l2_errors), 1)[0])
        h1_rate = float(np.polyfit(np.log(spacings), np.log(h1_errors), 1)[0])
        assert l2_rate > minimum_l2_rate
        assert h1_rate > minimum_h1_rate

    @pytest.mark.parametrize("wave_numbers", [(1, 1), (2, 1)])
    @pytest.mark.parametrize("seed", range(3))
    def test_matched_cost_poisson_records_vc2_gain_and_common_quadrature_gap(self, wave_numbers, seed):
        records = {
            "plain_scni": [],
            "centroidal_scni": [],
            "vc2": [],
            "common_quadrature": [],
        }
        kx, ky = wave_numbers
        for resolution in (7, 9, 11, 13, 17, 21):
            cloud, sandbox, E, W, arms, common_count, optimistic_vci_budget = _matched_cost_arms(resolution, seed)
            assert 0.75 * optimistic_vci_budget < common_count <= optimistic_vci_budget
            points = sandbox.evaluation_points
            exact_values = np.cos(kx * np.pi * points[:, 0]) * np.cos(ky * np.pi * points[:, 1])
            exact_gradient = np.column_stack(
                [
                    -kx * np.pi * np.sin(kx * np.pi * points[:, 0]) * np.cos(ky * np.pi * points[:, 1]),
                    -ky * np.pi * np.cos(kx * np.pi * points[:, 0]) * np.sin(ky * np.pi * points[:, 1]),
                ]
            )
            forcing_nodes = (
                (kx**2 + ky**2)
                * np.pi**2
                * np.cos(kx * np.pi * cloud.nodes[:, 0])
                * np.cos(ky * np.pi * cloud.nodes[:, 1])
            )
            for name, (K, M, gradients) in arms.items():
                one = np.ones(sandbox.n_dof)
                left_null = sandbox.left_null_vector() if name == "vc2" else one / len(one)
                solution, relative_residual, gauge_defect = _solve_neumann_system(
                    K,
                    M,
                    M @ forcing_nodes,
                    left_null,
                )
                assert relative_residual < 1e-12
                assert gauge_defect < 1e-12
                reconstructed_gradient = np.column_stack([gradient @ solution for gradient in gradients])
                l2_error, h1_error = _physical_poisson_errors(
                    E @ solution,
                    reconstructed_gradient,
                    exact_values,
                    exact_gradient,
                    W,
                )
                records[name].append((cloud.nominal_spacing, l2_error, h1_error))

        rates = {}
        for name, arm_records in records.items():
            spacings, l2_errors, h1_errors = map(np.asarray, zip(*arm_records, strict=True))
            assert np.all(np.diff(l2_errors) < 0.0)
            assert np.all(np.diff(h1_errors) < 0.0)
            rates[name] = (
                float(np.polyfit(np.log(spacings), np.log(l2_errors), 1)[0]),
                float(np.polyfit(np.log(spacings), np.log(h1_errors), 1)[0]),
            )

        assert rates["plain_scni"][1] < 1.6
        assert rates["centroidal_scni"][1] < 1.8
        assert rates["vc2"][1] > 2.0
        assert rates["common_quadrature"][1] > 2.0
        plain_finest = records["plain_scni"][-1]
        centroidal_finest = records["centroidal_scni"][-1]
        vc2_finest = records["vc2"][-1]
        common_finest = records["common_quadrature"][-1]
        assert vc2_finest[2] < 0.3 * plain_finest[2]
        assert vc2_finest[2] < 0.4 * centroidal_finest[2]
        assert common_finest[2] < 0.55 * vc2_finest[2]
        assert common_finest[1] < 0.06 * vc2_finest[1]

    @pytest.mark.parametrize(
        ("resolution", "seed"),
        [(resolution, seed) for resolution in (7, 9, 11, 13, 17, 21) for seed in range(3)],
    )
    def test_vc4_family_preserves_target_moments_and_stability(self, resolution, seed):
        _cloud, sandbox = _boundary_complete_vc4_case(resolution, seed)
        diagnostics = sandbox.diagnostics
        assert sandbox.vci_degree == 4
        assert diagnostics.local_rank_min == 14
        assert diagnostics.local_condition_max < 1700.0
        assert diagnostics.correction_ratio_max < 0.7
        assert diagnostics.mass_condition < 250.0
        assert diagnostics.quadratic_gradient_defect < 1e-8
        assert diagnostics.corrected_patch_defect < 1e-11
        assert diagnostics.discrete_patch_defect < 1e-2
        assert diagnostics.gauge_coercivity_min > 7.5
        assert diagnostics.stiffness_nullity == 1

    @pytest.mark.parametrize("seed", range(3))
    def test_vc4_discrete_quartic_patch_converges_without_claiming_trial_exactness(self, seed):
        records = []
        for resolution in (7, 9, 11, 13, 17, 21):
            cloud, sandbox = _boundary_complete_vc4_case(resolution, seed)
            diagnostics = sandbox.diagnostics
            assert diagnostics.vci_trial_gradient_defect > diagnostics.discrete_patch_defect
            records.append((cloud.nominal_spacing, diagnostics.discrete_patch_defect))
        spacings, defects = map(np.asarray, zip(*records, strict=True))
        assert np.all(np.diff(defects) < 0.0)
        rate = float(np.polyfit(np.log(spacings), np.log(defects), 1)[0])
        assert rate > 2.6

    @pytest.mark.parametrize("wave_numbers", [(1, 1), (2, 1), (2, 2), (3, 1)])
    @pytest.mark.parametrize("seed", range(3))
    def test_vc4_beats_mls_evaluation_budget_matched_common_in_h1_but_not_l2(self, wave_numbers, seed):
        vc4_records = []
        common_records = []
        kx, ky = wave_numbers
        for resolution in (7, 9, 11, 13, 17, 21):
            cloud, vc4 = _boundary_complete_vc4_case(resolution, seed)
            _, _, E, W, arms, common_count, optimistic_vci_budget = _matched_cost_arms(resolution, seed)
            assert 0.75 * optimistic_vci_budget < common_count <= optimistic_vci_budget
            points = vc4.evaluation_points
            exact_values = np.cos(kx * np.pi * points[:, 0]) * np.cos(ky * np.pi * points[:, 1])
            exact_gradient = np.column_stack(
                [
                    -kx * np.pi * np.sin(kx * np.pi * points[:, 0]) * np.cos(ky * np.pi * points[:, 1]),
                    -ky * np.pi * np.cos(kx * np.pi * points[:, 0]) * np.sin(ky * np.pi * points[:, 1]),
                ]
            )
            forcing_nodes = (
                (kx**2 + ky**2)
                * np.pi**2
                * np.cos(kx * np.pi * cloud.nodes[:, 0])
                * np.cos(ky * np.pi * cloud.nodes[:, 1])
            )
            vc4_K = vc4.stiffness().toarray()
            vc4_M = vc4.mass().toarray()
            vc4_solution, vc4_residual, vc4_gauge = _solve_neumann_system(
                vc4_K,
                vc4_M,
                vc4_M @ forcing_nodes,
                vc4.left_null_vector(),
            )
            assert vc4_residual < 1e-12
            assert vc4_gauge < 1e-12
            vc4_gradient = np.column_stack([gradient @ vc4_solution for gradient in vc4.trial_gradient()])
            vc4_records.append(
                (
                    cloud.nominal_spacing,
                    *_physical_poisson_errors(
                        E @ vc4_solution,
                        vc4_gradient,
                        exact_values,
                        exact_gradient,
                        W,
                    ),
                )
            )

            common_K, common_M, common_gradients = arms["common_quadrature"]
            one = np.ones(vc4.n_dof)
            common_solution, common_residual, common_gauge = _solve_neumann_system(
                common_K,
                common_M,
                common_M @ forcing_nodes,
                one / len(one),
            )
            assert common_residual < 1e-12
            assert common_gauge < 1e-12
            common_gradient = np.column_stack([gradient @ common_solution for gradient in common_gradients])
            common_records.append(
                (
                    cloud.nominal_spacing,
                    *_physical_poisson_errors(
                        E @ common_solution,
                        common_gradient,
                        exact_values,
                        exact_gradient,
                        W,
                    ),
                )
            )

        spacings, vc4_l2, vc4_h1 = map(np.asarray, zip(*vc4_records, strict=True))
        _, common_l2, common_h1 = map(np.asarray, zip(*common_records, strict=True))
        assert np.all(np.diff(vc4_l2) < 0.0)
        assert np.all(np.diff(vc4_h1) < 0.0)
        vc4_h1_rate = float(np.polyfit(np.log(spacings), np.log(vc4_h1), 1)[0])
        assert vc4_h1_rate > 2.75
        assert vc4_h1[-1] < 0.4 * common_h1[-1]
        assert common_l2[-1] < 0.35 * vc4_l2[-1]

    @pytest.mark.parametrize(
        ("resolution", "seed"),
        [(resolution, seed) for resolution in (7, 9, 11, 13, 17, 21) for seed in range(3)],
    )
    def test_stabilized_aligned_vc4_family_restores_coercivity_without_changing_p4(self, resolution, seed):
        _cloud, sandbox = _boundary_complete_stabilized_vc4_case(resolution, seed)
        diagnostics = sandbox.diagnostics
        assert sandbox.degree == 4
        assert sandbox.vci_degree == 4
        assert sandbox.trial_gradient_mode == "pointwise"
        assert sandbox.test_gradient_base == "trial"
        assert sandbox.polynomial_null_stabilization == 0.1
        assert diagnostics.local_rank_min == 14
        assert diagnostics.local_condition_max < 1700.0
        assert diagnostics.correction_ratio_max < 0.72
        assert diagnostics.mass_condition < 3600.0
        assert diagnostics.vci_trial_gradient_defect < 2e-12
        assert diagnostics.corrected_patch_defect < 1e-14
        assert diagnostics.discrete_patch_defect < 1e-12
        assert diagnostics.stabilization_rank_min == 15
        assert diagnostics.stabilization_condition_max < 4700.0
        assert diagnostics.stabilization_support_count_min >= 20
        assert diagnostics.stabilization_support_count_max <= 68
        assert diagnostics.stabilization_patch_defect < 1e-14
        assert diagnostics.raw_gauge_coercivity_min < -9.0
        assert diagnostics.gauge_coercivity_min > 7.7
        assert diagnostics.gauge_smallest_singular_value > 9.8
        assert diagnostics.stiffness_nullity == 1

    def test_local_polynomial_null_stabilization_is_symmetric_psd_and_annihilates_p4(self):
        cloud, sandbox = _boundary_complete_stabilized_vc4_case(11, 0)
        stabilization = sandbox.stabilization().toarray()
        node_values, _, _ = _polynomial_data(
            cloud.nodes,
            np.array([0.5, 0.5]),
            np.ones(2),
            degree=4,
        )
        assert np.allclose(stabilization, stabilization.T, rtol=0.0, atol=1e-14)
        assert np.linalg.eigvalsh(stabilization).min() > -1e-12
        assert np.max(np.abs(stabilization @ node_values)) < 1e-14

    @pytest.mark.parametrize(
        ("resolution", "seed"),
        [(resolution, seed) for resolution in (7, 9, 11, 13, 17, 21) for seed in range(3)],
    )
    def test_edge_energy_stabilized_vc4_family_has_physical_spectral_bounds(self, resolution, seed):
        cloud, sandbox = _boundary_complete_edge_stabilized_vc4_case(resolution, seed)
        diagnostics = sandbox.diagnostics
        spacing = cloud.nominal_spacing
        assert sandbox.stabilization_metric == "edge_energy"
        assert diagnostics.stabilization_rank_min == 15
        assert diagnostics.stabilization_condition_max < 5100.0
        assert diagnostics.stabilization_support_count_min >= 23
        assert diagnostics.stabilization_support_count_max <= 86
        assert diagnostics.stabilization_patch_defect < 5e-15
        assert diagnostics.raw_gauge_coercivity_min < -9.0
        assert diagnostics.gauge_coercivity_min > 7.6
        assert diagnostics.gauge_smallest_singular_value > 9.8
        assert diagnostics.discrete_patch_defect < 7e-14
        assert diagnostics.stiffness_nullity == 1
        assert diagnostics.value_weighted_column_sum_max / spacing**2 < 2.3
        assert diagnostics.test_gradient_weighted_column_sum_max / spacing < 6.0
        assert diagnostics.edge_value_weighted_column_sum_max / spacing < 9.5
        assert diagnostics.frame_gauge_euclidean_coercivity_min / spacing**2 > 0.35

        one = np.ones(sandbox.n_dof)
        mass = sandbox.mass().toarray()
        gauge_basis = linalg.null_space((mass @ one)[None, :])
        candidate = 0.5 * (sandbox.stiffness().toarray() + sandbox.stiffness().toarray().T)
        reference = _degree_four_reference_stiffness(resolution, seed)
        relative_spectrum = linalg.eigvalsh(
            gauge_basis.T @ candidate @ gauge_basis,
            gauge_basis.T @ reference @ gauge_basis,
            check_finite=False,
        )
        assert relative_spectrum[0] > 0.075
        assert relative_spectrum[-1] < 2.45

    def test_edge_energy_stabilization_is_symmetric_psd_and_annihilates_p4(self):
        cloud, sandbox = _boundary_complete_edge_stabilized_vc4_case(11, 0)
        stabilization = sandbox.stabilization().toarray()
        node_values, _, _ = _polynomial_data(
            cloud.nodes,
            np.array([0.5, 0.5]),
            np.ones(2),
            degree=4,
        )
        assert np.allclose(stabilization, stabilization.T, rtol=0.0, atol=1e-14)
        assert np.linalg.eigvalsh(stabilization).min() > -1e-12
        assert np.max(np.abs(stabilization @ node_values)) < 5e-15

    def test_explicit_unit_elliptic_coefficient_is_identical_to_default_path(self):
        cloud, baseline = _boundary_complete_edge_stabilized_vc4_case(7, 0)
        explicit = CentroidalVCISCNIOperatorSandbox(
            cloud.nodes,
            rho=4.5 * cloud.nominal_spacing,
            bounds=BOUNDS,
            degree=4,
            vci_degree=4,
            vci_support_radius=4.0 * cloud.nominal_spacing,
            trial_gradient_mode="pointwise",
            test_gradient_base="trial",
            polynomial_null_stabilization=0.1,
            stabilization_metric="edge_energy",
            elliptic_coefficient=lambda points: np.ones(len(points)),
            elliptic_coefficient_gradient=lambda points: np.zeros_like(points),
        )
        assert not baseline.has_spatial_elliptic_coefficient
        assert explicit.has_spatial_elliptic_coefficient
        assert explicit.sampled_elliptic_coefficient_bounds == (1.0, 1.0)
        assert np.array_equal(explicit.stiffness().toarray(), baseline.stiffness().toarray())
        assert np.array_equal(explicit.stabilization().toarray(), baseline.stabilization().toarray())
        assert all(
            np.array_equal(explicit_gradient.toarray(), baseline_gradient.toarray())
            for explicit_gradient, baseline_gradient in zip(
                explicit.test_gradient(),
                baseline.test_gradient(),
                strict=True,
            )
        )
        assert explicit.diagnostics == baseline.diagnostics

    @pytest.mark.parametrize(
        ("resolution", "seed"),
        [(resolution, seed) for resolution in (7, 9, 11, 13, 17, 21) for seed in range(3)],
    )
    def test_variable_coefficient_edge_vc4_family_preserves_patch_and_spectral_bounds(self, resolution, seed):
        cloud, sandbox = _boundary_complete_variable_edge_stabilized_vc4_case(resolution, seed)
        diagnostics = sandbox.diagnostics
        spacing = cloud.nominal_spacing
        sampled_min, sampled_max = sandbox.sampled_elliptic_coefficient_bounds
        assert sandbox.has_spatial_elliptic_coefficient
        assert 0.8 <= sampled_min < sampled_max <= 1.2
        assert sampled_max - sampled_min > 0.39
        assert diagnostics.local_rank_min == 14
        assert diagnostics.local_condition_max < 1700.0
        assert diagnostics.correction_ratio_max < 0.72
        assert diagnostics.corrected_patch_defect < 8e-15
        assert diagnostics.discrete_patch_defect < 7e-14
        assert diagnostics.stabilization_patch_defect < 1e-14
        assert diagnostics.raw_gauge_coercivity_min < -9.0
        assert diagnostics.gauge_coercivity_min > 7.55
        assert diagnostics.gauge_smallest_singular_value > 9.74
        assert diagnostics.stiffness_nullity == 1
        assert diagnostics.value_weighted_column_sum_max / spacing**2 < 2.3
        assert diagnostics.test_gradient_weighted_column_sum_max / spacing < 6.2
        assert diagnostics.edge_value_weighted_column_sum_max / spacing < 9.5
        assert diagnostics.frame_gauge_euclidean_coercivity_min / spacing**2 > 0.35

        one = np.ones(sandbox.n_dof)
        mass = sandbox.mass().toarray()
        gauge_basis = linalg.null_space((mass @ one)[None, :])
        stiffness = sandbox.stiffness().toarray()
        symmetric_stiffness = 0.5 * (stiffness + stiffness.T)
        reference = _degree_four_variable_reference_stiffness(resolution, seed)
        relative_spectrum = linalg.eigvalsh(
            gauge_basis.T @ symmetric_stiffness @ gauge_basis,
            gauge_basis.T @ reference @ gauge_basis,
            check_finite=False,
        )
        assert relative_spectrum[0] > 0.07
        assert relative_spectrum[-1] < 2.6

    @pytest.mark.parametrize("wave_numbers", [(1, 1), (2, 1), (2, 2), (3, 1)])
    @pytest.mark.parametrize("seed", range(3))
    def test_edge_energy_stabilization_is_smoothly_consistent(self, wave_numbers, seed):
        records = []
        kx, ky = wave_numbers
        for resolution in (7, 9, 11, 13, 17, 21):
            cloud, _sandbox = _boundary_complete_edge_stabilized_vc4_case(resolution, seed)
            stabilization, gauge_basis, reference_factor = _edge_stabilization_dual_data(resolution, seed)
            node_values = np.cos(kx * np.pi * cloud.nodes[:, 0]) * np.cos(ky * np.pi * cloud.nodes[:, 1])
            defect = stabilization @ node_values
            energy = float(np.sqrt(max(node_values @ defect, 0.0)))
            reduced_defect = gauge_basis.T @ defect
            dual = float(
                np.sqrt(reduced_defect @ linalg.cho_solve(reference_factor, reduced_defect, check_finite=False))
            )
            records.append((cloud.nominal_spacing, energy, dual))

        spacings, energy_defects, dual_defects = map(np.asarray, zip(*records, strict=True))
        energy_rate = float(np.polyfit(np.log(spacings), np.log(energy_defects), 1)[0])
        dual_rate = float(np.polyfit(np.log(spacings), np.log(dual_defects), 1)[0])
        assert energy_rate > 2.2
        assert dual_rate > 3.05
        assert np.all(np.diff(energy_defects) < 0.0)
        assert np.all(np.diff(dual_defects) < 0.0)

    def test_degree_three_mls_dispatch_preserves_the_vci_gate(self):
        sandbox = CentroidalVCISCNIOperatorSandbox(
            _grid(7),
            rho=3.5 / 6.0,
            bounds=BOUNDS,
            degree=3,
            vci_degree=3,
            vci_support_radius=3.5 / 6.0,
        )
        assert sandbox.degree == 3
        assert sandbox.diagnostics.corrected_patch_defect < 1e-12
        assert sandbox.diagnostics.gauge_coercivity_min > 9.0
        assert sandbox.diagnostics.stiffness_nullity == 1

    @pytest.mark.parametrize("test_gradient_base", ["trial", "scni"])
    def test_pointwise_trial_gradient_dispatch_preserves_the_quadratic_patch(self, test_gradient_base):
        sandbox = CentroidalVCISCNIOperatorSandbox(
            _grid(7),
            rho=0.5,
            bounds=BOUNDS,
            trial_gradient_mode="pointwise",
            test_gradient_base=test_gradient_base,
        )
        assert sandbox.trial_gradient_mode == "pointwise"
        assert sandbox.test_gradient_base == test_gradient_base
        assert sandbox.diagnostics.discrete_patch_defect < 1e-12
        assert sandbox.diagnostics.gauge_coercivity_min > 9.0

    @pytest.mark.parametrize("wave_numbers", [(1, 1), (2, 1), (2, 2), (3, 1)])
    @pytest.mark.parametrize("seed", range(3))
    def test_stabilized_aligned_vc4_poisson_converges_near_fourth_order(self, wave_numbers, seed):
        records = []
        kx, ky = wave_numbers
        for resolution in (7, 9, 11, 13, 17, 21):
            cloud, sandbox = _boundary_complete_stabilized_vc4_case(resolution, seed)
            points = sandbox.evaluation_points
            exact_values = np.cos(kx * np.pi * points[:, 0]) * np.cos(ky * np.pi * points[:, 1])
            forcing_nodes = (
                (kx**2 + ky**2)
                * np.pi**2
                * np.cos(kx * np.pi * cloud.nodes[:, 0])
                * np.cos(ky * np.pi * cloud.nodes[:, 1])
            )
            exact_gradient = np.column_stack(
                [
                    -kx * np.pi * np.sin(kx * np.pi * points[:, 0]) * np.cos(ky * np.pi * points[:, 1]),
                    -ky * np.pi * np.cos(kx * np.pi * points[:, 0]) * np.sin(ky * np.pi * points[:, 1]),
                ]
            )

            E = sandbox.value_operator().toarray()
            M = sandbox.mass().toarray()
            K = sandbox.stiffness().toarray()
            solution, relative_residual, gauge_defect = _solve_neumann_system(
                K,
                M,
                M @ forcing_nodes,
                sandbox.left_null_vector(),
            )
            assert relative_residual < 1e-12
            assert gauge_defect < 1e-12
            reconstructed_gradient = np.column_stack([gradient @ solution for gradient in sandbox.trial_gradient()])
            l2_error, h1_error = _physical_poisson_errors(
                E @ solution,
                reconstructed_gradient,
                exact_values,
                exact_gradient,
                sandbox.weights,
            )
            records.append((cloud.nominal_spacing, l2_error, h1_error))

        spacings, l2_errors, h1_errors = map(np.asarray, zip(*records, strict=True))
        l2_rate = float(np.polyfit(np.log(spacings), np.log(l2_errors), 1)[0])
        h1_rate = float(np.polyfit(np.log(spacings), np.log(h1_errors), 1)[0])
        assert l2_rate > 3.65
        assert h1_rate > 3.6
        assert l2_errors[-1] < 1e-3
        assert h1_errors[-1] < 0.022
        assert np.all(np.diff(h1_errors) < 0.0)

    @pytest.mark.parametrize("wave_numbers", [(1, 1), (2, 1), (2, 2), (3, 1)])
    @pytest.mark.parametrize("seed", range(3))
    def test_edge_energy_stabilized_vc4_poisson_converges(self, wave_numbers, seed):
        records = []
        kx, ky = wave_numbers
        for resolution in (7, 9, 11, 13, 17, 21):
            cloud, sandbox = _boundary_complete_edge_stabilized_vc4_case(resolution, seed)
            points = sandbox.evaluation_points
            exact_values = np.cos(kx * np.pi * points[:, 0]) * np.cos(ky * np.pi * points[:, 1])
            forcing_nodes = (
                (kx**2 + ky**2)
                * np.pi**2
                * np.cos(kx * np.pi * cloud.nodes[:, 0])
                * np.cos(ky * np.pi * cloud.nodes[:, 1])
            )
            exact_gradient = np.column_stack(
                [
                    -kx * np.pi * np.sin(kx * np.pi * points[:, 0]) * np.cos(ky * np.pi * points[:, 1]),
                    -ky * np.pi * np.cos(kx * np.pi * points[:, 0]) * np.sin(ky * np.pi * points[:, 1]),
                ]
            )
            mass = sandbox.mass().toarray()
            solution, relative_residual, gauge_defect = _solve_neumann_system(
                sandbox.stiffness().toarray(),
                mass,
                mass @ forcing_nodes,
                sandbox.left_null_vector(),
            )
            assert relative_residual < 1e-12
            assert gauge_defect < 1e-12
            reconstructed_gradient = np.column_stack([gradient @ solution for gradient in sandbox.trial_gradient()])
            errors = _physical_poisson_errors(
                sandbox.value_operator() @ solution,
                reconstructed_gradient,
                exact_values,
                exact_gradient,
                sandbox.weights,
            )
            records.append((cloud.nominal_spacing, *errors))

        spacings, l2_errors, h1_errors = map(np.asarray, zip(*records, strict=True))
        l2_rate = float(np.polyfit(np.log(spacings), np.log(l2_errors), 1)[0])
        h1_rate = float(np.polyfit(np.log(spacings), np.log(h1_errors), 1)[0])
        assert l2_rate > 3.7
        assert h1_rate > 3.0
        assert l2_errors[-1] < 2.1e-3
        assert h1_errors[-1] < 0.033
        assert np.all(np.diff(l2_errors) < 0.0)
        assert np.all(np.diff(h1_errors) < 0.0)

    @pytest.mark.parametrize("wave_numbers", [(1, 1), (2, 1), (2, 2), (3, 1)])
    @pytest.mark.parametrize("seed", range(3))
    def test_variable_coefficient_edge_vc4_poisson_converges(self, wave_numbers, seed):
        records = []
        kx, ky = wave_numbers
        for resolution in (7, 9, 11, 13, 17, 21):
            cloud, sandbox = _boundary_complete_variable_edge_stabilized_vc4_case(resolution, seed)
            points = sandbox.evaluation_points
            exact_values = np.cos(kx * np.pi * points[:, 0]) * np.cos(ky * np.pi * points[:, 1])
            exact_gradient = np.column_stack(
                [
                    -kx * np.pi * np.sin(kx * np.pi * points[:, 0]) * np.cos(ky * np.pi * points[:, 1]),
                    -ky * np.pi * np.cos(kx * np.pi * points[:, 0]) * np.sin(ky * np.pi * points[:, 1]),
                ]
            )
            coefficient = _smooth_elliptic_coefficient(points)
            coefficient_gradient = _smooth_elliptic_coefficient_gradient(points)
            forcing = (kx**2 + ky**2) * np.pi**2 * coefficient * exact_values - np.sum(
                coefficient_gradient * exact_gradient, axis=1
            )
            value_operator = sandbox.value_operator().toarray()
            mass = sandbox.mass().toarray()
            load = value_operator.T @ (sandbox.weights * forcing)
            solution, relative_residual, gauge_defect = _solve_neumann_system(
                sandbox.stiffness().toarray(),
                mass,
                load,
                sandbox.left_null_vector(),
            )
            assert relative_residual < 2e-12
            assert gauge_defect < 2e-12
            reconstructed_gradient = np.column_stack([gradient @ solution for gradient in sandbox.trial_gradient()])
            errors = _physical_poisson_errors(
                value_operator @ solution,
                reconstructed_gradient,
                exact_values,
                exact_gradient,
                sandbox.weights,
            )
            records.append((cloud.nominal_spacing, *errors))

        spacings, l2_errors, h1_errors = map(np.asarray, zip(*records, strict=True))
        l2_rate = float(np.polyfit(np.log(spacings), np.log(l2_errors), 1)[0])
        h1_rate = float(np.polyfit(np.log(spacings), np.log(h1_errors), 1)[0])
        assert l2_rate > 3.35
        assert h1_rate > 2.7
        assert l2_errors[-1] < 2.1e-3
        assert h1_errors[-1] < 0.033
        assert np.all(np.diff(l2_errors) < 0.0)
        assert np.all(np.diff(h1_errors) < 0.0)

    def test_variable_coefficient_vci_correction_beats_frozen_constant_correction(self):
        _cloud, variable = _boundary_complete_variable_edge_stabilized_vc4_case(21, 0)
        _, constant = _boundary_complete_edge_stabilized_vc4_case(21, 0)
        points = variable.evaluation_points
        coefficient = _smooth_elliptic_coefficient(points)
        elliptic_weights = variable.weights * coefficient
        frozen_stiffness = sum(
            test_gradient.toarray().T @ (elliptic_weights[:, None] * trial_gradient.toarray())
            for test_gradient, trial_gradient in zip(
                constant.test_gradient(),
                constant.trial_gradient(),
                strict=True,
            )
        )
        frozen_stiffness += 0.1 * variable.stabilization().toarray()

        exact_values = np.cos(np.pi * points[:, 0]) * np.cos(np.pi * points[:, 1])
        exact_gradient = np.column_stack(
            [
                -np.pi * np.sin(np.pi * points[:, 0]) * np.cos(np.pi * points[:, 1]),
                -np.pi * np.cos(np.pi * points[:, 0]) * np.sin(np.pi * points[:, 1]),
            ]
        )
        forcing = 2.0 * np.pi**2 * coefficient * exact_values - np.sum(
            _smooth_elliptic_coefficient_gradient(points) * exact_gradient, axis=1
        )
        value_operator = variable.value_operator().toarray()
        mass = variable.mass().toarray()
        load = value_operator.T @ (variable.weights * forcing)
        frozen_left_null = np.linalg.svd(frozen_stiffness)[0][:, -1]
        frozen_left_null /= frozen_left_null @ np.ones(variable.n_dof)
        errors = []
        for stiffness, left_null in (
            (variable.stiffness().toarray(), variable.left_null_vector()),
            (frozen_stiffness, frozen_left_null),
        ):
            solution, relative_residual, gauge_defect = _solve_neumann_system(
                stiffness,
                mass,
                load,
                left_null,
            )
            assert relative_residual < 2e-12
            assert gauge_defect < 2e-12
            reconstructed_gradient = np.column_stack([gradient @ solution for gradient in variable.trial_gradient()])
            errors.append(
                _physical_poisson_errors(
                    value_operator @ solution,
                    reconstructed_gradient,
                    exact_values,
                    exact_gradient,
                    variable.weights,
                )
            )
        variable_errors, frozen_errors = errors
        assert variable_errors[0] < 0.1 * frozen_errors[0]
        assert variable_errors[1] < 0.1 * frozen_errors[1]

    @pytest.mark.parametrize("wave_numbers", [(1, 1), (2, 1), (2, 2), (3, 1)])
    def test_stabilized_aligned_vc4_does_not_dominate_evaluation_matched_common(self, wave_numbers):
        candidate_records = []
        common_records = []
        kx, ky = wave_numbers
        for resolution in (7, 9, 11, 13, 17, 21):
            cloud, candidate = _boundary_complete_stabilized_vc4_case(resolution, 0)
            matched_candidate, common_K, common_M, common_count, candidate_budget = (
                _degree_four_evaluation_matched_common_case(resolution, 0)
            )
            assert matched_candidate is candidate
            assert 0.8 * candidate_budget < common_count <= candidate_budget
            points = candidate.evaluation_points
            exact_values = np.cos(kx * np.pi * points[:, 0]) * np.cos(ky * np.pi * points[:, 1])
            exact_gradient = np.column_stack(
                [
                    -kx * np.pi * np.sin(kx * np.pi * points[:, 0]) * np.cos(ky * np.pi * points[:, 1]),
                    -ky * np.pi * np.cos(kx * np.pi * points[:, 0]) * np.sin(ky * np.pi * points[:, 1]),
                ]
            )
            forcing_nodes = (
                (kx**2 + ky**2)
                * np.pi**2
                * np.cos(kx * np.pi * cloud.nodes[:, 0])
                * np.cos(ky * np.pi * cloud.nodes[:, 1])
            )
            E = candidate.value_operator().toarray()
            gradients = [gradient.toarray() for gradient in candidate.trial_gradient()]
            one = np.ones(candidate.n_dof)
            arms = {
                "candidate": (
                    candidate.stiffness().toarray(),
                    candidate.mass().toarray(),
                    candidate.left_null_vector(),
                ),
                "common": (common_K, common_M, one / len(one)),
            }
            for name, (K, M, left_null) in arms.items():
                solution, relative_residual, gauge_defect = _solve_neumann_system(
                    K,
                    M,
                    M @ forcing_nodes,
                    left_null,
                )
                assert relative_residual < 1e-12
                assert gauge_defect < 1e-12
                reconstructed_gradient = np.column_stack([gradient @ solution for gradient in gradients])
                errors = _physical_poisson_errors(
                    E @ solution,
                    reconstructed_gradient,
                    exact_values,
                    exact_gradient,
                    candidate.weights,
                )
                record = (cloud.nominal_spacing, *errors)
                if name == "candidate":
                    candidate_records.append(record)
                else:
                    common_records.append(record)

        _, candidate_l2, candidate_h1 = map(np.asarray, zip(*candidate_records, strict=True))
        _, common_l2, common_h1 = map(np.asarray, zip(*common_records, strict=True))
        assert common_l2[-1] < 0.4 * candidate_l2[-1]
        if wave_numbers == (1, 1):
            assert candidate_h1[-1] < 0.6 * common_h1[-1]
        else:
            assert common_h1[-1] < 0.5 * candidate_h1[-1]

    def test_interior_only_lloyd_cloud_fails_the_mass_gate(self):
        from mfgarchon.geometry.collocation import ImplicitDomainCollocation
        from mfgarchon.geometry.implicit import Hyperrectangle

        nodes = ImplicitDomainCollocation(Hyperrectangle(bounds=BOUNDS)).sample_interior(
            120,
            method="lloyd",
            seed=0,
        )
        h = 1.0 / np.sqrt(len(nodes))
        with pytest.raises(np.linalg.LinAlgError, match=r"mass gate failed: cond\(M\)"):
            CentroidalVCISCNIOperatorSandbox(nodes, rho=4.0 * h, bounds=BOUNDS)

    def test_mass_and_petrov_nullspaces_are_not_conflated(self, structured_sandbox):
        E = structured_sandbox.value_operator().toarray()
        W = structured_sandbox.weights
        M = structured_sandbox.mass().toarray()
        K = structured_sandbox.stiffness().toarray()
        one = np.ones(structured_sandbox.n_dof)
        assert np.allclose(M, E.T @ (W[:, None] * E), atol=1e-14)
        assert np.linalg.eigvalsh(M).min() > 0.0
        assert np.linalg.norm(K - K.T) / np.linalg.norm(K) > 1e-4
        assert np.max(np.abs(K @ one)) < 1e-10
        assert np.max(np.abs(one @ K)) > 1e-3
        left_null = structured_sandbox.left_null_vector()
        assert abs(left_null.sum() - 1.0) < 1e-12
        assert np.max(np.abs(left_null @ K)) < 1e-10

    def test_rank_deficient_local_support_fails_loud(self):
        n = 7
        with pytest.raises(ValueError, match="VCI local constraint rank failure"):
            CentroidalVCISCNIOperatorSandbox(
                _grid(n),
                rho=3.0 / (n - 1),
                bounds=BOUNDS,
                vci_support_radius=0.2 / (n - 1),
            )

    def test_duplicate_nodes_fail_before_geometry_assembly(self):
        nodes = _grid(7)
        nodes[1] = nodes[0]
        with pytest.raises(ValueError, match="does not admit duplicate nodes"):
            CentroidalVCISCNIOperatorSandbox(nodes, rho=0.5, bounds=BOUNDS)

    def test_unsupported_vci_degree_fails_before_geometry_assembly(self):
        with pytest.raises(ValueError, match=r"vci_degree in \{2, 3, 4\}"):
            CentroidalVCISCNIOperatorSandbox(_grid(7), rho=0.5, bounds=BOUNDS, vci_degree=5)
        with pytest.raises(ValueError, match=r"vci_degree in \{2, 3, 4\}"):
            CentroidalVCISCNIOperatorSandbox(_grid(7), rho=0.5, bounds=BOUNDS, vci_degree=4.0)

    def test_unsupported_mls_degree_fails_before_geometry_assembly(self):
        with pytest.raises(ValueError, match=r"MLS degree in \{2, 3, 4\}"):
            CentroidalVCISCNIOperatorSandbox(_grid(7), rho=0.5, bounds=BOUNDS, degree=5)
        with pytest.raises(ValueError, match=r"MLS degree in \{2, 3, 4\}"):
            CentroidalVCISCNIOperatorSandbox(_grid(7), rho=0.5, bounds=BOUNDS, degree=4.0)

    def test_mls_degree_requires_enough_nodes_before_geometry_assembly(self):
        with pytest.raises(ValueError, match="degree-4 MLS requires at least 15 nodes"):
            CentroidalVCISCNIOperatorSandbox(_grid(3), rho=0.75, bounds=BOUNDS, degree=4)

    @pytest.mark.parametrize(
        ("options", "message"),
        [
            ({"trial_gradient_mode": "unknown"}, "trial_gradient_mode"),
            ({"test_gradient_base": "unknown"}, "test_gradient_base"),
        ],
    )
    def test_unsupported_gradient_modes_fail_before_geometry_assembly(self, options, message):
        with pytest.raises(ValueError, match=message):
            CentroidalVCISCNIOperatorSandbox(_grid(7), rho=0.5, bounds=BOUNDS, **options)

    @pytest.mark.parametrize(
        ("options", "message"),
        [
            ({"polynomial_null_stabilization": -0.1}, "finite and nonnegative"),
            ({"polynomial_null_stabilization": np.nan}, "finite and nonnegative"),
            ({"stabilization_metric": "unknown"}, "stabilization_metric"),
            ({"stabilization_support_radius": 0.0}, "finite and positive"),
            (
                {"stabilization_metric": "edge_energy", "stabilization_support_radius": 0.5},
                "active edge MLS stencil determines its support",
            ),
            ({"max_stabilization_condition": 1.0}, "greater than one"),
        ],
    )
    def test_invalid_polynomial_null_stabilization_parameters_fail_before_geometry_assembly(
        self,
        options,
        message,
    ):
        with pytest.raises(ValueError, match=message):
            CentroidalVCISCNIOperatorSandbox(_grid(7), rho=0.5, bounds=BOUNDS, **options)

    @pytest.mark.parametrize(
        ("options", "exception", "message"),
        [
            (
                {"elliptic_coefficient": lambda points: np.ones(len(points))},
                ValueError,
                "must be supplied together",
            ),
            (
                {"elliptic_coefficient_gradient": lambda points: np.zeros_like(points)},
                ValueError,
                "must be supplied together",
            ),
            (
                {
                    "elliptic_coefficient": 1.0,
                    "elliptic_coefficient_gradient": lambda points: np.zeros_like(points),
                },
                TypeError,
                "must be callable",
            ),
            (
                {
                    "elliptic_coefficient": lambda points: np.ones((len(points), 1)),
                    "elliptic_coefficient_gradient": lambda points: np.zeros_like(points),
                },
                ValueError,
                "elliptic_coefficient\\(centroids\\) must return shape",
            ),
            (
                {
                    "elliptic_coefficient": lambda points: np.ones(len(points)),
                    "elliptic_coefficient_gradient": lambda points: np.zeros(len(points)),
                },
                ValueError,
                "elliptic_coefficient_gradient\\(centroids\\) must return shape",
            ),
            (
                {
                    "elliptic_coefficient": lambda points: np.full(len(points), np.nan),
                    "elliptic_coefficient_gradient": lambda points: np.zeros_like(points),
                },
                ValueError,
                "returned non-finite values",
            ),
            (
                {
                    "elliptic_coefficient": lambda points: np.ones(len(points)),
                    "elliptic_coefficient_gradient": lambda points: np.full_like(points, np.inf),
                },
                ValueError,
                "returned non-finite values",
            ),
            (
                {
                    "elliptic_coefficient": lambda points: np.where(points[:, 0] == 0.0, -1.0, 1.0),
                    "elliptic_coefficient_gradient": lambda points: np.zeros_like(points),
                },
                ValueError,
                "must be strictly positive",
            ),
        ],
    )
    def test_invalid_elliptic_coefficient_contract_fails_loudly(self, options, exception, message):
        aligned_options = {
            "degree": 4,
            "vci_degree": 4,
            "vci_support_radius": 4.0 / 6.0,
            "trial_gradient_mode": "pointwise",
            "test_gradient_base": "trial",
            "polynomial_null_stabilization": 0.1,
            "stabilization_metric": "edge_energy",
        }
        with pytest.raises(exception, match=message):
            CentroidalVCISCNIOperatorSandbox(
                _grid(7),
                rho=4.5 / 6.0,
                bounds=BOUNDS,
                **aligned_options,
                **options,
            )

    def test_spatial_elliptic_coefficient_rejects_untested_operator_branch(self):
        with pytest.raises(ValueError, match="only defined for the aligned degree-4 VCI candidate"):
            CentroidalVCISCNIOperatorSandbox(
                _grid(7),
                rho=0.5,
                bounds=BOUNDS,
                elliptic_coefficient=lambda points: np.ones(len(points)),
                elliptic_coefficient_gradient=lambda points: np.zeros_like(points),
            )

    def test_polynomial_null_stabilization_fails_on_rank_deficient_support(self):
        nodes = _grid(7)
        with pytest.raises(ValueError, match="Polynomial-null stabilization rank failure"):
            CentroidalVCISCNIOperatorSandbox(
                nodes,
                rho=0.5,
                bounds=BOUNDS,
                polynomial_null_stabilization=0.1,
                stabilization_support_radius=0.01,
            )

    def test_polynomial_null_stabilization_fails_on_ill_conditioned_projection(self):
        with pytest.raises(np.linalg.LinAlgError, match="Polynomial-null stabilization is ill-conditioned"):
            CentroidalVCISCNIOperatorSandbox(
                _grid(7),
                rho=0.5,
                bounds=BOUNDS,
                polynomial_null_stabilization=0.1,
                max_stabilization_condition=1.01,
            )

    def test_condition_gate_fails_loud(self):
        n = 7
        with pytest.raises(np.linalg.LinAlgError, match="VCI local constraint is ill-conditioned"):
            CentroidalVCISCNIOperatorSandbox(
                _grid(n),
                rho=3.0 / (n - 1),
                bounds=BOUNDS,
                max_local_condition=1.0,
            )


class TestPairedMFGOperator:
    def test_variable_elliptic_correction_cannot_be_reused_for_nonlinear_stabilization(self):
        _cloud, sandbox = _boundary_complete_variable_edge_stabilized_vc4_case(7, 0)
        with pytest.raises(ValueError, match="requires its own variational correction"):
            PairedMFGOperatorSandbox(
                sandbox,
                diffusion=0.07,
                hamiltonian=_hamiltonian,
                hamiltonian_gradient=_hamiltonian_gradient,
                stabilization_flux=_stabilization_flux,
                stabilization_jacobian=_stabilization_jacobian,
            )

    def test_complete_analytic_jacobian_matches_centered_difference(self, structured_sandbox):
        operator = PairedMFGOperatorSandbox(
            structured_sandbox,
            diffusion=0.07,
            hamiltonian=_hamiltonian,
            hamiltonian_gradient=_hamiltonian_gradient,
            stabilization_flux=_stabilization_flux,
            stabilization_jacobian=_stabilization_jacobian,
        )
        rng = np.random.default_rng(12)
        state = 0.1 * rng.standard_normal(structured_sandbox.n_dof)
        analytic = operator.jacobian(state)
        finite_difference = np.empty_like(analytic)
        step = 2e-7
        for column in range(structured_sandbox.n_dof):
            direction = np.zeros(structured_sandbox.n_dof)
            direction[column] = step
            finite_difference[:, column] = (
                operator.residual(state + direction) - operator.residual(state - direction)
            ) / (2.0 * step)
        relative_defect = np.linalg.norm(analytic - finite_difference) / np.linalg.norm(analytic)
        assert relative_defect < 2e-8

    def test_forward_block_is_exact_transpose_and_preserves_physical_mass(self, structured_sandbox):
        operator = PairedMFGOperatorSandbox(
            structured_sandbox,
            diffusion=0.05,
            hamiltonian=_hamiltonian,
            hamiltonian_gradient=_hamiltonian_gradient,
            stabilization_flux=_stabilization_flux,
            stabilization_jacobian=_stabilization_jacobian,
        )
        rng = np.random.default_rng(4)
        state = 0.05 * rng.standard_normal(structured_sandbox.n_dof)
        density = np.ones(structured_sandbox.n_dof) + 0.02 * rng.standard_normal(structured_sandbox.n_dof)
        jacobian = operator.jacobian(state)
        one = np.ones(structured_sandbox.n_dof)
        assert np.max(np.abs(jacobian @ one)) < 1e-10
        assert np.array_equal(operator.forward_spatial(state, density), jacobian.T @ density)

        M = structured_sandbox.mass().toarray()
        time_step = 1e-3
        advanced_density = np.linalg.solve(M / time_step + jacobian.T, (M / time_step) @ density)
        assert abs(operator.physical_mass(advanced_density) - operator.physical_mass(density)) < 1e-12

    def test_hamiltonian_pairing_equals_reconstructed_density_bregman_term(self, structured_sandbox):
        operator = PairedMFGOperatorSandbox(
            structured_sandbox,
            diffusion=0.0,
            hamiltonian=_hamiltonian,
            hamiltonian_gradient=_hamiltonian_gradient,
        )
        rng = np.random.default_rng(9)
        state_1 = 0.05 * rng.standard_normal(structured_sandbox.n_dof)
        state_2 = 0.05 * rng.standard_normal(structured_sandbox.n_dof)
        density_1 = np.ones(structured_sandbox.n_dof) + 0.02 * rng.standard_normal(structured_sandbox.n_dof)
        density_2 = np.ones(structured_sandbox.n_dof) + 0.02 * rng.standard_normal(structured_sandbox.n_dof)
        rho_1 = structured_sandbox.reconstructed_density(density_1)
        rho_2 = structured_sandbox.reconstructed_density(density_2)
        assert np.min(rho_1) > 0.0
        assert np.min(rho_2) > 0.0

        momentum_1 = operator.momentum(state_1)
        momentum_2 = operator.momentum(state_2)
        values_1 = _hamiltonian(structured_sandbox.evaluation_points, momentum_1)
        values_2 = _hamiltonian(structured_sandbox.evaluation_points, momentum_2)
        gradients_1 = _hamiltonian_gradient(structured_sandbox.evaluation_points, momentum_1)
        gradients_2 = _hamiltonian_gradient(structured_sandbox.evaluation_points, momentum_2)
        bregman_21 = values_2 - values_1 - np.sum(gradients_1 * (momentum_2 - momentum_1), axis=1)
        bregman_12 = values_1 - values_2 - np.sum(gradients_2 * (momentum_1 - momentum_2), axis=1)
        expected = structured_sandbox.weights @ (rho_1 * bregman_21 + rho_2 * bregman_12)

        residual_1 = operator.residual(state_1)
        residual_2 = operator.residual(state_2)
        state_delta = state_2 - state_1
        paired = density_1 @ (residual_2 - residual_1 - operator.jacobian(state_1) @ state_delta)
        paired += density_2 @ (residual_1 - residual_2 + operator.jacobian(state_2) @ state_delta)
        assert abs(paired - expected) < 1e-13
        assert expected >= 0.0

    def test_invalid_hamiltonian_shape_fails_loud(self, structured_sandbox):
        operator = PairedMFGOperatorSandbox(
            structured_sandbox,
            diffusion=0.0,
            hamiltonian=lambda _points, _momentum: np.zeros((structured_sandbox.n_dof, 1)),
            hamiltonian_gradient=_hamiltonian_gradient,
        )
        with pytest.raises(ValueError, match="Hamiltonian values must have shape"):
            operator.residual(np.zeros(structured_sandbox.n_dof))

    def test_stabilization_derivative_is_mandatory(self, structured_sandbox):
        with pytest.raises(ValueError, match="must be supplied together"):
            PairedMFGOperatorSandbox(
                structured_sandbox,
                diffusion=0.0,
                hamiltonian=_hamiltonian,
                hamiltonian_gradient=_hamiltonian_gradient,
                stabilization_flux=_stabilization_flux,
            )
