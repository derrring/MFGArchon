"""Admission tests for the experimental centroidal VC2-SCNI operator sandbox."""

from __future__ import annotations

from functools import cache

import pytest

import numpy as np

from mfgarchon.alg.numerical.meshless_galerkin.centroidal_vci_sandbox import (
    BoundaryCompleteCVTCloud,
    CentroidalVC2SCNIOperatorSandbox,
    PairedMFGOperatorSandbox,
    boundary_complete_cvt_rectangle,
)
from mfgarchon.alg.numerical.meshless_galerkin.voronoi_cells import clipped_voronoi_cells

BOUNDS = [(0.0, 1.0), (0.0, 1.0)]


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
) -> tuple[BoundaryCompleteCVTCloud, CentroidalVC2SCNIOperatorSandbox]:
    cloud = boundary_complete_cvt_rectangle(resolution, BOUNDS, seed=seed)
    sandbox = CentroidalVC2SCNIOperatorSandbox(
        cloud.nodes,
        rho=3.0 * cloud.nominal_spacing,
        bounds=BOUNDS,
    )
    return cloud, sandbox


@pytest.fixture(scope="module")
def structured_sandbox() -> CentroidalVC2SCNIOperatorSandbox:
    n = 7
    return CentroidalVC2SCNIOperatorSandbox(_grid(n), rho=3.0 / (n - 1), bounds=BOUNDS)


@pytest.fixture(scope="module")
def jittered_sandbox() -> CentroidalVC2SCNIOperatorSandbox:
    n = 7
    return CentroidalVC2SCNIOperatorSandbox(
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


class TestCentroidalVC2Geometry:
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
        sandbox = CentroidalVC2SCNIOperatorSandbox(
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
            exact_values -= float(sandbox.weights @ exact_values / np.sum(sandbox.weights))
            exact_gradient = np.column_stack(
                [
                    -kx * np.pi * np.sin(kx * np.pi * points[:, 0]) * np.cos(ky * np.pi * points[:, 1]),
                    -ky * np.pi * np.cos(kx * np.pi * points[:, 0]) * np.sin(ky * np.pi * points[:, 1]),
                ]
            )

            E = sandbox.value_operator().toarray()
            M = sandbox.mass().toarray()
            K = sandbox.stiffness().toarray()
            one = np.ones(sandbox.n_dof)
            mean_constraint = M @ one
            left_null = sandbox.left_null_vector()
            load = E.T @ (sandbox.weights * forcing)
            saddle = np.block(
                [
                    [K, left_null[:, None]],
                    [mean_constraint[None, :], np.zeros((1, 1))],
                ]
            )
            augmented_load = np.append(load, 0.0)
            augmented_solution = np.linalg.solve(saddle, augmented_load)
            solution = augmented_solution[:-1]
            relative_saddle_residual = np.linalg.norm(saddle @ augmented_solution - augmented_load) / np.linalg.norm(
                load
            )
            assert relative_saddle_residual < 1e-12
            assert abs(mean_constraint @ solution) < 1e-12

            reconstructed_values = E @ solution
            reconstructed_gradient = np.column_stack([gradient @ solution for gradient in sandbox.trial_gradient()])
            l2_error = float(np.sqrt(sandbox.weights @ (reconstructed_values - exact_values) ** 2))
            h1_error = float(np.sqrt(sandbox.weights @ np.sum((reconstructed_gradient - exact_gradient) ** 2, axis=1)))
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
            CentroidalVC2SCNIOperatorSandbox(nodes, rho=4.0 * h, bounds=BOUNDS)

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
        with pytest.raises(ValueError, match="VC2 local constraint rank failure"):
            CentroidalVC2SCNIOperatorSandbox(
                _grid(n),
                rho=3.0 / (n - 1),
                bounds=BOUNDS,
                vci_support_radius=0.2 / (n - 1),
            )

    def test_duplicate_nodes_fail_before_geometry_assembly(self):
        nodes = _grid(7)
        nodes[1] = nodes[0]
        with pytest.raises(ValueError, match="does not admit duplicate nodes"):
            CentroidalVC2SCNIOperatorSandbox(nodes, rho=0.5, bounds=BOUNDS)

    def test_condition_gate_fails_loud(self):
        n = 7
        with pytest.raises(np.linalg.LinAlgError, match="VC2 local constraint is ill-conditioned"):
            CentroidalVC2SCNIOperatorSandbox(
                _grid(n),
                rho=3.0 / (n - 1),
                bounds=BOUNDS,
                max_local_condition=1.0,
            )


class TestPairedMFGOperator:
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
