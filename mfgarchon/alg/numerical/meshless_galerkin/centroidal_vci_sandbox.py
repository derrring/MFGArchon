"""Operator sandbox for centroidal VCI-SCNI and an exactly paired MFG linearization.

This module is deliberately outside ``WeakFormDiscretization``. It is an admission
gate for a candidate Petrov-Galerkin method, not a production solver backend.

The construction uses clipped Voronoi centroids ``c_a`` as sampling points,
``M = E.T @ W @ E`` with ``E[a, i] = phi_i(c_a)``, a selectable SCNI
cell-average or centroid-pointwise trial gradient, and a local
variational-consistency correction of the test gradient.
The correction targets shifted/scaled weak moments through degree two, three,
or four, including boundary fluxes. The MLS trial degree is selected
independently, so target weak moments and the actual discrete patch are reported
separately.

An optional local polynomial-null stabilization is a research diagnostic, not a
production backend. It tests whether a symmetric positive-semidefinite correction
can remove non-polynomial negative modes without changing the target patch. The
stabilization metric is either the original Euclidean stencil metric or a physical
edge-gradient energy assembled from the already-required cell-edge evaluations.

All nonlinear MFG blocks are derived from one residual. The forward spatial
operator is therefore the transpose of the complete analytic Jacobian; no
independent advection assembly exists here.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import TYPE_CHECKING

import numpy as np
from scipy import linalg, sparse
from scipy.spatial import cKDTree

from mfgarchon.alg.numerical.meshless_galerkin.mls_basis import monomial_exponents, shape_functions_and_grads
from mfgarchon.alg.numerical.meshless_galerkin.voronoi_cells import CellGeometry, clipped_voronoi_cells

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray


@dataclass(frozen=True)
class VCIAdmissionDiagnostics:
    """Numerical invariants that decide whether a cloud is admitted by the sandbox.

    ``corrected_patch_defect`` uses exact target-polynomial gradients.
    ``discrete_patch_defect`` instead applies the assembled stiffness to nodal
    polynomial values and therefore includes any trial-gradient error at the
    selected VCI target degree.

    ``raw_gauge_coercivity_min`` is measured before polynomial-null
    stabilization; ``gauge_coercivity_min`` is measured after it.
    """

    local_rank_min: int
    local_condition_max: float
    correction_ratio_median: float
    correction_ratio_max: float
    mass_condition: float
    value_constant_defect: float
    trial_constant_defect: float
    quadratic_gradient_defect: float
    vci_trial_gradient_defect: float
    plain_patch_defect: float
    corrected_patch_defect: float
    discrete_patch_defect: float
    stabilization_rank_min: int
    stabilization_condition_max: float
    stabilization_support_count_min: int
    stabilization_support_count_max: int
    stabilization_patch_defect: float
    right_constant_defect: float
    left_constant_defect: float
    raw_gauge_coercivity_min: float
    gauge_coercivity_min: float
    gauge_smallest_singular_value: float
    stiffness_nullity: int


@dataclass(frozen=True)
class _EdgeQuadrature:
    points: NDArray
    cell_indices: NDArray
    normal_weights: NDArray


@dataclass(frozen=True)
class BoundaryCompleteCVTCloud:
    """A rectangle cloud with fixed boundary sites and Lloyd-relaxed interior sites."""

    nodes: NDArray
    boundary_mask: NDArray
    resolution: int
    seed: int
    iterations: int
    nominal_spacing: float
    max_interior_centroid_offset_ratio: float
    min_separation_ratio: float
    min_cell_area_ratio: float
    max_cell_area_ratio: float
    max_cell_diameter_ratio: float

    @property
    def cell_area_ratio(self) -> float:
        return self.max_cell_area_ratio / self.min_cell_area_ratio


def boundary_complete_cvt_rectangle(
    resolution: int,
    bounds: list[tuple[float, float]],
    *,
    seed: int = 0,
    iterations: int = 30,
    jitter_fraction: float = 0.35,
    relaxation: float = 1.0,
) -> BoundaryCompleteCVTCloud:
    """Build a deterministic boundary-complete constrained-CVT cloud.

    The boundary sites are the boundary of a tensor grid and remain fixed.
    Interior sites start from independently jittered tensor-grid positions and
    undergo exact clipped-Voronoi centroid updates. The construction has exactly
    ``resolution**2`` sites and keeps ``4 * (resolution - 1)`` fitted boundary
    sites at every level.

    This is a research family constructor. It does not claim that arbitrary
    interior-only clouds become admissible after adding boundary samples.
    """
    if resolution < 5:
        raise ValueError(f"Boundary-complete CVT requires resolution >= 5, got {resolution}.")
    if iterations < 0:
        raise ValueError(f"Boundary-complete CVT requires iterations >= 0, got {iterations}.")
    if not 0.0 <= jitter_fraction < 0.5:
        raise ValueError(f"jitter_fraction must lie in [0, 0.5), got {jitter_fraction}.")
    if not 0.0 < relaxation <= 1.0:
        raise ValueError(f"relaxation must lie in (0, 1], got {relaxation}.")
    bounds_array = np.asarray(bounds, dtype=np.float64)
    if bounds_array.shape != (2, 2):
        raise ValueError(f"Boundary-complete CVT requires bounds with shape (2, 2), got {bounds_array.shape}.")
    spans = bounds_array[:, 1] - bounds_array[:, 0]
    if not np.all(np.isfinite(bounds_array)) or np.any(spans <= 0.0):
        raise ValueError(f"Boundary-complete CVT requires finite increasing bounds, got {bounds}.")

    axes = [np.linspace(bounds_array[direction, 0], bounds_array[direction, 1], resolution) for direction in range(2)]
    coordinates = np.meshgrid(*axes, indexing="ij")
    nodes = np.stack([coordinate.ravel() for coordinate in coordinates], axis=1)
    indices: tuple[NDArray, NDArray] = np.meshgrid(np.arange(resolution), np.arange(resolution), indexing="ij")
    boundary_mask = (
        (indices[0] == 0) | (indices[0] == resolution - 1) | (indices[1] == 0) | (indices[1] == resolution - 1)
    ).ravel()
    interior_mask = ~boundary_mask
    grid_spacings = spans / (resolution - 1)
    rng = np.random.default_rng(seed)
    nodes[interior_mask] += (
        jitter_fraction
        * rng.uniform(-1.0, 1.0, size=(int(np.count_nonzero(interior_mask)), 2))
        * grid_spacings[None, :]
    )

    for _ in range(iterations):
        cells = clipped_voronoi_cells(nodes, bounds)
        centroids = np.vstack([cell.centroid for cell in cells])
        nodes[interior_mask] += relaxation * (centroids[interior_mask] - nodes[interior_mask])

    cells = clipped_voronoi_cells(nodes, bounds)
    centroids = np.vstack([cell.centroid for cell in cells])
    nominal_spacing = float(np.max(grid_spacings))
    centroid_offsets = np.linalg.norm(centroids[interior_mask] - nodes[interior_mask], axis=1)
    nearest_distances, _ = cKDTree(nodes).query(nodes, k=2)
    cell_areas = np.array([cell.area for cell in cells], dtype=np.float64)
    nominal_cell_area = float(np.prod(spans) / len(nodes))
    cell_diameters = np.array(
        [np.max(np.linalg.norm(cell.polygon[:, None, :] - cell.polygon[None, :, :], axis=2)) for cell in cells],
        dtype=np.float64,
    )
    nodes.setflags(write=False)
    boundary_mask.setflags(write=False)
    return BoundaryCompleteCVTCloud(
        nodes=nodes,
        boundary_mask=boundary_mask,
        resolution=resolution,
        seed=seed,
        iterations=iterations,
        nominal_spacing=nominal_spacing,
        max_interior_centroid_offset_ratio=float(np.max(centroid_offsets) / nominal_spacing),
        min_separation_ratio=float(np.min(nearest_distances[:, 1]) / nominal_spacing),
        min_cell_area_ratio=float(np.min(cell_areas) / nominal_cell_area),
        max_cell_area_ratio=float(np.max(cell_areas) / nominal_cell_area),
        max_cell_diameter_ratio=float(np.max(cell_diameters) / nominal_spacing),
    )


def _edge_quadrature(cells: list[CellGeometry], n_gauss: int) -> _EdgeQuadrature:
    if n_gauss < 2:
        raise ValueError("VCI edge quadrature requires at least two Gauss points per edge.")
    xi, wi = np.polynomial.legendre.leggauss(n_gauss)
    t = 0.5 * (xi + 1.0)
    points: list[NDArray] = []
    cell_indices: list[NDArray] = []
    normal_weights: list[NDArray] = []
    for a, cell in enumerate(cells):
        polygon = cell.polygon
        for edge in range(len(polygon)):
            v0 = polygon[edge]
            delta = polygon[(edge + 1) % len(polygon)] - v0
            length = float(np.linalg.norm(delta))
            if length <= 1e-14:
                continue
            normal = np.array([delta[1], -delta[0]], dtype=np.float64) / length
            edge_points = v0[None, :] + t[:, None] * delta[None, :]
            edge_weights = 0.5 * length * wi
            points.append(edge_points)
            cell_indices.append(np.full(n_gauss, a, dtype=np.int64))
            normal_weights.append(edge_weights[:, None] * normal[None, :])
    if not points:
        raise ValueError("VCI edge quadrature found no nondegenerate cell edges.")
    return _EdgeQuadrature(
        points=np.vstack(points),
        cell_indices=np.concatenate(cell_indices),
        normal_weights=np.vstack(normal_weights),
    )


def _polynomial_exponents(degree: int) -> NDArray:
    return np.asarray(
        [
            (x_degree, total_degree - x_degree)
            for total_degree in range(degree + 1)
            for x_degree in range(total_degree, -1, -1)
        ],
        dtype=np.int64,
    )


def _monomial_values(scaled_points: NDArray, exponents: NDArray) -> NDArray:
    return np.prod(scaled_points[:, None, :] ** exponents[None, :, :], axis=2)


def _polynomial_data(
    points: NDArray,
    center: NDArray,
    scales: NDArray,
    degree: int,
) -> tuple[NDArray, NDArray, NDArray]:
    """Values, gradients, and Laplacians of a shifted/scaled total-degree basis."""
    scaled_points = (points - center[None, :]) / scales[None, :]
    if degree == 2:
        sx = scaled_points[:, 0]
        sy = scaled_points[:, 1]
        values = np.column_stack([np.ones(len(points)), sx, sy, sx**2, sx * sy, sy**2])
        quadratic_gradients: NDArray = np.zeros((len(points), 6, 2), dtype=np.float64)
        quadratic_gradients[:, 1, 0] = 1.0 / scales[0]
        quadratic_gradients[:, 2, 1] = 1.0 / scales[1]
        quadratic_gradients[:, 3, 0] = 2.0 * sx / scales[0]
        quadratic_gradients[:, 4, 0] = sy / scales[0]
        quadratic_gradients[:, 4, 1] = sx / scales[1]
        quadratic_gradients[:, 5, 1] = 2.0 * sy / scales[1]
        quadratic_laplacians: NDArray = np.zeros((len(points), 6), dtype=np.float64)
        quadratic_laplacians[:, 3] = 2.0 / scales[0] ** 2
        quadratic_laplacians[:, 5] = 2.0 / scales[1] ** 2
        return values, quadratic_gradients, quadratic_laplacians

    exponents = _polynomial_exponents(degree)
    values = _monomial_values(scaled_points, exponents)
    gradients: NDArray = np.zeros((len(points), len(exponents), 2), dtype=np.float64)
    laplacians: NDArray = np.zeros((len(points), len(exponents)), dtype=np.float64)
    for basis_index, exponent in enumerate(exponents):
        for direction in range(2):
            if exponent[direction] > 0:
                gradient_exponent = exponent.copy()
                gradient_exponent[direction] -= 1
                gradients[:, basis_index, direction] = (
                    exponent[direction]
                    / scales[direction]
                    * _monomial_values(scaled_points, gradient_exponent[None, :])[:, 0]
                )
            if exponent[direction] > 1:
                laplacian_exponent = exponent.copy()
                laplacian_exponent[direction] -= 2
                laplacians[:, basis_index] += (
                    exponent[direction]
                    * (exponent[direction] - 1)
                    / scales[direction] ** 2
                    * _monomial_values(scaled_points, laplacian_exponent[None, :])[:, 0]
                )
    return values, gradients, laplacians


def _as_vector(values: NDArray, size: int, name: str) -> NDArray:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},), got {array.shape}.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values.")
    return array


class CentroidalVCISCNIOperatorSandbox:
    """Dense research sandbox for the centroidal VCI-SCNI operator gate.

    The MLS trial basis is two-dimensional and can have degree two through four.
    The shifted/scaled variational correction independently targets degrees two
    through four. The sandbox
    supports rectangular clipping and the existing polygonal SDF-chord clipping.
    Dense diagnostics are intentional: promotion to a production sparse backend
    is blocked until the operator gate passes across cloud families and refinement.
    """

    def __init__(
        self,
        nodes: NDArray,
        rho: float,
        bounds: list[tuple[float, float]],
        *,
        degree: int = 2,
        vci_degree: int = 2,
        sdf: Callable[[NDArray], NDArray] | None = None,
        backend: str = "numpy",
        n_edge_gauss: int = 4,
        vci_support_radius: float | None = None,
        trial_gradient_mode: str = "scni",
        test_gradient_base: str = "trial",
        polynomial_null_stabilization: float = 0.0,
        stabilization_metric: str = "euclidean",
        stabilization_support_radius: float | None = None,
        max_stabilization_condition: float = 1e8,
        rank_tolerance: float = 1e-11,
        max_local_condition: float = 1e8,
        max_mass_condition: float = 1e6,
        max_correction_ratio: float = 1.0,
        min_gauge_coercivity: float = 0.0,
        patch_tolerance: float = 1e-9,
    ) -> None:
        self._nodes = np.asarray(nodes, dtype=np.float64)
        if self._nodes.ndim != 2 or self._nodes.shape[1] != 2:
            raise ValueError(f"Centroidal VCI-SCNI requires nodes with shape (N, 2), got {self._nodes.shape}.")
        if not np.all(np.isfinite(self._nodes)):
            raise ValueError("Centroidal VCI-SCNI nodes must all be finite.")
        if len(np.unique(self._nodes, axis=0)) != len(self._nodes):
            raise ValueError("Centroidal VCI-SCNI does not admit duplicate nodes.")
        if not isinstance(degree, Integral) or degree not in (2, 3, 4):
            raise ValueError(f"Centroidal VCI-SCNI requires MLS degree in {{2, 3, 4}}, got {degree}.")
        required_nodes = len(monomial_exponents(2, int(degree)))
        if len(self._nodes) < required_nodes:
            raise ValueError(
                f"Centroidal VCI-SCNI degree-{degree} MLS requires at least {required_nodes} nodes, "
                f"got {len(self._nodes)}."
            )
        if not isinstance(vci_degree, Integral) or vci_degree not in (2, 3, 4):
            raise ValueError(f"Centroidal VCI-SCNI requires vci_degree in {{2, 3, 4}}, got {vci_degree}.")
        if backend != "numpy":
            raise ValueError("Centroidal VCI-SCNI admission diagnostics require backend='numpy'.")
        if trial_gradient_mode not in {"scni", "pointwise"}:
            raise ValueError(
                "Centroidal VCI-SCNI requires trial_gradient_mode in {'scni', 'pointwise'}, "
                f"got {trial_gradient_mode!r}."
            )
        if test_gradient_base not in {"trial", "scni"}:
            raise ValueError(
                f"Centroidal VCI-SCNI requires test_gradient_base in {{'trial', 'scni'}}, got {test_gradient_base!r}."
            )
        if not np.isfinite(polynomial_null_stabilization) or polynomial_null_stabilization < 0.0:
            raise ValueError(
                f"polynomial_null_stabilization must be finite and nonnegative, got {polynomial_null_stabilization}."
            )
        if stabilization_metric not in {"euclidean", "edge_energy"}:
            raise ValueError(
                "Centroidal VCI-SCNI requires stabilization_metric in "
                f"{{'euclidean', 'edge_energy'}}, got {stabilization_metric!r}."
            )
        if stabilization_metric == "edge_energy" and stabilization_support_radius is not None:
            raise ValueError(
                "stabilization_support_radius is not defined for stabilization_metric='edge_energy'; "
                "the active edge MLS stencil determines its support."
            )
        if stabilization_support_radius is not None and (
            not np.isfinite(stabilization_support_radius) or stabilization_support_radius <= 0.0
        ):
            raise ValueError(
                f"stabilization_support_radius must be finite and positive, got {stabilization_support_radius}."
            )
        if not np.isfinite(max_stabilization_condition) or max_stabilization_condition <= 1.0:
            raise ValueError(
                f"max_stabilization_condition must be finite and greater than one, got {max_stabilization_condition}."
            )
        if not np.isfinite(rho) or rho <= 0.0:
            raise ValueError(f"Centroidal VCI-SCNI requires rho > 0, got {rho}.")
        if len(bounds) != 2 or any(len(bound) != 2 or bound[1] <= bound[0] for bound in bounds):
            raise ValueError(f"Centroidal VCI-SCNI requires two increasing bounds, got {bounds}.")
        bounds_array = np.asarray(bounds, dtype=np.float64)
        if not np.all(np.isfinite(bounds_array)):
            raise ValueError(f"Centroidal VCI-SCNI bounds must be finite, got {bounds}.")
        bound_tolerance = 1e-12 * max(float(np.max(bounds_array[:, 1] - bounds_array[:, 0])), 1.0)
        outside_bounds = np.any(
            (self._nodes < bounds_array[:, 0] - bound_tolerance) | (self._nodes > bounds_array[:, 1] + bound_tolerance)
        )
        if outside_bounds:
            raise ValueError("Centroidal VCI-SCNI nodes must lie inside bounds.")
        if sdf is not None:
            signed_distances = np.asarray(sdf(self._nodes), dtype=np.float64).ravel()
            if signed_distances.shape != (len(self._nodes),) or not np.all(np.isfinite(signed_distances)):
                raise ValueError("Centroidal VCI-SCNI sdf(nodes) must return one finite value per node.")
            if np.any(signed_distances > bound_tolerance):
                raise ValueError("Centroidal VCI-SCNI nodes must lie inside the sdf domain.")
        if rank_tolerance <= 0.0 or rank_tolerance >= 1.0:
            raise ValueError(f"rank_tolerance must lie in (0, 1), got {rank_tolerance}.")

        self._n_dof = len(self._nodes)
        self._rho = float(rho)
        self._bounds = [(float(lower), float(upper)) for lower, upper in bounds]
        self._degree = int(degree)
        self._exponents = monomial_exponents(2, self._degree)
        self._trial_gradient_mode = trial_gradient_mode
        self._test_gradient_base = test_gradient_base
        self._polynomial_null_stabilization = float(polynomial_null_stabilization)
        self._stabilization_metric = stabilization_metric
        self._stabilization_support_radius = float(
            stabilization_support_radius if stabilization_support_radius is not None else rho
        )
        self._max_stabilization_condition = float(max_stabilization_condition)
        self._vci_degree = int(vci_degree)
        self._vci_constraint_count = len(_polynomial_exponents(self._vci_degree)) - 1
        self._correction_exponents = _polynomial_exponents(self._vci_degree - 1)
        self._rank_tolerance = float(rank_tolerance)
        self._max_local_condition = float(max_local_condition)
        self._max_mass_condition = float(max_mass_condition)
        self._max_correction_ratio = float(max_correction_ratio)
        self._min_gauge_coercivity = float(min_gauge_coercivity)
        self._patch_tolerance = float(patch_tolerance)
        self._vci_support_radius = float(vci_support_radius if vci_support_radius is not None else rho)
        if self._vci_support_radius <= 0.0:
            raise ValueError(f"vci_support_radius must be positive, got {self._vci_support_radius}.")

        cells = clipped_voronoi_cells(self._nodes, self._bounds, sdf)
        self._centroids = np.vstack([cell.centroid for cell in cells])
        self._weights = np.array([cell.area for cell in cells], dtype=np.float64)
        edge_rule = _edge_quadrature(cells, n_edge_gauss)
        self._edge_points = edge_rule.points
        self._edge_normal_weights = edge_rule.normal_weights
        self._edge_cell_indices = edge_rule.cell_indices
        edge_measures = np.linalg.norm(edge_rule.normal_weights, axis=1)
        cell_perimeters = np.bincount(
            edge_rule.cell_indices,
            weights=edge_measures,
            minlength=self._n_dof,
        )
        if np.any(cell_perimeters <= 0.0):
            bad = int(np.argmin(cell_perimeters))
            raise ValueError(f"VCI cell {bad} has nonpositive boundary measure.")
        self._edge_energy_weights = (
            edge_measures * self._weights[edge_rule.cell_indices] / cell_perimeters[edge_rule.cell_indices]
        )

        self._E, centroid_gradients = shape_functions_and_grads(
            self._centroids,
            self._nodes,
            self._rho,
            self._exponents,
            backend,
            check_conditioning=True,
        )
        pointwise_gradients = [centroid_gradients[:, :, direction] for direction in range(2)]
        self._edge_phi, edge_point_gradients = shape_functions_and_grads(
            self._edge_points,
            self._nodes,
            self._rho,
            self._exponents,
            backend,
            check_conditioning=True,
        )
        scni_gradients = self._assemble_scni_gradient(edge_rule)
        self._G = scni_gradients if trial_gradient_mode == "scni" else pointwise_gradients
        base_test_gradients = self._G if test_gradient_base == "trial" else scni_gradients
        self._Gbar, ranks, conditions, correction_ratios = self._assemble_test_gradient(base_test_gradients)
        self._M = self._E.T @ (self._weights[:, None] * self._E)
        self._raw_K = sum(
            test_gradient.T @ (self._weights[:, None] * trial_gradient)
            for test_gradient, trial_gradient in zip(self._Gbar, self._G, strict=True)
        )
        if self._polynomial_null_stabilization > 0.0:
            (
                self._stabilization,
                stabilization_ranks,
                stabilization_conditions,
                stabilization_support_counts,
                stabilization_patch_defect,
            ) = self._assemble_polynomial_null_stabilization(edge_point_gradients)
        else:
            self._stabilization = np.zeros_like(self._raw_K)
            stabilization_ranks = []
            stabilization_conditions = []
            stabilization_support_counts = []
            stabilization_patch_defect = 0.0
        self._K = self._raw_K + self._polynomial_null_stabilization * self._stabilization
        self._diagnostics = self._compute_diagnostics(
            ranks,
            conditions,
            correction_ratios,
            stabilization_ranks,
            stabilization_conditions,
            stabilization_support_counts,
            stabilization_patch_defect,
        )
        self._enforce_admission_gates()

    def _assemble_scni_gradient(self, edge_rule: _EdgeQuadrature) -> list[NDArray]:
        gradients = []
        for direction in range(2):
            accumulator: NDArray = np.zeros((self._n_dof, self._n_dof), dtype=np.float64)
            contribution = edge_rule.normal_weights[:, direction, None] * self._edge_phi
            np.add.at(accumulator, edge_rule.cell_indices, contribution)
            accumulator /= self._weights[:, None]
            gradients.append(accumulator)
        return gradients

    def _assemble_test_gradient(
        self,
        base_test_gradients: list[NDArray],
    ) -> tuple[list[NDArray], list[int], list[float], list[float]]:
        test_gradients = [gradient.copy() for gradient in base_test_gradients]
        nearest_distances, _ = cKDTree(self._nodes).query(self._nodes, k=2)
        local_scales = nearest_distances[:, 1]
        if np.any(local_scales <= 1e-14):
            bad = int(np.argmin(local_scales))
            raise ValueError(f"VCI local scale vanishes at node {bad}; duplicate or coincident nodes are not admitted.")

        ranks = []
        conditions = []
        correction_ratios = []
        correction_basis_count = len(self._correction_exponents)
        for i in range(self._n_dof):
            distances = np.linalg.norm(self._centroids - self._nodes[i], axis=1)
            support = distances <= self._vci_support_radius * (1.0 + 10.0 * np.finfo(float).eps)
            support_count = int(np.count_nonzero(support))
            if support_count < correction_basis_count:
                raise ValueError(
                    f"VCI local constraint rank failure at node {i}: support_cells={support_count} "
                    f"< correction_basis_size={correction_basis_count} "
                    f"(support_radius={self._vci_support_radius:.6g})."
                )
            h_i = float(local_scales[i])
            local_scale = np.array([h_i, h_i], dtype=np.float64)
            _, centroid_gradients, centroid_laplacians = _polynomial_data(
                self._centroids,
                self._nodes[i],
                local_scale,
                self._vci_degree,
            )
            _, edge_gradients, _ = _polynomial_data(
                self._edge_points,
                self._nodes[i],
                local_scale,
                self._vci_degree,
            )

            shifted = (self._centroids[support] - self._nodes[i]) / h_i
            correction_basis = _monomial_values(shifted, self._correction_exponents)
            weighted_gram = correction_basis.T @ (self._weights[support, None] * correction_basis)
            correction_metric = linalg.block_diag(weighted_gram, weighted_gram)
            try:
                metric_cholesky = np.linalg.cholesky(correction_metric)
            except np.linalg.LinAlgError as error:
                raise np.linalg.LinAlgError(
                    f"VCI local correction metric is singular at node {i} with {support_count} support cells."
                ) from error

            constraint = np.einsum(
                "a,al,akd->kdl",
                self._weights[support],
                correction_basis,
                centroid_gradients[support, 1:, :],
            ).reshape(self._vci_constraint_count, 2 * correction_basis_count)
            whitened_constraint = linalg.solve_triangular(
                metric_cholesky,
                constraint.T,
                lower=True,
                check_finite=False,
            ).T
            singular_values = np.linalg.svd(whitened_constraint, compute_uv=False)
            relative_cutoff = self._rank_tolerance * singular_values[0] if singular_values.size else np.inf
            rank = int(np.count_nonzero(singular_values > relative_cutoff))
            ranks.append(rank)
            if rank < self._vci_constraint_count:
                raise ValueError(
                    f"VCI local constraint rank failure at node {i}: rank={rank}, "
                    f"required={self._vci_constraint_count}, "
                    f"support_cells={support_count}, support_radius={self._vci_support_radius:.6g}."
                )
            condition = float(singular_values[0] / singular_values[-1])
            conditions.append(condition)
            if not np.isfinite(condition) or condition > self._max_local_condition:
                raise np.linalg.LinAlgError(
                    f"VCI local constraint is ill-conditioned at node {i}: cond={condition:.3e} "
                    f"> {self._max_local_condition:.3e}."
                )

            base_gradient = np.column_stack([gradient[:, i] for gradient in base_test_gradients])
            base_term = np.einsum(
                "a,ad,akd->k",
                self._weights,
                base_gradient,
                centroid_gradients,
            )
            volume_term = np.einsum(
                "a,a,ak->k",
                self._weights,
                self._E[:, i],
                centroid_laplacians,
            )
            boundary_term = np.einsum(
                "q,qkd,qd->k",
                self._edge_phi[:, i],
                edge_gradients,
                self._edge_normal_weights,
            )
            residual = (-volume_term + boundary_term - base_term)[1:]

            multiplier_matrix = whitened_constraint @ whitened_constraint.T
            multipliers = np.linalg.solve(multiplier_matrix, residual)
            whitened_coefficients = whitened_constraint.T @ multipliers
            coefficients = linalg.solve_triangular(
                metric_cholesky.T,
                whitened_coefficients,
                lower=False,
                check_finite=False,
            )
            correction_x = correction_basis @ coefficients[:correction_basis_count]
            correction_y = correction_basis @ coefficients[correction_basis_count:]
            test_gradients[0][support, i] += correction_x
            test_gradients[1][support, i] += correction_y

            correction_norm = np.sqrt(np.sum(self._weights[support] * (correction_x**2 + correction_y**2)))
            base_norm = np.sqrt(np.sum(self._weights * np.sum(base_gradient**2, axis=1)))
            if base_norm <= 1e-14:
                raise ValueError(f"VCI base test-gradient norm vanishes at node {i}.")
            correction_ratios.append(float(correction_norm / base_norm))
        return test_gradients, ranks, conditions, correction_ratios

    def _assemble_polynomial_null_stabilization(
        self,
        edge_point_gradients: NDArray,
    ) -> tuple[NDArray, list[int], list[float], list[int], float]:
        """Assemble a local PSD stabilization that annihilates target polynomials.

        The Euclidean metric divides each stencil projector by its support count;
        this remains the challenged baseline. The edge-energy metric sandwiches
        each cell's positive boundary-gradient energy with the same local
        polynomial-complement projector. Its boundary weights sum to the cell
        area, so the metric has the scale of a volume gradient energy.
        """
        stabilization = np.zeros((self._n_dof, self._n_dof), dtype=np.float64)
        tree = cKDTree(self._nodes)
        nearest_distances, _ = tree.query(self._nodes, k=2)
        local_scales = nearest_distances[:, 1]
        polynomial_count = len(_polynomial_exponents(self._vci_degree))
        ranks = []
        conditions = []
        support_counts = []
        for i, center in enumerate(self._nodes):
            if self._stabilization_metric == "euclidean":
                support = np.sort(
                    np.asarray(tree.query_ball_point(center, self._stabilization_support_radius), dtype=np.int64)
                )
                edge_rows = None
            else:
                edge_rows = np.flatnonzero(self._edge_cell_indices == i)
                local_edge_gradients = edge_point_gradients[edge_rows]
                support = np.flatnonzero(np.max(np.abs(local_edge_gradients), axis=(0, 2)) > 0.0)
            support_count = len(support)
            support_counts.append(support_count)
            if support_count < polynomial_count:
                support_description = (
                    f"support_radius={self._stabilization_support_radius:.6g}"
                    if edge_rows is None
                    else "support=active edge MLS stencil"
                )
                raise ValueError(
                    f"Polynomial-null stabilization rank failure at node {i}: support_nodes={support_count} "
                    f"< polynomial_basis_size={polynomial_count} "
                    f"({support_description})."
                )
            local_points = (self._nodes[support] - center[None, :]) / local_scales[i]
            polynomial_values, _, _ = _polynomial_data(
                local_points,
                np.zeros(2, dtype=np.float64),
                np.ones(2, dtype=np.float64),
                self._vci_degree,
            )
            left_vectors, singular_values, _ = np.linalg.svd(polynomial_values, full_matrices=False)
            relative_cutoff = self._rank_tolerance * singular_values[0]
            rank = int(np.count_nonzero(singular_values > relative_cutoff))
            ranks.append(rank)
            if rank < polynomial_count:
                support_description = (
                    f"support_radius={self._stabilization_support_radius:.6g}"
                    if edge_rows is None
                    else "support=active edge MLS stencil"
                )
                raise ValueError(
                    f"Polynomial-null stabilization rank failure at node {i}: rank={rank}, "
                    f"required={polynomial_count}, support_nodes={support_count}, "
                    f"{support_description}."
                )
            condition = float(singular_values[0] / singular_values[-1])
            conditions.append(condition)
            if not np.isfinite(condition) or condition > self._max_stabilization_condition:
                raise np.linalg.LinAlgError(
                    f"Polynomial-null stabilization is ill-conditioned at node {i}: cond={condition:.3e} "
                    f"> {self._max_stabilization_condition:.3e}."
                )
            polynomial_basis = left_vectors[:, :polynomial_count]
            residual_projector = np.eye(support_count) - polynomial_basis @ polynomial_basis.T
            if edge_rows is None:
                local_stabilization = residual_projector / support_count
            else:
                local_edge_gradients = edge_point_gradients[edge_rows][:, support, :]
                local_metric = np.einsum(
                    "q,qid,qjd->ij",
                    self._edge_energy_weights[edge_rows],
                    local_edge_gradients,
                    local_edge_gradients,
                )
                local_stabilization = residual_projector @ local_metric @ residual_projector
            stabilization[np.ix_(support, support)] += local_stabilization

        node_values, _, _, _ = self._global_patch_data(self._vci_degree)
        patch_defect = float(np.max(np.abs(stabilization @ node_values)))
        return stabilization, ranks, conditions, support_counts, patch_defect

    def _global_patch_data(self, degree: int) -> tuple[NDArray, NDArray, NDArray, NDArray]:
        center = np.array(
            [0.5 * (lower + upper) for lower, upper in self._bounds],
            dtype=np.float64,
        )
        scales = np.array([upper - lower for lower, upper in self._bounds], dtype=np.float64)
        node_values, _, _ = _polynomial_data(self._nodes, center, scales, degree)
        _, centroid_gradients, centroid_laplacians = _polynomial_data(self._centroids, center, scales, degree)
        _, edge_gradients, _ = _polynomial_data(self._edge_points, center, scales, degree)
        return node_values, centroid_gradients, centroid_laplacians, edge_gradients

    def _weak_patch_defect(self, test_gradients: list[NDArray], degree: int) -> float:
        _, centroid_gradients, centroid_laplacians, edge_gradients = self._global_patch_data(degree)
        volume_gradient = sum(
            gradient.T @ (self._weights[:, None] * centroid_gradients[:, :, direction])
            for direction, gradient in enumerate(test_gradients)
        )
        volume_laplacian = self._E.T @ (self._weights[:, None] * centroid_laplacians)
        normal_derivatives = np.einsum("qkd,qd->qk", edge_gradients, self._edge_normal_weights)
        boundary_flux = self._edge_phi.T @ normal_derivatives
        return float(np.max(np.abs(volume_gradient + volume_laplacian - boundary_flux)))

    def _discrete_patch_defect(self) -> float:
        node_values, _, centroid_laplacians, edge_gradients = self._global_patch_data(self._vci_degree)
        volume_laplacian = self._E.T @ (self._weights[:, None] * centroid_laplacians)
        normal_derivatives = np.einsum("qkd,qd->qk", edge_gradients, self._edge_normal_weights)
        boundary_flux = self._edge_phi.T @ normal_derivatives
        return float(np.max(np.abs(self._K @ node_values + volume_laplacian - boundary_flux)))

    def _compute_diagnostics(
        self,
        ranks: list[int],
        conditions: list[float],
        correction_ratios: list[float],
        stabilization_ranks: list[int],
        stabilization_conditions: list[float],
        stabilization_support_counts: list[int],
        stabilization_patch_defect: float,
    ) -> VCIAdmissionDiagnostics:
        one = np.ones(self._n_dof)
        quadratic_values, quadratic_gradients, _, _ = self._global_patch_data(2)
        quadratic_gradient_defect = max(
            float(np.max(np.abs(gradient @ quadratic_values - quadratic_gradients[:, :, direction])))
            for direction, gradient in enumerate(self._G)
        )
        vci_values, vci_gradients, _, _ = self._global_patch_data(self._vci_degree)
        vci_gradient_defect = max(
            float(np.max(np.abs(gradient @ vci_values - vci_gradients[:, :, direction])))
            for direction, gradient in enumerate(self._G)
        )
        mean_constraint = self._M @ one
        gauge_basis = linalg.null_space(mean_constraint[None, :])
        gauge_mass = gauge_basis.T @ self._M @ gauge_basis
        symmetric_stiffness = 0.5 * (self._K + self._K.T)
        gauge_stiffness = gauge_basis.T @ symmetric_stiffness @ gauge_basis
        gauge_coercivity = float(
            linalg.eigvalsh(gauge_stiffness, gauge_mass, subset_by_index=[0, 0], check_finite=False)[0]
        )
        if self._polynomial_null_stabilization == 0.0:
            raw_gauge_coercivity = gauge_coercivity
        else:
            raw_symmetric_stiffness = 0.5 * (self._raw_K + self._raw_K.T)
            raw_gauge_stiffness = gauge_basis.T @ raw_symmetric_stiffness @ gauge_basis
            raw_gauge_coercivity = float(
                linalg.eigvalsh(raw_gauge_stiffness, gauge_mass, subset_by_index=[0, 0], check_finite=False)[0]
            )
        gauge_cholesky = np.linalg.cholesky(gauge_mass)
        reduced_operator = gauge_basis.T @ self._K @ gauge_basis
        left_scaled = linalg.solve_triangular(
            gauge_cholesky,
            reduced_operator,
            lower=True,
            check_finite=False,
        )
        mass_scaled_operator = linalg.solve_triangular(
            gauge_cholesky,
            left_scaled.T,
            lower=True,
            check_finite=False,
        ).T
        gauge_smallest_singular = float(np.linalg.svd(mass_scaled_operator, compute_uv=False)[-1])
        stiffness_singular_values = np.linalg.svd(self._K, compute_uv=False)
        nullity_threshold = 1e-10 * float(stiffness_singular_values[0])
        stiffness_nullity = int(np.count_nonzero(stiffness_singular_values <= nullity_threshold))
        return VCIAdmissionDiagnostics(
            local_rank_min=min(ranks),
            local_condition_max=max(conditions),
            correction_ratio_median=float(np.median(correction_ratios)),
            correction_ratio_max=max(correction_ratios),
            mass_condition=float(np.linalg.cond(self._M)),
            value_constant_defect=float(np.max(np.abs(self._E @ one - 1.0))),
            trial_constant_defect=max(float(np.max(np.abs(gradient @ one))) for gradient in self._G),
            quadratic_gradient_defect=quadratic_gradient_defect,
            vci_trial_gradient_defect=vci_gradient_defect,
            plain_patch_defect=self._weak_patch_defect(self._G, self._vci_degree),
            corrected_patch_defect=self._weak_patch_defect(self._Gbar, self._vci_degree),
            discrete_patch_defect=self._discrete_patch_defect(),
            stabilization_rank_min=min(stabilization_ranks, default=0),
            stabilization_condition_max=max(stabilization_conditions, default=0.0),
            stabilization_support_count_min=min(stabilization_support_counts, default=0),
            stabilization_support_count_max=max(stabilization_support_counts, default=0),
            stabilization_patch_defect=stabilization_patch_defect,
            right_constant_defect=float(np.max(np.abs(self._K @ one))),
            left_constant_defect=float(np.max(np.abs(one @ self._K))),
            raw_gauge_coercivity_min=raw_gauge_coercivity,
            gauge_coercivity_min=gauge_coercivity,
            gauge_smallest_singular_value=gauge_smallest_singular,
            stiffness_nullity=stiffness_nullity,
        )

    def _enforce_admission_gates(self) -> None:
        diagnostics = self._diagnostics
        exactness_defects = {
            "value partition of unity": diagnostics.value_constant_defect,
            "trial-gradient constants": diagnostics.trial_constant_defect,
            "quadratic centroid gradient": diagnostics.quadratic_gradient_defect,
            f"degree-{self._vci_degree} corrected target weak patch": diagnostics.corrected_patch_defect,
            "polynomial-null stabilization patch": diagnostics.stabilization_patch_defect,
            "right constant nullity": diagnostics.right_constant_defect,
        }
        failed_exactness = {name: value for name, value in exactness_defects.items() if value > self._patch_tolerance}
        if failed_exactness:
            details = ", ".join(f"{name}={value:.3e}" for name, value in failed_exactness.items())
            raise ValueError(
                f"Centroidal VCI-SCNI exactness gate failed ({details}); tolerance={self._patch_tolerance:.3e}."
            )
        if not np.isfinite(diagnostics.mass_condition) or diagnostics.mass_condition > self._max_mass_condition:
            raise np.linalg.LinAlgError(
                f"Centroidal VCI-SCNI mass gate failed: cond(M)={diagnostics.mass_condition:.3e} "
                f"> {self._max_mass_condition:.3e}."
            )
        if diagnostics.correction_ratio_max > self._max_correction_ratio:
            raise ValueError(
                f"Centroidal VCI-SCNI correction gate failed: max ratio={diagnostics.correction_ratio_max:.3e} "
                f"> {self._max_correction_ratio:.3e}."
            )
        if diagnostics.gauge_coercivity_min <= self._min_gauge_coercivity:
            raise ValueError(
                f"Centroidal VCI-SCNI gauge-coercivity gate failed: lambda_min="
                f"{diagnostics.gauge_coercivity_min:.3e} <= {self._min_gauge_coercivity:.3e}."
            )
        if diagnostics.stiffness_nullity != 1:
            raise ValueError(
                f"Centroidal VCI-SCNI kernel gate failed: nullity={diagnostics.stiffness_nullity}, expected 1."
            )

    @property
    def n_dof(self) -> int:
        return self._n_dof

    @property
    def dim(self) -> int:
        return 2

    @property
    def vci_degree(self) -> int:
        return self._vci_degree

    @property
    def degree(self) -> int:
        return self._degree

    @property
    def trial_gradient_mode(self) -> str:
        return self._trial_gradient_mode

    @property
    def test_gradient_base(self) -> str:
        return self._test_gradient_base

    @property
    def polynomial_null_stabilization(self) -> float:
        return self._polynomial_null_stabilization

    @property
    def stabilization_metric(self) -> str:
        return self._stabilization_metric

    @property
    def nodes(self) -> NDArray:
        return self._nodes

    @property
    def evaluation_points(self) -> NDArray:
        return self._centroids

    @property
    def centroid_evaluation_count(self) -> int:
        return len(self._centroids)

    @property
    def edge_evaluation_count(self) -> int:
        return len(self._edge_points)

    @property
    def weights(self) -> NDArray:
        return self._weights

    @property
    def diagnostics(self) -> VCIAdmissionDiagnostics:
        return self._diagnostics

    def value_operator(self) -> sparse.csr_matrix:
        return sparse.csr_matrix(self._E)

    def trial_gradient(self) -> list[sparse.csr_matrix]:
        return [sparse.csr_matrix(gradient) for gradient in self._G]

    def test_gradient(self) -> list[sparse.csr_matrix]:
        return [sparse.csr_matrix(gradient) for gradient in self._Gbar]

    def mass(self) -> sparse.csr_matrix:
        return sparse.csr_matrix(self._M)

    def stiffness(self) -> sparse.csr_matrix:
        return sparse.csr_matrix(self._K)

    def stabilization(self) -> sparse.csr_matrix:
        return sparse.csr_matrix(self._stabilization)

    def reconstructed_density(self, coefficients: NDArray) -> NDArray:
        vector = _as_vector(coefficients, self._n_dof, "Density coefficients")
        return self._E @ vector

    def physical_mass(self, coefficients: NDArray) -> float:
        return float(self._weights @ self.reconstructed_density(coefficients))

    def left_null_vector(self) -> NDArray:
        left_singular_vectors, _, _ = np.linalg.svd(self._K)
        vector = left_singular_vectors[:, -1]
        normalization = float(vector @ np.ones(self._n_dof))
        if abs(normalization) <= 1e-14:
            raise ValueError("Centroidal VCI-SCNI left null vector has zero constant pairing.")
        return vector / normalization


class PairedMFGOperatorSandbox:
    """Nonlinear nodal MFG residual with one analytic Jacobian and its transpose."""

    def __init__(
        self,
        discretization: CentroidalVCISCNIOperatorSandbox,
        *,
        diffusion: float,
        hamiltonian: Callable[[NDArray, NDArray], NDArray],
        hamiltonian_gradient: Callable[[NDArray, NDArray], NDArray],
        stabilization_flux: Callable[[NDArray, NDArray], NDArray] | None = None,
        stabilization_jacobian: Callable[[NDArray, NDArray], NDArray] | None = None,
    ) -> None:
        if not np.isfinite(diffusion) or diffusion < 0.0:
            raise ValueError(f"diffusion must be finite and nonnegative, got {diffusion}.")
        if (stabilization_flux is None) != (stabilization_jacobian is None):
            raise ValueError("stabilization_flux and stabilization_jacobian must be supplied together.")
        self._discretization = discretization
        self._diffusion = float(diffusion)
        self._hamiltonian = hamiltonian
        self._hamiltonian_gradient = hamiltonian_gradient
        self._stabilization_flux = stabilization_flux
        self._stabilization_jacobian = stabilization_jacobian

    def momentum(self, state: NDArray) -> NDArray:
        vector = _as_vector(state, self._discretization.n_dof, "HJB state")
        return np.column_stack([gradient @ vector for gradient in self._discretization._G])

    def _hamiltonian_data(self, momentum: NDArray) -> tuple[NDArray, NDArray]:
        points = self._discretization.evaluation_points
        count = self._discretization.n_dof
        values = np.asarray(self._hamiltonian(points, momentum), dtype=np.float64)
        gradients = np.asarray(self._hamiltonian_gradient(points, momentum), dtype=np.float64)
        if values.shape != (count,):
            raise ValueError(f"Hamiltonian values must have shape ({count},), got {values.shape}.")
        if gradients.shape != (count, 2):
            raise ValueError(f"Hamiltonian gradient must have shape ({count}, 2), got {gradients.shape}.")
        if not np.all(np.isfinite(values)) or not np.all(np.isfinite(gradients)):
            raise ValueError("Hamiltonian evaluation contains non-finite values.")
        return values, gradients

    def _stabilization_data(self, momentum: NDArray) -> tuple[NDArray, NDArray] | None:
        if self._stabilization_flux is None or self._stabilization_jacobian is None:
            return None
        points = self._discretization.evaluation_points
        count = self._discretization.n_dof
        flux = np.asarray(self._stabilization_flux(points, momentum), dtype=np.float64)
        jacobian = np.asarray(self._stabilization_jacobian(points, momentum), dtype=np.float64)
        if flux.shape != (count, 2):
            raise ValueError(f"Stabilization flux must have shape ({count}, 2), got {flux.shape}.")
        if jacobian.shape != (count, 2, 2):
            raise ValueError(f"Stabilization Jacobian must have shape ({count}, 2, 2), got {jacobian.shape}.")
        if not np.all(np.isfinite(flux)) or not np.all(np.isfinite(jacobian)):
            raise ValueError("Stabilization evaluation contains non-finite values.")
        return flux, jacobian

    def residual(self, state: NDArray) -> NDArray:
        vector = _as_vector(state, self._discretization.n_dof, "HJB state")
        momentum = self.momentum(vector)
        hamiltonian, _ = self._hamiltonian_data(momentum)
        residual = self._diffusion * (self._discretization._K @ vector)
        residual += self._discretization._E.T @ (self._discretization._weights * hamiltonian)
        stabilization = self._stabilization_data(momentum)
        if stabilization is not None:
            flux, _ = stabilization
            for direction in range(2):
                residual += self._discretization._Gbar[direction].T @ (
                    self._discretization._weights * flux[:, direction]
                )
        return residual

    def jacobian(self, state: NDArray) -> NDArray:
        vector = _as_vector(state, self._discretization.n_dof, "HJB state")
        momentum = self.momentum(vector)
        _, hamiltonian_gradient = self._hamiltonian_data(momentum)
        jacobian = self._diffusion * self._discretization._K.copy()
        for direction in range(2):
            weighted_derivative = self._discretization._weights * hamiltonian_gradient[:, direction]
            jacobian += self._discretization._E.T @ (weighted_derivative[:, None] * self._discretization._G[direction])
        stabilization = self._stabilization_data(momentum)
        if stabilization is not None:
            _, stabilization_jacobian = stabilization
            for test_direction in range(2):
                for trial_direction in range(2):
                    weighted_derivative = (
                        self._discretization._weights * stabilization_jacobian[:, test_direction, trial_direction]
                    )
                    jacobian += self._discretization._Gbar[test_direction].T @ (
                        weighted_derivative[:, None] * self._discretization._G[trial_direction]
                    )
        return jacobian

    def forward_spatial(self, state: NDArray, density: NDArray) -> NDArray:
        density_vector = _as_vector(density, self._discretization.n_dof, "Density coefficients")
        return self.jacobian(state).T @ density_vector

    def physical_mass(self, density: NDArray) -> float:
        return self._discretization.physical_mass(density)
