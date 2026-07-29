"""Operator sandbox for centroidal VC2-SCNI and an exactly paired MFG linearization.

This module is deliberately outside ``WeakFormDiscretization``. It is an admission
gate for a candidate Petrov-Galerkin method, not a production solver backend.

The construction uses clipped Voronoi centroids ``c_a`` as sampling points,
``M = E.T @ W @ E`` with ``E[a, i] = phi_i(c_a)``, the SCNI cell-average trial
gradient, and a local variational-consistency correction of the test gradient.
The correction enforces the degree-two weak patch, including boundary fluxes,
through five shifted/scaled constraints per test function.

All nonlinear MFG blocks are derived from one residual. The forward spatial
operator is therefore the transpose of the complete analytic Jacobian; no
independent advection assembly exists here.
"""

from __future__ import annotations

from dataclasses import dataclass
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
class VC2AdmissionDiagnostics:
    """Numerical invariants that decide whether a cloud is admitted by the sandbox."""

    local_rank_min: int
    local_condition_max: float
    correction_ratio_median: float
    correction_ratio_max: float
    mass_condition: float
    value_constant_defect: float
    trial_constant_defect: float
    quadratic_gradient_defect: float
    plain_patch_defect: float
    corrected_patch_defect: float
    right_constant_defect: float
    left_constant_defect: float
    gauge_coercivity_min: float
    gauge_smallest_singular_value: float
    stiffness_nullity: int


@dataclass(frozen=True)
class _EdgeQuadrature:
    points: NDArray
    cell_indices: NDArray
    normal_weights: NDArray


def _edge_quadrature(cells: list[CellGeometry], n_gauss: int) -> _EdgeQuadrature:
    if n_gauss < 2:
        raise ValueError("VC2 edge quadrature requires at least two Gauss points per edge.")
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
        raise ValueError("VC2 edge quadrature found no nondegenerate cell edges.")
    return _EdgeQuadrature(
        points=np.vstack(points),
        cell_indices=np.concatenate(cell_indices),
        normal_weights=np.vstack(normal_weights),
    )


def _p2_data(points: NDArray, center: NDArray, scales: NDArray) -> tuple[NDArray, NDArray, NDArray]:
    """Values, gradients, and Laplacians of a shifted/scaled basis of ``P2``."""
    sx = (points[:, 0] - center[0]) / scales[0]
    sy = (points[:, 1] - center[1]) / scales[1]
    values = np.column_stack([np.ones(len(points)), sx, sy, sx**2, sx * sy, sy**2])
    gradients: NDArray = np.zeros((len(points), 6, 2), dtype=np.float64)
    gradients[:, 1, 0] = 1.0 / scales[0]
    gradients[:, 2, 1] = 1.0 / scales[1]
    gradients[:, 3, 0] = 2.0 * sx / scales[0]
    gradients[:, 4, 0] = sy / scales[0]
    gradients[:, 4, 1] = sx / scales[1]
    gradients[:, 5, 1] = 2.0 * sy / scales[1]
    laplacians: NDArray = np.zeros((len(points), 6), dtype=np.float64)
    laplacians[:, 3] = 2.0 / scales[0] ** 2
    laplacians[:, 5] = 2.0 / scales[1] ** 2
    return values, gradients, laplacians


def _as_vector(values: NDArray, size: int, name: str) -> NDArray:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},), got {array.shape}.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values.")
    return array


class CentroidalVC2SCNIOperatorSandbox:
    """Dense research sandbox for the centroidal VC2-SCNI operator gate.

    The current implementation is two-dimensional and degree-two. It supports
    rectangular clipping and the existing polygonal SDF-chord clipping. Dense
    diagnostics are intentional: promotion to a production sparse backend is
    blocked until the operator gate passes across cloud families and refinement.
    """

    def __init__(
        self,
        nodes: NDArray,
        rho: float,
        bounds: list[tuple[float, float]],
        *,
        degree: int = 2,
        sdf: Callable[[NDArray], NDArray] | None = None,
        backend: str = "numpy",
        n_edge_gauss: int = 4,
        vci_support_radius: float | None = None,
        rank_tolerance: float = 1e-11,
        max_local_condition: float = 1e8,
        max_mass_condition: float = 1e6,
        max_correction_ratio: float = 1.0,
        min_gauge_coercivity: float = 0.0,
        patch_tolerance: float = 1e-9,
    ) -> None:
        self._nodes = np.asarray(nodes, dtype=np.float64)
        if self._nodes.ndim != 2 or self._nodes.shape[1] != 2:
            raise ValueError(f"Centroidal VC2-SCNI requires nodes with shape (N, 2), got {self._nodes.shape}.")
        if len(self._nodes) < 6:
            raise ValueError("Centroidal VC2-SCNI requires at least six nodes for a degree-two MLS basis.")
        if not np.all(np.isfinite(self._nodes)):
            raise ValueError("Centroidal VC2-SCNI nodes must all be finite.")
        if len(np.unique(self._nodes, axis=0)) != len(self._nodes):
            raise ValueError("Centroidal VC2-SCNI does not admit duplicate nodes.")
        if degree != 2:
            raise ValueError(f"Centroidal VC2-SCNI implements degree=2 only, got degree={degree}.")
        if backend != "numpy":
            raise ValueError("Centroidal VC2-SCNI admission diagnostics require backend='numpy'.")
        if not np.isfinite(rho) or rho <= 0.0:
            raise ValueError(f"Centroidal VC2-SCNI requires rho > 0, got {rho}.")
        if len(bounds) != 2 or any(len(bound) != 2 or bound[1] <= bound[0] for bound in bounds):
            raise ValueError(f"Centroidal VC2-SCNI requires two increasing bounds, got {bounds}.")
        bounds_array = np.asarray(bounds, dtype=np.float64)
        if not np.all(np.isfinite(bounds_array)):
            raise ValueError(f"Centroidal VC2-SCNI bounds must be finite, got {bounds}.")
        bound_tolerance = 1e-12 * max(float(np.max(bounds_array[:, 1] - bounds_array[:, 0])), 1.0)
        outside_bounds = np.any(
            (self._nodes < bounds_array[:, 0] - bound_tolerance) | (self._nodes > bounds_array[:, 1] + bound_tolerance)
        )
        if outside_bounds:
            raise ValueError("Centroidal VC2-SCNI nodes must lie inside bounds.")
        if sdf is not None:
            signed_distances = np.asarray(sdf(self._nodes), dtype=np.float64).ravel()
            if signed_distances.shape != (len(self._nodes),) or not np.all(np.isfinite(signed_distances)):
                raise ValueError("Centroidal VC2-SCNI sdf(nodes) must return one finite value per node.")
            if np.any(signed_distances > bound_tolerance):
                raise ValueError("Centroidal VC2-SCNI nodes must lie inside the sdf domain.")
        if rank_tolerance <= 0.0 or rank_tolerance >= 1.0:
            raise ValueError(f"rank_tolerance must lie in (0, 1), got {rank_tolerance}.")

        self._n_dof = len(self._nodes)
        self._rho = float(rho)
        self._bounds = [(float(lower), float(upper)) for lower, upper in bounds]
        self._exponents = monomial_exponents(2, 2)
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

        self._E, _ = shape_functions_and_grads(
            self._centroids,
            self._nodes,
            self._rho,
            self._exponents,
            backend,
            check_conditioning=True,
        )
        self._edge_phi, _ = shape_functions_and_grads(
            self._edge_points,
            self._nodes,
            self._rho,
            self._exponents,
            backend,
            check_conditioning=True,
        )
        self._G = self._assemble_trial_gradient(edge_rule)
        self._Gbar, ranks, conditions, correction_ratios = self._assemble_test_gradient()
        self._M = self._E.T @ (self._weights[:, None] * self._E)
        self._K = sum(
            test_gradient.T @ (self._weights[:, None] * trial_gradient)
            for test_gradient, trial_gradient in zip(self._Gbar, self._G, strict=True)
        )
        self._diagnostics = self._compute_diagnostics(ranks, conditions, correction_ratios)
        self._enforce_admission_gates()

    def _assemble_trial_gradient(self, edge_rule: _EdgeQuadrature) -> list[NDArray]:
        gradients = []
        for direction in range(2):
            accumulator: NDArray = np.zeros((self._n_dof, self._n_dof), dtype=np.float64)
            contribution = edge_rule.normal_weights[:, direction, None] * self._edge_phi
            np.add.at(accumulator, edge_rule.cell_indices, contribution)
            accumulator /= self._weights[:, None]
            gradients.append(accumulator)
        return gradients

    def _assemble_test_gradient(self) -> tuple[list[NDArray], list[int], list[float], list[float]]:
        test_gradients = [gradient.copy() for gradient in self._G]
        nearest_distances, _ = cKDTree(self._nodes).query(self._nodes, k=2)
        local_scales = nearest_distances[:, 1]
        if np.any(local_scales <= 1e-14):
            bad = int(np.argmin(local_scales))
            raise ValueError(f"VC2 local scale vanishes at node {bad}; duplicate or coincident nodes are not admitted.")

        ranks = []
        conditions = []
        correction_ratios = []
        for i in range(self._n_dof):
            distances = np.linalg.norm(self._centroids - self._nodes[i], axis=1)
            support = distances <= self._vci_support_radius * (1.0 + 10.0 * np.finfo(float).eps)
            support_count = int(np.count_nonzero(support))
            if support_count < 3:
                raise ValueError(
                    f"VC2 local constraint rank failure at node {i}: fewer than three support cells "
                    f"(support_cells={support_count}, support_radius={self._vci_support_radius:.6g})."
                )
            h_i = float(local_scales[i])
            local_scale = np.array([h_i, h_i], dtype=np.float64)
            _, centroid_gradients, centroid_laplacians = _p2_data(
                self._centroids,
                self._nodes[i],
                local_scale,
            )
            _, edge_gradients, _ = _p2_data(self._edge_points, self._nodes[i], local_scale)

            shifted = (self._centroids[support] - self._nodes[i]) / h_i
            correction_basis = np.column_stack([np.ones(support_count), shifted])
            weighted_gram = correction_basis.T @ (self._weights[support, None] * correction_basis)
            correction_metric = linalg.block_diag(weighted_gram, weighted_gram)
            try:
                metric_cholesky = np.linalg.cholesky(correction_metric)
            except np.linalg.LinAlgError as error:
                raise np.linalg.LinAlgError(
                    f"VC2 local correction metric is singular at node {i} with {support_count} support cells."
                ) from error

            constraint = np.einsum(
                "a,al,akd->kdl",
                self._weights[support],
                correction_basis,
                centroid_gradients[support, 1:, :],
            ).reshape(5, 6)
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
            if rank < 5:
                raise ValueError(
                    f"VC2 local constraint rank failure at node {i}: rank={rank}, required=5, "
                    f"support_cells={support_count}, support_radius={self._vci_support_radius:.6g}."
                )
            condition = float(singular_values[0] / singular_values[-1])
            conditions.append(condition)
            if not np.isfinite(condition) or condition > self._max_local_condition:
                raise np.linalg.LinAlgError(
                    f"VC2 local constraint is ill-conditioned at node {i}: cond={condition:.3e} "
                    f"> {self._max_local_condition:.3e}."
                )

            base_gradient = np.column_stack([gradient[:, i] for gradient in self._G])
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
            correction_x = correction_basis @ coefficients[:3]
            correction_y = correction_basis @ coefficients[3:]
            test_gradients[0][support, i] += correction_x
            test_gradients[1][support, i] += correction_y

            correction_norm = np.sqrt(np.sum(self._weights[support] * (correction_x**2 + correction_y**2)))
            base_norm = np.sqrt(np.sum(self._weights * np.sum(base_gradient**2, axis=1)))
            if base_norm <= 1e-14:
                raise ValueError(f"VC2 base test-gradient norm vanishes at node {i}.")
            correction_ratios.append(float(correction_norm / base_norm))
        return test_gradients, ranks, conditions, correction_ratios

    def _global_patch_data(self) -> tuple[NDArray, NDArray, NDArray, NDArray]:
        center = np.array(
            [0.5 * (lower + upper) for lower, upper in self._bounds],
            dtype=np.float64,
        )
        scales = np.array([upper - lower for lower, upper in self._bounds], dtype=np.float64)
        node_values, _, _ = _p2_data(self._nodes, center, scales)
        _, centroid_gradients, centroid_laplacians = _p2_data(self._centroids, center, scales)
        _, edge_gradients, _ = _p2_data(self._edge_points, center, scales)
        return node_values, centroid_gradients, centroid_laplacians, edge_gradients

    def _weak_patch_defect(self, test_gradients: list[NDArray]) -> float:
        _, centroid_gradients, centroid_laplacians, edge_gradients = self._global_patch_data()
        volume_gradient = sum(
            gradient.T @ (self._weights[:, None] * centroid_gradients[:, :, direction])
            for direction, gradient in enumerate(test_gradients)
        )
        volume_laplacian = self._E.T @ (self._weights[:, None] * centroid_laplacians)
        normal_derivatives = np.einsum("qkd,qd->qk", edge_gradients, self._edge_normal_weights)
        boundary_flux = self._edge_phi.T @ normal_derivatives
        return float(np.max(np.abs(volume_gradient + volume_laplacian - boundary_flux)))

    def _compute_diagnostics(
        self,
        ranks: list[int],
        conditions: list[float],
        correction_ratios: list[float],
    ) -> VC2AdmissionDiagnostics:
        one = np.ones(self._n_dof)
        node_values, centroid_gradients, _, _ = self._global_patch_data()
        gradient_defect = max(
            float(np.max(np.abs(gradient @ node_values - centroid_gradients[:, :, direction])))
            for direction, gradient in enumerate(self._G)
        )
        symmetric_stiffness = 0.5 * (self._K + self._K.T)
        mean_constraint = self._M @ one
        gauge_basis = linalg.null_space(mean_constraint[None, :])
        gauge_stiffness = gauge_basis.T @ symmetric_stiffness @ gauge_basis
        gauge_mass = gauge_basis.T @ self._M @ gauge_basis
        gauge_coercivity = float(
            linalg.eigvalsh(gauge_stiffness, gauge_mass, subset_by_index=[0, 0], check_finite=False)[0]
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
        return VC2AdmissionDiagnostics(
            local_rank_min=min(ranks),
            local_condition_max=max(conditions),
            correction_ratio_median=float(np.median(correction_ratios)),
            correction_ratio_max=max(correction_ratios),
            mass_condition=float(np.linalg.cond(self._M)),
            value_constant_defect=float(np.max(np.abs(self._E @ one - 1.0))),
            trial_constant_defect=max(float(np.max(np.abs(gradient @ one))) for gradient in self._G),
            quadratic_gradient_defect=gradient_defect,
            plain_patch_defect=self._weak_patch_defect(self._G),
            corrected_patch_defect=self._weak_patch_defect(self._Gbar),
            right_constant_defect=float(np.max(np.abs(self._K @ one))),
            left_constant_defect=float(np.max(np.abs(one @ self._K))),
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
            "corrected weak patch": diagnostics.corrected_patch_defect,
            "right constant nullity": diagnostics.right_constant_defect,
        }
        failed_exactness = {name: value for name, value in exactness_defects.items() if value > self._patch_tolerance}
        if failed_exactness:
            details = ", ".join(f"{name}={value:.3e}" for name, value in failed_exactness.items())
            raise ValueError(
                f"Centroidal VC2-SCNI exactness gate failed ({details}); tolerance={self._patch_tolerance:.3e}."
            )
        if not np.isfinite(diagnostics.mass_condition) or diagnostics.mass_condition > self._max_mass_condition:
            raise np.linalg.LinAlgError(
                f"Centroidal VC2-SCNI mass gate failed: cond(M)={diagnostics.mass_condition:.3e} "
                f"> {self._max_mass_condition:.3e}."
            )
        if diagnostics.correction_ratio_max > self._max_correction_ratio:
            raise ValueError(
                f"Centroidal VC2-SCNI correction gate failed: max ratio={diagnostics.correction_ratio_max:.3e} "
                f"> {self._max_correction_ratio:.3e}."
            )
        if diagnostics.gauge_coercivity_min <= self._min_gauge_coercivity:
            raise ValueError(
                f"Centroidal VC2-SCNI gauge-coercivity gate failed: lambda_min="
                f"{diagnostics.gauge_coercivity_min:.3e} <= {self._min_gauge_coercivity:.3e}."
            )
        if diagnostics.stiffness_nullity != 1:
            raise ValueError(
                f"Centroidal VC2-SCNI kernel gate failed: nullity={diagnostics.stiffness_nullity}, expected 1."
            )

    @property
    def n_dof(self) -> int:
        return self._n_dof

    @property
    def dim(self) -> int:
        return 2

    @property
    def nodes(self) -> NDArray:
        return self._nodes

    @property
    def evaluation_points(self) -> NDArray:
        return self._centroids

    @property
    def weights(self) -> NDArray:
        return self._weights

    @property
    def diagnostics(self) -> VC2AdmissionDiagnostics:
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
            raise ValueError("Centroidal VC2-SCNI left null vector has zero constant pairing.")
        return vector / normalization


class PairedMFGOperatorSandbox:
    """Nonlinear nodal MFG residual with one analytic Jacobian and its transpose."""

    def __init__(
        self,
        discretization: CentroidalVC2SCNIOperatorSandbox,
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
