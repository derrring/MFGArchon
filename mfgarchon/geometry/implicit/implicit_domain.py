"""
Implicit Domain Infrastructure for Meshfree Methods

This module provides dimension-agnostic implicit domain representation via
signed distance functions (SDFs). This enables:

- No mesh generation required (O(d) storage instead of O(N^d))
- Natural obstacle representation via CSG operations
- Dimension-agnostic code (works for 2D, 3D, 4D, ..., 100D)
- Particle-friendly boundary conditions

Key concept: A domain D ⊂ ℝ^d is represented implicitly by a function:
    φ: ℝ^d → ℝ  where  φ(x) < 0 ⟺ x ∈ D

References:
- Osher & Fedkiw (2003): Level Set Methods and Dynamic Implicit Surfaces
- TECHNICAL_REFERENCE_HIGH_DIMENSIONAL_MFG.md Section 4
"""

import warnings
from abc import abstractmethod
from collections.abc import Callable
from itertools import combinations
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from mfgarchon.geometry.base import ImplicitGeometry
from mfgarchon.geometry.boundary.tolerances import SDF_BOUNDARY_TOL
from mfgarchon.geometry.protocol import GeometryType
from mfgarchon.geometry.protocols import (
    SupportsBoundaryDistance,
    SupportsBoundaryNormal,
    SupportsBoundaryProjection,
    SupportsLipschitz,
    SupportsManifold,
)
from mfgarchon.geometry.traits import BoundaryDef, ConnectivityType, StructureType
from mfgarchon.utils.deprecation import deprecated
from mfgarchon.utils.mfg_logging import get_logger

# Module logger
logger = get_logger(__name__)


def _leg_parameters(n_scan: int) -> NDArray[np.float64]:
    """The points of a leg the crossing search samples, as fractions of it: ``n_scan`` steps from the start, one more
    1e-9 of the leg before the end, and the end (`ImplicitDomain._leg_crossing`, `ImplicitDomain.legs_leave`)."""
    return np.concatenate([np.linspace(0.0, 1.0, n_scan + 1)[:-1], [1.0 - 1e-9, 1.0]])


class ImplicitDomain(
    ImplicitGeometry,
    # Boundary traits (Issue #590 Phase 1.2) - SDF-based geometries naturally support these
    SupportsBoundaryNormal,
    SupportsBoundaryProjection,
    SupportsBoundaryDistance,
    # Topology traits (Issue #590 Phase 1.2)
    SupportsManifold,  # SDFs with C² continuity form smooth manifolds
    SupportsLipschitz,  # SDFs are typically 1-Lipschitz
):
    """
    Abstract base class for n-dimensional implicit domains.

    An implicit domain D ⊂ ℝ^d is defined by a signed distance function φ:
        x ∈ D  ⟺  φ(x) < 0   (interior)
        x ∈ ∂D ⟺  φ(x) = 0   (boundary)
        x ∉ D  ⟺  φ(x) > 0   (exterior)

    Advantages over explicit mesh representation:
    - Memory: O(d) vs O(N^d) for mesh
    - Obstacles: Free with CSG operations
    - Dimension-agnostic: Same code for any d
    - Particle methods: Natural boundary handling

    Subclasses must implement:
    - signed_distance(x): Core SDF
    - get_bounding_box(): For efficient sampling

    Example:
        >>> domain = Hyperrectangle(np.array([[0, 1], [0, 1]]))  # [0,1]²
        >>> domain.contains(np.array([0.5, 0.5]))  # True (interior)
        >>> domain.contains(np.array([1.5, 0.5]))  # False (exterior)
        >>> particles = domain.sample_uniform(1000)  # Sample particles
    """

    # GeometryProtocol implementation
    @property
    def geometry_type(self) -> GeometryType:
        """
        Type of geometry (IMPLICIT for all ImplicitDomain subclasses).

        All implicit domains (defined by signed distance functions) return
        GeometryType.IMPLICIT regardless of dimension or specific shape.
        """
        return GeometryType.IMPLICIT

    # --- Trait properties (Issue #732 Tier 1b) ---

    @property
    def connectivity_type(self) -> ConnectivityType:
        """Dynamic: neighbors via runtime spatial search (meshfree)."""
        return ConnectivityType.DYNAMIC

    @property
    def structure_type(self) -> StructureType:
        """Unstructured: scattered point cloud."""
        return StructureType.UNSTRUCTURED

    @property
    def boundary_def(self) -> BoundaryDef:
        """Implicit: boundary defined by SDF phi(x) = 0."""
        return BoundaryDef.IMPLICIT

    @property
    def num_spatial_points(self) -> int:
        """
        Number of discrete spatial points.

        For implicit domains, this is not well-defined as they are continuous
        representations. This method estimates the number of points by computing
        the volume and assuming a typical mesh spacing of 0.1.

        The result is **cached on first call** (Issue #1037): ``compute_volume``
        uses Monte-Carlo sampling without a fixed seed, so subsequent calls would
        return slightly different values. Cached returns guarantee the property
        is deterministic within one process — required by callers like
        ``MFGComponents._setup_custom_initial_density`` that pre-allocate based
        on the value and then iterate the spatial grid.

        Note: For actual discretization, use sample_uniform() or meshfree methods.
        """
        # Issue #1037: cache to avoid MC-volume non-determinism across calls
        if self._num_spatial_points_cached is not None:
            return self._num_spatial_points_cached

        # Estimate number of points based on volume
        # Assume typical mesh spacing h = 0.1
        try:
            volume = self.compute_volume(n_monte_carlo=10000)
            h = 0.1  # Typical mesh spacing
            n_points = int(volume / (h**self.dimension))
            result = max(n_points, 100)  # Minimum of 100 points
        except (ValueError, RuntimeError) as e:
            # Issue #547: Volume computation can fail for complex implicit domains
            logger.warning(
                "Volume computation failed for %dD implicit domain: %s. "
                "Using bounding box approximation (may overestimate point count).",
                self.dimension,
                e,
            )
            # Fallback: use bounding box
            bounds = self.get_bounding_box()
            bbox_volume = np.prod(bounds[:, 1] - bounds[:, 0])
            h = 0.1
            result = max(int(bbox_volume / (h**self.dimension)), 100)

        self._num_spatial_points_cached = result
        return result

    def get_spatial_grid(self) -> NDArray[np.float64]:
        """
        Get spatial grid representation (sampled interior points).

        Since implicit domains are continuous (defined by SDF), we return
        a uniform sample of interior points as the "grid".

        Returns:
            Array of shape (N, dimension) with N sampled interior points
            where N is determined by num_spatial_points.

        Note: This is a sampling-based approximation. For more control over
        point distribution, use sample_uniform() directly.
        """
        n_points = self.num_spatial_points
        return self.sample_uniform(n_points)

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Spatial dimension of the domain."""

    @abstractmethod
    def signed_distance(self, x: NDArray[np.float64]) -> float | NDArray[np.float64]:
        """
        Compute signed distance function φ(x).

        Convention:
            φ(x) < 0  ⟺  x inside domain
            φ(x) = 0  ⟺  x on boundary
            φ(x) > 0  ⟺  x outside domain

        Args:
            x: Point(s) to evaluate - shape (d,) or (N, d)

        Returns:
            Signed distance(s) - scalar float or array of shape (N,)

        Notes:
            For exact SDFs, |φ(x)| = distance to boundary.
            For approximations, only sign matters for containment.
        """

    @abstractmethod
    def get_bounding_box(self) -> NDArray[np.float64]:
        """
        Get axis-aligned bounding box containing the domain.

        Returns:
            bounds: Array of shape (d, 2) where bounds[i] = [min_i, max_i]

        Example:
            >>> domain = Hypersphere(center=[0, 0], radius=1.0)
            >>> bounds = domain.get_bounding_box()
            >>> bounds  # array([[-1, 1], [-1, 1]])
        """

    # Note: subclasses that don't already maintain a `self.bounds` instance
    # attribute (Hyperrectangle does) should expose `.bounds` themselves —
    # see CSG composites (UnionDomain/IntersectionDomain/DifferenceDomain/
    # ComplementDomain) for the Issue #1041 fix that adds them.

    # Note: contains() and sample_uniform() now inherited from ImplicitGeometry

    @deprecated(
        since="v0.12.0",
        replacement="Use MeshfreeApplicator from mfgarchon.geometry.boundary instead.",
    )
    def apply_boundary_conditions(
        self,
        particles: NDArray[np.float64],
        bc_type: Literal["reflecting", "absorbing", "periodic"] = "reflecting",
    ) -> NDArray[np.float64]:
        """
        Apply boundary conditions to particles that left the domain.

        .. deprecated:: 0.12.0
            Use :class:`MeshfreeApplicator` from ``mfgarchon.geometry.boundary`` instead.
            This provides a unified interface for all geometry types.

        Args:
            particles: Particle positions - shape (N, d)
            bc_type: Boundary condition type
                - "reflecting": Reflect particles back into domain
                - "absorbing": Remove particles outside domain
                - "periodic": Wrap particles to opposite side (box domains only)

        Returns:
            Updated particle positions (may have fewer particles for "absorbing")

        Example:
            >>> # Deprecated usage:
            >>> domain = Hyperrectangle(np.array([[0, 1], [0, 1]]))
            >>> particles_updated = domain.apply_boundary_conditions(particles, "reflecting")
            >>>
            >>> # New preferred usage:
            >>> from mfgarchon.geometry.boundary import MeshfreeApplicator
            >>> applicator = MeshfreeApplicator(domain)
            >>> particles_updated = applicator.apply_particle_bc(particles, "reflecting")
        """
        if bc_type == "reflecting":
            return self.project_to_interior(particles)

        elif bc_type == "absorbing":
            inside = self.contains(particles)
            if np.isscalar(inside):
                inside = np.array([inside])
            return particles[inside]

        elif bc_type == "periodic":
            # Periodic BC requires special handling (only works for hyperrectangles)
            raise NotImplementedError("Periodic BC requires Hyperrectangle-specific implementation")

        else:
            raise ValueError(f"Unknown boundary condition type: {bc_type}")

    def compute_volume(self, n_monte_carlo: int = 100000) -> float:
        """
        Estimate domain volume via Monte Carlo integration.

        Args:
            n_monte_carlo: Number of samples for Monte Carlo

        Returns:
            Estimated volume

        Example:
            >>> sphere = Hypersphere(center=[0, 0], radius=1.0)
            >>> vol = sphere.compute_volume(n_monte_carlo=1000000)
            >>> assert np.abs(vol - np.pi) < 0.01  # π for unit circle in 2D
        """
        bounds = self.get_bounding_box()
        bbox_volume = np.prod(bounds[:, 1] - bounds[:, 0])

        # Sample uniformly from bounding box
        samples = np.random.uniform(low=bounds[:, 0], high=bounds[:, 1], size=(n_monte_carlo, self.dimension))

        # Fraction inside domain
        inside = self.contains(samples)
        if np.isscalar(inside):
            inside = np.array([inside])

        fraction_inside = np.mean(inside)

        return bbox_volume * fraction_inside

    def get_boundary_normal(self, x: NDArray[np.float64], eps: float = 1e-6) -> NDArray[np.float64]:
        """
        Compute outward boundary normal at point(s) using SDF gradient.

        The boundary normal is computed as n = ∇φ / ||∇φ|| where φ is the
        signed distance function. For an exact SDF, ||∇φ|| = 1, but we
        normalize to handle approximate SDFs.

        Universal Outward Normal Convention (Issue #661):
            - n points FROM domain interior TO exterior
            - ∂u/∂n > 0 means u increases in the outward direction
            - For SDF φ with convention φ < 0 inside: n = ∇φ / |∇φ|

        Args:
            x: Point(s) to evaluate - shape (d,) or (N, d)
            eps: Finite difference step size for gradient computation

        Returns:
            Unit normal vector(s) pointing outward - same shape as x

        Notes:
            - The normal points outward (direction of increasing φ)
            - For points not exactly on the boundary, this gives the normal
              to the closest boundary point
            - Uses central finite differences for gradient: O(eps²) accuracy
            - Implementation delegates to outward_normal_from_sdf() for
              consistency (Issue #662 consolidation)

        Example:
            >>> sphere = Hypersphere(center=[0, 0], radius=1.0)
            >>> normal = sphere.get_boundary_normal(np.array([1.0, 0.0]))
            >>> np.allclose(normal, [1.0, 0.0])  # Points radially outward
            True
        """
        # Use canonical implementation from operators/differential (Issue #662)
        from mfgarchon.operators.differential.function_gradient import (
            outward_normal_from_sdf,
        )

        return outward_normal_from_sdf(self.signed_distance, x, eps=eps)

    def project_to_boundary(
        self,
        x: NDArray[np.float64],
        max_iterations: int = 20,
        tol: float = 1e-8,
    ) -> NDArray[np.float64]:
        """
        Project point(s) onto the domain boundary using Newton iteration.

        Uses the iteration: x_{k+1} = x_k - φ(x_k) * n(x_k)
        where φ is the SDF and n is the unit outward normal.

        This converges quickly for smooth boundaries and is exact in one
        step for points where the SDF gradient is constant along the
        projection direction (e.g., spheres, planes).

        Args:
            x: Point(s) to project - shape (d,) or (N, d)
            max_iterations: Maximum Newton iterations
            tol: Convergence tolerance for |φ(x)|

        Returns:
            Projected point(s) on boundary - same shape as x

        Example:
            >>> sphere = Hypersphere(center=[0, 0], radius=1.0)
            >>> inside = np.array([0.5, 0.0])
            >>> on_boundary = sphere.project_to_boundary(inside)
            >>> np.allclose(on_boundary, [1.0, 0.0])
            True
        """
        is_single = x.ndim == 1
        if is_single:
            x = x.reshape(1, -1)

        x_proj = x.copy()
        n_points = x.shape[0]

        for i in range(n_points):
            xi = x_proj[i].copy()

            for _ in range(max_iterations):
                phi = self.signed_distance(xi)

                # Check convergence
                if np.abs(phi) < tol:
                    break

                # Get outward normal at current point
                normal = self.get_boundary_normal(xi)

                # Newton step: move along normal by -φ
                # (negative because we want to move toward boundary)
                xi = xi - phi * normal

            x_proj[i] = xi

        return x_proj[0] if is_single else x_proj

    def is_on_boundary(
        self,
        x: NDArray[np.float64],
        tolerance: float | None = None,
        tol: float | None = None,
    ) -> bool | NDArray[np.bool_]:
        """
        Check if point(s) are on the domain boundary using SDF.

        A point is on the boundary if |φ(x)| < tolerance where φ is the SDF.

        `tolerance`, matching `GeometryProtocol.is_on_boundary` and the eighteen other
        implementers (#1943). This family spelled it `tol`, so positional calls worked either way
        and **any keyword call broke on one side of the split**, with which side decided only by
        which geometry the caller happened to be handed. Measured before the fix:

            Hyperrectangle.is_on_boundary(pts, tol=0.03)        -> [True, False]
            Hyperrectangle.is_on_boundary(pts, tolerance=0.03)  -> TypeError

        and `ImplicitApplicator._detect_boundary_points` used the keyword form -- an applicator that
        exists *for* this family, calling it in the spelling this family did not accept.

        `tol=` is still accepted and deprecated. Six subclasses inherit this method rather than
        overriding it, so this is the one definition the whole family reads.

        Args:
            x: Point(s) to check - shape (d,) or (N, d)
            tolerance: Tolerance for boundary detection
            tol: Deprecated alias for `tolerance`

        Returns:
            Boolean or array of booleans indicating if points are on boundary

        Example:
            >>> sphere = Hypersphere(center=[0, 0], radius=1.0)
            >>> sphere.is_on_boundary(np.array([1.0, 0.0]))
            True
            >>> sphere.is_on_boundary(np.array([0.5, 0.0]))
            False
        """
        if tol is not None:
            if tolerance is not None:
                raise TypeError(
                    "is_on_boundary() received both `tolerance` and `tol`; they are the same "
                    "argument. Use `tolerance`, which is what GeometryProtocol declares (#1943)."
                )
            warnings.warn(
                "is_on_boundary(tol=...) is deprecated; use `tolerance=`, which is the spelling "
                "GeometryProtocol and the other eighteen implementers use (#1943).",
                DeprecationWarning,
                stacklevel=2,
            )
            tolerance = tol
        if tolerance is None:
            tolerance = SDF_BOUNDARY_TOL

        sd = self.signed_distance(x)
        return np.abs(sd) < tolerance

    # =========================================================================
    # Trait Protocol Implementations (Issue #590 Phase 1.2)
    # =========================================================================

    def get_outward_normal(
        self,
        points: NDArray[np.float64],
        boundary_name: str | None = None,
    ) -> NDArray[np.float64]:
        """
        Compute outward unit normal vectors at boundary points.

        Implements SupportsBoundaryNormal protocol.

        For SDF-based geometries, the outward normal is n = ∇φ / |∇φ| where
        φ is the signed distance function.

        Args:
            points: Points at which to evaluate normal, shape (num_points, dimension)
                    or (dimension,) for single point
            boundary_name: Ignored for implicit domains (SDF defines single boundary)

        Returns:
            Outward unit normals, shape (num_points, dimension) or (dimension,)

        Example:
            >>> sphere = Hypersphere(center=[0, 0], radius=1.0)
            >>> normal = sphere.get_outward_normal(np.array([1.0, 0.0]))
            >>> assert np.allclose(normal, [1.0, 0.0])  # Points radially outward

        Note:
            This method delegates to get_boundary_normal() which uses finite
            differences to compute the SDF gradient.
        """
        # Delegate to existing implementation
        return self.get_boundary_normal(points)

    def get_signed_distance(
        self,
        points: NDArray[np.float64],
    ) -> NDArray[np.float64] | float:
        """
        Compute signed distance to boundary for given points.

        Implements SupportsBoundaryDistance protocol.

        Args:
            points: Query points, shape (num_points, dimension) or (dimension,) for single point

        Returns:
            Signed distances, shape (num_points,) or scalar for single point
                - Negative: Inside domain
                - Zero: On boundary
                - Positive: Outside domain

        Example:
            >>> sphere = Hypersphere(center=[0, 0], radius=1.0)
            >>> points = np.array([[0, 0], [1, 0], [2, 0]])  # Center, boundary, outside
            >>> phi = sphere.get_signed_distance(points)
            >>> assert phi[0] < 0  # Inside
            >>> assert np.isclose(phi[1], 0)  # On boundary
            >>> assert phi[2] > 0  # Outside

        Note:
            This method delegates to signed_distance() which is the core SDF
            implementation provided by subclasses.
        """
        # Delegate to existing implementation
        return self.signed_distance(points)

    def project_to_interior(
        self,
        points: NDArray[np.float64],
        tolerance: float = 1e-10,
        previous: NDArray[np.float64] | None = None,
        report: dict[str, int] | None = None,
        via: NDArray[np.float64] | None = None,
    ) -> NDArray[np.float64]:
        """
        Project points from outside domain into interior.

        Implements SupportsBoundaryProjection protocol.

        Each outside point is projected onto the boundary along the SDF normal, then moved inward by
        `tolerance`, or by up to 1e4 * `tolerance` where the projection converged linearly and stopped
        short. That is the nearest interior point wherever the projection reaches the boundary point
        nearest it (#2679).

        The projection does not place a point where phi is not the true distance to the domain: a point
        trapped against the outer box where an obstacle crosses or is flush with it, and a point inside a
        union, overlapping obstacles or an approximate SDF (#2689). Such a point needs where it came from
        (#2688):

        - With ``previous`` (the point's position before the step that took it out), the first phi = 0
          crossing on the step's path names the face it crossed, and the point is projected onto that face's
          tangent plane, then moved inward by ``tolerance``. The path is the segment from ``previous``, or,
          with ``via``, the polyline through those waypoints -- a reflected step's wall hits, so that the face
          is one the step actually met (#2706, review 1). The crossing is found by a scan of 32 steps per leg
          and bisection on the sign of phi alone, so a leg whose path inside a feature is shorter than a scan step,
          1/32 of the leg, can step over it and name a later face: a thin feature, or a corner the leg clips
          (#2710). On a flat face this
          is the projected-Euler (Skorokhod) map: it keeps the step's displacement along the face. On a curved
          face, at an edge or a seam it is a local approximation, repeated from the new segment while the result
          is still outside, up to 8 faces, each time projecting onto the intersection of the faces met. A point
          still outside after 8 faces raises, naming it and the faces.
        - Without ``previous`` such a point is refused: nothing here can place it nearest, and the walk toward
          the bounding-box centre that used to place it crossed the obstacle (#2688).

        Args:
            points: Points to project, shape (num_points, dimension) or (dimension,)
            tolerance: Distance to move inside boundary (for numerical stability)
            previous: The points' positions before the step, same shape as ``points``, each inside the domain.
                Read only for the points the projection does not place.
            report: Optional counter, updated in place: ``"projected"`` (placed by the projection) and
                ``"face_of_entry"`` (by the face map).
            via: Optional waypoints between ``previous`` and each point, shape (num_points, k, dimension), rows of
                NaN unused: the step's actual path when it is not straight. Read only with ``previous``.

        Returns:
            Projected points in interior, same shape as input

        Raises:
            RuntimeError: a point the projection does not place, with no ``previous`` given; or one that 8
                faces do not place.
            ValueError: a ``previous`` position, for such a point, that is not inside the domain.

        Example:
            >>> sphere = Hypersphere(center=[0, 0], radius=1.0)
            >>> outside = np.array([2.0, 0.0])  # Outside
            >>> inside = sphere.project_to_interior(outside, tolerance=0.01)
            >>> assert sphere.signed_distance(inside) < 0  # Now inside

        Note:
            For points already inside, returns them unchanged.
        """
        is_single = points.ndim == 1
        if is_single:
            points = points.reshape(1, -1)
            if previous is not None:
                previous = np.asarray(previous, dtype=float).reshape(1, -1)

        outside = np.atleast_1d(self.signed_distance(points)) > 0

        if not np.any(outside):
            # All points already inside
            return points[0] if is_single else points

        points_interior = points.copy()
        moved, missed = self._projection(points[outside], tolerance)

        if missed.any():
            at = np.where(outside)[0][missed]
            if previous is None:
                raise RuntimeError(
                    f"project_to_interior: {at.size} point(s), the first {points[at[0]]}, are not placed by the "
                    f"projection onto the boundary: the domain's phi is not the true distance there (a point trapped "
                    f"against the outer box where an obstacle meets it, inside a union or overlapping obstacles, or "
                    f"an approximate SDF), and no previous position was given. Pass "
                    f"project_to_interior(points, previous=...) -- each point's position before the step -- and the "
                    f"point is placed on the face that step crossed (#2688)."
                )
            prev = np.asarray(previous, dtype=float)[at]
            if np.any(np.atleast_1d(self.signed_distance(prev)) > 0):
                bad = prev[np.atleast_1d(self.signed_distance(prev)) > 0][0]
                raise ValueError(
                    f"project_to_interior: previous position {bad} is not inside the domain, so the step from it "
                    f"names no face it crossed (#2688)."
                )
            waypoints = (
                None if via is None else np.asarray(via, dtype=float).reshape(len(points), -1, points.shape[1])[at]
            )
            moved[missed] = self._place_by_face_of_entry(prev, points[at], tolerance, via=waypoints)
        if report is not None:
            report["projected"] = report.get("projected", 0) + int(outside.sum() - missed.sum())
            report["face_of_entry"] = report.get("face_of_entry", 0) + int(missed.sum())
        points_interior[outside] = moved

        return points_interior[0] if is_single else points_interior

    def _projection(
        self, points: NDArray[np.float64], tolerance: float
    ) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
        """`project_to_interior`'s projection of outside ``points``, shape (n, d): each onto the boundary along the SDF
        normal, then `tolerance` inward; and which of them it does not place (#2679, #2689, #2707).

        Not placed: a point still outside, and a point where the boundary's normal vanishes. There phi's central
        difference is exactly 0 -- an internal seam of a union, where two touching components each have phi = 0, or an
        obstacle flush with a wall at a coordinate where the difference rounds symmetrically -- so the step inward
        moves the point nowhere, and it stays at phi = 0: inside the closed obstacle, which reads as navigable."""
        boundary_points = self.project_to_boundary(points)
        normals = np.atleast_2d(self.get_outward_normal(boundary_points))
        degenerate = np.linalg.norm(normals, axis=1) < 0.5
        moved = boundary_points - tolerance * normals
        # Where the projection converges linearly -- an obstacle's corner, the seam of a union -- it stops with
        # 0 < phi below its own tol, and a step of `tolerance` may not cross. Step further, up to 1e4 * tolerance.
        missed = (np.atleast_1d(self.signed_distance(moved)) > 0) | degenerate
        for step in tolerance * 10.0 ** np.arange(1, 5):
            idx = np.where(missed & ~degenerate)[0]
            if idx.size == 0:
                break
            moved[idx] = boundary_points[idx] - step * normals[idx]
            missed[idx] = np.atleast_1d(self.signed_distance(moved[idx])) > 0
        return moved, missed

    def place_step(
        self,
        previous: NDArray[np.float64] | None,
        end: NDArray[np.float64],
        via: NDArray[np.float64] | None = None,
        tolerance: float = 1e-10,
        report: dict[str, int] | None = None,
    ) -> NDArray[np.float64]:
        """Where each step from ``previous`` to ``end`` lands inside the domain: a point the particle reaches from
        ``previous`` without leaving the domain (#2707). The one owner of that answer for a particle step.

        The step's path is the segment from ``previous`` to ``end``, or, with ``via``, the polyline through those
        waypoints -- a reflected step's wall hits (`reflection_waypoints`). Each leg is searched with `legs_leave`'s
        scan, so a leg whose path inside a feature is shorter than a scan step reads as staying inside (#2710).

        - A path that stays inside lands at ``end``.
        - An ``end`` outside the domain, on a path that stays inside up to its last waypoint, lands at its projection,
          the nearest interior point (`project_to_interior`), when both hold on the leaf the path's first crossing
          entered (`crossing_leaf`, the primitive whose surface it crossed): (a) the segment from that crossing to
          ``end`` stays inside that leaf's far side, so ``end`` is in the obstacle the path entered, not beyond a wall
          the obstacle meets; and (b) that leaf's outward normals at the crossing and at the projection agree, so no
          medial axis of it lies between them (`_projection_reached`). Whether the projection lies on that leaf's own
          surface is not tested: the projection follows phi's gradient, which inside the leaf is the leaf's own, and
          where it ends elsewhere -- on a surface the leaf's is covered by -- it has, in every case measured, either
          missed or failed (b) (#2732, review 1).
        - Every other step lands by the face its path met first (`_place_by_face_of_entry`): an ``end`` past an
          obstacle's medial axis (a slab's middle, a disc's centre), whose projection is on the far side; an ``end``
          in a second obstacle after the path left the first; an ``end`` inside the domain beyond an obstacle the path
          crossed; a path that left before its last waypoint; and an ``end`` the projection does not place.

        With ``previous`` None, the step is `project_to_interior` with no previous position: an outside point is
        projected, and one the projection does not place is refused.

        Args:
            previous: The positions before the step, shape (n, d) or (d,), each inside the domain; or None.
            end: The positions the step reached, same shape.
            via: Optional waypoints, shape (n, k, d), rows of NaN unused. Read only with ``previous``.
            tolerance: How far inside the boundary a placed point lands.
            report: Optional counter, updated in place: ``"projected"`` and ``"face_of_entry"``.

        Returns:
            The landed positions, the shape of ``end``.

        Raises:
            ValueError: a ``previous`` position, of a step whose path leaves the domain, that is not inside it; and,
                from `project_to_boundary`, an ``end`` exactly on a medial axis, where phi's gradient is zero ("SDF
                gradient is zero").
            RuntimeError: as `_place_by_face_of_entry` and `project_to_interior`.
        """
        if previous is None:
            return self.project_to_interior(np.asarray(end, dtype=float), tolerance, report=report)
        points = np.atleast_2d(np.asarray(end, dtype=float))
        n, d = points.shape
        if n == 0:
            return points.copy()
        prev = np.asarray(previous, dtype=float).reshape(n, d)
        waypoints = None if via is None else np.asarray(via, dtype=float).reshape(n, -1, d)
        last, early = prev.copy(), np.zeros(n, dtype=bool)
        for k in range(0 if waypoints is None else waypoints.shape[1]):
            w = waypoints[:, k]
            used = np.all(np.isfinite(w), axis=1)
            if used.any():
                early[used] |= self.legs_leave(last[used], w[used])
                last[used] = w[used]
        leaves = early | self.legs_leave(last, points)
        if leaves.any():
            phi_prev = np.atleast_1d(self.signed_distance(prev[leaves]))
            if np.any(phi_prev > 0):
                raise ValueError(
                    f"place_step: previous position {prev[leaves][phi_prev > 0][0]} is not inside the domain, so the "
                    f"step from it names no face it crossed (#2688, #2707)."
                )
        landed = points.copy()
        projected = np.zeros(n, dtype=bool)
        tried = np.flatnonzero(leaves & ~early & (np.atleast_1d(self.signed_distance(points)) > 0))
        if tried.size:
            ends = points[tried]
            candidate, missed = self._projection(ends, tolerance)
            found, entry_in, entry_out, _ = self._leg_crossings(last[tried], ends)
            kept = ~missed & found
            at = np.flatnonzero(kept)
            if at.size:
                kept[at] = self._projection_reached(entry_in[at], entry_out[at], ends[at], candidate[at])
            landed[tried[kept]] = candidate[kept]
            projected[tried[kept]] = True
        face = leaves & ~projected
        if face.any():
            landed[face] = self._place_by_face_of_entry(
                prev[face], points[face], tolerance, via=None if waypoints is None else waypoints[face]
            )
        if report is not None:
            report["projected"] = report.get("projected", 0) + int(projected.sum())
            report["face_of_entry"] = report.get("face_of_entry", 0) + int(face.sum())
        return landed[0] if np.asarray(end).ndim == 1 else landed

    def crossing_leaf(
        self, inside: NDArray[np.float64], outside: NDArray[np.float64]
    ) -> tuple[NDArray[np.object_], NDArray[np.int_]]:
        """Which surface of this domain's boundary is crossed between ``inside`` (phi <= 0) and ``outside`` (phi > 0),
        for crossings of shape (n, d): per crossing, the primitive domain whose own surface it is -- the leaf -- and the
        parity of the complements between this domain and that leaf. The corner owner's one answer to which surface a
        path crossed (#2706, #2707); `crossing_normal` derives from it.

        Here, with no components to choose among, the leaf is this domain, at parity 0. A CSG domain overrides it: it
        decides which component is crossed by which component's own phi changes sign between the two points, and asks
        that component, so the answer is a primitive's surface, not a blend of two (#2698). A leaf at odd parity is
        complemented: the domain lies outside it, so it is crossed from its outside to its inside.
        """
        n = int(np.atleast_2d(np.asarray(outside)).shape[0])
        leaves = np.empty(n, dtype=object)
        leaves[:] = [self] * n
        return leaves, np.zeros(n, dtype=int)

    def crossing_normal(self, inside: NDArray[np.float64], outside: NDArray[np.float64]) -> NDArray[np.float64]:
        """The outward unit normal of this domain's boundary where it is crossed between ``inside`` (phi <= 0) and
        ``outside`` (phi > 0), two points bracketing the crossing closely: one crossing, shape (d,), or several, shape
        (n, d), which give shape (n, d).

        It is the crossed leaf's own gradient normal (`crossing_leaf`), at the bracket's point outside that leaf --
        ``outside`` at even parity, ``inside`` at odd -- signed by the parity. That is a central difference of step 1e-6
        on the leaf's phi (`get_boundary_normal`'s ``eps``), so within that distance of another surface of the same
        leaf -- a rectangle's corner -- it blends that leaf's two faces; it never blends two components (#2698, #2706).
        """
        single = np.ndim(outside) == 1
        ins = np.atleast_2d(np.asarray(inside, dtype=float))
        outs = np.atleast_2d(np.asarray(outside, dtype=float))
        leaves, parity = self.crossing_leaf(ins, outs)
        normals = np.empty_like(outs)
        for leaf in _unique(leaves):
            rows = np.array([candidate is leaf for candidate in leaves])
            for odd in (False, True):
                at = rows & ((parity % 2 == 1) == odd)
                if at.any():
                    points = ins[at] if odd else outs[at]
                    normal = np.atleast_2d(leaf.get_outward_normal(points))
                    normals[at] = -normal if odd else normal
        return normals[0] if single else normals

    def _leg_crossing(
        self, a: NDArray[np.float64], b: NDArray[np.float64], n_scan: int = 32, n_bisect: int = 60
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]] | None:
        """On the leg from ``a`` (taken as inside) to ``b``: the last inside and the first outside point bracketing
        its first phi = 0 crossing, or None when no scan point beyond ``a`` is outside. A scan of ``n_scan`` steps
        and ``n_bisect`` bisections on the sign of phi; a leg whose path inside a feature is shorter than a scan step
        -- a thin feature, or a corner the leg clips -- can pass it over, and the crossing found is then a later one
        (#2710). One more sample sits 1e-9 of the leg before ``b``: a leg that
        ends ON an obstacle's face -- a reflected step's wall hit where the obstacle is flush with the wall -- has
        phi = 0 there, which reads inside, and its excursion into the obstacle just before the end can be shorter
        than a scan step (#2706)."""
        found, inside, outside, _ = self._leg_crossings(a[None, :], b[None, :], n_scan, n_bisect)
        return (inside[0], outside[0]) if found[0] else None

    def _leg_crossings(
        self, a: NDArray[np.float64], b: NDArray[np.float64], n_scan: int = 32, n_bisect: int = 60
    ) -> tuple[NDArray[np.bool_], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """`_leg_crossing` for the legs ``a[i]`` -> ``b[i]``, shape (n, d), all at once: whether each has a crossing,
        its bracket, and the bracket's outside end as a fraction of the leg, NaN where it has none. Each leg's scan and
        bisections are its own, so a leg's bracket is the one it would get alone."""
        n, d = a.shape
        s = _leg_parameters(n_scan)
        samples = a[:, None, :] + s[None, :, None] * (b - a)[:, None, :]
        out = np.atleast_1d(self.signed_distance(samples.reshape(-1, d))).reshape(n, len(s)) > 0
        out[:, 0] = False
        found = out.any(axis=1)
        inside = np.full((n, d), np.nan)
        outside = np.full((n, d), np.nan)
        fraction = np.full(n, np.nan)
        if not found.any():
            return found, inside, outside, fraction
        at = np.flatnonzero(found)
        first = np.argmax(out[at], axis=1)
        lo, hi = s[first - 1].copy(), s[first].copy()
        a_at, step = a[at], (b - a)[at]
        for _ in range(n_bisect):
            mid = 0.5 * (lo + hi)
            beyond = np.atleast_1d(self.signed_distance(a_at + mid[:, None] * step)) > 0
            hi = np.where(beyond, mid, hi)
            lo = np.where(beyond, lo, mid)
        inside[at] = a_at + lo[:, None] * step
        outside[at] = a_at + hi[:, None] * step
        fraction[at] = hi
        return found, inside, outside, fraction

    def legs_leave(
        self,
        starts: NDArray[np.float64],
        ends: NDArray[np.float64],
        n_scan: int = 32,
        before: NDArray[np.float64] | None = None,
    ) -> NDArray[np.bool_]:
        """Whether each leg from ``starts[i]`` (taken as inside) to ``ends[i]`` leaves the domain, for every leg at once
        (#2707).

        A leg whose two ends are inside, and whose length is less than the sum of their depths -phi over phi's
        Lipschitz constant L (`get_lipschitz_constant`), stays inside: the balls of those radii about its ends are
        inside the domain and cover it. That is exact wherever phi is L-Lipschitz -- an exact SDF, and a CSG of exact
        SDFs, with L = 1 -- and it decides a short step without a scan. Every other leg leaves where a point of
        `_leg_crossing`'s scan beyond its start is outside: the same scan, so the same bound, and a leg whose path
        inside a feature is shorter than a scan step can pass it over (#2710).

        With ``before``, fractions of each leg, a leg leaves only where its first crossing -- the scan and bisection
        `place_step` itself uses on that leg -- lies before ``before[i]``; so a caller that asks about part of a leg
        and then places the whole leg gets one answer about which boundary it met first."""
        if len(starts) == 0:
            return np.zeros(0, dtype=bool)
        a = np.asarray(starts, dtype=float).reshape(len(starts), -1)
        b = np.asarray(ends, dtype=float).reshape(a.shape)
        phi_a = np.atleast_1d(self.signed_distance(a))
        phi_b = np.atleast_1d(self.signed_distance(b))
        lipschitz = self.get_lipschitz_constant("sdf")
        covered = (phi_a <= 0) & (phi_b <= 0) & (lipschitz * np.linalg.norm(b - a, axis=1) < -(phi_a + phi_b))
        leaves = np.zeros(len(a), dtype=bool)
        scan = np.flatnonzero(~covered)
        if scan.size:
            s = _leg_parameters(n_scan)[1:]
            samples = a[scan, None, :] + s[None, :, None] * (b[scan] - a[scan])[:, None, :]
            phi = np.atleast_1d(self.signed_distance(samples.reshape(-1, a.shape[1]))).reshape(scan.size, len(s))
            leaves[scan] = (phi > 0).any(axis=1)
        if before is not None:
            at = np.flatnonzero(leaves)
            if at.size:
                fraction = self._leg_crossings(a[at], b[at], n_scan)[3]
                leaves[at] = fraction < np.asarray(before, dtype=float).reshape(len(a))[at]
        return leaves

    def _projection_reached(
        self,
        entry_in: NDArray[np.float64],
        entry_out: NDArray[np.float64],
        ends: NDArray[np.float64],
        projections: NDArray[np.float64],
        n_scan: int = 32,
    ) -> NDArray[np.bool_]:
        """`place_step`'s test of whether a step reaches its end's projection, per step (#2707): read on the leaf the
        path entered (`crossing_leaf`), the primitive whose surface it crossed, not on the domain's whole complement.

        (a) Every point of `legs_leave`'s scan from the entry to the end lies inside that leaf's far side: phi_leaf > 0
        beyond a leaf at even parity, phi_leaf < 0 inside a complemented one (an obstacle). The end is in the obstacle
        the path entered, not in the region beyond a wall that obstacle meets. The same scan, so the same bound: a gap
        narrower than a scan step can pass unseen (#2710).
        (b) That leaf's own outward normal at the projection agrees with its normal at the entry, so no medial axis of
        the leaf lies between them. It is the same leaf both times, so the order of tied components never decides.
        """
        leaves, parity = self.crossing_leaf(entry_in, entry_out)
        sign = np.where(parity % 2 == 1, -1.0, 1.0)
        n_entry = np.atleast_2d(self.crossing_normal(entry_in, entry_out))
        s = _leg_parameters(n_scan)
        reached = np.zeros(len(ends), dtype=bool)
        for leaf in _unique(leaves):
            rows = np.flatnonzero(np.array([candidate is leaf for candidate in leaves]))
            a, b = entry_out[rows], ends[rows]
            samples = a[:, None, :] + s[None, :, None] * (b - a)[:, None, :]
            phi = np.atleast_1d(leaf.signed_distance(samples.reshape(-1, a.shape[1]))).reshape(len(rows), len(s))
            inside_entered = (sign[rows, None] * phi > 0).all(axis=1)
            n_projection = sign[rows, None] * np.atleast_2d(leaf.get_outward_normal(projections[rows]))
            agree = np.einsum("nd,nd->n", n_entry[rows], n_projection) > 0
            reached[rows] = inside_entered & agree
        return reached

    def _place_by_face_of_entry(
        self,
        previous: NDArray[np.float64],
        points: NDArray[np.float64],
        tolerance: float,
        max_faces: int = 8,
        via: NDArray[np.float64] | None = None,
    ) -> NDArray[np.float64]:
        """Place each outside point by the faces its step crosses (#2688); see `project_to_interior`.

        The first crossing on the step's path -- the segment from the previous position, or the polyline through
        ``via`` -- names a face, kept as the half-space on the
        domain's side of its tangent plane, and the point is projected onto the intersection of the faces kept so
        far, `tolerance` inside each. Where that is still outside, the segment from the last inside point to it
        crosses one more face, and the original point is projected again onto the larger intersection. One face is
        the flat-face projection; two at a seam or an edge are the projection onto the wedge they bound -- repeated
        projections onto one face at a time converge only geometrically there. Raises, naming the point and the
        faces, where ``max_faces`` faces do not place it, or where `_project_onto_half_spaces` finds no nearest
        point of their intersection (#2688)."""
        placed = np.empty_like(points)
        n = points.shape[0]
        # Each step's path: its previous position, its waypoints in order, its point. The first crossing on it is
        # searched for every step at once, leg by leg.
        paths = []
        for i in range(n):
            nodes = [previous[i]]
            if via is not None:
                nodes.extend(v for v in via[i] if np.all(np.isfinite(v)))
            nodes.append(points[i])
            paths.append(nodes)
        first: list[tuple[NDArray[np.float64], NDArray[np.float64]] | None] = [None] * n
        pending = np.arange(n)
        for k in range(max(len(nodes) for nodes in paths) - 1):
            pending = np.array([i for i in pending if k + 1 < len(paths[i])], dtype=int)
            if pending.size == 0:
                break
            found, inside, outside, _ = self._leg_crossings(
                np.array([paths[i][k] for i in pending]), np.array([paths[i][k + 1] for i in pending])
            )
            for j in np.flatnonzero(found):
                first[pending[j]] = (inside[j], outside[j])
            pending = pending[~found]
        for i in range(n):
            target = points[i]
            crossing = first[i]
            centres, normals = [], []
            candidate, why = None, ""
            for _ in range(max_faces):
                if crossing is None:
                    candidate = None
                    why = (
                        "no crossing of the domain's boundary was found where one was sought: on its path, or on the "
                        "segment from the last inside point to the point projected onto the faces met"
                    )
                    break
                inside, crossed = crossing
                centres.append(crossed)
                normals.append(self.crossing_normal(inside, crossed))
                candidate = _project_onto_half_spaces(target, np.array(centres), np.array(normals), tolerance)
                if candidate is None:
                    why = (
                        "no nearest point of their intersection was found. That is the active-set solve's limit "
                        "(`_project_onto_half_spaces`): faces so nearly parallel that the Gram matrix of the set "
                        "holding the nearest point is too close to singular -- two faces facing apart at half-angle "
                        "1e-7 or less, with the placement's gap 1e-10"
                    )
                    break
                if float(np.atleast_1d(self.signed_distance(candidate[None, :]))[0]) <= 0:
                    placed[i] = candidate
                    break
                crossing = self._leg_crossing(inside, candidate)
            else:
                candidate = None
                why = f"projected onto the intersection of these {max_faces} faces it is still outside the domain"
            if candidate is None:
                faces = "; ".join(f"at {c} with outward normal {n}" for c, n in zip(centres, normals, strict=True))
                raise RuntimeError(
                    f"project_to_interior: the point {target}, whose step began at {previous[i]}, is not placed by "
                    f"the {len(centres)} face(s) its step met ({faces}): {why}. The face-of-entry placement does "
                    f"not cover this geometry (#2688); report it with the domain."
                )
        return placed

    @property
    def manifold_dimension(self) -> int:
        """
        Intrinsic dimension of the domain manifold.

        Implements SupportsManifold protocol.

        For ImplicitDomain representing a full domain D subset R^d,
        manifold dimension equals ambient dimension d. The SDF boundary
        dD is (d-1)-dimensional, but the domain itself is d-dimensional.

        Note:
            Subclasses representing embedded surfaces (codimension-1)
            should override to return self.dimension - 1.
        """
        return self.dimension

    def get_metric_tensor(
        self,
        points: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """
        Get metric tensor for the domain manifold.

        Implements SupportsManifold protocol.

        For ImplicitDomain in Euclidean space, the metric tensor is the
        standard Euclidean metric g_ij = δ_ij (identity matrix).

        Args:
            points: Points at which to evaluate metric, shape (num_points, dimension)
                    or (dimension,) for single point

        Returns:
            Metric tensor(s), shape (num_points, dimension, dimension) or (dimension, dimension)

        Example:
            >>> rect = Hyperrectangle([[0, 1], [0, 1]])
            >>> g = rect.get_metric_tensor(np.array([0.5, 0.5]))
            >>> assert np.allclose(g, np.eye(2))  # Euclidean metric

        Note:
            ImplicitDomains are embedded in Euclidean space, so the metric is
            always Euclidean (flat). Subclasses with Riemannian geometry should
            override this method.
        """
        is_single = points.ndim == 1
        if is_single:
            points = points.reshape(1, -1)

        n_points, d = points.shape

        # Euclidean metric: g_ij = δ_ij at all points
        g = np.tile(np.eye(d), (n_points, 1, 1))

        return g[0] if is_single else g

    def get_lipschitz_constant(
        self,
        function_type: str = "sdf",
    ) -> float:
        """
        Get Lipschitz constant for specified function.

        Implements SupportsLipschitz protocol.

        Args:
            function_type: Type of function to query
                - "sdf": Signed distance function (default)
                - "metric": Metric tensor components
                - "projection": Boundary projection map

        Returns:
            Lipschitz constant L where |f(x) - f(y)| ≤ L|x - y|

        Example:
            >>> sphere = Hypersphere(center=[0, 0], radius=1.0)
            >>> L = sphere.get_lipschitz_constant("sdf")
            >>> assert L == 1.0  # Exact SDFs are 1-Lipschitz

        Raises:
            ValueError: If function_type not recognized

        Note:
            - Exact SDFs satisfy |∇φ| = 1, so L_sdf = 1.0
            - Metric tensor is constant (Euclidean), so L_metric = 0.0
            - Projection map is generally 1-Lipschitz for convex domains
        """
        if function_type == "sdf":
            # Exact SDFs are 1-Lipschitz: |φ(x) - φ(y)| ≤ |x - y|. `legs_leave` decides a short leg inside without a
            # scan by this constant (#2707), so a subclass whose phi is steeper -- an approximate SDF -- must return
            # its own bound, or legs it crosses are read as inside.
            return 1.0

        elif function_type == "metric":
            # Euclidean metric is constant, so derivative is zero
            return 0.0

        elif function_type == "projection":
            # Projection to convex set is 1-Lipschitz (non-expansive)
            # For non-convex sets, this is an upper bound
            return 1.0

        else:
            raise ValueError(f"Unknown function_type '{function_type}'. Valid options: 'sdf', 'metric', 'projection'")

    def get_tangent_space_basis(
        self,
        points: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """
        Compute orthonormal basis for tangent space at given points.

        Implements SupportsManifold protocol.

        For ImplicitDomain in flat Euclidean space, the tangent space is R^d
        with canonical basis at every point.

        Args:
            points: Query points, shape (num_points, dimension) or (dimension,)

        Returns:
            Tangent basis vectors:
                - Single point: (dimension, dimension) canonical basis
                - Multiple points: (num_points, dimension, dimension)
        """
        # Canonical basis = metric tensor for flat space
        return self.get_metric_tensor(points)

    def compute_christoffel_symbols(
        self,
        points: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """
        Compute Christoffel symbols for flat Euclidean space.

        Implements SupportsManifold protocol.

        For ImplicitDomain in flat Euclidean space, all Christoffel symbols
        vanish: Gamma^k_{ij} = 0. This reflects zero curvature.

        Args:
            points: Query points, shape (num_points, dimension) or (dimension,)

        Returns:
            Christoffel symbols (all zeros):
                - Single point: (dimension, dimension, dimension)
                - Multiple points: (num_points, dimension, dimension, dimension)
        """
        is_single = points.ndim == 1
        d = points.shape[-1] if points.ndim > 1 else len(points)

        if is_single:
            return np.zeros((d, d, d))

        n_points = points.shape[0]
        return np.zeros((n_points, d, d, d))

    def validate_lipschitz_regularity(
        self,
        tolerance: float = 1e-6,
    ) -> tuple[bool, str]:
        """
        Validate that boundary satisfies Lipschitz condition.

        Implements SupportsLipschitz protocol.

        For ImplicitDomain, regularity depends on the SDF. Exact SDFs
        (|nabla phi| = 1) are automatically 1-Lipschitz with smooth
        boundaries. Subclasses with non-smooth boundaries (corners, cusps)
        should override.

        Args:
            tolerance: Numerical tolerance for validation checks

        Returns:
            (is_valid, message): True with empty message for smooth SDFs.
        """
        L = self.get_lipschitz_constant("sdf")
        if not np.isfinite(L):
            return False, f"SDF Lipschitz constant is not finite: {L}"
        return True, ""

    # GeometryProtocol methods for solver interface
    def get_grid_shape(self) -> tuple[int]:
        """
        Get discretization shape for implicit domain.

        Returns:
            (N,) where N is an estimated number of points based on volume.

        Notes:
            Implicit domains are continuous - they don't have inherent grid structure.
            This method provides an estimate for compatibility with structured solvers.
            For actual discretization, use sample_uniform() or meshfree methods.
        """
        # Return estimated point count as (N,) tuple
        return (self.num_spatial_points,)

    def get_boundary_conditions(self):
        """
        Get boundary conditions for implicit domain.

        Returns:
            None - implicit domains don't have inherent boundary conditions.
            BCs should be specified by the problem or solver.

        Notes:
            Implicit domains support general BC via signed distance function. The one channel a problem on an
            implicit domain has is the components form's MFGComponents(boundary_conditions=), which is deprecated;
            the model/domain/conditions form has none (#2684). A solver may also take them directly.
        """
        return None

    def get_collocation_points(self) -> NDArray[np.float64]:
        """
        Get collocation points via uniform sampling of the domain.

        Returns:
            Array of shape (N, d) containing uniformly sampled points from domain.

        Notes:
            This samples the domain using the default number of points from
            num_spatial_points. For explicit control over sampling density,
            use sample_uniform(n_points) directly.

        Example:
            >>> rect = Hyperrectangle([[0, 1], [0, 1]])
            >>> points = rect.get_collocation_points()
            >>> points.shape  # (N, 2) where N ≈ volume / 0.1^2
        """
        return self.sample_uniform(self.num_spatial_points)

    # =========================================================================
    # Region Marking (SupportsRegionMarking protocol - Issue #590 Phase 1.3)
    # =========================================================================

    _regions_dict: dict | None = None  # Lazy-initialized by _regions property

    # Issue #1037: cache for num_spatial_points to prevent MC-volume non-determinism
    # across calls within one process. None until first computation.
    _num_spatial_points_cached: int | None = None

    @property
    def _regions(self) -> dict:
        """
        Lazy-initialized storage for named regions.

        For ImplicitDomain (meshfree), regions are stored as predicates/callables,
        not boolean masks (unlike TensorProductGrid which has discrete points).

        Returns:
            Dictionary mapping region names to predicates
        """
        if self._regions_dict is None:
            self._regions_dict = {}
        return self._regions_dict

    def mark_region(
        self,
        name: str,
        predicate: Callable[[NDArray], NDArray[np.bool_]] | None = None,
        sdf_region: Callable[[NDArray], NDArray[np.floating]] | None = None,
        **kwargs,
    ) -> None:
        """
        Mark a named spatial region for later reference.

        For implicit domains (meshfree), regions are specified via predicates
        or SDF functions and evaluated on-demand at query points.

        Args:
            name: Unique name for this region (e.g., "inlet", "obstacle")
            predicate: Function (N, d) → (N,) bool - True where point in region
            sdf_region: Alternative SDF-based specification (φ < 0 means inside region)
            **kwargs: Ignored (for compatibility with TensorProductGrid signature)

        Raises:
            ValueError: If name already exists
            ValueError: If neither predicate nor sdf_region provided

        Example:
            >>> circle = Hypersphere(center=[0.5, 0.5], radius=0.3)
            >>>
            >>> # Mark top half using predicate
            >>> circle.mark_region(
            ...     "top_half",
            ...     predicate=lambda x: x[:, 1] > 0.5
            ... )
            >>>
            >>> # Mark region using SDF
            >>> def sector_sdf(x):
            ...     on_boundary = np.abs(np.linalg.norm(x - [0.5, 0.5], axis=-1) - 0.3) < 0.01
            ...     in_sector = x[:, 1] > 0.65
            ...     return np.where(on_boundary & in_sector, -1.0, 1.0)
            >>> circle.mark_region("exit", sdf_region=sector_sdf)

        Note:
            Issue #549 / #590 Phase 1.3: Enables BC via region_name parameter
        """
        if name in self._regions:
            raise ValueError(f"Region '{name}' already exists. Available regions: {list(self._regions.keys())}")

        # Validate: exactly one specification method
        specified = sum(x is not None for x in [predicate, sdf_region])
        if specified == 0:
            raise ValueError("Must specify one of: predicate or sdf_region")
        if specified > 1:
            raise ValueError("Cannot specify both predicate and sdf_region")

        # Store predicate (or convert SDF to predicate)
        if predicate is not None:
            self._regions[name] = predicate
        elif sdf_region is not None:
            # Convert SDF to predicate: region is where SDF < 0
            self._regions[name] = lambda x: sdf_region(x) < 0

    def get_region_mask(
        self,
        name: str,
        points: NDArray[np.float64] | None = None,
    ) -> NDArray[np.bool_]:
        """
        Get boolean mask for named region.

        For implicit domains, evaluates the stored predicate at given points.

        Args:
            name: Region name (from mark_region call)
            points: Points to evaluate at - shape (N, d)
                    If None, uses get_collocation_points() (sampled discretization)

        Returns:
            Boolean mask of shape (N,) - True at points in region

        Raises:
            KeyError: If region name not found

        Example:
            >>> circle = Hypersphere(center=[0.5, 0.5], radius=0.3)
            >>> circle.mark_region("top", predicate=lambda x: x[:, 1] > 0.5)
            >>>
            >>> # Evaluate at specific points
            >>> pts = np.array([[0.5, 0.6], [0.5, 0.4]])
            >>> mask = circle.get_region_mask("top", points=pts)
            >>> # Returns: [True, False]

        Note:
            Unlike TensorProductGrid (which has fixed grid), implicit domains
            evaluate regions on-demand at provided points.
        """
        if name not in self._regions:
            raise KeyError(f"Region '{name}' not found. Available regions: {list(self._regions.keys())}")

        # Get points to evaluate at
        if points is None:
            points = self.get_collocation_points()

        # Ensure 2D array
        if points.ndim == 1:
            points = points.reshape(1, -1)

        # Evaluate predicate
        predicate = self._regions[name]
        return predicate(points)

    def intersect_regions(self, *names: str) -> Callable[[NDArray], NDArray[np.bool_]]:
        """
        Get intersection of multiple regions (boolean AND).

        For implicit domains, returns a combined predicate.

        Args:
            *names: Region names to intersect

        Returns:
            Combined predicate: callable (N, d) → (N,) bool

        Raises:
            KeyError: If any region name not found
            ValueError: If no region names provided

        Example:
            >>> circle.mark_region("top", predicate=lambda x: x[:, 1] > 0.5)
            >>> circle.mark_region("right", predicate=lambda x: x[:, 0] > 0.5)
            >>> combined = circle.intersect_regions("top", "right")
            >>> # combined(pts) returns True where BOTH top AND right
        """
        if not names:
            raise ValueError("Must provide at least one region name")

        predicates = [self._regions[name] for name in names]

        # Combine predicates with logical AND
        def combined_predicate(x: NDArray[np.float64]) -> NDArray[np.bool_]:
            result = predicates[0](x)
            for pred in predicates[1:]:
                result &= pred(x)
            return result

        return combined_predicate

    def get_region_predicate(self, name: str) -> Callable[[NDArray], NDArray[np.bool_]]:
        """
        Get predicate function for named region (meshfree helper).

        This is a convenience method for meshfree solvers that work directly
        with predicates rather than boolean masks.

        Args:
            name: Region name

        Returns:
            Predicate function: (N, d) → (N,) bool

        Raises:
            KeyError: If region name not found

        Example:
            >>> circle.mark_region("inlet", predicate=lambda x: x[:, 0] < 0.3)
            >>> inlet_pred = circle.get_region_predicate("inlet")
            >>> particles = np.array([[0.2, 0.5], [0.8, 0.5]])
            >>> in_inlet = inlet_pred(particles)  # [True, False]
        """
        if name not in self._regions:
            raise KeyError(f"Region '{name}' not found. Available regions: {list(self._regions.keys())}")
        return self._regions[name]

    def get_region_names(self) -> list[str]:
        """
        Get list of all marked region names.

        Returns:
            List of region names

        Example:
            >>> circle.mark_region("inlet", predicate=lambda x: x[:, 0] < 0.3)
            >>> circle.mark_region("outlet", predicate=lambda x: x[:, 0] > 0.7)
            >>> circle.get_region_names()
            ['inlet', 'outlet']
        """
        return list(self._regions.keys())

    def __repr__(self) -> str:
        """String representation of the domain."""
        return f"{self.__class__.__name__}(dimension={self.dimension})"


def _unique(objects: NDArray[np.object_]) -> list[Any]:
    """The distinct objects of an object array, by identity, in first-seen order."""
    seen: dict[int, Any] = {}
    for obj in objects:
        seen.setdefault(id(obj), obj)
    return list(seen.values())


def _project_onto_half_spaces(
    point: NDArray[np.float64], centres: NDArray[np.float64], normals: NDArray[np.float64], tolerance: float
) -> NDArray[np.float64] | None:
    """The nearest point to ``point`` in the intersection of the half-spaces ``(x - c_k) . n_k <= -tolerance``.

    Each half-space is a face's domain side, ``n_k`` its outward unit normal. The minimiser has at most ``d``
    active faces with independent normals, so it is found by trying every such set -- at most 92 for 8 faces in
    3-D -- and keeping the nearest candidate that satisfies all of them. Feasibility is judged with a slack
    relative to the problem's scale (#2706, review 1). None where no active set yields a feasible candidate: the
    intersection is empty, or the faces are so nearly parallel that the active set holding the nearest point has a
    Gram matrix too close to singular to solve. Two faces through one point, facing apart at half-angle 1e-7 or
    less, with ``tolerance`` 1e-10, are one: their intersection is a sliver whose apex lies ``tolerance / sin`` of the
    half-angle from that point, 0.1 at 1e-9 (#2706, review 2)."""
    d = point.shape[0]
    offsets = np.einsum("kd,kd->k", centres, normals) - tolerance
    slack = 64.0 * np.finfo(float).eps * max(1.0, float(np.abs(offsets).max()), float(np.abs(point).max()))
    if np.all(normals @ point <= offsets + slack):
        return point.copy()
    best, best_dist = None, np.inf
    k = len(normals)
    for size in range(1, min(d, k) + 1):
        for active in combinations(range(k), size):
            n_act = normals[list(active)]
            gram = n_act @ n_act.T
            if abs(np.linalg.det(gram)) < 1e-14:
                continue
            lam = np.linalg.solve(gram, n_act @ point - offsets[list(active)])
            if np.any(lam < 0):
                continue
            candidate = point - n_act.T @ lam
            if np.all(normals @ candidate <= offsets + slack):
                dist = float(np.linalg.norm(candidate - point))
                if dist < best_dist:
                    best, best_dist = candidate, dist
    return best
