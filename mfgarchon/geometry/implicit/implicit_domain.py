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
from dataclasses import dataclass
from itertools import combinations, pairwise
from typing import Literal

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

# An interval end that no surface bounds: a segment's own end, or the line's ends at infinity.
_NO_SURFACE = -1


@dataclass(frozen=True)
class SegmentIntervals:
    """The parts of segments a -> b inside a domain, as parameters t of a + t (b - a) in [0, 1] (#2733 D1).

    ``t[i, k]`` is the k-th interval (lo, hi) of segment i: closed, sorted and disjoint, with rows padded by NaN. A
    point of the segment is in an interval exactly where `contains` holds there, up to rounding within about an ulp of
    a surface, in space, and short of the overflow and underflow limits each leaf states: a segment that grazes a face
    can disagree with `contains` over a long stretch of its parameter.
    Each leaf solves its own form, a slab test or a quadratic, and a CSG node combines its children's intervals as Roth
    (1982) does. A complement enters as the closed set phi >= 0, not as the gaps between its child's
    intervals, so a segment that runs along an obstacle's face is in the domain there, as `contains` says.

    ``leaf[i, k, e]`` indexes ``leaves``, the domain's primitives depth first, for the surface that bounds end e of
    that interval, and is -1 where the end is the segment's own end and no surface. ``leaf_face`` is that leaf's face:
    on a Hyperrectangle 2 * axis for the min face and 2 * axis + 1 for the max face, on a Hypersphere 0.
    ``complemented`` marks a leaf that enters complemented, an obstacle, so that the domain's outward normal at that end
    is the leaf's inward one. Where the ends of several leaves coincide, at a corner or a wall two rooms share, the end
    carries one of them; a box's own tie, at an edge or a corner, goes to its lowest axis.
    """

    t: NDArray[np.float64]
    leaf: NDArray[np.int64]
    leaf_face: NDArray[np.int64]
    complemented: NDArray[np.bool_]
    leaves: tuple["ImplicitDomain", ...]


@dataclass(frozen=True)
class FirstContact:
    """Each segment's first contact with its domain's boundary, read from `ImplicitDomain.segment_intervals` (#2733 D1).

    ``inside_at_start`` is whether t = 0 lies in an interval: the admission of the start point, decided by the leaves'
    own forms rather than by a band around phi = 0. From a start inside, the contact is where the path leaves the
    domain, at t < 1, and at t = 0 where it leaves at once, from a face or a corner it starts on. From a start outside,
    it is where the path enters, at t <= 1. Elsewhere ``t`` is NaN and ``leaf`` and ``leaf_face`` are -1. ``point`` is
    a + t (b - a). ``normal`` is the domain's outward unit normal there, from the leaf's own form: +-e_axis on a box
    face, radial on a sphere, negated for a complemented leaf. ``leaf``, ``leaf_face`` and ``leaves`` are as in
    `SegmentIntervals`.
    """

    t: NDArray[np.float64]
    point: NDArray[np.float64]
    leaf: NDArray[np.int64]
    leaf_face: NDArray[np.int64]
    normal: NDArray[np.float64]
    inside_at_start: NDArray[np.bool_]
    leaves: tuple["ImplicitDomain", ...]


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
          (#2733). On a flat face this
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
        boundary_points = self.project_to_boundary(points[outside])
        normals = np.atleast_2d(self.get_outward_normal(boundary_points))
        moved = boundary_points - tolerance * normals
        # Where the projection converges linearly -- an obstacle's corner, the seam of a union -- it stops with
        # 0 < phi below its own tol, and a step of `tolerance` may not cross. Step further, up to 1e4 * tolerance.
        missed = np.atleast_1d(self.signed_distance(moved)) > 0
        for step in tolerance * 10.0 ** np.arange(1, 5):
            if not missed.any():
                break
            idx = np.where(missed)[0]
            moved[idx] = boundary_points[idx] - step * normals[idx]
            missed[idx] = np.atleast_1d(self.signed_distance(moved[idx])) > 0

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

    def crossing_normal(self, inside: NDArray[np.float64], outside: NDArray[np.float64]) -> NDArray[np.float64]:
        """The outward unit normal of this domain's boundary where it is crossed between ``inside`` (phi <= 0) and
        ``outside`` (phi > 0), two points bracketing the crossing closely.

        Here, with no components to choose among, it is the gradient normal at ``outside``
        (:meth:`get_outward_normal`). That is a central difference of step 1e-6 on phi (`get_boundary_normal`'s
        ``eps``), so within that distance of
        another surface of the boundary -- a corner, an obstacle meeting a wall -- it blends the two surfaces'
        normals (#2698). A CSG domain overrides this: it decides which component's surface is crossed by which
        component's own phi changes sign between the two points, and gives that component's normal, so no
        difference straddles another surface (#2706).
        """
        return np.atleast_2d(self.get_outward_normal(np.asarray(outside, dtype=float)[None, :]))[0]

    def _leg_crossing(
        self, a: NDArray[np.float64], b: NDArray[np.float64], n_scan: int = 32, n_bisect: int = 60
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]] | None:
        """On the leg from ``a`` (taken as inside) to ``b``: the last inside and the first outside point bracketing
        its first phi = 0 crossing, or None when no scan point beyond ``a`` is outside. A scan of ``n_scan`` steps
        and ``n_bisect`` bisections on the sign of phi; a leg whose path inside a feature is shorter than a scan step
        -- a thin feature, or a corner the leg clips -- can pass it over, and the crossing found is then a later one
        (#2733). One more sample sits 1e-9 of the leg before ``b``: a leg that
        ends ON an obstacle's face -- a reflected step's wall hit where the obstacle is flush with the wall -- has
        phi = 0 there, which reads inside, and its excursion into the obstacle just before the end can be shorter
        than a scan step (#2706)."""
        s = np.concatenate([np.linspace(0.0, 1.0, n_scan + 1)[:-1], [1.0 - 1e-9, 1.0]])
        out = np.atleast_1d(self.signed_distance(a[None, :] + s[:, None] * (b - a)[None, :])) > 0
        out[0] = False
        if not out.any():
            return None
        first = int(np.argmax(out))
        lo, hi = float(s[first - 1]), float(s[first])
        for _ in range(n_bisect):
            mid = 0.5 * (lo + hi)
            if float(np.atleast_1d(self.signed_distance((a + mid * (b - a))[None, :]))[0]) > 0:
                hi = mid
            else:
                lo = mid
        return a + lo * (b - a), a + hi * (b - a)

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
        for i in range(points.shape[0]):
            target = points[i]
            legs = [previous[i]]
            if via is not None:
                legs.extend(v for v in via[i] if np.all(np.isfinite(v)))
            legs.append(target)
            crossing = next(
                (c for a, b in pairwise(legs) if (c := self._leg_crossing(a, b)) is not None),
                None,
            )
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

    def segment_intervals(self, a: NDArray[np.float64], b: NDArray[np.float64]) -> SegmentIntervals:
        """The parts of each segment a -> b inside this domain, exactly, with the surfaces that bound them (#2733 D1).

        Args:
            a, b: The segments' ends, shape (N, d), or (d,) for one segment, which is returned as N = 1.

        Returns:
            `SegmentIntervals`, of shape (N, K, 2) with K the most intervals any segment has, and at least 1.

        Raises:
            NotImplementedError: a domain with no exact form, one given only by its phi (`_line_sets`), or a
                Hyperrectangle with periodic axes (#2704).
            ValueError: ends of different shapes, of another dimension, or not finite, or ends whose extent b - a
                overflows.
        """
        a, b = self._segment_ends(a, b)
        t, tag = self._line_sets(a, b - a, False, 0)
        lo, hi = t[..., 0], t[..., 1]
        kept = (hi >= 0.0) & (lo <= 1.0)
        ends = np.stack(
            [np.where(lo < 0.0, _NO_SURFACE, tag[..., 0]), np.where(hi > 1.0, _NO_SURFACE, tag[..., 1])], -1
        )
        t = np.stack([np.maximum(lo, 0.0), np.minimum(hi, 1.0)], axis=-1)
        k = max(int(kept.sum(axis=1).max(initial=0)), 1)
        order = np.argsort(~kept, axis=1, kind="stable")[:, :k]
        kept = np.take_along_axis(kept, order, axis=1)[..., None]
        t = np.where(kept, np.take_along_axis(t, order[..., None], axis=1), np.nan)
        tag = np.where(kept, np.take_along_axis(ends, order[..., None], axis=1), _NO_SURFACE)
        none = tag < 0
        return SegmentIntervals(
            t=t,
            leaf=np.where(none, _NO_SURFACE, tag >> 32),
            leaf_face=np.where(none, _NO_SURFACE, (tag & 0xFFFFFFFF) >> 1),
            complemented=~none & ((tag & 1) == 1),
            leaves=self._leaves(),
        )

    def first_contact(self, a: NDArray[np.float64], b: NDArray[np.float64]) -> FirstContact:
        """Each segment's first contact with this domain's boundary, from `segment_intervals` (#2733 D1); see
        `FirstContact`. Arguments and refusals are `segment_intervals`'s."""
        a, b = self._segment_ends(a, b)
        found = self.segment_intervals(a, b)
        lo, hi = found.t[:, 0, 0], found.t[:, 0, 1]
        inside = lo == 0.0
        exits = inside & (hi < 1.0)
        hit = exits | (~inside & ~np.isnan(lo))
        end = exits.astype(np.int64)
        rows = np.arange(len(a))
        t = np.where(hit, np.where(exits, hi, lo), np.nan)
        leaf = np.where(hit, found.leaf[rows, 0, end], _NO_SURFACE)
        face = np.where(hit, found.leaf_face[rows, 0, end], _NO_SURFACE)
        if np.any(leaf[hit] < 0):
            raise RuntimeError(
                f"first_contact: a contact at t = {t[hit][leaf[hit] < 0][0]} carries no surface; an exit before t = 1 "
                f"or an entry after t = 0 is always a leaf's end, so the classification is inconsistent (#2733)"
            )
        point = a + t[:, None] * (b - a)
        normal = np.full_like(a, np.nan)
        for index in np.unique(leaf[hit]):
            at = leaf == index
            normal[at] = found.leaves[index]._face_normal(point[at], face[at])
        normal[hit & found.complemented[rows, 0, end]] *= -1.0
        return FirstContact(
            t=t, point=point, leaf=leaf, leaf_face=face, normal=normal, inside_at_start=inside, leaves=found.leaves
        )

    def _segment_ends(
        self, a: NDArray[np.float64], b: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """The ends as (N, d) arrays, refused where they cannot be segments of this domain's space."""
        a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
        if a.shape != b.shape or a.ndim not in (1, 2) or a.shape[-1] != self.dimension:
            raise ValueError(
                f"segment ends must both have shape (d,) or (N, d), with d = {self.dimension} for this domain; got "
                f"{a.shape} and {b.shape}"
            )
        if not (np.isfinite(a).all() and np.isfinite(b).all()):
            raise ValueError(
                "segment ends must be finite: a NaN end would classify the whole segment as outside, with nothing "
                "raised"
            )
        with np.errstate(over="ignore"):
            extent = b - a
        if not np.isfinite(extent).all():
            raise ValueError(
                "a segment's extent b - a overflows: its direction is infinite, so the intervals along it and the "
                "start's admission are not the domain's"
            )
        return np.atleast_2d(a), np.atleast_2d(b)

    def _line_sets(
        self, a: NDArray[np.float64], d: NDArray[np.float64], outside: bool, leaf: int
    ) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
        """Along each line a + t d: the closed set where phi <= 0, or with ``outside`` where phi >= 0, as sorted
        disjoint intervals of shape (N, K, 2) with NaN rows, and each end's surface packed as one integer
        (`_surface_tag`). ``leaf`` is the index of this node's first leaf in the ``_leaves()`` of the domain the search
        began at. A leaf solves its own form and a CSG node combines its children's sets (#2733 D1).

        Here, with no form to solve, it is refused. A search that steps along the segment by phi, sphere tracing, is
        exact only where phi is a distance or bounds one, and that is stated (`get_lipschitz_constant`) and never
        checked: a phi such as |x|^2 - r^2 would let it step over a feature with nothing raised."""
        raise NotImplementedError(
            f"{type(self).__name__} has no exact segment classification: segment_intervals and first_contact are "
            f"defined for Hyperrectangle, Hypersphere and the CSG domains built from them. A domain given only by its "
            f"phi is refused, not sphere-traced, since phi is never checked to be a distance, and where it is not, a "
            f"search that steps by phi passes over a feature with nothing raised (#2733 D1)."
        )

    def _leaves(self) -> tuple["ImplicitDomain", ...]:
        """The primitives this domain is built from, depth first: the domain itself, unless it is a CSG node."""
        return (self,)

    def _face_normal(self, x: NDArray[np.float64], face: NDArray[np.int64]) -> NDArray[np.float64]:
        """The outward unit normal of this leaf's face ``face`` at the points ``x`` on it, shape (n, d). A leaf with
        no exact form has none; `_line_sets` refuses it before this is reached."""
        raise NotImplementedError(f"{type(self).__name__} has no exact face normals (#2733 D1).")

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
            # Exact SDFs are 1-Lipschitz: |φ(x) - φ(y)| ≤ |x - y|
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


def _surface_tag(leaf: int, face: NDArray[np.int64] | int, complemented: bool) -> NDArray[np.int64]:
    """An interval end's surface as one integer, which sorts and gathers with its parameter: the leaf in the high 32
    bits, then the face, then whether the leaf enters complemented (`SegmentIntervals`)."""
    return (np.int64(leaf) << 32) | (np.asarray(face, dtype=np.int64) << 1) | np.int64(complemented)


def _closed_inside(
    t_in: NDArray[np.float64],
    t_out: NDArray[np.float64],
    holds: NDArray[np.bool_],
    tag_in: NDArray[np.int64],
    tag_out: NDArray[np.int64],
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """A convex leaf's phi <= 0 along each line: [t_in, t_out] where ``holds``, else nothing. Shape (N, 1, 2)."""
    t = np.where(holds[:, None], np.stack([t_in, t_out], axis=1), np.nan)[:, None, :]
    tag = np.where(holds[:, None], np.stack([tag_in, tag_out], axis=1), _NO_SURFACE)[:, None, :]
    return t, tag


def _closed_outside(
    t_in: NDArray[np.float64],
    t_out: NDArray[np.float64],
    meets: NDArray[np.bool_],
    tag_in: NDArray[np.int64],
    tag_out: NDArray[np.int64],
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """A convex leaf's phi >= 0 along each line, the complement of its open interior (t_in, t_out) where the line
    ``meets`` it: the line before t_in and after t_out, nothing of either part at infinity; and the whole line where it
    misses the open interior, as a line that only grazes the leaf does. Shape (N, 2, 2)."""
    n = t_in.shape[0]
    before, after = meets & (t_in > -np.inf), meets & (t_out < np.inf)
    t = np.full((n, 2, 2), np.nan)
    tag = np.full((n, 2, 2), _NO_SURFACE, dtype=np.int64)
    t[~meets, 0] = (-np.inf, np.inf)
    t[before, 0, 0], t[before, 0, 1], tag[before, 0, 1] = -np.inf, t_in[before], tag_in[before]
    t[after, 1, 0], t[after, 1, 1], tag[after, 1, 0] = t_out[after], np.inf, tag_out[after]
    return t, tag


def _combine_line_sets(
    parts: list[tuple[NDArray[np.float64], NDArray[np.int64]]], need: int
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """The set of each line that at least ``need`` of ``parts`` cover: their union for 1, their intersection for
    ``len(parts)`` (#2733 D1). Each part is a `_line_sets` result.

    A sweep over every part's ends, in parameter order, counting the parts that cover; at an equal parameter an
    opening end comes before a closing one, so that closed intervals that touch are joined, and a point where one
    closes as another opens is kept, as the closed sets' intersection holds it."""
    n = parts[0][0].shape[0]
    # Each part's (N, K, 2) ends as (N, 2K) events. The width is given, not inferred, so N = 0 reshapes too.
    t = np.concatenate([part[0].reshape(n, 2 * part[0].shape[1]) for part in parts], axis=1)
    tag = np.concatenate([part[1].reshape(n, 2 * part[1].shape[1]) for part in parts], axis=1)
    closing = np.broadcast_to(np.arange(t.shape[1]) % 2, t.shape)
    order = np.lexsort((closing, t), axis=-1)
    t, tag, closing = (np.take_along_axis(x, order, axis=1) for x in (t, tag, closing))
    step = np.where(np.isnan(t), 0, 1 - 2 * closing)
    covered = np.cumsum(step, axis=1)
    opens, closes = (step == 1) & (covered == need), (step == -1) & (covered == need - 1)
    count = opens.sum(axis=1)
    k = max(int(count.max(initial=0)), 1)
    real = np.arange(k)[None, :] < count[:, None]
    first_opens = np.argsort(~opens, axis=1, kind="stable")[:, :k]
    first_closes = np.argsort(~closes, axis=1, kind="stable")[:, :k]
    out_t = np.stack([np.take_along_axis(t, first_opens, axis=1), np.take_along_axis(t, first_closes, axis=1)], -1)
    out_tag = np.stack(
        [np.take_along_axis(tag, first_opens, axis=1), np.take_along_axis(tag, first_closes, axis=1)], -1
    )
    return np.where(real[..., None], out_t, np.nan), np.where(real[..., None], out_tag, _NO_SURFACE)


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
